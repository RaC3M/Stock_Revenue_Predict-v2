from __future__ import annotations

"""Compare Rolling sequence backbones with fixed experiment settings.

This D1.5 runner keeps the rolling data, KMeans clusters, growth adjustment,
and evaluation path fixed while swapping only the recurrent backbone.
"""

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import rolling_lstm_engine as engine
    from .experiment_metrics import metric_record, summarize
    from .experiment_metadata import write_rolling_run_config
except ImportError:
    import rolling_lstm_engine as engine
    from experiment_metrics import metric_record, summarize
    from experiment_metadata import write_rolling_run_config


DEFAULT_STOCK_IDS = (1101, 1231, 3017)
DEFAULT_BACKBONES = ("lstm", "xlstm")
DEFAULT_K = 4
DEFAULT_EPOCHS = 5
DEFAULT_MAX_TRAIN_SAMPLES = 5_000
DEFAULT_UNDER_WEIGHT = 2.0
DEFAULT_GROWTH_ALPHA = 0.8
DEFAULT_MIN_2025_MONTHS = 12
DEFAULT_SAMPLE_SEED = 42

MODEL_COLUMNS = {
    "Rolling LSTM": "predicted_revenue_no_cluster",
    "Rolling LSTM + Cluster": "predicted_revenue_cluster",
    "Rolling LSTM + Cluster + Conditional Adjustment": "predicted_revenue_adjusted",
}

ERROR_METRICS = ("MSE", "RMSE", "MAE", "MAPE", "MedianAPE", "WMAPE", "SMAPE")
COMPARISON_METRICS = (*ERROR_METRICS, "Bias", "UnderestimateRate", "DirectionAccuracy", "runtime_seconds")


@dataclass(frozen=True)
class BackboneRunSpec:
    backbone: str
    k: int = DEFAULT_K
    epochs: int = DEFAULT_EPOCHS
    max_train_samples: int = DEFAULT_MAX_TRAIN_SAMPLES
    use_asymmetric_loss: bool = True
    under_weight: float = DEFAULT_UNDER_WEIGHT
    growth_enabled: bool = True
    growth_alpha: float = DEFAULT_GROWTH_ALPHA
    growth_conditional: bool = True
    growth_regime_strategy: bool = True


def parse_int_csv(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not value:
        return default
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("At least one integer value is required.")
    return parsed


def parse_optional_int_csv(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    return parse_int_csv(value, ())


def parse_backbones(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_BACKBONES
    backbones = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not backbones:
        raise ValueError("At least one sequence backbone is required.")
    unknown = sorted(set(backbones) - set(engine.SEQUENCE_BACKBONES))
    if unknown:
        raise ValueError(f"Unknown sequence backbones: {unknown}")
    return backbones


def resolve_effect_backbones(backbones: tuple[str, ...]) -> tuple[str, str] | None:
    distinct = tuple(dict.fromkeys(backbones))
    if len(distinct) < 2:
        return None
    if len(distinct) == 2:
        return distinct[0], distinct[1]
    if {"lstm", "xlstm"}.issubset(distinct):
        return "lstm", "xlstm"
    return distinct[0], distinct[1]


def clear_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def warm_shared_caches(k: int, stock_ids: tuple[int, ...]) -> None:
    revenue_mtime_ns = engine._revenue_file_mtime_ns()
    window_size = engine.DEFAULT_WINDOW_SIZE
    engine._cached_revenue_and_windows(window_size, revenue_mtime_ns)
    engine._cached_clustered_artifacts(k, window_size, revenue_mtime_ns)
    engine._cached_train_samples(k, window_size, revenue_mtime_ns)
    engine._cached_all_eval_samples(k, window_size, revenue_mtime_ns)
    for stock_id in stock_ids:
        engine._cached_eval_samples(k, window_size, int(stock_id), revenue_mtime_ns)


def load_stock_metadata(revenue: pd.DataFrame) -> pd.DataFrame:
    stock_meta = (
        revenue.sort_values("date")
        .groupby("stock_id", as_index=False)
        .agg(
            industry_category=("industry_category", "last"),
            available_months_2025=("revenue_year", lambda values: int((values == engine.FORECAST_YEAR).sum())),
        )
    )
    stock_list_path = Path(__file__).resolve().parent.parent / "data" / "stock_list_new.csv"
    if stock_list_path.exists():
        names = pd.read_csv(stock_list_path, usecols=["stock_id", "stock_name"])
        names["stock_id"] = pd.to_numeric(names["stock_id"], errors="coerce")
        names = names.dropna(subset=["stock_id"]).drop_duplicates("stock_id")
        names["stock_id"] = names["stock_id"].astype(int)
        stock_meta = stock_meta.merge(names, on="stock_id", how="left")
    else:
        stock_meta["stock_name"] = ""
    stock_meta["industry_category"] = stock_meta["industry_category"].fillna("unknown")
    stock_meta["stock_name"] = stock_meta["stock_name"].fillna("")
    return stock_meta


def build_candidate_stock_pool(
    revenue: pd.DataFrame,
    stock_meta: pd.DataFrame,
    min_2025_months: int,
) -> pd.DataFrame:
    revenue_2024 = (
        revenue[revenue["revenue_year"].astype(int).eq(engine.FORECAST_YEAR - 1)]
        .groupby("stock_id", as_index=False)["revenue_thousand"]
        .sum()
        .rename(columns={"revenue_thousand": "annual_revenue_2024"})
    )
    pool = stock_meta.merge(revenue_2024, on="stock_id", how="left")
    pool["annual_revenue_2024"] = pd.to_numeric(pool["annual_revenue_2024"], errors="coerce").fillna(0)
    pool = pool[pool["available_months_2025"].astype(int) >= int(min_2025_months)].copy()
    return pool.sort_values(["industry_category", "stock_id"]).reset_index(drop=True)


def select_auto_stock_ids(
    candidate_pool: pd.DataFrame,
    stock_limit: int,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> tuple[int, ...]:
    if stock_limit <= 0:
        raise ValueError("stock_limit must be positive.")
    if candidate_pool.empty:
        raise ValueError("No eligible stocks are available for automatic basket selection.")
    if stock_limit >= len(candidate_pool):
        return tuple(candidate_pool["stock_id"].astype(int).tolist())

    rng = np.random.default_rng(int(sample_seed))
    industries = candidate_pool["industry_category"].fillna("unknown").drop_duplicates().sort_values().tolist()
    selected_indices: list[int] = []
    shuffled_by_industry: dict[str, list[int]] = {}
    for industry in industries:
        industry_indices = candidate_pool.index[candidate_pool["industry_category"].fillna("unknown").eq(industry)]
        shuffled = list(rng.permutation(industry_indices.to_numpy()))
        shuffled_by_industry[str(industry)] = [int(index) for index in shuffled]

    while len(selected_indices) < stock_limit:
        made_progress = False
        for industry in industries:
            bucket = shuffled_by_industry[str(industry)]
            if not bucket:
                continue
            selected_indices.append(bucket.pop(0))
            made_progress = True
            if len(selected_indices) >= stock_limit:
                break
        if not made_progress:
            break

    selected = candidate_pool.loc[selected_indices].sort_values(["industry_category", "stock_id"])
    return tuple(selected["stock_id"].astype(int).tolist())


def resolve_stock_ids(
    revenue: pd.DataFrame,
    stock_meta: pd.DataFrame,
    explicit_stocks: str | None,
    stock_limit: int | None,
    min_2025_months: int,
    sample_seed: int,
) -> tuple[int, ...]:
    parsed = parse_optional_int_csv(explicit_stocks)
    if parsed is not None:
        return parsed
    if stock_limit is None:
        return DEFAULT_STOCK_IDS
    candidate_pool = build_candidate_stock_pool(revenue, stock_meta, min_2025_months=min_2025_months)
    return select_auto_stock_ids(candidate_pool, stock_limit=stock_limit, sample_seed=sample_seed)


def extract_sequence_note(notes: list[str]) -> str:
    for note in notes:
        if note.startswith("Sequence backbone="):
            return note
    return ""


def build_run_config(spec: BackboneRunSpec) -> engine.RollingExperimentConfig:
    backbone = str(spec.backbone).lower()
    return engine.RollingExperimentConfig(
        k=int(spec.k),
        window_size=engine.DEFAULT_WINDOW_SIZE,
        epochs=int(spec.epochs),
        max_train_samples=int(spec.max_train_samples),
        sequence_backbone=backbone,
        include_xlstm_plain=False,
        xlstm_backbone=(
            engine.get_xlstm_backbone_spec(backbone).key
            if backbone in engine.XLSTM_BACKBONES
            else engine.DEFAULT_XLSTM_BACKBONE
        ),
        use_asymmetric_loss=bool(spec.use_asymmetric_loss),
        under_weight=float(spec.under_weight),
        growth=engine.GrowthAdjustmentConfig(
            enabled=bool(spec.growth_enabled),
            alpha=float(spec.growth_alpha),
            conditional=bool(spec.growth_conditional),
            regime_strategy=bool(spec.growth_regime_strategy),
        ),
    )


def build_monthly_long_frame(
    forecast: pd.DataFrame,
    stock_id: int,
    stock_name: str,
    industry_category: str,
    backbone: str,
) -> pd.DataFrame:
    id_columns = [
        "target_date",
        "target_year",
        "target_month",
        "actual_revenue",
        "last_observed_revenue",
        "regime",
        "cluster",
        "growth_ratio",
        "growth_streak",
        "is_growth_phase",
        "adjustment_applied",
        "prediction_cap",
    ]
    parts = []
    for model, prediction_column in MODEL_COLUMNS.items():
        part = forecast[id_columns].copy()
        part.insert(0, "sequence_backbone", backbone)
        part.insert(1, "xlstm_backbone", backbone if backbone in engine.XLSTM_BACKBONES else pd.NA)
        part.insert(2, "stock_id", int(stock_id))
        part.insert(3, "stock_name", stock_name)
        part.insert(4, "industry_category", industry_category)
        part["model"] = model
        part["predicted_revenue"] = forecast[prediction_column].to_numpy()
        part["error"] = part["predicted_revenue"] - part["actual_revenue"]
        part["abs_error"] = part["error"].abs()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def add_runtime_columns(metrics: pd.DataFrame, runtime_seconds: float) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["runtime_seconds"] = float(round(runtime_seconds, 3))
    metrics["runtime_minutes"] = float(round(runtime_seconds / 60.0, 3))
    return metrics


def build_backbone_effects(
    stock_accuracy: pd.DataFrame,
    baseline_backbone: str = "lstm",
    challenger_backbone: str = "xlstm",
) -> pd.DataFrame:
    index_columns = ["stock_id", "stock_name", "industry_category", "model"]
    available_metrics = [metric for metric in COMPARISON_METRICS if metric in stock_accuracy.columns]
    if not available_metrics:
        return pd.DataFrame(columns=index_columns)

    wide = stock_accuracy.pivot_table(
        index=index_columns,
        columns="sequence_backbone",
        values=available_metrics,
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{backbone}" for metric, backbone in wide.columns]
    wide = wide.reset_index()
    for metric in available_metrics:
        baseline_column = f"{metric}_{baseline_backbone}"
        challenger_column = f"{metric}_{challenger_backbone}"
        if baseline_column in wide.columns and challenger_column in wide.columns:
            wide[f"{metric}_delta_{challenger_backbone}_minus_{baseline_backbone}"] = (
                wide[challenger_column] - wide[baseline_column]
            )
    for metric in ("MAPE", "WMAPE"):
        delta_column = f"{metric}_delta_{challenger_backbone}_minus_{baseline_backbone}"
        if delta_column not in wide.columns:
            continue
        delta = wide[delta_column]
        wide[f"{metric}_winner"] = np.select(
            [delta < 0, delta > 0],
            [challenger_backbone, baseline_backbone],
            default="tie",
        )
    return wide


def build_winner_summary(
    backbone_effects: pd.DataFrame,
    baseline_backbone: str = "lstm",
    challenger_backbone: str = "xlstm",
) -> pd.DataFrame:
    if backbone_effects.empty or "MAPE_winner" not in backbone_effects.columns:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for model, frame in backbone_effects.groupby("model", sort=True, dropna=False):
        winner_counts = frame["MAPE_winner"].value_counts()
        stock_count = int(frame["stock_id"].nunique())
        row = {
            "model": model,
            "stock_count": stock_count,
            f"{challenger_backbone}_wins": int(winner_counts.get(challenger_backbone, 0)),
            f"{baseline_backbone}_wins": int(winner_counts.get(baseline_backbone, 0)),
            "ties": int(winner_counts.get("tie", 0)),
            f"{challenger_backbone}_win_rate": (
                float(winner_counts.get(challenger_backbone, 0) / len(frame) * 100) if len(frame) else np.nan
            ),
        }
        for metric in ("MAPE", "WMAPE"):
            winner_column = f"{metric}_winner"
            if winner_column not in frame.columns:
                continue
            metric_winner_counts = winner_counts if metric == "MAPE" else frame[winner_column].value_counts()
            row[f"{metric}_{challenger_backbone}_wins"] = int(
                metric_winner_counts.get(challenger_backbone, 0)
            )
            row[f"{metric}_{baseline_backbone}_wins"] = int(
                metric_winner_counts.get(baseline_backbone, 0)
            )
            row[f"{metric}_ties"] = int(metric_winner_counts.get("tie", 0))
            row[f"{metric}_{challenger_backbone}_win_rate"] = (
                float(metric_winner_counts.get(challenger_backbone, 0) / len(frame) * 100)
                if len(frame)
                else np.nan
            )
        for metric in ["MAPE", "WMAPE", "MAE", "MedianAPE", "SMAPE", "Bias", "UnderestimateRate", "DirectionAccuracy"]:
            delta_column = f"{metric}_delta_{challenger_backbone}_minus_{baseline_backbone}"
            if delta_column in frame.columns:
                row[f"average_{metric}_delta"] = float(frame[delta_column].mean())
                row[f"median_{metric}_delta"] = float(frame[delta_column].median())
        rows.append(row)
    return pd.DataFrame(rows)


def build_underestimate_risk(stock_accuracy: pd.DataFrame) -> pd.DataFrame:
    if stock_accuracy.empty:
        return pd.DataFrame()
    required_columns = [
        "sequence_backbone",
        "stock_id",
        "stock_name",
        "industry_category",
        "model",
        "MAPE",
        "MAE",
        "Bias",
        "UnderestimateRate",
    ]
    missing = set(required_columns) - set(stock_accuracy.columns)
    if missing:
        return pd.DataFrame(columns=required_columns)

    risk = stock_accuracy[required_columns].copy()
    risk["is_underestimate_bias"] = risk["Bias"] < 0
    risk["underestimate_risk_level"] = np.select(
        [
            risk["is_underestimate_bias"] & (risk["UnderestimateRate"] >= 75),
            risk["is_underestimate_bias"] & (risk["UnderestimateRate"] >= 50),
        ],
        ["high", "medium"],
        default="low",
    )
    risk["underestimate_risk_score"] = risk["underestimate_risk_level"].map({"high": 3, "medium": 2, "low": 1})
    return risk.sort_values(
        ["underestimate_risk_score", "UnderestimateRate", "MAPE"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stock-limit", type=int, default=None)
    parser.add_argument("--min-2025-months", type=int, default=DEFAULT_MIN_2025_MONTHS)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--backbones", default=",".join(DEFAULT_BACKBONES))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max-train-samples", type=int, default=DEFAULT_MAX_TRAIN_SAMPLES)
    parser.add_argument("--under-weight", type=float, default=DEFAULT_UNDER_WEIGHT)
    parser.add_argument("--growth-alpha", type=float, default=DEFAULT_GROWTH_ALPHA)
    parser.add_argument("--disable-growth-adjustment", action="store_true")
    parser.add_argument("--disable-asymmetric-loss", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    backbones = parse_backbones(args.backbones)

    started = time.time()
    revenue = engine.load_revenue_data()
    stock_meta_frame = load_stock_metadata(revenue)
    stock_ids = resolve_stock_ids(
        revenue,
        stock_meta_frame,
        explicit_stocks=args.stocks,
        stock_limit=args.stock_limit,
        min_2025_months=int(args.min_2025_months),
        sample_seed=int(args.sample_seed),
    )
    stock_meta = stock_meta_frame.set_index("stock_id")
    available_stock_ids = set(engine.get_stock_list(revenue))
    missing_stocks = [stock_id for stock_id in stock_ids if stock_id not in available_stock_ids]
    if missing_stocks:
        raise ValueError(f"Stocks are not available in revenue data: {missing_stocks}")

    metric_rows: list[pd.DataFrame] = []
    monthly_rows: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    print(f"Running sequence backbone ablation: stocks={stock_ids}, backbones={backbones}", flush=True)
    print("Warming shared revenue, KMeans, training-sample, and eval-sample caches.", flush=True)
    warm_shared_caches(int(args.k), stock_ids)

    for backbone in backbones:
        spec = BackboneRunSpec(
            backbone=backbone,
            k=int(args.k),
            epochs=int(args.epochs),
            max_train_samples=int(args.max_train_samples),
            use_asymmetric_loss=not bool(args.disable_asymmetric_loss),
            under_weight=float(args.under_weight),
            growth_enabled=not bool(args.disable_growth_adjustment),
            growth_alpha=float(args.growth_alpha),
        )
        print(f"Starting backbone={backbone}", flush=True)
        for position, stock_id in enumerate(stock_ids, start=1):
            stock_started = time.time()
            try:
                result = engine.run_rolling_lstm_experiment(
                    selected_stock=int(stock_id),
                    config=build_run_config(spec),
                )
                runtime_seconds = time.time() - stock_started
                stock_name = str(stock_meta.at[stock_id, "stock_name"]) if stock_id in stock_meta.index else ""
                industry = (
                    str(stock_meta.at[stock_id, "industry_category"]) if stock_id in stock_meta.index else "unknown"
                )
                metrics = result.metrics[result.metrics["model"].isin(MODEL_COLUMNS)].copy()
                metrics = metrics.drop(columns=["sequence_backbone", "xlstm_backbone"], errors="ignore")
                metrics.insert(0, "sequence_backbone", backbone)
                metrics.insert(1, "xlstm_backbone", backbone if backbone in engine.XLSTM_BACKBONES else pd.NA)
                metrics.insert(2, "stock_id", int(stock_id))
                metrics.insert(3, "stock_name", stock_name)
                metrics.insert(4, "industry_category", industry)
                metrics["k"] = int(args.k)
                metrics["epochs"] = int(args.epochs)
                metrics["max_train_samples"] = int(args.max_train_samples)
                metrics["sequence_backend_note"] = extract_sequence_note(result.notes)
                metric_rows.append(add_runtime_columns(metrics, runtime_seconds))
                monthly_rows.append(
                    build_monthly_long_frame(
                        result.forecast,
                        stock_id=int(stock_id),
                        stock_name=stock_name,
                        industry_category=industry,
                        backbone=backbone,
                    )
                )
            except Exception as error:
                failures.append(
                    {
                        "sequence_backbone": backbone,
                        "stock_id": int(stock_id),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            clear_torch_cache()
            print(
                f"{backbone}: {position}/{len(stock_ids)} stocks processed, "
                f"failures={sum(item['sequence_backbone'] == backbone for item in failures)}, "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    stock_accuracy = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    monthly_predictions = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()
    overall_accuracy = (
        summarize(monthly_predictions, ["sequence_backbone", "model"]) if not monthly_predictions.empty else pd.DataFrame()
    )
    industry_backbone_accuracy = (
        summarize(monthly_predictions, ["sequence_backbone", "industry_category", "model"])
        if not monthly_predictions.empty
        else pd.DataFrame()
    )
    regime_backbone_accuracy = (
        summarize(monthly_predictions, ["sequence_backbone", "regime", "model"])
        if not monthly_predictions.empty
        else pd.DataFrame()
    )
    effect_backbones = resolve_effect_backbones(backbones)
    available_backbones = set(stock_accuracy.get("sequence_backbone", pd.Series(dtype=str)))
    if effect_backbones is not None and set(effect_backbones).issubset(available_backbones):
        baseline_backbone, challenger_backbone = effect_backbones
        backbone_effects = build_backbone_effects(
            stock_accuracy,
            baseline_backbone=baseline_backbone,
            challenger_backbone=challenger_backbone,
        )
        winner_summary = build_winner_summary(
            backbone_effects,
            baseline_backbone=baseline_backbone,
            challenger_backbone=challenger_backbone,
        )
    else:
        backbone_effects = pd.DataFrame()
        winner_summary = pd.DataFrame()
    underestimate_risk = build_underestimate_risk(stock_accuracy)
    failure_frame = pd.DataFrame(failures, columns=["sequence_backbone", "stock_id", "error_type", "error"])

    write_csv(stock_accuracy, output_dir / "stock_accuracy.csv")
    write_csv(overall_accuracy, output_dir / "overall_accuracy.csv")
    write_csv(industry_backbone_accuracy, output_dir / "industry_backbone_accuracy.csv")
    write_csv(regime_backbone_accuracy, output_dir / "regime_backbone_accuracy.csv")
    write_csv(backbone_effects, output_dir / "backbone_effects.csv")
    write_csv(winner_summary, output_dir / "winner_summary.csv")
    write_csv(underestimate_risk, output_dir / "underestimate_risk.csv")
    write_csv(monthly_predictions, output_dir / "monthly_predictions.csv")
    write_csv(failure_frame, output_dir / "failed_runs.csv")

    run_summary = {
        "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": round(time.time() - started, 3),
        "stocks": list(stock_ids),
        "backbones": list(backbones),
        "stock_selection": {
            "explicit_stocks": args.stocks,
            "stock_limit": args.stock_limit,
            "min_2025_months": int(args.min_2025_months),
            "sample_seed": int(args.sample_seed),
        },
        "parameters": {
            "k": int(args.k),
            "window_size": engine.DEFAULT_WINDOW_SIZE,
            "epochs": int(args.epochs),
            "max_train_samples": int(args.max_train_samples),
            "use_asymmetric_loss": not bool(args.disable_asymmetric_loss),
            "under_weight": float(args.under_weight),
            "growth_enabled": not bool(args.disable_growth_adjustment),
            "growth_alpha": float(args.growth_alpha),
        },
        "row_counts": {
            "stock_accuracy": int(len(stock_accuracy)),
            "overall_accuracy": int(len(overall_accuracy)),
            "industry_backbone_accuracy": int(len(industry_backbone_accuracy)),
            "regime_backbone_accuracy": int(len(regime_backbone_accuracy)),
            "backbone_effects": int(len(backbone_effects)),
            "winner_summary": int(len(winner_summary)),
            "underestimate_risk": int(len(underestimate_risk)),
            "monthly_predictions": int(len(monthly_predictions)),
            "failed_runs": int(len(failure_frame)),
        },
        "metric_notes": {
            "revenue_unit": "thousand TWD (same as revenue_thousand source)",
            "effect_delta": (
                f"{effect_backbones[1]} minus {effect_backbones[0]}; negative MAPE/MAE deltas mean "
                f"{effect_backbones[1]} improved error"
                if effect_backbones is not None
                else "Unavailable because fewer than two distinct backbones were requested."
            ),
            "runtime_seconds": (
                "per stock call runtime after shared revenue/KMeans/sample caches are warmed; "
                "first stock for each backbone still includes backbone-specific model training/cache build"
            ),
        },
    }
    run_summary = write_rolling_run_config(
        output_dir,
        run_summary,
        experiment_family="rolling_sequence_backbone",
        evidence_tier="C",
        selection_protocol="target-year-hindsight",
        report_ready=False,
        report_ready_reason="Backbone candidates were compared on the target-year evaluation set.",
    )

    print(json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
