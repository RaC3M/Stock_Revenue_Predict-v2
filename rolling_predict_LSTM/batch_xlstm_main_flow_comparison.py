from __future__ import annotations

"""Evaluate architecture-explicit Rolling xLSTM comparison rows in batch.

This runner keeps the main LSTM flow fixed and enables the optional no-cluster
xLSTM comparison row. The default follows the current Streamlit Hybrid choice;
``--xlstm-backbone xlstm`` reproduces the historical mLSTM-only D1 architecture.
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
    from . import batch_sequence_backbone_ablation as basket
    from .experiment_metadata import write_rolling_run_config
except ImportError:
    import rolling_lstm_engine as engine
    import batch_sequence_backbone_ablation as basket
    from experiment_metadata import write_rolling_run_config


DEFAULT_STOCK_IDS = (1101, 1231, 3017)
DEFAULT_K = 4
DEFAULT_EPOCHS = 5
DEFAULT_MAX_TRAIN_SAMPLES = 5_000
DEFAULT_UNDER_WEIGHT = 2.0
DEFAULT_GROWTH_ALPHA = 0.8
DEFAULT_XLSTM_ALPHA = engine.DEFAULT_XLSTM_GROWTH_ADJUSTMENT_ALPHA
DEFAULT_XLSTM_DECLINE_CAP_GROWTH_RATIO_MAX = engine.DEFAULT_XLSTM_DECLINE_CAP_GROWTH_RATIO_MAX
DEFAULT_XLSTM_DECLINE_CAP_PREDICTION_RATIO_MIN = engine.DEFAULT_XLSTM_DECLINE_CAP_PREDICTION_RATIO_MIN
DEFAULT_MIN_2025_MONTHS = 12
DEFAULT_SAMPLE_SEED = 42

MODEL_COLUMNS = {
    model_name: prediction_column
    for model_name, prediction_column, _, _ in engine.ROLLING_MODEL_OUTPUTS
}
MODEL_PROVENANCE_COLUMNS = ["xlstm_backbone", "sequence_backbone"]

MODEL_EFFECT_PAIRS = (
    (
        "xlstm_adjusted_minus_cluster_adjusted",
        "Rolling LSTM + Cluster + Conditional Adjustment",
        "Rolling xLSTM + Conditional Adjustment",
    ),
    (
        "xlstm_adjusted_minus_xlstm_plain",
        "Rolling xLSTM",
        "Rolling xLSTM + Conditional Adjustment",
    ),
    (
        "xlstm_plain_minus_lstm_plain",
        "Rolling LSTM",
        "Rolling xLSTM",
    ),
)

COMPARISON_METRICS = (
    "MSE",
    "RMSE",
    "MAE",
    "MAPE",
    "MedianAPE",
    "WMAPE",
    "SMAPE",
    "Bias",
    "UnderestimateRate",
    "DirectionAccuracy",
)
LOWER_IS_BETTER_METRICS = ("MSE", "RMSE", "MAE", "MAPE", "MedianAPE", "WMAPE", "SMAPE")
HIGHER_IS_BETTER_METRICS = ("DirectionAccuracy",)


@dataclass(frozen=True)
class MainFlowRunSpec:
    k: int = DEFAULT_K
    epochs: int = DEFAULT_EPOCHS
    max_train_samples: int = DEFAULT_MAX_TRAIN_SAMPLES
    use_asymmetric_loss: bool = True
    under_weight: float = DEFAULT_UNDER_WEIGHT
    growth_enabled: bool = True
    growth_alpha: float = DEFAULT_GROWTH_ALPHA
    xlstm_growth_alpha: float = DEFAULT_XLSTM_ALPHA
    xlstm_decline_cap_growth_ratio_max: float | None = DEFAULT_XLSTM_DECLINE_CAP_GROWTH_RATIO_MAX
    xlstm_decline_cap_prediction_ratio_min: float = DEFAULT_XLSTM_DECLINE_CAP_PREDICTION_RATIO_MIN
    growth_conditional: bool = True
    growth_regime_strategy: bool = True
    xlstm_backbone: str = engine.DEFAULT_STREAMLIT_XLSTM_BACKBONE


def clear_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def build_run_config(spec: MainFlowRunSpec) -> engine.RollingExperimentConfig:
    xlstm_backbone = engine.get_xlstm_backbone_spec(spec.xlstm_backbone).key
    return engine.RollingExperimentConfig(
        k=int(spec.k),
        window_size=engine.DEFAULT_WINDOW_SIZE,
        epochs=int(spec.epochs),
        max_train_samples=int(spec.max_train_samples),
        sequence_backbone="lstm",
        include_xlstm_plain=True,
        xlstm_backbone=xlstm_backbone,
        use_asymmetric_loss=bool(spec.use_asymmetric_loss),
        under_weight=float(spec.under_weight),
        growth=engine.GrowthAdjustmentConfig(
            enabled=bool(spec.growth_enabled),
            alpha=float(spec.growth_alpha),
            conditional=bool(spec.growth_conditional),
            regime_strategy=bool(spec.growth_regime_strategy),
        ),
        xlstm_growth=engine.GrowthAdjustmentConfig(
            enabled=bool(spec.growth_enabled),
            alpha=float(spec.xlstm_growth_alpha),
            conditional=bool(spec.growth_conditional),
            regime_strategy=bool(spec.growth_regime_strategy),
            decline_cap_growth_ratio_max=spec.xlstm_decline_cap_growth_ratio_max,
            decline_cap_prediction_ratio_min=float(spec.xlstm_decline_cap_prediction_ratio_min),
        ),
    )


def extract_sequence_note(notes: list[str]) -> str:
    for note in notes:
        if note.startswith("Sequence backbone="):
            return note
    return ""


def build_monthly_long_frame(
    forecast: pd.DataFrame,
    stock_id: int,
    stock_name: str,
    industry_category: str,
    xlstm_backbone: str,
) -> pd.DataFrame:
    xlstm_backbone = engine.get_xlstm_backbone_spec(xlstm_backbone).key
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
        "decline_cap_applied",
        "xlstm_adjustment_applied",
        "xlstm_decline_cap_applied",
        "prediction_cap",
        "decline_cap_growth_ratio_max",
        "decline_cap_prediction_ratio_min",
        "xlstm_decline_cap_growth_ratio_max",
        "xlstm_decline_cap_prediction_ratio_min",
    ]
    available_id_columns = [column for column in id_columns if column in forecast.columns]
    parts: list[pd.DataFrame] = []
    for model, prediction_column in MODEL_COLUMNS.items():
        part = forecast[available_id_columns].copy()
        part.insert(0, "stock_id", int(stock_id))
        part.insert(1, "stock_name", stock_name)
        part.insert(2, "industry_category", industry_category)
        part.insert(3, "xlstm_backbone", xlstm_backbone)
        part.insert(
            4,
            "sequence_backbone",
            engine.resolve_model_sequence_backbone(
                model,
                main_sequence_backbone="lstm",
                xlstm_backbone=xlstm_backbone,
                include_xlstm_plain=True,
            ),
        )
        part["model"] = model
        part["predicted_revenue"] = forecast[prediction_column].to_numpy()
        part["error"] = part["predicted_revenue"] - part["actual_revenue"]
        part["abs_error"] = part["error"].abs()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build_accuracy_summaries(
    monthly_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if monthly_predictions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    overall_accuracy = basket.summarize(
        monthly_predictions,
        [*MODEL_PROVENANCE_COLUMNS, "model"],
    )
    industry_accuracy = basket.summarize(
        monthly_predictions,
        [*MODEL_PROVENANCE_COLUMNS, "industry_category", "model"],
    )
    regime_accuracy = basket.summarize(
        monthly_predictions,
        [*MODEL_PROVENANCE_COLUMNS, "regime", "model"],
    )
    return overall_accuracy, industry_accuracy, regime_accuracy


def add_runtime_columns(metrics: pd.DataFrame, runtime_seconds: float) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["runtime_seconds"] = float(round(runtime_seconds, 3))
    metrics["runtime_minutes"] = float(round(runtime_seconds / 60.0, 3))
    return metrics


def build_model_effects(
    stock_accuracy: pd.DataFrame,
    effect_pairs: tuple[tuple[str, str, str], ...] = MODEL_EFFECT_PAIRS,
) -> pd.DataFrame:
    if stock_accuracy.empty:
        return pd.DataFrame()
    index_columns = ["stock_id", "stock_name", "industry_category"]
    if "xlstm_backbone" in stock_accuracy.columns:
        index_columns.append("xlstm_backbone")
    available_metrics = [metric for metric in COMPARISON_METRICS if metric in stock_accuracy.columns]
    wide = stock_accuracy.pivot_table(
        index=index_columns,
        columns="model",
        values=available_metrics,
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()

    rows: list[pd.DataFrame] = []
    for effect_name, baseline_model, challenger_model in effect_pairs:
        part = wide[index_columns].copy()
        part["effect"] = effect_name
        part["baseline_model"] = baseline_model
        part["challenger_model"] = challenger_model
        if "xlstm_backbone" in part.columns:
            part["baseline_sequence_backbone"] = [
                engine.resolve_model_sequence_backbone(
                    baseline_model,
                    main_sequence_backbone="lstm",
                    xlstm_backbone=backbone,
                    include_xlstm_plain=True,
                )
                for backbone in part["xlstm_backbone"]
            ]
            part["challenger_sequence_backbone"] = [
                engine.resolve_model_sequence_backbone(
                    challenger_model,
                    main_sequence_backbone="lstm",
                    xlstm_backbone=backbone,
                    include_xlstm_plain=True,
                )
                for backbone in part["xlstm_backbone"]
            ]
        for metric in available_metrics:
            baseline_column = f"{metric}_{baseline_model}"
            challenger_column = f"{metric}_{challenger_model}"
            if baseline_column in wide.columns and challenger_column in wide.columns:
                part[f"{metric}_baseline"] = wide[baseline_column]
                part[f"{metric}_challenger"] = wide[challenger_column]
                part[f"{metric}_delta_challenger_minus_baseline"] = wide[challenger_column] - wide[baseline_column]
        for metric in LOWER_IS_BETTER_METRICS:
            delta_column = f"{metric}_delta_challenger_minus_baseline"
            if delta_column in part:
                delta = part[delta_column]
                part[f"{metric}_winner"] = np.select(
                    [delta < 0, delta > 0],
                    ["challenger", "baseline"],
                    default="tie",
                )
        for metric in HIGHER_IS_BETTER_METRICS:
            delta_column = f"{metric}_delta_challenger_minus_baseline"
            if delta_column in part:
                delta = part[delta_column]
                part[f"{metric}_winner"] = np.select(
                    [delta > 0, delta < 0],
                    ["challenger", "baseline"],
                    default="tie",
                )
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_winner_summary(model_effects: pd.DataFrame) -> pd.DataFrame:
    if model_effects.empty or "MAPE_winner" not in model_effects.columns:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_columns = [
        column
        for column in [
            "xlstm_backbone",
            "baseline_sequence_backbone",
            "challenger_sequence_backbone",
            "effect",
        ]
        if column in model_effects.columns
    ]
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for group_key, frame in model_effects.groupby(grouper, sort=True, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        winner_counts = frame["MAPE_winner"].value_counts()
        stock_count = int(frame["stock_id"].nunique())
        row = dict(zip(group_columns, group_key, strict=True))
        row.update(
            {
                "baseline_model": frame["baseline_model"].iloc[0],
                "challenger_model": frame["challenger_model"].iloc[0],
                "stock_count": stock_count,
                "challenger_wins": int(winner_counts.get("challenger", 0)),
                "baseline_wins": int(winner_counts.get("baseline", 0)),
                "ties": int(winner_counts.get("tie", 0)),
                "challenger_win_rate": (
                    float(winner_counts.get("challenger", 0) / len(frame) * 100)
                    if len(frame)
                    else np.nan
                ),
            }
        )
        for winner_metric in ["WMAPE", "MAE", "DirectionAccuracy"]:
            winner_column = f"{winner_metric}_winner"
            if winner_column in frame.columns:
                metric_winner_counts = frame[winner_column].value_counts()
                row[f"{winner_metric}_challenger_wins"] = int(metric_winner_counts.get("challenger", 0))
                row[f"{winner_metric}_baseline_wins"] = int(metric_winner_counts.get("baseline", 0))
                row[f"{winner_metric}_ties"] = int(metric_winner_counts.get("tie", 0))
                row[f"{winner_metric}_challenger_win_rate"] = (
                    float(metric_winner_counts.get("challenger", 0) / len(frame) * 100) if len(frame) else np.nan
                )
        for metric in [
            "MAPE",
            "WMAPE",
            "MAE",
            "MedianAPE",
            "SMAPE",
            "Bias",
            "UnderestimateRate",
            "DirectionAccuracy",
        ]:
            delta_column = f"{metric}_delta_challenger_minus_baseline"
            if delta_column in frame.columns:
                row[f"average_{metric}_delta"] = float(frame[delta_column].mean())
                row[f"median_{metric}_delta"] = float(frame[delta_column].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stock-limit", type=int, default=None)
    parser.add_argument("--min-2025-months", type=int, default=DEFAULT_MIN_2025_MONTHS)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max-train-samples", type=int, default=DEFAULT_MAX_TRAIN_SAMPLES)
    parser.add_argument(
        "--xlstm-backbone",
        choices=engine.XLSTM_BACKBONES,
        default=engine.DEFAULT_STREAMLIT_XLSTM_BACKBONE,
        help="Use xlstm_hybrid for the current UI architecture or xlstm for historical D1 reproduction.",
    )
    parser.add_argument("--under-weight", type=float, default=DEFAULT_UNDER_WEIGHT)
    parser.add_argument("--growth-alpha", type=float, default=DEFAULT_GROWTH_ALPHA)
    parser.add_argument("--xlstm-growth-alpha", type=float, default=DEFAULT_XLSTM_ALPHA)
    parser.add_argument("--xlstm-decline-cap-growth-ratio-max", type=float, default=DEFAULT_XLSTM_DECLINE_CAP_GROWTH_RATIO_MAX)
    parser.add_argument(
        "--xlstm-decline-cap-prediction-ratio-min",
        type=float,
        default=DEFAULT_XLSTM_DECLINE_CAP_PREDICTION_RATIO_MIN,
    )
    parser.add_argument("--disable-growth-adjustment", action="store_true")
    parser.add_argument("--disable-asymmetric-loss", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()

    revenue = engine.load_revenue_data()
    stock_meta_frame = basket.load_stock_metadata(revenue)
    stock_ids = basket.resolve_stock_ids(
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

    spec = MainFlowRunSpec(
        xlstm_backbone=str(args.xlstm_backbone),
        k=int(args.k),
        epochs=int(args.epochs),
        max_train_samples=int(args.max_train_samples),
        use_asymmetric_loss=not bool(args.disable_asymmetric_loss),
        under_weight=float(args.under_weight),
        growth_enabled=not bool(args.disable_growth_adjustment),
        growth_alpha=float(args.growth_alpha),
        xlstm_growth_alpha=float(args.xlstm_growth_alpha),
        xlstm_decline_cap_growth_ratio_max=float(args.xlstm_decline_cap_growth_ratio_max),
        xlstm_decline_cap_prediction_ratio_min=float(args.xlstm_decline_cap_prediction_ratio_min),
    )
    config = build_run_config(spec)
    xlstm_spec = engine.get_xlstm_backbone_spec(config.xlstm_backbone)

    metric_rows: list[pd.DataFrame] = []
    monthly_rows: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []

    print(
        f"Running xLSTM main-flow comparison: stocks={stock_ids}, "
        f"architecture={xlstm_spec.display_name} ({xlstm_spec.key})",
        flush=True,
    )
    print("Warming shared revenue, KMeans, training-sample, and eval-sample caches.", flush=True)
    basket.warm_shared_caches(int(args.k), stock_ids)

    for position, stock_id in enumerate(stock_ids, start=1):
        stock_started = time.time()
        try:
            result = engine.run_rolling_lstm_experiment(
                selected_stock=int(stock_id),
                config=config,
            )
            runtime_seconds = time.time() - stock_started
            stock_name = str(stock_meta.at[stock_id, "stock_name"]) if stock_id in stock_meta.index else ""
            industry = str(stock_meta.at[stock_id, "industry_category"]) if stock_id in stock_meta.index else "unknown"
            metrics = result.metrics[result.metrics["model"].isin(MODEL_COLUMNS)].copy()
            metrics.insert(0, "stock_id", int(stock_id))
            metrics.insert(1, "stock_name", stock_name)
            metrics.insert(2, "industry_category", industry)
            metrics["k"] = int(args.k)
            metrics["epochs"] = int(args.epochs)
            metrics["max_train_samples"] = int(args.max_train_samples)
            metrics["growth_alpha"] = float(args.growth_alpha)
            metrics["xlstm_growth_alpha"] = float(args.xlstm_growth_alpha)
            metrics["xlstm_decline_cap_growth_ratio_max"] = float(args.xlstm_decline_cap_growth_ratio_max)
            metrics["xlstm_decline_cap_prediction_ratio_min"] = float(
                args.xlstm_decline_cap_prediction_ratio_min
            )
            metrics["xlstm_backbone"] = xlstm_spec.key
            metrics["sequence_backend_note"] = extract_sequence_note(result.notes)
            metric_rows.append(add_runtime_columns(metrics, runtime_seconds))
            monthly_rows.append(
                build_monthly_long_frame(
                    result.forecast,
                    stock_id=int(stock_id),
                    stock_name=stock_name,
                    industry_category=industry,
                    xlstm_backbone=xlstm_spec.key,
                )
            )
        except Exception as error:
            failures.append(
                {
                    "xlstm_backbone": xlstm_spec.key,
                    "stock_id": int(stock_id),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        print(
            f"{position}/{len(stock_ids)} stocks processed, "
            f"failures={len(failures)}, elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    # The model predictions are trained and cached for the full evaluation universe.
    # Clearing Python and CUDA caches per stock only repeats expensive collection work.
    clear_torch_cache()

    stock_accuracy = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    monthly_predictions = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()
    overall_accuracy, industry_accuracy, regime_accuracy = build_accuracy_summaries(
        monthly_predictions
    )
    model_effects = build_model_effects(stock_accuracy)
    winner_summary = build_winner_summary(model_effects)
    failure_frame = pd.DataFrame(
        failures,
        columns=["xlstm_backbone", "stock_id", "error_type", "error"],
    )

    write_csv(stock_accuracy, output_dir / "stock_accuracy.csv")
    write_csv(overall_accuracy, output_dir / "overall_accuracy.csv")
    write_csv(industry_accuracy, output_dir / "industry_accuracy.csv")
    write_csv(regime_accuracy, output_dir / "regime_accuracy.csv")
    write_csv(model_effects, output_dir / "model_effects.csv")
    write_csv(winner_summary, output_dir / "winner_summary.csv")
    write_csv(monthly_predictions, output_dir / "monthly_predictions.csv")
    write_csv(failure_frame, output_dir / "failed_runs.csv")

    completed = time.time()
    run_summary = {
        "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(completed)),
        "duration_sec": round(completed - started, 3),
        "stocks": list(stock_ids),
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
            "include_xlstm_plain": True,
            "xlstm_backbone": xlstm_spec.key,
            "xlstm_architecture": xlstm_spec.display_name,
            "use_asymmetric_loss": not bool(args.disable_asymmetric_loss),
            "under_weight": float(args.under_weight),
            "growth_enabled": not bool(args.disable_growth_adjustment),
            "growth_alpha": float(args.growth_alpha),
            "xlstm_growth_alpha": float(args.xlstm_growth_alpha),
            "xlstm_decline_cap_growth_ratio_max": float(args.xlstm_decline_cap_growth_ratio_max),
            "xlstm_decline_cap_prediction_ratio_min": float(args.xlstm_decline_cap_prediction_ratio_min),
            "xlstm_adjusted_default": (
                "balanced decline cap with no growth boost when xlstm_growth_alpha is 0.0 and regime strategy is enabled"
            ),
        },
        "row_counts": {
            "stock_accuracy": int(len(stock_accuracy)),
            "overall_accuracy": int(len(overall_accuracy)),
            "industry_accuracy": int(len(industry_accuracy)),
            "regime_accuracy": int(len(regime_accuracy)),
            "model_effects": int(len(model_effects)),
            "winner_summary": int(len(winner_summary)),
            "monthly_predictions": int(len(monthly_predictions)),
            "failed_runs": int(len(failure_frame)),
        },
        "metric_notes": {
            "revenue_unit": "thousand TWD (same as revenue_thousand source)",
            "MAPE": "mean absolute percentage error across nonzero actual revenue rows",
            "MedianAPE": "median absolute percentage error; less sensitive to low-denominator outliers than MAPE",
            "WMAPE": "sum absolute error divided by sum absolute actual revenue",
            "SMAPE": "symmetric MAPE: mean 2*abs(error)/(abs(actual)+abs(predicted))",
            "effect_delta": "challenger minus baseline; negative error-metric deltas mean challenger improved error",
            "actual_usage": "2025 actual revenue is used only after predictions are produced, for evaluation.",
        },
    }
    run_summary = write_rolling_run_config(
        output_dir,
        run_summary,
        experiment_family="rolling_main_flow",
        evidence_tier="C",
        selection_protocol="target-year-hindsight",
        report_ready=False,
        report_ready_reason=(
            "The D1.16 balanced cap default was selected after D1.15 replayed target-year outcomes."
        ),
    )

    print(json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
