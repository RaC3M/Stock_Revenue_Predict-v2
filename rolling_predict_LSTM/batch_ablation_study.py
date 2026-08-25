from __future__ import annotations

"""Run fixed-parameter Rolling LSTM ablation studies.

The runner trains each ablation with the same K, epoch count, and training
sample cap, then attaches 2025 actual revenue only for evaluation reports.
"""

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from . import rolling_lstm_engine as engine
    from .experiment_metrics import metric_record, summarize
    from .experiment_metadata import write_rolling_run_config
except ImportError:
    import rolling_lstm_engine as engine
    from experiment_metrics import metric_record, summarize
    from experiment_metadata import write_rolling_run_config


WINDOW_SIZE = 12
FIXED_K = 6
FIXED_EPOCHS = 35
FIXED_MAX_TRAIN_SAMPLES = 40_000
UNDER_WEIGHT = 2.0
GROWTH_ALPHA = 0.8
TREND_SLOPE_BETA = 0.35
MAX_VOLATILITY_SCALE = 2.5

TrendCycleMode = Literal["off", "cycle", "decline", "all"]


@dataclass(frozen=True)
class AblationSpec:
    experiment_id: str
    experiment_name: str
    ablation_group: Literal["method", "feature"]
    baseline_id: str
    include_cluster: bool = True
    numeric_features: tuple[str, ...] = tuple(engine.NUMERIC_SEQUENCE_FEATURES)
    use_asymmetric_loss: bool = True
    under_weight: float = UNDER_WEIGHT
    growth_enabled: bool = False
    growth_conditional: bool = True
    growth_regime_strategy: bool = True
    trend_cycle_mode: TrendCycleMode = "off"


DEFAULT_ABLATION_SPECS: tuple[AblationSpec, ...] = (
    AblationSpec(
        experiment_id="M00",
        experiment_name="cluster asymmetric raw",
        ablation_group="method",
        baseline_id="M00",
    ),
    AblationSpec(
        experiment_id="M01",
        experiment_name="no cluster asymmetric raw",
        ablation_group="method",
        baseline_id="M00",
        include_cluster=False,
    ),
    AblationSpec(
        experiment_id="M02",
        experiment_name="cluster huber raw",
        ablation_group="method",
        baseline_id="M00",
        use_asymmetric_loss=False,
        under_weight=1.0,
    ),
    AblationSpec(
        experiment_id="M03",
        experiment_name="cluster asymmetric + growth",
        ablation_group="method",
        baseline_id="M00",
        growth_enabled=True,
    ),
    AblationSpec(
        experiment_id="M05",
        experiment_name="growth without regime gate",
        ablation_group="method",
        baseline_id="M03",
        growth_enabled=True,
        growth_regime_strategy=False,
    ),
    AblationSpec(
        experiment_id="M07",
        experiment_name="growth + trend cycle on cycle",
        ablation_group="method",
        baseline_id="M03",
        growth_enabled=True,
        trend_cycle_mode="cycle",
    ),
    AblationSpec(
        experiment_id="M08",
        experiment_name="growth + trend cycle on decline",
        ablation_group="method",
        baseline_id="M03",
        growth_enabled=True,
        trend_cycle_mode="decline",
    ),
    AblationSpec(
        experiment_id="M09",
        experiment_name="growth + trend cycle on all regimes",
        ablation_group="method",
        baseline_id="M03",
        growth_enabled=True,
        trend_cycle_mode="all",
    ),
    AblationSpec(
        experiment_id="F00",
        experiment_name="all numeric features",
        ablation_group="feature",
        baseline_id="F00",
    ),
    AblationSpec(
        experiment_id="F01",
        experiment_name="remove log_revenue",
        ablation_group="feature",
        baseline_id="F00",
        numeric_features=tuple(feature for feature in engine.NUMERIC_SEQUENCE_FEATURES if feature != "log_revenue"),
    ),
    AblationSpec(
        experiment_id="F02",
        experiment_name="remove growth_rate",
        ablation_group="feature",
        baseline_id="F00",
        numeric_features=tuple(feature for feature in engine.NUMERIC_SEQUENCE_FEATURES if feature != "growth_rate"),
    ),
    AblationSpec(
        experiment_id="F03",
        experiment_name="remove momentum_3m",
        ablation_group="feature",
        baseline_id="F00",
        numeric_features=tuple(feature for feature in engine.NUMERIC_SEQUENCE_FEATURES if feature != "momentum_3m"),
    ),
    AblationSpec(
        experiment_id="F04",
        experiment_name="remove momentum_6m",
        ablation_group="feature",
        baseline_id="F00",
        numeric_features=tuple(feature for feature in engine.NUMERIC_SEQUENCE_FEATURES if feature != "momentum_6m"),
    ),
    AblationSpec(
        experiment_id="F05",
        experiment_name="remove cluster one-hot",
        ablation_group="feature",
        baseline_id="F00",
        include_cluster=False,
    ),
)


@dataclass(frozen=True)
class AblationContext:
    revenue: pd.DataFrame
    monthly: pd.DataFrame
    train_samples: list[dict[str, object]]
    eval_samples: list[dict[str, object]]
    actual_revenue: pd.DataFrame
    cluster_count: int
    stock_meta: pd.DataFrame
    revenue_mtime_ns: int


def parse_stock_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def clear_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def add_stock_metadata(monthly_predictions: pd.DataFrame, stock_meta: pd.DataFrame) -> pd.DataFrame:
    return monthly_predictions.merge(stock_meta, on="stock_id", how="left")


def build_stock_meta(revenue: pd.DataFrame) -> pd.DataFrame:
    target_year_revenue = revenue[revenue["revenue_year"].astype(int).eq(engine.FORECAST_YEAR)]
    actual_2025 = (
        target_year_revenue.groupby("stock_id", as_index=False)["revenue_thousand"]
        .sum()
        .rename(columns={"revenue_thousand": "actual_2025_revenue"})
    )
    stock_meta = (
        revenue.sort_values("date")
        .groupby("stock_id", as_index=False)
        .agg(
            industry_category=("industry_category", "last"),
        )
    )
    stock_meta = stock_meta.merge(actual_2025, on="stock_id", how="left")
    stock_meta["actual_2025_revenue"] = stock_meta["actual_2025_revenue"].fillna(0.0)
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


def select_stock_ids(
    revenue: pd.DataFrame,
    requested_stock_ids: set[int] | None,
    stock_limit: int | None,
) -> list[int]:
    available = sorted(
        revenue.loc[revenue["revenue_year"].eq(engine.FORECAST_YEAR), "stock_id"].dropna().astype(int).unique()
    )
    if requested_stock_ids is not None:
        available = [stock_id for stock_id in available if stock_id in requested_stock_ids]
    if stock_limit:
        available = available[: int(stock_limit)]
    return available


def prepare_context(
    *,
    k: int,
    window_size: int,
    max_train_samples: int,
    requested_stock_ids: set[int] | None,
    stock_limit: int | None,
) -> AblationContext:
    revenue_mtime_ns = engine._revenue_file_mtime_ns()
    revenue, _ = engine._cached_revenue_and_windows(window_size, revenue_mtime_ns)
    _, clustered_windows, _, monthly = engine._cached_clustered_artifacts(k, window_size, revenue_mtime_ns)
    train_samples = list(engine._cached_train_samples(k, window_size, revenue_mtime_ns))
    selected_stock_ids = select_stock_ids(revenue, requested_stock_ids, stock_limit)
    if not selected_stock_ids:
        raise ValueError("No 2025 stocks are available for ablation evaluation.")

    eval_samples: list[dict[str, object]] = []
    for stock_id in selected_stock_ids:
        eval_samples.extend(engine.build_eval_sequences_for_stock(monthly, stock_id, window_size=window_size))
    if not eval_samples:
        raise ValueError("No 2025 rolling evaluation samples are available for selected stocks.")

    capped_train_samples, _ = engine.cap_training_samples(train_samples, max_train_samples=max_train_samples, seed=42)
    return AblationContext(
        revenue=revenue,
        monthly=monthly,
        train_samples=capped_train_samples,
        eval_samples=eval_samples,
        actual_revenue=engine.build_actual_revenue_frame(revenue, target_year=engine.FORECAST_YEAR),
        cluster_count=int(clustered_windows["cluster"].max()) + 1,
        stock_meta=build_stock_meta(revenue),
        revenue_mtime_ns=revenue_mtime_ns,
    )


def build_experiment_specs(groups: set[str]) -> tuple[AblationSpec, ...]:
    return tuple(spec for spec in DEFAULT_ABLATION_SPECS if spec.ablation_group in groups)


def train_raw_prediction(
    context: AblationContext,
    spec: AblationSpec,
    *,
    epochs: int,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    numeric_scaler = StandardScaler()
    target_scaler = StandardScaler()
    x_train, y_train, _ = engine.make_lstm_arrays(
        context.train_samples,
        numeric_scaler,
        target_scaler,
        cluster_count=context.cluster_count,
        include_cluster=spec.include_cluster,
        fit_scalers=True,
        numeric_features=spec.numeric_features,
    )
    x_eval, _, metadata = engine.make_lstm_arrays(
        context.eval_samples,
        numeric_scaler,
        target_scaler,
        cluster_count=context.cluster_count,
        include_cluster=spec.include_cluster,
        fit_scalers=False,
        numeric_features=spec.numeric_features,
        require_target=False,
    )
    prediction, backend = engine.train_predict_lstm(
        x_train,
        y_train,
        x_eval,
        target_scaler,
        epochs=epochs,
        use_asymmetric_loss=spec.use_asymmetric_loss,
        under_weight=spec.under_weight,
    )
    return prediction, metadata, backend


def apply_trend_cycle_mode(
    base_prediction: np.ndarray,
    metadata: pd.DataFrame,
    context: AblationContext,
    spec: AblationSpec,
    *,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    regime = engine.classify_regime(metadata)
    default_length = len(metadata)
    if spec.trend_cycle_mode == "off":
        return (
            base_prediction,
            np.full(default_length, np.nan, dtype=float),
            np.zeros(default_length, dtype=float),
            np.ones(default_length, dtype=float),
            np.zeros(default_length, dtype=bool),
            "disabled",
            "disabled",
        )

    trend_pred, cycle_pred, trend_backend, cycle_backend = engine.train_predict_trend_cycle_components(
        context.train_samples,
        context.eval_samples,
        cluster_count=context.cluster_count,
        epochs=epochs,
    )
    cycle_std = float(np.nanstd(cycle_pred))
    actual_std = metadata["cycle_volatility_6m"].to_numpy(dtype=float)
    volatility_scale = np.divide(
        actual_std,
        cycle_std + 1e-6,
        out=np.ones_like(actual_std, dtype=float),
        where=np.isfinite(actual_std),
    )
    volatility_scale = np.clip(volatility_scale, 1.0 / MAX_VOLATILITY_SCALE, MAX_VOLATILITY_SCALE)
    if not np.isfinite(cycle_std) or cycle_std <= 1e-6:
        volatility_scale = np.ones_like(actual_std, dtype=float)

    adjusted_cycle = cycle_pred * volatility_scale
    trend_slope_rate = metadata["trend_slope_rate"].to_numpy(dtype=float)
    trend_boost = np.where(
        trend_slope_rate > 0,
        1.0 + TREND_SLOPE_BETA * trend_slope_rate,
        1.0,
    )
    trend_boost = np.clip(trend_boost, 1.0, 1.35)
    trend_cycle_prediction = np.clip((trend_pred * trend_boost) + adjusted_cycle, 0, None)
    if spec.trend_cycle_mode == "cycle":
        apply_mask = regime == "cycle"
    elif spec.trend_cycle_mode == "decline":
        apply_mask = regime == "decline"
    else:
        apply_mask = np.ones(default_length, dtype=bool)
    final_prediction = np.where(apply_mask, trend_cycle_prediction, base_prediction)
    return final_prediction, trend_pred, adjusted_cycle, trend_boost, apply_mask, trend_backend, cycle_backend


def finalize_prediction(
    prediction: np.ndarray,
    metadata: pd.DataFrame,
    context: AblationContext,
    spec: AblationSpec,
    *,
    trend_component: np.ndarray,
    cycle_component: np.ndarray,
    trend_boost: np.ndarray,
    trend_cycle_applied: np.ndarray,
    backend: str,
    trend_backend: str,
    cycle_backend: str,
    clipped_base_count: int,
    clipped_final_count: int,
) -> pd.DataFrame:
    forecast = metadata.copy()
    final_prediction = engine.safe_round_revenue(prediction)
    forecast["predicted_revenue"] = final_prediction
    evaluated = engine.attach_actual_revenue(forecast, context.actual_revenue)
    evaluated["error"] = evaluated["predicted_revenue"] - evaluated["actual_revenue"]
    evaluated["abs_error"] = evaluated["error"].abs()
    evaluated["predicted_return"] = evaluated["predicted_revenue"] / evaluated["last_observed_revenue"] - 1
    evaluated["actual_return"] = evaluated["actual_revenue"] / evaluated["last_observed_revenue"] - 1
    evaluated["underestimated"] = evaluated["predicted_revenue"] < evaluated["actual_revenue"]
    evaluated["direction_correct"] = np.sign(
        evaluated["predicted_revenue"] - evaluated["last_observed_revenue"]
    ) == np.sign(evaluated["actual_revenue"] - evaluated["last_observed_revenue"])
    evaluated["regime"] = engine.classify_regime(evaluated)
    evaluated["growth_phase"] = engine.calculate_growth_phase(evaluated)
    evaluated["high_growth_flag"] = engine.calculate_high_growth_flag(evaluated)
    evaluated["trend_component_prediction"] = np.round(trend_component, 2)
    evaluated["cycle_component_prediction"] = np.round(cycle_component, 2)
    evaluated["trend_boost"] = np.round(trend_boost, 4)
    evaluated["trend_cycle_applied"] = trend_cycle_applied
    evaluated["experiment_id"] = spec.experiment_id
    evaluated["experiment_name"] = spec.experiment_name
    evaluated["ablation_group"] = spec.ablation_group
    evaluated["baseline_id"] = spec.baseline_id
    evaluated["include_cluster"] = spec.include_cluster
    evaluated["numeric_features"] = ",".join(spec.numeric_features)
    evaluated["use_asymmetric_loss"] = spec.use_asymmetric_loss
    evaluated["growth_enabled"] = spec.growth_enabled
    evaluated["growth_conditional"] = spec.growth_conditional
    evaluated["growth_regime_strategy"] = spec.growth_regime_strategy
    evaluated["trend_cycle_mode"] = spec.trend_cycle_mode
    evaluated["backend"] = backend
    evaluated["trend_backend"] = trend_backend
    evaluated["cycle_backend"] = cycle_backend
    evaluated["clipped_base_count"] = clipped_base_count
    evaluated["clipped_final_count"] = clipped_final_count
    return evaluated


def run_ablation_spec(context: AblationContext, spec: AblationSpec, *, epochs: int) -> pd.DataFrame:
    started = time.perf_counter()
    raw_prediction, metadata, backend = train_raw_prediction(context, spec, epochs=epochs)
    guarded_prediction, clipped_base_count, _ = engine.apply_revenue_guardrails(
        raw_prediction,
        metadata,
    )
    trend_prediction, trend_component, cycle_component, trend_boost, trend_applied, trend_backend, cycle_backend = (
        apply_trend_cycle_mode(
            guarded_prediction,
            metadata,
            context,
            spec,
            epochs=epochs,
        )
    )
    trend_prediction, clipped_trend_count, _ = engine.apply_revenue_guardrails(
        trend_prediction,
        metadata,
    )
    if spec.growth_enabled:
        adjusted, _, _, _, _, _ = engine.apply_growth_adjustment(
            trend_prediction,
            metadata,
            alpha=GROWTH_ALPHA,
            enable_growth_adjustment=True,
            enable_conditional_adjustment=spec.growth_conditional,
            enable_regime_strategy=spec.growth_regime_strategy,
        )
    else:
        adjusted = trend_prediction
    adjusted, clipped_adjusted_count, _ = engine.apply_revenue_guardrails(
        adjusted,
        metadata,
    )
    result = finalize_prediction(
        adjusted,
        metadata,
        context,
        spec,
        trend_component=trend_component,
        cycle_component=cycle_component,
        trend_boost=trend_boost,
        trend_cycle_applied=trend_applied,
        backend=backend,
        trend_backend=trend_backend,
        cycle_backend=cycle_backend,
        clipped_base_count=clipped_base_count,
        clipped_final_count=clipped_trend_count + clipped_adjusted_count,
    )
    result["duration_sec"] = round(time.perf_counter() - started, 3)
    return result


def build_stock_type(monthly_predictions: pd.DataFrame) -> pd.DataFrame:
    baseline = monthly_predictions[monthly_predictions["experiment_id"].eq("M00")]
    if baseline.empty:
        baseline = monthly_predictions[monthly_predictions["experiment_id"].eq("F00")]
    stock_type = (
        baseline.sort_values(["stock_id", "target_date"])
        .groupby("stock_id")
        .agg(
            evaluated_months=("target_date", "size"),
            cycle_months=("regime", lambda values: int((values == "cycle").sum())),
            growth_months=("regime", lambda values: int((values == "growth").sum())),
            decline_months=("regime", lambda values: int((values == "decline").sum())),
            actual_2025_revenue=("actual_revenue", "sum"),
        )
        .reset_index()
    )
    for regime in ("cycle", "growth", "decline"):
        stock_type[f"{regime}_share"] = stock_type[f"{regime}_months"] / stock_type["evaluated_months"]
    share_columns = ["cycle_share", "growth_share", "decline_share"]
    stock_type["dominant_regime"] = stock_type[share_columns].idxmax(axis=1).str.replace("_share", "", regex=False)
    stock_type["regime_confidence"] = stock_type[share_columns].max(axis=1)
    if len(stock_type) >= 4:
        stock_type["revenue_size_quartile"] = pd.qcut(
            stock_type["actual_2025_revenue"].rank(method="first"),
            4,
            labels=["Q1_small", "Q2", "Q3", "Q4_large"],
        )
    else:
        stock_type["revenue_size_quartile"] = "all"
    return stock_type


def build_effects(summary: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metrics = [
        "RMSE",
        "MAE",
        "MAPE",
        "MedianAPE",
        "WMAPE",
        "SMAPE",
        "Bias",
        "UnderestimateRate",
        "DirectionAccuracy",
    ]
    rows: list[dict[str, object]] = []
    index_columns = ["ablation_group", *group_columns]
    indexed = summary.set_index(["experiment_id", *index_columns])
    config = summary[["experiment_id", "experiment_name", "baseline_id", "ablation_group"]].drop_duplicates()
    for row in config.itertuples(index=False):
        if row.experiment_id == row.baseline_id:
            continue
        treatment = summary[summary["experiment_id"].eq(row.experiment_id)]
        for treatment_row in treatment.itertuples(index=False):
            key = tuple(getattr(treatment_row, column) for column in index_columns)
            baseline_key = (row.baseline_id, *key)
            if baseline_key not in indexed.index:
                continue
            baseline = indexed.loc[baseline_key]
            output = {
                "experiment_id": row.experiment_id,
                "experiment_name": row.experiment_name,
                "baseline_id": row.baseline_id,
                "ablation_group": row.ablation_group,
            }
            output.update(dict(zip(group_columns, key[1:])))
            for metric in metrics:
                output[f"{metric}_base"] = float(baseline[metric])
                output[f"{metric}_treatment"] = float(getattr(treatment_row, metric))
                output[f"{metric}_delta"] = output[f"{metric}_treatment"] - output[f"{metric}_base"]
            output["MAE_pct_change"] = (
                (output["MAE_treatment"] / output["MAE_base"] - 1) * 100
                if output["MAE_base"]
                else np.nan
            )
            output["WMAPE_pct_change"] = (
                (output["WMAPE_treatment"] / output["WMAPE_base"] - 1) * 100
                if output["WMAPE_base"]
                else np.nan
            )
            rows.append(output)
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--groups", default="method,feature", help="Comma-separated: method,feature")
    parser.add_argument("--k", type=int, default=FIXED_K)
    parser.add_argument("--epochs", type=int, default=FIXED_EPOCHS)
    parser.add_argument("--max-train-samples", type=int, default=FIXED_MAX_TRAIN_SAMPLES)
    parser.add_argument("--stock-ids", default="")
    parser.add_argument("--stock-limit", type=int, default=0)
    parser.add_argument("--skip-monthly", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    groups = {group.strip() for group in args.groups.split(",") if group.strip()}
    unknown_groups = groups - {"method", "feature"}
    if unknown_groups:
        raise ValueError(f"Unknown ablation groups: {sorted(unknown_groups)}")

    specs = build_experiment_specs(groups)
    if not specs:
        raise ValueError("No ablation specs selected. Use --groups method,feature or one of them.")
    context = prepare_context(
        k=int(args.k),
        window_size=WINDOW_SIZE,
        max_train_samples=int(args.max_train_samples),
        requested_stock_ids=parse_stock_ids(args.stock_ids),
        stock_limit=int(args.stock_limit) or None,
    )

    monthly_parts: list[pd.DataFrame] = []
    for position, spec in enumerate(specs, start=1):
        print(f"[{position}/{len(specs)}] Running {spec.experiment_id} {spec.experiment_name}", flush=True)
        monthly_parts.append(run_ablation_spec(context, spec, epochs=int(args.epochs)))
        clear_torch_cache()

    monthly = pd.concat(monthly_parts, ignore_index=True)
    monthly = add_stock_metadata(monthly, context.stock_meta)
    stock_type = build_stock_type(monthly)
    monthly = monthly.merge(
        stock_type[["stock_id", "dominant_regime", "regime_confidence", "revenue_size_quartile"]],
        on="stock_id",
        how="left",
    )

    overall = summarize(monthly, ["ablation_group", "experiment_id", "experiment_name", "baseline_id"])
    stock_accuracy = summarize(
        monthly,
        ["ablation_group", "experiment_id", "experiment_name", "baseline_id", "stock_id", "stock_name", "industry_category"],
    )
    regime_accuracy = summarize(
        monthly,
        ["ablation_group", "experiment_id", "experiment_name", "baseline_id", "regime"],
    )
    dominant_regime_accuracy = summarize(
        monthly,
        ["ablation_group", "experiment_id", "experiment_name", "baseline_id", "dominant_regime"],
    )
    size_accuracy = summarize(
        monthly,
        ["ablation_group", "experiment_id", "experiment_name", "baseline_id", "revenue_size_quartile"],
    )
    industry_accuracy = summarize(
        monthly,
        ["ablation_group", "experiment_id", "experiment_name", "baseline_id", "industry_category"],
    )
    overall_effects = build_effects(overall, [])
    dominant_regime_effects = build_effects(dominant_regime_accuracy, ["dominant_regime"])
    stock_effects = build_effects(stock_accuracy, ["stock_id", "stock_name", "industry_category"])

    experiment_config = pd.DataFrame([asdict(spec) for spec in specs])
    experiment_config["numeric_features"] = experiment_config["numeric_features"].apply(lambda values: ",".join(values))
    run_config = {
        "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": round(time.time() - started, 3),
        "k": int(args.k),
        "epochs": int(args.epochs),
        "max_train_samples": int(args.max_train_samples),
        "groups": sorted(groups),
        "stock_ids": sorted({int(sample["stock_id"]) for sample in context.eval_samples}),
        "train_samples_used": len(context.train_samples),
        "eval_samples": len(context.eval_samples),
        "method_note": (
            "Ablations keep K, epochs, and sample cap fixed. "
            "2025 actual revenue is attached only after predictions for metric calculation."
        ),
    }

    write_csv(experiment_config, output_dir / "ablation_config.csv")
    write_csv(overall, output_dir / "overall_accuracy.csv")
    write_csv(overall_effects, output_dir / "overall_effects.csv")
    write_csv(stock_accuracy, output_dir / "stock_accuracy.csv")
    write_csv(stock_effects, output_dir / "stock_effects.csv")
    write_csv(regime_accuracy, output_dir / "regime_accuracy.csv")
    write_csv(dominant_regime_accuracy, output_dir / "dominant_regime_accuracy.csv")
    write_csv(dominant_regime_effects, output_dir / "dominant_regime_effects.csv")
    write_csv(size_accuracy, output_dir / "revenue_size_accuracy.csv")
    write_csv(industry_accuracy, output_dir / "industry_accuracy.csv")
    write_csv(stock_type, output_dir / "stock_type.csv")
    if not args.skip_monthly:
        write_csv(monthly, output_dir / "monthly_predictions.csv")
    run_config = write_rolling_run_config(
        output_dir,
        run_config,
        experiment_family="rolling_ablation",
        evidence_tier="C",
        selection_protocol="target-year-hindsight",
        report_ready=False,
        report_ready_reason="Ablation winners are ranked on target-year results and are exploratory.",
    )

    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
