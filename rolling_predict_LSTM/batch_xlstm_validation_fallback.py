from __future__ import annotations

"""Select xLSTM plain vs balanced adjusted with prior-year validation.

This runner trains one no-cluster xLSTM validation model on samples whose
targets end at or before 2023, validates xLSTM plain vs balanced adjusted on
2024 targets, then applies the source-model choice to a saved 2025 main-flow
monthly_predictions.csv file. The 2025 actual revenue values are used only for
final evaluation, not for selecting the fallback source model.

D1.18 also supports a stock-regime selection scope so a stock can use the
adjusted source only in regimes where prior-year validation supports it.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import batch_sequence_backbone_ablation as basket
    from . import batch_xlstm_main_flow_comparison as main_flow
    from . import rolling_lstm_engine as engine
    from .experiment_metadata import write_rolling_run_config
except ImportError:
    import batch_sequence_backbone_ablation as basket
    import batch_xlstm_main_flow_comparison as main_flow
    import rolling_lstm_engine as engine
    from experiment_metadata import write_rolling_run_config


VALIDATION_MODEL_PLAIN = "Rolling xLSTM"
VALIDATION_MODEL_ADJUSTED = "Rolling xLSTM + Conditional Adjustment"
FALLBACK_MODEL = "Rolling xLSTM Validation Fallback"
SELECTION_SCOPE_STOCK = "stock"
SELECTION_SCOPE_STOCK_REGIME = "stock-regime"
SELECTION_SCOPES = {SELECTION_SCOPE_STOCK, SELECTION_SCOPE_STOCK_REGIME}
DEFAULT_SELECTION_METRIC = "WMAPE"
DEFAULT_VALIDATION_YEAR = engine.TRAIN_END_YEAR
DEFAULT_VALIDATION_TRAIN_END_YEAR = DEFAULT_VALIDATION_YEAR - 1
DEFAULT_K = 6
DEFAULT_EPOCHS = 35
DEFAULT_MAX_TRAIN_SAMPLES = 40_000
DEFAULT_UNDER_WEIGHT = engine.DEFAULT_UNDER_WEIGHT
LOWER_IS_BETTER_METRICS = {"MSE", "RMSE", "MAE", "MAPE", "MedianAPE", "WMAPE", "SMAPE"}

FALLBACK_EFFECT_PAIRS = (
    (
        "fallback_minus_cluster_adjusted",
        "Rolling LSTM + Cluster + Conditional Adjustment",
        FALLBACK_MODEL,
    ),
    (
        "fallback_minus_xlstm_adjusted",
        VALIDATION_MODEL_ADJUSTED,
        FALLBACK_MODEL,
    ),
    (
        "fallback_minus_xlstm_plain",
        VALIDATION_MODEL_PLAIN,
        FALLBACK_MODEL,
    ),
)


@dataclass(frozen=True)
class ValidationFallbackSpec:
    k: int = DEFAULT_K
    window_size: int = engine.DEFAULT_WINDOW_SIZE
    epochs: int = DEFAULT_EPOCHS
    max_train_samples: int = DEFAULT_MAX_TRAIN_SAMPLES
    validation_year: int = DEFAULT_VALIDATION_YEAR
    validation_train_end_year: int = DEFAULT_VALIDATION_TRAIN_END_YEAR
    selection_metric: str = DEFAULT_SELECTION_METRIC
    selection_scope: str = SELECTION_SCOPE_STOCK
    min_improvement: float = 0.0
    fallback_default: str = VALIDATION_MODEL_PLAIN
    use_asymmetric_loss: bool = True
    under_weight: float = DEFAULT_UNDER_WEIGHT
    xlstm_growth_alpha: float = engine.DEFAULT_XLSTM_GROWTH_ADJUSTMENT_ALPHA
    xlstm_decline_cap_growth_ratio_max: float | None = engine.DEFAULT_XLSTM_DECLINE_CAP_GROWTH_RATIO_MAX
    xlstm_decline_cap_prediction_ratio_min: float = engine.DEFAULT_XLSTM_DECLINE_CAP_PREDICTION_RATIO_MIN


def _sample_from_window(stock_id: int, stock_df: pd.DataFrame, end_idx: int, window_size: int) -> dict[str, object]:
    target_idx = end_idx + 1
    sequence_frame = stock_df.iloc[end_idx - window_size + 1 : end_idx + 1]
    return {
        "stock_id": int(stock_id),
        "sequence_frame": sequence_frame,
        "cluster": int(stock_df.loc[end_idx, "cluster"]),
        "sequence_start_date": stock_df.loc[end_idx - window_size + 1, "date"],
        "sequence_end_date": stock_df.loc[end_idx, "date"],
        "target_date": stock_df.loc[target_idx, "date"],
        "target_year": int(stock_df.loc[target_idx, "revenue_year"]),
        "target_month": int(stock_df.loc[target_idx, "revenue_month"]),
        "target_revenue": float(stock_df.loc[target_idx, "revenue_thousand"]),
        "target_trend": float(stock_df.loc[target_idx, "trend_component"]),
        "target_cycle": float(stock_df.loc[target_idx, "cycle_component"]),
    }


def build_year_validation_samples(
    monthly: pd.DataFrame,
    stock_ids: tuple[int, ...],
    window_size: int,
    validation_train_end_year: int,
    validation_year: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected_stocks = {int(stock_id) for stock_id in stock_ids}
    train_samples: list[dict[str, object]] = []
    validation_samples: list[dict[str, object]] = []

    for stock_id, stock_df in monthly.groupby("stock_id", sort=False):
        stock_df = stock_df.sort_values("date").reset_index(drop=True)
        if len(stock_df) <= window_size:
            continue
        for end_idx in range(window_size - 1, len(stock_df) - 1):
            target_idx = end_idx + 1
            target_year = int(stock_df.loc[target_idx, "revenue_year"])
            target_revenue = float(stock_df.loc[target_idx, "revenue_thousand"])
            if not np.isfinite(target_revenue) or target_revenue < 0:
                continue
            sample = _sample_from_window(int(stock_id), stock_df, end_idx, window_size)
            if target_year <= int(validation_train_end_year):
                train_samples.append(sample)
            elif target_year == int(validation_year) and int(stock_id) in selected_stocks:
                validation_samples.append(sample)

    return train_samples, validation_samples


def build_validation_predictions(
    stock_ids: tuple[int, ...],
    spec: ValidationFallbackSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.preprocessing import StandardScaler

    revenue_mtime_ns = engine._revenue_file_mtime_ns()
    _, clustered_windows, _, monthly = engine._cached_clustered_artifacts(
        int(spec.k),
        int(spec.window_size),
        revenue_mtime_ns,
    )
    cluster_count = int(clustered_windows["cluster"].max()) + 1
    train_samples, validation_samples = build_year_validation_samples(
        monthly,
        stock_ids=stock_ids,
        window_size=int(spec.window_size),
        validation_train_end_year=int(spec.validation_train_end_year),
        validation_year=int(spec.validation_year),
    )
    if not train_samples:
        raise ValueError("No validation training samples are available.")
    if not validation_samples:
        raise ValueError(f"No validation samples are available for {spec.validation_year}.")

    sampled_train_samples, sample_capped = engine.cap_training_samples(
        train_samples,
        max_train_samples=int(spec.max_train_samples),
        seed=42,
    )
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train, y_train, _ = engine.make_lstm_arrays(
        sampled_train_samples,
        x_scaler,
        y_scaler,
        cluster_count=cluster_count,
        include_cluster=False,
        fit_scalers=True,
    )
    x_validation, _, validation_meta = engine.make_lstm_arrays(
        validation_samples,
        x_scaler,
        y_scaler,
        cluster_count=cluster_count,
        include_cluster=False,
        fit_scalers=False,
        require_target=True,
    )
    pred_plain, backend = engine.train_predict_lstm(
        x_train,
        y_train,
        x_validation,
        y_scaler,
        epochs=int(spec.epochs),
        sequence_backbone="xlstm",
        use_asymmetric_loss=bool(spec.use_asymmetric_loss),
        under_weight=float(spec.under_weight),
    )
    pred_plain, _, _ = engine.apply_revenue_guardrails(pred_plain, validation_meta)
    (
        pred_adjusted,
        growth_signal,
        adjustment_ratio,
        regime,
        is_growth_phase,
        adjustment_applied,
    ) = engine.apply_growth_adjustment(
        pred_plain,
        validation_meta,
        alpha=float(spec.xlstm_growth_alpha),
        enable_growth_adjustment=bool(spec.xlstm_growth_alpha != 0.0),
        enable_conditional_adjustment=True,
        enable_regime_strategy=True,
        decline_cap_growth_ratio_max=spec.xlstm_decline_cap_growth_ratio_max,
        decline_cap_prediction_ratio_min=float(spec.xlstm_decline_cap_prediction_ratio_min),
    )
    pred_adjusted, _, _ = engine.apply_revenue_guardrails(pred_adjusted, validation_meta)
    validation_meta = validation_meta.copy()
    validation_meta["regime"] = regime
    validation_meta["is_growth_phase"] = is_growth_phase
    validation_meta["growth_signal"] = growth_signal
    validation_meta["xlstm_adjustment_ratio"] = adjustment_ratio
    validation_meta["xlstm_adjustment_applied"] = adjustment_applied
    validation_meta["xlstm_decline_cap_applied"] = engine.calculate_decline_cap_mask(
        pred_plain,
        validation_meta,
        regime,
        enable_regime_strategy=True,
        decline_cap_growth_ratio_max=spec.xlstm_decline_cap_growth_ratio_max,
        decline_cap_prediction_ratio_min=float(spec.xlstm_decline_cap_prediction_ratio_min),
    )

    validation_long = build_validation_long_frame(validation_meta, pred_plain, pred_adjusted)
    validation_long["validation_backend"] = backend
    validation_long["validation_train_samples_used"] = len(sampled_train_samples)
    validation_long["validation_train_samples_available"] = len(train_samples)
    validation_long["validation_sample_capped"] = bool(sample_capped)
    validation_long["validation_year"] = int(spec.validation_year)
    validation_long["validation_train_end_year"] = int(spec.validation_train_end_year)
    return validation_long, monthly


def build_validation_long_frame(
    validation_meta: pd.DataFrame,
    pred_plain: np.ndarray,
    pred_adjusted: np.ndarray,
) -> pd.DataFrame:
    id_columns = [
        "stock_id",
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
        "growth_signal",
        "xlstm_adjustment_ratio",
        "xlstm_adjustment_applied",
        "xlstm_decline_cap_applied",
    ]
    available_id_columns = [column for column in id_columns if column in validation_meta.columns]
    parts: list[pd.DataFrame] = []
    for model, predicted in [
        (VALIDATION_MODEL_PLAIN, engine.safe_round_revenue(pred_plain).astype(float)),
        (VALIDATION_MODEL_ADJUSTED, engine.safe_round_revenue(pred_adjusted).astype(float)),
    ]:
        part = validation_meta[available_id_columns].copy()
        part["model"] = model
        part["predicted_revenue"] = predicted
        part["error"] = part["predicted_revenue"] - part["actual_revenue"]
        part["abs_error"] = part["error"].abs()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def normalize_selection_scope(selection_scope: str) -> str:
    normalized = str(selection_scope).strip().lower().replace("_", "-")
    if normalized not in SELECTION_SCOPES:
        raise ValueError(f"Unsupported selection scope: {selection_scope}")
    return normalized


def selection_group_columns(selection_scope: str) -> list[str]:
    scope = normalize_selection_scope(selection_scope)
    columns = ["stock_id"]
    if scope == SELECTION_SCOPE_STOCK_REGIME:
        columns.append("regime")
    return columns


def validation_selection_index_columns(selection_scope: str, frame: pd.DataFrame) -> list[str]:
    group_columns = selection_group_columns(selection_scope)
    missing_columns = [column for column in group_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Validation accuracy is missing selection columns: {missing_columns}")
    metadata_columns = [column for column in ["stock_name", "industry_category"] if column in frame.columns]
    return [*group_columns, *metadata_columns]


def build_fallback_selection(
    validation_accuracy: pd.DataFrame,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    selection_scope: str = SELECTION_SCOPE_STOCK,
    min_improvement: float = 0.0,
    fallback_default: str = VALIDATION_MODEL_PLAIN,
) -> pd.DataFrame:
    scope = normalize_selection_scope(selection_scope)
    selection_metric = str(selection_metric)
    if selection_metric not in LOWER_IS_BETTER_METRICS:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")
    if selection_metric not in validation_accuracy.columns:
        raise ValueError(f"Validation accuracy is missing selection metric: {selection_metric}")
    if min_improvement < 0:
        raise ValueError("min_improvement must be non-negative.")
    if fallback_default not in {VALIDATION_MODEL_PLAIN, VALIDATION_MODEL_ADJUSTED}:
        raise ValueError(f"Unknown fallback_default: {fallback_default}")
    models_present = set(validation_accuracy["model"].dropna().astype(str).unique())
    if not {VALIDATION_MODEL_PLAIN, VALIDATION_MODEL_ADJUSTED}.issubset(models_present):
        raise ValueError("Validation accuracy must include both xLSTM plain and adjusted rows.")

    index_columns = validation_selection_index_columns(scope, validation_accuracy)
    metric_columns = list(dict.fromkeys([selection_metric, "MAE", "MAPE", "WMAPE", "DirectionAccuracy"]))
    available_metric_columns = [column for column in metric_columns if column in validation_accuracy.columns]
    wide = validation_accuracy.pivot_table(
        index=index_columns,
        columns="model",
        values=available_metric_columns,
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    for model in [VALIDATION_MODEL_PLAIN, VALIDATION_MODEL_ADJUSTED]:
        for metric in available_metric_columns:
            column = f"{metric}_{model}"
            if column not in wide.columns:
                wide[column] = np.nan

    adjusted_metric = f"{selection_metric}_{VALIDATION_MODEL_ADJUSTED}"
    plain_metric = f"{selection_metric}_{VALIDATION_MODEL_PLAIN}"
    adjusted_available = adjusted_metric in wide.columns
    plain_available = plain_metric in wide.columns
    if not adjusted_available or not plain_available:
        raise ValueError("Validation accuracy must include both xLSTM plain and adjusted rows.")

    delta = wide[adjusted_metric] - wide[plain_metric]
    improvement_required = float(min_improvement)
    adjusted_wins = delta < 0 if improvement_required == 0.0 else delta <= -improvement_required
    wide["selected_model"] = np.where(adjusted_wins, VALIDATION_MODEL_ADJUSTED, VALIDATION_MODEL_PLAIN)
    missing_metric = ~np.isfinite(delta.to_numpy(dtype=float))
    if missing_metric.any():
        wide.loc[missing_metric, "selected_model"] = fallback_default
    wide["selection_metric"] = selection_metric
    wide["selection_scope"] = scope
    wide["selection_metric_delta_adjusted_minus_plain"] = delta
    wide["min_improvement_required"] = float(min_improvement)
    wide["selection_reason"] = np.select(
        [missing_metric, adjusted_wins],
        ["missing_validation_metric", "adjusted_validation_metric_improved"],
        default="plain_validation_metric_kept",
    )
    return wide


def complete_fallback_selection(
    selection: pd.DataFrame,
    target_monthly: pd.DataFrame,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    selection_scope: str = SELECTION_SCOPE_STOCK,
    min_improvement: float = 0.0,
    fallback_default: str = VALIDATION_MODEL_PLAIN,
) -> pd.DataFrame:
    scope = normalize_selection_scope(selection_scope)
    group_columns = selection_group_columns(scope)
    missing_columns = [column for column in group_columns if column not in target_monthly.columns]
    if missing_columns:
        raise ValueError(f"Target predictions must include selection columns: {missing_columns}")

    metadata_columns = [column for column in ["stock_name", "industry_category"] if column in target_monthly.columns]
    target_groups = target_monthly[[*group_columns, *metadata_columns]].drop_duplicates(group_columns).copy()
    target_groups["stock_id"] = target_groups["stock_id"].astype(int)

    selection_payload = selection.copy()
    for column in metadata_columns:
        if column in selection_payload.columns:
            selection_payload = selection_payload.rename(columns={column: f"{column}_validation"})

    completed = target_groups.merge(selection_payload, on=group_columns, how="left")
    for column in metadata_columns:
        validation_column = f"{column}_validation"
        if validation_column in completed.columns:
            completed[column] = completed[column].fillna(completed[validation_column])
            completed = completed.drop(columns=[validation_column])

    missing_selection = completed["selected_model"].isna()
    completed.loc[missing_selection, "selected_model"] = fallback_default
    completed.loc[missing_selection, "selection_metric"] = selection_metric
    completed.loc[missing_selection, "selection_scope"] = scope
    completed.loc[missing_selection, "min_improvement_required"] = float(min_improvement)
    if scope == SELECTION_SCOPE_STOCK_REGIME and "stock_id" in selection:
        stocks_with_validation = set(selection["stock_id"].dropna().astype(int).tolist())
        missing_stock = missing_selection & ~completed["stock_id"].astype(int).isin(stocks_with_validation)
        missing_regime = missing_selection & ~missing_stock
        completed.loc[missing_stock, "selection_reason"] = "missing_validation_stock"
        completed.loc[missing_regime, "selection_reason"] = "missing_validation_regime"
    else:
        completed.loc[missing_selection, "selection_reason"] = "missing_validation_stock"
    if "selection_metric_delta_adjusted_minus_plain" not in completed.columns:
        completed["selection_metric_delta_adjusted_minus_plain"] = np.nan
    sort_columns = [column for column in ["stock_id", "regime"] if column in completed.columns]
    return completed.sort_values(sort_columns).reset_index(drop=True)


def build_fallback_predictions(
    target_monthly: pd.DataFrame,
    selection: pd.DataFrame,
    selection_scope: str = SELECTION_SCOPE_STOCK,
) -> pd.DataFrame:
    scope = normalize_selection_scope(selection_scope)
    selection_keys = selection_group_columns(scope)
    missing_columns = [column for column in selection_keys if column not in target_monthly.columns]
    if missing_columns:
        raise ValueError(f"Target predictions must include selection columns: {missing_columns}")
    plain = target_monthly[target_monthly["model"].eq(VALIDATION_MODEL_PLAIN)].copy()
    adjusted = target_monthly[target_monthly["model"].eq(VALIDATION_MODEL_ADJUSTED)].copy()
    if plain.empty or adjusted.empty:
        raise ValueError("Target predictions must include both xLSTM plain and adjusted rows.")
    if "xlstm_decline_cap_applied" not in plain.columns:
        plain["xlstm_decline_cap_applied"] = False
    if "xlstm_decline_cap_applied" not in adjusted.columns:
        adjusted["xlstm_decline_cap_applied"] = False

    key_columns = ["stock_id", "target_year", "target_month"]
    adjusted_values = adjusted[key_columns + ["predicted_revenue", "xlstm_decline_cap_applied"]].rename(
        columns={
            "predicted_revenue": "predicted_revenue_adjusted_source",
            "xlstm_decline_cap_applied": "xlstm_decline_cap_applied_adjusted_source",
        }
    )
    selection_columns = [
        *selection_keys,
        "selected_model",
        "selection_metric",
        "selection_scope",
        "selection_metric_delta_adjusted_minus_plain",
        "selection_reason",
    ]
    selection_payload = selection.copy()
    if "selection_scope" not in selection_payload.columns:
        selection_payload["selection_scope"] = scope
    if "selection_metric" not in selection_payload.columns:
        selection_payload["selection_metric"] = DEFAULT_SELECTION_METRIC
    if "selection_metric_delta_adjusted_minus_plain" not in selection_payload.columns:
        selection_payload["selection_metric_delta_adjusted_minus_plain"] = np.nan
    if "selection_reason" not in selection_payload.columns:
        selection_payload["selection_reason"] = "selection_reason_unavailable"
    fallback = plain.merge(adjusted_values, on=key_columns, how="left").merge(
        selection_payload[selection_columns],
        on=selection_keys,
        how="left",
    )
    fallback["selected_model"] = fallback["selected_model"].fillna(VALIDATION_MODEL_PLAIN)
    fallback["selection_scope"] = fallback["selection_scope"].fillna(scope)
    selected_adjusted = fallback["selected_model"].eq(VALIDATION_MODEL_ADJUSTED)
    adjusted_prediction = pd.to_numeric(fallback["predicted_revenue_adjusted_source"], errors="coerce")
    use_adjusted = selected_adjusted & np.isfinite(adjusted_prediction.to_numpy(dtype=float))
    fallback["fallback_source_missing"] = selected_adjusted & ~use_adjusted
    fallback["source_model"] = np.where(use_adjusted, VALIDATION_MODEL_ADJUSTED, VALIDATION_MODEL_PLAIN)
    fallback["model"] = FALLBACK_MODEL
    fallback["predicted_revenue"] = np.where(
        use_adjusted,
        adjusted_prediction,
        fallback["predicted_revenue"],
    )
    adjusted_decline_cap_applied = (
        fallback["xlstm_decline_cap_applied_adjusted_source"]
        .astype("boolean")
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    plain_decline_cap_applied = (
        fallback["xlstm_decline_cap_applied"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    )
    fallback["xlstm_decline_cap_applied"] = np.where(
        use_adjusted,
        adjusted_decline_cap_applied,
        plain_decline_cap_applied,
    )
    fallback["error"] = fallback["predicted_revenue"] - fallback["actual_revenue"]
    fallback["abs_error"] = fallback["error"].abs()
    return fallback.drop(
        columns=["predicted_revenue_adjusted_source", "xlstm_decline_cap_applied_adjusted_source"],
        errors="ignore",
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-metric", default=DEFAULT_SELECTION_METRIC)
    parser.add_argument(
        "--selection-scope",
        choices=sorted(SELECTION_SCOPES),
        default=SELECTION_SCOPE_STOCK,
    )
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--fallback-default", choices=[VALIDATION_MODEL_PLAIN, VALIDATION_MODEL_ADJUSTED], default=VALIDATION_MODEL_PLAIN)
    parser.add_argument("--validation-year", type=int, default=DEFAULT_VALIDATION_YEAR)
    parser.add_argument("--validation-train-end-year", type=int, default=DEFAULT_VALIDATION_TRAIN_END_YEAR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max-train-samples", type=int, default=DEFAULT_MAX_TRAIN_SAMPLES)
    parser.add_argument("--under-weight", type=float, default=DEFAULT_UNDER_WEIGHT)
    args = parser.parse_args()

    started = time.time()
    target_predictions_path = Path(args.target_predictions)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    target_monthly = pd.read_csv(target_predictions_path)
    stock_ids = tuple(sorted(target_monthly["stock_id"].dropna().astype(int).unique().tolist()))
    spec = ValidationFallbackSpec(
        k=int(args.k),
        epochs=int(args.epochs),
        max_train_samples=int(args.max_train_samples),
        validation_year=int(args.validation_year),
        validation_train_end_year=int(args.validation_train_end_year),
        selection_metric=str(args.selection_metric),
        selection_scope=str(args.selection_scope),
        min_improvement=float(args.min_improvement),
        fallback_default=str(args.fallback_default),
        under_weight=float(args.under_weight),
    )

    print(
        f"Running xLSTM validation fallback: stocks={len(stock_ids)}, "
        f"validation_year={spec.validation_year}, metric={spec.selection_metric}, "
        f"scope={spec.selection_scope}",
        flush=True,
    )
    validation_monthly, _ = build_validation_predictions(stock_ids, spec)
    revenue = engine.load_revenue_data()
    stock_meta = basket.load_stock_metadata(revenue)
    validation_monthly = validation_monthly.merge(
        stock_meta[["stock_id", "stock_name", "industry_category"]],
        on="stock_id",
        how="left",
    )
    validation_group_columns = validation_selection_index_columns(spec.selection_scope, validation_monthly)
    validation_accuracy = basket.summarize(validation_monthly, [*validation_group_columns, "model"])
    selection = build_fallback_selection(
        validation_accuracy,
        selection_metric=spec.selection_metric,
        selection_scope=spec.selection_scope,
        min_improvement=float(spec.min_improvement),
        fallback_default=spec.fallback_default,
    )
    selection = complete_fallback_selection(
        selection,
        target_monthly,
        selection_metric=spec.selection_metric,
        selection_scope=spec.selection_scope,
        min_improvement=float(spec.min_improvement),
        fallback_default=spec.fallback_default,
    )
    fallback_monthly = build_fallback_predictions(target_monthly, selection, selection_scope=spec.selection_scope)
    combined_monthly = pd.concat([target_monthly, fallback_monthly], ignore_index=True)
    overall_accuracy = basket.summarize(combined_monthly, ["model"])
    stock_accuracy = basket.summarize(combined_monthly, ["stock_id", "stock_name", "industry_category", "model"])
    regime_accuracy = basket.summarize(combined_monthly, ["regime", "model"]) if "regime" in combined_monthly else pd.DataFrame()
    model_effects = main_flow.build_model_effects(stock_accuracy, effect_pairs=FALLBACK_EFFECT_PAIRS)
    winner_summary = main_flow.build_winner_summary(model_effects)

    write_csv(validation_monthly, output_dir / "validation_monthly_predictions.csv")
    write_csv(validation_accuracy, output_dir / "validation_accuracy.csv")
    write_csv(selection, output_dir / "fallback_selection.csv")
    write_csv(fallback_monthly, output_dir / "fallback_monthly_predictions.csv")
    write_csv(combined_monthly, output_dir / "combined_monthly_predictions.csv")
    write_csv(overall_accuracy, output_dir / "overall_accuracy.csv")
    write_csv(stock_accuracy, output_dir / "stock_accuracy.csv")
    write_csv(regime_accuracy, output_dir / "regime_accuracy.csv")
    write_csv(model_effects, output_dir / "model_effects.csv")
    write_csv(winner_summary, output_dir / "winner_summary.csv")

    completed = time.time()
    run_config = {
        "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(completed)),
        "duration_sec": round(completed - started, 3),
        "target_predictions": str(target_predictions_path),
        "stock_count": len(stock_ids),
        "parameters": {
            "k": int(spec.k),
            "window_size": int(spec.window_size),
            "epochs": int(spec.epochs),
            "max_train_samples": int(spec.max_train_samples),
            "validation_year": int(spec.validation_year),
            "validation_train_end_year": int(spec.validation_train_end_year),
            "selection_metric": spec.selection_metric,
            "selection_scope": spec.selection_scope,
            "min_improvement": float(spec.min_improvement),
            "fallback_default": spec.fallback_default,
            "under_weight": float(spec.under_weight),
            "xlstm_growth_alpha": float(spec.xlstm_growth_alpha),
            "xlstm_decline_cap_growth_ratio_max": spec.xlstm_decline_cap_growth_ratio_max,
            "xlstm_decline_cap_prediction_ratio_min": float(spec.xlstm_decline_cap_prediction_ratio_min),
        },
        "selection_counts": selection["selected_model"].value_counts().sort_index().to_dict(),
        "selection_reason_counts": selection["selection_reason"].value_counts().sort_index().to_dict(),
        "row_counts": {
            "validation_monthly_predictions": int(len(validation_monthly)),
            "validation_accuracy": int(len(validation_accuracy)),
            "fallback_selection": int(len(selection)),
            "fallback_monthly_predictions": int(len(fallback_monthly)),
            "combined_monthly_predictions": int(len(combined_monthly)),
            "overall_accuracy": int(len(overall_accuracy)),
            "stock_accuracy": int(len(stock_accuracy)),
            "model_effects": int(len(model_effects)),
            "winner_summary": int(len(winner_summary)),
        },
        "metric_notes": {
            "selection": "Source model is selected at the configured scope from prior-year validation only.",
            "actual_usage": "Target-year actual revenue is used only after fallback predictions are selected.",
        },
    }
    run_config = write_rolling_run_config(
        output_dir,
        run_config,
        experiment_family="rolling_xlstm_validation_fallback",
        evidence_tier="B",
        selection_protocol="historical-validation",
        report_ready=False,
        report_ready_reason=(
            "Fallback selection is time-safe, but its upstream xLSTM/default policy was developed "
            "after inspecting the target-year evaluation."
        ),
    )

    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
