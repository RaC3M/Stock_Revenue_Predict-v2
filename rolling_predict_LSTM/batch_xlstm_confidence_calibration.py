from __future__ import annotations

"""Calibrate xLSTM decline-cap confidence threshold on prior-year validation.

This D1.20 runner closes the loop after D1.19. It uses validation predictions
whose targets are in 2024 to select a confidence threshold, then applies only
that threshold to a saved 2025 main-flow monthly_predictions.csv file. Target
year actual revenue is used only after the threshold is selected.
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
    from . import batch_xlstm_decline_cap_confidence as confidence
    from . import batch_xlstm_main_flow_comparison as main_flow
    from . import batch_xlstm_validation_fallback as validation
    from . import rolling_lstm_engine as engine
    from .experiment_metadata import write_rolling_run_config
except ImportError:
    import batch_sequence_backbone_ablation as basket
    import batch_xlstm_decline_cap_confidence as confidence
    import batch_xlstm_main_flow_comparison as main_flow
    import batch_xlstm_validation_fallback as validation
    import rolling_lstm_engine as engine
    from experiment_metadata import write_rolling_run_config


CALIBRATED_MODEL = "Rolling xLSTM Decline Confidence Calibrated"
DEFAULT_THRESHOLDS = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
DEFAULT_SELECTION_METRIC = "WMAPE"
LOWER_IS_BETTER_METRICS = {"MSE", "RMSE", "MAE", "MAPE", "MedianAPE", "WMAPE", "SMAPE"}

CALIBRATED_EFFECT_PAIRS = (
    (
        "calibrated_minus_cluster_adjusted",
        confidence.MODEL_CLUSTER_ADJUSTED,
        CALIBRATED_MODEL,
    ),
    (
        "calibrated_minus_xlstm_adjusted",
        confidence.MODEL_XLSTM_ADJUSTED,
        CALIBRATED_MODEL,
    ),
    (
        "calibrated_minus_xlstm_plain",
        confidence.MODEL_XLSTM_PLAIN,
        CALIBRATED_MODEL,
    ),
)


@dataclass(frozen=True)
class ConfidenceCalibrationSpec:
    k: int = validation.DEFAULT_K
    window_size: int = engine.DEFAULT_WINDOW_SIZE
    epochs: int = validation.DEFAULT_EPOCHS
    max_train_samples: int = validation.DEFAULT_MAX_TRAIN_SAMPLES
    validation_year: int = validation.DEFAULT_VALIDATION_YEAR
    validation_train_end_year: int = validation.DEFAULT_VALIDATION_TRAIN_END_YEAR
    selection_metric: str = DEFAULT_SELECTION_METRIC
    use_asymmetric_loss: bool = True
    under_weight: float = validation.DEFAULT_UNDER_WEIGHT
    score: confidence.ConfidenceScoreSpec = confidence.ConfidenceScoreSpec()


def parse_thresholds(value: str | None) -> tuple[float, ...]:
    return confidence.parse_float_csv(value, DEFAULT_THRESHOLDS)


def _metric_value(row: pd.Series, metric: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
    return float(value)


def build_threshold_selection(
    validation_accuracy: pd.DataFrame,
    thresholds: tuple[float, ...],
    selection_metric: str = DEFAULT_SELECTION_METRIC,
) -> pd.DataFrame:
    metric = str(selection_metric)
    if metric not in LOWER_IS_BETTER_METRICS:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")
    if metric not in validation_accuracy.columns:
        raise ValueError(f"Validation accuracy is missing metric: {selection_metric}")

    rows: list[dict[str, object]] = []
    by_model = validation_accuracy.set_index("model", drop=False)
    for threshold in thresholds:
        model = confidence.confidence_model_name(float(threshold))
        if model not in by_model.index:
            rows.append(
                {
                    "threshold": float(threshold),
                    "model": model,
                    "selection_metric": metric,
                    "validation_metric": np.nan,
                    "selection_reason": "missing_validation_model",
                    "selected": False,
                }
            )
            continue
        row = by_model.loc[model]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        metric_value = _metric_value(row, metric)
        rows.append(
            {
                "threshold": float(threshold),
                "model": model,
                "selection_metric": metric,
                "validation_metric": metric_value,
                "selection_reason": "candidate",
                "selected": False,
            }
        )
    selection = pd.DataFrame(rows)
    valid = selection[np.isfinite(selection["validation_metric"].to_numpy(dtype=float))].copy()
    if valid.empty:
        raise ValueError("No valid threshold validation metric is available.")

    valid = valid.sort_values(["validation_metric", "threshold"], ascending=[True, False])
    selected_index = valid.index[0]
    selection.loc[selected_index, "selected"] = True
    selection.loc[selected_index, "selection_reason"] = "lowest_validation_metric"
    selection["selected_threshold"] = float(selection.loc[selected_index, "threshold"])
    selection["selected_model"] = str(selection.loc[selected_index, "model"])
    return selection.sort_values("threshold").reset_index(drop=True)


def selected_threshold(selection: pd.DataFrame) -> float:
    selected = selection[selection["selected"].astype(bool)]
    if len(selected) != 1:
        raise ValueError("Threshold selection must contain exactly one selected row.")
    return float(selected.iloc[0]["threshold"])


def build_calibrated_predictions(
    target_confidence_predictions: pd.DataFrame,
    threshold_selection: pd.DataFrame,
    validation_year: int,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
) -> pd.DataFrame:
    threshold = selected_threshold(threshold_selection)
    source_model = confidence.confidence_model_name(threshold)
    calibrated = target_confidence_predictions[target_confidence_predictions["model"].eq(source_model)].copy()
    if calibrated.empty:
        raise ValueError(f"Target confidence predictions are missing selected model: {source_model}")
    calibrated["source_model"] = source_model
    calibrated["model"] = CALIBRATED_MODEL
    calibrated["calibrated_threshold"] = float(threshold)
    calibrated["calibration_year"] = int(validation_year)
    calibrated["calibration_metric"] = str(selection_metric)
    calibrated["calibration_selected_model"] = source_model
    calibrated["error"] = calibrated["predicted_revenue"] - calibrated["actual_revenue"]
    calibrated["abs_error"] = calibrated["error"].abs()
    return calibrated


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--selection-metric", default=DEFAULT_SELECTION_METRIC)
    parser.add_argument("--validation-year", type=int, default=validation.DEFAULT_VALIDATION_YEAR)
    parser.add_argument("--validation-train-end-year", type=int, default=validation.DEFAULT_VALIDATION_TRAIN_END_YEAR)
    parser.add_argument("--k", type=int, default=validation.DEFAULT_K)
    parser.add_argument("--epochs", type=int, default=validation.DEFAULT_EPOCHS)
    parser.add_argument("--max-train-samples", type=int, default=validation.DEFAULT_MAX_TRAIN_SAMPLES)
    parser.add_argument("--under-weight", type=float, default=validation.DEFAULT_UNDER_WEIGHT)
    parser.add_argument("--overshoot-weight", type=float, default=confidence.DEFAULT_OVERSHOOT_WEIGHT)
    parser.add_argument("--decline-depth-weight", type=float, default=confidence.DEFAULT_DECLINE_DEPTH_WEIGHT)
    parser.add_argument("--low-streak-weight", type=float, default=confidence.DEFAULT_LOW_STREAK_WEIGHT)
    parser.add_argument("--overshoot-scale", type=float, default=confidence.DEFAULT_OVERSHOOT_SCALE)
    parser.add_argument("--decline-ratio-reference", type=float, default=confidence.DEFAULT_DECLINE_RATIO_REFERENCE)
    parser.add_argument("--streak-reference", type=float, default=confidence.DEFAULT_STREAK_REFERENCE)
    args = parser.parse_args()

    started = time.time()
    target_predictions_path = Path(args.target_predictions)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    thresholds = parse_thresholds(args.thresholds)
    score_spec = confidence.ConfidenceScoreSpec(
        overshoot_weight=float(args.overshoot_weight),
        decline_depth_weight=float(args.decline_depth_weight),
        low_streak_weight=float(args.low_streak_weight),
        overshoot_scale=float(args.overshoot_scale),
        decline_ratio_reference=float(args.decline_ratio_reference),
        streak_reference=float(args.streak_reference),
    )
    confidence.validate_score_spec(score_spec)
    target_monthly = confidence.load_monthly_predictions(target_predictions_path)
    stock_ids = tuple(sorted(target_monthly["stock_id"].dropna().astype(int).unique().tolist()))
    spec = ConfidenceCalibrationSpec(
        k=int(args.k),
        epochs=int(args.epochs),
        max_train_samples=int(args.max_train_samples),
        validation_year=int(args.validation_year),
        validation_train_end_year=int(args.validation_train_end_year),
        selection_metric=str(args.selection_metric),
        under_weight=float(args.under_weight),
        score=score_spec,
    )

    print(
        f"Running xLSTM confidence calibration: stocks={len(stock_ids)}, "
        f"validation_year={spec.validation_year}, metric={spec.selection_metric}",
        flush=True,
    )
    validation_spec = validation.ValidationFallbackSpec(
        k=int(spec.k),
        epochs=int(spec.epochs),
        max_train_samples=int(spec.max_train_samples),
        validation_year=int(spec.validation_year),
        validation_train_end_year=int(spec.validation_train_end_year),
        under_weight=float(spec.under_weight),
    )
    validation_monthly, _ = validation.build_validation_predictions(stock_ids, validation_spec)
    revenue = engine.load_revenue_data()
    stock_meta = basket.load_stock_metadata(revenue)
    validation_monthly = validation_monthly.merge(
        stock_meta[["stock_id", "stock_name", "industry_category"]],
        on="stock_id",
        how="left",
    )

    validation_confidence = confidence.build_confidence_predictions(validation_monthly, thresholds, score_spec)
    validation_combined = pd.concat([validation_monthly, validation_confidence], ignore_index=True)
    validation_accuracy = basket.summarize(validation_combined, ["model"])
    threshold_selection = build_threshold_selection(validation_accuracy, thresholds, spec.selection_metric)
    chosen_threshold = selected_threshold(threshold_selection)

    target_confidence = confidence.build_confidence_predictions(target_monthly, thresholds, score_spec)
    calibrated_monthly = build_calibrated_predictions(
        target_confidence,
        threshold_selection,
        validation_year=int(spec.validation_year),
        selection_metric=spec.selection_metric,
    )
    target_threshold_accuracy = basket.summarize(
        pd.concat([target_monthly, target_confidence], ignore_index=True),
        ["model"],
    )
    combined_monthly = pd.concat([target_monthly, calibrated_monthly], ignore_index=True)
    overall_accuracy = basket.summarize(combined_monthly, ["model"])
    stock_accuracy = basket.summarize(combined_monthly, ["stock_id", "stock_name", "industry_category", "model"])
    regime_accuracy = (
        basket.summarize(combined_monthly, ["regime", "model"]) if "regime" in combined_monthly else pd.DataFrame()
    )
    model_effects = main_flow.build_model_effects(stock_accuracy, effect_pairs=CALIBRATED_EFFECT_PAIRS)
    winner_summary = main_flow.build_winner_summary(model_effects)

    write_csv(validation_monthly, output_dir / "validation_monthly_predictions.csv")
    write_csv(validation_confidence, output_dir / "validation_confidence_predictions.csv")
    write_csv(validation_accuracy, output_dir / "validation_accuracy.csv")
    write_csv(threshold_selection, output_dir / "threshold_selection.csv")
    write_csv(target_confidence, output_dir / "target_confidence_predictions.csv")
    write_csv(target_threshold_accuracy, output_dir / "target_threshold_accuracy.csv")
    write_csv(calibrated_monthly, output_dir / "calibrated_monthly_predictions.csv")
    write_csv(combined_monthly, output_dir / "combined_monthly_predictions.csv")
    write_csv(overall_accuracy, output_dir / "overall_accuracy.csv")
    write_csv(stock_accuracy, output_dir / "stock_accuracy.csv")
    write_csv(regime_accuracy, output_dir / "regime_accuracy.csv")
    write_csv(model_effects, output_dir / "model_effects.csv")
    write_csv(winner_summary, output_dir / "winner_summary.csv")

    completed = time.time()
    cap_counts = target_confidence.groupby("model", sort=True)["confidence_cap_applied"].sum().astype(int).to_dict()
    calibrated_cap_count = int(calibrated_monthly["confidence_cap_applied"].sum())
    run_config = {
        "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(completed)),
        "duration_sec": round(completed - started, 3),
        "target_predictions": str(target_predictions_path),
        "stock_count": len(stock_ids),
        "thresholds": [float(threshold) for threshold in thresholds],
        "selected_threshold": float(chosen_threshold),
        "selected_threshold_model": confidence.confidence_model_name(chosen_threshold),
        "parameters": {
            "k": int(spec.k),
            "window_size": int(spec.window_size),
            "epochs": int(spec.epochs),
            "max_train_samples": int(spec.max_train_samples),
            "validation_year": int(spec.validation_year),
            "validation_train_end_year": int(spec.validation_train_end_year),
            "selection_metric": spec.selection_metric,
            "under_weight": float(spec.under_weight),
        },
        "score_spec": {
            "overshoot_weight": float(score_spec.overshoot_weight),
            "decline_depth_weight": float(score_spec.decline_depth_weight),
            "low_streak_weight": float(score_spec.low_streak_weight),
            "overshoot_scale": float(score_spec.overshoot_scale),
            "decline_ratio_reference": float(score_spec.decline_ratio_reference),
            "streak_reference": float(score_spec.streak_reference),
        },
        "target_threshold_cap_counts": cap_counts,
        "calibrated_cap_count": calibrated_cap_count,
        "row_counts": {
            "validation_monthly_predictions": int(len(validation_monthly)),
            "validation_confidence_predictions": int(len(validation_confidence)),
            "validation_accuracy": int(len(validation_accuracy)),
            "threshold_selection": int(len(threshold_selection)),
            "target_confidence_predictions": int(len(target_confidence)),
            "calibrated_monthly_predictions": int(len(calibrated_monthly)),
            "combined_monthly_predictions": int(len(combined_monthly)),
            "overall_accuracy": int(len(overall_accuracy)),
            "stock_accuracy": int(len(stock_accuracy)),
            "model_effects": int(len(model_effects)),
            "winner_summary": int(len(winner_summary)),
        },
        "metric_notes": {
            "selection": "Confidence threshold is selected from validation-year predictions only.",
            "actual_usage": "Target-year actual revenue is used only after threshold selection.",
            "target_threshold_accuracy": "Diagnostic only; not used to select the calibrated threshold.",
        },
    }
    run_config = write_rolling_run_config(
        output_dir,
        run_config,
        experiment_family="rolling_xlstm_confidence_calibration",
        evidence_tier="B",
        selection_protocol="historical-validation",
        report_ready=False,
        report_ready_reason=(
            "Threshold selection uses historical validation, but the upstream xLSTM/default policy "
            "was developed after target-year inspection."
        ),
    )

    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
