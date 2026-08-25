from __future__ import annotations

"""Replay confidence-gated xLSTM decline caps on saved monthly predictions.

This D1.19 runner is post-hoc: it does not retrain xLSTM. It starts from the
saved main-flow monthly_predictions.csv, scores whether each xLSTM plain
prediction looks like a credible decline-cap candidate using only target-month
available metadata, then applies a cap when the score clears a threshold.
Actual revenue is used only after the variant prediction is selected.
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


MODEL_XLSTM_PLAIN = "Rolling xLSTM"
MODEL_XLSTM_ADJUSTED = "Rolling xLSTM + Conditional Adjustment"
MODEL_CLUSTER_ADJUSTED = "Rolling LSTM + Cluster + Conditional Adjustment"
CONFIDENCE_MODEL_PREFIX = "Rolling xLSTM Decline Confidence"

DEFAULT_THRESHOLDS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_OVERSHOOT_WEIGHT = 0.50
DEFAULT_DECLINE_DEPTH_WEIGHT = 0.35
DEFAULT_LOW_STREAK_WEIGHT = 0.15
DEFAULT_OVERSHOOT_SCALE = 0.50
DEFAULT_DECLINE_RATIO_REFERENCE = engine.DECLINE_REGIME_RATIO_THRESHOLD
DEFAULT_STREAK_REFERENCE = 4.0

REQUIRED_COLUMNS = {
    "stock_id",
    "target_year",
    "target_month",
    "actual_revenue",
    "last_observed_revenue",
    "regime",
    "growth_ratio",
    "growth_streak",
    "model",
    "predicted_revenue",
}
NUMERIC_COLUMNS = (
    "stock_id",
    "target_year",
    "target_month",
    "actual_revenue",
    "last_observed_revenue",
    "growth_ratio",
    "growth_streak",
    "predicted_revenue",
)


@dataclass(frozen=True)
class ConfidenceScoreSpec:
    overshoot_weight: float = DEFAULT_OVERSHOOT_WEIGHT
    decline_depth_weight: float = DEFAULT_DECLINE_DEPTH_WEIGHT
    low_streak_weight: float = DEFAULT_LOW_STREAK_WEIGHT
    overshoot_scale: float = DEFAULT_OVERSHOOT_SCALE
    decline_ratio_reference: float = DEFAULT_DECLINE_RATIO_REFERENCE
    streak_reference: float = DEFAULT_STREAK_REFERENCE


def parse_float_csv(value: str | None, default: tuple[float, ...] = DEFAULT_THRESHOLDS) -> tuple[float, ...]:
    if not value:
        return default
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("At least one float value is required.")
    return parsed


def _threshold_token(threshold: float) -> str:
    return f"{float(threshold):g}".replace("-", "m").replace(".", "p")


def confidence_model_name(threshold: float) -> str:
    return f"{CONFIDENCE_MODEL_PREFIX} >= {_threshold_token(threshold)}"


def validate_score_spec(spec: ConfidenceScoreSpec) -> None:
    weights = [float(spec.overshoot_weight), float(spec.decline_depth_weight), float(spec.low_streak_weight)]
    if any(weight < 0 for weight in weights):
        raise ValueError("Confidence weights must be non-negative.")
    if np.isclose(sum(weights), 0.0):
        raise ValueError("At least one confidence weight must be positive.")
    if spec.overshoot_scale <= 0:
        raise ValueError("overshoot_scale must be positive.")
    if spec.decline_ratio_reference <= 0:
        raise ValueError("decline_ratio_reference must be positive.")
    if spec.streak_reference <= 0:
        raise ValueError("streak_reference must be positive.")


def load_monthly_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {missing}")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["stock_id", "target_year", "target_month", "predicted_revenue"]).copy()
    frame["stock_id"] = frame["stock_id"].astype(int)
    frame["target_year"] = frame["target_year"].astype(int)
    frame["target_month"] = frame["target_month"].astype(int)
    models = set(frame["model"].dropna().astype(str).unique())
    required_models = {MODEL_XLSTM_PLAIN, MODEL_XLSTM_ADJUSTED}
    if not required_models.issubset(models):
        raise ValueError(f"Prediction file must include models: {sorted(required_models)}")
    return frame.sort_values(["stock_id", "target_year", "target_month", "model"]).reset_index(drop=True)


def calculate_decline_cap_confidence(
    frame: pd.DataFrame,
    spec: ConfidenceScoreSpec = ConfidenceScoreSpec(),
) -> pd.DataFrame:
    validate_score_spec(spec)
    predicted = frame["predicted_revenue"].to_numpy(dtype=float)
    last_observed = frame["last_observed_revenue"].to_numpy(dtype=float)
    growth_ratio = frame["growth_ratio"].to_numpy(dtype=float)
    growth_streak = frame["growth_streak"].to_numpy(dtype=float)

    prediction_ratio = np.divide(
        predicted,
        last_observed,
        out=np.ones_like(predicted, dtype=float),
        where=np.isfinite(last_observed) & (last_observed > 0),
    )
    overshoot_score = np.clip((prediction_ratio - 1.0) / float(spec.overshoot_scale), 0.0, 1.0)
    decline_depth_score = np.clip(
        (float(spec.decline_ratio_reference) - growth_ratio) / float(spec.decline_ratio_reference),
        0.0,
        1.0,
    )
    low_streak_score = np.clip((float(spec.streak_reference) - growth_streak) / float(spec.streak_reference), 0.0, 1.0)

    weight_sum = float(spec.overshoot_weight + spec.decline_depth_weight + spec.low_streak_weight)
    raw_score = (
        float(spec.overshoot_weight) * overshoot_score
        + float(spec.decline_depth_weight) * decline_depth_score
        + float(spec.low_streak_weight) * low_streak_score
    ) / weight_sum
    is_decline = frame["regime"].astype(str).eq("decline").to_numpy(dtype=bool)
    confidence = np.where(is_decline, raw_score, 0.0)

    result = frame.copy()
    result["prediction_ratio_to_last"] = prediction_ratio
    result["confidence_overshoot_score"] = overshoot_score
    result["confidence_decline_depth_score"] = decline_depth_score
    result["confidence_low_streak_score"] = low_streak_score
    result["decline_cap_confidence"] = confidence
    result["decline_cap_confidence_candidate"] = confidence > 0
    return result


def build_variant_catalog(thresholds: tuple[float, ...], spec: ConfidenceScoreSpec) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": confidence_model_name(threshold),
                "threshold": float(threshold),
                "overshoot_weight": float(spec.overshoot_weight),
                "decline_depth_weight": float(spec.decline_depth_weight),
                "low_streak_weight": float(spec.low_streak_weight),
                "overshoot_scale": float(spec.overshoot_scale),
                "decline_ratio_reference": float(spec.decline_ratio_reference),
                "streak_reference": float(spec.streak_reference),
            }
            for threshold in thresholds
        ]
    )


def build_confidence_predictions(
    monthly_predictions: pd.DataFrame,
    thresholds: tuple[float, ...],
    spec: ConfidenceScoreSpec = ConfidenceScoreSpec(),
) -> pd.DataFrame:
    if any(float(threshold) < 0 for threshold in thresholds):
        raise ValueError("Thresholds must be non-negative.")
    plain = monthly_predictions[monthly_predictions["model"].eq(MODEL_XLSTM_PLAIN)].copy()
    if plain.empty:
        raise ValueError("No xLSTM plain rows are available.")

    scored = calculate_decline_cap_confidence(plain, spec)
    predicted_plain = scored["predicted_revenue"].to_numpy(dtype=float)
    last_observed = scored["last_observed_revenue"].to_numpy(dtype=float)
    actual = scored["actual_revenue"].to_numpy(dtype=float)
    confidence = scored["decline_cap_confidence"].to_numpy(dtype=float)

    rows: list[pd.DataFrame] = []
    for threshold in thresholds:
        threshold = float(threshold)
        cap_applied = (confidence > 0) & (confidence >= threshold)
        capped = np.where(cap_applied, np.minimum(predicted_plain, last_observed), predicted_plain)
        predicted = engine.safe_round_revenue(capped).astype(float)
        part = scored.copy()
        part["model"] = confidence_model_name(threshold)
        part["source_model"] = MODEL_XLSTM_PLAIN
        part["confidence_threshold"] = threshold
        part["confidence_cap_applied"] = cap_applied
        part["predicted_revenue_plain"] = predicted_plain
        part["predicted_revenue"] = predicted
        part["error"] = predicted - actual
        part["abs_error"] = np.abs(part["error"].to_numpy(dtype=float))
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_score_distribution(scored_plain: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("all", scored_plain)]
    if "regime" in scored_plain.columns:
        groups.extend((str(regime), group) for regime, group in scored_plain.groupby("regime", sort=True, dropna=False))
    for regime, frame in groups:
        score = frame["decline_cap_confidence"].to_numpy(dtype=float)
        ratio = frame["prediction_ratio_to_last"].to_numpy(dtype=float)
        rows.append(
            {
                "regime": regime,
                "observations": int(len(frame)),
                "candidate_observations": int((score > 0).sum()),
                "mean_confidence": float(np.nanmean(score)) if len(score) else np.nan,
                "p50_confidence": float(np.nanpercentile(score, 50)) if len(score) else np.nan,
                "p75_confidence": float(np.nanpercentile(score, 75)) if len(score) else np.nan,
                "p90_confidence": float(np.nanpercentile(score, 90)) if len(score) else np.nan,
                "max_confidence": float(np.nanmax(score)) if len(score) else np.nan,
                "mean_prediction_ratio_to_last": float(np.nanmean(ratio)) if len(ratio) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_effect_pairs(confidence_models: tuple[str, ...]) -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    for model in confidence_models:
        token = model.removeprefix(CONFIDENCE_MODEL_PREFIX).strip().replace(" ", "_").replace(">=", "gte")
        rows.extend(
            [
                (f"confidence_{token}_minus_xlstm_plain", MODEL_XLSTM_PLAIN, model),
                (f"confidence_{token}_minus_xlstm_adjusted", MODEL_XLSTM_ADJUSTED, model),
                (f"confidence_{token}_minus_cluster_adjusted", MODEL_CLUSTER_ADJUSTED, model),
            ]
        )
    return tuple(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--overshoot-weight", type=float, default=DEFAULT_OVERSHOOT_WEIGHT)
    parser.add_argument("--decline-depth-weight", type=float, default=DEFAULT_DECLINE_DEPTH_WEIGHT)
    parser.add_argument("--low-streak-weight", type=float, default=DEFAULT_LOW_STREAK_WEIGHT)
    parser.add_argument("--overshoot-scale", type=float, default=DEFAULT_OVERSHOOT_SCALE)
    parser.add_argument("--decline-ratio-reference", type=float, default=DEFAULT_DECLINE_RATIO_REFERENCE)
    parser.add_argument("--streak-reference", type=float, default=DEFAULT_STREAK_REFERENCE)
    args = parser.parse_args()

    started = time.time()
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    thresholds = parse_float_csv(args.thresholds)
    spec = ConfidenceScoreSpec(
        overshoot_weight=float(args.overshoot_weight),
        decline_depth_weight=float(args.decline_depth_weight),
        low_streak_weight=float(args.low_streak_weight),
        overshoot_scale=float(args.overshoot_scale),
        decline_ratio_reference=float(args.decline_ratio_reference),
        streak_reference=float(args.streak_reference),
    )
    validate_score_spec(spec)

    monthly_predictions = load_monthly_predictions(predictions_path)
    plain = monthly_predictions[monthly_predictions["model"].eq(MODEL_XLSTM_PLAIN)].copy()
    scored_plain = calculate_decline_cap_confidence(plain, spec)
    confidence_predictions = build_confidence_predictions(monthly_predictions, thresholds, spec)
    combined_monthly = pd.concat([monthly_predictions, confidence_predictions], ignore_index=True)

    confidence_models = tuple(confidence_model_name(threshold) for threshold in thresholds)
    variant_catalog = build_variant_catalog(thresholds, spec)
    score_distribution = build_score_distribution(scored_plain)
    overall_accuracy = basket.summarize(combined_monthly, ["model"])
    stock_accuracy = basket.summarize(combined_monthly, ["stock_id", "stock_name", "industry_category", "model"])
    regime_accuracy = (
        basket.summarize(combined_monthly, ["regime", "model"]) if "regime" in combined_monthly else pd.DataFrame()
    )
    model_effects = main_flow.build_model_effects(stock_accuracy, effect_pairs=build_effect_pairs(confidence_models))
    winner_summary = main_flow.build_winner_summary(model_effects)

    write_csv(variant_catalog, output_dir / "variant_catalog.csv")
    write_csv(scored_plain, output_dir / "scored_xlstm_plain.csv")
    write_csv(score_distribution, output_dir / "score_distribution.csv")
    write_csv(confidence_predictions, output_dir / "monthly_confidence_predictions.csv")
    write_csv(combined_monthly, output_dir / "combined_monthly_predictions.csv")
    write_csv(overall_accuracy, output_dir / "overall_accuracy.csv")
    write_csv(stock_accuracy, output_dir / "stock_accuracy.csv")
    write_csv(regime_accuracy, output_dir / "regime_accuracy.csv")
    write_csv(model_effects, output_dir / "model_effects.csv")
    write_csv(winner_summary, output_dir / "winner_summary.csv")

    completed = time.time()
    cap_counts = (
        confidence_predictions.groupby("model", sort=True)["confidence_cap_applied"].sum().astype(int).to_dict()
        if not confidence_predictions.empty
        else {}
    )
    run_config = {
        "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(completed)),
        "duration_sec": round(completed - started, 3),
        "predictions": str(predictions_path),
        "thresholds": [float(threshold) for threshold in thresholds],
        "score_spec": {
            "overshoot_weight": float(spec.overshoot_weight),
            "decline_depth_weight": float(spec.decline_depth_weight),
            "low_streak_weight": float(spec.low_streak_weight),
            "overshoot_scale": float(spec.overshoot_scale),
            "decline_ratio_reference": float(spec.decline_ratio_reference),
            "streak_reference": float(spec.streak_reference),
        },
        "row_counts": {
            "source_monthly_predictions": int(len(monthly_predictions)),
            "scored_xlstm_plain": int(len(scored_plain)),
            "monthly_confidence_predictions": int(len(confidence_predictions)),
            "combined_monthly_predictions": int(len(combined_monthly)),
            "overall_accuracy": int(len(overall_accuracy)),
            "stock_accuracy": int(len(stock_accuracy)),
            "model_effects": int(len(model_effects)),
            "winner_summary": int(len(winner_summary)),
        },
        "cap_counts": cap_counts,
        "metric_notes": {
            "confidence_score": "Uses predicted/plain-to-last ratio, prior-window growth ratio, growth streak, and regime only.",
            "actual_usage": "Actual revenue is used only after confidence-gated predictions are produced.",
        },
    }
    run_config = write_rolling_run_config(
        output_dir,
        run_config,
        experiment_family="rolling_xlstm_confidence_scan",
        evidence_tier="C",
        selection_protocol="target-year-hindsight",
        report_ready=False,
        report_ready_reason="Confidence thresholds are ranked directly on target-year metrics.",
    )

    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
