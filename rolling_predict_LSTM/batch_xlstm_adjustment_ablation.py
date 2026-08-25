from __future__ import annotations

"""Replay xLSTM adjustment variants on saved monthly predictions.

This D1.11 runner does not retrain the sequence model. It takes a
monthly_predictions.csv file produced with include_xlstm_plain=True and
separates growth-boost and decline-cap effects on the plain xLSTM forecast.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import rolling_lstm_engine as engine
    from .experiment_metadata import write_rolling_run_config
except ImportError:
    import rolling_lstm_engine as engine
    from experiment_metadata import write_rolling_run_config


DEFAULT_ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5)
DEFAULT_CONDITIONAL_OPTIONS = (True, False)
DEFAULT_REGIME_OPTIONS = (True, False)
DEFAULT_BALANCED_DECLINE_CAP_GROWTH_RATIO_MAX = engine.DEFAULT_XLSTM_DECLINE_CAP_GROWTH_RATIO_MAX
DEFAULT_BALANCED_DECLINE_CAP_PREDICTION_RATIO_MIN = engine.DEFAULT_XLSTM_DECLINE_CAP_PREDICTION_RATIO_MIN

REQUIRED_COLUMNS = {
    "stock_id",
    "target_year",
    "target_month",
    "actual_revenue",
    "predicted_revenue_xlstm",
    "last_observed_revenue",
    "sequence_max_revenue",
    "growth_rate_at_end",
    "momentum_3m_at_end",
    "momentum_6m_at_end",
    "growth_ratio",
    "growth_streak",
}

NUMERIC_REQUIRED_COLUMNS = tuple(sorted(REQUIRED_COLUMNS - {"target_year", "target_month"}))

METADATA_COLUMNS = [
    "stock_id",
    "target_date",
    "target_year",
    "target_month",
    "actual_revenue",
    "last_observed_revenue",
    "sequence_max_revenue",
    "cluster",
    "regime",
    "growth_ratio",
    "growth_streak",
    "growth_rate_at_end",
    "momentum_3m_at_end",
    "momentum_6m_at_end",
    "is_growth_phase",
    "is_high_growth_flag",
]

BASELINE_MODEL_COLUMNS = {
    "Rolling LSTM": "predicted_revenue_no_cluster",
    "Rolling LSTM + Cluster": "predicted_revenue_cluster",
    "Rolling LSTM + Cluster + Conditional Adjustment": "predicted_revenue_adjusted",
    "Rolling xLSTM": "predicted_revenue_xlstm",
    "Rolling xLSTM + Conditional Adjustment": "predicted_revenue_xlstm_adjusted",
}

LOWER_IS_BETTER_METRICS = ("MSE", "RMSE", "MAE", "MAPE", "MedianAPE", "WMAPE", "SMAPE")
HIGHER_IS_BETTER_METRICS = ("DirectionAccuracy",)


@dataclass(frozen=True)
class AdjustmentSpec:
    name: str
    alpha: float
    conditional: bool
    regime_strategy: bool
    enabled: bool = True
    growth_boost_enabled: bool = True
    decline_cap_enabled: bool = True
    effect_component: str = "growth_boost_and_decline_cap"
    decline_cap_growth_ratio_max: float | None = None
    decline_cap_prediction_ratio_min: float = engine.DEFAULT_DECLINE_CAP_PREDICTION_RATIO_MIN


def parse_float_csv(value: str | None, default: tuple[float, ...] = DEFAULT_ALPHAS) -> tuple[float, ...]:
    if not value:
        return default
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("At least one alpha value is required.")
    return parsed


def parse_bool_csv(
    value: str | None,
    default: tuple[bool, ...] = DEFAULT_CONDITIONAL_OPTIONS,
) -> tuple[bool, ...]:
    if not value:
        return default
    parsed: list[bool] = []
    for part in value.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token in {"1", "true", "t", "yes", "y", "on"}:
            parsed.append(True)
        elif token in {"0", "false", "f", "no", "n", "off"}:
            parsed.append(False)
        else:
            raise ValueError(f"Unknown boolean value: {part!r}")
    if not parsed:
        raise ValueError("At least one boolean option is required.")
    return tuple(parsed)


def _alpha_token(alpha: float) -> str:
    return f"{float(alpha):g}".replace("-", "m").replace(".", "p")


def _bool_token(value: bool) -> str:
    return "on" if value else "off"


def build_adjustment_specs(
    alphas: tuple[float, ...],
    conditional_options: tuple[bool, ...],
    regime_options: tuple[bool, ...],
) -> tuple[AdjustmentSpec, ...]:
    specs = [
        AdjustmentSpec(
            name="plain",
            alpha=0.0,
            conditional=True,
            regime_strategy=True,
            enabled=False,
            growth_boost_enabled=False,
            decline_cap_enabled=False,
            effect_component="plain",
        ),
        AdjustmentSpec(
            name="decline_cap_only",
            alpha=0.0,
            conditional=True,
            regime_strategy=True,
            enabled=True,
            growth_boost_enabled=False,
            decline_cap_enabled=True,
            effect_component="decline_cap_only",
        ),
        AdjustmentSpec(
            name="decline_cap_balanced",
            alpha=0.0,
            conditional=True,
            regime_strategy=True,
            enabled=True,
            growth_boost_enabled=False,
            decline_cap_enabled=True,
            effect_component="decline_cap_balanced",
            decline_cap_growth_ratio_max=DEFAULT_BALANCED_DECLINE_CAP_GROWTH_RATIO_MAX,
            decline_cap_prediction_ratio_min=DEFAULT_BALANCED_DECLINE_CAP_PREDICTION_RATIO_MIN,
        ),
    ]
    seen = {spec.name for spec in specs}
    for alpha in alphas:
        if np.isclose(float(alpha), 0.0):
            continue
        for conditional in conditional_options:
            for regime_strategy in regime_options:
                for effect_component, growth_boost_enabled, decline_cap_enabled in [
                    ("growth_boost_only", True, False),
                    ("growth_boost_and_decline_cap", True, True),
                ]:
                    name = (
                        f"{effect_component}"
                        f"_alpha_{_alpha_token(alpha)}"
                        f"_cond_{_bool_token(conditional)}"
                        f"_regime_{_bool_token(regime_strategy)}"
                    )
                    if name in seen:
                        continue
                    specs.append(
                        AdjustmentSpec(
                            name=name,
                            alpha=float(alpha),
                            conditional=bool(conditional),
                            regime_strategy=bool(regime_strategy),
                            enabled=True,
                            growth_boost_enabled=bool(growth_boost_enabled),
                            decline_cap_enabled=bool(decline_cap_enabled),
                            effect_component=effect_component,
                        )
                    )
                    seen.add(name)
    return tuple(specs)


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {missing}")

    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    finite_mask = np.ones(len(frame), dtype=bool)
    for column in NUMERIC_REQUIRED_COLUMNS:
        finite_mask &= np.isfinite(frame[column].to_numpy(dtype=float))
    frame = frame[finite_mask].copy()
    if frame.empty:
        raise ValueError("Prediction file has no valid xLSTM prediction rows after numeric cleanup.")

    frame["stock_id"] = frame["stock_id"].astype(int)
    frame["target_year"] = frame["target_year"].astype(int)
    frame["target_month"] = frame["target_month"].astype(int)
    return frame.sort_values(["stock_id", "target_year", "target_month"]).reset_index(drop=True)


def apply_adjustment_variant(frame: pd.DataFrame, spec: AdjustmentSpec) -> pd.DataFrame:
    predicted_plain = frame["predicted_revenue_xlstm"].to_numpy(dtype=float)
    growth_signal = engine.calculate_growth_signal(frame)
    regime = engine.classify_regime(frame)
    is_growth_phase = engine.calculate_growth_phase(frame)
    direction = frame["growth_rate_at_end"].to_numpy(dtype=float)
    positive_signal = np.clip(growth_signal, 0.0, None)
    growth_boost_active = bool(spec.enabled and spec.growth_boost_enabled and not np.isclose(float(spec.alpha), 0.0))
    raw_ratio = 1.0 + (float(spec.alpha) * positive_signal if growth_boost_active else 0.0)

    if growth_boost_active:
        if spec.regime_strategy:
            adjustment_allowed = regime == "growth"
        else:
            adjustment_allowed = np.ones(len(frame), dtype=bool)
        if spec.conditional:
            adjustment_allowed = adjustment_allowed & is_growth_phase
        adjustment_allowed = adjustment_allowed & (direction > 0) & (positive_signal > 0)
    else:
        adjustment_allowed = np.zeros(len(frame), dtype=bool)

    boosted = np.where(adjustment_allowed, predicted_plain * raw_ratio, predicted_plain)
    decline_cap_applied = engine.calculate_decline_cap_mask(
        boosted,
        frame,
        regime,
        enable_regime_strategy=bool(spec.enabled and spec.decline_cap_enabled),
        decline_cap_growth_ratio_max=spec.decline_cap_growth_ratio_max,
        decline_cap_prediction_ratio_min=float(spec.decline_cap_prediction_ratio_min),
    )
    last_observed = frame["last_observed_revenue"].to_numpy(dtype=float)
    adjusted = np.where(
        decline_cap_applied,
        np.minimum(boosted, last_observed),
        boosted,
    )

    if not bool(spec.enabled):
        adjusted = predicted_plain.copy()
        raw_ratio = np.ones(len(frame), dtype=float)
        adjustment_allowed = np.zeros(len(frame), dtype=bool)
        decline_cap_applied = np.zeros(len(frame), dtype=bool)

    guarded, _, _ = engine.apply_revenue_guardrails(adjusted, frame)
    predicted = engine.safe_round_revenue(guarded).astype(float)
    cleaned_adjusted = np.nan_to_num(adjusted, nan=0.0, posinf=0.0, neginf=0.0)
    guardrail_clipped = np.abs(guarded - cleaned_adjusted) > 1e-9
    if not bool(spec.decline_cap_enabled):
        decline_cap_applied = np.zeros(len(frame), dtype=bool)
    adjustment_ratio = np.divide(
        predicted,
        np.where(predicted_plain == 0, np.nan, predicted_plain),
        out=np.ones_like(predicted, dtype=float),
        where=predicted_plain != 0,
    )

    component_effect = np.select(
        [
            adjustment_allowed & decline_cap_applied,
            adjustment_allowed,
            decline_cap_applied,
        ],
        [
            "growth_boost_and_decline_cap",
            "growth_boost_only",
            "decline_cap_only",
        ],
        default="none",
    )

    actual = frame["actual_revenue"].to_numpy(dtype=float)
    error = predicted - actual
    abs_error = np.abs(error)
    ape = np.divide(
        abs_error,
        actual,
        out=np.full_like(abs_error, np.nan, dtype=float),
        where=actual != 0,
    )

    columns = [column for column in METADATA_COLUMNS if column in frame.columns]
    result = frame[columns].copy()
    result.insert(0, "variant", spec.name)
    result.insert(1, "effect_component", spec.effect_component)
    result.insert(2, "alpha", float(spec.alpha))
    result.insert(3, "conditional", bool(spec.conditional))
    result.insert(4, "regime_strategy", bool(spec.regime_strategy))
    result.insert(5, "adjustment_enabled", bool(spec.enabled))
    result.insert(6, "growth_boost_enabled", bool(spec.growth_boost_enabled))
    result.insert(7, "decline_cap_enabled", bool(spec.decline_cap_enabled))
    result.insert(8, "decline_cap_growth_ratio_max", spec.decline_cap_growth_ratio_max)
    result.insert(9, "decline_cap_prediction_ratio_min", float(spec.decline_cap_prediction_ratio_min))
    result["predicted_revenue_plain"] = predicted_plain
    result["predicted_revenue"] = predicted
    result["growth_signal_recomputed"] = growth_signal
    result["regime_recomputed"] = regime
    result["is_growth_phase_recomputed"] = is_growth_phase
    result["raw_adjustment_ratio"] = raw_ratio
    result["adjustment_ratio"] = adjustment_ratio
    result["adjustment_applied"] = adjustment_allowed
    result["decline_cap_applied"] = decline_cap_applied
    result["component_effect_applied"] = component_effect
    result["guardrail_clipped"] = guardrail_clipped
    result["error"] = error
    result["abs_error"] = abs_error
    result["absolute_percentage_error"] = ape * 100
    return result


def build_variant_catalog(specs: tuple[AdjustmentSpec, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant": spec.name,
                "effect_component": spec.effect_component,
                "alpha": float(spec.alpha),
                "conditional": bool(spec.conditional),
                "regime_strategy": bool(spec.regime_strategy),
                "adjustment_enabled": bool(spec.enabled),
                "growth_boost_enabled": bool(spec.growth_boost_enabled),
                "decline_cap_enabled": bool(spec.decline_cap_enabled),
                "decline_cap_growth_ratio_max": spec.decline_cap_growth_ratio_max,
                "decline_cap_prediction_ratio_min": float(spec.decline_cap_prediction_ratio_min),
            }
            for spec in specs
        ]
    )


def attach_variant_catalog(frame: pd.DataFrame, variant_catalog: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        "variant",
        "effect_component",
        "alpha",
        "conditional",
        "regime_strategy",
        "adjustment_enabled",
        "growth_boost_enabled",
        "decline_cap_enabled",
        "decline_cap_growth_ratio_max",
        "decline_cap_prediction_ratio_min",
    ]
    existing_metadata = [column for column in metadata_columns if column in frame.columns and column != "variant"]
    frame = frame.drop(columns=existing_metadata, errors="ignore")
    return frame.merge(variant_catalog[metadata_columns], on="variant", how="left")


def build_component_best_summary(overall_accuracy: pd.DataFrame) -> pd.DataFrame:
    if overall_accuracy.empty or "effect_component" not in overall_accuracy.columns:
        return pd.DataFrame()
    return (
        overall_accuracy.sort_values(["effect_component", "MAPE", "MAE"], ascending=[True, True, True])
        .groupby("effect_component", as_index=False, sort=True)
        .head(1)
        .sort_values(["MAPE", "MAE"])
        .reset_index(drop=True)
    )


def metric_record(frame: pd.DataFrame) -> dict[str, float | int]:
    actual = frame["actual_revenue"].to_numpy(dtype=float)
    predicted = frame["predicted_revenue"].to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    last_observed = (
        frame["last_observed_revenue"].to_numpy(dtype=float) if "last_observed_revenue" in frame.columns else None
    )
    if last_observed is not None:
        valid &= np.isfinite(last_observed)
    if not valid.any():
        return {
            "observations": 0,
            "stock_count": 0,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE": np.nan,
            "MedianAPE": np.nan,
            "WMAPE": np.nan,
            "SMAPE": np.nan,
            "Bias": np.nan,
            "UnderestimateRate": np.nan,
            "DirectionAccuracy": np.nan,
            "AdjustedRate": np.nan,
            "GrowthBoostRate": np.nan,
            "DeclineCapRate": np.nan,
            "GuardrailClipRate": np.nan,
        }

    metrics = engine.compute_metrics(actual, predicted, last_observed)
    return {
        "observations": int(valid.sum()),
        "stock_count": int(frame.loc[valid, "stock_id"].nunique()),
        **metrics,
        "AdjustedRate": float(frame.loc[valid, "adjustment_applied"].mean() * 100),
        "GrowthBoostRate": float(frame.loc[valid, "adjustment_applied"].mean() * 100),
        "DeclineCapRate": float(frame.loc[valid, "decline_cap_applied"].mean() * 100),
        "GuardrailClipRate": float(frame.loc[valid, "guardrail_clipped"].mean() * 100),
    }


def summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in frame.groupby(grouper, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(metric_record(group))
        rows.append(row)
    return pd.DataFrame(rows)


def build_baseline_long_frame(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    id_columns = [column for column in METADATA_COLUMNS if column in frame.columns]
    for model, prediction_column in BASELINE_MODEL_COLUMNS.items():
        if prediction_column not in frame.columns:
            continue
        part = frame[id_columns].copy()
        part.insert(0, "model", model)
        part["predicted_revenue"] = pd.to_numeric(frame[prediction_column], errors="coerce")
        part["adjustment_applied"] = False
        part["decline_cap_applied"] = False
        part["guardrail_clipped"] = False
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_variant_effects(stock_accuracy: pd.DataFrame) -> pd.DataFrame:
    if stock_accuracy.empty:
        return pd.DataFrame()
    plain = stock_accuracy[stock_accuracy["variant"].eq("plain")]
    if plain.empty:
        return pd.DataFrame()
    metrics = [
        "MSE",
        "RMSE",
        "MAPE",
        "MedianAPE",
        "WMAPE",
        "SMAPE",
        "MAE",
        "Bias",
        "UnderestimateRate",
        "DirectionAccuracy",
        "AdjustedRate",
        "GrowthBoostRate",
        "DeclineCapRate",
        "GuardrailClipRate",
    ]
    metrics = [metric for metric in metrics if metric in stock_accuracy.columns]
    plain_columns = ["stock_id", *metrics]
    plain = plain[plain_columns].rename(columns={column: f"{column}_plain" for column in plain_columns if column != "stock_id"})
    effects = stock_accuracy.merge(plain, on="stock_id", how="left")
    for metric in metrics:
        plain_column = f"{metric}_plain"
        if plain_column in effects.columns:
            effects[f"{metric}_delta_vs_plain"] = effects[metric] - effects[plain_column]
    for metric in LOWER_IS_BETTER_METRICS:
        delta_column = f"{metric}_delta_vs_plain"
        if delta_column in effects.columns:
            delta = effects[delta_column]
            effects[f"{metric}_winner"] = np.select(
                [delta < 0, delta > 0],
                ["variant", "plain"],
                default="tie",
            )
    for metric in HIGHER_IS_BETTER_METRICS:
        delta_column = f"{metric}_delta_vs_plain"
        if delta_column in effects.columns:
            delta = effects[delta_column]
            effects[f"{metric}_winner"] = np.select(
                [delta > 0, delta < 0],
                ["variant", "plain"],
                default="tie",
            )
    return effects


def build_winner_summary(variant_effects: pd.DataFrame) -> pd.DataFrame:
    if variant_effects.empty or "MAPE_winner" not in variant_effects.columns:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for variant, frame in variant_effects[~variant_effects["variant"].eq("plain")].groupby("variant", sort=True):
        winner_counts = frame["MAPE_winner"].value_counts()
        stock_count = int(frame["stock_id"].nunique())
        row = {
            "variant": variant,
            "effect_component": frame["effect_component"].iloc[0] if "effect_component" in frame else "",
            "alpha": float(frame["alpha"].iloc[0]) if "alpha" in frame else np.nan,
            "conditional": bool(frame["conditional"].iloc[0]) if "conditional" in frame else False,
            "regime_strategy": bool(frame["regime_strategy"].iloc[0]) if "regime_strategy" in frame else False,
            "growth_boost_enabled": (
                bool(frame["growth_boost_enabled"].iloc[0]) if "growth_boost_enabled" in frame else False
            ),
            "decline_cap_enabled": (
                bool(frame["decline_cap_enabled"].iloc[0]) if "decline_cap_enabled" in frame else False
            ),
            "decline_cap_growth_ratio_max": (
                float(frame["decline_cap_growth_ratio_max"].iloc[0])
                if "decline_cap_growth_ratio_max" in frame and pd.notna(frame["decline_cap_growth_ratio_max"].iloc[0])
                else np.nan
            ),
            "decline_cap_prediction_ratio_min": (
                float(frame["decline_cap_prediction_ratio_min"].iloc[0])
                if "decline_cap_prediction_ratio_min" in frame
                else np.nan
            ),
            "stock_count": stock_count,
            "variant_wins": int(winner_counts.get("variant", 0)),
            "plain_wins": int(winner_counts.get("plain", 0)),
            "ties": int(winner_counts.get("tie", 0)),
            "variant_win_rate": float(winner_counts.get("variant", 0) / len(frame) * 100) if len(frame) else np.nan,
        }
        for winner_metric in ["WMAPE", "MAE", "DirectionAccuracy"]:
            winner_column = f"{winner_metric}_winner"
            if winner_column in frame.columns:
                metric_winner_counts = frame[winner_column].value_counts()
                row[f"{winner_metric}_variant_wins"] = int(metric_winner_counts.get("variant", 0))
                row[f"{winner_metric}_plain_wins"] = int(metric_winner_counts.get("plain", 0))
                row[f"{winner_metric}_ties"] = int(metric_winner_counts.get("tie", 0))
                row[f"{winner_metric}_variant_win_rate"] = (
                    float(metric_winner_counts.get("variant", 0) / len(frame) * 100) if len(frame) else np.nan
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
            "DeclineCapRate",
        ]:
            delta_column = f"{metric}_delta_vs_plain"
            if delta_column in frame.columns:
                row[f"average_{metric}_delta_vs_plain"] = float(frame[delta_column].mean())
                row[f"median_{metric}_delta_vs_plain"] = float(frame[delta_column].median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["average_MAPE_delta_vs_plain", "average_MAE_delta_vs_plain"],
        ascending=[True, True],
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to monthly_predictions.csv with xLSTM columns.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alphas", default=",".join(str(alpha) for alpha in DEFAULT_ALPHAS))
    parser.add_argument("--conditional-options", default="true,false")
    parser.add_argument("--regime-options", default="true,false")
    args = parser.parse_args()

    started = time.time()
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alphas = parse_float_csv(args.alphas)
    conditional_options = parse_bool_csv(args.conditional_options)
    regime_options = parse_bool_csv(args.regime_options, DEFAULT_REGIME_OPTIONS)
    specs = build_adjustment_specs(alphas, conditional_options, regime_options)
    variant_catalog = build_variant_catalog(specs)

    source = load_predictions(predictions_path)
    monthly_predictions = pd.concat(
        [apply_adjustment_variant(source, spec) for spec in specs],
        ignore_index=True,
    )
    stock_accuracy = attach_variant_catalog(summarize(monthly_predictions, ["variant", "stock_id"]), variant_catalog)
    overall_accuracy = attach_variant_catalog(summarize(monthly_predictions, ["variant"]), variant_catalog)
    overall_accuracy = overall_accuracy.sort_values(["MAPE", "MAE"]).reset_index(drop=True)
    variant_effects = build_variant_effects(stock_accuracy)
    winner_summary = build_winner_summary(variant_effects)
    component_best_summary = build_component_best_summary(overall_accuracy)
    regime_accuracy = summarize(monthly_predictions, ["variant", "regime_recomputed"]).sort_values(
        ["variant", "regime_recomputed"]
    )
    regime_accuracy = attach_variant_catalog(regime_accuracy, variant_catalog)

    baseline_long = build_baseline_long_frame(source)
    baseline_overall = summarize(baseline_long, ["model"]) if not baseline_long.empty else pd.DataFrame()
    baseline_stock = summarize(baseline_long, ["model", "stock_id"]) if not baseline_long.empty else pd.DataFrame()

    write_csv(variant_catalog, output_dir / "variant_catalog.csv")
    write_csv(monthly_predictions, output_dir / "monthly_adjustment_predictions.csv")
    write_csv(stock_accuracy, output_dir / "stock_accuracy.csv")
    write_csv(overall_accuracy, output_dir / "overall_accuracy.csv")
    write_csv(variant_effects, output_dir / "variant_effects.csv")
    write_csv(winner_summary, output_dir / "winner_summary.csv")
    write_csv(component_best_summary, output_dir / "component_best_summary.csv")
    write_csv(regime_accuracy, output_dir / "regime_accuracy.csv")
    write_csv(baseline_overall, output_dir / "baseline_overall_accuracy.csv")
    write_csv(baseline_stock, output_dir / "baseline_stock_accuracy.csv")

    completed = time.time()
    run_config = {
        "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(completed)),
        "duration_sec": round(completed - started, 3),
        "predictions": str(predictions_path),
        "alphas": list(alphas),
        "conditional_options": list(conditional_options),
        "regime_options": list(regime_options),
        "variant_count": len(specs),
        "source_rows": int(len(source)),
        "monthly_rows": int(len(monthly_predictions)),
        "component_counts": variant_catalog["effect_component"].value_counts().sort_index().to_dict(),
        "thresholds": {
            "growth_phase_ratio": engine.GROWTH_PHASE_RATIO_THRESHOLD,
            "growth_phase_streak": engine.GROWTH_PHASE_STREAK_THRESHOLD,
            "decline_regime_ratio": engine.DECLINE_REGIME_RATIO_THRESHOLD,
        },
        "metric_notes": {
            "posthoc": "No sequence model is retrained; variants replay separated growth-boost and decline-cap effects on saved plain xLSTM predictions.",
            "MAPE": "mean absolute percentage error across nonzero actual revenue rows",
            "MedianAPE": "median absolute percentage error; less sensitive to low-denominator outliers than MAPE",
            "WMAPE": "sum absolute error divided by sum absolute actual revenue",
            "SMAPE": "symmetric MAPE: mean 2*abs(error)/(abs(actual)+abs(predicted))",
            "actual_usage": "2025 actual revenue is used only for evaluation metrics after variant predictions are produced.",
        },
    }
    run_config = write_rolling_run_config(
        output_dir,
        run_config,
        experiment_family="rolling_xlstm_adjustment_ablation",
        evidence_tier="C",
        selection_protocol="target-year-hindsight",
        report_ready=False,
        report_ready_reason="Adjustment variants are replayed and ranked on target-year actuals.",
    )

    summary = {
        **run_config,
        "best_variants_by_MAPE": overall_accuracy.head(8).to_dict(orient="records"),
        "best_components_by_MAPE": component_best_summary.to_dict(orient="records"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
