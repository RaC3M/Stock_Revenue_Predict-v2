from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


INT64_GUARD = float(np.iinfo(np.int64).max) * 0.99
SARIMA_UPPER_MULTIPLIER = 2.0
MERGE_KEYS = ["stock_id", "target_date", "target_year", "target_month"]


@dataclass(frozen=True)
class HybridConfig:
    sarima_weight: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.sarima_weight) <= 1.0:
            raise ValueError("sarima_weight must be between 0 and 1.")


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _prepare_formula(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        *MERGE_KEYS,
        "actual_revenue",
        "last_observed_revenue",
        "formula_adjusted_revenue",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Formula predictions are missing columns: {missing}")
    columns = [
        *MERGE_KEYS,
        "actual_revenue",
        "last_observed_revenue",
        "formula_adjusted_revenue",
        "mase_scale",
        "forecast_method",
    ]
    columns = [column for column in columns if column in frame.columns]
    output = frame[columns].copy()
    output = output.rename(columns={"forecast_method": "formula_forecast_method"})
    output["target_date"] = pd.to_datetime(output["target_date"])
    return output


def _prepare_sarima(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*MERGE_KEYS, "predicted_revenue_sarima"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"SARIMA predictions are missing columns: {missing}")
    columns = [
        *MERGE_KEYS,
        "actual_revenue",
        "last_observed_revenue",
        "predicted_revenue_sarima",
        "forecast_method",
        "fallback_reason",
        "numeric_valid",
        "selected_order",
        "selected_seasonal_order",
    ]
    columns = [column for column in columns if column in frame.columns]
    output = frame[columns].copy()
    output = output.rename(
        columns={
            "actual_revenue": "sarima_actual_revenue",
            "last_observed_revenue": "sarima_last_observed_revenue",
            "forecast_method": "sarima_forecast_method",
        }
    )
    output["target_date"] = pd.to_datetime(output["target_date"])
    return output


def combine_predictions(
    formula_predictions: pd.DataFrame,
    sarima_predictions: pd.DataFrame,
    config: HybridConfig | None = None,
) -> pd.DataFrame:
    config = config or HybridConfig()
    formula = _prepare_formula(formula_predictions)
    sarima = _prepare_sarima(sarima_predictions)
    merged = formula.merge(sarima, on=MERGE_KEYS, how="outer", validate="one_to_one")

    actual = _numeric(merged.get("actual_revenue", pd.Series(np.nan, index=merged.index)))
    sarima_actual = _numeric(
        merged.get("sarima_actual_revenue", pd.Series(np.nan, index=merged.index))
    )
    merged["actual_revenue"] = actual.fillna(sarima_actual)

    last = _numeric(
        merged.get("last_observed_revenue", pd.Series(np.nan, index=merged.index))
    )
    sarima_last = _numeric(
        merged.get("sarima_last_observed_revenue", pd.Series(np.nan, index=merged.index))
    )
    merged["last_observed_revenue"] = last.fillna(sarima_last)

    formula_value = _numeric(merged["formula_adjusted_revenue"])
    sarima_value = _numeric(merged["predicted_revenue_sarima"])
    last_value = _numeric(merged["last_observed_revenue"])
    formula_valid = np.isfinite(formula_value) & (formula_value >= 0)
    sarima_valid = np.isfinite(sarima_value) & (sarima_value >= 0) & (sarima_value < INT64_GUARD)
    if "numeric_valid" in merged.columns:
        reported_valid = merged["numeric_valid"].astype(str).str.lower().isin({"true", "1"})
        sarima_valid &= reported_valid
    if "sarima_forecast_method" in merged.columns:
        sarima_valid &= merged["sarima_forecast_method"].eq("sarima")
    guardrail_anchor = np.maximum(
        np.where(formula_valid, formula_value, 0.0),
        np.where(np.isfinite(last_value) & (last_value >= 0), last_value, 0.0),
    )
    guardrail_upper = SARIMA_UPPER_MULTIPLIER * np.maximum(guardrail_anchor, 1.0)
    sarima_within_guardrail = sarima_value <= guardrail_upper
    sarima_valid &= sarima_within_guardrail

    weight = float(config.sarima_weight)
    both = formula_valid & sarima_valid
    prediction = pd.Series(np.nan, index=merged.index, dtype=float)
    prediction.loc[both] = (
        weight * sarima_value.loc[both]
        + (1.0 - weight) * formula_value.loc[both]
    )
    prediction.loc[formula_valid & ~sarima_valid] = formula_value.loc[
        formula_valid & ~sarima_valid
    ]
    prediction.loc[sarima_valid & ~formula_valid] = sarima_value.loc[
        sarima_valid & ~formula_valid
    ]

    merged["formula_valid"] = formula_valid
    merged["sarima_valid"] = sarima_valid
    merged["sarima_guardrail_upper"] = guardrail_upper
    merged["sarima_within_guardrail"] = sarima_within_guardrail
    merged["both_models_valid"] = both
    merged["sarima_weight"] = weight
    merged["hybrid_predicted_revenue"] = prediction.clip(lower=0)
    merged["hybrid_method"] = np.select(
        [both, formula_valid & ~sarima_valid, sarima_valid & ~formula_valid],
        ["weighted_hybrid", "formula_fallback", "sarima_fallback"],
        default="unavailable",
    )
    merged["hybrid_error"] = merged["hybrid_predicted_revenue"] - merged["actual_revenue"]
    merged["hybrid_abs_error"] = merged["hybrid_error"].abs()
    merged["hybrid_ape"] = np.divide(
        merged["hybrid_abs_error"],
        merged["actual_revenue"].abs(),
        out=np.full(len(merged), np.nan, dtype=float),
        where=merged["actual_revenue"].to_numpy(dtype=float) != 0,
    ) * 100
    return merged.sort_values(MERGE_KEYS).reset_index(drop=True)


def compute_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    valid = frame.dropna(subset=["actual_revenue", prediction_column]).copy()
    if valid.empty:
        return {
            "observations": 0,
            "stock_count": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE": np.nan,
            "MedianAPE": np.nan,
            "P90APE": np.nan,
            "WMAPE": np.nan,
            "SMAPE": np.nan,
            "MASE": np.nan,
            "Bias": np.nan,
            "UnderestimateRate": np.nan,
            "DirectionAccuracy": np.nan,
        }
    actual = valid["actual_revenue"].to_numpy(dtype=float)
    predicted = valid[prediction_column].to_numpy(dtype=float)
    error = predicted - actual
    absolute_error = np.abs(error)
    ape = np.divide(
        absolute_error,
        np.abs(actual),
        out=np.full_like(absolute_error, np.nan),
        where=actual != 0,
    )
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.divide(
        2.0 * absolute_error,
        denominator,
        out=np.full_like(absolute_error, np.nan),
        where=denominator != 0,
    )
    mase = np.array([], dtype=float)
    if "mase_scale" in valid.columns:
        scale = valid["mase_scale"].to_numpy(dtype=float)
        mase = np.divide(
            absolute_error,
            scale,
            out=np.full_like(absolute_error, np.nan),
            where=np.isfinite(scale) & (scale > 0),
        )
        mase = mase[np.isfinite(mase)]
    last = valid.get("last_observed_revenue", pd.Series(np.nan, index=valid.index)).to_numpy(dtype=float)
    direction_rows = np.isfinite(last)
    direction = (
        float(
            np.mean(
                np.sign(predicted[direction_rows] - last[direction_rows])
                == np.sign(actual[direction_rows] - last[direction_rows])
            )
            * 100
        )
        if direction_rows.any()
        else np.nan
    )
    return {
        "observations": int(len(valid)),
        "stock_count": int(valid["stock_id"].nunique()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(absolute_error)),
        "MAPE": float(np.nanmean(ape) * 100),
        "MedianAPE": float(np.nanmedian(ape) * 100),
        "P90APE": float(np.nanpercentile(ape, 90) * 100),
        "WMAPE": float(absolute_error.sum() / np.abs(actual).sum() * 100),
        "SMAPE": float(np.nanmean(smape) * 100),
        "MASE": float(np.mean(mase)) if len(mase) else np.nan,
        "Bias": float(np.mean(error)),
        "UnderestimateRate": float(np.mean(predicted < actual) * 100),
        "DirectionAccuracy": direction,
    }


def stock_metrics(frame: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stock_id, group in frame.groupby("stock_id", sort=True):
        metrics = compute_metrics(group, prediction_column)
        rows.append({"stock_id": int(stock_id), **metrics})
    return pd.DataFrame(rows)


def search_sarima_weight(
    formula_predictions: pd.DataFrame,
    sarima_predictions: pd.DataFrame,
    weights: tuple[float, ...] = tuple(np.round(np.arange(0.0, 1.01, 0.1), 2)),
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for weight in weights:
        combined = combine_predictions(
            formula_predictions,
            sarima_predictions,
            HybridConfig(sarima_weight=float(weight)),
        )
        common = combined[combined["both_models_valid"]].copy()
        pooled = compute_metrics(common, "hybrid_predicted_revenue")
        per_stock = stock_metrics(common, "hybrid_predicted_revenue")
        median_stock_wmape = (
            float(per_stock["WMAPE"].median()) if not per_stock.empty else np.nan
        )
        balanced_score = (
            0.5 * float(pooled["WMAPE"]) + 0.5 * median_stock_wmape
            if np.isfinite(pooled["WMAPE"]) and np.isfinite(median_stock_wmape)
            else np.nan
        )
        rows.append(
            {
                "sarima_weight": float(weight),
                "formula_weight": 1.0 - float(weight),
                "balanced_score": balanced_score,
                "median_stock_WMAPE": median_stock_wmape,
                **pooled,
            }
        )
    sweep = pd.DataFrame(rows).sort_values("sarima_weight").reset_index(drop=True)
    valid = sweep[np.isfinite(sweep["balanced_score"])].copy()
    if valid.empty:
        raise ValueError("No valid common validation rows were available for weight search.")
    best = valid.sort_values(
        ["balanced_score", "WMAPE", "sarima_weight"], ascending=[True, True, True]
    ).iloc[0]
    return float(best["sarima_weight"]), sweep


def model_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    mappings = {
        "營收調整公式": "formula_adjusted_revenue",
        "SARIMA": "predicted_revenue_sarima",
        "SARIMA＋營收公式": "hybrid_predicted_revenue",
    }
    rows = []
    for model, column in mappings.items():
        model_frame = frame
        if model == "SARIMA":
            model_frame = frame[frame["sarima_valid"]]
        rows.append({"model": model, **compute_metrics(model_frame, column)})
    return pd.DataFrame(rows)
