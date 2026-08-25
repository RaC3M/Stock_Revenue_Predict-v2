from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


EPSILON = 1.0


@dataclass(frozen=True)
class StructuralBreakConfig:
    yoy_ratio_threshold: float = 0.10
    level_ratio_threshold: float = 0.20
    formula_retention: float = 0.10
    recent_ratio_threshold: float = 0.20
    required_recent_breaks: int = 2

    def __post_init__(self) -> None:
        for name in [
            "yoy_ratio_threshold",
            "level_ratio_threshold",
            "formula_retention",
            "recent_ratio_threshold",
        ]:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if int(self.required_recent_breaks) < 1:
            raise ValueError("required_recent_breaks must be positive.")

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan
    return float((max(numerator, 0.0) + EPSILON) / (max(denominator, 0.0) + EPSILON))


def add_structural_break_features(
    formula_predictions: pd.DataFrame,
    revenue_data: pd.DataFrame,
) -> pd.DataFrame:
    predictions = formula_predictions.copy()
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    revenue = revenue_data[["stock_id", "date", "revenue_thousand"]].copy()
    revenue["date"] = pd.to_datetime(revenue["date"])
    revenue["revenue_thousand"] = pd.to_numeric(revenue["revenue_thousand"], errors="coerce")

    features: list[dict[str, object]] = []
    for stock_id, stock_predictions in predictions.groupby("stock_id", sort=False):
        stock_revenue = revenue[revenue["stock_id"].astype(int).eq(int(stock_id))]
        value_by_date = {
            pd.Timestamp(row.date): float(row.revenue_thousand)
            for row in stock_revenue.itertuples()
            if np.isfinite(row.revenue_thousand)
        }
        for index, row in stock_predictions.iterrows():
            target = pd.Timestamp(row["target_date"])
            last_date = target - pd.DateOffset(months=1)
            last_year_date = target - pd.DateOffset(months=13)
            last_value = value_by_date.get(last_date, np.nan)
            last_year_value = value_by_date.get(last_year_date, np.nan)
            last_yoy_ratio = _ratio(last_value, last_year_value)

            trailing_values = [
                value_by_date.get(target - pd.DateOffset(months=offset), np.nan)
                for offset in range(1, 13)
            ]
            finite_trailing = np.asarray(trailing_values, dtype=float)
            finite_trailing = finite_trailing[np.isfinite(finite_trailing)]
            trailing_median = (
                float(np.median(finite_trailing)) if len(finite_trailing) else np.nan
            )
            level_ratio = _ratio(last_value, trailing_median)

            recent_ratios: list[float] = []
            for offset in range(1, 4):
                recent_value = value_by_date.get(
                    target - pd.DateOffset(months=offset), np.nan
                )
                prior_value = value_by_date.get(
                    target - pd.DateOffset(months=offset + 12), np.nan
                )
                recent_ratios.append(_ratio(recent_value, prior_value))
            features.append(
                {
                    "_row_index": index,
                    "break_last_yoy_ratio": last_yoy_ratio,
                    "break_level_ratio": level_ratio,
                    "break_recent_ratio_1": recent_ratios[0],
                    "break_recent_ratio_2": recent_ratios[1],
                    "break_recent_ratio_3": recent_ratios[2],
                }
            )

    feature_frame = pd.DataFrame(features).set_index("_row_index")
    for column in feature_frame.columns:
        predictions[column] = feature_frame[column]
    return predictions


def apply_structural_break_adjustment(
    featured_predictions: pd.DataFrame,
    config: StructuralBreakConfig,
) -> pd.DataFrame:
    output = featured_predictions.copy()
    recent_columns = [
        "break_recent_ratio_1",
        "break_recent_ratio_2",
        "break_recent_ratio_3",
    ]
    recent_break_count = (
        output[recent_columns]
        .lt(float(config.recent_ratio_threshold))
        .sum(axis=1)
    )
    last_yoy_break = output["break_last_yoy_ratio"].lt(
        float(config.yoy_ratio_threshold)
    )
    level_break = output["break_level_ratio"].lt(
        float(config.level_ratio_threshold)
    )
    confirmed_recent_break = recent_break_count.ge(
        int(config.required_recent_breaks)
    )
    detected = last_yoy_break & (level_break | confirmed_recent_break)

    original = pd.to_numeric(output["formula_adjusted_revenue"], errors="coerce")
    last = pd.to_numeric(output["last_observed_revenue"], errors="coerce")
    retention = float(config.formula_retention)
    break_prediction = retention * original + (1.0 - retention) * last
    valid_break = detected & np.isfinite(last) & (last >= 0)
    output["structural_break_detected"] = valid_break
    output["recent_break_count"] = recent_break_count.astype(int)
    output["original_formula_revenue"] = original
    output["formula_adjusted_revenue"] = np.where(
        valid_break,
        np.maximum(break_prediction, 0.0),
        original,
    )
    output["structural_formula_method"] = np.where(
        valid_break,
        "structural_break_recent_level",
        "original_formula",
    )
    return output
