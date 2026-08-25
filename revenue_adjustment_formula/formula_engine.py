from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SYSTEM_DIR.parent
REVENUE_PATH = PROJECT_ROOT / "data" / "Stock_revenue_2019~2025.csv"
EPSILON = 1e-6
MIN_FORMULA_HISTORY = 15


@dataclass(frozen=True)
class FormulaConfig:
    seasonal_weight: float = 0.75
    residual_alpha: float = 0.20
    residual_strength: float = 0.50
    growth_log_cap: float = math.log(2.0)
    correction_log_cap: float = 0.50

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def load_revenue_data(path: str | Path | None = None) -> pd.DataFrame:
    revenue_path = Path(path) if path is not None else REVENUE_PATH
    frame = pd.read_csv(revenue_path)
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")].copy()
    if "revenue_thousand" not in frame.columns:
        if "revenue" not in frame.columns:
            raise ValueError("Revenue data must include revenue_thousand or revenue.")
        frame["revenue_thousand"] = _to_numeric(frame["revenue"]) / 1000.0

    for column in ["stock_id", "revenue_year", "revenue_month", "revenue_thousand"]:
        frame[column] = _to_numeric(frame[column])
    frame = frame.dropna(
        subset=["stock_id", "revenue_year", "revenue_month", "revenue_thousand"]
    )
    frame = frame[
        np.isfinite(frame["revenue_thousand"]) & (frame["revenue_thousand"] >= 0)
    ].copy()
    frame["stock_id"] = frame["stock_id"].astype(int)
    frame["revenue_year"] = frame["revenue_year"].astype(int)
    frame["revenue_month"] = frame["revenue_month"].astype(int)
    frame["date"] = pd.to_datetime(
        frame["revenue_year"].astype(str)
        + "-"
        + frame["revenue_month"].astype(str).str.zfill(2)
        + "-01"
    )
    return (
        frame.sort_values(["stock_id", "date"])
        .drop_duplicates(["stock_id", "date"], keep="last")
        .reset_index(drop=True)
    )


def _to_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )


def _formula_base(
    history: np.ndarray,
    config: FormulaConfig,
) -> tuple[float, float, float]:
    values = np.asarray(history, dtype=float)
    if len(values) < MIN_FORMULA_HISTORY:
        raise ValueError(f"Formula requires at least {MIN_FORMULA_HISTORY} months.")

    recent = values[-3:]
    prior_year = values[-15:-12]
    yoy_log_growth = np.log1p(recent) - np.log1p(prior_year)
    growth_log = float(
        np.clip(
            np.nanmedian(yoy_log_growth),
            -float(config.growth_log_cap),
            float(config.growth_log_cap),
        )
    )
    seasonal_anchor = float(values[-12])
    seasonal_growth_forecast = float(
        np.expm1(np.log1p(seasonal_anchor) + growth_log)
    )
    current_level = float(values[-1])
    base_log = (
        float(config.seasonal_weight) * np.log1p(max(seasonal_growth_forecast, 0.0))
        + (1.0 - float(config.seasonal_weight)) * np.log1p(current_level)
    )
    base = float(max(np.expm1(base_log), 0.0))
    return base, seasonal_growth_forecast, growth_log


def _seasonal_mase_scale(history: np.ndarray) -> float:
    values = np.asarray(history, dtype=float)
    if len(values) >= 24:
        differences = np.abs(values[12:] - values[:-12])
    elif len(values) >= 2:
        differences = np.abs(np.diff(values))
    else:
        return np.nan
    finite = differences[np.isfinite(differences)]
    scale = float(np.mean(finite)) if len(finite) else np.nan
    return scale if np.isfinite(scale) and scale > 0 else np.nan


def _segment_ids(stock_frame: pd.DataFrame) -> pd.Series:
    month_index = stock_frame["revenue_year"] * 12 + stock_frame["revenue_month"] - 1
    return month_index.diff().ne(1).cumsum()


def _predict_segment(
    stock_id: int,
    segment: pd.DataFrame,
    config: FormulaConfig,
) -> list[dict[str, object]]:
    segment = segment.sort_values("date").reset_index(drop=True)
    values = segment["revenue_thousand"].to_numpy(dtype=float)
    dates = segment["date"].to_numpy()
    correction_state = 0.0
    rows: list[dict[str, object]] = []

    for index in range(len(values)):
        target_date = pd.Timestamp(dates[index])
        history = values[:index]
        actual = float(values[index])
        last_observed = float(history[-1]) if len(history) else np.nan
        seasonal_naive = float(history[-12]) if len(history) >= 12 else np.nan
        base = np.nan
        adjusted = np.nan
        seasonal_growth = np.nan
        growth_log = np.nan
        applied_correction = np.nan
        method = "missing_prediction"

        if len(history) >= MIN_FORMULA_HISTORY:
            base, seasonal_growth, growth_log = _formula_base(history, config)
            applied_correction = float(
                np.clip(
                    float(config.residual_strength) * correction_state,
                    -float(config.correction_log_cap),
                    float(config.correction_log_cap),
                )
            )
            adjusted = float(
                max(np.expm1(np.log1p(base) + applied_correction), 0.0)
            )
            method = "revenue_adjustment_formula"
        elif len(history) >= 12:
            base = seasonal_naive
            adjusted = seasonal_naive
            applied_correction = 0.0
            method = "seasonal_naive_fallback"
        elif len(history):
            base = last_observed
            adjusted = last_observed
            applied_correction = 0.0
            method = "last_observed_fallback"

        rows.append(
            {
                "stock_id": int(stock_id),
                "target_date": target_date,
                "target_year": int(target_date.year),
                "target_month": int(target_date.month),
                "history_months": int(len(history)),
                "actual_revenue": actual,
                "last_observed_revenue": last_observed,
                "seasonal_naive_revenue": seasonal_naive,
                "formula_base_revenue": base,
                "formula_adjusted_revenue": adjusted,
                "seasonal_growth_forecast": seasonal_growth,
                "growth_log": growth_log,
                "residual_state": correction_state if len(history) >= MIN_FORMULA_HISTORY else np.nan,
                "applied_correction_log": applied_correction,
                "mase_scale": _seasonal_mase_scale(history),
                "forecast_method": method,
            }
        )

        if len(history) >= MIN_FORMULA_HISTORY and np.isfinite(base):
            residual = float(np.log1p(actual) - np.log1p(base))
            correction_state = (
                float(config.residual_alpha) * residual
                + (1.0 - float(config.residual_alpha)) * correction_state
            )

    return rows


def build_rolling_predictions(
    revenue_data: pd.DataFrame,
    config: FormulaConfig,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    stock_ids: list[int] | None = None,
) -> pd.DataFrame:
    frame = revenue_data.copy()
    if stock_ids is not None:
        selected = {int(stock_id) for stock_id in stock_ids}
        frame = frame[frame["stock_id"].isin(selected)].copy()

    rows: list[dict[str, object]] = []
    for stock_id, stock_frame in frame.groupby("stock_id", sort=True):
        stock_frame = stock_frame.sort_values("date").reset_index(drop=True)
        segment_ids = _segment_ids(stock_frame)
        for _, segment in stock_frame.groupby(segment_ids, sort=False):
            rows.extend(_predict_segment(int(stock_id), segment, config))

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        return predictions
    if start_date is not None:
        predictions = predictions[predictions["target_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        predictions = predictions[predictions["target_date"] <= pd.Timestamp(end_date)]
    return predictions.sort_values(["stock_id", "target_date"]).reset_index(drop=True)


def compute_metrics(
    predictions: pd.DataFrame,
    prediction_column: str,
) -> dict[str, float]:
    valid = predictions.dropna(
        subset=["actual_revenue", prediction_column]
    ).copy()
    if valid.empty:
        return {
            "observations": 0,
            "stock_count": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE": np.nan,
            "MedianAPE": np.nan,
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
    abs_error = np.abs(error)
    ape = np.divide(
        abs_error,
        np.abs(actual),
        out=np.full_like(abs_error, np.nan),
        where=actual != 0,
    )
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.divide(
        2.0 * abs_error,
        denominator,
        out=np.full_like(abs_error, np.nan),
        where=denominator != 0,
    )
    mase_scale = valid["mase_scale"].to_numpy(dtype=float)
    scaled_error = np.divide(
        abs_error,
        mase_scale,
        out=np.full_like(abs_error, np.nan),
        where=np.isfinite(mase_scale) & (mase_scale > 0),
    )
    finite_scaled_error = scaled_error[np.isfinite(scaled_error)]
    last_observed = valid["last_observed_revenue"].to_numpy(dtype=float)
    direction_valid = np.isfinite(last_observed)
    direction_accuracy = (
        float(
            np.mean(
                np.sign(predicted[direction_valid] - last_observed[direction_valid])
                == np.sign(actual[direction_valid] - last_observed[direction_valid])
            )
            * 100
        )
        if direction_valid.any()
        else np.nan
    )
    return {
        "observations": int(len(valid)),
        "stock_count": int(valid["stock_id"].nunique()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(abs_error)),
        "MAPE": float(np.nanmean(ape) * 100),
        "MedianAPE": float(np.nanmedian(ape) * 100),
        "WMAPE": float(abs_error.sum() / np.abs(actual).sum() * 100),
        "SMAPE": float(np.nanmean(smape) * 100),
        "MASE": (
            float(np.mean(finite_scaled_error))
            if len(finite_scaled_error)
            else np.nan
        ),
        "Bias": float(np.mean(error)),
        "UnderestimateRate": float(np.mean(predicted < actual) * 100),
        "DirectionAccuracy": direction_accuracy,
    }


def stock_wmape(
    predictions: pd.DataFrame,
    prediction_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stock_id, group in predictions.groupby("stock_id"):
        valid = group.dropna(subset=["actual_revenue", prediction_column])
        if valid.empty:
            continue
        denominator = valid["actual_revenue"].abs().sum()
        wmape = (
            float(
                (valid[prediction_column] - valid["actual_revenue"]).abs().sum()
                / denominator
                * 100
            )
            if denominator > 0
            else np.nan
        )
        rows.append(
            {
                "stock_id": int(stock_id),
                "observations": int(len(valid)),
                "WMAPE": wmape,
            }
        )
    return pd.DataFrame(rows)
