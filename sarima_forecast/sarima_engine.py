from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SYSTEM_DIR.parent
DATA_DIR_ENV_VAR = "PREDICT_DATA_DIR"
REVENUE_FILENAME = "Stock_revenue_2019~2025.csv"
FORECAST_YEAR = 2025
SEASONAL_PERIOD = 12
MIN_HISTORY_MONTHS = 36

SARIMA_ORDERS = (
    (0, 1, 1),
    (1, 1, 0),
    (1, 1, 1),
)
SARIMA_SEASONAL_ORDERS = (
    (0, 1, 1, SEASONAL_PERIOD),
    (1, 1, 0, SEASONAL_PERIOD),
    (0, 1, 0, SEASONAL_PERIOD),
)


def _resolve_data_dir(raw_path: str | os.PathLike[str] | None = None) -> Path:
    raw_path = os.environ.get(DATA_DIR_ENV_VAR) if raw_path is None else raw_path
    if raw_path:
        path = Path(raw_path)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return PROJECT_ROOT / "data"


DATA_DIR = _resolve_data_dir()


@dataclass(frozen=True)
class SarimaConfig:
    forecast_year: int = FORECAST_YEAR
    min_history_months: int = MIN_HISTORY_MONTHS
    confidence_level: float = 0.95
    maxiter: int = 100


@dataclass
class SarimaResult:
    forecast: pd.DataFrame
    metrics: pd.DataFrame
    order_search: pd.DataFrame
    selected_order: tuple[int, int, int] | None
    selected_seasonal_order: tuple[int, int, int, int] | None
    notes: list[str]


def load_revenue_data(path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    revenue_path = Path(path) if path is not None else DATA_DIR / REVENUE_FILENAME
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
        values.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def get_stock_list(frame: pd.DataFrame) -> list[int]:
    return sorted(frame["stock_id"].dropna().astype(int).unique().tolist())


def _trailing_consecutive_history(
    stock_frame: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    history = stock_frame[stock_frame["date"] < pd.Timestamp(target_date)].copy()
    history = history.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if history.empty:
        return history
    month_index = history["revenue_year"] * 12 + history["revenue_month"] - 1
    segment = month_index.diff().ne(1).cumsum()
    return history[segment.eq(segment.iloc[-1])].reset_index(drop=True)


def _fit_sarima(
    log_revenue: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    maxiter: int,
):
    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as error:
        raise ImportError(
            "SARIMA requires statsmodels. Install sarima_forecast/requirements.txt."
        ) from error

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        model = SARIMAX(
            log_revenue,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=int(maxiter))


def select_sarima_order(
    revenue: np.ndarray,
    maxiter: int = 100,
    min_history_months: int = MIN_HISTORY_MONTHS,
) -> tuple[
    tuple[int, int, int] | None,
    tuple[int, int, int, int] | None,
    pd.DataFrame,
]:
    values = np.asarray(revenue, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    rows: list[dict[str, object]] = []
    if len(values) < int(min_history_months):
        return None, None, pd.DataFrame(rows)

    log_revenue = np.log1p(values)
    for order in SARIMA_ORDERS:
        for seasonal_order in SARIMA_SEASONAL_ORDERS:
            row: dict[str, object] = {
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "aic": np.nan,
                "converged": False,
                "status": "fit_failed",
            }
            try:
                fitted = _fit_sarima(log_revenue, order, seasonal_order, maxiter)
                converged = bool(getattr(fitted, "mle_retvals", {}).get("converged", True))
                aic = float(fitted.aic)
                row.update(
                    {
                        "aic": aic,
                        "converged": converged,
                        "status": "ok" if converged and np.isfinite(aic) else "not_converged",
                    }
                )
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
                row["error"] = str(error)
            rows.append(row)

    search = pd.DataFrame(rows).sort_values(
        ["status", "aic"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    successful = search[search["status"].eq("ok") & np.isfinite(search["aic"])]
    if successful.empty:
        return None, None, search
    best = successful.iloc[0]
    selected_order = _parse_order(str(best["order"]), expected_length=3)
    selected_seasonal_order = _parse_order(str(best["seasonal_order"]), expected_length=4)
    return selected_order, selected_seasonal_order, search


def _parse_order(value: str, expected_length: int) -> tuple[int, ...]:
    parts = [int(part.strip()) for part in value.strip("() ").split(",")]
    if len(parts) != expected_length:
        raise ValueError(f"Invalid SARIMA order: {value!r}")
    return tuple(parts)


def forecast_sarima_one_step(
    revenue: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    confidence_level: float = 0.95,
    maxiter: int = 100,
) -> tuple[float, float, float]:
    values = np.asarray(revenue, dtype=float)
    fitted = _fit_sarima(np.log1p(values), order, seasonal_order, maxiter)
    prediction = fitted.get_forecast(steps=1)
    mean_log = float(np.asarray(prediction.predicted_mean).reshape(-1)[0])
    alpha = 1.0 - float(confidence_level)
    interval = np.asarray(prediction.conf_int(alpha=alpha), dtype=float).reshape(-1, 2)[0]
    max_safe_log = float(np.log1p(np.iinfo(np.int64).max - 1))
    point = float(np.expm1(np.clip(mean_log, 0.0, max_safe_log)))
    lower = float(np.expm1(np.clip(interval[0], 0.0, max_safe_log)))
    upper = float(np.expm1(np.clip(interval[1], 0.0, max_safe_log)))
    return point, lower, upper


def _seasonal_naive(history: pd.DataFrame) -> float:
    values = history["revenue_thousand"].to_numpy(dtype=float)
    if len(values) >= SEASONAL_PERIOD:
        return float(values[-SEASONAL_PERIOD])
    if len(values):
        return float(values[-1])
    return np.nan


def build_rolling_sarima_forecast(
    revenue_data: pd.DataFrame,
    selected_stock: int,
    config: SarimaConfig | None = None,
) -> SarimaResult:
    config = config or SarimaConfig()
    stock_frame = revenue_data[
        revenue_data["stock_id"].astype(int).eq(int(selected_stock))
    ].sort_values("date").reset_index(drop=True)
    if stock_frame.empty:
        raise ValueError(f"Stock {selected_stock} is not available in the revenue data.")

    forecast_start = pd.Timestamp(year=int(config.forecast_year), month=1, day=1)
    selection_history = _trailing_consecutive_history(stock_frame, forecast_start)
    if len(selection_history) >= int(config.min_history_months):
        selected_order, selected_seasonal_order, order_search = select_sarima_order(
            selection_history["revenue_thousand"].to_numpy(dtype=float),
            maxiter=int(config.maxiter),
            min_history_months=int(config.min_history_months),
        )
    else:
        selected_order, selected_seasonal_order = None, None
        order_search = pd.DataFrame()

    forecast_rows: list[dict[str, object]] = []
    target_dates = pd.date_range(forecast_start, periods=12, freq="MS")
    for target_date in target_dates:
        history = _trailing_consecutive_history(stock_frame, target_date)
        previous_month = target_date - pd.offsets.MonthBegin(1)
        has_previous_month = bool(not history.empty and history.iloc[-1]["date"] == previous_month)
        fallback_prediction = _seasonal_naive(history)
        point = fallback_prediction
        lower = np.nan
        upper = np.nan
        status = "seasonal_naive_fallback"
        error_message = ""

        can_fit = bool(
            selected_order is not None
            and selected_seasonal_order is not None
            and len(history) >= int(config.min_history_months)
            and has_previous_month
        )
        if can_fit:
            try:
                point, lower, upper = forecast_sarima_one_step(
                    history["revenue_thousand"].to_numpy(dtype=float),
                    selected_order,
                    selected_seasonal_order,
                    confidence_level=float(config.confidence_level),
                    maxiter=int(config.maxiter),
                )
                status = "sarima"
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
                error_message = str(error)
        elif not has_previous_month:
            error_message = "Previous calendar month is unavailable."
        elif len(history) < int(config.min_history_months):
            error_message = "Insufficient consecutive history."
        else:
            error_message = "No converged SARIMA candidate was available."

        last_observed = (
            float(history.iloc[-1]["revenue_thousand"])
            if has_previous_month
            else np.nan
        )
        forecast_rows.append(
            {
                "stock_id": int(selected_stock),
                "target_date": pd.Timestamp(target_date),
                "target_year": int(target_date.year),
                "target_month": int(target_date.month),
                "last_observed_revenue": last_observed,
                "history_months": int(len(history)),
                "predicted_revenue_sarima": point,
                "sarima_lower": lower,
                "sarima_upper": upper,
                "forecast_method": status,
                "fallback_reason": error_message,
            }
        )

    forecast = pd.DataFrame(forecast_rows)
    actual = stock_frame[
        stock_frame["revenue_year"].astype(int).eq(int(config.forecast_year))
    ][["date", "revenue_thousand"]].rename(
        columns={"date": "target_date", "revenue_thousand": "actual_revenue"}
    )
    forecast = forecast.merge(actual, on="target_date", how="left")
    forecast["error"] = forecast["predicted_revenue_sarima"] - forecast["actual_revenue"]
    forecast["abs_error"] = forecast["error"].abs()
    forecast["absolute_percentage_error"] = np.divide(
        forecast["abs_error"],
        forecast["actual_revenue"].abs(),
        out=np.full(len(forecast), np.nan, dtype=float),
        where=forecast["actual_revenue"].to_numpy(dtype=float) != 0,
    ) * 100
    metrics = pd.DataFrame([{"model": "Rolling SARIMA", **compute_metrics(forecast)}])
    metric_columns = [column for column in metrics.columns if column != "model"]
    metrics[metric_columns] = metrics[metric_columns].round(3)

    notes = [
        f"SARIMA order selection used only consecutive data ending before {config.forecast_year}-01.",
        "Each target month is a one-step rolling replay using only revenue dated before that target month.",
        "Target-year actual revenue is merged only after predictions are produced for evaluation.",
        "Rows with insufficient history or failed convergence use a clearly labelled prior-year seasonal fallback.",
    ]
    return SarimaResult(
        forecast=forecast,
        metrics=metrics,
        order_search=order_search,
        selected_order=selected_order,
        selected_seasonal_order=selected_seasonal_order,
        notes=notes,
    )


def compute_metrics(forecast: pd.DataFrame) -> dict[str, float]:
    valid = forecast.dropna(
        subset=["actual_revenue", "predicted_revenue_sarima"]
    ).copy()
    if valid.empty:
        return {
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE": np.nan,
            "MedianAPE": np.nan,
            "WMAPE": np.nan,
            "SMAPE": np.nan,
            "Bias": np.nan,
            "UnderestimateRate": np.nan,
            "DirectionAccuracy": np.nan,
        }

    actual = valid["actual_revenue"].to_numpy(dtype=float)
    predicted = valid["predicted_revenue_sarima"].to_numpy(dtype=float)
    error = predicted - actual
    abs_error = np.abs(error)
    ape = np.divide(
        abs_error,
        np.abs(actual),
        out=np.full_like(abs_error, np.nan),
        where=actual != 0,
    )
    smape = np.divide(
        2.0 * abs_error,
        np.abs(actual) + np.abs(predicted),
        out=np.full_like(abs_error, np.nan),
        where=(np.abs(actual) + np.abs(predicted)) != 0,
    )
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
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(abs_error)),
        "MAPE": float(np.nanmean(ape) * 100),
        "MedianAPE": float(np.nanmedian(ape) * 100),
        "WMAPE": float(abs_error.sum() / np.abs(actual).sum() * 100),
        "SMAPE": float(np.nanmean(smape) * 100),
        "Bias": float(np.mean(error)),
        "UnderestimateRate": float(np.mean(predicted < actual) * 100),
        "DirectionAccuracy": direction_accuracy,
    }
