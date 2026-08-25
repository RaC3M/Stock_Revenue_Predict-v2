"""Rolling adapter for the shared financial-forecast module.

The adapter owns Rolling column normalization and Streamlit-facing result names.  EPS,
dividend, price-evidence, and yield rules live behind ``financial_forecast``'s interface.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


SYSTEM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SYSTEM_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from financial_forecast import FinancialForecastPolicy, forecast_financials


@dataclass
class RollingYieldForecastResult:
    monthly: pd.DataFrame
    summary: pd.DataFrame
    notes: list[str]


def build_rolling_yield_forecast(
    revenue_forecast: pd.DataFrame,
    selected_stock: int,
    target_year: int,
    model_columns: Mapping[str, str],
    data_dir: str | Path,
    *,
    as_of_date: str | pd.Timestamp | None = None,
) -> RollingYieldForecastResult:
    """Map wide Rolling outputs through the shared financial-forecast interface."""

    if "target_month" not in revenue_forecast.columns:
        raise ValueError("Rolling revenue forecast is missing target_month.")
    if not model_columns:
        raise ValueError("At least one Rolling revenue model column is required.")
    if revenue_forecast["target_month"].duplicated().any():
        raise ValueError("Rolling revenue forecast contains duplicate target_month rows.")

    cutoff = pd.Timestamp(as_of_date) if as_of_date is not None else pd.Timestamp(target_year, 1, 10)
    normalized_parts: list[pd.DataFrame] = []
    for model_name, prediction_column in model_columns.items():
        if prediction_column not in revenue_forecast.columns:
            continue
        part = revenue_forecast[["target_month", prediction_column]].rename(
            columns={prediction_column: "predicted_revenue"}
        )
        part["predicted_revenue"] = pd.to_numeric(part["predicted_revenue"], errors="coerce")
        part["source_family"] = "rolling_lstm"
        part["model"] = str(model_name)
        part["stock_id"] = int(selected_stock)
        part["target_year"] = int(target_year)
        normalized_parts.append(part)
    normalized = (
        pd.concat(normalized_parts, ignore_index=True)
        if normalized_parts
        else pd.DataFrame(
            columns=[
                "source_family",
                "model",
                "stock_id",
                "target_year",
                "target_month",
                "predicted_revenue",
            ]
        )
    )

    shared = forecast_financials(
        normalized,
        target_year=int(target_year),
        as_of_date=cutoff,
        data_dir=data_dir,
        policy=FinancialForecastPolicy(
            eps_methods=("current_ratio",),
            dividend_methods=("announcement_safe_payout_ratio",),
            yield_modes=("as_of_price_yield", "target_month_end_yield"),
        ),
    )
    monthly = _build_rolling_monthly(shared.yield_estimates, int(target_year))
    summary = _build_rolling_summary(
        revenue_forecast,
        model_columns,
        shared.summary,
        shared.failures,
        monthly,
        cutoff,
    )
    notes = list(shared.notes)
    notes.extend(
        [
            "Target-year actual cash dividend is attached only for evaluation after the estimate is built.",
            "The annual revenue total sums rolling monthly forecasts that may use newly available "
            "revenue each month; it is a rolling evaluation total, not a single-vintage January forecast.",
        ]
    )
    return RollingYieldForecastResult(monthly=monthly, summary=summary, notes=notes)


def _build_rolling_monthly(yields: pd.DataFrame, target_year: int) -> pd.DataFrame:
    if yields.empty:
        return pd.DataFrame()
    monthly = yields[yields["yield_mode"].eq("target_month_end_yield")].copy()
    if monthly.empty:
        return pd.DataFrame()
    monthly = monthly.rename(
        columns={
            "price_date": "stock_price_date",
            "price_source": "stock_price_source",
            "dividend_source": "payout_source",
            "estimated_yield_percent": "predicted_dividend_yield_percent",
            "actual_yield_percent": "actual_dividend_yield_percent",
        }
    )
    monthly["target_date"] = pd.to_datetime(
        str(target_year) + "-" + monthly["target_month"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    monthly["yield_abs_error_percent_point"] = monthly["yield_error_percent_point"].abs()
    desired_columns = [
        "target_month",
        "predicted_revenue",
        "stock_price_date",
        "stock_price",
        "stock_price_source",
        "model",
        "stock_id",
        "target_year",
        "target_date",
        "predicted_annual_revenue",
        "eps_reference_year",
        "estimated_eps",
        "eps_source",
        "payout_ratio",
        "payout_source",
        "estimated_cash_dividend",
        "actual_cash_dividend",
        "actual_cash_dividend_source",
        "predicted_dividend_yield_percent",
        "actual_dividend_yield_percent",
        "yield_error_percent_point",
        "yield_abs_error_percent_point",
    ]
    return monthly[desired_columns].sort_values(["model", "target_month"]).reset_index(drop=True)


def _build_rolling_summary(
    revenue_forecast: pd.DataFrame,
    model_columns: Mapping[str, str],
    shared_summary: pd.DataFrame,
    failures: pd.DataFrame,
    monthly: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, prediction_column in model_columns.items():
        matched = (
            shared_summary[shared_summary["model"].eq(str(model_name))]
            if not shared_summary.empty
            else pd.DataFrame()
        )
        if matched.empty:
            if prediction_column not in revenue_forecast.columns:
                status = "prediction column missing"
            else:
                values = pd.to_numeric(revenue_forecast[prediction_column], errors="coerce")
                months = pd.to_numeric(
                    revenue_forecast.loc[values.notna(), "target_month"], errors="coerce"
                ).dropna().astype(int).unique()
                status = f"incomplete monthly predictions ({len(months)}/12)"
                if not failures.empty:
                    failure = failures[failures["model"].eq(str(model_name))]
                    if not failure.empty:
                        status = str(failure.iloc[0]["status"])
            rows.append(_unavailable_summary(model_name, prediction_column, cutoff, status))
            continue

        row = matched.iloc[0]
        model_monthly = (
            monthly[monthly["model"].eq(str(model_name))].sort_values("target_month")
            if not monthly.empty
            else pd.DataFrame()
        )
        priced = model_monthly.dropna(subset=["stock_price"])
        latest = priced.iloc[-1] if not priced.empty else pd.Series(dtype=object)
        rows.append(
            {
                "model": str(model_name),
                "prediction_column": str(prediction_column),
                "as_of_date": cutoff,
                "status": row["status"],
                "predicted_annual_revenue": row["predicted_annual_revenue"],
                "eps_reference_year": row["eps_reference_year"],
                "estimated_eps": row["estimated_eps"],
                "eps_source": row["eps_source"],
                "payout_ratio": row["payout_ratio"],
                "payout_source": row["dividend_source"],
                "estimated_cash_dividend": row["estimated_cash_dividend"],
                "actual_cash_dividend": row["actual_cash_dividend"],
                "actual_cash_dividend_source": row["actual_cash_dividend_source"],
                "as_of_price_date": row.get("as_of_price_date", pd.NaT),
                "as_of_stock_price": row.get("as_of_stock_price", np.nan),
                "as_of_price_yield_percent": row.get("as_of_price_yield_percent", np.nan),
                "latest_price_date": latest.get("stock_price_date", pd.NaT),
                "latest_stock_price": latest.get("stock_price", np.nan),
                "price_source": latest.get("stock_price_source", ""),
                "average_predicted_yield_percent": model_monthly[
                    "predicted_dividend_yield_percent"
                ].mean(),
                "latest_predicted_yield_percent": latest.get(
                    "predicted_dividend_yield_percent", np.nan
                ),
                "latest_actual_yield_percent": latest.get(
                    "actual_dividend_yield_percent", np.nan
                ),
            }
        )
    summary = pd.DataFrame(rows)
    for column in ["predicted_annual_revenue", "eps_reference_year"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").round().astype("Int64")
    return summary


def _unavailable_summary(
    model_name: str,
    prediction_column: str,
    cutoff: pd.Timestamp,
    status: str,
) -> dict[str, object]:
    return {
        "model": str(model_name),
        "prediction_column": str(prediction_column),
        "as_of_date": cutoff,
        "status": status,
        "predicted_annual_revenue": np.nan,
        "eps_reference_year": pd.NA,
        "estimated_eps": np.nan,
        "eps_source": "",
        "payout_ratio": np.nan,
        "payout_source": "",
        "estimated_cash_dividend": np.nan,
        "actual_cash_dividend": np.nan,
        "actual_cash_dividend_source": "",
        "as_of_price_date": pd.NaT,
        "as_of_stock_price": np.nan,
        "as_of_price_yield_percent": np.nan,
        "latest_price_date": pd.NaT,
        "latest_stock_price": np.nan,
        "price_source": "",
        "average_predicted_yield_percent": np.nan,
        "latest_predicted_yield_percent": np.nan,
        "latest_actual_yield_percent": np.nan,
    }
