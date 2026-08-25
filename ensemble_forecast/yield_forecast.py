"""Ensemble adapter for the shared financial-forecast module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SYSTEM_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from financial_forecast import FinancialForecastPolicy, forecast_financials


def build_ensemble_yield_forecast(
    revenue_forecast: pd.DataFrame,
    *,
    selected_stock: int,
    target_year: int,
    model_family: str,
    model_name: str,
    data_dir: str | Path,
    actual_revenue: pd.DataFrame | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Normalize one Ensemble revenue model and restore its legacy output contract."""

    required = {"revenue_month", "predicted_revenue"}
    if not required.issubset(revenue_forecast.columns):
        raise ValueError(
            f"Revenue forecast is missing columns: {sorted(required - set(revenue_forecast.columns))}"
        )
    cutoff = (
        pd.Timestamp(as_of_date)
        if as_of_date is not None
        else pd.Timestamp(int(target_year), 1, 10)
    )
    normalized = revenue_forecast[["revenue_month", "predicted_revenue"]].rename(
        columns={"revenue_month": "target_month"}
    )
    normalized["source_family"] = str(model_family)
    normalized["model"] = str(model_name)
    normalized["stock_id"] = int(selected_stock)
    normalized["target_year"] = int(target_year)

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
    monthly = shared.yield_estimates[
        shared.yield_estimates["yield_mode"].eq("target_month_end_yield")
    ].copy()
    if monthly.empty:
        return pd.DataFrame()

    monthly = monthly.rename(
        columns={
            "target_year": "revenue_year",
            "target_month": "revenue_month",
            "price_date": "stock_price_date",
            "price_source": "stock_price_source",
            "estimated_yield_percent": "predicted_dividend_yield_percent",
            "actual_yield_percent": "actual_dividend_yield_percent",
        }
    )
    monthly["model_family"] = str(model_family)
    monthly["date"] = pd.to_datetime(
        str(target_year)
        + "-"
        + monthly["revenue_month"].astype(int).astype(str).str.zfill(2)
        + "-01"
    )
    monthly["annual_eps_reference_year"] = monthly["eps_reference_year"]
    monthly["annual_eps"] = monthly["estimated_eps"]
    monthly["cash_dividend_per_share"] = monthly["estimated_cash_dividend"]
    monthly["cash_dividend_source"] = (
        monthly["eps_source"].astype(str)
        + "; payout="
        + monthly["dividend_source"].astype(str)
    )
    monthly["actual_cash_dividend_per_share"] = monthly["actual_cash_dividend"]
    monthly["forecast_annual_revenue"] = monthly["predicted_annual_revenue"]
    monthly["dividend_yield_percent"] = monthly["predicted_dividend_yield_percent"]

    summary = shared.summary.iloc[0] if not shared.summary.empty else pd.Series(dtype=object)
    monthly["as_of_price_date"] = summary.get("as_of_price_date", pd.NaT)
    monthly["as_of_stock_price"] = summary.get("as_of_stock_price", np.nan)
    monthly["as_of_price_yield_percent"] = summary.get(
        "as_of_price_yield_percent", np.nan
    )

    actual = _normalize_actual_revenue(actual_revenue, int(selected_stock), int(target_year))
    monthly = monthly.merge(actual, on="revenue_month", how="left")
    actual_annual_revenue = (
        float(actual["actual_revenue"].sum()) if not actual.empty else np.nan
    )
    monthly["actual_annual_revenue"] = actual_annual_revenue
    monthly["revenue_error"] = monthly["predicted_revenue"] - monthly["actual_revenue"]
    monthly["revenue_abs_error"] = monthly["revenue_error"].abs()
    monthly["yield_abs_error_percent_point"] = monthly["yield_error_percent_point"].abs()

    monthly["predicted_revenue"] = monthly["predicted_revenue"].round().astype("Int64")
    monthly["predicted_annual_revenue"] = (
        monthly["predicted_annual_revenue"].round().astype("Int64")
    )
    monthly["forecast_annual_revenue"] = (
        monthly["forecast_annual_revenue"].round().astype("Int64")
    )
    monthly["actual_annual_revenue"] = (
        pd.to_numeric(monthly["actual_annual_revenue"], errors="coerce")
        .round()
        .astype("Int64")
    )
    return monthly.sort_values("revenue_month").reset_index(drop=True)


def _normalize_actual_revenue(
    actual_revenue: pd.DataFrame | None,
    selected_stock: int,
    target_year: int,
) -> pd.DataFrame:
    if actual_revenue is None or actual_revenue.empty:
        return pd.DataFrame(columns=["revenue_month", "actual_revenue"])
    actual = actual_revenue.copy()
    if "stock_id" in actual.columns:
        actual = actual[pd.to_numeric(actual["stock_id"], errors="coerce").eq(selected_stock)]
    if "revenue_year" in actual.columns:
        actual = actual[
            pd.to_numeric(actual["revenue_year"], errors="coerce").eq(target_year)
        ]
    required = {"revenue_month", "actual_revenue"}
    if not required.issubset(actual.columns):
        return pd.DataFrame(columns=["revenue_month", "actual_revenue"])
    actual["revenue_month"] = pd.to_numeric(actual["revenue_month"], errors="coerce")
    actual["actual_revenue"] = pd.to_numeric(actual["actual_revenue"], errors="coerce")
    actual = actual.dropna(subset=["revenue_month", "actual_revenue"])
    actual["revenue_month"] = actual["revenue_month"].astype(int)
    return actual[["revenue_month", "actual_revenue"]].drop_duplicates(
        "revenue_month", keep="last"
    )
