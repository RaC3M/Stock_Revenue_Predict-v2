from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import (
    FinancialForecastPolicy, FinancialForecastResult,
    EPS_METHOD_KNOWN_QUARTERS, DIVIDEND_METHOD_FIVE_YEAR_MEAN, DIVIDEND_METHOD_CLASSIFIED,
)
from .dividend import estimate_dividends
from .eps import estimate_eps
from .evidence import load_financial_evidence
from .yield_calc import calculate_yields
from .live_methods import estimate_known_quarters, estimate_five_year_dividends, estimate_classified_dividends


REQUIRED_PREDICTION_COLUMNS = {
    "source_family",
    "model",
    "stock_id",
    "target_year",
    "target_month",
    "predicted_revenue",
}


def forecast_financials(
    revenue_predictions: pd.DataFrame,
    *,
    target_year: int,
    as_of_date: str | pd.Timestamp,
    data_dir: str | Path,
    policy: FinancialForecastPolicy | None = None,
) -> FinancialForecastResult:
    """Forecast EPS, cash dividend, and distinct deployable/evaluation yields.

    Input predictions use a normalized long-form monthly contract.  Every source/model/stock
    group must contain exactly one finite prediction for each month 1..12; incomplete groups
    are returned as failures rather than silently annualized.
    """

    policy = policy or FinancialForecastPolicy()
    cutoff = pd.Timestamp(as_of_date)
    normalized = _normalize_predictions(revenue_predictions, int(target_year))
    stock_ids = set(int(value) for value in normalized["stock_id"].dropna().unique())
    evidence = load_financial_evidence(
        data_dir,
        stock_ids=stock_ids,
        target_year=int(target_year),
        as_of_date=cutoff,
        live=(EPS_METHOD_KNOWN_QUARTERS in policy.eps_methods
              or DIVIDEND_METHOD_FIVE_YEAR_MEAN in policy.dividend_methods
              or DIVIDEND_METHOD_CLASSIFIED in policy.dividend_methods),
    )
    annual_predictions, failures = _build_annual_predictions(normalized)
    if annual_predictions.empty:
        return FinancialForecastResult(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), failures,
            ["No complete finite annual revenue input.", *evidence.issues], data_status=evidence.data_status,
        )

    eps_parts = []
    quarter_details = pd.DataFrame()
    for method in policy.eps_methods:
        if method == EPS_METHOD_KNOWN_QUARTERS:
            estimate, quarter_details = estimate_known_quarters(annual_predictions, normalized, evidence)
        else:
            estimate = estimate_eps(
                annual_predictions,
                normalized,
                evidence.revenue,
                evidence.annual_eps,
                evidence.quarterly_eps,
                method,
            )
        eps_parts.append(estimate)
    eps_estimates = pd.concat(eps_parts, ignore_index=True) if eps_parts else pd.DataFrame()

    dividend_parts = []
    payout_details = pd.DataFrame()
    for method in policy.dividend_methods:
        if method == DIVIDEND_METHOD_CLASSIFIED:
            estimate, payout_details = estimate_classified_dividends(eps_estimates, evidence, cutoff)
        elif method == DIVIDEND_METHOD_FIVE_YEAR_MEAN:
            estimate, payout_details = estimate_five_year_dividends(eps_estimates, evidence, cutoff)
        else:
            estimate = estimate_dividends(
                eps_estimates,
                evidence.dividends,
                evidence.annual_eps,
                target_year=int(target_year),
                as_of_date=cutoff,
                method=method,
            )
        dividend_parts.append(estimate)
    dividend_estimates = (
        pd.concat(dividend_parts, ignore_index=True) if dividend_parts else pd.DataFrame()
    )
    yield_estimates = calculate_yields(
        dividend_estimates,
        evidence.prices,
        target_year=int(target_year),
        as_of_date=cutoff,
        yield_modes=policy.yield_modes,
        min_stock_price=policy.min_stock_price,
    )
    if not yield_estimates.empty:
        monthly_values = normalized[
            [
                "source_family",
                "model",
                "stock_id",
                "target_year",
                "target_month",
                "predicted_revenue",
            ]
        ].copy()
        yield_estimates["target_month"] = pd.to_numeric(
            yield_estimates["target_month"], errors="coerce"
        ).astype("Int64")
        monthly_values["target_month"] = monthly_values["target_month"].astype("Int64")
        yield_estimates = yield_estimates.merge(
            monthly_values,
            on=["source_family", "model", "stock_id", "target_year", "target_month"],
            how="left",
        )
    summary = _build_summary(dividend_estimates, yield_estimates)
    notes = [
        f"Financial evidence is restricted to information available by {cutoff.date()}.",
        "as_of_price_yield uses the latest observed close at the cutoff and is deployable.",
        "target_month_end_yield uses target-year observed closes and is evaluation-only.",
        *evidence.issues,
    ]
    return FinancialForecastResult(
        eps_estimates=eps_estimates,
        dividend_estimates=dividend_estimates,
        yield_estimates=yield_estimates,
        summary=summary,
        failures=failures,
        notes=notes,
        quarterly_eps_estimates=quarter_details,
        payout_history=payout_details,
        data_status=evidence.data_status,
    )


def _normalize_predictions(predictions: pd.DataFrame, target_year: int) -> pd.DataFrame:
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Revenue predictions are missing columns: {sorted(missing)}")
    frame = predictions.copy()
    for column in ["stock_id", "target_year", "target_month", "predicted_revenue"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(
        subset=[
            "source_family",
            "model",
            "stock_id",
            "target_year",
            "target_month",
            "predicted_revenue",
        ]
    )
    frame["stock_id"] = frame["stock_id"].astype(int)
    frame["target_year"] = frame["target_year"].astype(int)
    frame["target_month"] = frame["target_month"].astype(int)
    frame = frame[frame["target_year"].eq(int(target_year))]
    frame["predicted_revenue"] = frame["predicted_revenue"].clip(lower=0)
    return frame.sort_values(
        ["source_family", "model", "stock_id", "target_month"]
    ).reset_index(drop=True)


def _build_annual_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    keys = ["source_family", "model", "stock_id", "target_year"]
    for key, group in predictions.groupby(keys, dropna=False):
        source_family, model, stock_id, target_year = key
        months = sorted(group["target_month"].unique().tolist())
        duplicate_months = bool(group["target_month"].duplicated().any())
        finite = bool(np.isfinite(group["predicted_revenue"]).all())
        if months != list(range(1, 13)) or duplicate_months or not finite:
            failures.append(
                {
                    "source_family": source_family,
                    "model": model,
                    "stock_id": int(stock_id),
                    "target_year": int(target_year),
                    "stage": "annual_revenue",
                    "status": (
                        "duplicate monthly predictions"
                        if duplicate_months
                        else ("nonfinite monthly predictions" if not finite
                              else f"incomplete monthly predictions ({len(months)}/12)")
                    ),
                }
            )
            continue
        rows.append(
            {
                "source_family": source_family,
                "model": model,
                "stock_id": int(stock_id),
                "target_year": int(target_year),
                "monthly_observations": 12,
                "predicted_annual_revenue": float(group["predicted_revenue"].sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(failures)


def _build_summary(
    dividends: pd.DataFrame,
    yields: pd.DataFrame,
) -> pd.DataFrame:
    if dividends.empty:
        return pd.DataFrame()
    key_columns = [
        "source_family",
        "model",
        "stock_id",
        "target_year",
        "eps_method",
        "dividend_method",
    ]
    summary = dividends.copy()
    as_of = yields[yields["yield_mode"].eq("as_of_price_yield")][
        key_columns + ["price_date", "stock_price", "estimated_yield_percent"]
    ].rename(
        columns={
            "price_date": "as_of_price_date",
            "stock_price": "as_of_stock_price",
            "estimated_yield_percent": "as_of_price_yield_percent",
        }
    )
    evaluation = yields[yields["yield_mode"].eq("target_month_end_yield")].groupby(
        key_columns, as_index=False
    ).agg(
        latest_target_month_end_yield_percent=("estimated_yield_percent", "last"),
        average_target_month_end_yield_percent=("estimated_yield_percent", "mean"),
        target_month_end_observations=("estimated_yield_percent", "count"),
    )
    summary = summary.merge(as_of, on=key_columns, how="left")
    summary = summary.merge(evaluation, on=key_columns, how="left")
    for column in ["predicted_annual_revenue", "eps_reference_year"]:
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce").round().astype("Int64")
    return summary
