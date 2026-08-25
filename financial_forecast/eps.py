from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import EPS_METHOD_CURRENT_RATIO, EPS_METHOD_SEASONAL_QUARTER_MEDIAN


def estimate_eps(
    annual_predictions: pd.DataFrame,
    monthly_predictions: pd.DataFrame,
    revenue_history: pd.DataFrame,
    annual_eps: pd.DataFrame,
    quarterly_eps: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    if method not in {EPS_METHOD_CURRENT_RATIO, EPS_METHOD_SEASONAL_QUARTER_MEDIAN}:
        raise ValueError(f"Unsupported EPS method: {method}")

    historical_revenue = revenue_history.groupby(
        ["stock_id", "revenue_year"], as_index=False
    ).agg(
        annual_revenue=("revenue_thousand", "sum"),
        month_count=("revenue_month", "nunique"),
    )
    historical_revenue = historical_revenue[historical_revenue["month_count"] >= 12]
    history = historical_revenue.merge(
        annual_eps,
        left_on=["stock_id", "revenue_year"],
        right_on=["stock_id", "eps_year"],
        how="inner",
    )
    history = history[(history["annual_revenue"] > 0) & history["annual_eps"].notna()].copy()
    history["eps_to_revenue_ratio"] = history["annual_eps"] / history["annual_revenue"]
    history = history.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["eps_to_revenue_ratio"]
    )

    if method == EPS_METHOD_SEASONAL_QUARTER_MEDIAN:
        return _estimate_seasonal_quarter_eps(
            annual_predictions,
            monthly_predictions,
            revenue_history,
            quarterly_eps,
            history,
        )

    rows: list[dict[str, object]] = []
    for _, prediction in annual_predictions.iterrows():
        row = prediction.to_dict()
        stock_history = history[history["stock_id"].eq(int(prediction["stock_id"]))]
        recent = stock_history.sort_values("revenue_year").tail(3)
        if recent.empty:
            row.update(
                {
                    "eps_method": method,
                    "estimated_eps": np.nan,
                    "eps_reference_year": pd.NA,
                    "eps_to_revenue_ratio": np.nan,
                    "eps_source": "annual EPS unavailable",
                    "status": "EPS unavailable",
                }
            )
        else:
            ratio = float(recent["eps_to_revenue_ratio"].median())
            row.update(
                {
                    "eps_method": method,
                    "estimated_eps": float(prediction["predicted_annual_revenue"]) * ratio,
                    "eps_reference_year": int(recent["revenue_year"].max()),
                    "eps_to_revenue_ratio": ratio,
                    "eps_source": (
                        "forecast annual revenue × median historical EPS/revenue "
                        "(up to 3 complete availability-safe years)"
                    ),
                    "status": "ok",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _estimate_seasonal_quarter_eps(
    annual_predictions: pd.DataFrame,
    monthly_predictions: pd.DataFrame,
    revenue_history: pd.DataFrame,
    quarterly_eps: pd.DataFrame,
    annual_history: pd.DataFrame,
) -> pd.DataFrame:
    historical = revenue_history.copy()
    historical["eps_quarter"] = ((historical["revenue_month"] - 1) // 3 + 1).astype(int)
    quarterly_revenue = historical.groupby(
        ["stock_id", "revenue_year", "eps_quarter"], as_index=False
    ).agg(
        quarter_revenue=("revenue_thousand", "sum"),
        month_count=("revenue_month", "nunique"),
    )
    quarterly_revenue = quarterly_revenue[quarterly_revenue["month_count"] >= 3]
    quarter_history = quarterly_revenue.merge(
        quarterly_eps,
        left_on=["stock_id", "revenue_year", "eps_quarter"],
        right_on=["stock_id", "eps_year", "eps_quarter"],
        how="inner",
    )
    quarter_history = quarter_history[
        (quarter_history["quarter_revenue"] > 0) & quarter_history["quarter_eps"].notna()
    ].copy()
    quarter_history["quarter_eps_to_revenue_ratio"] = (
        quarter_history["quarter_eps"] / quarter_history["quarter_revenue"]
    )
    quarter_history = quarter_history.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["quarter_eps_to_revenue_ratio"]
    )

    rows: list[dict[str, object]] = []
    key_columns = ["source_family", "model", "stock_id", "target_year"]
    for _, prediction in annual_predictions.iterrows():
        row = prediction.to_dict()
        selected_monthly = monthly_predictions.copy()
        for column in key_columns:
            selected_monthly = selected_monthly[
                selected_monthly[column].eq(prediction[column])
            ]
        selected_monthly = selected_monthly.copy()
        selected_monthly["eps_quarter"] = (
            (selected_monthly["target_month"] - 1) // 3 + 1
        ).astype(int)
        target_quarters = selected_monthly.groupby("eps_quarter", as_index=False).agg(
            predicted_quarter_revenue=("predicted_revenue", "sum")
        )

        stock_quarters = quarter_history[
            quarter_history["stock_id"].eq(int(prediction["stock_id"]))
        ]
        annual_candidates = annual_history[
            annual_history["stock_id"].eq(int(prediction["stock_id"]))
        ].sort_values("revenue_year").tail(3)
        annual_fallback = (
            float(annual_candidates["eps_to_revenue_ratio"].median())
            if not annual_candidates.empty
            else np.nan
        )
        estimated_eps = 0.0
        reference_years: list[int] = []
        used_quarters = 0
        fallback_quarters = 0
        for _, target_quarter in target_quarters.iterrows():
            quarter = int(target_quarter["eps_quarter"])
            candidates = stock_quarters[stock_quarters["eps_quarter"].eq(quarter)].sort_values(
                "revenue_year"
            ).tail(3)
            if candidates.empty:
                ratio = annual_fallback
                fallback_quarters += 1
            else:
                ratio = float(candidates["quarter_eps_to_revenue_ratio"].median())
                reference_years.append(int(candidates["revenue_year"].max()))
            if pd.notna(ratio):
                estimated_eps += float(target_quarter["predicted_quarter_revenue"]) * float(ratio)
                used_quarters += 1

        if used_quarters == 4:
            row.update(
                {
                    "eps_method": EPS_METHOD_SEASONAL_QUARTER_MEDIAN,
                    "estimated_eps": estimated_eps,
                    "eps_reference_year": max(reference_years) if reference_years else pd.NA,
                    "eps_to_revenue_ratio": (
                        estimated_eps / float(prediction["predicted_annual_revenue"])
                        if float(prediction["predicted_annual_revenue"]) > 0
                        else np.nan
                    ),
                    "eps_source": (
                        "same-quarter EPS/revenue median; "
                        f"fallback_quarters={fallback_quarters}"
                    ),
                    "status": "ok",
                }
            )
        else:
            row.update(
                {
                    "eps_method": EPS_METHOD_SEASONAL_QUARTER_MEDIAN,
                    "estimated_eps": np.nan,
                    "eps_reference_year": pd.NA,
                    "eps_to_revenue_ratio": np.nan,
                    "eps_source": "seasonal EPS evidence unavailable",
                    "status": "EPS unavailable",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)
