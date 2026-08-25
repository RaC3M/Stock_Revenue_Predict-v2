from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import (
    DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_LAST,
    DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_MEDIAN,
    DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_PAYOUT,
    DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_SMOOTHED,
)


SUPPORTED_DIVIDEND_METHODS = {
    DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_PAYOUT,
    DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_LAST,
    DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_MEDIAN,
    DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_SMOOTHED,
}


def estimate_dividends(
    eps_estimates: pd.DataFrame,
    dividends: pd.DataFrame,
    annual_eps: pd.DataFrame,
    *,
    target_year: int,
    as_of_date: pd.Timestamp,
    method: str,
) -> pd.DataFrame:
    if method not in SUPPORTED_DIVIDEND_METHODS:
        raise ValueError(f"Unsupported dividend method: {method}")

    available = dividends[
        dividends["available_date"].notna() & dividends["available_date"].le(as_of_date)
    ].copy()
    annual_cash = available.dropna(subset=["fiscal_year"]).groupby(
        ["stock_id", "fiscal_year"], as_index=False
    ).agg(cash_dividend=("TotalCashDividend", "sum"))
    if not annual_cash.empty:
        annual_cash["fiscal_year"] = annual_cash["fiscal_year"].astype(int)
    payout_history = annual_cash.merge(
        annual_eps,
        left_on=["stock_id", "fiscal_year"],
        right_on=["stock_id", "eps_year"],
        how="inner",
    )
    payout_history = payout_history[
        (payout_history["annual_eps"] > 0) & (payout_history["cash_dividend"] >= 0)
    ].copy()
    payout_history["payout_ratio"] = (
        payout_history["cash_dividend"] / payout_history["annual_eps"]
    )
    payout_history = payout_history.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["payout_ratio"]
    )
    payout_history["payout_ratio"] = payout_history["payout_ratio"].clip(0, 1.5)

    cross_sectional = _cross_sectional_payout(payout_history)
    actual = _actual_dividend_lookup(dividends, target_year)
    rows: list[dict[str, object]] = []
    for _, eps_row in eps_estimates.iterrows():
        stock_id = int(eps_row["stock_id"])
        if method == DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_PAYOUT:
            stock_history = payout_history[payout_history["stock_id"].eq(stock_id)].sort_values(
                "fiscal_year"
            ).tail(3)
            if not stock_history.empty:
                payout_ratio = float(stock_history["payout_ratio"].median())
                reference_year = int(stock_history["fiscal_year"].max())
                source = f"announcement-safe stock payout through fiscal {reference_year}"
            elif pd.notna(cross_sectional):
                payout_ratio = float(cross_sectional)
                reference_year = pd.NA
                source = "announcement-safe cross-sectional payout median"
            else:
                payout_ratio = np.nan
                reference_year = pd.NA
                source = "historical payout unavailable"

            estimated_eps = pd.to_numeric(
                pd.Series([eps_row["estimated_eps"]]), errors="coerce"
            ).iloc[0]
            estimated_cash = (
                float(max(float(estimated_eps) * payout_ratio, 0.0))
                if pd.notna(estimated_eps) and pd.notna(payout_ratio)
                else np.nan
            )
        else:
            payout_ratio = np.nan
            cash_history = annual_cash[annual_cash["stock_id"].eq(stock_id)].sort_values(
                "fiscal_year"
            ).tail(3)
            reference_year = (
                int(cash_history["fiscal_year"].max()) if not cash_history.empty else pd.NA
            )
            values = pd.to_numeric(
                cash_history["cash_dividend"], errors="coerce"
            ).dropna()
            if values.empty:
                estimated_cash = np.nan
                source = "announcement-safe cash dividend history unavailable"
            elif method == DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_LAST:
                estimated_cash = float(values.iloc[-1])
                source = "last announcement-safe cash dividend"
            elif method == DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_MEDIAN:
                estimated_cash = float(values.median())
                source = "median of up to 3 announcement-safe cash dividends"
            else:
                latest_first = values.iloc[::-1].to_numpy(dtype=float)
                weights = np.array([0.5, 0.3, 0.2], dtype=float)[: len(latest_first)]
                weights = weights / weights.sum()
                estimated_cash = float(np.sum(latest_first * weights))
                source = "0.5/0.3/0.2 weighted announcement-safe cash dividends"

        actual_row = actual[actual["stock_id"].eq(stock_id)]
        actual_cash = (
            float(actual_row.iloc[0]["actual_cash_dividend"])
            if not actual_row.empty
            else np.nan
        )
        row = eps_row.to_dict()
        row.update(
            {
                "dividend_method": method,
                "payout_ratio": payout_ratio,
                "dividend_reference_year": reference_year,
                "estimated_cash_dividend": estimated_cash,
                "dividend_source": source,
                "actual_cash_dividend": actual_cash,
                "actual_cash_dividend_source": (
                    "target-year actual (evaluation)"
                    if pd.notna(actual_cash)
                    else f"no {target_year} ex-dividend record"
                ),
                "status": (
                    eps_row["status"]
                    if eps_row["status"] != "ok"
                    else ("ok" if pd.notna(estimated_cash) else "payout unavailable")
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _cross_sectional_payout(history: pd.DataFrame) -> float:
    if history.empty:
        return np.nan
    stock_medians = history.groupby("stock_id")["payout_ratio"].median()
    return float(stock_medians.median()) if not stock_medians.empty else np.nan


def _actual_dividend_lookup(dividends: pd.DataFrame, target_year: int) -> pd.DataFrame:
    ex_year = dividends["ex_dividend_date"].dt.year
    matched = dividends[ex_year.eq(int(target_year))]
    if matched.empty:
        return pd.DataFrame(columns=["stock_id", "actual_cash_dividend"])
    return matched.groupby("stock_id", as_index=False).agg(
        actual_cash_dividend=("TotalCashDividend", "sum")
    )
