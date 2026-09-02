from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import DIVIDEND_METHOD_FIVE_YEAR_MEAN, DIVIDEND_METHOD_CLASSIFIED, EPS_METHOD_KNOWN_QUARTERS
from .dividend_patterns import analyze_dividend_patterns
from .evidence import FinancialEvidence


def estimate_known_quarters(
    annual_predictions: pd.DataFrame,
    monthly_predictions: pd.DataFrame,
    evidence: FinancialEvidence,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    revenue = evidence.revenue.copy()
    revenue["eps_quarter"] = (revenue["revenue_month"] - 1) // 3 + 1
    quarter_revenue = revenue.groupby(["stock_id", "revenue_year", "eps_quarter"], as_index=False).agg(
        quarter_revenue=("revenue_thousand", "sum"), months=("revenue_month", "nunique"),
    )
    quarter_history = quarter_revenue[quarter_revenue["months"].eq(3)].merge(
        evidence.quarterly_eps,
        left_on=["stock_id", "revenue_year", "eps_quarter"],
        right_on=["stock_id", "eps_year", "eps_quarter"],
    )
    quarter_history = quarter_history[quarter_history["quarter_revenue"].gt(0)].copy()
    quarter_history["ratio"] = quarter_history["quarter_eps"] / quarter_history["quarter_revenue"]
    year_revenue = revenue.groupby(["stock_id", "revenue_year"], as_index=False).agg(
        revenue=("revenue_thousand", "sum"), months=("revenue_month", "nunique"),
    )
    year_history = year_revenue[year_revenue["months"].eq(12)].merge(
        evidence.annual_eps, left_on=["stock_id", "revenue_year"], right_on=["stock_id", "eps_year"],
    )
    year_history = year_history[year_history["revenue"].gt(0)].copy()
    year_history["ratio"] = year_history["annual_eps"] / year_history["revenue"]
    annual_rows, quarter_rows = [], []
    keys = ["source_family", "model", "stock_id", "target_year"]
    for _, prediction in annual_predictions.iterrows():
        stock, year = int(prediction["stock_id"]), int(prediction["target_year"])
        monthly = monthly_predictions
        for key in keys:
            monthly = monthly[monthly[key].eq(prediction[key])]
        known = evidence.quarterly_eps[
            evidence.quarterly_eps["stock_id"].eq(stock) & evidence.quarterly_eps["eps_year"].eq(year)
        ]
        stock_quarters = quarter_history[
            quarter_history["stock_id"].eq(stock) & quarter_history["eps_year"].lt(year)
        ]
        stock_years = year_history[
            year_history["stock_id"].eq(stock) & year_history["eps_year"].lt(year)
        ].sort_values("eps_year").tail(3)
        values, reference_years = [], []
        known_count = 0
        for quarter in range(1, 5):
            quarter_value = float(monthly.loc[
                monthly["target_month"].between(quarter * 3 - 2, quarter * 3), "predicted_revenue"
            ].sum())
            actual = known[known["eps_quarter"].eq(quarter)]
            ratio, eps, basis, years = np.nan, np.nan, "unavailable", []
            if not actual.empty:
                eps, basis, years = float(actual.iloc[0]["quarter_eps"]), "actual", [year]
                known_count += 1
            else:
                candidates = stock_quarters[stock_quarters["eps_quarter"].eq(quarter)].sort_values("eps_year").tail(3)
                if not candidates.empty:
                    ratio, basis = float(candidates["ratio"].median()), "seasonal_estimate"
                    years = candidates["eps_year"].astype(int).tolist()
                elif not stock_years.empty:
                    ratio, basis = float(stock_years["ratio"].median()), "annual_ratio_fallback"
                    years = stock_years["eps_year"].astype(int).tolist()
                if np.isfinite(ratio):
                    eps = quarter_value * ratio
            values.append(eps)
            reference_years.extend(years)
            quarter_rows.append({
                **{key: prediction[key] for key in keys}, "eps_quarter": quarter,
                "quarter_revenue": quarter_value, "quarter_eps": eps,
                "eps_basis": basis, "eps_to_revenue_ratio": ratio,
                "reference_years": ", ".join(map(str, years)),
                "status": "ok" if np.isfinite(eps) else "EPS unavailable",
            })
        total = float(sum(values)) if np.isfinite(values).all() else np.nan
        annual_rows.append({
            **prediction.to_dict(), "eps_method": EPS_METHOD_KNOWN_QUARTERS,
            "estimated_eps": total, "known_quarters": known_count,
            "estimated_quarters": sum(np.isfinite(values)) - known_count,
            "eps_reference_year": max(reference_years) if reference_years else pd.NA,
            "eps_to_revenue_ratio": total / float(prediction["predicted_annual_revenue"])
                if float(prediction["predicted_annual_revenue"]) > 0 else np.nan,
            "eps_source": "reported after-tax quarterly EPS + historical same-quarter revenue-ratio estimates",
            "status": "ok" if np.isfinite(total) else "EPS unavailable",
        })
    return pd.DataFrame(annual_rows), pd.DataFrame(quarter_rows)


def estimate_five_year_dividends(
    eps_estimates: pd.DataFrame, evidence: FinancialEvidence, as_of_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = list(range(as_of_date.year - 5, as_of_date.year))
    cash = evidence.dividends.dropna(subset=["fiscal_year"]).groupby(
        ["stock_id", "fiscal_year"]
    )["TotalCashDividend"].sum(min_count=1)
    annual_eps = evidence.annual_eps.set_index(["stock_id", "eps_year"])
    detail_rows, summaries = [], {}
    for stock in eps_estimates["stock_id"].unique():
        ratios, used = [], []
        for year in years:
            key = (stock, year)
            eps = float(annual_eps.loc[key, "annual_eps"]) if key in annual_eps.index else np.nan
            dividend = float(cash.loc[key]) if key in cash.index else np.nan
            reasons = []
            if not np.isfinite(eps):
                reasons.append("缺少完整四季 EPS")
            elif eps <= 0:
                reasons.append("EPS 為零或負數")
            if not np.isfinite(dividend):
                reasons.append("缺少已公告現金股利紀錄")
            elif dividend < 0:
                reasons.append("現金股利為負數")
            ratio = dividend / eps if not reasons else np.nan
            if np.isfinite(ratio):
                ratios.append(ratio)
                used.append(year)
            detail_rows.append({
                "stock_id": int(stock), "fiscal_year": year, "annual_eps": eps,
                "cash_dividend": dividend, "payout_ratio": ratio,
                "included": not reasons, "reason": "採用" if not reasons else "；".join(reasons),
            })
        summaries[stock] = (float(np.mean(ratios)) if ratios else np.nan, used)
    rows = []
    for _, eps_row in eps_estimates.iterrows():
        ratio, used = summaries[eps_row["stock_id"]]
        eps = float(eps_row["estimated_eps"])
        amount = max(eps * ratio, 0.0) if np.isfinite(eps) and np.isfinite(ratio) else np.nan
        rows.append({
            **eps_row.to_dict(), "dividend_method": DIVIDEND_METHOD_FIVE_YEAR_MEAN,
            "payout_ratio": ratio, "payout_valid_years": len(used),
            "payout_window": f"{years[0]}–{years[-1]}",
            "dividend_reference_year": max(used) if used else pd.NA,
            "estimated_cash_dividend": amount,
            "dividend_source": f"five-year arithmetic mean; used years={used}",
            "actual_cash_dividend": np.nan, "actual_cash_dividend_source": "not used in live forecasts",
            "status": eps_row["status"] if eps_row["status"] != "ok"
                else ("ok" if np.isfinite(amount) else "payout unavailable"),
        })
    return pd.DataFrame(rows), pd.DataFrame(detail_rows)


def estimate_classified_dividends(eps_estimates, evidence, as_of_date):
    catalog, detail = analyze_dividend_patterns(
        eps_estimates["stock_id"].unique(), evidence.annual_eps, evidence.dividends, as_of_date,
    )
    lookup = catalog.set_index("stock_id")
    rows = []
    for _, eps_row in eps_estimates.iterrows():
        classification = lookup.loc[eps_row["stock_id"]].to_dict()
        pattern = classification["dividend_pattern"]
        eps, ratio = float(eps_row["estimated_eps"]), classification["payout_ratio"]
        if pattern == "fixed":
            amount, rule = classification["fixed_cash_dividend"], "五年每股現金股利中位數，不乘 EPS"
        elif pattern == "none":
            amount, rule = 0.0, "五年明確零現金股利，預估為零"
        elif classification["positive_cash_years"] > 0:
            amount = max(eps * ratio, 0.0) if np.isfinite(eps) and np.isfinite(ratio) else np.nan
            rule = "預估 EPS × 前五年有效配息率算術平均"
            if pattern == "insufficient":
                rule += "（模式待確認，暫用有限年度資料）"
        else:
            amount, rule = np.nan, "沒有足夠配息紀錄，保留未知，不預設零股利"
        status = "ok" if np.isfinite(amount) else ("EPS unavailable" if not np.isfinite(eps) else "payout unavailable")
        rows.append({
            **eps_row.to_dict(), **classification,
            "dividend_method": DIVIDEND_METHOD_CLASSIFIED,
            "estimated_cash_dividend": amount, "dividend_calculation": rule,
            "dividend_source": f"{classification['dividend_pattern_label']}；{rule}",
            "eps_status": eps_row["status"], "status": status,
            "actual_cash_dividend": np.nan, "actual_cash_dividend_source": "not used in live forecasts",
        })
    return pd.DataFrame(rows), detail
