from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evidence import _load_dividends, _load_eps, resolve_data_files


PATTERN_LABELS = {
    "normal": "1. 正常配息（依盈餘估算）",
    "fixed": "2. 固定配息（歷史近似）",
    "none": "3. 不配現金股利（歷史）",
    "insufficient": "資料不足／待確認",
}
FIXED_DIVIDEND_TOLERANCE = 0.05
PATTERN_RULE_VERSION = "five_year_cash_pattern_v1"


def analyze_dividend_patterns(
    stock_ids, annual_eps: pd.DataFrame, dividends: pd.DataFrame, as_of_date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(as_of_date)
    years = list(range(cutoff.year - 5, cutoff.year))
    dividends = dividends.copy()
    annual_eps = annual_eps.copy()
    if "available_date" in dividends:
        dividends = dividends[pd.to_datetime(dividends["available_date"]).le(cutoff)]
    if "latest_available_date" in annual_eps:
        annual_eps = annual_eps[pd.to_datetime(annual_eps["latest_available_date"]).le(cutoff)]
    dividends = dividends[dividends["fiscal_year"].isin(years)]
    grouped = dividends.groupby(["stock_id", "fiscal_year"])["TotalCashDividend"]
    cash = grouped.sum(min_count=1)
    # An invalid installment must not turn a partially observed year into a valid total.
    valid_cash = grouped.agg(lambda values: bool(np.isfinite(values).all() and values.ge(0).all()))
    cash = cash.where(valid_cash)
    eps_lookup = annual_eps.set_index(["stock_id", "eps_year"])["annual_eps"]
    details, summaries = [], []
    for stock in sorted(set(map(int, stock_ids))):
        stock_rows = []
        for year in years:
            key = (stock, year)
            eps = float(eps_lookup.get(key, np.nan))
            dividend = float(cash.get(key, np.nan))
            reasons = []
            if not np.isfinite(eps):
                reasons.append("缺少完整四季 EPS")
            elif eps <= 0:
                reasons.append("EPS 為零或負數")
            if not np.isfinite(dividend):
                reasons.append("現金股利紀錄缺漏或無效，不能當作零")
            ratio = dividend / eps if not reasons else np.nan
            stock_rows.append({
                "stock_id": stock, "fiscal_year": year, "annual_eps": eps,
                "cash_dividend": dividend, "payout_ratio": ratio,
                "included": bool(np.isfinite(ratio)),
                "reason": "採用" if np.isfinite(ratio) else "；".join(reasons),
            })
        history = pd.DataFrame(stock_rows)
        known = history[history["cash_dividend"].notna()]
        positive = known[known["cash_dividend"].gt(0)]
        zero = known[known["cash_dividend"].eq(0)]
        used = history[history["included"]]
        median = float(known["cash_dividend"].median()) if not known.empty else np.nan
        deviation = float((known["cash_dividend"] / median - 1).abs().max()) if median > 0 else np.nan
        fixed = len(positive) == 5 and deviation <= FIXED_DIVIDEND_TOLERANCE + 1e-12
        if len(zero) == 5:
            pattern, reason = "none", "五個年度都有明確零現金股利紀錄；不代表沒有股票股利"
        elif fixed:
            pattern, reason = "fixed", "五年皆有正現金股利，逐年金額與五年中位數差距皆不超過 5%"
        elif len(known) >= 3 and not positive.empty:
            pattern, reason = "normal", "至少三年有現金股利紀錄且曾配現金，未符合五年固定金額規則"
        else:
            pattern, reason = "insufficient", "現金股利紀錄少於三年，或僅有部分年度零配息，無法確認模式"
        missing = history.loc[history["cash_dividend"].isna(), "fiscal_year"].tolist()
        if missing:
            reason += f"；股利缺漏年度：{', '.join(map(str, missing))}"
        if not zero.empty and not positive.empty:
            reason += "；含零配息年度，屬間歇配息紀錄"
        summaries.append({
            "stock_id": stock, "dividend_pattern": pattern,
            "dividend_pattern_label": PATTERN_LABELS[pattern], "pattern_reason": reason,
            "pattern_as_of_date": cutoff, "pattern_rule_version": PATTERN_RULE_VERSION,
            "pattern_basis": "歷史資料推估，非公司未來配息承諾",
            "payout_window": f"{years[0]}–{years[-1]}",
            "cash_history_years": len(known), "positive_cash_years": len(positive),
            "zero_cash_years": len(zero), "missing_cash_years": ", ".join(map(str, missing)),
            "payout_ratio": float(used["payout_ratio"].mean()) if not used.empty else np.nan,
            "payout_valid_years": len(used),
            "dividend_reference_year": int(known["fiscal_year"].max()) if not known.empty else pd.NA,
            "fixed_cash_dividend": median if fixed else np.nan,
            "cash_max_deviation_percent": deviation * 100,
            **{f"cash_dividend_{r['fiscal_year']}": r["cash_dividend"] for r in stock_rows},
        })
        details.extend({**row, "dividend_pattern_label": PATTERN_LABELS[pattern]} for row in stock_rows)
    return pd.DataFrame(summaries), pd.DataFrame(details)


def load_dividend_patterns(data_dir: str | Path, stock_ids, as_of_date):
    paths = resolve_data_files(data_dir)
    stocks, cutoff = set(map(int, stock_ids)), pd.Timestamp(as_of_date)
    issues = []
    try:
        annual, _ = _load_eps(paths["eps"], stocks, cutoff, strict=True)
    except (OSError, ValueError, KeyError) as error:
        annual = pd.DataFrame(columns=["stock_id", "eps_year", "annual_eps"])
        issues.append(f"eps: {error}")
    try:
        dividends = _load_dividends(paths["dividends"], stocks, cutoff)
    except (OSError, ValueError, KeyError) as error:
        dividends = pd.DataFrame(columns=["stock_id", "fiscal_year", "TotalCashDividend"])
        issues.append(f"dividends: {error}")
    catalog, detail = analyze_dividend_patterns(stocks, annual, dividends, cutoff)
    return catalog, detail, issues
