from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REVENUE_FILENAME = "Stock_revenue_2019~2025.csv"
EPS_FILENAME = "EPS2020~2025.csv"
DIVIDEND_FILENAME = "Dividend2019~2025.csv"
PRICE_FILENAME = "day K2020~2025.csv"
PRICE_CHUNK_SIZE = 250_000


@dataclass(frozen=True)
class FinancialEvidence:
    revenue: pd.DataFrame
    annual_eps: pd.DataFrame
    quarterly_eps: pd.DataFrame
    dividends: pd.DataFrame
    prices: pd.DataFrame


def load_financial_evidence(
    data_dir: str | Path,
    *,
    stock_ids: set[int],
    target_year: int,
    as_of_date: pd.Timestamp,
) -> FinancialEvidence:
    root = Path(data_dir)
    annual_eps, quarterly_eps = _load_eps(root / EPS_FILENAME, stock_ids, as_of_date)
    return FinancialEvidence(
        revenue=_load_revenue(root / REVENUE_FILENAME, stock_ids),
        annual_eps=annual_eps,
        quarterly_eps=quarterly_eps,
        dividends=_load_dividends(root / DIVIDEND_FILENAME, stock_ids),
        prices=_load_prices(root / PRICE_FILENAME, stock_ids, target_year),
    )


def _load_revenue(path: Path, stock_ids: set[int]) -> pd.DataFrame:
    revenue = pd.read_csv(
        path,
        usecols=["stock_id", "revenue_year", "revenue_month", "revenue_thousand"],
    )
    for column in ["stock_id", "revenue_year", "revenue_month", "revenue_thousand"]:
        revenue[column] = pd.to_numeric(revenue[column], errors="coerce")
    revenue = revenue.dropna(
        subset=["stock_id", "revenue_year", "revenue_month", "revenue_thousand"]
    )
    revenue["stock_id"] = revenue["stock_id"].astype(int)
    revenue["revenue_year"] = revenue["revenue_year"].astype(int)
    revenue["revenue_month"] = revenue["revenue_month"].astype(int)
    return revenue[revenue["stock_id"].isin(stock_ids)].reset_index(drop=True)


def _load_eps(
    path: Path,
    stock_ids: set[int],
    as_of_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps = pd.read_csv(path)
    required = {"stock_id", "date", "EPS"}
    if not required.issubset(eps.columns):
        raise ValueError(f"{path.name} is missing columns: {sorted(required - set(eps.columns))}")
    eps["stock_id"] = pd.to_numeric(eps["stock_id"], errors="coerce")
    eps["date"] = pd.to_datetime(eps["date"], errors="coerce")
    eps["EPS"] = pd.to_numeric(eps["EPS"], errors="coerce")
    if "statement_available_date" in eps.columns:
        eps["available_date"] = pd.to_datetime(eps["statement_available_date"], errors="coerce")
    else:
        eps["available_date"] = eps["date"].map(_statement_available_date)
    eps = eps.dropna(subset=["stock_id", "date", "EPS", "available_date"])
    eps["stock_id"] = eps["stock_id"].astype(int)
    eps = eps[eps["stock_id"].isin(stock_ids) & eps["available_date"].le(as_of_date)].copy()
    eps["eps_year"] = eps["date"].dt.year.astype(int)
    eps["eps_quarter"] = eps["date"].dt.quarter.astype(int)
    quarterly = eps.groupby(["stock_id", "eps_year", "eps_quarter"], as_index=False).agg(
        quarter_eps=("EPS", "sum"),
        latest_available_date=("available_date", "max"),
    )
    annual = eps.groupby(["stock_id", "eps_year"], as_index=False).agg(
        annual_eps=("EPS", "sum"),
        quarter_count=("eps_quarter", "nunique"),
        latest_available_date=("available_date", "max"),
    )
    return (
        annual[annual["quarter_count"] >= 4].reset_index(drop=True),
        quarterly.reset_index(drop=True),
    )


def _load_dividends(path: Path, stock_ids: set[int]) -> pd.DataFrame:
    dividends = pd.read_csv(path)
    required = {"stock_id", "TotalCashDividend"}
    if not required.issubset(dividends.columns):
        raise ValueError(
            f"{path.name} is missing columns: {sorted(required - set(dividends.columns))}"
        )
    dividends["stock_id"] = pd.to_numeric(dividends["stock_id"], errors="coerce")
    dividends["TotalCashDividend"] = pd.to_numeric(
        dividends["TotalCashDividend"], errors="coerce"
    )
    if "fiscal_year" in dividends.columns:
        dividends["fiscal_year"] = pd.to_numeric(dividends["fiscal_year"], errors="coerce")
    else:
        roc_year = dividends.get(
            "year", pd.Series(index=dividends.index, dtype=object)
        ).astype(str)
        dividends["fiscal_year"] = (
            pd.to_numeric(roc_year.str.extract(r"(\d{2,3})")[0], errors="coerce") + 1911
        )
    dividends["ex_dividend_date"] = pd.to_datetime(
        dividends.get("CashExDividendTradingDate"), errors="coerce"
    )
    available = pd.to_datetime(dividends.get("DividendAvailableDate"), errors="coerce")
    if available is None:
        available = pd.Series(pd.NaT, index=dividends.index, dtype="datetime64[ns]")
    if "AnnouncementDate" in dividends.columns:
        available = available.fillna(
            pd.to_datetime(dividends["AnnouncementDate"], errors="coerce")
        )
    dividends["available_date"] = available.fillna(dividends["ex_dividend_date"])
    dividends = dividends.dropna(subset=["stock_id", "TotalCashDividend"])
    dividends["stock_id"] = dividends["stock_id"].astype(int)
    return dividends[dividends["stock_id"].isin(stock_ids)].reset_index(drop=True)


def _load_prices(path: Path, stock_ids: set[int], target_year: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=["date", "stock_id", "close"],
        chunksize=PRICE_CHUNK_SIZE,
    ):
        chunk["stock_id"] = pd.to_numeric(chunk["stock_id"], errors="coerce")
        matched = chunk[chunk["stock_id"].isin(stock_ids)].copy()
        if matched.empty:
            continue
        matched["date"] = pd.to_datetime(matched["date"], errors="coerce")
        matched["close"] = pd.to_numeric(matched["close"], errors="coerce")
        matched = matched.dropna(subset=["stock_id", "date", "close"])
        matched = matched[
            matched["date"].dt.year.le(int(target_year)) & matched["close"].gt(0)
        ]
        if not matched.empty:
            parts.append(matched)
    if not parts:
        return pd.DataFrame(columns=["stock_id", "date", "close"])
    prices = pd.concat(parts, ignore_index=True)
    prices["stock_id"] = prices["stock_id"].astype(int)
    return prices.sort_values(["stock_id", "date"]).reset_index(drop=True)


def _statement_available_date(value: object) -> pd.Timestamp:
    date = pd.Timestamp(value)
    if date.month <= 3:
        return pd.Timestamp(date.year, 5, 15)
    if date.month <= 6:
        return pd.Timestamp(date.year, 8, 14)
    if date.month <= 9:
        return pd.Timestamp(date.year, 11, 14)
    return pd.Timestamp(date.year + 1, 3, 31)
