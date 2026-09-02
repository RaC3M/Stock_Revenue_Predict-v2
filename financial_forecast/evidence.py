from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import numpy as np


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
    issues: list[str] = field(default_factory=list)
    data_status: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_financial_evidence(
    data_dir: str | Path,
    *,
    stock_ids: set[int],
    target_year: int,
    as_of_date: pd.Timestamp,
    live: bool = False,
) -> FinancialEvidence:
    root = Path(data_dir)
    if live:
        return _load_available_evidence(root, stock_ids, target_year, as_of_date)
    annual_eps, quarterly_eps = _load_eps(root / EPS_FILENAME, stock_ids, as_of_date)
    return FinancialEvidence(
        revenue=_load_revenue(root / REVENUE_FILENAME, stock_ids),
        annual_eps=annual_eps,
        quarterly_eps=quarterly_eps,
        dividends=_load_dividends(root / DIVIDEND_FILENAME, stock_ids),
        prices=_load_prices(root / PRICE_FILENAME, stock_ids, target_year),
    )


def _load_revenue(
    path: Path, stock_ids: set[int], as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    columns = ["stock_id", "revenue_year", "revenue_month", "revenue_thousand"]
    if as_of_date is not None:
        columns.append("revenue_available_date")
    revenue = pd.read_csv(path, usecols=columns)
    for column in ["stock_id", "revenue_year", "revenue_month", "revenue_thousand"]:
        revenue[column] = pd.to_numeric(revenue[column], errors="coerce")
    revenue = revenue.dropna(
        subset=["stock_id", "revenue_year", "revenue_month", "revenue_thousand"]
    )
    revenue["stock_id"] = revenue["stock_id"].astype(int)
    revenue["revenue_year"] = revenue["revenue_year"].astype(int)
    revenue["revenue_month"] = revenue["revenue_month"].astype(int)
    if as_of_date is not None:
        revenue = revenue[revenue["stock_id"].isin(stock_ids)].copy()
        revenue["available_date"] = pd.to_datetime(revenue["revenue_available_date"], errors="coerce")
        revenue["date"] = pd.to_datetime(dict(
            year=revenue["revenue_year"], month=revenue["revenue_month"], day=1,
        ))
        revenue = revenue[
            revenue["available_date"].le(as_of_date) & revenue["date"].lt(as_of_date.to_period("M").start_time)
        ]
        if revenue.duplicated(["stock_id", "revenue_year", "revenue_month"]).any():
            raise ValueError("營收有重複的股票／月份")
        revenue = revenue[np.isfinite(revenue["revenue_thousand"]) & revenue["revenue_thousand"].ge(0)]
    return revenue[revenue["stock_id"].isin(stock_ids)].reset_index(drop=True)


def _load_eps(
    path: Path,
    stock_ids: set[int],
    as_of_date: pd.Timestamp,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps = pd.read_csv(path)
    required = {"stock_id", "date", "EPS"}
    if not required.issubset(eps.columns):
        raise ValueError(f"{path.name} is missing columns: {sorted(required - set(eps.columns))}")
    if strict and "statement_available_date" not in eps.columns:
        raise ValueError("EPS 缺少 statement_available_date")
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
    if strict:
        eps = eps[eps["date"].le(as_of_date) & np.isfinite(eps["EPS"])].copy()
        if eps.duplicated(["stock_id", "eps_year", "eps_quarter"]).any():
            raise ValueError("EPS 有重複的股票／季度，請提供單季 EPS")
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


def _load_dividends(
    path: Path, stock_ids: set[int], as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    dividends = pd.read_csv(path)
    required = {"stock_id", "TotalCashDividend"}
    if not required.issubset(dividends.columns):
        raise ValueError(
            f"{path.name} is missing columns: {sorted(required - set(dividends.columns))}"
        )
    if as_of_date is not None and "DividendAvailableDate" not in dividends.columns:
        raise ValueError("股利缺少 DividendAvailableDate")
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
    if as_of_date is not None:
        dividends["available_date"] = pd.to_datetime(dividends["DividendAvailableDate"], errors="coerce")
        dividends = dividends[
            dividends["available_date"].le(as_of_date) & dividends["stock_id"].isin(stock_ids)
        ].copy()
        if dividends.duplicated().any():
            raise ValueError("股利有完全重複的紀錄")
    dividends = dividends.dropna(subset=["stock_id"] if as_of_date is not None else ["stock_id", "TotalCashDividend"])
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


def resolve_data_files(data_dir: str | Path) -> dict[str, Path]:
    root = Path(data_dir).expanduser().resolve()
    names = {
        "revenue": REVENUE_FILENAME, "eps": EPS_FILENAME,
        "dividends": DIVIDEND_FILENAME, "daily_prices": PRICE_FILENAME,
        "stock_list": "stock_list_new.csv",
    }
    manifest = root / "manifest.json"
    if manifest.is_file():
        names.update({k: v for k, v in json.loads(manifest.read_text(encoding="utf-8-sig")).get("files", {}).items() if k in names})
    return {kind: root / name for kind, name in names.items()}


def _load_available_evidence(
    root: Path, stock_ids: set[int], target_year: int, as_of_date: pd.Timestamp,
) -> FinancialEvidence:
    paths = resolve_data_files(root)
    issues: list[str] = []

    def capture(kind, loader, empty):
        try:
            return loader()
        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as error:
            issues.append(f"{kind}: {error}")
            return empty

    revenue = capture("revenue", lambda: _load_revenue(paths["revenue"], stock_ids, as_of_date),
        pd.DataFrame(columns=["stock_id", "revenue_year", "revenue_month", "revenue_thousand", "date", "available_date"]))
    annual, quarterly = capture("eps", lambda: _load_eps(paths["eps"], stock_ids, as_of_date, strict=True), (
        pd.DataFrame(columns=["stock_id", "eps_year", "annual_eps", "quarter_count", "latest_available_date"]),
        pd.DataFrame(columns=["stock_id", "eps_year", "eps_quarter", "quarter_eps", "latest_available_date"]),
    ))
    dividends = capture("dividends", lambda: _load_dividends(paths["dividends"], stock_ids, as_of_date),
        pd.DataFrame(columns=["stock_id", "fiscal_year", "TotalCashDividend", "available_date", "ex_dividend_date"]))
    prices = capture("daily_prices", lambda: _load_prices(paths["daily_prices"], stock_ids, target_year),
        pd.DataFrame(columns=["stock_id", "date", "close"]))
    prices = prices[pd.to_datetime(prices["date"]).le(as_of_date)].copy()
    prices = prices[np.isfinite(pd.to_numeric(prices["close"], errors="coerce"))]
    if prices.duplicated(["stock_id", "date"]).any():
        issues.append("daily_prices: 股價有重複的股票／日期")
        prices = prices.iloc[:0]
    eps_period = (pd.to_datetime(dict(year=quarterly["eps_year"], month=quarterly["eps_quarter"] * 3, day=1))
        + pd.offsets.MonthEnd(0)).max() if not quarterly.empty else pd.NaT
    status = []
    for kind, frame, period, available in [
        ("revenue", revenue, pd.to_datetime(revenue["date"]).max(), pd.to_datetime(revenue["available_date"]).max()),
        ("eps", quarterly, eps_period, pd.to_datetime(quarterly["latest_available_date"]).max()),
        ("dividends", dividends, pd.NaT, pd.to_datetime(dividends["available_date"]).max()),
        ("daily_prices", prices, pd.to_datetime(prices["date"]).max(), pd.to_datetime(prices["date"]).max()),
    ]:
        status.append({"dataset": kind, "source": str(paths[kind]), "latest_period": period,
            "latest_available_date": available, "rows": len(frame),
            "status": "；".join(i for i in issues if i.startswith(kind + ":")) or ("可用" if not frame.empty else "無可用資料")})
    return FinancialEvidence(revenue, annual, quarterly, dividends, prices, issues, pd.DataFrame(status))
