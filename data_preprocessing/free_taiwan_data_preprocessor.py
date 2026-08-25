"""Preprocess free_taiwan_data into the project's canonical CSV inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocessing.canonical_data_contract import build_canonical_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "free_taiwan_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_preprocessing" / "outputs" / "processed"
DEFAULT_START_YEAR = 2019
DEFAULT_END_YEAR = 2025
DEFAULT_TARGET_YEAR = 2025
DEFAULT_PRICE_START_YEAR = 2020

REVENUE_FILENAME_TEMPLATE = "Stock_revenue_{start_year}~{end_year}.csv"
EPS_FILENAME_TEMPLATE = "EPS{start_year}~{end_year}.csv"
DIVIDEND_FILENAME_TEMPLATE = "Dividend{start_year}~{end_year}.csv"
DAILY_PRICE_FILENAME_TEMPLATE = "day K{start_year}~{end_year}.csv"
STOCK_LIST_FILENAME = "stock_list_new.csv"
TARGET_STOCKS_FILENAME_TEMPLATE = "target_stocks_{target_year}.csv"


@dataclass(frozen=True)
class FreeTaiwanPreprocessConfig:
    source_dir: Path = DEFAULT_SOURCE_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    start_year: int = DEFAULT_START_YEAR
    end_year: int = DEFAULT_END_YEAR
    target_year: int = DEFAULT_TARGET_YEAR
    price_start_year: int = DEFAULT_PRICE_START_YEAR
    stock_ids: tuple[int, ...] | None = None
    stock_limit: int | None = None


def parse_int_csv(value: str | None) -> tuple[int, ...] | None:
    if value is None or not str(value).strip():
        return None
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def parse_roc_year(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.match(r"^\s*(\d{2,3})", str(value).strip())
    if not match:
        return np.nan
    parsed = pd.to_numeric(match.group(1), errors="coerce")
    return float(parsed + 1911) if pd.notna(parsed) else np.nan


def month_revenue_available_date(revenue_year: object, revenue_month: object) -> pd.Timestamp:
    year = pd.to_numeric(revenue_year, errors="coerce")
    month = pd.to_numeric(revenue_month, errors="coerce")
    if pd.isna(year) or pd.isna(month):
        return pd.NaT
    year = int(year)
    month = int(month)
    if month == 12:
        return pd.Timestamp(year + 1, 1, 10)
    return pd.Timestamp(year, month + 1, 10)


def financial_statement_available_date(statement_date: object) -> pd.Timestamp:
    date = pd.to_datetime(statement_date, errors="coerce")
    if pd.isna(date):
        return pd.NaT
    year = int(date.year)
    month = int(date.month)
    if month == 3:
        return pd.Timestamp(year, 5, 15)
    if month == 6:
        return pd.Timestamp(year, 8, 14)
    if month == 9:
        return pd.Timestamp(year, 11, 14)
    if month == 12:
        return pd.Timestamp(year + 1, 3, 31)
    return pd.NaT


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator - 1


def _candidate_csv_paths(folder: Path, stock_ids: tuple[int, ...] | None = None) -> list[Path]:
    if not folder.is_dir():
        return []
    paths = sorted(folder.glob("*.csv"))
    if stock_ids is None:
        return paths
    stock_id_set = {str(int(stock_id)) for stock_id in stock_ids}
    return [path for path in paths if path.stem in stock_id_set]


def read_per_stock_csvs(
    folder: Path,
    stock_ids: tuple[int, ...] | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    frames = []
    for path in _candidate_csv_paths(folder, stock_ids=stock_ids):
        try:
            frame = pd.read_csv(path, usecols=usecols)
        except ValueError:
            frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _select_stock_ids(stock_info: pd.DataFrame, config: FreeTaiwanPreprocessConfig) -> tuple[int, ...] | None:
    if config.stock_ids is not None:
        stock_ids = tuple(sorted({int(stock_id) for stock_id in config.stock_ids}))
    else:
        if stock_info.empty:
            return None
        stock_ids = tuple(sorted(stock_info["stock_id"].dropna().astype(int).unique().tolist()))
    if config.stock_limit is not None:
        stock_ids = stock_ids[: int(config.stock_limit)]
    return stock_ids


def load_stock_info(source_dir: Path) -> pd.DataFrame:
    path = source_dir / "technical" / "TaiwanStockInfo" / "TaiwanStockInfo.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["stock_id", "stock_name", "industry_category", "market_type", "stock_info_date"])
    info = pd.read_csv(path).copy()
    info["stock_id"] = pd.to_numeric(info["stock_id"], errors="coerce")
    info = info.dropna(subset=["stock_id"])
    info["stock_id"] = info["stock_id"].astype(int)
    info["stock_info_date"] = pd.to_datetime(info.get("date"), errors="coerce")
    info = info.sort_values(["stock_id", "stock_info_date"])
    info = info.drop_duplicates("stock_id", keep="last")
    info = info.rename(columns={"type": "market_type"})
    return info[
        ["stock_id", "stock_name", "industry_category", "market_type", "stock_info_date"]
    ].sort_values("stock_id")


def build_stock_list_frame(stock_info: pd.DataFrame) -> pd.DataFrame:
    if stock_info.empty:
        return pd.DataFrame(columns=["stock_id", "stock_name", "industry_category"])
    return stock_info[["stock_id", "stock_name", "industry_category"]].sort_values("stock_id")


def build_revenue_frame(
    source_dir: Path,
    stock_info: pd.DataFrame,
    config: FreeTaiwanPreprocessConfig,
) -> pd.DataFrame:
    stock_ids = _select_stock_ids(stock_info, config)
    revenue = read_per_stock_csvs(
        source_dir / "fundamental" / "TaiwanStockMonthRevenue",
        stock_ids=stock_ids,
    )
    if revenue.empty:
        return pd.DataFrame()

    revenue = revenue.copy()
    revenue["stock_id"] = _to_numeric(revenue["stock_id"])
    revenue["revenue_year"] = _to_numeric(revenue["revenue_year"])
    revenue["revenue_month"] = _to_numeric(revenue["revenue_month"])
    revenue["revenue"] = _to_numeric(revenue["revenue"])
    revenue = revenue.dropna(subset=["stock_id", "revenue_year", "revenue_month", "revenue"])
    revenue["stock_id"] = revenue["stock_id"].astype(int)
    revenue["revenue_year"] = revenue["revenue_year"].astype(int)
    revenue["revenue_month"] = revenue["revenue_month"].astype(int)
    revenue = revenue[
        revenue["revenue_year"].between(int(config.start_year), int(config.end_year))
    ].copy()
    revenue["revenue_thousand"] = revenue["revenue"] / 1000.0
    revenue["date"] = pd.to_datetime(
        revenue["revenue_year"].astype(str)
        + "-"
        + revenue["revenue_month"].astype(str).str.zfill(2)
        + "-01",
        errors="coerce",
    )
    revenue["revenue_available_date"] = [
        month_revenue_available_date(year, month)
        for year, month in zip(revenue["revenue_year"], revenue["revenue_month"], strict=True)
    ]
    revenue = revenue.sort_values(["stock_id", "revenue_year", "revenue_month"]).reset_index(drop=True)

    previous_year = revenue[["stock_id", "revenue_year", "revenue_month", "revenue"]].copy()
    previous_year["revenue_year"] = previous_year["revenue_year"] + 1
    previous_year = previous_year.rename(columns={"revenue": "last_year_revenue"})
    revenue = revenue.merge(previous_year, on=["stock_id", "revenue_year", "revenue_month"], how="left")

    revenue["_calendar_month_index"] = revenue["revenue_year"] * 12 + revenue["revenue_month"] - 1
    month_gap = revenue.groupby("stock_id")["_calendar_month_index"].diff().ne(1)
    revenue["_calendar_segment"] = month_gap.groupby(revenue["stock_id"]).cumsum().astype(int)
    segment_grouped = revenue.groupby(["stock_id", "_calendar_segment"], group_keys=False)
    revenue["mom"] = segment_grouped["revenue"].pct_change(fill_method=None)
    revenue["yoy"] = _safe_pct(revenue["revenue"], revenue["last_year_revenue"])
    revenue["last_3m_revenue"] = (
        segment_grouped["revenue"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )
    revenue["last_12m_revenue"] = (
        segment_grouped["revenue"]
        .rolling(12, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )
    previous_rolling = revenue[
        ["stock_id", "revenue_year", "revenue_month", "last_3m_revenue", "last_12m_revenue"]
    ].copy()
    previous_rolling["revenue_year"] = previous_rolling["revenue_year"] + 1
    previous_rolling = previous_rolling.rename(
        columns={
            "last_3m_revenue": "previous_year_last_3m_revenue",
            "last_12m_revenue": "previous_year_last_12m_revenue",
        }
    )
    revenue = revenue.merge(
        previous_rolling,
        on=["stock_id", "revenue_year", "revenue_month"],
        how="left",
        validate="many_to_one",
    )
    revenue["last_3m_revenue_yoy"] = _safe_pct(
        revenue["last_3m_revenue"],
        revenue["previous_year_last_3m_revenue"],
    )
    revenue["last_12m_revenue_yoy"] = _safe_pct(
        revenue["last_12m_revenue"],
        revenue["previous_year_last_12m_revenue"],
    )
    revenue["acc_revenue"] = revenue.groupby(["stock_id", "revenue_year"])["revenue"].cumsum()
    previous_acc = revenue[["stock_id", "revenue_year", "revenue_month", "acc_revenue"]].copy()
    previous_acc["revenue_year"] = previous_acc["revenue_year"] + 1
    previous_acc = previous_acc.rename(columns={"acc_revenue": "previous_year_acc_revenue"})
    revenue = revenue.merge(previous_acc, on=["stock_id", "revenue_year", "revenue_month"], how="left")
    revenue["acc_revenue_yoy"] = _safe_pct(revenue["acc_revenue"], revenue["previous_year_acc_revenue"])
    for ratio_column in ["mom", "yoy", "last_3m_revenue_yoy", "last_12m_revenue_yoy", "acc_revenue_yoy"]:
        revenue[ratio_column] = revenue[ratio_column].round(4)

    metadata = stock_info[["stock_id", "industry_category"]].drop_duplicates("stock_id")
    revenue = revenue.merge(metadata, on="stock_id", how="left")
    output_columns = [
        "date",
        "stock_id",
        "revenue_year",
        "revenue_month",
        "revenue",
        "revenue_thousand",
        "mom",
        "last_year_revenue",
        "yoy",
        "last_3m_revenue",
        "last_3m_revenue_yoy",
        "last_12m_revenue",
        "last_12m_revenue_yoy",
        "acc_revenue",
        "acc_revenue_yoy",
        "industry_category",
        "revenue_available_date",
    ]
    return revenue[output_columns].sort_values(["stock_id", "revenue_year", "revenue_month"])


def build_target_stocks_frame(revenue: pd.DataFrame, target_year: int) -> pd.DataFrame:
    if revenue.empty:
        return pd.DataFrame(columns=["date", "stock_id", "country", "revenue", "revenue_month", "revenue_year"])
    target = revenue[revenue["revenue_year"].eq(int(target_year))].copy()
    target["country"] = "Taiwan"
    return target[
        ["date", "stock_id", "country", "revenue", "revenue_month", "revenue_year", "revenue_available_date"]
    ].sort_values(["stock_id", "revenue_month"])


def build_eps_frame(source_dir: Path, config: FreeTaiwanPreprocessConfig) -> pd.DataFrame:
    eps = read_per_stock_csvs(
        source_dir / "fundamental" / "TaiwanStockFinancialStatements",
        stock_ids=config.stock_ids,
    )
    if eps.empty:
        return pd.DataFrame(columns=["date", "stock_id", "EPS", "statement_available_date"])
    eps = eps[eps["type"].eq("EPS")].copy()
    eps["date"] = pd.to_datetime(eps["date"], errors="coerce")
    eps["stock_id"] = _to_numeric(eps["stock_id"])
    eps["EPS"] = _to_numeric(eps["value"])
    eps = eps.dropna(subset=["date", "stock_id", "EPS"])
    eps["stock_id"] = eps["stock_id"].astype(int)
    eps = eps[eps["date"].dt.year.between(int(config.start_year), int(config.end_year))].copy()
    eps["statement_available_date"] = eps["date"].map(financial_statement_available_date)
    return eps[["date", "stock_id", "EPS", "statement_available_date"]].sort_values(["stock_id", "date"])


def build_dividend_frame(source_dir: Path, config: FreeTaiwanPreprocessConfig) -> pd.DataFrame:
    dividends = read_per_stock_csvs(
        source_dir / "fundamental" / "TaiwanStockDividend",
        stock_ids=config.stock_ids,
    )
    if dividends.empty:
        return pd.DataFrame()
    dividends = dividends.copy()
    dividends["stock_id"] = _to_numeric(dividends["stock_id"])
    dividends["fiscal_year"] = dividends["year"].map(parse_roc_year)
    for column in ["CashEarningsDistribution", "CashStatutorySurplus"]:
        dividends[column] = _to_numeric(dividends[column]).fillna(0)
    dividends["TotalCashDividend"] = dividends["CashEarningsDistribution"] + dividends["CashStatutorySurplus"]
    for column in [
        "date",
        "CashExDividendTradingDate",
        "CashDividendPaymentDate",
        "StockExDividendTradingDate",
        "AnnouncementDate",
    ]:
        if column in dividends.columns:
            dividends[column] = pd.to_datetime(dividends[column], errors="coerce")
    dividends = dividends.dropna(subset=["stock_id", "year"])
    dividends["stock_id"] = dividends["stock_id"].astype(int)
    ex_year = dividends["CashExDividendTradingDate"].dt.year
    announcement_year = dividends["AnnouncementDate"].dt.year
    fiscal_year = _to_numeric(dividends["fiscal_year"])
    year_mask = (
        ex_year.between(int(config.start_year), int(config.end_year))
        | announcement_year.between(int(config.start_year), int(config.end_year))
        | fiscal_year.between(int(config.start_year) - 1, int(config.end_year))
    )
    dividends = dividends[year_mask].copy()
    dividends["DividendAvailableDate"] = dividends["AnnouncementDate"].fillna(
        dividends["CashExDividendTradingDate"]
    )
    dividends["dividend_available_source"] = np.where(
        dividends["AnnouncementDate"].notna(),
        "AnnouncementDate",
        np.where(dividends["CashExDividendTradingDate"].notna(), "CashExDividendTradingDate", "missing"),
    )
    output_columns = [
        "stock_id",
        "year",
        "fiscal_year",
        "TotalCashDividend",
        "CashEarningsDistribution",
        "CashStatutorySurplus",
        "CashExDividendTradingDate",
        "CashDividendPaymentDate",
        "StockEarningsDistribution",
        "StockExDividendTradingDate",
        "AnnouncementDate",
        "AnnouncementTime",
        "DividendAvailableDate",
        "dividend_available_source",
    ]
    existing = [column for column in output_columns if column in dividends.columns]
    return dividends[existing].sort_values(["stock_id", "DividendAvailableDate", "CashExDividendTradingDate"])


def build_daily_price_frame(source_dir: Path, config: FreeTaiwanPreprocessConfig) -> pd.DataFrame:
    prices = read_per_stock_csvs(
        source_dir / "technical" / "TaiwanStockPrice",
        stock_ids=config.stock_ids,
    )
    if prices.empty:
        return pd.DataFrame()
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["stock_id"] = _to_numeric(prices["stock_id"])
    for column in ["Trading_Volume", "Trading_money", "open", "max", "min", "close", "spread", "Trading_turnover"]:
        if column in prices.columns:
            prices[column] = _to_numeric(prices[column])
    prices = prices.dropna(subset=["date", "stock_id", "close"])
    prices["stock_id"] = prices["stock_id"].astype(int)
    prices = prices[
        prices["date"].dt.year.between(int(config.price_start_year), int(config.end_year))
    ].copy()
    output_columns = [
        "date",
        "stock_id",
        "Trading_Volume",
        "Trading_money",
        "open",
        "max",
        "min",
        "close",
        "spread",
        "Trading_turnover",
    ]
    existing = [column for column in output_columns if column in prices.columns]
    return prices[existing].sort_values(["stock_id", "date"])


def build_manifest(
    config: FreeTaiwanPreprocessConfig,
    frames: dict[str, pd.DataFrame],
    *,
    filenames: dict[str, str] | None = None,
    file_sha256: dict[str, str] | None = None,
) -> dict[str, object]:
    config_dict = {
        **asdict(config),
        "source_dir": str(config.source_dir),
        "output_dir": str(config.output_dir),
        "stock_ids": list(config.stock_ids) if config.stock_ids is not None else None,
    }
    manifest = build_canonical_manifest(
        config=config_dict,
        frames=frames,
        generator="data_preprocessing.free_taiwan_data_preprocessor",
        filenames=filenames,
        file_sha256=file_sha256,
    )
    dividends = frames.get("dividends", pd.DataFrame())
    if not dividends.empty and "AnnouncementDate" in dividends.columns:
        manifest["dividend_announcement_coverage"] = {
            "rows_with_announcement_date": int(dividends["AnnouncementDate"].notna().sum()),
            "rows_with_cash_ex_dividend_date": int(dividends["CashExDividendTradingDate"].notna().sum()),
            "rows_with_total_cash_dividend": int(dividends["TotalCashDividend"].notna().sum()),
        }
    return manifest


def preprocess_free_taiwan_data(config: FreeTaiwanPreprocessConfig) -> dict[str, pd.DataFrame]:
    stock_info = load_stock_info(config.source_dir)
    selected_stock_ids = _select_stock_ids(stock_info, config)
    effective_config = config
    if selected_stock_ids is not None and selected_stock_ids != config.stock_ids:
        effective_config = FreeTaiwanPreprocessConfig(
            source_dir=config.source_dir,
            output_dir=config.output_dir,
            start_year=config.start_year,
            end_year=config.end_year,
            target_year=config.target_year,
            price_start_year=config.price_start_year,
            stock_ids=selected_stock_ids,
            stock_limit=None,
        )
    stock_info = stock_info[stock_info["stock_id"].isin(selected_stock_ids or [])] if selected_stock_ids else stock_info
    stock_list = build_stock_list_frame(stock_info)
    revenue = build_revenue_frame(effective_config.source_dir, stock_info, effective_config)
    frames = {
        "stock_list": stock_list,
        "revenue": revenue,
        "target_stocks": build_target_stocks_frame(revenue, effective_config.target_year),
        "eps": build_eps_frame(effective_config.source_dir, effective_config),
        "dividends": build_dividend_frame(effective_config.source_dir, effective_config),
        "daily_prices": build_daily_price_frame(effective_config.source_dir, effective_config),
    }
    return frames


def write_processed_outputs(
    frames: dict[str, pd.DataFrame],
    config: FreeTaiwanPreprocessConfig,
) -> dict[str, Path]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "stock_list": output_dir / STOCK_LIST_FILENAME,
        "revenue": output_dir / REVENUE_FILENAME_TEMPLATE.format(
            start_year=config.start_year, end_year=config.end_year
        ),
        "target_stocks": output_dir / TARGET_STOCKS_FILENAME_TEMPLATE.format(target_year=config.target_year),
        "eps": output_dir / EPS_FILENAME_TEMPLATE.format(start_year=max(2020, config.start_year), end_year=config.end_year),
        "dividends": output_dir / DIVIDEND_FILENAME_TEMPLATE.format(
            start_year=config.start_year, end_year=config.end_year
        ),
        "daily_prices": output_dir / DAILY_PRICE_FILENAME_TEMPLATE.format(
            start_year=config.price_start_year, end_year=config.end_year
        ),
    }
    filenames = {name: path.name for name, path in paths.items()}
    prewrite_manifest = build_manifest(config, frames, filenames=filenames)
    validation = prewrite_manifest.get("validation", {})
    if not isinstance(validation, dict) or not bool(validation.get("is_valid")):
        issues = validation.get("issues", []) if isinstance(validation, dict) else []
        raise ValueError(f"Canonical frames failed validation: {issues}")
    for name, path in paths.items():
        frames[name].to_csv(path, index=False, encoding="utf-8-sig")
    file_sha256 = {}
    for name, path in paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        file_sha256[name] = digest.hexdigest()
    manifest = build_manifest(
        config,
        frames,
        filenames=filenames,
        file_sha256=file_sha256,
    )
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    paths["manifest"] = manifest_path
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR)
    parser.add_argument("--price-start-year", type=int, default=DEFAULT_PRICE_START_YEAR)
    parser.add_argument("--stock-ids", help="Comma-separated stock IDs.")
    parser.add_argument("--stock-limit", type=int, help="Limit selected stock pool for smoke runs.")
    parser.add_argument(
        "--full-universe",
        action="store_true",
        help="Explicitly build every stock available in TaiwanStockInfo. This is the default without filters.",
    )
    parser.add_argument("--no-write", action="store_true", help="Build frames but skip writing CSV outputs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.full_universe and (args.stock_ids or args.stock_limit is not None):
        parser.error("--full-universe cannot be combined with --stock-ids or --stock-limit.")
    config = FreeTaiwanPreprocessConfig(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        target_year=args.target_year,
        price_start_year=args.price_start_year,
        stock_ids=parse_int_csv(args.stock_ids),
        stock_limit=args.stock_limit,
    )
    frames = preprocess_free_taiwan_data(config)
    print("Built free_taiwan_data canonical frames:")
    for name, frame in frames.items():
        stock_count = int(frame["stock_id"].nunique()) if "stock_id" in frame.columns and not frame.empty else 0
        print(f"- {name}: rows={len(frame):,}, stocks={stock_count:,}")
    if not args.no_write:
        paths = write_processed_outputs(frames, config)
        print("Wrote processed outputs:")
        for name, path in paths.items():
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
