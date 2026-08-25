from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from .data_contracts import (
        DAILY_STOCK_PRICE_FILENAME,
        DATA_DIR,
        DATA_DIR_ENV_VAR,
        DIVIDEND_CASH_FILENAME,
        DIVIDEND_POLICY_FILENAME,
        EPS_FILENAME,
        LEGACY_REVENUE_FILENAME,
        MODEL_REVENUE_DATA_CONTRACT,
        MONETARY_REVENUE_FEATURE_COLUMNS,
        PROJECT_ROOT,
        REVENUE_FEATURE_UNIT_DIVISORS,
        REVENUE_FILENAME,
        SHARED_REVENUE_DATA_CONTRACT,
        SYSTEM_DIR,
        RevenueAmountUnit,
        RevenueDataContract,
        _resolve_data_dir,
        apply_revenue_data_contract,
    )
except ImportError:
    from data_contracts import (
        DAILY_STOCK_PRICE_FILENAME,
        DATA_DIR,
        DATA_DIR_ENV_VAR,
        DIVIDEND_CASH_FILENAME,
        DIVIDEND_POLICY_FILENAME,
        EPS_FILENAME,
        LEGACY_REVENUE_FILENAME,
        MODEL_REVENUE_DATA_CONTRACT,
        MONETARY_REVENUE_FEATURE_COLUMNS,
        PROJECT_ROOT,
        REVENUE_FEATURE_UNIT_DIVISORS,
        REVENUE_FILENAME,
        SHARED_REVENUE_DATA_CONTRACT,
        SYSTEM_DIR,
        RevenueAmountUnit,
        RevenueDataContract,
        _resolve_data_dir,
        apply_revenue_data_contract,
    )

try:
    from .yield_forecast import build_ensemble_yield_forecast as _build_ensemble_yield_forecast
except ImportError:
    from yield_forecast import build_ensemble_yield_forecast as _build_ensemble_yield_forecast


TRAIN_START_YEAR = 2020
FORECAST_YEAR = 2025
SOURCE_YEAR = FORECAST_YEAR - 1
FORECAST_MODEL_NAMES = ("XGBoost", "LightGBM", "CatBoost", "SeasonalQuantile")
YIELD_FORECAST_COLUMNS = [
    "forecast_annual_revenue",
    "annual_eps_reference_year",
    "annual_eps",
    "payout_ratio",
    "cash_dividend_per_share",
    "cash_dividend_source",
    "estimated_eps",
    "estimated_cash_dividend",
    "stock_price_date",
    "stock_price",
    "stock_price_source",
    "dividend_yield_percent",
    "as_of_price_date",
    "as_of_stock_price",
    "as_of_price_yield_percent",
]

FEATURES = [
    "stock_id",
    "revenue_month",
    #"revenue_thousand",

    "mom",
    "last_year_revenue",
    "yoy",
    "last_3m_revenue",
    "last_3m_revenue_yoy",
    "last_12m_revenue",
    "last_12m_revenue_yoy",

    "acc_revenue",
    "acc_revenue_yoy",
    
    # log 類特徵
    "log_revenue",
    "log_revenue_lag1",
    "log_mom",
    "log_mom_3",

    # 春節特徵
    "is_cny_month",

]

@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    yield_comparison: pd.DataFrame
    backtest: pd.DataFrame
    metrics: pd.DataFrame
    weights: pd.DataFrame
    recommendation: dict[str, str]
    notes: list[str]


def _apply_revenue_data_contract(df: pd.DataFrame, data_contract: RevenueDataContract) -> pd.DataFrame:
    return apply_revenue_data_contract(df, data_contract)


RATIO_FEATURE_COLUMNS = {
    "mom",
    "yoy",
    "last_3m_revenue_yoy",
    "last_12m_revenue_yoy",
    "acc_revenue_yoy",
}


def _coerce_numeric_column(series: pd.Series, *, ratio: bool = False) -> pd.Series:
    """Parse a numeric column without guessing the unit from unrelated rows."""
    text = series.astype(str).str.strip()
    percent_mask = text.str.endswith("%", na=False) if ratio else pd.Series(False, index=series.index)
    values = pd.to_numeric(text.str.replace("%", "", regex=False), errors="coerce")
    if ratio:
        values.loc[percent_mask] = values.loc[percent_mask] / 100.0
    return values


def _fill_feature_from_past(df: pd.DataFrame, column: str) -> pd.Series:
    """Fill a model feature using only earlier observations for the same stock."""
    values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    historical_median = values.groupby(df["stock_id"], sort=False).transform(
        lambda stock_values: stock_values.expanding(min_periods=1).median().shift(1)
    )
    return values.fillna(historical_median).fillna(0.0)


def _attach_next_year_target(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the same calendar month's revenue one year later as the target."""
    target = df[["stock_id", "revenue_year", "revenue_month", "log_revenue"]].copy()
    target["revenue_year"] = target["revenue_year"] - 1
    target = target.rename(columns={"log_revenue": "target_next_year"})
    target = target.drop_duplicates(["stock_id", "revenue_year", "revenue_month"], keep="last")
    return df.merge(
        target,
        on=["stock_id", "revenue_year", "revenue_month"],
        how="left",
        validate="many_to_one",
    )


def _attach_previous_year_value(
    df: pd.DataFrame,
    value_column: str,
    output_column: str,
) -> pd.DataFrame:
    previous = df[["stock_id", "revenue_year", "revenue_month", value_column]].copy()
    previous["revenue_year"] = previous["revenue_year"] + 1
    previous = previous.rename(columns={value_column: output_column})
    previous = previous.drop_duplicates(["stock_id", "revenue_year", "revenue_month"], keep="last")
    return df.merge(
        previous,
        on=["stock_id", "revenue_year", "revenue_month"],
        how="left",
        validate="many_to_one",
    )


def load_revenue_data(
    path: str | None = None,
    data_contract: RevenueDataContract = SHARED_REVENUE_DATA_CONTRACT,
) -> pd.DataFrame:
    if path is None:
        path = os.path.join(DATA_DIR, REVENUE_FILENAME)
        if not os.path.exists(path):
            path = os.path.join(DATA_DIR, LEGACY_REVENUE_FILENAME)
    return prepare_revenue_data(pd.read_csv(path), data_contract=data_contract)


def load_actual_2025_data(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(DATA_DIR, REVENUE_FILENAME)
        if not os.path.exists(path):
            path = os.path.join(DATA_DIR, "target_stocks_2025.csv")
    actual = pd.read_csv(path)
    actual = actual.copy()
    actual["stock_id"] = pd.to_numeric(actual["stock_id"], errors="coerce").astype("Int64")
    actual["revenue_year"] = pd.to_numeric(actual["revenue_year"], errors="coerce").astype("Int64")
    actual["revenue_month"] = pd.to_numeric(actual["revenue_month"], errors="coerce").astype("Int64")
    if "revenue_thousand" in actual.columns:
        actual["actual_revenue"] = pd.to_numeric(actual["revenue_thousand"], errors="coerce").round()
    else:
        actual["revenue"] = pd.to_numeric(actual["revenue"], errors="coerce")
        actual["actual_revenue"] = (actual["revenue"] / 1000).round()
    actual = actual.dropna(subset=["stock_id", "revenue_year", "revenue_month", "actual_revenue"])
    actual["stock_id"] = actual["stock_id"].astype(int)
    actual["revenue_year"] = actual["revenue_year"].astype(int)
    actual["revenue_month"] = actual["revenue_month"].astype(int)
    actual["actual_revenue"] = actual["actual_revenue"].astype(int)
    actual = actual[actual["revenue_year"] == FORECAST_YEAR]
    actual["date"] = pd.to_datetime(
        actual["revenue_year"].astype(str) + "-" + actual["revenue_month"].astype(str).str.zfill(2) + "-01"
    )
    return actual[["date", "stock_id", "revenue_year", "revenue_month", "actual_revenue"]].sort_values(
        ["stock_id", "revenue_year", "revenue_month"]
    )


def load_eps_data(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(DATA_DIR, EPS_FILENAME)
    eps = pd.read_csv(path).copy()
    eps["date"] = pd.to_datetime(eps["date"], errors="coerce")
    eps["stock_id"] = pd.to_numeric(eps["stock_id"], errors="coerce").astype("Int64")
    eps["EPS"] = pd.to_numeric(eps["EPS"], errors="coerce")
    if "statement_available_date" in eps.columns:
        eps["available_date"] = pd.to_datetime(eps["statement_available_date"], errors="coerce")
    else:
        eps["available_date"] = eps["date"].map(_eps_statement_available_date)
    eps = eps.dropna(subset=["date", "stock_id", "EPS"])
    eps["stock_id"] = eps["stock_id"].astype(int)
    eps["eps_year"] = eps["date"].dt.year.astype(int)
    eps["eps_quarter"] = eps["date"].dt.quarter.astype(int)
    eps = eps.sort_values(["stock_id", "date"]).reset_index(drop=True)
    grouped = eps.groupby("stock_id", group_keys=False)
    eps["latest_eps"] = eps["EPS"]
    eps["eps_ttm"] = (
        grouped["EPS"]
        .rolling(4, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    eps["eps_yoy"] = eps["EPS"] / grouped["EPS"].shift(4) - 1
    eps["eps_ttm_yoy"] = eps["eps_ttm"] / grouped["eps_ttm"].shift(4) - 1
    return eps[
        [
            "stock_id",
            "date",
            "available_date",
            "eps_year",
            "eps_quarter",
            "latest_eps",
            "eps_ttm",
            "eps_yoy",
            "eps_ttm_yoy",
        ]
    ]


def _eps_statement_available_date(value: object) -> pd.Timestamp:
    date = pd.Timestamp(value)
    if date.month <= 3:
        return pd.Timestamp(date.year, 5, 15)
    if date.month <= 6:
        return pd.Timestamp(date.year, 8, 14)
    if date.month <= 9:
        return pd.Timestamp(date.year, 11, 14)
    return pd.Timestamp(date.year + 1, 3, 31)


def _filter_eps_available_as_of(eps: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    if "available_date" in eps.columns:
        available_date = pd.to_datetime(eps["available_date"], errors="coerce")
    elif "date" in eps.columns:
        available_date = pd.to_datetime(eps["date"], errors="coerce").map(
            _eps_statement_available_date
        )
    else:
        return eps.copy()
    return eps[available_date.notna() & available_date.le(cutoff)].copy()


def load_cash_dividend_data(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(DATA_DIR, DIVIDEND_CASH_FILENAME)
    dividends = pd.read_csv(path).copy()
    dividends["stock_id"] = pd.to_numeric(dividends["stock_id"], errors="coerce").astype("Int64")
    dividends["TotalCashDividend"] = pd.to_numeric(dividends["TotalCashDividend"], errors="coerce")
    dividends["CashExDividendTradingDate"] = pd.to_datetime(
        dividends["CashExDividendTradingDate"], errors="coerce"
    )
    dividends = dividends.dropna(subset=["stock_id", "TotalCashDividend"])
    dividends["stock_id"] = dividends["stock_id"].astype(int)
    dividends["ex_dividend_year"] = dividends["CashExDividendTradingDate"].dt.year.astype("Int64")
    dividends["fiscal_year_roc"] = (
        dividends["year"].astype(str).str.replace("年", "", regex=False).pipe(pd.to_numeric, errors="coerce")
    )
    dividends["fiscal_year"] = dividends["fiscal_year_roc"] + 1911
    return dividends.sort_values(["stock_id", "CashExDividendTradingDate"]).reset_index(drop=True)


def load_dividend_policy_data(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(DATA_DIR, DIVIDEND_POLICY_FILENAME)
    policy = pd.read_csv(path).copy()
    policy["stock_id"] = pd.to_numeric(policy["stock_id"], errors="coerce").astype("Int64")
    for column in ["net_margin_assumption", "payout_ratio"]:
        policy[column] = pd.to_numeric(policy[column], errors="coerce")
    policy = policy.dropna(subset=["stock_id", "net_margin_assumption", "payout_ratio"])
    policy["stock_id"] = policy["stock_id"].astype(int)
    policy["net_margin_assumption"] = policy["net_margin_assumption"].clip(lower=0, upper=1)
    policy["payout_ratio"] = policy["payout_ratio"].clip(lower=0, upper=1)
    return policy.sort_values("stock_id").reset_index(drop=True)


def load_stock_price_data(
    path: str | None = None,
    selected_stock: int | None = None,
    target_year: int | None = None,
) -> pd.DataFrame:
    if path is None:
        path = os.path.join(DATA_DIR, DAILY_STOCK_PRICE_FILENAME)

    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if {"date", "stock_id", "close"}.issubset(columns):
        return _load_daily_stock_prices(path, selected_stock=selected_stock, target_year=target_year)

    prices = pd.read_csv(path).copy()
    prices["stock_id"] = pd.to_numeric(prices["stock_id"], errors="coerce").astype("Int64")
    prices["price_year"] = pd.to_numeric(prices["price_year"], errors="coerce").astype("Int64")
    prices["price_month"] = pd.to_numeric(prices["price_month"], errors="coerce").astype("Int64")
    prices["close_price"] = pd.to_numeric(prices["close_price"], errors="coerce")
    prices = prices.dropna(subset=["stock_id", "price_year", "price_month", "close_price"])
    prices["stock_id"] = prices["stock_id"].astype(int)
    prices["price_year"] = prices["price_year"].astype(int)
    prices["price_month"] = prices["price_month"].astype(int)
    prices["close_price"] = prices["close_price"].clip(lower=0.01)
    if "price_date" not in prices.columns:
        prices["price_date"] = pd.NaT
    prices["price_source"] = os.path.basename(path)
    if selected_stock is not None:
        prices = prices[prices["stock_id"] == int(selected_stock)]
    if target_year is not None:
        prices = prices[prices["price_year"] == int(target_year)]
    return prices.sort_values(["stock_id", "price_year", "price_month"]).reset_index(drop=True)


def _load_daily_stock_prices(
    path: str,
    selected_stock: int | None = None,
    target_year: int | None = None,
) -> pd.DataFrame:
    frames = []
    for chunk in pd.read_csv(path, usecols=["date", "stock_id", "close"], chunksize=250_000):
        chunk["stock_id"] = pd.to_numeric(chunk["stock_id"], errors="coerce")
        if selected_stock is not None:
            chunk = chunk[chunk["stock_id"] == int(selected_stock)]
        if chunk.empty:
            continue

        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        if target_year is not None:
            chunk = chunk[chunk["date"].dt.year == int(target_year)]
        chunk["close"] = pd.to_numeric(chunk["close"], errors="coerce")
        chunk = chunk.dropna(subset=["date", "stock_id", "close"])
        if chunk.empty:
            continue
        frames.append(chunk)

    if not frames:
        return pd.DataFrame(
            columns=["stock_id", "price_year", "price_month", "price_date", "close_price", "price_source"]
        )

    prices = pd.concat(frames, ignore_index=True)
    prices["stock_id"] = prices["stock_id"].astype(int)
    prices["price_year"] = prices["date"].dt.year.astype(int)
    prices["price_month"] = prices["date"].dt.month.astype(int)
    prices = prices.sort_values(["stock_id", "price_year", "price_month", "date"])
    monthly = prices.groupby(["stock_id", "price_year", "price_month"], as_index=False).tail(1)
    monthly = monthly.rename(columns={"date": "price_date", "close": "close_price"})
    monthly["close_price"] = monthly["close_price"].clip(lower=0.01)
    monthly["price_source"] = os.path.basename(path)
    return monthly[
        ["stock_id", "price_year", "price_month", "price_date", "close_price", "price_source"]
    ].sort_values(["stock_id", "price_year", "price_month"]).reset_index(drop=True)


def prepare_revenue_data(
    df: pd.DataFrame,
    data_contract: RevenueDataContract = SHARED_REVENUE_DATA_CONTRACT,
) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    if "1000" in df.columns:
        df = df.drop(columns=["1000"])


    numeric_columns = [
        "stock_id",
        "revenue_year",
        "revenue_month",
        "revenue",
        "revenue_thousand",   # ✅ 這個一定要保留
        "mom",
        "last_year_revenue",
        "yoy",
        "last_3m_revenue",
        "last_3m_revenue_yoy",
        "last_12m_revenue",
        "last_12m_revenue_yoy",
        "acc_revenue",
        "acc_revenue_yoy",
        "outstanding_shares",
        "outstanding_shares_thousand",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = _coerce_numeric_column(
                df[column],
                ratio=column in RATIO_FEATURE_COLUMNS,
            )

    if "revenue_thousand" not in df.columns and "revenue" in df.columns:
        df["revenue_thousand"] = pd.to_numeric(df["revenue"], errors="coerce") / 1000.0
    df = df.dropna(subset=["stock_id", "revenue_year", "revenue_month", "revenue_thousand"])
    df["stock_id"] = df["stock_id"].astype(int)
    df["revenue_year"] = df["revenue_year"].astype(int)
    df["revenue_month"] = df["revenue_month"].astype(int)
    df = df.sort_values(["stock_id", "revenue_year", "revenue_month"]).reset_index(drop=True)

    df = _apply_revenue_data_contract(df, data_contract)
    if "last_year_revenue" in df.columns:
        df["_source_last_year_revenue"] = df["last_year_revenue"]
    else:
        df["_source_last_year_revenue"] = np.nan

    # Calendar segments prevent row shifts and rolling windows from crossing a missing month.
    df["_calendar_month_index"] = df["revenue_year"] * 12 + df["revenue_month"] - 1
    month_gap = df.groupby("stock_id")["_calendar_month_index"].diff().ne(1)
    df["_calendar_segment"] = month_gap.groupby(df["stock_id"]).cumsum().astype(int)
    grouped = df.groupby("stock_id", group_keys=False)
    segment_grouped = df.groupby(["stock_id", "_calendar_segment"], group_keys=False)
    df["_consecutive_month_count"] = segment_grouped.cumcount() + 1
    df["_history_12m_complete"] = df["_consecutive_month_count"].ge(12)

    df["time_idx"] = grouped.cumcount()
    time_max = grouped["time_idx"].transform("max").replace(0, 1)
    df["time_idx_norm"] = df["time_idx"] / time_max

    # Rebuild derived features from the canonical revenue amount. This makes their
    # calendar semantics independent of stale or partially populated CSV columns.
    df = df.drop(
        columns=[
            "last_year_revenue",
            "mom",
            "yoy",
            "last_3m_revenue",
            "last_3m_revenue_yoy",
            "last_12m_revenue",
            "last_12m_revenue_yoy",
            "acc_revenue",
            "acc_revenue_yoy",
        ],
        errors="ignore",
    )
    df = _attach_previous_year_value(df, "revenue_thousand", "last_year_revenue")
    df["last_year_revenue"] = df["last_year_revenue"].fillna(df["_source_last_year_revenue"])
    segment_grouped = df.groupby(["stock_id", "_calendar_segment"], group_keys=False)
    df["mom"] = segment_grouped["revenue_thousand"].pct_change(fill_method=None)
    df["yoy"] = df["revenue_thousand"] / df["last_year_revenue"] - 1
    df["last_3m_revenue"] = (
        segment_grouped["revenue_thousand"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )
    df["last_12m_revenue"] = (
        segment_grouped["revenue_thousand"]
        .rolling(12, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
    )
    df["acc_revenue"] = df.groupby(["stock_id", "revenue_year"])["revenue_thousand"].cumsum()
    df = _attach_previous_year_value(df, "last_3m_revenue", "_previous_year_last_3m")
    df = _attach_previous_year_value(df, "last_12m_revenue", "_previous_year_last_12m")
    df = _attach_previous_year_value(df, "acc_revenue", "_previous_year_acc_revenue")
    df["last_3m_revenue_yoy"] = df["last_3m_revenue"] / df["_previous_year_last_3m"] - 1
    df["last_12m_revenue_yoy"] = df["last_12m_revenue"] / df["_previous_year_last_12m"] - 1
    df["acc_revenue_yoy"] = df["acc_revenue"] / df["_previous_year_acc_revenue"] - 1

    # ✅ log target 與 log features
    df["log_revenue"] = np.log1p(df["revenue_thousand"].clip(lower=0))

    grouped = df.groupby("stock_id", group_keys=False)
    segment_grouped = df.groupby(["stock_id", "_calendar_segment"], group_keys=False)
    df["log_revenue_lag1"] = segment_grouped["log_revenue"].shift(1)
    df["log_revenue_lag1"] = df["log_revenue_lag1"].fillna(df["log_revenue"])

    # ✅ 成長速度 features
    df["log_mom"] = df["log_revenue"] - segment_grouped["log_revenue"].shift(1)
    df["log_mom"] = df["log_mom"].fillna(0)

    df["log_mom_3"] = (
        df.groupby(["stock_id", "_calendar_segment"])["log_mom"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # ✅ per share
    if "outstanding_shares" in df.columns:
        shares = df["outstanding_shares"].replace(0, np.nan)
    else:
        shares = pd.Series(np.nan, index=df.index)

    df["revenue_per_share"] = (df["revenue_thousand"] * 1000) / shares
    df["last_3m_revenue_per_share"] = (df["last_3m_revenue"] * 1000) / shares
    df["last_12m_revenue_per_share"] = (df["last_12m_revenue"] * 1000) / shares
    df["acc_revenue_per_share"] = (df["acc_revenue"] * 1000) / shares

    # ✅ 春節特徵要在 clean FEATURES 前產生
    CNY_MONTH_BY_YEAR = {
        2019: 2,
        2020: 1,
        2021: 2,
        2022: 2,
        2023: 1,
        2024: 2,
        2025: 1,
    }

    df["is_cny_month"] = df.apply(
        lambda r: 1
        if CNY_MONTH_BY_YEAR.get(int(r["revenue_year"])) == int(r["revenue_month"])
        else 0,
        axis=1,
    )

    df["date"] = pd.to_datetime(
        df["revenue_year"].astype(str)
        + "-"
        + df["revenue_month"].astype(str).str.zfill(2)
        + "-01"
    )

    # Fill model features from each stock's past only. Whole-series medians would
    # leak future observations into earlier validation rows.
    for column in FEATURES:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = _fill_feature_from_past(df, column)

    return df



def get_stock_list(df: pd.DataFrame) -> list[int]:
    return sorted(df["stock_id"].dropna().astype(int).unique().tolist())


def build_forecast(selected_stock: int) -> ForecastResult:
    df = load_revenue_data()
    actual_2025 = load_actual_2025_data()
    forecast_parts, notes = _run_model_suite(df, selected_stock, FORECAST_YEAR)
    weights = _build_validation_weights(df, selected_stock, [part["model"].iloc[0] for part in forecast_parts])
    forecast = _combine_model_outputs(forecast_parts, FORECAST_YEAR, weights)
    evaluation = forecast.merge(
        actual_2025[actual_2025["stock_id"] == selected_stock][
            ["revenue_year", "revenue_month", "actual_revenue"]
        ],
        on=["revenue_year", "revenue_month"],
        how="left",
    )
    metrics = _build_metrics(evaluation)
    recommendation = _build_model_recommendation(metrics, weights)
    forecast = _attach_dividend_yield(forecast, df, selected_stock, FORECAST_YEAR)
    yield_comparison = _build_yield_comparison(forecast, actual_2025, selected_stock, FORECAST_YEAR)
    notes.append(
        "正式模型訓練已排除 outstanding_shares 衍生的每股營收特徵；"
        "殖利率改用每股現金股利資料，不再依賴 outstanding_shares。"
    )
    notes.append(
        f"殖利率優先使用 {DIVIDEND_CASH_FILENAME} 的每股現金股利，股價使用 "
        f"{DAILY_STOCK_PRICE_FILENAME} 的每月最後交易日收盤價。"
    )
    return ForecastResult(
        forecast=forecast,
        yield_comparison=yield_comparison,
        backtest=evaluation,
        metrics=metrics,
        weights=weights,
        recommendation=recommendation,
        notes=notes,
    )


def _run_model_suite(
    df: pd.DataFrame,
    selected_stock: int,
    target_year: int,
) -> tuple[list[pd.DataFrame], list[str]]:
    model_jobs = [
        ("XGBoost", lambda: _xgb_annual_forecast(df, selected_stock, target_year)),
        ("LightGBM", lambda: _lightgbm_annual_forecast(df, selected_stock, target_year)),
        ("CatBoost", lambda: _catboost_annual_forecast(df, selected_stock, target_year)),
        ("SeasonalQuantile", lambda: _seasonal_quantile_forecast(df, selected_stock, target_year)),
    ]
    forecasts: list[pd.DataFrame] = []
    notes: list[str] = []
    for model_name, runner in model_jobs:
        try:
            forecast = runner()
            if forecast is None or forecast.empty:
                notes.append(f"{model_name} 未納入：模型沒有產生預測結果。")
                continue
            forecasts.append(forecast)
        except ImportError as error:
            notes.append(f"{model_name} 未納入：缺少套件 {error.name}。")
        except Exception as error:
            notes.append(f"{model_name} 未納入：{error}")

    if not forecasts:
        raise RuntimeError("沒有任何模型成功產生預測結果。")
    notes.append("MAPE 加權集成使用 2023、2024 歷史驗證 MAPE 估權重；沒有使用 2025 真實值決定權重。")
    return forecasts, notes


def _xgb_annual_forecast(
    df: pd.DataFrame,
    selected_stock: int,
    target_year: int,
    features: list[str] | None = None,
) -> pd.DataFrame:
    features = features or FEATURES
    from sklearn.model_selection import GridSearchCV, PredefinedSplit
    from xgboost import XGBRegressor

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=160,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
    )
    param_grid = {
        "n_estimators": [80, 160],
        "max_depth": [2, 3],
        "learning_rate": [0.03, 0.08],
    }

    model_df = _attach_next_year_target(df)
    complete_history = model_df.get(
        "_history_12m_complete",
        pd.Series(True, index=model_df.index),
    )
    train_df = model_df[
        (model_df["revenue_year"] >= TRAIN_START_YEAR)
        & (model_df["revenue_year"] <= target_year - 2)
        & complete_history
    ].dropna(subset=features + ["target_next_year"])
    predict_df = model_df[
        (model_df["stock_id"] == selected_stock)
        & (model_df["revenue_year"] == target_year - 1)
        & complete_history
    ].sort_values("revenue_month")

    if len(train_df) < 24 or len(predict_df) != 12:
        raise ValueError(f"{selected_stock} 缺少連續且足夠的歷史資料，無法預測 {target_year} 年 12 個月。")

    validation_year = int(train_df["revenue_year"].max())
    test_fold = np.where(train_df["revenue_year"].to_numpy() == validation_year, 0, -1)
    if (test_fold == 0).any() and (test_fold == -1).any():
        splitter = PredefinedSplit(test_fold)
        search = GridSearchCV(
            base_model,
            param_grid=param_grid,
            cv=splitter,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        search.fit(train_df[features], train_df["target_next_year"])
        fitted_model = search.best_estimator_
    else:
        fitted_model = base_model.fit(train_df[features], train_df["target_next_year"])

    prediction_log = fitted_model.predict(predict_df[features])

    # ✅ 重點：轉回原始營收
    prediction = np.expm1(prediction_log)

    return pd.DataFrame(
        {
            "revenue_year": target_year,
            "revenue_month": predict_df["revenue_month"].to_numpy(),
            "model": "XGBoost",
            "predicted_revenue": np.maximum(prediction, 0).round().astype(int),
        }
    )


def _lightgbm_annual_forecast(
    df: pd.DataFrame,
    selected_stock: int,
    target_year: int,
    features: list[str] | None = None,
) -> pd.DataFrame:
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        objective="regression",
        random_state=42,
        n_estimators=160,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=15,
        min_child_samples=5,
        verbosity=-1,
    )
    return _tree_annual_forecast(df, selected_stock, target_year, model, "LightGBM", features=features)


def _catboost_annual_forecast(
    df: pd.DataFrame,
    selected_stock: int,
    target_year: int,
    features: list[str] | None = None,
) -> pd.DataFrame:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=42,
        iterations=220,
        learning_rate=0.05,
        depth=4,
        verbose=False,
    )
    return _tree_annual_forecast(df, selected_stock, target_year, model, "CatBoost", features=features)


def _tree_annual_forecast(
    df: pd.DataFrame,
    selected_stock: int,
    target_year: int,
    model,
    model_name: str,
    features: list[str] | None = None,
) -> pd.DataFrame:
    features = features or FEATURES
    model_df = _attach_next_year_target(df)
    complete_history = model_df.get(
        "_history_12m_complete",
        pd.Series(True, index=model_df.index),
    )
    train_df = model_df[
        (model_df["revenue_year"] >= TRAIN_START_YEAR)
        & (model_df["revenue_year"] <= target_year - 2)
        & complete_history
    ].dropna(subset=features + ["target_next_year"])
    predict_df = model_df[
        (model_df["stock_id"] == selected_stock)
        & (model_df["revenue_year"] == target_year - 1)
        & complete_history
    ].sort_values("revenue_month")

    if len(train_df) < 24 or len(predict_df) != 12:
        raise ValueError(f"{selected_stock} 缺少連續且足夠的歷史資料，無法預測 {target_year} 年 12 個月。")

    model.fit(train_df[features], train_df["target_next_year"])
    prediction_log = model.predict(predict_df[features])
    prediction = np.expm1(prediction_log)
    return pd.DataFrame(
        {
            "revenue_year": target_year,
            "revenue_month": predict_df["revenue_month"].to_numpy(),
            "model": model_name,
            "predicted_revenue": np.maximum(prediction, 0).round().astype(int),
        }
    )


def _seasonal_quantile_forecast(df: pd.DataFrame, selected_stock: int, target_year: int) -> pd.DataFrame:
    stock_df = df[df["stock_id"] == selected_stock].copy()
    stock_df = stock_df[stock_df["revenue_year"] < target_year].sort_values(
        ["revenue_year", "revenue_month"]
    )
    if stock_df.empty:
        raise ValueError(f"{selected_stock} 缺少 {target_year} 年以前的營收資料，無法建立季節 fallback。")
    source = stock_df[stock_df["revenue_year"] == target_year - 1].sort_values("revenue_month")
    fallback_base = pd.to_numeric(source["revenue_thousand"], errors="coerce").dropna()
    if fallback_base.empty:
        fallback_base = pd.to_numeric(stock_df["revenue_thousand"], errors="coerce").dropna().tail(12)
    if fallback_base.empty:
        raise ValueError(f"{selected_stock} 缺少可用營收基準，無法建立季節 fallback。")
    default_base = float(fallback_base.median())
    rows = []
    for month in range(1, 13):
        history = stock_df[
            (stock_df["revenue_month"] == month)
            & (stock_df["revenue_year"] >= TRAIN_START_YEAR)
            & (stock_df["revenue_year"] <= target_year - 1)
        ].sort_values("revenue_year")
        yoy = history["revenue_thousand"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        growth = yoy.tail(3).median() if not yoy.empty else 0.0
        source_base = pd.to_numeric(
            source[source["revenue_month"] == month]["revenue_thousand"],
            errors="coerce",
        ).dropna()
        if not source_base.empty:
            base = float(source_base.iloc[-1])
        else:
            historical_base = pd.to_numeric(history["revenue_thousand"], errors="coerce").dropna()
            base = float(historical_base.iloc[-1]) if not historical_base.empty else default_base
        prediction = max(base * (1 + growth), 0)
        rows.append(
            {
                "revenue_year": target_year,
                "revenue_month": month,
                "model": "SeasonalQuantile",
                "predicted_revenue": int(round(prediction)),
            }
        )
    return pd.DataFrame(rows)


def _combine_model_outputs(parts: list[pd.DataFrame], target_year: int, weights: pd.DataFrame | None = None) -> pd.DataFrame:
    all_predictions = pd.concat(parts, ignore_index=True)
    pivot = all_predictions.pivot_table(
        index=["revenue_year", "revenue_month"],
        columns="model",
        values="predicted_revenue",
        aggfunc="mean",
    ).reset_index()
    model_columns = [column for column in pivot.columns if column not in ["revenue_year", "revenue_month"]]
    weight_map = _normalize_weight_map(model_columns, weights)
    weighted_values = sum(pivot[column] * weight_map[column] for column in model_columns)
    pivot["ensemble_revenue"] = weighted_values.round().astype(int)
    pivot["model_spread"] = pivot[model_columns].std(axis=1).fillna(0).round().astype(int)
    pivot["lower_bound"] = (pivot["ensemble_revenue"] - 1.28 * pivot["model_spread"]).clip(lower=0).round().astype(int)
    pivot["upper_bound"] = (pivot["ensemble_revenue"] + 1.28 * pivot["model_spread"]).round().astype(int)
    pivot["date"] = pd.to_datetime(str(target_year) + "-" + pivot["revenue_month"].astype(str).str.zfill(2) + "-01")
    return pivot.sort_values("revenue_month")


def _attach_dividend_yield(
    forecast: pd.DataFrame,
    revenue_data: pd.DataFrame,
    selected_stock: int,
    target_year: int,
) -> pd.DataFrame:
    del revenue_data  # The shared financial module owns canonical evidence loading.
    normalized = forecast[["revenue_month", "ensemble_revenue"]].rename(
        columns={"ensemble_revenue": "predicted_revenue"}
    )
    yield_rows = _build_ensemble_yield_forecast(
        normalized,
        selected_stock=int(selected_stock),
        target_year=int(target_year),
        model_family="ensemble_forecast",
        model_name="ensemble_revenue",
        data_dir=DATA_DIR,
        actual_revenue=None,
        as_of_date=f"{int(target_year)}-01-10",
    )
    if yield_rows.empty:
        return forecast.copy()

    yield_columns = [
        "revenue_month",
        "forecast_annual_revenue",
        "annual_eps_reference_year",
        "annual_eps",
        "payout_ratio",
        "cash_dividend_per_share",
        "cash_dividend_source",
        "estimated_eps",
        "estimated_cash_dividend",
        "stock_price_date",
        "stock_price",
        "stock_price_source",
        "dividend_yield_percent",
        "as_of_price_date",
        "as_of_stock_price",
        "as_of_price_yield_percent",
    ]
    enriched = forecast.merge(
        yield_rows[yield_columns].drop_duplicates("revenue_month"),
        on="revenue_month",
        how="left",
        validate="one_to_one",
    )

    for column in [
        "forecast_annual_revenue",
        "annual_eps_reference_year",
        "annual_eps",
        "cash_dividend_per_share",
        "estimated_eps",
        "estimated_cash_dividend",
        "stock_price",
        "dividend_yield_percent",
    ]:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
    for column in ["payout_ratio"]:
        enriched[column] = enriched[column].round(4)
    for column in [
        "annual_eps",
        "cash_dividend_per_share",
        "estimated_eps",
        "estimated_cash_dividend",
        "stock_price",
        "dividend_yield_percent",
    ]:
        enriched[column] = enriched[column].round(2)
    enriched["forecast_annual_revenue"] = enriched["forecast_annual_revenue"].round().astype("Int64")
    enriched["annual_eps_reference_year"] = enriched["annual_eps_reference_year"].round().astype("Int64")
    return enriched.sort_values("revenue_month")


def build_yield_forecast(
    revenue_forecast: pd.DataFrame,
    selected_stock: int,
    target_year: int,
    model_family: str,
    model_name: str,
    revenue_data: pd.DataFrame | None = None,
    actual_2025: pd.DataFrame | None = None,
) -> pd.DataFrame:
    del revenue_data  # Kept for compatibility; shared evidence is loaded from DATA_DIR.
    actual_2025 = actual_2025 if actual_2025 is not None else load_actual_2025_data()
    return _build_ensemble_yield_forecast(
        revenue_forecast,
        selected_stock=int(selected_stock),
        target_year=int(target_year),
        model_family=str(model_family),
        model_name=str(model_name),
        data_dir=DATA_DIR,
        actual_revenue=actual_2025,
        as_of_date=f"{int(target_year)}-01-10",
    )


def _get_forecast_dividend_info(
    selected_stock: int,
    target_year: int,
    forecast_annual_revenue_thousand: float,
    revenue_data: pd.DataFrame,
) -> dict[str, float | str]:
    estimated_eps, annual_eps_year, eps_source = _estimate_eps_from_revenue_forecast(
        selected_stock,
        target_year,
        forecast_annual_revenue_thousand,
        revenue_data,
    )
    payout_ratio, payout_source = _get_historical_payout_ratio(selected_stock, target_year)
    if pd.isna(payout_ratio):
        policy = _get_dividend_policy(selected_stock, target_year)
        payout_ratio = policy["payout_ratio"]
        payout_source = str(policy["source"])

    cash_dividend = estimated_eps * payout_ratio if pd.notna(estimated_eps) else np.nan
    if pd.notna(cash_dividend):
        cash_dividend = float(max(cash_dividend, 0))

    return {
        "annual_eps_reference_year": annual_eps_year,
        "annual_eps": estimated_eps,
        "payout_ratio": payout_ratio,
        "cash_dividend_per_share": cash_dividend,
        "cash_dividend_source": f"{eps_source}; payout={payout_source}",
    }


def _estimate_eps_from_revenue_forecast(
    selected_stock: int,
    target_year: int,
    forecast_annual_revenue_thousand: float,
    revenue_data: pd.DataFrame,
) -> tuple[float, float, str]:
    stock_revenue = revenue_data[
        (revenue_data["stock_id"] == selected_stock) & (revenue_data["revenue_year"] < target_year)
    ]
    annual_revenue = (
        stock_revenue.groupby("revenue_year", as_index=False)["revenue_thousand"]
        .sum()
        .rename(columns={"revenue_thousand": "annual_revenue_thousand"})
    )

    try:
        eps = load_eps_data()
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        annual_eps, annual_eps_year = _get_annual_eps(
            selected_stock,
            target_year - 1,
            as_of_date=pd.Timestamp(target_year, 1, 10),
        )
        return annual_eps, annual_eps_year, "latest EPS fallback"

    eps = _filter_eps_available_as_of(eps, pd.Timestamp(target_year, 1, 10))
    stock_eps = eps[(eps["stock_id"] == selected_stock) & (eps["eps_year"] < target_year)]
    annual_eps = stock_eps.groupby("eps_year", as_index=False).agg(
        annual_eps=("latest_eps", "sum"),
        quarter_count=("latest_eps", "count"),
    )
    merged = annual_revenue.merge(annual_eps, left_on="revenue_year", right_on="eps_year", how="inner")
    if merged.empty:
        annual_eps, annual_eps_year = _get_annual_eps(
            selected_stock,
            target_year - 1,
            as_of_date=pd.Timestamp(target_year, 1, 10),
        )
        return annual_eps, annual_eps_year, "latest EPS fallback"

    candidates = merged[
        (merged["annual_revenue_thousand"] > 0)
        & (merged["annual_eps"].notna())
        & (merged["quarter_count"] >= 4)
    ].copy()
    if candidates.empty:
        annual_eps, annual_eps_year = _get_annual_eps(
            selected_stock,
            target_year - 1,
            as_of_date=pd.Timestamp(target_year, 1, 10),
        )
        return annual_eps, annual_eps_year, "latest EPS fallback"

    candidates["eps_to_revenue_ratio"] = candidates["annual_eps"] / candidates["annual_revenue_thousand"]
    candidates = candidates.replace([np.inf, -np.inf], np.nan).dropna(subset=["eps_to_revenue_ratio"])
    if candidates.empty:
        annual_eps, annual_eps_year = _get_annual_eps(
            selected_stock,
            target_year - 1,
            as_of_date=pd.Timestamp(target_year, 1, 10),
        )
        return annual_eps, annual_eps_year, "latest EPS fallback"

    recent = candidates.sort_values("revenue_year").tail(3)
    ratio = float(recent["eps_to_revenue_ratio"].median())
    reference_year = float(recent["revenue_year"].max())
    estimated_eps = float(forecast_annual_revenue_thousand * ratio)
    return estimated_eps, reference_year, "forecast revenue x historical EPS/revenue"


def _get_historical_payout_ratio(selected_stock: int, target_year: int) -> tuple[float, str]:
    candidates = _build_historical_payout_candidates(target_year)
    if candidates.empty or "stock_id" not in candidates.columns:
        return np.nan, ""
    candidates = candidates[candidates["stock_id"] == int(selected_stock)].copy()
    if candidates.empty:
        return np.nan, ""

    recent = candidates.sort_values("fiscal_year").tail(3)
    payout_ratio = float(np.clip(recent["payout_ratio"].median(), 0, 1.5))
    latest_year = int(recent["fiscal_year"].max())
    return payout_ratio, f"{DIVIDEND_CASH_FILENAME} time-safe historical payout through fiscal {latest_year}"


def _build_historical_payout_candidates(target_year: int) -> pd.DataFrame:
    try:
        dividends = load_cash_dividend_data()
        eps = load_eps_data()
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        return pd.DataFrame(
            columns=[
                "stock_id",
                "fiscal_year",
                "cash_dividend_per_share",
                "annual_eps",
                "quarter_count",
                "payout_ratio",
            ]
        )

    safe_dividends = dividends.copy()
    eps = _filter_eps_available_as_of(eps, pd.Timestamp(target_year, 1, 10))
    ex_year = pd.to_numeric(safe_dividends["ex_dividend_year"], errors="coerce")
    fiscal_year = pd.to_numeric(safe_dividends["fiscal_year"], errors="coerce")
    time_safe_mask = ex_year.lt(int(target_year)) | (ex_year.isna() & fiscal_year.lt(int(target_year) - 1))
    safe_dividends = safe_dividends[time_safe_mask].copy()
    if safe_dividends.empty:
        return pd.DataFrame()

    dividend_by_year = safe_dividends.groupby(["stock_id", "fiscal_year"], as_index=False).agg(
        cash_dividend_per_share=("TotalCashDividend", "sum")
    )
    safe_eps = eps[eps["eps_year"] < target_year]
    annual_eps = safe_eps.groupby(["stock_id", "eps_year"], as_index=False).agg(
        annual_eps=("latest_eps", "sum"),
        quarter_count=("latest_eps", "count"),
    )
    merged = dividend_by_year.merge(
        annual_eps,
        left_on=["stock_id", "fiscal_year"],
        right_on=["stock_id", "eps_year"],
        how="inner",
    )
    candidates = merged[
        (merged["annual_eps"] > 0)
        & (merged["cash_dividend_per_share"] >= 0)
        & (merged["quarter_count"] >= 4)
    ].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates["payout_ratio"] = candidates["cash_dividend_per_share"] / candidates["annual_eps"]
    candidates = candidates.replace([np.inf, -np.inf], np.nan).dropna(subset=["payout_ratio"])
    candidates["payout_ratio"] = candidates["payout_ratio"].clip(lower=0, upper=1.5)
    return candidates


def _get_cross_sectional_historical_payout_ratio(target_year: int) -> tuple[float, str]:
    candidates = _build_historical_payout_candidates(target_year)
    if candidates.empty:
        return np.nan, "no availability-safe historical payout evidence"

    per_stock_ratios = []
    latest_year = int(candidates["fiscal_year"].max())
    for _, stock_rows in candidates.groupby("stock_id"):
        per_stock_ratios.append(float(stock_rows.sort_values("fiscal_year").tail(3)["payout_ratio"].median()))

    payout_ratio = float(np.clip(pd.Series(per_stock_ratios).median(), 0, 1.5))
    return payout_ratio, f"{DIVIDEND_CASH_FILENAME} time-safe cross-sectional payout through fiscal {latest_year}"


def _build_yield_comparison(
    forecast: pd.DataFrame,
    actual_2025: pd.DataFrame,
    selected_stock: int,
    target_year: int,
) -> pd.DataFrame:
    required_columns = {
        "date",
        "revenue_year",
        "revenue_month",
        "ensemble_revenue",
        "forecast_annual_revenue",
        "estimated_cash_dividend",
        "cash_dividend_source",
        "stock_price_date",
        "stock_price",
        "stock_price_source",
        "dividend_yield_percent",
    }
    if forecast.empty or not required_columns.issubset(forecast.columns):
        return pd.DataFrame()

    comparison = forecast[
        [
            "date",
            "revenue_year",
            "revenue_month",
            "ensemble_revenue",
            "forecast_annual_revenue",
            "estimated_cash_dividend",
            "cash_dividend_source",
            "stock_price_date",
            "stock_price",
            "stock_price_source",
            "dividend_yield_percent",
        ]
    ].copy()
    comparison = comparison.rename(
        columns={
            "ensemble_revenue": "predicted_revenue",
            "forecast_annual_revenue": "predicted_annual_revenue",
            "estimated_cash_dividend": "predicted_cash_dividend_per_share",
            "cash_dividend_source": "predicted_cash_dividend_source",
            "dividend_yield_percent": "predicted_dividend_yield_percent",
        }
    )

    actual = actual_2025[actual_2025["stock_id"] == selected_stock][
        ["revenue_year", "revenue_month", "actual_revenue"]
    ].copy()
    actual_annual_revenue = float(actual["actual_revenue"].sum()) if not actual.empty else np.nan
    comparison = comparison.merge(actual, on=["revenue_year", "revenue_month"], how="left")

    actual_dividend = _get_actual_cash_dividend_info(selected_stock, target_year)
    comparison["actual_annual_revenue"] = actual_annual_revenue
    comparison["actual_cash_dividend_per_share"] = actual_dividend["actual_cash_dividend_per_share"]
    comparison["actual_cash_dividend_source"] = actual_dividend["actual_cash_dividend_source"]
    comparison["actual_dividend_yield_percent"] = np.where(
        comparison["stock_price"] > 0,
        comparison["actual_cash_dividend_per_share"] / comparison["stock_price"] * 100,
        np.nan,
    )
    comparison["yield_error_percent_point"] = (
        comparison["predicted_dividend_yield_percent"] - comparison["actual_dividend_yield_percent"]
    )
    comparison["yield_abs_error_percent_point"] = comparison["yield_error_percent_point"].abs()

    integer_columns = [
        "predicted_revenue",
        "predicted_annual_revenue",
        "actual_revenue",
        "actual_annual_revenue",
    ]
    for column in integer_columns:
        comparison[column] = pd.to_numeric(comparison[column], errors="coerce").round().astype("Int64")

    decimal_columns = [
        "predicted_cash_dividend_per_share",
        "actual_cash_dividend_per_share",
        "stock_price",
        "predicted_dividend_yield_percent",
        "actual_dividend_yield_percent",
        "yield_error_percent_point",
        "yield_abs_error_percent_point",
    ]
    for column in decimal_columns:
        comparison[column] = pd.to_numeric(comparison[column], errors="coerce").round(2)

    return comparison.sort_values("revenue_month").reset_index(drop=True)


def _get_actual_cash_dividend_info(selected_stock: int, target_year: int) -> dict[str, float | str]:
    try:
        dividends = load_cash_dividend_data()
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        return {
            "actual_cash_dividend_per_share": np.nan,
            "actual_cash_dividend_source": "actual cash dividend unavailable",
        }

    stock_dividends = dividends[dividends["stock_id"] == selected_stock]
    matched = stock_dividends[stock_dividends["ex_dividend_year"] == target_year]
    if matched.empty:
        return {
            "actual_cash_dividend_per_share": np.nan,
            "actual_cash_dividend_source": f"{DIVIDEND_CASH_FILENAME} no {target_year} ex-dividend record",
        }

    cash_dividend = float(matched["TotalCashDividend"].sum())
    return {
        "actual_cash_dividend_per_share": cash_dividend,
        "actual_cash_dividend_source": DIVIDEND_CASH_FILENAME,
    }


def _get_dividend_policy(selected_stock: int, target_year: int = FORECAST_YEAR) -> dict[str, float | str]:
    try:
        policy = load_dividend_policy_data()
        matched = policy[policy["stock_id"] == selected_stock]
        if not matched.empty:
            row = matched.iloc[0]
            return {
                "net_margin_assumption": float(row["net_margin_assumption"]),
                "payout_ratio": float(row["payout_ratio"]),
                "source": DIVIDEND_POLICY_FILENAME,
            }
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        pass
    payout_ratio, source = _get_cross_sectional_historical_payout_ratio(target_year)
    return {
        "net_margin_assumption": np.nan,
        "payout_ratio": payout_ratio,
        "source": source,
    }


def _get_cash_dividend_info(selected_stock: int, target_year: int) -> dict[str, float | str]:
    try:
        dividends = load_cash_dividend_data()
        stock_dividends = dividends[dividends["stock_id"] == selected_stock]
        matched = stock_dividends[stock_dividends["ex_dividend_year"] == target_year]
        source = DIVIDEND_CASH_FILENAME
        if matched.empty:
            matched = stock_dividends[stock_dividends["fiscal_year"] == target_year - 1]
        if matched.empty:
            prior = stock_dividends[stock_dividends["ex_dividend_year"] <= target_year].dropna(
                subset=["ex_dividend_year"]
            )
            if not prior.empty:
                latest_ex_year = int(prior["ex_dividend_year"].max())
                matched = prior[prior["ex_dividend_year"] == latest_ex_year]
                source = f"{DIVIDEND_CASH_FILENAME} (最近年度 {latest_ex_year})"
        if not matched.empty:
            cash_dividend = float(matched["TotalCashDividend"].sum())
            fiscal_years = pd.to_numeric(matched["fiscal_year"], errors="coerce").dropna()
            annual_eps_target_year = int(fiscal_years.max()) if not fiscal_years.empty else target_year - 1
            annual_eps, annual_eps_year = _get_annual_eps(selected_stock, annual_eps_target_year)
            payout_ratio = cash_dividend / annual_eps if pd.notna(annual_eps) and annual_eps > 0 else np.nan
            return {
                "annual_eps_reference_year": annual_eps_year,
                "annual_eps": annual_eps,
                "payout_ratio": payout_ratio,
                "cash_dividend_per_share": cash_dividend,
                "cash_dividend_source": source,
            }
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        pass

    annual_eps, annual_eps_year = _get_annual_eps(selected_stock, target_year - 1)
    policy = _get_dividend_policy(selected_stock, target_year)
    payout_ratio = policy["payout_ratio"]
    cash_dividend = annual_eps * payout_ratio if pd.notna(annual_eps) else np.nan
    return {
        "annual_eps_reference_year": annual_eps_year,
        "annual_eps": annual_eps,
        "payout_ratio": payout_ratio,
        "cash_dividend_per_share": cash_dividend,
        "cash_dividend_source": str(policy["source"]),
    }


def _get_annual_eps(
    selected_stock: int,
    fiscal_year: int,
    *,
    as_of_date: pd.Timestamp | None = None,
) -> tuple[float, float]:
    try:
        eps = load_eps_data()
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        return np.nan, np.nan

    if as_of_date is not None:
        eps = _filter_eps_available_as_of(eps, as_of_date)
    stock_eps = eps[eps["stock_id"] == selected_stock].copy()
    if stock_eps.empty:
        return np.nan, np.nan

    annual = stock_eps.groupby("eps_year", as_index=False).agg(
        annual_eps=("latest_eps", "sum"),
        quarter_count=("latest_eps", "count"),
    )
    candidates = annual[(annual["eps_year"] <= fiscal_year) & (annual["quarter_count"] >= 4)]
    if candidates.empty:
        return np.nan, np.nan

    exact = candidates[candidates["eps_year"] == fiscal_year]
    row = exact.iloc[-1] if not exact.empty else candidates.sort_values("eps_year").iloc[-1]
    return float(row["annual_eps"]), float(row["eps_year"])


def _get_stock_prices(revenue_data: pd.DataFrame, selected_stock: int, target_year: int) -> pd.DataFrame:
    del revenue_data  # Kept in the signature for compatibility with the public yield builders.
    empty = pd.DataFrame(
        {
            "price_month": range(1, 13),
            "price_date": [pd.NaT] * 12,
            "close_price": [np.nan] * 12,
            "price_source": ["stock price unavailable"] * 12,
        }
    )
    try:
        prices = load_stock_price_data(selected_stock=selected_stock, target_year=None)
        prices = prices[prices["stock_id"] == selected_stock].copy()
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        return empty

    if prices.empty:
        return empty

    prices["price_year"] = pd.to_numeric(prices["price_year"], errors="coerce")
    prices["price_month"] = pd.to_numeric(prices["price_month"], errors="coerce")
    prices["close_price"] = pd.to_numeric(prices["close_price"], errors="coerce")
    prices["price_date"] = pd.to_datetime(prices["price_date"], errors="coerce")
    prices = prices.dropna(subset=["price_year", "price_month", "close_price"])
    prices = prices[prices["close_price"] > 0].copy()
    if prices.empty:
        return empty

    prices["price_year"] = prices["price_year"].astype(int)
    prices["price_month"] = prices["price_month"].astype(int)
    prices = prices.sort_values(["price_year", "price_month", "price_date"])

    rows: list[dict[str, object]] = []
    for month in range(1, 13):
        available = prices[
            (prices["price_year"] < target_year)
            | ((prices["price_year"] == target_year) & (prices["price_month"] <= month))
        ]
        if available.empty:
            rows.append(
                {
                    "price_month": month,
                    "price_date": pd.NaT,
                    "close_price": np.nan,
                    "price_source": "stock price unavailable",
                }
            )
            continue

        latest = available.iloc[-1]
        is_exact_month = int(latest["price_year"]) == target_year and int(latest["price_month"]) == month
        source = str(latest["price_source"])
        if not is_exact_month:
            source = f"{source} (last known close)"
        rows.append(
            {
                "price_month": month,
                "price_date": latest["price_date"],
                "close_price": float(latest["close_price"]),
                "price_source": source,
            }
        )
    return pd.DataFrame(rows)


def _normalize_weight_map(model_columns: list[str], weights: pd.DataFrame | None) -> dict[str, float]:
    if weights is None or weights.empty:
        return {model: 1 / len(model_columns) for model in model_columns}
    raw = {
        row["model"]: float(row["weight"])
        for _, row in weights.iterrows()
        if row["model"] in model_columns and pd.notna(row["weight"])
    }
    missing = [model for model in model_columns if model not in raw]
    if missing:
        fallback = min(raw.values()) if raw else 1.0
        for model in missing:
            raw[model] = fallback
    total = sum(raw.values())
    if total <= 0:
        return {model: 1 / len(model_columns) for model in model_columns}
    return {model: raw[model] / total for model in model_columns}


def _mean_absolute_error(actual, predicted) -> float:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual_values - predicted_values)))


def _mean_absolute_percentage_error(actual, predicted) -> float:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    valid_mask = np.isfinite(actual_values) & np.isfinite(predicted_values) & (actual_values != 0)
    if not valid_mask.any():
        return np.nan
    return float(np.mean(np.abs((actual_values[valid_mask] - predicted_values[valid_mask]) / actual_values[valid_mask])))


def _build_validation_weights(
    df: pd.DataFrame,
    selected_stock: int,
    model_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[pd.DataFrame] = []
    for validation_year in [2023, 2024]:
        actual = df[
            (df["stock_id"] == selected_stock) & (df["revenue_year"] == validation_year)
        ][["revenue_month", "revenue_thousand"]].rename(columns={"revenue_thousand": "actual_revenue"})
        if len(actual) != 12:
            continue
        for model_name in model_names:
            try:
                forecast = _forecast_model_by_name(df, selected_stock, validation_year, model_name)
                evaluated = forecast.merge(actual, on="revenue_month", how="inner")
                if len(evaluated) == 12:
                    canonical_model_name = str(forecast["model"].iloc[0])
                    rows.append(
                        {
                            "model": canonical_model_name,
                            "validation_year": validation_year,
                            "MAPE": _mean_absolute_percentage_error(
                                evaluated["actual_revenue"], evaluated["predicted_revenue"]
                            )
                            * 100,
                        }
                    )
                    evaluated = evaluated[
                        ["revenue_month", "actual_revenue", "predicted_revenue"]
                    ].copy()
                    evaluated["validation_year"] = validation_year
                    evaluated["model"] = canonical_model_name
                    prediction_rows.append(evaluated)
            except Exception:
                continue

    if not rows:
        return pd.DataFrame(
            {
                "model": model_names,
                "validation_mape": [np.nan] * len(model_names),
                "weight": [1 / len(model_names)] * len(model_names),
                "ensemble_validation_mape": [np.nan] * len(model_names),
                "validation_year_count": [0] * len(model_names),
            }
        )

    score_rows = pd.DataFrame(rows)
    report = (
        score_rows.groupby("model", as_index=False)
        .agg(
            validation_mape=("MAPE", "mean"),
            validation_year_count=("validation_year", "nunique"),
        )
    )
    report = report[report["model"].isin(model_names)]
    missing = [model for model in model_names if model not in report["model"].tolist()]
    if missing:
        worst_mape = report["validation_mape"].max() if not report.empty else 100.0
        report = pd.concat(
            [
                report,
                pd.DataFrame(
                    {
                        "model": missing,
                        "validation_mape": [worst_mape] * len(missing),
                        "validation_year_count": [0] * len(missing),
                    }
                ),
            ],
            ignore_index=True,
        )
    report["raw_weight"] = 1 / report["validation_mape"].clip(lower=0.01)
    report["weight"] = report["raw_weight"] / report["raw_weight"].sum()

    predictions = pd.concat(prediction_rows, ignore_index=True)
    prediction_matrix = predictions.pivot_table(
        index=["validation_year", "revenue_month", "actual_revenue"],
        columns="model",
        values="predicted_revenue",
        aggfunc="first",
    )
    weight_lookup = report.set_index("model")["weight"]
    available_models = [model for model in weight_lookup.index if model in prediction_matrix.columns]
    if available_models:
        weighted_values = prediction_matrix[available_models].mul(
            weight_lookup.reindex(available_models),
            axis=1,
        )
        available_weight = prediction_matrix[available_models].notna().mul(
            weight_lookup.reindex(available_models),
            axis=1,
        ).sum(axis=1)
        ensemble_prediction = weighted_values.sum(axis=1) / available_weight.replace(0, np.nan)
        ensemble_validation_mape = (
            _mean_absolute_percentage_error(
                prediction_matrix.index.get_level_values("actual_revenue"),
                ensemble_prediction,
            )
            * 100
        )
    else:
        ensemble_validation_mape = np.nan
    report["ensemble_validation_mape"] = ensemble_validation_mape
    return report[
        [
            "model",
            "validation_mape",
            "weight",
            "ensemble_validation_mape",
            "validation_year_count",
        ]
    ].sort_values("weight", ascending=False)


def _forecast_model_by_name(
    df: pd.DataFrame,
    selected_stock: int,
    target_year: int,
    model_name: str,
    features: list[str] | None = None,
) -> pd.DataFrame:
    features = features or FEATURES
    if model_name == "XGBoost":
        return _xgb_annual_forecast(df, selected_stock, target_year, features=features)
    if model_name == "LightGBM":
        return _lightgbm_annual_forecast(df, selected_stock, target_year, features=features)
    if model_name == "CatBoost":
        return _catboost_annual_forecast(df, selected_stock, target_year, features=features)
    if model_name == "SeasonalQuantile":
        return _seasonal_quantile_forecast(df, selected_stock, target_year)
    raise ValueError(f"未知模型：{model_name}")


def _build_metrics(backtest: pd.DataFrame) -> pd.DataFrame:
    valid = backtest.dropna(subset=["actual_revenue"])
    if valid.empty:
        return pd.DataFrame(columns=["model", "MAE", "MAPE"])
    actual = valid["actual_revenue"].to_numpy()
    model_columns = [
        column
        for column in valid.columns
        if column not in ["revenue_year", "revenue_month", "date", "actual_revenue", "lower_bound", "upper_bound", "model_spread"]
    ]
    rows = []
    for column in model_columns:
        predicted = valid[column].to_numpy()
        rows.append(
            {
                "model": column,
                "MAE": round(_mean_absolute_error(actual, predicted), 0),
                "MAPE": round(_mean_absolute_percentage_error(actual, predicted) * 100, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("MAPE")


def _build_model_recommendation(metrics: pd.DataFrame, weights: pd.DataFrame) -> dict[str, str]:
    if weights.empty:
        return {
            "recommendation": "目前缺少歷史驗證資料，無法產生個股模型建議。",
            "historical_best_model": "無",
            "historical_best_mape": "無",
            "historical_ensemble_mape": "無",
            "actual_best_model": "無",
            "actual_best_mape": "無",
            "highest_weight_model": "無",
            "highest_weight": "無",
            "reason": "weights 為空。",
        }

    historical = weights.dropna(subset=["validation_mape", "weight"]).copy()
    actual_best_model = "無"
    actual_best_mape = "無"
    if not metrics.empty:
        single_metrics = metrics[metrics["model"] != "ensemble_revenue"].copy()
        if not single_metrics.empty:
            actual_best = single_metrics.sort_values("MAPE").iloc[0]
            actual_best_model = str(actual_best["model"])
            actual_best_mape = f"{float(actual_best['MAPE']):.2f}%"

    weight_candidates = weights.dropna(subset=["weight"]).copy()
    if not weight_candidates.empty:
        highest_weight_row = weight_candidates.sort_values("weight", ascending=False).iloc[0]
        highest_weight_model = str(highest_weight_row["model"])
        highest_weight = f"{float(highest_weight_row['weight']) * 100:.2f}%"
    else:
        highest_weight_model = "無"
        highest_weight = "無"

    if historical.empty:
        return {
            "recommendation": "目前缺少足夠歷史驗證資料，無法產生個股模型建議。",
            "historical_best_model": "無",
            "historical_best_mape": "無",
            "historical_ensemble_mape": "無",
            "actual_best_model": actual_best_model,
            "actual_best_mape": actual_best_mape,
            "highest_weight_model": highest_weight_model,
            "highest_weight": highest_weight,
            "reason": "2023/2024 歷史驗證資料不足，權重為 fallback 權重。",
        }

    historical_best = historical.sort_values("validation_mape").iloc[0]
    historical_best_mape = float(historical_best["validation_mape"])
    ensemble_scores = pd.to_numeric(
        historical.get("ensemble_validation_mape", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    historical_ensemble_mape = (
        float(ensemble_scores.iloc[0])
        if not ensemble_scores.empty
        else float((historical["validation_mape"] * historical["weight"]).sum())
    )
    close_models = historical[historical["validation_mape"] <= historical_best_mape + 1.0]["model"].tolist()

    if historical_best_mape + 3 < historical_ensemble_mape:
        recommendation = "此股票較適合單一模型"
        reason = (
            f"{historical_best['model']} 的歷史驗證 MAPE 比加權集成估計誤差低超過 3 個百分點，"
            "正式預測時應優先參考該單一模型。"
        )
    elif len(close_models) >= 2:
        recommendation = "此股票適合集成"
        reason = "多個模型的歷史驗證 MAPE 差距在 1 個百分點內，代表沒有明顯唯一勝出模型。"
    else:
        recommendation = "此股票建議以加權集成為主，並觀察最佳單一模型"
        reason = "歷史最佳單一模型沒有明顯贏過加權集成超過 3 個百分點，集成較能降低單一模型失準風險。"

    return {
        "recommendation": recommendation,
        "historical_best_model": str(historical_best["model"]),
        "historical_best_mape": f"{historical_best_mape:.2f}%",
        "historical_ensemble_mape": f"{historical_ensemble_mape:.2f}%",
        "actual_best_model": actual_best_model,
        "actual_best_mape": actual_best_mape,
        "highest_weight_model": highest_weight_model,
        "highest_weight": highest_weight,
        "reason": reason,
    }


def make_revenue_summary(
    forecast: pd.DataFrame,
    backtest_metrics: pd.DataFrame,
    validation_weights: pd.DataFrame | None = None,
) -> dict[str, str]:
    annual_total = int(forecast["ensemble_revenue"].sum())
    validation = pd.DataFrame()
    if validation_weights is not None and "validation_mape" in validation_weights.columns:
        validation = validation_weights.dropna(subset=["validation_mape"])
    if not validation.empty:
        best_row = validation.sort_values("validation_mape").iloc[0]
        best_model = best_row["model"]
        best_mape = best_row["validation_mape"]
    else:
        best_model = "無"
        best_mape = np.nan
    peak_month = int(forecast.loc[forecast["ensemble_revenue"].idxmax(), "revenue_month"])
    low_month = int(forecast.loc[forecast["ensemble_revenue"].idxmin(), "revenue_month"])
    model_columns = [
        column
        for column in forecast.columns
        if column
        not in [
            "revenue_year",
            "revenue_month",
            "date",
            "ensemble_revenue",
            "lower_bound",
            "upper_bound",
            "model_spread",
            *YIELD_FORECAST_COLUMNS,
        ]
    ]
    return {
        "annual_total": f"{annual_total:,}",
        "best_model": str(best_model),
        "best_mape": f"{best_mape:.2f}%" if pd.notna(best_mape) else "無",
        "peak_month": f"{peak_month} 月",
        "low_month": f"{low_month} 月",
        "model_count": str(len(model_columns)),
        "model_names": "、".join(model_columns),
    }


def make_yield_summary(forecast: pd.DataFrame) -> dict[str, str]:
    required_columns = {"estimated_cash_dividend", "stock_price", "dividend_yield_percent"}
    if not required_columns.issubset(forecast.columns):
        return {
            "estimated_cash_dividend": "無",
            "annual_eps": "無",
            "latest_price": "無",
            "latest_price_date": "無",
            "price_source": "無",
            "cash_dividend_source": "無",
            "average_yield": "無",
            "latest_yield": "無",
            "as_of_price": "無",
            "as_of_price_date": "無",
            "as_of_yield": "無",
            "payout_ratio": "無",
        }

    latest = forecast.sort_values("revenue_month").iloc[-1]
    average_yield = pd.to_numeric(forecast["dividend_yield_percent"], errors="coerce").mean()
    latest_price_date = pd.to_datetime(latest.get("stock_price_date"), errors="coerce")
    as_of_price_date = pd.to_datetime(latest.get("as_of_price_date"), errors="coerce")
    as_of_price = pd.to_numeric(latest.get("as_of_stock_price"), errors="coerce")
    as_of_yield = pd.to_numeric(latest.get("as_of_price_yield_percent"), errors="coerce")
    return {
        "estimated_cash_dividend": f"{float(latest['estimated_cash_dividend']):.2f}",
        "annual_eps": f"{float(latest['annual_eps']):.2f}" if pd.notna(latest["annual_eps"]) else "無",
        "latest_price": f"{float(latest['stock_price']):.2f}",
        "latest_price_date": latest_price_date.strftime("%Y-%m-%d") if pd.notna(latest_price_date) else "無",
        "price_source": str(latest.get("stock_price_source", "無")),
        "cash_dividend_source": str(latest.get("cash_dividend_source", "無")),
        "average_yield": f"{average_yield:.2f}%" if pd.notna(average_yield) else "無",
        "latest_yield": f"{float(latest['dividend_yield_percent']):.2f}%"
        if pd.notna(latest["dividend_yield_percent"])
        else "無",
        "as_of_price": f"{float(as_of_price):.2f}" if pd.notna(as_of_price) else "無",
        "as_of_price_date": (
            as_of_price_date.strftime("%Y-%m-%d") if pd.notna(as_of_price_date) else "無"
        ),
        "as_of_yield": f"{float(as_of_yield):.2f}%" if pd.notna(as_of_yield) else "無",
        "payout_ratio": f"{float(latest['payout_ratio']) * 100:.1f}%"
        if pd.notna(latest["payout_ratio"])
        else "無",
    }


def make_yield_comparison_summary(yield_comparison: pd.DataFrame) -> dict[str, str]:
    required_columns = {
        "predicted_dividend_yield_percent",
        "actual_dividend_yield_percent",
        "yield_abs_error_percent_point",
        "yield_error_percent_point",
    }
    if yield_comparison.empty or not required_columns.issubset(yield_comparison.columns):
        return {
            "average_predicted_yield": "無",
            "average_actual_yield": "無",
            "average_abs_error": "無",
            "latest_error": "無",
        }

    predicted = pd.to_numeric(yield_comparison["predicted_dividend_yield_percent"], errors="coerce")
    actual = pd.to_numeric(yield_comparison["actual_dividend_yield_percent"], errors="coerce")
    abs_error = pd.to_numeric(yield_comparison["yield_abs_error_percent_point"], errors="coerce")
    latest = yield_comparison.sort_values("revenue_month").iloc[-1]
    latest_error = pd.to_numeric(latest["yield_error_percent_point"], errors="coerce")

    return {
        "average_predicted_yield": f"{predicted.mean():.2f}%" if pd.notna(predicted.mean()) else "無",
        "average_actual_yield": f"{actual.mean():.2f}%" if pd.notna(actual.mean()) else "無",
        "average_abs_error": f"{abs_error.mean():.2f} 個百分點" if pd.notna(abs_error.mean()) else "無",
        "latest_error": f"{latest_error:.2f} 個百分點" if pd.notna(latest_error) else "無",
    }
