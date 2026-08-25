"""Benchmark dividend-layer methods for downstream yield estimates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_benchmark.adapters.ensemble_evidence import (
    get_dividend_policy,
    load_cash_dividend_data,
    load_eps_data,
    load_revenue_data,
)
from forecast_benchmark.benchmark_config import DEFAULT_TARGET_YEAR, PROJECT_ROOT
from forecast_benchmark.eps_benchmark import build_eps_predictions
from forecast_benchmark.experiment_registry import (
    add_registry_arguments,
    enrich_run_config_from_args,
    write_run_config_and_registry,
)
from forecast_benchmark.run_benchmark import parse_int_csv, parse_str_csv
from forecast_benchmark.yield_benchmark import (
    DEFAULT_INPUT_PREDICTIONS,
    DEFAULT_MIN_STOCK_PRICE,
    DEFAULT_MODEL_NAMES,
    ENTRY_YIELD_RATES,
    build_stock_price_lookup,
    is_observed_stock_price_source,
    load_prediction_input,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forecast_benchmark" / "outputs" / "dividend_layer_benchmark"
DEFAULT_EPS_METHODS = ("current_ratio", "elastic_net_annual")
DEFAULT_DIVIDEND_METHODS = (
    "current_system_payout_ratio",
    "time_safe_payout_ratio",
    "announcement_safe_payout_ratio",
    "last_cash_dividend",
    "recent_cash_dividend_median",
    "smoothed_cash_dividend",
    "announcement_safe_last_cash_dividend",
    "announcement_safe_cash_dividend_median",
    "announcement_safe_smoothed_cash_dividend",
    "eps_sign_guard_last_cash_dividend",
)
PAYOUT_METHODS = {
    "current_system_payout_ratio",
    "time_safe_payout_ratio",
    "announcement_safe_payout_ratio",
}


def default_as_of_date(target_year: int) -> pd.Timestamp:
    return pd.Timestamp(int(target_year), 1, 10)


def _round_numeric_columns(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").round(digits)
    return result


def _first_valid(values: pd.Series) -> object:
    valid = values.dropna()
    return valid.iloc[0] if not valid.empty else np.nan


def load_dividend_source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_revenue_data(), load_eps_data(), load_cash_dividend_data()


def build_annual_eps_lookup(eps: pd.DataFrame) -> pd.DataFrame:
    return (
        eps.groupby(["stock_id", "eps_year"], as_index=False)
        .agg(annual_eps=("latest_eps", "sum"), eps_quarter_count=("latest_eps", "count"))
        .sort_values(["stock_id", "eps_year"])
    )


def build_annual_cash_dividend(dividends: pd.DataFrame, target_year: int) -> pd.DataFrame:
    source = dividends.copy()
    if "CashExDividendTradingDate" not in source.columns:
        source["CashExDividendTradingDate"] = pd.NaT
    if "DividendAvailableDate" in source.columns:
        source["dividend_available_date"] = pd.to_datetime(source["DividendAvailableDate"], errors="coerce")
        source["dividend_available_source"] = "DividendAvailableDate"
    elif "AnnouncementDate" in source.columns:
        source["dividend_available_date"] = pd.to_datetime(source["AnnouncementDate"], errors="coerce")
        source["dividend_available_source"] = "AnnouncementDate"
    else:
        source["dividend_available_date"] = pd.to_datetime(source["CashExDividendTradingDate"], errors="coerce")
        source["dividend_available_source"] = "CashExDividendTradingDate fallback"
    source["dividend_available_date"] = source["dividend_available_date"].fillna(
        pd.to_datetime(source["CashExDividendTradingDate"], errors="coerce")
    )
    annual = (
        source
        .groupby(["stock_id", "fiscal_year"], as_index=False)
        .agg(
            cash_dividend_per_share=("TotalCashDividend", "sum"),
            ex_dividend_year_min=("ex_dividend_year", "min"),
            ex_dividend_year_max=("ex_dividend_year", "max"),
            dividend_available_date_min=("dividend_available_date", "min"),
            dividend_available_date_max=("dividend_available_date", "max"),
            dividend_available_source=("dividend_available_source", _first_valid),
            dividend_record_count=("TotalCashDividend", "count"),
        )
    )
    annual["fiscal_year"] = pd.to_numeric(annual["fiscal_year"], errors="coerce")
    annual["ex_dividend_year_min"] = pd.to_numeric(annual["ex_dividend_year_min"], errors="coerce")
    annual["ex_dividend_year_max"] = pd.to_numeric(annual["ex_dividend_year_max"], errors="coerce")
    annual["uses_target_year_ex_dividend"] = annual["ex_dividend_year_max"].eq(int(target_year))
    return annual.dropna(subset=["fiscal_year"]).sort_values(["stock_id", "fiscal_year"])


def build_actual_cash_dividend_lookup(dividends: pd.DataFrame, stock_ids: list[int], target_year: int) -> pd.DataFrame:
    actual = (
        dividends[
            dividends["stock_id"].isin(stock_ids)
            & pd.to_numeric(dividends["ex_dividend_year"], errors="coerce").eq(int(target_year))
        ]
        .groupby("stock_id", as_index=False)
        .agg(
            actual_cash_dividend_per_share=("TotalCashDividend", "sum"),
            actual_cash_dividend_record_count=("TotalCashDividend", "count"),
        )
    )
    stock_frame = pd.DataFrame({"stock_id": stock_ids})
    return stock_frame.merge(actual, on="stock_id", how="left")


def _policy_payout_ratio(stock_id: int) -> tuple[float, str]:
    policy = get_dividend_policy(int(stock_id))
    return float(policy["payout_ratio"]), "dividend_policy.csv fallback"


def select_cash_history(
    annual_cash: pd.DataFrame,
    stock_id: int,
    target_year: int,
    mode: str,
    as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    stock = annual_cash[annual_cash["stock_id"].eq(int(stock_id))].copy()
    if mode == "current_system":
        selected = stock[stock["fiscal_year"].lt(int(target_year))].copy()
    elif mode == "time_safe":
        selected = stock[stock["ex_dividend_year_max"].lt(int(target_year))].copy()
    elif mode == "announcement_safe":
        if as_of_date is None:
            raise ValueError("announcement_safe cash history requires as_of_date.")
        available = pd.to_datetime(stock["dividend_available_date_max"], errors="coerce")
        selected = stock[available.le(pd.Timestamp(as_of_date))].copy()
    else:
        raise ValueError(f"Unknown cash history mode: {mode}")
    return selected.sort_values("fiscal_year")


def estimate_payout_ratio(
    annual_cash: pd.DataFrame,
    annual_eps: pd.DataFrame,
    stock_id: int,
    target_year: int,
    mode: str,
    as_of_date: pd.Timestamp | None = None,
) -> dict[str, object]:
    history = select_cash_history(annual_cash, stock_id, target_year, mode=mode, as_of_date=as_of_date)
    merged = history.merge(
        annual_eps[annual_eps["stock_id"].eq(int(stock_id))],
        left_on=["stock_id", "fiscal_year"],
        right_on=["stock_id", "eps_year"],
        how="inner",
    )
    candidates = merged[
        (merged["annual_eps"] > 0)
        & (merged["cash_dividend_per_share"] >= 0)
        & (merged["eps_quarter_count"] >= 4)
    ].copy()
    if candidates.empty:
        candidates = merged[(merged["annual_eps"] > 0) & (merged["cash_dividend_per_share"] >= 0)].copy()
    if candidates.empty:
        payout_ratio, source = _policy_payout_ratio(stock_id)
        return {
            "payout_ratio": payout_ratio,
            "dividend_reference_year": np.nan,
            "dividend_history_count": int(len(history)),
            "dividend_method_source": source,
            "uses_target_year_ex_dividend": False,
            "uses_post_as_of_dividend": False,
        }

    candidates["payout_ratio"] = candidates["cash_dividend_per_share"] / candidates["annual_eps"]
    candidates = candidates.replace([np.inf, -np.inf], np.nan).dropna(subset=["payout_ratio"])
    if candidates.empty:
        payout_ratio, source = _policy_payout_ratio(stock_id)
        return {
            "payout_ratio": payout_ratio,
            "dividend_reference_year": np.nan,
            "dividend_history_count": int(len(history)),
            "dividend_method_source": source,
            "uses_target_year_ex_dividend": False,
            "uses_post_as_of_dividend": False,
        }

    recent = candidates.sort_values("fiscal_year").tail(3)
    payout_ratio = float(np.clip(recent["payout_ratio"].median(), 0, 1.5))
    reference_year = int(recent["fiscal_year"].max())
    uses_target = bool(recent["uses_target_year_ex_dividend"].any())
    if as_of_date is not None:
        available = pd.to_datetime(recent["dividend_available_date_max"], errors="coerce")
        uses_post_as_of = bool(available.gt(pd.Timestamp(as_of_date)).any())
    else:
        uses_post_as_of = False
    labels = {
        "current_system": "current system payout",
        "time_safe": "ex-date-safe payout",
        "announcement_safe": f"announcement-safe payout as of {pd.Timestamp(as_of_date).date()}",
    }
    label = labels[mode]
    return {
        "payout_ratio": payout_ratio,
        "dividend_reference_year": reference_year,
        "dividend_history_count": int(len(history)),
        "dividend_method_source": f"{label} through fiscal {reference_year}",
        "uses_target_year_ex_dividend": uses_target,
        "uses_post_as_of_dividend": uses_post_as_of,
    }


def estimate_recent_cash_dividend(
    annual_cash: pd.DataFrame,
    stock_id: int,
    target_year: int,
    method: str,
    mode: str = "time_safe",
    as_of_date: pd.Timestamp | None = None,
) -> dict[str, object]:
    history = select_cash_history(annual_cash, stock_id, target_year, mode=mode, as_of_date=as_of_date)
    recent = history.dropna(subset=["cash_dividend_per_share"]).sort_values("fiscal_year").tail(3)
    if recent.empty:
        return {
            "estimated_cash_dividend": np.nan,
            "payout_ratio": np.nan,
            "dividend_reference_year": np.nan,
            "dividend_history_count": 0,
            "dividend_method_source": "missing time-safe cash dividend history",
            "uses_target_year_ex_dividend": False,
            "uses_post_as_of_dividend": False,
        }

    values = pd.to_numeric(recent["cash_dividend_per_share"], errors="coerce").dropna()
    if values.empty:
        estimated = np.nan
    elif method == "last_cash_dividend":
        estimated = float(values.iloc[-1])
    elif method == "recent_cash_dividend_median":
        estimated = float(values.median())
    elif method == "smoothed_cash_dividend":
        latest_first = values.iloc[::-1].to_numpy(dtype=float)
        weights = np.array([0.5, 0.3, 0.2], dtype=float)[: len(latest_first)]
        weights = weights / weights.sum()
        estimated = float(np.sum(latest_first * weights))
    else:
        raise ValueError(f"Unsupported recent cash dividend method: {method}")

    if as_of_date is not None:
        available = pd.to_datetime(recent["dividend_available_date_max"], errors="coerce")
        uses_post_as_of = bool(available.gt(pd.Timestamp(as_of_date)).any())
    else:
        uses_post_as_of = False
    source_label = (
        f"{method} from announcement-safe cash dividend history as of {pd.Timestamp(as_of_date).date()}"
        if mode == "announcement_safe"
        else f"{method} from time-safe cash dividend history"
    )
    return {
        "estimated_cash_dividend": estimated,
        "payout_ratio": np.nan,
        "dividend_reference_year": int(recent["fiscal_year"].max()),
        "dividend_history_count": int(len(history)),
        "dividend_method_source": source_label,
        "uses_target_year_ex_dividend": bool(recent["uses_target_year_ex_dividend"].any()),
        "uses_post_as_of_dividend": uses_post_as_of,
    }


def estimate_cash_dividend(
    annual_cash: pd.DataFrame,
    annual_eps: pd.DataFrame,
    stock_id: int,
    target_year: int,
    estimated_eps: float,
    dividend_method: str,
    as_of_date: pd.Timestamp | None = None,
) -> dict[str, object]:
    if dividend_method == "current_system_payout_ratio":
        estimate = estimate_payout_ratio(
            annual_cash,
            annual_eps,
            stock_id,
            target_year,
            mode="current_system",
            as_of_date=as_of_date,
        )
        cash_dividend = float(estimated_eps) * float(estimate["payout_ratio"]) if pd.notna(estimated_eps) else np.nan
        estimate["estimated_cash_dividend"] = float(max(cash_dividend, 0)) if pd.notna(cash_dividend) else np.nan
        return estimate

    if dividend_method == "time_safe_payout_ratio":
        estimate = estimate_payout_ratio(
            annual_cash,
            annual_eps,
            stock_id,
            target_year,
            mode="time_safe",
            as_of_date=as_of_date,
        )
        cash_dividend = float(estimated_eps) * float(estimate["payout_ratio"]) if pd.notna(estimated_eps) else np.nan
        estimate["estimated_cash_dividend"] = float(max(cash_dividend, 0)) if pd.notna(cash_dividend) else np.nan
        return estimate

    if dividend_method == "announcement_safe_payout_ratio":
        estimate = estimate_payout_ratio(
            annual_cash,
            annual_eps,
            stock_id,
            target_year,
            mode="announcement_safe",
            as_of_date=as_of_date,
        )
        cash_dividend = float(estimated_eps) * float(estimate["payout_ratio"]) if pd.notna(estimated_eps) else np.nan
        estimate["estimated_cash_dividend"] = float(max(cash_dividend, 0)) if pd.notna(cash_dividend) else np.nan
        return estimate

    if dividend_method in {"last_cash_dividend", "recent_cash_dividend_median", "smoothed_cash_dividend"}:
        return estimate_recent_cash_dividend(
            annual_cash,
            stock_id,
            target_year,
            dividend_method,
            mode="time_safe",
            as_of_date=as_of_date,
        )

    announcement_safe_recent_methods = {
        "announcement_safe_last_cash_dividend": "last_cash_dividend",
        "announcement_safe_cash_dividend_median": "recent_cash_dividend_median",
        "announcement_safe_smoothed_cash_dividend": "smoothed_cash_dividend",
    }
    if dividend_method in announcement_safe_recent_methods:
        return estimate_recent_cash_dividend(
            annual_cash,
            stock_id,
            target_year,
            announcement_safe_recent_methods[dividend_method],
            mode="announcement_safe",
            as_of_date=as_of_date,
        )

    if dividend_method == "eps_sign_guard_last_cash_dividend":
        estimate = estimate_recent_cash_dividend(
            annual_cash,
            stock_id,
            target_year,
            "last_cash_dividend",
            mode="time_safe",
            as_of_date=as_of_date,
        )
        if pd.notna(estimated_eps) and float(estimated_eps) <= 0:
            estimate["estimated_cash_dividend"] = 0.0
            estimate["dividend_method_source"] = "zero if estimated EPS <= 0; otherwise last cash dividend"
        return estimate

    raise ValueError(f"Unsupported dividend method: {dividend_method}")


def build_dividend_estimates(
    eps_predictions: pd.DataFrame,
    annual_cash: pd.DataFrame,
    annual_eps: pd.DataFrame,
    target_year: int,
    dividend_methods: list[str],
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    base_columns = [
        "source_family",
        "model",
        "eps_method",
        "stock_id",
        "stock_name",
        "industry_category",
        "target_year",
        "predicted_annual_revenue",
        "actual_annual_revenue",
        "annual_revenue_abs_percent_error",
        "estimated_eps",
        "actual_annual_eps",
        "eps_abs_error",
        "eps_abs_percent_error",
    ]
    for _, eps_row in eps_predictions[base_columns].drop_duplicates().iterrows():
        for dividend_method in dividend_methods:
            estimate = estimate_cash_dividend(
                annual_cash,
                annual_eps,
                int(eps_row["stock_id"]),
                int(target_year),
                float(eps_row["estimated_eps"]) if pd.notna(eps_row["estimated_eps"]) else np.nan,
                dividend_method,
                as_of_date=as_of_date,
            )
            row = eps_row.to_dict()
            row.update(
                {
                    "as_of_date": pd.Timestamp(as_of_date).date().isoformat(),
                    "dividend_method": dividend_method,
                    "estimated_cash_dividend": estimate.get("estimated_cash_dividend", np.nan),
                    "payout_ratio": estimate.get("payout_ratio", np.nan),
                    "dividend_reference_year": estimate.get("dividend_reference_year", np.nan),
                    "dividend_history_count": estimate.get("dividend_history_count", np.nan),
                    "dividend_method_source": estimate.get("dividend_method_source", ""),
                    "uses_target_year_ex_dividend": bool(estimate.get("uses_target_year_ex_dividend", False)),
                    "uses_post_as_of_dividend": bool(estimate.get("uses_post_as_of_dividend", False)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_dividend_layer_predictions(
    predictions: pd.DataFrame,
    target_year: int,
    eps_methods: list[str] | None = None,
    dividend_methods: list[str] | None = None,
    as_of_date: pd.Timestamp | None = None,
    min_stock_price: float = DEFAULT_MIN_STOCK_PRICE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    revenue_data, eps, dividends = load_dividend_source_data()
    eps_methods = list(eps_methods or DEFAULT_EPS_METHODS)
    dividend_methods = list(dividend_methods or DEFAULT_DIVIDEND_METHODS)
    as_of_date = pd.Timestamp(as_of_date) if as_of_date is not None else default_as_of_date(target_year)
    stock_ids = sorted(int(stock_id) for stock_id in predictions["stock_id"].dropna().unique())
    prices = build_stock_price_lookup(revenue_data, stock_ids, target_year)
    annual_eps = build_annual_eps_lookup(eps)
    annual_cash = build_annual_cash_dividend(dividends, target_year)
    actual_cash = build_actual_cash_dividend_lookup(dividends, stock_ids, target_year)
    eps_predictions, eps_failures = build_eps_predictions(
        predictions,
        target_year=target_year,
        eps_methods=eps_methods,
        include_oracle=False,
    )
    eps_predictions = eps_predictions[~eps_predictions["is_oracle"].astype(bool)].copy()
    dividend_estimates = build_dividend_estimates(
        eps_predictions,
        annual_cash,
        annual_eps,
        target_year,
        dividend_methods,
        as_of_date,
    )

    monthly = predictions.merge(dividend_estimates, on=["source_family", "model", "stock_id"], how="inner")
    monthly = monthly[monthly["eps_method"].notna()].copy()
    monthly = monthly.merge(
        prices[["stock_id", "price_month", "price_date", "close_price", "price_source"]],
        left_on=["stock_id", "target_month"],
        right_on=["stock_id", "price_month"],
        how="left",
    )
    monthly = monthly.merge(actual_cash, on="stock_id", how="left")
    monthly["cash_dividend_error"] = (
        monthly["estimated_cash_dividend"] - monthly["actual_cash_dividend_per_share"]
    )
    monthly["cash_dividend_abs_error"] = monthly["cash_dividend_error"].abs()
    monthly["stock_price_date"] = monthly["price_date"]
    monthly["stock_price"] = monthly["close_price"]
    monthly["stock_price_source"] = monthly["price_source"]
    monthly["stock_price_is_observed"] = is_observed_stock_price_source(
        monthly["stock_price_source"]
    )
    monthly["stock_price_valid_for_yield"] = (
        monthly["stock_price_is_observed"]
        & (monthly["stock_price"] > float(min_stock_price))
    )
    monthly["predicted_dividend_yield_percent"] = np.where(
        monthly["stock_price_valid_for_yield"],
        monthly["estimated_cash_dividend"] / monthly["stock_price"] * 100,
        np.nan,
    )
    monthly["actual_dividend_yield_percent"] = np.where(
        monthly["stock_price_valid_for_yield"],
        monthly["actual_cash_dividend_per_share"] / monthly["stock_price"] * 100,
        np.nan,
    )
    monthly["yield_error_percent_point"] = (
        monthly["predicted_dividend_yield_percent"] - monthly["actual_dividend_yield_percent"]
    )
    monthly["yield_abs_error_percent_point"] = monthly["yield_error_percent_point"].abs()
    monthly["annual_revenue_error"] = (
        monthly["predicted_annual_revenue"] - monthly["actual_annual_revenue"]
    )
    monthly["annual_revenue_abs_error"] = monthly["annual_revenue_error"].abs()
    for rate in ENTRY_YIELD_RATES:
        label = int(rate * 100)
        monthly[f"entry_price_at_{label}_percent"] = monthly["estimated_cash_dividend"] / rate
        monthly[f"actual_entry_price_at_{label}_percent"] = (
            monthly["actual_cash_dividend_per_share"] / rate
        )
    monthly = monthly.drop(columns=["price_month", "price_date", "close_price", "price_source"])
    monthly = monthly.drop(columns=[column for column in monthly.columns if column.endswith("_y")])
    monthly = monthly.rename(columns={column: column[:-2] for column in monthly.columns if column.endswith("_x")})
    failures = eps_failures.copy()
    return monthly.sort_values(
        ["stock_id", "source_family", "model", "eps_method", "dividend_method", "target_month"]
    ), failures


def build_dividend_layer_stock_accuracy(dividend_predictions: pd.DataFrame) -> pd.DataFrame:
    if dividend_predictions.empty:
        return pd.DataFrame()

    rows = []
    group_columns = [
        "source_family",
        "model",
        "eps_method",
        "dividend_method",
        "stock_id",
        "stock_name",
        "industry_category",
    ]
    for group_key, group in dividend_predictions.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_key, strict=True))
        valid_yield = group.dropna(
            subset=["predicted_dividend_yield_percent", "actual_dividend_yield_percent"]
        )
        yield_error = pd.to_numeric(valid_yield["yield_error_percent_point"], errors="coerce")
        yield_abs_error = yield_error.abs()
        first = group.iloc[0]
        row.update(
            {
                "monthly_observations": int(len(valid_yield)),
                "predicted_annual_revenue": first.get("predicted_annual_revenue", np.nan),
                "actual_annual_revenue": first.get("actual_annual_revenue", np.nan),
                "annual_revenue_abs_percent_error": first.get("annual_revenue_abs_percent_error", np.nan),
                "estimated_eps": first.get("estimated_eps", np.nan),
                "actual_annual_eps": first.get("actual_annual_eps", np.nan),
                "eps_abs_error": first.get("eps_abs_error", np.nan),
                "estimated_cash_dividend": first.get("estimated_cash_dividend", np.nan),
                "actual_cash_dividend_per_share": first.get("actual_cash_dividend_per_share", np.nan),
                "cash_dividend_abs_error": first.get("cash_dividend_abs_error", np.nan),
                "payout_ratio": first.get("payout_ratio", np.nan),
                "dividend_reference_year": first.get("dividend_reference_year", np.nan),
                "dividend_history_count": first.get("dividend_history_count", np.nan),
                "uses_target_year_ex_dividend": bool(first.get("uses_target_year_ex_dividend", False)),
                "uses_post_as_of_dividend": bool(first.get("uses_post_as_of_dividend", False)),
                "yield_mae_percent_point": float(yield_abs_error.mean())
                if yield_abs_error.notna().any()
                else np.nan,
                "yield_median_ae_percent_point": float(yield_abs_error.median())
                if yield_abs_error.notna().any()
                else np.nan,
                "yield_rmse_percent_point": float(np.sqrt(np.mean(yield_error.dropna() ** 2)))
                if yield_error.notna().any()
                else np.nan,
                "mean_predicted_yield_percent": float(valid_yield["predicted_dividend_yield_percent"].mean())
                if not valid_yield.empty
                else np.nan,
                "mean_actual_yield_percent": float(valid_yield["actual_dividend_yield_percent"].mean())
                if not valid_yield.empty
                else np.nan,
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    numeric_columns = [
        "predicted_annual_revenue",
        "actual_annual_revenue",
        "annual_revenue_abs_percent_error",
        "estimated_eps",
        "actual_annual_eps",
        "eps_abs_error",
        "estimated_cash_dividend",
        "actual_cash_dividend_per_share",
        "cash_dividend_abs_error",
        "payout_ratio",
        "dividend_reference_year",
        "dividend_history_count",
        "yield_mae_percent_point",
        "yield_median_ae_percent_point",
        "yield_rmse_percent_point",
        "mean_predicted_yield_percent",
        "mean_actual_yield_percent",
    ]
    result = _round_numeric_columns(result, numeric_columns)
    return result.sort_values(
        ["stock_id", "yield_mae_percent_point", "source_family", "model", "eps_method", "dividend_method"],
        na_position="last",
    )


def build_dividend_layer_overall_accuracy(stock_accuracy: pd.DataFrame) -> pd.DataFrame:
    if stock_accuracy.empty:
        return pd.DataFrame()

    metric_columns = [
        "annual_revenue_abs_percent_error",
        "eps_abs_error",
        "cash_dividend_abs_error",
        "yield_mae_percent_point",
        "yield_median_ae_percent_point",
        "yield_rmse_percent_point",
    ]
    rows = []
    group_columns = ["source_family", "model", "eps_method", "dividend_method"]
    for group_key, group in stock_accuracy.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_key, strict=True))
        row.update(
            {
                "stock_count": int(group["stock_id"].nunique()),
                "valid_revenue_stock_count": int(group["annual_revenue_abs_percent_error"].notna().sum()),
                "valid_eps_stock_count": int(group["eps_abs_error"].notna().sum()),
                "valid_cash_dividend_stock_count": int(group["cash_dividend_abs_error"].notna().sum()),
                "valid_yield_stock_count": int(group["yield_mae_percent_point"].notna().sum()),
                "uses_target_year_ex_dividend_stock_count": int(group["uses_target_year_ex_dividend"].sum()),
                "uses_post_as_of_dividend_stock_count": int(group["uses_post_as_of_dividend"].sum()),
                "monthly_observations": int(group["monthly_observations"].sum()),
            }
        )
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"average_{column}"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"median_{column}"] = float(values.median()) if values.notna().any() else np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    numeric_columns = [
        column for column in result.columns if column.startswith("average_") or column.startswith("median_")
    ]
    result = _round_numeric_columns(result, numeric_columns)
    return result.sort_values(
        [
            "average_yield_mae_percent_point",
            "average_cash_dividend_abs_error",
            "source_family",
            "model",
            "eps_method",
            "dividend_method",
        ],
        na_position="last",
    )


def build_dividend_layer_winner_summary(
    stock_accuracy: pd.DataFrame,
    primary_metric: str = "yield_mae_percent_point",
) -> pd.DataFrame:
    valid = stock_accuracy.dropna(subset=[primary_metric]).copy()
    if valid.empty:
        return pd.DataFrame()

    group_columns = ["source_family", "model", "eps_method", "dividend_method"]
    compared_stocks = int(valid["stock_id"].nunique())
    winners = valid.loc[valid.groupby("stock_id")[primary_metric].idxmin()]
    winner_counts = winners.groupby(group_columns, as_index=False).size().rename(columns={"size": "stock_wins"})
    metric_summary = (
        valid.groupby(group_columns, as_index=False)[primary_metric]
        .agg(average_primary_metric="mean", median_primary_metric="median")
    )
    summary = metric_summary.merge(winner_counts, on=group_columns, how="left")
    summary["stock_wins"] = summary["stock_wins"].fillna(0).astype(int)
    summary["compared_stocks"] = compared_stocks
    summary["stock_win_rate"] = np.where(
        compared_stocks > 0,
        summary["stock_wins"] / compared_stocks * 100,
        np.nan,
    )
    return _round_numeric_columns(
        summary,
        ["average_primary_metric", "median_primary_metric", "stock_win_rate"],
    ).sort_values(
        ["stock_wins", "average_primary_metric", "source_family", "model", "eps_method", "dividend_method"],
        ascending=[False, True, True, True, True, True],
    )


def build_improvement_vs_baseline(
    overall_accuracy: pd.DataFrame,
    baseline_eps_method: str = "current_ratio",
    baseline_dividend_method: str = "current_system_payout_ratio",
) -> pd.DataFrame:
    if overall_accuracy.empty:
        return pd.DataFrame()

    baseline = overall_accuracy[
        overall_accuracy["eps_method"].eq(baseline_eps_method)
        & overall_accuracy["dividend_method"].eq(baseline_dividend_method)
    ][
        [
            "source_family",
            "model",
            "average_eps_abs_error",
            "average_cash_dividend_abs_error",
            "average_yield_mae_percent_point",
        ]
    ].rename(
        columns={
            "average_eps_abs_error": "baseline_average_eps_abs_error",
            "average_cash_dividend_abs_error": "baseline_average_cash_dividend_abs_error",
            "average_yield_mae_percent_point": "baseline_average_yield_mae_percent_point",
        }
    )
    compared = overall_accuracy.merge(baseline, on=["source_family", "model"], how="left")
    compared["average_cash_dividend_abs_error_delta_vs_baseline"] = (
        compared["baseline_average_cash_dividend_abs_error"] - compared["average_cash_dividend_abs_error"]
    )
    compared["average_yield_mae_delta_vs_baseline"] = (
        compared["baseline_average_yield_mae_percent_point"] - compared["average_yield_mae_percent_point"]
    )
    compared["average_yield_mae_improvement_pct_vs_baseline"] = np.where(
        compared["baseline_average_yield_mae_percent_point"] > 0,
        compared["average_yield_mae_delta_vs_baseline"]
        / compared["baseline_average_yield_mae_percent_point"]
        * 100,
        np.nan,
    )
    numeric_columns = [
        "baseline_average_eps_abs_error",
        "baseline_average_cash_dividend_abs_error",
        "baseline_average_yield_mae_percent_point",
        "average_cash_dividend_abs_error_delta_vs_baseline",
        "average_yield_mae_delta_vs_baseline",
        "average_yield_mae_improvement_pct_vs_baseline",
    ]
    compared = _round_numeric_columns(compared, numeric_columns)
    return compared.sort_values(
        ["average_yield_mae_improvement_pct_vs_baseline", "average_yield_mae_percent_point"],
        ascending=[False, True],
        na_position="last",
    )


def build_leakage_diagnostic(stock_accuracy: pd.DataFrame) -> pd.DataFrame:
    diagnostic_methods = {"current_system_payout_ratio", "time_safe_payout_ratio", "announcement_safe_payout_ratio"}
    current = stock_accuracy[stock_accuracy["dividend_method"].isin(diagnostic_methods)].copy()
    if current.empty:
        return pd.DataFrame()
    rows = []
    group_columns = ["source_family", "model", "eps_method", "dividend_method"]
    for group_key, group in current.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_key, strict=True))
        row.update(
            {
                "stock_count": int(group["stock_id"].nunique()),
                "uses_target_year_ex_dividend_stock_count": int(group["uses_target_year_ex_dividend"].sum()),
                "uses_target_year_ex_dividend_rate": float(group["uses_target_year_ex_dividend"].mean() * 100),
                "uses_post_as_of_dividend_stock_count": int(group["uses_post_as_of_dividend"].sum()),
                "uses_post_as_of_dividend_rate": float(group["uses_post_as_of_dividend"].mean() * 100),
                "average_yield_mae_percent_point": float(group["yield_mae_percent_point"].mean()),
                "average_cash_dividend_abs_error": float(group["cash_dividend_abs_error"].mean()),
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    return _round_numeric_columns(
        result,
        [
            "uses_target_year_ex_dividend_rate",
            "uses_post_as_of_dividend_rate",
            "average_yield_mae_percent_point",
            "average_cash_dividend_abs_error",
        ],
    )


def write_outputs(
    output_dir: Path,
    dividend_predictions: pd.DataFrame,
    stock_accuracy: pd.DataFrame,
    overall_accuracy: pd.DataFrame,
    winner_summary: pd.DataFrame,
    improvement_vs_baseline: pd.DataFrame,
    leakage_diagnostic: pd.DataFrame,
    failures: pd.DataFrame,
    run_config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dividend_predictions.to_csv(output_dir / "dividend_layer_predictions.csv", index=False, encoding="utf-8-sig")
    stock_accuracy.to_csv(output_dir / "dividend_layer_stock_accuracy.csv", index=False, encoding="utf-8-sig")
    overall_accuracy.to_csv(output_dir / "dividend_layer_overall_accuracy.csv", index=False, encoding="utf-8-sig")
    winner_summary.to_csv(output_dir / "dividend_layer_winner_summary.csv", index=False, encoding="utf-8-sig")
    improvement_vs_baseline.to_csv(
        output_dir / "dividend_layer_improvement_vs_baseline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    leakage_diagnostic.to_csv(
        output_dir / "dividend_layer_leakage_diagnostic.csv",
        index=False,
        encoding="utf-8-sig",
    )
    failures.to_csv(output_dir / "dividend_layer_failed_runs.csv", index=False, encoding="utf-8-sig")
    write_run_config_and_registry(output_dir, run_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-predictions", type=Path, default=DEFAULT_INPUT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODEL_NAMES),
        help="Comma-separated revenue model names. Use --all-models to ignore this.",
    )
    parser.add_argument(
        "--eps-methods",
        default=",".join(DEFAULT_EPS_METHODS),
        help="Comma-separated EPS methods.",
    )
    parser.add_argument(
        "--dividend-methods",
        default=",".join(DEFAULT_DIVIDEND_METHODS),
        help="Comma-separated dividend methods.",
    )
    parser.add_argument(
        "--as-of-date",
        help="Information cutoff date for announcement-safe dividend methods. Defaults to Jan 10 of target year.",
    )
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--stock-ids", help="Comma-separated stock IDs.")
    parser.add_argument("--stock-limit", type=int, help="Limit stock pool for smoke runs.")
    parser.add_argument(
        "--min-stock-price",
        type=float,
        default=DEFAULT_MIN_STOCK_PRICE,
        help="Exclude monthly prices at or below this value from yield-error metrics.",
    )
    return add_registry_arguments(parser)


def run_dividend_layer_benchmark(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    model_names = None if args.all_models else parse_str_csv(args.models)
    eps_methods = parse_str_csv(args.eps_methods) or list(DEFAULT_EPS_METHODS)
    dividend_methods = parse_str_csv(args.dividend_methods) or list(DEFAULT_DIVIDEND_METHODS)
    as_of_date = pd.Timestamp(args.as_of_date) if args.as_of_date else default_as_of_date(args.target_year)
    stock_ids = parse_int_csv(args.stock_ids)
    predictions = load_prediction_input(
        args.input_predictions,
        target_year=args.target_year,
        model_names=model_names,
        stock_ids=stock_ids,
        stock_limit=args.stock_limit,
    )
    dividend_predictions, failures = build_dividend_layer_predictions(
        predictions,
        target_year=args.target_year,
        eps_methods=eps_methods,
        dividend_methods=dividend_methods,
        as_of_date=as_of_date,
        min_stock_price=args.min_stock_price,
    )
    stock_accuracy = build_dividend_layer_stock_accuracy(dividend_predictions)
    overall_accuracy = build_dividend_layer_overall_accuracy(stock_accuracy)
    winner_summary = build_dividend_layer_winner_summary(stock_accuracy)
    improvement_vs_baseline = build_improvement_vs_baseline(overall_accuracy)
    leakage_diagnostic = build_leakage_diagnostic(stock_accuracy)
    run_config = {
        "input_predictions": str(args.input_predictions),
        "output_dir": str(args.output_dir),
        "target_year": int(args.target_year),
        "models": model_names,
        "eps_methods": eps_methods,
        "dividend_methods": dividend_methods,
        "as_of_date": pd.Timestamp(as_of_date).date().isoformat(),
        "all_models": bool(args.all_models),
        "stock_ids": sorted(int(stock_id) for stock_id in predictions["stock_id"].unique()),
        "stock_count": int(predictions["stock_id"].nunique()),
        "prediction_rows": int(len(predictions)),
        "dividend_prediction_rows": int(len(dividend_predictions)),
        "min_stock_price": float(args.min_stock_price),
        "failed_runs": int(len(failures)),
    }
    run_config = enrich_run_config_from_args(
        run_config,
        args,
        experiment_family="dividend_layer_benchmark",
        extra={"input_predictions": str(args.input_predictions)},
    )
    write_outputs(
        args.output_dir,
        dividend_predictions,
        stock_accuracy,
        overall_accuracy,
        winner_summary,
        improvement_vs_baseline,
        leakage_diagnostic,
        failures,
        run_config,
    )
    return {
        "dividend_layer_predictions": dividend_predictions,
        "dividend_layer_stock_accuracy": stock_accuracy,
        "dividend_layer_overall_accuracy": overall_accuracy,
        "dividend_layer_winner_summary": winner_summary,
        "dividend_layer_improvement_vs_baseline": improvement_vs_baseline,
        "dividend_layer_leakage_diagnostic": leakage_diagnostic,
        "dividend_layer_failed_runs": failures,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = run_dividend_layer_benchmark(args)
    print("Wrote dividend layer benchmark outputs to", args.output_dir)
    print(results["dividend_layer_overall_accuracy"].to_string(index=False))
    print("\nDividend availability diagnostic:")
    print(results["dividend_layer_leakage_diagnostic"].to_string(index=False))


if __name__ == "__main__":
    main()
