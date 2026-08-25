"""Time-safe direct cash-dividend model benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_benchmark.adapters.ensemble_evidence import (
    load_cash_dividend_data,
    load_eps_data,
    load_revenue_data,
)
from forecast_benchmark.benchmark_config import DEFAULT_TARGET_YEAR, PROJECT_ROOT
from forecast_benchmark.dividend_layer_benchmark import (
    DEFAULT_EPS_METHODS as DEFAULT_BASELINE_EPS_METHODS,
    build_annual_cash_dividend,
    build_dividend_layer_overall_accuracy,
    build_dividend_layer_predictions,
    build_dividend_layer_stock_accuracy,
    build_dividend_layer_winner_summary,
    default_as_of_date,
)
from forecast_benchmark.eps_benchmark import build_annual_revenue_predictions
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


DEFAULT_VALIDATION_YEAR = 2024
DEFAULT_VALIDATION_YEARS = (2022, 2023, 2024)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forecast_benchmark" / "outputs" / "direct_dividend_model_benchmark"
DIRECT_EPS_METHOD_LABEL = "time_safe_features"
DEFAULT_DIRECT_THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60)
DEFAULT_MIN_BUCKET_FOLDS = 2
DEFAULT_MIN_BUCKET_STOCK_YEARS = 15
DIRECT_HEURISTIC_METHODS = (
    "direct_hurdle_last_known",
    "direct_hurdle_recent_median",
    "direct_hurdle_smoothed",
)
DIRECT_ML_METHOD_FAMILIES = ("ridge", "elastic_net")
DEFAULT_DIRECT_METHODS = (
    *DIRECT_HEURISTIC_METHODS,
    "direct_hurdle_ridge_t025",
    "direct_hurdle_ridge_t030",
    "direct_hurdle_ridge_t035",
    "direct_hurdle_ridge_t040",
    "direct_hurdle_ridge_t045",
    "direct_hurdle_ridge_t050",
    "direct_hurdle_ridge_t060",
    "direct_hurdle_elastic_net_t025",
    "direct_hurdle_elastic_net_t030",
    "direct_hurdle_elastic_net_t035",
    "direct_hurdle_elastic_net_t040",
    "direct_hurdle_elastic_net_t045",
    "direct_hurdle_elastic_net_t050",
    "direct_hurdle_elastic_net_t060",
)
SELECTED_BUCKET_DIVIDEND_METHOD = "bucket_validation_best"
GLOBAL_SELECTION_STRATEGY = "global_multi_year_validation_best"
BUCKET_SELECTION_STRATEGY = "bucket_multi_year_validation_best"
BUCKET_SUPPORT_STATUS_SUPPORTED = "supported"
BUCKET_SUPPORT_STATUS_FALLBACK = "fallback_to_global"
DEFAULT_BASELINE_DIVIDEND_METHODS = (
    "announcement_safe_payout_ratio",
    "announcement_safe_last_cash_dividend",
)
DIRECT_FEATURE_COLUMNS = [
    "last_known_cash_dividend",
    "recent_cash_dividend_median",
    "recent_cash_dividend_mean",
    "recent_cash_dividend_smoothed",
    "recent_paid_rate",
    "dividend_history_count",
    "years_since_last_dividend_reference",
    "latest_available_eps",
    "available_eps_ttm",
    "available_eps_ttm_yoy",
    "eps_history_count",
    "known_revenue_ltm",
    "known_revenue_recent_3m",
    "known_revenue_yoy_mean",
    "known_revenue_mom_mean",
    "revenue_history_count",
]
DIRECT_BUCKET_COLUMNS = [
    "paid_rate_bucket",
    "dividend_history_bucket",
    "latest_dividend_bucket",
    "dividend_selection_bucket",
]


def _round_numeric_columns(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").round(digits)
    return result


def _first_valid(values: pd.Series) -> object:
    valid = values.dropna()
    return valid.iloc[0] if not valid.empty else np.nan


def _safe_divide(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0:
        return np.nan
    return float(numerator) / float(denominator)


def parse_float_csv(value: str | None) -> list[float] | None:
    if value is None or not str(value).strip():
        return None
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def threshold_suffix(threshold: float) -> str:
    return f"{int(round(float(threshold) * 100)):03d}"


def build_threshold_direct_methods(thresholds: list[float] | tuple[float, ...]) -> list[str]:
    methods = list(DIRECT_HEURISTIC_METHODS)
    for family in DIRECT_ML_METHOD_FAMILIES:
        for threshold in thresholds:
            methods.append(f"direct_hurdle_{family}_t{threshold_suffix(float(threshold))}")
    return methods


def direct_method_threshold(method: str) -> float:
    if "_t" not in method:
        return 0.5
    suffix = method.rsplit("_t", 1)[-1]
    try:
        return float(int(suffix)) / 100.0
    except ValueError:
        return 0.5


def is_ml_hurdle_method(method: str) -> bool:
    return method.startswith("direct_hurdle_ridge") or method.startswith("direct_hurdle_elastic_net")


def validation_as_of_date_for_year(validation_year: int, explicit_date: str | None = None) -> pd.Timestamp:
    if not explicit_date:
        return default_as_of_date(validation_year)
    date = pd.Timestamp(explicit_date)
    return pd.Timestamp(int(validation_year), int(date.month), int(date.day))


def statement_available_date(statement_date: object) -> pd.Timestamp:
    date = pd.to_datetime(statement_date, errors="coerce")
    if pd.isna(date):
        return pd.NaT
    year = int(date.year)
    quarter = int(date.quarter)
    if quarter == 1:
        return pd.Timestamp(year, 5, 15)
    if quarter == 2:
        return pd.Timestamp(year, 8, 14)
    if quarter == 3:
        return pd.Timestamp(year, 11, 14)
    return pd.Timestamp(year + 1, 3, 31)


def revenue_available_date(revenue_year: object, revenue_month: object) -> pd.Timestamp:
    year = pd.to_numeric(pd.Series([revenue_year]), errors="coerce").iloc[0]
    month = pd.to_numeric(pd.Series([revenue_month]), errors="coerce").iloc[0]
    if pd.isna(year) or pd.isna(month):
        return pd.NaT
    year = int(year)
    month = int(month)
    if month == 12:
        return pd.Timestamp(year + 1, 1, 10)
    return pd.Timestamp(year, month + 1, 10)


def load_direct_source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    revenue = prepare_time_safe_revenue_data(load_revenue_data())
    dividends = load_cash_dividend_data()
    eps = load_eps_data()
    return revenue, prepare_time_safe_eps_data(eps), dividends


def prepare_time_safe_eps_data(eps: pd.DataFrame) -> pd.DataFrame:
    frame = eps.copy()
    if "EPS" not in frame.columns and "latest_eps" in frame.columns:
        frame["EPS"] = frame["latest_eps"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["stock_id"] = pd.to_numeric(frame["stock_id"], errors="coerce")
    frame["EPS"] = pd.to_numeric(frame["EPS"], errors="coerce")
    if "statement_available_date" in frame.columns:
        frame["statement_available_date"] = pd.to_datetime(frame["statement_available_date"], errors="coerce")
    else:
        frame["statement_available_date"] = frame["date"].map(statement_available_date)
    frame = frame.dropna(subset=["stock_id", "date", "EPS"])
    frame["stock_id"] = frame["stock_id"].astype(int)
    return frame[["stock_id", "date", "EPS", "statement_available_date"]].sort_values(["stock_id", "date"])


def prepare_time_safe_revenue_data(revenue: pd.DataFrame) -> pd.DataFrame:
    frame = revenue.copy()
    if "revenue_available_date" in frame.columns:
        frame["revenue_available_date"] = pd.to_datetime(frame["revenue_available_date"], errors="coerce")
    else:
        frame["revenue_available_date"] = [
            revenue_available_date(year, month)
            for year, month in zip(frame["revenue_year"], frame["revenue_month"], strict=False)
        ]
    for column in ["stock_id", "revenue_year", "revenue_month", "revenue_thousand", "mom", "yoy"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["stock_id", "revenue_year", "revenue_month", "revenue_thousand"])
    frame["stock_id"] = frame["stock_id"].astype(int)
    frame["revenue_year"] = frame["revenue_year"].astype(int)
    frame["revenue_month"] = frame["revenue_month"].astype(int)
    return frame.sort_values(["stock_id", "revenue_year", "revenue_month"])


def build_stock_metadata(predictions: pd.DataFrame, stock_ids: list[int]) -> pd.DataFrame:
    frame = predictions.copy()
    for column in ["stock_name", "industry_category"]:
        if column not in frame.columns:
            frame[column] = np.nan
    metadata = (
        frame[frame["stock_id"].isin(stock_ids)]
        .groupby("stock_id", as_index=False)
        .agg(stock_name=("stock_name", _first_valid), industry_category=("industry_category", _first_valid))
    )
    return pd.DataFrame({"stock_id": stock_ids}).merge(metadata, on="stock_id", how="left")


def build_actual_revenue_prediction_rows(
    revenue_data: pd.DataFrame,
    stock_ids: list[int],
    target_year: int,
    stock_metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows = revenue_data[
        revenue_data["stock_id"].isin(stock_ids) & revenue_data["revenue_year"].eq(int(target_year))
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    rows = rows.merge(
        stock_metadata[["stock_id", "stock_name", "industry_category"]],
        on="stock_id",
        how="left",
        suffixes=("", "_metadata"),
    )
    for column in ["stock_name", "industry_category"]:
        metadata_column = f"{column}_metadata"
        if metadata_column in rows.columns:
            rows[column] = rows[column].where(rows[column].notna(), rows[metadata_column])
    rows["source_family"] = "validation_actual_revenue"
    rows["model"] = f"actual_{int(target_year)}_revenue"
    rows["target_year"] = int(target_year)
    rows["target_month"] = rows["revenue_month"]
    rows["predicted_revenue"] = rows["revenue_thousand"]
    rows["actual_revenue"] = rows["revenue_thousand"]
    rows["last_observed_revenue"] = np.nan
    rows["source_path"] = "time-safe actual revenue validation frame"
    return rows[
        [
            "source_family",
            "model",
            "stock_id",
            "stock_name",
            "industry_category",
            "target_year",
            "target_month",
            "predicted_revenue",
            "actual_revenue",
            "last_observed_revenue",
            "source_path",
        ]
    ].sort_values(["stock_id", "target_month"])


def build_annual_context(monthly_predictions: pd.DataFrame) -> pd.DataFrame:
    annual = build_annual_revenue_predictions(monthly_predictions)
    annual["annual_revenue_abs_percent_error"] = np.where(
        pd.to_numeric(annual["actual_annual_revenue"], errors="coerce").abs() > 0,
        (
            pd.to_numeric(annual["predicted_annual_revenue"], errors="coerce")
            - pd.to_numeric(annual["actual_annual_revenue"], errors="coerce")
        ).abs()
        / pd.to_numeric(annual["actual_annual_revenue"], errors="coerce").abs()
        * 100,
        np.nan,
    )
    return annual


def build_historical_context(
    stock_ids: list[int],
    stock_metadata: pd.DataFrame,
    prediction_years: list[int],
) -> pd.DataFrame:
    metadata = stock_metadata.set_index("stock_id", drop=False)
    rows = []
    for year in prediction_years:
        for stock_id in stock_ids:
            stock_row = metadata.loc[int(stock_id)] if int(stock_id) in metadata.index else {}
            rows.append(
                {
                    "source_family": "historical_time_safe",
                    "model": "time_safe_feature_history",
                    "stock_id": int(stock_id),
                    "stock_name": stock_row.get("stock_name", np.nan)
                    if hasattr(stock_row, "get")
                    else np.nan,
                    "industry_category": stock_row.get("industry_category", np.nan)
                    if hasattr(stock_row, "get")
                    else np.nan,
                    "target_year": int(year),
                    "predicted_annual_revenue": np.nan,
                    "actual_annual_revenue": np.nan,
                    "annual_revenue_abs_percent_error": np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_actual_cash_dividend_targets(
    dividends: pd.DataFrame,
    stock_ids: list[int],
    target_years: list[int] | tuple[int, ...] | int,
) -> pd.DataFrame:
    years = [int(target_years)] if isinstance(target_years, int) else [int(year) for year in target_years]
    frame = dividends.copy()
    frame["stock_id"] = pd.to_numeric(frame["stock_id"], errors="coerce")
    frame["TotalCashDividend"] = pd.to_numeric(frame["TotalCashDividend"], errors="coerce")
    frame["ex_dividend_year"] = pd.to_numeric(frame["ex_dividend_year"], errors="coerce")
    known_dividend_stocks = set(frame["stock_id"].dropna().astype(int).unique())
    actual = (
        frame[frame["ex_dividend_year"].isin(years)]
        .dropna(subset=["stock_id", "ex_dividend_year"])
        .groupby(["stock_id", "ex_dividend_year"], as_index=False)
        .agg(
            actual_cash_dividend_per_share=("TotalCashDividend", "sum"),
            actual_cash_dividend_record_count=("TotalCashDividend", "count"),
        )
        .rename(columns={"ex_dividend_year": "target_year"})
    )
    base = pd.MultiIndex.from_product([stock_ids, years], names=["stock_id", "target_year"]).to_frame(index=False)
    result = base.merge(actual, on=["stock_id", "target_year"], how="left")
    result["has_known_dividend_data"] = result["stock_id"].isin(known_dividend_stocks)
    missing_known = result["actual_cash_dividend_per_share"].isna() & result["has_known_dividend_data"]
    result.loc[missing_known, "actual_cash_dividend_per_share"] = 0.0
    result.loc[missing_known, "actual_cash_dividend_record_count"] = 0
    result["actual_dividend_paid"] = np.where(
        result["actual_cash_dividend_per_share"].notna(),
        result["actual_cash_dividend_per_share"].gt(0),
        np.nan,
    )
    result["actual_cash_dividend_source"] = np.where(
        result["actual_cash_dividend_record_count"].fillna(0).gt(0),
        "target-year ex-dividend records",
        np.where(
            result["has_known_dividend_data"],
            "no target-year ex-dividend record treated as zero",
            "missing dividend coverage",
        ),
    )
    return result.sort_values(["stock_id", "target_year"])


def select_available_cash_history(
    annual_cash: pd.DataFrame,
    stock_id: int,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    stock = annual_cash[annual_cash["stock_id"].eq(int(stock_id))].copy()
    available = pd.to_datetime(stock["dividend_available_date_max"], errors="coerce")
    return stock[available.le(pd.Timestamp(as_of_date))].sort_values("fiscal_year")


def _recent_numeric_values(values: pd.Series, limit: int = 3) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").dropna().tail(limit)


def _weighted_recent(values: pd.Series) -> float:
    if values.empty:
        return np.nan
    latest_first = values.iloc[::-1].to_numpy(dtype=float)
    weights = np.array([0.5, 0.3, 0.2], dtype=float)[: len(latest_first)]
    weights = weights / weights.sum()
    return float(np.sum(latest_first * weights))


def build_time_safe_features(
    revenue_data: pd.DataFrame,
    eps: pd.DataFrame,
    annual_cash: pd.DataFrame,
    stock_id: int,
    target_year: int,
    as_of_date: pd.Timestamp,
) -> dict[str, object]:
    cash_history = select_available_cash_history(annual_cash, stock_id, as_of_date)
    cash_values = _recent_numeric_values(cash_history["cash_dividend_per_share"], limit=3)
    latest_cash = cash_values.iloc[-1] if not cash_values.empty else np.nan
    latest_reference_year = (
        int(pd.to_numeric(cash_history["fiscal_year"], errors="coerce").dropna().iloc[-1])
        if not cash_history.empty and pd.to_numeric(cash_history["fiscal_year"], errors="coerce").notna().any()
        else np.nan
    )

    stock_eps = eps[
        eps["stock_id"].eq(int(stock_id))
        & pd.to_datetime(eps["statement_available_date"], errors="coerce").le(pd.Timestamp(as_of_date))
    ].sort_values("statement_available_date")
    eps_values = pd.to_numeric(stock_eps["EPS"], errors="coerce").dropna()
    latest_eps = float(eps_values.iloc[-1]) if not eps_values.empty else np.nan
    eps_ttm = float(eps_values.tail(4).sum()) if len(eps_values) >= 1 else np.nan
    prev_eps_ttm = float(eps_values.iloc[-8:-4].sum()) if len(eps_values) >= 8 else np.nan

    stock_revenue = revenue_data[
        revenue_data["stock_id"].eq(int(stock_id))
        & pd.to_datetime(revenue_data["revenue_available_date"], errors="coerce").le(pd.Timestamp(as_of_date))
    ].sort_values(["revenue_year", "revenue_month"])
    revenue_values = pd.to_numeric(stock_revenue["revenue_thousand"], errors="coerce").dropna()

    return {
        "as_of_date": pd.Timestamp(as_of_date).date().isoformat(),
        "last_known_cash_dividend": float(latest_cash) if pd.notna(latest_cash) else np.nan,
        "recent_cash_dividend_median": float(cash_values.median()) if not cash_values.empty else np.nan,
        "recent_cash_dividend_mean": float(cash_values.mean()) if not cash_values.empty else np.nan,
        "recent_cash_dividend_smoothed": _weighted_recent(cash_values),
        "recent_paid_rate": float(cash_values.gt(0).mean()) if not cash_values.empty else np.nan,
        "dividend_history_count": int(len(cash_history)),
        "years_since_last_dividend_reference": float(int(target_year) - latest_reference_year)
        if pd.notna(latest_reference_year)
        else np.nan,
        "latest_available_eps": latest_eps,
        "available_eps_ttm": eps_ttm,
        "available_eps_ttm_yoy": _safe_divide(eps_ttm, prev_eps_ttm) - 1 if pd.notna(prev_eps_ttm) else np.nan,
        "eps_history_count": int(len(eps_values)),
        "known_revenue_ltm": float(revenue_values.tail(12).sum()) if len(revenue_values) >= 1 else np.nan,
        "known_revenue_recent_3m": float(revenue_values.tail(3).sum()) if len(revenue_values) >= 1 else np.nan,
        "known_revenue_yoy_mean": float(pd.to_numeric(stock_revenue["yoy"], errors="coerce").tail(12).mean())
        if "yoy" in stock_revenue.columns
        else np.nan,
        "known_revenue_mom_mean": float(pd.to_numeric(stock_revenue["mom"], errors="coerce").tail(3).mean())
        if "mom" in stock_revenue.columns
        else np.nan,
        "revenue_history_count": int(len(revenue_values)),
    }


def attach_dividend_selection_buckets(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return samples.copy()
    result = samples.copy()
    paid_rate = pd.to_numeric(result["recent_paid_rate"], errors="coerce")
    history_count = pd.to_numeric(result["dividend_history_count"], errors="coerce").fillna(0)
    latest = pd.to_numeric(result["last_known_cash_dividend"], errors="coerce")
    result["paid_rate_bucket"] = np.select(
        [
            history_count.le(0),
            paid_rate.ge(0.67),
            paid_rate.gt(0),
        ],
        [
            "paid_no_history",
            "paid_high",
            "paid_mixed",
        ],
        default="paid_none",
    )
    result["dividend_history_bucket"] = np.select(
        [
            history_count.ge(3),
            history_count.gt(0),
        ],
        [
            "history_enough",
            "history_sparse",
        ],
        default="history_none",
    )
    result["latest_dividend_bucket"] = np.select(
        [
            latest.gt(0),
            latest.eq(0),
        ],
        [
            "latest_positive",
            "latest_zero",
        ],
        default="latest_missing",
    )
    result["dividend_selection_bucket"] = (
        result["paid_rate_bucket"].astype(str)
        + "|"
        + result["dividend_history_bucket"].astype(str)
        + "|"
        + result["latest_dividend_bucket"].astype(str)
    )
    return result


def build_direct_samples(
    context: pd.DataFrame,
    revenue_data: pd.DataFrame,
    eps: pd.DataFrame,
    annual_cash: pd.DataFrame,
    actual_targets: pd.DataFrame,
    default_as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    actual_lookup = actual_targets.set_index(["stock_id", "target_year"], drop=False)
    rows = []
    for _, context_row in context.iterrows():
        stock_id = int(context_row["stock_id"])
        target_year = int(context_row["target_year"])
        as_of = pd.Timestamp(context_row.get("as_of_date", default_as_of or default_as_of_date(target_year)))
        actual_key = (stock_id, target_year)
        actual = actual_lookup.loc[actual_key] if actual_key in actual_lookup.index else pd.Series(dtype=object)
        features = build_time_safe_features(revenue_data, eps, annual_cash, stock_id, target_year, as_of)
        row = context_row.to_dict()
        row.update(features)
        row.update(
            {
                "actual_cash_dividend_per_share": actual.get("actual_cash_dividend_per_share", np.nan),
                "actual_cash_dividend_record_count": actual.get("actual_cash_dividend_record_count", np.nan),
                "actual_dividend_paid": actual.get("actual_dividend_paid", np.nan),
                "actual_cash_dividend_source": actual.get("actual_cash_dividend_source", "missing dividend coverage"),
                "has_known_dividend_data": actual.get("has_known_dividend_data", False),
            }
        )
        rows.append(row)
    return attach_dividend_selection_buckets(pd.DataFrame(rows))


def infer_training_years(dividends: pd.DataFrame, target_year: int, minimum_year: int = 2020) -> list[int]:
    years = pd.to_numeric(dividends.get("ex_dividend_year", pd.Series(dtype=float)), errors="coerce")
    selected = sorted(int(year) for year in years.dropna().unique() if minimum_year <= int(year) < int(target_year))
    return selected


def _constant_probability(probability: float, index: pd.Index) -> pd.Series:
    return pd.Series(float(probability), index=index)


def _fit_ml_hurdle(training_samples: pd.DataFrame, method: str) -> dict[str, object]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    threshold = direct_method_threshold(method)
    training = training_samples.dropna(subset=["actual_cash_dividend_per_share"]).copy()
    feature_columns = [
        column
        for column in DIRECT_FEATURE_COLUMNS
        if column in training.columns and training[column].notna().any()
    ]
    if training.empty:
        return {
            "method": method,
            "threshold": threshold,
            "feature_columns": feature_columns,
            "classifier": None,
            "classifier_constant_probability": 0.0,
            "regressor": None,
            "regressor_constant_amount": 0.0,
            "training_sample_count": 0,
            "positive_training_sample_count": 0,
        }

    x_train = training[feature_columns]
    y_paid = training["actual_cash_dividend_per_share"].gt(0).astype(int)
    if y_paid.nunique() >= 2 and feature_columns:
        classifier = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        classifier.fit(x_train, y_paid)
        constant_probability = np.nan
    else:
        classifier = None
        constant_probability = float(y_paid.mean())

    paid_training = training[training["actual_cash_dividend_per_share"].gt(0)].copy()
    if paid_training.empty:
        regressor = None
        constant_amount = 0.0
    else:
        if "elastic_net" in method:
            estimator = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000)
        else:
            estimator = Ridge(alpha=1.0)
        constant_amount = float(paid_training["actual_cash_dividend_per_share"].median())
        if feature_columns:
            regressor = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
            regressor.fit(paid_training[feature_columns], paid_training["actual_cash_dividend_per_share"])
        else:
            regressor = None

    return {
        "method": method,
        "threshold": threshold,
        "feature_columns": feature_columns,
        "classifier": classifier,
        "classifier_constant_probability": constant_probability,
        "regressor": regressor,
        "regressor_constant_amount": constant_amount,
        "training_sample_count": int(len(training)),
        "positive_training_sample_count": int(len(paid_training)),
    }


def _predict_ml_hurdle(samples: pd.DataFrame, fitted: dict[str, object]) -> pd.DataFrame:
    feature_columns = list(fitted.get("feature_columns", DIRECT_FEATURE_COLUMNS))
    x = samples[feature_columns] if feature_columns else pd.DataFrame(index=samples.index)
    classifier = fitted["classifier"]
    if classifier is None:
        probability = _constant_probability(float(fitted["classifier_constant_probability"]), samples.index)
    else:
        probability = pd.Series(classifier.predict_proba(x)[:, 1], index=samples.index)

    regressor = fitted["regressor"]
    if regressor is None:
        amount = pd.Series(float(fitted["regressor_constant_amount"]), index=samples.index)
    else:
        amount = pd.Series(regressor.predict(x), index=samples.index)

    paid = probability.ge(float(fitted["threshold"]))
    estimated = amount.clip(lower=0).where(paid, 0.0)
    return pd.DataFrame(
        {
            "predicted_dividend_paid_probability": probability,
            "predicted_dividend_paid": paid,
            "estimated_cash_dividend": estimated,
            "training_sample_count": int(fitted["training_sample_count"]),
            "positive_training_sample_count": int(fitted["positive_training_sample_count"]),
        },
        index=samples.index,
    )


def _predict_heuristic_hurdle(samples: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "direct_hurdle_last_known":
        amount = pd.to_numeric(samples["last_known_cash_dividend"], errors="coerce").fillna(0).clip(lower=0)
        probability = amount.gt(0).astype(float)
    elif method == "direct_hurdle_recent_median":
        amount = pd.to_numeric(samples["recent_cash_dividend_median"], errors="coerce").fillna(0).clip(lower=0)
        probability = pd.to_numeric(samples["recent_paid_rate"], errors="coerce").fillna(0)
    elif method == "direct_hurdle_smoothed":
        amount = pd.to_numeric(samples["recent_cash_dividend_smoothed"], errors="coerce").fillna(0).clip(lower=0)
        probability = pd.to_numeric(samples["recent_paid_rate"], errors="coerce").fillna(0)
    else:
        raise ValueError(f"Unsupported direct dividend heuristic: {method}")
    paid = probability.ge(0.5)
    estimated = amount.where(paid, 0.0)
    return pd.DataFrame(
        {
            "predicted_dividend_paid_probability": probability,
            "predicted_dividend_paid": paid,
            "estimated_cash_dividend": estimated,
            "training_sample_count": 0,
            "positive_training_sample_count": 0,
        },
        index=samples.index,
    )


def build_direct_dividend_estimates(
    samples: pd.DataFrame,
    training_samples: pd.DataFrame,
    methods: list[str],
) -> pd.DataFrame:
    samples = attach_dividend_selection_buckets(samples)
    frames = []
    base_columns = [
        "source_family",
        "model",
        "stock_id",
        "stock_name",
        "industry_category",
        "target_year",
        "predicted_annual_revenue",
        "actual_annual_revenue",
        "annual_revenue_abs_percent_error",
        "as_of_date",
        "actual_cash_dividend_per_share",
        "actual_cash_dividend_record_count",
        "actual_dividend_paid",
        "actual_cash_dividend_source",
        "has_known_dividend_data",
    ]
    for method in methods:
        if is_ml_hurdle_method(method):
            prediction = _predict_ml_hurdle(samples, _fit_ml_hurdle(training_samples, method))
            method_source = "time-safe hurdle classifier + paid-only regression"
        else:
            prediction = _predict_heuristic_hurdle(samples, method)
            method_source = "time-safe hurdle heuristic from known dividend history"
        frame = samples[base_columns + DIRECT_BUCKET_COLUMNS + DIRECT_FEATURE_COLUMNS].copy()
        frame["eps_method"] = DIRECT_EPS_METHOD_LABEL
        frame["dividend_method"] = method
        frame["dividend_method_source"] = method_source
        frame["dividend_reference_year"] = (
            pd.to_numeric(frame["target_year"], errors="coerce")
            - pd.to_numeric(frame["years_since_last_dividend_reference"], errors="coerce")
        )
        frame["dividend_history_count"] = frame["dividend_history_count"].fillna(0).astype(int)
        frame["uses_target_year_ex_dividend"] = False
        frame["uses_post_as_of_dividend"] = False
        frame = pd.concat([frame, prediction.reset_index(drop=True)], axis=1)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def enrich_direct_stock_accuracy(stock_accuracy: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if stock_accuracy.empty or predictions.empty:
        return stock_accuracy.copy()
    group_columns = ["source_family", "model", "eps_method", "dividend_method", "stock_id"]
    metadata_columns = [
        "target_year",
        "validation_fold_year",
        "selection_strategy",
        "selection_source",
        "selected_dividend_method",
        "selected_underlying_dividend_method",
        "bucket_winner_dividend_method",
        "bucket_support_status",
        "fallback_to_global",
        "min_bucket_folds",
        "min_bucket_stock_years",
        "validation_fold_count",
        "validation_stock_year_count",
        "validation_primary_metric",
        *DIRECT_BUCKET_COLUMNS,
    ]
    available_columns = [column for column in metadata_columns if column in predictions.columns]
    if not available_columns:
        return stock_accuracy.copy()
    metadata = (
        predictions[group_columns + available_columns]
        .drop_duplicates(group_columns)
        .copy()
    )
    return stock_accuracy.merge(metadata, on=group_columns, how="left")


def build_direct_monthly_predictions(
    monthly_context: pd.DataFrame,
    estimates: pd.DataFrame,
    revenue_data: pd.DataFrame,
    target_year: int,
    min_stock_price: float = DEFAULT_MIN_STOCK_PRICE,
) -> pd.DataFrame:
    if monthly_context.empty or estimates.empty:
        return pd.DataFrame()
    stock_ids = sorted(int(stock_id) for stock_id in monthly_context["stock_id"].dropna().unique())
    prices = build_stock_price_lookup(revenue_data, stock_ids, target_year)
    merge_keys = ["source_family", "model", "stock_id", "target_year"]
    monthly = monthly_context.merge(estimates, on=merge_keys, how="inner", suffixes=("", "_estimate"))
    for column in ["stock_name", "industry_category"]:
        estimate_column = f"{column}_estimate"
        if estimate_column in monthly.columns:
            monthly[column] = monthly[column].where(monthly[column].notna(), monthly[estimate_column])
            monthly = monthly.drop(columns=[estimate_column])
    monthly = monthly.merge(
        prices[["stock_id", "price_month", "price_date", "close_price", "price_source"]],
        left_on=["stock_id", "target_month"],
        right_on=["stock_id", "price_month"],
        how="left",
    )
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
    for rate in ENTRY_YIELD_RATES:
        label = int(rate * 100)
        monthly[f"entry_price_at_{label}_percent"] = monthly["estimated_cash_dividend"] / rate
        monthly[f"actual_entry_price_at_{label}_percent"] = monthly["actual_cash_dividend_per_share"] / rate
    monthly = monthly.drop(columns=["price_month", "price_date", "close_price", "price_source"])
    return monthly.sort_values(["stock_id", "source_family", "model", "dividend_method", "target_month"])


def align_predictions_to_actual_targets(predictions: pd.DataFrame, actual_targets: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    actual = actual_targets[
        [
            "stock_id",
            "target_year",
            "actual_cash_dividend_per_share",
            "actual_cash_dividend_record_count",
            "actual_dividend_paid",
            "actual_cash_dividend_source",
            "has_known_dividend_data",
        ]
    ]
    drop_columns = [
        column
        for column in [
            "actual_cash_dividend_per_share",
            "actual_cash_dividend_record_count",
            "actual_dividend_paid",
            "actual_cash_dividend_source",
            "has_known_dividend_data",
        ]
        if column in predictions.columns
    ]
    result = predictions.drop(columns=drop_columns).merge(actual, on=["stock_id", "target_year"], how="left")
    result["cash_dividend_error"] = (
        result["estimated_cash_dividend"] - result["actual_cash_dividend_per_share"]
    )
    result["cash_dividend_abs_error"] = result["cash_dividend_error"].abs()
    result["actual_dividend_yield_percent"] = np.where(
        result["stock_price_valid_for_yield"],
        result["actual_cash_dividend_per_share"] / result["stock_price"] * 100,
        np.nan,
    )
    result["yield_error_percent_point"] = (
        result["predicted_dividend_yield_percent"] - result["actual_dividend_yield_percent"]
    )
    result["yield_abs_error_percent_point"] = result["yield_error_percent_point"].abs()
    for rate in ENTRY_YIELD_RATES:
        label = int(rate * 100)
        result[f"actual_entry_price_at_{label}_percent"] = result["actual_cash_dividend_per_share"] / rate
    return result


def _stock_year_count(group: pd.DataFrame) -> int:
    columns = [column for column in ["stock_id", "validation_fold_year"] if column in group.columns]
    if "validation_fold_year" not in columns:
        return int(group["stock_id"].nunique())
    return int(group[columns].drop_duplicates().shape[0])


def build_validation_method_scores(
    validation_stock_accuracy: pd.DataFrame,
) -> pd.DataFrame:
    if validation_stock_accuracy.empty:
        return pd.DataFrame()
    direct = validation_stock_accuracy[
        validation_stock_accuracy["dividend_method"].astype(str).str.startswith("direct_")
    ].copy()
    if direct.empty:
        return pd.DataFrame()
    rows = []
    group_columns = ["eps_method", "dividend_method"]
    for group_key, group in direct.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_key, strict=True))
        cash_error = pd.to_numeric(group["cash_dividend_abs_error"], errors="coerce")
        yield_error = pd.to_numeric(group["yield_mae_percent_point"], errors="coerce")
        row.update(
            {
                "validation_fold_count": int(group["validation_fold_year"].nunique())
                if "validation_fold_year" in group.columns
                else 1,
                "validation_stock_year_count": _stock_year_count(group),
                "average_cash_dividend_abs_error": float(cash_error.mean()) if cash_error.notna().any() else np.nan,
                "median_cash_dividend_abs_error": float(cash_error.median()) if cash_error.notna().any() else np.nan,
                "average_yield_mae_percent_point": float(yield_error.mean()) if yield_error.notna().any() else np.nan,
                "median_yield_mae_percent_point": float(yield_error.median()) if yield_error.notna().any() else np.nan,
            }
        )
        rows.append(row)
    return _round_numeric_columns(
        pd.DataFrame(rows),
        [
            "average_cash_dividend_abs_error",
            "median_cash_dividend_abs_error",
            "average_yield_mae_percent_point",
            "median_yield_mae_percent_point",
        ],
    ).sort_values(["average_cash_dividend_abs_error", "average_yield_mae_percent_point", "dividend_method"])


def select_validation_direct_method(
    validation_scores: pd.DataFrame,
    primary_metric: str = "average_cash_dividend_abs_error",
) -> pd.DataFrame:
    direct = validation_scores[
        validation_scores["dividend_method"].astype(str).str.startswith("direct_")
        & validation_scores[primary_metric].notna()
    ].copy()
    if direct.empty:
        return pd.DataFrame(
            columns=[
                "selection_strategy",
                "selected_dividend_method",
                "primary_metric",
                "validation_primary_metric",
            ]
        )
    winner = direct.sort_values(
        [primary_metric, "average_yield_mae_percent_point", "dividend_method"],
        na_position="last",
    ).iloc[0]
    return pd.DataFrame(
        [
            {
                "selection_strategy": GLOBAL_SELECTION_STRATEGY,
                "selection_source": "multi-year validation cash_dividend_abs_error",
                "selected_dividend_method": winner["dividend_method"],
                "selected_eps_method": winner["eps_method"],
                "primary_metric": primary_metric,
                "validation_primary_metric": winner[primary_metric],
                "validation_average_yield_mae_percent_point": winner.get(
                    "average_yield_mae_percent_point", np.nan
                ),
                "validation_fold_count": winner.get("validation_fold_count", np.nan),
                "validation_stock_year_count": winner.get("validation_stock_year_count", np.nan),
            }
        ]
    )


def build_validation_bucket_method_scores(validation_stock_accuracy: pd.DataFrame) -> pd.DataFrame:
    if validation_stock_accuracy.empty or "dividend_selection_bucket" not in validation_stock_accuracy.columns:
        return pd.DataFrame()
    direct = validation_stock_accuracy[
        validation_stock_accuracy["dividend_method"].astype(str).str.startswith("direct_")
        & validation_stock_accuracy["dividend_selection_bucket"].notna()
    ].copy()
    if direct.empty:
        return pd.DataFrame()
    rows = []
    group_columns = ["dividend_selection_bucket", "eps_method", "dividend_method"]
    for group_key, group in direct.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_key, strict=True))
        first = group.iloc[0]
        cash_error = pd.to_numeric(group["cash_dividend_abs_error"], errors="coerce")
        yield_error = pd.to_numeric(group["yield_mae_percent_point"], errors="coerce")
        row.update(
            {
                "paid_rate_bucket": first.get("paid_rate_bucket", np.nan),
                "dividend_history_bucket": first.get("dividend_history_bucket", np.nan),
                "latest_dividend_bucket": first.get("latest_dividend_bucket", np.nan),
                "validation_fold_count": int(group["validation_fold_year"].nunique())
                if "validation_fold_year" in group.columns
                else 1,
                "validation_stock_year_count": _stock_year_count(group),
                "average_cash_dividend_abs_error": float(cash_error.mean()) if cash_error.notna().any() else np.nan,
                "median_cash_dividend_abs_error": float(cash_error.median()) if cash_error.notna().any() else np.nan,
                "average_yield_mae_percent_point": float(yield_error.mean()) if yield_error.notna().any() else np.nan,
                "median_yield_mae_percent_point": float(yield_error.median()) if yield_error.notna().any() else np.nan,
            }
        )
        rows.append(row)
    return _round_numeric_columns(
        pd.DataFrame(rows),
        [
            "average_cash_dividend_abs_error",
            "median_cash_dividend_abs_error",
            "average_yield_mae_percent_point",
            "median_yield_mae_percent_point",
        ],
    ).sort_values(
        ["dividend_selection_bucket", "average_cash_dividend_abs_error", "average_yield_mae_percent_point"]
    )


def select_validation_bucket_methods(
    bucket_scores: pd.DataFrame,
    primary_metric: str = "average_cash_dividend_abs_error",
    global_selected_method: str | None = None,
    min_bucket_folds: int = DEFAULT_MIN_BUCKET_FOLDS,
    min_bucket_stock_years: int = DEFAULT_MIN_BUCKET_STOCK_YEARS,
) -> pd.DataFrame:
    if bucket_scores.empty:
        return pd.DataFrame(
            columns=[
                "selection_strategy",
                "dividend_selection_bucket",
                "selected_dividend_method",
                "bucket_winner_dividend_method",
                "bucket_support_status",
                "fallback_to_global",
                "primary_metric",
                "validation_primary_metric",
            ]
        )
    candidates = bucket_scores[bucket_scores[primary_metric].notna()].copy()
    if candidates.empty:
        return pd.DataFrame()
    winners = candidates.loc[
        candidates.groupby("dividend_selection_bucket")[primary_metric].idxmin()
    ].copy()
    winners["selection_strategy"] = BUCKET_SELECTION_STRATEGY
    winners["selection_source"] = "multi-year validation bucket cash_dividend_abs_error"
    winners["bucket_winner_dividend_method"] = winners["dividend_method"]
    winners["selected_eps_method"] = winners["eps_method"]
    winners["primary_metric"] = primary_metric
    winners["validation_primary_metric"] = winners[primary_metric]
    has_support = (
        pd.to_numeric(winners["validation_fold_count"], errors="coerce").ge(int(min_bucket_folds))
        & pd.to_numeric(winners["validation_stock_year_count"], errors="coerce").ge(int(min_bucket_stock_years))
    )
    winners["bucket_support_status"] = np.where(
        has_support,
        BUCKET_SUPPORT_STATUS_SUPPORTED,
        BUCKET_SUPPORT_STATUS_FALLBACK,
    )
    winners["fallback_to_global"] = ~has_support
    if global_selected_method:
        winners["selected_dividend_method"] = winners["bucket_winner_dividend_method"].where(
            has_support,
            global_selected_method,
        )
    else:
        winners["selected_dividend_method"] = winners["bucket_winner_dividend_method"]
    winners["min_bucket_folds"] = int(min_bucket_folds)
    winners["min_bucket_stock_years"] = int(min_bucket_stock_years)
    result_columns = [
        "selection_strategy",
        "selection_source",
        "dividend_selection_bucket",
        "paid_rate_bucket",
        "dividend_history_bucket",
        "latest_dividend_bucket",
        "selected_dividend_method",
        "bucket_winner_dividend_method",
        "bucket_support_status",
        "fallback_to_global",
        "min_bucket_folds",
        "min_bucket_stock_years",
        "selected_eps_method",
        "primary_metric",
        "validation_primary_metric",
        "average_yield_mae_percent_point",
        "validation_fold_count",
        "validation_stock_year_count",
    ]
    return winners[result_columns].sort_values(["dividend_selection_bucket", "selected_dividend_method"])


def filter_selected_direct_predictions(
    predictions: pd.DataFrame,
    method_selection: pd.DataFrame,
    bucket_method_selection: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if predictions.empty or method_selection.empty:
        return pd.DataFrame()
    global_selection = method_selection[
        method_selection["selection_strategy"].eq(GLOBAL_SELECTION_STRATEGY)
    ].copy()
    if global_selection.empty:
        global_selection = method_selection.head(1).copy()
    selected_method = global_selection.iloc[0]["selected_dividend_method"]
    global_selected = predictions[predictions["dividend_method"].eq(selected_method)].copy()
    global_selected["selection_strategy"] = global_selection.iloc[0]["selection_strategy"]
    global_selected["selection_source"] = global_selection.iloc[0]["selection_source"]
    global_selected["selected_dividend_method"] = selected_method

    frames = [global_selected]
    if bucket_method_selection is not None and not bucket_method_selection.empty:
        bucket_columns = [
            "dividend_selection_bucket",
            "selected_dividend_method",
            "bucket_winner_dividend_method",
            "bucket_support_status",
            "fallback_to_global",
            "min_bucket_folds",
            "min_bucket_stock_years",
            "validation_fold_count",
            "validation_stock_year_count",
            "validation_primary_metric",
        ]
        bucket_columns = [column for column in bucket_columns if column in bucket_method_selection.columns]
        bucket_selected = predictions.merge(
            bucket_method_selection[bucket_columns].drop_duplicates("dividend_selection_bucket"),
            on="dividend_selection_bucket",
            how="left",
        )
        missing_bucket_selection = bucket_selected["selected_dividend_method"].isna()
        bucket_selected["selected_dividend_method"] = bucket_selected["selected_dividend_method"].fillna(
            selected_method
        )
        if "bucket_winner_dividend_method" not in bucket_selected.columns:
            bucket_selected["bucket_winner_dividend_method"] = np.nan
        if "bucket_support_status" not in bucket_selected.columns:
            bucket_selected["bucket_support_status"] = np.nan
        fallback_status = pd.Series(np.nan, index=bucket_selected.index, dtype=object)
        fallback_status.loc[missing_bucket_selection] = BUCKET_SUPPORT_STATUS_FALLBACK
        bucket_selected["bucket_support_status"] = bucket_selected["bucket_support_status"].where(
            bucket_selected["bucket_support_status"].notna(),
            fallback_status,
        )
        if "fallback_to_global" not in bucket_selected.columns:
            bucket_selected["fallback_to_global"] = False
        bucket_selected["fallback_to_global"] = (
            bucket_selected["fallback_to_global"].fillna(missing_bucket_selection).astype(bool)
        )
        bucket_selected = bucket_selected[
            bucket_selected["dividend_method"].eq(bucket_selected["selected_dividend_method"])
        ].copy()
        bucket_selected["selected_underlying_dividend_method"] = bucket_selected["dividend_method"]
        bucket_selected["dividend_method"] = SELECTED_BUCKET_DIVIDEND_METHOD
        bucket_selected["selection_strategy"] = BUCKET_SELECTION_STRATEGY
        bucket_selected["selection_source"] = "multi-year validation bucket cash_dividend_abs_error"
        frames.append(bucket_selected)

    return pd.concat(frames, ignore_index=True, sort=False)


def build_comparison_vs_baselines(
    selected_direct_overall: pd.DataFrame,
    baseline_overall: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    if not selected_direct_overall.empty:
        direct = selected_direct_overall.copy()
        direct["benchmark_family"] = "direct_dividend_model_selected"
        frames.append(direct)
    if not baseline_overall.empty:
        baseline = baseline_overall.copy()
        baseline["benchmark_family"] = "announcement_safe_baseline"
        frames.append(baseline)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True, sort=False)
    return result.sort_values(
        [
            "average_cash_dividend_abs_error",
            "average_yield_mae_percent_point",
            "benchmark_family",
            "source_family",
            "model",
            "eps_method",
            "dividend_method",
        ],
        na_position="last",
    )


def _write_outputs(output_dir: Path, outputs: dict[str, pd.DataFrame], run_config: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_run_config_and_registry(output_dir, run_config)


def _stock_year_context_with_asof(context: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    result = context.copy()
    result["as_of_date"] = pd.Timestamp(as_of_date).date().isoformat()
    return result


def run_direct_dividend_model_benchmark(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    model_names = None if args.all_models else parse_str_csv(args.models)
    direct_thresholds = parse_float_csv(args.threshold_values) or list(DEFAULT_DIRECT_THRESHOLDS)
    direct_methods = parse_str_csv(args.direct_methods) or build_threshold_direct_methods(direct_thresholds)
    baseline_eps_methods = parse_str_csv(args.baseline_eps_methods) or list(DEFAULT_BASELINE_EPS_METHODS)
    baseline_dividend_methods = parse_str_csv(args.baseline_dividend_methods) or list(DEFAULT_BASELINE_DIVIDEND_METHODS)
    validation_years = parse_int_csv(args.validation_years) or [int(args.validation_year)]
    validation_years = sorted({int(year) for year in validation_years if int(year) < int(args.target_year)})
    stock_ids = parse_int_csv(args.stock_ids)
    target_as_of = pd.Timestamp(args.as_of_date) if args.as_of_date else default_as_of_date(args.target_year)

    test_monthly = load_prediction_input(
        args.input_predictions,
        target_year=args.target_year,
        model_names=model_names,
        stock_ids=stock_ids,
        stock_limit=args.stock_limit,
    )
    selected_stock_ids = sorted(int(stock_id) for stock_id in test_monthly["stock_id"].dropna().unique())
    revenue_data, eps, dividends = load_direct_source_data()
    stock_metadata = build_stock_metadata(test_monthly, selected_stock_ids)
    annual_cash = build_annual_cash_dividend(dividends, target_year=args.target_year)

    validation_folds = []
    for validation_year in validation_years:
        validation_as_of = validation_as_of_date_for_year(validation_year, args.validation_as_of_date)
        validation_monthly = build_actual_revenue_prediction_rows(
            revenue_data,
            selected_stock_ids,
            validation_year,
            stock_metadata,
        )
        if validation_monthly.empty:
            continue
        validation_training_years = infer_training_years(dividends, validation_year)
        validation_folds.append(
            {
                "validation_year": int(validation_year),
                "validation_as_of": validation_as_of,
                "validation_monthly": validation_monthly,
                "validation_annual_context": _stock_year_context_with_asof(
                    build_annual_context(validation_monthly),
                    validation_as_of,
                ),
                "validation_training_years": validation_training_years,
                "validation_training_context": build_historical_context(
                    selected_stock_ids,
                    stock_metadata,
                    validation_training_years,
                ),
            }
        )

    test_annual_context = _stock_year_context_with_asof(build_annual_context(test_monthly), target_as_of)

    test_training_years = infer_training_years(dividends, args.target_year)
    test_training_context = build_historical_context(selected_stock_ids, stock_metadata, test_training_years)

    validation_training_year_union = sorted(
        {
            int(year)
            for fold in validation_folds
            for year in fold["validation_training_years"]
        }
    )
    all_years = sorted(
        set(
            validation_training_year_union
            + test_training_years
            + [fold["validation_year"] for fold in validation_folds]
            + [args.target_year]
        )
    )
    actual_targets = build_actual_cash_dividend_targets(dividends, selected_stock_ids, all_years)

    validation_prediction_frames = []
    validation_stock_accuracy_frames = []
    validation_overall_accuracy_frames = []
    for fold in validation_folds:
        validation_training_samples = build_direct_samples(
            fold["validation_training_context"],
            revenue_data,
            eps,
            annual_cash,
            actual_targets,
        )
        validation_samples = build_direct_samples(
            fold["validation_annual_context"],
            revenue_data,
            eps,
            annual_cash,
            actual_targets,
        )
        validation_estimates = build_direct_dividend_estimates(
            validation_samples,
            validation_training_samples,
            direct_methods,
        )
        fold_predictions = build_direct_monthly_predictions(
            fold["validation_monthly"],
            validation_estimates,
            revenue_data,
            fold["validation_year"],
            min_stock_price=args.min_stock_price,
        )
        fold_predictions["validation_fold_year"] = int(fold["validation_year"])
        fold_stock_accuracy = enrich_direct_stock_accuracy(
            build_dividend_layer_stock_accuracy(fold_predictions),
            fold_predictions,
        )
        fold_stock_accuracy["validation_fold_year"] = int(fold["validation_year"])
        fold_overall_accuracy = build_dividend_layer_overall_accuracy(fold_stock_accuracy)
        fold_overall_accuracy["validation_fold_year"] = int(fold["validation_year"])
        validation_prediction_frames.append(fold_predictions)
        validation_stock_accuracy_frames.append(fold_stock_accuracy)
        validation_overall_accuracy_frames.append(fold_overall_accuracy)

    validation_predictions = (
        pd.concat(validation_prediction_frames, ignore_index=True, sort=False)
        if validation_prediction_frames
        else pd.DataFrame()
    )
    validation_stock_accuracy = (
        pd.concat(validation_stock_accuracy_frames, ignore_index=True, sort=False)
        if validation_stock_accuracy_frames
        else pd.DataFrame()
    )
    validation_overall_accuracy = (
        pd.concat(validation_overall_accuracy_frames, ignore_index=True, sort=False)
        if validation_overall_accuracy_frames
        else pd.DataFrame()
    )
    validation_method_scores = build_validation_method_scores(validation_stock_accuracy)
    global_method_selection = select_validation_direct_method(
        validation_method_scores,
        primary_metric=args.primary_selection_metric,
    )
    global_selected_method = (
        global_method_selection.iloc[0]["selected_dividend_method"]
        if not global_method_selection.empty
        else None
    )
    bucket_method_scores = build_validation_bucket_method_scores(validation_stock_accuracy)
    bucket_method_selection = select_validation_bucket_methods(
        bucket_method_scores,
        primary_metric=args.primary_selection_metric,
        global_selected_method=global_selected_method,
        min_bucket_folds=args.min_bucket_folds,
        min_bucket_stock_years=args.min_bucket_stock_years,
    )
    method_selection = pd.concat(
        [global_method_selection, bucket_method_selection],
        ignore_index=True,
        sort=False,
    )

    test_training_samples = build_direct_samples(
        test_training_context,
        revenue_data,
        eps,
        annual_cash,
        actual_targets,
    )
    test_samples = build_direct_samples(
        test_annual_context,
        revenue_data,
        eps,
        annual_cash,
        actual_targets,
    )
    test_estimates = build_direct_dividend_estimates(test_samples, test_training_samples, direct_methods)
    test_predictions = build_direct_monthly_predictions(
        test_monthly,
        test_estimates,
        revenue_data,
        args.target_year,
        min_stock_price=args.min_stock_price,
    )
    test_stock_accuracy = enrich_direct_stock_accuracy(
        build_dividend_layer_stock_accuracy(test_predictions),
        test_predictions,
    )
    test_overall_accuracy = build_dividend_layer_overall_accuracy(test_stock_accuracy)
    test_winner_summary = build_dividend_layer_winner_summary(
        test_stock_accuracy,
        primary_metric="cash_dividend_abs_error",
    )

    selected_test_predictions = filter_selected_direct_predictions(
        test_predictions,
        global_method_selection,
        bucket_method_selection=bucket_method_selection,
    )
    selected_test_stock_accuracy = enrich_direct_stock_accuracy(
        build_dividend_layer_stock_accuracy(selected_test_predictions),
        selected_test_predictions,
    )
    selected_test_overall_accuracy = build_dividend_layer_overall_accuracy(selected_test_stock_accuracy)

    baseline_predictions, baseline_failures = build_dividend_layer_predictions(
        test_monthly,
        target_year=args.target_year,
        eps_methods=baseline_eps_methods,
        dividend_methods=baseline_dividend_methods,
        as_of_date=target_as_of,
        min_stock_price=args.min_stock_price,
    )
    target_actual = actual_targets[actual_targets["target_year"].eq(int(args.target_year))].copy()
    baseline_predictions = align_predictions_to_actual_targets(baseline_predictions, target_actual)
    baseline_stock_accuracy = build_dividend_layer_stock_accuracy(baseline_predictions)
    baseline_overall_accuracy = build_dividend_layer_overall_accuracy(baseline_stock_accuracy)
    comparison = build_comparison_vs_baselines(selected_test_overall_accuracy, baseline_overall_accuracy)

    failed_runs = baseline_failures.copy()
    if failed_runs.empty:
        failed_runs = pd.DataFrame(columns=["source_family", "model", "stock_id", "error"])

    outputs = {
        "direct_dividend_validation_predictions": validation_predictions,
        "direct_dividend_validation_stock_accuracy": validation_stock_accuracy,
        "direct_dividend_validation_overall_accuracy": validation_overall_accuracy,
        "direct_dividend_validation_method_scores": validation_method_scores,
        "direct_dividend_bucket_method_scores": bucket_method_scores,
        "direct_dividend_bucket_method_selection": bucket_method_selection,
        "direct_dividend_method_selection": method_selection,
        "direct_dividend_test_predictions": test_predictions,
        "direct_dividend_test_stock_accuracy": test_stock_accuracy,
        "direct_dividend_test_overall_accuracy": test_overall_accuracy,
        "direct_dividend_test_winner_summary": test_winner_summary,
        "direct_dividend_selected_test_predictions": selected_test_predictions,
        "direct_dividend_selected_test_stock_accuracy": selected_test_stock_accuracy,
        "direct_dividend_selected_test_overall_accuracy": selected_test_overall_accuracy,
        "direct_dividend_baseline_predictions": baseline_predictions,
        "direct_dividend_baseline_stock_accuracy": baseline_stock_accuracy,
        "direct_dividend_baseline_overall_accuracy": baseline_overall_accuracy,
        "direct_dividend_comparison_vs_baselines": comparison,
        "direct_dividend_failed_runs": failed_runs,
    }
    run_config = {
        "input_predictions": str(Path(args.input_predictions).resolve()),
        "output_dir": str(args.output_dir),
        "target_year": int(args.target_year),
        "validation_year": int(args.validation_year),
        "validation_years": [fold["validation_year"] for fold in validation_folds],
        "as_of_date": pd.Timestamp(target_as_of).date().isoformat(),
        "validation_as_of_dates": {
            str(fold["validation_year"]): pd.Timestamp(fold["validation_as_of"]).date().isoformat()
            for fold in validation_folds
        },
        "models": model_names,
        "direct_methods": direct_methods,
        "direct_thresholds": direct_thresholds,
        "selected_direct_method": global_selected_method,
        "selected_bucket_count": int(len(bucket_method_selection)),
        "supported_bucket_count": int(
            bucket_method_selection["fallback_to_global"].eq(False).sum()
        )
        if "fallback_to_global" in bucket_method_selection.columns
        else 0,
        "fallback_bucket_count": int(
            bucket_method_selection["fallback_to_global"].eq(True).sum()
        )
        if "fallback_to_global" in bucket_method_selection.columns
        else 0,
        "min_bucket_folds": int(args.min_bucket_folds),
        "min_bucket_stock_years": int(args.min_bucket_stock_years),
        "primary_selection_metric": args.primary_selection_metric,
        "baseline_eps_methods": baseline_eps_methods,
        "baseline_dividend_methods": baseline_dividend_methods,
        "all_models": bool(args.all_models),
        "stock_ids": selected_stock_ids,
        "stock_count": int(len(selected_stock_ids)),
        "validation_training_years": {
            str(fold["validation_year"]): fold["validation_training_years"]
            for fold in validation_folds
        },
        "test_training_years": test_training_years,
        "test_prediction_rows": int(len(test_predictions)),
        "baseline_prediction_rows": int(len(baseline_predictions)),
        "min_stock_price": float(args.min_stock_price),
        "failed_runs": int(len(failed_runs)),
    }
    run_config = enrich_run_config_from_args(
        run_config,
        args,
        experiment_family="direct_dividend_model_benchmark",
        extra={"input_predictions": str(args.input_predictions)},
    )
    _write_outputs(Path(args.output_dir), outputs, run_config)
    outputs["run_config"] = pd.DataFrame([run_config])
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-predictions", type=Path, default=DEFAULT_INPUT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR)
    parser.add_argument("--validation-year", type=int, default=DEFAULT_VALIDATION_YEAR)
    parser.add_argument(
        "--validation-years",
        default=",".join(str(year) for year in DEFAULT_VALIDATION_YEARS),
        help="Comma-separated validation years. Defaults to expanding folds 2022,2023,2024.",
    )
    parser.add_argument("--as-of-date", help="Information cutoff date for target-year test.")
    parser.add_argument(
        "--validation-as-of-date",
        help="Validation cutoff month/day. For multiple validation years, the month/day is applied to each year.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODEL_NAMES),
        help="Comma-separated revenue model names to include.",
    )
    parser.add_argument(
        "--direct-methods",
        help="Comma-separated direct dividend methods. Defaults to heuristics plus threshold sweep methods.",
    )
    parser.add_argument(
        "--threshold-values",
        default=",".join(str(value) for value in DEFAULT_DIRECT_THRESHOLDS),
        help="Comma-separated hurdle thresholds used when --direct-methods is omitted.",
    )
    parser.add_argument(
        "--baseline-eps-methods",
        default=",".join(DEFAULT_BASELINE_EPS_METHODS),
        help="Comma-separated EPS methods for announcement-safe baselines.",
    )
    parser.add_argument(
        "--baseline-dividend-methods",
        default=",".join(DEFAULT_BASELINE_DIVIDEND_METHODS),
        help="Comma-separated announcement-safe baseline dividend methods.",
    )
    parser.add_argument(
        "--primary-selection-metric",
        default="average_cash_dividend_abs_error",
        help="Validation overall metric used to select the direct method.",
    )
    parser.add_argument(
        "--min-bucket-folds",
        type=int,
        default=DEFAULT_MIN_BUCKET_FOLDS,
        help="Minimum validation folds required before a bucket can use its own selected method.",
    )
    parser.add_argument(
        "--min-bucket-stock-years",
        type=int,
        default=DEFAULT_MIN_BUCKET_STOCK_YEARS,
        help="Minimum validation stock-year support required before a bucket can use its own selected method.",
    )
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--stock-ids", help="Comma-separated stock IDs.")
    parser.add_argument("--stock-limit", type=int, help="Limit stock pool for smoke runs.")
    parser.add_argument("--min-stock-price", type=float, default=DEFAULT_MIN_STOCK_PRICE)
    return add_registry_arguments(parser)


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_direct_dividend_model_benchmark(args)
    print("Wrote direct dividend model benchmark outputs to", args.output_dir)
    print("\nValidation method selection:")
    print(outputs["direct_dividend_method_selection"].to_string(index=False))
    print("\n2025 selected direct model vs announcement-safe baselines:")
    print(outputs["direct_dividend_comparison_vs_baselines"].to_string(index=False))


if __name__ == "__main__":
    main()
