"""Benchmark EPS transforms applied to comparable revenue predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_benchmark.adapters.ensemble_evidence import load_eps_data, load_revenue_data
from forecast_benchmark.benchmark_config import DEFAULT_TARGET_YEAR, PROJECT_ROOT
from forecast_benchmark.experiment_registry import (
    add_registry_arguments,
    enrich_run_config_from_args,
    write_run_config_and_registry,
)
from forecast_benchmark.run_benchmark import parse_int_csv, parse_str_csv
from forecast_benchmark.yield_benchmark import (
    DEFAULT_INPUT_PREDICTIONS,
    DEFAULT_MODEL_NAMES,
    load_prediction_input,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forecast_benchmark" / "outputs" / "eps_benchmark"
DEFAULT_MIN_ABS_ACTUAL_EPS_FOR_PERCENT = 0.01
DEFAULT_EPS_METHODS = (
    "current_ratio",
    "seasonal_quarter_median",
    "ridge_annual",
    "lasso_annual",
    "elastic_net_annual",
)
ML_EPS_METHODS = {"ridge_annual", "lasso_annual", "elastic_net_annual"}
ML_FEATURE_COLUMNS = [
    "annual_revenue_thousand",
    "revenue_yoy",
    "annual_revenue_change",
    "prev_annual_revenue_thousand",
    "prev_annual_eps",
    "prev_eps_to_revenue_ratio",
]


def _first_valid(values: pd.Series) -> object:
    valid = values.dropna()
    return valid.iloc[0] if not valid.empty else np.nan


def _safe_percent_error(
    predicted: pd.Series,
    actual: pd.Series,
    min_abs_actual: float = DEFAULT_MIN_ABS_ACTUAL_EPS_FOR_PERCENT,
) -> pd.Series:
    actual_abs = actual.abs()
    return np.where(actual_abs >= float(min_abs_actual), (predicted - actual).abs() / actual_abs * 100, np.nan)


def _round_numeric_columns(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").round(digits)
    return result


def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_revenue_data(), load_eps_data()


def build_actual_annual_eps(eps: pd.DataFrame, target_year: int) -> pd.DataFrame:
    actual_eps = (
        eps[eps["eps_year"].eq(int(target_year))]
        .groupby("stock_id", as_index=False)
        .agg(actual_annual_eps=("latest_eps", "sum"), actual_eps_quarter_count=("latest_eps", "count"))
    )
    return actual_eps


def build_annual_revenue_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    for column in ["stock_name", "industry_category"]:
        if column not in frame.columns:
            frame[column] = np.nan

    metadata = (
        frame.groupby("stock_id", as_index=False)
        .agg(stock_name=("stock_name", _first_valid), industry_category=("industry_category", _first_valid))
    )
    annual = (
        frame.groupby(["source_family", "model", "stock_id"], as_index=False)
        .agg(
            target_year=("target_year", _first_valid),
            monthly_observations=("predicted_revenue", "count"),
            actual_monthly_observations=("actual_revenue", "count"),
            predicted_annual_revenue=("predicted_revenue", "sum"),
            actual_annual_revenue=("actual_revenue", "sum"),
        )
        .merge(metadata, on="stock_id", how="left")
    )
    annual["target_year"] = pd.to_numeric(annual["target_year"], errors="coerce").astype("Int64")
    return annual[
        [
            "source_family",
            "model",
            "stock_id",
            "stock_name",
            "industry_category",
            "target_year",
            "monthly_observations",
            "actual_monthly_observations",
            "predicted_annual_revenue",
            "actual_annual_revenue",
        ]
    ].sort_values(["stock_id", "source_family", "model"])


def build_quarterly_revenue_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["target_quarter"] = ((pd.to_numeric(frame["target_month"], errors="coerce") - 1) // 3 + 1).astype(
        "Int64"
    )
    quarterly = (
        frame.dropna(subset=["target_quarter"])
        .groupby(["source_family", "model", "stock_id", "target_quarter"], as_index=False)
        .agg(
            predicted_quarter_revenue=("predicted_revenue", "sum"),
            actual_quarter_revenue=("actual_revenue", "sum"),
            monthly_observations=("predicted_revenue", "count"),
        )
    )
    quarterly["target_quarter"] = quarterly["target_quarter"].astype(int)
    return quarterly


def build_historical_annual_frame(
    revenue_data: pd.DataFrame,
    eps: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:
    annual_revenue = (
        revenue_data[revenue_data["revenue_year"].lt(int(target_year))]
        .groupby(["stock_id", "revenue_year"], as_index=False)
        .agg(annual_revenue_thousand=("revenue_thousand", "sum"))
    )
    annual_eps = (
        eps[eps["eps_year"].lt(int(target_year))]
        .groupby(["stock_id", "eps_year"], as_index=False)
        .agg(annual_eps=("latest_eps", "sum"), eps_quarter_count=("latest_eps", "count"))
    )
    history = annual_revenue.merge(
        annual_eps,
        left_on=["stock_id", "revenue_year"],
        right_on=["stock_id", "eps_year"],
        how="left",
    )
    history = history.sort_values(["stock_id", "revenue_year"]).reset_index(drop=True)
    grouped = history.groupby("stock_id", group_keys=False)
    history["prev_annual_revenue_thousand"] = grouped["annual_revenue_thousand"].shift(1)
    history["prev_annual_eps"] = grouped["annual_eps"].shift(1)
    history["revenue_yoy"] = history["annual_revenue_thousand"] / history["prev_annual_revenue_thousand"] - 1
    history["annual_revenue_change"] = (
        history["annual_revenue_thousand"] - history["prev_annual_revenue_thousand"]
    )
    history["eps_to_revenue_ratio"] = history["annual_eps"] / history["annual_revenue_thousand"]
    history["prev_eps_to_revenue_ratio"] = (
        history["prev_annual_eps"] / history["prev_annual_revenue_thousand"]
    )
    return history.replace([np.inf, -np.inf], np.nan)


def build_historical_quarter_frame(
    revenue_data: pd.DataFrame,
    eps: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:
    revenue = revenue_data[revenue_data["revenue_year"].lt(int(target_year))].copy()
    revenue["eps_quarter"] = ((pd.to_numeric(revenue["revenue_month"], errors="coerce") - 1) // 3 + 1).astype(
        "Int64"
    )
    quarterly_revenue = (
        revenue.dropna(subset=["eps_quarter"])
        .groupby(["stock_id", "revenue_year", "eps_quarter"], as_index=False)
        .agg(quarter_revenue_thousand=("revenue_thousand", "sum"))
    )
    quarterly_revenue["eps_quarter"] = quarterly_revenue["eps_quarter"].astype(int)
    quarterly_eps = (
        eps[eps["eps_year"].lt(int(target_year))]
        .groupby(["stock_id", "eps_year", "eps_quarter"], as_index=False)
        .agg(quarter_eps=("latest_eps", "sum"), eps_rows=("latest_eps", "count"))
    )
    history = quarterly_revenue.merge(
        quarterly_eps,
        left_on=["stock_id", "revenue_year", "eps_quarter"],
        right_on=["stock_id", "eps_year", "eps_quarter"],
        how="inner",
    )
    history["quarter_eps_to_revenue_ratio"] = history["quarter_eps"] / history["quarter_revenue_thousand"]
    return history.replace([np.inf, -np.inf], np.nan)


def select_annual_ratio_candidates(annual_history: pd.DataFrame, stock_id: int) -> pd.DataFrame:
    stock_history = annual_history[
        (annual_history["stock_id"].eq(int(stock_id)))
        & (annual_history["annual_revenue_thousand"] > 0)
        & (annual_history["annual_eps"].notna())
    ].copy()
    full_years = stock_history[stock_history["eps_quarter_count"] >= 4].copy()
    candidates = full_years if not full_years.empty else stock_history
    candidates["eps_to_revenue_ratio"] = candidates["annual_eps"] / candidates["annual_revenue_thousand"]
    return candidates.replace([np.inf, -np.inf], np.nan).dropna(subset=["eps_to_revenue_ratio"])


def estimate_current_ratio_eps(
    stock_id: int,
    annual_revenue_thousand: float,
    annual_history: pd.DataFrame,
) -> dict[str, object]:
    if not np.isfinite(float(annual_revenue_thousand)):
        return {
            "estimated_eps": np.nan,
            "eps_reference_year": np.nan,
            "eps_to_revenue_ratio": np.nan,
            "eps_transform_source": "missing annual revenue",
        }

    candidates = select_annual_ratio_candidates(annual_history, int(stock_id))
    if candidates.empty:
        return {
            "estimated_eps": np.nan,
            "eps_reference_year": np.nan,
            "eps_to_revenue_ratio": np.nan,
            "eps_transform_source": "missing historical EPS/revenue ratio",
        }

    recent = candidates.sort_values("revenue_year").tail(3)
    ratio = float(recent["eps_to_revenue_ratio"].median())
    reference_year = int(recent["revenue_year"].max())
    return {
        "estimated_eps": float(annual_revenue_thousand) * ratio,
        "eps_reference_year": reference_year,
        "eps_to_revenue_ratio": ratio,
        "eps_transform_source": "forecast revenue x historical EPS/revenue median",
    }


def estimate_seasonal_quarter_eps(
    stock_id: int,
    quarter_predictions: pd.DataFrame,
    quarter_history: pd.DataFrame,
    annual_ratio_estimate: dict[str, object],
) -> dict[str, object]:
    if quarter_predictions.empty:
        return {
            "estimated_eps": np.nan,
            "eps_reference_year": np.nan,
            "eps_to_revenue_ratio": np.nan,
            "eps_transform_source": "missing quarterly revenue prediction",
        }

    annual_fallback_ratio = annual_ratio_estimate.get("eps_to_revenue_ratio", np.nan)
    estimated_eps = 0.0
    estimated_revenue = 0.0
    used_quarters = 0
    fallback_quarters = 0
    reference_years = []

    by_quarter = quarter_predictions.set_index("target_quarter")["predicted_quarter_revenue"].to_dict()
    for quarter in range(1, 5):
        quarter_revenue = float(by_quarter.get(quarter, 0.0))
        if not np.isfinite(quarter_revenue):
            continue

        candidates = quarter_history[
            (quarter_history["stock_id"].eq(int(stock_id)))
            & (quarter_history["eps_quarter"].eq(int(quarter)))
            & (quarter_history["quarter_revenue_thousand"] > 0)
            & (quarter_history["quarter_eps_to_revenue_ratio"].notna())
        ].copy()
        if not candidates.empty:
            recent = candidates.sort_values("revenue_year").tail(3)
            ratio = float(recent["quarter_eps_to_revenue_ratio"].median())
            reference_years.append(int(recent["revenue_year"].max()))
        else:
            ratio = float(annual_fallback_ratio) if pd.notna(annual_fallback_ratio) else np.nan
            fallback_quarters += 1

        if np.isfinite(ratio):
            estimated_eps += quarter_revenue * ratio
            estimated_revenue += quarter_revenue
            used_quarters += 1

    if used_quarters == 0:
        return {
            "estimated_eps": np.nan,
            "eps_reference_year": np.nan,
            "eps_to_revenue_ratio": np.nan,
            "eps_transform_source": "missing seasonal and annual EPS/revenue ratios",
        }

    weighted_ratio = estimated_eps / estimated_revenue if estimated_revenue else np.nan
    reference_year = max(reference_years) if reference_years else annual_ratio_estimate.get("eps_reference_year")
    return {
        "estimated_eps": float(estimated_eps),
        "eps_reference_year": reference_year,
        "eps_to_revenue_ratio": float(weighted_ratio) if np.isfinite(weighted_ratio) else np.nan,
        "eps_transform_source": (
            f"same-quarter EPS/revenue median; fallback_quarters={fallback_quarters}"
        ),
    }


def build_ml_training_frame(annual_history: pd.DataFrame) -> pd.DataFrame:
    training = annual_history.copy()
    training = training[
        (training["annual_revenue_thousand"] > 0)
        & (training["annual_eps"].notna())
        & (training["eps_quarter_count"] >= 4)
    ].copy()
    if training.empty:
        training = annual_history[
            (annual_history["annual_revenue_thousand"] > 0) & (annual_history["annual_eps"].notna())
        ].copy()
    return training.replace([np.inf, -np.inf], np.nan)


def build_ml_target_frame(
    annual_predictions: pd.DataFrame,
    annual_history: pd.DataFrame,
) -> pd.DataFrame:
    previous = (
        annual_history.sort_values(["stock_id", "revenue_year"])
        .groupby("stock_id", as_index=False)
        .tail(1)[
            [
                "stock_id",
                "revenue_year",
                "annual_revenue_thousand",
                "annual_eps",
                "eps_to_revenue_ratio",
            ]
        ]
        .rename(
            columns={
                "revenue_year": "previous_reference_year",
                "annual_revenue_thousand": "prev_annual_revenue_thousand",
                "annual_eps": "prev_annual_eps",
                "eps_to_revenue_ratio": "prev_eps_to_revenue_ratio",
            }
        )
    )
    target = annual_predictions.merge(previous, on="stock_id", how="left").copy()
    target["annual_revenue_thousand"] = target["predicted_annual_revenue"]
    target["revenue_yoy"] = target["annual_revenue_thousand"] / target["prev_annual_revenue_thousand"] - 1
    target["annual_revenue_change"] = (
        target["annual_revenue_thousand"] - target["prev_annual_revenue_thousand"]
    )
    return target.replace([np.inf, -np.inf], np.nan)


def predict_ml_eps(
    annual_predictions: pd.DataFrame,
    annual_history: pd.DataFrame,
    eps_method: str,
) -> pd.DataFrame:
    if eps_method not in ML_EPS_METHODS:
        raise ValueError(f"Unsupported ML EPS method: {eps_method}")

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    training = build_ml_training_frame(annual_history)
    if len(training) < 10:
        raise ValueError("Not enough historical EPS rows to train ML EPS model.")

    target = build_ml_target_frame(annual_predictions, annual_history)
    if target.empty:
        return pd.DataFrame()

    estimator = Ridge(alpha=1.0)
    if eps_method == "lasso_annual":
        estimator = Lasso(alpha=0.001, max_iter=20000)
    elif eps_method == "elastic_net_annual":
        estimator = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000)

    pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
    pipeline.fit(training[ML_FEATURE_COLUMNS], training["annual_eps"])
    result = target.copy()
    result["estimated_eps"] = pipeline.predict(result[ML_FEATURE_COLUMNS])
    result["eps_reference_year"] = result["previous_reference_year"]
    result["eps_to_revenue_ratio"] = np.where(
        result["annual_revenue_thousand"] != 0,
        result["estimated_eps"] / result["annual_revenue_thousand"],
        np.nan,
    )
    result["eps_transform_source"] = f"{eps_method} fitted on historical annual revenue/EPS"
    return result


def _base_prediction_row(
    annual_row: pd.Series,
    eps_method: str,
    estimate: dict[str, object],
    is_oracle: bool = False,
) -> dict[str, object]:
    return {
        "source_family": annual_row["source_family"],
        "model": annual_row["model"],
        "eps_method": eps_method,
        "is_oracle": bool(is_oracle),
        "stock_id": int(annual_row["stock_id"]),
        "stock_name": annual_row.get("stock_name", np.nan),
        "industry_category": annual_row.get("industry_category", np.nan),
        "target_year": annual_row.get("target_year", np.nan),
        "monthly_observations": annual_row.get("monthly_observations", np.nan),
        "actual_monthly_observations": annual_row.get("actual_monthly_observations", np.nan),
        "predicted_annual_revenue": annual_row.get("predicted_annual_revenue", np.nan),
        "actual_annual_revenue": annual_row.get("actual_annual_revenue", np.nan),
        "estimated_eps": estimate.get("estimated_eps", np.nan),
        "eps_reference_year": estimate.get("eps_reference_year", np.nan),
        "eps_to_revenue_ratio": estimate.get("eps_to_revenue_ratio", np.nan),
        "eps_transform_source": estimate.get("eps_transform_source", ""),
    }


def build_eps_predictions(
    predictions: pd.DataFrame,
    target_year: int,
    eps_methods: list[str] | None = None,
    include_oracle: bool = True,
    min_abs_actual_eps_for_percent: float = DEFAULT_MIN_ABS_ACTUAL_EPS_FOR_PERCENT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    revenue_data, eps = load_source_data()
    eps_methods = list(eps_methods or DEFAULT_EPS_METHODS)
    unknown_methods = sorted(set(eps_methods) - (set(DEFAULT_EPS_METHODS) | {"oracle_current_ratio"}))
    if unknown_methods:
        raise ValueError(f"Unknown EPS methods: {unknown_methods}")

    annual_history = build_historical_annual_frame(revenue_data, eps, target_year)
    quarter_history = build_historical_quarter_frame(revenue_data, eps, target_year)
    annual_predictions = build_annual_revenue_predictions(predictions)
    quarterly_predictions = build_quarterly_revenue_predictions(predictions)
    actual_eps = build_actual_annual_eps(eps, target_year)
    rows = []
    failures = []

    for _, annual_row in annual_predictions.iterrows():
        stock_id = int(annual_row["stock_id"])
        base_key = {
            "source_family": annual_row["source_family"],
            "model": annual_row["model"],
            "stock_id": stock_id,
        }
        try:
            annual_ratio = estimate_current_ratio_eps(
                stock_id,
                float(annual_row["predicted_annual_revenue"]),
                annual_history,
            )
            if "current_ratio" in eps_methods:
                rows.append(_base_prediction_row(annual_row, "current_ratio", annual_ratio))

            if "seasonal_quarter_median" in eps_methods:
                quarter_rows = quarterly_predictions[
                    (quarterly_predictions["source_family"].eq(annual_row["source_family"]))
                    & (quarterly_predictions["model"].eq(annual_row["model"]))
                    & (quarterly_predictions["stock_id"].eq(stock_id))
                ]
                seasonal = estimate_seasonal_quarter_eps(stock_id, quarter_rows, quarter_history, annual_ratio)
                rows.append(_base_prediction_row(annual_row, "seasonal_quarter_median", seasonal))
        except Exception as error:  # pragma: no cover - integration safety net.
            failures.append(
                {
                    **base_key,
                    "eps_method": "ratio_methods",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    for eps_method in [method for method in eps_methods if method in ML_EPS_METHODS]:
        try:
            ml_predictions = predict_ml_eps(annual_predictions, annual_history, eps_method)
            for _, ml_row in ml_predictions.iterrows():
                estimate = {
                    "estimated_eps": ml_row["estimated_eps"],
                    "eps_reference_year": ml_row["eps_reference_year"],
                    "eps_to_revenue_ratio": ml_row["eps_to_revenue_ratio"],
                    "eps_transform_source": ml_row["eps_transform_source"],
                }
                rows.append(_base_prediction_row(ml_row, eps_method, estimate))
        except Exception as error:  # pragma: no cover - optional dependency/integration safety net.
            failures.append(
                {
                    "source_family": "all",
                    "model": "all",
                    "stock_id": np.nan,
                    "eps_method": eps_method,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    if include_oracle:
        oracle_base = (
            annual_predictions.sort_values(["stock_id", "source_family", "model"])
            .groupby("stock_id", as_index=False)
            .first()
        )
        for _, annual_row in oracle_base.iterrows():
            annual_row = annual_row.copy()
            annual_row["source_family"] = "oracle"
            annual_row["model"] = "actual_revenue"
            annual_row["predicted_annual_revenue"] = annual_row["actual_annual_revenue"]
            estimate = estimate_current_ratio_eps(
                int(annual_row["stock_id"]),
                float(annual_row["actual_annual_revenue"]),
                annual_history,
            )
            estimate["eps_transform_source"] = "actual revenue x historical EPS/revenue median"
            rows.append(_base_prediction_row(annual_row, "oracle_current_ratio", estimate, is_oracle=True))

    if not rows:
        failures_frame = pd.DataFrame(
            failures,
            columns=["source_family", "model", "stock_id", "eps_method", "error_type", "error"],
        )
        return pd.DataFrame(), failures_frame

    eps_predictions = pd.DataFrame(rows).merge(actual_eps, on="stock_id", how="left")
    eps_predictions["annual_revenue_error"] = (
        eps_predictions["predicted_annual_revenue"] - eps_predictions["actual_annual_revenue"]
    )
    eps_predictions["annual_revenue_abs_error"] = eps_predictions["annual_revenue_error"].abs()
    eps_predictions["annual_revenue_abs_percent_error"] = np.where(
        eps_predictions["actual_annual_revenue"].abs() > 0,
        eps_predictions["annual_revenue_abs_error"] / eps_predictions["actual_annual_revenue"].abs() * 100,
        np.nan,
    )
    eps_predictions["eps_error"] = eps_predictions["estimated_eps"] - eps_predictions["actual_annual_eps"]
    eps_predictions["eps_abs_error"] = eps_predictions["eps_error"].abs()
    eps_predictions["eps_percent_error_valid"] = (
        eps_predictions["actual_annual_eps"].abs() >= float(min_abs_actual_eps_for_percent)
    )
    eps_predictions["eps_abs_percent_error"] = _safe_percent_error(
        eps_predictions["estimated_eps"],
        eps_predictions["actual_annual_eps"],
        min_abs_actual=min_abs_actual_eps_for_percent,
    )
    eps_predictions["eps_underestimated"] = eps_predictions["estimated_eps"] < eps_predictions["actual_annual_eps"]
    failures_frame = pd.DataFrame(
        failures,
        columns=["source_family", "model", "stock_id", "eps_method", "error_type", "error"],
    )
    return eps_predictions.sort_values(["stock_id", "source_family", "model", "eps_method"]), failures_frame


def build_eps_stock_accuracy(eps_predictions: pd.DataFrame) -> pd.DataFrame:
    if eps_predictions.empty:
        return pd.DataFrame()

    group_columns = [
        "source_family",
        "model",
        "eps_method",
        "is_oracle",
        "stock_id",
        "stock_name",
        "industry_category",
    ]
    rows = []
    for group_key, group in eps_predictions.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_key, strict=True))
        first = group.iloc[0]
        row.update(
            {
                "monthly_observations": int(first.get("monthly_observations", 0))
                if pd.notna(first.get("monthly_observations", np.nan))
                else 0,
                "predicted_annual_revenue": first.get("predicted_annual_revenue", np.nan),
                "actual_annual_revenue": first.get("actual_annual_revenue", np.nan),
                "annual_revenue_abs_percent_error": first.get("annual_revenue_abs_percent_error", np.nan),
                "estimated_eps": first.get("estimated_eps", np.nan),
                "actual_annual_eps": first.get("actual_annual_eps", np.nan),
                "actual_eps_quarter_count": first.get("actual_eps_quarter_count", np.nan),
                "eps_percent_error_valid": bool(first.get("eps_percent_error_valid", False))
                if pd.notna(first.get("eps_percent_error_valid", np.nan))
                else False,
                "eps_error": first.get("eps_error", np.nan),
                "eps_abs_error": first.get("eps_abs_error", np.nan),
                "eps_abs_percent_error": first.get("eps_abs_percent_error", np.nan),
                "eps_underestimated": bool(first.get("eps_underestimated", False))
                if pd.notna(first.get("eps_underestimated", np.nan))
                else False,
                "eps_reference_year": first.get("eps_reference_year", np.nan),
                "eps_to_revenue_ratio": first.get("eps_to_revenue_ratio", np.nan),
                "eps_transform_source": first.get("eps_transform_source", ""),
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
        "eps_error",
        "eps_abs_error",
        "eps_abs_percent_error",
        "eps_reference_year",
        "eps_to_revenue_ratio",
    ]
    result = _round_numeric_columns(result, numeric_columns)
    return result.sort_values(
        ["is_oracle", "stock_id", "eps_abs_error", "source_family", "model", "eps_method"],
        na_position="last",
    )


def build_eps_overall_accuracy(stock_accuracy: pd.DataFrame) -> pd.DataFrame:
    if stock_accuracy.empty:
        return pd.DataFrame()

    metric_columns = [
        "annual_revenue_abs_percent_error",
        "eps_abs_error",
        "eps_abs_percent_error",
    ]
    rows = []
    group_columns = ["source_family", "model", "eps_method", "is_oracle"]
    for group_key, group in stock_accuracy.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_key, strict=True))
        valid_eps = group.dropna(subset=["estimated_eps", "actual_annual_eps"])
        row.update(
            {
                "stock_count": int(group["stock_id"].nunique()),
                "valid_revenue_stock_count": int(group["annual_revenue_abs_percent_error"].notna().sum()),
                "valid_eps_stock_count": int(group["eps_abs_error"].notna().sum()),
                "valid_eps_percent_stock_count": int(group["eps_abs_percent_error"].notna().sum()),
                "average_eps_bias": float(valid_eps["eps_error"].mean()) if not valid_eps.empty else np.nan,
                "eps_underestimate_rate": float(valid_eps["eps_underestimated"].mean() * 100)
                if not valid_eps.empty
                else np.nan,
            }
        )
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"average_{column}"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"median_{column}"] = float(values.median()) if values.notna().any() else np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    numeric_columns = [
        "average_eps_bias",
        "eps_underestimate_rate",
        *[column for column in result.columns if column.startswith("average_") or column.startswith("median_")],
    ]
    result = _round_numeric_columns(result, numeric_columns)
    return result.sort_values(
        ["is_oracle", "average_eps_abs_error", "average_eps_abs_percent_error", "source_family", "model"],
        na_position="last",
    )


def build_eps_method_winner_summary(
    stock_accuracy: pd.DataFrame,
    primary_metric: str = "eps_abs_error",
) -> pd.DataFrame:
    if stock_accuracy.empty:
        return pd.DataFrame()
    if primary_metric not in stock_accuracy.columns:
        raise ValueError(f"Unknown primary metric: {primary_metric}")

    valid = stock_accuracy[
        (~stock_accuracy["is_oracle"].astype(bool)) & stock_accuracy[primary_metric].notna()
    ].copy()
    if valid.empty:
        return pd.DataFrame()

    compared_stocks = int(valid["stock_id"].nunique())
    winners = valid.loc[valid.groupby("stock_id")[primary_metric].idxmin()]
    group_columns = ["source_family", "model", "eps_method"]
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
        ["stock_wins", "average_primary_metric", "source_family", "model", "eps_method"],
        ascending=[False, True, True, True, True],
    )


def build_eps_error_decomposition(overall_accuracy: pd.DataFrame) -> pd.DataFrame:
    stage_map = {
        "annual_revenue_percent": "average_annual_revenue_abs_percent_error",
        "eps_absolute": "average_eps_abs_error",
        "eps_percent": "average_eps_abs_percent_error",
    }
    rows = []
    for _, row in overall_accuracy.iterrows():
        for stage, column in stage_map.items():
            rows.append(
                {
                    "source_family": row["source_family"],
                    "model": row["model"],
                    "eps_method": row["eps_method"],
                    "is_oracle": bool(row["is_oracle"]),
                    "error_stage": stage,
                    "average_abs_error": row.get(column, np.nan),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    output_dir: Path,
    eps_predictions: pd.DataFrame,
    stock_accuracy: pd.DataFrame,
    overall_accuracy: pd.DataFrame,
    winner_summary: pd.DataFrame,
    error_decomposition: pd.DataFrame,
    failures: pd.DataFrame,
    run_config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    eps_predictions.to_csv(output_dir / "eps_predictions.csv", index=False, encoding="utf-8-sig")
    stock_accuracy.to_csv(output_dir / "eps_stock_accuracy.csv", index=False, encoding="utf-8-sig")
    overall_accuracy.to_csv(output_dir / "eps_overall_accuracy.csv", index=False, encoding="utf-8-sig")
    winner_summary.to_csv(output_dir / "eps_method_winner_summary.csv", index=False, encoding="utf-8-sig")
    error_decomposition.to_csv(output_dir / "eps_error_decomposition.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(output_dir / "eps_failed_runs.csv", index=False, encoding="utf-8-sig")
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
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--stock-ids", help="Comma-separated stock IDs.")
    parser.add_argument("--stock-limit", type=int, help="Limit stock pool for smoke runs.")
    parser.add_argument("--skip-oracle", action="store_true", help="Skip actual-revenue diagnostic rows.")
    parser.add_argument(
        "--min-abs-actual-eps-for-percent",
        type=float,
        default=DEFAULT_MIN_ABS_ACTUAL_EPS_FOR_PERCENT,
        help="Exclude stocks with smaller absolute actual EPS from EPS percent-error metrics.",
    )
    return add_registry_arguments(parser)


def run_eps_benchmark(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    model_names = None if args.all_models else parse_str_csv(args.models)
    eps_methods = parse_str_csv(args.eps_methods) or list(DEFAULT_EPS_METHODS)
    stock_ids = parse_int_csv(args.stock_ids)
    predictions = load_prediction_input(
        args.input_predictions,
        target_year=args.target_year,
        model_names=model_names,
        stock_ids=stock_ids,
        stock_limit=args.stock_limit,
    )
    eps_predictions, failures = build_eps_predictions(
        predictions,
        target_year=args.target_year,
        eps_methods=eps_methods,
        include_oracle=not args.skip_oracle,
        min_abs_actual_eps_for_percent=args.min_abs_actual_eps_for_percent,
    )
    stock_accuracy = build_eps_stock_accuracy(eps_predictions)
    overall_accuracy = build_eps_overall_accuracy(stock_accuracy)
    winner_summary = build_eps_method_winner_summary(stock_accuracy)
    error_decomposition = build_eps_error_decomposition(overall_accuracy)
    run_config = {
        "input_predictions": str(args.input_predictions),
        "output_dir": str(args.output_dir),
        "target_year": int(args.target_year),
        "models": model_names,
        "eps_methods": eps_methods,
        "all_models": bool(args.all_models),
        "include_oracle": not args.skip_oracle,
        "stock_ids": sorted(int(stock_id) for stock_id in predictions["stock_id"].unique()),
        "stock_count": int(predictions["stock_id"].nunique()),
        "prediction_rows": int(len(predictions)),
        "eps_prediction_rows": int(len(eps_predictions)),
        "min_abs_actual_eps_for_percent": float(args.min_abs_actual_eps_for_percent),
        "failed_runs": int(len(failures)),
    }
    run_config = enrich_run_config_from_args(
        run_config,
        args,
        experiment_family="eps_benchmark",
        extra={"input_predictions": str(args.input_predictions)},
    )
    write_outputs(
        args.output_dir,
        eps_predictions,
        stock_accuracy,
        overall_accuracy,
        winner_summary,
        error_decomposition,
        failures,
        run_config,
    )
    return {
        "eps_predictions": eps_predictions,
        "eps_stock_accuracy": stock_accuracy,
        "eps_overall_accuracy": overall_accuracy,
        "eps_method_winner_summary": winner_summary,
        "eps_error_decomposition": error_decomposition,
        "eps_failed_runs": failures,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = run_eps_benchmark(args)
    print("Wrote EPS benchmark outputs to", args.output_dir)
    print(results["eps_overall_accuracy"].to_string(index=False))


if __name__ == "__main__":
    main()
