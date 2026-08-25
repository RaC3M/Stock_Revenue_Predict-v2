"""Benchmark downstream EPS, dividend, and yield estimates from revenue predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_benchmark.adapters.ensemble_evidence import (
    get_actual_cash_dividend_info,
    get_forecast_dividend_info,
    load_actual_revenue_data,
    load_eps_data,
    load_revenue_data,
    load_stock_price_data,
)
from forecast_benchmark.benchmark_config import DEFAULT_TARGET_YEAR, PROJECT_ROOT
from forecast_benchmark.experiment_registry import (
    add_registry_arguments,
    enrich_run_config_from_args,
    write_run_config_and_registry,
)
from forecast_benchmark.run_benchmark import parse_int_csv, parse_str_csv


DEFAULT_INPUT_PREDICTIONS = (
    PROJECT_ROOT
    / "forecast_benchmark"
    / "outputs"
    / "data_migration_revenue_20260730"
    / "comparable_monthly_predictions.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forecast_benchmark" / "outputs" / "yield_benchmark"
DEFAULT_MODEL_NAMES = (
    "Rolling xLSTM",
    "Rolling xLSTM + Conditional Adjustment",
    "ensemble_revenue",
    "LightGBM",
)
ENTRY_YIELD_RATES = (0.07, 0.08, 0.09)
DEFAULT_MIN_STOCK_PRICE = 1.0


def load_prediction_input(
    path: str | Path,
    target_year: int,
    model_names: list[str] | None = None,
    stock_ids: list[int] | None = None,
    stock_limit: int | None = None,
) -> pd.DataFrame:
    predictions = pd.read_csv(path).copy()
    required_columns = {
        "source_family",
        "model",
        "stock_id",
        "target_year",
        "target_month",
        "predicted_revenue",
        "actual_revenue",
    }
    missing = required_columns - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction input missing columns: {sorted(missing)}")

    for column in ["stock_id", "target_year", "target_month", "predicted_revenue", "actual_revenue"]:
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")
    if "last_observed_revenue" in predictions.columns:
        predictions["last_observed_revenue"] = pd.to_numeric(
            predictions["last_observed_revenue"], errors="coerce"
        )
    else:
        predictions["last_observed_revenue"] = np.nan

    predictions = predictions.dropna(
        subset=["source_family", "model", "stock_id", "target_year", "target_month", "predicted_revenue"]
    )
    predictions["stock_id"] = predictions["stock_id"].astype(int)
    predictions["target_year"] = predictions["target_year"].astype(int)
    predictions["target_month"] = predictions["target_month"].astype(int)
    predictions = predictions[predictions["target_year"].eq(int(target_year))]

    if model_names is not None:
        predictions = predictions[predictions["model"].isin(model_names)]

    if stock_ids is None:
        selected_stock_ids = sorted(int(stock_id) for stock_id in predictions["stock_id"].unique())
    else:
        available = set(int(stock_id) for stock_id in predictions["stock_id"].unique())
        selected_stock_ids = [int(stock_id) for stock_id in stock_ids if int(stock_id) in available]
    if stock_limit is not None:
        selected_stock_ids = selected_stock_ids[: int(stock_limit)]

    return predictions[predictions["stock_id"].isin(selected_stock_ids)].sort_values(
        ["stock_id", "source_family", "model", "target_month"]
    )


def build_actual_eps_lookup(target_year: int) -> pd.DataFrame:
    try:
        eps = load_eps_data()
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["stock_id", "actual_annual_eps", "actual_eps_quarter_count"])

    actual_eps = (
        eps[eps["eps_year"].eq(int(target_year))]
        .groupby("stock_id", as_index=False)
        .agg(actual_annual_eps=("latest_eps", "sum"), actual_eps_quarter_count=("latest_eps", "count"))
    )
    return actual_eps


def is_observed_stock_price_source(price_source: pd.Series) -> pd.Series:
    """Return False for synthetic/simulated prices, which are not valid evidence."""
    normalized = price_source.fillna("").astype(str).str.lower()
    return ~normalized.str.contains(r"synthetic|simulated", regex=True)


def build_stock_price_lookup(revenue_data: pd.DataFrame, stock_ids: list[int], target_year: int) -> pd.DataFrame:
    try:
        prices = load_stock_price_data(target_year=target_year)
        prices = prices[prices["stock_id"].isin(stock_ids) & prices["price_year"].eq(int(target_year))]
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        prices = pd.DataFrame(
            columns=["stock_id", "price_year", "price_month", "price_date", "close_price", "price_source"]
        )

    observed = prices[
        prices["stock_id"].isin([int(stock_id) for stock_id in stock_ids])
        & is_observed_stock_price_source(prices["price_source"])
    ][
        ["stock_id", "price_year", "price_month", "price_date", "close_price", "price_source"]
    ].copy()
    return observed.sort_values(["stock_id", "price_month"]).reset_index(drop=True)


def build_yield_predictions(
    predictions: pd.DataFrame,
    target_year: int,
    min_stock_price: float = DEFAULT_MIN_STOCK_PRICE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    revenue_data = load_revenue_data()
    actual_2025 = load_actual_revenue_data()
    stock_ids = sorted(int(stock_id) for stock_id in predictions["stock_id"].dropna().unique())
    actual_eps = build_actual_eps_lookup(target_year)
    prices = build_stock_price_lookup(revenue_data, stock_ids, target_year)
    failures = []
    rows = []

    actual_annual_revenue = (
        actual_2025[actual_2025["stock_id"].isin(stock_ids)]
        .groupby("stock_id", as_index=False)["actual_revenue"]
        .sum()
        .rename(columns={"actual_revenue": "actual_annual_revenue"})
    )

    group_columns = ["source_family", "model", "stock_id"]
    for (source_family, model_name, stock_id), group in predictions.groupby(group_columns, dropna=False):
        try:
            group = group.sort_values("target_month").copy()
            forecast_annual_revenue = float(pd.to_numeric(group["predicted_revenue"], errors="coerce").sum())
            dividend_info = get_forecast_dividend_info(
                int(stock_id),
                int(target_year),
                forecast_annual_revenue,
                revenue_data,
            )
            actual_dividend = get_actual_cash_dividend_info(int(stock_id), int(target_year))

            monthly = group[
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
                ]
            ].copy()
            monthly = monthly.merge(
                prices[prices["stock_id"].eq(int(stock_id))][
                    ["price_month", "price_date", "close_price", "price_source"]
                ],
                left_on="target_month",
                right_on="price_month",
                how="left",
            )
            monthly = monthly.merge(actual_annual_revenue, on="stock_id", how="left")
            monthly = monthly.merge(actual_eps, on="stock_id", how="left")
            monthly["predicted_annual_revenue"] = forecast_annual_revenue
            monthly["annual_eps_reference_year"] = dividend_info["annual_eps_reference_year"]
            monthly["estimated_eps"] = dividend_info["annual_eps"]
            monthly["payout_ratio"] = dividend_info["payout_ratio"]
            monthly["estimated_cash_dividend"] = dividend_info["cash_dividend_per_share"]
            monthly["cash_dividend_source"] = dividend_info["cash_dividend_source"]
            monthly["actual_cash_dividend_per_share"] = actual_dividend["actual_cash_dividend_per_share"]
            monthly["actual_cash_dividend_source"] = actual_dividend["actual_cash_dividend_source"]
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
            monthly["annual_revenue_abs_percent_error"] = np.where(
                monthly["actual_annual_revenue"] != 0,
                monthly["annual_revenue_abs_error"] / monthly["actual_annual_revenue"] * 100,
                np.nan,
            )
            monthly["eps_error"] = monthly["estimated_eps"] - monthly["actual_annual_eps"]
            monthly["eps_abs_error"] = monthly["eps_error"].abs()
            monthly["cash_dividend_error"] = (
                monthly["estimated_cash_dividend"] - monthly["actual_cash_dividend_per_share"]
            )
            monthly["cash_dividend_abs_error"] = monthly["cash_dividend_error"].abs()
            for rate in ENTRY_YIELD_RATES:
                label = int(rate * 100)
                monthly[f"entry_price_at_{label}_percent"] = monthly["estimated_cash_dividend"] / rate
                monthly[f"actual_entry_price_at_{label}_percent"] = (
                    monthly["actual_cash_dividend_per_share"] / rate
                )
            monthly = monthly.drop(columns=["price_month", "price_date", "close_price", "price_source"])
            rows.append(monthly)
        except Exception as error:  # pragma: no cover - integration safety net.
            failures.append(
                {
                    "source_family": source_family,
                    "model": model_name,
                    "stock_id": int(stock_id),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    if rows:
        yield_predictions = pd.concat(rows, ignore_index=True)
    else:
        yield_predictions = pd.DataFrame()
    failures_frame = pd.DataFrame(
        failures,
        columns=["source_family", "model", "stock_id", "error_type", "error"],
    )
    return yield_predictions, failures_frame


def build_yield_stock_accuracy(yield_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["source_family", "model", "stock_id", "stock_name", "industry_category"]
    for group_key, group in yield_predictions.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_key, strict=True))
        valid_yield = group.dropna(
            subset=["predicted_dividend_yield_percent", "actual_dividend_yield_percent"]
        )
        yield_error = pd.to_numeric(valid_yield["yield_error_percent_point"], errors="coerce")
        yield_abs_error = yield_error.abs()
        row.update(
            {
                "monthly_observations": int(len(valid_yield)),
                "predicted_annual_revenue": float(group["predicted_annual_revenue"].dropna().iloc[0])
                if group["predicted_annual_revenue"].notna().any()
                else np.nan,
                "actual_annual_revenue": float(group["actual_annual_revenue"].dropna().iloc[0])
                if group["actual_annual_revenue"].notna().any()
                else np.nan,
                "annual_revenue_abs_percent_error": float(
                    group["annual_revenue_abs_percent_error"].dropna().iloc[0]
                )
                if group["annual_revenue_abs_percent_error"].notna().any()
                else np.nan,
                "estimated_eps": float(group["estimated_eps"].dropna().iloc[0])
                if group["estimated_eps"].notna().any()
                else np.nan,
                "actual_annual_eps": float(group["actual_annual_eps"].dropna().iloc[0])
                if group["actual_annual_eps"].notna().any()
                else np.nan,
                "eps_abs_error": float(group["eps_abs_error"].dropna().iloc[0])
                if group["eps_abs_error"].notna().any()
                else np.nan,
                "estimated_cash_dividend": float(group["estimated_cash_dividend"].dropna().iloc[0])
                if group["estimated_cash_dividend"].notna().any()
                else np.nan,
                "actual_cash_dividend_per_share": float(
                    group["actual_cash_dividend_per_share"].dropna().iloc[0]
                )
                if group["actual_cash_dividend_per_share"].notna().any()
                else np.nan,
                "cash_dividend_abs_error": float(group["cash_dividend_abs_error"].dropna().iloc[0])
                if group["cash_dividend_abs_error"].notna().any()
                else np.nan,
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
        "yield_mae_percent_point",
        "yield_median_ae_percent_point",
        "yield_rmse_percent_point",
        "mean_predicted_yield_percent",
        "mean_actual_yield_percent",
    ]
    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").round(4)
    return result.sort_values(["stock_id", "yield_mae_percent_point", "source_family", "model"])


def build_yield_overall_accuracy(stock_accuracy: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "annual_revenue_abs_percent_error",
        "eps_abs_error",
        "cash_dividend_abs_error",
        "yield_mae_percent_point",
        "yield_median_ae_percent_point",
        "yield_rmse_percent_point",
    ]
    rows = []
    for (source_family, model), group in stock_accuracy.groupby(["source_family", "model"], dropna=False):
        row = {
            "source_family": source_family,
            "model": model,
            "stock_count": int(group["stock_id"].nunique()),
            "valid_revenue_stock_count": int(group["annual_revenue_abs_percent_error"].notna().sum()),
            "valid_eps_stock_count": int(group["eps_abs_error"].notna().sum()),
            "valid_cash_dividend_stock_count": int(group["cash_dividend_abs_error"].notna().sum()),
            "valid_yield_stock_count": int(group["yield_mae_percent_point"].notna().sum()),
            "monthly_observations": int(group["monthly_observations"].sum()),
        }
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"average_{column}"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"median_{column}"] = float(values.median()) if values.notna().any() else np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    for column in result.columns:
        if column.startswith("average_") or column.startswith("median_"):
            result[column] = pd.to_numeric(result[column], errors="coerce").round(4)
    return result.sort_values(
        ["average_yield_mae_percent_point", "average_cash_dividend_abs_error", "source_family", "model"],
        na_position="last",
    )


def build_yield_winner_summary(
    stock_accuracy: pd.DataFrame,
    primary_metric: str = "yield_mae_percent_point",
) -> pd.DataFrame:
    valid = stock_accuracy.dropna(subset=[primary_metric]).copy()
    if valid.empty:
        return pd.DataFrame()

    compared_stocks = int(valid["stock_id"].nunique())
    winners = valid.loc[valid.groupby("stock_id")[primary_metric].idxmin()]
    winner_counts = (
        winners.groupby(["source_family", "model"], as_index=False)
        .size()
        .rename(columns={"size": "stock_wins"})
    )
    metric_summary = (
        valid.groupby(["source_family", "model"], as_index=False)[primary_metric]
        .agg(average_primary_metric="mean", median_primary_metric="median")
    )
    summary = metric_summary.merge(winner_counts, on=["source_family", "model"], how="left")
    summary["stock_wins"] = summary["stock_wins"].fillna(0).astype(int)
    summary["compared_stocks"] = compared_stocks
    summary["stock_win_rate"] = np.where(
        compared_stocks > 0,
        summary["stock_wins"] / compared_stocks * 100,
        np.nan,
    )
    for column in ["average_primary_metric", "median_primary_metric", "stock_win_rate"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").round(4)
    return summary.sort_values(
        ["stock_wins", "average_primary_metric", "source_family", "model"],
        ascending=[False, True, True, True],
    )


def build_error_decomposition(overall_accuracy: pd.DataFrame) -> pd.DataFrame:
    stage_map = {
        "revenue": "average_annual_revenue_abs_percent_error",
        "eps": "average_eps_abs_error",
        "cash_dividend": "average_cash_dividend_abs_error",
        "yield": "average_yield_mae_percent_point",
    }
    rows = []
    for _, row in overall_accuracy.iterrows():
        for stage, column in stage_map.items():
            rows.append(
                {
                    "source_family": row["source_family"],
                    "model": row["model"],
                    "error_stage": stage,
                    "average_abs_error": row.get(column, np.nan),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    output_dir: Path,
    yield_predictions: pd.DataFrame,
    stock_accuracy: pd.DataFrame,
    overall_accuracy: pd.DataFrame,
    winner_summary: pd.DataFrame,
    error_decomposition: pd.DataFrame,
    failures: pd.DataFrame,
    run_config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    yield_predictions.to_csv(output_dir / "yield_predictions.csv", index=False, encoding="utf-8-sig")
    stock_accuracy.to_csv(output_dir / "yield_stock_accuracy.csv", index=False, encoding="utf-8-sig")
    overall_accuracy.to_csv(output_dir / "yield_overall_accuracy.csv", index=False, encoding="utf-8-sig")
    winner_summary.to_csv(output_dir / "yield_winner_summary.csv", index=False, encoding="utf-8-sig")
    error_decomposition.to_csv(output_dir / "yield_error_decomposition.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(output_dir / "yield_failed_runs.csv", index=False, encoding="utf-8-sig")
    write_run_config_and_registry(output_dir, run_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-predictions", type=Path, default=DEFAULT_INPUT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODEL_NAMES),
        help="Comma-separated model names. Use --all-models to ignore this.",
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


def run_yield_benchmark(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    model_names = None if args.all_models else parse_str_csv(args.models)
    stock_ids = parse_int_csv(args.stock_ids)
    predictions = load_prediction_input(
        args.input_predictions,
        target_year=args.target_year,
        model_names=model_names,
        stock_ids=stock_ids,
        stock_limit=args.stock_limit,
    )
    yield_predictions, failures = build_yield_predictions(
        predictions,
        target_year=args.target_year,
        min_stock_price=args.min_stock_price,
    )
    stock_accuracy = build_yield_stock_accuracy(yield_predictions)
    overall_accuracy = build_yield_overall_accuracy(stock_accuracy)
    winner_summary = build_yield_winner_summary(stock_accuracy)
    error_decomposition = build_error_decomposition(overall_accuracy)
    run_config = {
        "input_predictions": str(args.input_predictions),
        "output_dir": str(args.output_dir),
        "target_year": int(args.target_year),
        "models": model_names,
        "all_models": bool(args.all_models),
        "stock_ids": sorted(int(stock_id) for stock_id in predictions["stock_id"].unique()),
        "stock_count": int(predictions["stock_id"].nunique()),
        "prediction_rows": int(len(predictions)),
        "yield_prediction_rows": int(len(yield_predictions)),
        "min_stock_price": float(args.min_stock_price),
        "failed_runs": int(len(failures)),
    }
    run_config = enrich_run_config_from_args(
        run_config,
        args,
        experiment_family="yield_benchmark",
        extra={"input_predictions": str(args.input_predictions)},
    )
    write_outputs(
        args.output_dir,
        yield_predictions,
        stock_accuracy,
        overall_accuracy,
        winner_summary,
        error_decomposition,
        failures,
        run_config,
    )
    return {
        "yield_predictions": yield_predictions,
        "yield_stock_accuracy": stock_accuracy,
        "yield_overall_accuracy": overall_accuracy,
        "yield_winner_summary": winner_summary,
        "yield_error_decomposition": error_decomposition,
        "yield_failed_runs": failures,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = run_yield_benchmark(args)
    print("Wrote yield benchmark outputs to", args.output_dir)
    print(results["yield_overall_accuracy"].to_string(index=False))


if __name__ == "__main__":
    main()
