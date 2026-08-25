"""Benchmark dividend-yield results from alternative EPS layers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_benchmark.adapters.ensemble_evidence import (
    get_actual_cash_dividend_info,
    get_dividend_policy,
    get_historical_payout_ratio,
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


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forecast_benchmark" / "outputs" / "yield_eps_layer_benchmark"
DEFAULT_EPS_METHODS = ("current_ratio", "lasso_annual", "elastic_net_annual")


def _round_numeric_columns(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").round(digits)
    return result


def build_payout_ratio_lookup(stock_ids: list[int], target_year: int) -> pd.DataFrame:
    rows = []
    for stock_id in stock_ids:
        payout_ratio, payout_source = get_historical_payout_ratio(int(stock_id), int(target_year))
        if pd.isna(payout_ratio):
            policy = get_dividend_policy(int(stock_id))
            payout_ratio = policy["payout_ratio"]
            payout_source = "dividend_policy.csv"
        rows.append(
            {
                "stock_id": int(stock_id),
                "payout_ratio": payout_ratio,
                "payout_ratio_source": payout_source,
            }
        )
    return pd.DataFrame(rows)


def build_actual_cash_dividend_lookup(stock_ids: list[int], target_year: int) -> pd.DataFrame:
    rows = []
    for stock_id in stock_ids:
        actual = get_actual_cash_dividend_info(int(stock_id), int(target_year))
        rows.append(
            {
                "stock_id": int(stock_id),
                "actual_cash_dividend_per_share": actual["actual_cash_dividend_per_share"],
                "actual_cash_dividend_source": actual["actual_cash_dividend_source"],
            }
        )
    return pd.DataFrame(rows)


def build_yield_eps_layer_predictions(
    predictions: pd.DataFrame,
    target_year: int,
    eps_methods: list[str] | None = None,
    min_stock_price: float = DEFAULT_MIN_STOCK_PRICE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eps_methods = list(eps_methods or DEFAULT_EPS_METHODS)
    revenue_data = load_revenue_data()
    stock_ids = sorted(int(stock_id) for stock_id in predictions["stock_id"].dropna().unique())
    prices = build_stock_price_lookup(revenue_data, stock_ids, target_year)
    payout_ratios = build_payout_ratio_lookup(stock_ids, target_year)
    actual_dividends = build_actual_cash_dividend_lookup(stock_ids, target_year)
    eps_predictions, eps_failures = build_eps_predictions(
        predictions,
        target_year=target_year,
        eps_methods=eps_methods,
        include_oracle=False,
    )
    eps_predictions = eps_predictions[~eps_predictions["is_oracle"].astype(bool)].copy()
    eps_columns = [
        "source_family",
        "model",
        "stock_id",
        "eps_method",
        "predicted_annual_revenue",
        "actual_annual_revenue",
        "annual_revenue_abs_percent_error",
        "estimated_eps",
        "actual_annual_eps",
        "actual_eps_quarter_count",
        "eps_error",
        "eps_abs_error",
        "eps_abs_percent_error",
        "eps_reference_year",
        "eps_to_revenue_ratio",
        "eps_transform_source",
    ]
    monthly = predictions.merge(
        eps_predictions[eps_columns],
        on=["source_family", "model", "stock_id"],
        how="inner",
    )
    monthly = monthly.merge(
        prices[["stock_id", "price_month", "price_date", "close_price", "price_source"]],
        left_on=["stock_id", "target_month"],
        right_on=["stock_id", "price_month"],
        how="left",
    )
    monthly = monthly.merge(payout_ratios, on="stock_id", how="left")
    monthly = monthly.merge(actual_dividends, on="stock_id", how="left")
    monthly["estimated_cash_dividend"] = monthly["estimated_eps"] * monthly["payout_ratio"]
    monthly["estimated_cash_dividend"] = np.where(
        monthly["estimated_cash_dividend"].notna(),
        np.maximum(monthly["estimated_cash_dividend"], 0.0),
        np.nan,
    )
    monthly["cash_dividend_source"] = monthly["eps_transform_source"].fillna("") + "; payout=" + monthly[
        "payout_ratio_source"
    ].fillna("")
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
    monthly["cash_dividend_error"] = (
        monthly["estimated_cash_dividend"] - monthly["actual_cash_dividend_per_share"]
    )
    monthly["cash_dividend_abs_error"] = monthly["cash_dividend_error"].abs()
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
    failures = eps_failures.copy()
    return monthly.sort_values(["stock_id", "source_family", "model", "eps_method", "target_month"]), failures


def build_yield_eps_layer_stock_accuracy(yield_predictions: pd.DataFrame) -> pd.DataFrame:
    if yield_predictions.empty:
        return pd.DataFrame()

    rows = []
    group_columns = ["source_family", "model", "eps_method", "stock_id", "stock_name", "industry_category"]
    for group_key, group in yield_predictions.groupby(group_columns, dropna=False):
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
    result = _round_numeric_columns(result, numeric_columns)
    return result.sort_values(
        ["stock_id", "yield_mae_percent_point", "source_family", "model", "eps_method"],
        na_position="last",
    )


def build_yield_eps_layer_overall_accuracy(stock_accuracy: pd.DataFrame) -> pd.DataFrame:
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
    group_columns = ["source_family", "model", "eps_method"]
    for group_key, group in stock_accuracy.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_key, strict=True))
        row.update(
            {
                "stock_count": int(group["stock_id"].nunique()),
                "valid_revenue_stock_count": int(group["annual_revenue_abs_percent_error"].notna().sum()),
                "valid_eps_stock_count": int(group["eps_abs_error"].notna().sum()),
                "valid_cash_dividend_stock_count": int(group["cash_dividend_abs_error"].notna().sum()),
                "valid_yield_stock_count": int(group["yield_mae_percent_point"].notna().sum()),
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
            "average_eps_abs_error",
            "source_family",
            "model",
            "eps_method",
        ],
        na_position="last",
    )


def build_yield_eps_layer_winner_summary(
    stock_accuracy: pd.DataFrame,
    primary_metric: str = "yield_mae_percent_point",
) -> pd.DataFrame:
    if stock_accuracy.empty:
        return pd.DataFrame()
    if primary_metric not in stock_accuracy.columns:
        raise ValueError(f"Unknown primary metric: {primary_metric}")

    valid = stock_accuracy.dropna(subset=[primary_metric]).copy()
    if valid.empty:
        return pd.DataFrame()

    group_columns = ["source_family", "model", "eps_method"]
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
        ["stock_wins", "average_primary_metric", "source_family", "model", "eps_method"],
        ascending=[False, True, True, True, True],
    )


def build_yield_eps_layer_improvement_vs_current(overall_accuracy: pd.DataFrame) -> pd.DataFrame:
    if overall_accuracy.empty:
        return pd.DataFrame()

    baseline = overall_accuracy[
        overall_accuracy["eps_method"].eq("current_ratio")
    ][
        [
            "source_family",
            "model",
            "average_eps_abs_error",
            "average_cash_dividend_abs_error",
            "average_yield_mae_percent_point",
            "median_yield_mae_percent_point",
        ]
    ].rename(
        columns={
            "average_eps_abs_error": "baseline_average_eps_abs_error",
            "average_cash_dividend_abs_error": "baseline_average_cash_dividend_abs_error",
            "average_yield_mae_percent_point": "baseline_average_yield_mae_percent_point",
            "median_yield_mae_percent_point": "baseline_median_yield_mae_percent_point",
        }
    )
    compared = overall_accuracy.merge(baseline, on=["source_family", "model"], how="left")
    compared["average_yield_mae_delta_vs_current"] = (
        compared["baseline_average_yield_mae_percent_point"]
        - compared["average_yield_mae_percent_point"]
    )
    compared["average_yield_mae_improvement_pct_vs_current"] = np.where(
        compared["baseline_average_yield_mae_percent_point"] > 0,
        compared["average_yield_mae_delta_vs_current"]
        / compared["baseline_average_yield_mae_percent_point"]
        * 100,
        np.nan,
    )
    compared["average_eps_abs_error_delta_vs_current"] = (
        compared["baseline_average_eps_abs_error"] - compared["average_eps_abs_error"]
    )
    numeric_columns = [
        "baseline_average_eps_abs_error",
        "baseline_average_cash_dividend_abs_error",
        "baseline_average_yield_mae_percent_point",
        "baseline_median_yield_mae_percent_point",
        "average_yield_mae_delta_vs_current",
        "average_yield_mae_improvement_pct_vs_current",
        "average_eps_abs_error_delta_vs_current",
    ]
    compared = _round_numeric_columns(compared, numeric_columns)
    return compared.sort_values(
        ["average_yield_mae_improvement_pct_vs_current", "average_yield_mae_percent_point"],
        ascending=[False, True],
        na_position="last",
    )


def build_yield_eps_layer_error_decomposition(overall_accuracy: pd.DataFrame) -> pd.DataFrame:
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
                    "eps_method": row["eps_method"],
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
    improvement_vs_current: pd.DataFrame,
    error_decomposition: pd.DataFrame,
    failures: pd.DataFrame,
    run_config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    yield_predictions.to_csv(output_dir / "yield_eps_layer_predictions.csv", index=False, encoding="utf-8-sig")
    stock_accuracy.to_csv(output_dir / "yield_eps_layer_stock_accuracy.csv", index=False, encoding="utf-8-sig")
    overall_accuracy.to_csv(output_dir / "yield_eps_layer_overall_accuracy.csv", index=False, encoding="utf-8-sig")
    winner_summary.to_csv(output_dir / "yield_eps_layer_winner_summary.csv", index=False, encoding="utf-8-sig")
    improvement_vs_current.to_csv(
        output_dir / "yield_eps_layer_improvement_vs_current.csv",
        index=False,
        encoding="utf-8-sig",
    )
    error_decomposition.to_csv(
        output_dir / "yield_eps_layer_error_decomposition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    failures.to_csv(output_dir / "yield_eps_layer_failed_runs.csv", index=False, encoding="utf-8-sig")
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
    parser.add_argument(
        "--min-stock-price",
        type=float,
        default=DEFAULT_MIN_STOCK_PRICE,
        help="Exclude monthly prices at or below this value from yield-error metrics.",
    )
    return add_registry_arguments(parser)


def run_yield_eps_layer_benchmark(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
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
    yield_predictions, failures = build_yield_eps_layer_predictions(
        predictions,
        target_year=args.target_year,
        eps_methods=eps_methods,
        min_stock_price=args.min_stock_price,
    )
    stock_accuracy = build_yield_eps_layer_stock_accuracy(yield_predictions)
    overall_accuracy = build_yield_eps_layer_overall_accuracy(stock_accuracy)
    winner_summary = build_yield_eps_layer_winner_summary(stock_accuracy)
    improvement_vs_current = build_yield_eps_layer_improvement_vs_current(overall_accuracy)
    error_decomposition = build_yield_eps_layer_error_decomposition(overall_accuracy)
    run_config = {
        "input_predictions": str(args.input_predictions),
        "output_dir": str(args.output_dir),
        "target_year": int(args.target_year),
        "models": model_names,
        "eps_methods": eps_methods,
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
        experiment_family="yield_eps_layer_benchmark",
        extra={"input_predictions": str(args.input_predictions)},
    )
    write_outputs(
        args.output_dir,
        yield_predictions,
        stock_accuracy,
        overall_accuracy,
        winner_summary,
        improvement_vs_current,
        error_decomposition,
        failures,
        run_config,
    )
    return {
        "yield_eps_layer_predictions": yield_predictions,
        "yield_eps_layer_stock_accuracy": stock_accuracy,
        "yield_eps_layer_overall_accuracy": overall_accuracy,
        "yield_eps_layer_winner_summary": winner_summary,
        "yield_eps_layer_improvement_vs_current": improvement_vs_current,
        "yield_eps_layer_error_decomposition": error_decomposition,
        "yield_eps_layer_failed_runs": failures,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = run_yield_eps_layer_benchmark(args)
    print("Wrote yield EPS-layer benchmark outputs to", args.output_dir)
    print(results["yield_eps_layer_overall_accuracy"].to_string(index=False))
    print("\nImprovement vs current_ratio:")
    print(results["yield_eps_layer_improvement_vs_current"].to_string(index=False))


if __name__ == "__main__":
    main()
