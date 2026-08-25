"""Time-safe validation for EPS method selection rules."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_benchmark.benchmark_config import DEFAULT_TARGET_YEAR, PROJECT_ROOT
from forecast_benchmark.eps_benchmark import (
    DEFAULT_EPS_METHODS,
    DEFAULT_INPUT_PREDICTIONS,
    DEFAULT_MODEL_NAMES,
    build_eps_overall_accuracy,
    build_eps_predictions,
    build_eps_stock_accuracy,
    build_historical_annual_frame,
    load_prediction_input,
    load_source_data,
)
from forecast_benchmark.eps_diagnostics import build_ratio_stability, build_stock_metadata
from forecast_benchmark.experiment_registry import (
    add_registry_arguments,
    enrich_run_config_from_args,
    write_run_config_and_registry,
)
from forecast_benchmark.run_benchmark import parse_int_csv, parse_str_csv


DEFAULT_VALIDATION_YEAR = 2024
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forecast_benchmark" / "outputs" / "eps_layer_validation"


def _first_valid(values: pd.Series) -> object:
    valid = values.dropna()
    return valid.iloc[0] if not valid.empty else np.nan


def build_actual_revenue_prediction_rows(
    revenue_data: pd.DataFrame,
    stock_ids: list[int],
    target_year: int,
    stock_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = revenue_data[
        revenue_data["stock_id"].isin(stock_ids) & revenue_data["revenue_year"].eq(int(target_year))
    ].copy()
    if rows.empty:
        return pd.DataFrame()

    rows["source_family"] = "actual_revenue"
    rows["model"] = f"actual_{int(target_year)}_revenue"
    rows["target_year"] = rows["revenue_year"]
    rows["target_month"] = rows["revenue_month"]
    rows["predicted_revenue"] = rows["revenue_thousand"]
    rows["actual_revenue"] = rows["revenue_thousand"]
    rows["last_observed_revenue"] = np.nan
    rows["source_path"] = "data/Stock_revenue_2019~2025.csv"

    metadata_columns = ["stock_id", "stock_name", "industry_category"]
    if stock_metadata is not None and set(metadata_columns).issubset(stock_metadata.columns):
        rows = rows.drop(
            columns=[column for column in ["stock_name", "industry_category"] if column in rows.columns]
        )
        rows = rows.merge(stock_metadata[metadata_columns].drop_duplicates("stock_id"), on="stock_id", how="left")
    else:
        if "stock_name" not in rows.columns:
            rows["stock_name"] = np.nan
        if "industry_category" not in rows.columns:
            rows["industry_category"] = np.nan

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


def select_stock_methods(
    validation_stock_accuracy: pd.DataFrame,
    eps_methods: list[str],
) -> pd.DataFrame:
    candidates = validation_stock_accuracy[
        (~validation_stock_accuracy["is_oracle"].astype(bool))
        & validation_stock_accuracy["eps_method"].isin(eps_methods)
        & validation_stock_accuracy["eps_abs_error"].notna()
    ].copy()
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "stock_id",
                "selected_eps_method",
                "validation_eps_abs_error",
                "validation_eps_abs_percent_error",
            ]
        )

    winners = candidates.loc[candidates.groupby("stock_id")["eps_abs_error"].idxmin()].copy()
    return winners[
        [
            "stock_id",
            "eps_method",
            "eps_abs_error",
            "eps_abs_percent_error",
            "estimated_eps",
            "actual_annual_eps",
        ]
    ].rename(
        columns={
            "eps_method": "selected_eps_method",
            "eps_abs_error": "validation_eps_abs_error",
            "eps_abs_percent_error": "validation_eps_abs_percent_error",
            "estimated_eps": "validation_estimated_eps",
            "actual_annual_eps": "validation_actual_eps",
        }
    )


def score_bucket_methods(
    validation_stock_accuracy: pd.DataFrame,
    validation_ratio_stability: pd.DataFrame,
    eps_methods: list[str],
) -> pd.DataFrame:
    candidates = validation_stock_accuracy[
        (~validation_stock_accuracy["is_oracle"].astype(bool))
        & validation_stock_accuracy["eps_method"].isin(eps_methods)
        & validation_stock_accuracy["eps_abs_error"].notna()
    ].copy()
    candidates = candidates.merge(
        validation_ratio_stability[["stock_id", "ratio_stability_bucket"]],
        on="stock_id",
        how="left",
    )
    scores = (
        candidates.groupby(["ratio_stability_bucket", "eps_method"], as_index=False)
        .agg(
            validation_stock_count=("stock_id", "nunique"),
            average_validation_eps_abs_error=("eps_abs_error", "mean"),
            median_validation_eps_abs_error=("eps_abs_error", "median"),
        )
        .sort_values(["ratio_stability_bucket", "average_validation_eps_abs_error", "eps_method"])
    )
    return scores


def select_bucket_methods(bucket_scores: pd.DataFrame) -> pd.DataFrame:
    if bucket_scores.empty:
        return pd.DataFrame(columns=["ratio_stability_bucket", "selected_eps_method"])

    winners = bucket_scores.loc[
        bucket_scores.groupby("ratio_stability_bucket")["average_validation_eps_abs_error"].idxmin()
    ].copy()
    return winners[
        [
            "ratio_stability_bucket",
            "eps_method",
            "validation_stock_count",
            "average_validation_eps_abs_error",
            "median_validation_eps_abs_error",
        ]
    ].rename(columns={"eps_method": "selected_eps_method"})


def _strategy_rows(
    selected: pd.DataFrame,
    selection_strategy: str,
    selection_source: str,
    is_hindsight_strategy: bool = False,
) -> pd.DataFrame:
    result = selected.copy()
    result["selection_strategy"] = selection_strategy
    result["selected_eps_method"] = result["eps_method"]
    result["selection_source"] = selection_source
    result["is_hindsight_strategy"] = bool(is_hindsight_strategy)
    return result


def build_selected_test_accuracy(
    test_stock_accuracy: pd.DataFrame,
    stock_method_selection: pd.DataFrame,
    bucket_method_selection: pd.DataFrame,
    test_ratio_stability: pd.DataFrame,
    eps_methods: list[str],
) -> pd.DataFrame:
    base = test_stock_accuracy[
        (~test_stock_accuracy["is_oracle"].astype(bool))
        & test_stock_accuracy["eps_method"].isin(eps_methods)
    ].copy()
    frames = []

    for eps_method in eps_methods:
        fixed = base[base["eps_method"].eq(eps_method)].copy()
        frames.append(
            _strategy_rows(
                fixed,
                selection_strategy=f"fixed_{eps_method}",
                selection_source="fixed_method",
            )
        )

    if not stock_method_selection.empty:
        stock_selected = base.merge(
            stock_method_selection[["stock_id", "selected_eps_method"]],
            on="stock_id",
            how="left",
        )
        stock_selected["selected_eps_method"] = stock_selected["selected_eps_method"].fillna("current_ratio")
        stock_selected = stock_selected[
            stock_selected["eps_method"].eq(stock_selected["selected_eps_method"])
        ].copy()
        frames.append(
            _strategy_rows(
                stock_selected,
                selection_strategy="stock_validation_best",
                selection_source="2024_stock_validation",
            )
        )

    if not bucket_method_selection.empty:
        bucket_selected = base.merge(
            test_ratio_stability[["stock_id", "ratio_stability_bucket"]],
            on="stock_id",
            how="left",
        ).merge(
            bucket_method_selection[["ratio_stability_bucket", "selected_eps_method"]],
            on="ratio_stability_bucket",
            how="left",
        )
        bucket_selected["selected_eps_method"] = bucket_selected["selected_eps_method"].fillna("current_ratio")
        bucket_selected = bucket_selected[
            bucket_selected["eps_method"].eq(bucket_selected["selected_eps_method"])
        ].copy()
        frames.append(
            _strategy_rows(
                bucket_selected,
                selection_strategy="ratio_bucket_validation_best",
                selection_source="2024_bucket_validation",
            )
        )

    hindsight = base.dropna(subset=["eps_abs_error"]).copy()
    if not hindsight.empty:
        hindsight = hindsight.loc[
            hindsight.groupby(["source_family", "model", "stock_id"])["eps_abs_error"].idxmin()
        ].copy()
        frames.append(
            _strategy_rows(
                hindsight,
                selection_strategy="oracle_2025_best_method",
                selection_source="2025_actual_eps_hindsight",
                is_hindsight_strategy=True,
            )
        )

    if not frames:
        return pd.DataFrame()

    selected = pd.concat(frames, ignore_index=True)
    return selected.sort_values(
        ["source_family", "model", "selection_strategy", "stock_id"],
        na_position="last",
    )


def build_strategy_overall_accuracy(selected_test_accuracy: pd.DataFrame) -> pd.DataFrame:
    if selected_test_accuracy.empty:
        return pd.DataFrame()

    rows = []
    group_columns = ["source_family", "model", "selection_strategy", "selection_source", "is_hindsight_strategy"]
    metric_columns = [
        "annual_revenue_abs_percent_error",
        "eps_abs_error",
        "eps_abs_percent_error",
    ]
    for group_key, group in selected_test_accuracy.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_key, strict=True))
        valid_eps = group.dropna(subset=["estimated_eps", "actual_annual_eps"])
        row.update(
            {
                "stock_count": int(group["stock_id"].nunique()),
                "valid_eps_stock_count": int(group["eps_abs_error"].notna().sum()),
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
        column
        for column in result.columns
        if column.startswith("average_") or column.startswith("median_") or column.endswith("_rate")
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").round(4)
    return result.sort_values(
        [
            "is_hindsight_strategy",
            "average_eps_abs_error",
            "average_eps_abs_percent_error",
            "source_family",
            "model",
            "selection_strategy",
        ],
        na_position="last",
    )


def build_strategy_winner_summary(
    selected_test_accuracy: pd.DataFrame,
    primary_metric: str = "eps_abs_error",
) -> pd.DataFrame:
    if selected_test_accuracy.empty:
        return pd.DataFrame()

    valid = selected_test_accuracy[
        (~selected_test_accuracy["is_hindsight_strategy"].astype(bool))
        & selected_test_accuracy[primary_metric].notna()
    ].copy()
    if valid.empty:
        return pd.DataFrame()

    group_columns = ["source_family", "model", "selection_strategy"]
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
    for column in ["average_primary_metric", "median_primary_metric", "stock_win_rate"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").round(4)
    return summary.sort_values(
        ["stock_wins", "average_primary_metric", "source_family", "model", "selection_strategy"],
        ascending=[False, True, True, True, True],
    )


def build_improvement_vs_current(strategy_overall_accuracy: pd.DataFrame) -> pd.DataFrame:
    if strategy_overall_accuracy.empty:
        return pd.DataFrame()

    baseline = strategy_overall_accuracy[
        strategy_overall_accuracy["selection_strategy"].eq("fixed_current_ratio")
    ][["source_family", "model", "average_eps_abs_error", "median_eps_abs_error"]].rename(
        columns={
            "average_eps_abs_error": "baseline_average_eps_abs_error",
            "median_eps_abs_error": "baseline_median_eps_abs_error",
        }
    )
    compared = strategy_overall_accuracy.merge(baseline, on=["source_family", "model"], how="left")
    compared["average_eps_abs_error_delta_vs_current"] = (
        compared["baseline_average_eps_abs_error"] - compared["average_eps_abs_error"]
    )
    compared["average_eps_abs_error_improvement_pct_vs_current"] = np.where(
        compared["baseline_average_eps_abs_error"] > 0,
        compared["average_eps_abs_error_delta_vs_current"] / compared["baseline_average_eps_abs_error"] * 100,
        np.nan,
    )
    numeric_columns = [
        "baseline_average_eps_abs_error",
        "baseline_median_eps_abs_error",
        "average_eps_abs_error_delta_vs_current",
        "average_eps_abs_error_improvement_pct_vs_current",
    ]
    for column in numeric_columns:
        compared[column] = pd.to_numeric(compared[column], errors="coerce").round(4)
    return compared.sort_values(
        [
            "is_hindsight_strategy",
            "average_eps_abs_error_improvement_pct_vs_current",
            "average_eps_abs_error",
        ],
        ascending=[True, False, True],
        na_position="last",
    )


def write_outputs(
    output_dir: Path,
    validation_stock_accuracy: pd.DataFrame,
    validation_ratio_stability: pd.DataFrame,
    validation_bucket_method_scores: pd.DataFrame,
    validation_bucket_method_selection: pd.DataFrame,
    validation_stock_method_selection: pd.DataFrame,
    test_all_method_stock_accuracy: pd.DataFrame,
    test_ratio_stability: pd.DataFrame,
    test_selected_stock_accuracy: pd.DataFrame,
    test_strategy_overall_accuracy: pd.DataFrame,
    test_strategy_winner_summary: pd.DataFrame,
    test_strategy_improvement_vs_current: pd.DataFrame,
    run_config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_stock_accuracy.to_csv(
        output_dir / "validation_eps_stock_accuracy.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_ratio_stability.to_csv(
        output_dir / "validation_ratio_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_bucket_method_scores.to_csv(
        output_dir / "validation_bucket_method_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_bucket_method_selection.to_csv(
        output_dir / "validation_bucket_method_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_stock_method_selection.to_csv(
        output_dir / "validation_stock_method_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_all_method_stock_accuracy.to_csv(
        output_dir / "test_all_method_stock_accuracy.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_ratio_stability.to_csv(
        output_dir / "test_ratio_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_selected_stock_accuracy.to_csv(
        output_dir / "test_selected_stock_accuracy.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_strategy_overall_accuracy.to_csv(
        output_dir / "test_strategy_overall_accuracy.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_strategy_winner_summary.to_csv(
        output_dir / "test_strategy_winner_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_strategy_improvement_vs_current.to_csv(
        output_dir / "test_strategy_improvement_vs_current.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_run_config_and_registry(output_dir, run_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-predictions", type=Path, default=DEFAULT_INPUT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR)
    parser.add_argument("--validation-year", type=int, default=DEFAULT_VALIDATION_YEAR)
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
    return add_registry_arguments(parser)


def run_eps_layer_validation(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    model_names = None if args.all_models else parse_str_csv(args.models)
    eps_methods = parse_str_csv(args.eps_methods) or list(DEFAULT_EPS_METHODS)
    stock_ids = parse_int_csv(args.stock_ids)
    test_predictions = load_prediction_input(
        args.input_predictions,
        target_year=args.target_year,
        model_names=model_names,
        stock_ids=stock_ids,
        stock_limit=args.stock_limit,
    )
    selected_stock_ids = sorted(int(stock_id) for stock_id in test_predictions["stock_id"].unique())

    revenue_data, eps = load_source_data()
    metadata = (
        test_predictions.groupby("stock_id", as_index=False)
        .agg(stock_name=("stock_name", _first_valid), industry_category=("industry_category", _first_valid))
    )
    validation_predictions = build_actual_revenue_prediction_rows(
        revenue_data,
        selected_stock_ids,
        target_year=args.validation_year,
        stock_metadata=metadata,
    )
    validation_eps_predictions, validation_failures = build_eps_predictions(
        validation_predictions,
        target_year=args.validation_year,
        eps_methods=eps_methods,
        include_oracle=False,
    )
    validation_stock_accuracy = build_eps_stock_accuracy(validation_eps_predictions)
    validation_overall_accuracy = build_eps_overall_accuracy(validation_stock_accuracy)

    validation_history = build_historical_annual_frame(revenue_data, eps, args.validation_year)
    validation_stock_metadata = build_stock_metadata(validation_stock_accuracy)
    validation_ratio_stability = build_ratio_stability(validation_history, validation_stock_metadata)
    validation_stock_method_selection = select_stock_methods(validation_stock_accuracy, eps_methods)
    validation_bucket_method_scores = score_bucket_methods(
        validation_stock_accuracy,
        validation_ratio_stability,
        eps_methods,
    )
    validation_bucket_method_selection = select_bucket_methods(validation_bucket_method_scores)

    test_eps_predictions, test_failures = build_eps_predictions(
        test_predictions,
        target_year=args.target_year,
        eps_methods=eps_methods,
        include_oracle=False,
    )
    test_all_method_stock_accuracy = build_eps_stock_accuracy(test_eps_predictions)
    test_history = build_historical_annual_frame(revenue_data, eps, args.target_year)
    test_stock_metadata = build_stock_metadata(test_all_method_stock_accuracy)
    test_ratio_stability = build_ratio_stability(test_history, test_stock_metadata)

    test_selected_stock_accuracy = build_selected_test_accuracy(
        test_all_method_stock_accuracy,
        validation_stock_method_selection,
        validation_bucket_method_selection,
        test_ratio_stability,
        eps_methods,
    )
    test_strategy_overall_accuracy = build_strategy_overall_accuracy(test_selected_stock_accuracy)
    test_strategy_winner_summary = build_strategy_winner_summary(test_selected_stock_accuracy)
    test_strategy_improvement_vs_current = build_improvement_vs_current(test_strategy_overall_accuracy)

    run_config = {
        "input_predictions": str(args.input_predictions),
        "output_dir": str(args.output_dir),
        "target_year": int(args.target_year),
        "validation_year": int(args.validation_year),
        "models": model_names,
        "eps_methods": eps_methods,
        "stock_ids": selected_stock_ids,
        "stock_count": int(len(selected_stock_ids)),
        "validation_prediction_rows": int(len(validation_predictions)),
        "validation_eps_prediction_rows": int(len(validation_eps_predictions)),
        "test_prediction_rows": int(len(test_predictions)),
        "test_eps_prediction_rows": int(len(test_eps_predictions)),
        "validation_failed_runs": int(len(validation_failures)),
        "test_failed_runs": int(len(test_failures)),
    }
    run_config = enrich_run_config_from_args(
        run_config,
        args,
        experiment_family="eps_layer_validation",
        extra={"input_predictions": str(args.input_predictions)},
    )
    write_outputs(
        args.output_dir,
        validation_stock_accuracy,
        validation_ratio_stability,
        validation_bucket_method_scores,
        validation_bucket_method_selection,
        validation_stock_method_selection,
        test_all_method_stock_accuracy,
        test_ratio_stability,
        test_selected_stock_accuracy,
        test_strategy_overall_accuracy,
        test_strategy_winner_summary,
        test_strategy_improvement_vs_current,
        run_config,
    )
    return {
        "validation_eps_stock_accuracy": validation_stock_accuracy,
        "validation_eps_overall_accuracy": validation_overall_accuracy,
        "validation_ratio_stability": validation_ratio_stability,
        "validation_bucket_method_scores": validation_bucket_method_scores,
        "validation_bucket_method_selection": validation_bucket_method_selection,
        "validation_stock_method_selection": validation_stock_method_selection,
        "test_all_method_stock_accuracy": test_all_method_stock_accuracy,
        "test_ratio_stability": test_ratio_stability,
        "test_selected_stock_accuracy": test_selected_stock_accuracy,
        "test_strategy_overall_accuracy": test_strategy_overall_accuracy,
        "test_strategy_winner_summary": test_strategy_winner_summary,
        "test_strategy_improvement_vs_current": test_strategy_improvement_vs_current,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = run_eps_layer_validation(args)
    print("Wrote EPS layer validation outputs to", args.output_dir)
    print("\nValidation bucket method selection:")
    print(results["validation_bucket_method_selection"].to_string(index=False))
    print("\n2025 test strategy overall accuracy:")
    print(results["test_strategy_overall_accuracy"].to_string(index=False))


if __name__ == "__main__":
    main()
