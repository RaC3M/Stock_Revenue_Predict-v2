"""Diagnose direct dividend model errors from direct dividend benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_benchmark.benchmark_config import DEFAULT_TARGET_YEAR, PROJECT_ROOT
from forecast_benchmark.direct_dividend_model_benchmark import (
    SELECTED_BUCKET_DIVIDEND_METHOD,
)
from forecast_benchmark.experiment_registry import (
    add_registry_arguments,
    enrich_run_config_from_args,
    write_run_config_and_registry,
)


DEFAULT_DIRECT_BENCHMARK_DIR = PROJECT_ROOT / "forecast_benchmark" / "outputs" / "direct_dividend_model_benchmark"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forecast_benchmark" / "outputs" / "direct_dividend_error_diagnostics"
DEFAULT_DIRECT_SOURCE_FAMILY = "ensemble_forecast"
DEFAULT_DIRECT_MODEL = "LightGBM"
DEFAULT_DIRECT_EPS_METHOD = "time_safe_features"
DEFAULT_BASELINE_SOURCE_FAMILY = "ensemble_forecast"
DEFAULT_BASELINE_MODEL = "LightGBM"
DEFAULT_BASELINE_EPS_METHOD = "current_ratio"
DEFAULT_BASELINE_DIVIDEND_METHOD = "announcement_safe_payout_ratio"
NUMERIC_COLUMNS = [
    "stock_id",
    "target_year",
    "estimated_cash_dividend",
    "actual_cash_dividend_per_share",
    "cash_dividend_abs_error",
    "yield_mae_percent_point",
    "yield_abs_error_percent_point",
    "predicted_dividend_paid_probability",
    "validation_fold_count",
    "validation_stock_year_count",
    "validation_primary_metric",
]


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    text = series.astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(series, errors="coerce")
    return text.isin({"true", "1", "1.0", "yes"}) | numeric.fillna(0).ne(0)


def _round_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].round(4)
    return result


def _coerce_common_types(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in [
        "actual_dividend_paid",
        "predicted_dividend_paid",
        "fallback_to_global",
        "has_known_dividend_data",
        "stock_price_valid_for_yield",
    ]:
        if column in result.columns:
            result[column] = normalize_bool(result[column])
    if "stock_id" in result.columns:
        result = result.dropna(subset=["stock_id"]).copy()
        result["stock_id"] = result["stock_id"].astype(int)
    return result


def _read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return _coerce_common_types(frame)


def load_direct_benchmark_outputs(benchmark_dir: Path) -> dict[str, pd.DataFrame]:
    benchmark_dir = Path(benchmark_dir)
    return {
        "selected_stock_accuracy": _read_csv(
            benchmark_dir / "direct_dividend_selected_test_stock_accuracy.csv",
            {"stock_id", "dividend_method", "cash_dividend_abs_error", "yield_mae_percent_point"},
        ),
        "baseline_stock_accuracy": _read_csv(
            benchmark_dir / "direct_dividend_baseline_stock_accuracy.csv",
            {
                "stock_id",
                "source_family",
                "model",
                "eps_method",
                "dividend_method",
                "cash_dividend_abs_error",
                "yield_mae_percent_point",
            },
        ),
        "selected_predictions": _read_csv(
            benchmark_dir / "direct_dividend_selected_test_predictions.csv",
            {
                "stock_id",
                "dividend_method",
                "target_month",
                "estimated_cash_dividend",
                "actual_cash_dividend_per_share",
                "predicted_dividend_paid",
                "actual_dividend_paid",
            },
        ),
    }


def _filter_rows(frame: pd.DataFrame, filters: dict[str, str | None]) -> pd.DataFrame:
    result = frame.copy()
    for column, value in filters.items():
        if value is not None and column in result.columns:
            result = result[result[column].astype(str).eq(str(value))]
    return result


def _canonical_stock_rows(
    frame: pd.DataFrame,
    filters: dict[str, str | None],
    fallback_filters: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    filtered = _filter_rows(frame, filters)
    if filtered.empty and fallback_filters:
        filtered = _filter_rows(frame, fallback_filters)
    if filtered.empty:
        return pd.DataFrame()
    sort_columns = [
        column
        for column in ["source_family", "model", "eps_method", "dividend_method", "stock_id"]
        if column in filtered.columns
    ]
    return filtered.sort_values(sort_columns).drop_duplicates("stock_id", keep="first").copy()


def _prefixed_columns(
    frame: pd.DataFrame,
    prefix: str,
    columns: list[str],
) -> pd.DataFrame:
    if frame.empty or "stock_id" not in frame.columns:
        return pd.DataFrame(columns=["stock_id", *[f"{prefix}_{column}" for column in columns]])
    available = [column for column in columns if column in frame.columns]
    result = frame[["stock_id", *available]].copy()
    return result.rename(columns={column: f"{prefix}_{column}" for column in available})


def build_stock_error_comparison(
    selected_stock_accuracy: pd.DataFrame,
    baseline_stock_accuracy: pd.DataFrame,
    direct_global_method: str,
    direct_source_family: str = DEFAULT_DIRECT_SOURCE_FAMILY,
    direct_model: str = DEFAULT_DIRECT_MODEL,
    direct_eps_method: str = DEFAULT_DIRECT_EPS_METHOD,
    baseline_source_family: str = DEFAULT_BASELINE_SOURCE_FAMILY,
    baseline_model: str = DEFAULT_BASELINE_MODEL,
    baseline_eps_method: str = DEFAULT_BASELINE_EPS_METHOD,
    baseline_dividend_method: str = DEFAULT_BASELINE_DIVIDEND_METHOD,
) -> pd.DataFrame:
    direct_common_filters = {
        "source_family": direct_source_family,
        "model": direct_model,
        "eps_method": direct_eps_method,
    }
    bucket = _canonical_stock_rows(
        selected_stock_accuracy,
        {**direct_common_filters, "dividend_method": SELECTED_BUCKET_DIVIDEND_METHOD},
        {"dividend_method": SELECTED_BUCKET_DIVIDEND_METHOD},
    )
    global_direct = _canonical_stock_rows(
        selected_stock_accuracy,
        {**direct_common_filters, "dividend_method": direct_global_method},
        {"dividend_method": direct_global_method},
    )
    baseline = _canonical_stock_rows(
        baseline_stock_accuracy,
        {
            "source_family": baseline_source_family,
            "model": baseline_model,
            "eps_method": baseline_eps_method,
            "dividend_method": baseline_dividend_method,
        },
    )
    if bucket.empty:
        return pd.DataFrame()

    metadata_columns = [
        "stock_name",
        "industry_category",
        "target_year",
        "actual_cash_dividend_per_share",
        "dividend_selection_bucket",
        "paid_rate_bucket",
        "dividend_history_bucket",
        "latest_dividend_bucket",
        "bucket_support_status",
        "fallback_to_global",
        "selected_underlying_dividend_method",
        "bucket_winner_dividend_method",
        "validation_fold_count",
        "validation_stock_year_count",
        "validation_primary_metric",
    ]
    comparison = bucket[["stock_id", *[column for column in metadata_columns if column in bucket.columns]]].copy()
    metric_columns = [
        "estimated_cash_dividend",
        "cash_dividend_abs_error",
        "yield_mae_percent_point",
        "mean_predicted_yield_percent",
        "mean_actual_yield_percent",
    ]
    comparison = comparison.merge(
        _prefixed_columns(bucket, "bucket", metric_columns),
        on="stock_id",
        how="left",
    )
    comparison = comparison.merge(
        _prefixed_columns(global_direct, "global", metric_columns),
        on="stock_id",
        how="left",
    )
    comparison = comparison.merge(
        _prefixed_columns(baseline, "baseline", metric_columns + ["eps_abs_error"]),
        on="stock_id",
        how="left",
    )
    comparison["bucket_cash_improvement_vs_baseline"] = (
        comparison["baseline_cash_dividend_abs_error"] - comparison["bucket_cash_dividend_abs_error"]
    )
    comparison["bucket_yield_improvement_vs_baseline"] = (
        comparison["baseline_yield_mae_percent_point"] - comparison["bucket_yield_mae_percent_point"]
    )
    comparison["bucket_cash_improvement_vs_global"] = (
        comparison["global_cash_dividend_abs_error"] - comparison["bucket_cash_dividend_abs_error"]
    )
    comparison["bucket_yield_improvement_vs_global"] = (
        comparison["global_yield_mae_percent_point"] - comparison["bucket_yield_mae_percent_point"]
    )
    has_cash_comparison = (
        comparison["bucket_cash_dividend_abs_error"].notna()
        & comparison["baseline_cash_dividend_abs_error"].notna()
    )
    comparison["bucket_vs_baseline_cash_result"] = np.select(
        [
            ~has_cash_comparison,
            comparison["bucket_cash_improvement_vs_baseline"].gt(1e-9),
            comparison["bucket_cash_improvement_vs_baseline"].lt(-1e-9),
        ],
        ["missing_metric", "improved", "worse"],
        default="tied",
    )
    return _round_numeric_columns(
        comparison.sort_values(["bucket_cash_improvement_vs_baseline", "stock_id"], ascending=[False, True])
    )


def build_improvement_leaders(comparison: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    leaders = comparison[comparison["bucket_cash_improvement_vs_baseline"].gt(0)].copy()
    return leaders.sort_values(
        ["bucket_cash_improvement_vs_baseline", "bucket_yield_improvement_vs_baseline"],
        ascending=[False, False],
    ).head(int(top_n))


def build_regression_hotspots(comparison: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    regressions = comparison[comparison["bucket_cash_improvement_vs_baseline"].lt(0)].copy()
    return regressions.sort_values(
        ["bucket_cash_improvement_vs_baseline", "bucket_yield_improvement_vs_baseline"],
        ascending=[True, True],
    ).head(int(top_n))


def build_bucket_error_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty or "dividend_selection_bucket" not in comparison.columns:
        return pd.DataFrame()
    rows = []
    for bucket, group in comparison.groupby("dividend_selection_bucket", dropna=False):
        fallback = normalize_bool(group.get("fallback_to_global", pd.Series(False, index=group.index)))
        rows.append(
            {
                "dividend_selection_bucket": bucket,
                "stock_count": int(group["stock_id"].nunique()),
                "valid_cash_stock_count": int(
                    (
                        group["bucket_cash_dividend_abs_error"].notna()
                        & group["baseline_cash_dividend_abs_error"].notna()
                    ).sum()
                ),
                "missing_metric_stock_count": int(group["bucket_vs_baseline_cash_result"].eq("missing_metric").sum()),
                "fallback_stock_count": int(fallback.sum()),
                "supported_stock_count": int((~fallback).sum()),
                "average_bucket_cash_dividend_abs_error": group["bucket_cash_dividend_abs_error"].mean(),
                "average_baseline_cash_dividend_abs_error": group["baseline_cash_dividend_abs_error"].mean(),
                "average_cash_improvement_vs_baseline": group["bucket_cash_improvement_vs_baseline"].mean(),
                "average_bucket_yield_mae_percent_point": group["bucket_yield_mae_percent_point"].mean(),
                "average_baseline_yield_mae_percent_point": group["baseline_yield_mae_percent_point"].mean(),
                "average_yield_improvement_vs_baseline": group["bucket_yield_improvement_vs_baseline"].mean(),
                "improved_stock_count": int(group["bucket_vs_baseline_cash_result"].eq("improved").sum()),
                "worse_stock_count": int(group["bucket_vs_baseline_cash_result"].eq("worse").sum()),
            }
        )
    return _round_numeric_columns(
        pd.DataFrame(rows).sort_values(["average_cash_improvement_vs_baseline", "stock_count"], ascending=[False, False])
    )


def classify_dividend_outcome(actual_paid: bool, predicted_paid: bool) -> str:
    if actual_paid and predicted_paid:
        return "correct_paid"
    if actual_paid and not predicted_paid:
        return "false_negative_missed_dividend"
    if not actual_paid and predicted_paid:
        return "false_positive_extra_dividend"
    return "correct_no_dividend"


def build_classification_outcomes(
    selected_predictions: pd.DataFrame,
    direct_source_family: str = DEFAULT_DIRECT_SOURCE_FAMILY,
    direct_model: str = DEFAULT_DIRECT_MODEL,
    direct_eps_method: str = DEFAULT_DIRECT_EPS_METHOD,
) -> pd.DataFrame:
    bucket_predictions = _canonical_stock_rows(
        selected_predictions,
        {
            "source_family": direct_source_family,
            "model": direct_model,
            "eps_method": direct_eps_method,
            "dividend_method": SELECTED_BUCKET_DIVIDEND_METHOD,
        },
        {"dividend_method": SELECTED_BUCKET_DIVIDEND_METHOD},
    )
    if bucket_predictions.empty:
        return pd.DataFrame()
    rows = bucket_predictions.copy()
    rows["classification_outcome"] = [
        classify_dividend_outcome(bool(actual), bool(predicted))
        for actual, predicted in zip(rows["actual_dividend_paid"], rows["predicted_dividend_paid"], strict=True)
    ]
    columns = [
        "stock_id",
        "stock_name",
        "industry_category",
        "target_year",
        "dividend_selection_bucket",
        "bucket_support_status",
        "fallback_to_global",
        "selected_underlying_dividend_method",
        "bucket_winner_dividend_method",
        "predicted_dividend_paid_probability",
        "predicted_dividend_paid",
        "actual_dividend_paid",
        "classification_outcome",
        "estimated_cash_dividend",
        "actual_cash_dividend_per_share",
        "cash_dividend_abs_error",
        "predicted_dividend_yield_percent",
        "actual_dividend_yield_percent",
        "yield_abs_error_percent_point",
    ]
    available = [column for column in columns if column in rows.columns]
    return _round_numeric_columns(rows[available].sort_values(["classification_outcome", "stock_id"]))


def build_classification_summary(classification_outcomes: pd.DataFrame) -> pd.DataFrame:
    if classification_outcomes.empty:
        return pd.DataFrame()
    total = classification_outcomes["stock_id"].nunique()
    rows = []
    for outcome, group in classification_outcomes.groupby("classification_outcome", dropna=False):
        rows.append(
            {
                "classification_outcome": outcome,
                "stock_count": int(group["stock_id"].nunique()),
                "share_percent": group["stock_id"].nunique() / total * 100 if total else np.nan,
                "average_cash_dividend_abs_error": group["cash_dividend_abs_error"].mean()
                if "cash_dividend_abs_error" in group.columns
                else np.nan,
                "average_yield_abs_error_percent_point": group["yield_abs_error_percent_point"].mean()
                if "yield_abs_error_percent_point" in group.columns
                else np.nan,
            }
        )
    return _round_numeric_columns(pd.DataFrame(rows).sort_values("stock_count", ascending=False))


def build_classification_errors(classification_outcomes: pd.DataFrame) -> pd.DataFrame:
    if classification_outcomes.empty:
        return pd.DataFrame()
    errors = classification_outcomes[
        classification_outcomes["classification_outcome"].isin(
            ["false_negative_missed_dividend", "false_positive_extra_dividend"]
        )
    ].copy()
    return errors.sort_values(["cash_dividend_abs_error", "stock_id"], ascending=[False, True])


def build_amount_error_hotspots(classification_outcomes: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if classification_outcomes.empty:
        return pd.DataFrame()
    true_positive = classification_outcomes[
        classification_outcomes["classification_outcome"].eq("correct_paid")
    ].copy()
    if true_positive.empty:
        return pd.DataFrame()
    true_positive["cash_dividend_error"] = (
        true_positive["estimated_cash_dividend"] - true_positive["actual_cash_dividend_per_share"]
    )
    true_positive["cash_dividend_underestimate"] = true_positive["cash_dividend_error"].lt(0)
    return _round_numeric_columns(
        true_positive.sort_values(["cash_dividend_abs_error", "stock_id"], ascending=[False, True]).head(int(top_n))
    )


def build_diagnostic_summary(
    comparison: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    classification_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    total = int(comparison["stock_id"].nunique()) if not comparison.empty else 0
    rows.append(
        {
            "summary_type": "stock_count",
            "bucket": "all_analyzed_stocks",
            "count": total,
            "share_percent": 100.0 if total else np.nan,
            "average_cash_improvement_vs_baseline": comparison["bucket_cash_improvement_vs_baseline"].mean()
            if not comparison.empty
            else np.nan,
            "average_yield_improvement_vs_baseline": comparison["bucket_yield_improvement_vs_baseline"].mean()
            if not comparison.empty
            else np.nan,
        }
    )
    if not comparison.empty:
        for result, group in comparison.groupby("bucket_vs_baseline_cash_result", dropna=False):
            rows.append(
                {
                    "summary_type": "cash_result_vs_baseline",
                    "bucket": result,
                    "count": int(group["stock_id"].nunique()),
                    "share_percent": group["stock_id"].nunique() / total * 100 if total else np.nan,
                    "average_cash_improvement_vs_baseline": group["bucket_cash_improvement_vs_baseline"].mean(),
                    "average_yield_improvement_vs_baseline": group["bucket_yield_improvement_vs_baseline"].mean(),
                }
            )
    if not bucket_summary.empty:
        rows.append(
            {
                "summary_type": "bucket_support",
                "bucket": "fallback_buckets",
                "count": int(bucket_summary["fallback_stock_count"].gt(0).sum()),
                "share_percent": np.nan,
            }
        )
    if not classification_summary.empty:
        for _, row in classification_summary.iterrows():
            rows.append(
                {
                    "summary_type": "classification_outcome",
                    "bucket": row["classification_outcome"],
                    "count": int(row["stock_count"]),
                    "share_percent": row["share_percent"],
                    "average_cash_improvement_vs_baseline": np.nan,
                    "average_yield_improvement_vs_baseline": np.nan,
                }
            )
    return _round_numeric_columns(pd.DataFrame(rows))


def write_outputs(
    output_dir: Path,
    outputs: dict[str, pd.DataFrame],
    run_config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_run_config_and_registry(output_dir, run_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-benchmark-dir", type=Path, default=DEFAULT_DIRECT_BENCHMARK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR)
    parser.add_argument("--direct-source-family", default=DEFAULT_DIRECT_SOURCE_FAMILY)
    parser.add_argument("--direct-model", default=DEFAULT_DIRECT_MODEL)
    parser.add_argument("--direct-eps-method", default=DEFAULT_DIRECT_EPS_METHOD)
    parser.add_argument("--baseline-source-family", default=DEFAULT_BASELINE_SOURCE_FAMILY)
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--baseline-eps-method", default=DEFAULT_BASELINE_EPS_METHOD)
    parser.add_argument("--baseline-dividend-method", default=DEFAULT_BASELINE_DIVIDEND_METHOD)
    parser.add_argument("--top-n", type=int, default=20)
    return add_registry_arguments(parser)


def _load_run_config(benchmark_dir: Path) -> dict[str, object]:
    path = Path(benchmark_dir) / "run_config.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def run_direct_dividend_error_diagnostics(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    benchmark_outputs = load_direct_benchmark_outputs(args.direct_benchmark_dir)
    benchmark_config = _load_run_config(args.direct_benchmark_dir)
    direct_global_method = str(benchmark_config.get("selected_direct_method") or "direct_hurdle_ridge_t060")
    comparison = build_stock_error_comparison(
        benchmark_outputs["selected_stock_accuracy"],
        benchmark_outputs["baseline_stock_accuracy"],
        direct_global_method=direct_global_method,
        direct_source_family=args.direct_source_family,
        direct_model=args.direct_model,
        direct_eps_method=args.direct_eps_method,
        baseline_source_family=args.baseline_source_family,
        baseline_model=args.baseline_model,
        baseline_eps_method=args.baseline_eps_method,
        baseline_dividend_method=args.baseline_dividend_method,
    )
    improvement_leaders = build_improvement_leaders(comparison, args.top_n)
    regression_hotspots = build_regression_hotspots(comparison, args.top_n)
    bucket_summary = build_bucket_error_summary(comparison)
    classification_outcomes = build_classification_outcomes(
        benchmark_outputs["selected_predictions"],
        direct_source_family=args.direct_source_family,
        direct_model=args.direct_model,
        direct_eps_method=args.direct_eps_method,
    )
    classification_summary = build_classification_summary(classification_outcomes)
    classification_errors = build_classification_errors(classification_outcomes)
    amount_error_hotspots = build_amount_error_hotspots(classification_outcomes, args.top_n)
    diagnostic_summary = build_diagnostic_summary(comparison, bucket_summary, classification_summary)
    outputs = {
        "direct_dividend_stock_error_comparison": comparison,
        "direct_dividend_improvement_leaders": improvement_leaders,
        "direct_dividend_regression_hotspots": regression_hotspots,
        "direct_dividend_bucket_error_summary": bucket_summary,
        "direct_dividend_classification_outcomes": classification_outcomes,
        "direct_dividend_classification_summary": classification_summary,
        "direct_dividend_classification_errors": classification_errors,
        "direct_dividend_amount_error_hotspots": amount_error_hotspots,
        "direct_dividend_diagnostic_summary": diagnostic_summary,
    }
    run_config = {
        "direct_benchmark_dir": str(args.direct_benchmark_dir),
        "output_dir": str(args.output_dir),
        "target_year": int(args.target_year),
        "direct_global_method": direct_global_method,
        "direct_bucket_method": SELECTED_BUCKET_DIVIDEND_METHOD,
        "direct_source_family": args.direct_source_family,
        "direct_model": args.direct_model,
        "direct_eps_method": args.direct_eps_method,
        "baseline_source_family": args.baseline_source_family,
        "baseline_model": args.baseline_model,
        "baseline_eps_method": args.baseline_eps_method,
        "baseline_dividend_method": args.baseline_dividend_method,
        "top_n": int(args.top_n),
        "stock_count": int(comparison["stock_id"].nunique()) if not comparison.empty else 0,
    }
    run_config = enrich_run_config_from_args(
        run_config,
        args,
        experiment_family="direct_dividend_error_diagnostics",
        extra={"direct_benchmark_dir": str(args.direct_benchmark_dir)},
    )
    write_outputs(args.output_dir, outputs, run_config)
    outputs["run_config"] = pd.DataFrame([run_config])
    return outputs


def main() -> None:
    args = build_parser().parse_args()
    results = run_direct_dividend_error_diagnostics(args)
    print("Wrote direct dividend error diagnostics outputs to", args.output_dir)
    print("\nDiagnostic summary:")
    print(results["direct_dividend_diagnostic_summary"].to_string(index=False))
    print("\nLargest direct dividend improvements:")
    display_columns = [
        "stock_id",
        "stock_name",
        "dividend_selection_bucket",
        "bucket_cash_dividend_abs_error",
        "baseline_cash_dividend_abs_error",
        "bucket_cash_improvement_vs_baseline",
    ]
    leaders = results["direct_dividend_improvement_leaders"]
    if not leaders.empty:
        print(leaders[[column for column in display_columns if column in leaders.columns]].head(10).to_string(index=False))
    else:
        print("(none)")


if __name__ == "__main__":
    main()
