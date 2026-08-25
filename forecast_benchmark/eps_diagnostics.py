"""Diagnose stock-level EPS transform errors from EPS benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_benchmark.benchmark_config import DEFAULT_TARGET_YEAR, PROJECT_ROOT
from forecast_benchmark.eps_benchmark import (
    build_historical_annual_frame,
    load_source_data,
    select_annual_ratio_candidates,
)
from forecast_benchmark.experiment_registry import (
    add_registry_arguments,
    enrich_run_config_from_args,
    write_run_config_and_registry,
)


DEFAULT_EPS_BENCHMARK_DIR = PROJECT_ROOT / "forecast_benchmark" / "outputs" / "eps_benchmark"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forecast_benchmark" / "outputs" / "eps_diagnostics"
EPS_METHOD_FAMILY = {
    "current_ratio": "current_ratio",
    "seasonal_quarter_median": "seasonal_ratio",
    "ridge_annual": "ml_eps_layer",
    "lasso_annual": "ml_eps_layer",
    "elastic_net_annual": "ml_eps_layer",
}


def _first_valid(values: pd.Series) -> object:
    valid = values.dropna()
    return valid.iloc[0] if not valid.empty else np.nan


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _safe_float(value: object) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else np.nan


def classify_ratio_stability(
    ratio_count: int,
    ratio_std_to_median: float,
    latest_deviation_from_recent_median_pct: float,
) -> str:
    if int(ratio_count) < 3:
        return "insufficient_history"
    if not np.isfinite(ratio_std_to_median):
        return "unstable_ratio"
    if ratio_std_to_median <= 0.35 and (
        not np.isfinite(latest_deviation_from_recent_median_pct)
        or latest_deviation_from_recent_median_pct <= 50
    ):
        return "stable_ratio"
    if ratio_std_to_median <= 1.0 and (
        not np.isfinite(latest_deviation_from_recent_median_pct)
        or latest_deviation_from_recent_median_pct <= 150
    ):
        return "moderate_ratio"
    return "unstable_ratio"


def recommend_eps_path(
    ratio_stability_bucket: str,
    best_current_error: float,
    best_seasonal_error: float,
    best_ml_error: float,
) -> str:
    errors = {
        "keep_current_ratio": best_current_error,
        "test_seasonal_quarter_ratio": best_seasonal_error,
        "test_ml_eps_layer": best_ml_error,
    }
    valid_errors = {key: value for key, value in errors.items() if np.isfinite(value)}
    if not valid_errors:
        return "manual_review"

    best_path = min(valid_errors, key=valid_errors.get)
    best_error = valid_errors[best_path]
    current_error = valid_errors.get("keep_current_ratio", np.inf)
    seasonal_error = valid_errors.get("test_seasonal_quarter_ratio", np.inf)
    ml_error = valid_errors.get("test_ml_eps_layer", np.inf)

    if ratio_stability_bucket == "stable_ratio" and current_error <= min(seasonal_error, ml_error) * 1.10:
        return "keep_current_ratio"
    if ratio_stability_bucket == "insufficient_history":
        return "manual_or_cross_sectional_ml"
    if best_path == "test_seasonal_quarter_ratio" and seasonal_error <= min(current_error, ml_error) * 0.90:
        return "test_seasonal_quarter_ratio"
    if best_path == "test_ml_eps_layer" and ml_error <= min(current_error, seasonal_error) * 0.90:
        return "test_ml_eps_layer"
    if ratio_stability_bucket == "unstable_ratio":
        return "stock_specific_or_ml_eps_layer"
    return "stock_specific_validation"


def classify_current_ratio_driver(
    current_ratio_error: float,
    oracle_current_ratio_error: float,
    annual_revenue_abs_percent_error: float,
) -> str:
    if not np.isfinite(current_ratio_error):
        return "missing_current_ratio_result"
    if not np.isfinite(oracle_current_ratio_error):
        return "missing_oracle_diagnostic"
    if current_ratio_error < oracle_current_ratio_error * 0.75:
        return "forecast_error_offset_formula_error"
    if oracle_current_ratio_error >= current_ratio_error * 0.75:
        return "eps_ratio_formula_error"
    if np.isfinite(annual_revenue_abs_percent_error) and annual_revenue_abs_percent_error >= 20:
        return "revenue_forecast_error"
    return "mixed_error"


def load_eps_stock_accuracy(eps_benchmark_dir: Path) -> pd.DataFrame:
    path = Path(eps_benchmark_dir) / "eps_stock_accuracy.csv"
    stock_accuracy = pd.read_csv(path)
    required = {
        "source_family",
        "model",
        "eps_method",
        "is_oracle",
        "stock_id",
        "eps_abs_error",
        "estimated_eps",
        "actual_annual_eps",
    }
    missing = required - set(stock_accuracy.columns)
    if missing:
        raise ValueError(f"EPS stock accuracy missing columns: {sorted(missing)}")
    stock_accuracy["is_oracle"] = normalize_bool(stock_accuracy["is_oracle"])
    stock_accuracy["stock_id"] = pd.to_numeric(stock_accuracy["stock_id"], errors="coerce").astype("Int64")
    numeric_columns = [
        "eps_abs_error",
        "eps_abs_percent_error",
        "annual_revenue_abs_percent_error",
        "predicted_annual_revenue",
        "actual_annual_revenue",
        "estimated_eps",
        "actual_annual_eps",
    ]
    for column in numeric_columns:
        if column in stock_accuracy.columns:
            stock_accuracy[column] = pd.to_numeric(stock_accuracy[column], errors="coerce")
    return stock_accuracy.dropna(subset=["stock_id"]).assign(stock_id=lambda frame: frame["stock_id"].astype(int))


def build_stock_metadata(stock_accuracy: pd.DataFrame) -> pd.DataFrame:
    return (
        stock_accuracy.groupby("stock_id", as_index=False)
        .agg(
            stock_name=("stock_name", _first_valid) if "stock_name" in stock_accuracy.columns else ("model", _first_valid),
            industry_category=("industry_category", _first_valid)
            if "industry_category" in stock_accuracy.columns
            else ("model", _first_valid),
        )
        .replace("", np.nan)
    )


def build_ratio_stability(
    annual_history: pd.DataFrame,
    stock_metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, stock_row in stock_metadata.iterrows():
        stock_id = int(stock_row["stock_id"])
        candidates = select_annual_ratio_candidates(annual_history, stock_id)
        ratios = pd.to_numeric(candidates["eps_to_revenue_ratio"], errors="coerce").dropna()
        years = candidates.loc[ratios.index, "revenue_year"] if not candidates.empty else pd.Series(dtype=float)
        ratio_count = int(len(ratios))

        ratio_median = float(ratios.median()) if ratio_count else np.nan
        ratio_mean = float(ratios.mean()) if ratio_count else np.nan
        ratio_std = float(ratios.std(ddof=0)) if ratio_count else np.nan
        ratio_std_to_median = (
            abs(ratio_std / ratio_median) if np.isfinite(ratio_std) and abs(ratio_median) > 1e-12 else np.nan
        )
        recent = candidates.sort_values("revenue_year").tail(3)
        recent_ratio_median = (
            float(pd.to_numeric(recent["eps_to_revenue_ratio"], errors="coerce").median())
            if not recent.empty
            else np.nan
        )
        latest_ratio = float(ratios.iloc[-1]) if ratio_count else np.nan
        latest_deviation = (
            abs(latest_ratio - recent_ratio_median) / abs(recent_ratio_median) * 100
            if np.isfinite(latest_ratio) and abs(recent_ratio_median) > 1e-12
            else np.nan
        )
        max_abs_yoy_change = (
            float(pd.to_numeric(candidates.sort_values("revenue_year")["eps_to_revenue_ratio"], errors="coerce").diff().abs().max())
            if ratio_count >= 2
            else np.nan
        )
        bucket = classify_ratio_stability(ratio_count, ratio_std_to_median, latest_deviation)
        rows.append(
            {
                "stock_id": stock_id,
                "stock_name": stock_row.get("stock_name", np.nan),
                "industry_category": stock_row.get("industry_category", np.nan),
                "ratio_year_count": ratio_count,
                "ratio_year_min": int(years.min()) if ratio_count else np.nan,
                "ratio_year_max": int(years.max()) if ratio_count else np.nan,
                "eps_to_revenue_ratio_mean": ratio_mean,
                "eps_to_revenue_ratio_median": ratio_median,
                "eps_to_revenue_ratio_std": ratio_std,
                "ratio_std_to_median": ratio_std_to_median,
                "latest_eps_to_revenue_ratio": latest_ratio,
                "recent_three_ratio_median": recent_ratio_median,
                "latest_deviation_from_recent_median_pct": latest_deviation,
                "max_abs_ratio_yoy_change": max_abs_yoy_change,
                "eps_per_1b_revenue_median": ratio_median * 1_000_000 if np.isfinite(ratio_median) else np.nan,
                "eps_per_1b_revenue_recent_three_median": recent_ratio_median * 1_000_000
                if np.isfinite(recent_ratio_median)
                else np.nan,
                "ratio_stability_bucket": bucket,
            }
        )
    return pd.DataFrame(rows).sort_values(["ratio_stability_bucket", "stock_id"])


def _best_row(group: pd.DataFrame, subset: pd.Series) -> pd.Series | None:
    candidates = group[subset & group["eps_abs_error"].notna()].copy()
    if candidates.empty:
        return None
    return candidates.loc[candidates["eps_abs_error"].idxmin()]


def _row_value(row: pd.Series | None, column: str) -> object:
    if row is None:
        return np.nan
    return row.get(column, np.nan)


def build_method_recommendations(
    stock_accuracy: pd.DataFrame,
    ratio_stability: pd.DataFrame,
) -> pd.DataFrame:
    non_oracle = stock_accuracy[~stock_accuracy["is_oracle"]].copy()
    non_oracle["eps_method_family"] = non_oracle["eps_method"].map(EPS_METHOD_FAMILY).fillna("other")
    oracle = stock_accuracy[stock_accuracy["is_oracle"]].copy()
    rows = []

    for stock_id, group in non_oracle.groupby("stock_id"):
        current_row = _best_row(group, group["eps_method"].eq("current_ratio"))
        seasonal_row = _best_row(group, group["eps_method"].eq("seasonal_quarter_median"))
        ml_row = _best_row(group, group["eps_method_family"].eq("ml_eps_layer"))
        best_row = _best_row(group, pd.Series(True, index=group.index))
        oracle_row = _best_row(oracle[oracle["stock_id"].eq(stock_id)], pd.Series(True, index=oracle[oracle["stock_id"].eq(stock_id)].index))

        stability_row = ratio_stability[ratio_stability["stock_id"].eq(int(stock_id))]
        ratio_bucket = (
            str(stability_row["ratio_stability_bucket"].iloc[0])
            if not stability_row.empty
            else "missing_ratio_stability"
        )
        best_current_error = _safe_float(_row_value(current_row, "eps_abs_error"))
        best_seasonal_error = _safe_float(_row_value(seasonal_row, "eps_abs_error"))
        best_ml_error = _safe_float(_row_value(ml_row, "eps_abs_error"))
        oracle_error = _safe_float(_row_value(oracle_row, "eps_abs_error"))
        current_revenue_ape = _safe_float(_row_value(current_row, "annual_revenue_abs_percent_error"))
        recommended = recommend_eps_path(
            ratio_bucket,
            best_current_error,
            best_seasonal_error,
            best_ml_error,
        )
        driver = classify_current_ratio_driver(best_current_error, oracle_error, current_revenue_ape)

        rows.append(
            {
                "stock_id": int(stock_id),
                "stock_name": _row_value(best_row, "stock_name"),
                "industry_category": _row_value(best_row, "industry_category"),
                "actual_annual_eps": _row_value(best_row, "actual_annual_eps"),
                "ratio_stability_bucket": ratio_bucket,
                "best_source_family": _row_value(best_row, "source_family"),
                "best_model": _row_value(best_row, "model"),
                "best_eps_method": _row_value(best_row, "eps_method"),
                "best_eps_abs_error": _row_value(best_row, "eps_abs_error"),
                "best_eps_abs_percent_error": _row_value(best_row, "eps_abs_percent_error"),
                "best_current_model": _row_value(current_row, "model"),
                "best_current_eps_abs_error": best_current_error,
                "best_seasonal_model": _row_value(seasonal_row, "model"),
                "best_seasonal_eps_abs_error": best_seasonal_error,
                "best_ml_model": _row_value(ml_row, "model"),
                "best_ml_method": _row_value(ml_row, "eps_method"),
                "best_ml_eps_abs_error": best_ml_error,
                "oracle_current_ratio_eps_abs_error": oracle_error,
                "best_current_annual_revenue_abs_percent_error": current_revenue_ape,
                "current_ratio_error_driver": driver,
                "hindsight_recommended_eps_path": recommended,
            }
        )

    result = pd.DataFrame(rows)
    numeric_columns = [
        "actual_annual_eps",
        "best_eps_abs_error",
        "best_eps_abs_percent_error",
        "best_current_eps_abs_error",
        "best_seasonal_eps_abs_error",
        "best_ml_eps_abs_error",
        "oracle_current_ratio_eps_abs_error",
        "best_current_annual_revenue_abs_percent_error",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").round(4)
    return result.sort_values(["best_eps_abs_error", "stock_id"], ascending=[False, True])


def build_current_ratio_driver_by_model(stock_accuracy: pd.DataFrame) -> pd.DataFrame:
    current = stock_accuracy[
        (~stock_accuracy["is_oracle"]) & stock_accuracy["eps_method"].eq("current_ratio")
    ].copy()
    oracle = (
        stock_accuracy[stock_accuracy["is_oracle"]][["stock_id", "eps_abs_error"]]
        .rename(columns={"eps_abs_error": "oracle_current_ratio_eps_abs_error"})
        .drop_duplicates("stock_id")
    )
    current = current.merge(oracle, on="stock_id", how="left")
    current["current_ratio_error_driver"] = current.apply(
        lambda row: classify_current_ratio_driver(
            _safe_float(row["eps_abs_error"]),
            _safe_float(row["oracle_current_ratio_eps_abs_error"]),
            _safe_float(row.get("annual_revenue_abs_percent_error", np.nan)),
        ),
        axis=1,
    )
    return current.sort_values(["eps_abs_error", "stock_id"], ascending=[False, True])


def build_diagnostic_summary(
    ratio_stability: pd.DataFrame,
    recommendations: pd.DataFrame,
    current_ratio_driver_by_model: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "summary_type": "stock_count",
            "bucket": "all_analyzed_stocks",
            "count": int(recommendations["stock_id"].nunique()),
            "share_percent": 100.0,
        }
    )
    for bucket, group in ratio_stability.groupby("ratio_stability_bucket", dropna=False):
        rows.append(
            {
                "summary_type": "ratio_stability_bucket",
                "bucket": bucket,
                "count": int(group["stock_id"].nunique()),
                "share_percent": float(group["stock_id"].nunique() / ratio_stability["stock_id"].nunique() * 100)
                if ratio_stability["stock_id"].nunique()
                else np.nan,
            }
        )
    for path, group in recommendations.groupby("hindsight_recommended_eps_path", dropna=False):
        rows.append(
            {
                "summary_type": "hindsight_recommended_eps_path",
                "bucket": path,
                "count": int(group["stock_id"].nunique()),
                "share_percent": float(
                    group["stock_id"].nunique() / recommendations["stock_id"].nunique() * 100
                )
                if recommendations["stock_id"].nunique()
                else np.nan,
            }
        )
    for driver, group in current_ratio_driver_by_model.groupby("current_ratio_error_driver", dropna=False):
        rows.append(
            {
                "summary_type": "current_ratio_driver_by_model_row",
                "bucket": driver,
                "count": int(len(group)),
                "share_percent": float(len(group) / len(current_ratio_driver_by_model) * 100)
                if len(current_ratio_driver_by_model)
                else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    summary["share_percent"] = pd.to_numeric(summary["share_percent"], errors="coerce").round(4)
    return summary


def write_outputs(
    output_dir: Path,
    ratio_stability: pd.DataFrame,
    recommendations: pd.DataFrame,
    hotspots: pd.DataFrame,
    current_ratio_driver_by_model: pd.DataFrame,
    diagnostic_summary: pd.DataFrame,
    run_config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ratio_stability.to_csv(output_dir / "eps_ratio_stability.csv", index=False, encoding="utf-8-sig")
    recommendations.to_csv(output_dir / "eps_method_recommendations.csv", index=False, encoding="utf-8-sig")
    hotspots.to_csv(output_dir / "eps_error_hotspots.csv", index=False, encoding="utf-8-sig")
    current_ratio_driver_by_model.to_csv(
        output_dir / "eps_current_ratio_driver_by_model.csv",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostic_summary.to_csv(output_dir / "eps_diagnostic_summary.csv", index=False, encoding="utf-8-sig")
    write_run_config_and_registry(output_dir, run_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps-benchmark-dir", type=Path, default=DEFAULT_EPS_BENCHMARK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR)
    parser.add_argument("--top-n", type=int, default=30)
    return add_registry_arguments(parser)


def run_eps_diagnostics(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    stock_accuracy = load_eps_stock_accuracy(args.eps_benchmark_dir)
    revenue_data, eps = load_source_data()
    annual_history = build_historical_annual_frame(revenue_data, eps, args.target_year)
    stock_metadata = build_stock_metadata(stock_accuracy)
    ratio_stability = build_ratio_stability(annual_history, stock_metadata)
    recommendations = build_method_recommendations(stock_accuracy, ratio_stability)
    current_ratio_driver_by_model = build_current_ratio_driver_by_model(stock_accuracy)
    hotspots = recommendations.head(int(args.top_n)).copy()
    diagnostic_summary = build_diagnostic_summary(
        ratio_stability,
        recommendations,
        current_ratio_driver_by_model,
    )
    run_config = {
        "eps_benchmark_dir": str(args.eps_benchmark_dir),
        "output_dir": str(args.output_dir),
        "target_year": int(args.target_year),
        "top_n": int(args.top_n),
        "stock_count": int(recommendations["stock_id"].nunique()),
        "current_ratio_model_rows": int(len(current_ratio_driver_by_model)),
    }
    run_config = enrich_run_config_from_args(
        run_config,
        args,
        experiment_family="eps_diagnostics",
        extra={"eps_benchmark_dir": str(args.eps_benchmark_dir)},
    )
    write_outputs(
        args.output_dir,
        ratio_stability,
        recommendations,
        hotspots,
        current_ratio_driver_by_model,
        diagnostic_summary,
        run_config,
    )
    return {
        "eps_ratio_stability": ratio_stability,
        "eps_method_recommendations": recommendations,
        "eps_error_hotspots": hotspots,
        "eps_current_ratio_driver_by_model": current_ratio_driver_by_model,
        "eps_diagnostic_summary": diagnostic_summary,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = run_eps_diagnostics(args)
    print("Wrote EPS diagnostics outputs to", args.output_dir)
    print(results["eps_diagnostic_summary"].to_string(index=False))
    print("\nLargest remaining EPS-error stocks:")
    display_columns = [
        "stock_id",
        "best_model",
        "best_eps_method",
        "best_eps_abs_error",
        "ratio_stability_bucket",
        "current_ratio_error_driver",
        "hindsight_recommended_eps_path",
    ]
    print(results["eps_error_hotspots"][display_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
