from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPRESENTATIVE_MODELS = [
    "Last observed revenue",
    "SeasonalQuantile",
    "LightGBM",
    "ensemble_revenue",
    "Rolling xLSTM + Conditional Adjustment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact report tables from full-universe benchmark outputs."
    )
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--financial-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def calculate_metrics(frame: pd.DataFrame) -> pd.Series:
    actual = frame["actual_revenue"].astype(float).to_numpy()
    predicted = frame["predicted_revenue"].astype(float).to_numpy()
    absolute_error = np.abs(predicted - actual)
    denominator = np.abs(actual)
    valid = denominator > 0
    ape = np.full(len(frame), np.nan, dtype=float)
    ape[valid] = absolute_error[valid] / denominator[valid] * 100
    return pd.Series(
        {
            "observations": int(len(frame)),
            "stock_count": int(frame["stock_id"].nunique()),
            "MAE": float(absolute_error.mean()),
            "MAPE": float(np.nanmean(ape)),
            "MedianAPE": float(np.nanmedian(ape)),
            "WMAPE": float(absolute_error.sum() / denominator.sum() * 100),
            "Bias": float((predicted - actual).mean()),
        }
    )


def build_size_bucket_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    actual_by_stock = (
        predictions[["stock_id", "target_month", "actual_revenue"]]
        .drop_duplicates(["stock_id", "target_month"])
        .groupby("stock_id", as_index=False)["actual_revenue"]
        .mean()
        .rename(columns={"actual_revenue": "average_monthly_actual_revenue"})
    )
    actual_by_stock["size_decile"] = pd.qcut(
        actual_by_stock["average_monthly_actual_revenue"],
        q=10,
        labels=[f"Q{i}" for i in range(1, 11)],
        duplicates="drop",
    )
    merged = predictions.merge(actual_by_stock, on="stock_id", how="left")
    representative = merged[merged["model"].isin(REPRESENTATIVE_MODELS)].copy()
    result = (
        representative.groupby(["source_family", "model", "size_decile"], observed=True)
        .apply(calculate_metrics, include_groups=False)
        .reset_index()
    )
    result["size_decile_number"] = result["size_decile"].astype(str).str[1:].astype(int)
    return result.sort_values(["size_decile_number", "model"]).reset_index(drop=True)


def build_error_quantiles(stock_accuracy: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (source_family, model), group in stock_accuracy.groupby(["source_family", "model"]):
        values = group["WMAPE"].astype(float)
        rows.append(
            {
                "source_family": source_family,
                "model": model,
                "stock_count": int(group["stock_id"].nunique()),
                "WMAPE_p10": float(values.quantile(0.10)),
                "WMAPE_p25": float(values.quantile(0.25)),
                "WMAPE_median": float(values.quantile(0.50)),
                "WMAPE_p75": float(values.quantile(0.75)),
                "WMAPE_p90": float(values.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows).sort_values("WMAPE_median").reset_index(drop=True)


def build_learned_winners(stock_accuracy: pd.DataFrame) -> pd.DataFrame:
    learned = stock_accuracy[stock_accuracy["source_family"] != "baseline"].copy()
    winners = learned.loc[learned.groupby("stock_id")["WMAPE"].idxmin()]
    summary = (
        winners.groupby(["source_family", "model"], as_index=False)
        .agg(stock_wins=("stock_id", "nunique"))
        .sort_values("stock_wins", ascending=False)
    )
    total = int(winners["stock_id"].nunique())
    summary["compared_stocks"] = total
    summary["stock_win_rate_percent"] = summary["stock_wins"] / total * 100
    return summary.reset_index(drop=True)


def build_month_extremes(month_accuracy: pd.DataFrame) -> pd.DataFrame:
    representative = month_accuracy[month_accuracy["model"].isin(REPRESENTATIVE_MODELS)]
    rows: list[dict[str, object]] = []
    for (source_family, model), group in representative.groupby(["source_family", "model"]):
        best = group.loc[group["WMAPE"].idxmin()]
        worst = group.loc[group["WMAPE"].idxmax()]
        rows.append(
            {
                "source_family": source_family,
                "model": model,
                "best_month": int(best["target_month"]),
                "best_month_WMAPE": float(best["WMAPE"]),
                "worst_month": int(worst["target_month"]),
                "worst_month_WMAPE": float(worst["WMAPE"]),
                "average_month_WMAPE": float(group["WMAPE"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("average_month_WMAPE").reset_index(drop=True)


def build_representative_stock_errors(stock_accuracy: pd.DataFrame) -> pd.DataFrame:
    representative = stock_accuracy[stock_accuracy["model"].isin(REPRESENTATIVE_MODELS)].copy()
    result_parts: list[pd.DataFrame] = []
    for (_, _), group in representative.groupby(["source_family", "model"]):
        ordered = group.sort_values("WMAPE")
        best = ordered.head(10).copy()
        best["error_group"] = "best_10"
        worst = ordered.tail(10).sort_values("WMAPE", ascending=False).copy()
        worst["error_group"] = "worst_10"
        result_parts.extend([best, worst])
    columns = [
        "source_family",
        "model",
        "error_group",
        "stock_id",
        "stock_name",
        "industry_category",
        "WMAPE",
        "MedianAPE",
        "Bias",
        "UnderestimateRate",
        "DirectionAccuracy",
    ]
    return pd.concat(result_parts, ignore_index=True)[columns]


def build_failure_summary(
    rolling_failures: pd.DataFrame, ensemble_failures: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not rolling_failures.empty:
        grouped = rolling_failures.groupby("error_type", dropna=False)
        for error_type, group in grouped:
            rows.append(
                {
                    "workflow": "Rolling/xLSTM",
                    "target_year": 2025,
                    "model": "all rolling models",
                    "error_type": error_type,
                    "count": int(len(group)),
                    "example": group.iloc[0]["error"],
                }
            )
    if not ensemble_failures.empty:
        grouped = ensemble_failures.groupby(
            ["target_year", "model", "error_type"], dropna=False
        )
        for (target_year, model, error_type), group in grouped:
            rows.append(
                {
                    "workflow": "Ensemble",
                    "target_year": int(target_year),
                    "model": model,
                    "error_type": error_type,
                    "count": int(len(group)),
                    "example": group.iloc[0]["error"],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.benchmark_dir / "comparison_with_baseline.csv")
    stock_accuracy = pd.read_csv(args.benchmark_dir / "stock_accuracy.csv")
    month_accuracy = pd.read_csv(args.benchmark_dir / "month_accuracy.csv")
    rolling_failures = pd.read_csv(args.benchmark_dir / "rolling_failed_runs.csv")
    ensemble_failures = pd.read_csv(args.benchmark_dir / "ensemble_failed_runs.csv")

    build_size_bucket_accuracy(predictions).to_csv(
        args.output_dir / "size_bucket_accuracy.csv", index=False, encoding="utf-8-sig"
    )
    build_error_quantiles(stock_accuracy).to_csv(
        args.output_dir / "stock_error_quantiles.csv", index=False, encoding="utf-8-sig"
    )
    build_learned_winners(stock_accuracy).to_csv(
        args.output_dir / "learned_model_winner_summary.csv", index=False, encoding="utf-8-sig"
    )
    build_month_extremes(month_accuracy).to_csv(
        args.output_dir / "month_extremes.csv", index=False, encoding="utf-8-sig"
    )
    build_representative_stock_errors(stock_accuracy).to_csv(
        args.output_dir / "representative_stock_errors.csv", index=False, encoding="utf-8-sig"
    )
    build_failure_summary(rolling_failures, ensemble_failures).to_csv(
        args.output_dir / "failure_summary.csv", index=False, encoding="utf-8-sig"
    )

    financial = pd.read_csv(args.financial_dir / "financial_overall_accuracy.csv")
    financial.to_csv(
        args.output_dir / "financial_overall_accuracy.csv", index=False, encoding="utf-8-sig"
    )
    audit = pd.read_csv(args.financial_dir / "dividend_alignment_audit.csv")
    audit.to_csv(
        args.output_dir / "dividend_alignment_audit.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
