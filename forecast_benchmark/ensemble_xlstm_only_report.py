from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_MODELS = ("ensemble_revenue", "Rolling xLSTM")
REFERENCE_MODEL = "Last observed revenue"
REPORT_MODELS = (*PRIMARY_MODELS, REFERENCE_MODEL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a full-universe report limited to ensemble_revenue and Rolling xLSTM."
    )
    parser.add_argument("--benchmark-dir", type=Path, required=True)
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
    direction_actual = np.sign(actual - frame["last_observed_revenue"].astype(float).to_numpy())
    direction_predicted = np.sign(
        predicted - frame["last_observed_revenue"].astype(float).to_numpy()
    )
    return pd.Series(
        {
            "observations": int(len(frame)),
            "stock_count": int(frame["stock_id"].nunique()),
            "MAE": float(absolute_error.mean()),
            "MAPE": float(np.nanmean(ape)),
            "MedianAPE": float(np.nanmedian(ape)),
            "WMAPE": float(absolute_error.sum() / denominator.sum() * 100),
            "Bias": float((predicted - actual).mean()),
            "UnderestimateRate": float((predicted < actual).mean() * 100),
            "DirectionAccuracy": float((direction_actual == direction_predicted).mean() * 100),
        }
    )


def build_size_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
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
    result = (
        merged.groupby(["source_family", "model", "size_decile"], observed=True)
        .apply(calculate_metrics, include_groups=False)
        .reset_index()
    )
    result["size_decile_number"] = result["size_decile"].astype(str).str[1:].astype(int)
    return result.sort_values(["size_decile_number", "model"]).reset_index(drop=True)


def build_pairwise(stock_accuracy: pd.DataFrame) -> pd.DataFrame:
    pivot = stock_accuracy.pivot(index="stock_id", columns="model", values="WMAPE").dropna()
    ensemble = pivot[PRIMARY_MODELS[0]]
    xlstm = pivot[PRIMARY_MODELS[1]]
    delta = xlstm - ensemble
    return pd.DataFrame(
        [
            {
                "baseline_model": PRIMARY_MODELS[0],
                "challenger_model": PRIMARY_MODELS[1],
                "compared_stocks": int(len(pivot)),
                "ensemble_wins": int((delta > 1e-12).sum()),
                "xlstm_wins": int((delta < -1e-12).sum()),
                "ties": int((delta.abs() <= 1e-12).sum()),
                "ensemble_win_rate_percent": float((delta > 1e-12).mean() * 100),
                "xlstm_win_rate_percent": float((delta < -1e-12).mean() * 100),
                "average_xlstm_minus_ensemble_WMAPE": float(delta.mean()),
                "median_xlstm_minus_ensemble_WMAPE": float(delta.median()),
            }
        ]
    )


def build_winner_summary(stock_accuracy: pd.DataFrame) -> pd.DataFrame:
    primary = stock_accuracy[stock_accuracy["model"].isin(PRIMARY_MODELS)]
    winners = primary.loc[primary.groupby("stock_id")["WMAPE"].idxmin()]
    summary = (
        winners.groupby(["source_family", "model"], as_index=False)
        .agg(stock_wins=("stock_id", "nunique"))
        .sort_values("stock_wins", ascending=False)
    )
    total = int(winners["stock_id"].nunique())
    summary["compared_stocks"] = total
    summary["stock_win_rate_percent"] = summary["stock_wins"] / total * 100
    return summary.reset_index(drop=True)


def build_prediction_ratio(predictions: pd.DataFrame) -> pd.DataFrame:
    valid = predictions[predictions["actual_revenue"].abs().gt(0)].copy()
    valid["prediction_actual_ratio"] = valid["predicted_revenue"] / valid["actual_revenue"]
    rows: list[dict[str, object]] = []
    for (source_family, model), group in valid.groupby(["source_family", "model"]):
        ratio = group["prediction_actual_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "source_family": source_family,
                "model": model,
                "observations": int(len(ratio)),
                "ratio_p10": float(ratio.quantile(0.10)),
                "ratio_p25": float(ratio.quantile(0.25)),
                "ratio_median": float(ratio.quantile(0.50)),
                "ratio_p75": float(ratio.quantile(0.75)),
                "ratio_p90": float(ratio.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows).sort_values("ratio_median").reset_index(drop=True)


def build_annual_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    annual = (
        predictions.groupby(
            ["source_family", "model", "stock_id", "stock_name", "industry_category"],
            as_index=False,
        )
        .agg(
            predicted_annual_revenue=("predicted_revenue", "sum"),
            actual_annual_revenue=("actual_revenue", "sum"),
            monthly_observations=("target_month", "nunique"),
        )
    )
    annual["annual_abs_error"] = (
        annual["predicted_annual_revenue"] - annual["actual_annual_revenue"]
    ).abs()
    annual["annual_APE"] = np.where(
        annual["actual_annual_revenue"].abs().gt(0),
        annual["annual_abs_error"] / annual["actual_annual_revenue"].abs() * 100,
        np.nan,
    )
    annual["annual_bias"] = (
        annual["predicted_annual_revenue"] - annual["actual_annual_revenue"]
    )
    return annual.sort_values(["model", "annual_APE"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    predictions = pd.read_csv(
        args.benchmark_dir / "comparison_with_baseline.csv", low_memory=False
    )
    predictions = predictions[predictions["model"].isin(REPORT_MODELS)].copy()
    model_counts = predictions.groupby("model").agg(
        stocks=("stock_id", "nunique"), months=("target_month", "count")
    )
    expected_stocks = int(model_counts["stocks"].min())
    if expected_stocks <= 0 or not model_counts["stocks"].eq(expected_stocks).all():
        raise ValueError(f"Two-model cohort is not aligned:\n{model_counts}")

    overall = pd.read_csv(args.benchmark_dir / "overall_accuracy.csv")
    overall = overall[overall["model"].isin(REPORT_MODELS)].copy()
    stock_accuracy = pd.read_csv(args.benchmark_dir / "stock_accuracy.csv")
    stock_accuracy = stock_accuracy[stock_accuracy["model"].isin(REPORT_MODELS)].copy()
    month_accuracy = pd.read_csv(args.benchmark_dir / "month_accuracy.csv")
    month_accuracy = month_accuracy[month_accuracy["model"].isin(REPORT_MODELS)].copy()
    industry_accuracy = pd.read_csv(args.benchmark_dir / "industry_accuracy.csv")
    industry_accuracy = industry_accuracy[industry_accuracy["model"].isin(REPORT_MODELS)].copy()
    error_distribution = pd.read_csv(args.benchmark_dir / "error_distribution.csv")
    error_distribution = error_distribution[
        error_distribution["model"].isin(REPORT_MODELS)
    ].copy()

    predictions.to_csv(
        args.output_dir / "monthly_predictions.csv", index=False, encoding="utf-8-sig"
    )
    overall.to_csv(args.output_dir / "overall_accuracy.csv", index=False, encoding="utf-8-sig")
    stock_accuracy.to_csv(
        args.output_dir / "stock_accuracy.csv", index=False, encoding="utf-8-sig"
    )
    month_accuracy.to_csv(
        args.output_dir / "month_accuracy.csv", index=False, encoding="utf-8-sig"
    )
    industry_accuracy.to_csv(
        args.output_dir / "industry_accuracy.csv", index=False, encoding="utf-8-sig"
    )
    error_distribution.to_csv(
        args.output_dir / "error_distribution.csv", index=False, encoding="utf-8-sig"
    )
    build_size_accuracy(predictions).to_csv(
        args.output_dir / "size_bucket_accuracy.csv", index=False, encoding="utf-8-sig"
    )
    build_pairwise(stock_accuracy).to_csv(
        args.output_dir / "pairwise_summary.csv", index=False, encoding="utf-8-sig"
    )
    build_winner_summary(stock_accuracy).to_csv(
        args.output_dir / "primary_winner_summary.csv", index=False, encoding="utf-8-sig"
    )
    build_prediction_ratio(predictions).to_csv(
        args.output_dir / "prediction_actual_ratio.csv", index=False, encoding="utf-8-sig"
    )
    build_annual_accuracy(predictions).to_csv(
        args.output_dir / "annual_stock_accuracy.csv", index=False, encoding="utf-8-sig"
    )

    config = {
        "target_year": 2025,
        "primary_models": list(PRIMARY_MODELS),
        "reference_only": REFERENCE_MODEL,
        "stock_count": expected_stocks,
        "stock_month_count_per_model": int(model_counts.loc[PRIMARY_MODELS[0], "months"]),
        "source_benchmark_dir": str(args.benchmark_dir.resolve()),
        "xlstm_backbone": "xlstm_hybrid (mLSTM + sLSTM)",
        "evidence_tier": "C",
        "scope_note": (
            "Only the final ensemble forecast and plain Hybrid xLSTM are primary methods. "
            "The last-observed baseline is retained only as an error reference."
        ),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(overall.to_string(index=False))
    print(build_pairwise(stock_accuracy).to_string(index=False))


if __name__ == "__main__":
    main()
