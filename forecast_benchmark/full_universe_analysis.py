from __future__ import annotations

"""Build exact-cohort full-universe reports from precomputed system outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_benchmark.adapters.rolling_adapter import load_rolling_predictions
from forecast_benchmark.metrics import (
    build_accuracy_frame,
    build_overall_accuracy,
    build_stock_accuracy,
    build_winner_summary,
)
from forecast_benchmark.run_benchmark import filter_comparable_predictions


BASELINE_MODEL = "Last observed revenue"
EFFECT_PAIRS = (
    ("ensemble_vs_lightgbm", "LightGBM", "ensemble_revenue"),
    ("cluster_vs_plain_lstm", "Rolling LSTM", "Rolling LSTM + Cluster"),
    (
        "adjustment_vs_cluster",
        "Rolling LSTM + Cluster",
        "Rolling LSTM + Cluster + Conditional Adjustment",
    ),
    ("hybrid_xlstm_vs_plain_lstm", "Rolling LSTM", "Rolling xLSTM"),
    (
        "xlstm_adjustment_vs_xlstm",
        "Rolling xLSTM",
        "Rolling xLSTM + Conditional Adjustment",
    ),
    (
        "xlstm_adjusted_vs_ensemble",
        "ensemble_revenue",
        "Rolling xLSTM + Conditional Adjustment",
    ),
)


def load_ensemble_predictions(path: str | Path, target_year: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "source_family",
        "model",
        "stock_id",
        "target_year",
        "target_month",
        "predicted_revenue",
        "actual_revenue",
        "last_observed_revenue",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Ensemble predictions are missing columns: {sorted(missing)}")
    for column in [
        "stock_id",
        "target_year",
        "target_month",
        "predicted_revenue",
        "actual_revenue",
        "last_observed_revenue",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(
        subset=["source_family", "model", "stock_id", "target_year", "target_month", "predicted_revenue"]
    )
    frame["stock_id"] = frame["stock_id"].astype(int)
    frame["target_year"] = frame["target_year"].astype(int)
    frame["target_month"] = frame["target_month"].astype(int)
    for column in ["stock_name", "industry_category", "sequence_backbone", "xlstm_backbone"]:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[frame["target_year"].eq(int(target_year))]


def add_last_observed_baseline(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["stock_id", "target_year", "target_month"]
    labels = [column for column in ["stock_name", "industry_category"] if column in predictions.columns]
    baseline = predictions[keys + labels + ["actual_revenue", "last_observed_revenue"]].drop_duplicates(keys)
    baseline["source_family"] = "baseline"
    baseline["model"] = BASELINE_MODEL
    baseline["predicted_revenue"] = baseline["last_observed_revenue"]
    baseline["sequence_backbone"] = pd.NA
    baseline["xlstm_backbone"] = pd.NA
    baseline["source_path"] = "derived from last_observed_revenue"
    columns = list(predictions.columns)
    for column in columns:
        if column not in baseline.columns:
            baseline[column] = pd.NA
    return pd.concat([predictions, baseline[columns]], ignore_index=True)


def build_complete_year_cohort(predictions: pd.DataFrame) -> pd.DataFrame:
    month_counts = predictions.groupby("stock_id")["target_month"].nunique()
    complete_stocks = month_counts[month_counts.eq(12)].index
    return predictions[predictions["stock_id"].isin(complete_stocks)].copy()


def build_error_distribution(stock_accuracy: pd.DataFrame) -> pd.DataFrame:
    frame = stock_accuracy.dropna(subset=["WMAPE"]).copy()
    frame["error_band"] = pd.cut(
        frame["WMAPE"],
        bins=[-np.inf, 10, 20, 30, 50, np.inf],
        labels=["<10%", "10-20%", "20-30%", "30-50%", ">=50%"],
        right=False,
    )
    summary = (
        frame.groupby(["source_family", "model", "error_band"], observed=True)
        .size()
        .rename("stock_count")
        .reset_index()
    )
    totals = summary.groupby(["source_family", "model"])["stock_count"].transform("sum")
    summary["stock_share_percent"] = summary["stock_count"] / totals * 100
    return summary


def build_pairwise_effects(stock_accuracy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = stock_accuracy.copy()
    normalized["stock_name"] = normalized["stock_name"].fillna("")
    normalized["industry_category"] = normalized["industry_category"].fillna("unknown")
    values = normalized.pivot_table(
        index=["stock_id", "stock_name", "industry_category"],
        columns="model",
        values=["WMAPE", "MAPE", "MAE", "DirectionAccuracy"],
        aggfunc="first",
    )
    values.columns = [f"{metric}__{model}" for metric, model in values.columns]
    values = values.reset_index()
    rows: list[pd.DataFrame] = []
    for effect, baseline, challenger in EFFECT_PAIRS:
        required = [f"WMAPE__{baseline}", f"WMAPE__{challenger}"]
        if not set(required).issubset(values.columns):
            continue
        part = values[["stock_id", "stock_name", "industry_category"]].copy()
        part["effect"] = effect
        part["baseline_model"] = baseline
        part["challenger_model"] = challenger
        for metric in ["WMAPE", "MAPE", "MAE", "DirectionAccuracy"]:
            baseline_column = f"{metric}__{baseline}"
            challenger_column = f"{metric}__{challenger}"
            if baseline_column not in values or challenger_column not in values:
                continue
            part[f"{metric}_baseline"] = values[baseline_column]
            part[f"{metric}_challenger"] = values[challenger_column]
            part[f"{metric}_delta"] = values[challenger_column] - values[baseline_column]
        part["WMAPE_winner"] = np.select(
            [part["WMAPE_delta"] < 0, part["WMAPE_delta"] > 0],
            ["challenger", "baseline"],
            default="tie",
        )
        rows.append(part)
    details = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if details.empty:
        return details, pd.DataFrame()
    summaries: list[dict[str, object]] = []
    for effect, group in details.groupby("effect"):
        counts = group["WMAPE_winner"].value_counts()
        valid = group["WMAPE_delta"].notna()
        denominator = int(valid.sum())
        summaries.append(
            {
                "effect": effect,
                "baseline_model": group["baseline_model"].iloc[0],
                "challenger_model": group["challenger_model"].iloc[0],
                "compared_stocks": denominator,
                "challenger_wins": int(counts.get("challenger", 0)),
                "baseline_wins": int(counts.get("baseline", 0)),
                "ties": int(counts.get("tie", 0)),
                "challenger_win_rate_percent": (
                    float(counts.get("challenger", 0) / denominator * 100) if denominator else np.nan
                ),
                "average_WMAPE_delta": float(group["WMAPE_delta"].mean()),
                "median_WMAPE_delta": float(group["WMAPE_delta"].median()),
                "average_MAPE_delta": float(group["MAPE_delta"].mean()),
                "average_DirectionAccuracy_delta": float(group["DirectionAccuracy_delta"].mean()),
            }
        )
    return details, pd.DataFrame(summaries)


def build_universe_summary(
    rolling_all: pd.DataFrame,
    ensemble_all: pd.DataFrame,
    comparable: pd.DataFrame,
    complete_year: pd.DataFrame,
    rolling_failures: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item": "Rolling attempted stocks",
                "value": int(rolling_all["stock_id"].nunique() + rolling_failures["stock_id"].nunique()),
            },
            {"item": "Rolling successful stocks", "value": int(rolling_all["stock_id"].nunique())},
            {"item": "Rolling failed stocks", "value": int(len(rolling_failures))},
            {"item": "Ensemble eligible stocks", "value": int(ensemble_all["stock_id"].nunique())},
            {"item": "Exact cross-system stocks", "value": int(comparable["stock_id"].nunique())},
            {"item": "Exact cross-system stock-months", "value": int(comparable[["stock_id", "target_month"]].drop_duplicates().shape[0])},
            {"item": "Complete 12-month cross-system stocks", "value": int(complete_year["stock_id"].nunique())},
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling-output-dir", required=True)
    parser.add_argument("--ensemble-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-year", type=int, default=2025)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    rolling_dir = Path(args.rolling_output_dir).resolve()
    ensemble_dir = Path(args.ensemble_output_dir).resolve()
    rolling = load_rolling_predictions(rolling_dir, target_year=int(args.target_year))
    ensemble = load_ensemble_predictions(
        ensemble_dir / "monthly_predictions.csv",
        target_year=int(args.target_year),
    )
    combined = pd.concat([rolling, ensemble], ignore_index=True, sort=False)
    required_pairs = set(combined[["source_family", "model"]].itertuples(index=False, name=None))
    comparable = filter_comparable_predictions(combined, required_pairs=required_pairs)
    complete_year = build_complete_year_cohort(comparable)
    comparison = add_last_observed_baseline(comparable)
    complete_comparison = add_last_observed_baseline(complete_year)

    overall = build_overall_accuracy(comparison)
    complete_overall = build_overall_accuracy(complete_comparison)
    stock_accuracy = build_stock_accuracy(comparison)
    winner_summary = build_winner_summary(stock_accuracy, primary_metric="WMAPE")
    industry_accuracy = build_accuracy_frame(
        comparison,
        ["industry_category", "source_family", "model"],
    )
    month_accuracy = build_accuracy_frame(
        comparison,
        ["target_month", "source_family", "model"],
    )
    error_distribution = build_error_distribution(stock_accuracy)
    pairwise_details, pairwise_summary = build_pairwise_effects(stock_accuracy)

    raw_rolling = pd.read_csv(rolling_dir / "monthly_predictions.csv")
    regime_lookup = raw_rolling[["stock_id", "target_year", "target_month", "regime"]].drop_duplicates(
        ["stock_id", "target_year", "target_month"]
    )
    comparison_with_regime = comparison.merge(
        regime_lookup,
        on=["stock_id", "target_year", "target_month"],
        how="left",
    )
    regime_accuracy = build_accuracy_frame(
        comparison_with_regime.dropna(subset=["regime"]),
        ["regime", "source_family", "model"],
    )
    rolling_failures_path = rolling_dir / "failed_runs.csv"
    rolling_failures = pd.read_csv(rolling_failures_path) if rolling_failures_path.exists() else pd.DataFrame()
    ensemble_failures_path = ensemble_dir / "failed_runs.csv"
    ensemble_failures = pd.read_csv(ensemble_failures_path) if ensemble_failures_path.exists() else pd.DataFrame()
    universe_summary = build_universe_summary(
        rolling,
        ensemble,
        comparable,
        complete_year,
        rolling_failures,
    )

    outputs = {
        "comparable_monthly_predictions.csv": comparable,
        "complete_year_predictions.csv": complete_year,
        "comparison_with_baseline.csv": comparison,
        "overall_accuracy.csv": overall,
        "complete_year_overall_accuracy.csv": complete_overall,
        "stock_accuracy.csv": stock_accuracy,
        "winner_summary.csv": winner_summary,
        "industry_accuracy.csv": industry_accuracy,
        "month_accuracy.csv": month_accuracy,
        "regime_accuracy.csv": regime_accuracy,
        "error_distribution.csv": error_distribution,
        "pairwise_effect_details.csv": pairwise_details,
        "pairwise_effect_summary.csv": pairwise_summary,
        "universe_summary.csv": universe_summary,
        "rolling_failed_runs.csv": rolling_failures,
        "ensemble_failed_runs.csv": ensemble_failures,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    config = {
        "target_year": int(args.target_year),
        "rolling_output_dir": str(rolling_dir),
        "ensemble_output_dir": str(ensemble_dir),
        "required_source_model_pairs": sorted([list(pair) for pair in required_pairs]),
        "comparable_stock_count": int(comparable["stock_id"].nunique()),
        "comparable_stock_month_count": int(comparable[["stock_id", "target_month"]].drop_duplicates().shape[0]),
        "complete_year_stock_count": int(complete_year["stock_id"].nunique()),
        "primary_metric": "WMAPE",
        "evidence_tier": "C",
        "report_ready": False,
        "reason": "2025 is the xLSTM development/replay year, not an independent unseen holdout.",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
