"""Run a fair comparison between Ensemble Forecast and Rolling LSTM outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from forecast_benchmark.adapters.ensemble_adapter import run_ensemble_predictions
from forecast_benchmark.adapters.rolling_adapter import load_rolling_predictions
from forecast_benchmark.benchmark_config import BenchmarkConfig
from forecast_benchmark.experiment_registry import (
    add_registry_arguments,
    enrich_run_config_from_args,
    write_run_config_and_registry,
)
from forecast_benchmark.metrics import (
    build_overall_accuracy,
    build_stock_accuracy,
    build_winner_summary,
)


def parse_int_csv(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_csv(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    defaults = BenchmarkConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling-output-dir", type=Path, default=defaults.rolling_output_dir)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--target-year", type=int, default=defaults.target_year)
    parser.add_argument("--primary-metric", default=defaults.primary_metric)
    parser.add_argument("--stock-ids", help="Comma-separated stock IDs. Defaults to rolling output stock pool.")
    parser.add_argument("--stock-limit", type=int, help="Limit stock pool for smoke runs.")
    parser.add_argument("--rolling-models", help="Comma-separated Rolling model names to include.")
    parser.add_argument(
        "--ensemble-models",
        default=",".join(defaults.ensemble_models),
        help="Comma-separated Ensemble forecast columns to include.",
    )
    parser.add_argument(
        "--skip-ensemble",
        action="store_true",
        help="Only normalize Rolling outputs and compute metrics.",
    )
    return add_registry_arguments(parser)


def select_stock_pool(
    rolling_predictions: pd.DataFrame,
    explicit_stock_ids: list[int] | None,
    stock_limit: int | None,
) -> list[int]:
    if explicit_stock_ids is None:
        stock_ids = sorted(int(stock_id) for stock_id in rolling_predictions["stock_id"].dropna().unique())
    else:
        available = set(int(stock_id) for stock_id in rolling_predictions["stock_id"].dropna().unique())
        stock_ids = [int(stock_id) for stock_id in explicit_stock_ids if int(stock_id) in available]
    if stock_limit is not None:
        stock_ids = stock_ids[: int(stock_limit)]
    return stock_ids


def filter_comparable_predictions(
    predictions: pd.DataFrame,
    required_families: set[str] | None = None,
    required_pairs: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()

    required_columns = {
        "source_family",
        "model",
        "stock_id",
        "target_year",
        "target_month",
    }
    missing = required_columns.difference(predictions.columns)
    if missing:
        raise ValueError(f"Comparable predictions missing columns: {sorted(missing)}")

    scoped = predictions.copy()
    if required_pairs is None:
        if required_families is None:
            required_families = set(scoped["source_family"].dropna().astype(str).unique())
        scoped = scoped[scoped["source_family"].isin(required_families)].copy()
        required_pairs = set(
            scoped[["source_family", "model"]]
            .dropna()
            .itertuples(index=False, name=None)
        )
    else:
        required_pairs = {(str(family), str(model)) for family, model in required_pairs}
        pair_index = pd.MultiIndex.from_frame(scoped[["source_family", "model"]].astype(str))
        scoped = scoped[pair_index.isin(pd.MultiIndex.from_tuples(sorted(required_pairs)))].copy()
    if not required_pairs:
        return scoped

    observed_pairs = set(
        scoped[["source_family", "model"]].dropna().itertuples(index=False, name=None)
    )
    missing_pairs = sorted(required_pairs.difference(observed_pairs))
    if missing_pairs:
        raise ValueError(f"Predictions are missing required source/model pairs: {missing_pairs}")

    pair_observation_keys = [
        "source_family",
        "model",
        "stock_id",
        "target_year",
        "target_month",
    ]
    duplicate_mask = scoped.duplicated(pair_observation_keys, keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            f"Predictions contain {duplicate_count} duplicate source/model observation rows."
        )

    observation_keys = ["stock_id", "target_year", "target_month"]
    if "actual_revenue" in scoped.columns:
        actual_conflicts = (
            scoped.groupby(observation_keys, dropna=False)["actual_revenue"]
            .nunique(dropna=True)
            .gt(1)
        )
        if actual_conflicts.any():
            raise ValueError(
                f"Predictions contain conflicting actual_revenue values for "
                f"{int(actual_conflicts.sum())} observations."
            )

    common_keys: set[tuple[object, ...]] | None = None
    for source_family, model in sorted(required_pairs):
        model_keys = set(
            scoped[
                scoped["source_family"].astype(str).eq(source_family)
                & scoped["model"].astype(str).eq(model)
            ][observation_keys].itertuples(index=False, name=None)
        )
        common_keys = model_keys if common_keys is None else common_keys.intersection(model_keys)
    if not common_keys:
        return scoped.iloc[0:0].copy()

    observation_index = pd.MultiIndex.from_frame(scoped[observation_keys])
    return scoped[
        observation_index.isin(pd.MultiIndex.from_tuples(sorted(common_keys)))
    ].copy()


def write_outputs(
    output_dir: Path,
    predictions: pd.DataFrame,
    comparable_predictions: pd.DataFrame,
    overall_accuracy: pd.DataFrame,
    stock_accuracy: pd.DataFrame,
    winner_summary: pd.DataFrame,
    all_attempted_overall_accuracy: pd.DataFrame,
    all_attempted_stock_accuracy: pd.DataFrame,
    all_attempted_winner_summary: pd.DataFrame,
    failures: pd.DataFrame,
    run_config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "monthly_predictions.csv", index=False, encoding="utf-8-sig")
    comparable_predictions.to_csv(
        output_dir / "comparable_monthly_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall_accuracy.to_csv(output_dir / "overall_accuracy.csv", index=False, encoding="utf-8-sig")
    stock_accuracy.to_csv(output_dir / "stock_accuracy.csv", index=False, encoding="utf-8-sig")
    winner_summary.to_csv(output_dir / "winner_summary.csv", index=False, encoding="utf-8-sig")
    all_attempted_overall_accuracy.to_csv(
        output_dir / "all_attempted_overall_accuracy.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_attempted_stock_accuracy.to_csv(
        output_dir / "all_attempted_stock_accuracy.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_attempted_winner_summary.to_csv(
        output_dir / "all_attempted_winner_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    failures.to_csv(output_dir / "failed_runs.csv", index=False, encoding="utf-8-sig")
    write_run_config_and_registry(output_dir, run_config)


def run_benchmark(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    rolling_model_names = parse_str_csv(args.rolling_models)
    ensemble_model_names = parse_str_csv(args.ensemble_models) or []
    explicit_stock_ids = parse_int_csv(args.stock_ids)

    rolling_predictions_all = load_rolling_predictions(
        args.rolling_output_dir,
        target_year=args.target_year,
        model_names=rolling_model_names,
    )
    stock_ids = select_stock_pool(rolling_predictions_all, explicit_stock_ids, args.stock_limit)
    rolling_predictions = rolling_predictions_all[rolling_predictions_all["stock_id"].isin(stock_ids)]

    frames = [rolling_predictions]
    failures = pd.DataFrame(columns=["stock_id", "error_type", "error"])
    if not args.skip_ensemble:
        ensemble_predictions, failures = run_ensemble_predictions(
            stock_ids,
            target_year=args.target_year,
            model_names=ensemble_model_names,
        )
        frames.append(ensemble_predictions)

    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions.sort_values(["stock_id", "target_month", "source_family", "model"])
    required_pairs = {
        ("rolling_lstm", str(model))
        for model in (
            rolling_model_names
            or sorted(rolling_predictions["model"].dropna().astype(str).unique())
        )
    }
    if not args.skip_ensemble:
        required_pairs.update(
            ("ensemble_forecast", str(model)) for model in ensemble_model_names
        )
    comparable_predictions = filter_comparable_predictions(
        predictions,
        required_pairs=required_pairs,
    )
    if comparable_predictions.empty:
        raise ValueError(
            "No identical stock/year/month observations are available across every requested model."
        )
    all_attempted_overall_accuracy = build_overall_accuracy(predictions)
    all_attempted_stock_accuracy = build_stock_accuracy(predictions)
    all_attempted_winner_summary = build_winner_summary(
        all_attempted_stock_accuracy,
        primary_metric=args.primary_metric,
    )
    overall_accuracy = build_overall_accuracy(comparable_predictions)
    stock_accuracy = build_stock_accuracy(comparable_predictions)
    winner_summary = build_winner_summary(stock_accuracy, primary_metric=args.primary_metric)

    run_config = {
        "target_year": args.target_year,
        "primary_metric": args.primary_metric,
        "rolling_output_dir": str(args.rolling_output_dir),
        "output_dir": str(args.output_dir),
        "stock_ids": stock_ids,
        "requested_stock_count": len(stock_ids),
        "all_attempted_stock_count": int(predictions["stock_id"].nunique()),
        "comparable_stock_count": int(comparable_predictions["stock_id"].nunique()),
        "rolling_models": rolling_model_names,
        "ensemble_models": ensemble_model_names,
        "skip_ensemble": bool(args.skip_ensemble),
    }
    run_config = enrich_run_config_from_args(
        run_config,
        args,
        experiment_family="revenue_benchmark",
        report_ready_reason=(
            "Full cross-system benchmark using the available Rolling output pool."
            if args.report_ready or (args.report_ready is None and args.stock_limit is None and not args.skip_ensemble)
            else "Limited, rolling-only, or otherwise diagnostic benchmark run."
        ),
        extra={
            "rolling_output_stock_count": int(rolling_predictions_all["stock_id"].nunique()),
            "selected_stock_count": len(stock_ids),
        },
    )
    write_outputs(
        args.output_dir,
        predictions,
        comparable_predictions,
        overall_accuracy,
        stock_accuracy,
        winner_summary,
        all_attempted_overall_accuracy,
        all_attempted_stock_accuracy,
        all_attempted_winner_summary,
        failures,
        run_config,
    )
    return {
        "monthly_predictions": predictions,
        "comparable_monthly_predictions": comparable_predictions,
        "overall_accuracy": overall_accuracy,
        "stock_accuracy": stock_accuracy,
        "winner_summary": winner_summary,
        "all_attempted_overall_accuracy": all_attempted_overall_accuracy,
        "all_attempted_stock_accuracy": all_attempted_stock_accuracy,
        "all_attempted_winner_summary": all_attempted_winner_summary,
        "failed_runs": failures,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = run_benchmark(args)
    print("Wrote benchmark outputs to", args.output_dir)
    print(results["overall_accuracy"].to_string(index=False))


if __name__ == "__main__":
    main()
