"""Adapter for running Ensemble Forecast predictions in benchmark format."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forecast_benchmark.adapters.ensemble_evidence import (
    build_forecast,
    forecast_year,
    load_revenue_data,
)
from forecast_benchmark.benchmark_config import DEFAULT_ENSEMBLE_MODELS


def _build_last_observed_lookup(revenue_data: pd.DataFrame, target_year: int) -> pd.DataFrame:
    history = revenue_data[["stock_id", "revenue_year", "revenue_month", "revenue_thousand"]].copy()
    history["stock_id"] = pd.to_numeric(history["stock_id"], errors="coerce")
    history["revenue_year"] = pd.to_numeric(history["revenue_year"], errors="coerce")
    history["revenue_month"] = pd.to_numeric(history["revenue_month"], errors="coerce")
    history["revenue_thousand"] = pd.to_numeric(history["revenue_thousand"], errors="coerce")
    history = history.dropna(subset=["stock_id", "revenue_year", "revenue_month", "revenue_thousand"])
    history["stock_id"] = history["stock_id"].astype(int)
    history["revenue_year"] = history["revenue_year"].astype(int)
    history["revenue_month"] = history["revenue_month"].astype(int)
    history["target_year"] = history["revenue_year"]
    history["target_month"] = history["revenue_month"] + 1
    december_mask = history["target_month"].eq(13)
    history.loc[december_mask, "target_year"] = history.loc[december_mask, "target_year"] + 1
    history.loc[december_mask, "target_month"] = 1
    history = history[history["target_year"].eq(int(target_year))]
    return history.rename(columns={"revenue_thousand": "last_observed_revenue"})[
        ["stock_id", "target_year", "target_month", "last_observed_revenue"]
    ]


def run_ensemble_predictions(
    stock_ids: list[int],
    target_year: int,
    model_names: tuple[str, ...] | list[str] = DEFAULT_ENSEMBLE_MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    supported_year = forecast_year()
    if int(target_year) != supported_year:
        raise ValueError(
            f"Ensemble Forecast currently supports FORECAST_YEAR={supported_year}; got {target_year}."
        )

    revenue_data = load_revenue_data()
    last_observed = _build_last_observed_lookup(revenue_data, target_year)
    rows = []
    failures = []
    for stock_id in stock_ids:
        try:
            result = build_forecast(int(stock_id))
        except Exception as error:  # pragma: no cover - covered by integration use.
            failures.append(
                {
                    "stock_id": int(stock_id),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue

        forecast = result.forecast.copy()
        actual = result.backtest[["revenue_month", "actual_revenue"]].drop_duplicates("revenue_month")
        forecast = forecast.merge(actual, on="revenue_month", how="left")
        forecast = forecast.merge(
            last_observed[last_observed["stock_id"].eq(int(stock_id))],
            left_on=["revenue_year", "revenue_month"],
            right_on=["target_year", "target_month"],
            how="left",
        )

        for model_name in model_names:
            if model_name not in forecast.columns:
                continue
            model_frame = forecast[
                [
                    "revenue_year",
                    "revenue_month",
                    model_name,
                    "actual_revenue",
                    "last_observed_revenue",
                ]
            ].copy()
            model_frame = model_frame.rename(
                columns={
                    "revenue_year": "target_year",
                    "revenue_month": "target_month",
                    model_name: "predicted_revenue",
                }
            )
            model_frame["stock_id"] = int(stock_id)
            model_frame["model"] = model_name
            model_frame["source_family"] = "ensemble_forecast"
            model_frame["source_path"] = str(Path("ensemble_forecast") / "forecast_engine.py")
            rows.append(model_frame)

    if rows:
        predictions = pd.concat(rows, ignore_index=True)
    else:
        predictions = pd.DataFrame(
            columns=[
                "source_family",
                "model",
                "stock_id",
                "target_year",
                "target_month",
                "predicted_revenue",
                "actual_revenue",
                "last_observed_revenue",
                "source_path",
            ]
        )

    for optional_column in ["stock_name", "industry_category"]:
        predictions[optional_column] = pd.NA

    predictions = predictions[
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
    ].sort_values(["stock_id", "target_month", "source_family", "model"])

    failures_frame = pd.DataFrame(failures, columns=["stock_id", "error_type", "error"])
    return predictions, failures_frame
