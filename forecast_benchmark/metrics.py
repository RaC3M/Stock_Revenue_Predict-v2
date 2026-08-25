"""Shared metrics for comparing forecast systems."""

from __future__ import annotations

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "MSE",
    "RMSE",
    "MAE",
    "MAPE",
    "MedianAPE",
    "WMAPE",
    "SMAPE",
    "Bias",
    "UnderestimateRate",
    "DirectionAccuracy",
]


def compute_metrics(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    last_observed: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    empty_metrics = {column: np.nan for column in METRIC_COLUMNS}
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual_values) & np.isfinite(predicted_values)

    last_observed_values = None
    if last_observed is not None:
        last_observed_values = np.asarray(last_observed, dtype=float)
        if last_observed_values.shape != actual_values.shape:
            raise ValueError("last_observed must have the same shape as actual.")
        valid &= np.isfinite(last_observed_values)

    if not valid.any():
        return empty_metrics

    actual_values = actual_values[valid]
    predicted_values = predicted_values[valid]
    if last_observed_values is not None:
        last_observed_values = last_observed_values[valid]

    error = predicted_values - actual_values
    abs_error = np.abs(error)
    nonzero_actual = actual_values != 0
    absolute_percentage_error = np.divide(
        abs_error,
        np.abs(actual_values),
        out=np.full_like(abs_error, np.nan, dtype=float),
        where=nonzero_actual,
    )
    wmape_denominator = float(np.abs(actual_values).sum())
    smape_denominator = np.abs(actual_values) + np.abs(predicted_values)
    smape_terms = np.divide(
        2.0 * abs_error,
        smape_denominator,
        out=np.full_like(abs_error, np.nan, dtype=float),
        where=smape_denominator != 0,
    )

    if last_observed_values is None:
        direction_accuracy = np.nan
    else:
        direction_accuracy = float(
            np.mean(
                np.sign(predicted_values - last_observed_values)
                == np.sign(actual_values - last_observed_values)
            )
            * 100
        )

    return {
        "MSE": float(np.mean(error**2)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(abs_error)),
        "MAPE": (
            float(np.nanmean(absolute_percentage_error) * 100)
            if np.isfinite(absolute_percentage_error).any()
            else np.nan
        ),
        "MedianAPE": (
            float(np.nanmedian(absolute_percentage_error) * 100)
            if np.isfinite(absolute_percentage_error).any()
            else np.nan
        ),
        "WMAPE": float(abs_error.sum() / wmape_denominator * 100) if wmape_denominator else np.nan,
        "SMAPE": float(np.nanmean(smape_terms) * 100) if np.isfinite(smape_terms).any() else np.nan,
        "Bias": float(np.mean(error)),
        "UnderestimateRate": float(np.mean(predicted_values < actual_values) * 100),
        "DirectionAccuracy": direction_accuracy,
    }


def build_accuracy_frame(
    predictions: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for group_key, group in predictions.groupby(group_columns, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_columns, group_key, strict=True))
        metrics = compute_metrics(
            group["actual_revenue"],
            group["predicted_revenue"],
            group["last_observed_revenue"] if "last_observed_revenue" in group.columns else None,
        )
        row.update(
            {
                "observations": int(group[["actual_revenue", "predicted_revenue"]].dropna().shape[0]),
                "stock_count": int(group["stock_id"].nunique()) if "stock_id" in group.columns else np.nan,
                **metrics,
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    for column in METRIC_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").round(3)
    return result


def build_overall_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    return build_accuracy_frame(predictions, ["source_family", "model"]).sort_values(
        ["WMAPE", "MAPE", "source_family", "model"],
        na_position="last",
    )


def build_stock_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    label_columns = [
        column
        for column in ["stock_id", "stock_name", "industry_category", "source_family", "model"]
        if column in predictions.columns
    ]
    return build_accuracy_frame(predictions, label_columns).sort_values(
        ["stock_id", "WMAPE", "MAPE", "source_family", "model"],
        na_position="last",
    )


def build_winner_summary(
    stock_accuracy: pd.DataFrame,
    primary_metric: str = "WMAPE",
) -> pd.DataFrame:
    if stock_accuracy.empty:
        return pd.DataFrame(
            columns=[
                "source_family",
                "model",
                "compared_stocks",
                "stock_wins",
                "stock_win_rate",
                f"average_{primary_metric}",
                f"median_{primary_metric}",
            ]
        )
    if primary_metric not in stock_accuracy.columns:
        raise ValueError(f"Unknown primary metric: {primary_metric}")

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
        .agg(**{f"average_{primary_metric}": "mean", f"median_{primary_metric}": "median"})
    )
    summary = metric_summary.merge(winner_counts, on=["source_family", "model"], how="left")
    summary["stock_wins"] = summary["stock_wins"].fillna(0).astype(int)
    summary["compared_stocks"] = compared_stocks
    summary["stock_win_rate"] = np.where(
        compared_stocks > 0,
        summary["stock_wins"] / compared_stocks * 100,
        np.nan,
    )
    numeric_columns = [f"average_{primary_metric}", f"median_{primary_metric}", "stock_win_rate"]
    for column in numeric_columns:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").round(3)
    return summary.sort_values(
        ["stock_wins", f"average_{primary_metric}", "source_family", "model"],
        ascending=[False, True, True, True],
    )

