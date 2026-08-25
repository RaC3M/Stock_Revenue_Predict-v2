"""Shared metric aggregation for Rolling batch experiments."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

try:
    from . import rolling_lstm_engine as engine
except ImportError:
    import rolling_lstm_engine as engine


CountOrder = Literal["observations_first", "stock_count_first"]


def metric_record(
    frame: pd.DataFrame,
    *,
    count_order: CountOrder = "observations_first",
) -> dict[str, float | int]:
    actual = frame["actual_revenue"].to_numpy(dtype=float)
    predicted = frame["predicted_revenue"].to_numpy(dtype=float)
    last_observed = (
        frame["last_observed_revenue"].to_numpy(dtype=float)
        if "last_observed_revenue" in frame.columns
        else None
    )
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if last_observed is not None:
        valid &= np.isfinite(last_observed)
    metrics = engine.compute_metrics(actual, predicted, last_observed)
    counts = {
        "observations": int(valid.sum()),
        "stock_count": int(frame.loc[valid, "stock_id"].nunique()) if "stock_id" in frame.columns else 0,
    }
    if count_order == "stock_count_first":
        counts = {"stock_count": counts["stock_count"], "observations": counts["observations"]}
    elif count_order != "observations_first":
        raise ValueError(f"Unknown count order: {count_order}")
    return {**counts, **metrics}


def summarize(
    frame: pd.DataFrame,
    group_columns: list[str],
    *,
    count_order: CountOrder = "observations_first",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in frame.groupby(grouper, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(metric_record(group, count_order=count_order))
        rows.append(row)
    return pd.DataFrame(rows)
