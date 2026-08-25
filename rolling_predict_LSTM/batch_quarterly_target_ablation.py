from __future__ import annotations

"""Compare rolling monthly forecasts with direct 3-month revenue targets.

Direct-quarter experiments train on `past 12 months -> next 3 months total`.
The monthly-sum benchmark aggregates the existing one-month model outputs and is
therefore a rolling-updated reference, not a strict quarter-ahead forecast.
"""

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from . import rolling_lstm_engine as engine
    from .experiment_metrics import metric_record, summarize
    from .experiment_metadata import write_rolling_run_config
except ImportError:
    import rolling_lstm_engine as engine
    from experiment_metrics import metric_record, summarize
    from experiment_metadata import write_rolling_run_config


WINDOW_SIZE = 12
HORIZON_MONTHS = 3
FIXED_K = 6
FIXED_EPOCHS = 35
FIXED_MAX_TRAIN_SAMPLES = 40_000
UNDER_WEIGHT = 2.0
GROWTH_ALPHA = 0.8

TargetFamily = Literal["monthly_sum", "direct_3m"]


@dataclass(frozen=True)
class QuarterlySpec:
    experiment_id: str
    experiment_name: str
    target_family: TargetFamily
    include_cluster: bool = True
    use_asymmetric_loss: bool = True
    under_weight: float = UNDER_WEIGHT
    growth_enabled: bool = False
    growth_conditional: bool = True
    growth_regime_strategy: bool = True


DEFAULT_QUARTERLY_SPECS: tuple[QuarterlySpec, ...] = (
    QuarterlySpec(
        experiment_id="MS00",
        experiment_name="rolling monthly sum: cluster + growth",
        target_family="monthly_sum",
        growth_enabled=True,
    ),
    QuarterlySpec(
        experiment_id="MS01",
        experiment_name="rolling monthly sum: cluster raw",
        target_family="monthly_sum",
    ),
    QuarterlySpec(
        experiment_id="MS02",
        experiment_name="rolling monthly sum: no cluster raw",
        target_family="monthly_sum",
        include_cluster=False,
    ),
    QuarterlySpec(
        experiment_id="Q00",
        experiment_name="direct 3m target: cluster raw",
        target_family="direct_3m",
    ),
    QuarterlySpec(
        experiment_id="Q01",
        experiment_name="direct 3m target: no cluster raw",
        target_family="direct_3m",
        include_cluster=False,
    ),
    QuarterlySpec(
        experiment_id="Q02",
        experiment_name="direct 3m target: cluster huber",
        target_family="direct_3m",
        use_asymmetric_loss=False,
        under_weight=1.0,
    ),
    QuarterlySpec(
        experiment_id="Q03",
        experiment_name="direct 3m target: cluster + growth",
        target_family="direct_3m",
        growth_enabled=True,
    ),
)

EFFECT_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("E01", "Direct raw vs rolling monthly-sum growth", "MS00", "Q00"),
    ("E02", "Direct growth vs rolling monthly-sum growth", "MS00", "Q03"),
    ("E03", "Direct no-cluster vs direct cluster raw", "Q00", "Q01"),
    ("E04", "Direct huber vs direct cluster raw", "Q00", "Q02"),
    ("E05", "Direct growth vs direct cluster raw", "Q00", "Q03"),
    ("E06", "Monthly cluster raw vs monthly cluster growth", "MS00", "MS01"),
    ("E07", "Monthly no-cluster raw vs monthly cluster growth", "MS00", "MS02"),
)


@dataclass(frozen=True)
class QuarterlyContext:
    revenue: pd.DataFrame
    monthly: pd.DataFrame
    train_samples: list[dict[str, object]]
    eval_samples: list[dict[str, object]]
    actual_revenue: pd.DataFrame
    cluster_count: int
    stock_meta: pd.DataFrame
    selected_stock_ids: list[int]
    revenue_mtime_ns: int


def parse_stock_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def clear_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def safe_period_label(start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
    return f"{start_date:%Y-%m}~{end_date:%Y-%m}"


def rolling_sum(values: pd.Series, horizon: int) -> pd.Series:
    return values.rolling(horizon, min_periods=horizon).sum()


def build_stock_meta(revenue: pd.DataFrame) -> pd.DataFrame:
    target_year_revenue = revenue[revenue["revenue_year"].astype(int).eq(engine.FORECAST_YEAR)]
    actual_2025 = (
        target_year_revenue.groupby("stock_id", as_index=False)["revenue_thousand"]
        .sum()
        .rename(columns={"revenue_thousand": "actual_2025_revenue"})
    )
    stock_meta = (
        revenue.sort_values("date")
        .groupby("stock_id", as_index=False)
        .agg(industry_category=("industry_category", "last"))
        .merge(actual_2025, on="stock_id", how="left")
    )
    stock_meta["actual_2025_revenue"] = stock_meta["actual_2025_revenue"].fillna(0.0)
    stock_list_path = Path(__file__).resolve().parent.parent / "data" / "stock_list_new.csv"
    if stock_list_path.exists():
        names = pd.read_csv(stock_list_path, usecols=["stock_id", "stock_name"])
        names["stock_id"] = pd.to_numeric(names["stock_id"], errors="coerce")
        names = names.dropna(subset=["stock_id"]).drop_duplicates("stock_id")
        names["stock_id"] = names["stock_id"].astype(int)
        stock_meta = stock_meta.merge(names, on="stock_id", how="left")
    else:
        stock_meta["stock_name"] = ""
    stock_meta["industry_category"] = stock_meta["industry_category"].fillna("unknown")
    stock_meta["stock_name"] = stock_meta["stock_name"].fillna("")
    return stock_meta


def select_stock_ids(
    revenue: pd.DataFrame,
    requested_stock_ids: set[int] | None,
    stock_limit: int | None,
) -> list[int]:
    available = sorted(
        revenue.loc[revenue["revenue_year"].eq(engine.FORECAST_YEAR), "stock_id"].dropna().astype(int).unique()
    )
    if requested_stock_ids is not None:
        available = [stock_id for stock_id in available if stock_id in requested_stock_ids]
    if stock_limit:
        available = available[: int(stock_limit)]
    return [int(stock_id) for stock_id in available]


def enrich_quarterly_monthly_frame(monthly: pd.DataFrame, horizon: int) -> pd.DataFrame:
    enriched = monthly.sort_values(["stock_id", "date"]).copy()
    group_keys = (
        ["stock_id", "_calendar_segment"]
        if "_calendar_segment" in enriched.columns
        else ["stock_id"]
    )
    grouped = enriched.groupby(group_keys, group_keys=False)
    enriched["trailing_period_revenue"] = grouped["revenue_thousand"].transform(lambda values: rolling_sum(values, horizon))
    enriched["sequence_max_period_revenue"] = grouped["trailing_period_revenue"].transform(
        lambda values: values.rolling(WINDOW_SIZE, min_periods=1).max()
    )
    return enriched


def build_quarterly_sequences_for_stock(
    stock_df: pd.DataFrame,
    selected_stock: int,
    *,
    window_size: int = WINDOW_SIZE,
    horizon: int = HORIZON_MONTHS,
    train_end_year: int = engine.TRAIN_END_YEAR,
    eval_year: int = engine.FORECAST_YEAR,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    stock_df = stock_df[stock_df["stock_id"].astype(int).eq(int(selected_stock))].sort_values("date").reset_index(drop=True)
    train_samples: list[dict[str, object]] = []
    eval_samples: list[dict[str, object]] = []
    if len(stock_df) < window_size + horizon:
        return train_samples, eval_samples

    for end_idx in range(window_size - 1, len(stock_df) - horizon):
        target_start_idx = end_idx + 1
        target_end_idx = end_idx + horizon
        full_period = stock_df.iloc[end_idx - window_size + 1 : target_end_idx + 1]
        if not engine._months_are_consecutive(full_period):
            continue
        sequence_frame = stock_df.iloc[end_idx - window_size + 1 : end_idx + 1]
        target_frame = stock_df.iloc[target_start_idx : target_end_idx + 1]
        target_start = pd.Timestamp(stock_df.loc[target_start_idx, "date"])
        target_end = pd.Timestamp(stock_df.loc[target_end_idx, "date"])
        target_start_year = int(stock_df.loc[target_start_idx, "revenue_year"])
        target_end_year = int(stock_df.loc[target_end_idx, "revenue_year"])
        target_revenue = float(target_frame["revenue_thousand"].sum())
        trailing_start_idx = end_idx - horizon + 1
        if trailing_start_idx < 0:
            continue
        trailing_frame = stock_df.iloc[trailing_start_idx : end_idx + 1]
        trailing_period_revenue = float(trailing_frame["revenue_thousand"].sum())
        historical_period = stock_df["revenue_thousand"].rolling(horizon, min_periods=horizon).sum()
        sequence_max_period_revenue = float(historical_period.iloc[end_idx - window_size + 1 : end_idx + 1].max())

        sample = {
            "stock_id": int(selected_stock),
            "sequence_frame": sequence_frame,
            "cluster": int(stock_df.loc[end_idx, "cluster"]),
            "sequence_start_date": stock_df.loc[end_idx - window_size + 1, "date"],
            "sequence_end_date": stock_df.loc[end_idx, "date"],
            "target_date": target_start,
            "target_year": target_start_year,
            "target_month": int(stock_df.loc[target_start_idx, "revenue_month"]),
            "target_end_date": target_end,
            "target_end_year": target_end_year,
            "target_end_month": int(stock_df.loc[target_end_idx, "revenue_month"]),
            "target_period_label": safe_period_label(target_start, target_end),
            "horizon_months": int(horizon),
            "last_observed_period_revenue": trailing_period_revenue,
            "last_observed_month_revenue": float(stock_df.loc[end_idx, "revenue_thousand"]),
            "sequence_max_period_revenue": sequence_max_period_revenue,
            "quarter_start_month": int(stock_df.loc[target_start_idx, "revenue_month"]),
        }
        if target_end_year <= train_end_year:
            train_samples.append({**sample, "target_revenue": target_revenue})
        elif target_start_year == eval_year and target_end_year == eval_year:
            eval_samples.append(sample)

    return train_samples, eval_samples


def build_quarterly_sequences(
    monthly: pd.DataFrame,
    selected_stock_ids: list[int],
    *,
    window_size: int,
    horizon: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_samples: list[dict[str, object]] = []
    eval_samples: list[dict[str, object]] = []
    selected = set(int(stock_id) for stock_id in selected_stock_ids)
    for stock_id, stock_df in monthly.groupby("stock_id", sort=False):
        train, eval_ = build_quarterly_sequences_for_stock(
            stock_df,
            int(stock_id),
            window_size=window_size,
            horizon=horizon,
        )
        train_samples.extend(train)
        if int(stock_id) in selected:
            eval_samples.extend(eval_)
    return train_samples, eval_samples


def extra_eval_metadata(samples: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        rows.append(
            {
                "target_end_date": sample["target_end_date"],
                "target_end_year": sample["target_end_year"],
                "target_end_month": sample["target_end_month"],
                "target_period_label": sample["target_period_label"],
                "horizon_months": sample["horizon_months"],
                "last_observed_period_revenue": sample["last_observed_period_revenue"],
                "last_observed_month_revenue": sample["last_observed_month_revenue"],
                "sequence_max_period_revenue": sample["sequence_max_period_revenue"],
                "quarter_start_month": sample["quarter_start_month"],
            }
        )
    return pd.DataFrame(rows)


def make_quarterly_arrays(
    samples: list[dict[str, object]],
    numeric_scaler: StandardScaler,
    target_scaler: StandardScaler,
    *,
    cluster_count: int,
    include_cluster: bool,
    fit_scalers: bool,
    require_target: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x, y, metadata = engine.make_lstm_arrays(
        samples,
        numeric_scaler,
        target_scaler,
        cluster_count=cluster_count,
        include_cluster=include_cluster,
        fit_scalers=fit_scalers,
        require_target=require_target,
    )
    if not fit_scalers:
        extra = extra_eval_metadata(samples)
        metadata = pd.concat([metadata.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        metadata["last_observed_revenue"] = metadata["last_observed_period_revenue"]
        metadata["sequence_max_revenue"] = metadata["sequence_max_period_revenue"]
    return x, y, metadata


def build_actual_quarter_frame(eval_samples: list[dict[str, object]], revenue: pd.DataFrame) -> pd.DataFrame:
    lookup = revenue.set_index(["stock_id", "date"])["revenue_thousand"]
    rows = []
    for sample in eval_samples:
        start = pd.Timestamp(sample["target_date"])
        end = pd.Timestamp(sample["target_end_date"])
        dates = pd.date_range(start, end, freq="MS")
        actual_values = [
            float(lookup.loc[(int(sample["stock_id"]), date)])
            for date in dates
            if (int(sample["stock_id"]), date) in lookup.index
        ]
        if len(actual_values) != int(sample["horizon_months"]):
            continue
        rows.append(
            {
                "stock_id": int(sample["stock_id"]),
                "target_year": int(sample["target_year"]),
                "target_month": int(sample["target_month"]),
                "actual_revenue": float(sum(actual_values)),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(["stock_id", "target_year", "target_month"], keep="last")


def prepare_context(
    *,
    k: int,
    window_size: int,
    horizon: int,
    max_train_samples: int,
    requested_stock_ids: set[int] | None,
    stock_limit: int | None,
) -> QuarterlyContext:
    revenue_mtime_ns = engine._revenue_file_mtime_ns()
    revenue, _ = engine._cached_revenue_and_windows(window_size, revenue_mtime_ns)
    _, clustered_windows, _, monthly = engine._cached_clustered_artifacts(k, window_size, revenue_mtime_ns)
    monthly = enrich_quarterly_monthly_frame(monthly, horizon)
    selected_stock_ids = select_stock_ids(revenue, requested_stock_ids, stock_limit)
    if not selected_stock_ids:
        raise ValueError("No 2025 stocks are available for quarterly evaluation.")

    train_samples, eval_samples = build_quarterly_sequences(
        monthly,
        selected_stock_ids,
        window_size=window_size,
        horizon=horizon,
    )
    if not train_samples:
        raise ValueError("No quarterly training samples are available.")
    if not eval_samples:
        raise ValueError("No quarterly 2025 evaluation samples are available.")
    capped_train_samples, _ = engine.cap_training_samples(train_samples, max_train_samples=max_train_samples, seed=42)
    return QuarterlyContext(
        revenue=revenue,
        monthly=monthly,
        train_samples=capped_train_samples,
        eval_samples=eval_samples,
        actual_revenue=build_actual_quarter_frame(eval_samples, revenue),
        cluster_count=int(clustered_windows["cluster"].max()) + 1,
        stock_meta=build_stock_meta(revenue),
        selected_stock_ids=selected_stock_ids,
        revenue_mtime_ns=revenue_mtime_ns,
    )


def apply_quarterly_growth(prediction: np.ndarray, metadata: pd.DataFrame, spec: QuarterlySpec) -> np.ndarray:
    guarded, _, _ = engine.apply_revenue_guardrails(prediction, metadata)
    if not spec.growth_enabled:
        return guarded
    adjusted, _, _, _, _, _ = engine.apply_growth_adjustment(
        guarded,
        metadata,
        alpha=GROWTH_ALPHA,
        enable_growth_adjustment=True,
        enable_conditional_adjustment=spec.growth_conditional,
        enable_regime_strategy=spec.growth_regime_strategy,
    )
    adjusted, _, _ = engine.apply_revenue_guardrails(adjusted, metadata)
    return adjusted


def finalize_prediction(prediction: np.ndarray, metadata: pd.DataFrame, context: QuarterlyContext, spec: QuarterlySpec) -> pd.DataFrame:
    forecast = metadata.copy()
    forecast["predicted_revenue"] = engine.safe_round_revenue(prediction)
    evaluated = engine.attach_actual_revenue(forecast, context.actual_revenue)
    evaluated["error"] = evaluated["predicted_revenue"] - evaluated["actual_revenue"]
    evaluated["abs_error"] = evaluated["error"].abs()
    evaluated["predicted_return"] = evaluated["predicted_revenue"] / evaluated["last_observed_revenue"] - 1
    evaluated["actual_return"] = evaluated["actual_revenue"] / evaluated["last_observed_revenue"] - 1
    evaluated["underestimated"] = evaluated["predicted_revenue"] < evaluated["actual_revenue"]
    evaluated["direction_correct"] = np.sign(
        evaluated["predicted_revenue"] - evaluated["last_observed_revenue"]
    ) == np.sign(evaluated["actual_revenue"] - evaluated["last_observed_revenue"])
    evaluated["regime"] = engine.classify_regime(evaluated)
    evaluated["growth_phase"] = engine.calculate_growth_phase(evaluated)
    evaluated["experiment_id"] = spec.experiment_id
    evaluated["experiment_name"] = spec.experiment_name
    evaluated["target_family"] = spec.target_family
    evaluated["include_cluster"] = spec.include_cluster
    evaluated["use_asymmetric_loss"] = spec.use_asymmetric_loss
    evaluated["growth_enabled"] = spec.growth_enabled
    evaluated["growth_conditional"] = spec.growth_conditional
    evaluated["growth_regime_strategy"] = spec.growth_regime_strategy
    return evaluated


def run_direct_quarterly_spec(context: QuarterlyContext, spec: QuarterlySpec, *, epochs: int) -> pd.DataFrame:
    started = time.perf_counter()
    numeric_scaler = StandardScaler()
    target_scaler = StandardScaler()
    x_train, y_train, _ = make_quarterly_arrays(
        context.train_samples,
        numeric_scaler,
        target_scaler,
        cluster_count=context.cluster_count,
        include_cluster=spec.include_cluster,
        fit_scalers=True,
        require_target=True,
    )
    x_eval, _, metadata = make_quarterly_arrays(
        context.eval_samples,
        numeric_scaler,
        target_scaler,
        cluster_count=context.cluster_count,
        include_cluster=spec.include_cluster,
        fit_scalers=False,
        require_target=False,
    )
    prediction, backend = engine.train_predict_lstm(
        x_train,
        y_train,
        x_eval,
        target_scaler,
        epochs=epochs,
        use_asymmetric_loss=spec.use_asymmetric_loss,
        under_weight=spec.under_weight,
    )
    prediction = apply_quarterly_growth(prediction, metadata, spec)
    result = finalize_prediction(prediction, metadata, context, spec)
    result["backend"] = backend
    result["duration_sec"] = round(time.perf_counter() - started, 3)
    result["benchmark_note"] = "direct quarter-ahead target"
    return result


def build_monthly_metadata_for_aggregation(monthly_rows: pd.DataFrame, context: QuarterlyContext) -> pd.DataFrame:
    monthly = context.monthly[
        [
            "stock_id",
            "date",
            "trailing_period_revenue",
            "sequence_max_period_revenue",
            "revenue_thousand",
        ]
    ].copy()
    monthly = monthly.rename(
        columns={
            "date": "target_date",
            "trailing_period_revenue": "last_observed_period_revenue",
            "sequence_max_period_revenue": "sequence_max_period_revenue",
            "revenue_thousand": "last_observed_month_revenue",
        }
    )
    monthly["target_date"] = monthly["target_date"] + pd.DateOffset(months=1)
    return monthly_rows.merge(monthly, on=["stock_id", "target_date"], how="left")


def aggregate_monthly_predictions(monthly_predictions: pd.DataFrame, spec: QuarterlySpec, context: QuarterlyContext) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = set(context.selected_stock_ids)
    frame = monthly_predictions[monthly_predictions["stock_id"].isin(selected)].copy()
    frame = frame.sort_values(["stock_id", "target_date"])
    for stock_id, stock_frame in frame.groupby("stock_id", sort=False):
        stock_frame = stock_frame.reset_index(drop=True)
        for start_idx in range(0, len(stock_frame) - HORIZON_MONTHS + 1):
            chunk = stock_frame.iloc[start_idx : start_idx + HORIZON_MONTHS]
            if int(chunk.iloc[0]["target_year"]) != engine.FORECAST_YEAR:
                continue
            if int(chunk.iloc[-1]["target_year"]) != engine.FORECAST_YEAR:
                continue
            expected = pd.date_range(chunk.iloc[0]["target_date"], periods=HORIZON_MONTHS, freq="MS")
            if list(pd.to_datetime(chunk["target_date"])) != list(expected):
                continue
            first = chunk.iloc[0]
            last = chunk.iloc[-1]
            monthly_abs_sum = float(chunk["monthly_abs_error"].sum())
            quarter_prediction = float(chunk["predicted_revenue"].sum())
            quarter_actual = float(chunk["actual_revenue"].sum())
            quarter_abs_error = abs(quarter_prediction - quarter_actual)
            rows.append(
                {
                    "stock_id": int(stock_id),
                    "input_start_date": first["input_start_date"],
                    "input_end_date": first["input_end_date"],
                    "target_date": first["target_date"],
                    "target_year": int(first["target_year"]),
                    "target_month": int(first["target_month"]),
                    "target_end_date": last["target_date"],
                    "target_end_year": int(last["target_year"]),
                    "target_end_month": int(last["target_month"]),
                    "target_period_label": safe_period_label(pd.Timestamp(first["target_date"]), pd.Timestamp(last["target_date"])),
                    "horizon_months": HORIZON_MONTHS,
                    "quarter_start_month": int(first["target_month"]),
                    "last_observed_revenue": float(first["last_observed_period_revenue"]),
                    "last_observed_period_revenue": float(first["last_observed_period_revenue"]),
                    "last_observed_month_revenue": float(first["last_observed_month_revenue"]),
                    "sequence_max_revenue": float(first["sequence_max_period_revenue"]),
                    "sequence_max_period_revenue": float(first["sequence_max_period_revenue"]),
                    "cluster": int(first["cluster"]),
                    "growth_rate_at_end": float(first["growth_rate_at_end"]),
                    "momentum_3m_at_end": float(first["momentum_3m_at_end"]),
                    "momentum_6m_at_end": float(first["momentum_6m_at_end"]),
                    "growth_ratio": float(first["growth_ratio"]),
                    "growth_streak": int(first["growth_streak"]),
                    "trend_component": float(first["trend_component"]),
                    "cycle_component": float(first["cycle_component"]),
                    "cycle_volatility_6m": float(first["cycle_volatility_6m"]),
                    "trend_slope": float(first["trend_slope"]),
                    "trend_slope_rate": float(first["trend_slope_rate"]),
                    "predicted_revenue": engine.safe_round_revenue(np.array([quarter_prediction]))[0],
                    "actual_revenue": quarter_actual,
                    "error": quarter_prediction - quarter_actual,
                    "abs_error": quarter_abs_error,
                    "source_monthly_abs_error_sum": monthly_abs_sum,
                    "error_cancelled_pct": (1.0 - (quarter_abs_error / monthly_abs_sum)) * 100 if monthly_abs_sum else np.nan,
                }
            )
    result = pd.DataFrame(rows)
    result["predicted_return"] = result["predicted_revenue"] / result["last_observed_revenue"] - 1
    result["actual_return"] = result["actual_revenue"] / result["last_observed_revenue"] - 1
    result["underestimated"] = result["predicted_revenue"] < result["actual_revenue"]
    result["direction_correct"] = np.sign(
        result["predicted_revenue"] - result["last_observed_revenue"]
    ) == np.sign(result["actual_revenue"] - result["last_observed_revenue"])
    result["regime"] = engine.classify_regime(result)
    result["growth_phase"] = engine.calculate_growth_phase(result)
    result["experiment_id"] = spec.experiment_id
    result["experiment_name"] = spec.experiment_name
    result["target_family"] = spec.target_family
    result["include_cluster"] = spec.include_cluster
    result["use_asymmetric_loss"] = spec.use_asymmetric_loss
    result["growth_enabled"] = spec.growth_enabled
    result["growth_conditional"] = spec.growth_conditional
    result["growth_regime_strategy"] = spec.growth_regime_strategy
    result["backend"] = "monthly rolling LSTM aggregation"
    result["duration_sec"] = 0.0
    result["benchmark_note"] = "rolling-updated monthly predictions summed over 3 months"
    return result


def build_monthly_sum_predictions(context: QuarterlyContext, specs: tuple[QuarterlySpec, ...], *, k: int, epochs: int, max_train_samples: int) -> pd.DataFrame:
    monthly_specs = [spec for spec in specs if spec.target_family == "monthly_sum"]
    if not monthly_specs:
        return pd.DataFrame()
    raw_frame, backend_cluster, backend_plain, _, _ = engine._cached_lstm_predictions(
        k,
        WINDOW_SIZE,
        max_train_samples,
        epochs,
        True,
        UNDER_WEIGHT,
        False,
        context.revenue_mtime_ns,
    )
    actual_monthly = engine.build_actual_revenue_frame(context.revenue, target_year=engine.FORECAST_YEAR)
    raw_frame = engine.attach_actual_revenue(raw_frame, actual_monthly)
    raw_frame = raw_frame[raw_frame["stock_id"].isin(context.selected_stock_ids)].copy()
    raw_frame["target_date"] = pd.to_datetime(raw_frame["target_date"])
    raw_frame = build_monthly_metadata_for_aggregation(raw_frame, context)

    prediction_parts: list[pd.DataFrame] = []
    metadata = raw_frame.drop(columns=["raw_pred_cluster", "raw_pred_plain", "raw_pred_trend", "raw_pred_cycle"], errors="ignore").copy()
    for spec in monthly_specs:
        if spec.include_cluster:
            raw_prediction = raw_frame["raw_pred_cluster"].to_numpy(dtype=float)
            backend = backend_cluster
        else:
            raw_prediction = raw_frame["raw_pred_plain"].to_numpy(dtype=float)
            backend = backend_plain
        monthly_prediction = apply_quarterly_growth(raw_prediction, metadata, spec)
        part = raw_frame.copy()
        part["predicted_revenue"] = engine.safe_round_revenue(monthly_prediction)
        part["monthly_error"] = part["predicted_revenue"] - part["actual_revenue"]
        part["monthly_abs_error"] = part["monthly_error"].abs()
        aggregated = aggregate_monthly_predictions(part, spec, context)
        aggregated["backend"] = backend if spec.include_cluster else backend_plain
        prediction_parts.append(aggregated)
    return pd.concat(prediction_parts, ignore_index=True)


def build_stock_type(quarter_predictions: pd.DataFrame, stock_meta: pd.DataFrame) -> pd.DataFrame:
    baseline = quarter_predictions[quarter_predictions["experiment_id"].eq("MS00")]
    if baseline.empty:
        baseline = quarter_predictions[quarter_predictions["experiment_id"].eq("Q00")]
    stock_type = (
        baseline.sort_values(["stock_id", "target_date"])
        .groupby("stock_id")
        .agg(
            evaluated_windows=("target_date", "size"),
            cycle_windows=("regime", lambda values: int((values == "cycle").sum())),
            growth_windows=("regime", lambda values: int((values == "growth").sum())),
            decline_windows=("regime", lambda values: int((values == "decline").sum())),
        )
        .reset_index()
    )
    for regime in ("cycle", "growth", "decline"):
        stock_type[f"{regime}_share"] = stock_type[f"{regime}_windows"] / stock_type["evaluated_windows"]
    share_columns = ["cycle_share", "growth_share", "decline_share"]
    stock_type["dominant_regime"] = stock_type[share_columns].idxmax(axis=1).str.replace("_share", "", regex=False)
    stock_type["regime_confidence"] = stock_type[share_columns].max(axis=1)
    stock_type = stock_type.merge(
        stock_meta[["stock_id", "stock_name", "industry_category", "actual_2025_revenue"]],
        on="stock_id",
        how="left",
    )
    if len(stock_type) >= 4:
        stock_type["revenue_size_quartile"] = pd.qcut(
            stock_type["actual_2025_revenue"].rank(method="first"),
            4,
            labels=["Q1_small", "Q2", "Q3", "Q4_large"],
        )
    else:
        stock_type["revenue_size_quartile"] = "all"
    return stock_type


def build_pair_effects(summary: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metrics = [
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
    rows: list[dict[str, object]] = []
    if group_columns:
        indexed = summary.set_index(["experiment_id", *group_columns])
    else:
        indexed = summary.set_index("experiment_id")
    for effect_id, effect_name, baseline_id, treatment_id in EFFECT_PAIRS:
        treatment_rows = summary[summary["experiment_id"].eq(treatment_id)]
        for treatment_row in treatment_rows.itertuples(index=False):
            key = tuple(getattr(treatment_row, column) for column in group_columns)
            baseline_key = (baseline_id, *key) if group_columns else baseline_id
            if baseline_key not in indexed.index:
                continue
            baseline = indexed.loc[baseline_key]
            output = {
                "effect_id": effect_id,
                "effect_name": effect_name,
                "baseline_id": baseline_id,
                "treatment_id": treatment_id,
                "treatment_name": treatment_row.experiment_name,
            }
            output.update(dict(zip(group_columns, key)))
            for metric in metrics:
                output[f"{metric}_base"] = float(baseline[metric])
                output[f"{metric}_treatment"] = float(getattr(treatment_row, metric))
                output[f"{metric}_delta"] = output[f"{metric}_treatment"] - output[f"{metric}_base"]
            output["MAE_pct_change"] = (
                (output["MAE_treatment"] / output["MAE_base"] - 1) * 100 if output["MAE_base"] else np.nan
            )
            output["WMAPE_pct_change"] = (
                (output["WMAPE_treatment"] / output["WMAPE_base"] - 1) * 100 if output["WMAPE_base"] else np.nan
            )
            rows.append(output)
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=FIXED_K)
    parser.add_argument("--epochs", type=int, default=FIXED_EPOCHS)
    parser.add_argument("--max-train-samples", type=int, default=FIXED_MAX_TRAIN_SAMPLES)
    parser.add_argument("--horizon", type=int, default=HORIZON_MONTHS)
    parser.add_argument("--stock-ids", default="")
    parser.add_argument("--stock-limit", type=int, default=0)
    parser.add_argument("--skip-predictions", action="store_true")
    args = parser.parse_args()

    if int(args.horizon) != HORIZON_MONTHS:
        raise ValueError("This experiment currently supports --horizon 3 only.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    specs = DEFAULT_QUARTERLY_SPECS
    context = prepare_context(
        k=int(args.k),
        window_size=WINDOW_SIZE,
        horizon=int(args.horizon),
        max_train_samples=int(args.max_train_samples),
        requested_stock_ids=parse_stock_ids(args.stock_ids),
        stock_limit=int(args.stock_limit) or None,
    )

    prediction_parts: list[pd.DataFrame] = []
    print("Running rolling monthly-sum benchmark", flush=True)
    prediction_parts.append(
        build_monthly_sum_predictions(
            context,
            specs,
            k=int(args.k),
            epochs=int(args.epochs),
            max_train_samples=int(args.max_train_samples),
        )
    )
    direct_specs = [spec for spec in specs if spec.target_family == "direct_3m"]
    for position, spec in enumerate(direct_specs, start=1):
        print(f"[{position}/{len(direct_specs)}] Running {spec.experiment_id} {spec.experiment_name}", flush=True)
        prediction_parts.append(run_direct_quarterly_spec(context, spec, epochs=int(args.epochs)))
        clear_torch_cache()

    predictions = pd.concat([part for part in prediction_parts if not part.empty], ignore_index=True)
    for column in ["target_date", "target_end_date", "input_start_date", "input_end_date"]:
        if column in predictions.columns:
            predictions[column] = pd.to_datetime(predictions[column])
    stock_type = build_stock_type(predictions, context.stock_meta)
    predictions = predictions.merge(
        stock_type[
            [
                "stock_id",
                "stock_name",
                "industry_category",
                "dominant_regime",
                "regime_confidence",
                "actual_2025_revenue",
                "revenue_size_quartile",
            ]
        ],
        on="stock_id",
        how="left",
        suffixes=("", "_stock"),
    )

    overall = summarize(predictions, ["experiment_id", "experiment_name", "target_family"])
    target_family_accuracy = summarize(predictions, ["target_family", "experiment_id", "experiment_name"])
    dominant_regime_accuracy = summarize(
        predictions,
        ["experiment_id", "experiment_name", "target_family", "dominant_regime"],
    )
    size_accuracy = summarize(
        predictions,
        ["experiment_id", "experiment_name", "target_family", "revenue_size_quartile"],
    )
    stock_accuracy = summarize(
        predictions,
        ["experiment_id", "experiment_name", "target_family", "stock_id", "stock_name", "industry_category"],
    )
    industry_accuracy = summarize(
        predictions,
        ["experiment_id", "experiment_name", "target_family", "industry_category"],
    )
    overall_effects = build_pair_effects(overall, [])
    dominant_regime_effects = build_pair_effects(dominant_regime_accuracy, ["dominant_regime"])
    size_effects = build_pair_effects(size_accuracy, ["revenue_size_quartile"])
    stock_effects = build_pair_effects(stock_accuracy, ["stock_id", "stock_name", "industry_category"])

    cycle_necessity = dominant_regime_effects[
        dominant_regime_effects["dominant_regime"].eq("cycle")
        & dominant_regime_effects["effect_id"].isin(["E01", "E02"])
    ].copy()
    cycle_necessity["direct_quarterly_beats_monthly_sum"] = cycle_necessity["WMAPE_delta"] < 0

    experiment_config = pd.DataFrame([asdict(spec) for spec in specs])
    run_config = {
        "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": round(time.time() - started, 3),
        "k": int(args.k),
        "epochs": int(args.epochs),
        "max_train_samples": int(args.max_train_samples),
        "horizon_months": int(args.horizon),
        "stock_ids": [int(stock_id) for stock_id in context.selected_stock_ids],
        "train_samples_used": len(context.train_samples),
        "eval_samples": len(context.eval_samples),
        "method_note": (
            "Direct 3M uses past 12 months to predict the next 3-month revenue sum. "
            "MS experiments aggregate rolling one-month predictions and may use actual updates inside the quarter."
        ),
    }

    write_csv(experiment_config, output_dir / "quarterly_ablation_config.csv")
    write_csv(overall, output_dir / "overall_accuracy.csv")
    write_csv(target_family_accuracy, output_dir / "target_family_accuracy.csv")
    write_csv(overall_effects, output_dir / "overall_effects.csv")
    write_csv(dominant_regime_accuracy, output_dir / "dominant_regime_accuracy.csv")
    write_csv(dominant_regime_effects, output_dir / "dominant_regime_effects.csv")
    write_csv(size_accuracy, output_dir / "revenue_size_accuracy.csv")
    write_csv(size_effects, output_dir / "revenue_size_effects.csv")
    write_csv(stock_accuracy, output_dir / "stock_accuracy.csv")
    write_csv(stock_effects, output_dir / "stock_effects.csv")
    write_csv(industry_accuracy, output_dir / "industry_accuracy.csv")
    write_csv(stock_type, output_dir / "stock_type.csv")
    write_csv(cycle_necessity, output_dir / "cycle_necessity_summary.csv")
    if not args.skip_predictions:
        write_csv(predictions, output_dir / "quarter_predictions.csv")
    run_config = write_rolling_run_config(
        output_dir,
        run_config,
        experiment_family="rolling_quarterly_target",
        evidence_tier="A",
        selection_protocol="fixed-before-target",
        report_ready=True,
        report_ready_reason="Monthly-sum and direct-3M variants were fixed before target-year scoring.",
    )

    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
