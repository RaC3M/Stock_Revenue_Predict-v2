from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import experiment_metrics
    from .experiment_metadata import write_rolling_run_config
    from .rolling_lstm_engine import (
        get_stock_list,
        load_revenue_data,
        run_rolling_lstm_experiment,
    )
except ImportError:
    import experiment_metrics
    from experiment_metadata import write_rolling_run_config
    from rolling_lstm_engine import (
        get_stock_list,
        load_revenue_data,
        run_rolling_lstm_experiment,
    )


MODEL_COLUMNS = {
    "Rolling LSTM": "predicted_revenue_no_cluster",
    "Rolling LSTM + Cluster": "predicted_revenue_cluster",
    "Rolling LSTM + Cluster + Conditional Adjustment": "predicted_revenue_adjusted",
}

EXPERIMENTS = (
    {"penalty_setting": "off_huber", "use_asymmetric_loss": False, "under_weight": 1.0},
    {"penalty_setting": "on_under_weight_2", "use_asymmetric_loss": True, "under_weight": 2.0},
)

PARAMETERS = {
    "k": 6,
    "window_size": 12,
    "epochs": 35,
    "max_train_samples": 40_000,
    "backend": "torch",
    "enable_growth_adjustment": True,
    "growth_adjustment_alpha": 0.8,
    "enable_conditional_adjustment": True,
    "enable_regime_strategy": True,
}


def metric_record(frame: pd.DataFrame) -> dict[str, float | int]:
    return experiment_metrics.metric_record(frame, count_order="stock_count_first")


def summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return experiment_metrics.summarize(frame, group_columns, count_order="stock_count_first")


def build_impact(summary: pd.DataFrame, index_columns: list[str]) -> pd.DataFrame:
    metrics = [
        "observations",
        "stock_count",
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
    available_metrics = [column for column in metrics if column in summary.columns]
    wide = summary.pivot(index=index_columns, columns="penalty_setting", values=available_metrics)
    wide.columns = [f"{metric}_{setting}" for metric, setting in wide.columns]
    wide = wide.reset_index()
    for metric in [
        "RMSE",
        "MAE",
        "MAPE",
        "MedianAPE",
        "WMAPE",
        "SMAPE",
        "Bias",
        "UnderestimateRate",
        "DirectionAccuracy",
    ]:
        off = f"{metric}_off_huber"
        on = f"{metric}_on_under_weight_2"
        if off in wide.columns and on in wide.columns:
            wide[f"{metric}_delta_on_minus_off"] = wide[on] - wide[off]
    if "MAE_delta_on_minus_off" in wide.columns:
        delta = wide["MAE_delta_on_minus_off"]
        wide["MAE_winner"] = np.select(
            [delta < 0, delta > 0],
            ["penalty_on", "penalty_off"],
            default="tie",
        )
    return wide


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    started = time.time()
    revenue = load_revenue_data()
    available_2025 = set(revenue.loc[revenue["revenue_year"].eq(2025), "stock_id"].astype(int))
    stock_ids = [stock_id for stock_id in get_stock_list(revenue) if stock_id in available_2025]
    stock_meta = (
        revenue.sort_values("date")
        .groupby("stock_id", as_index=False)
        .agg(
            industry_category=("industry_category", "last"),
            available_months_2025=("revenue_year", lambda values: int((values == 2025).sum())),
        )
    )
    stock_list_path = Path(__file__).resolve().parent.parent / "data" / "stock_list_new.csv"
    if stock_list_path.exists():
        names = pd.read_csv(stock_list_path, usecols=["stock_id", "stock_name"])
        names["stock_id"] = pd.to_numeric(names["stock_id"], errors="coerce")
        names = names.dropna(subset=["stock_id"]).drop_duplicates("stock_id")
        names["stock_id"] = names["stock_id"].astype(int)
        stock_meta = stock_meta.merge(names, on="stock_id", how="left")
    else:
        stock_meta["stock_name"] = ""
    stock_meta["industry_category"] = stock_meta["industry_category"].fillna("未分類")
    stock_meta["stock_name"] = stock_meta["stock_name"].fillna("")
    meta_lookup = stock_meta.set_index("stock_id")

    all_forecasts: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    print(f"Eligible stocks with 2025 data: {len(stock_ids)}", flush=True)

    for experiment in EXPERIMENTS:
        setting = str(experiment["penalty_setting"])
        experiment_started = time.time()
        print(f"Starting experiment: {setting}", flush=True)
        for position, stock_id in enumerate(stock_ids, start=1):
            try:
                result = run_rolling_lstm_experiment(
                    selected_stock=stock_id,
                    **PARAMETERS,
                    use_asymmetric_loss=bool(experiment["use_asymmetric_loss"]),
                    under_weight=float(experiment["under_weight"]),
                )
                forecast = result.forecast.copy()
                forecast.insert(0, "penalty_setting", setting)
                forecast["stock_id"] = stock_id
                forecast.insert(2, "stock_name", meta_lookup.at[stock_id, "stock_name"])
                forecast.insert(3, "industry_category", meta_lookup.at[stock_id, "industry_category"])
                forecast["use_asymmetric_loss"] = bool(experiment["use_asymmetric_loss"])
                forecast["under_weight"] = float(experiment["under_weight"])
                all_forecasts.append(forecast)
            except Exception as error:
                failures.append(
                    {
                        "penalty_setting": setting,
                        "stock_id": stock_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            if position == 1 or position % 100 == 0 or position == len(stock_ids):
                elapsed = time.time() - experiment_started
                print(
                    f"{setting}: {position}/{len(stock_ids)} stocks processed, "
                    f"failures={sum(item['penalty_setting'] == setting for item in failures)}, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    monthly = pd.concat(all_forecasts, ignore_index=True)
    monthly["target_date"] = pd.to_datetime(monthly["target_date"])
    monthly = monthly.sort_values(["penalty_setting", "stock_id", "target_date"]).reset_index(drop=True)

    id_columns = [
        "penalty_setting",
        "stock_id",
        "stock_name",
        "industry_category",
        "target_date",
        "target_month",
        "actual_revenue",
        "last_observed_revenue",
        "regime",
        "cluster",
    ]
    long_parts = []
    for model, prediction_column in MODEL_COLUMNS.items():
        part = monthly[id_columns].copy()
        part["model"] = model
        part["predicted_revenue"] = monthly[prediction_column].to_numpy()
        long_parts.append(part)
    long_predictions = pd.concat(long_parts, ignore_index=True)

    stock_accuracy = summarize(
        long_predictions,
        ["penalty_setting", "stock_id", "stock_name", "industry_category", "model"],
    )
    dominant_regime = (
        monthly.loc[monthly["penalty_setting"].eq("off_huber")]
        .groupby("stock_id")["regime"]
        .agg(lambda values: values.value_counts().index[0])
        .rename("dominant_regime_2025")
        .reset_index()
    )
    stock_accuracy = stock_accuracy.merge(dominant_regime, on="stock_id", how="left")

    overall_accuracy = summarize(long_predictions, ["penalty_setting", "model"])
    industry_accuracy = summarize(
        long_predictions,
        ["penalty_setting", "industry_category", "model"],
    )
    regime_accuracy = summarize(long_predictions, ["penalty_setting", "regime", "model"])
    cluster_accuracy = summarize(long_predictions, ["penalty_setting", "cluster", "model"])

    stock_impact = build_impact(
        stock_accuracy,
        ["stock_id", "stock_name", "industry_category", "dominant_regime_2025", "model"],
    )
    overall_impact = build_impact(overall_accuracy, ["model"])
    industry_impact = build_impact(industry_accuracy, ["industry_category", "model"])
    regime_impact = build_impact(regime_accuracy, ["regime", "model"])

    completed_stock_counts = (
        monthly.groupby("penalty_setting")["stock_id"].nunique().rename("completed_stocks").reset_index()
    )
    failure_frame = pd.DataFrame(failures, columns=["penalty_setting", "stock_id", "error_type", "error"])
    run_summary = {
        "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": round(time.time() - started, 3),
        "eligible_stocks": len(stock_ids),
        "completed_stocks_by_setting": completed_stock_counts.to_dict("records"),
        "failures": len(failure_frame),
        "monthly_rows": len(monthly),
        "parameters": PARAMETERS,
        "experiments": list(EXPERIMENTS),
        "metric_notes": {
            "revenue_unit": "thousand TWD (same as revenue_thousand source)",
            "MAPE": "mean absolute percentage error across nonzero actual revenue rows",
            "WMAPE": "sum absolute error divided by sum absolute actual revenue",
            "UnderestimateRate": "percentage of predictions below actual revenue",
            "DirectionAccuracy": "percentage with correct sign versus last observed month",
            "impact_delta": "penalty on minus penalty off; negative error delta is improvement",
            "penalty_comparison": "off uses Huber loss; on uses asymmetric squared loss with under_weight=2.0",
        },
    }

    selected_monthly_columns = [
        "penalty_setting",
        "stock_id",
        "stock_name",
        "industry_category",
        "target_date",
        "target_month",
        "actual_revenue",
        "last_observed_revenue",
        "regime",
        "cluster",
        "growth_ratio",
        "growth_streak",
        "is_growth_phase",
        "adjustment_applied",
        "prediction_cap",
        "predicted_revenue_no_cluster",
        "predicted_revenue_cluster",
        "predicted_revenue_adjusted",
        "no_cluster_error",
        "cluster_error",
        "adjusted_error",
        "use_asymmetric_loss",
        "under_weight",
    ]
    write_csv(monthly[selected_monthly_columns], output_dir / "monthly_predictions.csv")
    write_csv(stock_accuracy, output_dir / "stock_accuracy.csv")
    write_csv(stock_impact, output_dir / "stock_penalty_impact.csv")
    write_csv(overall_accuracy, output_dir / "overall_accuracy.csv")
    write_csv(overall_impact, output_dir / "overall_penalty_impact.csv")
    write_csv(industry_accuracy, output_dir / "industry_accuracy.csv")
    write_csv(industry_impact, output_dir / "industry_penalty_impact.csv")
    write_csv(regime_accuracy, output_dir / "regime_accuracy.csv")
    write_csv(regime_impact, output_dir / "regime_penalty_impact.csv")
    write_csv(cluster_accuracy, output_dir / "cluster_accuracy.csv")
    write_csv(failure_frame, output_dir / "failed_stocks.csv")
    run_summary = write_rolling_run_config(
        output_dir,
        run_summary,
        experiment_family="rolling_asymmetric_penalty",
        evidence_tier="C",
        selection_protocol="target-year-hindsight",
        report_ready=False,
        report_ready_reason="Penalty settings are compared and ranked on target-year actuals.",
    )

    print(json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
