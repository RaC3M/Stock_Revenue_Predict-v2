from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

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
FIXED_K = 6
FIXED_EPOCHS = 35
FIXED_MAX_TRAIN_SAMPLES = 40_000
GROWTH_ALPHA = 0.8
UNDER_WEIGHT = 2.0
TREND_SLOPE_BETA = 0.35
MAX_VOLATILITY_SCALE = 2.5
TUNING_YEAR = 2024
TUNING_EPOCHS = 8
K_CANDIDATES = (4, 6, 8)
SAMPLE_CANDIDATES = (10_000, 40_000)


SCENARIOS = [
    {
        "scenario_id": "S01",
        "scenario_name": "Fixed Base",
        "auto_tune": False,
        "tuning_family": "fixed",
        "growth": False,
        "trend_cycle": False,
        "asymmetric_loss": False,
    },
    {
        "scenario_id": "S02",
        "scenario_name": "Fixed + Growth",
        "auto_tune": False,
        "tuning_family": "fixed",
        "growth": True,
        "trend_cycle": False,
        "asymmetric_loss": True,
    },
    {
        "scenario_id": "S03",
        "scenario_name": "Fixed + Growth + TrendCycle",
        "auto_tune": False,
        "tuning_family": "fixed",
        "growth": True,
        "trend_cycle": True,
        "asymmetric_loss": True,
    },
    {
        "scenario_id": "S06",
        "scenario_name": "AutoTune Base",
        "auto_tune": True,
        "tuning_family": "base",
        "growth": False,
        "trend_cycle": False,
        "asymmetric_loss": False,
    },
    {
        "scenario_id": "S07",
        "scenario_name": "AutoTune + Growth",
        "auto_tune": True,
        "tuning_family": "growth_static",
        "growth": True,
        "trend_cycle": False,
        "asymmetric_loss": True,
    },
    {
        "scenario_id": "S08",
        "scenario_name": "AutoTune + Growth + TrendCycle",
        "auto_tune": True,
        "tuning_family": "growth_static",
        "growth": True,
        "trend_cycle": True,
        "asymmetric_loss": True,
    },
]


EFFECT_PAIRS = [
    ("E01", "Growth package on fixed model", "S01", "S02"),
    ("E02", "TrendCycle on fixed model", "S02", "S03"),
    ("E06", "Auto tuning on base model", "S01", "S06"),
    ("E07", "Auto tuning on growth model", "S02", "S07"),
    ("E08", "TrendCycle on auto-tuned model", "S07", "S08"),
]


def build_all_year_sequences(
    monthly: pd.DataFrame,
    train_end_year: int,
    eval_year: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_samples: list[dict[str, object]] = []
    eval_samples: list[dict[str, object]] = []
    for stock_id, stock_df in monthly.groupby("stock_id", sort=False):
        stock_df = stock_df.sort_values("date").reset_index(drop=True)
        if len(stock_df) <= WINDOW_SIZE:
            continue
        for end_idx in range(WINDOW_SIZE - 1, len(stock_df) - 1):
            target_idx = end_idx + 1
            full_window = stock_df.iloc[end_idx - WINDOW_SIZE + 1 : target_idx + 1]
            if not engine._months_are_consecutive(full_window):
                continue
            target_year = int(stock_df.loc[target_idx, "revenue_year"])
            target_revenue = float(stock_df.loc[target_idx, "revenue_thousand"])
            if not np.isfinite(target_revenue) or target_revenue < 0:
                continue
            sequence_frame = stock_df.iloc[end_idx - WINDOW_SIZE + 1 : end_idx + 1]
            sample = {
                "stock_id": int(stock_id),
                "sequence_frame": sequence_frame,
                "cluster": int(stock_df.loc[end_idx, "cluster"]),
                "sequence_start_date": stock_df.loc[end_idx - WINDOW_SIZE + 1, "date"],
                "sequence_end_date": stock_df.loc[end_idx, "date"],
                "target_date": stock_df.loc[target_idx, "date"],
                "target_year": target_year,
                "target_month": int(stock_df.loc[target_idx, "revenue_month"]),
                "target_revenue": target_revenue,
                "target_trend": float(stock_df.loc[target_idx, "trend_component"]),
                "target_cycle": float(stock_df.loc[target_idx, "cycle_component"]),
            }
            if target_year <= train_end_year:
                train_samples.append(sample)
            elif target_year == eval_year:
                eval_samples.append(sample)
    return train_samples, eval_samples


def clear_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def apply_candidate_postprocessing(
    prediction: np.ndarray,
    metadata: pd.DataFrame,
    *,
    growth: bool,
) -> np.ndarray:
    guarded, _, _ = engine.apply_revenue_guardrails(
        prediction,
        metadata,
    )
    if not growth:
        return guarded
    adjusted, _, _, _, _, _ = engine.apply_growth_adjustment(
        guarded,
        metadata,
        alpha=GROWTH_ALPHA,
        enable_growth_adjustment=True,
        enable_conditional_adjustment=True,
        enable_regime_strategy=True,
    )
    adjusted, _, _ = engine.apply_revenue_guardrails(
        adjusted,
        metadata,
    )
    return adjusted


def candidate_stock_metrics(
    metadata: pd.DataFrame,
    prediction: np.ndarray,
    *,
    family: str,
    k: int,
    max_train_samples: int,
) -> list[dict[str, object]]:
    working = metadata[
        ["stock_id", "actual_revenue", "last_observed_revenue", "target_date"]
    ].copy()
    working["predicted_revenue"] = engine.safe_round_revenue(prediction)
    rows: list[dict[str, object]] = []
    for stock_id, stock_frame in working.groupby("stock_id", sort=False):
        metrics = metric_record(stock_frame)
        rows.append(
            {
                "tuning_family": family,
                "stock_id": int(stock_id),
                "k": int(k),
                "max_train_samples": int(max_train_samples),
                "tuning_year": TUNING_YEAR,
                "tuning_epochs": TUNING_EPOCHS,
                **metrics,
            }
        )
    return rows


def run_efficient_tuning(
    revenue: pd.DataFrame,
    windows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tuning_rows: list[dict[str, object]] = []
    base_train_samples: list[dict[str, object]] | None = None
    base_eval_samples: list[dict[str, object]] | None = None
    for k in K_CANDIDATES:
        print(f"Tuning K={k}: preparing 2023 train / 2024 validation samples", flush=True)
        _, clustered_windows, _ = engine.fit_kmeans_clusters(windows, k=k, train_end_year=TUNING_YEAR - 1)
        cluster_lookup = {
            (int(row.stock_id), pd.Timestamp(row.window_end_date)): int(row.cluster)
            for row in clustered_windows[
                ["stock_id", "window_end_date", "cluster"]
            ].itertuples(index=False)
        }
        if base_train_samples is None or base_eval_samples is None:
            monthly = engine.attach_clusters_to_monthly(revenue, clustered_windows)
            train_samples, eval_samples = build_all_year_sequences(
                monthly,
                train_end_year=TUNING_YEAR - 1,
                eval_year=TUNING_YEAR,
            )
            base_train_samples = train_samples
            base_eval_samples = eval_samples
        else:
            train_samples = [
                {
                    **sample,
                    "cluster": cluster_lookup.get(
                        (int(sample["stock_id"]), pd.Timestamp(sample["sequence_end_date"])),
                        0,
                    ),
                }
                for sample in base_train_samples
            ]
            eval_samples = [
                {
                    **sample,
                    "cluster": cluster_lookup.get(
                        (int(sample["stock_id"]), pd.Timestamp(sample["sequence_end_date"])),
                        0,
                    ),
                }
                for sample in base_eval_samples
            ]
        cluster_count = int(clustered_windows["cluster"].max()) + 1
        print(
            f"Tuning K={k}: train_samples={len(train_samples):,}, validation_samples={len(eval_samples):,}",
            flush=True,
        )
        for sample_limit in SAMPLE_CANDIDATES:
            sampled_train, _ = engine.cap_training_samples(train_samples, sample_limit)
            numeric_scaler = StandardScaler()
            target_scaler = StandardScaler()
            x_train, y_train, _ = engine.make_lstm_arrays(
                sampled_train,
                numeric_scaler,
                target_scaler,
                cluster_count=cluster_count,
                include_cluster=True,
                fit_scalers=True,
            )
            x_eval, _, eval_meta = engine.make_lstm_arrays(
                eval_samples,
                numeric_scaler,
                target_scaler,
                cluster_count=cluster_count,
                include_cluster=True,
                fit_scalers=False,
            )
            for loss_name, asymmetric in (("huber", False), ("asymmetric_2", True)):
                started = time.perf_counter()
                raw_prediction, backend_used = engine.train_predict_lstm(
                    x_train,
                    y_train,
                    x_eval,
                    target_scaler,
                    epochs=TUNING_EPOCHS,
                    use_asymmetric_loss=asymmetric,
                    under_weight=UNDER_WEIGHT if asymmetric else 1.0,
                )
                families = (
                    [("base", False, False)]
                    if not asymmetric
                    else [
                        ("growth_static", True, False),
                    ]
                )
                for family, growth, _ in families:
                    prediction = apply_candidate_postprocessing(
                        raw_prediction,
                        eval_meta,
                        growth=growth,
                    )
                    tuning_rows.extend(
                        candidate_stock_metrics(
                            eval_meta,
                            prediction,
                            family=family,
                            k=k,
                            max_train_samples=sample_limit,
                        )
                    )
                print(
                    f"Tuning candidate K={k}, samples={sample_limit:,}, loss={loss_name}, "
                    f"backend={backend_used}, elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
                clear_torch_cache()
            del x_train, y_train, x_eval, eval_meta
            clear_torch_cache()
        del train_samples, eval_samples, clustered_windows
        clear_torch_cache()

    tuning_results = pd.DataFrame(tuning_rows)
    selected = (
        tuning_results.sort_values(
            ["tuning_family", "stock_id", "MAE", "MAPE", "max_train_samples"],
            kind="stable",
        )
        .groupby(["tuning_family", "stock_id"], as_index=False)
        .first()
    )
    selected["tuning_status"] = "auto_selected"
    return tuning_results, selected


def finalize_one_stock(
    stock_frame: pd.DataFrame,
    scenario: dict[str, object],
) -> pd.DataFrame:
    metadata = stock_frame.drop(
        columns=["raw_pred_cluster", "raw_pred_plain", "raw_pred_trend", "raw_pred_cycle"]
    ).copy()
    raw_base = stock_frame["raw_pred_cluster"].to_numpy(dtype=float)
    base_prediction, _, cap = engine.apply_revenue_guardrails(
        raw_base,
        metadata,
    )
    regime = engine.classify_regime(metadata)
    trend_applied = np.zeros(len(metadata), dtype=bool)
    trend_cycle_prediction = base_prediction.copy()
    if bool(scenario["trend_cycle"]):
        trend_cycle_prediction, _, _, _, trend_applied = engine.apply_trend_cycle_adjustment(
            base_prediction,
            stock_frame["raw_pred_trend"].to_numpy(dtype=float),
            stock_frame["raw_pred_cycle"].to_numpy(dtype=float),
            metadata,
            regime,
            enable_trend_cycle_model=True,
            trend_slope_beta=TREND_SLOPE_BETA,
            max_volatility_scale=MAX_VOLATILITY_SCALE,
        )
        trend_cycle_prediction, _, _ = engine.apply_revenue_guardrails(
            trend_cycle_prediction,
            metadata,
        )

    growth_applied = np.zeros(len(metadata), dtype=bool)
    final_prediction = trend_cycle_prediction.copy()
    if bool(scenario["growth"]):
        final_prediction, _, _, regime, _, growth_applied = engine.apply_growth_adjustment(
            trend_cycle_prediction,
            metadata,
            alpha=GROWTH_ALPHA,
            enable_growth_adjustment=True,
            enable_conditional_adjustment=True,
            enable_regime_strategy=True,
        )
        final_prediction, _, cap = engine.apply_revenue_guardrails(
            final_prediction,
            metadata,
        )

    result = metadata[
        [
            "stock_id",
            "target_date",
            "target_month",
            "actual_revenue",
            "last_observed_revenue",
            "cluster",
            "growth_ratio",
            "growth_streak",
        ]
    ].copy()
    result["regime"] = regime
    result["growth_adjustment_applied"] = growth_applied
    result["trend_cycle_applied"] = trend_applied
    result["prediction_cap"] = cap
    result["predicted_revenue"] = engine.safe_round_revenue(final_prediction)
    result["error"] = result["predicted_revenue"] - result["actual_revenue"]
    result["abs_error"] = result["error"].abs()
    result["underestimated"] = result["predicted_revenue"] < result["actual_revenue"]
    result["direction_correct"] = np.sign(
        result["predicted_revenue"] - result["last_observed_revenue"]
    ) == np.sign(result["actual_revenue"] - result["last_observed_revenue"])
    return result


def finalize_raw_frame(
    raw_frame: pd.DataFrame,
    scenario: dict[str, object],
) -> pd.DataFrame:
    pieces = [
        finalize_one_stock(stock_frame, scenario)
        for _, stock_frame in raw_frame.groupby("stock_id", sort=False)
    ]
    return pd.concat(pieces, ignore_index=True)


def add_scenario_columns(
    frame: pd.DataFrame,
    scenario: dict[str, object],
    *,
    k: int,
    sample_limit: int,
    tuning_status: str,
) -> pd.DataFrame:
    frame.insert(0, "scenario_id", str(scenario["scenario_id"]))
    frame.insert(1, "scenario_name", str(scenario["scenario_name"]))
    frame["selected_k"] = int(k)
    frame["selected_max_train_samples"] = int(sample_limit)
    frame["tuning_status"] = tuning_status
    frame["auto_tune"] = bool(scenario["auto_tune"])
    frame["growth_enabled"] = bool(scenario["growth"])
    frame["trend_cycle_enabled"] = bool(scenario["trend_cycle"])
    return frame


def build_scenario_predictions(
    scenario: dict[str, object],
    selection: pd.DataFrame,
    revenue_mtime_ns: int,
) -> pd.DataFrame:
    revenue = engine._cached_revenue_and_windows(WINDOW_SIZE, revenue_mtime_ns)[0]
    actual_revenue = engine.build_actual_revenue_frame(revenue, target_year=engine.FORECAST_YEAR)
    if not bool(scenario["auto_tune"]):
        raw_frame = engine._cached_lstm_predictions(
            FIXED_K,
            WINDOW_SIZE,
            FIXED_MAX_TRAIN_SAMPLES,
            FIXED_EPOCHS,
            bool(scenario["asymmetric_loss"]),
            UNDER_WEIGHT if bool(scenario["asymmetric_loss"]) else 1.0,
            bool(scenario["trend_cycle"]),
            revenue_mtime_ns,
        )[0]
        raw_frame = engine.attach_actual_revenue(raw_frame, actual_revenue)
        finalized = finalize_raw_frame(raw_frame, scenario)
        return add_scenario_columns(
            finalized,
            scenario,
            k=FIXED_K,
            sample_limit=FIXED_MAX_TRAIN_SAMPLES,
            tuning_status="fixed",
        )

    family = str(scenario["tuning_family"])
    family_selection = selection[selection["tuning_family"].eq(family)].copy()
    pieces: list[pd.DataFrame] = []
    for (k, sample_limit), selected_group in family_selection.groupby(
        ["k", "max_train_samples"],
        sort=True,
    ):
        stock_ids = set(selected_group["stock_id"].astype(int))
        raw_frame = engine._cached_lstm_predictions(
            int(k),
            WINDOW_SIZE,
            int(sample_limit),
            FIXED_EPOCHS,
            bool(scenario["asymmetric_loss"]),
            UNDER_WEIGHT if bool(scenario["asymmetric_loss"]) else 1.0,
            bool(scenario["trend_cycle"]),
            revenue_mtime_ns,
        )[0]
        raw_subset = raw_frame[raw_frame["stock_id"].isin(stock_ids)].copy()
        if raw_subset.empty:
            continue
        raw_subset = engine.attach_actual_revenue(raw_subset, actual_revenue)
        finalized = finalize_raw_frame(raw_subset, scenario)
        status_lookup = selected_group.set_index("stock_id")["tuning_status"]
        finalized = add_scenario_columns(
            finalized,
            scenario,
            k=int(k),
            sample_limit=int(sample_limit),
            tuning_status="auto_selected",
        )
        finalized["tuning_status"] = finalized["stock_id"].map(status_lookup).fillna("auto_selected")
        pieces.append(finalized)
    if not pieces:
        raise ValueError(f"No predictions were assembled for {scenario['scenario_id']}.")
    return pd.concat(pieces, ignore_index=True)


def build_parameter_effects(
    summary: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
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
    extra_groups = [column for column in group_columns if column != "scenario_id"]
    for effect_id, effect_name, base_id, treatment_id in EFFECT_PAIRS:
        if extra_groups:
            base = summary[summary["scenario_id"].eq(base_id)].set_index(extra_groups)
            treatment = summary[summary["scenario_id"].eq(treatment_id)].set_index(extra_groups)
            pairs = [
                (key, base.loc[key], treatment.loc[key])
                for key in base.index.intersection(treatment.index)
            ]
        else:
            base_rows = summary[summary["scenario_id"].eq(base_id)]
            treatment_rows = summary[summary["scenario_id"].eq(treatment_id)]
            pairs = (
                [((), base_rows.iloc[0], treatment_rows.iloc[0])]
                if not base_rows.empty and not treatment_rows.empty
                else []
            )
        for key, base_row, treatment_row in pairs:
            row: dict[str, object] = {
                "effect_id": effect_id,
                "effect_name": effect_name,
                "base_scenario": base_id,
                "treatment_scenario": treatment_id,
            }
            keys = key if isinstance(key, tuple) else (key,)
            row.update(dict(zip(extra_groups, keys)))
            for metric in metrics:
                row[f"{metric}_base"] = float(base_row[metric])
                row[f"{metric}_treatment"] = float(treatment_row[metric])
                row[f"{metric}_delta"] = float(treatment_row[metric] - base_row[metric])
            row["MAE_pct_change"] = (
                float((treatment_row["MAE"] / base_row["MAE"] - 1) * 100)
                if base_row["MAE"]
                else np.nan
            )
            row["WMAPE_pct_change"] = (
                float((treatment_row["WMAPE"] / base_row["WMAPE"] - 1) * 100)
                if base_row["WMAPE"]
                else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    revenue_mtime_ns = engine._revenue_file_mtime_ns()
    revenue, windows = engine._cached_revenue_and_windows(WINDOW_SIZE, revenue_mtime_ns)
    eligible_2025 = sorted(
        revenue.loc[revenue["revenue_year"].eq(2025), "stock_id"].dropna().astype(int).unique()
    )
    print(f"2025 stocks in source data: {len(eligible_2025)}", flush=True)

    stock_meta = (
        revenue.sort_values("date")
        .groupby("stock_id", as_index=False)
        .agg(industry_category=("industry_category", "last"))
    )
    stock_list_path = Path(__file__).resolve().parent.parent / "data" / "stock_list_new.csv"
    names = pd.read_csv(stock_list_path, usecols=["stock_id", "stock_name"])
    names["stock_id"] = pd.to_numeric(names["stock_id"], errors="coerce")
    names = names.dropna(subset=["stock_id"]).drop_duplicates("stock_id")
    names["stock_id"] = names["stock_id"].astype(int)
    stock_meta = stock_meta.merge(names, on="stock_id", how="left")
    stock_meta["stock_name"] = stock_meta["stock_name"].fillna("")
    stock_meta["industry_category"] = stock_meta["industry_category"].fillna("未分類")

    tuning_checkpoint = output_dir / "tuning_candidate_results.csv"
    selected_checkpoint = output_dir / "tuning_selected_core.csv"
    if tuning_checkpoint.exists() and selected_checkpoint.exists():
        print("Loading completed tuning checkpoints", flush=True)
        tuning_results = pd.read_csv(tuning_checkpoint)
        selected = pd.read_csv(selected_checkpoint)
    else:
        tuning_results, selected = run_efficient_tuning(revenue, windows)
        write_csv(tuning_results, tuning_checkpoint)
        write_csv(selected, selected_checkpoint)
    prediction_stock_ids = set(
        engine._cached_lstm_predictions(
            FIXED_K,
            WINDOW_SIZE,
            FIXED_MAX_TRAIN_SAMPLES,
            FIXED_EPOCHS,
            False,
            1.0,
            False,
            revenue_mtime_ns,
        )[0]["stock_id"].astype(int)
    )
    selection_pieces = []
    for family in ("base", "growth_static"):
        family_selection = selected[selected["tuning_family"].eq(family)].copy()
        missing = prediction_stock_ids - set(family_selection["stock_id"].astype(int))
        if missing:
            fallback = pd.DataFrame(
                {
                    "tuning_family": family,
                    "stock_id": sorted(missing),
                    "k": FIXED_K,
                    "max_train_samples": FIXED_MAX_TRAIN_SAMPLES,
                    "tuning_year": TUNING_YEAR,
                    "tuning_epochs": TUNING_EPOCHS,
                    "tuning_status": "fallback_no_2024_validation",
                }
            )
            family_selection = pd.concat([family_selection, fallback], ignore_index=True)
        selection_pieces.append(family_selection)
    selected_all = pd.concat(selection_pieces, ignore_index=True)
    write_csv(selected_all, output_dir / "tuning_selected_params.csv")

    execution_ids = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S09", "S08", "S10"]
    scenarios_by_id = {str(scenario["scenario_id"]): scenario for scenario in SCENARIOS}
    for scenario_id in execution_ids:
        scenario = scenarios_by_id[scenario_id]
        scenario_checkpoint = output_dir / f"monthly_{scenario_id}.csv"
        if scenario_checkpoint.exists():
            print(f"Loading completed checkpoint for {scenario_id}", flush=True)
            continue
        scenario_started = time.time()
        print(f"Running {scenario['scenario_id']} {scenario['scenario_name']}", flush=True)
        predictions = build_scenario_predictions(scenario, selected_all, revenue_mtime_ns)
        predictions = predictions.merge(stock_meta, on="stock_id", how="left")
        write_csv(predictions, scenario_checkpoint)
        print(
            f"Completed {scenario['scenario_id']}: stocks={predictions['stock_id'].nunique()}, "
            f"rows={len(predictions):,}, elapsed={time.time() - scenario_started:.1f}s",
            flush=True,
        )
        clear_torch_cache()

    monthly = pd.concat(
        [pd.read_csv(output_dir / f"monthly_{scenario['scenario_id']}.csv") for scenario in SCENARIOS],
        ignore_index=True,
    )
    monthly["target_date"] = pd.to_datetime(monthly["target_date"])
    monthly = monthly.sort_values(["scenario_id", "stock_id", "target_date"]).reset_index(drop=True)

    scenario_config = pd.DataFrame(SCENARIOS)
    scenario_config["fixed_k"] = np.where(scenario_config["auto_tune"], np.nan, FIXED_K)
    scenario_config["fixed_epochs"] = FIXED_EPOCHS
    scenario_config["fixed_max_train_samples"] = np.where(
        scenario_config["auto_tune"],
        np.nan,
        FIXED_MAX_TRAIN_SAMPLES,
    )
    scenario_config["growth_alpha"] = np.where(scenario_config["growth"], GROWTH_ALPHA, 0.0)
    scenario_config["under_weight"] = np.where(
        scenario_config["asymmetric_loss"],
        UNDER_WEIGHT,
        1.0,
    )
    scenario_config["trend_slope_beta"] = np.where(
        scenario_config["trend_cycle"],
        TREND_SLOPE_BETA,
        0.0,
    )
    scenario_config["max_volatility_scale"] = np.where(
        scenario_config["trend_cycle"],
        MAX_VOLATILITY_SCALE,
        1.0,
    )
    scenario_config["base_static_guardrail_note"] = (
        "All scenarios retain the engine's static 5x last / 4x sequence cap."
    )

    overall = summarize(monthly, ["scenario_id", "scenario_name"])
    overall["WMAPE_rank"] = overall["WMAPE"].rank(method="min").astype(int)
    overall["MAE_rank"] = overall["MAE"].rank(method="min").astype(int)
    overall = overall.sort_values(["WMAPE_rank", "MAE_rank"]).reset_index(drop=True)
    stock_accuracy = summarize(
        monthly,
        ["scenario_id", "scenario_name", "stock_id", "stock_name", "industry_category"],
    )
    regime_accuracy = summarize(monthly, ["scenario_id", "scenario_name", "regime"])
    industry_accuracy = summarize(
        monthly,
        ["scenario_id", "scenario_name", "industry_category"],
    )

    stock_type = (
        monthly[monthly["scenario_id"].eq("S01")]
        .sort_values(["stock_id", "target_date"])
        .groupby("stock_id")
        .agg(
            evaluated_months=("target_date", "size"),
            cycle_months=("regime", lambda values: int((values == "cycle").sum())),
            growth_months=("regime", lambda values: int((values == "growth").sum())),
            decline_months=("regime", lambda values: int((values == "decline").sum())),
            regime_transitions=("regime", lambda values: int((values != values.shift()).sum() - 1)),
            actual_2025_revenue=("actual_revenue", "sum"),
        )
        .reset_index()
    )
    for regime in ("cycle", "growth", "decline"):
        stock_type[f"{regime}_share"] = (
            stock_type[f"{regime}_months"] / stock_type["evaluated_months"]
        )
    share_columns = ["cycle_share", "growth_share", "decline_share"]
    stock_type["dominant_regime"] = (
        stock_type[share_columns].idxmax(axis=1).str.replace("_share", "", regex=False)
    )
    stock_type["regime_confidence"] = stock_type[share_columns].max(axis=1)
    stock_type["revenue_size_quartile"] = pd.qcut(
        stock_type["actual_2025_revenue"].rank(method="first"),
        4,
        labels=["Q1_small", "Q2", "Q3", "Q4_large"],
    )
    stock_type = stock_type.merge(stock_meta, on="stock_id", how="left")

    best_scenario = (
        stock_accuracy.sort_values(["stock_id", "WMAPE", "MAE", "scenario_id"], kind="stable")
        .groupby("stock_id", as_index=False)
        .first()
        .rename(
            columns={
                "scenario_id": "best_scenario_id",
                "scenario_name": "best_scenario_name",
                "WMAPE": "best_WMAPE",
                "MAE": "best_MAE",
                "DirectionAccuracy": "best_DirectionAccuracy",
            }
        )
    )
    stock_best = stock_type.merge(
        best_scenario[
            [
                "stock_id",
                "best_scenario_id",
                "best_scenario_name",
                "best_WMAPE",
                "best_MAE",
                "best_DirectionAccuracy",
            ]
        ],
        on="stock_id",
        how="left",
    )
    scenario_win_counts = (
        stock_best.groupby(["best_scenario_id", "best_scenario_name"], as_index=False)
        .agg(stock_wins=("stock_id", "size"))
        .sort_values("stock_wins", ascending=False)
    )
    scenario_win_counts["win_rate"] = scenario_win_counts["stock_wins"] / len(stock_best) * 100

    parameter_effects = build_parameter_effects(overall, ["scenario_id"])
    parameter_effects_by_regime = build_parameter_effects(
        regime_accuracy,
        ["scenario_id", "regime"],
    )
    parameter_effects_by_industry = build_parameter_effects(
        industry_accuracy,
        ["scenario_id", "industry_category"],
    )
    stock_parameter_effects = build_parameter_effects(
        stock_accuracy,
        ["scenario_id", "stock_id", "stock_name", "industry_category"],
    )
    stock_effect_summary = (
        stock_parameter_effects.groupby(
            ["effect_id", "effect_name", "base_scenario", "treatment_scenario"],
            as_index=False,
        )
        .agg(
            stocks=("stock_id", "size"),
            stocks_MAE_improved=("MAE_delta", lambda values: int((values < 0).sum())),
            stocks_MAE_worsened=("MAE_delta", lambda values: int((values > 0).sum())),
            stocks_MAE_tied=("MAE_delta", lambda values: int((values == 0).sum())),
            median_MAE_delta=("MAE_delta", "median"),
            median_WMAPE_delta=("WMAPE_delta", "median"),
            median_DirectionAccuracy_delta=("DirectionAccuracy_delta", "median"),
        )
    )
    stock_effect_summary["MAE_improved_rate"] = (
        stock_effect_summary["stocks_MAE_improved"] / stock_effect_summary["stocks"] * 100
    )
    stock_effect_summary["MAE_worsened_rate"] = (
        stock_effect_summary["stocks_MAE_worsened"] / stock_effect_summary["stocks"] * 100
    )

    typed_monthly = monthly.merge(
        stock_type[
            ["stock_id", "dominant_regime", "regime_confidence", "revenue_size_quartile"]
        ],
        on="stock_id",
        how="left",
    )
    dominant_regime_accuracy = summarize(
        typed_monthly,
        ["scenario_id", "scenario_name", "dominant_regime"],
    )
    parameter_effects_by_dominant_regime = build_parameter_effects(
        dominant_regime_accuracy,
        ["scenario_id", "dominant_regime"],
    )
    stock_type_scenario_accuracy = summarize(
        typed_monthly,
        ["scenario_id", "scenario_name", "dominant_regime", "revenue_size_quartile"],
    )
    stock_wmape_wide = stock_accuracy.pivot(
        index="stock_id",
        columns="scenario_id",
        values="WMAPE",
    ).add_prefix("WMAPE_")
    cycle_candidates = stock_best[
        stock_best["dominant_regime"].eq("cycle")
    ].merge(stock_wmape_wide, on="stock_id", how="left")
    cycle_candidates["TrendCycle_delta_fixed"] = (
        cycle_candidates["WMAPE_S03"] - cycle_candidates["WMAPE_S02"]
    )
    cycle_candidates["TrendCycle_delta_auto"] = (
        cycle_candidates["WMAPE_S08"] - cycle_candidates["WMAPE_S07"]
    )
    cycle_candidates["AutoTune_delta_base"] = (
        cycle_candidates["WMAPE_S06"] - cycle_candidates["WMAPE_S01"]
    )
    cycle_candidates["AutoTune_delta_growth"] = (
        cycle_candidates["WMAPE_S07"] - cycle_candidates["WMAPE_S02"]
    )
    cycle_candidates["ML_benchmark_priority"] = pd.qcut(
        cycle_candidates["best_WMAPE"].rank(method="first"),
        4,
        labels=["low", "medium", "high", "very_high"],
    )
    cycle_candidates = cycle_candidates.sort_values(
        ["best_WMAPE", "regime_confidence"],
        ascending=[False, False],
    )
    tuning_selection_summary = (
        selected_all.groupby(
            ["tuning_family", "tuning_status", "k", "max_train_samples"],
            as_index=False,
        )
        .agg(stock_count=("stock_id", "nunique"))
    )
    tuning_selection_summary["stock_share"] = (
        tuning_selection_summary["stock_count"]
        / tuning_selection_summary.groupby("tuning_family")["stock_count"].transform("sum")
        * 100
    )

    failed_2025 = pd.DataFrame(
        {
            "stock_id": sorted(set(eligible_2025) - prediction_stock_ids),
            "reason": "No 2025 rolling evaluation samples",
        }
    ).merge(stock_meta, on="stock_id", how="left")

    scenario_counts = monthly.groupby("scenario_id").agg(
        stock_count=("stock_id", "nunique"),
        observations=("stock_id", "size"),
    )
    run_config = {
        "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": round(time.time() - started, 3),
        "source_2025_stocks": len(eligible_2025),
        "forecastable_2025_stocks": len(prediction_stock_ids),
        "unforecastable_2025_stocks": len(failed_2025),
        "scenario_counts": scenario_counts.reset_index().to_dict("records"),
        "fixed_parameters": {
            "k": FIXED_K,
            "epochs": FIXED_EPOCHS,
            "max_train_samples": FIXED_MAX_TRAIN_SAMPLES,
            "growth_alpha": GROWTH_ALPHA,
            "under_weight": UNDER_WEIGHT,
            "trend_slope_beta": TREND_SLOPE_BETA,
            "max_volatility_scale": MAX_VOLATILITY_SCALE,
        },
        "tuning": {
            "year": TUNING_YEAR,
            "epochs": TUNING_EPOCHS,
            "metric": "MAE",
            "k_candidates": K_CANDIDATES,
            "sample_candidates": SAMPLE_CANDIDATES,
            "optimization_note": (
                "Each candidate model was trained once and evaluated per stock; "
                "this is equivalent to repeating the same globally trained candidate "
                "for each stock, without redundant retraining."
            ),
        },
        "method_notes": {
            "primary_model": "Rolling LSTM with KMeans cluster feature",
            "base_loss": "Huber",
            "growth_loss": "asymmetric squared loss with under_weight=2.0",
            "guardrail": "Static 5x last / 4x sequence cap.",
            "regime": "Dynamic monthly label from past-only growth_ratio, not a permanent stock type.",
            "revenue_unit": "thousand TWD",
        },
    }

    write_csv(scenario_config, output_dir / "scenario_config.csv")
    write_csv(overall, output_dir / "overall_accuracy.csv")
    write_csv(parameter_effects, output_dir / "parameter_effects.csv")
    write_csv(parameter_effects_by_regime, output_dir / "parameter_effects_by_regime.csv")
    write_csv(parameter_effects_by_industry, output_dir / "parameter_effects_by_industry.csv")
    write_csv(stock_parameter_effects, output_dir / "stock_parameter_effects.csv")
    write_csv(stock_effect_summary, output_dir / "stock_effect_summary.csv")
    write_csv(regime_accuracy, output_dir / "regime_accuracy.csv")
    write_csv(dominant_regime_accuracy, output_dir / "dominant_regime_accuracy.csv")
    write_csv(
        parameter_effects_by_dominant_regime,
        output_dir / "parameter_effects_by_dominant_regime.csv",
    )
    write_csv(industry_accuracy, output_dir / "industry_accuracy.csv")
    write_csv(stock_type_scenario_accuracy, output_dir / "stock_type_scenario_accuracy.csv")
    write_csv(cycle_candidates, output_dir / "cycle_ml_benchmark_candidates.csv")
    write_csv(stock_accuracy, output_dir / "stock_accuracy.csv")
    write_csv(stock_best, output_dir / "stock_best_scenario.csv")
    write_csv(scenario_win_counts, output_dir / "scenario_win_counts.csv")
    write_csv(tuning_results, output_dir / "tuning_candidate_results.csv")
    write_csv(selected_all, output_dir / "tuning_selected_params.csv")
    write_csv(tuning_selection_summary, output_dir / "tuning_selection_summary.csv")
    write_csv(failed_2025, output_dir / "failed_stocks.csv")
    write_csv(monthly, output_dir / "monthly_predictions.csv")
    run_config = write_rolling_run_config(
        output_dir,
        run_config,
        experiment_family="rolling_ten_scenarios",
        evidence_tier="C",
        selection_protocol="target-year-hindsight",
        report_ready=False,
        report_ready_reason="Scenario winners are compared and ranked on target-year actuals.",
    )

    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
