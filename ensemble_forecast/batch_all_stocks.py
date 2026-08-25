from __future__ import annotations

"""Run the Ensemble revenue models once per target year for a full stock universe.

The interactive forecast path fits the same cross-sectional tree models again for
every selected stock.  This batch runner preserves the model definitions and the
stock-specific 2023/2024 validation weights, while fitting each shared tree model
only once per target year.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_forecast import forecast_engine as engine
from forecast_benchmark.metrics import (
    build_accuracy_frame,
    build_overall_accuracy,
    build_stock_accuracy,
    build_winner_summary,
)


VALIDATION_YEARS = (2023, 2024)


def parse_stock_ids(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_universe_audit(
    revenue: pd.DataFrame,
    actual: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:
    history_counts = (
        revenue[revenue["revenue_year"].isin([target_year - 1, *VALIDATION_YEARS])]
        .groupby(["stock_id", "revenue_year"])["revenue_month"]
        .nunique()
        .unstack(fill_value=0)
    )
    actual_counts = actual.groupby("stock_id")["revenue_month"].nunique().rename("actual_months")
    stock_ids = sorted(set(revenue["stock_id"].astype(int)).union(actual["stock_id"].astype(int)))
    audit = pd.DataFrame({"stock_id": stock_ids}).set_index("stock_id")
    audit = audit.join(actual_counts, how="left")
    for year in [target_year - 1, *VALIDATION_YEARS]:
        values = history_counts[year] if year in history_counts.columns else pd.Series(dtype=float)
        audit[f"months_{year}"] = values
    audit = audit.fillna(0).astype(int).reset_index()
    audit["complete_actual_year"] = audit["actual_months"].eq(12)
    audit["complete_source_year"] = audit[f"months_{target_year - 1}"].eq(12)
    audit["eligible"] = audit["complete_actual_year"] & audit["complete_source_year"]
    audit["exclusion_reason"] = np.select(
        [
            ~audit["complete_actual_year"],
            ~audit["complete_source_year"],
        ],
        [
            "target year has fewer than 12 actual months",
            "source year has fewer than 12 model-input months",
        ],
        default="",
    )
    return audit


def select_stock_ids(
    audit: pd.DataFrame,
    explicit_stocks: list[int] | None,
    stock_limit: int | None,
) -> list[int]:
    eligible = audit[audit["eligible"]]["stock_id"].astype(int).sort_values().tolist()
    if explicit_stocks is not None:
        eligible_set = set(eligible)
        missing = [stock_id for stock_id in explicit_stocks if stock_id not in eligible_set]
        if missing:
            raise ValueError(f"Requested stocks are not eligible: {missing}")
        eligible = explicit_stocks
    if stock_limit is not None:
        eligible = eligible[: int(stock_limit)]
    return eligible


def fit_shared_tree_predictions(
    revenue: pd.DataFrame,
    stock_ids: list[int],
    target_year: int,
) -> pd.DataFrame:
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import GridSearchCV, PredefinedSplit
    from xgboost import XGBRegressor

    model_df = engine._attach_next_year_target(revenue)
    complete_history = model_df.get(
        "_history_12m_complete",
        pd.Series(True, index=model_df.index),
    )
    train_df = model_df[
        (model_df["revenue_year"] >= engine.TRAIN_START_YEAR)
        & (model_df["revenue_year"] <= target_year - 2)
        & complete_history
    ].dropna(subset=engine.FEATURES + ["target_next_year"])
    predict_df = model_df[
        model_df["stock_id"].isin(stock_ids)
        & model_df["revenue_year"].eq(target_year - 1)
        & complete_history
    ].sort_values(["stock_id", "revenue_month"])
    complete_stocks = predict_df.groupby("stock_id")["revenue_month"].nunique()
    complete_stocks = complete_stocks[complete_stocks.eq(12)].index
    predict_df = predict_df[predict_df["stock_id"].isin(complete_stocks)].copy()
    if train_df.empty or predict_df.empty:
        return pd.DataFrame()

    xgb_base = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=160,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
    )
    validation_year = int(train_df["revenue_year"].max())
    test_fold = np.where(train_df["revenue_year"].to_numpy() == validation_year, 0, -1)
    if (test_fold == 0).any() and (test_fold == -1).any():
        search = GridSearchCV(
            xgb_base,
            param_grid={
                "n_estimators": [80, 160],
                "max_depth": [2, 3],
                "learning_rate": [0.03, 0.08],
            },
            cv=PredefinedSplit(test_fold),
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        search.fit(train_df[engine.FEATURES], train_df["target_next_year"])
        xgb_model = search.best_estimator_
    else:
        xgb_model = xgb_base.fit(train_df[engine.FEATURES], train_df["target_next_year"])

    models = {
        "XGBoost": xgb_model,
        "LightGBM": LGBMRegressor(
            objective="regression",
            random_state=42,
            n_estimators=160,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=15,
            min_child_samples=5,
            verbosity=-1,
        ).fit(train_df[engine.FEATURES], train_df["target_next_year"]),
        "CatBoost": CatBoostRegressor(
            loss_function="RMSE",
            random_seed=42,
            iterations=220,
            learning_rate=0.05,
            depth=4,
            verbose=False,
        ).fit(train_df[engine.FEATURES], train_df["target_next_year"]),
    }
    rows: list[pd.DataFrame] = []
    for model_name, model in models.items():
        predicted = np.maximum(np.expm1(model.predict(predict_df[engine.FEATURES])), 0)
        rows.append(
            pd.DataFrame(
                {
                    "stock_id": predict_df["stock_id"].astype(int).to_numpy(),
                    "target_year": int(target_year),
                    "target_month": predict_df["revenue_month"].astype(int).to_numpy(),
                    "model": model_name,
                    "predicted_revenue": np.rint(predicted).astype(int),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def build_seasonal_predictions(
    revenue: pd.DataFrame,
    stock_ids: list[int],
    target_year: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    for stock_id in stock_ids:
        try:
            part = engine._seasonal_quantile_forecast(revenue, stock_id, target_year).rename(
                columns={"revenue_year": "target_year", "revenue_month": "target_month"}
            )
            part.insert(0, "stock_id", int(stock_id))
            rows.append(part)
        except Exception as error:
            failures.append(
                {
                    "stock_id": int(stock_id),
                    "target_year": int(target_year),
                    "model": "SeasonalQuantile",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    return (
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
        failures,
    )


def build_year_predictions(
    revenue: pd.DataFrame,
    stock_ids: list[int],
    target_year: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    trees = fit_shared_tree_predictions(revenue, stock_ids, target_year)
    seasonal, failures = build_seasonal_predictions(revenue, stock_ids, target_year)
    return pd.concat([trees, seasonal], ignore_index=True), failures


def _mape(actual: pd.Series, predicted: pd.Series) -> float:
    actual_values = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(actual_values) & np.isfinite(predicted_values) & (actual_values != 0)
    if not valid.any():
        return np.nan
    return float(np.mean(np.abs((actual_values[valid] - predicted_values[valid]) / actual_values[valid])) * 100)


def build_stock_weights(
    validation_predictions: pd.DataFrame,
    actual_by_year: pd.DataFrame,
    stock_ids: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stock_id in stock_ids:
        score_rows: list[dict[str, object]] = []
        for year in VALIDATION_YEARS:
            actual = actual_by_year[
                actual_by_year["stock_id"].eq(stock_id)
                & actual_by_year["target_year"].eq(year)
            ][["target_month", "actual_revenue"]]
            if len(actual) != 12:
                continue
            stock_year = validation_predictions[
                validation_predictions["stock_id"].eq(stock_id)
                & validation_predictions["target_year"].eq(year)
            ]
            for model_name in engine.FORECAST_MODEL_NAMES:
                evaluated = stock_year[stock_year["model"].eq(model_name)].merge(
                    actual,
                    on="target_month",
                    how="inner",
                )
                if len(evaluated) == 12:
                    score_rows.append(
                        {
                            "model": model_name,
                            "validation_year": year,
                            "MAPE": _mape(evaluated["actual_revenue"], evaluated["predicted_revenue"]),
                        }
                    )
        if score_rows:
            report = (
                pd.DataFrame(score_rows)
                .groupby("model", as_index=False)
                .agg(
                    validation_mape=("MAPE", "mean"),
                    validation_year_count=("validation_year", "nunique"),
                )
            )
            missing = [model for model in engine.FORECAST_MODEL_NAMES if model not in set(report["model"])]
            if missing:
                worst = float(report["validation_mape"].max()) if not report.empty else 100.0
                report = pd.concat(
                    [
                        report,
                        pd.DataFrame(
                            {
                                "model": missing,
                                "validation_mape": [worst] * len(missing),
                                "validation_year_count": [0] * len(missing),
                            }
                        ),
                    ],
                    ignore_index=True,
                )
            report["raw_weight"] = 1 / report["validation_mape"].clip(lower=0.01)
            report["weight"] = report["raw_weight"] / report["raw_weight"].sum()
        else:
            report = pd.DataFrame(
                {
                    "model": engine.FORECAST_MODEL_NAMES,
                    "validation_mape": [np.nan] * len(engine.FORECAST_MODEL_NAMES),
                    "validation_year_count": [0] * len(engine.FORECAST_MODEL_NAMES),
                    "weight": [1 / len(engine.FORECAST_MODEL_NAMES)] * len(engine.FORECAST_MODEL_NAMES),
                }
            )
        report.insert(0, "stock_id", int(stock_id))
        rows.extend(report[["stock_id", "model", "validation_mape", "validation_year_count", "weight"]].to_dict("records"))
    return pd.DataFrame(rows)


def build_ensemble_predictions(
    target_predictions: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    weighted = target_predictions.merge(weights[["stock_id", "model", "weight"]], on=["stock_id", "model"], how="left")
    weighted["weighted_prediction"] = weighted["predicted_revenue"] * weighted["weight"]
    ensemble = weighted.groupby(["stock_id", "target_year", "target_month"], as_index=False).agg(
        weighted_prediction=("weighted_prediction", "sum"),
        available_weight=("weight", "sum"),
    )
    ensemble["model"] = "ensemble_revenue"
    ensemble["predicted_revenue"] = np.rint(
        ensemble["weighted_prediction"] / ensemble["available_weight"].replace(0, np.nan)
    )
    return ensemble[["stock_id", "target_year", "target_month", "model", "predicted_revenue"]]


def attach_evaluation_context(
    predictions: pd.DataFrame,
    revenue: pd.DataFrame,
    actual: pd.DataFrame,
    stock_meta: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:
    evaluated = predictions.merge(
        actual[["stock_id", "revenue_year", "revenue_month", "actual_revenue"]].rename(
            columns={"revenue_year": "target_year", "revenue_month": "target_month"}
        ),
        on=["stock_id", "target_year", "target_month"],
        how="left",
    )
    history = revenue[["stock_id", "revenue_year", "revenue_month", "revenue_thousand"]].copy()
    history["target_year"] = history["revenue_year"]
    history["target_month"] = history["revenue_month"] + 1
    december = history["target_month"].eq(13)
    history.loc[december, "target_year"] += 1
    history.loc[december, "target_month"] = 1
    history = history[history["target_year"].eq(target_year)].rename(
        columns={"revenue_thousand": "last_observed_revenue"}
    )
    evaluated = evaluated.merge(
        history[["stock_id", "target_year", "target_month", "last_observed_revenue"]],
        on=["stock_id", "target_year", "target_month"],
        how="left",
    )
    evaluated = evaluated.merge(stock_meta, on="stock_id", how="left")
    evaluated["source_family"] = "ensemble_forecast"
    evaluated["source_path"] = str(Path("ensemble_forecast") / "batch_all_stocks.py")
    evaluated["predicted_revenue"] = pd.to_numeric(evaluated["predicted_revenue"], errors="coerce")
    return evaluated[
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
    ].sort_values(["stock_id", "target_month", "model"])


def load_stock_metadata(revenue: pd.DataFrame) -> pd.DataFrame:
    metadata = (
        revenue.sort_values("date")
        .groupby("stock_id", as_index=False)
        .agg(industry_category=("industry_category", "last"))
    )
    stock_list_path = Path(engine.DATA_DIR) / "stock_list_new.csv"
    names = pd.read_csv(stock_list_path, usecols=["stock_id", "stock_name"])
    names["stock_id"] = pd.to_numeric(names["stock_id"], errors="coerce")
    names = names.dropna(subset=["stock_id"]).drop_duplicates("stock_id")
    names["stock_id"] = names["stock_id"].astype(int)
    metadata = metadata.merge(names, on="stock_id", how="left")
    metadata["stock_name"] = metadata["stock_name"].fillna("")
    metadata["industry_category"] = metadata["industry_category"].fillna("unknown")
    return metadata[["stock_id", "stock_name", "industry_category"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stock-limit", type=int, default=None)
    parser.add_argument("--target-year", type=int, default=engine.FORECAST_YEAR)
    args = parser.parse_args()

    if int(args.target_year) != engine.FORECAST_YEAR:
        raise ValueError(f"Ensemble Forecast currently supports target year {engine.FORECAST_YEAR}.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()

    revenue = engine.load_revenue_data()
    actual = engine.load_actual_2025_data()
    audit = build_universe_audit(revenue, actual, int(args.target_year))
    stock_ids = select_stock_ids(audit, parse_stock_ids(args.stocks), args.stock_limit)
    metadata = load_stock_metadata(revenue)
    actual_history = revenue.rename(
        columns={
            "revenue_year": "target_year",
            "revenue_month": "target_month",
            "revenue_thousand": "actual_revenue",
        }
    )[["stock_id", "target_year", "target_month", "actual_revenue"]]

    year_predictions: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    for year in [*VALIDATION_YEARS, int(args.target_year)]:
        print(f"Fitting shared Ensemble models for target_year={year}", flush=True)
        predictions, year_failures = build_year_predictions(revenue, stock_ids, year)
        year_predictions.append(predictions)
        failures.extend(year_failures)
    all_year_predictions = pd.concat(year_predictions, ignore_index=True)
    validation_predictions = all_year_predictions[
        all_year_predictions["target_year"].isin(VALIDATION_YEARS)
    ]
    weights = build_stock_weights(validation_predictions, actual_history, stock_ids)
    target_predictions = all_year_predictions[
        all_year_predictions["target_year"].eq(int(args.target_year))
    ].copy()
    ensemble = build_ensemble_predictions(target_predictions, weights)
    predictions = pd.concat([target_predictions, ensemble], ignore_index=True)
    evaluated = attach_evaluation_context(
        predictions,
        revenue,
        actual,
        metadata,
        int(args.target_year),
    )

    overall = build_overall_accuracy(evaluated)
    stock_accuracy = build_stock_accuracy(evaluated)
    winner_summary = build_winner_summary(stock_accuracy, primary_metric="WMAPE")
    industry_accuracy = build_accuracy_frame(
        evaluated,
        ["industry_category", "source_family", "model"],
    ).sort_values(["industry_category", "WMAPE", "MAPE", "model"])
    failure_frame = pd.DataFrame(
        failures,
        columns=["stock_id", "target_year", "model", "error_type", "error"],
    )

    evaluated.to_csv(output_dir / "monthly_predictions.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(output_dir / "validation_weights.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output_dir / "overall_accuracy.csv", index=False, encoding="utf-8-sig")
    stock_accuracy.to_csv(output_dir / "stock_accuracy.csv", index=False, encoding="utf-8-sig")
    winner_summary.to_csv(output_dir / "winner_summary.csv", index=False, encoding="utf-8-sig")
    industry_accuracy.to_csv(output_dir / "industry_accuracy.csv", index=False, encoding="utf-8-sig")
    failure_frame.to_csv(output_dir / "failed_runs.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output_dir / "universe_audit.csv", index=False, encoding="utf-8-sig")

    run_config = {
        "target_year": int(args.target_year),
        "validation_years": list(VALIDATION_YEARS),
        "requested_stock_count": len(stock_ids),
        "eligible_stock_count": int(audit["eligible"].sum()),
        "complete_actual_stock_count": int(audit["complete_actual_year"].sum()),
        "model_names": [*engine.FORECAST_MODEL_NAMES, "ensemble_revenue"],
        "duration_seconds": round(time.time() - started, 3),
        "optimization_note": (
            "Tree models are fit once per target year because their training data is cross-sectional "
            "and independent of selected_stock; validation weights remain stock-specific."
        ),
        "actual_usage": "2025 actual revenue is attached only after predictions for evaluation.",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
