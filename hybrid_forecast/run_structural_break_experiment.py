from __future__ import annotations

import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


SYSTEM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SYSTEM_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_forecast.hybrid_engine import (  # noqa: E402
    HybridConfig,
    combine_predictions,
    compute_metrics,
    search_sarima_weight,
    stock_metrics,
)
from hybrid_forecast.structural_break_engine import (  # noqa: E402
    StructuralBreakConfig,
    add_structural_break_features,
    apply_structural_break_adjustment,
)
from revenue_adjustment_formula.formula_engine import (  # noqa: E402
    FormulaConfig,
    build_rolling_predictions,
    load_revenue_data,
)


OUTPUT_DIR = SYSTEM_DIR / "outputs" / "structural_break_20260820"
BASELINE_DIR = SYSTEM_DIR / "outputs" / "full_universe_20260820"
SARIMA_2025_PATH = (
    PROJECT_ROOT
    / "sarima_forecast"
    / "outputs"
    / "full_universe_20260818"
    / "monthly_predictions.csv"
)
REVENUE_PATH = PROJECT_ROOT / "data" / "Stock_revenue_2019~2025.csv"
FORMULA_CONFIG = FormulaConfig(
    seasonal_weight=0.5,
    residual_alpha=0.1,
    residual_strength=0.0,
    growth_log_cap=float(np.log(2.0)),
    correction_log_cap=0.5,
)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _rename_stock_metrics(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return frame.rename(
        columns={column: f"{prefix}_{column}" for column in frame.columns if column != "stock_id"}
    )


def _model_table(frame: pd.DataFrame) -> pd.DataFrame:
    mappings = {
        "原營收公式": "baseline_formula_revenue",
        "原混合模型": "baseline_hybrid_revenue",
        "結構斷點公式": "structural_formula_revenue",
        "改善混合模型": "structural_hybrid_revenue",
    }
    return pd.DataFrame(
        [
            {"model": model, **compute_metrics(frame, column)}
            for model, column in mappings.items()
        ]
    )


def _monthly_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month, group in frame.groupby("target_month"):
        monthly = _model_table(group)
        monthly.insert(0, "target_month", int(month))
        rows.extend(monthly.to_dict(orient="records"))
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    revenue = load_revenue_data(REVENUE_PATH)
    formula = build_rolling_predictions(
        revenue,
        FORMULA_CONFIG,
        start_date="2023-01-01",
        end_date="2025-12-01",
    )
    featured = add_structural_break_features(formula, revenue)
    validation_sarima = pd.read_csv(
        BASELINE_DIR / "validation_sarima_predictions.csv"
    )
    validation_features = featured[featured["target_year"].isin([2023, 2024])].copy()

    sweep_path = OUTPUT_DIR / "parameter_sweep.csv"
    cached_sweep = pd.read_csv(sweep_path) if sweep_path.exists() else None
    parameter_rows: list[dict[str, object]] = []
    for yoy_threshold, level_threshold, retention in product(
        [] if cached_sweep is not None else [0.05, 0.10, 0.20],
        [] if cached_sweep is not None else [0.10, 0.20, 0.30],
        [] if cached_sweep is not None else [0.00, 0.10, 0.25],
    ):
        config = StructuralBreakConfig(
            yoy_ratio_threshold=yoy_threshold,
            level_ratio_threshold=level_threshold,
            formula_retention=retention,
        )
        adjusted = apply_structural_break_adjustment(validation_features, config)
        selected_weight, weight_sweep = search_sarima_weight(
            adjusted,
            validation_sarima,
        )
        best = weight_sweep[
            weight_sweep["sarima_weight"].round(6).eq(round(selected_weight, 6))
        ].iloc[0]
        parameter_rows.append(
            {
                **config.as_dict(),
                "selected_sarima_weight": selected_weight,
                "selected_formula_weight": 1.0 - selected_weight,
                "detected_rows": int(adjusted["structural_break_detected"].sum()),
                "balanced_score": float(best["balanced_score"]),
                "WMAPE": float(best["WMAPE"]),
                "SMAPE": float(best["SMAPE"]),
                "MedianAPE": float(best["MedianAPE"]),
                "P90APE": float(best["P90APE"]),
                "median_stock_WMAPE": float(best["median_stock_WMAPE"]),
                "DirectionAccuracy": float(best["DirectionAccuracy"]),
            }
        )
        print(
            f"config yoy={yoy_threshold:.2f}, level={level_threshold:.2f}, "
            f"retention={retention:.2f}, weight={selected_weight:.1f}, "
            f"score={float(best['balanced_score']):.4f}",
            flush=True,
        )

    parameter_sweep = (
        cached_sweep
        if cached_sweep is not None
        else pd.DataFrame(parameter_rows).sort_values(
            ["balanced_score", "WMAPE", "yoy_ratio_threshold"]
        ).reset_index(drop=True)
    )
    selected = parameter_sweep.iloc[0]
    selected_config = StructuralBreakConfig(
        yoy_ratio_threshold=float(selected["yoy_ratio_threshold"]),
        level_ratio_threshold=float(selected["level_ratio_threshold"]),
        formula_retention=float(selected["formula_retention"]),
        recent_ratio_threshold=float(selected["recent_ratio_threshold"]),
        required_recent_breaks=int(selected["required_recent_breaks"]),
    )
    selected_weight = float(selected["selected_sarima_weight"])
    _write_csv(parameter_sweep, OUTPUT_DIR / "parameter_sweep.csv")

    test_features = featured[featured["target_year"].eq(2025)].copy()
    adjusted_formula = apply_structural_break_adjustment(
        test_features,
        selected_config,
    )
    sarima_2025 = pd.read_csv(SARIMA_2025_PATH)
    improved = combine_predictions(
        adjusted_formula,
        sarima_2025,
        HybridConfig(sarima_weight=selected_weight),
    )

    baseline = pd.read_csv(BASELINE_DIR / "monthly_predictions.csv")
    metadata = baseline[["stock_id", "stock_name", "industry_category"]].drop_duplicates(
        "stock_id", keep="last"
    )
    improved = metadata.merge(improved, on="stock_id", how="right")
    feature_columns = [
        "stock_id",
        "target_date",
        "structural_break_detected",
        "recent_break_count",
        "break_last_yoy_ratio",
        "break_level_ratio",
        "original_formula_revenue",
        "structural_formula_method",
    ]
    improved = improved.merge(
        adjusted_formula[feature_columns],
        on=["stock_id", "target_date"],
        how="left",
        validate="one_to_one",
    )
    _write_csv(improved, OUTPUT_DIR / "monthly_predictions.csv")

    baseline_small = baseline[
        [
            "stock_id",
            "stock_name",
            "industry_category",
            "target_date",
            "target_year",
            "target_month",
            "actual_revenue",
            "last_observed_revenue",
            "formula_adjusted_revenue",
            "hybrid_predicted_revenue",
        ]
    ].rename(
        columns={
            "formula_adjusted_revenue": "baseline_formula_revenue",
            "hybrid_predicted_revenue": "baseline_hybrid_revenue",
        }
    )
    baseline_small["target_date"] = pd.to_datetime(baseline_small["target_date"])
    improved_small = improved[
        [
            "stock_id",
            "target_date",
            "formula_adjusted_revenue",
            "hybrid_predicted_revenue",
            "structural_break_detected",
            "structural_formula_method",
            "break_last_yoy_ratio",
            "break_level_ratio",
            "recent_break_count",
            "hybrid_method",
        ]
    ].rename(
        columns={
            "formula_adjusted_revenue": "structural_formula_revenue",
            "hybrid_predicted_revenue": "structural_hybrid_revenue",
            "hybrid_method": "structural_hybrid_method",
        }
    )
    improved_small["target_date"] = pd.to_datetime(improved_small["target_date"])
    comparison = baseline_small.merge(
        improved_small,
        on=["stock_id", "target_date"],
        how="inner",
        validate="one_to_one",
    )
    required = [
        "actual_revenue",
        "baseline_hybrid_revenue",
        "structural_hybrid_revenue",
    ]
    valid = comparison.dropna(subset=required)
    complete_counts = valid.groupby("stock_id").size()
    complete_ids = complete_counts[complete_counts.eq(12)].index
    complete = comparison[comparison["stock_id"].isin(complete_ids)].copy()
    complete["mase_scale"] = np.nan
    _write_csv(complete, OUTPUT_DIR / "operational_comparison_monthly.csv")

    overall = _model_table(complete)
    monthly_accuracy = _monthly_table(complete)
    baseline_stock = _rename_stock_metrics(
        stock_metrics(complete, "baseline_hybrid_revenue"), "baseline"
    )
    improved_stock = _rename_stock_metrics(
        stock_metrics(complete, "structural_hybrid_revenue"), "improved"
    )
    formula_stock = _rename_stock_metrics(
        stock_metrics(complete, "structural_formula_revenue"), "structural_formula"
    )
    stock_accuracy = baseline_stock.merge(improved_stock, on="stock_id").merge(
        formula_stock, on="stock_id"
    )
    break_summary = complete.groupby("stock_id", as_index=False).agg(
        structural_break_months=("structural_break_detected", "sum")
    )
    stock_accuracy = metadata.merge(stock_accuracy, on="stock_id", how="right").merge(
        break_summary, on="stock_id", how="left"
    )
    stock_accuracy["WMAPE_improvement_points"] = (
        stock_accuracy["baseline_WMAPE"] - stock_accuracy["improved_WMAPE"]
    )
    stock_accuracy["improved_beats_baseline"] = (
        stock_accuracy["improved_WMAPE"] < stock_accuracy["baseline_WMAPE"]
    )
    stock_accuracy["baseline_over_15"] = stock_accuracy["baseline_WMAPE"] > 15
    stock_accuracy["improved_over_15"] = stock_accuracy["improved_WMAPE"] > 15
    stock_accuracy = stock_accuracy.sort_values(
        "WMAPE_improvement_points", ascending=False
    ).reset_index(drop=True)
    _write_csv(overall, OUTPUT_DIR / "overall_accuracy.csv")
    _write_csv(monthly_accuracy, OUTPUT_DIR / "monthly_accuracy.csv")
    _write_csv(stock_accuracy, OUTPUT_DIR / "stock_accuracy.csv")
    _write_csv(
        stock_accuracy[stock_accuracy["improved_over_15"]].sort_values("improved_WMAPE"),
        OUTPUT_DIR / "over_15pct_ascending.csv",
    )

    baseline_row = overall[overall["model"].eq("原混合模型")].iloc[0]
    improved_row = overall[overall["model"].eq("改善混合模型")].iloc[0]
    rescued = int(
        (stock_accuracy["baseline_over_15"] & ~stock_accuracy["improved_over_15"]).sum()
    )
    config_output = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection_period": "2023-01 to 2024-12",
        "test_period": "2025-01 to 2025-12",
        "selected_structural_break_config": selected_config.as_dict(),
        "selected_sarima_weight": selected_weight,
        "selected_formula_weight": 1.0 - selected_weight,
        "validation_balanced_score": float(selected["balanced_score"]),
        "complete_operational_stock_count": int(len(complete_ids)),
        "test_break_rows": int(complete["structural_break_detected"].sum()),
        "stocks_improved": int(stock_accuracy["improved_beats_baseline"].sum()),
        "stocks_rescued_below_15pct": rescued,
        "baseline_WMAPE": float(baseline_row["WMAPE"]),
        "improved_WMAPE": float(improved_row["WMAPE"]),
        "exploratory_note": (
            "The structural-break concept was added after reviewing a 2025 failure case; "
            "parameters and weights were selected only on 2023-2024, but the 2025 result "
            "should still be treated as an exploratory comparison."
        ),
    }
    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(config_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# 結構性崩落偵測版比較

- 參數選擇：2023–2024。
- 比較期間：2025。
- 完整共同樣本：{len(complete_ids):,} 檔。
- 原混合模型 WMAPE：{baseline_row['WMAPE']:.3f}%。
- 改善混合模型 WMAPE：{improved_row['WMAPE']:.3f}%。
- 改善個股數：{int(stock_accuracy['improved_beats_baseline'].sum()):,} 檔。
- 從 WMAPE > 15% 降回 15% 以下：{rescued:,} 檔。
- 2025 觸發結構性崩落調整：{int(complete['structural_break_detected'].sum()):,} 個股票月份。

改善規則只使用預測當時已知的上月營收、去年同期與過去 12 個月中位數。偵測到崩落後，降低原季節公式的保留比例，改以最近已知營收為主。

## 實際上線時如何觸發

這個機制不是提前預知第一次營收崩落，而是「已公布營收出現崩落後的修正機制」。預測目標月時，只能使用預測當下已公布的歷史營收，不會使用目標月實際答案。

觸發條件為：最後已公布月份營收除以去年同期營收低於 20%，而且最後已公布月份營收除以過去 12 個月中位數低於 30%；最近三個月若已有至少兩次明顯崩落，也可作為持續斷點的確認訊號。

- 上月營收已公布後預測當月：可以在第一個崩落月份公布後立即修正。
- 月初而上月營收尚未公布：只能使用前兩個月資料，觸發至少延後一個月。
- 一次預測未來 12 個月：期間沒有新實際營收，斷點狀態無法逐月更新，必須在新營收公布後重跑。
- 第一次毫無前兆的崩落無法只靠歷史營收提前知道；若要提前預警，需要訂單、出貨、停產、重大訊息或資產處分等領先資料。

正式部署時應使用 `revenue_available_date <= forecast_timestamp` 作為資料截止條件，並在每筆輸出保存預測時間、最後已知營收月份及觸發原因。

注意：此概念是在檢視 2025 失敗案例後提出；雖然所有數值門檻及混合權重仍只由 2023–2024 選擇，2025 比較仍應視為探索性結果，後續需以新年度資料再確認。
"""
    (OUTPUT_DIR / "結構性崩落改善版比較報告.md").write_text(report, encoding="utf-8")
    print(json.dumps(config_output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
