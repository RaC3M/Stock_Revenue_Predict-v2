from __future__ import annotations

import argparse
import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .formula_engine import (
        FormulaConfig,
        build_rolling_predictions,
        compute_metrics,
        load_revenue_data,
        stock_wmape,
    )
except ImportError:
    from formula_engine import (
        FormulaConfig,
        build_rolling_predictions,
        compute_metrics,
        load_revenue_data,
        stock_wmape,
    )


SYSTEM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SYSTEM_DIR.parent
DEFAULT_OUTPUT_DIR = SYSTEM_DIR / "outputs" / f"formula_experiment_{date.today():%Y%m%d}"
STOCK_LIST_PATH = PROJECT_ROOT / "data" / "stock_list_new.csv"
XLSTM_PATH = (
    PROJECT_ROOT
    / "forecast_benchmark"
    / "outputs"
    / "parameter_sweep_20260813_report"
    / "exact_monthly_predictions.csv"
)
SARIMA_PATH = (
    PROJECT_ROOT
    / "sarima_forecast"
    / "outputs"
    / "full_universe_20260818"
    / "monthly_predictions.csv"
)

MODEL_COLUMNS = {
    "沿用上月營收": "last_observed_revenue",
    "去年同月營收": "seasonal_naive_revenue",
    "營收公式（未做殘差校正）": "formula_base_revenue",
    "營收公式（含殘差校正）": "formula_adjusted_revenue",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune and evaluate the revenue formula.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stock-limit", type=int, default=None)
    return parser.parse_args()


def load_stock_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(STOCK_LIST_PATH, dtype={"stock_id": str})
    metadata["stock_id"] = pd.to_numeric(metadata["stock_id"], errors="coerce")
    metadata = metadata.dropna(subset=["stock_id"]).copy()
    metadata["stock_id"] = metadata["stock_id"].astype(int)
    columns = [
        column
        for column in ["stock_id", "stock_name", "industry_category"]
        if column in metadata.columns
    ]
    return metadata[columns].drop_duplicates("stock_id", keep="last")


def parameter_grid() -> list[FormulaConfig]:
    return [
        FormulaConfig(
            seasonal_weight=seasonal_weight,
            residual_alpha=residual_alpha,
            residual_strength=residual_strength,
        )
        for seasonal_weight, residual_alpha, residual_strength in itertools.product(
            [0.50, 0.75, 1.00],
            [0.10, 0.20, 0.30],
            [0.00, 0.50, 1.00],
        )
    ]


def tune_formula(revenue: pd.DataFrame) -> tuple[FormulaConfig, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for index, config in enumerate(parameter_grid(), start=1):
        predictions = build_rolling_predictions(
            revenue,
            config,
            start_date="2023-01-01",
            end_date="2024-12-01",
        )
        formula_rows = predictions[
            predictions["forecast_method"] == "revenue_adjustment_formula"
        ]
        metrics = compute_metrics(formula_rows, "formula_adjusted_revenue")
        per_stock = stock_wmape(formula_rows, "formula_adjusted_revenue")
        median_stock_wmape = float(per_stock["WMAPE"].median())
        balanced_score = 0.5 * metrics["WMAPE"] + 0.5 * median_stock_wmape
        rows.append(
            {
                "candidate": index,
                **config.as_dict(),
                **metrics,
                "median_stock_WMAPE": median_stock_wmape,
                "balanced_score": balanced_score,
            }
        )
        print(
            f"[{index:02d}/27] seasonal={config.seasonal_weight:.2f}, "
            f"alpha={config.residual_alpha:.2f}, strength={config.residual_strength:.2f}, "
            f"WMAPE={metrics['WMAPE']:.3f}, median stock={median_stock_wmape:.3f}"
        )

    sweep = pd.DataFrame(rows).sort_values(
        ["balanced_score", "WMAPE", "MedianAPE"]
    ).reset_index(drop=True)
    best = sweep.iloc[0]
    selected = FormulaConfig(
        seasonal_weight=float(best["seasonal_weight"]),
        residual_alpha=float(best["residual_alpha"]),
        residual_strength=float(best["residual_strength"]),
        growth_log_cap=float(best["growth_log_cap"]),
        correction_log_cap=float(best["correction_log_cap"]),
    )
    return selected, sweep


def evaluate_models(predictions: pd.DataFrame) -> pd.DataFrame:
    common = predictions.dropna(subset=list(MODEL_COLUMNS.values())).copy()
    rows: list[dict[str, object]] = []
    for model_name, column in MODEL_COLUMNS.items():
        rows.append(
            {
                "model": model_name,
                "evaluation_scope": "共同有效股票月份",
                **compute_metrics(common, column),
            }
        )
    return pd.DataFrame(rows).sort_values("WMAPE").reset_index(drop=True)


def stock_accuracy(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stock_id, group in predictions.groupby("stock_id", sort=True):
        metrics = compute_metrics(group, "formula_adjusted_revenue")
        rows.append({"stock_id": int(stock_id), **metrics})
    result = pd.DataFrame(rows).merge(metadata, on="stock_id", how="left")
    columns = [
        "stock_id",
        "stock_name",
        "industry_category",
        "observations",
        "WMAPE",
        "SMAPE",
        "MedianAPE",
        "MAPE",
        "MAE",
        "RMSE",
        "MASE",
        "Bias",
        "UnderestimateRate",
        "DirectionAccuracy",
    ]
    return result[[column for column in columns if column in result.columns]].sort_values(
        ["WMAPE", "stock_id"]
    )


def monthly_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    common = predictions.dropna(subset=list(MODEL_COLUMNS.values())).copy()
    for month, group in common.groupby("target_month", sort=True):
        for model_name, column in MODEL_COLUMNS.items():
            rows.append(
                {
                    "target_month": int(month),
                    "model": model_name,
                    **compute_metrics(group, column),
                }
            )
    return pd.DataFrame(rows)


def compare_existing_models(formula: pd.DataFrame) -> pd.DataFrame:
    if not XLSTM_PATH.exists() or not SARIMA_PATH.exists():
        return pd.DataFrame()

    keys = ["stock_id", "target_year", "target_month"]
    base_columns = keys + [
        "actual_revenue",
        "last_observed_revenue",
        "mase_scale",
        "formula_adjusted_revenue",
    ]
    comparison = formula[base_columns].copy()

    xlstm = pd.read_csv(XLSTM_PATH)
    if "model" in xlstm.columns:
        xlstm = xlstm[xlstm["model"].eq("Rolling xLSTM")].copy()
    xlstm = xlstm[keys + ["predicted_revenue"]].rename(
        columns={"predicted_revenue": "predicted_revenue_xlstm"}
    )

    sarima = pd.read_csv(SARIMA_PATH)
    if "numeric_valid" in sarima.columns:
        valid = sarima["numeric_valid"].astype(str).str.lower().eq("true")
        sarima = sarima[valid].copy()
    sarima = sarima[keys + ["predicted_revenue_sarima"]]

    comparison = comparison.merge(xlstm, on=keys, how="inner").merge(
        sarima, on=keys, how="inner"
    )
    model_columns = {
        "營收調整公式": "formula_adjusted_revenue",
        "Rolling SARIMA": "predicted_revenue_sarima",
        "Rolling xLSTM": "predicted_revenue_xlstm",
    }
    comparison = comparison.dropna(subset=list(model_columns.values()))
    rows: list[dict[str, object]] = []
    for model_name, column in model_columns.items():
        rows.append(
            {
                "model": model_name,
                "evaluation_scope": "三模型共同股票月份",
                **compute_metrics(comparison, column),
            }
        )
    return pd.DataFrame(rows).sort_values("WMAPE").reset_index(drop=True)


def percentage_change(new: float, old: float) -> float:
    return (old - new) / old * 100 if np.isfinite(old) and old != 0 else np.nan


def format_metrics_table(frame: pd.DataFrame) -> str:
    columns = [
        "model",
        "observations",
        "stock_count",
        "WMAPE",
        "SMAPE",
        "MedianAPE",
        "MASE",
        "DirectionAccuracy",
    ]
    view = frame[[column for column in columns if column in frame.columns]].copy()
    for column in view.select_dtypes(include=["number"]).columns:
        if column not in ["observations", "stock_count"]:
            view[column] = view[column].map(lambda value: f"{value:.3f}")
    headers = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for values in view.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|") for value in values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(
    selected: FormulaConfig,
    sweep: pd.DataFrame,
    overall: pd.DataFrame,
    stocks: pd.DataFrame,
    predictions: pd.DataFrame,
    existing: pd.DataFrame,
) -> str:
    metric_by_model = overall.set_index("model")
    adjusted = metric_by_model.loc["營收公式（含殘差校正）"]
    base = metric_by_model.loc["營收公式（未做殘差校正）"]
    seasonal = metric_by_model.loc["去年同月營收"]
    last = metric_by_model.loc["沿用上月營收"]
    complete = stocks[stocks["observations"] >= 12]
    high_error = complete[complete["WMAPE"] > 15]
    formula_count = int(
        predictions["forecast_method"].eq("revenue_adjustment_formula").sum()
    )
    formula_share = formula_count / len(predictions) * 100 if len(predictions) else 0.0
    strict_formula = predictions[
        predictions["forecast_method"].eq("revenue_adjustment_formula")
    ]
    strict_metrics = compute_metrics(strict_formula, "formula_adjusted_revenue")

    if adjusted["WMAPE"] < seasonal["WMAPE"] and adjusted["WMAPE"] < last["WMAPE"]:
        verdict = "公式在全市場共同樣本的 WMAPE 同時優於兩個簡單基準，可保留為候選模型。"
    else:
        verdict = "公式尚未同時超越兩個簡單基準，不應直接取代現有模型；其結果可作為集成輸入或依股票類型啟用。"

    existing_section = (
        "## 與既有 SARIMA／xLSTM 的共同樣本比較\n\n"
        + format_metrics_table(existing)
        + "\n\n此表只保留三種方法都有預測值的同一批股票月份，避免樣本不同造成誤判。\n"
        if not existing.empty
        else "## 與既有 SARIMA／xLSTM 的比較\n\n找不到既有逐月預測檔，本次未產生直接比較。\n"
    )

    return f"""# 營收調整公式實驗報告

產生日期：{date.today():%Y-%m-%d}

## 實驗設計

- 參數選擇期間：2023-01 至 2024-12，逐月滾動，預測當下只使用以前月份。
- 最終評估期間：2025-01 至 2025-12；2025 實際營收只用於事後計分。
- 候選參數：27 組。
- 選擇分數：`0.5 × 全市場 WMAPE + 0.5 × 個股 WMAPE 中位數`。
- 公式有效預測占 2025 資料：{formula_count:,}/{len(predictions):,}（{formula_share:.2f}%）。

## 選中的參數

- 去年同月／近期水準權重：`{selected.seasonal_weight:.2f}`／`{1-selected.seasonal_weight:.2f}`
- 殘差指數平滑係數：`{selected.residual_alpha:.2f}`
- 殘差校正強度：`{selected.residual_strength:.2f}`
- 年增率 log 上下限：`±{selected.growth_log_cap:.6f}`
- 最終殘差校正 log 上下限：`±{selected.correction_log_cap:.2f}`
- 驗證期最佳平衡分數：`{sweep.iloc[0]['balanced_score']:.3f}`

## 2025 預測表現

{format_metrics_table(overall)}

上表是可運行的整體流程：公式所需歷史不足時會回退到去年同月。若嚴格只看真正套用公式的 `{strict_metrics['observations']:,}` 筆資料，WMAPE 為 `{strict_metrics['WMAPE']:.3f}%`、MedianAPE 為 `{strict_metrics['MedianAPE']:.3f}%`。部分營收接近 0 的股票會使一般 MAPE 極端放大，所以模型選擇以 WMAPE 與個股 WMAPE 中位數為主。

公式含校正相較：

- 去年同月基準的 WMAPE 改善：`{percentage_change(adjusted['WMAPE'], seasonal['WMAPE']):.2f}%`
- 沿用上月基準的 WMAPE 改善：`{percentage_change(adjusted['WMAPE'], last['WMAPE']):.2f}%`
- 未校正公式的 WMAPE 改善：`{percentage_change(adjusted['WMAPE'], base['WMAPE']):.2f}%`

完整 12 個月個股共 `{len(complete):,}` 檔；其中公式 WMAPE 超過 15% 的有 `{len(high_error):,}` 檔。

## 判讀

{verdict}

殘差強度若被選成 0，代表在驗證期中「追加誤差校正」沒有穩定增益；這也是有效實驗結果，不應為了讓公式更複雜而強制使用。

{existing_section}

## 下一步

1. 在同一個時間切分下加入 Holt-Winters／ETS 與非季節 ARIMA。
2. 對 SARIMA、公式、xLSTM 做只使用驗證資料決定權重的集成。
3. 依營收規模、季節性強度與波動度學習 gating，不永久標記股票類型。
4. 若再測小型 LSTM，改做跨股票共用模型並預測公式／ETS 的殘差，不為每檔股票各訓練深網路。

詳細公式與改善路線請見 `IMPROVEMENT_PLAN.md`；逐月、逐股和參數搜尋結果請見同資料夾 CSV。
"""


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    revenue = load_revenue_data()
    if args.stock_limit is not None:
        stock_ids = sorted(revenue["stock_id"].unique())[: args.stock_limit]
        revenue = revenue[revenue["stock_id"].isin(stock_ids)].copy()
    metadata = load_stock_metadata()

    selected, sweep = tune_formula(revenue)
    predictions = build_rolling_predictions(
        revenue,
        selected,
        start_date="2025-01-01",
        end_date="2025-12-01",
    )
    overall = evaluate_models(predictions)
    stocks = stock_accuracy(predictions, metadata)
    monthly = monthly_accuracy(predictions)
    existing = compare_existing_models(predictions)

    predictions = predictions.merge(metadata, on="stock_id", how="left")
    prediction_columns = [
        "stock_id",
        "stock_name",
        "industry_category",
        "target_date",
        "target_year",
        "target_month",
        "history_months",
        "actual_revenue",
        "last_observed_revenue",
        "seasonal_naive_revenue",
        "seasonal_growth_forecast",
        "formula_base_revenue",
        "formula_adjusted_revenue",
        "growth_log",
        "residual_state",
        "applied_correction_log",
        "mase_scale",
        "forecast_method",
    ]
    predictions[prediction_columns].to_csv(
        output_dir / "monthly_predictions.csv", index=False, encoding="utf-8-sig"
    )
    sweep.to_csv(output_dir / "parameter_sweep.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output_dir / "overall_accuracy.csv", index=False, encoding="utf-8-sig")
    stocks.to_csv(output_dir / "stock_accuracy.csv", index=False, encoding="utf-8-sig")
    stocks[(stocks["observations"] >= 12) & (stocks["WMAPE"] > 15)].sort_values(
        ["WMAPE", "stock_id"]
    ).to_csv(
        output_dir / "stocks_over_15pct_wmape.csv", index=False, encoding="utf-8-sig"
    )
    monthly.to_csv(output_dir / "monthly_accuracy.csv", index=False, encoding="utf-8-sig")
    if not existing.empty:
        existing.to_csv(
            output_dir / "comparison_with_sarima_xlstm.csv",
            index=False,
            encoding="utf-8-sig",
        )

    run_config = {
        "selection_period": ["2023-01-01", "2024-12-01"],
        "evaluation_period": ["2025-01-01", "2025-12-01"],
        "candidate_count": len(sweep),
        "selected_config": selected.as_dict(),
        "stock_limit": args.stock_limit,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = build_report(selected, sweep, overall, stocks, predictions, existing)
    report_path = output_dir / "營收調整公式實驗報告.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Selected config: {selected.as_dict()}")
    print(overall.to_string(index=False))
    if not existing.empty:
        print(existing.to_string(index=False))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
