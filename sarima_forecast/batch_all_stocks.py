from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import sarima_engine as engine


SYSTEM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SYSTEM_DIR.parent
DEFAULT_OUTPUT_DIR = SYSTEM_DIR / "outputs" / "full_universe_20260818"
_WORKER_REVENUE: pd.DataFrame | None = None


def _worker_init(revenue_path: str) -> None:
    global _WORKER_REVENUE
    _WORKER_REVENUE = engine.load_revenue_data(revenue_path)


def _run_stock(stock_id: int, config_values: dict[str, object]) -> dict[str, object]:
    if _WORKER_REVENUE is None:
        raise RuntimeError("Worker revenue data is not initialized.")
    started = time.perf_counter()
    config = engine.SarimaConfig(**config_values)
    result = engine.build_rolling_sarima_forecast(
        _WORKER_REVENUE,
        selected_stock=int(stock_id),
        config=config,
    )
    metric = result.metrics.iloc[0].to_dict()
    monthly = result.forecast.copy()
    monthly["target_date"] = monthly["target_date"].dt.strftime("%Y-%m-%d")
    successful_search = result.order_search[
        result.order_search.get("status", pd.Series(dtype=str)).eq("ok")
    ]
    best_aic = (
        float(successful_search.iloc[0]["aic"])
        if not successful_search.empty
        else np.nan
    )
    return {
        "stock_id": int(stock_id),
        "metric": metric,
        "monthly": monthly.to_dict(orient="records"),
        "selected_order": str(result.selected_order) if result.selected_order else "",
        "selected_seasonal_order": (
            str(result.selected_seasonal_order)
            if result.selected_seasonal_order
            else ""
        ),
        "best_aic": best_aic,
        "runtime_seconds": time.perf_counter() - started,
    }


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _metric_row(frame: pd.DataFrame, cohort: str) -> dict[str, object]:
    metrics = engine.compute_metrics(frame)
    return {
        "cohort": cohort,
        "stock_count": int(frame["stock_id"].nunique()),
        "prediction_count": int(
            frame.dropna(subset=["actual_revenue", "predicted_revenue_sarima"]).shape[0]
        ),
        **metrics,
    }


def _group_metrics(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_value, group in frame.groupby(group_column, dropna=False):
        output_group_value: object
        if group_column == "target_month" and pd.notna(group_value):
            output_group_value = int(group_value)
        else:
            output_group_value = _safe_text(group_value) or "未分類"
        row = {
            group_column: output_group_value,
            "stock_count": int(group["stock_id"].nunique()),
            "prediction_count": int(
                group.dropna(
                    subset=["actual_revenue", "predicted_revenue_sarima"]
                ).shape[0]
            ),
            **engine.compute_metrics(group),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _stock_metadata(revenue: pd.DataFrame) -> pd.DataFrame:
    stock_list_path = PROJECT_ROOT / "data" / "stock_list_new.csv"
    if stock_list_path.exists():
        metadata = pd.read_csv(stock_list_path)
        metadata = metadata.loc[
            :, [column for column in ["stock_id", "stock_name", "industry_category"] if column in metadata]
        ].copy()
        metadata["stock_id"] = pd.to_numeric(metadata["stock_id"], errors="coerce")
        metadata = metadata.dropna(subset=["stock_id"])
        metadata["stock_id"] = metadata["stock_id"].astype(int)
        metadata = metadata.drop_duplicates("stock_id", keep="last")
    else:
        metadata = pd.DataFrame(columns=["stock_id", "stock_name", "industry_category"])

    revenue_metadata = (
        revenue.sort_values("date")
        .groupby("stock_id", as_index=False)
        .last()[["stock_id", "industry_category"]]
        if "industry_category" in revenue.columns
        else pd.DataFrame(columns=["stock_id", "industry_category"])
    )
    metadata = revenue_metadata.merge(metadata, on="stock_id", how="outer", suffixes=("_revenue", ""))
    if "industry_category_revenue" in metadata:
        metadata["industry_category"] = metadata["industry_category"].fillna(
            metadata["industry_category_revenue"]
        )
        metadata = metadata.drop(columns=["industry_category_revenue"])
    if "stock_name" not in metadata:
        metadata["stock_name"] = ""
    if "industry_category" not in metadata:
        metadata["industry_category"] = "未分類"
    metadata["stock_name"] = metadata["stock_name"].fillna("").astype(str)
    metadata["industry_category"] = metadata["industry_category"].fillna("未分類").astype(str)
    return metadata[["stock_id", "stock_name", "industry_category"]]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_checkpoints(
    stock_rows: list[dict[str, object]],
    monthly_rows: list[dict[str, object]],
    failures: list[dict[str, object]],
    output_dir: Path,
) -> None:
    _write_csv(pd.DataFrame(stock_rows), output_dir / "stock_accuracy.partial.csv")
    _write_csv(pd.DataFrame(monthly_rows), output_dir / "monthly_predictions.partial.csv")
    _write_csv(pd.DataFrame(failures), output_dir / "failed_runs.partial.csv")


def _load_checkpoints(output_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    def load(name: str) -> list[dict]:
        path = output_dir / name
        if not path.exists() or path.stat().st_size == 0:
            return []
        try:
            return pd.read_csv(path).to_dict(orient="records")
        except pd.errors.EmptyDataError:
            return []

    return (
        load("stock_accuracy.partial.csv"),
        load("monthly_predictions.partial.csv"),
        load("failed_runs.partial.csv"),
    )


def _accuracy_buckets(
    stock_accuracy: pd.DataFrame,
    metric_column: str = "WMAPE_numeric_valid",
) -> pd.DataFrame:
    bins = [-np.inf, 5, 10, 15, 20, 30, 50, np.inf]
    labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-50%", "50%以上"]
    bucket = pd.cut(stock_accuracy[metric_column], bins=bins, labels=labels, right=True)
    counts = bucket.value_counts(sort=False).reindex(labels, fill_value=0)
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "WMAPE區間": labels,
            "股票數": counts.to_numpy(dtype=int),
            "占比(%)": np.where(total > 0, counts.to_numpy(dtype=float) / total * 100, np.nan),
        }
    )


def _format_pct(value: object) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _markdown_report(
    overall: pd.DataFrame,
    stock_accuracy: pd.DataFrame,
    monthly_accuracy: pd.DataFrame,
    industry_accuracy: pd.DataFrame,
    order_frequency: pd.DataFrame,
    accuracy_buckets: pd.DataFrame,
    method_accuracy: pd.DataFrame,
    quality_issues: pd.DataFrame,
    failures: pd.DataFrame,
    run_config: dict[str, object],
) -> str:
    all_row = overall.loc[overall["cohort"].eq("全部可評估月份（原始）")].iloc[0]
    raw_complete = overall.loc[overall["cohort"].eq("完整12個月股票（原始）")].iloc[0]
    complete = overall.loc[overall["cohort"].eq("完整12個月股票（排除數值飽和）")].iloc[0]
    over_15 = stock_accuracy[stock_accuracy["WMAPE_numeric_valid"] > 15].copy()
    best = stock_accuracy.sort_values("WMAPE_numeric_valid").head(10)
    worst = stock_accuracy.sort_values("WMAPE_numeric_valid", ascending=False).head(20)

    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        view = frame[columns].copy()
        for column in [
            "WMAPE", "SMAPE", "MedianAPE", "MAPE", "UnderestimateRate",
            "DirectionAccuracy", "WMAPE_numeric_valid", "SMAPE_numeric_valid",
            "MedianAPE_numeric_valid",
        ]:
            if column in view:
                view[column] = view[column].map(_format_pct)
        headers = [str(column) for column in view.columns]
        rows = [[_safe_text(value).replace("|", "\\|") for value in row] for row in view.itertuples(index=False, name=None)]
        output = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        output.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(output)

    lines = [
        "# SARIMA 全市場 2025 月營收滾動預測評估報告",
        "",
        f"- 產生時間：{run_config['finished_at']}",
        f"- 資料來源：`{run_config['revenue_file']}`",
        f"- 預測方法：SARIMA，季節週期 12 個月，使用 log1p 轉換。",
        f"- 評估範圍：{int(all_row['stock_count']):,} 檔股票、{int(all_row['prediction_count']):,} 筆有實際值的月預測。",
        "- 防止偷看答案：參數只以 2025 年以前資料依 AIC 選擇；每個 2025 月份只使用該月以前的營收重新做一步預測。",
        "",
        "## 一、整體結論",
        "",
        f"完整 12 個月股票共 {int(complete['stock_count']):,} 檔；共有 {len(quality_issues):,} 筆數值品質失效（缺值或飽和），穩健 pooled WMAPE 為 {_format_pct(complete['WMAPE'])}、SMAPE 為 {_format_pct(complete['SMAPE'])}、MedianAPE 為 {_format_pct(complete['MedianAPE'])}。",
        f"原始完整樣本 pooled WMAPE 為 {_format_pct(raw_complete['WMAPE'])}；此數字被數值飽和值嚴重扭曲，不可單獨用來判斷一般股票表現。",
        f"以每檔股票的數值有效 WMAPE 判斷，共 {len(over_15):,} 檔超過 15%，占已完成股票 {len(over_15) / max(len(stock_accuracy), 1) * 100:.2f}%。",
        f"SARIMA 實際使用月份共 {int(stock_accuracy['sarima_months'].sum()):,} 筆；季節天真 fallback 共 {int(stock_accuracy['fallback_months'].sum()):,} 筆。",
        "",
        "## 二、整體誤差",
        "",
        table(overall, ["cohort", "stock_count", "prediction_count", "WMAPE", "SMAPE", "MedianAPE", "MAPE", "UnderestimateRate", "DirectionAccuracy"]),
        "",
        "### 預測方式拆分（已排除數值飽和）",
        "",
        table(method_accuracy, ["forecast_method", "stock_count", "prediction_count", "WMAPE", "SMAPE", "MedianAPE", "UnderestimateRate", "DirectionAccuracy"]),
        "",
        "## 三、月份誤差",
        "",
        table(monthly_accuracy, ["target_month", "stock_count", "prediction_count", "WMAPE", "SMAPE", "MedianAPE", "UnderestimateRate", "DirectionAccuracy"]),
        "",
        "## 四、個股誤差分布",
        "",
        table(accuracy_buckets.round({"占比(%)": 2}), list(accuracy_buckets.columns)),
        "",
        "### WMAPE 最小的 10 檔",
        "",
        table(best, ["stock_id", "stock_name", "industry_category", "actual_months", "WMAPE_numeric_valid", "SMAPE_numeric_valid", "MedianAPE_numeric_valid"]),
        "",
        "### WMAPE 最大的 20 檔",
        "",
        table(worst, ["stock_id", "stock_name", "industry_category", "actual_months", "WMAPE_numeric_valid", "SMAPE_numeric_valid", "MedianAPE_numeric_valid"]),
        "",
        "完整的逐檔清單與全部數值有效 WMAPE 超過 15% 清單已放在 CSV 與 Excel 分頁，並按 WMAPE 由小到大排列。",
        "",
        "## 五、產業與參數概況",
        "",
        "### 產業 WMAPE 較低的 15 類",
        "",
        table(industry_accuracy.sort_values("WMAPE").head(15), ["industry_category", "stock_count", "prediction_count", "WMAPE", "SMAPE", "MedianAPE"]),
        "",
        "### 常見 SARIMA 參數",
        "",
        table(order_frequency.head(15), list(order_frequency.columns)),
        "",
        "## 六、數值品質異常",
        "",
        f"共有 {len(quality_issues):,} 筆預測為缺值或碰到 `expm1` 的 int64 上限，屬於數值失效而非合理營收預測；原始值完整保留，主要穩健指標另行排除。",
        "",
        table(quality_issues, ["stock_id", "stock_name", "target_month", "actual_revenue", "predicted_revenue_sarima", "quality_flag"]) if not quality_issues.empty else "無數值品質異常。",
        "",
        "## 七、改善方向",
        "",
        "1. 以 rolling-origin validation 取代單純 AIC 選模：AIC 衡量樣本內配適，未必直接對應 2025 預測誤差。",
        "2. 依股票穩定度決定模型：季節性穩定股票保留 SARIMA；結構突變、高成長與低基期股票改用 damped trend、Theta、ETS 或集成方式。",
        "3. 擴充但限制參數搜尋：加入少量 drift、不同差分組合與季節性檢定，同時以時間序列驗證避免過度擬合。",
        "4. 對異常營收做事件處理：合併、處分、認列時點與一次性大單會破壞固定季節模式，可加入 robust clipping 或事件旗標。",
        "5. 以產業或營收型態分群後再挑模型：不是把 cluster 當永久股票標籤，而是用過去視窗判斷當期型態。",
        "6. 報告同時保留 WMAPE、SMAPE、MedianAPE：低營收月份會使單月 MAPE 爆大，不能只看一個百分比指標。",
        "",
        "7. 在引擎加入收斂檢查與合理上限 guardrail：若預測碰到數值飽和，應自動改用前一年同月，而不是輸出極端值。",
        "",
        "## 八、限制",
        "",
        "- 本報告是 2025 歷史滾動回測，不代表未來實際報酬或投資建議。",
        "- SARIMA 僅使用單一股票自身月營收，沒有加入總體、產業、價格或公司事件資料。",
        "- 個股 WMAPE 以該股票可取得的 2025 實際月份計算；跨股票比較時應優先查看 `actual_months=12` 的完整樣本。",
        f"- 執行失敗股票數：{len(failures):,}。",
    ]
    return "\n".join(lines) + "\n"


def _finalize_outputs(
    stock_rows: list[dict[str, object]],
    monthly_rows: list[dict[str, object]],
    failures: list[dict[str, object]],
    metadata: pd.DataFrame,
    output_dir: Path,
    run_config: dict[str, object],
) -> None:
    stock_accuracy = pd.DataFrame(stock_rows).drop_duplicates("stock_id", keep="last")
    monthly = pd.DataFrame(monthly_rows).drop_duplicates(
        ["stock_id", "target_year", "target_month"], keep="last"
    )
    failures_frame = pd.DataFrame(
        failures, columns=["stock_id", "error_type", "error"]
    ).drop_duplicates("stock_id", keep="last")

    stock_accuracy = metadata.merge(stock_accuracy, on="stock_id", how="right")
    monthly = metadata.merge(monthly, on="stock_id", how="right")
    monthly["target_date"] = pd.to_datetime(monthly["target_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    numeric_monthly = [
        "last_observed_revenue", "predicted_revenue_sarima", "sarima_lower", "sarima_upper",
        "actual_revenue", "error", "abs_error", "absolute_percentage_error",
    ]
    for column in numeric_monthly:
        if column in monthly:
            monthly[column] = pd.to_numeric(monthly[column], errors="coerce")

    saturation_threshold = float(np.iinfo(np.int64).max) * 0.99
    predicted = monthly["predicted_revenue_sarima"]
    monthly["numeric_valid"] = np.isfinite(predicted) & (predicted < saturation_threshold)
    monthly["quality_flag"] = np.select(
        [predicted >= saturation_threshold, ~np.isfinite(predicted)],
        ["numeric_saturation", "missing_prediction"],
        default="ok",
    )
    quality_issues = monthly[
        monthly["actual_revenue"].notna() & ~monthly["numeric_valid"]
    ].copy()
    numeric_valid_monthly = monthly[monthly["numeric_valid"]].copy()

    valid_stock_rows: list[dict[str, object]] = []
    for stock_id, group in numeric_valid_monthly.groupby("stock_id"):
        metrics = engine.compute_metrics(group)
        valid_stock_rows.append(
            {
                "stock_id": int(stock_id),
                **{f"{name}_numeric_valid": value for name, value in metrics.items()},
            }
        )
    stock_accuracy = stock_accuracy.merge(
        pd.DataFrame(valid_stock_rows), on="stock_id", how="left"
    )

    complete_ids = set(
        stock_accuracy.loc[stock_accuracy["actual_months"].eq(12), "stock_id"].astype(int)
    )
    complete_monthly = monthly[monthly["stock_id"].isin(complete_ids)].copy()
    complete_numeric_valid = numeric_valid_monthly[
        numeric_valid_monthly["stock_id"].isin(complete_ids)
    ].copy()
    overall = pd.DataFrame(
        [
            _metric_row(monthly, "全部可評估月份（原始）"),
            _metric_row(complete_monthly, "完整12個月股票（原始）"),
            _metric_row(numeric_valid_monthly, "全部可評估月份（排除數值飽和）"),
            _metric_row(complete_numeric_valid, "完整12個月股票（排除數值飽和）"),
        ]
    )
    monthly_accuracy = _group_metrics(
        complete_numeric_valid, "target_month"
    ).sort_values("target_month")
    industry_accuracy = _group_metrics(
        complete_numeric_valid, "industry_category"
    ).sort_values("WMAPE")
    method_accuracy = _group_metrics(
        numeric_valid_monthly, "forecast_method"
    ).sort_values("WMAPE")

    order_frequency = (
        stock_accuracy.groupby(["selected_order", "selected_seasonal_order"], dropna=False)
        .agg(
            stock_count=("stock_id", "nunique"),
            median_stock_WMAPE=("WMAPE_numeric_valid", "median"),
            pooled_fallback_months=("fallback_months", "sum"),
        )
        .reset_index()
        .sort_values(["stock_count", "median_stock_WMAPE"], ascending=[False, True])
    )
    order_frequency["median_stock_WMAPE"] = order_frequency["median_stock_WMAPE"].round(3)
    accuracy_buckets = _accuracy_buckets(stock_accuracy)
    over_15 = stock_accuracy[stock_accuracy["WMAPE_numeric_valid"] > 15].sort_values(
        "WMAPE_numeric_valid"
    )

    preferred_stock_columns = [
        "stock_id", "stock_name", "industry_category", "actual_months", "WMAPE", "SMAPE",
        "MedianAPE", "MAPE", "RMSE", "MAE", "Bias", "UnderestimateRate",
        "DirectionAccuracy", "WMAPE_numeric_valid", "SMAPE_numeric_valid",
        "MedianAPE_numeric_valid", "MAPE_numeric_valid", "RMSE_numeric_valid",
        "MAE_numeric_valid", "Bias_numeric_valid", "UnderestimateRate_numeric_valid",
        "DirectionAccuracy_numeric_valid", "sarima_months", "fallback_months", "selected_order",
        "selected_seasonal_order", "best_aic", "runtime_seconds",
    ]
    over_15_columns = [
        "stock_id", "stock_name", "industry_category", "actual_months",
        "WMAPE_numeric_valid", "SMAPE_numeric_valid", "MedianAPE_numeric_valid",
        "MAPE_numeric_valid", "WMAPE", "SMAPE", "MedianAPE", "MAPE", "RMSE", "MAE",
        "Bias", "UnderestimateRate", "DirectionAccuracy", "sarima_months",
        "fallback_months", "selected_order", "selected_seasonal_order", "best_aic",
        "runtime_seconds",
    ]
    over_15 = over_15[[column for column in over_15_columns if column in over_15]]
    stock_accuracy = stock_accuracy[[column for column in preferred_stock_columns if column in stock_accuracy]]
    monthly_columns = [
        "stock_id", "stock_name", "industry_category", "target_date", "target_year", "target_month",
        "history_months", "last_observed_revenue", "predicted_revenue_sarima", "sarima_lower",
        "sarima_upper", "actual_revenue", "error", "abs_error", "absolute_percentage_error",
        "forecast_method", "fallback_reason", "numeric_valid", "quality_flag",
    ]
    monthly = monthly[[column for column in monthly_columns if column in monthly]]

    outputs = {
        "overall_accuracy.csv": overall,
        "method_accuracy.csv": method_accuracy,
        "monthly_accuracy.csv": monthly_accuracy,
        "industry_accuracy.csv": industry_accuracy,
        "stock_accuracy.csv": stock_accuracy.sort_values("WMAPE_numeric_valid"),
        "over_15pct_ascending.csv": over_15,
        "quality_issues.csv": quality_issues,
        "accuracy_buckets.csv": accuracy_buckets,
        "order_frequency.csv": order_frequency,
        "monthly_predictions.csv": monthly.sort_values(["stock_id", "target_month"]),
        "failed_runs.csv": failures_frame,
    }
    for filename, frame in outputs.items():
        _write_csv(frame, output_dir / filename)

    run_config["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    run_config["completed_stock_count"] = int(len(stock_accuracy))
    run_config["failed_stock_count"] = int(len(failures_frame))
    run_config["complete_12m_stock_count"] = int(len(complete_ids))
    run_config["quality_issue_count"] = int(len(quality_issues))
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = _markdown_report(
        overall, stock_accuracy, monthly_accuracy, industry_accuracy,
        order_frequency, accuracy_buckets, method_accuracy, quality_issues,
        failures_frame, run_config,
    )
    (output_dir / "SARIMA全市場預測評估報告_20260818.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the independent SARIMA workflow for all evaluable stocks.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maxiter", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Use --resume or a new path.")
    output_dir.mkdir(parents=True, exist_ok=True)

    revenue_path = PROJECT_ROOT / "data" / engine.REVENUE_FILENAME
    revenue = engine.load_revenue_data(revenue_path)
    metadata = _stock_metadata(revenue)
    evaluable = (
        revenue[revenue["revenue_year"].eq(engine.FORECAST_YEAR)]
        .groupby("stock_id")["revenue_thousand"]
        .count()
    )
    stocks = sorted(evaluable[evaluable > 0].index.astype(int).tolist())
    if args.limit is not None:
        stocks = stocks[: max(0, int(args.limit))]

    stock_rows, monthly_rows, failures = _load_checkpoints(output_dir) if args.resume else ([], [], [])
    completed = {int(row["stock_id"]) for row in stock_rows}
    failed_ids = {int(row["stock_id"]) for row in failures}
    pending = [stock_id for stock_id in stocks if stock_id not in completed and stock_id not in failed_ids]
    config_values: dict[str, object] = {
        "forecast_year": engine.FORECAST_YEAR,
        "min_history_months": engine.MIN_HISTORY_MONTHS,
        "confidence_level": 0.95,
        "maxiter": int(args.maxiter),
    }
    run_config: dict[str, object] = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "revenue_file": str(revenue_path),
        "requested_stock_count": len(stocks),
        "worker_count": int(args.workers),
        "sarima_orders": [str(value) for value in engine.SARIMA_ORDERS],
        "seasonal_orders": [str(value) for value in engine.SARIMA_SEASONAL_ORDERS],
        **config_values,
    }
    print(
        f"Starting SARIMA batch: requested={len(stocks)}, pending={len(pending)}, "
        f"workers={args.workers}, output={output_dir}",
        flush=True,
    )
    batch_started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_worker_init,
        initargs=(str(revenue_path),),
    ) as executor:
        futures = {
            executor.submit(_run_stock, stock_id, config_values): stock_id
            for stock_id in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            stock_id = futures[future]
            try:
                result = future.result()
                monthly = result.pop("monthly")
                metric = result.pop("metric")
                metric.update(result)
                metric["actual_months"] = int(
                    sum(pd.notna(row.get("actual_revenue")) for row in monthly)
                )
                metric["sarima_months"] = int(
                    sum(row.get("forecast_method") == "sarima" for row in monthly)
                )
                metric["fallback_months"] = int(
                    sum(row.get("forecast_method") != "sarima" for row in monthly)
                )
                stock_rows.append(metric)
                monthly_rows.extend(monthly)
            except Exception as error:
                failures.append(
                    {
                        "stock_id": int(stock_id),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            if index % max(1, int(args.checkpoint_every)) == 0 or index == len(pending):
                _write_checkpoints(stock_rows, monthly_rows, failures, output_dir)
                elapsed = time.perf_counter() - batch_started
                print(
                    f"Progress {index}/{len(pending)}; completed={len(stock_rows)}, "
                    f"failed={len(failures)}, elapsed={elapsed:.1f}s",
                    flush=True,
                )

    _finalize_outputs(stock_rows, monthly_rows, failures, metadata, output_dir, run_config)
    print(
        f"Finished: completed={len(stock_rows)}, failed={len(failures)}, "
        f"elapsed={time.perf_counter() - batch_started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
