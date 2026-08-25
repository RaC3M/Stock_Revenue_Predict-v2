from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
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
    model_comparison,
    search_sarima_weight,
    stock_metrics,
)
from revenue_adjustment_formula.formula_engine import (  # noqa: E402
    FormulaConfig,
    build_rolling_predictions,
    load_revenue_data as load_formula_revenue,
)
from sarima_forecast import sarima_engine  # noqa: E402


DEFAULT_OUTPUT_DIR = SYSTEM_DIR / "outputs" / "full_universe_20260820"
EXISTING_2025_SARIMA = (
    PROJECT_ROOT
    / "sarima_forecast"
    / "outputs"
    / "full_universe_20260818"
    / "monthly_predictions.csv"
)
FORMULA_CONFIG = FormulaConfig(
    seasonal_weight=0.5,
    residual_alpha=0.1,
    residual_strength=0.0,
    growth_log_cap=float(np.log(2.0)),
    correction_log_cap=0.5,
)
VALIDATION_YEARS = (2023, 2024)
_WORKER_REVENUE: pd.DataFrame | None = None


def _worker_init(revenue_path: str) -> None:
    global _WORKER_REVENUE
    warnings.filterwarnings("ignore", module="statsmodels")
    _WORKER_REVENUE = sarima_engine.load_revenue_data(revenue_path)


def _run_stock_validation(
    stock_id: int,
    maxiter: int,
) -> dict[str, object]:
    if _WORKER_REVENUE is None:
        raise RuntimeError("Worker revenue data is not initialized.")
    rows: list[dict[str, object]] = []
    for year in VALIDATION_YEARS:
        result = sarima_engine.build_rolling_sarima_forecast(
            _WORKER_REVENUE,
            selected_stock=int(stock_id),
            config=sarima_engine.SarimaConfig(
                forecast_year=int(year),
                min_history_months=sarima_engine.MIN_HISTORY_MONTHS,
                confidence_level=0.95,
                maxiter=int(maxiter),
            ),
        )
        forecast = result.forecast.copy()
        forecast["selected_order"] = str(result.selected_order or "")
        forecast["selected_seasonal_order"] = str(result.selected_seasonal_order or "")
        forecast["target_date"] = forecast["target_date"].dt.strftime("%Y-%m-%d")
        rows.extend(forecast.to_dict(orient="records"))
    return {"stock_id": int(stock_id), "monthly": rows}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _stock_metadata(revenue: pd.DataFrame) -> pd.DataFrame:
    list_path = PROJECT_ROOT / "data" / "stock_list_new.csv"
    metadata = pd.read_csv(list_path) if list_path.exists() else pd.DataFrame()
    keep = [column for column in ["stock_id", "stock_name", "industry_category"] if column in metadata]
    metadata = metadata[keep].copy() if keep else pd.DataFrame(columns=["stock_id"])
    if "stock_id" in metadata:
        metadata["stock_id"] = pd.to_numeric(metadata["stock_id"], errors="coerce")
        metadata = metadata.dropna(subset=["stock_id"])
        metadata["stock_id"] = metadata["stock_id"].astype(int)
        metadata = metadata.drop_duplicates("stock_id", keep="last")
    revenue_meta = revenue.sort_values("date").groupby("stock_id", as_index=False).last()
    revenue_meta = revenue_meta[["stock_id", "industry_category"]]
    output = revenue_meta.merge(metadata, on="stock_id", how="outer", suffixes=("_data", ""))
    if "industry_category_data" in output:
        if "industry_category" not in output:
            output["industry_category"] = output["industry_category_data"]
        else:
            output["industry_category"] = output["industry_category"].fillna(
                output["industry_category_data"]
            )
    if "stock_name" not in output:
        output["stock_name"] = ""
    if "industry_category" not in output:
        output["industry_category"] = "未知產業"
    return output[["stock_id", "stock_name", "industry_category"]].fillna("")


def _generate_validation_sarima(
    revenue_path: Path,
    stocks: list[int],
    output_dir: Path,
    workers: int,
    maxiter: int,
    checkpoint_every: int,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_path = output_dir / "validation_sarima_predictions.csv"
    failure_path = output_dir / "validation_sarima_failures.csv"
    if final_path.exists():
        print(f"Reusing validation cache: {final_path}", flush=True)
        try:
            failures = pd.read_csv(failure_path) if failure_path.exists() else pd.DataFrame()
        except pd.errors.EmptyDataError:
            failures = pd.DataFrame(columns=["stock_id", "error_type", "error"])
        return pd.read_csv(final_path), failures

    partial_path = output_dir / "validation_sarima_predictions.partial.csv"
    partial_failure_path = output_dir / "validation_sarima_failures.partial.csv"
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    if resume and partial_path.exists():
        rows = pd.read_csv(partial_path).to_dict(orient="records")
    if resume and partial_failure_path.exists() and partial_failure_path.stat().st_size:
        failures = pd.read_csv(partial_failure_path).to_dict(orient="records")
    completed = {int(row["stock_id"]) for row in rows}
    failed = {int(row["stock_id"]) for row in failures}
    pending = [stock_id for stock_id in stocks if stock_id not in completed and stock_id not in failed]
    started = time.perf_counter()
    print(
        f"Validation SARIMA: stocks={len(stocks)}, pending={len(pending)}, workers={workers}",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=max(1, int(workers)),
        initializer=_worker_init,
        initargs=(str(revenue_path),),
    ) as executor:
        futures = {
            executor.submit(_run_stock_validation, stock_id, int(maxiter)): stock_id
            for stock_id in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            stock_id = futures[future]
            try:
                rows.extend(future.result()["monthly"])
            except Exception as error:
                failures.append(
                    {
                        "stock_id": int(stock_id),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            if index % max(1, int(checkpoint_every)) == 0 or index == len(pending):
                _write_csv(pd.DataFrame(rows), partial_path)
                _write_csv(pd.DataFrame(failures), partial_failure_path)
                print(
                    f"Validation progress {index}/{len(pending)}; rows={len(rows)}, "
                    f"failed={len(failures)}, elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    validation = pd.DataFrame(rows)
    _write_csv(validation, final_path)
    _write_csv(pd.DataFrame(failures), failure_path)
    return validation, pd.DataFrame(failures)


def _comparison_by_month(common: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    mappings = {
        "營收調整公式": "formula_adjusted_revenue",
        "SARIMA": "predicted_revenue_sarima",
        "SARIMA＋營收公式": "hybrid_predicted_revenue",
    }
    for month, group in common.groupby("target_month"):
        for model, column in mappings.items():
            rows.append(
                {
                    "target_month": int(month),
                    "model": model,
                    **compute_metrics(group, column),
                }
            )
    return pd.DataFrame(rows)


def _stock_comparison(complete: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    hybrid = stock_metrics(complete, "hybrid_predicted_revenue").add_prefix("hybrid_")
    hybrid = hybrid.rename(columns={"hybrid_stock_id": "stock_id"})
    formula = stock_metrics(complete, "formula_adjusted_revenue").add_prefix("formula_")
    formula = formula.rename(columns={"formula_stock_id": "stock_id"})
    sarima = stock_metrics(complete, "predicted_revenue_sarima").add_prefix("sarima_")
    sarima = sarima.rename(columns={"sarima_stock_id": "stock_id"})
    output = hybrid.merge(formula, on="stock_id").merge(sarima, on="stock_id")
    output = metadata.merge(output, on="stock_id", how="right")
    output["hybrid_beats_formula"] = output["hybrid_WMAPE"] < output["formula_WMAPE"]
    output["hybrid_beats_sarima"] = output["hybrid_WMAPE"] < output["sarima_WMAPE"]
    return output.sort_values("hybrid_WMAPE").reset_index(drop=True)


def _report(
    weight: float,
    validation_sweep: pd.DataFrame,
    overall: pd.DataFrame,
    stock_accuracy: pd.DataFrame,
    test: pd.DataFrame,
) -> str:
    lookup = overall.set_index("model")
    hybrid = lookup.loc["SARIMA＋營收公式"]
    formula = lookup.loc["營收調整公式"]
    sarima = lookup.loc["SARIMA"]
    complete_count = int(stock_accuracy["stock_id"].nunique())
    under_15 = int((stock_accuracy["hybrid_WMAPE"] <= 15).sum())
    fallback_count = int(test["hybrid_method"].ne("weighted_hybrid").sum())
    best_row = validation_sweep.loc[validation_sweep["sarima_weight"].eq(weight)].iloc[0]
    return f"""# SARIMA＋營收調整公式：全市場預測評估

## 實驗設計

- 驗證期間：2023–2024，只用來選擇混合權重。
- 測試期間：2025，權重凍結後只評估一次。
- 權重搜尋：SARIMA 權重 0.0～1.0，每次增加 0.1。
- 選擇準則：`0.5 × 全體 WMAPE + 0.5 × 個股 WMAPE 中位數`。
- 最終公式：`{weight:.1f} × SARIMA + {1-weight:.1f} × 營收調整公式`。
- SARIMA 無法擬合、數值不合法，或高於「公式預測與上月營收較大值的 2 倍」時，自動使用營收調整公式。

## 2023–2024 驗證結果

- 選定 SARIMA 權重：{weight:.1f}
- 選定公式權重：{1-weight:.1f}
- 驗證 WMAPE：{best_row['WMAPE']:.3f}%
- 驗證個股 WMAPE 中位數：{best_row['median_stock_WMAPE']:.3f}%
- 驗證平衡分數：{best_row['balanced_score']:.3f}

## 2025 公平比較

以下三種方法只比較兩個基礎模型都有效、且個股有完整 12 個月的共同樣本。

| 模型 | WMAPE | SMAPE | MedianAPE | P90APE | 方向準確率 |
|---|---:|---:|---:|---:|---:|
| 營收調整公式 | {formula['WMAPE']:.3f}% | {formula['SMAPE']:.3f}% | {formula['MedianAPE']:.3f}% | {formula['P90APE']:.3f}% | {formula['DirectionAccuracy']:.3f}% |
| SARIMA | {sarima['WMAPE']:.3f}% | {sarima['SMAPE']:.3f}% | {sarima['MedianAPE']:.3f}% | {sarima['P90APE']:.3f}% | {sarima['DirectionAccuracy']:.3f}% |
| SARIMA＋營收公式 | {hybrid['WMAPE']:.3f}% | {hybrid['SMAPE']:.3f}% | {hybrid['MedianAPE']:.3f}% | {hybrid['P90APE']:.3f}% | {hybrid['DirectionAccuracy']:.3f}% |

## 個股層級

- 完整 12 個月共同樣本：{complete_count:,} 檔。
- 混合模型 WMAPE ≤ 15%：{under_15:,} 檔（{under_15 / complete_count * 100 if complete_count else 0:.2f}%）。
- 混合模型勝過營收公式：{int(stock_accuracy['hybrid_beats_formula'].sum()):,} 檔。
- 混合模型勝過 SARIMA：{int(stock_accuracy['hybrid_beats_sarima'].sum()):,} 檔。
- 2025 營運預測中自動退回單一模型：{fallback_count:,} 筆。

## 解讀與後續改善

混合模型的主要作用是讓 SARIMA 的季節性與公式的穩定性互相抵消部分誤差。若 WMAPE 與 P90APE 同時下降，代表不只總體誤差改善，極端誤差也受到抑制。方向準確率不一定會同步提高，因為線性加權主要修正預測幅度，不是專門分類營收上升或下降。

下一步可在不動 2025 測試集的前提下，使用產業別權重、波動度分組權重，或加入指數平滑 ETS 作為第三個低資料量模型；每個改動都應重新以 2023–2024 選參數，再用 2025 做一次性比較。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run time-safe SARIMA + formula hybrid evaluation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 1)))
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    revenue_path = PROJECT_ROOT / "data" / sarima_engine.REVENUE_FILENAME
    revenue = load_formula_revenue(revenue_path)
    metadata = _stock_metadata(revenue)
    validation_ids = sorted(
        revenue.loc[revenue["revenue_year"].isin(VALIDATION_YEARS), "stock_id"]
        .astype(int)
        .unique()
        .tolist()
    )
    if args.limit is not None:
        validation_ids = validation_ids[: max(0, int(args.limit))]

    run_config = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "validation_years": list(VALIDATION_YEARS),
        "test_year": 2025,
        "requested_stock_count": len(validation_ids),
        "workers": int(args.workers),
        "maxiter": int(args.maxiter),
        "formula_config": FORMULA_CONFIG.as_dict(),
        "weight_grid": [round(value, 1) for value in np.arange(0.0, 1.01, 0.1)],
        "selection_score": "0.5 * pooled WMAPE + 0.5 * median stock WMAPE",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    validation_sarima, failures = _generate_validation_sarima(
        revenue_path,
        validation_ids,
        output_dir,
        int(args.workers),
        int(args.maxiter),
        int(args.checkpoint_every),
        bool(args.resume),
    )
    formula = build_rolling_predictions(
        revenue,
        FORMULA_CONFIG,
        start_date="2023-01-01",
        end_date="2025-12-01",
        stock_ids=validation_ids if args.limit is not None else None,
    )
    validation_formula = formula[formula["target_year"].isin(VALIDATION_YEARS)]
    selected_weight, sweep = search_sarima_weight(validation_formula, validation_sarima)
    _write_csv(sweep, output_dir / "validation_weight_search.csv")

    if not EXISTING_2025_SARIMA.exists():
        raise FileNotFoundError(f"Existing 2025 SARIMA output not found: {EXISTING_2025_SARIMA}")
    test_sarima = pd.read_csv(EXISTING_2025_SARIMA)
    if args.limit is not None:
        test_sarima = test_sarima[test_sarima["stock_id"].isin(validation_ids)]
    test_formula = formula[formula["target_year"].eq(2025)]
    test = combine_predictions(
        test_formula,
        test_sarima,
        HybridConfig(sarima_weight=selected_weight),
    )
    test = metadata.merge(test, on="stock_id", how="right")
    _write_csv(test, output_dir / "monthly_predictions.csv")

    common_counts = test[test["both_models_valid"]].groupby("stock_id").size()
    complete_ids = common_counts[common_counts.eq(12)].index
    complete = test[test["stock_id"].isin(complete_ids) & test["both_models_valid"]].copy()
    overall = model_comparison(complete)
    operational = {
        "model": "混合模型（含自動退回）",
        **compute_metrics(test, "hybrid_predicted_revenue"),
    }
    overall = pd.concat([overall, pd.DataFrame([operational])], ignore_index=True)
    stock_accuracy = _stock_comparison(complete, metadata)
    monthly_accuracy = _comparison_by_month(complete)
    over_15 = stock_accuracy[stock_accuracy["hybrid_WMAPE"] > 15].sort_values("hybrid_WMAPE")
    _write_csv(overall, output_dir / "overall_accuracy.csv")
    _write_csv(stock_accuracy, output_dir / "stock_accuracy.csv")
    _write_csv(monthly_accuracy, output_dir / "monthly_accuracy.csv")
    _write_csv(over_15, output_dir / "over_15pct_ascending.csv")
    _write_csv(failures, output_dir / "failed_runs.csv")

    report = _report(selected_weight, sweep, overall, stock_accuracy, test)
    (output_dir / "SARIMA加營收公式_全市場預測報告.md").write_text(report, encoding="utf-8")
    run_config.update(
        {
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "selected_sarima_weight": selected_weight,
            "selected_formula_weight": 1.0 - selected_weight,
            "complete_common_stock_count": int(len(complete_ids)),
            "failed_validation_stock_count": int(len(failures)),
            "test_sarima_source": str(EXISTING_2025_SARIMA),
        }
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Finished hybrid experiment: SARIMA weight={selected_weight:.1f}, "
        f"complete stocks={len(complete_ids)}, output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
