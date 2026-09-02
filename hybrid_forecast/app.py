from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


SYSTEM_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SYSTEM_DIR / "outputs" / "full_universe_20260820"
BREAK_OUTPUT_DIR = SYSTEM_DIR / "outputs" / "structural_break_20260820"


st.set_page_config(page_title="SARIMA＋營收公式", page_icon="📈", layout="wide")

sys.path.insert(0, str(SYSTEM_DIR.parent))
mode = st.sidebar.radio("運作模式", ["2025 測試沙盒", "實作模式"], index=1, key="forecast_mode")
if mode == "實作模式":
    from hybrid_forecast.live_ui import render_live_app

    render_live_app()
    st.stop()


@st.cache_data(show_spinner=False)
def load_results() -> tuple[dict[str, pd.DataFrame], dict[str, object], dict[str, object]]:
    files = {
        "overall": "overall_accuracy.csv",
        "monthly": "monthly_accuracy.csv",
        "predictions": "monthly_predictions.csv",
        "stocks": "stock_accuracy.csv",
        "high_error": "over_15pct_ascending.csv",
        "weights": "validation_weight_search.csv",
    }
    missing = [filename for filename in files.values() if not (OUTPUT_DIR / filename).exists()]
    if missing:
        raise FileNotFoundError("缺少實驗輸出：" + "、".join(missing))
    frames = {key: pd.read_csv(OUTPUT_DIR / filename) for key, filename in files.items()}
    break_files = {
        "break_overall": "overall_accuracy.csv",
        "break_monthly": "monthly_accuracy.csv",
        "break_predictions": "operational_comparison_monthly.csv",
        "break_stocks": "stock_accuracy.csv",
        "break_parameters": "parameter_sweep.csv",
    }
    break_missing = [
        filename
        for filename in break_files.values()
        if not (BREAK_OUTPUT_DIR / filename).exists()
    ]
    if break_missing:
        raise FileNotFoundError("缺少結構斷點輸出：" + "、".join(break_missing))
    frames.update(
        {
            key: pd.read_csv(BREAK_OUTPUT_DIR / filename)
            for key, filename in break_files.items()
        }
    )
    config = json.loads((OUTPUT_DIR / "run_config.json").read_text(encoding="utf-8"))
    break_config = json.loads(
        (BREAK_OUTPUT_DIR / "run_config.json").read_text(encoding="utf-8")
    )
    return frames, config, break_config


try:
    results, config, break_config = load_results()
except FileNotFoundError as error:
    st.error(str(error))
    st.info("請先執行 run_experiment.py 產生 2023–2024 驗證及 2025 測試結果。")
    st.stop()


overall = results["overall"]
selected_weight = float(config["selected_sarima_weight"])
comparison = overall[overall["model"].isin(["營收調整公式", "SARIMA", "SARIMA＋營收公式"])].copy()
metric_lookup = comparison.set_index("model")
hybrid = metric_lookup.loc["SARIMA＋營收公式"]
formula = metric_lookup.loc["營收調整公式"]
sarima = metric_lookup.loc["SARIMA"]


def delta_text(base_value: float, hybrid_value: float) -> str:
    difference = float(base_value) - float(hybrid_value)
    label = "改善" if difference >= 0 else "增加"
    return f"{label} {abs(difference):.2f} 個百分點"

st.title("SARIMA＋營收調整公式")
st.caption(
    "權重只由 2023–2024 滾動驗證決定；2025 是凍結參數後的獨立測試。"
    "SARIMA 無效時會自動退回營收公式。"
)

summary = st.columns(6)
summary[0].metric("SARIMA 權重", f"{selected_weight:.1f}")
summary[1].metric("公式權重", f"{1-selected_weight:.1f}")
summary[2].metric("混合 WMAPE", f"{hybrid['WMAPE']:.2f}%")
summary[3].metric(
    "相較 SARIMA",
    delta_text(sarima["WMAPE"], hybrid["WMAPE"]),
)
summary[4].metric(
    "相較公式",
    delta_text(formula["WMAPE"], hybrid["WMAPE"]),
)
summary[5].metric("完整共同樣本", f"{int(hybrid['stock_count']):,} 檔")

overview_tab, break_tab, stock_tab, monthly_tab, weight_tab, error_tab = st.tabs(
    ["整體結果", "結構斷點改善版", "個股預測", "月份表現", "權重選擇", "誤差超過 15%"]
)

percent_columns = [
    "WMAPE",
    "SMAPE",
    "MedianAPE",
    "P90APE",
    "MAPE",
    "UnderestimateRate",
    "DirectionAccuracy",
]

with overview_tab:
    st.subheader("2025 公平比較")
    st.write("三種模型只比較兩個基礎模型都有效、且全年 12 個月完整的共同樣本。")
    display = comparison[
        ["model", "observations", "stock_count", "WMAPE", "SMAPE", "MedianAPE", "P90APE", "MASE", "DirectionAccuracy"]
    ]
    st.dataframe(
        display.style.format({column: "{:.3f}%" for column in percent_columns if column in display}),
        width="stretch",
        hide_index=True,
    )
    st.bar_chart(comparison.set_index("model")[["WMAPE", "SMAPE", "MedianAPE", "P90APE"]])
    operational = overall[overall["model"].eq("混合模型（含自動退回）")]
    if not operational.empty:
        row = operational.iloc[0]
        st.info(
            f"納入 SARIMA 失敗時的公式退回後，共評估 {int(row['observations']):,} 筆、"
            f"{int(row['stock_count']):,} 檔，營運版 WMAPE 為 {row['WMAPE']:.3f}%。"
        )

with break_tab:
    st.subheader("結構性崩落偵測版：原版與改善版並排比較")
    st.write(
        "若上月營收相較去年同期低於 20%，且也低於近 12 個月中位數的 30%，"
        "或最近三個月已有兩次崩落，就暫停去年同月季節基準，改採最近已知營收。"
    )
    break_overall = results["break_overall"].copy()
    break_lookup = break_overall.set_index("model")
    original_hybrid = break_lookup.loc["原混合模型"]
    improved_hybrid = break_lookup.loc["改善混合模型"]
    break_stocks = results["break_stocks"].copy()
    affected = int((break_stocks["structural_break_months"] > 0).sum())
    improved_count = int((break_stocks["WMAPE_improvement_points"] > 1e-9).sum())
    worsened_count = int((break_stocks["WMAPE_improvement_points"] < -1e-9).sum())
    break_cards = st.columns(6)
    break_cards[0].metric("原版 WMAPE", f"{original_hybrid['WMAPE']:.3f}%")
    break_cards[1].metric("改善版 WMAPE", f"{improved_hybrid['WMAPE']:.3f}%")
    break_cards[2].metric(
        "SMAPE 改善",
        f"{original_hybrid['SMAPE'] - improved_hybrid['SMAPE']:.3f} 個百分點",
    )
    break_cards[3].metric(
        "P90APE 改善",
        f"{original_hybrid['P90APE'] - improved_hybrid['P90APE']:.3f} 個百分點",
    )
    break_cards[4].metric("觸發股票", f"{affected} 檔")
    break_cards[5].metric("改善／變差", f"{improved_count}／{worsened_count} 檔")

    st.dataframe(
        break_overall[[
            "model", "observations", "stock_count", "WMAPE", "SMAPE",
            "MedianAPE", "P90APE", "DirectionAccuracy",
        ]].style.format(
            {
                "WMAPE": "{:.3f}%",
                "SMAPE": "{:.3f}%",
                "MedianAPE": "{:.3f}%",
                "P90APE": "{:.3f}%",
                "DirectionAccuracy": "{:.3f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.bar_chart(
        break_overall.set_index("model")[["WMAPE", "SMAPE", "MedianAPE", "P90APE"]]
    )

    break_stocks["label"] = (
        break_stocks["stock_id"].astype(int).astype(str)
        + " "
        + break_stocks["stock_name"].fillna("").astype(str)
    ).str.strip()
    break_labels = break_stocks.sort_values("stock_id")["label"].tolist()
    default_label = next(
        (label for label in break_labels if label.startswith("6405 ")),
        break_labels[0],
    )
    selected_break_label = st.selectbox(
        "選擇股票比較改善前後",
        break_labels,
        index=break_labels.index(default_label),
        key="structural_break_stock",
    )
    selected_break_id = int(selected_break_label.split()[0])
    break_row = break_stocks[break_stocks["stock_id"].eq(selected_break_id)].iloc[0]
    stock_cards = st.columns(5)
    stock_cards[0].metric("原版 WMAPE", f"{break_row['baseline_WMAPE']:.2f}%")
    stock_cards[1].metric("改善版 WMAPE", f"{break_row['improved_WMAPE']:.2f}%")
    stock_cards[2].metric(
        "改善幅度", f"{break_row['WMAPE_improvement_points']:.2f} 個百分點"
    )
    stock_cards[3].metric("觸發月份", f"{int(break_row['structural_break_months'])} 個月")
    stock_cards[4].metric(
        "是否改善", "是" if break_row["WMAPE_improvement_points"] > 0 else "否"
    )

    break_predictions = results["break_predictions"].copy()
    selected_break = break_predictions[
        break_predictions["stock_id"].eq(selected_break_id)
    ].copy()
    selected_break["target_date"] = pd.to_datetime(selected_break["target_date"])
    st.line_chart(
        selected_break.set_index("target_date")[[
            "actual_revenue",
            "baseline_hybrid_revenue",
            "structural_hybrid_revenue",
            "baseline_formula_revenue",
            "structural_formula_revenue",
        ]].rename(
            columns={
                "actual_revenue": "實際營收",
                "baseline_hybrid_revenue": "原混合模型",
                "structural_hybrid_revenue": "改善混合模型",
                "baseline_formula_revenue": "原營收公式",
                "structural_formula_revenue": "結構斷點公式",
            }
        )
    )
    st.dataframe(
        selected_break[[
            "target_date",
            "actual_revenue",
            "baseline_hybrid_revenue",
            "structural_hybrid_revenue",
            "baseline_formula_revenue",
            "structural_formula_revenue",
            "structural_break_detected",
            "break_last_yoy_ratio",
            "break_level_ratio",
            "structural_hybrid_method",
        ]].style.format(
            {
                "actual_revenue": "{:,.0f}",
                "baseline_hybrid_revenue": "{:,.0f}",
                "structural_hybrid_revenue": "{:,.0f}",
                "baseline_formula_revenue": "{:,.0f}",
                "structural_formula_revenue": "{:,.0f}",
                "break_last_yoy_ratio": "{:.3f}",
                "break_level_ratio": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("實際上線時如何觸發")
    st.info(
        "這不是提前預知第一次崩落，而是『已公布營收出現崩落後的修正機制』。"
        "預測目標月時，只能使用預測當下已公布的歷史營收，不會讀取目標月實際答案。"
    )
    st.markdown(
        """
觸發條件只使用最後已公布月份的資料：

```text
最後已公布月份營收 ÷ 去年同期營收 < 20%

而且

最後已公布月份營收 ÷ 過去 12 個月營收中位數 < 30%
```

- 如果預測時上月營收已公布，可以用上月資料判斷是否觸發。
- 如果月初預測、上月營收尚未公布，只能使用前兩個月資料，觸發會延後。
- 如果一次預測未來 12 個月、期間沒有新實際營收，斷點狀態也無法逐月更新。
- 第一次毫無前兆的崩落無法只靠歷史營收提前知道；若要提前預警，需要訂單、出貨、停產、重大訊息或資產處分等領先資料。
"""
    )
    timing_table = pd.DataFrame(
        [
            {
                "預測情境": "上月營收公布後預測當月",
                "最後可用營收": "上個月",
                "斷點反應": "可在第一個崩落月份公布後立即修正",
            },
            {
                "預測情境": "月初、上月營收尚未公布",
                "最後可用營收": "前兩個月",
                "斷點反應": "至少延後一個月",
            },
            {
                "預測情境": "一次預測未來 12 個月",
                "最後可用營收": "預測日起固定",
                "斷點反應": "無法自動更新，需等新營收公布後重跑",
            },
        ]
    )
    st.dataframe(timing_table, width="stretch", hide_index=True)
    st.caption(
        "正式部署時應以 revenue_available_date 作為資料截止條件，並在輸出中保存"
        "預測時間、最後已知月份及觸發原因，才能證明沒有使用未來資料。"
    )
    st.warning(
        "這是探索性改善：概念是在檢視 2025 失敗案例後提出。數值門檻與權重雖只用 "
        "2023–2024 選擇，仍需用 2026 或之後的新資料做真正的樣本外確認。"
    )

    with st.expander("查看 2023–2024 參數搜尋結果"):
        parameters = results["break_parameters"].head(27)
        st.dataframe(
            parameters.style.format(
                {
                    "yoy_ratio_threshold": "{:.2f}",
                    "level_ratio_threshold": "{:.2f}",
                    "formula_retention": "{:.2f}",
                    "selected_sarima_weight": "{:.1f}",
                    "balanced_score": "{:.4f}",
                    "WMAPE": "{:.3f}%",
                    "median_stock_WMAPE": "{:.3f}%",
                }
            ),
            width="stretch",
            hide_index=True,
        )

with stock_tab:
    stocks = results["stocks"].copy()
    stocks["label"] = (
        stocks["stock_id"].astype(int).astype(str)
        + " "
        + stocks["stock_name"].fillna("").astype(str)
    ).str.strip()
    selected_label = st.selectbox("選擇股票", stocks.sort_values("stock_id")["label"].tolist())
    selected_id = int(selected_label.split()[0])
    row = stocks[stocks["stock_id"].eq(selected_id)].iloc[0]
    cards = st.columns(5)
    cards[0].metric("混合 WMAPE", f"{row['hybrid_WMAPE']:.2f}%")
    cards[1].metric("SARIMA WMAPE", f"{row['sarima_WMAPE']:.2f}%")
    cards[2].metric("公式 WMAPE", f"{row['formula_WMAPE']:.2f}%")
    cards[3].metric("混合勝過 SARIMA", "是" if row["hybrid_beats_sarima"] else "否")
    cards[4].metric("混合勝過公式", "是" if row["hybrid_beats_formula"] else "否")

    predictions = results["predictions"]
    selected = predictions[predictions["stock_id"].eq(selected_id)].copy()
    selected["target_date"] = pd.to_datetime(selected["target_date"])
    chart = selected.set_index("target_date")[[
        "actual_revenue",
        "hybrid_predicted_revenue",
        "predicted_revenue_sarima",
        "formula_adjusted_revenue",
    ]].rename(
        columns={
            "actual_revenue": "實際營收",
            "hybrid_predicted_revenue": "混合預測",
            "predicted_revenue_sarima": "SARIMA",
            "formula_adjusted_revenue": "營收公式",
        }
    )
    st.line_chart(chart)
    st.dataframe(
        selected[[
            "target_date",
            "actual_revenue",
            "hybrid_predicted_revenue",
            "predicted_revenue_sarima",
            "formula_adjusted_revenue",
            "hybrid_ape",
            "hybrid_method",
        ]].style.format(
            {
                "actual_revenue": "{:,.0f}",
                "hybrid_predicted_revenue": "{:,.0f}",
                "predicted_revenue_sarima": "{:,.0f}",
                "formula_adjusted_revenue": "{:,.0f}",
                "hybrid_ape": "{:.2f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )

with monthly_tab:
    metric = st.radio(
        "顯示指標",
        ["WMAPE", "SMAPE", "MedianAPE", "P90APE", "DirectionAccuracy"],
        horizontal=True,
    )
    pivot = results["monthly"].pivot(index="target_month", columns="model", values=metric)
    pivot.index = [f"{int(month)} 月" for month in pivot.index]
    st.line_chart(pivot)
    st.dataframe(pivot.style.format("{:.3f}%"), width="stretch")

with weight_tab:
    st.subheader("2023–2024 權重搜尋")
    st.write(
        "每一組權重都只在 2023–2024 評估；平衡分數越低越好。"
        "選定後凍結權重，再套到 2025。"
    )
    weights = results["weights"].copy()
    weights["SARIMA 權重"] = weights["sarima_weight"].map(lambda value: f"{value:.1f}")
    st.line_chart(weights.set_index("SARIMA 權重")[["balanced_score", "WMAPE", "median_stock_WMAPE"]])
    best_mask = weights["sarima_weight"].round(6).eq(round(selected_weight, 6))
    styled = weights[[
        "sarima_weight", "formula_weight", "balanced_score", "WMAPE", "SMAPE", "MedianAPE", "P90APE", "median_stock_WMAPE", "DirectionAccuracy"
    ]].style.format(
        {
            "sarima_weight": "{:.1f}",
            "formula_weight": "{:.1f}",
            "balanced_score": "{:.3f}",
            "WMAPE": "{:.3f}%",
            "SMAPE": "{:.3f}%",
            "MedianAPE": "{:.3f}%",
            "P90APE": "{:.3f}%",
            "median_stock_WMAPE": "{:.3f}%",
            "DirectionAccuracy": "{:.3f}%",
        }
    ).apply(lambda row: ["background-color: #d9ead3" if best_mask.iloc[row.name] else ""] * len(row), axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)

with error_tab:
    st.subheader("2025 個股 WMAPE 超過 15%，由小到大")
    high_error = results["high_error"].copy()
    keyword = st.text_input("搜尋股票代號或名稱")
    if keyword.strip():
        mask = (
            high_error["stock_id"].astype(str).str.contains(keyword.strip(), na=False)
            | high_error["stock_name"].astype(str).str.contains(keyword.strip(), na=False)
        )
        high_error = high_error[mask]
    columns = [
        "stock_id",
        "stock_name",
        "industry_category",
        "hybrid_WMAPE",
        "sarima_WMAPE",
        "formula_WMAPE",
        "hybrid_MedianAPE",
        "hybrid_P90APE",
        "hybrid_DirectionAccuracy",
    ]
    st.dataframe(
        high_error[columns].style.format({column: "{:.3f}%" for column in columns if "WMAPE" in column or "APE" in column or "Accuracy" in column}),
        width="stretch",
        hide_index=True,
    )

st.caption("研究原型，模型結果不構成投資建議。")
