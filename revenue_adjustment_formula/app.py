from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


SYSTEM_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SYSTEM_DIR / "outputs" / "full_universe_20260818"


st.set_page_config(
    page_title="營收調整公式｜2025 回測結果",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_results() -> dict[str, pd.DataFrame]:
    required = {
        "overall": "overall_accuracy.csv",
        "monthly": "monthly_accuracy.csv",
        "predictions": "monthly_predictions.csv",
        "stocks": "stock_accuracy.csv",
        "high_error": "stocks_over_15pct_wmape.csv",
        "sweep": "parameter_sweep.csv",
        "comparison": "comparison_with_sarima_xlstm.csv",
    }
    missing = [name for name in required.values() if not (OUTPUT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError("缺少結果檔：" + "、".join(missing))
    return {
        key: pd.read_csv(OUTPUT_DIR / filename)
        for key, filename in required.items()
    }


def metric_value(frame: pd.DataFrame, model: str, column: str) -> float:
    return float(frame.loc[frame["model"].eq(model), column].iloc[0])


def stock_label(row: pd.Series) -> str:
    name = "" if pd.isna(row.get("stock_name")) else str(row.get("stock_name"))
    return f"{int(row['stock_id'])} {name}".strip()


try:
    results = load_results()
except FileNotFoundError as error:
    st.error(str(error))
    st.info("請先執行 run_experiment.py 產生完整回測結果。")
    st.stop()

overall = results["overall"]
formula_model = "營收公式（含殘差校正）"
last_model = "沿用上月營收"
seasonal_model = "去年同月營收"
formula_wmape = metric_value(overall, formula_model, "WMAPE")
last_wmape = metric_value(overall, last_model, "WMAPE")
seasonal_wmape = metric_value(overall, seasonal_model, "WMAPE")
formula_median = metric_value(overall, formula_model, "MedianAPE")

st.title("營收調整公式｜2025 全市場回測")
st.caption(
    "參數只使用 2023–2024 滾動驗證選擇；2025 實際營收僅用於事後評分。"
    "現有集成、xLSTM 與 SARIMA 引擎沒有被修改。"
)

summary_columns = st.columns(5)
summary_columns[0].metric("公式 WMAPE", f"{formula_wmape:.2f}%")
summary_columns[1].metric("MedianAPE", f"{formula_median:.2f}%")
summary_columns[2].metric(
    "相較沿用上月",
    f"改善 {(last_wmape - formula_wmape) / last_wmape * 100:.2f}%",
)
summary_columns[3].metric(
    "相較去年同月",
    f"改善 {(seasonal_wmape - formula_wmape) / seasonal_wmape * 100:.2f}%",
)
summary_columns[4].metric(
    "誤差 > 15%",
    f"{len(results['high_error']):,} 檔",
    help="只計算具有完整 12 個月結果的股票。",
)

overview_tab, monthly_tab, stock_tab, error_tab, parameter_tab = st.tabs(
    ["整體比較", "逐月表現", "個股結果", "大誤差股票", "參數搜尋"]
)

with overview_tab:
    st.subheader("公式與簡單基準")
    display_columns = [
        "model",
        "observations",
        "stock_count",
        "WMAPE",
        "SMAPE",
        "MedianAPE",
        "MASE",
        "DirectionAccuracy",
    ]
    st.dataframe(
        overall[display_columns].style.format(
            {
                "WMAPE": "{:.3f}%",
                "SMAPE": "{:.3f}%",
                "MedianAPE": "{:.3f}%",
                "MASE": "{:.3f}",
                "DirectionAccuracy": "{:.3f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("與既有 SARIMA、xLSTM 的共同樣本比較")
    comparison = results["comparison"]
    st.bar_chart(comparison.set_index("model")[["WMAPE", "SMAPE", "MedianAPE"]])
    st.dataframe(
        comparison[display_columns].style.format(
            {
                "WMAPE": "{:.3f}%",
                "SMAPE": "{:.3f}%",
                "MedianAPE": "{:.3f}%",
                "MASE": "{:.3f}",
                "DirectionAccuracy": "{:.3f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.info(
        "結論：公式的 WMAPE 略高於 SARIMA，但 SMAPE、MedianAPE 較低，"
        "而且比 xLSTM 更不容易出現少數極端預測拉高總誤差。"
    )

with monthly_tab:
    monthly = results["monthly"]
    metric_choice = st.radio(
        "顯示指標",
        ["WMAPE", "SMAPE", "MedianAPE", "DirectionAccuracy"],
        horizontal=True,
    )
    monthly_pivot = monthly.pivot(
        index="target_month", columns="model", values=metric_choice
    )
    monthly_pivot.index = [f"{month} 月" for month in monthly_pivot.index]
    st.line_chart(monthly_pivot)
    st.dataframe(
        monthly_pivot.style.format("{:.3f}%"),
        use_container_width=True,
    )

with stock_tab:
    stocks = results["stocks"].copy()
    stocks["label"] = stocks.apply(stock_label, axis=1)
    selected_label = st.selectbox(
        "選擇股票",
        stocks.sort_values("stock_id")["label"].tolist(),
        index=0,
    )
    selected_id = int(selected_label.split()[0])
    selected_metrics = stocks[stocks["stock_id"].eq(selected_id)].iloc[0]
    stock_metrics = st.columns(5)
    stock_metrics[0].metric("股票代碼", selected_id)
    stock_metrics[1].metric("月份數", int(selected_metrics["observations"]))
    stock_metrics[2].metric("WMAPE", f"{selected_metrics['WMAPE']:.2f}%")
    stock_metrics[3].metric("SMAPE", f"{selected_metrics['SMAPE']:.2f}%")
    stock_metrics[4].metric(
        "方向準確率", f"{selected_metrics['DirectionAccuracy']:.2f}%"
    )

    stock_predictions = results["predictions"][
        results["predictions"]["stock_id"].eq(selected_id)
    ].copy()
    stock_predictions["target_date"] = pd.to_datetime(
        stock_predictions["target_date"]
    )
    chart = stock_predictions.set_index("target_date")[[
        "actual_revenue",
        "formula_adjusted_revenue",
        "last_observed_revenue",
        "seasonal_naive_revenue",
    ]].rename(
        columns={
            "actual_revenue": "實際營收",
            "formula_adjusted_revenue": "公式預測",
            "last_observed_revenue": "沿用上月",
            "seasonal_naive_revenue": "去年同月",
        }
    )
    st.line_chart(chart)
    st.dataframe(
        stock_predictions[[
            "target_year",
            "target_month",
            "actual_revenue",
            "formula_adjusted_revenue",
            "last_observed_revenue",
            "seasonal_naive_revenue",
            "forecast_method",
        ]],
        use_container_width=True,
        hide_index=True,
    )

with error_tab:
    st.subheader("完整 12 個月且 WMAPE 超過 15%")
    keyword = st.text_input("搜尋股票代碼或名稱")
    high_error = results["high_error"].copy()
    if keyword.strip():
        mask = (
            high_error["stock_id"].astype(str).str.contains(keyword.strip(), na=False)
            | high_error["stock_name"].astype(str).str.contains(keyword.strip(), na=False)
        )
        high_error = high_error[mask]
    st.dataframe(
        high_error.style.format(
            {
                "WMAPE": "{:.3f}%",
                "SMAPE": "{:.3f}%",
                "MedianAPE": "{:.3f}%",
                "MAPE": "{:.3f}%",
                "DirectionAccuracy": "{:.3f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=520,
    )
    st.download_button(
        "下載大誤差股票 CSV",
        results["high_error"].to_csv(index=False).encode("utf-8-sig"),
        file_name="stocks_over_15pct_wmape.csv",
        mime="text/csv",
    )

with parameter_tab:
    st.subheader("2023–2024 的 27 組參數搜尋")
    best = results["sweep"].iloc[0]
    best_columns = st.columns(4)
    best_columns[0].metric("季節權重", f"{best['seasonal_weight']:.2f}")
    best_columns[1].metric("殘差 α", f"{best['residual_alpha']:.2f}")
    best_columns[2].metric("殘差校正強度", f"{best['residual_strength']:.2f}")
    best_columns[3].metric("驗證平衡分數", f"{best['balanced_score']:.3f}")
    st.warning(
        "殘差校正強度被選成 0，代表這次驗證中追加殘差修正沒有穩定增益。"
    )
    st.dataframe(
        results["sweep"].style.format(
            {
                "WMAPE": "{:.3f}%",
                "median_stock_WMAPE": "{:.3f}%",
                "balanced_score": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=520,
    )
