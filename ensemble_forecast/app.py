from __future__ import annotations

import pandas as pd
import streamlit as st

from forecast_engine import (
    FORECAST_YEAR,
    build_forecast,
    get_stock_list,
    load_revenue_data,
    make_revenue_summary,
    make_yield_summary,
)


st.set_page_config(
    page_title="多模型集成營收與殖利率預測系統",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return load_revenue_data()


@st.cache_data(show_spinner=False)
def run_forecast(selected_stock: int):
    return build_forecast(selected_stock)


def format_int(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{int(round(float(value))):,}"


revenue_data = load_data()
stock_list = get_stock_list(revenue_data)

st.title("多模型集成營收與殖利率預測系統")
st.caption("比較 XGBoost、LightGBM、CatBoost 與 Seasonal Quantile，並依歷史驗證誤差建立加權預測。")

with st.sidebar:
    st.header("預測設定")
    stock_keyword = st.text_input("搜尋股票代號", placeholder="例如 1101")
    filtered_stock_list = [stock_id for stock_id in stock_list if stock_keyword.strip() in str(stock_id)]
    if not filtered_stock_list:
        st.warning("找不到符合的股票，已恢復完整清單。")
        filtered_stock_list = stock_list
    selected_stock = st.selectbox("選擇股票", filtered_stock_list)
    run_button = st.button("執行集成預測", type="primary", use_container_width=True)

stock_data = revenue_data[revenue_data["stock_id"] == selected_stock].sort_values("date")
overview_cols = st.columns(4)
overview_cols[0].metric("資料起始年", int(stock_data["revenue_year"].min()))
overview_cols[1].metric("資料結束年", int(stock_data["revenue_year"].max()))
overview_cols[2].metric("月資料筆數", f"{len(stock_data):,}")
overview_cols[3].metric(
    "2024 年營收",
    f"{format_int(stock_data[stock_data['revenue_year'] == 2024]['revenue_thousand'].sum())} 千元",
)

if run_button:
    with st.spinner("正在訓練多模型、建立驗證權重並估算殖利率..."):
        try:
            st.session_state["ensemble_result"] = run_forecast(int(selected_stock))
            st.session_state["ensemble_stock"] = int(selected_stock)
        except Exception as error:
            st.error("預測執行失敗，請檢查資料欄位與模型套件。")
            st.exception(error)

if "ensemble_result" not in st.session_state:
    st.info("選擇股票後按下「執行集成預測」。")
else:
    result = st.session_state["ensemble_result"]
    result_stock = st.session_state["ensemble_stock"]
    revenue_summary = make_revenue_summary(result.forecast, result.metrics, result.weights)
    yield_summary = make_yield_summary(result.forecast)

    summary_cols = st.columns(5)
    summary_cols[0].metric(f"{FORECAST_YEAR} 預測年營收", f"{revenue_summary['annual_total']} 千元")
    summary_cols[1].metric("驗證最佳模型", revenue_summary["best_model"])
    summary_cols[2].metric("驗證最佳 MAPE", revenue_summary["best_mape"])
    summary_cols[3].metric("Cutoff 可部署殖利率", yield_summary["as_of_yield"])
    summary_cols[4].metric("最新回測估算殖利率", yield_summary["latest_yield"])

    revenue_tab, yield_tab, diagnostics_tab = st.tabs(["營收預測", "殖利率比較", "模型診斷"])

    with revenue_tab:
        st.subheader(f"{result_stock} 的 {FORECAST_YEAR} 月營收預測")
        model_columns = [
            column
            for column in ["XGBoost", "LightGBM", "CatBoost", "SeasonalQuantile", "ensemble_revenue"]
            if column in result.forecast.columns
        ]
        revenue_chart = result.forecast.set_index("date")[model_columns].copy()
        if "actual_revenue" in result.backtest.columns:
            actual_by_month = result.backtest.set_index("revenue_month")["actual_revenue"]
            revenue_chart["實際營收"] = result.forecast["revenue_month"].map(actual_by_month).to_numpy()
        st.line_chart(revenue_chart)
        display_columns = [
            "date",
            "revenue_month",
            *model_columns,
            "lower_bound",
            "upper_bound",
        ]
        st.dataframe(
            result.forecast[[column for column in display_columns if column in result.forecast.columns]],
            use_container_width=True,
            hide_index=True,
        )

    with yield_tab:
        st.subheader("預測股利對應的可部署殖利率與回測")
        st.caption(
            "預估現金股利由預測營收、歷史 EPS/revenue 與歷史 payout 推導；"
            "可部署殖利率使用 2025-01-10 cutoff 前最後真實收盤價；"
            "月線使用 2025 各月實際月末收盤價，只供回測評估。"
        )
        deployable_cols = st.columns(3)
        deployable_cols[0].metric("Cutoff 參考股價", yield_summary["as_of_price"])
        deployable_cols[1].metric("Cutoff 股價日期", yield_summary["as_of_price_date"])
        deployable_cols[2].metric("可部署估算殖利率", yield_summary["as_of_yield"])
        if result.yield_comparison.empty:
            st.warning("目前資料不足，無法建立殖利率比較。")
        else:
            yield_chart = result.yield_comparison.set_index("date")[
                ["predicted_dividend_yield_percent", "actual_dividend_yield_percent"]
            ].rename(
                columns={
                    "predicted_dividend_yield_percent": "預估殖利率（預測股利／實際股價）",
                    "actual_dividend_yield_percent": "實際殖利率（評估）",
                }
            )
            st.line_chart(yield_chart)
            st.dataframe(result.yield_comparison, use_container_width=True, hide_index=True)
            yield_download = result.yield_comparison.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下載殖利率回測 CSV",
                data=yield_download,
                file_name=f"{result_stock}_{FORECAST_YEAR}_ensemble_yield_evaluation.csv",
                mime="text/csv",
            )

    with diagnostics_tab:
        st.subheader("回測指標")
        st.dataframe(result.metrics, use_container_width=True, hide_index=True)
        st.subheader("模型權重")
        st.dataframe(result.weights, use_container_width=True, hide_index=True)
        st.subheader("模型建議")
        st.write(result.recommendation.get("recommendation", ""))
        st.caption(result.recommendation.get("reason", ""))
        if result.notes:
            with st.expander("執行資訊"):
                for note in result.notes:
                    st.write(f"- {note}")

    download = result.forecast.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下載集成預測 CSV",
        data=download,
        file_name=f"{result_stock}_{FORECAST_YEAR}_ensemble_forecast.csv",
        mime="text/csv",
    )
