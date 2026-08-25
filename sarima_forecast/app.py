from __future__ import annotations

import pandas as pd
import streamlit as st

from sarima_engine import (
    FORECAST_YEAR,
    SarimaConfig,
    build_rolling_sarima_forecast,
    get_stock_list,
    load_revenue_data,
)


st.set_page_config(page_title="Rolling SARIMA 月營收預測", layout="wide")
CACHE_VERSION = "sarima_v1"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return load_revenue_data()


@st.cache_data(show_spinner=False)
def run_experiment(selected_stock: int, confidence_level: float, cache_version: str):
    _ = cache_version
    return build_rolling_sarima_forecast(
        load_revenue_data(),
        selected_stock=selected_stock,
        config=SarimaConfig(confidence_level=confidence_level),
    )


def format_number(value: object, decimals: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):,.{decimals}f}"


data = load_data()
stock_list = get_stock_list(data)

st.title("Rolling SARIMA 月營收預測")
st.caption(
    "獨立的傳統時間序列預測系統；使用 log(月營收) 與 12 個月季節週期，"
    "2025 每個月只使用該月以前已知的營收。"
)

with st.sidebar:
    st.header("設定")
    keyword = st.text_input("股票代號搜尋", placeholder="例如 2330")
    filtered = [stock_id for stock_id in stock_list if keyword.strip() in str(stock_id)]
    if not filtered:
        filtered = stock_list
    selected_stock = st.selectbox("股票代號", filtered)
    confidence_level = st.slider("預測區間信心水準", 0.80, 0.99, 0.95, 0.01)
    run_button = st.button("開始 SARIMA 滾動預測", type="primary", use_container_width=True)

stock_data = data[data["stock_id"].eq(int(selected_stock))].sort_values("date")
overview = st.columns(4)
overview[0].metric("資料起始月份", stock_data["date"].min().strftime("%Y-%m"))
overview[1].metric("資料結束月份", stock_data["date"].max().strftime("%Y-%m"))
overview[2].metric("月份筆數", f"{len(stock_data):,}")
overview[3].metric(
    "2024 年營收",
    f"{stock_data[stock_data['revenue_year'].eq(2024)]['revenue_thousand'].sum():,.0f} 千元",
)

st.subheader("歷史月營收")
history_chart = stock_data[["date", "revenue_thousand"]].rename(
    columns={"date": "日期", "revenue_thousand": "營收"}
)
st.line_chart(history_chart.set_index("日期"))

cache_matches = (
    "sarima_result" in st.session_state
    and st.session_state.get("sarima_stock") == int(selected_stock)
    and st.session_state.get("sarima_confidence") == float(confidence_level)
)

if run_button or cache_matches:
    try:
        if run_button:
            with st.spinner("正在選擇 SARIMA 參數並執行 2025 逐月滾動預測..."):
                result = run_experiment(
                    int(selected_stock),
                    float(confidence_level),
                    CACHE_VERSION,
                )
                st.session_state["sarima_result"] = result
                st.session_state["sarima_stock"] = int(selected_stock)
                st.session_state["sarima_confidence"] = float(confidence_level)
        else:
            result = st.session_state["sarima_result"]
    except ImportError as error:
        st.error("缺少 SARIMA 套件，請先安裝 requirements.txt。")
        st.code("pip install -r sarima_forecast/requirements.txt", language="bash")
        st.exception(error)
    except Exception as error:
        st.error("SARIMA 執行失敗，請檢查資料或模型參數。")
        st.exception(error)
    else:
        metric = result.metrics.iloc[0]
        st.subheader(f"{FORECAST_YEAR} 預測結果")
        metric_columns = st.columns(6)
        metric_columns[0].metric("WMAPE", f"{format_number(metric.get('WMAPE'))}%")
        metric_columns[1].metric("SMAPE", f"{format_number(metric.get('SMAPE'))}%")
        metric_columns[2].metric("Median MAPE", f"{format_number(metric.get('MedianAPE'))}%")
        metric_columns[3].metric("MAE", f"{format_number(metric.get('MAE'), 0)} 千元")
        metric_columns[4].metric("方向準確率", f"{format_number(metric.get('DirectionAccuracy'))}%")
        metric_columns[5].metric("低估比例", f"{format_number(metric.get('UnderestimateRate'))}%")

        order_columns = st.columns(3)
        order_columns[0].metric("非季節參數", str(result.selected_order or "fallback"))
        order_columns[1].metric("季節參數", str(result.selected_seasonal_order or "fallback"))
        fallback_count = int(result.forecast["forecast_method"].ne("sarima").sum())
        order_columns[2].metric("Fallback 月數", str(fallback_count))

        chart = result.forecast[
            [
                "target_date",
                "actual_revenue",
                "predicted_revenue_sarima",
                "sarima_lower",
                "sarima_upper",
            ]
        ].rename(
            columns={
                "target_date": "日期",
                "actual_revenue": "實際營收",
                "predicted_revenue_sarima": "SARIMA 預測",
                "sarima_lower": "預測區間下限",
                "sarima_upper": "預測區間上限",
            }
        )
        st.line_chart(chart.set_index("日期"))

        display = result.forecast.rename(
            columns={
                "target_date": "目標月份",
                "actual_revenue": "實際營收（千元）",
                "predicted_revenue_sarima": "SARIMA 預測（千元）",
                "sarima_lower": "區間下限",
                "sarima_upper": "區間上限",
                "absolute_percentage_error": "APE（%）",
                "forecast_method": "預測方式",
                "fallback_reason": "Fallback 原因",
                "history_months": "連續歷史月數",
            }
        )
        st.dataframe(display.round(3), use_container_width=True, hide_index=True)

        with st.expander("SARIMA 參數搜尋結果"):
            if result.order_search.empty:
                st.warning("連續歷史資料不足或沒有可用的 SARIMA 參數。")
            else:
                st.dataframe(result.order_search.round(3), use_container_width=True, hide_index=True)

        with st.expander("方法與資料時序說明"):
            for note in result.notes:
                st.write(f"- {note}")

        st.download_button(
            "下載 SARIMA 月預測 CSV",
            result.forecast.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"sarima_{selected_stock}_{FORECAST_YEAR}.csv",
            mime="text/csv",
        )

