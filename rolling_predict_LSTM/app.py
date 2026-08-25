# -*- coding: utf-8 -*-
#開啟:.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
from __future__ import annotations

import pandas as pd
import streamlit as st

from rolling_lstm_engine import (
    DEFAULT_STREAMLIT_XLSTM_BACKBONE,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_XLSTM_GROWTH_ADJUSTMENT_ALPHA,
    DEFAULT_XLSTM_DECLINE_CAP_GROWTH_RATIO_MAX,
    DEFAULT_XLSTM_DECLINE_CAP_PREDICTION_RATIO_MIN,
    FORECAST_YEAR,
    GrowthAdjustmentConfig,
    RollingExperimentConfig,
    XLSTM_BACKBONES,
    build_growth_windows,
    get_xlstm_backbone_spec,
    get_xlstm_backbone_status,
    get_stock_list,
    load_revenue_data,
    run_rolling_lstm_experiment,
)


st.set_page_config(page_title="Rolling LSTM 營收與殖利率預測", layout="wide")
CACHE_VERSION = "rolling_xlstm_hybrid_v2"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return load_revenue_data()


@st.cache_data(show_spinner=False)
def load_window_preview() -> pd.DataFrame:
    return build_growth_windows(load_revenue_data(), window_size=DEFAULT_WINDOW_SIZE)


@st.cache_data(show_spinner=False)
def run_experiment(
    selected_stock: int,
    config: RollingExperimentConfig,
    cache_version: str,
):
    _ = cache_version
    return run_rolling_lstm_experiment(selected_stock=selected_stock, config=config)


def format_int(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{int(round(float(value))):,}"


def format_percent(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.2f}%"


def format_decimal(value) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def detect_torch_cuda() -> tuple[bool, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
        return False, "CUDA not available"
    except Exception as error:
        return False, str(error)


df = load_data()
stock_list = get_stock_list(df)
torch_cuda_available, torch_cuda_name = detect_torch_cuda()

st.title("Rolling LSTM + KMeans 營收與殖利率預測")
st.caption(
    "Rolling 系統獨立完成月營收、年營收、EPS、現金股利與殖利率估算；"
    "不 import Ensemble，跨系統公平比較仍由 forecast_benchmark 負責。"
)

with st.sidebar:
    st.header("設定")
    stock_keyword = st.text_input("股票代號搜尋", placeholder="例如 2330")
    filtered_stock_list = [stock_id for stock_id in stock_list if stock_keyword.strip() in str(stock_id)]
    if not filtered_stock_list:
        st.warning("找不到符合的股票代號，先顯示全部清單。")
        filtered_stock_list = stock_list

    selected_stock = st.selectbox("股票代號", filtered_stock_list)
    k = st.slider("KMeans 群數 K", min_value=4, max_value=8, value=6, step=1)
    epochs = st.slider("LSTM epochs", min_value=5, max_value=100, value=35, step=5)
    xlstm_backbone_options = (
        DEFAULT_STREAMLIT_XLSTM_BACKBONE,
        *(value for value in XLSTM_BACKBONES if value != DEFAULT_STREAMLIT_XLSTM_BACKBONE),
    )
    xlstm_backbone = st.selectbox(
        "xLSTM 架構",
        xlstm_backbone_options,
        index=0,
        format_func=lambda value: get_xlstm_backbone_spec(value).display_name,
        help="Hybrid 依序堆疊一個 mLSTM block 與一個 sLSTM block；舊 D1 選項保留既有 mLSTM-only 結果。",
    )
    xlstm_architecture_label = get_xlstm_backbone_spec(xlstm_backbone).display_name
    xlstm_status = get_xlstm_backbone_status(xlstm_backbone)
    enable_xlstm_plain = st.checkbox(
        "加入 Rolling xLSTM 比較",
        value=bool(xlstm_status["available"]),
        disabled=not bool(xlstm_status["available"]),
        help="加入 no-cluster xLSTM 與同一套 time-safe conditional adjustment；原本 LSTM / Cluster / Adjustment 仍固定保留。",
    )
    if xlstm_status["available"]:
        st.success(f"{xlstm_architecture_label} 可用")
        st.caption(str(xlstm_status["detail"]))
    else:
        st.warning(f"xLSTM 目前不可用：{xlstm_status['detail']}")
    max_train_samples = st.slider(
        "最大訓練樣本數",
        min_value=5_000,
        max_value=80_000,
        value=40_000,
        step=5_000,
        help="資料量很大時先固定抽樣，讓實驗可以較快重跑。",
    )
    if torch_cuda_available:
        st.success(f"PyTorch 裝置：CUDA（{torch_cuda_name}）")
    else:
        st.info(f"PyTorch 裝置：CPU（{torch_cuda_name}）")
    st.divider()
    st.caption("飆漲股低估改善參數")
    enable_growth_adjustment = st.checkbox("啟用 Growth Adjustment", value=True)
    growth_adjustment_alpha = st.slider("Growth Adjustment alpha", 0.0, 2.0, 0.8, 0.1)
    xlstm_growth_adjustment_alpha = st.slider(
        "xLSTM Adjustment alpha",
        0.0,
        2.0,
        float(DEFAULT_XLSTM_GROWTH_ADJUSTMENT_ALPHA),
        0.1,
        disabled=not enable_xlstm_plain,
        help=(
            "只影響 Rolling xLSTM + Conditional Adjustment；0.0 代表不做 growth boost，"
            "但 regime strategy 的 balanced decline cap 仍可生效。"
        ),
    )
    enable_conditional_adjustment = st.checkbox("啟用條件式成長修正", value=True)
    enable_regime_strategy = st.checkbox("啟用 regime 自動策略", value=True)
    use_asymmetric_loss = st.checkbox("啟用非對稱 loss", value=True)
    under_weight = st.slider("低估懲罰 under_weight", 1.0, 5.0, 2.0, 0.25)
    st.caption("Growth Adjustment 固定需要最後一個輸入月份為正成長；基本營收 guardrail 固定保留。")
    st.caption("Regime strategy 會在 decline regime 觸發 decline cap；xLSTM 預設使用較嚴格的 balanced gate。")
    st.caption(f"視窗長度固定為 {DEFAULT_WINDOW_SIZE} 個月，預測目標為下一期月營收。")
    run_button = st.button("開始 Rolling LSTM 實驗", type="primary", use_container_width=True)

stock_df = df[df["stock_id"] == selected_stock].sort_values("date")

overview_cols = st.columns(4)
overview_cols[0].metric("資料起始年", f"{int(stock_df['revenue_year'].min())}")
overview_cols[1].metric("資料結束年", f"{int(stock_df['revenue_year'].max())}")
overview_cols[2].metric("月份筆數", f"{len(stock_df):,}")
overview_cols[3].metric(
    "2024 營收合計",
    f"{format_int(stock_df[stock_df['revenue_year'] == 2024]['revenue_thousand'].sum())} 千元",
)

st.subheader("歷史月營收")
history_chart = stock_df[stock_df["revenue_year"] >= 2020][["date", "revenue_thousand"]].rename(
    columns={"date": "日期", "revenue_thousand": "營收"}
)
st.line_chart(history_chart.set_index("日期"))

cached_result_available = (
    "rolling_result" in st.session_state
    and st.session_state.get("rolling_stock") == int(selected_stock)
)

if run_button or cached_result_available:
    try:
        if run_button:
            with st.spinner("正在訓練 Rolling LSTM，並建立 EPS、現金股利與殖利率估算..."):
                experiment_config = RollingExperimentConfig(
                    k=int(k),
                    window_size=DEFAULT_WINDOW_SIZE,
                    epochs=int(epochs),
                    max_train_samples=int(max_train_samples),
                    sequence_backbone="lstm",
                    include_xlstm_plain=enable_xlstm_plain,
                    xlstm_backbone=xlstm_backbone,
                    include_yield_forecast=True,
                    yield_as_of_date=f"{FORECAST_YEAR}-01-10",
                    use_asymmetric_loss=use_asymmetric_loss,
                    under_weight=float(under_weight),
                    growth=GrowthAdjustmentConfig(
                        enabled=enable_growth_adjustment,
                        alpha=float(growth_adjustment_alpha),
                        conditional=enable_conditional_adjustment,
                        regime_strategy=enable_regime_strategy,
                    ),
                    xlstm_growth=GrowthAdjustmentConfig(
                        enabled=enable_growth_adjustment,
                        alpha=float(xlstm_growth_adjustment_alpha),
                        conditional=enable_conditional_adjustment,
                        regime_strategy=enable_regime_strategy,
                        decline_cap_growth_ratio_max=DEFAULT_XLSTM_DECLINE_CAP_GROWTH_RATIO_MAX,
                        decline_cap_prediction_ratio_min=DEFAULT_XLSTM_DECLINE_CAP_PREDICTION_RATIO_MIN,
                    ),
                )
                result = run_experiment(
                    selected_stock=int(selected_stock),
                    config=experiment_config,
                    cache_version=CACHE_VERSION,
                )
                st.session_state["rolling_result"] = result
                st.session_state["rolling_stock"] = int(selected_stock)
        else:
            result = st.session_state["rolling_result"]
    except ImportError as error:
        st.error("目前環境缺少必要套件，無法訓練 Rolling LSTM。")
        st.code("pip install -r rolling_predict_LSTM/requirements.txt", language="bash")
        st.code("pip install -r rolling_predict_LSTM/requirements-xlstm.txt", language="bash")
        st.exception(error)
    except Exception as error:
        st.error("實驗執行失敗，請檢查資料欄位或套件環境。")
        st.exception(error)
    else:
        result_xlstm_spec = get_xlstm_backbone_spec(
            result.selected_params.get("xlstm_backbone", "xlstm")
        )
        result_xlstm_backbone = result_xlstm_spec.key
        result_xlstm_label = result_xlstm_spec.display_name
        metric_lookup = result.metrics.set_index("model").to_dict("index")
        cluster_metric = metric_lookup.get("Rolling LSTM + Cluster", {})
        plain_metric = metric_lookup.get("Rolling LSTM", {})
        adjusted_metric = metric_lookup.get("Rolling LSTM + Cluster + Conditional Adjustment", {})
        xlstm_metric = metric_lookup.get("Rolling xLSTM", {})
        xlstm_adjusted_metric = metric_lookup.get("Rolling xLSTM + Conditional Adjustment", {})
        mae_improvement = cluster_metric.get("MAE", 0) - adjusted_metric.get("MAE", 0)
        improve_pct = mae_improvement / cluster_metric.get("MAE", 1) * 100 if cluster_metric.get("MAE", 0) else 0
        xlstm_adjustment_improvement = xlstm_metric.get("MAE", 0) - xlstm_adjusted_metric.get("MAE", 0)

        st.subheader("2025 預測效果比較")
        metric_cols = st.columns(6)
        metric_cols[0].metric("Adjustment MAE", f"{format_int(adjusted_metric.get('MAE'))} 千元")
        metric_cols[1].metric("Cluster MAE", f"{format_int(cluster_metric.get('MAE'))} 千元")
        metric_cols[2].metric("Adj. vs Cluster", f"{format_int(mae_improvement)} 千元", f"{improve_pct:.2f}%")
        metric_cols[3].metric(f"{result_xlstm_label} MAE", f"{format_int(xlstm_metric.get('MAE'))} 千元")
        metric_cols[4].metric(f"{result_xlstm_label} Adj. MAE", f"{format_int(xlstm_adjusted_metric.get('MAE'))} 千元")
        metric_cols[5].metric("xAdj. vs xLSTM", f"{format_int(xlstm_adjustment_improvement)} 千元")

        prediction_chart = result.forecast[
            [
                "target_date",
                "actual_revenue",
                "predicted_revenue_no_cluster",
                "predicted_revenue_base",
                "predicted_revenue_adjusted",
                "predicted_revenue_xlstm",
                "predicted_revenue_xlstm_adjusted",
            ]
        ].rename(
            columns={
                "target_date": "日期",
                "actual_revenue": "2025 實際營收",
                "predicted_revenue_no_cluster": "Rolling LSTM",
                "predicted_revenue_base": "Rolling LSTM + Cluster",
                "predicted_revenue_adjusted": "Cluster + Conditional Adjustment",
                "predicted_revenue_xlstm": f"Rolling {result_xlstm_label}",
                "predicted_revenue_xlstm_adjusted": f"Rolling {result_xlstm_label} + Conditional Adjustment",
            }
        )
        st.line_chart(prediction_chart.set_index("日期"))

        st.subheader("2025 EPS、現金股利與殖利率估算")
        st.caption(
            "EPS 與 payout 僅使用 2025-01-10 前可得資料。可部署殖利率使用 cutoff 前最後真實收盤價；"
            "月線使用 2025 各月實際月末收盤價，只供回測評估。年營收是逐月 Rolling 預測的合計，"
            "不是單一時點全年預測。"
        )
        if result.yield_summary.empty:
            st.warning("目前 EPS、股利或股價資料不足，無法建立殖利率估算。")
        else:
            yield_summary = result.yield_summary.copy()
            ok_rows = yield_summary[yield_summary["status"].eq("ok")]
            preferred = ok_rows[
                ok_rows["model"].eq("Rolling LSTM + Cluster + Conditional Adjustment")
            ]
            selected_yield_row = (
                preferred.iloc[0]
                if not preferred.empty
                else (ok_rows.iloc[0] if not ok_rows.empty else yield_summary.iloc[0])
            )
            yield_cols = st.columns(6)
            yield_cols[0].metric(
                "預測年營收",
                f"{format_int(selected_yield_row.get('predicted_annual_revenue'))} 千元",
            )
            yield_cols[1].metric(
                "預估 EPS",
                format_decimal(selected_yield_row.get("estimated_eps")),
            )
            yield_cols[2].metric(
                "預估現金股利",
                format_decimal(selected_yield_row.get("estimated_cash_dividend")),
            )
            yield_cols[3].metric(
                "Cutoff 參考股價",
                format_decimal(selected_yield_row.get("as_of_stock_price")),
            )
            yield_cols[4].metric(
                "可部署估算殖利率",
                format_percent(selected_yield_row.get("as_of_price_yield_percent")),
            )
            yield_cols[5].metric(
                "最新回測估算殖利率",
                format_percent(selected_yield_row.get("latest_predicted_yield_percent")),
            )

            display_yield_summary = yield_summary[
                [
                    "model",
                    "status",
                    "predicted_annual_revenue",
                    "eps_reference_year",
                    "estimated_eps",
                    "payout_ratio",
                    "estimated_cash_dividend",
                    "actual_cash_dividend",
                    "as_of_price_date",
                    "as_of_stock_price",
                    "as_of_price_yield_percent",
                    "latest_stock_price",
                    "average_predicted_yield_percent",
                    "latest_predicted_yield_percent",
                    "latest_actual_yield_percent",
                ]
            ].copy()
            display_yield_summary["payout_ratio"] = display_yield_summary["payout_ratio"] * 100
            display_yield_summary = display_yield_summary.rename(
                columns={
                    "model": "營收模型",
                    "status": "狀態",
                    "predicted_annual_revenue": "預測年營收（千元）",
                    "eps_reference_year": "EPS 參考年",
                    "estimated_eps": "預估 EPS",
                    "payout_ratio": "Payout（%）",
                    "estimated_cash_dividend": "預估現金股利",
                    "actual_cash_dividend": "實際現金股利（評估）",
                    "as_of_price_date": "Cutoff 股價日期",
                    "as_of_stock_price": "Cutoff 參考股價",
                    "as_of_price_yield_percent": "可部署估算殖利率（%）",
                    "latest_stock_price": "最新回測股價",
                    "average_predicted_yield_percent": "平均估算殖利率（%）",
                    "latest_predicted_yield_percent": "最新估算殖利率（%）",
                    "latest_actual_yield_percent": "最新實際殖利率（%）",
                }
            )
            st.dataframe(display_yield_summary.round(3), use_container_width=True, hide_index=True)

            if not result.yield_forecast.empty:
                predicted_yield_chart = result.yield_forecast.pivot(
                    index="target_date",
                    columns="model",
                    values="predicted_dividend_yield_percent",
                )
                actual_yield = (
                    result.yield_forecast.sort_values("target_month")
                    .drop_duplicates("target_date")
                    .set_index("target_date")["actual_dividend_yield_percent"]
                )
                predicted_yield_chart["實際殖利率（評估）"] = actual_yield
                st.line_chart(predicted_yield_chart)
                with st.expander("殖利率月資料與來源"):
                    st.dataframe(result.yield_forecast, use_container_width=True, hide_index=True)
        if result.yield_notes:
            with st.expander("殖利率方法與限制"):
                for note in result.yield_notes:
                    st.write(f"- {note}")

        return_chart = result.forecast[
            [
                "target_date",
                "actual_return",
                "predicted_return_no_cluster",
                "predicted_return_cluster",
                "predicted_return_adjusted",
                "predicted_return_xlstm",
                "predicted_return_xlstm_adjusted",
            ]
        ].copy()
        for column in [
            "actual_return",
            "predicted_return_no_cluster",
            "predicted_return_cluster",
            "predicted_return_adjusted",
            "predicted_return_xlstm",
            "predicted_return_xlstm_adjusted",
        ]:
            return_chart[column] = return_chart[column] * 100
        return_chart = return_chart.rename(
            columns={
                "target_date": "日期",
                "actual_return": "實際月營收變化率",
                "predicted_return_no_cluster": "無分群預測變化率",
                "predicted_return_cluster": "有分群預測變化率",
                "predicted_return_adjusted": "成長修正預測變化率",
                "predicted_return_xlstm": "xLSTM 預測變化率",
                "predicted_return_xlstm_adjusted": "xLSTM 成長修正變化率",
            }
        )
        st.caption("下圖的 return 是月營收相對於上一個已知月份的變化率，不是股價報酬率。")
        st.line_chart(return_chart.set_index("日期"))

        st.subheader("評估指標")
        st.dataframe(result.metrics, use_container_width=True, hide_index=True)

        if result.selected_params:
            st.caption(f"本次可選 xLSTM 架構：{result_xlstm_label}（{result_xlstm_backbone}）")
            selected_cols = st.columns(4)
            selected_cols[0].metric("最終 K", result.selected_params.get("k", "-"))
            selected_cols[1].metric("最終樣本數", format_int(result.selected_params.get("max_train_samples")))
            selected_cols[2].metric("Cluster alpha", f"{float(result.selected_params.get('growth_alpha', 0)):.2f}")
            selected_cols[3].metric("xLSTM alpha", f"{float(result.selected_params.get('xlstm_growth_alpha', 0)):.2f}")

        st.subheader("分群對預測效果的影響")
        effect_cols = st.columns(2)
        with effect_cols[0]:
            st.caption("選定股票在 2025 各 cluster 的誤差比較")
            st.dataframe(result.cluster_effect, use_container_width=True, hide_index=True)
        with effect_cols[1]:
            st.caption("選定股票的 rolling cluster timeline")
            st.dataframe(result.selected_cluster_timeline, use_container_width=True, hide_index=True)

        st.subheader("Cluster 行為形狀")
        profile_cols = st.columns([1.2, 1])
        with profile_cols[0]:
            st.dataframe(result.cluster_profile, use_container_width=True, hide_index=True)
        with profile_cols[1]:
            vector_columns = [column for column in result.cluster_profile.columns if column.startswith("g_")]
            if vector_columns:
                pattern_chart = result.cluster_profile.set_index("cluster")[vector_columns].T
                pattern_chart.index = list(range(1, len(vector_columns) + 1))
                st.caption("每個 cluster 在 12 個月視窗中的平均上升機率")
                st.line_chart(pattern_chart)

        if not result.elbow_scores.empty:
            st.subheader("KMeans elbow 參考")
            elbow_chart = result.elbow_scores.rename(columns={"k": "K", "inertia": "Inertia"})
            st.line_chart(elbow_chart.set_index("K"))
            st.dataframe(result.elbow_scores, use_container_width=True, hide_index=True)

        st.subheader("2025 Rolling 預測明細")
        display_forecast = result.forecast.copy()
        round_columns = [
            "actual_return",
            "predicted_return_cluster",
            "predicted_return_no_cluster",
            "predicted_return_adjusted",
            "predicted_return_xlstm",
            "predicted_return_xlstm_adjusted",
            "growth_rate_at_end",
            "momentum_3m_at_end",
            "momentum_6m_at_end",
            "growth_ratio",
            "growth_signal",
            "adjustment_ratio",
            "xlstm_adjustment_ratio",
            "trend_slope_rate",
        ]
        for column in round_columns:
            if column in display_forecast.columns:
                display_forecast[column] = display_forecast[column].round(4)
        st.dataframe(display_forecast, use_container_width=True, hide_index=True)

        if result.notes:
            with st.expander("模型說明"):
                for note in result.notes:
                    st.write(f"- {note}")

        csv = result.forecast.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下載 Rolling LSTM 預測 CSV",
            data=csv,
            file_name=f"{selected_stock}_{FORECAST_YEAR}_rolling_lstm_forecast.csv",
            mime="text/csv",
        )
        if not result.yield_forecast.empty:
            yield_csv = result.yield_forecast.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下載 Rolling LSTM 殖利率 CSV",
                data=yield_csv,
                file_name=f"{selected_stock}_{FORECAST_YEAR}_rolling_lstm_yield_forecast.csv",
                mime="text/csv",
            )
else:
    st.info("選擇股票後按下左側按鈕，系統會完成營收、EPS、現金股利與殖利率估算。")
