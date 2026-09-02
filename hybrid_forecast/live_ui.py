from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from financial_forecast.evidence import resolve_data_files
from financial_forecast.dividend_patterns import load_dividend_patterns, PATTERN_LABELS
from hybrid_forecast.dividend_report import format_dividend_catalog
from hybrid_forecast.live_engine import (
    LIVE_CACHE_VERSION, build_live_forecast, data_fingerprint, load_revenue_snapshot,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@st.cache_data(show_spinner=False)
def load_inputs(data_dir: str, as_of: str, fingerprint: tuple):
    revenue = load_revenue_snapshot(data_dir, as_of)
    metadata_path = resolve_data_files(data_dir)["stock_list"]
    names = {}
    if metadata_path.is_file():
        metadata = pd.read_csv(metadata_path)
        if {"stock_id", "stock_name"}.issubset(metadata.columns):
            metadata["stock_id"] = pd.to_numeric(metadata["stock_id"], errors="coerce")
            names = metadata.dropna(subset=["stock_id"]).set_index("stock_id")["stock_name"].to_dict()
    return revenue, names


@st.cache_data(show_spinner=False)
def run_live(stock: int, as_of: str, data_dir: str, fingerprint: tuple, version: str):
    return build_live_forecast(stock, as_of, data_dir)


@st.cache_data(show_spinner=False)
def load_patterns(data_dir: str, stocks: tuple, as_of: str, fingerprint: tuple, version: str):
    return load_dividend_patterns(data_dir, stocks, as_of)


def _number(value, decimals=2):
    return "無法估算" if pd.isna(value) else f"{float(value):,.{decimals}f}"


def render_live_app():
    st.title("SARIMA＋營收公式｜實作模式")
    st.caption("以已公布資料估算當年與次年全年營收；兩年殖利率共用同一筆基準日股價。")
    with st.sidebar:
        st.subheader("實作設定")
        selected_date = st.date_input("預測基準日", value=datetime.now(timezone(timedelta(hours=8))).date(), key="live_as_of")
        raw_dir = st.text_input("CSV 資料夾", value=str(PROJECT_ROOT / "new data"), key="live_data_dir")
    directory = Path(raw_dir).expanduser()
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    directory = str(directory.resolve())
    as_of = selected_date.isoformat()
    try:
        fingerprint = data_fingerprint(directory)
        revenue, names = load_inputs(directory, as_of, fingerprint)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"無法讀取營收資料：{error}")
        return
    if revenue.empty:
        st.warning("基準日前沒有可用營收，請檢查資料夾、公布日期及基準日。")
        return
    if revenue.attrs.get("invalid_rows", 0):
        st.warning(f"有 {revenue.attrs['invalid_rows']} 筆營收缺少有效數值或公布日期，未納入。")
    all_stocks = tuple(sorted(revenue["stock_id"].unique().tolist()))
    catalog, catalog_detail, catalog_issues = load_patterns(directory, all_stocks, as_of, fingerprint, LIVE_CACHE_VERSION)
    pattern_lookup = catalog.set_index("stock_id")
    with st.sidebar:
        keyword = st.text_input("搜尋股票代號或名稱", key="live_search").strip()
        stocks = list(all_stocks)
        stocks = [s for s in stocks if not keyword or keyword in str(s) or keyword in str(names.get(s, ""))]
        if not stocks:
            st.info("找不到符合的股票。")
            return
        stock = st.selectbox("股票", stocks, index=stocks.index(2330) if 2330 in stocks else 0,
            format_func=lambda s: f"{s} {names.get(s, '')}｜{pattern_lookup.loc[s, 'dividend_pattern_label']}", key="live_stock")
        run = st.button("執行實作預測", type="primary", width="stretch", key="live_run")
    stock_revenue = revenue[revenue["stock_id"].eq(stock)]
    st.write(f"**{stock} {names.get(stock, '')}**　營收最新月份：{stock_revenue['date'].max():%Y/%m}　｜　基準日：{as_of}")
    st.caption("已公布月份保留實際營收；其餘月份使用 SARIMA 10%＋營收公式 90% 預測。")
    selected_pattern = pattern_lookup.loc[stock]
    st.write(f"**配息分類：{selected_pattern['dividend_pattern_label']}**")
    st.caption(selected_pattern["pattern_reason"])
    if selected_pattern["dividend_pattern"] == "fixed":
        st.info(f"每股現金股利採五年中位數 {selected_pattern['fixed_cash_dividend']:.4g} 元／年，不隨預估 EPS 倍增。")
    if selected_pattern["dividend_pattern"] == "insufficient" or selected_pattern["cash_history_years"] < 5:
        st.warning("歷史紀錄不完整，不能把缺漏年度當作零配息；有限年度平均可能偏高。")
    with st.expander("全部股票配息分類與名單", expanded=False):
        st.caption(f"{selected_date.year - 5}–{selected_date.year - 1}，共 {len(catalog)} 檔有歷史營收的股票。分類是歷史推估，非未來配息承諾。")
        for col, (pattern, label) in zip(st.columns(4), PATTERN_LABELS.items()):
            col.metric(label, int(catalog["dividend_pattern"].eq(pattern).sum()))
        st.write("固定：五年皆配現金，金額與中位數差距均在 ±5% 內。零配息：五年皆有明確零現金股利。正常：至少三年有紀錄且曾配現金，其餘待確認。")
        st.caption("金額為各獲利年度已公告紀錄合計；原始資料沒有全年已公告完畢標記。零現金股利不代表沒有股票股利。")
        for issue in catalog_issues:
            st.warning(issue)
        filters = st.multiselect("篩選配息分類", list(PATTERN_LABELS.values()), key="live_pattern_filter")
        catalog_display = format_dividend_catalog(catalog, names)
        shown = catalog_display if not filters else catalog_display[catalog_display["配息分類"].isin(filters)]
        leading = ["股票代號", "股票名稱", "配息分類", "固定每股現金股利估計（元）", "五年有效平均配息率（%）",
            "有效配息率年數", "有股利紀錄年數", "缺漏股利年度", "判定依據"]
        st.dataframe(shown[leading + [c for c in shown if "每股現金股利（元）" in c]], width="stretch", hide_index=True)
        st.download_button("下載全部股票分類 CSV", catalog_display.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"stock_dividend_labels_{as_of}.csv", mime="text/csv", key="live_download_catalog")
        st.download_button("下載篩選名單 CSV", shown.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"filtered_dividend_labels_{as_of}.csv", mime="text/csv", key="live_download_filtered")
        st.download_button("下載全體五年判讀依據 CSV", catalog_detail.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"dividend_evidence_{as_of}.csv", mime="text/csv", key="live_download_evidence")
    identity = (int(stock), as_of, directory, fingerprint, LIVE_CACHE_VERSION)
    if run:
        try:
            with st.spinner("正在預測至次年年底，並依配息分類計算股利與殖利率…"):
                result = run_live(*identity)
            st.session_state["live_result"] = result
            st.session_state["live_result_key"] = identity
        except (OSError, ValueError, KeyError, ImportError, ArithmeticError) as error:
            st.error(f"這檔股票暫時無法完成預測：{error}")
            return
    if st.session_state.get("live_result_key") != identity:
        st.info("按「執行實作預測」產生結果。更換股票、日期或 CSV 後須重新執行。")
        return
    result = st.session_state["live_result"]
    summary = result.summary
    priced_rows = summary.dropna(subset=["as_of_stock_price"]) if "as_of_stock_price" in summary else pd.DataFrame()
    price_row = priced_rows.iloc[0] if not priced_rows.empty else pd.Series(dtype=object)
    price, price_date = price_row.get("as_of_stock_price"), price_row.get("as_of_price_date")
    price_text = _number(price)
    date_text = pd.Timestamp(price_date).strftime("%Y/%m/%d") if pd.notna(price_date) else "無可用日期"
    st.info(f"兩年共同計價股價：{price_text} 元｜價格日期：{date_text}。使用 CSV 最新可得收盤價。")
    for pane, (_, row) in zip(st.columns(2), summary.iterrows()):
        with pane:
            year = int(row["target_year"])
            st.subheader(f"{year} 獲利年度")
            st.caption(f"實際營收 {row['actual_months']} 個月＋預測 {row['forecast_months']} 個月")
            cards = st.columns(2)
            cards[0].metric("全年營收（千元）", _number(row.get("predicted_annual_revenue"), 0))
            cards[1].metric("全年稅後 EPS（元）", _number(row.get("estimated_eps")))
            cards = st.columns(2)
            cards[0].metric("預估現金股利（元／股）", _number(row.get("estimated_cash_dividend")))
            value = row.get("as_of_price_yield_percent")
            cards[1].metric("預估現金殖利率", f"{_number(value)}%" if pd.notna(value) else "無法估算")
            if row.get("dividend_calculation"):
                st.caption(row["dividend_calculation"])
            if row.get("eps_status") == "EPS unavailable" and row.get("status") == "ok":
                st.warning("EPS 資料不足；此股利估計直接依歷史固定／零配息模式計算。")
            status = row.get("status", "")
            if status != "ok":
                st.warning({"EPS unavailable": "季度 EPS 估計依據不足", "payout unavailable": "配息紀錄或有效 EPS 配對不足，無法估算股利",
                    "price unavailable": "缺少有效股價"}.get(status, status))
    st.caption("EPS 為公司稅後獲利；股利金額未另扣個人所得稅或補充保費。年度表示獲利所屬年，不表示領息年。")
    monthly_tab, eps_tab, payout_tab, source_tab = st.tabs(["月營收", "季度 EPS", "配息分類與五年依據", "資料與方法"])
    with monthly_tab:
        frame = result.monthly
        chart = frame.set_index("target_date")[["actual_revenue", "hybrid_predicted_revenue"]].rename(
            columns={"actual_revenue": "已公布營收", "hybrid_predicted_revenue": "混合預測營收"})
        st.line_chart(chart)
        display = frame[["target_date", "actual_revenue", "predicted_revenue_sarima", "formula_adjusted_revenue",
            "hybrid_predicted_revenue", "revenue_used", "revenue_basis", "hybrid_method"]].copy()
        display["revenue_basis"] = display["revenue_basis"].map({"actual": "實際", "forecast": "預估", "unavailable": "缺資料"})
        display = display.rename(columns={"target_date": "月份", "actual_revenue": "實際營收", "predicted_revenue_sarima": "SARIMA",
            "formula_adjusted_revenue": "營收公式", "hybrid_predicted_revenue": "混合預測", "revenue_used": "年度計算採用營收",
            "revenue_basis": "資料類型", "hybrid_method": "混合方式"})
        st.dataframe(display, width="stretch", hide_index=True)
    with eps_tab:
        if result.quarterly_eps.empty:
            st.info("沒有可用的季度 EPS 結果。")
        else:
            frame = result.quarterly_eps.copy()
            frame["eps_basis"] = frame["eps_basis"].map({"actual": "已公布稅後 EPS", "seasonal_estimate": "歷史同季比率推估",
                "annual_ratio_fallback": "完整年度比率備援", "unavailable": "無法估算"})
            st.dataframe(frame[["target_year", "eps_quarter", "quarter_revenue", "quarter_eps", "eps_basis", "reference_years"]].rename(
                columns={"target_year": "年度", "eps_quarter": "季度", "quarter_revenue": "季營收（千元）", "quarter_eps": "稅後 EPS",
                    "eps_basis": "依據", "reference_years": "參考年份"}), width="stretch", hide_index=True)
    with payout_tab:
        if result.payout_history.empty:
            st.info("沒有可用的配息率明細。")
        else:
            payout_rows = summary.dropna(subset=["payout_valid_years"])
            payout_row = payout_rows.iloc[0]
            ratio = payout_row.get("payout_ratio")
            st.metric(f"{payout_row.get('payout_window', '')} 平均配息率", f"{float(ratio) * 100:.2f}%" if pd.notna(ratio) else "無法估算")
            st.write(f"有效年度：{int(payout_row.get('payout_valid_years', 0))}／5。先逐年計算現金股利 ÷ 稅後 EPS，再取有效年度的算術平均。")
            st.info(f"本次採用：{payout_row.get('dividend_pattern_label', '')}；{payout_row.get('dividend_calculation', '')}")
            st.caption("固定配息與零配息類不使用上方平均配息率乘 EPS；年度現金股利為已公告紀錄合計。")
            display = result.payout_history.copy()
            display["payout_ratio"] *= 100
            st.dataframe(display.rename(columns={"fiscal_year": "獲利年度", "annual_eps": "全年稅後 EPS", "cash_dividend": "已公告現金股利",
                "payout_ratio": "配息率（%）", "included": "是否採用", "reason": "採用／排除原因"}), width="stretch", hide_index=True)
    with source_tab:
        st.dataframe(result.data_status.rename(columns={"dataset": "資料", "source": "檔案", "latest_period": "最新資料期間",
            "latest_available_date": "最新可用日期", "rows": "採用筆數", "status": "狀態"}), width="stretch", hide_index=True)
        for note in result.notes:
            st.write(note)
        st.dataframe(result.order_search, width="stretch", hide_index=True)
    export_prefix = f"hybrid_{stock}_{as_of}"
    for column, (label, frame, suffix) in zip(st.columns(4), [
        ("年度摘要", result.summary, "annual"), ("月營收", result.monthly, "monthly"),
        ("季度 EPS", result.quarterly_eps, "eps"), ("配息率明細", result.payout_history, "payout"),
    ]):
        column.download_button(f"下載{label} CSV", frame.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{export_prefix}_{suffix}.csv", mime="text/csv", key=f"live_download_{suffix}")
