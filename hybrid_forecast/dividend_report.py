from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from financial_forecast.dividend_patterns import load_dividend_patterns, PATTERN_LABELS
from financial_forecast.evidence import resolve_data_files
from hybrid_forecast.live_engine import data_fingerprint, load_revenue_snapshot


def format_dividend_catalog(catalog: pd.DataFrame, names: dict) -> pd.DataFrame:
    frame = catalog.copy()
    frame.insert(1, "stock_name", frame["stock_id"].map(names).fillna(""))
    frame["payout_ratio"] *= 100
    return frame.rename(columns={
        "stock_id": "股票代號", "stock_name": "股票名稱", "dividend_pattern": "分類代碼",
        "dividend_pattern_label": "配息分類", "pattern_reason": "判定依據",
        "pattern_as_of_date": "判定基準日", "pattern_basis": "分類性質",
        "pattern_rule_version": "規則版本", "payout_window": "歷史年度",
        "cash_history_years": "有股利紀錄年數", "positive_cash_years": "正現金股利年數",
        "zero_cash_years": "零現金股利年數", "missing_cash_years": "缺漏股利年度",
        "payout_ratio": "五年有效平均配息率（%）", "payout_valid_years": "有效配息率年數",
        "dividend_reference_year": "最新股利獲利年度", "fixed_cash_dividend": "固定每股現金股利估計（元）",
        "cash_max_deviation_percent": "股利相對中位數最大差距（%）",
        **{c: f"{c[-4:]} 每股現金股利（元）" for c in frame if c.startswith("cash_dividend_")},
    })


def main():
    parser = argparse.ArgumentParser(description="依基準日前五年資料輸出配息分類及逐年依據")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--data-dir", default="new data")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    cutoff = pd.Timestamp(args.as_of)
    revenue = load_revenue_snapshot(args.data_dir, cutoff)
    metadata = pd.read_csv(resolve_data_files(args.data_dir)["stock_list"])
    names = metadata.set_index("stock_id")["stock_name"].to_dict()
    catalog, detail, issues = load_dividend_patterns(args.data_dir, revenue["stock_id"].unique(), cutoff)
    output = Path(args.output_dir or f"hybrid_forecast/outputs/dividend_patterns_{cutoff:%Y%m%d}")
    output.mkdir(parents=True, exist_ok=True)
    display = format_dividend_catalog(catalog, names)
    display.to_csv(output / "stock_dividend_labels.csv", index=False, encoding="utf-8-sig")
    detail.insert(1, "stock_name", detail["stock_id"].map(names).fillna(""))
    detail.to_csv(output / "five_year_evidence.csv", index=False, encoding="utf-8-sig")
    counts = []
    for pattern, label in PATTERN_LABELS.items():
        group = display[catalog["dividend_pattern"].eq(pattern)]
        group.to_csv(output / f"{pattern}_stocks.csv", index=False, encoding="utf-8-sig")
        counts.append(f"- {label}：{len(group)} 檔")
    lines = [
        f"# 配息分類清單（基準日 {cutoff:%Y-%m-%d}）", "",
        f"範圍：CSV 中基準日前有可用營收的 {len(catalog)} 檔股票，並非目前掛牌名單。",
        f"歷史窗口：{cutoff.year - 5}–{cutoff.year - 1}；同一基準日的兩個預測年度使用同一分類。", "",
        *counts, "", "## 判定及計算", "",
        "- 固定配息：五年每年皆有正現金股利，與五年中位數差距均在 ±5% 內；股利預估採該中位數，不乘 EPS。",
        "- 不配現金股利：五年都有明確零現金股利紀錄；股利估計為零，不代表沒有股票股利。",
        "- 正常配息：至少三年有股利紀錄且曾配現金，未符合固定金額規則；EPS 乘五年有效配息率算術平均。",
        "- 資料不足：不強制分入三類；若曾配現金且有有效 EPS／股利配對，暫用有限年度平均並標示依據，否則不估算。",
        "- 年度股利採基準日前已公告紀錄合計；未知、負數或無效紀錄不作零處理。EPS 非正的年度不計入配息率平均。",
        "- 分類是歷史近似，未經公司確認，±5% 為判讀規則而非驗證得出的最佳門檻。",
        "- 資料沒有全年股利已公告完畢標記，已知金額未必等於最終全年股利；後續公告應更新資料再重算。",
        "- EPS 維持已公布單季 EPS 與營收比率推估，沒有新增淨利／股數資料；全年 EPS 仍採四季合計。",
        "- 2025 沙盒與原始 CSV 不套用這份 2026 分類；實作模式依使用者選定基準日重新判讀。", "",
        "## 固定配息名單", "",
    ]
    for _, row in catalog[catalog["dividend_pattern"].eq("fixed")].iterrows():
        stock = int(row["stock_id"])
        lines.append(f"- {stock} {names.get(stock, '')}：預估每股 {row['fixed_cash_dividend']:g} 元／年")
    lines += ["", "## 資料問題", "", *(issues or ["未遇到檔案讀取錯誤；個股缺漏年度請看清單。"]),
        "", "## 輸入檔案指紋", "", "```text", repr(data_fingerprint(args.data_dir)), "```", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join([str(output.resolve()), *counts, *issues]))


if __name__ == "__main__":
    main()
