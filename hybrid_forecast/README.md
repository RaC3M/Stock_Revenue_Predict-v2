# SARIMA＋營收調整公式

這個資料夾是獨立的低資料量混合預測實驗，不修改既有 `sarima_forecast/` 或 `revenue_adjustment_formula/`。

正式流程：

1. 對 2023、2024 做逐月時間安全回放。
2. 搜尋 SARIMA 權重 0.0～1.0，步長 0.1。
3. 以全體 WMAPE 與個股 WMAPE 中位數的平均選權重。
4. 凍結權重後評估 2025。
5. SARIMA 無法擬合、數值不合法，或高於「公式預測與上月營收較大值的 2 倍」時，自動退回營收調整公式。

執行全市場實驗：

```powershell
& '..\sarima_forecast\.venv\Scripts\python.exe' run_experiment.py --workers 8 --resume
```

啟動結果頁：

```powershell
.\run_app.ps1
```

網址：`http://localhost:8505`

## 雙模式

同一個頁面左側可切換「實作模式」（預設）與「2025 測試沙盒」。沙盒保留原有
2025 報告及結構斷點比較；實作模式不依賴這些歷史輸出檔。

實作模式使用 `new data/`，也可在介面指定其他 CSV 資料夾。若有 `manifest.json`，
依其中的 `files` 對應讀取檔案；否則沿用現有 CSV 檔名。檔名含 2025 不代表內容只能到 2025。
CSV 更新後按「執行實作預測」即可重新計算，快取會依路徑、檔案大小及修改時間失效。

### 無 UI 介面、持久化 artifact 與 LLM 資料包

`hybrid_forecast.live_engine` 可直接由 Python 呼叫，不依賴 Streamlit：

```python
from hybrid_forecast.live_engine import (
    ForecastRequest, apply_price_scenario, load_or_build_forecast,
)

request = ForecastRequest(2330, "2026-09-19", "new data")
core = load_or_build_forecast(request)
result = apply_price_scenario(
    core, stock_price=1200, price_date="2026-09-19", price_source="manual_scenario",
)
```

- `load_or_build_forecast()` 只快取營收、EPS 與股利階段，artifact 位於
  `outputs/live_artifacts/<stock>/<as-of>/<artifact-id>/`。
- artifact ID 包含模型版本、參數與營收／EPS／股利檔案內容雜湊，不包含股價。
- 外部月營收預測的 artifact ID 另包含外部預測內容、`source_family`、`model` 與財務證據雜湊；
  匯出 JSON 會保留外部模型來源，不會把內部混合公式誤標成外部模型公式。
- `apply_price_scenario()` 只以已估算現金股利重算殖利率，不重跑 SARIMA、營收公式、EPS 或股利。
- `build_financial_core_from_predictions()` 可接收其他組別的標準月預測欄位；每個年度必須恰有 1–12 月。
- LLM JSON 包含式子、資料截止日、價格來源、缺漏與限制，不輸出買賣建議。

批次匯出每檔股票的 JSON 與 CSV 資料包：

```powershell
& '.\sarima_forecast\.venv\Scripts\python.exe' -m hybrid_forecast.export_forecast `
  --data-dir '.\new data' --as-of 2026-09-19 --stock 2330 --stock 1101
```

手動價格情境可重複使用 `--manual-price STOCK=PRICE`；未指定時使用基準日以前
CSV 最新可得收盤價。單檔失敗不會中斷其他股票，程式會在最後輸出完成與失敗清單。

- 基準日預設台北當天，可自行調整；輸出該年及次年的完整年度。
- 營收必須包含 `revenue_available_date`，EPS 使用 `statement_available_date`，
  股利使用 `DividendAvailableDate`；只採基準日以前可得的資料。
- 已公布營收保留實際值。SARIMA 用最後連續歷史區段選參並一次預測至次年年底，
  公式以自己的預測逐月遞推，再以 SARIMA 0.1＋公式 0.9 組合。
- 公式固定採既有驗證參數：seasonal_weight=0.5、residual_alpha=0.1、
  residual_strength=0、growth_log_cap=ln(2)、correction_log_cap=0.5。
- 實作模式不啟用結構斷點改善版，不使用預測值重新訓練 SARIMA。
- EPS CSV 的 `EPS` 必須是**單季公司稅後 EPS**。已公布季度直接採用；其餘季度
  使用最近最多三個有效歷史同季的 EPS／營收中位比率，必要時退回完整年度比率。
- 配息率取基準日前五個已結束會計年度的「逐年現金股利／全年 EPS」算術平均。
  缺資料、EPS 非正數的年度排除並列出原因；明確零配息採用，超過 100% 的有效比率不截斷。
- 實作模式依五年歷史判讀配息分類：五年現金股利皆為正且與中位數差距在 ±5% 內，
  標記「固定配息（歷史近似）」，直接使用五年現金股利中位數，不再乘預估 EPS。
  五年皆有明確零現金股利才標記「不配現金股利」，股利預估為零（不代表沒有股票股利）。
  其餘至少三年有紀錄且曾配現金者標記「正常配息」，使用上述平均配息率。
- 不足以確認分類者獨立標記「資料不足／待確認」；若曾配現金且有有效股利／EPS
  配對，保留有限年度平均估計並提醒，否則不估股利。缺漏年度不當作零。
- 分類只使用基準日之前已公告資料，與股價無關，不從殖利率反推；兩個預測年度
  共用同一歷史窗口。±5% 是透明判讀門檻，未經回測選優；分類不是公司配息承諾。
- 年度股利為已公告紀錄合計，來源未提供全年已公告完畢標記。尚未公布的分期股利
  可能使金額不完整，需隨 CSV 更新重算；不以單筆零現金股利宣告整個目標年度不配息。
- 兩年使用同一筆基準日前最新有效收盤價；也可輸入不寫回 CSV 的手動價格情境。價格改變只重算殖利率。
- 年份代表獲利所屬年度，不表示股利領取年；不另扣個人所得稅或補充保費。
- 財務資料缺漏時保留營收預測。全年營收不足 12 個有效月份，不默認年化或補零。

介面提供全體股票分類篩選／CSV、個股分類與判定理由、月營收、季度 EPS、五年配息依據
及資料來源明細。分類清單不必執行營收預測即可查看。另可產生完整清單與各類分檔：

```powershell
.\sarima_forecast\.venv\Scripts\python.exe -m hybrid_forecast.dividend_report --as-of 2026-09-02
```

輸出至 `hybrid_forecast/outputs/dividend_patterns_20260902/`，保留原始 CSV 不變。

實作模式的固定起點多步預測與沙盒逐月更新實績的一步回測不同；沙盒的 WMAPE
不能直接當作未來全年預測的準確度。長期預測應另外以固定起點回放評估。

驗證（在專案根目錄）：

```powershell
.\sarima_forecast\.venv\Scripts\python.exe -W error::FutureWarning -m unittest hybrid_forecast.test_live_engine hybrid_forecast.test_export_forecast financial_forecast.tests.test_dividend_patterns financial_forecast.tests.test_live_pipeline financial_forecast.tests.test_pipeline_contract sarima_forecast.tests.test_sarima_engine -v
```
