# 營收調整公式實驗

這個資料夾獨立於集成、xLSTM 與 SARIMA，不會改動既有預測程式。

## 執行方式

在專案根目錄執行：

```powershell
& '.\sarima_forecast\.venv\Scripts\python.exe' -m unittest revenue_adjustment_formula.tests.test_formula_engine
& '.\sarima_forecast\.venv\Scripts\python.exe' '.\revenue_adjustment_formula\run_experiment.py'
```

程式會：

1. 用 2023–2024 滾動回測搜尋公式參數。
2. 鎖定最佳參數後，只回放一次 2025。
3. 與沿用上月、去年同月、公式未校正值比較。
4. 若找到既有結果檔，再以共同股票月份比較 SARIMA 與 xLSTM。
5. 在 `outputs/` 產生 CSV、JSON 與中文 Markdown 報告。

## 開啟回測結果網頁

```powershell
.\revenue_adjustment_formula\run_app.ps1
```

瀏覽器網址：`http://localhost:8504`

公式與後續改善方向詳見 [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md)。
