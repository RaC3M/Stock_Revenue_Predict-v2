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
