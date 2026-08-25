# Rolling SARIMA 月營收預測

這是一個與 `ensemble_forecast/`、`rolling_predict_LSTM/` 分離的傳統時間序列預測系統。

## 方法

- 對月營收使用 `log1p` 轉換。
- 固定 12 個月季節週期。
- 只用 2024 年底以前的資料，以 AIC 從小型 SARIMA 組合選參數。
- 2025 每個月重新使用目標月以前的已知營收做一步滾動預測。
- 2025 實際營收只在預測完成後合併計算誤差。
- 連續歷史不足或模型未收斂時，標記並退回去年同月營收。

## 建立環境

```powershell
cd sarima_forecast
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 啟動

```powershell
.\sarima_forecast\run_app.ps1
```

網址：`http://localhost:8503`

