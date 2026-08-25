# 多模型集成營收與殖利率預測系統

獨立的 Streamlit 預測系統，使用多種非 LSTM 模型預測台灣股票月營收，依歷史驗證誤差建立 ensemble 權重，再估算股利與殖利率。

本系統與 `rolling_predict_LSTM/` 平行，兩者只共用 root `data/`，不得互相 import。跨系統比較放在 `forecast_benchmark/`。

## Models

- XGBoost
- LightGBM
- CatBoost
- Seasonal Quantile
- Validation-weighted ensemble

2025 實際營收只用於評估，不用於訓練特徵或模型權重。

目前正式殖利率路徑由 `yield_forecast.py` 薄 adapter 呼叫 root `financial_forecast/`。它以 target year 的 `01-10` 作為 EPS 可得性 cutoff，只接受 cutoff 前可得的完整年度 EPS，並使用 announcement-safe historical payout evidence。個股 payout 證據不足時使用歷史橫斷面中位數；若仍無證據就標示無法估算，不生成隨機股利政策。Benchmark 中的 `current_system_payout_ratio` 僅保留為 legacy hindsight diagnostic，不代表本系統目前正式行為。

UI 分開顯示 cutoff 當下真實價格的 `as_of_price_yield` 可部署殖利率，以及 target-year 月末真實價格的 `target_month_end_yield` 回測殖利率。後者不是股價預測。缺月時只沿用前次真實收盤價；完全沒有真實價格時殖利率保持 unavailable，不生成模擬價格。

## Data and forecast contract

- 預設資料目錄：root `data/`；可用 `PREDICT_DATA_DIR` 明確覆寫。
- `RevenueDataContract` 將 raw-NTD 衍生營收欄位轉成模型使用的千元單位。
- 下一年 target 依股票、calendar month 與 target year 對齊，不使用 row offset 猜測。
- 缺值填補只使用同一股票較早的 observations。
- Ensemble 權重評估的是實際 weighted ensemble predictions，不把個別模型平均誤差當作 ensemble error。
- Partial-history 股票可由 SeasonalQuantile fallback 產生完整 target-month 預測。

## Environment

From the repository root:

```powershell
python --version  # must be Python 3.11.x
python -m venv ensemble_forecast\.venv
.\ensemble_forecast\.venv\Scripts\python.exe -m pip install --upgrade pip
.\ensemble_forecast\.venv\Scripts\python.exe -m pip install -r ensemble_forecast\requirements.txt
```

To recreate the verified package versions:

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m pip install -r ensemble_forecast\requirements-lock.txt
```

## Run

```powershell
.\ensemble_forecast\run_app.ps1
```

The app uses port `8501`.

## Tests

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m unittest discover -s ensemble_forecast\tests -v
```

Quick smoke：

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -c "from ensemble_forecast.forecast_engine import build_forecast; r=build_forecast(1101); print(r.metrics.to_string(index=False)); print(r.weights.to_string(index=False))"
```

## Evidence and outputs

- 本機輸出放在 `ensemble_forecast/outputs/`，由 Git ignore。
- 跨系統 accuracy、EPS、股利與殖利率結論以 `forecast_benchmark/` 和 `docs/experiments/experiment_registry.md` 為準。
- `forecast_benchmark/` 可以透過 adapter 使用 Ensemble evidence；不要為了 benchmark 改變本系統的 public behavior。

## Ownership

This system owns:

- `app.py`: Streamlit interface
- `forecast_engine.py`: data preparation, model forecasts, validation weights, yield calculation
- `tests/`: Ensemble behavior and isolation tests
- `outputs/`: optional local outputs, ignored by Git

It reads source data from the repository root `data/` directory and must not import `rolling_predict_LSTM`.
