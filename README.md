# Yield-Predict

台灣股票月營收、EPS、現金股利與殖利率預測研究專案。Repository 由三套彼此獨立的營收預測系統、一個中立的財務轉換模組，以及一個隔離的比較層組成：

| 區域 | 角色 | 主要內容 |
|---|---|---|
| [`ensemble_forecast/`](ensemble_forecast/) | 傳統機器學習預測系統 | XGBoost、LightGBM、CatBoost、SeasonalQuantile 與歷史驗證加權 |
| [`rolling_predict_LSTM/`](rolling_predict_LSTM/) | 序列模型預測系統 | Rolling LSTM、KMeans pattern cluster、動態 regime，以及可選歷史 mLSTM-only／`mLSTM → sLSTM` Hybrid xLSTM |
| [`sarima_forecast/`](sarima_forecast/) | 傳統時間序列預測系統 | Log-SARIMA、12 個月季節週期、AIC 小型選參與 2025 逐月一步滾動回測 |
| [`revenue_adjustment_formula/`](revenue_adjustment_formula/) | 可解釋營收調整公式 | 去年同月季節性、近三個月年增趨勢與上月營收水準的穩健混合 |
| [`hybrid_forecast/`](hybrid_forecast/) | SARIMA＋營收公式實驗 | 驗證式權重、guardrail、fallback 與結構性崩落偵測改善版 |
| [`financial_forecast/`](financial_forecast/) | 中立財務轉換模組 | availability-safe EPS、現金股利，以及 deployable／evaluation 殖利率 |
| [`forecast_benchmark/`](forecast_benchmark/) | 跨系統評估層 | 統一股票池、欄位、metrics、EPS／股利／殖利率下游 benchmark 與 evidence metadata |

這是研究原型，不是投資系統，也不構成投資建議。

## 2026-08-25 更新紀錄

- 新增獨立 Rolling SARIMA 流程：營收使用 `log1p`、季節週期 12 個月、9 組小型候選參數依 AIC 選擇，並以一步滾動方式回測；歷史不足或模型失敗時退回去年同月基準。
- 新增可解釋營收調整公式：以最近三個月 YoY log 成長率中位數調整去年同月營收，再與上月營收於 log 空間各取 50% 混合；成長倍率限制在 0.5～2.0 倍，殘差校正強度為 0。
- 新增 `hybrid_forecast/`：使用 2023–2024 滾動驗證選擇 SARIMA／公式權重，選中 `SARIMA 0.1＋公式 0.9`，並加入有限值、非負、上限與方法別 guardrail／fallback。
- 新增結構性崩落改善版：只使用預測當下已知月份偵測連續營收縮水，不偷看目標月份實際值；保留原版並在 Streamlit 介面並排比較。
- 完成全市場誤差、高誤差股票與改善前後分析。公平 1,635 檔樣本中，純 SARIMA WMAPE 為 9.712%，原始混合為 10.069%；結構斷點版在其可比較全樣本中由 10.5788% 改為 10.5763%。6405 悅城的個股 WMAPE 由 796.04% 降至 74.83%，但改善不代表第一個突發斷點可以預知。
- 新增完整進度文件 [`docs/SARIMA至結構斷點改善_完整進度報告_20260821.docx`](docs/SARIMA至結構斷點改善_完整進度報告_20260821.docx) 與 [`docs/WEEKLY_PROGRESS_20260820.md`](docs/WEEKLY_PROGRESS_20260820.md)，保留參數、評估設計、誤差股票與限制說明。

## Current status

截至 2026-08-06：

- 三套預測系統共用 root [`data/`](data/) canonical CSV，但不得互相 import。
- 兩套 UI 透過各自的薄 adapter 使用 [`financial_forecast/`](financial_forecast/)；該模組不訓練營收模型，也不負責選模。
- 跨系統比較只放在 `forecast_benchmark/`，它不是第三套預測模型。
- Rolling UI 的 optional xLSTM 預設為 `mLSTM → sLSTM` Hybrid；歷史 mLSTM-only 仍可選來重現 D1 實驗。
- D1.21 已補上 Hybrid frozen-parameter basket-100 結果：100/100 檔成功，Hybrid adjusted WMAPE `17.448%`，優於同次 run 的 cluster adjusted `19.359%`。它仍是帶有 2025 policy-development 歷史的 Tier C 證據，且不可沿用或直接對比舊 mLSTM-only D1 指標。
- D1.22 已在同一批股票與月份直接比較 Hybrid 和歷史 mLSTM-only。Hybrid no-cluster pooled WMAPE 為 `17.598%`，略差於歷史版本 `17.166%`，但股票層級 WMAPE 勝出 63/100 檔並通過預先登記的非重大退步門檻；結論是保留 Hybrid，但不宣稱 plain aggregate 全面勝出。
- EPS 變準不保證殖利率同步變準。正式下游結論必須使用 time-safe／announcement-safe 股利證據。
- Ensemble 與 Rolling UI 都提供營收到殖利率的完整鏈；`as_of_price_yield` 使用 cutoff 當下已知真實價格，`target_month_end_yield` 使用 target-year 實際月末價格且只作回測，不代表系統有預測股價。

最新結論與證據限制請以 [`docs/experiments/experiment_registry.md`](docs/experiments/experiment_registry.md) 和 [`docs/experiments/benchmark_protocol.md`](docs/experiments/benchmark_protocol.md) 為準。

## Quick start

### 1. Clone and fetch data

`data/day K2020~2025.csv` 使用 Git LFS。第一次 clone 後執行：

```powershell
git lfs install
git lfs pull
```

若未安裝 Git LFS，只會取得 pointer file，系統無法讀取完整日股價資料。

### 2. Python environments

三套系統維持獨立虛擬環境。先執行 `python --version`，確認目前
`python` 是 3.11.x；本專案不假設 Windows 已安裝 `py` launcher。

Ensemble：

```powershell
python -m venv ensemble_forecast\.venv
.\ensemble_forecast\.venv\Scripts\python.exe -m pip install --upgrade pip
.\ensemble_forecast\.venv\Scripts\python.exe -m pip install -r ensemble_forecast\requirements-lock.txt
```

Rolling LSTM（已驗證的 Windows CUDA 13.0 環境）：

```powershell
python -m venv rolling_predict_LSTM\.venv
.\rolling_predict_LSTM\.venv\Scripts\python.exe -m pip install --upgrade pip
.\rolling_predict_LSTM\.venv\Scripts\python.exe -m pip install -r rolling_predict_LSTM\requirements-lock.txt
```

`requirements-lock.txt` 直接使用 CUDA 13.0 wheel index 並固定 `torch==2.11.0+cu130`，不是
CPU-only lock。沒有 CUDA 13.0 相容環境時，改裝 base／CPU 路徑：

```powershell
.\rolling_predict_LSTM\.venv\Scripts\python.exe -m pip install -r rolling_predict_LSTM\requirements.txt
```

GPU 與 optional xLSTM 的額外安裝方式請看 [`rolling_predict_LSTM/README.md`](rolling_predict_LSTM/README.md)。
Rolling lockfile 固定 `pandas==2.2.3` 與 `pytz==2025.2`，以避開目前 Windows 應用程式控制原則封鎖新版 pandas DLL 的環境問題。

SARIMA：

```powershell
python -m venv sarima_forecast\.venv
.\sarima_forecast\.venv\Scripts\python.exe -m pip install --upgrade pip
.\sarima_forecast\.venv\Scripts\python.exe -m pip install -r sarima_forecast\requirements.txt
```

### 3. Run the apps

```powershell
.\ensemble_forecast\run_app.ps1
.\rolling_predict_LSTM\run_app.ps1
.\sarima_forecast\run_app.ps1
.\revenue_adjustment_formula\run_app.ps1
.\hybrid_forecast\run_app.ps1
```

- Ensemble Streamlit：`http://localhost:8501`
- Rolling LSTM Streamlit：`http://localhost:8502`
- Rolling SARIMA Streamlit：`http://localhost:8503`
- 營收調整公式 Streamlit：`http://localhost:8504`
- SARIMA＋營收公式／結構斷點比較 Streamlit：`http://localhost:8505`

## Validation

從 repository root 執行完整驗證：

```powershell
python tools\validate_project.py
```

這個入口會使用正確的兩個虛擬環境，先編譯檢查所有專案 Python 原始碼，再依序驗證 tracked
canonical `data/` 與 manifest、以 `pip check` 檢查 Ensemble 與 Rolling 的已安裝依賴，並執行
tooling、data preprocessing、Ensemble、financial forecast、benchmark 與 Rolling 測試；預設將
`FutureWarning` 視為失敗。
局部驗證可重複使用 `--suite`，例如 `--suite rolling --suite benchmark`；只有在調查第三方
套件遷移時才使用 `--allow-future-warnings`。

完整模型訓練可能耗時；小型 smoke 指令記錄在各系統 README。

## Data and generated outputs

三套系統預設讀取以下 canonical files：

- `data/Stock_revenue_2019~2025.csv`
- `data/EPS2020~2025.csv`
- `data/Dividend2019~2025.csv`
- `data/day K2020~2025.csv`
- `data/stock_list_new.csv`
- `data/target_stocks_2025.csv`
- `data/manifest.json`

`data/` 是由 ignored `free_taiwan_data/` 經 `data_preprocessing/` 產生的 tracked interface。Raw 目錄只保留目前 adapters 需要的五種來源；candidate 與 audit 暫存集中在 ignored `data_preprocessing/outputs/`。重新整理資料前，請依 [`docs/adr/0002-canonical-data-generation.md`](docs/adr/0002-canonical-data-generation.md) 執行 manifest validation 與 preprocessing audit。

`outputs/`、cache、虛擬環境與 raw `free_taiwan_data/` 不納入 Git。Fresh clone 可以執行三套預測系統，但不會包含本機完整實驗輸出；可引用的結果與 provenance 應整理到 tracked `docs/`，完整 raw artifacts 另行保存。

## Documentation

文件入口在 [`docs/README.md`](docs/README.md)：

- 架構決策與系統邊界
- 最新實驗結論與 evidence tier
- 資料更新流程
- 實驗 protocol、結果狀態與證據限制
- AI/coding agent 協作規則

協作者請另外閱讀 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## Architecture rule

`ensemble_forecast/`、`rolling_predict_LSTM/` 與 `sarima_forecast/` 是 peer systems，不得直接 import 彼此。它們可共用 `data/`；Ensemble 與 Rolling 可使用不含營收訓練／選模邏輯的 `financial_forecast/`。任何跨系統比較與方法選擇必須留在 `forecast_benchmark/` 或另一個明確隔離的分析工具中。
