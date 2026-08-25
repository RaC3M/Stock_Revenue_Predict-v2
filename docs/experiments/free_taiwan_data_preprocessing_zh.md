# free_taiwan_data 預處理層

> 文件狀態：更新至 2026-07-31。目前追蹤中的 canonical data 是 2026-07-30
> 產生的版本；它早於 manifest 的逐檔 SHA-256 功能，因此既有 manifest 尚未包含
> `file_sha256`。下次因資料更新而重新產生 `data/` 時會自動補上，不需要只為更新
> 文件而重建資料。`config.source_dir` 是來源追溯資訊，可能保留產生者的本機絕對路徑。
>
> 2026-07-31 data locality cleanup：`free_taiwan_data/` 只保留目前 adapter 會讀取的
> 五種 raw datasets；新的 candidate 與 audit 預設寫到 ignored
> `data_preprocessing/outputs/`。舊的 full/smoke/audit artifacts 已刪除。

## 目的

`data/` 現在是由 `free_taiwan_data/` 透過 preprocessing layer 產生的 canonical CSV interface。Ensemble、Rolling、benchmark 預設都讀 root `data/`；原始 `free_taiwan_data/` 保持 ignored，不直接 commit。

這份文件記錄資料遷移的處理規則、audit gate、root `data/` 目前狀態，以及後續重新產生資料時要跑的檢查。

`data_preprocessing/outputs/` 是 smoke、candidate 與 audit 的臨時輸出位置，正式程式路徑不需要用它取代 `data/`。歷史 `free_taiwan_data/processed_benchmark_82/` 暫時保留，直到 82 檔 cohort 另存成 tracked config。

## 新增入口

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m data_preprocessing.free_taiwan_data_preprocessor --stock-ids 1101,1231,3017 --output-dir data_preprocessing\outputs\processed_smoke_3
```

共同 benchmark 82 檔預處理：

```powershell
$ids = (Import-Csv forecast_benchmark\outputs\data_migration_revenue_20260730\comparable_monthly_predictions.csv | Select-Object -ExpandProperty stock_id -Unique) -join ','
.\ensemble_forecast\.venv\Scripts\python.exe -m data_preprocessing.free_taiwan_data_preprocessor --stock-ids $ids --output-dir data_preprocessing\outputs\processed_benchmark_82
```

完整 universe 預處理候選：

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m data_preprocessing.free_taiwan_data_preprocessor --full-universe --output-dir data_preprocessing\outputs\processed_full
```

`--full-universe` 不可與 `--stock-ids` 或 `--stock-limit` 併用。若完全不指定 stock filter，目前也等同完整 universe，但建議正式遷移時明確加上 `--full-universe`。

## 輸出檔案

預處理會輸出：

```text
stock_list_new.csv
Stock_revenue_2019~2025.csv
target_stocks_2025.csv
EPS2020~2025.csv
Dividend2019~2025.csv
day K2020~2025.csv
manifest.json
```

開發或 audit 輸出集中在 `data_preprocessing/outputs/`。正式 root 資料可明確指定 `--output-dir data`，並一起 commit `data/manifest.json`。`free_taiwan_data/` 與所有 `outputs/` 都已 ignored，避免把 raw inputs 或臨時產物 commit 進 repo。

`manifest.json` 現在是 canonical data contract 的一部分，至少包含：

- `data_contract_version`
- `dataset_role`
- `generator`
- `generated_at_utc`
- `files`
- `revenue_unit_contract`
- `row_counts`
- `stock_counts`
- `stock_coverage`
- `validation`

可以單獨驗證一個 candidate data directory：

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m data_preprocessing.canonical_data_contract data_preprocessing\outputs\processed_full --require-manifest --minimum-stock-counts stock_list=1900,revenue=1900,daily_prices=1900,dividends=1800
```

## 欄位對應

| Canonical 檔案 | free_taiwan_data 來源 | 重點欄位 |
|---|---|---|
| `stock_list_new.csv` | `technical/TaiwanStockInfo` | `stock_id`, `stock_name`, `industry_category` |
| `Stock_revenue_2019~2025.csv` | `fundamental/TaiwanStockMonthRevenue` | `revenue_year`, `revenue_month`, `revenue`, `revenue_thousand`, lag / rolling features |
| `target_stocks_2025.csv` | `fundamental/TaiwanStockMonthRevenue` | `2025` 年實際月營收 |
| `EPS2020~2025.csv` | `fundamental/TaiwanStockFinancialStatements` | `type = EPS`, `value -> EPS` |
| `Dividend2019~2025.csv` | `fundamental/TaiwanStockDividend` | `CashEarningsDistribution + CashStatutorySurplus -> TotalCashDividend` |
| `day K2020~2025.csv` | `technical/TaiwanStockPrice` | `date`, `open`, `max`, `min`, `close`, volume 欄位 |

## 目前保留的 raw inputs

`free_taiwan_data/` 只保留目前 preprocessor implementation 實際讀取的五個路徑：

- `technical/TaiwanStockInfo`
- `technical/TaiwanStockPrice`
- `fundamental/TaiwanStockMonthRevenue`
- `fundamental/TaiwanStockFinancialStatements`
- `fundamental/TaiwanStockDividend`

未被任何現行 adapter 讀取的 chip、derivative，以及其他 fundamental／technical datasets
已在 2026-07-31 移除。若未來新增籌碼面、資產負債表、現金流、PER 或 intraday features，
必須先重新取得對應 raw dataset，再明確擴充 preprocessor。

## Time-safe 日期欄位

預處理層會保留之後做 time-safe benchmark 需要的日期：

| 資料 | 新增 / 保留欄位 | 規則 |
|---|---|---|
| 月營收 | `revenue_available_date` | 該營收月份次月 10 日 |
| EPS / 財報 | `statement_available_date` | Q1: 5/15, Q2: 8/14, Q3: 11/14, 年報: 次年 3/31 |
| 現金股利 | `DividendAvailableDate` | 優先使用 `AnnouncementDate`，若缺失才 fallback 到 `CashExDividendTradingDate` |
| 現金股利 | `AnnouncementDate`, `AnnouncementTime` | 直接保留 free data 原始公告時間 |
| 除息 | `CashExDividendTradingDate` | 除息交易日，只適合 event / evaluation，不應取代公告日 |

正式殖利率預測應以：

```text
available_date <= as_of_date
```

作為資料可用性判斷，而不是只看 `fiscal_year < target_year`。

## 82 檔候選預處理結果（歷史 artifact 暫時保留）

這是資料遷移前用共同 benchmark 股票池做的 subset compatibility check；現在正式資料源已是 root `data/`。

輸出位置：

```text
free_taiwan_data/processed_benchmark_82
```

結果：

| frame | rows | stocks |
|---|---:|---:|
| stock_list | 82 | 82 |
| revenue | 6,659 | 82 |
| target_stocks | 984 | 82 |
| eps | 2,191 | 81 |
| dividends | 391 | 68 |
| daily_prices | 116,054 | 82 |

現有 Ensemble loader compatibility check：

| loader | rows | stocks |
|---|---:|---:|
| `load_revenue_data` | 6,659 | 82 |
| `load_actual_2025_data` | 984 | 82 |
| `load_eps_data` | 2,191 | 81 |
| `load_cash_dividend_data` | 391 | 68 |
| `load_stock_price_data(target_year=2025)` | 984 | 82 |

## 完整 universe 預處理結果（歷史 artifact 已刪除）

執行日期：2026-07-30

輸出位置：

```text
free_taiwan_data/processed_full
```

結果：

| frame | rows | stocks |
|---|---:|---:|
| stock_list | 2,788 | 2,788 |
| revenue | 153,021 | 2,016 |
| target_stocks | 23,020 | 1,958 |
| eps | 52,419 | 2,038 |
| dividends | 11,840 | 1,850 |
| daily_prices | 2,743,211 | 2,035 |

Manifest validation：

```text
is_valid = true
issues = []
warnings = []
```

## 與舊 root data 的 preprocessing audit（歷史 artifact 已刪除）

執行指令：

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m data_preprocessing.preprocessing_audit --baseline-dir data --candidate-dir free_taiwan_data\processed_benchmark_82 --output-dir free_taiwan_data\audit_benchmark_82
```

完整 universe candidate 的替換 gate；此 gate 已在 root data migration 前通過：

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m data_preprocessing.preprocessing_audit --baseline-dir data --candidate-dir free_taiwan_data\processed_full --output-dir free_taiwan_data\audit_full
```

輸出位置：

```text
free_taiwan_data/audit_benchmark_82
```

audit 會額外輸出：

```text
replacement_readiness_summary.csv
```

只要 `replacement_readiness_summary.csv` 仍有 `status = fail`，就不能更新根目錄 `data/`。

覆蓋範圍摘要：

| dataset | baseline rows | candidate rows | common keys | 說明 |
|---|---:|---:|---:|---|
| revenue | 152,760 | 6,659 | 6,659 | 新資料只取 benchmark 82 檔，因此 baseline-only 很多是正常現象 |
| eps | 44,835 | 2,191 | 1,889 | free data EPS 口徑仍需後續檢查 |
| dividends | 11,090 | 391 | 354 | 股利資料有重複 / 拆分 / 公告欄位口徑差異 |
| daily_prices | 2,730,622 | 116,054 | 116,054 | 共同 key 價格欄位完全一致 |
| stock_list | 2,985 | 82 | 82 | benchmark 82 檔皆可對應 |

Revenue 數值一致性：

| 欄位 | common rows | mismatch | mismatch rate |
|---|---:|---:|---:|
| `revenue` | 6,659 | 0 | 0.0000% |
| `revenue_thousand` | 6,659 | 0 | 0.0000% |
| `mom` | 6,659 | 0 | 0.0000% |
| `last_3m_revenue` | 6,659 | 0 | 0.0000% |
| `last_3m_revenue_yoy` | 6,659 | 0 | 0.0000% |
| `last_12m_revenue` | 6,659 | 0 | 0.0000% |
| `last_12m_revenue_yoy` | 6,659 | 0 | 0.0000% |
| `acc_revenue` | 6,659 | 0 | 0.0000% |
| `last_year_revenue` | 6,659 | 28 | 0.4205% |
| `yoy` | 6,659 | 28 | 0.4205% |
| `acc_revenue_yoy` | 6,659 | 28 | 0.4205% |

解讀：

- 核心營收數值與大部分衍生特徵已可重建既有 `data/` 口徑。
- `last_year_revenue` / `yoy` / `acc_revenue_yoy` 的 28 筆差異集中在少數早期歷史較短或對齊邊界股票，比例約 `0.42%`，目前不需要為了新上市或特殊資料硬補。
- EPS mismatch 為 `321 / 1,889 = 16.9931%`，股利 mismatch 為 `32 / 402 = 7.9602%`。這比較像舊資料更新、EPS 口徑、或股利拆分口徑差異，不應混在 revenue model 公平比較裡解讀。
- `target_stocks_2025.csv` 沒有 common key，是因為舊 `data/target_stocks_2025.csv` 只有 10 檔，而新 processed 版本輸出 benchmark 82 檔。現有 engine 主要仍可從完整 revenue file 載入 2025 actual，因此不是 blocker。

## 完整 universe audit 結果（歷史 artifact 已刪除）

執行日期：2026-07-30

輸出位置：

```text
free_taiwan_data/audit_full
```

覆蓋範圍摘要：

| dataset | baseline rows | candidate rows | common keys | baseline-only keys | candidate-only keys |
|---|---:|---:|---:|---:|---:|
| revenue | 152,760 | 153,021 | 152,760 | 0 | 261 |
| target_stocks | 120 | 23,020 | 120 | 0 | 22,900 |
| eps | 44,835 | 52,419 | 44,835 | 0 | 7,584 |
| dividends | 11,090 | 11,840 | 10,677 | 0 | 739 |
| daily_prices | 2,730,622 | 2,743,211 | 2,730,622 | 0 | 12,589 |
| stock_list | 2,985 | 2,788 | 2,115 | 0 | 673 |

Replacement readiness：

| dataset | check | observed | threshold | status |
|---|---|---:|---:|---|
| revenue | common key coverage | 1.0000 | 0.99 | pass |
| revenue | candidate stock coverage | 1.0035 | 0.99 | pass |
| target_stocks | common key coverage | 1.0000 | 0.99 | pass |
| eps | common key coverage | 1.0000 | 0.99 | pass |
| dividends | common key coverage | 1.0000 | 0.99 | pass |
| dividends | candidate stock coverage | 1.0137 | 0.99 | pass |
| daily_prices | common key coverage | 1.0000 | 0.99 | pass |
| stock_list | common key coverage | 1.0000 | 0.99 | pass |

解讀：

- `processed_full` 已經可以重建 revenue、daily prices、EPS、stock list、dividends 的既有 key coverage。
- revenue 數值核心欄位完全一致；`last_year_revenue` / `yoy` / `acc_revenue_yoy` 的 mismatch rate 為 `0.2520%`，低於替換 gate 的 `1%`。
- `dividends` 缺口的主因是 `year` 欄位不一定是單純民國年，例如 `113年後半年度`、`113年第4季`、`不適用`。預處理現在會保留 fiscal year 無法解析但有 `AnnouncementDate` 或 `CashExDividendTradingDate` 落在 2019-2025 的現金股利 row，並繼續保留 `DividendAvailableDate` 供下游做 time-safe 判斷。
- replacement readiness summary 目前沒有 `fail` rows，因此 full generated candidate 已通過替換 gate。

## Root data migration

執行日期：2026-07-30

Root `data/` 已由同一個 preprocessor 直接重建：

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m data_preprocessing.free_taiwan_data_preprocessor --full-universe --output-dir data
```

遷移後驗證：

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m data_preprocessing.canonical_data_contract data --require-manifest --minimum-stock-counts stock_list=1900,revenue=1900,daily_prices=1900,dividends=1800
.\ensemble_forecast\.venv\Scripts\python.exe -m data_preprocessing.preprocessing_audit --baseline-dir free_taiwan_data\processed_full --candidate-dir data --output-dir free_taiwan_data\audit_data_migration
```

結果：

- `data/manifest.json` validation 通過，`issues = []`, `warnings = []`。
- `free_taiwan_data\audit_data_migration\replacement_readiness_summary.csv` 沒有 `fail` rows。
- `processed_full` 與 root `data/` 的共同 keys 完全一致。
- revenue、target_stocks、EPS、dividends、daily_prices 的 numeric diff 都是 `0`。

## 已知限制

- 股利資料不是 82 檔全覆蓋，目前 `Dividend2019~2025.csv` 覆蓋 68 檔，因此後續 dividend layer 必須有 fallback 或 unavailable 標記。
- 完整 universe candidate 的股利資料已通過 replacement readiness gate，但 `TotalCashDividend` 仍有 `424 / 12,106 = 3.5024%` 的共同 row 數值差異。這比較像舊手工資料與 free data 更新 / 拆分口徑差異，若要做殖利率研究仍需在 dividend layer benchmark 中保留實測比較。
- EPS 資料覆蓋 81 檔，缺 1 檔需在 EPS benchmark 中保留 failed/unavailable rows。
- free data 的 EPS 口徑目前沿用既有 `EPS2020~2025.csv` 的形式；後續若要嚴格區分單季 EPS 與累計 EPS，需要另做 EPS 口徑驗證。
- 主程式預設讀 root `data/`，而 root `data/` 已是 generated canonical CSV。需要做臨時 subset audit 時，可用 `PREDICT_DATA_DIR` 指向 `data_preprocessing/outputs/processed_*`，不需要改程式碼。

## 下一步

1. Ensemble 與 Rolling loader 已加入可設定資料根目錄：`PREDICT_DATA_DIR`；未設定時讀 root generated `data/`。
2. `free_taiwan_data/` 是 ignored raw source，`data/` 是已追蹤的 generated canonical CSV interface。
3. 歷史 `processed_full` 已通過 manifest validation；完成 root migration 並確認六份 CSV hash 一致後，重複 artifact 已刪除。
4. full audit 顯示 revenue / daily prices / EPS / stock list / dividends 都已通過 replacement readiness gate。
5. root `data/` 已完成 migration，並加入 `data/manifest.json` 作為 generated data metadata。
6. 後續資料更新應重跑 preprocessor、manifest validation、preprocessing audit、Ensemble / Rolling tests，再做資料更新 commit；新版 manifest 會包含逐檔 SHA-256。
7. 已將 dividend layer benchmark 改成可使用 `DividendAvailableDate <= as_of_date`。
8. 保留 `current_system_payout_ratio` 作為 legacy hindsight diagnostic；目前 Ensemble Forecast System 的正式路徑已改成 time-safe historical payout，後續更嚴格的殖利率結果可再使用 announcement-safe dividend layer。

例如：

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m data_preprocessing.free_taiwan_data_preprocessor --full-universe --output-dir data
.\ensemble_forecast\.venv\Scripts\python.exe -m data_preprocessing.canonical_data_contract data --require-manifest --minimum-stock-counts stock_list=1900,revenue=1900,daily_prices=1900,dividends=1800
.\ensemble_forecast\.venv\Scripts\python.exe -m forecast_benchmark.run_benchmark --stock-limit 3 --output-dir forecast_benchmark\outputs\data_refresh_smoke_3 --report-ready false
```
