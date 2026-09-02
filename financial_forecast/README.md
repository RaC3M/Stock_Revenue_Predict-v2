# Shared Financial Forecast Module

`financial_forecast` is the neutral downstream transformation module used by both forecasting
systems. It converts a complete 12-month revenue forecast into availability-safe EPS and cash
dividend estimates, then calculates two explicitly different yield views.

Public interface:

```python
from financial_forecast import FinancialForecastPolicy, forecast_financials
```

The monthly input contract requires:

- `source_family`
- `model`
- `stock_id`
- `target_year`
- `target_month`
- `predicted_revenue`

Every source/model/stock group must contain exactly months 1–12. Incomplete or duplicate groups
are returned in `result.failures`; partial years are never silently annualized.

## Stages

- EPS: annual revenue-ratio median or same-quarter seasonal ratio median.
- Cash dividend: announcement-safe payout ratio, last known cash dividend, historical median, or
  recency-weighted cash dividend.
- Yield:
  - `as_of_price_yield`: latest real close at or before the cutoff; deployable estimate.
  - `target_month_end_yield`: target-year observed month-end close; evaluation only.

The module owns financial evidence loading and transformations only. It does not train revenue
models, choose benchmark cohorts, select a winning method, or compare forecasting systems. Those
responsibilities stay in the owning forecast system and `forecast_benchmark/`.

## Live hybrid policies

The hybrid live adapter selects `known_quarters_plus_seasonal`, `historical_pattern_dividend`, and
`as_of_price_yield` through the same `forecast_financials` interface. Existing policy defaults
remain unchanged. These live policies read manifest file mappings and enforce explicit revenue,
EPS, and dividend availability dates. Missing financial files produce unavailable estimates and
data-status notes instead of fabricated inputs.

- `known_quarters_plus_seasonal`: retain reported single-quarter after-tax EPS; estimate missing
  quarters from up to three prior same-quarter EPS/revenue ratios, with a complete-year ratio fallback.
- `five_year_mean_payout`: arithmetic mean of valid yearly cash-dividend/EPS ratios within
  `as_of_date.year - 5` through `as_of_date.year - 1`. Both forecast years use the same window.
  Nonpositive EPS and missing evidence are excluded; valid zero payouts are retained, and ratios
  are not capped. No personal income tax or health-insurance deduction is applied.
- `historical_pattern_dividend`: classifies the same five-year window before estimating cash.
  Five positive annual cash totals all within 5% of their median imply a historically fixed
  amount; estimate that median independently of EPS. Five explicit zero annual totals imply
  zero cash dividends, also independently of EPS. At least three observed years and one positive
  year otherwise imply the normal payout-ratio path. Insufficient history remains labeled
  unknown; positive cash history with valid EPS pairings may use a clearly labeled limited-year
  mean, but absent cash records and incomplete zero histories never imply zero future cash.
  Missing or invalid installments invalidate the year rather than becoming partial sums.
  Summary rows expose the classification, dates, reasons, history, calculation, and EPS status.
  A fixed or zero dividend can remain estimable when EPS is unavailable. The current annual
  revenue completeness gate still applies. Existing defaults and `five_year_mean_payout`
  are retained for compatibility.

Classifications are historical heuristics, not confirmed company policy. Annual totals represent
available announcement rows; the CSV has no complete-year dividend announcement marker. A zero
cash dividend does not rule out stock dividends. EPS still uses reported quarterly sums and the
historical EPS/revenue proxy; no net-income or weighted-share dataset has been added.

`FinancialForecastResult` adds `quarterly_eps_estimates`, `payout_history`, and `data_status`
tables with empty defaults for compatibility. Future-year observations do not serve as historical
evidence merely because the requested target year is later than the as-of date.

## Validation

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m unittest discover -s financial_forecast\tests -v
```
