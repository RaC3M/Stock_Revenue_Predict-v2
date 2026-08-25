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

## Validation

```powershell
.\ensemble_forecast\.venv\Scripts\python.exe -m unittest discover -s financial_forecast\tests -v
```
