from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


EPS_METHOD_CURRENT_RATIO = "current_ratio"
EPS_METHOD_SEASONAL_QUARTER_MEDIAN = "seasonal_quarter_median"
EPS_METHOD_KNOWN_QUARTERS = "known_quarters_plus_seasonal"
DIVIDEND_METHOD_FIVE_YEAR_MEAN = "five_year_mean_payout"
DIVIDEND_METHOD_CLASSIFIED = "historical_pattern_dividend"
DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_PAYOUT = "announcement_safe_payout_ratio"
DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_LAST = "announcement_safe_last_cash_dividend"
DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_MEDIAN = "announcement_safe_cash_dividend_median"
DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_SMOOTHED = "announcement_safe_smoothed_cash_dividend"
YIELD_MODE_AS_OF_PRICE = "as_of_price_yield"
YIELD_MODE_TARGET_MONTH_END = "target_month_end_yield"


@dataclass(frozen=True)
class FinancialForecastPolicy:
    eps_methods: tuple[str, ...] = (EPS_METHOD_CURRENT_RATIO,)
    dividend_methods: tuple[str, ...] = (DIVIDEND_METHOD_ANNOUNCEMENT_SAFE_PAYOUT,)
    yield_modes: tuple[str, ...] = (YIELD_MODE_AS_OF_PRICE, YIELD_MODE_TARGET_MONTH_END)
    min_stock_price: float = 1.0


@dataclass
class FinancialForecastResult:
    eps_estimates: pd.DataFrame
    dividend_estimates: pd.DataFrame
    yield_estimates: pd.DataFrame
    summary: pd.DataFrame
    failures: pd.DataFrame
    notes: list[str]
    quarterly_eps_estimates: pd.DataFrame = field(default_factory=pd.DataFrame)
    payout_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_status: pd.DataFrame = field(default_factory=pd.DataFrame)
