"""Single integration seam for Ensemble Forecast data and evidence.

Benchmark modules must import Ensemble capabilities through this adapter. Private
engine helpers are intentionally contained here so their churn cannot spread
through every downstream benchmark.
"""

from __future__ import annotations

import sys
from typing import Any

from forecast_benchmark.benchmark_config import PROJECT_ROOT


def _engine():
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from ensemble_forecast import forecast_engine

    return forecast_engine


def forecast_year() -> int:
    return int(_engine().FORECAST_YEAR)


def build_forecast(stock_id: int):
    return _engine().build_forecast(int(stock_id))


def load_revenue_data():
    return _engine().load_revenue_data()


def load_actual_revenue_data():
    return _engine().load_actual_2025_data()


def load_eps_data():
    return _engine().load_eps_data()


def load_cash_dividend_data():
    return _engine().load_cash_dividend_data()


def load_stock_price_data(*, target_year: int):
    return _engine().load_stock_price_data(target_year=int(target_year))


def get_dividend_policy(stock_id: int) -> dict[str, Any]:
    return _engine()._get_dividend_policy(int(stock_id))


def get_historical_payout_ratio(stock_id: int, target_year: int) -> tuple[float, str]:
    return _engine()._get_historical_payout_ratio(int(stock_id), int(target_year))


def get_actual_cash_dividend_info(stock_id: int, target_year: int) -> dict[str, Any]:
    return _engine()._get_actual_cash_dividend_info(int(stock_id), int(target_year))


def get_forecast_dividend_info(
    stock_id: int,
    target_year: int,
    forecast_annual_revenue_thousand: float,
    revenue_data,
) -> dict[str, Any]:
    return _engine()._get_forecast_dividend_info(
        int(stock_id),
        int(target_year),
        float(forecast_annual_revenue_thousand),
        revenue_data,
    )
