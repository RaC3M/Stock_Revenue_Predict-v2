"""Shared financial-forecast application module.

Forecast systems use this module through one public interface.  Revenue-model training and
cross-system comparison remain owned by their existing packages.
"""

from .contracts import FinancialForecastPolicy, FinancialForecastResult
from .pipeline import forecast_financial_components, forecast_financials
from .yield_calc import calculate_as_of_yields

__all__ = [
    "FinancialForecastPolicy",
    "FinancialForecastResult",
    "calculate_as_of_yields",
    "forecast_financial_components",
    "forecast_financials",
]
