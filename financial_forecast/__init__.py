"""Shared financial-forecast application module.

Forecast systems use this module through one public interface.  Revenue-model training and
cross-system comparison remain owned by their existing packages.
"""

from .contracts import FinancialForecastPolicy, FinancialForecastResult
from .pipeline import forecast_financials

__all__ = ["FinancialForecastPolicy", "FinancialForecastResult", "forecast_financials"]
