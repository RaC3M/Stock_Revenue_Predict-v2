from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ensemble_forecast.yield_forecast import build_ensemble_yield_forecast


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


class EnsembleYieldAdapterContractTests(unittest.TestCase):
    def test_adapter_preserves_ensemble_columns_and_exposes_both_yield_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            revenue_rows: list[dict[str, object]] = []
            eps_rows: list[dict[str, object]] = []
            dividends: list[dict[str, object]] = []
            prices: list[dict[str, object]] = [
                {"stock_id": 1101, "date": "2024-12-31", "close": 80.0}
            ]
            for year in [2021, 2022, 2023]:
                for month in range(1, 13):
                    revenue_rows.append(
                        {
                            "stock_id": 1101,
                            "revenue_year": year,
                            "revenue_month": month,
                            "revenue_thousand": 100.0,
                        }
                    )
                for quarter, month in enumerate([3, 6, 9, 12], start=1):
                    eps_rows.append(
                        {
                            "stock_id": 1101,
                            "date": f"{year}-{month:02d}-28",
                            "EPS": 3.0,
                            "statement_available_date": (
                                f"{year + 1}-03-31"
                                if quarter == 4
                                else f"{year}-{[5, 8, 11][quarter - 1]:02d}-14"
                            ),
                        }
                    )
                dividends.append(
                    {
                        "stock_id": 1101,
                        "fiscal_year": year,
                        "TotalCashDividend": 6.0,
                        "DividendAvailableDate": f"{year + 1}-06-01",
                        "CashExDividendTradingDate": f"{year + 1}-07-01",
                    }
                )
            for month in range(1, 13):
                prices.append(
                    {"stock_id": 1101, "date": f"2025-{month:02d}-28", "close": 100.0}
                )

            _write_csv(data_dir / "Stock_revenue_2019~2025.csv", revenue_rows)
            _write_csv(data_dir / "EPS2020~2025.csv", eps_rows)
            _write_csv(data_dir / "Dividend2019~2025.csv", dividends)
            _write_csv(data_dir / "day K2020~2025.csv", prices)
            forecast = pd.DataFrame(
                {"revenue_month": range(1, 13), "predicted_revenue": [200.0] * 12}
            )
            actual = pd.DataFrame(
                {
                    "stock_id": 1101,
                    "revenue_year": 2025,
                    "revenue_month": range(1, 13),
                    "actual_revenue": [180.0] * 12,
                }
            )

            result = build_ensemble_yield_forecast(
                forecast,
                selected_stock=1101,
                target_year=2025,
                model_family="ensemble_forecast",
                model_name="ensemble_revenue",
                data_dir=data_dir,
                actual_revenue=actual,
                as_of_date="2025-01-10",
            )

            self.assertEqual(len(result), 12)
            self.assertTrue((result["predicted_annual_revenue"] == 2400).all())
            self.assertTrue((result["estimated_eps"] == 24.0).all())
            self.assertTrue((result["estimated_cash_dividend"] == 12.0).all())
            self.assertTrue((result["predicted_dividend_yield_percent"] == 12.0).all())
            self.assertTrue((result["as_of_price_yield_percent"] == 15.0).all())
            self.assertTrue((result["as_of_stock_price"] == 80.0).all())
            self.assertTrue((result["actual_revenue"] == 180.0).all())
            self.assertIn("stock_price_source", result.columns)
            self.assertIn("cash_dividend_source", result.columns)


if __name__ == "__main__":
    unittest.main()
