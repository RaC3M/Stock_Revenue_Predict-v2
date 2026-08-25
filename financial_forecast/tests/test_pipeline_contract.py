from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from financial_forecast import FinancialForecastPolicy, forecast_financials


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


class FinancialForecastContractTests(unittest.TestCase):
    def test_forecasts_eps_dividend_and_distinct_as_of_and_evaluation_yields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            revenue_rows: list[dict[str, object]] = []
            eps_rows: list[dict[str, object]] = []
            dividend_rows: list[dict[str, object]] = []
            price_rows: list[dict[str, object]] = [
                {"stock_id": 1101, "date": "2024-12-31", "close": 80.0}
            ]
            for year in [2021, 2022, 2023, 2024]:
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
                            "EPS": 3.0 if year < 2024 else 100.0,
                            "statement_available_date": (
                                f"{year + 1}-03-31"
                                if quarter == 4
                                else f"{year}-{[5, 8, 11][quarter - 1]:02d}-14"
                            ),
                        }
                    )
            for year in [2021, 2022, 2023]:
                dividend_rows.append(
                    {
                        "stock_id": 1101,
                        "fiscal_year": year,
                        "TotalCashDividend": 6.0,
                        "DividendAvailableDate": f"{year + 1}-06-01",
                        "CashExDividendTradingDate": f"{year + 1}-07-01",
                    }
                )
            dividend_rows.append(
                {
                    "stock_id": 1101,
                    "fiscal_year": 2024,
                    "TotalCashDividend": 99.0,
                    "DividendAvailableDate": "2025-06-01",
                    "CashExDividendTradingDate": "2025-07-01",
                }
            )
            for month in range(1, 13):
                price_rows.append(
                    {"stock_id": 1101, "date": f"2025-{month:02d}-28", "close": 100.0}
                )

            _write_csv(data_dir / "Stock_revenue_2019~2025.csv", revenue_rows)
            _write_csv(data_dir / "EPS2020~2025.csv", eps_rows)
            _write_csv(data_dir / "Dividend2019~2025.csv", dividend_rows)
            _write_csv(data_dir / "day K2020~2025.csv", price_rows)

            predictions = pd.DataFrame(
                {
                    "source_family": "rolling_lstm",
                    "model": "Rolling adjusted",
                    "stock_id": 1101,
                    "target_year": 2025,
                    "target_month": range(1, 13),
                    "predicted_revenue": [200.0] * 12,
                }
            )
            result = forecast_financials(
                predictions,
                target_year=2025,
                as_of_date="2025-01-10",
                data_dir=data_dir,
                policy=FinancialForecastPolicy(
                    eps_methods=("current_ratio",),
                    dividend_methods=("announcement_safe_payout_ratio",),
                    yield_modes=("as_of_price_yield", "target_month_end_yield"),
                ),
            )

            eps = result.eps_estimates.iloc[0]
            dividend = result.dividend_estimates.iloc[0]
            as_of_yield = result.yield_estimates[
                result.yield_estimates["yield_mode"].eq("as_of_price_yield")
            ].iloc[0]
            evaluation_yield = result.yield_estimates[
                result.yield_estimates["yield_mode"].eq("target_month_end_yield")
            ]

            self.assertEqual(eps["status"], "ok")
            self.assertEqual(int(eps["predicted_annual_revenue"]), 2400)
            self.assertEqual(int(eps["eps_reference_year"]), 2023)
            self.assertAlmostEqual(float(eps["estimated_eps"]), 24.0)
            self.assertAlmostEqual(float(dividend["payout_ratio"]), 0.5)
            self.assertAlmostEqual(float(dividend["estimated_cash_dividend"]), 12.0)
            self.assertAlmostEqual(float(dividend["actual_cash_dividend"]), 99.0)
            self.assertEqual(as_of_yield["price_date"], pd.Timestamp("2024-12-31"))
            self.assertAlmostEqual(float(as_of_yield["estimated_yield_percent"]), 15.0)
            self.assertEqual(len(evaluation_yield), 12)
            self.assertTrue((evaluation_yield["stock_price"] == 100.0).all())
            self.assertTrue((evaluation_yield["estimated_yield_percent"] == 12.0).all())
            self.assertTrue((evaluation_yield["is_evaluation"] == True).all())
            self.assertFalse(bool(as_of_yield["is_evaluation"]))
            self.assertFalse(result.failures.shape[0])

    def test_eps_method_matrix_keeps_annual_and_seasonal_estimates_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            revenue_rows: list[dict[str, object]] = []
            eps_rows: list[dict[str, object]] = []
            quarterly_revenue = [300.0, 600.0, 900.0, 1_200.0]
            quarterly_eps = [3.0, 12.0, 27.0, 48.0]
            for year in [2021, 2022, 2023]:
                for quarter, (quarter_revenue, quarter_eps) in enumerate(
                    zip(quarterly_revenue, quarterly_eps, strict=True),
                    start=1,
                ):
                    for month in range((quarter - 1) * 3 + 1, quarter * 3 + 1):
                        revenue_rows.append(
                            {
                                "stock_id": 1101,
                                "revenue_year": year,
                                "revenue_month": month,
                                "revenue_thousand": quarter_revenue / 3,
                            }
                        )
                    eps_rows.append(
                        {
                            "stock_id": 1101,
                            "date": f"{year}-{quarter * 3:02d}-28",
                            "EPS": quarter_eps,
                            "statement_available_date": (
                                f"{year + 1}-03-31"
                                if quarter == 4
                                else f"{year}-{[5, 8, 11][quarter - 1]:02d}-14"
                            ),
                        }
                    )

            _write_csv(data_dir / "Stock_revenue_2019~2025.csv", revenue_rows)
            _write_csv(data_dir / "EPS2020~2025.csv", eps_rows)
            _write_csv(
                data_dir / "Dividend2019~2025.csv",
                [
                    {
                        "stock_id": 1101,
                        "fiscal_year": 2023,
                        "TotalCashDividend": 45.0,
                        "DividendAvailableDate": "2024-06-01",
                        "CashExDividendTradingDate": "2024-07-01",
                    }
                ],
            )
            _write_csv(
                data_dir / "day K2020~2025.csv",
                [{"stock_id": 1101, "date": "2024-12-31", "close": 100.0}],
            )
            predictions = pd.DataFrame(
                {
                    "source_family": "rolling_lstm",
                    "model": "Rolling adjusted",
                    "stock_id": 1101,
                    "target_year": 2025,
                    "target_month": range(1, 13),
                    "predicted_revenue": [100.0] * 12,
                }
            )

            result = forecast_financials(
                predictions,
                target_year=2025,
                as_of_date="2025-01-10",
                data_dir=data_dir,
                policy=FinancialForecastPolicy(
                    eps_methods=("current_ratio", "seasonal_quarter_median"),
                    dividend_methods=("announcement_safe_payout_ratio",),
                    yield_modes=("as_of_price_yield",),
                ),
            )

            estimates = result.eps_estimates.set_index("eps_method")["estimated_eps"]
            self.assertAlmostEqual(float(estimates["current_ratio"]), 36.0)
            self.assertAlmostEqual(float(estimates["seasonal_quarter_median"]), 30.0)
            self.assertEqual(len(result.dividend_estimates), 2)

    def test_dividend_method_matrix_separates_payout_and_cash_history_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            revenue_rows: list[dict[str, object]] = []
            eps_rows: list[dict[str, object]] = []
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
                            "EPS": 2.0,
                            "statement_available_date": (
                                f"{year + 1}-03-31"
                                if quarter == 4
                                else f"{year}-{[5, 8, 11][quarter - 1]:02d}-14"
                            ),
                        }
                    )
            dividends = []
            for year, cash in [(2021, 2.0), (2022, 4.0), (2023, 10.0)]:
                dividends.append(
                    {
                        "stock_id": 1101,
                        "fiscal_year": year,
                        "TotalCashDividend": cash,
                        "DividendAvailableDate": f"{year + 1}-06-01",
                        "CashExDividendTradingDate": f"{year + 1}-07-01",
                    }
                )
            _write_csv(data_dir / "Stock_revenue_2019~2025.csv", revenue_rows)
            _write_csv(data_dir / "EPS2020~2025.csv", eps_rows)
            _write_csv(data_dir / "Dividend2019~2025.csv", dividends)
            _write_csv(
                data_dir / "day K2020~2025.csv",
                [{"stock_id": 1101, "date": "2024-12-31", "close": 100.0}],
            )
            predictions = pd.DataFrame(
                {
                    "source_family": "ensemble_forecast",
                    "model": "LightGBM",
                    "stock_id": 1101,
                    "target_year": 2025,
                    "target_month": range(1, 13),
                    "predicted_revenue": [100.0] * 12,
                }
            )

            result = forecast_financials(
                predictions,
                target_year=2025,
                as_of_date="2025-01-10",
                data_dir=data_dir,
                policy=FinancialForecastPolicy(
                    eps_methods=("current_ratio",),
                    dividend_methods=(
                        "announcement_safe_payout_ratio",
                        "announcement_safe_last_cash_dividend",
                        "announcement_safe_cash_dividend_median",
                        "announcement_safe_smoothed_cash_dividend",
                    ),
                    yield_modes=("as_of_price_yield",),
                ),
            )

            estimates = result.dividend_estimates.set_index("dividend_method")[
                "estimated_cash_dividend"
            ]
            self.assertAlmostEqual(float(estimates["announcement_safe_last_cash_dividend"]), 10.0)
            self.assertAlmostEqual(
                float(estimates["announcement_safe_cash_dividend_median"]), 4.0
            )
            self.assertAlmostEqual(
                float(estimates["announcement_safe_smoothed_cash_dividend"]), 6.6
            )
            self.assertEqual(len(result.yield_estimates), 4)


if __name__ == "__main__":
    unittest.main()
