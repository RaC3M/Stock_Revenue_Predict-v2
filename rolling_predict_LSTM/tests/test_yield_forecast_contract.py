from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from rolling_predict_LSTM.yield_forecast import build_rolling_yield_forecast


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


class RollingYieldForecastContractTests(unittest.TestCase):
    def test_builds_complete_chain_and_excludes_information_after_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            revenue_rows = []
            eps_rows = []
            dividend_rows = []
            price_rows = []
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
                                f"{year + 1}-03-31" if quarter == 4 else f"{year}-{[5, 8, 11][quarter - 1]:02d}-14"
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
            price_rows.append(
                {
                    "stock_id": 1101,
                    "date": "2024-12-31",
                    "close": 80.0,
                }
            )
            for month in range(1, 13):
                price_rows.append(
                    {
                        "stock_id": 1101,
                        "date": f"2025-{month:02d}-28",
                        "close": 100.0,
                    }
                )
            _write_csv(data_dir / "Stock_revenue_2019~2025.csv", revenue_rows)
            _write_csv(data_dir / "EPS2020~2025.csv", eps_rows)
            _write_csv(data_dir / "Dividend2019~2025.csv", dividend_rows)
            _write_csv(data_dir / "day K2020~2025.csv", price_rows)

            forecast = pd.DataFrame(
                {
                    "target_month": range(1, 13),
                    "predicted_revenue_adjusted": [200.0] * 12,
                    "predicted_revenue_xlstm": [pd.NA] * 12,
                }
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                result = build_rolling_yield_forecast(
                    forecast,
                    selected_stock=1101,
                    target_year=2025,
                    model_columns={
                        "Rolling adjusted": "predicted_revenue_adjusted",
                        "Rolling xLSTM": "predicted_revenue_xlstm",
                    },
                    data_dir=data_dir,
                    as_of_date="2025-01-10",
                )

            adjusted = result.summary[result.summary["model"] == "Rolling adjusted"].iloc[0]
            xlstm = result.summary[result.summary["model"] == "Rolling xLSTM"].iloc[0]
            self.assertEqual(adjusted["status"], "ok")
            self.assertEqual(int(adjusted["predicted_annual_revenue"]), 2400)
            self.assertEqual(int(adjusted["eps_reference_year"]), 2023)
            self.assertAlmostEqual(float(adjusted["estimated_eps"]), 24.0)
            self.assertAlmostEqual(float(adjusted["payout_ratio"]), 0.5)
            self.assertAlmostEqual(float(adjusted["estimated_cash_dividend"]), 12.0)
            self.assertAlmostEqual(float(adjusted["latest_predicted_yield_percent"]), 12.0)
            self.assertAlmostEqual(float(adjusted["as_of_price_yield_percent"]), 15.0)
            self.assertEqual(adjusted["as_of_price_date"], pd.Timestamp("2024-12-31"))
            self.assertAlmostEqual(float(adjusted["actual_cash_dividend"]), 99.0)
            self.assertEqual(xlstm["status"], "incomplete monthly predictions (0/12)")
            self.assertEqual(len(result.monthly), 12)
            self.assertTrue(result.monthly["stock_price_source"].str.contains("evaluation").all())

    def test_incomplete_months_are_not_silently_annualized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _write_csv(
                data_dir / "Stock_revenue_2019~2025.csv",
                [
                    {
                        "stock_id": 1101,
                        "revenue_year": 2023,
                        "revenue_month": month,
                        "revenue_thousand": 100.0,
                    }
                    for month in range(1, 13)
                ],
            )
            _write_csv(
                data_dir / "EPS2020~2025.csv",
                [
                    {
                        "stock_id": 1101,
                        "date": f"2023-{month:02d}-28",
                        "EPS": 1.0,
                        "statement_available_date": "2024-03-31",
                    }
                    for month in [3, 6, 9, 12]
                ],
            )
            _write_csv(
                data_dir / "Dividend2019~2025.csv",
                [
                    {
                        "stock_id": 1101,
                        "fiscal_year": 2023,
                        "TotalCashDividend": 2.0,
                        "DividendAvailableDate": "2024-06-01",
                        "CashExDividendTradingDate": "2024-07-01",
                    }
                ],
            )
            _write_csv(
                data_dir / "day K2020~2025.csv",
                [{"stock_id": 1101, "date": "2025-01-31", "close": 50.0}],
            )
            forecast = pd.DataFrame({"target_month": range(1, 12), "model": [100.0] * 11})

            result = build_rolling_yield_forecast(
                forecast,
                selected_stock=1101,
                target_year=2025,
                model_columns={"Model": "model"},
                data_dir=data_dir,
            )

            self.assertTrue(result.monthly.empty)
            self.assertEqual(result.summary.iloc[0]["status"], "incomplete monthly predictions (11/12)")


if __name__ == "__main__":
    unittest.main()
