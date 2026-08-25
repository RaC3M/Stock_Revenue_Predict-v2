from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from forecast_benchmark.financial_ablation import run_financial_ablation


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_financial_fixture(data_dir: Path) -> None:
    revenue_rows: list[dict[str, object]] = []
    eps_rows: list[dict[str, object]] = []
    dividend_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []

    for year in range(2020, 2026):
        annual_scale = float(year - 2019)
        for month in range(1, 13):
            for stock_id in (1101, 9999):
                revenue_rows.append(
                    {
                        "stock_id": stock_id,
                        "revenue_year": year,
                        "revenue_month": month,
                        "revenue_thousand": 100.0 * annual_scale,
                    }
                )
                price_rows.append(
                    {
                        "stock_id": stock_id,
                        "date": f"{year}-{month:02d}-28",
                        "close": 50.0 + annual_scale,
                    }
                )
        for quarter, month in enumerate((3, 6, 9, 12), start=1):
            eps_rows.append(
                {
                    "stock_id": 1101,
                    "date": f"{year}-{month:02d}-28",
                    "EPS": annual_scale,
                    "statement_available_date": (
                        f"{year + 1}-03-31"
                        if quarter == 4
                        else f"{year}-{(5, 8, 11)[quarter - 1]:02d}-14"
                    ),
                }
            )
        dividend_rows.append(
            {
                "stock_id": 1101,
                "fiscal_year": year - 1,
                "TotalCashDividend": annual_scale,
                "DividendAvailableDate": f"{year}-06-01",
                "CashExDividendTradingDate": f"{year}-07-01",
            }
        )
        dividend_rows.append(
            {
                "stock_id": 9999,
                "fiscal_year": year - 1,
                "TotalCashDividend": annual_scale,
                "DividendAvailableDate": f"{year}-06-01",
                "CashExDividendTradingDate": f"{year}-07-01",
            }
        )

    _write_csv(data_dir / "Stock_revenue_2019~2025.csv", revenue_rows)
    _write_csv(data_dir / "EPS2020~2025.csv", eps_rows)
    _write_csv(data_dir / "Dividend2019~2025.csv", dividend_rows)
    _write_csv(data_dir / "day K2020~2025.csv", price_rows)


class FinancialAblationTests(unittest.TestCase):
    def test_selects_on_historical_years_and_evaluates_frozen_2025_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            _build_financial_fixture(data_dir)
            frozen_predictions = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "source_family": "rolling_lstm",
                            "model": "Frozen Rolling",
                            "stock_id": stock_id,
                            "target_year": 2025,
                            "target_month": range(1, 13),
                            "predicted_revenue": [625.0] * 12,
                        }
                    )
                    for stock_id in (1101, 9999)
                ],
                ignore_index=True,
            )

            result = run_financial_ablation(
                frozen_predictions,
                data_dir=data_dir,
                target_year=2025,
                validation_years=(2023, 2024),
            )

            self.assertEqual(
                set(result["validation_eps_estimates"]["target_year"]),
                {2023, 2024},
            )
            self.assertEqual(
                set(result["validation_eps_estimates"]["validation_source"]),
                {"actual_revenue_replay"},
            )
            self.assertEqual(int(result["method_selection"].iloc[0]["selection_max_year"]), 2024)
            self.assertLess(
                int(result["method_selection"].iloc[0]["selection_max_year"]),
                2025,
            )
            self.assertIn("selected_eps_method", result["method_selection"].columns)
            self.assertIn("selected_dividend_method", result["method_selection"].columns)
            self.assertEqual(set(result["test_eps_estimates"]["model"]), {"Frozen Rolling"})
            self.assertEqual(set(result["selected_test_estimates"]["model"]), {"Frozen Rolling"})
            self.assertEqual(
                result["selected_test_estimates"]["eps_method"].nunique(),
                1,
            )
            self.assertEqual(
                result["selected_test_estimates"]["dividend_method"].nunique(),
                1,
            )
            self.assertIn("cash_dividend_mae", result["validation_end_to_end_scores"].columns)
            self.assertIn("yield_mae_percent_point", result["validation_end_to_end_scores"].columns)
            self.assertIn("cash_dividend_abs_error", result["selected_test_estimates"].columns)
            self.assertIn("yield_abs_error_percent_point", result["selected_test_estimates"].columns)
            self.assertEqual(
                result["validation_end_to_end_scores"][
                    "cash_dividend_observations"
                ].nunique(),
                1,
            )
            self.assertEqual(
                result["validation_end_to_end_scores"]["yield_observations"].nunique(),
                1,
            )
            self.assertEqual(
                set(result["test_yield_estimates"]["yield_mode"]),
                {"as_of_price_yield", "target_month_end_yield"},
            )


if __name__ == "__main__":
    unittest.main()
