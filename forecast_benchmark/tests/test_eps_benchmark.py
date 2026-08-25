from __future__ import annotations

import unittest

import pandas as pd

from forecast_benchmark.eps_benchmark import (
    build_eps_method_winner_summary,
    build_eps_overall_accuracy,
    build_eps_stock_accuracy,
    build_historical_annual_frame,
    build_historical_quarter_frame,
    estimate_current_ratio_eps,
    estimate_seasonal_quarter_eps,
)


def _annual_revenue_rows(stock_id: int, year: int, annual_revenue: float) -> list[dict[str, float]]:
    monthly_revenue = annual_revenue / 12
    return [
        {
            "stock_id": stock_id,
            "revenue_year": year,
            "revenue_month": month,
            "revenue_thousand": monthly_revenue,
        }
        for month in range(1, 13)
    ]


def _annual_eps_rows(stock_id: int, year: int, annual_eps: float) -> list[dict[str, float]]:
    quarter_eps = annual_eps / 4
    return [
        {
            "stock_id": stock_id,
            "eps_year": year,
            "eps_quarter": quarter,
            "latest_eps": quarter_eps,
        }
        for quarter in range(1, 5)
    ]


class EpsBenchmarkTests(unittest.TestCase):
    def test_current_ratio_uses_recent_three_year_median(self) -> None:
        revenue = pd.DataFrame(
            _annual_revenue_rows(1, 2020, 1000.0)
            + _annual_revenue_rows(1, 2021, 1000.0)
            + _annual_revenue_rows(1, 2022, 1000.0)
            + _annual_revenue_rows(1, 2023, 1000.0)
            + _annual_revenue_rows(1, 2024, 1000.0)
        )
        eps = pd.DataFrame(
            _annual_eps_rows(1, 2020, 10.0)
            + _annual_eps_rows(1, 2021, 20.0)
            + _annual_eps_rows(1, 2022, 30.0)
            + _annual_eps_rows(1, 2023, 40.0)
            + _annual_eps_rows(1, 2024, 50.0)
        )
        history = build_historical_annual_frame(revenue, eps, target_year=2025)

        estimate = estimate_current_ratio_eps(1, 2000.0, history)

        self.assertAlmostEqual(float(estimate["eps_to_revenue_ratio"]), 0.04)
        self.assertAlmostEqual(float(estimate["estimated_eps"]), 80.0)
        self.assertEqual(estimate["eps_reference_year"], 2024)

    def test_seasonal_quarter_eps_uses_same_quarter_ratios(self) -> None:
        revenue = []
        for month in range(1, 13):
            revenue.append(
                {
                    "stock_id": 1,
                    "revenue_year": 2024,
                    "revenue_month": month,
                    "revenue_thousand": 100.0,
                }
            )
        eps = pd.DataFrame(
            [
                {"stock_id": 1, "eps_year": 2024, "eps_quarter": 1, "latest_eps": 3.0},
                {"stock_id": 1, "eps_year": 2024, "eps_quarter": 2, "latest_eps": 6.0},
                {"stock_id": 1, "eps_year": 2024, "eps_quarter": 3, "latest_eps": 9.0},
                {"stock_id": 1, "eps_year": 2024, "eps_quarter": 4, "latest_eps": 12.0},
            ]
        )
        quarter_history = build_historical_quarter_frame(pd.DataFrame(revenue), eps, target_year=2025)
        quarter_predictions = pd.DataFrame(
            {
                "target_quarter": [1, 2, 3, 4],
                "predicted_quarter_revenue": [300.0, 300.0, 300.0, 300.0],
            }
        )

        estimate = estimate_seasonal_quarter_eps(
            1,
            quarter_predictions,
            quarter_history,
            {"eps_to_revenue_ratio": 0.0, "eps_reference_year": 2024},
        )

        self.assertAlmostEqual(float(estimate["estimated_eps"]), 30.0)
        self.assertAlmostEqual(float(estimate["eps_to_revenue_ratio"]), 0.025)

    def test_winner_summary_excludes_oracle_rows(self) -> None:
        predictions = pd.DataFrame(
            {
                "source_family": ["a", "a", "oracle"],
                "model": ["A", "A", "actual_revenue"],
                "eps_method": ["current_ratio", "ridge_annual", "oracle_current_ratio"],
                "is_oracle": [False, False, True],
                "stock_id": [1, 1, 1],
                "stock_name": ["S", "S", "S"],
                "industry_category": ["I", "I", "I"],
                "monthly_observations": [12, 12, 12],
                "predicted_annual_revenue": [1000.0, 1000.0, 900.0],
                "actual_annual_revenue": [900.0, 900.0, 900.0],
                "annual_revenue_abs_percent_error": [11.1111, 11.1111, 0.0],
                "estimated_eps": [2.0, 1.1, 1.0],
                "actual_annual_eps": [1.0, 1.0, 1.0],
                "actual_eps_quarter_count": [4, 4, 4],
                "eps_error": [1.0, 0.1, 0.0],
                "eps_abs_error": [1.0, 0.1, 0.0],
                "eps_abs_percent_error": [100.0, 10.0, 0.0],
                "eps_underestimated": [False, False, False],
                "eps_reference_year": [2024, 2024, 2024],
                "eps_to_revenue_ratio": [0.002, 0.0011, 0.0011],
                "eps_transform_source": ["ratio", "ridge", "oracle"],
            }
        )

        stock_accuracy = build_eps_stock_accuracy(predictions)
        overall = build_eps_overall_accuracy(stock_accuracy)
        winners = build_eps_method_winner_summary(stock_accuracy)

        self.assertIn(True, set(overall["is_oracle"]))
        self.assertEqual(winners.iloc[0]["eps_method"], "ridge_annual")
        self.assertNotIn("oracle_current_ratio", set(winners["eps_method"]))


if __name__ == "__main__":
    unittest.main()
