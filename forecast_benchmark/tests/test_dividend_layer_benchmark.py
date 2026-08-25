from __future__ import annotations

import unittest

import pandas as pd

from forecast_benchmark.dividend_layer_benchmark import (
    build_annual_cash_dividend,
    build_dividend_layer_overall_accuracy,
    build_dividend_layer_stock_accuracy,
    build_dividend_layer_winner_summary,
    build_leakage_diagnostic,
    estimate_cash_dividend,
    estimate_recent_cash_dividend,
    select_cash_history,
)


class DividendLayerBenchmarkTests(unittest.TestCase):
    def test_time_safe_cash_history_excludes_target_year_ex_dividend(self) -> None:
        annual_cash = build_annual_cash_dividend(
            pd.DataFrame(
                {
                    "stock_id": [1, 1, 1],
                    "fiscal_year": [2022, 2023, 2024],
                    "TotalCashDividend": [1.0, 2.0, 9.0],
                    "ex_dividend_year": [2023, 2024, 2025],
                }
            ),
            target_year=2025,
        )

        current_system = select_cash_history(annual_cash, 1, 2025, mode="current_system")
        time_safe = select_cash_history(annual_cash, 1, 2025, mode="time_safe")

        self.assertEqual(current_system["fiscal_year"].tolist(), [2022, 2023, 2024])
        self.assertEqual(time_safe["fiscal_year"].tolist(), [2022, 2023])
        self.assertTrue(bool(current_system["uses_target_year_ex_dividend"].iloc[-1]))
        self.assertFalse(bool(time_safe["uses_target_year_ex_dividend"].any()))

    def test_announcement_safe_cash_history_excludes_post_as_of_announcements(self) -> None:
        annual_cash = build_annual_cash_dividend(
            pd.DataFrame(
                {
                    "stock_id": [1, 1],
                    "fiscal_year": [2023, 2024],
                    "TotalCashDividend": [2.0, 9.0],
                    "ex_dividend_year": [2024, 2025],
                    "CashExDividendTradingDate": ["2024-07-01", "2025-07-01"],
                    "AnnouncementDate": ["2024-06-01", "2025-06-01"],
                }
            ),
            target_year=2025,
        )
        annual_eps = pd.DataFrame(
            {
                "stock_id": [1, 1],
                "eps_year": [2023, 2024],
                "annual_eps": [4.0, 10.0],
                "eps_quarter_count": [4, 4],
            }
        )

        current_system = select_cash_history(annual_cash, 1, 2025, mode="current_system")
        announcement_safe = select_cash_history(
            annual_cash,
            1,
            2025,
            mode="announcement_safe",
            as_of_date=pd.Timestamp(2025, 1, 10),
        )
        estimate = estimate_cash_dividend(
            annual_cash,
            annual_eps,
            1,
            2025,
            estimated_eps=8.0,
            dividend_method="announcement_safe_payout_ratio",
            as_of_date=pd.Timestamp(2025, 1, 10),
        )

        self.assertEqual(current_system["fiscal_year"].tolist(), [2023, 2024])
        self.assertEqual(announcement_safe["fiscal_year"].tolist(), [2023])
        self.assertAlmostEqual(float(estimate["payout_ratio"]), 0.5)
        self.assertAlmostEqual(float(estimate["estimated_cash_dividend"]), 4.0)
        self.assertFalse(bool(estimate["uses_post_as_of_dividend"]))

    def test_recent_cash_dividend_methods_use_time_safe_history(self) -> None:
        annual_cash = build_annual_cash_dividend(
            pd.DataFrame(
                {
                    "stock_id": [1, 1, 1, 1],
                    "fiscal_year": [2021, 2022, 2023, 2024],
                    "TotalCashDividend": [3.0, 1.0, 2.0, 9.0],
                    "ex_dividend_year": [2022, 2023, 2024, 2025],
                }
            ),
            target_year=2025,
        )

        last = estimate_recent_cash_dividend(annual_cash, 1, 2025, "last_cash_dividend")
        median = estimate_recent_cash_dividend(annual_cash, 1, 2025, "recent_cash_dividend_median")
        smoothed = estimate_recent_cash_dividend(annual_cash, 1, 2025, "smoothed_cash_dividend")

        self.assertAlmostEqual(float(last["estimated_cash_dividend"]), 2.0)
        self.assertAlmostEqual(float(median["estimated_cash_dividend"]), 2.0)
        self.assertAlmostEqual(float(smoothed["estimated_cash_dividend"]), 1.9)
        self.assertEqual(int(last["dividend_reference_year"]), 2023)

    def test_stock_and_overall_accuracy_keep_dividend_method_dimension(self) -> None:
        predictions = pd.DataFrame(
            {
                "source_family": ["a", "a", "a", "a"],
                "model": ["M", "M", "M", "M"],
                "eps_method": ["current_ratio", "current_ratio", "current_ratio", "current_ratio"],
                "dividend_method": [
                    "current_system_payout_ratio",
                    "current_system_payout_ratio",
                    "time_safe_payout_ratio",
                    "time_safe_payout_ratio",
                ],
                "stock_id": [1, 1, 1, 1],
                "stock_name": ["S", "S", "S", "S"],
                "industry_category": ["I", "I", "I", "I"],
                "predicted_annual_revenue": [1000.0, 1000.0, 1000.0, 1000.0],
                "actual_annual_revenue": [900.0, 900.0, 900.0, 900.0],
                "annual_revenue_abs_percent_error": [11.1111, 11.1111, 11.1111, 11.1111],
                "estimated_eps": [2.0, 2.0, 2.0, 2.0],
                "actual_annual_eps": [1.0, 1.0, 1.0, 1.0],
                "eps_abs_error": [1.0, 1.0, 1.0, 1.0],
                "estimated_cash_dividend": [1.0, 1.0, 0.6, 0.6],
                "actual_cash_dividend_per_share": [0.5, 0.5, 0.5, 0.5],
                "cash_dividend_abs_error": [0.5, 0.5, 0.1, 0.1],
                "payout_ratio": [0.5, 0.5, 0.3, 0.3],
                "dividend_reference_year": [2024, 2024, 2023, 2023],
                "dividend_history_count": [3, 3, 2, 2],
                "uses_target_year_ex_dividend": [True, True, False, False],
                "predicted_dividend_yield_percent": [2.0, 4.0, 1.2, 2.4],
                "actual_dividend_yield_percent": [1.0, 2.0, 1.0, 2.0],
                "yield_error_percent_point": [1.0, 2.0, 0.2, 0.4],
            }
        )

        stock_accuracy = build_dividend_layer_stock_accuracy(predictions)
        overall = build_dividend_layer_overall_accuracy(stock_accuracy)
        winners = build_dividend_layer_winner_summary(stock_accuracy)
        leakage = build_leakage_diagnostic(stock_accuracy)

        self.assertEqual(set(overall["dividend_method"]), {"current_system_payout_ratio", "time_safe_payout_ratio"})
        time_safe = overall[overall["dividend_method"].eq("time_safe_payout_ratio")].iloc[0]
        self.assertAlmostEqual(float(time_safe["average_yield_mae_percent_point"]), 0.3)
        self.assertEqual(winners.iloc[0]["dividend_method"], "time_safe_payout_ratio")
        self.assertEqual(int(leakage.iloc[0]["uses_target_year_ex_dividend_stock_count"]), 1)


if __name__ == "__main__":
    unittest.main()
