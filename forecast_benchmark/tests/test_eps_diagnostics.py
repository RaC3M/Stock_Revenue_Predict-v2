from __future__ import annotations

import unittest

from forecast_benchmark.eps_diagnostics import (
    classify_current_ratio_driver,
    classify_ratio_stability,
    recommend_eps_path,
)


class EpsDiagnosticsTests(unittest.TestCase):
    def test_ratio_stability_requires_enough_history(self) -> None:
        bucket = classify_ratio_stability(
            ratio_count=2,
            ratio_std_to_median=0.1,
            latest_deviation_from_recent_median_pct=10,
        )

        self.assertEqual(bucket, "insufficient_history")

    def test_ratio_stability_marks_low_variation_as_stable(self) -> None:
        bucket = classify_ratio_stability(
            ratio_count=5,
            ratio_std_to_median=0.2,
            latest_deviation_from_recent_median_pct=20,
        )

        self.assertEqual(bucket, "stable_ratio")

    def test_recommendation_keeps_current_ratio_when_stable_and_close(self) -> None:
        recommendation = recommend_eps_path(
            ratio_stability_bucket="stable_ratio",
            best_current_error=1.0,
            best_seasonal_error=0.95,
            best_ml_error=1.1,
        )

        self.assertEqual(recommendation, "keep_current_ratio")

    def test_recommendation_switches_to_ml_when_materially_better(self) -> None:
        recommendation = recommend_eps_path(
            ratio_stability_bucket="moderate_ratio",
            best_current_error=2.0,
            best_seasonal_error=1.9,
            best_ml_error=1.0,
        )

        self.assertEqual(recommendation, "test_ml_eps_layer")

    def test_current_ratio_driver_identifies_formula_error(self) -> None:
        driver = classify_current_ratio_driver(
            current_ratio_error=4.0,
            oracle_current_ratio_error=3.5,
            annual_revenue_abs_percent_error=5.0,
        )

        self.assertEqual(driver, "eps_ratio_formula_error")

    def test_current_ratio_driver_identifies_revenue_error(self) -> None:
        driver = classify_current_ratio_driver(
            current_ratio_error=4.0,
            oracle_current_ratio_error=1.0,
            annual_revenue_abs_percent_error=35.0,
        )

        self.assertEqual(driver, "revenue_forecast_error")


if __name__ == "__main__":
    unittest.main()
