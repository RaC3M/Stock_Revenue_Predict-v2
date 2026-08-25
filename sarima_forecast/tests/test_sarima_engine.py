import unittest
from unittest.mock import patch

import pandas as pd

from sarima_forecast import sarima_engine


class SarimaEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2019-01-01", periods=84, freq="MS")
        self.frame = pd.DataFrame(
            {
                "stock_id": [1101] * len(dates),
                "revenue_year": dates.year,
                "revenue_month": dates.month,
                "date": dates,
                "revenue_thousand": [100.0 + index for index in range(len(dates))],
            }
        )

    def test_each_target_uses_only_prior_months(self) -> None:
        observed_history_ends = []

        def fake_forecast(values, order, seasonal_order, confidence_level, maxiter):
            observed_history_ends.append(float(values[-1]))
            return float(values[-1]), float(values[-1] * 0.9), float(values[-1] * 1.1)

        with patch.object(
            sarima_engine,
            "select_sarima_order",
            return_value=((0, 1, 1), (0, 1, 1, 12), pd.DataFrame()),
        ), patch.object(
            sarima_engine,
            "forecast_sarima_one_step",
            side_effect=fake_forecast,
        ):
            result = sarima_engine.build_rolling_sarima_forecast(self.frame, 1101)

        self.assertEqual(observed_history_ends[0], 171.0)
        self.assertEqual(observed_history_ends[1], 172.0)
        self.assertEqual(result.forecast.loc[0, "target_date"], pd.Timestamp("2025-01-01"))
        self.assertEqual(result.forecast.loc[1, "target_date"], pd.Timestamp("2025-02-01"))

    def test_missing_history_uses_labelled_seasonal_fallback(self) -> None:
        short_frame = self.frame[self.frame["date"] >= "2024-01-01"].copy()
        result = sarima_engine.build_rolling_sarima_forecast(short_frame, 1101)

        self.assertIsNone(result.selected_order)
        self.assertTrue(result.forecast["forecast_method"].eq("seasonal_naive_fallback").all())
        self.assertTrue(result.forecast["fallback_reason"].str.len().gt(0).all())


if __name__ == "__main__":
    unittest.main()

