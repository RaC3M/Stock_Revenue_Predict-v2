from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from revenue_adjustment_formula.formula_engine import (
    FormulaConfig,
    build_rolling_predictions,
)


def make_monthly_data(months: int = 84) -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", periods=months, freq="MS")
    trend = np.arange(months, dtype=float) * 8.0
    season = 80.0 * np.sin(np.arange(months) * 2.0 * np.pi / 12.0)
    values = 1000.0 + trend + season
    return pd.DataFrame(
        {
            "stock_id": 9999,
            "revenue_year": dates.year,
            "revenue_month": dates.month,
            "revenue_thousand": values,
            "date": dates,
        }
    )


class FormulaEngineTests(unittest.TestCase):
    def test_future_actual_does_not_change_past_prediction(self) -> None:
        data = make_monthly_data()
        target = pd.Timestamp("2024-06-01")
        original = build_rolling_predictions(data, FormulaConfig())

        changed = data.copy()
        changed.loc[changed["date"] > target, "revenue_thousand"] *= 100.0
        replay = build_rolling_predictions(changed, FormulaConfig())

        first = original.loc[
            original["target_date"] == target, "formula_adjusted_revenue"
        ].iloc[0]
        second = replay.loc[
            replay["target_date"] == target, "formula_adjusted_revenue"
        ].iloc[0]
        self.assertAlmostEqual(first, second)

    def test_zero_residual_strength_equals_formula_base(self) -> None:
        predictions = build_rolling_predictions(
            make_monthly_data(), FormulaConfig(residual_strength=0.0)
        )
        formula_rows = predictions[
            predictions["forecast_method"] == "revenue_adjustment_formula"
        ]
        np.testing.assert_allclose(
            formula_rows["formula_adjusted_revenue"],
            formula_rows["formula_base_revenue"],
        )

    def test_predictions_are_finite_and_nonnegative(self) -> None:
        predictions = build_rolling_predictions(make_monthly_data(), FormulaConfig())
        values = predictions.loc[
            predictions["forecast_method"] == "revenue_adjustment_formula",
            "formula_adjusted_revenue",
        ].to_numpy()
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values >= 0).all())

    def test_gap_starts_a_new_history_segment(self) -> None:
        data = make_monthly_data().drop(index=40).reset_index(drop=True)
        predictions = build_rolling_predictions(data, FormulaConfig())
        first_after_gap = predictions.loc[
            predictions["target_date"] == pd.Timestamp("2022-06-01")
        ].iloc[0]
        self.assertEqual(first_after_gap["forecast_method"], "missing_prediction")


if __name__ == "__main__":
    unittest.main()
