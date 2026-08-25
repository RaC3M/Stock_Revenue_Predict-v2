import unittest

import pandas as pd

from rolling_predict_LSTM import batch_xlstm_adjustment_ablation as ablation


class XLSTMAdjustmentAblationContractTests(unittest.TestCase):
    def test_parse_float_csv_uses_default_for_empty_value(self) -> None:
        self.assertEqual(ablation.parse_float_csv(None, (0.1, 0.2)), (0.1, 0.2))
        self.assertEqual(ablation.parse_float_csv("0, 0.8,1.2"), (0.0, 0.8, 1.2))

    def test_parse_bool_csv_accepts_common_tokens(self) -> None:
        self.assertEqual(ablation.parse_bool_csv("true,off,1"), (True, False, True))

        with self.assertRaises(ValueError):
            ablation.parse_bool_csv("maybe")

    def test_build_adjustment_specs_includes_plain_once(self) -> None:
        specs = ablation.build_adjustment_specs(
            alphas=(0.0, 0.8),
            conditional_options=(True,),
            regime_options=(True,),
        )

        self.assertEqual(specs[0].name, "plain")
        self.assertEqual(sum(spec.name == "plain" for spec in specs), 1)
        self.assertEqual(
            [spec.name for spec in specs[1:]],
            [
                "decline_cap_only",
                "decline_cap_balanced",
                "growth_boost_only_alpha_0p8_cond_on_regime_on",
                "growth_boost_and_decline_cap_alpha_0p8_cond_on_regime_on",
            ],
        )
        self.assertEqual(
            [spec.effect_component for spec in specs],
            [
                "plain",
                "decline_cap_only",
                "decline_cap_balanced",
                "growth_boost_only",
                "growth_boost_and_decline_cap",
            ],
        )
        balanced = [spec for spec in specs if spec.name == "decline_cap_balanced"][0]
        self.assertEqual(balanced.decline_cap_growth_ratio_max, ablation.DEFAULT_BALANCED_DECLINE_CAP_GROWTH_RATIO_MAX)
        self.assertEqual(
            balanced.decline_cap_prediction_ratio_min,
            ablation.DEFAULT_BALANCED_DECLINE_CAP_PREDICTION_RATIO_MIN,
        )

    def test_apply_adjustment_variant_uses_time_safe_growth_gate(self) -> None:
        frame = pd.DataFrame(
            {
                "stock_id": [1, 1, 1],
                "target_year": [2025, 2025, 2025],
                "target_month": [1, 2, 3],
                "actual_revenue": [130.0, 90.0, 100.0],
                "predicted_revenue_xlstm": [100.0, 100.0, 120.0],
                "last_observed_revenue": [100.0, 100.0, 100.0],
                "sequence_max_revenue": [110.0, 110.0, 120.0],
                "growth_rate_at_end": [0.2, -0.2, -0.2],
                "momentum_3m_at_end": [0.2, 0.2, -0.1],
                "momentum_6m_at_end": [0.2, 0.2, -0.1],
                "growth_ratio": [0.8, 0.8, 0.2],
                "growth_streak": [4, 4, 0],
            }
        )
        boost_only = ablation.AdjustmentSpec(
            name="growth_boost_only_alpha_1_cond_on_regime_on",
            alpha=1.0,
            conditional=True,
            regime_strategy=True,
            enabled=True,
            growth_boost_enabled=True,
            decline_cap_enabled=False,
            effect_component="growth_boost_only",
        )
        cap_only = ablation.AdjustmentSpec(
            name="decline_cap_only",
            alpha=0.0,
            conditional=True,
            regime_strategy=True,
            enabled=True,
            growth_boost_enabled=False,
            decline_cap_enabled=True,
            effect_component="decline_cap_only",
        )
        combined = ablation.AdjustmentSpec(
            name="growth_boost_and_decline_cap_alpha_1_cond_on_regime_on",
            alpha=1.0,
            conditional=True,
            regime_strategy=True,
            enabled=True,
            growth_boost_enabled=True,
            decline_cap_enabled=True,
            effect_component="growth_boost_and_decline_cap",
        )

        boost_result = ablation.apply_adjustment_variant(frame, boost_only)
        cap_result = ablation.apply_adjustment_variant(frame, cap_only)
        combined_result = ablation.apply_adjustment_variant(frame, combined)

        self.assertGreater(float(boost_result.loc[0, "predicted_revenue"]), 100.0)
        self.assertEqual(float(boost_result.loc[2, "predicted_revenue"]), 120.0)
        self.assertTrue(bool(boost_result.loc[0, "adjustment_applied"]))
        self.assertFalse(bool(boost_result.loc[2, "decline_cap_applied"]))

        self.assertEqual(float(cap_result.loc[0, "predicted_revenue"]), 100.0)
        self.assertEqual(float(cap_result.loc[2, "predicted_revenue"]), 100.0)
        self.assertFalse(bool(cap_result.loc[0, "adjustment_applied"]))
        self.assertTrue(bool(cap_result.loc[2, "decline_cap_applied"]))

        self.assertGreater(float(combined_result.loc[0, "predicted_revenue"]), 100.0)
        self.assertEqual(float(combined_result.loc[2, "predicted_revenue"]), 100.0)
        self.assertTrue(bool(combined_result.loc[0, "adjustment_applied"]))
        self.assertTrue(bool(combined_result.loc[2, "decline_cap_applied"]))

    def test_variant_effects_compare_against_plain_per_stock(self) -> None:
        stock_accuracy = pd.DataFrame(
            {
                "variant": ["plain", "plain", "alpha_0p8_cond_on_regime_on", "alpha_0p8_cond_on_regime_on"],
                "stock_id": [1, 2, 1, 2],
                "MAPE": [10.0, 20.0, 8.0, 22.0],
                "WMAPE": [9.0, 18.0, 7.0, 23.0],
                "MAE": [100.0, 200.0, 80.0, 220.0],
                "Bias": [0.0, 0.0, 5.0, -5.0],
                "UnderestimateRate": [50.0, 50.0, 40.0, 60.0],
                "AdjustedRate": [0.0, 0.0, 25.0, 25.0],
                "GrowthBoostRate": [0.0, 0.0, 25.0, 25.0],
                "DeclineCapRate": [0.0, 0.0, 0.0, 0.0],
                "GuardrailClipRate": [0.0, 0.0, 0.0, 0.0],
            }
        )

        effects = ablation.build_variant_effects(stock_accuracy)
        winner_summary = ablation.build_winner_summary(effects)

        row = winner_summary.iloc[0]
        self.assertEqual(int(row["variant_wins"]), 1)
        self.assertEqual(int(row["plain_wins"]), 1)
        self.assertAlmostEqual(float(row["average_MAPE_delta_vs_plain"]), 0.0)
        self.assertEqual(int(row["WMAPE_variant_wins"]), 1)
        self.assertEqual(int(row["WMAPE_plain_wins"]), 1)
        self.assertAlmostEqual(float(row["average_WMAPE_delta_vs_plain"]), 1.5)

    def test_component_best_summary_keeps_best_variant_per_component(self) -> None:
        overall = pd.DataFrame(
            {
                "variant": ["plain", "cap_a", "cap_b", "boost_a"],
                "effect_component": ["plain", "decline_cap_only", "decline_cap_only", "growth_boost_only"],
                "MAPE": [30.0, 20.0, 18.0, 25.0],
                "MAE": [300.0, 200.0, 210.0, 250.0],
            }
        )

        summary = ablation.build_component_best_summary(overall)

        self.assertEqual(summary[summary["effect_component"].eq("decline_cap_only")].iloc[0]["variant"], "cap_b")


if __name__ == "__main__":
    unittest.main()
