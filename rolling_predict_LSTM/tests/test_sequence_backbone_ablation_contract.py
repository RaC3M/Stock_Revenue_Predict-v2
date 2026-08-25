import unittest

import pandas as pd

from rolling_predict_LSTM import batch_sequence_backbone_ablation as backbone_ablation


class SequenceBackboneAblationContractTests(unittest.TestCase):
    def test_default_backbones_compare_lstm_and_xlstm(self) -> None:
        self.assertEqual(backbone_ablation.DEFAULT_BACKBONES, ("lstm", "xlstm"))
        self.assertEqual(backbone_ablation.DEFAULT_STOCK_IDS, (1101, 1231, 3017))

    def test_parse_backbones_rejects_unknown_values(self) -> None:
        self.assertEqual(backbone_ablation.parse_backbones("lstm,xlstm"), ("lstm", "xlstm"))

        with self.assertRaises(ValueError):
            backbone_ablation.parse_backbones("lstm,transformer")

    def test_parse_backbones_accepts_hybrid_xlstm(self) -> None:
        self.assertEqual(
            backbone_ablation.parse_backbones("xlstm,xlstm_hybrid"),
            ("xlstm", "xlstm_hybrid"),
        )

    def test_effect_backbones_follow_requested_two_backbones(self) -> None:
        self.assertEqual(
            backbone_ablation.resolve_effect_backbones(("xlstm", "xlstm_hybrid")),
            ("xlstm", "xlstm_hybrid"),
        )

    def test_hybrid_run_config_records_architecture(self) -> None:
        config = backbone_ablation.build_run_config(
            backbone_ablation.BackboneRunSpec(backbone="xlstm_hybrid")
        )

        self.assertEqual(config.sequence_backbone, "xlstm_hybrid")
        self.assertEqual(config.xlstm_backbone, "xlstm_hybrid")

    def test_parse_int_csv_uses_default_for_empty_value(self) -> None:
        self.assertEqual(backbone_ablation.parse_int_csv(None, (1101,)), (1101,))
        self.assertEqual(backbone_ablation.parse_int_csv("1101, 3017", (1231,)), (1101, 3017))

    def test_auto_stock_selection_is_deterministic_and_limited(self) -> None:
        candidate_pool = pd.DataFrame(
            {
                "stock_id": [1, 2, 3, 4, 5],
                "industry_category": ["A", "A", "B", "B", "C"],
                "available_months_2025": [12, 12, 12, 12, 12],
            }
        )

        first = backbone_ablation.select_auto_stock_ids(candidate_pool, stock_limit=3, sample_seed=7)
        second = backbone_ablation.select_auto_stock_ids(candidate_pool, stock_limit=3, sample_seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_candidate_pool_requires_minimum_2025_months(self) -> None:
        revenue = pd.DataFrame(
            {
                "stock_id": [1, 1, 2, 2],
                "revenue_year": [2024, 2025, 2024, 2025],
                "revenue_thousand": [100, 110, 200, 210],
            }
        )
        stock_meta = pd.DataFrame(
            {
                "stock_id": [1, 2],
                "stock_name": ["A", "B"],
                "industry_category": ["X", "Y"],
                "available_months_2025": [12, 3],
            }
        )

        pool = backbone_ablation.build_candidate_stock_pool(revenue, stock_meta, min_2025_months=12)

        self.assertEqual(pool["stock_id"].tolist(), [1])

    def test_backbone_effects_define_xlstm_minus_lstm_delta(self) -> None:
        stock_accuracy = pd.DataFrame(
            {
                "sequence_backbone": ["lstm", "xlstm"],
                "stock_id": [1101, 1101],
                "stock_name": ["A", "A"],
                "industry_category": ["cement", "cement"],
                "model": ["Rolling LSTM", "Rolling LSTM"],
                "MAPE": [12.0, 10.0],
                "WMAPE": [11.0, 9.0],
                "MAE": [120.0, 90.0],
                "Bias": [5.0, -2.0],
                "UnderestimateRate": [50.0, 40.0],
                "runtime_seconds": [1.0, 3.0],
            }
        )

        effects = backbone_ablation.build_backbone_effects(stock_accuracy)

        self.assertEqual(float(effects.loc[0, "MAPE_delta_xlstm_minus_lstm"]), -2.0)
        self.assertEqual(float(effects.loc[0, "WMAPE_delta_xlstm_minus_lstm"]), -2.0)
        self.assertEqual(float(effects.loc[0, "MAE_delta_xlstm_minus_lstm"]), -30.0)
        self.assertEqual(effects.loc[0, "MAPE_winner"], "xlstm")

    def test_backbone_effects_can_compare_historical_and_hybrid_xlstm(self) -> None:
        stock_accuracy = pd.DataFrame(
            {
                "sequence_backbone": ["xlstm", "xlstm_hybrid"],
                "stock_id": [1101, 1101],
                "stock_name": ["A", "A"],
                "industry_category": ["cement", "cement"],
                "model": ["Rolling LSTM", "Rolling LSTM"],
                "MAPE": [12.0, 9.0],
                "MAE": [120.0, 90.0],
            }
        )

        effects = backbone_ablation.build_backbone_effects(
            stock_accuracy,
            baseline_backbone="xlstm",
            challenger_backbone="xlstm_hybrid",
        )

        self.assertEqual(
            float(effects.loc[0, "MAPE_delta_xlstm_hybrid_minus_xlstm"]),
            -3.0,
        )
        self.assertEqual(effects.loc[0, "MAPE_winner"], "xlstm_hybrid")

    def test_winner_summary_counts_backbone_wins_by_model(self) -> None:
        effects = pd.DataFrame(
            {
                "stock_id": [1, 2, 3],
                "model": ["Rolling LSTM", "Rolling LSTM", "Rolling LSTM"],
                "MAPE_winner": ["xlstm", "lstm", "xlstm"],
                "MAPE_delta_xlstm_minus_lstm": [-1.0, 2.0, -3.0],
                "MAE_delta_xlstm_minus_lstm": [-10.0, 20.0, -30.0],
            }
        )

        summary = backbone_ablation.build_winner_summary(effects)

        self.assertEqual(int(summary.loc[0, "xlstm_wins"]), 2)
        self.assertEqual(int(summary.loc[0, "lstm_wins"]), 1)
        self.assertAlmostEqual(float(summary.loc[0, "xlstm_win_rate"]), 66.6666666667)

    def test_winner_summary_counts_primary_wmape_wins_and_ties(self) -> None:
        stock_accuracy = pd.DataFrame(
            {
                "sequence_backbone": ["lstm", "xlstm"] * 3,
                "stock_id": [1, 1, 2, 2, 3, 3],
                "stock_name": ["A", "A", "B", "B", "C", "C"],
                "industry_category": ["X", "X", "Y", "Y", "Z", "Z"],
                "model": ["Rolling LSTM"] * 6,
                "MAPE": [10.0, 9.0, 10.0, 11.0, 10.0, 10.0],
                "WMAPE": [10.0, 9.0, 10.0, 11.0, 10.0, 10.0],
            }
        )

        effects = backbone_ablation.build_backbone_effects(stock_accuracy)
        summary = backbone_ablation.build_winner_summary(effects)

        self.assertEqual(int(summary.loc[0, "WMAPE_xlstm_wins"]), 1)
        self.assertEqual(int(summary.loc[0, "WMAPE_lstm_wins"]), 1)
        self.assertEqual(int(summary.loc[0, "WMAPE_ties"]), 1)
        self.assertAlmostEqual(float(summary.loc[0, "WMAPE_xlstm_win_rate"]), 33.3333333333)

    def test_underestimate_risk_flags_negative_bias_high_underestimate_rate(self) -> None:
        stock_accuracy = pd.DataFrame(
            {
                "sequence_backbone": ["xlstm", "lstm"],
                "stock_id": [3017, 1101],
                "stock_name": ["A", "B"],
                "industry_category": ["thermal", "cement"],
                "model": ["Rolling LSTM", "Rolling LSTM"],
                "MAPE": [12.0, 9.0],
                "MAE": [100.0, 80.0],
                "Bias": [-50.0, 10.0],
                "UnderestimateRate": [83.0, 10.0],
            }
        )

        risk = backbone_ablation.build_underestimate_risk(stock_accuracy)

        self.assertEqual(risk.loc[0, "underestimate_risk_level"], "high")
        self.assertEqual(int(risk.loc[0, "stock_id"]), 3017)


if __name__ == "__main__":
    unittest.main()
