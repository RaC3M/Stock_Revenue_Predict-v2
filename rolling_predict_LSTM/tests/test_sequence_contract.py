import unittest

import pandas as pd

from rolling_predict_LSTM import rolling_lstm_engine


class RollingSequenceContractTests(unittest.TestCase):
    def test_evaluation_sequence_ends_before_its_target_month(self) -> None:
        dates = pd.date_range("2024-01-01", periods=13, freq="MS")
        frame = pd.DataFrame(
            {
                "stock_id": [1101] * 13,
                "date": dates,
                "revenue_year": dates.year,
                "revenue_month": dates.month,
                "revenue_thousand": list(range(100, 113)),
                "cluster": [2] * 13,
                "trend_component": list(range(90, 103)),
                "cycle_component": [10.0] * 13,
            }
        )

        samples = rolling_lstm_engine.build_eval_sequences_for_stock(frame, selected_stock=1101, window_size=12)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["sequence_end_date"], pd.Timestamp("2024-12-01"))
        self.assertEqual(samples[0]["target_date"], pd.Timestamp("2025-01-01"))
        self.assertEqual(samples[0]["sequence_frame"]["date"].max(), pd.Timestamp("2024-12-01"))
        self.assertNotIn("target_revenue", samples[0])
        self.assertNotIn("target_trend", samples[0])
        self.assertNotIn("target_cycle", samples[0])

    def test_sequence_builders_reject_windows_that_cross_a_missing_month(self) -> None:
        dates = pd.to_datetime(
            [
                "2023-12-01",
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
                "2024-04-01",
                "2024-05-01",
                "2024-07-01",
                "2024-08-01",
                "2024-09-01",
                "2024-10-01",
                "2024-11-01",
                "2024-12-01",
                "2025-01-01",
            ]
        )
        frame = pd.DataFrame(
            {
                "stock_id": [1101] * len(dates),
                "date": dates,
                "revenue_year": dates.year,
                "revenue_month": dates.month,
                "revenue_thousand": range(100, 100 + len(dates)),
                "growth_direction": [1] * len(dates),
                "cluster": [2] * len(dates),
                "trend_component": range(90, 90 + len(dates)),
                "cycle_component": [10.0] * len(dates),
            }
        )

        windows = rolling_lstm_engine.build_growth_windows(frame, window_size=12)
        samples = rolling_lstm_engine.build_eval_sequences_for_stock(
            frame,
            selected_stock=1101,
            window_size=12,
        )

        self.assertTrue(windows.empty)
        self.assertEqual(samples, [])

    def test_prepare_revenue_resets_growth_features_after_a_calendar_gap(self) -> None:
        raw = pd.DataFrame(
            {
                "stock_id": [1, 1, 1],
                "revenue_year": [2024, 2024, 2024],
                "revenue_month": [1, 2, 4],
                "revenue_thousand": [100.0, 120.0, 300.0],
            }
        )

        prepared = rolling_lstm_engine.prepare_revenue_data(raw)

        self.assertAlmostEqual(float(prepared.loc[1, "growth_rate"]), 0.2)
        self.assertEqual(float(prepared.loc[2, "growth_rate"]), 0.0)
        self.assertNotEqual(
            int(prepared.loc[1, "_calendar_segment"]),
            int(prepared.loc[2, "_calendar_segment"]),
        )

    def test_forward_validation_split_uses_latest_year_without_overlap(self) -> None:
        samples = [
            {"target_year": 2022, "id": "a"},
            {"target_year": 2023, "id": "b"},
            {"target_year": 2024, "id": "c"},
            {"target_year": 2024, "id": "d"},
        ]

        train, validation, was_capped = rolling_lstm_engine.split_forward_validation_samples(
            samples,
            max_train_samples=0,
        )

        self.assertFalse(was_capped)
        self.assertEqual([sample["id"] for sample in train], ["a", "b"])
        self.assertEqual([sample["id"] for sample in validation], ["c", "d"])

    def test_forecast_assembly_owns_guardrails_adjustment_and_evaluation(self) -> None:
        prediction_frame = pd.DataFrame(
            {
                "stock_id": [1101],
                "target_date": pd.to_datetime(["2025-01-01"]),
                "target_year": [2025],
                "target_month": [1],
                "last_observed_revenue": [100.0],
                "sequence_max_revenue": [120.0],
                "cluster": [2],
                "growth_rate_at_end": [0.1],
                "momentum_3m_at_end": [0.1],
                "momentum_6m_at_end": [0.1],
                "growth_ratio": [0.8],
                "growth_streak": [5],
                "trend_component": [95.0],
                "cycle_component": [5.0],
                "cycle_volatility_6m": [2.0],
                "trend_slope": [3.0],
                "trend_slope_rate": [0.03],
                "raw_pred_cluster": [110.0],
                "raw_pred_plain": [105.0],
                "raw_pred_xlstm": [108.0],
            }
        )
        actual = pd.DataFrame(
            {
                "stock_id": [1101],
                "target_year": [2025],
                "target_month": [1],
                "actual_revenue": [115.0],
            }
        )

        result = rolling_lstm_engine.assemble_rolling_forecast(
            prediction_frame,
            actual,
            rolling_lstm_engine.GrowthAdjustmentConfig(),
            rolling_lstm_engine.GrowthAdjustmentConfig(alpha=0.0),
            include_xlstm_plain=True,
        )

        self.assertEqual(len(result.forecast), 1)
        self.assertIn("predicted_revenue_adjusted", result.forecast)
        self.assertIn("adjusted_abs_error", result.forecast)
        self.assertEqual(len(result.metrics), len(rolling_lstm_engine.ROLLING_MODEL_OUTPUTS))
        self.assertEqual(set(result.clip_counts), {"cluster", "plain", "xlstm", "xlstm_adjusted", "adjusted"})

    def test_actual_revenue_is_attached_during_evaluation(self) -> None:
        forecast = pd.DataFrame(
            {
                "stock_id": [1101],
                "target_year": [2025],
                "target_month": [1],
                "last_observed_revenue": [100.0],
                "predicted_revenue_no_cluster": [110],
                "predicted_revenue_cluster": [120],
                "predicted_revenue_adjusted": [140],
                "predicted_revenue_xlstm": [130],
                "predicted_revenue_xlstm_adjusted": [150],
            }
        )
        actual_revenue = pd.DataFrame(
            {
                "stock_id": [1101],
                "target_year": [2025],
                "target_month": [1],
                "actual_revenue": [125.0],
            }
        )

        evaluated, metrics = rolling_lstm_engine.evaluate_rolling_forecast(forecast, actual_revenue)

        self.assertEqual(float(evaluated.loc[0, "actual_revenue"]), 125.0)
        self.assertEqual(float(evaluated.loc[0, "cluster_error"]), -5.0)
        self.assertEqual(float(evaluated.loc[0, "adjusted_abs_error"]), 15.0)
        self.assertEqual(float(evaluated.loc[0, "xlstm_error"]), 5.0)
        self.assertEqual(float(evaluated.loc[0, "xlstm_adjusted_error"]), 25.0)
        self.assertEqual(float(evaluated.loc[0, "actual_return"]), 0.25)
        cluster_metric = metrics[metrics["model"] == "Rolling LSTM + Cluster"].iloc[0]
        self.assertEqual(float(cluster_metric["MAPE"]), 4.0)
        self.assertEqual(float(cluster_metric["MedianAPE"]), 4.0)
        self.assertEqual(float(cluster_metric["WMAPE"]), 4.0)
        self.assertAlmostEqual(float(cluster_metric["SMAPE"]), 4.082, places=3)
        self.assertEqual(float(cluster_metric["DirectionAccuracy"]), 100.0)
        self.assertEqual(
            metrics["model"].tolist(),
            [
                "Rolling LSTM",
                "Rolling LSTM + Cluster",
                "Rolling LSTM + Cluster + Conditional Adjustment",
                "Rolling xLSTM",
                "Rolling xLSTM + Conditional Adjustment",
            ],
        )

    def test_missing_xlstm_prediction_column_yields_nan_metrics(self) -> None:
        forecast = pd.DataFrame(
            {
                "stock_id": [1101],
                "target_year": [2025],
                "target_month": [1],
                "last_observed_revenue": [100.0],
                "predicted_revenue_no_cluster": [110],
                "predicted_revenue_cluster": [120],
                "predicted_revenue_adjusted": [140],
            }
        )
        actual_revenue = pd.DataFrame(
            {
                "stock_id": [1101],
                "target_year": [2025],
                "target_month": [1],
                "actual_revenue": [125.0],
            }
        )

        evaluated, metrics = rolling_lstm_engine.evaluate_rolling_forecast(forecast, actual_revenue)

        self.assertIn("predicted_revenue_xlstm", evaluated.columns)
        self.assertIn("predicted_revenue_xlstm_adjusted", evaluated.columns)
        xlstm_metric = metrics[metrics["model"] == "Rolling xLSTM"].iloc[0]
        xlstm_adjusted_metric = metrics[metrics["model"] == "Rolling xLSTM + Conditional Adjustment"].iloc[0]
        self.assertTrue(pd.isna(xlstm_metric["MAPE"]))
        self.assertTrue(pd.isna(xlstm_metric["WMAPE"]))
        self.assertTrue(pd.isna(xlstm_metric["SMAPE"]))
        self.assertTrue(pd.isna(xlstm_adjusted_metric["MAPE"]))

    def test_balanced_decline_cap_requires_deeper_decline_and_prediction_overshoot(self) -> None:
        metadata = pd.DataFrame(
            {
                "last_observed_revenue": [100.0, 100.0, 100.0, 100.0],
                "growth_ratio": [0.20, 0.38, 0.20, 0.20],
            }
        )
        regime = ["decline", "decline", "decline", "cycle"]
        predicted = [120.0, 120.0, 105.0, 120.0]

        old_mask = rolling_lstm_engine.calculate_decline_cap_mask(predicted, metadata, regime)
        balanced_mask = rolling_lstm_engine.calculate_decline_cap_mask(
            predicted,
            metadata,
            regime,
            decline_cap_growth_ratio_max=0.35,
            decline_cap_prediction_ratio_min=1.10,
        )

        self.assertEqual(old_mask.tolist(), [True, True, True, False])
        self.assertEqual(balanced_mask.tolist(), [True, False, False, False])

    def test_cluster_profile_is_built_from_training_windows_only(self) -> None:
        windows = pd.DataFrame(
            {
                "stock_id": [1, 2, 1, 2],
                "window_end_year": [2024, 2024, 2025, 2025],
                "window_end_month": [12, 12, 1, 1],
                "window_end_date": pd.to_datetime(["2024-12-01", "2024-12-01", "2025-01-01", "2025-01-01"]),
                "growth_ratio": [0.0, 1.0, 1.0, 1.0],
                "growth_streak": [0, 2, 2, 2],
                "g_1": [0, 1, 1, 1],
                "g_2": [0, 1, 1, 1],
            }
        )

        _, clustered, profile = rolling_lstm_engine.fit_kmeans_clusters(windows, k=2, train_end_year=2024)

        self.assertEqual(len(clustered), 4)
        self.assertEqual(int(profile["window_count"].sum()), 2)
        self.assertEqual(set(profile["profile_train_end_year"]), {2024})


if __name__ == "__main__":
    unittest.main()
