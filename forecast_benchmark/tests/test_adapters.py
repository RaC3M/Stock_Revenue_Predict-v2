from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from forecast_benchmark.adapters.rolling_adapter import load_rolling_predictions
from forecast_benchmark.run_benchmark import filter_comparable_predictions, select_stock_pool


class AdapterTests(unittest.TestCase):
    def test_load_rolling_predictions_normalizes_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pd.DataFrame(
                {
                    "stock_id": [2, 1],
                    "stock_name": ["B", "A"],
                    "industry_category": ["x", "y"],
                    "target_year": [2025, 2025],
                    "target_month": [1, 1],
                    "model": ["Rolling LSTM", "Rolling xLSTM"],
                    "predicted_revenue": [100.0, 200.0],
                    "actual_revenue": [110.0, 190.0],
                    "last_observed_revenue": [90.0, 180.0],
                    "sequence_backbone": ["lstm", "xlstm_hybrid"],
                    "xlstm_backbone": ["xlstm_hybrid", "xlstm_hybrid"],
                }
            ).to_csv(output_dir / "monthly_predictions.csv", index=False)

            normalized = load_rolling_predictions(output_dir, target_year=2025, stock_ids=[1])

        self.assertEqual(normalized["stock_id"].tolist(), [1])
        self.assertEqual(normalized["source_family"].unique().tolist(), ["rolling_lstm"])
        self.assertIn("last_observed_revenue", normalized.columns)
        self.assertEqual(normalized["sequence_backbone"].tolist(), ["xlstm_hybrid"])
        self.assertEqual(normalized["xlstm_backbone"].tolist(), ["xlstm_hybrid"])

    def test_select_stock_pool_uses_sorted_rolling_pool_and_limit(self) -> None:
        rolling_predictions = pd.DataFrame({"stock_id": [3, 1, 2, 1]})
        self.assertEqual(select_stock_pool(rolling_predictions, None, 2), [1, 2])
        self.assertEqual(select_stock_pool(rolling_predictions, [2, 9, 1], None), [2, 1])

    def test_filter_comparable_predictions_requires_all_source_families(self) -> None:
        predictions = pd.DataFrame(
            {
                "stock_id": [1, 1, 2],
                "target_year": [2025, 2025, 2025],
                "target_month": [1, 1, 1],
                "actual_revenue": [100.0, 100.0, 200.0],
                "source_family": ["rolling_lstm", "ensemble_forecast", "rolling_lstm"],
                "model": ["R", "E", "R"],
            }
        )

        comparable = filter_comparable_predictions(predictions)

        self.assertEqual(comparable["stock_id"].tolist(), [1, 1])

    def test_filter_comparable_predictions_intersects_exact_model_observations(self) -> None:
        predictions = pd.DataFrame(
            {
                "stock_id": [1, 1, 1, 1, 1],
                "target_year": [2025] * 5,
                "target_month": [1, 2, 1, 2, 1],
                "actual_revenue": [100.0, 200.0, 100.0, 200.0, 100.0],
                "source_family": ["rolling", "rolling", "ensemble", "ensemble", "extra"],
                "model": ["R", "R", "E", "E", "X"],
            }
        )

        comparable = filter_comparable_predictions(
            predictions,
            required_pairs={("rolling", "R"), ("ensemble", "E"), ("extra", "X")},
        )

        self.assertEqual(set(comparable["target_month"]), {1})
        self.assertEqual(len(comparable), 3)

    def test_filter_comparable_predictions_rejects_duplicate_model_observations(self) -> None:
        predictions = pd.DataFrame(
            {
                "stock_id": [1, 1],
                "target_year": [2025, 2025],
                "target_month": [1, 1],
                "actual_revenue": [100.0, 100.0],
                "source_family": ["rolling", "rolling"],
                "model": ["R", "R"],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            filter_comparable_predictions(predictions)

    def test_load_rolling_predictions_rejects_missing_requested_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pd.DataFrame(
                {
                    "stock_id": [1],
                    "target_year": [2025],
                    "target_month": [1],
                    "model": ["Rolling LSTM"],
                    "predicted_revenue": [100.0],
                    "actual_revenue": [110.0],
                }
            ).to_csv(output_dir / "monthly_predictions.csv", index=False)

            with self.assertRaisesRegex(ValueError, "requested models"):
                load_rolling_predictions(
                    output_dir,
                    target_year=2025,
                    model_names=["Rolling LSTM", "Missing Model"],
                )

    def test_load_rolling_predictions_requires_xlstm_architecture_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pd.DataFrame(
                {
                    "stock_id": [1],
                    "target_year": [2025],
                    "target_month": [1],
                    "model": ["Rolling xLSTM"],
                    "predicted_revenue": [100.0],
                    "actual_revenue": [110.0],
                }
            ).to_csv(output_dir / "monthly_predictions.csv", index=False)

            with self.assertRaisesRegex(ValueError, "require architecture provenance"):
                load_rolling_predictions(output_dir, target_year=2025)

    def test_load_rolling_predictions_keeps_legacy_lstm_only_outputs_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pd.DataFrame(
                {
                    "stock_id": [1],
                    "target_year": [2025],
                    "target_month": [1],
                    "model": ["Rolling LSTM"],
                    "predicted_revenue": [100.0],
                    "actual_revenue": [110.0],
                }
            ).to_csv(output_dir / "monthly_predictions.csv", index=False)

            normalized = load_rolling_predictions(output_dir, target_year=2025)

        self.assertTrue(normalized["sequence_backbone"].isna().all())
        self.assertTrue(normalized["xlstm_backbone"].isna().all())

    def test_load_rolling_predictions_rejects_mismatched_xlstm_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pd.DataFrame(
                {
                    "stock_id": [1],
                    "target_year": [2025],
                    "target_month": [1],
                    "model": ["Rolling xLSTM"],
                    "predicted_revenue": [100.0],
                    "actual_revenue": [110.0],
                    "sequence_backbone": ["xlstm_hybrid"],
                    "xlstm_backbone": ["xlstm"],
                }
            ).to_csv(output_dir / "monthly_predictions.csv", index=False)

            with self.assertRaisesRegex(ValueError, "must match"):
                load_rolling_predictions(output_dir, target_year=2025)

    def test_load_rolling_predictions_rejects_mixed_xlstm_architectures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pd.DataFrame(
                {
                    "stock_id": [1, 1],
                    "target_year": [2025, 2025],
                    "target_month": [1, 2],
                    "model": ["Rolling xLSTM", "Rolling xLSTM"],
                    "predicted_revenue": [100.0, 120.0],
                    "actual_revenue": [110.0, 115.0],
                    "sequence_backbone": ["xlstm", "xlstm_hybrid"],
                    "xlstm_backbone": ["xlstm", "xlstm_hybrid"],
                }
            ).to_csv(output_dir / "monthly_predictions.csv", index=False)

            with self.assertRaisesRegex(ValueError, "mixes xLSTM architectures"):
                load_rolling_predictions(output_dir, target_year=2025)

    def test_ensemble_engine_imports_are_confined_to_adapter_package(self) -> None:
        benchmark_root = Path(__file__).resolve().parents[1]
        violations = []
        for path in benchmark_root.rglob("*.py"):
            if "adapters" in path.parts or "tests" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            if (
                "ensemble_forecast.forecast_engine" in source
                or "from ensemble_forecast import forecast_engine" in source
            ):
                violations.append(str(path.relative_to(benchmark_root)))

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
