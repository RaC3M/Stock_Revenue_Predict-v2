import builtins
import importlib.util
import inspect
import unittest
from unittest import mock

from rolling_predict_LSTM import rolling_lstm_engine


class RollingBackendContractTests(unittest.TestCase):
    def test_training_backend_is_automatic_pytorch(self) -> None:
        parameters = inspect.signature(rolling_lstm_engine.run_rolling_lstm_experiment).parameters
        self.assertNotIn("backend", parameters)
        self.assertEqual(rolling_lstm_engine.TRAINING_BACKEND, "torch")

    def test_public_experiment_interface_uses_config_object(self) -> None:
        parameters = inspect.signature(rolling_lstm_engine.run_rolling_lstm_experiment).parameters

        self.assertEqual(tuple(parameters), ("selected_stock", "config", "legacy_options"))
        self.assertEqual(parameters["legacy_options"].kind, inspect.Parameter.VAR_KEYWORD)

    def test_legacy_options_are_normalized_into_config(self) -> None:
        config = rolling_lstm_engine._normalize_experiment_config(
            None,
            {
                "k": 4,
                "epochs": 5,
                "enable_growth_adjustment": False,
                "backend": "torch",
                "sequence_backbone": "xlstm",
                "include_xlstm_plain": True,
                "xlstm_growth_adjustment_alpha": 0.2,
            },
        )

        self.assertIsInstance(config, rolling_lstm_engine.RollingExperimentConfig)
        self.assertEqual(config.k, 4)
        self.assertEqual(config.epochs, 5)
        self.assertEqual(config.sequence_backbone, "xlstm")
        self.assertTrue(config.include_xlstm_plain)
        self.assertFalse(config.growth.enabled)
        self.assertEqual(config.xlstm_growth.alpha, 0.2)

    def test_default_sequence_backbone_remains_lstm(self) -> None:
        config = rolling_lstm_engine._normalize_experiment_config(None)

        self.assertEqual(config.sequence_backbone, "lstm")
        self.assertEqual(config.xlstm_backbone, "xlstm")
        self.assertEqual(
            rolling_lstm_engine.DEFAULT_STREAMLIT_XLSTM_BACKBONE,
            "xlstm_hybrid",
        )
        self.assertEqual(config.growth.alpha, rolling_lstm_engine.DEFAULT_GROWTH_ADJUSTMENT_ALPHA)
        self.assertIsNone(config.growth.decline_cap_growth_ratio_max)
        self.assertEqual(
            config.growth.decline_cap_prediction_ratio_min,
            rolling_lstm_engine.DEFAULT_DECLINE_CAP_PREDICTION_RATIO_MIN,
        )
        self.assertEqual(rolling_lstm_engine.DEFAULT_XLSTM_GROWTH_ADJUSTMENT_ALPHA, 0.0)
        self.assertEqual(config.xlstm_growth.alpha, rolling_lstm_engine.DEFAULT_XLSTM_GROWTH_ADJUSTMENT_ALPHA)
        self.assertEqual(
            config.xlstm_growth.decline_cap_growth_ratio_max,
            rolling_lstm_engine.DEFAULT_XLSTM_DECLINE_CAP_GROWTH_RATIO_MAX,
        )
        self.assertEqual(
            config.xlstm_growth.decline_cap_prediction_ratio_min,
            rolling_lstm_engine.DEFAULT_XLSTM_DECLINE_CAP_PREDICTION_RATIO_MIN,
        )

    def test_legacy_options_can_override_xlstm_decline_cap_gate(self) -> None:
        config = rolling_lstm_engine._normalize_experiment_config(
            None,
            {
                "include_xlstm_plain": True,
                "xlstm_decline_cap_growth_ratio_max": 0.25,
                "xlstm_decline_cap_prediction_ratio_min": 1.2,
            },
        )

        self.assertEqual(config.xlstm_growth.decline_cap_growth_ratio_max, 0.25)
        self.assertEqual(config.xlstm_growth.decline_cap_prediction_ratio_min, 1.2)

    def test_unknown_sequence_backbone_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rolling_lstm_engine._normalize_experiment_config(None, {"sequence_backbone": "transformer"})

    def test_hybrid_xlstm_backbone_is_normalized_separately(self) -> None:
        config = rolling_lstm_engine._normalize_experiment_config(
            None,
            {
                "include_xlstm_plain": True,
                "xlstm_backbone": "XLSTM_HYBRID",
            },
        )

        self.assertEqual(config.sequence_backbone, "lstm")
        self.assertEqual(config.xlstm_backbone, "xlstm_hybrid")

    def test_unknown_optional_xlstm_backbone_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rolling_lstm_engine._normalize_experiment_config(None, {"xlstm_backbone": "slstm"})

    def test_xlstm_backbone_spec_owns_label_and_block_layout(self) -> None:
        historical = rolling_lstm_engine.get_xlstm_backbone_spec("XLSTM")
        hybrid = rolling_lstm_engine.get_xlstm_backbone_spec("XLSTM_HYBRID")

        self.assertEqual(historical.block_types, ("mlstm",))
        self.assertEqual(hybrid.block_types, ("mlstm", "slstm"))
        self.assertEqual(hybrid.slstm_backend, "vanilla")
        self.assertIn("Hybrid", hybrid.display_name)

    def test_model_backbone_provenance_uses_the_central_model_registry(self) -> None:
        self.assertEqual(
            rolling_lstm_engine.resolve_model_sequence_backbone(
                rolling_lstm_engine.ROLLING_LSTM_MODEL,
                main_sequence_backbone="lstm",
                xlstm_backbone="xlstm_hybrid",
                include_xlstm_plain=True,
            ),
            "lstm",
        )
        self.assertEqual(
            rolling_lstm_engine.resolve_model_sequence_backbone(
                rolling_lstm_engine.ROLLING_XLSTM_MODEL,
                main_sequence_backbone="lstm",
                xlstm_backbone="xlstm_hybrid",
                include_xlstm_plain=True,
            ),
            "xlstm_hybrid",
        )
        with self.assertRaisesRegex(ValueError, "Unknown rolling model"):
            rolling_lstm_engine.resolve_model_sequence_backbone(
                "Renamed xLSTM display label",
                main_sequence_backbone="lstm",
                xlstm_backbone="xlstm_hybrid",
                include_xlstm_plain=True,
            )

    def test_xlstm_status_probe_never_raises(self) -> None:
        status = rolling_lstm_engine.get_xlstm_backbone_status()

        self.assertIn("available", status)
        self.assertIn("detail", status)

    @unittest.skipUnless(importlib.util.find_spec("xlstm"), "optional xLSTM dependency is not installed")
    def test_xlstm_compat_import_does_not_replace_cuda_availability_probe(self) -> None:
        import torch

        original_cuda_available = torch.cuda.is_available
        original_import = builtins.__import__
        observed_probe_identity: list[bool] = []

        def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "xlstm":
                observed_probe_identity.append(torch.cuda.is_available is original_cuda_available)
            return original_import(name, globals, locals, fromlist, level)

        rolling_lstm_engine._clear_partial_xlstm_imports()
        with mock.patch("builtins.__import__", side_effect=tracking_import):
            components = rolling_lstm_engine._import_xlstm_components()

        self.assertEqual(len(components), 4)
        self.assertTrue(observed_probe_identity)
        self.assertTrue(all(observed_probe_identity))
        self.assertIs(torch.cuda.is_available, original_cuda_available)

    @unittest.skipUnless(importlib.util.find_spec("xlstm"), "optional xLSTM dependency is not installed")
    def test_hybrid_backbone_contains_mlstm_then_slstm_and_preserves_shape(self) -> None:
        import torch
        from torch import nn

        model = rolling_lstm_engine._build_revenue_sequence_model(
            nn,
            input_size=4,
            hidden_units=48,
            window_size=12,
            sequence_backbone="xlstm_hybrid",
        )
        output = model(torch.randn(2, 12, 4))

        self.assertEqual([type(block).__name__ for block in model.backbone.blocks], ["mLSTMBlock", "sLSTMBlock"])
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_removed_main_flow_options_are_rejected(self) -> None:
        removed_options = [
            "enable_direction_filter",
            "enable_trend_cycle_model",
            "enable_dynamic_guardrail",
            "auto_tune_hyperparameters",
        ]

        for option in removed_options:
            with self.subTest(option=option):
                with self.assertRaises(TypeError):
                    rolling_lstm_engine._normalize_experiment_config(None, {option: True})


if __name__ == "__main__":
    unittest.main()
