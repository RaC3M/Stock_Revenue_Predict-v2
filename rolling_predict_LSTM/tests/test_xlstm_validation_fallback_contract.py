import unittest
import warnings

import numpy as np
import pandas as pd

from rolling_predict_LSTM import batch_xlstm_validation_fallback as fallback


class XLSTMValidationFallbackContractTests(unittest.TestCase):
    def test_build_fallback_selection_uses_prior_year_validation_metric(self) -> None:
        validation_accuracy = pd.DataFrame(
            {
                "stock_id": [1, 1, 2, 2, 3, 3],
                "stock_name": ["A", "A", "B", "B", "C", "C"],
                "industry_category": ["X", "X", "Y", "Y", "Z", "Z"],
                "model": [
                    fallback.VALIDATION_MODEL_PLAIN,
                    fallback.VALIDATION_MODEL_ADJUSTED,
                    fallback.VALIDATION_MODEL_PLAIN,
                    fallback.VALIDATION_MODEL_ADJUSTED,
                    fallback.VALIDATION_MODEL_PLAIN,
                    fallback.VALIDATION_MODEL_ADJUSTED,
                ],
                "MAE": [100.0, 80.0, 100.0, 120.0, 100.0, 100.0],
                "MAPE": [10.0, 8.0, 10.0, 12.0, 10.0, 10.0],
                "WMAPE": [9.0, 7.0, 9.0, 11.0, 9.0, 9.0],
                "DirectionAccuracy": [50.0, 60.0, 50.0, 40.0, 50.0, 60.0],
            }
        )

        selection = fallback.build_fallback_selection(validation_accuracy, selection_metric="WMAPE")

        by_stock = selection.set_index("stock_id")
        self.assertEqual(by_stock.loc[1, "selected_model"], fallback.VALIDATION_MODEL_ADJUSTED)
        self.assertEqual(by_stock.loc[1, "selection_reason"], "adjusted_validation_metric_improved")
        self.assertEqual(by_stock.loc[2, "selected_model"], fallback.VALIDATION_MODEL_PLAIN)
        self.assertEqual(by_stock.loc[2, "selection_reason"], "plain_validation_metric_kept")
        self.assertEqual(by_stock.loc[3, "selected_model"], fallback.VALIDATION_MODEL_PLAIN)
        self.assertEqual(by_stock.loc[3, "selection_reason"], "plain_validation_metric_kept")
        self.assertAlmostEqual(float(by_stock.loc[1, "selection_metric_delta_adjusted_minus_plain"]), -2.0)

    def test_build_fallback_selection_uses_default_when_metric_is_missing(self) -> None:
        validation_accuracy = pd.DataFrame(
            {
                "stock_id": [1, 1],
                "model": [fallback.VALIDATION_MODEL_PLAIN, fallback.VALIDATION_MODEL_ADJUSTED],
                "MAE": [100.0, 80.0],
                "MAPE": [10.0, np.nan],
                "WMAPE": [9.0, np.nan],
                "DirectionAccuracy": [50.0, 60.0],
            }
        )

        selection = fallback.build_fallback_selection(
            validation_accuracy,
            selection_metric="WMAPE",
            fallback_default=fallback.VALIDATION_MODEL_ADJUSTED,
        )

        self.assertEqual(selection.loc[0, "selected_model"], fallback.VALIDATION_MODEL_ADJUSTED)
        self.assertEqual(selection.loc[0, "selection_reason"], "missing_validation_metric")

    def test_complete_fallback_selection_records_stocks_missing_validation(self) -> None:
        selection = pd.DataFrame(
            {
                "stock_id": [1],
                "selected_model": [fallback.VALIDATION_MODEL_ADJUSTED],
                "selection_metric": ["WMAPE"],
                "selection_metric_delta_adjusted_minus_plain": [-2.0],
                "min_improvement_required": [0.0],
                "selection_reason": ["adjusted_validation_metric_improved"],
            }
        )
        target_monthly = pd.DataFrame(
            {
                "stock_id": [1, 1, 2, 2],
                "stock_name": ["A", "A", "B", "B"],
                "industry_category": ["X", "X", "Y", "Y"],
                "model": [
                    fallback.VALIDATION_MODEL_PLAIN,
                    fallback.VALIDATION_MODEL_ADJUSTED,
                    fallback.VALIDATION_MODEL_PLAIN,
                    fallback.VALIDATION_MODEL_ADJUSTED,
                ],
            }
        )

        completed = fallback.complete_fallback_selection(selection, target_monthly)
        by_stock = completed.set_index("stock_id")

        self.assertEqual(len(completed), 2)
        self.assertEqual(by_stock.loc[1, "selected_model"], fallback.VALIDATION_MODEL_ADJUSTED)
        self.assertEqual(by_stock.loc[2, "selected_model"], fallback.VALIDATION_MODEL_PLAIN)
        self.assertEqual(by_stock.loc[2, "selection_reason"], "missing_validation_stock")
        self.assertEqual(by_stock.loc[2, "stock_name"], "B")

    def test_stock_regime_selection_can_choose_different_sources_per_regime(self) -> None:
        validation_accuracy = pd.DataFrame(
            {
                "stock_id": [1, 1, 1, 1],
                "stock_name": ["A", "A", "A", "A"],
                "industry_category": ["X", "X", "X", "X"],
                "regime": ["decline", "decline", "cycle", "cycle"],
                "model": [
                    fallback.VALIDATION_MODEL_PLAIN,
                    fallback.VALIDATION_MODEL_ADJUSTED,
                    fallback.VALIDATION_MODEL_PLAIN,
                    fallback.VALIDATION_MODEL_ADJUSTED,
                ],
                "MAE": [200.0, 100.0, 50.0, 50.0],
                "MAPE": [20.0, 10.0, 5.0, 5.0],
                "WMAPE": [18.0, 9.0, 4.0, 4.0],
                "DirectionAccuracy": [40.0, 50.0, 60.0, 70.0],
            }
        )

        selection = fallback.build_fallback_selection(
            validation_accuracy,
            selection_metric="WMAPE",
            selection_scope=fallback.SELECTION_SCOPE_STOCK_REGIME,
        )
        by_regime = selection.set_index(["stock_id", "regime"])

        self.assertEqual(by_regime.loc[(1, "decline"), "selected_model"], fallback.VALIDATION_MODEL_ADJUSTED)
        self.assertEqual(by_regime.loc[(1, "cycle"), "selected_model"], fallback.VALIDATION_MODEL_PLAIN)
        self.assertEqual(by_regime.loc[(1, "cycle"), "selection_reason"], "plain_validation_metric_kept")

    def test_complete_stock_regime_selection_records_missing_regime_separately(self) -> None:
        selection = pd.DataFrame(
            {
                "stock_id": [1],
                "regime": ["decline"],
                "selected_model": [fallback.VALIDATION_MODEL_ADJUSTED],
                "selection_metric": ["WMAPE"],
                "selection_scope": [fallback.SELECTION_SCOPE_STOCK_REGIME],
                "selection_metric_delta_adjusted_minus_plain": [-5.0],
                "min_improvement_required": [0.0],
                "selection_reason": ["adjusted_validation_metric_improved"],
            }
        )
        target_monthly = pd.DataFrame(
            {
                "stock_id": [1, 1, 1, 2],
                "stock_name": ["A", "A", "A", "B"],
                "industry_category": ["X", "X", "X", "Y"],
                "regime": ["decline", "cycle", "growth", "decline"],
                "model": [
                    fallback.VALIDATION_MODEL_PLAIN,
                    fallback.VALIDATION_MODEL_PLAIN,
                    fallback.VALIDATION_MODEL_PLAIN,
                    fallback.VALIDATION_MODEL_PLAIN,
                ],
            }
        )

        completed = fallback.complete_fallback_selection(
            selection,
            target_monthly,
            selection_scope=fallback.SELECTION_SCOPE_STOCK_REGIME,
        )
        by_group = completed.set_index(["stock_id", "regime"])

        self.assertEqual(len(completed), 4)
        self.assertEqual(by_group.loc[(1, "decline"), "selected_model"], fallback.VALIDATION_MODEL_ADJUSTED)
        self.assertEqual(by_group.loc[(1, "cycle"), "selected_model"], fallback.VALIDATION_MODEL_PLAIN)
        self.assertEqual(by_group.loc[(1, "cycle"), "selection_reason"], "missing_validation_regime")
        self.assertEqual(by_group.loc[(2, "decline"), "selection_reason"], "missing_validation_stock")

    def test_build_fallback_predictions_selects_source_and_recomputes_error(self) -> None:
        target_monthly = pd.DataFrame(
            {
                "stock_id": [1, 1, 2, 2],
                "target_year": [2025, 2025, 2025, 2025],
                "target_month": [1, 1, 1, 1],
                "actual_revenue": [80.0, 80.0, 240.0, 240.0],
                "model": [
                    fallback.VALIDATION_MODEL_PLAIN,
                    fallback.VALIDATION_MODEL_ADJUSTED,
                    fallback.VALIDATION_MODEL_PLAIN,
                    fallback.VALIDATION_MODEL_ADJUSTED,
                ],
                "predicted_revenue": [100.0, 90.0, 200.0, 250.0],
                "xlstm_decline_cap_applied": [False, True, False, True],
            }
        )
        selection = pd.DataFrame(
            {
                "stock_id": [1, 2],
                "selected_model": [fallback.VALIDATION_MODEL_ADJUSTED, fallback.VALIDATION_MODEL_PLAIN],
                "selection_metric": ["WMAPE", "WMAPE"],
                "selection_metric_delta_adjusted_minus_plain": [-2.0, 2.0],
                "selection_reason": [
                    "adjusted_validation_metric_improved",
                    "plain_validation_metric_kept",
                ],
            }
        )

        result = fallback.build_fallback_predictions(target_monthly, selection).sort_values("stock_id")

        self.assertEqual(result["model"].unique().tolist(), [fallback.FALLBACK_MODEL])
        self.assertEqual(result["source_model"].tolist(), [fallback.VALIDATION_MODEL_ADJUSTED, fallback.VALIDATION_MODEL_PLAIN])
        self.assertEqual(result["predicted_revenue"].tolist(), [90.0, 200.0])
        self.assertEqual(result["error"].tolist(), [10.0, -40.0])
        self.assertEqual(result["abs_error"].tolist(), [10.0, 40.0])
        self.assertEqual(result["xlstm_decline_cap_applied"].tolist(), [True, False])

    def test_build_fallback_predictions_uses_plain_when_selected_adjusted_source_is_missing(self) -> None:
        target_monthly = pd.DataFrame(
            {
                "stock_id": [1, 1, 1],
                "target_year": [2025, 2025, 2025],
                "target_month": [1, 1, 2],
                "actual_revenue": [80.0, 80.0, 120.0],
                "model": [
                    fallback.VALIDATION_MODEL_PLAIN,
                    fallback.VALIDATION_MODEL_ADJUSTED,
                    fallback.VALIDATION_MODEL_PLAIN,
                ],
                "predicted_revenue": [100.0, 90.0, 130.0],
            }
        )
        selection = pd.DataFrame(
            {
                "stock_id": [1],
                "selected_model": [fallback.VALIDATION_MODEL_ADJUSTED],
                "selection_metric": ["WMAPE"],
                "selection_metric_delta_adjusted_minus_plain": [-2.0],
                "selection_reason": ["adjusted_validation_metric_improved"],
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = fallback.build_fallback_predictions(target_monthly, selection).sort_values("target_month")

        self.assertEqual(result["source_model"].tolist(), [fallback.VALIDATION_MODEL_ADJUSTED, fallback.VALIDATION_MODEL_PLAIN])
        self.assertEqual(result["predicted_revenue"].tolist(), [90.0, 130.0])
        self.assertEqual(result["fallback_source_missing"].tolist(), [False, True])
        self.assertEqual(result["xlstm_decline_cap_applied"].tolist(), [False, False])
        self.assertTrue(pd.api.types.is_bool_dtype(result["xlstm_decline_cap_applied"]))

    def test_stock_regime_fallback_predictions_select_by_target_regime(self) -> None:
        target_monthly = pd.DataFrame(
            {
                "stock_id": [1, 1, 1, 1],
                "target_year": [2025, 2025, 2025, 2025],
                "target_month": [1, 1, 2, 2],
                "actual_revenue": [80.0, 80.0, 120.0, 120.0],
                "regime": ["decline", "decline", "cycle", "cycle"],
                "model": [
                    fallback.VALIDATION_MODEL_PLAIN,
                    fallback.VALIDATION_MODEL_ADJUSTED,
                    fallback.VALIDATION_MODEL_PLAIN,
                    fallback.VALIDATION_MODEL_ADJUSTED,
                ],
                "predicted_revenue": [100.0, 90.0, 130.0, 110.0],
            }
        )
        selection = pd.DataFrame(
            {
                "stock_id": [1, 1],
                "regime": ["decline", "cycle"],
                "selected_model": [fallback.VALIDATION_MODEL_ADJUSTED, fallback.VALIDATION_MODEL_PLAIN],
            }
        )

        result = fallback.build_fallback_predictions(
            target_monthly,
            selection,
            selection_scope=fallback.SELECTION_SCOPE_STOCK_REGIME,
        ).sort_values("target_month")

        self.assertEqual(result["source_model"].tolist(), [fallback.VALIDATION_MODEL_ADJUSTED, fallback.VALIDATION_MODEL_PLAIN])
        self.assertEqual(result["predicted_revenue"].tolist(), [90.0, 130.0])


if __name__ == "__main__":
    unittest.main()
