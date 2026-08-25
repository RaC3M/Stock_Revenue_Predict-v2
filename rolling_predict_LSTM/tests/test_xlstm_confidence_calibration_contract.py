import unittest

import pandas as pd

from rolling_predict_LSTM import batch_xlstm_confidence_calibration as calibration
from rolling_predict_LSTM import batch_xlstm_decline_cap_confidence as confidence


class XLSTMConfidenceCalibrationContractTests(unittest.TestCase):
    def test_threshold_selection_uses_validation_metric_only(self) -> None:
        accuracy = pd.DataFrame(
            {
                "model": [
                    confidence.confidence_model_name(0.35),
                    confidence.confidence_model_name(0.45),
                    confidence.confidence_model_name(0.55),
                ],
                "WMAPE": [12.0, 10.0, 11.0],
                "MAPE": [20.0, 30.0, 10.0],
            }
        )

        selection = calibration.build_threshold_selection(accuracy, (0.35, 0.45, 0.55), "WMAPE")

        selected = selection[selection["selected"]]
        self.assertEqual(float(selected.iloc[0]["threshold"]), 0.45)
        self.assertEqual(float(selection.iloc[0]["selected_threshold"]), 0.45)

    def test_threshold_selection_breaks_ties_toward_higher_threshold(self) -> None:
        accuracy = pd.DataFrame(
            {
                "model": [confidence.confidence_model_name(0.35), confidence.confidence_model_name(0.55)],
                "WMAPE": [10.0, 10.0],
            }
        )

        selection = calibration.build_threshold_selection(accuracy, (0.35, 0.55), "WMAPE")

        self.assertEqual(calibration.selected_threshold(selection), 0.55)

    def test_build_calibrated_predictions_renames_selected_threshold_model(self) -> None:
        target_confidence = pd.DataFrame(
            {
                "stock_id": [1, 1],
                "target_year": [2025, 2025],
                "target_month": [1, 1],
                "actual_revenue": [80.0, 80.0],
                "model": [confidence.confidence_model_name(0.35), confidence.confidence_model_name(0.45)],
                "predicted_revenue": [100.0, 90.0],
                "confidence_cap_applied": [False, True],
            }
        )
        selection = pd.DataFrame(
            {
                "threshold": [0.35, 0.45],
                "model": [confidence.confidence_model_name(0.35), confidence.confidence_model_name(0.45)],
                "selected": [False, True],
            }
        )

        calibrated = calibration.build_calibrated_predictions(
            target_confidence,
            selection,
            validation_year=2024,
            selection_metric="WMAPE",
        )

        self.assertEqual(calibrated["model"].unique().tolist(), [calibration.CALIBRATED_MODEL])
        self.assertEqual(calibrated["source_model"].tolist(), [confidence.confidence_model_name(0.45)])
        self.assertEqual(float(calibrated.iloc[0]["calibrated_threshold"]), 0.45)
        self.assertEqual(float(calibrated.iloc[0]["error"]), 10.0)


if __name__ == "__main__":
    unittest.main()
