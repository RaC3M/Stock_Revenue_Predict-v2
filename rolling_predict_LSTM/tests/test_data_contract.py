from pathlib import Path
import unittest

from rolling_predict_LSTM import rolling_lstm_engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RollingDataContractTests(unittest.TestCase):
    def test_default_data_directory_is_the_shared_root_data_directory(self) -> None:
        self.assertEqual(Path(rolling_lstm_engine.DATA_DIR).resolve(), (PROJECT_ROOT / "data").resolve())
        self.assertTrue((Path(rolling_lstm_engine.DATA_DIR) / rolling_lstm_engine.REVENUE_FILENAME).is_file())

    def test_relative_data_directory_override_resolves_from_project_root(self) -> None:
        resolved = rolling_lstm_engine._resolve_data_dir("free_taiwan_data/processed_benchmark_82")
        self.assertEqual(resolved, PROJECT_ROOT / "free_taiwan_data" / "processed_benchmark_82")


if __name__ == "__main__":
    unittest.main()
