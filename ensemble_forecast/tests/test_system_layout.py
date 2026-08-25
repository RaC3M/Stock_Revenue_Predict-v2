import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ForecastSystemLayoutTests(unittest.TestCase):
    def test_forecast_systems_are_peer_directories_with_no_root_app(self) -> None:
        self.assertTrue((PROJECT_ROOT / "ensemble_forecast" / "app.py").is_file())
        self.assertTrue((PROJECT_ROOT / "ensemble_forecast" / "forecast_engine.py").is_file())
        self.assertTrue((PROJECT_ROOT / "rolling_predict_LSTM" / "app.py").is_file())
        self.assertTrue((PROJECT_ROOT / "rolling_predict_LSTM" / "rolling_lstm_engine.py").is_file())
        self.assertFalse((PROJECT_ROOT / "app.py").exists())

    def test_forecast_systems_do_not_import_each_other(self) -> None:
        systems = {
            "ensemble_forecast": "rolling_predict_LSTM",
            "rolling_predict_LSTM": "ensemble_forecast",
        }
        for system_name, forbidden_import in systems.items():
            system_dir = PROJECT_ROOT / system_name
            for source_path in system_dir.glob("*.py"):
                tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
                imported_names = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported_names.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported_names.append(node.module)
                self.assertFalse(
                    any(name == forbidden_import or name.startswith(f"{forbidden_import}.") for name in imported_names),
                    f"{source_path} imports peer system {forbidden_import}",
                )


if __name__ == "__main__":
    unittest.main()
