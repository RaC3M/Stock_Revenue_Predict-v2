import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from tools import validate_project


class ProjectValidationRunnerTests(unittest.TestCase):
    def test_venv_python_uses_platform_specific_layout(self) -> None:
        venv = Path("repo") / "system" / ".venv"

        self.assertEqual(venv / "Scripts" / "python.exe", validate_project.venv_python(venv, "win32"))
        self.assertEqual(venv / "bin" / "python", validate_project.venv_python(venv, "linux"))

    def test_default_suites_cover_every_documented_test_layer(self) -> None:
        root = Path("repo")

        suites = validate_project.build_suites(root, platform="win32")

        self.assertEqual(
            [
                "source-compile",
                "tooling",
                "canonical-data",
                "ensemble-dependencies",
                "rolling-dependencies",
                "data",
                "ensemble",
                "financial",
                "benchmark",
                "rolling",
            ],
            [suite.name for suite in suites],
        )
        self.assertTrue(
            all(
                suite.python_path == root / "ensemble_forecast" / ".venv" / "Scripts" / "python.exe"
                for suite in suites
                if suite.name not in {"rolling-dependencies", "rolling"}
            )
        )
        self.assertEqual(
            root / "rolling_predict_LSTM" / ".venv" / "Scripts" / "python.exe",
            suites[-1].python_path,
        )

    def test_source_compile_suite_covers_code_directories_and_excludes_virtualenvs(self) -> None:
        root = Path("repo")
        suite = next(
            suite for suite in validate_project.build_suites(root, platform="win32") if suite.name == "source-compile"
        )

        self.assertEqual(
            [
                str(root / "ensemble_forecast" / ".venv" / "Scripts" / "python.exe"),
                "-W",
                "error::FutureWarning",
                "-m",
                "compileall",
                "-q",
                "-f",
                "-x",
                r"(?:^|[\\/])\.venv(?:[\\/]|$)",
                str(root / "data_preprocessing"),
                str(root / "ensemble_forecast"),
                str(root / "financial_forecast"),
                str(root / "forecast_benchmark"),
                str(root / "rolling_predict_LSTM"),
                str(root / "tools"),
            ],
            validate_project.build_command(suite),
        )

    def test_source_compile_rejects_bad_project_source_and_ignores_virtualenv_source(self) -> None:
        with TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "project"
            virtualenv_dir = source_dir / ".venv"
            virtualenv_dir.mkdir(parents=True)
            (source_dir / "valid.py").write_text("value = 1\n", encoding="utf-8")
            (virtualenv_dir / "invalid.py").write_text("def broken(:\n", encoding="utf-8")
            suite = validate_project.ValidationSuite.source_compile(
                "source-compile",
                Path(sys.executable),
                (source_dir,),
            )

            valid_result = subprocess.run(
                validate_project.build_command(suite),
                cwd=temp_dir,
                check=False,
                capture_output=True,
            )
            (source_dir / "invalid.py").write_text("def broken(:\n", encoding="utf-8")
            invalid_result = subprocess.run(
                validate_project.build_command(suite),
                cwd=temp_dir,
                check=False,
                capture_output=True,
            )

        self.assertEqual(0, valid_result.returncode)
        self.assertNotEqual(0, invalid_result.returncode)

    def test_dependency_suites_use_each_virtualenv_pip_check(self) -> None:
        root = Path("repo")
        suites = {suite.name: suite for suite in validate_project.build_suites(root, platform="win32")}

        for suite_name, system_dir in (
            ("ensemble-dependencies", "ensemble_forecast"),
            ("rolling-dependencies", "rolling_predict_LSTM"),
        ):
            with self.subTest(suite=suite_name):
                self.assertEqual(
                    [
                        str(root / system_dir / ".venv" / "Scripts" / "python.exe"),
                        "-W",
                        "error::FutureWarning",
                        "-m",
                        "pip",
                        "check",
                    ],
                    validate_project.build_command(suites[suite_name]),
                )

    def test_canonical_data_suite_validates_tracked_manifest(self) -> None:
        root = Path("repo")
        suite = next(
            suite for suite in validate_project.build_suites(root, platform="win32") if suite.name == "canonical-data"
        )

        command = validate_project.build_command(suite)

        self.assertEqual(
            [
                str(root / "ensemble_forecast" / ".venv" / "Scripts" / "python.exe"),
                "-W",
                "error::FutureWarning",
                "-m",
                "data_preprocessing.canonical_data_contract",
                str(root / "data"),
                "--require-manifest",
            ],
            command,
        )

    def test_build_command_enables_future_warning_failures_by_default(self) -> None:
        suite = validate_project.ValidationSuite.unit_tests(
            "rolling",
            Path("python"),
            Path("rolling_predict_LSTM/tests"),
        )

        command = validate_project.build_command(suite)

        self.assertEqual(
            [
                "python",
                "-W",
                "error::FutureWarning",
                "-m",
                "unittest",
                "discover",
                "-s",
                str(Path("rolling_predict_LSTM/tests")),
                "-v",
            ],
            command,
        )

    def test_run_suites_runs_all_selected_suites_and_returns_failure(self) -> None:
        suites = (
            validate_project.ValidationSuite.unit_tests("one", Path("python-one"), Path("tests/one")),
            validate_project.ValidationSuite.unit_tests("two", Path("python-two"), Path("tests/two")),
        )
        runner = Mock(
            side_effect=[
                subprocess.CompletedProcess([], 0),
                subprocess.CompletedProcess([], 1),
            ]
        )

        with redirect_stdout(StringIO()):
            exit_code = validate_project.run_suites(suites, Path("repo"), runner=runner)

        self.assertEqual(1, exit_code)
        self.assertEqual(2, runner.call_count)

    def test_select_suites_filters_deduplicates_and_preserves_defined_order(self) -> None:
        suites = tuple(
            validate_project.ValidationSuite.unit_tests(name, Path(f"python-{name}"), Path(f"tests/{name}"))
            for name in ("one", "two", "three")
        )

        selected = validate_project.select_suites(suites, ["three", "one", "three"])

        self.assertEqual(["one", "three"], [suite.name for suite in selected])

    def test_main_returns_two_without_running_when_selected_interpreter_is_missing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            missing_python = Path(temp_dir) / "missing-python"
            suites = (
                validate_project.ValidationSuite.unit_tests("rolling", missing_python, Path(temp_dir) / "tests"),
            )
            with (
                patch.object(validate_project, "build_suites", return_value=suites),
                patch.object(validate_project, "run_suites") as run_suites,
                redirect_stderr(StringIO()),
            ):
                exit_code = validate_project.main(["--suite", "rolling"])

        self.assertEqual(2, exit_code)
        run_suites.assert_not_called()


if __name__ == "__main__":
    unittest.main()
