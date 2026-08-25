from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidationSuite:
    name: str
    python_path: Path
    arguments: tuple[str, ...]

    @classmethod
    def unit_tests(cls, name: str, python_path: Path, test_dir: Path) -> ValidationSuite:
        return cls(
            name=name,
            python_path=python_path,
            arguments=("-m", "unittest", "discover", "-s", str(test_dir), "-v"),
        )

    @classmethod
    def dependency_check(cls, name: str, python_path: Path) -> ValidationSuite:
        return cls(name=name, python_path=python_path, arguments=("-m", "pip", "check"))

    @classmethod
    def source_compile(cls, name: str, python_path: Path, source_dirs: Sequence[Path]) -> ValidationSuite:
        return cls(
            name=name,
            python_path=python_path,
            arguments=(
                "-m",
                "compileall",
                "-q",
                "-f",
                "-x",
                r"(?:^|[\\/])\.venv(?:[\\/]|$)",
                *(str(source_dir) for source_dir in source_dirs),
            ),
        )


def venv_python(venv_dir: Path, platform: str | None = None) -> Path:
    platform = sys.platform if platform is None else platform
    if platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def build_suites(root: Path, platform: str | None = None) -> tuple[ValidationSuite, ...]:
    ensemble_python = venv_python(root / "ensemble_forecast" / ".venv", platform)
    rolling_python = venv_python(root / "rolling_predict_LSTM" / ".venv", platform)
    return (
        ValidationSuite.source_compile(
            "source-compile",
            ensemble_python,
            tuple(
                root / source_dir
                for source_dir in (
                    "data_preprocessing",
                    "ensemble_forecast",
                    "financial_forecast",
                    "forecast_benchmark",
                    "rolling_predict_LSTM",
                    "tools",
                )
            ),
        ),
        ValidationSuite.unit_tests("tooling", ensemble_python, root / "tools" / "tests"),
        ValidationSuite(
            "canonical-data",
            ensemble_python,
            ("-m", "data_preprocessing.canonical_data_contract", str(root / "data"), "--require-manifest"),
        ),
        ValidationSuite.dependency_check("ensemble-dependencies", ensemble_python),
        ValidationSuite.dependency_check("rolling-dependencies", rolling_python),
        ValidationSuite.unit_tests("data", ensemble_python, root / "data_preprocessing" / "tests"),
        ValidationSuite.unit_tests("ensemble", ensemble_python, root / "ensemble_forecast" / "tests"),
        ValidationSuite.unit_tests("financial", ensemble_python, root / "financial_forecast" / "tests"),
        ValidationSuite.unit_tests("benchmark", ensemble_python, root / "forecast_benchmark" / "tests"),
        ValidationSuite.unit_tests("rolling", rolling_python, root / "rolling_predict_LSTM" / "tests"),
    )


def build_command(suite: ValidationSuite, *, strict_future_warnings: bool = True) -> list[str]:
    command = [str(suite.python_path)]
    if strict_future_warnings:
        command.extend(["-W", "error::FutureWarning"])
    command.extend(suite.arguments)
    return command


Runner = Callable[..., subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]]


def run_suites(
    suites: Iterable[ValidationSuite],
    root: Path,
    *,
    strict_future_warnings: bool = True,
    runner: Runner = subprocess.run,
) -> int:
    failed: list[str] = []
    for suite in suites:
        print(f"\n=== {suite.name} ===", flush=True)
        completed = runner(
            build_command(suite, strict_future_warnings=strict_future_warnings),
            cwd=root,
            check=False,
        )
        if completed.returncode != 0:
            failed.append(suite.name)
    if failed:
        print(f"\nValidation failed: {', '.join(failed)}", flush=True)
        return 1
    print("\nAll selected validation suites passed.", flush=True)
    return 0


def select_suites(suites: Sequence[ValidationSuite], names: Sequence[str] | None) -> tuple[ValidationSuite, ...]:
    if not names:
        return tuple(suites)
    selected = set(names)
    return tuple(suite for suite in suites if suite.name in selected)


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    suites = build_suites(root)
    parser = argparse.ArgumentParser(description="Run all repository validation suites with the correct virtualenv.")
    parser.add_argument(
        "--suite",
        action="append",
        choices=[suite.name for suite in suites],
        help="Run only the named suite; repeat the option to select multiple suites.",
    )
    parser.add_argument(
        "--allow-future-warnings",
        action="store_true",
        help="Do not promote FutureWarning messages to test failures.",
    )
    args = parser.parse_args(argv)
    selected = select_suites(suites, args.suite)
    missing_interpreters = sorted({str(suite.python_path) for suite in selected if not suite.python_path.is_file()})
    if missing_interpreters:
        for interpreter in missing_interpreters:
            print(f"Missing virtualenv interpreter: {interpreter}", file=sys.stderr)
        return 2
    return run_suites(
        selected,
        root,
        strict_future_warnings=not args.allow_future_warnings,
    )


if __name__ == "__main__":
    raise SystemExit(main())
