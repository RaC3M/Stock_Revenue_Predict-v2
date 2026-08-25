from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from forecast_benchmark.experiment_registry import (
    add_registry_arguments,
    build_experiment_registry_entry,
    enrich_run_config_from_args,
    enrich_run_config_with_registry,
    write_run_config_and_registry,
    write_registry_entry,
)


class ExperimentRegistryTests(unittest.TestCase):
    def test_registry_entry_includes_data_manifest_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            data_dir = project_root / "data"
            data_dir.mkdir()
            revenue_bytes = b"stock_id,revenue\n1,100\n"
            (data_dir / "revenue.csv").write_bytes(revenue_bytes)
            revenue_hash = hashlib.sha256(revenue_bytes).hexdigest()
            (data_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "data_contract_version": "canonical-csv-v1",
                        "generated_at_utc": "2026-07-30T12:05:12Z",
                        "generator": "data_preprocessing.free_taiwan_data_preprocessor",
                        "files": {"revenue": "revenue.csv"},
                        "file_sha256": {"revenue": revenue_hash},
                        "row_counts": {"revenue": 10},
                        "stock_counts": {"revenue": 2},
                    }
                ),
                encoding="utf-8",
            )

            entry = build_experiment_registry_entry(
                experiment_family="revenue_benchmark",
                output_dir=project_root / "forecast_benchmark" / "outputs" / "example",
                evidence_tier="A",
                report_ready=True,
                report_ready_reason="full run",
                command=["python", "-m", "forecast_benchmark.run_benchmark"],
                data_dir=data_dir,
                created_at_utc="2026-07-30T13:00:00Z",
                git_commit="abc123",
                git_is_dirty=False,
                project_root=project_root,
            )

        self.assertEqual(entry["experiment_id"], "revenue_benchmark:example")
        self.assertEqual(entry["evidence_tier"], "A")
        self.assertTrue(entry["report_ready"])
        self.assertEqual(entry["git_commit"], "abc123")
        self.assertFalse(entry["git_is_dirty"])
        self.assertEqual(entry["data"]["data_dir"], "data")
        self.assertEqual(entry["data"]["data_contract_version"], "canonical-csv-v1")
        self.assertEqual(entry["data"]["stock_counts"]["revenue"], 2)
        self.assertRegex(entry["data"]["manifest_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(entry["data"]["dataset_file_sha256"]["revenue"], revenue_hash)
        self.assertTrue(entry["data"]["dataset_hashes_match_manifest"])
        self.assertRegex(entry["data"]["dataset_bundle_sha256"], r"^[0-9a-f]{64}$")

    def test_enrich_run_config_copies_registry_summary_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            data_dir = project_root / "data"
            data_dir.mkdir()

            config = enrich_run_config_with_registry(
                {"target_year": 2025},
                experiment_family="revenue_benchmark",
                output_dir=project_root / "outputs" / "smoke",
                evidence_tier="B",
                report_ready=False,
                report_ready_reason="smoke",
                command="python -m forecast_benchmark.run_benchmark --stock-limit 3",
                data_dir=data_dir,
                project_root=project_root,
            )

            written_path = write_registry_entry(
                project_root / "outputs" / "smoke",
                config["experiment_registry"],
            )
            written = json.loads(written_path.read_text(encoding="utf-8"))

        self.assertEqual(config["experiment_id"], "revenue_benchmark:smoke")
        self.assertEqual(config["evidence_tier"], "B")
        self.assertFalse(config["report_ready"])
        self.assertEqual(written["experiment_id"], "revenue_benchmark:smoke")
        self.assertFalse(written["data"]["manifest_present"])

    def test_registry_arguments_enrich_downstream_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            parser = argparse.ArgumentParser()
            parser.add_argument("--output-dir", type=Path, required=True)
            parser.add_argument("--stock-limit", type=int)
            add_registry_arguments(parser)
            args = parser.parse_args(
                [
                    "--output-dir",
                    str(project_root / "outputs" / "yield_smoke"),
                    "--stock-limit",
                    "3",
                    "--registry-notes",
                    "downstream smoke",
                ]
            )

            config = enrich_run_config_from_args(
                {"target_year": 2025},
                args,
                experiment_family="yield_benchmark",
                project_root=project_root,
                extra={"input_predictions": "forecast_benchmark/outputs/example.csv"},
            )

        registry = config["experiment_registry"]
        self.assertEqual(config["experiment_id"], "yield_benchmark:yield_smoke")
        self.assertEqual(config["evidence_tier"], "B")
        self.assertFalse(config["report_ready"])
        self.assertEqual(registry["extra"]["registry_notes"], "downstream smoke")
        self.assertEqual(
            registry["extra"]["input_predictions"],
            "forecast_benchmark/outputs/example.csv",
        )

    def test_target_year_hindsight_cannot_be_marked_report_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            parser = argparse.ArgumentParser()
            parser.add_argument("--output-dir", type=Path, required=True)
            add_registry_arguments(parser)
            args = parser.parse_args(
                [
                    "--output-dir",
                    str(project_root / "outputs" / "hindsight"),
                    "--report-ready",
                    "true",
                    "--evidence-tier",
                    "A",
                    "--selection-protocol",
                    "target-year-hindsight",
                ]
            )

            config = enrich_run_config_from_args(
                {"target_year": 2025},
                args,
                experiment_family="rolling_replay",
                project_root=project_root,
            )

        self.assertFalse(config["report_ready"])
        self.assertFalse(config["report_ready_eligible"])
        self.assertEqual(config["evidence_tier"], "C")
        self.assertIn("Not report-ready", config["report_ready_reason"])

    def test_write_run_config_and_registry_writes_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "outputs" / "run"
            run_config = {
                "target_year": 2025,
                "experiment_registry": {
                    "experiment_id": "eps_benchmark:run",
                    "report_ready": True,
                },
            }

            config_path = write_run_config_and_registry(output_dir, run_config)
            registry_path = output_dir / "experiment_registry_entry.json"

            written_config = json.loads(config_path.read_text(encoding="utf-8"))
            written_registry = json.loads(registry_path.read_text(encoding="utf-8"))

        self.assertEqual(written_config["experiment_registry"]["experiment_id"], "eps_benchmark:run")
        self.assertEqual(written_registry["experiment_id"], "eps_benchmark:run")


if __name__ == "__main__":
    unittest.main()
