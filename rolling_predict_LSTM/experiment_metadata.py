"""Shared provenance writer for Rolling LSTM research runners."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def write_rolling_run_config(
    output_dir: str | Path,
    run_config: Mapping[str, Any],
    *,
    experiment_family: str,
    evidence_tier: str,
    selection_protocol: str,
    report_ready: bool,
    report_ready_reason: str,
    registry_notes: str = "",
) -> dict[str, Any]:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from forecast_benchmark.experiment_registry import (
        enrich_run_config_from_args,
        write_run_config_and_registry,
    )

    args = argparse.Namespace(
        output_dir=Path(output_dir),
        stock_limit=None,
        skip_ensemble=False,
        evidence_tier=str(evidence_tier),
        report_ready=bool(report_ready),
        registry_notes=str(registry_notes),
        selection_protocol=str(selection_protocol),
    )
    enriched = enrich_run_config_from_args(
        run_config,
        args,
        experiment_family=experiment_family,
        report_ready_reason=report_ready_reason,
        data_dir="data",
        project_root=PROJECT_ROOT,
    )
    write_run_config_and_registry(output_dir, enriched)
    return enriched
