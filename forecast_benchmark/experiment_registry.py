"""Experiment metadata helpers for benchmark outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def _display_path(path: str | Path, project_root: Path = PROJECT_ROOT) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit(project_root: Path = PROJECT_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def get_git_is_dirty(project_root: Path = PROJECT_ROOT) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def load_data_manifest_summary(
    data_dir: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    resolved_data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if not resolved_data_dir.is_absolute():
        resolved_data_dir = project_root / resolved_data_dir

    manifest_path = resolved_data_dir / "manifest.json"
    summary: dict[str, Any] = {
        "data_dir": _display_path(resolved_data_dir, project_root),
        "manifest_present": manifest_path.is_file(),
    }
    if not manifest_path.is_file():
        return summary

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    dataset_hashes: dict[str, str] = {}
    missing_dataset_files: list[str] = []
    for dataset_name, filename in dict(manifest.get("files", {})).items():
        dataset_path = resolved_data_dir / str(filename)
        if dataset_path.is_file():
            dataset_hashes[str(dataset_name)] = _hash_file(dataset_path)
        else:
            missing_dataset_files.append(str(filename))
    declared_hashes = {
        str(key): str(value)
        for key, value in dict(manifest.get("file_sha256", {})).items()
    }
    hash_comparison_keys = set(declared_hashes).union(dataset_hashes)
    dataset_hashes_match_manifest = (
        all(dataset_hashes.get(key) == declared_hashes.get(key) for key in hash_comparison_keys)
        if declared_hashes
        else None
    )
    combined_digest = hashlib.sha256(
        json.dumps(dataset_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary.update(
        {
            "manifest_path": _display_path(manifest_path, project_root),
            "manifest_sha256": _hash_file(manifest_path),
            "data_contract_version": manifest.get("data_contract_version"),
            "generated_at_utc": manifest.get("generated_at_utc"),
            "generator": manifest.get("generator"),
            "row_counts": manifest.get("row_counts", {}),
            "stock_counts": manifest.get("stock_counts", {}),
            "dataset_file_sha256": dataset_hashes,
            "declared_file_sha256": declared_hashes,
            "dataset_hashes_match_manifest": dataset_hashes_match_manifest,
            "missing_dataset_files": missing_dataset_files,
            "dataset_bundle_sha256": combined_digest,
        }
    )
    return summary


def normalize_command(command: str | Sequence[str] | None = None) -> dict[str, Any]:
    if command is None:
        main_module = sys.modules.get("__main__")
        main_spec = getattr(main_module, "__spec__", None)
        module_name = getattr(main_spec, "name", None)
        if module_name:
            args = [sys.executable, "-m", module_name, *sys.argv[1:]]
        else:
            args = [sys.executable, *sys.argv]
        return {"command": " ".join(args), "command_args": args}
    if isinstance(command, str):
        return {"command": command, "command_args": None}
    args = [str(part) for part in command]
    return {"command": " ".join(args), "command_args": args}


def build_experiment_registry_entry(
    *,
    experiment_family: str,
    output_dir: str | Path,
    evidence_tier: str,
    report_ready: bool,
    report_ready_reason: str = "",
    command: str | Sequence[str] | None = None,
    data_dir: str | Path | None = None,
    created_at_utc: str | None = None,
    git_commit: str | None = None,
    git_is_dirty: bool | None = None,
    project_root: Path = PROJECT_ROOT,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_slug = output_path.name or experiment_family
    command_info = normalize_command(command)
    entry = {
        "experiment_id": f"{experiment_family}:{output_slug}",
        "experiment_family": experiment_family,
        "created_at_utc": created_at_utc or _utc_now(),
        "evidence_tier": str(evidence_tier),
        "report_ready": bool(report_ready),
        "report_ready_reason": str(report_ready_reason),
        "output_dir": _display_path(output_path, project_root),
        "git_commit": git_commit if git_commit is not None else get_git_commit(project_root),
        "git_is_dirty": (
            bool(git_is_dirty) if git_is_dirty is not None else get_git_is_dirty(project_root)
        ),
        "data": load_data_manifest_summary(data_dir=data_dir, project_root=project_root),
        **command_info,
    }
    if extra:
        entry["extra"] = dict(extra)
    return entry


def enrich_run_config_with_registry(
    run_config: Mapping[str, Any],
    *,
    experiment_family: str,
    output_dir: str | Path,
    evidence_tier: str,
    report_ready: bool,
    report_ready_reason: str = "",
    command: str | Sequence[str] | None = None,
    data_dir: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry_entry = build_experiment_registry_entry(
        experiment_family=experiment_family,
        output_dir=output_dir,
        evidence_tier=evidence_tier,
        report_ready=report_ready,
        report_ready_reason=report_ready_reason,
        command=command,
        data_dir=data_dir,
        project_root=project_root,
        extra=extra,
    )
    enriched = dict(run_config)
    enriched["experiment_id"] = registry_entry["experiment_id"]
    enriched["evidence_tier"] = registry_entry["evidence_tier"]
    enriched["report_ready"] = registry_entry["report_ready"]
    enriched["report_ready_reason"] = registry_entry["report_ready_reason"]
    enriched["experiment_registry"] = registry_entry
    return enriched


def parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected one of: true, false, yes, no, 1, 0.")


def add_registry_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--evidence-tier",
        help="Experiment evidence tier. Defaults to A for full runs and B for smoke/diagnostic runs.",
    )
    parser.add_argument(
        "--report-ready",
        type=parse_optional_bool,
        help="Whether this run is intended as report-ready evidence.",
    )
    parser.add_argument(
        "--registry-notes",
        default="",
        help="Short notes stored in the experiment registry metadata.",
    )
    parser.add_argument(
        "--selection-protocol",
        choices=(
            "fixed-before-target",
            "historical-validation",
            "target-year-hindsight",
            "diagnostic",
            "unspecified",
        ),
        default="unspecified",
        help=(
            "How model/parameter choices were made. Report-ready runs require "
            "fixed-before-target or historical-validation."
        ),
    )
    return parser


def default_report_ready_from_args(args: argparse.Namespace) -> bool:
    return getattr(args, "stock_limit", None) is None and not bool(getattr(args, "skip_ensemble", False))


def default_evidence_tier_from_args(args: argparse.Namespace, report_ready: bool) -> str:
    evidence_tier = getattr(args, "evidence_tier", None)
    if evidence_tier:
        return str(evidence_tier)
    return "A" if report_ready else "B"


def default_report_ready_reason(report_ready: bool) -> str:
    if report_ready:
        return "Full benchmark run with no stock-limit smoke cap."
    return "Limited, skipped, or diagnostic benchmark run."


def registry_data_dir_from_environment() -> str:
    return os.environ.get("PREDICT_DATA_DIR") or "data"


def enrich_run_config_from_args(
    run_config: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    experiment_family: str,
    report_ready_reason: str | None = None,
    command: str | Sequence[str] | None = None,
    data_dir: str | Path | None = None,
    project_root: Path = PROJECT_ROOT,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    requested_report_ready = getattr(args, "report_ready", None)
    if requested_report_ready is None:
        requested_report_ready = default_report_ready_from_args(args)
    selection_protocol = str(getattr(args, "selection_protocol", "unspecified"))
    report_ready_eligible = selection_protocol in {
        "fixed-before-target",
        "historical-validation",
    }
    report_ready = bool(requested_report_ready) and report_ready_eligible
    evidence_tier = default_evidence_tier_from_args(args, report_ready)
    if not report_ready_eligible and str(evidence_tier).upper() == "A":
        evidence_tier = "C" if selection_protocol == "target-year-hindsight" else "B"
    effective_reason = report_ready_reason or default_report_ready_reason(report_ready)
    if bool(requested_report_ready) and not report_ready_eligible:
        effective_reason = (
            f"Not report-ready: selection_protocol={selection_protocol!r} is not an "
            f"independent pre-target selection protocol. {effective_reason}"
        )
    registry_extra = {
        "registry_notes": str(getattr(args, "registry_notes", "")),
        "selection_protocol": selection_protocol,
        "report_ready_requested": bool(requested_report_ready),
        "report_ready_eligible": report_ready_eligible,
    }
    if extra:
        registry_extra.update(dict(extra))
    run_config_with_selection = dict(run_config)
    run_config_with_selection.update(
        {
            "selection_protocol": selection_protocol,
            "report_ready_requested": bool(requested_report_ready),
            "report_ready_eligible": report_ready_eligible,
        }
    )
    return enrich_run_config_with_registry(
        run_config_with_selection,
        experiment_family=experiment_family,
        output_dir=getattr(args, "output_dir"),
        evidence_tier=evidence_tier,
        report_ready=bool(report_ready),
        report_ready_reason=effective_reason,
        command=command,
        data_dir=data_dir if data_dir is not None else registry_data_dir_from_environment(),
        project_root=project_root,
        extra=registry_extra,
    )


def write_registry_entry(output_dir: str | Path, registry_entry: Mapping[str, Any]) -> Path:
    path = Path(output_dir) / "experiment_registry_entry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(dict(registry_entry), stream, ensure_ascii=False, indent=2)
    return path


def write_run_config_and_registry(output_dir: str | Path, run_config: Mapping[str, Any]) -> Path:
    path = Path(output_dir) / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(dict(run_config), stream, ensure_ascii=False, indent=2)
    registry_entry = run_config.get("experiment_registry")
    if isinstance(registry_entry, Mapping):
        write_registry_entry(output_dir, registry_entry)
    return path
