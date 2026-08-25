"""Canonical CSV data contract helpers for forecasting datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CONTRACT_VERSION = "canonical-csv-v1"
MANIFEST_FILENAME = "manifest.json"

CANONICAL_FILENAMES = {
    "stock_list": "stock_list_new.csv",
    "revenue": "Stock_revenue_2019~2025.csv",
    "target_stocks": "target_stocks_2025.csv",
    "eps": "EPS2020~2025.csv",
    "dividends": "Dividend2019~2025.csv",
    "daily_prices": "day K2020~2025.csv",
}

REVENUE_UNIT_CONTRACT = {
    "revenue": "raw NTD",
    "revenue_thousand": "thousand NTD",
    "last_year_revenue": "raw NTD",
    "last_3m_revenue": "raw NTD",
    "last_12m_revenue": "raw NTD",
    "acc_revenue": "raw NTD",
    "mom": "decimal ratio",
    "yoy": "decimal ratio",
    "last_3m_revenue_yoy": "decimal ratio",
    "last_12m_revenue_yoy": "decimal ratio",
    "acc_revenue_yoy": "decimal ratio",
}


@dataclass(frozen=True)
class CanonicalDatasetContract:
    name: str
    filename: str
    required_columns: tuple[str, ...]
    stock_id_column: str | None = "stock_id"


DATASET_CONTRACTS = {
    "stock_list": CanonicalDatasetContract(
        name="stock_list",
        filename=CANONICAL_FILENAMES["stock_list"],
        required_columns=("stock_id", "stock_name", "industry_category"),
    ),
    "revenue": CanonicalDatasetContract(
        name="revenue",
        filename=CANONICAL_FILENAMES["revenue"],
        required_columns=(
            "date",
            "stock_id",
            "revenue_year",
            "revenue_month",
            "revenue",
            "revenue_thousand",
            "mom",
            "last_year_revenue",
            "yoy",
            "last_3m_revenue",
            "last_3m_revenue_yoy",
            "last_12m_revenue",
            "last_12m_revenue_yoy",
            "acc_revenue",
            "acc_revenue_yoy",
            "industry_category",
        ),
    ),
    "target_stocks": CanonicalDatasetContract(
        name="target_stocks",
        filename=CANONICAL_FILENAMES["target_stocks"],
        required_columns=("date", "stock_id", "country", "revenue", "revenue_month", "revenue_year"),
    ),
    "eps": CanonicalDatasetContract(
        name="eps",
        filename=CANONICAL_FILENAMES["eps"],
        required_columns=("date", "stock_id", "EPS"),
    ),
    "dividends": CanonicalDatasetContract(
        name="dividends",
        filename=CANONICAL_FILENAMES["dividends"],
        required_columns=("stock_id", "year", "TotalCashDividend"),
    ),
    "daily_prices": CanonicalDatasetContract(
        name="daily_prices",
        filename=CANONICAL_FILENAMES["daily_prices"],
        required_columns=("date", "stock_id", "open", "max", "min", "close"),
    ),
}

TIME_SAFE_OPTIONAL_COLUMNS = {
    "revenue": ("revenue_available_date",),
    "eps": ("statement_available_date",),
    "dividends": ("AnnouncementDate", "DividendAvailableDate", "dividend_available_source"),
}


@dataclass(frozen=True)
class CanonicalValidationResult:
    data_dir: str | None
    row_counts: dict[str, int]
    stock_counts: dict[str, int]
    issues: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, object]:
        return {
            "data_dir": self.data_dir,
            "is_valid": self.is_valid,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "row_counts": self.row_counts,
            "stock_counts": self.stock_counts,
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _stock_count(frame: pd.DataFrame, stock_id_column: str | None = "stock_id") -> int:
    if not stock_id_column or stock_id_column not in frame.columns or frame.empty:
        return 0
    return int(pd.to_numeric(frame[stock_id_column], errors="coerce").dropna().nunique())


def _validate_revenue_frame(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    keys = ["stock_id", "revenue_year", "revenue_month"]
    if not set(keys).issubset(frame.columns):
        return issues, warnings

    key_frame = frame[keys].apply(pd.to_numeric, errors="coerce")
    if key_frame.isna().any(axis=None):
        issues.append("revenue: null or non-numeric stock/year/month keys")
        return issues, warnings
    if not key_frame["revenue_month"].between(1, 12).all():
        issues.append("revenue: revenue_month must be between 1 and 12")
    duplicate_count = int(key_frame.duplicated(keys).sum())
    if duplicate_count:
        issues.append(f"revenue: {duplicate_count} duplicate stock/year/month rows")

    ordered = key_frame.sort_values(keys).copy()
    month_index = ordered["revenue_year"] * 12 + ordered["revenue_month"]
    month_gap = month_index.groupby(ordered["stock_id"]).diff()
    interior_gap_count = int(month_gap.gt(1).sum())
    if interior_gap_count:
        affected_stocks = int(ordered.loc[month_gap.gt(1), "stock_id"].nunique())
        warnings.append(
            f"revenue: {interior_gap_count} interior calendar gaps across {affected_stocks} stocks"
        )

    if {"revenue", "revenue_thousand"}.issubset(frame.columns):
        revenue = pd.to_numeric(frame["revenue"], errors="coerce")
        revenue_thousand = pd.to_numeric(frame["revenue_thousand"], errors="coerce")
        valid = revenue.notna() & revenue_thousand.notna()
        tolerance = revenue.abs().clip(lower=1.0) * 1e-9
        mismatch = valid & (revenue_thousand * 1000.0 - revenue).abs().gt(tolerance)
        if mismatch.any():
            issues.append(
                f"revenue: {int(mismatch.sum())} rows violate revenue_thousand = revenue / 1000"
            )
    return issues, warnings


def validate_canonical_frames(
    frames: Mapping[str, pd.DataFrame],
    *,
    minimum_stock_counts: Mapping[str, int] | None = None,
) -> CanonicalValidationResult:
    minimum_stock_counts = minimum_stock_counts or {}
    issues: list[str] = []
    warnings: list[str] = []
    row_counts: dict[str, int] = {}
    stock_counts: dict[str, int] = {}

    for dataset, contract in DATASET_CONTRACTS.items():
        frame = frames.get(dataset)
        if frame is None:
            issues.append(f"{dataset}: missing frame")
            row_counts[dataset] = 0
            stock_counts[dataset] = 0
            continue

        row_counts[dataset] = int(len(frame))
        stock_counts[dataset] = _stock_count(frame, contract.stock_id_column)

        missing_columns = [column for column in contract.required_columns if column not in frame.columns]
        if missing_columns:
            issues.append(f"{dataset}: missing required columns {missing_columns}")

        optional_columns = TIME_SAFE_OPTIONAL_COLUMNS.get(dataset, ())
        missing_optional = [column for column in optional_columns if column not in frame.columns]
        if missing_optional:
            warnings.append(f"{dataset}: missing time-safe columns {missing_optional}")

        minimum_stock_count = minimum_stock_counts.get(dataset)
        if minimum_stock_count is not None and stock_counts[dataset] < int(minimum_stock_count):
            issues.append(
                f"{dataset}: stock_count {stock_counts[dataset]} below minimum {int(minimum_stock_count)}"
            )
        if dataset == "revenue":
            revenue_issues, revenue_warnings = _validate_revenue_frame(frame)
            issues.extend(revenue_issues)
            warnings.extend(revenue_warnings)

    return CanonicalValidationResult(
        data_dir=None,
        row_counts=row_counts,
        stock_counts=stock_counts,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def _read_stock_id_only(path: Path, contract: CanonicalDatasetContract) -> tuple[list[str], int, int]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [contract.stock_id_column] if contract.stock_id_column in columns else None
    frame = pd.read_csv(path, usecols=usecols)
    return columns, int(len(frame)), _stock_count(frame, contract.stock_id_column)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_canonical_data_dir(
    data_dir: Path,
    *,
    minimum_stock_counts: Mapping[str, int] | None = None,
    require_manifest: bool = False,
) -> CanonicalValidationResult:
    minimum_stock_counts = minimum_stock_counts or {}
    data_dir = Path(data_dir)
    issues: list[str] = []
    warnings: list[str] = []
    row_counts: dict[str, int] = {}
    stock_counts: dict[str, int] = {}

    manifest_path = data_dir / MANIFEST_FILENAME
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                loaded_manifest = json.load(stream)
            if isinstance(loaded_manifest, dict):
                manifest = loaded_manifest
            else:
                issues.append("manifest: root must be a JSON object")
        except (OSError, json.JSONDecodeError) as error:
            issues.append(f"manifest: unreadable {type(error).__name__}: {error}")
    elif require_manifest:
        issues.append(f"manifest: missing {MANIFEST_FILENAME}")
    else:
        warnings.append(f"manifest: missing {MANIFEST_FILENAME}")

    manifest_files = manifest.get("files", {})
    if not isinstance(manifest_files, Mapping):
        issues.append("manifest: files must be an object")
        manifest_files = {}
    filenames = dict(CANONICAL_FILENAMES)
    for dataset, filename in manifest_files.items():
        if dataset not in filenames or not isinstance(filename, str):
            continue
        if Path(filename).is_absolute() or Path(filename).name != filename:
            issues.append(f"manifest: unsafe filename for {dataset}: {filename!r}")
            continue
        filenames[dataset] = filename

    for dataset, contract in DATASET_CONTRACTS.items():
        path = data_dir / filenames[dataset]
        if not path.is_file():
            issues.append(f"{dataset}: missing file {filenames[dataset]}")
            row_counts[dataset] = 0
            stock_counts[dataset] = 0
            continue

        columns, row_count, stock_count = _read_stock_id_only(path, contract)
        row_counts[dataset] = row_count
        stock_counts[dataset] = stock_count

        missing_columns = [column for column in contract.required_columns if column not in columns]
        if missing_columns:
            issues.append(f"{dataset}: missing required columns {missing_columns}")

        optional_columns = TIME_SAFE_OPTIONAL_COLUMNS.get(dataset, ())
        missing_optional = [column for column in optional_columns if column not in columns]
        if missing_optional:
            warnings.append(f"{dataset}: missing time-safe columns {missing_optional}")

        minimum_stock_count = minimum_stock_counts.get(dataset)
        if minimum_stock_count is not None and stock_count < int(minimum_stock_count):
            issues.append(f"{dataset}: stock_count {stock_count} below minimum {int(minimum_stock_count)}")

    if manifest:
        manifest_version = manifest.get("data_contract_version")
        if manifest_version != DATA_CONTRACT_VERSION:
            issues.append(
                f"manifest: data_contract_version {manifest_version!r} does not match {DATA_CONTRACT_VERSION!r}"
            )
        for count_name, observed_counts in [
            ("row_counts", row_counts),
            ("stock_counts", stock_counts),
        ]:
            expected_counts = manifest.get(count_name, {})
            if not isinstance(expected_counts, Mapping):
                continue
            singular = "row_count" if count_name == "row_counts" else "stock_count"
            for dataset, expected in expected_counts.items():
                if dataset not in observed_counts:
                    continue
                try:
                    expected_value = int(expected)
                except (TypeError, ValueError):
                    issues.append(f"manifest {singular}: invalid {dataset} value {expected!r}")
                    continue
                if observed_counts[dataset] != expected_value:
                    issues.append(
                        f"manifest {singular}: {dataset} expected {expected_value}, "
                        f"observed {observed_counts[dataset]}"
                    )

        expected_hashes = manifest.get("file_sha256", {})
        if isinstance(expected_hashes, Mapping):
            for dataset, expected_digest in expected_hashes.items():
                if dataset not in filenames or not isinstance(expected_digest, str):
                    continue
                path = data_dir / filenames[dataset]
                if path.is_file():
                    observed_digest = _hash_file(path)
                    if observed_digest != expected_digest:
                        issues.append(
                            f"manifest sha256: {dataset} expected {expected_digest}, "
                            f"observed {observed_digest}"
                        )

    return CanonicalValidationResult(
        data_dir=str(data_dir),
        row_counts=row_counts,
        stock_counts=stock_counts,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def build_stock_coverage(stock_counts: Mapping[str, int]) -> dict[str, dict[str, float | int]]:
    universe_count = int(stock_counts.get("stock_list", 0))
    coverage: dict[str, dict[str, float | int]] = {}
    for dataset, stock_count in stock_counts.items():
        ratio = float(stock_count / universe_count) if universe_count else 0.0
        coverage[dataset] = {
            "stock_count": int(stock_count),
            "stock_list_coverage_ratio": ratio,
        }
    return coverage


def build_canonical_manifest(
    *,
    config: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    generator: str,
    generated_at_utc: datetime | None = None,
    filenames: Mapping[str, str] | None = None,
    file_sha256: Mapping[str, str] | None = None,
) -> dict[str, object]:
    generated_at_utc = generated_at_utc or datetime.now(UTC)
    validation = validate_canonical_frames(frames)
    manifest = {
        "data_contract_version": DATA_CONTRACT_VERSION,
        "dataset_role": "canonical_generated_csv",
        "generator": generator,
        "generated_at_utc": generated_at_utc.isoformat().replace("+00:00", "Z"),
        "config": _json_safe(dict(config)),
        "files": dict(filenames or CANONICAL_FILENAMES),
        "revenue_unit_contract": REVENUE_UNIT_CONTRACT,
        "row_counts": validation.row_counts,
        "stock_counts": validation.stock_counts,
        "stock_coverage": build_stock_coverage(validation.stock_counts),
        "validation": validation.as_dict(),
    }
    if file_sha256 is not None:
        manifest["file_sha256"] = dict(file_sha256)
    return manifest


def _parse_minimum_stock_counts(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    result: dict[str, int] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        dataset, _, value = item.partition("=")
        if not dataset or not value:
            raise ValueError("Minimum stock counts must use dataset=count pairs.")
        result[dataset.strip()] = int(value)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Canonical data directory to validate.")
    parser.add_argument(
        "--minimum-stock-counts",
        help="Comma-separated dataset=count thresholds, for example revenue=1900,daily_prices=1900.",
    )
    parser.add_argument("--require-manifest", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = validate_canonical_data_dir(
        args.data_dir,
        minimum_stock_counts=_parse_minimum_stock_counts(args.minimum_stock_counts),
        require_manifest=args.require_manifest,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    if not result.is_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
