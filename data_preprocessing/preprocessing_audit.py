"""Audit canonical CSVs produced by different preprocessing runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocessing.canonical_data_contract import (
    DATASET_CONTRACTS,
    validate_canonical_data_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "data"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "data_preprocessing" / "outputs"
DEFAULT_CANDIDATE_DIR = DEFAULT_ARTIFACT_DIR / "processed"
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACT_DIR / "audit"
DEFAULT_ABS_TOLERANCE = 1e-6
DEFAULT_REL_TOLERANCE = 1e-6
DEFAULT_MIN_COMMON_KEY_COVERAGE = 0.99
DEFAULT_MIN_CANDIDATE_STOCK_COVERAGE = 0.99
DEFAULT_MAX_CRITICAL_NUMERIC_MISMATCH_RATE = 0.01
DEFAULT_CRITICAL_NUMERIC_DATASETS = ("revenue", "daily_prices")


@dataclass(frozen=True)
class PreprocessingAuditConfig:
    baseline_dir: Path = DEFAULT_BASELINE_DIR
    candidate_dir: Path = DEFAULT_CANDIDATE_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    abs_tolerance: float = DEFAULT_ABS_TOLERANCE
    rel_tolerance: float = DEFAULT_REL_TOLERANCE
    min_common_key_coverage: float = DEFAULT_MIN_COMMON_KEY_COVERAGE
    min_candidate_stock_coverage: float = DEFAULT_MIN_CANDIDATE_STOCK_COVERAGE
    max_critical_numeric_mismatch_rate: float = DEFAULT_MAX_CRITICAL_NUMERIC_MISMATCH_RATE
    critical_numeric_datasets: tuple[str, ...] = DEFAULT_CRITICAL_NUMERIC_DATASETS


DATASET_SPECS = {
    "revenue": {
        "filename": "Stock_revenue_2019~2025.csv",
        "keys": ["stock_id", "revenue_year", "revenue_month"],
        "numeric": [
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
        ],
    },
    "target_stocks": {
        "filename": "target_stocks_2025.csv",
        "keys": ["stock_id", "revenue_year", "revenue_month"],
        "numeric": ["revenue"],
    },
    "eps": {
        "filename": "EPS2020~2025.csv",
        "keys": ["stock_id", "date"],
        "numeric": ["EPS"],
    },
    "dividends": {
        "filename": "Dividend2019~2025.csv",
        "keys": ["stock_id", "year"],
        "numeric": ["TotalCashDividend"],
    },
    "daily_prices": {
        "filename": "day K2020~2025.csv",
        "keys": ["stock_id", "date"],
        "numeric": ["open", "max", "min", "close"],
    },
    "stock_list": {
        "filename": "stock_list_new.csv",
        "keys": ["stock_id"],
        "numeric": [],
    },
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _normalize_keys(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for key in keys:
        if key not in result.columns:
            continue
        if key == "date":
            result[key] = pd.to_datetime(result[key], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            numeric = pd.to_numeric(result[key], errors="coerce")
            if numeric.notna().all():
                result[key] = numeric.astype(int).astype(str)
            else:
                result[key] = result[key].astype(str)
    return result


def _stock_count(frame: pd.DataFrame) -> int:
    return int(frame["stock_id"].nunique()) if "stock_id" in frame.columns and not frame.empty else 0


def _key_frame(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty or not set(keys).issubset(frame.columns):
        return pd.DataFrame(columns=keys)
    return _normalize_keys(frame, keys)[keys].drop_duplicates()


def build_coverage_summary(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    dataset: str,
    keys: list[str],
) -> dict[str, object]:
    baseline_keys = _key_frame(baseline, keys)
    candidate_keys = _key_frame(candidate, keys)
    common = baseline_keys.merge(candidate_keys, on=keys, how="inner")
    baseline_only = baseline_keys.merge(candidate_keys, on=keys, how="left", indicator=True)
    candidate_only = candidate_keys.merge(baseline_keys, on=keys, how="left", indicator=True)
    return {
        "dataset": dataset,
        "baseline_rows": int(len(baseline)),
        "candidate_rows": int(len(candidate)),
        "baseline_stock_count": _stock_count(baseline),
        "candidate_stock_count": _stock_count(candidate),
        "baseline_key_count": int(len(baseline_keys)),
        "candidate_key_count": int(len(candidate_keys)),
        "common_key_count": int(len(common)),
        "baseline_only_key_count": int(baseline_only["_merge"].eq("left_only").sum()) if "_merge" in baseline_only else 0,
        "candidate_only_key_count": int(candidate_only["_merge"].eq("left_only").sum()) if "_merge" in candidate_only else 0,
    }


def compare_numeric_columns(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    dataset: str,
    keys: list[str],
    numeric_columns: list[str],
    abs_tolerance: float,
    rel_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if baseline.empty or candidate.empty or not numeric_columns:
        return pd.DataFrame(), pd.DataFrame()
    if not set(keys).issubset(baseline.columns) or not set(keys).issubset(candidate.columns):
        return pd.DataFrame(), pd.DataFrame()

    shared_numeric = [column for column in numeric_columns if column in baseline.columns and column in candidate.columns]
    if not shared_numeric:
        return pd.DataFrame(), pd.DataFrame()

    left = _build_numeric_comparison_frame(baseline, keys, shared_numeric)
    right = _build_numeric_comparison_frame(candidate, keys, shared_numeric)
    merged = left.merge(right, on=[*keys, "_key_ordinal"], how="inner", suffixes=("_baseline", "_candidate"))
    rows = []
    summary_rows = []
    for column in shared_numeric:
        baseline_values = pd.to_numeric(merged[f"{column}_baseline"], errors="coerce")
        candidate_values = pd.to_numeric(merged[f"{column}_candidate"], errors="coerce")
        abs_diff = (candidate_values - baseline_values).abs()
        denom = baseline_values.abs().replace(0, np.nan)
        rel_diff = abs_diff / denom
        mismatch = (
            abs_diff.gt(float(abs_tolerance))
            & rel_diff.fillna(abs_diff).gt(float(rel_tolerance))
            & ~(baseline_values.isna() & candidate_values.isna())
        )
        summary_rows.append(
            {
                "dataset": dataset,
                "column": column,
                "common_rows": int(len(merged)),
                "mismatch_count": int(mismatch.sum()),
                "mismatch_rate": float(mismatch.mean() * 100) if len(merged) else np.nan,
                "max_abs_diff": float(abs_diff.max()) if abs_diff.notna().any() else np.nan,
                "mean_abs_diff": float(abs_diff.mean()) if abs_diff.notna().any() else np.nan,
            }
        )
        if mismatch.any():
            detail = merged.loc[mismatch, keys].copy()
            detail["dataset"] = dataset
            detail["column"] = column
            detail["baseline_value"] = baseline_values.loc[mismatch].to_numpy()
            detail["candidate_value"] = candidate_values.loc[mismatch].to_numpy()
            detail["abs_diff"] = abs_diff.loc[mismatch].to_numpy()
            detail["rel_diff"] = rel_diff.loc[mismatch].to_numpy()
            rows.append(detail)

    differences = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return summary, differences


def _build_numeric_comparison_frame(
    frame: pd.DataFrame,
    keys: list[str],
    numeric_columns: list[str],
) -> pd.DataFrame:
    result = _normalize_keys(frame[[*keys, *numeric_columns]].copy(), keys)
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.sort_values([*keys, *numeric_columns], kind="mergesort", na_position="last").reset_index(drop=True)
    result["_key_ordinal"] = result.groupby(keys).cumcount()
    return result


def build_column_presence_summary(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    columns = sorted(set(baseline.columns) | set(candidate.columns))
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "column": column,
                "in_baseline": column in baseline.columns,
                "in_candidate": column in candidate.columns,
            }
            for column in columns
        ]
    )


def _coverage_ratio(numerator: object, denominator: object) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0:
        return 1.0
    return float(numerator) / denominator_value


def _status(observed: float, threshold: float, direction: str) -> str:
    if direction == "min":
        return "pass" if observed >= threshold else "fail"
    if direction == "max":
        return "pass" if observed <= threshold else "fail"
    raise ValueError(f"Unsupported threshold direction: {direction}")


def build_replacement_readiness_summary(
    results: dict[str, pd.DataFrame],
    config: PreprocessingAuditConfig,
) -> pd.DataFrame:
    """Summarize whether a candidate can replace the baseline canonical data dir."""

    rows: list[dict[str, object]] = []
    coverage = results["coverage_summary"]
    for _, row in coverage.iterrows():
        dataset = str(row["dataset"])
        key_coverage = _coverage_ratio(row["common_key_count"], row["baseline_key_count"])
        stock_coverage = _coverage_ratio(row["candidate_stock_count"], row["baseline_stock_count"])
        rows.append(
            {
                "dataset": dataset,
                "check": "common_key_coverage",
                "observed": key_coverage,
                "threshold": float(config.min_common_key_coverage),
                "status": _status(key_coverage, float(config.min_common_key_coverage), "min"),
                "message": (
                    f"{int(row['common_key_count'])} common keys out of "
                    f"{int(row['baseline_key_count'])} baseline keys"
                ),
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "check": "candidate_stock_coverage",
                "observed": stock_coverage,
                "threshold": float(config.min_candidate_stock_coverage),
                "status": _status(stock_coverage, float(config.min_candidate_stock_coverage), "min"),
                "message": (
                    f"{int(row['candidate_stock_count'])} candidate stocks vs "
                    f"{int(row['baseline_stock_count'])} baseline stocks"
                ),
            }
        )

    numeric_summary = results["numeric_diff_summary"]
    if not numeric_summary.empty:
        critical_datasets = set(config.critical_numeric_datasets)
        critical_numeric = numeric_summary[numeric_summary["dataset"].isin(critical_datasets)]
        for _, row in critical_numeric.iterrows():
            mismatch_rate = float(row["mismatch_rate"]) / 100.0
            rows.append(
                {
                    "dataset": str(row["dataset"]),
                    "check": f"{row['column']}_numeric_mismatch_rate",
                    "observed": mismatch_rate,
                    "threshold": float(config.max_critical_numeric_mismatch_rate),
                    "status": _status(
                        mismatch_rate,
                        float(config.max_critical_numeric_mismatch_rate),
                        "max",
                    ),
                    "message": (
                        f"{int(row['mismatch_count'])} mismatches out of "
                        f"{int(row['common_rows'])} common rows"
                    ),
                }
            )

    column_presence = results.get("column_presence_summary", pd.DataFrame())
    if not column_presence.empty:
        for dataset, contract in DATASET_CONTRACTS.items():
            required = set(contract.required_columns)
            missing_required = column_presence[
                column_presence["dataset"].eq(dataset)
                & column_presence["column"].isin(required)
                & column_presence["in_baseline"].astype(bool)
                & ~column_presence["in_candidate"].astype(bool)
            ]
            for column in sorted(missing_required["column"].astype(str).tolist()):
                rows.append(
                    {
                        "dataset": dataset,
                        "check": f"required_column:{column}",
                        "observed": 0.0,
                        "threshold": 1.0,
                        "status": "fail",
                        "message": f"candidate is missing required baseline column {column}",
                    }
                )

    contract_summary = results.get("candidate_contract_summary", pd.DataFrame())
    if not contract_summary.empty:
        for _, row in contract_summary.iterrows():
            rows.append(
                {
                    "dataset": str(row["dataset"]),
                    "check": f"canonical_contract:{row['check_id']}",
                    "observed": 0.0 if row["status"] == "fail" else 1.0,
                    "threshold": 1.0,
                    "status": str(row["status"]),
                    "message": str(row["message"]),
                }
            )

    return pd.DataFrame(rows)


def run_preprocessing_audit(config: PreprocessingAuditConfig) -> dict[str, pd.DataFrame]:
    coverage_rows = []
    numeric_summaries = []
    numeric_differences = []
    column_summaries = []

    for dataset, spec in DATASET_SPECS.items():
        baseline = _read_csv(config.baseline_dir / spec["filename"])
        candidate = _read_csv(config.candidate_dir / spec["filename"])
        keys = list(spec["keys"])
        coverage_rows.append(build_coverage_summary(baseline, candidate, dataset, keys))
        column_summaries.append(build_column_presence_summary(baseline, candidate, dataset))
        numeric_summary, differences = compare_numeric_columns(
            baseline,
            candidate,
            dataset,
            keys,
            list(spec["numeric"]),
            abs_tolerance=config.abs_tolerance,
            rel_tolerance=config.rel_tolerance,
        )
        if not numeric_summary.empty:
            numeric_summaries.append(numeric_summary)
        if not differences.empty:
            numeric_differences.append(differences)

    results = {
        "coverage_summary": pd.DataFrame(coverage_rows),
        "column_presence_summary": pd.concat(column_summaries, ignore_index=True),
        "numeric_diff_summary": pd.concat(numeric_summaries, ignore_index=True)
        if numeric_summaries
        else pd.DataFrame(),
        "numeric_differences": pd.concat(numeric_differences, ignore_index=True)
        if numeric_differences
        else pd.DataFrame(),
    }
    contract_result = validate_canonical_data_dir(
        config.candidate_dir,
        require_manifest=True,
    )
    contract_rows = [
        {
            "dataset": "canonical_data",
            "check_id": f"issue_{index + 1}",
            "status": "fail",
            "message": issue,
        }
        for index, issue in enumerate(contract_result.issues)
    ]
    contract_rows.extend(
        {
            "dataset": "canonical_data",
            "check_id": f"warning_{index + 1}",
            "status": "warn",
            "message": warning,
        }
        for index, warning in enumerate(contract_result.warnings)
    )
    results["candidate_contract_summary"] = pd.DataFrame(
        contract_rows,
        columns=["dataset", "check_id", "status", "message"],
    )
    results["replacement_readiness_summary"] = build_replacement_readiness_summary(results, config)
    return results


def write_outputs(results: dict[str, pd.DataFrame], config: PreprocessingAuditConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "coverage_summary": config.output_dir / "coverage_summary.csv",
        "column_presence_summary": config.output_dir / "column_presence_summary.csv",
        "numeric_diff_summary": config.output_dir / "numeric_diff_summary.csv",
        "numeric_differences": config.output_dir / "numeric_differences.csv",
        "replacement_readiness_summary": config.output_dir / "replacement_readiness_summary.csv",
        "candidate_contract_summary": config.output_dir / "candidate_contract_summary.csv",
    }
    for name, path in paths.items():
        results[name].to_csv(path, index=False, encoding="utf-8-sig")
    manifest_path = config.output_dir / "run_config.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                **asdict(config),
                "baseline_dir": str(config.baseline_dir),
                "candidate_dir": str(config.candidate_dir),
                "output_dir": str(config.output_dir),
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    paths["run_config"] = manifest_path
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--abs-tolerance", type=float, default=DEFAULT_ABS_TOLERANCE)
    parser.add_argument("--rel-tolerance", type=float, default=DEFAULT_REL_TOLERANCE)
    parser.add_argument("--min-common-key-coverage", type=float, default=DEFAULT_MIN_COMMON_KEY_COVERAGE)
    parser.add_argument("--min-candidate-stock-coverage", type=float, default=DEFAULT_MIN_CANDIDATE_STOCK_COVERAGE)
    parser.add_argument(
        "--max-critical-numeric-mismatch-rate",
        type=float,
        default=DEFAULT_MAX_CRITICAL_NUMERIC_MISMATCH_RATE,
    )
    parser.add_argument(
        "--critical-numeric-datasets",
        default=",".join(DEFAULT_CRITICAL_NUMERIC_DATASETS),
        help="Comma-separated datasets whose numeric diffs are replacement gates.",
    )
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = PreprocessingAuditConfig(
        baseline_dir=args.baseline_dir,
        candidate_dir=args.candidate_dir,
        output_dir=args.output_dir,
        abs_tolerance=args.abs_tolerance,
        rel_tolerance=args.rel_tolerance,
        min_common_key_coverage=args.min_common_key_coverage,
        min_candidate_stock_coverage=args.min_candidate_stock_coverage,
        max_critical_numeric_mismatch_rate=args.max_critical_numeric_mismatch_rate,
        critical_numeric_datasets=tuple(
            item.strip() for item in args.critical_numeric_datasets.split(",") if item.strip()
        ),
    )
    results = run_preprocessing_audit(config)
    print("Coverage summary:")
    print(results["coverage_summary"].to_string(index=False))
    print("\nNumeric diff summary:")
    print(results["numeric_diff_summary"].to_string(index=False))
    print("\nReplacement readiness summary:")
    print(results["replacement_readiness_summary"].to_string(index=False))
    if not args.no_write:
        paths = write_outputs(results, config)
        print("\nWrote audit outputs:")
        for name, path in paths.items():
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
