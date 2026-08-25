# ADR 0002: Canonical data is generated from free_taiwan_data

## Status

Accepted

## Context

The repository has two peer forecasting systems that read shared CSV files from `data/`.
The ignored `free_taiwan_data/` directory contains the upstream raw data family and can
rebuild those CSV shapes through `data_preprocessing/free_taiwan_data_preprocessor.py`.

Deleting `data/` immediately would break the default interface for fresh clones because
`free_taiwan_data/` is intentionally not tracked.

## Decision

Treat `free_taiwan_data/` as the ignored raw source and `data/` as the tracked canonical
CSV interface. It is regenerated from `free_taiwan_data/` only after a full-universe
candidate passes the preprocessing audit.

Generated candidates must include:

- `manifest.json` with `data_contract_version`
- required canonical CSV files and columns
- row and stock counts
- stock coverage by dataset
- the revenue monetary-unit contract
- per-file SHA-256 hashes after outputs are written
- validation results for required columns, duplicate revenue keys, unit consistency, and coverage

The replacement gate is `data_preprocessing.preprocessing_audit`. A candidate should not
replace or delete `data/` while `replacement_readiness_summary.csv` contains `fail` rows.

## Consequences

- Ensemble and Rolling LSTM keep depending on one small data interface.
- Raw data parsing stays local to `data_preprocessing/`.
- Ignored candidate and audit artifacts stay under `data_preprocessing/outputs/`, not inside the raw input directory.
- The local raw source retains only the five datasets consumed by current preprocessing adapters;
  adding a new raw feature family requires an explicit adapter change and source refresh.
- `data/` was replaced by generated outputs on 2026-07-30 after full coverage and
  mismatch checks passed.
- The tracked 2026-07-30 manifest predates the later per-file hash addition; it remains the current
  canonical-v1 snapshot, while the next regeneration will include hashes automatically.
- Future data refreshes must be explicit migrations with manifest validation, audit
  outputs, and system tests.
