# Contributing

## Workflow

1. Create a feature branch from the current shared branch.
2. Keep changes inside the owning system unless the task explicitly concerns shared data or the benchmark layer.
3. Run the relevant tests before pushing.
4. Use a Pull Request for review when collaborating; do not force-push the shared default branch.

## Ownership boundaries

- `ensemble_forecast/`: non-LSTM revenue and formal dividend-yield application logic.
- `rolling_predict_LSTM/`: Rolling LSTM, pattern clusters, xLSTM research paths, and ablations.
- `forecast_benchmark/`: cross-system normalization, comparable cohorts, shared metrics, and downstream evaluation.
- `data_preprocessing/`: raw-to-canonical conversion and audit gates.
- `data/`: tracked canonical interface; change only through an explicit data refresh.

The two forecast systems must not import one another. Cross-system dependencies point into
`forecast_benchmark/`, never from one forecast engine to the other.

## Validation

```powershell
python tools\validate_project.py
```

The repository-level runner compiles all project Python sources, validates the tracked canonical
`data/` and manifest, runs `pip check` for both owned virtualenvs, invokes each test suite with its
owning virtualenv, and treats `FutureWarning` as a failure. Use repeated `--suite` options for scoped
iteration, for example `--suite rolling --suite benchmark`; run the full command before merging
cross-cutting changes.

Run the relevant smoke test when model execution changes. Full training is not required for every
documentation or test-only change, but state clearly when it was not run.

## Data and artifacts

- Install Git LFS and run `git lfs pull` after cloning.
- Do not commit `.venv/`, cache files, local secrets, raw `free_taiwan_data/`, or generated `outputs/`.
- Keep report-citable conclusions and provenance in tracked `docs/experiments/`.
- Share large raw experiment artifacts separately; do not unignore all output directories.
- Keep preprocessing candidates and audits under ignored `data_preprocessing/outputs/`.
- Regenerate canonical data and pass the preprocessing audit before replacing files in `data/`.

## Evidence and documentation

- Never use target-year actual values to build prediction features, regimes, thresholds, or model selection rules.
- Distinguish prediction-time safety from experiment-level independence: a model can avoid actuals at inference while still having a policy selected from target-year replay.
- Treat `docs/experiments/experiment_registry.md` as the source of truth for evidence status.
- Update the relevant README, result summary, and registry entry when behavior or report-citable conclusions change.
