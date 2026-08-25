# ADR 0001: Keep forecast systems independent

## Status

Accepted; comparison layer implemented. Downstream financial transformation sharing is clarified
by ADR 0003.

## Decision

The Ensemble Forecast System and Rolling LSTM Forecast System are peer systems with separate
application, revenue engine, test, and output ownership. They may read the same root `data/` files
and use the neutral `financial_forecast/` transformation package, but must not import one another.

Cross-system comparison belongs in the isolated `forecast_benchmark/` package. That package may
consume public evidence from both systems through adapters, normalize prediction schemas, construct
the comparable cohort, and evaluate downstream EPS/dividend/yield methods. It must not become a
shared model engine or move one system's training behavior into the other.

## Consequences

- A model change remains local to its owning forecast system.
- Shared evaluation logic lives in `forecast_benchmark/`, not in either Streamlit application.
- The benchmark can fail or evolve without changing the two forecast interfaces.
- Future cross-system tools must use the same one-way dependency direction.
