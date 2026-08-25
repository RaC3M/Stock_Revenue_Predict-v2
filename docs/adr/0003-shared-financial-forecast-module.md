# ADR 0003: Share downstream financial transformations

## Status

Accepted.

## Context

Ensemble and Rolling LSTM previously implemented overlapping EPS, dividend, price, and yield
rules. The duplicated rules drifted: a correction to availability cutoffs or yield semantics had
to be made twice, and the two UIs could display different meanings under the same label.

## Decision

Create a neutral `financial_forecast/` package with one public orchestration interface. Both
forecast systems keep independent revenue engines and call the shared package through local thin
adapters. Neither forecasting system imports the other.

The shared package owns:

- canonical financial evidence loading and availability cutoffs;
- EPS and cash-dividend transformation strategies;
- complete-12-month enforcement;
- `as_of_price_yield` deployable yield and `target_month_end_yield` evaluation yield.

It does not own revenue training, model selection, comparable-cohort construction, experiment
selection, or reporting claims. Cross-system scoring and 2022–2024 method selection remain in
`forecast_benchmark/`.

## Consequences

- Ensemble and Rolling display the same financial semantics without sharing revenue-model code.
- Characterization tests protect existing outputs while internal strategies can evolve.
- New EPS or dividend methods are added behind the policy contract and can be ablated without UI
  duplication.
- Historical actual-revenue replay validates only the downstream transformation; it is not
  evidence that an upstream revenue model generalizes.
- Target-year observed prices remain evaluation evidence and are never described as forecasts.
