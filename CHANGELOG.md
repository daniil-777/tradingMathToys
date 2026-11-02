# Changelog

## Unreleased

- Offline interactive replay explorer with synchronized quote/fill, inventory, marked PnL, and drawdown panels.
- Currency-isolated account views, light/dark themes, exact nanosecond hover details, and downloadable report data.
- Source/output manifests, full-ledger statistics, extrema/gap-preserving display reduction, and explicit display budgets.
- Visualization design research, a heterogeneous-market example, and real Chromium regression checks in CI.

- Per-instrument position caps and independent order/acknowledgement latency overrides.
- JSON risk configuration through the CLI; replay exports preserve actual limits.
- Labels, research solvers, and reports honor the selected instrument controls.
- Exact decimal grid checks independent of Decimal context precision, plus settlement overflow checks.
- Heterogeneous portfolio regressions and a sourced library architecture review.

## 0.2.0.dev0

Initial public preview of the generalized library.

- Explicit spot, linear, and inverse instrument conventions, quantity grids, multipliers, and currencies.
- Proportional and per-native-quantity commissions; spot shorting disabled by default.
- Currency-aware replay, labels, marked values, fee rescoring, and exact cash audits.
- Instrument JSON ingestion and scenario JSON execution configuration.
- Arbitrary instrument specifications in normalized CSV ingestion and synthetic markets.
- Portable tests, optional acceleration/reporting/API extras, and distribution checks.
- Reproducible trading animation with explanatory captions and a verified numerical trace.

The API uses neutral quantity/cash field names. Tape cache schema 2 requires complete instrument
metadata; rebuild older caches from their source. This development version does not promise API stability.
