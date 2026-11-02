# Statistical evaluation

Evaluate strategies on chronological data with explicit development, validation, and held-out roles.
A result selected from many candidates is subject to selection bias, even when every individual replay
is causal. Statistical diagnostics complement execution validation; neither proves future profitability.

## Account increments

`account_increments(result, market, grid_ns=...)` samples actual positions and cash on a fixed grid.
It values held inventory at received midpoints (`valuation="mid"`) or the appropriate exit touch
(`valuation="exit"`). Both conventions omit liquidation depth and fees. Cash arithmetic follows each
instrument's spot, linear, or inverse definition.

The grid must cover all fills and begin flat. Missing books while inventory is held produce NaN marks;
they are not replaced with zero. Choose an appropriate grid explicitly. The final mark of a flat account
must reconcile with settlement. `increment_matrix(paths)` requires matching clocks, valuation rules,
and settlement currencies.

## Dependence-aware comparisons

- `paired_block_bootstrap` resamples the same stationary-bootstrap blocks across candidates, preserving
  their shared market path. Compare several plausible block lengths and record the seed.
- `hac_mean` estimates uncertainty around the mean with autocorrelation-consistent variance.
- `selection_diagnostics` reports multiple-comparison and Sharpe-related diagnostics with explicit trial counts.
- The lower-level `stats` module provides deterministic bootstrap and change-point utilities.

Do not interpret per-interval cash increments as investment returns without a justified capital
normalization. A strategy that starts flat with unconstrained collateral has no automatically meaningful
percentage-return denominator. Annualization and Sharpe interpretation depend on the sampling interval,
capital convention, and dependence assumptions.

## Research records

`Campaign` records data exposure, candidate trials, versioned artifacts, configuration freezes, and
held-out evaluations. Retries and diagnostics remain distinguishable from new evidence. The records
make the research process auditable; they cannot turn already inspected data into an unseen test set.

[Executable labels](labels.md) record when an outcome becomes knowable. Respect those knowledge times
when constructing chronological splits, especially when labels overlap across horizons.
