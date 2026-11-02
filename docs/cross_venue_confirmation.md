# Cross-venue confirmation

A price move or large trade on one venue can reflect a local liquidity shock. A confirmation rule
can require activity on another venue, and a freshness gate can exclude stale quotes. These are
hypotheses to evaluate on chronological training and validation data, not guarantees of profitability.

`CrossVenueConfirmationSpec` defines a sweep component and a lead-gap component, each with optional
volume gates on either instrument. `build_cross_venue_signal_spec(spec, source=..., leader=...)`
accepts arbitrary instrument keys. `build_cross_venue_strategy_config` derives the keys from a
provided two-instrument strategy configuration.

```python
from financeai.research.cross_venue_confirmation import (
    CrossVenueConfirmationSpec, build_cross_venue_signal_spec,
)

spec = CrossVenueConfirmationSpec(name="my-confirmation", sweep_min_units=1000)
signal = build_cross_venue_signal_spec(spec, source="exchange_a/ASSET", leader="exchange_b/ASSET")
```

Thresholds use stored quantity units. Compare compatible prices across venues and choose scales
for each instrument's quantity convention. The example constructors are illustrative configurations.

`diagnose_episode_confirmation` pairs completed episodes with causal entry-time volume diagnostics.
`certify_fill_ledger_interval` checks lots, position limits, terminal flatness, and cash with exact
rational arithmetic. It supports instrument-specific settlement and requires a common currency for
a scalar certificate. Its rounded interval is an enclosure of the supplied ledger, not a statistical
confidence interval or proof of future profitability.
