# Executable labels

A label asks: if an order were submitted using information available at a particular receipt time,
and a close were attempted after a given horizon, what would the episode return under the execution model?

`executable_labels` evaluates those hypothetical episodes independently. `label_fills` exposes their
ledgers, and `hold_value_labels` compares immediate liquidation with waiting. Labels deliberately use
future outcomes and belong in training/evaluation targets, never as contemporaneous decision inputs.

The same lot rules, order limits, arrival books, depth consumption, fees, and instrument cash conventions
apply as in replay. Entry and exit averages are harmonic for inverse contracts and arithmetic otherwise.
Short-entry labels require a shortable instrument.

| Status | Meaning |
| --- | --- |
| `FILLED` | Full requested entry and a completed close |
| `PARTIAL_ENTRY` | Part of the entry filled and that exposure was closed |
| `NO_FILL` | An actual submitted order filled nothing; zero realized cash |
| `NO_ORDER` | No eligible order was submitted; do not treat it as an observed entry outcome |
| `CENSORED` | Unresolved inventory at the cutoff or retry limit; outcome fields are NaN |

Times are integer nanoseconds. Sizes are stored quantity units. Cash fields are in the selected
instrument's settlement currency. `knowable_ns` records when the result's final acknowledgement
could be known, which matters when splitting chronological training data.

Independent label calls do not compete for shared inventory or liquidity. A replay of all those
entries together can therefore produce different results. Use a `SessionPlan` execution cutoff and
an explicit retry budget appropriate for the experiment. A censored label is never replaced with zero.
