# Instrument conventions

`InstrumentSpec` is the source of truth for units and cash arithmetic. An exchange name or symbol
never selects a hidden preset. `to_mapping()` and `from_mapping()` round-trip JSON-ready metadata.

| Field | Meaning |
| --- | --- |
| `tick_size` | Quote-price increment, an exact positive `Decimal` |
| `quantity_step` | Native quantity represented by one stored integer unit |
| `lot_size` | Executable multiple of stored quantity units |
| `contract_multiplier` | Exposure per native quantity; defaults to one |
| `contract_type` | `SPOT`, `LINEAR`, or `INVERSE` |
| `base_currency`, `quote_currency` | Explicit unit labels |
| `settlement_currency` | Base for inverse, quote otherwise; inconsistent declarations are rejected |
| `shortable` | Whether negative inventory is allowed; spot defaults to false |

For example, `quantity_step=Decimal("0.001")`, `lot_size=10` means orders execute in multiples
of 0.010 native units. A tape quantity of 250 represents 0.250 native units. A linear future
with `contract_multiplier=Decimal("50")` changes settlement value by 50 currency units per
price point per native contract.

## Cash equations

Let `q` be a positive fill quantity in stored units, `u=quantity_step`, `m=contract_multiplier`,
`p` the price, `s=+1` for a buy or `-1` for a sell, `f` the proportional fee, and `c` the fixed
commission in settlement currency per native quantity.

```text
exposure = q × u × m
linear / spot fill cash = (−s − f) × exposure × p − q × u × c
inverse fill cash       = ( s − f) × exposure ÷ p − q × u × c
```

Cash is not PnL while inventory remains open. At a valid observed midpoint:

```text
linear / spot marked value = cash + position × u × m × midpoint
inverse marked value       = cash − position × u × m ÷ midpoint
```

These marks omit the spread and fees of a hypothetical liquidation. A terminal-flat account's
cash is its realized result. Settlement also checks the position cap and shorting permission.
Inverse entry/exit averages are quantity-weighted harmonic prices; linear averages are
quantity-weighted arithmetic prices.

Different settlement currencies are reported separately. A scalar `net_pnl` is `None` when
currencies differ; use `pnl_by_currency`. No implicit FX conversion takes place.

## Storage and ingestion

Book prices and quantities use positive/nonnegative int32 ticks/units, and clocks use int64
nanoseconds. Inputs outside those ranges or off the declared grids are rejected. Fills and
positions use wider integer quantities. Very small price ticks can require the exact decimal
parser instead of the optimized text path.

Cache schema 2 stores the full instrument definition and refuses incomplete metadata. Older
cache schemas must be rebuilt from source data. Numerical kernels use float64 cash; use
`exact_score` when checking a flat ledger against rational arithmetic.

Instrument configuration describes the supported cash-flow model. Funding, corporate actions,
margin, borrowing charges over time, and derivative lifecycle events require additional models.

## Instrument-specific caps and delays

Different instruments can share an exchange while using different position limits and transport
assumptions. Configure overrides by the full `venue/symbol` key:

```python
from financeai.domains.market import InstrumentLatency, RiskLimits

limits = RiskLimits(
    max_abs_position_units=10_000,
    per_instrument={"exchange/SHARE-CHF": 2_000, "exchange/INDEX-CHF": 10},
)
# Native exposure still depends on each instrument's quantity_step and multiplier.
# A zero per-instrument cap disables trading that instrument.
```

`Scenario.latency_by_instrument` accepts the same keys and `InstrumentLatency` values. Each value
specifies `order_latency_ns` and `ack_latency_ns`; the latter defaults to the order delay.
Either leg can be zero, but their sum must be positive. Instruments without an override inherit
the scenario defaults. Unknown override keys fail validation when replaying a market.

`scenario.for_instrument(spec)` resolves a single-instrument scenario for research or label
calculations. `limits.cap_for(spec)` returns that instrument's cap. `to_mapping`/`from_mapping`
round-trip both configurations, including their overrides. The replay result retains its actual
limits, and `summary.json` records them as `risk_limits`.

For CLI runs, supply `--risk-file risk.json` with this structure:

```json
{
  "max_abs_position_units": 10000,
  "per_instrument": {"exchange/SHARE-CHF": 2000, "exchange/INDEX-CHF": 10}
}
```

A scenario JSON file may include:

```json
{
  "name": "custom",
  "fees": {"exchange/SHARE-CHF": "0.0002", "exchange/INDEX-CHF": "0.0001"},
  "order_latency_ns": 3000000,
  "latency_by_instrument": {
    "exchange/INDEX-CHF": {"order_latency_ns": 1000000, "ack_latency_ns": 2000000}
  }
}
```

Use the same instrument keys in your caches and strategy. `run` accepts `--scenario-file` and
`--risk-file`; `suite` accepts `--risk-file` for its built-in demonstration scenarios.
