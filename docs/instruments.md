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
