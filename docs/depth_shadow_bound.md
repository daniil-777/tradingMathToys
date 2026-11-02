# A historical upper bound using actual order-book depth

`financeai.research.depth_shadow_bound.certify_depth_shadow` evaluates a supplied
shadow price against every displayed level. It is a certificate of an upper
bound on historical profit, not an executable strategy or a future-profit forecast.

```python
from financeai.research.depth_shadow_bound import certify_depth_shadow

certificate = certify_depth_shadow(
    book, shadow_rates,                 # BookTape and one int64 rate per bucket
    start_ns=start, end_ns=end,          # all arrivals lie in this half-open range
    bucket_ns=10_000_000,
    minimum_arrival_gap_ns=10_000_000,
    cap_units=1_000_000, lot_units=100,
    tick_size="0.5", taker_fee="0.00005",
    rate_scale=10**14,
)
# Keep objective_units as an integer for comparisons and stored certificates.
# Dividing by rate_scale expresses the upper bound in settlement currency.
```

## What gets tighter

Suppose the ask ladder contains one unit at a good price and 99 units at a worse
price. Multiplying the good price's advantage by 100 exaggerates the opportunity.
This calculation prices each of those 100 units at its own level. It also keeps
prices and sizes from the **same arrival book** together.

For an integer shadow rate `x`, a buy level contributes
`quantity * (rounded_buy_credit - x)`. A sell level contributes
`quantity * (x - rounded_sell_cost)`. Sorted levels make these marginal advantages
nonincreasing. The best whole-lot quantity therefore takes positive-advantage
levels until the order cap or the first nonpositive advantage. The bucket retains
the better side of the best candidate book, including the option to do nothing.

The final bound is

```
inventory_cap * sum(abs(next_shadow - current_shadow))
+ sum(best_bucket_residual)
```

Fees and tick sizes accept exact strings, `Decimal`, or `Fraction`; floats are
rejected. Buy credits round up and sell costs round down, favouring the trader.
Per-level face is rounded down to whole lots, matching the engine's availability
rule. This remains optimistic: previously consumed quantity, IOC price limits,
session entry limits, and receipt-time information restrictions are removed.

## Required trading scope

- The contracts are inverse contracts with cash change `(side - fee)*face/price`.
- Initial and terminal positions are zero, and absolute inventory is at most the cap.
- Every order has one side and at most the cap's face value.
- Orders and fills are whole lots; displayed depth can be reduced but never enlarged.
- The minimum same-venue arrival gap is at least the bucket width.
- Every arrival is inside the supplied time range and uses the supplied tape's
  running-maximum exchange clock.

The generic local replay engine enforces the order cap by closing through zero
before reversal. Its single pending order and acknowledgement rule imply an
arrival gap of order latency plus acknowledgement latency. These additional
constraints must be checked for any other execution model.

## Why consumed liquidity cannot invalidate the bound

Consumption may make a real fill skip the best level. The certificate does not
assume actual fills form a fresh prefix. At each level their quantity is no larger
than the fresh whole-lot display. Hold total filled face `q` fixed and move those
units to the best available fresh levels. Buy credit cannot decrease, sell cost
cannot increase, and the shadow term is unchanged. Maximizing over all `q` up to
the cap therefore bounds every consumed-book fill pattern.

## Audit and numerical guarantees

All engine-valid rows are audited for positive active prices/sizes, valid level
counts, and sorted ladders. Malformed executable rows raise `InvalidDataError`;
they are never silently discarded. Every bucket must have exactly one shadow.
The scan includes the book effective at the bucket's left edge and all updates
before its right edge. Intermediate equal-exchange-clock updates inside a bucket
may be included optimistically, matching the existing bucket relaxation.

The code uses exact reciprocal fractions and integer outward rounding. It checks
integer products before compiled execution and aggregates the final sums with
Python integers. This prevents overflow even when the whole-day objective exceeds
int64. `bucket_residual_units` is read-only and supports independent comparison.
The monetary model is the exact rational cash-flow formula; floating ledger
presentation is not used to compute the certificate.

Input file hashes and complete tape provenance remain the caller's responsibility.
The report runner performs those checks before using this API. The API evaluates
the supplied shadow exactly; it does not optimize the stronger depth objective.
