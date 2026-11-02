# Consumption memory can tighten a historical upper bound

`financeai.research.ttl_shadow_bound.certify_ttl_shadow` extends the actual-depth
fixed-shadow certificate. It uses the fact that a filled price stays consumed,
subject to the local engine's memory-table rules.

```python
from financeai.research.ttl_shadow_bound import certify_ttl_shadow

result = certify_ttl_shadow(
    book, shadow_rates,
    start_ns=start, end_ns=end,
    bucket_ns=10_000_000, minimum_arrival_gap_ns=10_000_000,
    cap_units=1_000_000, lot_units=100,
    tick_size="0.5", taker_fee="0.00005", rate_scale=10**14,
    consumption_ttl_ns=1_000_000_000,
    block_buckets=100, phase_buckets=0,
    consumption_slots_per_side=32,
)
```

This inherits all assumptions in [the depth certificate documentation](depth_shadow_bound.md).
The base certificate is recomputed and audited internally. No precomputed,
potentially mismatched residual array is trusted.

## A simple example

A price displays four contracts. Buying those four consumes them. Seeing the same
four-contract quote again does not create another four available contracts while
the consumption entry remains active. Over a protected block, the total fill at
that price is at most the **largest displayed size**, rather than the sum of all
repeated quote sizes.

For each price and side in a block, the algorithm finds that largest whole-lot
size. It multiplies ask capacity by its best buy advantage over the block's
smallest shadow, and bid capacity by its best sell advantage over the largest
shadow. Summing these optimistic capacities bounds the block's cash-minus-shadow
residual. Taking the smaller of this bound and the existing depth residual sum
can only improve or retain the certificate. The inventory variation term stays
unchanged.

## The 32-price memory guard

The reference engine uses an unbounded dictionary. The compiled engine has only
32 consumption slots per venue and side. When full, it evicts the entry with the
earliest expiry, even if that entry has not yet expired. Consequently the one-second
TTL cannot be assumed to hold for every compiled-engine fill.

The default API guards each block before applying the TTL argument. It proves
that no within-block consumption entry can be evicted if either:

- the candidate book rows contain at most 32 distinct prices on each side; or
- the maximum number of arrivals, displayed levels, and whole lots per order
  permits at most 32 distinct fills on each side.

Untouched entries from before the block expire earlier than every entry touched
inside it. The engine therefore evicts those older entries first. Once all older
entries are gone, the protected within-block price set fits in the table.
Both sides count all candidate prices, including unprofitable levels.

If protection cannot be proved, the block retains the original depth bound.
`unprotected_blocks_retained` reports how often this happened. Set
`consumption_slots_per_side=None` only for the unbounded reference-memory model.

## Boundaries and arithmetic

Blocks are half-open, align with shadow buckets, and are no longer than the TTL.
Fills at the beginning and end of a full TTL interval therefore lie in different
blocks. Positive `phase_buckets` produces a partial first block; the final block
can also be partial. Every shadow bucket is included exactly once.

Sparse price dictionaries handle extreme deep prices without enormous dense
arrays. The capacity sum is saturated at the base block residual, using division
before multiplication when a product would already reach that ceiling. Returned
`capacity_capped_residual_units` stores this exact minimum. All final totals use
Python integers. The function evaluates the supplied shadow; it does not optimize
it, prove that the resulting bound is attainable, or predict future profits.
