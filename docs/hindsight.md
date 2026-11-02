# Hindsight benchmarks

These methods read future prices to solve a specified historical optimization problem. They
are useful benchmarks for execution constraints and search algorithms, not deployable strategies.
The tape-based optimizers currently require shortable inverse instruments with unit cash exposure
and proportional fees. They reject other conventions instead of applying inverse algebra silently.

## Quote-only relaxation

`quote_oracle` chooses a terminal-flat inventory path in the set `{-C, 0, C}` using top quotes.
Under this relaxation, intermediate lot-multiple inventory does not improve the optimum. It ignores
depth and receipt-time information constraints. Its float64 objective is a numerical benchmark,
not an outward-rounded mathematical certificate.

## Executable depth paths

`depth_oracle` uses an inventory grid and receipt-time decisions. Each candidate transition sweeps
the actual arrival book at the order's limit. Order spacing clears acknowledgements and consumed
liquidity, so inventory is sufficient state for the recurrence. Grid quantities, timing, and complete
fills are restrictions of this benchmark, not a proof of optimality over every engine policy.

`adaptive_depth_oracle` chooses order times from a denser candidate union while imposing a cooldown
on the orders it selects. `retime_depth_oracle` can improve a fixed sequence locally by moving its
order times within valid fences. See [adaptive timing](adaptive_hindsight.md) and
[retiming](retimed_hindsight.md).

Every resulting schedule should be checked with `replay_depth_oracle(..., full_stream=True)` and
its fill ledger audited with `exact_score`. A numerical optimizer's reported objective alone is not
execution evidence. Exact certificates must state their covered graph, grid, and rounding rules;
see [certified hindsight](certified_hindsight.md).

```bash
python examples/hindsight.py --out runs/synthetic-hindsight
python examples/hindsight.py --cache cache/my-inverse-market --out runs/cached-hindsight
```

The example exports actual replay orders/fills and an exact cash result. Its output explicitly
identifies future-data use. No example result is a claim about attainable live or future returns.
