# Retiming a historical executable path

`financeai.research.retimed_hindsight.retime_depth_oracle` takes an existing
`DepthOracle` witness and improves its order times while retaining all target
inventories. It reads future data and is intended for attainability research.

```python
from financeai.research.retimed_hindsight import retime_depth_oracle
from financeai.research.hindsight import replay_depth_oracle

improved = tuple(
    retime_depth_oracle(tapes, plan, scenario, session, max_passes=6)
    for tapes, plan in zip(data.venues, initial_plans, strict=True)
)
plans = tuple(result.plan for result in improved)
verified = replay_depth_oracle(data, plans, scenario, session, full_stream=True)
```

## Why orders can be closer together

The reference engine maintains separate consumed-liquidity dictionaries for buys
and sells at each venue. An order is independent of earlier consumed depth if
the preceding **same-side** order arrived at least one consumption lifetime
earlier. With fixed latency, this is also the required submission-time gap.
An opposite-side order only needs to wait for the previous acknowledgement.

The helper enforces both constraints. It uses the same observed limit price,
arrival book, depth haircut, lot rounding, fees, and full-fill IOC curve as the
existing depth oracle. It preserves caps, close-through-zero target transitions,
entry cutoff, session coverage, and final flatness. Its returned
`plan.spacing_ns` is the acknowledgement round trip; the additional same-side
constraint is recorded by `RetimingResult.minimum_same_side_gap_ns`.

## Search and evidence

Each coordinate sweep scans all book receipt cohorts between the neighboring
order fences, then improves one order's float64 cash with every other order time
held fixed. The direction alternates. It stops at `max_passes` or an unchanged
sweep. `RetimingResult.passes` records changes and gains per sweep.

The helper supplies a feasible witness under its mathematical assumptions;
`replay_depth_oracle(..., full_stream=True)` independently validates the complete
engine path. Exact rational enclosures of the saved fill ledger should establish
any claimed gain. Search arithmetic is float64. The method does not certify
global optimality or future profitability. An unchanged sweep is only numerical
coordinate stationarity; changing several times together or changing quantities
can still improve the result.
