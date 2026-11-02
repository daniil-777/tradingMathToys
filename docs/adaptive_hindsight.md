# Adaptive timing with a liquidity cooldown

`financeai.research.adaptive_depth_oracle` improves the executable hindsight
benchmark by optimizing **when to act**. It deliberately uses future prices.
It returns the same `DepthOracle` schedule consumed by `replay_depth_oracle`.

## The new state recurrence

Let `t[i]` be candidate receipt times and let

```
tau = max(order_latency + ack_latency, consumption_ttl)
j[i] = last index j with t[j] <= t[i] - tau
```

The previous method allowed decisions only at widely separated points. The new
method may inspect every receipt while imposing that separation on the orders
it actually selects. A decision to wait starts no cooldown.

Define `F[i,q]` as the best score ending at inventory q using candidates through i:

```
F[i,q] = max(F[i-1,q], max_p {F[j[i],p] + IOC[i](q-p)})
```

Initial inventory is flat; other initial states have score negative infinity.
The final state must also be flat. An IOC edge uses the original arrival book,
actual limit price, fees, lots and haircut. It must fill the entire grid quantity.
Direct reversals across zero and exposure increases after the entry cutoff are
excluded, as in the earlier optimizer.

**Proof of feasibility.** Any prefix selected from `F[j[i],p]` has its last trade
at or before `t[i]-tau`. The previous acknowledgement has arrived, and all
liquidity it consumed has expired. Earlier trades have expired too. Thus the
inventory p suffices to compute the next feasible IOC, even though unused
candidate times may be much closer than tau. Induction gives a valid path for
every finite state value. Backtracking records an order and jumps to `j[i]`,
or waits and jumps to `i-1`.

**Proof of graph optimality.** A feasible prefix either makes no trade at i,
or its previous selected trade is among candidates through `j[i]`. The two
terms enumerate those exhaustive possibilities. Induction and terminal-flat
backtracking establish the exact-arithmetic optimum of the restricted graph.

The graph includes the old solution when all old candidate times and inventory
states are retained, and the new cooldown is no larger than the old spacing.
Independently resampling at a finer interval is insufficient: receipt times
are irregular, so the original points must be unioned explicitly. Setting
`candidate_spacing_ns=1` considers every unique book receipt cohort.

## Why the transition search is faster

For ordered book levels, the IOC reward `g(d)` is discretely concave: additional
buy quantity earns a smaller base-currency cash rate, while additional sell quantity costs a
larger rate. The bid/ask spread gives the correctly ordered slope at zero.

Consequently `A[q,p] = F[j,p] + g(q-p)` has the anti-Monge inequality

```
A[q1,p1] + A[q2,p2] >= A[q1,p2] + A[q2,p1]
```

for increasing row and column indices wherever the four edges are feasible.
Column constants `F[j,p]` cancel; they do not need to be concave. The maximizing
predecessor index is monotone in q. Divide-and-conquer searches those maxima,
reducing the typical transition cost from `K²` to `K log K`.

The implementation treats positive targets and negative targets separately;
flat inventory is handled independently. Fill capacities and the entry cutoff
clip feasible predecessor intervals. Consistent leftmost ties are used. An
entirely unreachable midpoint never narrows its children's search bounds.
That fallback can increase worst-case work; an unconditional `O(K log K)`
runtime bound for every unreachable-state pattern is not claimed.

The public `accelerated=False` option uses the plain quadratic transition search.
Tests compare the two on 600 adversarial random transition problems and compare
complete schedules against exhaustive small-market paths.

## Storage and numerical qualifications

Predecessor choices use `N*K` int16 entries. Score vectors are stored in a ring
of size `max(i-j[i])+2`, calculated from the actual unioned candidate times.
This preserves the required prefix across long receipt gaps. State indices are
limited to 32,767; very fine inventory grids can require substantial memory.

Pass `checkpoint_rows=B` to retain only B rows of parents while reconstructing
the schedule. A first pass writes each block's complete score ring to a private
temporary directory. During backtracking, the solver loads the corresponding
frontier and recomputes that block with the identical transition kernel and tie
order. This reproduces the full parent-table path, including jumps across blocks.
Snapshots include the entire ring, so delayed eligible prefixes remain available.
Temporary files are removed on completion or error. Parent RAM is O(BK), score
RAM O(RK), and temporary disk O(ceil(N/B)RK), where R is the ring length. Work is
at most about two forward scans, plus disk I/O. The tests compare full and
checkpointed paths for both transition methods and five block sizes.

For nested inventory refinement, choose steps that divide the previous step:
`10000 -> 5000 -> 2500` preserves every old state. `5000 -> 2000` does not.

The mathematical recurrence is exact; implementation arithmetic is float64.
Rounding can perturb discrete concavity slightly. Full replay establishes the
chosen path's feasibility and score, not rational optimality of its Bellman table.
The study independently encloses the actual fill score by exact integer division.

This is a lower bound on the unrestricted engine optimum. Same-venue orders
less than tau apart, partial grid fills, finer inventory quantities and decisions
at trade-only receipts remain outside its graph. The saved schedules cannot be
used as causal forecasts on a new day.

## Usage

```python
from financeai.research import adaptive_depth_oracle, replay_depth_oracle

plans = tuple(adaptive_depth_oracle(
    tapes, scenario, session,
    step_units=10_000,
    candidate_spacing_ns=25_000_000,
    include_spacing_ns=(1_000_000_000, 250_000_000, 100_000_000),
) for tapes in data.venues)
result = replay_depth_oracle(data, plans, scenario, session, full_stream=True)
```

Run `python examples/hindsight.py --out runs/hindsight` for a synthetic end-to-end example.
