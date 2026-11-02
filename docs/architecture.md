# Architecture

| Package | Responsibility |
| --- | --- |
| `financeai.core` | Errors, clocks, hashing, optional JIT |
| `financeai.domains.market` | Instruments, tapes, parsers, caches, event streams, execution, ledgers, labels |
| `financeai.strategies` | Causal signals and feature construction |
| `financeai.models` | Small numerical forecasting models |
| `financeai.application` | Strategy configuration, replay workflows, grids, exports |
| `financeai.research` | Statistical analysis, evidence records, and scoped mathematical benchmarks |
| `financeai.reporting` | Optional figures and decision traces |
| `financeai.service` | Optional local job API and artifact management |

The dependency tests enforce this direction: numerical kernels do not import plotting or HTTP
packages. The NumPy-only path runs without Numba; accelerated kernels use the same data contracts.

## Three clocks

1. **Receipt:** new market information becomes available to the controller.
2. **Arrival:** a submitted order reaches the exchange and sweeps the applicable book.
3. **Acknowledgement:** fills become known to the controller and can trigger another decision.

`TimingMode.EXCHANGE_CLOCK` uses the exchange-clock envelope for execution. The alternative
receipt-replay timing mode is an explicit modeling choice. Receipt cohorts and acknowledgement
ordering are deterministic; validated event streams cannot invent, duplicate, or retime rows.

Orders are marketable IOC limits. They can fill fully, fill partially, or expire unfilled. The
execution model consumes available lot-multiple depth and tracks recently consumed liquidity.
Policies use acknowledged inventory. Account settlement uses actual fills.

## Boundaries

Input adapters normalize external formats into immutable `MarketTapes`. The cache hashes its
columns and records instrument metadata. `PreparedMarket` validates and indexes tapes for repeated
replays. A `Scenario` provides costs and timing; a `SessionPlan` provides entry and execution
cutoffs; `RiskLimits` provides a cap in stored units.

`run_reference` is an independent Python controller/execution path. `exact_score` computes the
cash of a flat ledger with rational numbers. Agreement between implementations is useful evidence,
while model assumptions and input quality still determine what the simulation represents.
