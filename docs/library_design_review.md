# Designing a general market research library

Reviewed 2 November 2026. This review uses public documentation and source code; it makes no
claims about the internal practices of trading firms. The architectural recommendations below
are our engineering conclusions, not endorsements or performance rankings.

## Decision

FinanceAI should remain a focused, deterministic market-replay library with explicit instrument
and execution contracts. Its useful distinction is the separation between information available
to a controller and the historical book encountered by an order, checked against an independent
execution implementation. Preserve that foundation while removing assumptions that every asset
has identical units, risk limits, or execution delays.

Generality comes from well-defined boundaries and tested economic conventions. A larger catalog
of symbols alone does not provide market-specific settlement, trading hours, collateral,
corporate actions, or derivative lifecycle behavior.

## What established projects demonstrate

| Project | Public design pattern | Implication for FinanceAI |
| --- | --- | --- |
| QuantConnect LEAN | Security-level fill, slippage, fee, brokerage, buying-power, settlement, and short-availability models; separate portfolio-level models. | Attach economic rules to explicit instruments and account models. Avoid treating one fee, limit, or market convention as universal. |
| NautilusTrader | Instrument definitions, normalized data and execution adapters, explicit sequencing, and typed configuration boundaries. | Keep vendor parsing outside the numerical engine; preserve exchange/receipt clocks, declared increments, and deterministic order handling. |
| Backtrader | Configurable commission information distinguishes percentage charges, fixed charges, and contract multipliers; schemes can apply to a named data feed. | Keep commissions and exposure conversion explicit. A quantity of one share and a quantity of one futures contract are different economic exposures. |
| vectorbt | Labeled array research, explicit order-array simulation, and callbacks for decisions that depend on evolving state. | Keep parameter sweeps efficient and inspectable. Use the event-driven engine when execution depends on acknowledgements, pending orders, or changing depth. |

Sources: [LEAN reality models](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/key-concepts),
[LEAN Security implementation](https://github.com/QuantConnect/Lean/blob/master/Common/Securities/Security.cs),
[NautilusTrader instruments](https://nautilustrader.io/docs/latest/concepts/instruments/),
[NautilusTrader adapters](https://nautilustrader.io/docs/latest/concepts/adapters/),
[NautilusTrader execution sequencing](https://nautilustrader.io/docs/latest/concepts/backtesting/execution-flow/),
[Backtrader commissions](https://www.backtrader.com/docu/commission-schemes/commission-schemes/),
[Backtrader commission implementation](https://github.com/mementum/backtrader/blob/master/backtrader/comminfo.py),
[vectorbt research and simulation features](https://vectorbt.dev/getting-started/features/).

These systems address different workloads. Their documentation does not establish that one engine
is universally faster, more realistic, or more profitable. Comparing them fairly requires the
same inputs, economic assumptions, order policy, and measured output.

## Engineering practices worth adopting

NautilusTrader's contribution guide calls for complete, locally validated changes, relevant tests,
formatting checks, generated-artifact consistency, and coherent updates before CI. LEAN's
contributor documentation requires accompanying tests, and its regression runner drives complete
algorithms through the engine. Those are useful public examples of evidence-based development.
See [NautilusTrader contributing](https://github.com/nautechsystems/nautilus_trader/blob/develop/CONTRIBUTING.md),
[LEAN contributing](https://github.com/QuantConnect/Lean/blob/master/CONTRIBUTING.md), and
[LEAN regression runner](https://github.com/QuantConnect/Lean/blob/master/Tests/AlgorithmRunner.cs).

For this smaller library, the practical equivalents are:

- Test economic invariants against independently calculated cash and an independent controller.
- Check configuration failures at the public boundary, before entering compiled kernels.
- Preserve deterministic ledger comparisons and exact cash audits.
- Test both accelerated and NumPy-only installations, plus installation outside the checkout.
- Verify exported configuration, schemas, and documentation against the behavior that actually runs.
- Keep synthetic demonstrations reproducible and distinguish them from historical performance evidence.

An extensive inheritance hierarchy, a new language implementation, or many exchange connectors
would introduce maintenance obligations. Add those only when a concrete use case and a measurable
benefit justify them.

## Changes implemented from this review

### Instrument-specific controls

`RiskLimits.per_instrument` overrides a default absolute-position cap in stored quantity units.
A zero override disables an instrument. `Scenario.latency_by_instrument` assigns an
`InstrumentLatency` with independent order and acknowledgement delays. Omitted keys inherit the
default; unknown override keys are rejected when replaying the selected market.

The compiled engine resolves these values into arrays once per replay configuration. The Python
reference retains its own control and execution logic. Labels and the inverse-only research
solvers resolve the selected instrument's settings as well. Reports show the applicable caps,
and exported summaries preserve the actual risk and execution configuration.

Instrument identity is the full `venue/symbol` key. Multiple assets may share one exchange name;
legacy integer `venue` columns identify these instrument slots, not unique exchange names.
The current storage supports at most 128 slots.

### Exact normalization boundaries

Price and quantity grids are checked with exact integer ratios. An off-grid decimal must not
become valid because the caller lowered Python's Decimal precision or because a division rounded
away extra digits. The optimized text decoder is selected only when the micro-price grid is
exactly representable. Extreme exponents are rejected before allocating an enormous scaled integer.

### Audit behavior

Settlement validates the per-instrument cap vector and rejects integer position wraparound.
Configuration values are immutable, JSON-compatible, and serializable across Python process
boundaries. Disabling an instrument does not disable other instruments in the same portfolio.

## Validation strategy

The regression suite covers one-, three-, and eight-instrument portfolios; multiple symbols on
one venue; spot, linear, and inverse contracts; fractional quantities; multipliers; mixed
settlement currencies; zero caps; and asymmetric latency legs. Randomized portfolios compare
every order/fill field and terminal settlement with the independent engine. Reordering
instruments must preserve each instrument's execution history after matching by key.

Additional cases check labels and research plans against explicitly resolved configurations,
JSON/process serialization, unknown instruments, clock headroom, exact grid membership under
different Decimal contexts, ledger overflow, CLI configuration, and report annotations.
The existing causality, fill, fee, cache-integrity, and optional-dependency suites remain required.

## Roadmap and boundaries

| Priority | Area | Required contract before claiming support |
| --- | --- | --- |
| Implemented | Heterogeneous instrument settings | Declared units, payoff convention, currencies, costs, caps, and deterministic delays. |
| Next | Additional data adapters | Explicit schemas, timestamps, increments, sequence/disconnect semantics, and adapter conformance fixtures. Bar data must not silently manufacture depth. |
| Next | Account and lifecycle models | Cash constraints, margin, funding/borrow costs, corporate actions, exchange calendars, expirations, and consistent mark/settlement events. |
| Later | Additional execution models | Documented order lifecycles, price/volume assumptions, cancellation semantics, and independent execution checks. |
| Later | Live integration | Paper/sandbox reconciliation, recoverable order state, risk admission, and operational ownership before connecting a broker. |

The present engine models positive-price displayed-depth taker execution. It does not claim
universal instrument lifecycle support. Negative-price contracts, currency conversion, passive
queue priority, and the account/lifecycle features above require explicit additional models.
That boundary is part of the API contract and should stay visible in the README.
