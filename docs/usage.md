# Usage

The [quickstart](../README.md#quickstart) runs without external data. A custom experiment follows
the same sequence: define instruments, normalize data, construct a signal, replay a policy, inspect
its ledger, and evaluate it on chronological data that was not used for parameter selection.

## Python replay

```python
from financeai.application.strategy import StrategyConfig, load_market, run_strategy, write_run_outputs
from financeai.domains.market import Scenario

config = StrategyConfig.load("strategy.json")
market = load_market(["cache/market-a", "cache/market-b"], config.instruments)
scenario = Scenario.from_mapping(scenario_document)
result = run_strategy(market, config, scenario)
write_run_outputs(result, market, config, scenario, "runs/experiment")
```

`config.instruments` fixes instrument order. Cache metadata defines the units and payoff conventions.
The exported `summary.json` also records `instrument_specs` and `scenario_config`; both round-trip
through their respective `from_mapping` constructors, including optional costs and timing settings.
`StrategyConfig.from_threshold_params` exports a Python threshold policy without dropping fields;
its integer-millisecond JSON clocks reject finer values rather than truncate them.

## Signals

`build_signal(market, recipe)` supports lead-gap, EWMA deviation, flow, sweeps, moves, quote
confirmation, and order-flow imbalance. `any`/`all` composites combine scaled components; freshness
and volume gates restrict eligible events.

```python
recipe = {"kind": "flow", "venue": "exchange/ASSET", "tau_ms": 1000, "measure": "native"}
```

Flow can use `units` (stored integers), `native` quantity, or `notional` in quote currency. Inverse
face amounts already express quote notional; linear notional includes the trade price and multiplier.
Choose thresholds and feature scales appropriate for the declared units. Cross-venue price comparisons
need compatible quotes; the library does not silently perform currency conversions.

## Costs and timing

Supply a `Scenario` or `--scenario-file` with explicit fees and latencies. `fees` is proportional;
`fees_per_unit` charges settlement currency per native quantity. `SessionPlan` separates entry cutoff,
forced flattening, execution cutoff, and the final observation drain. `RiskLimits` caps absolute
positions in stored units.

`financeai suite` runs four fictional stress presets for the supplied instruments. It is a convenient
software demonstration, not a claim that those fees or delays apply to a particular exchange.

## Diagnostics

`result.orders` records submissions and outcomes. `result.fills` records execution prices, quantities,
arrival times, acknowledgements, and cash. Settlement requires terminal flatness and compliance with
position and shorting rules. Multiple currencies produce separate `pnl_by_currency` results.

For a common settlement currency, the reporting extra can create an account figure and provenance:

```python
from financeai.reporting.charts import position_pnl_figure

artifact = position_pnl_figure(result, market, title="Synthetic replay",
                               scenario_name=scenario.name, scenario=scenario,
                               out_path="figures/replay.png", evidence_scope="SYNTHETIC")
```

Use [executable labels](labels.md) for cost-aware targets and [statistics](statistics.md) for temporal
resampling and candidate comparisons. The [local API](api/README.md) provides an optional bounded job
service. [`render_dashboard_html`](../src/financeai/research/dashboard.py) renders portable reports from
named sections without assuming particular markets, currencies, dates, or results.
