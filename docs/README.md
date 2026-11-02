# Documentation

Start with the [repository quickstart](../README.md), then follow the path that fits your work.

| Guide | Purpose |
| --- | --- |
| [Instrument conventions](instruments.md) | Price grids, quantity units, multipliers, fees, and settlement |
| [Usage](usage.md) | Data → signals → replay → exported results |
| [Animation](trading_animation.md) | Beginner explanation, transcript, and reproducible rendering |
| [Architecture](architecture.md) | Package layers and execution semantics |
| [CSV ingestion](tardis.md) | Normalized snapshot/trade inputs and validation |
| [Executable labels](labels.md) | Training targets with execution costs and knowledge times |
| [Statistics](statistics.md) | Time dependence, resampling, and candidate comparisons |
| [Local API](api/README.md) | Optional research job service |
| [Cross-venue confirmation](cross_venue_confirmation.md) | Combining causal signals across instruments |

## Mathematical research tools

Hindsight methods deliberately use future information. They are benchmarks for a defined
historical optimization problem, never signals a live strategy could observe.

- [Quote/depth hindsight](hindsight.md), [adaptive timing](adaptive_hindsight.md), and
  [retiming](retimed_hindsight.md).
- [Certified hindsight](certified_hindsight.md), [shadow bounds](shadow_bound.md),
  [depth bounds](depth_shadow_bound.md), [TTL bounds](ttl_shadow_bound.md), and
  [multiscale TTL bounds](multiscale_ttl_shadow.md).
- [Cash-denominator certificates](cash_denominator.md) and
  [nonattainment certificates](impossibility_certificates.md).

The inverse-specific solvers require their documented payoff and fee conventions. General
replay, labels, and accounting support spot, linear, and inverse instruments.
