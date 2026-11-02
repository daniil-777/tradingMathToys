"""Nonfinancial delayed demand/sensor domain on the generic forecast, ticket and maturity machinery.

The domain owns its data, causal features, baseline adapters, metrics, protocol and evaluation. It reuses
:mod:`financeai.core.forecast` (targets, queries, tickets, outcomes) and :mod:`financeai.online` (maturity-gated
feedback, score-once ledger, delayed-feedback learner) and never imports the market domain, an inverse-perpetual
validator or a BTC ledger.
"""

from financeai.domains.demand.adapters import (
    ADAPTER_KINDS,
    DemandAdapter,
    RidgeLagAdapter,
    SeasonalEWMAAdapter,
    SeasonalNaiveAdapter,
    build_adapter,
)
from financeai.domains.demand.data import DEFAULT_START_NS, HOURLY_NS, DemandSeries, DemandSpec, generate_demand
from financeai.domains.demand.evaluation import RUN_SCHEMA, DemandReport, DemandRun, RunResult, evaluate
from financeai.domains.demand.features import (
    FEATURE_VERSION,
    DemandFeatures,
    FeatureSpec,
    compute_features,
    demand_target,
    demand_target_for,
    make_query,
    mature_batch,
)
from financeai.domains.demand.metrics import DemandMetrics, summarize
from financeai.domains.demand.protocol import (
    PROTOCOL_SCHEMA,
    PROTOCOL_SCHEMA_VERSION,
    AdapterSpec,
    ChronologicalSplit,
    DemandProtocol,
    default_protocol,
)

__all__ = [
    "ADAPTER_KINDS",
    "DEFAULT_START_NS",
    "FEATURE_VERSION",
    "HOURLY_NS",
    "PROTOCOL_SCHEMA",
    "PROTOCOL_SCHEMA_VERSION",
    "RUN_SCHEMA",
    "AdapterSpec",
    "ChronologicalSplit",
    "DemandAdapter",
    "DemandFeatures",
    "DemandMetrics",
    "DemandProtocol",
    "DemandReport",
    "DemandRun",
    "DemandSeries",
    "DemandSpec",
    "FeatureSpec",
    "RidgeLagAdapter",
    "RunResult",
    "SeasonalEWMAAdapter",
    "SeasonalNaiveAdapter",
    "build_adapter",
    "compute_features",
    "default_protocol",
    "demand_target",
    "demand_target_for",
    "evaluate",
    "generate_demand",
    "make_query",
    "mature_batch",
    "summarize",
]
