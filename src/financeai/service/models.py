"""Pydantic v2 transport models of the local research API v1 (the source of the OpenAPI snapshot).

Wire rules: unknown fields are rejected (``extra="forbid"``) and request models are strict
(no int/str/float coercion). Clocks are UTC epoch nanoseconds carried as canonical decimal
strings (int64, no sign/exponent/leading zeros); exact decimals (fees) are decimal strings.
Duplicate JSON keys and non-finite numbers are rejected before these models run (the route
parses bodies with :func:`financeai.core.serialization.loads_strict`).

Enumerations (venues, scenarios, artifact kinds, data roles) are derived from the domain and
artifact registries, so the snapshot changes whenever the domain does. Request models only
check the transport shape; semantic validation (strategy config, coverage, units) is done by
the same application validators Python and the CLI use, behind the strict canonical strategy
check of :mod:`financeai.service.canonical`. The synthetic ``start_ns`` is capped at ``2**62``
(:data:`~financeai.domains.market.synthetic.MAX_START_NS`) so replay-derived clocks fit int64.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Final, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, WithJsonSchema

from financeai.domains.market.instruments import INSTRUMENTS
from financeai.domains.market.scenario import demo_scenarios
from financeai.domains.market.synthetic import MAX_ROWS_PER_VENUE, MAX_SEED, MAX_START_NS
from financeai.service.artifacts import ARTIFACT_KINDS, DATA_ROLES

API_VERSION: Final = "1.0"
INT64_MAX: Final = 2**63 - 1
EPOCH_NS_PATTERN: Final = r"^(0|[1-9][0-9]{0,18})$"
DECIMAL_PATTERN: Final = r"^(0|[1-9][0-9]*)(\.[0-9]+)?$"
IDEMPOTENCY_KEY_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
VENUES: Final = tuple(sorted(INSTRUMENTS))
SCENARIOS: Final = tuple(demo_scenarios())
JOB_STATUSES: Final = ("QUEUED", "RUNNING", "CANCEL_REQUESTED", "SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED")


def _int64(value: str) -> str:
    if int(value) > INT64_MAX:
        raise ValueError("clock exceeds the int64 range")
    return value


def _synthetic_start(value: str) -> str:
    if int(value) > MAX_START_NS:
        raise ValueError(f"start_ns exceeds {MAX_START_NS} (2**62): replay-derived clocks would leave int64")
    return value


def _one_of(name: str, allowed: tuple[str, ...]) -> AfterValidator:
    def check(value: str) -> str:
        if value not in allowed:
            raise ValueError(f"unknown {name}; expected one of {list(allowed)}")
        return value

    return AfterValidator(check)


EpochNs = Annotated[
    str,
    StringConstraints(pattern=EPOCH_NS_PATTERN),
    AfterValidator(_int64),
    WithJsonSchema(
        {
            "type": "string",
            "pattern": EPOCH_NS_PATTERN,
            "description": "UTC epoch nanoseconds (int64) as a canonical decimal string",
        }
    ),
]
SyntheticStartNs = Annotated[
    str,
    StringConstraints(pattern=EPOCH_NS_PATTERN),
    AfterValidator(_synthetic_start),
    WithJsonSchema({"type": "string", "pattern": EPOCH_NS_PATTERN}),
]
DecimalString = Annotated[str, StringConstraints(pattern=DECIMAL_PATTERN)]
RunId = Annotated[str, StringConstraints(pattern=r"^run_[0-9a-f]{32}$")]
ArtifactId = Annotated[str, StringConstraints(pattern=r"^art_[0-9a-f]{32}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CampaignId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")]
VenueKey = Annotated[str, _one_of("venue", VENUES), WithJsonSchema({"type": "string", "enum": list(VENUES)})]
ScenarioName = Annotated[
    str, _one_of("scenario", SCENARIOS), WithJsonSchema({"type": "string", "enum": list(SCENARIOS)})
]
JobStatusName = Literal["QUEUED", "RUNNING", "CANCEL_REQUESTED", "SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"]
EvidenceScope = Literal["SYNTHETIC", "DEVELOPMENT", "FRESH_TEST", "NONE"]
EconomicStatus = Literal["NOT_EVALUATED", "VALID_ACCOUNT", "INVALID_ACCOUNT", "INCOMPLETE"]
ArtifactKind = Annotated[
    str,
    _one_of("artifact kind", tuple(ARTIFACT_KINDS)),
    WithJsonSchema({"type": "string", "enum": list(ARTIFACT_KINDS)}),
]
DataRole = Annotated[
    str, _one_of("data role", DATA_ROLES), WithJsonSchema({"type": "string", "enum": list(DATA_ROLES)})
]

EXAMPLE_STRATEGY: Final[dict[str, Any]] = {
    "schema": "financeai.strategy",
    "schema_version": "1.0",
    "name": "demo",
    "instruments": ["venue_a/ALPHA-USD", "venue_b/ALPHA-PERP"],
    "signal": {
        "kind": "lead_gap",
        "leader": "venue_b/ALPHA-PERP",
        "follower": "venue_a/ALPHA-USD",
        "window_ms": 20,
        "depth_units": 1000,
        "gate_ticks": 2,
    },
    "policy": {
        "direction": {"venue_a/ALPHA-USD": 1, "venue_b/ALPHA-PERP": 0},
        "size_units": {"venue_a/ALPHA-USD": 1000, "venue_b/ALPHA-PERP": 0},
        "enter": 0.5,
        "exit": -1000000000.0,
        "allow_short": True,
        "min_hold_ms": 0,
        "max_hold_ms": 50,
        "cooldown_ms": 0,
        "stop_bps": 0.0,
        "entry_slip_ticks": 1,
        "exit_slip_ticks": 20,
    },
    "session": {"warmup_s": 0, "entry_stop_s": 0.4, "flatten_s": 0.3, "exec_stop_s": 0.06},
}
EXAMPLE_REQUESTS: Final[tuple[dict[str, Any], ...]] = (
    {
        "schema_version": API_VERSION,
        "campaign_id": "demo",
        "job": {
            "kind": "synthetic_replay",
            "seed": 7,
            "venues": ["venue_a/ALPHA-USD", "venue_b/ALPHA-PERP"],
            "rows_per_venue": 300,
            "start_ns": "1700000000000000000",
            "scenario": "base",
            "strategy": EXAMPLE_STRATEGY,
        },
    },
    {
        "schema_version": API_VERSION,
        "campaign_id": "demo",
        "job": {
            "kind": "strategy_replay",
            "strategy_artifact_id": "art_" + "1" * 32,
            "cache_artifact_ids": ["art_" + "2" * 32, "art_" + "3" * 32],
            "scenario": "higher_fees",
        },
    },
)


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SyntheticReplayRequest(_Request):
    """Replay an inline strategy on a deterministic synthetic two-venue market."""

    kind: Literal["synthetic_replay"]
    seed: int = Field(ge=0, le=MAX_SEED)
    venues: list[VenueKey] = Field(min_length=1, max_length=len(VENUES))
    rows_per_venue: int = Field(default=300, ge=2, le=MAX_ROWS_PER_VENUE)
    start_ns: SyntheticStartNs = Field(
        default="0",
        description=f"Clock offset of the generated market: UTC epoch nanoseconds as a canonical decimal "
        f"string, at most {MAX_START_NS} (2**62, 2116-02-20) so that the clocks a replay "
        f"derives (latency, acknowledgement, cooldown) stay inside int64",
    )
    scenario: ScenarioName
    strategy: dict[str, Any] = Field(
        description="A financeai.strategy document, checked strictly and canonicalized before StrategyConfig "
        "validates it: integer parameters (direction, size_units, size_units_cap, max_clips, *_ms "
        "including abs_cap_ms and early_check_ms, *_ticks, unit-suffixed signal parameters and the "
        "counts ticks and levels) are JSON integers (no float or bool spellings), binary64 parameters "
        "(enter, exit, stop_bps, early_markout_bps, session seconds, composite component scale) accept "
        "integers or floats and are stored as floats, allow_short, refresh_hold and reverse are "
        "booleans; nested signal objects (gate, fresh, components) and exit_signal follow the signal "
        "rules. The canonical form is what is hashed for idempotency, reported as the admitted job "
        "and run."
    )

    def to_domain(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "seed": self.seed,
            "venues": list(self.venues),
            "rows_per_venue": self.rows_per_venue,
            "start_ns": int(self.start_ns),
            "scenario": self.scenario,
            "strategy": dict(self.strategy),
        }


class StrategyReplayRequest(_Request):
    """Replay a registered strategy artifact on registered tape-cache artifacts."""

    kind: Literal["strategy_replay"]
    strategy_artifact_id: ArtifactId
    cache_artifact_ids: list[ArtifactId] = Field(min_length=1, max_length=16)
    scenario: ScenarioName

    def to_domain(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "strategy_artifact_id": self.strategy_artifact_id,
            "cache_artifact_ids": list(self.cache_artifact_ids),
            "scenario": self.scenario,
        }


JobRequest = Annotated[SyntheticReplayRequest | StrategyReplayRequest, Field(discriminator="kind")]
_JOB_ADAPTER: Final[TypeAdapter[SyntheticReplayRequest | StrategyReplayRequest]] = TypeAdapter(JobRequest)


def job_from_domain(doc: Mapping[str, Any]) -> SyntheticReplayRequest | StrategyReplayRequest:
    """Transport view of a normalized job mapping (clocks become decimal strings)."""
    wire = dict(doc)
    if wire.get("kind") == "synthetic_replay":
        wire["start_ns"] = str(wire["start_ns"])
    return _JOB_ADAPTER.validate_python(wire)


class RunRequest(_Request):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, json_schema_extra={"examples": list(EXAMPLE_REQUESTS)}
    )

    schema_version: Literal["1.0"]
    campaign_id: CampaignId = Field(default="default", description="Idempotency keys are unique per campaign")
    job: JobRequest


class RunAccepted(_Response):
    schema_version: Literal["1.0"] = API_VERSION
    run_id: RunId
    status: JobStatusName
    status_url: str
    cancel_url: str
    replayed: bool = Field(description="True when the idempotency key matched an existing identical request")


class Progress(_Response):
    phase: str
    completed_steps: int
    total_steps: int


class Failure(_Response):
    code: str
    message: str


class RunResult(_Response):
    scenario: str
    account_valid: bool = Field(description="Every venue ended flat and the position cap held")
    invalid_reason: str | None
    net_pnl: float | None = Field(
        description="Net result (binary64); null for an invalid account or multiple settlement currencies"
    )
    settlement_currencies: list[str] = Field(default_factory=list)
    pnl_by_currency: dict[str, float] | None = None
    fees_per_unit: dict[str, DecimalString] = Field(default_factory=dict)
    round_trips: int
    fills: int
    orders: int
    decisions: int
    first_event_ns: EpochNs
    last_event_ns: EpochNs
    fees: dict[str, DecimalString] = Field(description="Taker fee per instrument as an exact decimal string")


class RunStatus(_Response):
    schema_version: Literal["1.0"] = API_VERSION
    run_id: RunId
    campaign_id: str
    kind: Literal["synthetic_replay", "strategy_replay"]
    status: JobStatusName
    attempt_id: str | None
    generation: int
    job: JobRequest = Field(description="The normalized job that was admitted (defaults filled in)")
    progress: Progress | None
    failure: Failure | None
    result: RunResult | None
    artifact_ids: list[ArtifactId]
    evidence_scope: list[EvidenceScope]
    economic_status: EconomicStatus = Field(
        description="Account validity of a finished replay; never a claim of economic success"
    )
    created_at_ns: EpochNs
    updated_at_ns: EpochNs
    started_at_ns: EpochNs | None
    finished_at_ns: EpochNs | None
    status_url: str
    cancel_url: str


class ArtifactMetadata(_Response):
    schema_version: Literal["1.0"] = API_VERSION
    artifact_id: ArtifactId
    kind: ArtifactKind
    media_type: str
    schema_type: str = Field(description="Schema identifier of the content, e.g. financeai.run_summary/1")
    sha256: Sha256
    size_bytes: int
    data_role: DataRole
    run_id: RunId | None
    created_at_ns: EpochNs
    provenance: dict[str, Any]


class ErrorBody(_Response):
    code: str
    message: str
    request_id: str
    details: dict[str, Any]


class ErrorEnvelope(_Response):
    error: ErrorBody


class Health(_Response):
    status: Literal["ok"]
    readiness: Literal["READY", "NOT_READY"] = Field(description="Worker admission readiness (not economic validity)")
    api_version: Literal["1.0"] = API_VERSION
