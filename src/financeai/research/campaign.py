"""Reproducible research records with atomic writes, versioned artifacts, and sealed evaluations.

Each campaign records data exposure, trials, artifacts, configuration freezes, and
held-out evaluations. Hash chains detect edits; canonical records give repeatable
identities. POSIX advisory locks serialize concurrent writers. Retry and diagnostic
records preserve the distinction between new evidence and repeated computation.
"""

from __future__ import annotations

import csv
import functools
import io
import json
import math
import numbers
import os
import re
import sys
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO, Final

from financeai.core.errors import (
    IncompatibleSchemaError,
    InvalidConfigError,
    InvalidDataError,
    UnsupportedCapabilityError,
)
from financeai.core.hashing import sha256_bytes, sha256_file, sha256_json
from financeai.core.serialization import canonical_json, loads_strict


class ImmutableRecordError(InvalidDataError):
    """An immutable record (attempt, seal, opening, artifact version, trial id) exists."""

    code = "immutable_record"


# ---------------------------------------------------------------------------
# Research record vocabularies
# ---------------------------------------------------------------------------


class ExecutionStatus(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    DEFERRED = "DEFERRED"
    BLOCKED = "BLOCKED"


class EngineeringStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"


class EvidenceScope(StrEnum):
    SYNTHETIC = "SYNTHETIC"
    DEVELOPMENT = "DEVELOPMENT"
    FRESH_TEST = "FRESH_TEST"
    NONE = "NONE"


class EconomicOutcome(StrEnum):
    NOT_EVALUATED = "NOT_EVALUATED"
    NO_CANDIDATE_PASSED = "NO_CANDIDATE_PASSED"
    DEVELOPMENT_POSITIVE = "DEVELOPMENT_POSITIVE"
    FROZEN_FOR_TEST = "FROZEN_FOR_TEST"
    TEST_PASSED = "TEST_PASSED"
    TEST_FAILED = "TEST_FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"


class TrialStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


class TrialValidity(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    INCOMPLETE_OBSERVATION = "INCOMPLETE_OBSERVATION"


class ResearchTimingMode(StrEnum):
    """Two explicitly distinguished timing modes (never an unlabelled mixture)."""

    HISTORICAL_REPLAY = "HISTORICAL_REPLAY"
    FULL_PIPELINE = "FULL_PIPELINE"


class DataRole(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    DEVELOPMENT = "development"
    FRESH_TEST = "fresh_test"
    RESERVED = "reserved"
    UNASSIGNED = "unassigned"
    UNAVAILABLE = "unavailable"


class ExposureKind(StrEnum):
    HISTORICAL_PRIOR_CAMPAIGN = "HISTORICAL_PRIOR_CAMPAIGN"
    METADATA_ONLY = "METADATA_ONLY"
    MECHANICS_SAMPLE = "MECHANICS_SAMPLE"
    RAW_VALUES = "RAW_VALUES"
    DESCRIPTIVE_SUMMARY = "DESCRIPTIVE_SUMMARY"
    PRIOR_RESULTS = "PRIOR_RESULTS"
    DERIVED_CACHE = "DERIVED_CACHE"
    SELECTION = "SELECTION"
    FRESH_TEST_OPENING = "FRESH_TEST_OPENING"


class DeliveryProfile(StrEnum):
    RESEARCH_RUN = "research_run"
    FULL_LIBRARY = "full_library"


class AdmissionDecision(StrEnum):
    READY_TO_OPEN = "READY_TO_OPEN"
    NO_TEST_OPENING = "NO_TEST_OPENING"


class EvaluationMode(StrEnum):
    FIXED = "FIXED"
    PREQUENTIAL = "PREQUENTIAL"


class EvaluationStatus(StrEnum):
    NOT_EVALUATED = "NOT_EVALUATED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


# ---------------------------------------------------------------------------
# Versioned record templates; callers receive fresh copies
# ---------------------------------------------------------------------------

_STAGE_TEMPLATE_JSON: Final = """{
  "schema_version": 5, "campaign_id": null, "stage_id": null, "attempt_id": null,
  "execution_status": "NOT_STARTED", "engineering_status": "NOT_RUN",
  "evidence_scope": ["NONE"], "economic_outcome": "NOT_EVALUATED",
  "consumed_artifacts": [], "produced_artifacts": [], "commands": [], "changed_files": [],
  "data_exposures": [], "failed_checks": [], "defer_or_block_reason": null,
  "reentry_criterion": null, "next_stage": null, "started_at": null, "finished_at": null,
  "delivery_profile": "research_run", "development_record": null,
  "source_provenance": null
}"""

_FREEZE_TEMPLATE_JSON: Final = """{
  "schema_version": 5, "status": "DRAFT", "campaign_id": null, "protocol_hash": null,
  "selected_artifact_closure": [], "optional_branch_dispositions": {}, "code_hash": null,
  "environment_hash": null, "configuration_hash": null, "initial_model_hash": null,
  "initial_online_state_hash": null, "seed_policy": null,
  "selection_refit_calibration_rule_hash": null, "evaluation_mode": "FIXED",
  "online_rule_hash": null, "eligible_feedback": null, "pre_test_gate_result": "NOT_EVALUATED",
  "test_data_role_manifest": null, "suite_plan": null, "reason": null,
  "scenario_initial_state_manifest": null, "scenario_order": [], "package_release_hash": null,
  "performance_implementation_hash": null, "stack_decision_hash": null,
  "engineering_plan_hash": null, "engineering_verification_hash": null,
  "api_conformance_hash": null, "schema_version_manifest": null,
  "delivery_profile": "research_run", "phase": "PRE_EVALUATION", "sealed_at": null,
  "admission_decision": "NOT_EVALUATED", "opening_policy_hash": null,
  "selected_components_hash": null, "source_provenance": null
}"""

_EVALUATION_TEMPLATE_JSON: Final = """{
  "schema_version": 1, "phase": "EVALUATION_DISPOSITION", "status": "NOT_EVALUATED",
  "freeze_record_sha256": null, "opening_record_id": null, "result_artifacts": [],
  "economic_decision": "NOT_EVALUATED", "reason": null, "created_at": null
}"""


def stage_record_template() -> dict[str, Any]:
    """Fresh copy of ``templates/stage-record.json`` (schema 5)."""
    doc: dict[str, Any] = loads_strict(_STAGE_TEMPLATE_JSON)
    return doc


def freeze_template() -> dict[str, Any]:
    """Fresh copy of ``templates/freeze.json`` (schema 5)."""
    doc: dict[str, Any] = loads_strict(_FREEZE_TEMPLATE_JSON)
    return doc


def evaluation_record_template() -> dict[str, Any]:
    """Fresh copy of ``templates/evaluation-record.json`` (schema 1)."""
    doc: dict[str, Any] = loads_strict(_EVALUATION_TEMPLATE_JSON)
    return doc


STAGE_RECORD_FIELDS: Final = tuple(stage_record_template())
FREEZE_FIELDS: Final = tuple(freeze_template())
EVALUATION_RECORD_FIELDS: Final = tuple(evaluation_record_template())
ARTIFACT_RECORD_FIELDS: Final = (
    "artifact_id",
    "producer_stage",
    "producer_attempt",
    "path",
    "sha256",
    "evidence_scope",
    "depends_on",
    "description",
)
TRIAL_LEDGER_HEADER: Final = (
    "campaign_id",
    "protocol_id",
    "trial_id",
    "stage_id",
    "fit_job_id",
    "policy_config_id",
    "account_id",
    "scenario",
    "timing_mode",
    "dataset_id",
    "seed",
    "status",
    "validity",
    "net_pnl",
    "fees",
    "turnover_units",
    "residual_inventory",
    "evidence_scope",
    "artifact_path",
    "notes",
)
EXPOSURE_LEDGER_HEADER: Final = (
    "campaign_id",
    "dataset_id",
    "source",
    "period",
    "assigned_role",
    "previously_exposed",
    "exposure_kind",
    "exposed_at",
    "reason",
    "protocol_id",
    "manifest_hash",
)
REQUIRED_SEAL_HASHES: Final = (
    "protocol_hash",
    "code_hash",
    "environment_hash",
    "configuration_hash",
    "selected_components_hash",
    "package_release_hash",
    "performance_implementation_hash",
)
"""Digests required by a reproducible evaluation seal."""
FREEZE_HASH_FIELDS: Final = tuple(k for k in FREEZE_FIELDS if k.endswith("_hash"))
FORBIDDEN_SEAL_KEYS: Final = frozenset({"opening_record", "result_paths", "decision", "economic_decision"})
"""Future results never belong in the seal, not even as null."""
LEDGER_FILES: Final = MappingProxyType(
    {
        "artifact-registry": "artifact-registry.jsonl",
        "data-exposure": "data-exposure.csv",
        "trial-ledger": "trial-ledger.csv",
    }
)
MANIFEST_EXCLUDED_NAMES: Final = frozenset({"__pycache__", ".DS_Store"})
"""Interpreter/OS noise left out of directory manifests (plus this module's temp files)."""

DIAGNOSTIC_LABEL: Final = "DIAGNOSTIC"
"""First word of a trial's ``notes`` marking a labelled diagnostic rerun."""

REGISTRY_SCHEMA_VERSION: Final = 1
OPENING_SCHEMA_VERSION: Final = 1
GENESIS_SHA256: Final = "0" * 64
_DIRECTORY_MANIFEST_SCHEMA: Final = "financeai.directory_manifest"
_LOCK_NAME: Final = ".campaign.lock"
_SEED_LIMIT: Final = 2**64

_ID_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")
_INT_PATTERN: Final = re.compile(r"0|-?[1-9][0-9]*")
_FORBIDDEN_CHARS: Final = re.compile("[\x00-\x1f\x7f-\x9f\u2028\u2029\ud800-\udfff]")
"""Unicode categories Cc, Zl, Zp and Cs: anything that could break a CSV/JSONL line."""
_TEMP_PATTERN: Final = re.compile(r"\..+\.tmp")
_DIAGNOSTIC_PATTERN: Final = re.compile(rf"{DIAGNOSTIC_LABEL}\b")
_RETRY_PATTERN: Final = re.compile(r"RETRY_OF=(\S+)")
"""A retry citation in a trial's notes: ``RETRY_OF=<trial_id>`` (ids carry no whitespace)."""
_RETRY_IDENTITY: Final = (
    "stage_id",
    "fit_job_id",
    "policy_config_id",
    "account_id",
    "scenario",
    "timing_mode",
    "dataset_id",
    "seed",
    "evidence_scope",
)
"""The columns that make two ledger rows the same trial: a retry re-runs exactly this; outcome
columns (status, validity, scores, inventory, artifact path, notes) may differ."""
_FINAL_DISPOSITIONS: Final = frozenset({EvaluationStatus.COMPLETED, EvaluationStatus.NOT_EVALUATED})
"""The suite is evaluated once (COMPLETED) or not at all (NOT_EVALUATED); only a FAILED or
INTERRUPTED disposition may be followed by a continuation."""

# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidDataError(f"{name} must be a non-empty string (use None when absent)", context={"got": value})
    if value != value.strip() or _FORBIDDEN_CHARS.search(value):
        raise InvalidDataError(
            f"{name} has surrounding whitespace or control/line-break characters", context={"got": value}
        )
    return value


def _optional_text(name: str, value: object) -> str | None:
    return None if value is None else _text(name, value)


def _note(name: str, value: object) -> str | None:
    """Optional free text for JSON records (line breaks allowed, blank refused)."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InvalidDataError(f"{name} must be a non-blank string or None", context={"got": value})
    return value


def _identifier(
    name: str, value: object, error: type[InvalidConfigError | InvalidDataError] = InvalidConfigError
) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise error(f"{name} must be a safe path component ([A-Za-z0-9][A-Za-z0-9._-]*)", context={"got": value})
    return value


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise InvalidDataError(f"{name} must be a lowercase 64-hex SHA-256", context={"got": value})
    return value


def _member[E: StrEnum](enum: type[E], value: object, name: str) -> E:
    if isinstance(value, enum):
        return value
    if not isinstance(value, str):
        raise InvalidDataError(f"{name} must be a string", context={"got": value})
    try:
        return enum(value)
    except ValueError:
        raise InvalidDataError(
            f"{name} is not an allowed value", context={"got": value, "allowed": [m.value for m in enum]}
        ) from None


def _utc_text(name: str, value: object) -> str:
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str):
        try:
            stamp = datetime.fromisoformat(value)
        except ValueError as exc:
            raise InvalidDataError(f"{name} is not an ISO-8601 timestamp", context={"got": value}) from exc
    else:
        raise InvalidDataError(f"{name} must be a timezone-aware timestamp", context={"got": value})
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise InvalidDataError(f"{name} needs an explicit timezone", context={"got": value})
    return stamp.astimezone(UTC).isoformat(timespec="microseconds")


def _instant(text: str) -> datetime:
    """The UTC instant of a normalised timestamp (see :func:`_utc_text`)."""
    return datetime.fromisoformat(text)


def _not_before(name: str, stamp: str, bound: str, what: str) -> None:
    """Refuse a record time earlier than the record it depends on."""
    if _instant(stamp) < _instant(bound):
        raise InvalidDataError(f"{name} is earlier than {what}", context={name: stamp, what: bound})


def _relative_path(name: str, value: str) -> str:
    """A canonical POSIX path relative to the campaign root: no leading ``/``, no ``..``, ``.`` or
    empty component and no trailing slash, so one location has one spelling in the records."""
    if value.startswith("/") or ".." in value.split("/"):
        raise InvalidDataError(f"{name} must be relative to the campaign root", context={name: value})
    if value == "." or PurePosixPath(value).as_posix() != value:
        raise InvalidDataError(
            f"{name} must be a canonical relative path (no '.', '//' or trailing '/')", context={name: value}
        )
    return value


def _integer(name: str, value: object, *, minimum: int | None = None, limit: int | None = None) -> int:
    if type(value) is int:
        out = value
    elif isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise InvalidDataError(f"{name} must be an integer", context={"got": value})
    else:
        out = int(value)
    if (minimum is not None and out < minimum) or (limit is not None and out >= limit):
        raise InvalidDataError(f"{name} is out of range", context={"got": out, "minimum": minimum, "limit": limit})
    return out


def _finite(name: str, value: object) -> float:
    if type(value) is float:
        out = value
    elif isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise InvalidDataError(f"{name} must be a real number", context={"got": value})
    else:
        out = float(value)
    if not math.isfinite(out):
        raise InvalidDataError(f"{name} must be finite (absent values are None, never NaN)", context={"got": value})
    return out


def _scopes(value: object, *, allow_text: bool) -> tuple[EvidenceScope, ...]:
    items: object = [value] if allow_text and isinstance(value, str) else value
    if not isinstance(items, list | tuple) or not items:
        raise InvalidDataError("evidence_scope must be a non-empty list", context={"got": value})
    members = [_member(EvidenceScope, item, "evidence_scope") for item in items]
    if len(set(members)) != len(members):
        raise InvalidDataError("evidence_scope repeats a value", context={"got": value})
    if EvidenceScope.NONE in members and len(members) > 1:
        raise InvalidDataError("evidence_scope NONE excludes every other scope", context={"got": value})
    order = tuple(EvidenceScope)
    return tuple(sorted(members, key=order.index))


def _inventory(value: object) -> tuple[tuple[str, int], ...] | None:
    if value is None:
        return None
    pairs: Iterable[tuple[object, object]]
    if isinstance(value, Mapping):
        pairs = value.items()
    elif isinstance(value, tuple):
        pairs = value
    else:
        raise InvalidDataError(
            "residual_inventory must map instruments to integer stored quantity units", context={"got": value}
        )
    out = {_text("residual_inventory key", k): _integer("residual_inventory value", v) for k, v in pairs}
    return tuple(sorted(out.items()))


def _strict_json(value: Any, what: str) -> Any:
    """Round-trip through canonical JSON: refuses NaN/inf and non-JSON types."""
    try:
        return loads_strict(canonical_json(value))
    except (InvalidConfigError, TypeError) as exc:
        raise InvalidDataError(f"{what} is not strict JSON", context={"error": str(exc)}) from exc


def _load_json(data: bytes, what: str) -> Any:
    try:
        return loads_strict(data.decode("utf-8"))
    except (InvalidConfigError, UnicodeDecodeError) as exc:
        raise InvalidDataError(f"{what} is not strict JSON", context={"error": str(exc)}) from exc


def _pretty_json(doc: Any) -> bytes:
    text = json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _utc_now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A pinned artifact version: what consumers, stage records and evaluations cite."""

    artifact_id: str
    version: int
    sha256: str

    def __post_init__(self) -> None:
        _text("artifact_id", self.artifact_id)
        _integer("version", self.version, minimum=1)
        _sha256("sha256", self.sha256)

    def to_json(self) -> dict[str, Any]:
        return {"artifact_id": self.artifact_id, "version": self.version, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One registered artifact version (standard record fields plus identity data)."""

    campaign_id: str
    artifact_id: str
    version: int
    producer_stage: str | None
    producer_attempt: str
    path: str
    """POSIX path relative to the campaign root."""
    kind: str
    """``"file"`` or ``"directory"`` (hash of a sorted file manifest)."""
    sha256: str
    size_bytes: int
    evidence_scope: tuple[EvidenceScope, ...]
    depends_on: tuple[ArtifactRef, ...]
    description: str | None
    registered_at: str

    def __post_init__(self) -> None:
        _identifier("campaign_id", self.campaign_id, InvalidDataError)
        _text("artifact_id", self.artifact_id)
        _integer("version", self.version, minimum=1)
        _optional_text("producer_stage", self.producer_stage)
        _text("producer_attempt", self.producer_attempt)
        _relative_path("path", _text("path", self.path))
        if self.kind not in ("file", "directory"):
            raise InvalidDataError("artifact kind must be file or directory", context={"got": self.kind})
        _sha256("sha256", self.sha256)
        _integer("size_bytes", self.size_bytes, minimum=0)
        object.__setattr__(self, "evidence_scope", _scopes(self.evidence_scope, allow_text=False))
        if not all(isinstance(d, ArtifactRef) for d in self.depends_on):
            raise InvalidDataError("depends_on must hold ArtifactRef values")
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        _note("description", self.description)
        object.__setattr__(self, "registered_at", _utc_text("registered_at", self.registered_at))

    @property
    def ref(self) -> ArtifactRef:
        return ArtifactRef(self.artifact_id, self.version, self.sha256)

    def registration_key(self) -> tuple[object, ...]:
        """Everything but the version number and registration time."""
        return (
            self.artifact_id,
            self.producer_stage,
            self.producer_attempt,
            self.path,
            self.kind,
            self.sha256,
            self.size_bytes,
            self.evidence_scope,
            self.depends_on,
            self.description,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "producer_stage": self.producer_stage,
            "producer_attempt": self.producer_attempt,
            "path": self.path,
            "kind": self.kind,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "evidence_scope": [s.value for s in self.evidence_scope],
            "depends_on": [d.to_json() for d in self.depends_on],
            "description": self.description,
            "registered_at": self.registered_at,
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> ArtifactRecord:
        expected = {
            "campaign_id",
            "artifact_id",
            "version",
            "producer_stage",
            "producer_attempt",
            "path",
            "kind",
            "sha256",
            "size_bytes",
            "evidence_scope",
            "depends_on",
            "description",
            "registered_at",
        }
        if set(doc) != expected:
            raise IncompatibleSchemaError("artifact record fields differ from schema", context={"fields": sorted(doc)})
        deps = doc["depends_on"]
        if not isinstance(deps, list) or not all(isinstance(d, dict) for d in deps):
            raise InvalidDataError("depends_on must be a list of references")
        return cls(
            campaign_id=doc["campaign_id"],
            artifact_id=doc["artifact_id"],
            version=doc["version"],
            producer_stage=doc["producer_stage"],
            producer_attempt=doc["producer_attempt"],
            path=doc["path"],
            kind=doc["kind"],
            sha256=doc["sha256"],
            size_bytes=doc["size_bytes"],
            evidence_scope=_scopes(doc["evidence_scope"], allow_text=False),
            depends_on=tuple(_ref_from_json(d) for d in deps),
            description=doc["description"],
            registered_at=doc["registered_at"],
        )


def _ref_from_json(doc: Mapping[str, Any]) -> ArtifactRef:
    if set(doc) != {"artifact_id", "version", "sha256"}:
        raise InvalidDataError("artifact reference fields differ from schema", context={"fields": sorted(doc)})
    return ArtifactRef(doc["artifact_id"], doc["version"], doc["sha256"])


ArtifactSpec = str | ArtifactRef | ArtifactRecord | Mapping[str, Any]
"""An artifact citation: an id (latest version), a pinned ref/record, or a ref mapping; a
mapping without ``version`` pins the latest version whose hash is its ``sha256``."""


@dataclass(frozen=True, slots=True)
class ExposureRecord:
    """One row of ``data-exposure.csv``."""

    campaign_id: str
    dataset_id: str
    source: str
    period: str
    assigned_role: DataRole
    previously_exposed: bool
    exposure_kind: ExposureKind
    exposed_at: str
    reason: str
    protocol_id: str | None = None
    manifest_hash: str | None = None

    def __post_init__(self) -> None:
        _identifier("campaign_id", self.campaign_id, InvalidDataError)
        for name in ("dataset_id", "source", "period", "reason"):
            _text(name, getattr(self, name))
        role = _member(DataRole, self.assigned_role, "assigned_role")
        kind = _member(ExposureKind, self.exposure_kind, "exposure_kind")
        object.__setattr__(self, "assigned_role", role)
        object.__setattr__(self, "exposure_kind", kind)
        if not isinstance(self.previously_exposed, bool):
            raise InvalidDataError("previously_exposed must be a bool", context={"got": self.previously_exposed})
        object.__setattr__(self, "exposed_at", _utc_text("exposed_at", self.exposed_at))
        _optional_text("protocol_id", self.protocol_id)
        if self.manifest_hash is not None:
            _sha256("manifest_hash", self.manifest_hash)
        if kind is ExposureKind.FRESH_TEST_OPENING and role is not DataRole.FRESH_TEST:
            raise InvalidDataError("a fresh-test opening must carry the fresh_test role")
        if role is DataRole.FRESH_TEST and (
            self.previously_exposed or kind not in (ExposureKind.METADATA_ONLY, ExposureKind.FRESH_TEST_OPENING)
        ):
            raise InvalidDataError(
                "fresh test data admits only metadata lookups before its single opening",
                context={"dataset_id": self.dataset_id},
            )
        if kind is ExposureKind.HISTORICAL_PRIOR_CAMPAIGN and not self.previously_exposed:
            raise InvalidDataError("a prior-campaign row documents earlier exposure (previously_exposed must be true)")

    def to_row(self) -> list[str]:
        return [
            self.campaign_id,
            self.dataset_id,
            self.source,
            self.period,
            self.assigned_role.value,
            "true" if self.previously_exposed else "false",
            self.exposure_kind.value,
            self.exposed_at,
            self.reason,
            _cell(self.protocol_id),
            _cell(self.manifest_hash),
        ]

    @classmethod
    def from_row(cls, row: Sequence[str]) -> ExposureRecord:
        return cls(
            campaign_id=row[0],
            dataset_id=row[1],
            source=row[2],
            period=row[3],
            assigned_role=_member(DataRole, row[4], "assigned_role"),
            previously_exposed=_bool_cell("previously_exposed", row[5]),
            exposure_kind=_member(ExposureKind, row[6], "exposure_kind"),
            exposed_at=row[7],
            reason=row[8],
            protocol_id=_text_cell(row[9]),
            manifest_hash=_text_cell(row[10]),
        )


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One row of ``trial-ledger.csv``: one counted fit job, configuration or replay.

    A score (``net_pnl``) exists only for a completed, valid account replay whose recorded
    terminal inventory (``residual_inventory``, at least one venue) is zero everywhere;
    otherwise it is ``None`` and the CSV cell is empty (never ``0``, never ``nan``).
    ``artifact_path`` is relative to the campaign root.
    """

    campaign_id: str
    protocol_id: str
    trial_id: str
    stage_id: str
    status: TrialStatus
    validity: TrialValidity
    evidence_scope: EvidenceScope
    fit_job_id: str | None = None
    policy_config_id: str | None = None
    account_id: str | None = None
    scenario: str | None = None
    timing_mode: ResearchTimingMode | None = None
    dataset_id: str | None = None
    seed: int | None = None
    net_pnl: float | None = None
    fees: float | None = None
    turnover_units: int | None = None
    residual_inventory: tuple[tuple[str, int], ...] | None = None
    artifact_path: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        _identifier("campaign_id", self.campaign_id, InvalidDataError)
        for name in ("protocol_id", "trial_id", "stage_id"):
            _text(name, getattr(self, name))
        for name in (
            "fit_job_id",
            "policy_config_id",
            "account_id",
            "scenario",
            "dataset_id",
            "artifact_path",
            "notes",
        ):
            _optional_text(name, getattr(self, name))
        if self.artifact_path is not None:
            _relative_path("artifact_path", self.artifact_path)
        status = _member(TrialStatus, self.status, "status")
        validity = _member(TrialValidity, self.validity, "validity")
        scope = _member(EvidenceScope, self.evidence_scope, "evidence_scope")
        timing = None if self.timing_mode is None else _member(ResearchTimingMode, self.timing_mode, "timing_mode")
        seed = None if self.seed is None else _integer("seed", self.seed, minimum=0, limit=_SEED_LIMIT)
        net = None if self.net_pnl is None else _finite("net_pnl", self.net_pnl)
        fees = None if self.fees is None else _finite("fees", self.fees)
        turnover = None if self.turnover_units is None else _integer("turnover_units", self.turnover_units, minimum=0)
        inventory = _inventory(self.residual_inventory)
        for name, value in (
            ("status", status),
            ("validity", validity),
            ("evidence_scope", scope),
            ("timing_mode", timing),
            ("seed", seed),
            ("net_pnl", net),
            ("fees", fees),
            ("turnover_units", turnover),
            ("residual_inventory", inventory),
        ):
            object.__setattr__(self, name, value)
        if self.fit_job_id is None and self.policy_config_id is None and self.account_id is None:
            raise InvalidDataError(
                "a trial row must count a fit job, a configuration or a replay", context={"trial_id": self.trial_id}
            )
        if self.account_id is not None and self.policy_config_id is None:
            raise InvalidDataError("an account replay needs its policy_config_id", context={"trial_id": self.trial_id})
        if net is not None:
            if self.account_id is None or status is not TrialStatus.COMPLETED or validity is not TrialValidity.VALID:
                raise InvalidDataError(
                    "net_pnl exists only for a completed, valid account replay", context={"trial_id": self.trial_id}
                )
            # VALID alone is no evidence of flatness: the terminal inventory itself must be on record.
            if not inventory or any(q != 0 for _, q in inventory):
                raise InvalidDataError(
                    "a score requires the recorded terminal inventory of a flat account "
                    "(residual_inventory with every venue at 0)",
                    context={"trial_id": self.trial_id},
                )
        if scope in (EvidenceScope.DEVELOPMENT, EvidenceScope.FRESH_TEST) and self.dataset_id is None:
            raise InvalidDataError(
                "development or fresh-test evidence must name its dataset", context={"trial_id": self.trial_id}
            )

    @property
    def counts_account(self) -> bool:
        return self.account_id is not None

    @property
    def retry_of(self) -> str | None:
        """The trial id cited as ``RETRY_OF=<trial_id>`` in the notes, if any (unchecked)."""
        cited = None if self.notes is None else _RETRY_PATTERN.search(self.notes)
        return None if cited is None else cited.group(1)

    def same_trial_as(self, other: TrialRecord) -> bool:
        """``True`` when every identity column (:data:`_RETRY_IDENTITY`) equals ``other``'s."""
        return all(getattr(self, name) == getattr(other, name) for name in _RETRY_IDENTITY)

    def to_row(self) -> list[str]:
        inventory = "" if self.residual_inventory is None else canonical_json(dict(self.residual_inventory))
        return [
            self.campaign_id,
            self.protocol_id,
            self.trial_id,
            self.stage_id,
            _cell(self.fit_job_id),
            _cell(self.policy_config_id),
            _cell(self.account_id),
            _cell(self.scenario),
            _cell(self.timing_mode),
            _cell(self.dataset_id),
            _int_cell(self.seed),
            self.status.value,
            self.validity.value,
            _float_cell(self.net_pnl),
            _float_cell(self.fees),
            _int_cell(self.turnover_units),
            inventory,
            self.evidence_scope.value,
            _cell(self.artifact_path),
            _cell(self.notes),
        ]

    @classmethod
    def from_row(cls, row: Sequence[str]) -> TrialRecord:
        timing = _text_cell(row[8])
        return cls(
            campaign_id=row[0],
            protocol_id=row[1],
            trial_id=row[2],
            stage_id=row[3],
            fit_job_id=_text_cell(row[4]),
            policy_config_id=_text_cell(row[5]),
            account_id=_text_cell(row[6]),
            scenario=_text_cell(row[7]),
            timing_mode=None if timing is None else _member(ResearchTimingMode, timing, "timing_mode"),
            dataset_id=_text_cell(row[9]),
            seed=_int_from_cell("seed", row[10]),
            status=_member(TrialStatus, row[11], "status"),
            validity=_member(TrialValidity, row[12], "validity"),
            net_pnl=_float_from_cell("net_pnl", row[13]),
            fees=_float_from_cell("fees", row[14]),
            turnover_units=_int_from_cell("turnover_units", row[15]),
            residual_inventory=_inventory_from_cell(row[16]),
            evidence_scope=_member(EvidenceScope, row[17], "evidence_scope"),
            artifact_path=_text_cell(row[18]),
            notes=_text_cell(row[19]),
        )


@dataclass(frozen=True, slots=True)
class TrialCounts:
    """Budget counters: fit jobs, configurations and account replays are distinct objects.

    Idempotent retries (rows whose notes cite ``RETRY_OF=<trial_id>`` of an earlier row with the
    same identity columns) are counted in ``retries`` only; ``rows`` counts every ledger row.
    A chain of retries is one trial: ``account_replays`` counts chains with an account and
    ``valid_accounts`` the chains with at least one COMPLETED and VALID row.
    """

    rows: int
    fit_jobs: int
    policy_configs: int
    account_replays: int
    valid_accounts: int
    retries: int = 0


@dataclass(frozen=True, slots=True)
class OpenedDataset:
    """A dataset identity (content hash) opened as fresh test data."""

    dataset_id: str
    sha256: str
    source: str
    period: str

    def __post_init__(self) -> None:
        for name in ("dataset_id", "source", "period"):
            _text(name, getattr(self, name))
        _sha256("sha256", self.sha256)

    def to_json(self) -> dict[str, str]:
        return {"dataset_id": self.dataset_id, "sha256": self.sha256, "source": self.source, "period": self.period}


@dataclass(frozen=True, slots=True)
class RecordFile:
    """A written record: where it lives and the SHA-256 of its exact bytes."""

    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class FreezeSeal:
    path: Path
    sha256: str
    """SHA-256 of the seal's exact bytes (equal to the hash of its canonical JSON)."""
    document: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    """Prefix identity of an append-only ledger at one moment."""

    name: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _TestState:
    """The seal plus every opening and evaluation record citing it (read under the lock)."""

    seal: FreezeSeal
    opened: Mapping[str, frozenset[str]]
    """Opening record id -> the dataset ids it opened."""
    opened_at: Mapping[str, str]
    """Opening record id -> its normalised UTC ``opened_at``."""
    dispositions: tuple[EvaluationStatus, ...]

    @property
    def sealed_at(self) -> str:
        stamp: str = self.seal.document["sealed_at"]
        return stamp

    @property
    def final(self) -> bool:
        return any(status in _FINAL_DISPOSITIONS for status in self.dispositions)


# ---------------------------------------------------------------------------
# CSV cells (canonical forms only; anything else is refused on read)
# ---------------------------------------------------------------------------


def _cell(value: str | None) -> str:
    return "" if value is None else str(value)


def _text_cell(cell: str) -> str | None:
    return None if cell == "" else cell


def _int_cell(value: int | None) -> str:
    return "" if value is None else str(value)


def _float_cell(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _bool_cell(name: str, cell: str) -> bool:
    if cell not in ("true", "false"):
        raise InvalidDataError(f"{name} must be true or false", context={"got": cell})
    return cell == "true"


def _int_from_cell(name: str, cell: str) -> int | None:
    if cell == "":
        return None
    if not _INT_PATTERN.fullmatch(cell):
        raise InvalidDataError(f"{name} is not a canonical integer", context={"got": cell})
    return int(cell)


def _float_from_cell(name: str, cell: str) -> float | None:
    if cell == "":
        return None
    try:
        value = float(cell)
    except ValueError:
        raise InvalidDataError(f"{name} is not a number", context={"got": cell}) from None
    if not math.isfinite(value) or repr(value) != cell:
        raise InvalidDataError(f"{name} is not a canonical finite number", context={"got": cell})
    return value


@functools.lru_cache(maxsize=1024)
def _inventory_from_cell(cell: str) -> tuple[tuple[str, int], ...] | None:
    """Parse one inventory cell (memoised: most rows repeat the same flat inventory)."""
    if cell == "":
        return None
    doc = _load_json(cell.encode("utf-8"), "residual_inventory")
    if not isinstance(doc, dict) or canonical_json(doc) != cell:
        raise InvalidDataError("residual_inventory is not a canonical JSON object", context={"got": cell})
    return _inventory(doc)


def _csv_lines(rows: Iterable[Sequence[str]]) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()


def _csv_line(fields: Sequence[str]) -> str:
    return _csv_lines([fields])


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------


def _excluded(name: str) -> bool:
    return name in MANIFEST_EXCLUDED_NAMES or bool(_TEMP_PATTERN.fullmatch(name))


def _directory_manifest(root: Path) -> list[tuple[str, int, str]]:
    entries: list[tuple[str, int, str]] = []
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as listing:
            for entry in listing:
                if _excluded(entry.name):
                    continue
                if entry.is_symlink():
                    raise InvalidDataError(
                        "directory artifacts may not contain symlinks", context={"entry": entry.name}
                    )
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    entries.append((path.relative_to(root).as_posix(), entry.stat().st_size, sha256_file(path)))
                else:
                    raise InvalidDataError(
                        "unsupported file type in a directory artifact", context={"entry": entry.name}
                    )
    entries.sort()
    return entries


def _content_identity(path: Path) -> tuple[str, str, int]:
    """``(kind, sha256, size_bytes)`` of a regular file or a directory manifest."""
    if path.is_file():
        return "file", sha256_file(path), path.stat().st_size
    if path.is_dir():
        entries = _directory_manifest(path)
        manifest = {"schema": _DIRECTORY_MANIFEST_SCHEMA, "version": 1, "files": [list(e) for e in entries]}
        return "directory", sha256_json(manifest), sum(e[1] for e in entries)
    raise InvalidDataError("artifact path is not an existing file or directory", context={"path": path.name})


def content_sha256(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a file's bytes, or of a directory's canonical file manifest.

    The manifest is ``{"schema": "financeai.directory_manifest", "version": 1, "files":
    [[relative_posix_path, size, sha256], ...]}`` sorted by path and hashed as canonical
    JSON, so it is independent of creation order and file-system listing order. Empty
    directories are not represented; ``__pycache__``, ``.DS_Store`` and this module's
    temporary files are excluded; symlinks inside the directory are refused.
    """
    return _content_identity(Path(path))[1]


# ---------------------------------------------------------------------------
# Atomic file publication and locking
# ---------------------------------------------------------------------------


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes, *, create_only: bool) -> None:
    """Publish ``data`` at ``path`` via a fsynced temporary sibling and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if create_only and path.exists():
        raise ImmutableRecordError("record already exists and is never overwritten", context={"record": path.name})
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _lock_exclusive(handle: BinaryIO) -> None:
    if sys.platform == "win32":
        raise UnsupportedCapabilityError("campaign records need POSIX advisory file locks")
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


@dataclass(frozen=True, slots=True)
class _Parsed:
    """Parse cache of an append-only file: its exact bytes, records and chain tail."""

    data: bytes
    records: tuple[Any, ...]
    tail: str


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


class Campaign:
    """Writer and reader of one campaign's evidence under ``<root>/runs/<campaign_id>``.

    ``clock`` returns the current timezone-aware time (injectable for reproducible records).
    Instances are thread-safe; processes serialise on the campaign's lock file.
    """

    __slots__ = ("_campaign_id", "_clock", "_depth", "_dir", "_handle", "_mutex", "_parsed", "_root")

    def __init__(
        self, root: str | os.PathLike[str], campaign_id: str, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        self._campaign_id = _identifier("campaign_id", campaign_id)
        self._root = Path(root).resolve()
        self._dir = self._root / "runs" / self._campaign_id
        self._clock: Callable[[], datetime] = clock if clock is not None else _utc_now
        self._mutex = threading.RLock()
        self._depth = 0
        self._handle: BinaryIO | None = None
        self._parsed: dict[Path, _Parsed] = {}

    def __repr__(self) -> str:
        return f"Campaign(campaign_id={self._campaign_id!r})"

    # -- locations ------------------------------------------------------------

    @property
    def campaign_id(self) -> str:
        return self._campaign_id

    @property
    def root(self) -> Path:
        return self._root

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def registry_path(self) -> Path:
        return self._dir / "registry" / LEDGER_FILES["artifact-registry"]

    @property
    def exposure_ledger_path(self) -> Path:
        return self._dir / "registry" / LEDGER_FILES["data-exposure"]

    @property
    def trial_ledger_path(self) -> Path:
        return self._dir / "registry" / LEDGER_FILES["trial-ledger"]

    @property
    def freeze_path(self) -> Path:
        return self._dir / "freeze" / "freeze.json"

    def stage_record_path(self, stage_id: str, attempt_id: str) -> Path:
        stage = _identifier("stage_id", stage_id)
        attempt = _identifier("attempt_id", attempt_id)
        return self._dir / "stages" / stage / attempt / "stage-record.json"

    def _relative(self, path: Path) -> str:
        return path.relative_to(self._root).as_posix()

    # -- infrastructure ---------------------------------------------------------

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._mutex:
            if self._depth == 0:
                self._dir.mkdir(parents=True, exist_ok=True)
                handle = (self._dir / _LOCK_NAME).open("a+b")
                try:
                    _lock_exclusive(handle)
                except BaseException:
                    handle.close()
                    raise
                self._handle = handle
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0 and self._handle is not None:
                    held, self._handle = self._handle, None
                    held.close()  # closing the descriptor releases the flock

    def _now(self) -> str:
        stamp = self._clock()
        if not isinstance(stamp, datetime) or stamp.tzinfo is None:
            raise InvalidConfigError("campaign clock must return timezone-aware datetimes")
        return stamp.astimezone(UTC).isoformat(timespec="microseconds")

    def _cached_prefix(self, path: Path, data: bytes) -> _Parsed | None:
        cached = self._parsed.get(path)
        if cached is not None and data.startswith(cached.data):
            return cached
        return None

    def _csv_state[R](
        self, path: Path, header: tuple[str, ...], parse_row: Callable[[list[str]], R]
    ) -> tuple[bytes, tuple[R, ...]]:
        if not path.exists():
            return b"", ()
        data = path.read_bytes()
        if not data.endswith(b"\n"):
            raise InvalidDataError("ledger ends with a torn line", context={"ledger": path.name})
        cached = self._cached_prefix(path, data)
        records: list[R]
        if cached is not None:
            start, records = len(cached.data), list(cached.records)
        else:
            header_line = _csv_line(header).encode("utf-8")
            if not data.startswith(header_line):
                raise IncompatibleSchemaError(
                    "ledger header differs from the record schema", context={"ledger": path.name}
                )
            start, records = len(header_line), []
        if start < len(data):
            try:
                text = data[start:].decode("utf-8")
                for row in csv.reader(io.StringIO(text, newline=""), strict=True):
                    if len(row) != len(header):
                        raise InvalidDataError(
                            "ledger row has the wrong number of fields",
                            context={"ledger": path.name, "fields": len(row)},
                        )
                    if row[0] != self._campaign_id:  # both ledgers lead with campaign_id
                        raise InvalidDataError(
                            "ledger row belongs to another campaign", context={"ledger": path.name, "got": row[0]}
                        )
                    records.append(parse_row(row))
            except (UnicodeDecodeError, csv.Error) as exc:
                raise InvalidDataError(
                    "ledger is not valid UTF-8 CSV", context={"ledger": path.name, "error": str(exc)}
                ) from exc
        state = _Parsed(data, tuple(records), "")
        self._parsed[path] = state
        return data, state.records

    def _append_csv[R](
        self,
        path: Path,
        header: tuple[str, ...],
        data: bytes,
        existing: tuple[R, ...],
        new: Sequence[R],
        to_row: Callable[[R], list[str]],
    ) -> None:
        prefix = data if data else _csv_line(header).encode("utf-8")
        body = _csv_lines(to_row(r) for r in new).encode("utf-8")
        content = prefix + body
        _atomic_write(path, content, create_only=False)
        self._parsed[path] = _Parsed(content, (*existing, *new), "")

    # -- artifact registry ------------------------------------------------------

    def _registry_state(self) -> tuple[bytes, tuple[ArtifactRecord, ...], str]:
        path = self.registry_path
        if not path.exists():
            return b"", (), GENESIS_SHA256
        data = path.read_bytes()
        if not data.endswith(b"\n"):
            raise InvalidDataError("artifact registry ends with a torn line")
        cached = self._cached_prefix(path, data)
        if cached is not None:
            start, records, tail = len(cached.data), list(cached.records), cached.tail
        else:
            start, records, tail = 0, [], GENESIS_SHA256
        latest: dict[str, int] = {}
        for record in records:
            latest[record.artifact_id] = record.version
        if start < len(data):
            for raw in data[start:-1].split(b"\n"):
                record, tail = self._parse_registry_line(raw, tail, latest)
                records.append(record)
        state = _Parsed(data, tuple(records), tail)
        self._parsed[path] = state
        return data, state.records, tail

    def _parse_registry_line(self, raw: bytes, previous: str, latest: dict[str, int]) -> tuple[ArtifactRecord, str]:
        doc = _load_json(raw, "artifact registry line")
        if not isinstance(doc, dict) or canonical_json(doc).encode("utf-8") != raw:
            raise InvalidDataError("artifact registry line is not canonical JSON")
        if doc.pop("schema_version", None) != REGISTRY_SCHEMA_VERSION:
            raise IncompatibleSchemaError("unsupported artifact registry schema version")
        if doc.pop("prev_sha256", None) != previous:
            raise InvalidDataError("artifact registry hash chain is broken (history was edited)")
        record = ArtifactRecord.from_json(doc)
        if record.campaign_id != self._campaign_id:
            raise InvalidDataError(
                "artifact registry line belongs to another campaign", context={"campaign_id": record.campaign_id}
            )
        if record.version != latest.get(record.artifact_id, 0) + 1:
            raise InvalidDataError("artifact versions are not contiguous", context={"artifact_id": record.artifact_id})
        latest[record.artifact_id] = record.version
        return record, sha256_bytes(raw)

    def _inside_root(self, path: str | os.PathLike[str]) -> tuple[Path, str]:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self._root / candidate
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(self._root).as_posix()
        except ValueError:
            raise InvalidConfigError(
                "artifacts must live inside the campaign root", context={"path": str(path)}
            ) from None
        if not resolved.exists():
            raise InvalidDataError("artifact path does not exist", context={"path": relative})
        return resolved, relative

    @staticmethod
    def _resolve(spec: object, records: Sequence[ArtifactRecord]) -> ArtifactRecord:
        if isinstance(spec, ArtifactRecord):
            spec = spec.ref
        artifact_id: object
        version: object = None
        digest: object = None
        if isinstance(spec, str):
            artifact_id = spec
        elif isinstance(spec, ArtifactRef):
            artifact_id, version, digest = spec.artifact_id, spec.version, spec.sha256
        elif isinstance(spec, Mapping):
            artifact_id, version, digest = (spec.get("artifact_id"), spec.get("version"), spec.get("sha256"))
        else:
            raise InvalidDataError("unsupported artifact citation", context={"got": spec})
        if version is not None:
            version = _integer("version", version, minimum=1)
        chain = [r for r in records if r.artifact_id == artifact_id]
        if version is not None:
            chain = [r for r in chain if r.version == version]
        elif digest is not None and chain:
            # A version-less {artifact_id, sha256} citation pins content, not "latest": it
            # selects the newest version carrying that hash, so a closure can still cite the
            # selected version after a later attempt registered another one.
            chain = [r for r in chain if r.sha256 == digest]
            if not chain:
                raise InvalidDataError(
                    "cited artifact hash matches no registered version", context={"artifact_id": artifact_id}
                )
        if not chain:
            raise InvalidDataError(
                "cited artifact version is not registered", context={"artifact_id": artifact_id, "version": version}
            )
        record = chain[-1]
        if digest is not None and digest != record.sha256:
            raise InvalidDataError(
                "cited artifact hash differs from the registry",
                context={"artifact_id": artifact_id, "version": record.version},
            )
        return record

    def register_artifact(
        self,
        artifact_id: str,
        path: str | os.PathLike[str],
        *,
        producer_attempt: str,
        evidence_scope: str | Sequence[str],
        dependencies: Sequence[ArtifactSpec] = (),
        producer_stage: str | None = None,
        description: str | None = None,
        version: int | None = None,
    ) -> ArtifactRecord:
        """Register (or idempotently re-confirm) an immutable artifact version.

        ``path`` (absolute, or relative to the root) must lie inside the campaign root; a
        directory is hashed through its file manifest (:func:`content_sha256`).
        Dependencies are pinned to registered versions (an id means its latest version, an
        ``{artifact_id, sha256}`` mapping the latest version with that hash).
        Without ``version`` an identical registration returns the existing record and
        anything else appends the next version; an explicit existing ``version`` must match
        exactly (:class:`ImmutableRecordError` otherwise) and new versions are contiguous.
        """
        target, relative = self._inside_root(path)
        kind, digest, size = _content_identity(target)
        scopes = _scopes(evidence_scope, allow_text=True)
        with self._locked():
            data, records, tail = self._registry_state()
            deps = tuple(self._resolve(spec, records).ref for spec in dependencies)
            chain = [r for r in records if r.artifact_id == artifact_id]
            next_version = chain[-1].version + 1 if chain else 1
            wanted = next_version if version is None else _integer("version", version, minimum=1)
            candidate = ArtifactRecord(
                campaign_id=self._campaign_id,
                artifact_id=artifact_id,
                version=wanted,
                producer_stage=producer_stage,
                producer_attempt=producer_attempt,
                path=relative,
                kind=kind,
                sha256=digest,
                size_bytes=size,
                evidence_scope=scopes,
                depends_on=deps,
                description=description,
                registered_at=self._now(),
            )
            existing = next((r for r in chain if r.version == wanted), None)
            if existing is not None:
                if existing.registration_key() == candidate.registration_key():
                    return existing
                raise ImmutableRecordError(
                    "artifact version is already registered differently",
                    context={"artifact_id": artifact_id, "version": wanted},
                )
            if wanted != next_version:
                raise InvalidDataError(
                    "artifact versions must be contiguous", context={"artifact_id": artifact_id, "next": next_version}
                )
            if version is None and chain and chain[-1].registration_key() == candidate.registration_key():
                return chain[-1]
            line = canonical_json(
                {**candidate.to_json(), "schema_version": REGISTRY_SCHEMA_VERSION, "prev_sha256": tail}
            ).encode("utf-8")
            content = data + line + b"\n"
            _atomic_write(self.registry_path, content, create_only=False)
            self._parsed[self.registry_path] = _Parsed(content, (*records, candidate), sha256_bytes(line))
        return candidate

    def artifacts(self) -> tuple[ArtifactRecord, ...]:
        """Every registered version, in registration order (hash chain verified)."""
        return self._registry_state()[1]

    def artifact(self, artifact_id: str, version: int | None = None) -> ArtifactRecord:
        """The latest (or the given) registered version of ``artifact_id``."""
        spec: dict[str, Any] = {"artifact_id": artifact_id, "version": version}
        return self._resolve(spec, self.artifacts())

    def verify_artifact(self, artifact_id: str, version: int | None = None) -> ArtifactRecord:
        """Recompute the content hash on disk; raise unless it still matches the registry."""
        record = self.artifact(artifact_id, version)
        target = self._root / record.path
        if not target.exists():
            raise InvalidDataError(
                "registered artifact is missing on disk", context={"artifact_id": artifact_id, "path": record.path}
            )
        kind, digest, _ = _content_identity(target)
        if (kind, digest) != (record.kind, record.sha256):
            raise InvalidDataError(
                "artifact content no longer matches its registered hash",
                context={"artifact_id": artifact_id, "version": record.version},
            )
        return record

    # -- data-exposure ledger ---------------------------------------------------

    def _exposure_state(self) -> tuple[bytes, tuple[ExposureRecord, ...]]:
        return self._csv_state(self.exposure_ledger_path, EXPOSURE_LEDGER_HEADER, ExposureRecord.from_row)

    def _check_exposure(self, record: ExposureRecord, history: Sequence[ExposureRecord]) -> None:
        if record.campaign_id != self._campaign_id:
            raise InvalidDataError("exposure row belongs to another campaign")
        exposed = any(
            h.dataset_id == record.dataset_id and h.exposure_kind is not ExposureKind.METADATA_ONLY for h in history
        )
        if not exposed:
            return
        if record.assigned_role is DataRole.FRESH_TEST:
            raise InvalidDataError(
                "dataset was already exposed; it can never be fresh test data",
                context={"dataset_id": record.dataset_id},
            )
        if not record.previously_exposed:
            raise InvalidDataError(
                "exposure history cannot be reset: previously_exposed must be true",
                context={"dataset_id": record.dataset_id},
            )

    def _append_exposures(self, new: Sequence[ExposureRecord]) -> None:
        data, existing = self._exposure_state()
        accepted: list[ExposureRecord] = []
        for record in new:
            self._check_exposure(record, (*existing, *accepted))
            accepted.append(record)
        self._append_csv(
            self.exposure_ledger_path, EXPOSURE_LEDGER_HEADER, data, existing, accepted, ExposureRecord.to_row
        )

    def record_exposure(
        self,
        dataset_id: str,
        *,
        source: str,
        period: str,
        assigned_role: str,
        previously_exposed: bool,
        exposure_kind: str,
        reason: str,
        protocol_id: str | None = None,
        manifest_hash: str | None = None,
        exposed_at: datetime | str | None = None,
    ) -> ExposureRecord:
        """Append one ``data-exposure.csv`` row (validated against the dataset's history).

        ``FRESH_TEST_OPENING`` is refused here: only :meth:`append_opening` records an
        opening, under a READY_TO_OPEN seal and together with its opening record.
        """
        kind = _member(ExposureKind, exposure_kind, "exposure_kind")
        if kind is ExposureKind.FRESH_TEST_OPENING:
            raise InvalidDataError(
                "a fresh-test opening is recorded only by append_opening, under the seal",
                context={"dataset_id": dataset_id},
            )
        with self._locked():
            stamp = self._now() if exposed_at is None else _utc_text("exposed_at", exposed_at)
            record = ExposureRecord(
                campaign_id=self._campaign_id,
                dataset_id=dataset_id,
                source=source,
                period=period,
                assigned_role=_member(DataRole, assigned_role, "assigned_role"),
                previously_exposed=previously_exposed,
                exposure_kind=kind,
                exposed_at=stamp,
                reason=reason,
                protocol_id=protocol_id,
                manifest_hash=manifest_hash,
            )
            self._append_exposures([record])
        return record

    def exposures(self) -> tuple[ExposureRecord, ...]:
        return self._exposure_state()[1]

    # -- trial ledger -----------------------------------------------------------

    def _trial_state(self) -> tuple[bytes, tuple[TrialRecord, ...]]:
        return self._csv_state(self.trial_ledger_path, TRIAL_LEDGER_HEADER, TrialRecord.from_row)

    def make_trial(
        self,
        *,
        protocol_id: str,
        trial_id: str,
        stage_id: str,
        status: str,
        validity: str,
        evidence_scope: str,
        fit_job_id: str | None = None,
        policy_config_id: str | None = None,
        account_id: str | None = None,
        scenario: str | None = None,
        timing_mode: str | None = None,
        dataset_id: str | None = None,
        seed: int | None = None,
        net_pnl: float | None = None,
        fees: float | None = None,
        turnover_units: int | None = None,
        residual_inventory: Mapping[str, int] | None = None,
        artifact_path: str | None = None,
        notes: str | None = None,
    ) -> TrialRecord:
        """Build and validate a trial row for this campaign without writing it."""
        return TrialRecord(
            campaign_id=self._campaign_id,
            protocol_id=protocol_id,
            trial_id=trial_id,
            stage_id=stage_id,
            status=_member(TrialStatus, status, "status"),
            validity=_member(TrialValidity, validity, "validity"),
            evidence_scope=_member(EvidenceScope, evidence_scope, "evidence_scope"),
            fit_job_id=fit_job_id,
            policy_config_id=policy_config_id,
            account_id=account_id,
            scenario=scenario,
            timing_mode=None if timing_mode is None else _member(ResearchTimingMode, timing_mode, "timing_mode"),
            dataset_id=dataset_id,
            seed=seed,
            net_pnl=net_pnl,
            fees=fees,
            turnover_units=turnover_units,
            residual_inventory=_inventory(residual_inventory),
            artifact_path=artifact_path,
            notes=notes,
        )

    def record_trial(self, **fields: Any) -> TrialRecord:
        """Validate (see :meth:`make_trial`) and append one ``trial-ledger.csv`` row."""
        record = self.make_trial(**fields)
        self.record_trials([record])
        return record

    def record_trials(self, records: Iterable[TrialRecord]) -> tuple[TrialRecord, ...]:
        """Append a batch atomically: every row is accepted or none is.

        ``trial_id`` is unique per campaign (a retry is a new row whose notes cite
        ``RETRY_OF=<trial_id>``). A citation of a recorded row (or of an earlier row of this
        batch) is accepted only when the identity columns (:data:`_RETRY_IDENTITY`) equal the
        cited row's: a retry re-runs *that* trial, so a note alone cannot take another account
        or configuration out of the budget. Rows claiming DEVELOPMENT evidence need a recorded
        non-metadata exposure of their dataset. FRESH_TEST rows need their dataset in the
        seal's recorded opening and no final (COMPLETED or NOT_EVALUATED) evaluation
        disposition. A dataset opened as this campaign's fresh test admits no other rows but
        labelled diagnostics: evidence scope NONE with notes starting with
        :data:`DIAGNOSTIC_LABEL` (e.g. ``"DIAGNOSTIC: rerun with corrected fees"``).
        """
        batch = tuple(records)
        if not batch:
            return ()
        with self._locked():
            data, existing = self._trial_state()
            seen = {t.trial_id: t for t in existing}
            opened: frozenset[str] | None = None
            exposures: tuple[ExposureRecord, ...] = ()
            admitted: frozenset[str] | None = None
            for record in batch:
                if not isinstance(record, TrialRecord) or record.campaign_id != self._campaign_id:
                    raise InvalidDataError("trial rows must be TrialRecords of this campaign")
                if record.trial_id in seen:
                    raise ImmutableRecordError(
                        "trial_id already recorded; record a retry under a new trial_id",
                        context={"trial_id": record.trial_id},
                    )
                cited = seen.get(record.retry_of or "")
                if cited is not None and not record.same_trial_as(cited):
                    raise InvalidDataError(
                        "a RETRY_OF citation must name the same trial (equal identity columns); "
                        "a different account, configuration or dataset is a new trial",
                        context={
                            "trial_id": record.trial_id,
                            "retry_of": cited.trial_id,
                            "differs": [n for n in _RETRY_IDENTITY if getattr(record, n) != getattr(cited, n)],
                        },
                    )
                seen[record.trial_id] = record
                if record.dataset_id is None:
                    continue  # SYNTHETIC/NONE rows only: real-data scopes must name a dataset
                if opened is None:
                    exposures = self._exposure_state()[1]
                    opened = frozenset(
                        e.dataset_id for e in exposures if e.exposure_kind is ExposureKind.FRESH_TEST_OPENING
                    )
                if record.evidence_scope is EvidenceScope.FRESH_TEST and admitted is None:
                    admitted = self._admitted_test_datasets()
                self._check_trial_exposure(record, exposures, opened, admitted or frozenset())
            self._append_csv(self.trial_ledger_path, TRIAL_LEDGER_HEADER, data, existing, batch, TrialRecord.to_row)
        return batch

    def _admitted_test_datasets(self) -> frozenset[str]:
        """Datasets that may carry FRESH_TEST trials: those of the sealed opening, until the
        final disposition (empty without a READY_TO_OPEN seal and its opening record)."""
        if not self.freeze_path.is_file():
            return frozenset()
        current = self._test_state(None)
        if current.seal.document["admission_decision"] != AdmissionDecision.READY_TO_OPEN or current.final:
            return frozenset()
        return frozenset(d for datasets in current.opened.values() for d in datasets)

    @staticmethod
    def _check_trial_exposure(
        record: TrialRecord, exposures: Sequence[ExposureRecord], opened: frozenset[str], admitted: frozenset[str]
    ) -> None:
        """``opened``: datasets with a FRESH_TEST_OPENING row; ``admitted``: datasets of the
        sealed opening while the evaluation is not final (only consulted for FRESH_TEST rows)."""
        context = {"trial_id": record.trial_id, "dataset_id": record.dataset_id}
        scope = record.evidence_scope
        diagnostic = record.notes is not None and bool(_DIAGNOSTIC_PATTERN.match(record.notes))
        if scope is EvidenceScope.FRESH_TEST:
            if diagnostic:
                raise InvalidDataError(
                    "a labelled diagnostic never claims fresh-test evidence (record it with evidence_scope NONE)",
                    context=context,
                )
            # Both records are written by append_opening: a ledger row alone (or an opening of
            # another seal) is not an opening, and nothing follows a final disposition.
            if record.dataset_id not in opened or record.dataset_id not in admitted:
                raise InvalidDataError(
                    "fresh-test trials need their dataset in the sealed, "
                    "recorded opening and no final evaluation disposition",
                    context=context,
                )
        elif record.dataset_id in opened:
            if scope is not EvidenceScope.NONE or not diagnostic:
                raise InvalidDataError(
                    "the dataset was opened as this campaign's fresh test: it admits no tuning, "
                    "only its evaluation and labelled diagnostics (evidence_scope NONE, notes "
                    f"starting with {DIAGNOSTIC_LABEL})",
                    context=context,
                )
        elif scope is EvidenceScope.DEVELOPMENT and not any(
            e.dataset_id == record.dataset_id and e.exposure_kind is not ExposureKind.METADATA_ONLY for e in exposures
        ):
            raise InvalidDataError("development trials need a recorded exposure of their dataset", context=context)

    def trials(self) -> tuple[TrialRecord, ...]:
        return self._trial_state()[1]

    def trial_counts(self) -> TrialCounts:
        """Counted objects for budget checks (failed and invalid rows included).

        A row whose notes cite ``RETRY_OF=<trial_id>`` of an *earlier* row with the same
        identity columns is an idempotent retry of that trial (retries are "not new
        trials"): it is reported in ``retries`` and adds no fit job, configuration or account
        replay. Citations are followed to the chain's root, so a retry of a retry is still the
        root trial, and the root's account counts as valid when any row of the chain is
        COMPLETED and VALID (the motivating case: the original FAILED after a worker restart,
        the retry succeeded). A citation of an unrecorded id, or of a row with different
        identity columns (a ledger written before :meth:`record_trials` checked them), is an
        ordinary new trial, which is the conservative reading for a budget.
        """
        rows = self.trials()
        seen: dict[str, TrialRecord] = {}
        root_of: dict[str, str] = {}
        counted: list[TrialRecord] = []
        retries = 0
        for t in rows:
            cited = seen.get(t.retry_of or "")
            if cited is not None and t.same_trial_as(cited):
                retries += 1
                root_of[t.trial_id] = root_of[cited.trial_id]
            else:
                counted.append(t)
                root_of[t.trial_id] = t.trial_id
            seen[t.trial_id] = t
        valid_roots = {
            root_of[t.trial_id]
            for t in rows
            if t.counts_account and t.status is TrialStatus.COMPLETED and t.validity is TrialValidity.VALID
        }
        return TrialCounts(
            rows=len(rows),
            fit_jobs=len({t.fit_job_id for t in counted if t.fit_job_id is not None}),
            policy_configs=len({t.policy_config_id for t in counted if t.policy_config_id is not None}),
            account_replays=sum(1 for t in counted if t.counts_account),
            valid_accounts=len(valid_roots),
            retries=retries,
        )

    # -- ledger audit -----------------------------------------------------------

    def _ledger_path(self, name: str) -> Path:
        if name not in LEDGER_FILES:
            raise InvalidConfigError("unknown ledger", context={"got": name, "allowed": sorted(LEDGER_FILES)})
        return self._dir / "registry" / LEDGER_FILES[name]

    def ledger_snapshot(self, name: str) -> LedgerSnapshot:
        """Prefix identity of ``artifact-registry``, ``data-exposure`` or ``trial-ledger``."""
        path = self._ledger_path(name)
        data = path.read_bytes() if path.exists() else b""
        return LedgerSnapshot(name=name, size=len(data), sha256=sha256_bytes(data))

    def verify_append_only(self, snapshot: LedgerSnapshot) -> None:
        """Raise unless the ledger still starts with exactly the snapshotted bytes."""
        path = self._ledger_path(snapshot.name)
        data = path.read_bytes() if path.exists() else b""
        if len(data) < snapshot.size or sha256_bytes(data[: snapshot.size]) != snapshot.sha256:
            raise InvalidDataError(
                "ledger history changed since the snapshot", context={"ledger": snapshot.name, "bytes": snapshot.size}
            )

    # -- stage records ----------------------------------------------------------

    def _artifact_entries(
        self, value: object, records: Sequence[ArtifactRecord], name: str, producer: tuple[str, str] | None
    ) -> list[dict[str, Any]]:
        if not isinstance(value, list | tuple):
            raise InvalidDataError(f"{name} must be a list", context={"got": value})
        entries: list[dict[str, Any]] = []
        for spec in value:
            record = self._resolve(spec, records)
            if producer is not None and (
                record.producer_attempt != producer[1] or record.producer_stage not in (None, producer[0])
            ):
                raise InvalidDataError(
                    "produced artifacts must be registered by this attempt",
                    context={"artifact_id": record.artifact_id, "producer_attempt": record.producer_attempt},
                )
            entries.append({**record.ref.to_json(), "producer_attempt": record.producer_attempt})
        return entries

    def _validated_stage(self, doc: dict[str, Any], stage: str, attempt: str) -> dict[str, Any]:
        execution = _member(ExecutionStatus, doc["execution_status"], "execution_status")
        engineering = _member(EngineeringStatus, doc["engineering_status"], "engineering_status")
        scopes = _scopes(doc["evidence_scope"], allow_text=False)
        outcome = _member(EconomicOutcome, doc["economic_outcome"], "economic_outcome")
        profile = _member(DeliveryProfile, doc["delivery_profile"], "delivery_profile")
        reason = _note("defer_or_block_reason", doc["defer_or_block_reason"])
        reentry = _note("reentry_criterion", doc["reentry_criterion"])
        if execution is ExecutionStatus.DEFERRED and (reason is None or reentry is None):
            raise InvalidDataError("a DEFERRED stage needs a reason and a re-entry criterion")
        if execution is ExecutionStatus.BLOCKED and reason is None:
            raise InvalidDataError("a BLOCKED stage needs its blocking reason")
        if execution is ExecutionStatus.COMPLETE and engineering is EngineeringStatus.FAIL:
            raise InvalidDataError("an unresolved engineering failure cannot be COMPLETE (record BLOCKED or DEFERRED)")
        if (
            outcome in (EconomicOutcome.TEST_PASSED, EconomicOutcome.TEST_FAILED)
            and EvidenceScope.FRESH_TEST not in scopes
        ):
            raise InvalidDataError("a test outcome needs FRESH_TEST evidence")
        if outcome is EconomicOutcome.DEVELOPMENT_POSITIVE and EvidenceScope.DEVELOPMENT not in scopes:
            raise InvalidDataError("a development outcome needs DEVELOPMENT evidence")
        times: dict[str, str | None] = {}
        for key in ("started_at", "finished_at"):
            times[key] = None if doc[key] is None else _utc_text(key, doc[key])
        if (
            times["started_at"] is not None
            and times["finished_at"] is not None
            and times["finished_at"] < times["started_at"]
        ):
            raise InvalidDataError("finished_at precedes started_at")
        for key in ("commands", "changed_files", "data_exposures", "failed_checks"):
            if not isinstance(doc[key], list | tuple):
                raise InvalidDataError(f"{key} must be a list", context={"got": doc[key]})
        if doc["next_stage"] is not None:
            _text("next_stage", doc["next_stage"])
        consumed, produced = doc["consumed_artifacts"], doc["produced_artifacts"]
        records: tuple[ArtifactRecord, ...] = ()
        if consumed or produced:
            records = self._registry_state()[1]
        doc.update(
            execution_status=execution.value,
            engineering_status=engineering.value,
            evidence_scope=[s.value for s in scopes],
            economic_outcome=outcome.value,
            delivery_profile=profile.value,
            **times,
            consumed_artifacts=self._artifact_entries(consumed, records, "consumed_artifacts", None),
            produced_artifacts=self._artifact_entries(produced, records, "produced_artifacts", (stage, attempt)),
        )
        checked: dict[str, Any] = _strict_json(doc, "stage record")
        return checked

    def write_stage_record(self, stage_id: str, attempt_id: str, **fields: Any) -> RecordFile:
        """Write ``stages/<stage>/<attempt>/stage-record.json`` (schema 5) exactly once.

        Unspecified fields keep the template defaults. The four status axes are validated
        independently and for consistency (e.g. COMPLETE never carries an engineering FAIL;
        a test outcome needs FRESH_TEST evidence); cited artifacts must match the registry
        and produced ones must have been registered by this attempt.
        """
        stage = _identifier("stage_id", stage_id)
        attempt = _identifier("attempt_id", attempt_id)
        unknown = sorted(set(fields) - set(STAGE_RECORD_FIELDS))
        if unknown:
            raise InvalidDataError("unknown stage-record fields", context={"fields": unknown})
        managed = sorted(set(fields) & {"schema_version", "campaign_id", "stage_id", "attempt_id"})
        if managed:
            raise InvalidDataError("these stage-record fields are set by the writer", context={"fields": managed})
        doc = stage_record_template()
        doc.update(fields)
        doc.update(campaign_id=self._campaign_id, stage_id=stage, attempt_id=attempt)
        path = self.stage_record_path(stage, attempt)
        with self._locked():
            payload = _pretty_json(self._validated_stage(doc, stage, attempt))
            _atomic_write(path, payload, create_only=True)
        return RecordFile(path, sha256_bytes(payload))

    def read_stage_record(self, stage_id: str, attempt_id: str) -> dict[str, Any]:
        path = self.stage_record_path(stage_id, attempt_id)
        if not path.is_file():
            raise InvalidDataError("no such stage record", context={"stage_id": stage_id, "attempt_id": attempt_id})
        doc: dict[str, Any] = _load_json(path.read_bytes(), "stage record")
        return doc

    def stage_records(self) -> tuple[Path, ...]:
        return tuple(sorted(self._dir.glob("stages/*/*/stage-record.json")))

    # -- freeze seal, opening and evaluation records ----------------------------

    def _seal_document(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise InvalidDataError("freeze payload must be a mapping")
        forbidden = sorted(set(payload) & FORBIDDEN_SEAL_KEYS)
        if forbidden:
            raise InvalidDataError("future openings/results do not belong in the seal", context={"fields": forbidden})
        unknown = sorted(set(payload) - set(FREEZE_FIELDS))
        if unknown:
            raise InvalidDataError("unknown freeze fields", context={"fields": unknown})
        doc = freeze_template()
        doc.update(payload)
        if type(doc["schema_version"]) is not int or doc["schema_version"] != 5:
            raise InvalidDataError("freeze schema_version must be 5")
        if doc["status"] not in ("DRAFT", "SEALED") or doc["phase"] != "PRE_EVALUATION":
            raise InvalidDataError(
                "only a pre-evaluation draft can be sealed", context={"status": doc["status"], "phase": doc["phase"]}
            )
        if doc["campaign_id"] not in (None, self._campaign_id):
            raise InvalidDataError("freeze payload names another campaign")
        for key in FREEZE_HASH_FIELDS:
            if key in REQUIRED_SEAL_HASHES or doc[key] is not None:
                _sha256(key, doc[key])
        _member(AdmissionDecision, doc["admission_decision"], "admission_decision")
        _member(EvaluationMode, doc["evaluation_mode"], "evaluation_mode")
        _member(DeliveryProfile, doc["delivery_profile"], "delivery_profile")
        for key in ("selected_artifact_closure", "scenario_order"):
            if not isinstance(doc[key], list | tuple):
                raise InvalidDataError(f"{key} must be a list")
        scenarios = [_text("scenario_order", name) for name in doc["scenario_order"]]
        if len(set(scenarios)) != len(scenarios):
            raise InvalidDataError("scenario_order lists a scenario twice", context={"got": scenarios})
        if not isinstance(doc["optional_branch_dispositions"], Mapping):
            raise InvalidDataError("optional_branch_dispositions must be an object")
        doc["selected_artifact_closure"] = self._pinned_closure(doc["selected_artifact_closure"])
        doc["status"] = "SEALED"
        doc["campaign_id"] = self._campaign_id
        if doc["sealed_at"] is None:
            doc["sealed_at"] = self._now()
        else:
            # A caller-supplied time may back-date the seal (the freeze happened earlier) but never
            # post-date it: openings and dispositions are ordered against this instant.
            doc["sealed_at"] = _utc_text("sealed_at", doc["sealed_at"])
            if _instant(doc["sealed_at"]) > _instant(self._now()):
                raise InvalidDataError(
                    "sealed_at is later than the campaign clock", context={"sealed_at": doc["sealed_at"]}
                )
        sealed: dict[str, Any] = _strict_json(doc, "freeze record")
        return sealed

    def _pinned_closure(self, closure: Sequence[object]) -> list[dict[str, Any]]:
        """Closure entries must pin ``{artifact_id, sha256}`` and match the registry."""
        entries: list[dict[str, Any]] = []
        records: tuple[ArtifactRecord, ...] | None = None
        for item in closure:
            entry = item.ref.to_json() if isinstance(item, ArtifactRecord) else item
            entry = entry.to_json() if isinstance(entry, ArtifactRef) else entry
            if not isinstance(entry, Mapping) or "artifact_id" not in entry or "sha256" not in entry:
                raise InvalidDataError("closure entries must pin {artifact_id, sha256}", context={"got": item})
            if records is None:
                records = self._registry_state()[1]
            self._resolve(entry, records)
            entries.append(dict(entry))
        return entries

    def seal_freeze(self, payload: Mapping[str, Any]) -> FreezeSeal:
        """Write ``freeze/freeze.json``, the campaign's single pre-evaluation seal.

        The payload fills template fields (choices, hashes, planned suite); status, phase,
        campaign id and ``sealed_at`` (default: now) are set here. Future results are
        refused, as are unknown fields, malformed digests and a missing admission decision.
        The file holds exactly the canonical JSON bytes; their SHA-256 is the seal.
        """
        with self._locked():
            doc = self._seal_document(payload)
            data = canonical_json(doc).encode("utf-8")
            _atomic_write(self.freeze_path, data, create_only=True)
        return FreezeSeal(self.freeze_path, sha256_bytes(data), MappingProxyType(doc))

    def _load_seal(self, expected: str | None) -> FreezeSeal:
        if expected is not None:
            _sha256("freeze_hash", expected)
        if not self.freeze_path.is_file():
            raise InvalidDataError("the campaign has no freeze seal")
        data = self.freeze_path.read_bytes()
        digest = sha256_bytes(data)
        if expected is not None and digest != expected:
            raise InvalidDataError(
                "freeze hash does not match the sealed record", context={"expected": expected, "actual": digest}
            )
        doc = _load_json(data, "freeze record")
        if not isinstance(doc, dict) or canonical_json(doc).encode("utf-8") != data:
            raise InvalidDataError("the seal is no longer canonical JSON (it was edited)")
        if doc.get("status") != "SEALED" or doc.get("campaign_id") != self._campaign_id:
            raise InvalidDataError("freeze.json is not this campaign's seal")
        return FreezeSeal(self.freeze_path, digest, MappingProxyType(doc))

    def read_freeze(self) -> FreezeSeal:
        """The seal and its hash (refuses a seal that is no longer canonical)."""
        return self._load_seal(None)

    def _opening_dir(self) -> Path:
        return self._dir / "freeze" / "openings"

    def _evaluation_dir(self) -> Path:
        return self._dir / "freeze" / "evaluations"

    def openings(self) -> tuple[Path, ...]:
        return tuple(sorted(p for p in self._opening_dir().glob("*.json") if not p.name.startswith(".")))

    def evaluations(self) -> tuple[Path, ...]:
        return tuple(sorted(self._evaluation_dir().glob("evaluation-*.json")))

    def _opened(self, seal: FreezeSeal) -> tuple[dict[str, frozenset[str]], dict[str, str]]:
        """Opening record id -> opened dataset ids, and id -> ``opened_at``; every opening
        record must cite ``seal``."""
        opened: dict[str, frozenset[str]] = {}
        opened_at: dict[str, str] = {}
        for path in self.openings():
            doc = _load_json(path.read_bytes(), "opening record")
            if not isinstance(doc, dict) or doc.get("freeze_record_sha256") != seal.sha256:
                raise InvalidDataError("opening record does not cite the current seal", context={"record": path.name})
            datasets = doc.get("datasets")
            if (
                doc.get("campaign_id") != self._campaign_id
                or not isinstance(datasets, list)
                or not all(isinstance(d, dict) and isinstance(d.get("dataset_id"), str) for d in datasets)
            ):
                raise InvalidDataError("opening record is malformed", context={"record": path.name})
            opening_id = _identifier("opening_record_id", doc.get("opening_record_id"), InvalidDataError)
            opened[opening_id] = frozenset(d["dataset_id"] for d in datasets)
            opened_at[opening_id] = _utc_text("opened_at", doc.get("opened_at"))
        return opened, opened_at

    def _dispositions(self, seal: FreezeSeal) -> tuple[EvaluationStatus, ...]:
        """Statuses of the evaluation records in order; each must cite ``seal``."""
        statuses: list[EvaluationStatus] = []
        for path in self.evaluations():
            doc = _load_json(path.read_bytes(), "evaluation record")
            if not isinstance(doc, dict) or doc.get("freeze_record_sha256") != seal.sha256:
                raise InvalidDataError(
                    "evaluation record does not cite the current seal", context={"record": path.name}
                )
            statuses.append(_member(EvaluationStatus, doc.get("status"), "status"))
        return tuple(statuses)

    def _test_state(self, freeze_hash: str | None) -> _TestState:
        seal = self._load_seal(freeze_hash)
        opened, opened_at = self._opened(seal)
        return _TestState(seal, MappingProxyType(opened), MappingProxyType(opened_at), self._dispositions(seal))

    def append_opening(
        self,
        freeze_hash: str,
        *,
        opening_record_id: str,
        datasets: Sequence[OpenedDataset],
        authorization: str,
        opened_at: datetime | str | None = None,
    ) -> RecordFile:
        """Record the seal's single test opening and its FRESH_TEST_OPENING exposures.

        Refused unless ``freeze_hash`` is the current seal's hash and the seal admits an
        opening (``READY_TO_OPEN``); a second opening, an opening after any evaluation
        disposition (a no-test audit is final), data with any earlier exposure and an
        ``opened_at`` before the seal's ``sealed_at`` are refused. Exposure rows are written
        first: an exposure recorded for an opening that then failed to publish is
        conservative, the reverse would not be.
        """
        opening_id = _identifier("opening_record_id", opening_record_id)
        _text("authorization", authorization)
        items = tuple(datasets)
        if not items or not all(isinstance(d, OpenedDataset) for d in items):
            raise InvalidDataError("an opening needs one or more OpenedDataset identities")
        if len({d.dataset_id for d in items}) != len(items):
            raise InvalidDataError("an opening lists each dataset once")
        with self._locked():
            current = self._test_state(freeze_hash)
            seal = current.seal
            if seal.document["admission_decision"] != AdmissionDecision.READY_TO_OPEN:
                raise InvalidDataError(
                    "the seal admits no test opening", context={"admission": seal.document["admission_decision"]}
                )
            if current.opened:
                raise ImmutableRecordError("the seal already has its one declared test opening")
            if current.dispositions:
                raise InvalidDataError(
                    "an evaluation disposition is already recorded; the test can no longer be opened",
                    context={"evaluations": len(current.dispositions)},
                )
            stamp = self._now() if opened_at is None else _utc_text("opened_at", opened_at)
            _not_before("opened_at", stamp, current.sealed_at, "sealed_at")
            rows = [
                ExposureRecord(
                    campaign_id=self._campaign_id,
                    dataset_id=d.dataset_id,
                    source=d.source,
                    period=d.period,
                    assigned_role=DataRole.FRESH_TEST,
                    previously_exposed=False,
                    exposure_kind=ExposureKind.FRESH_TEST_OPENING,
                    exposed_at=stamp,
                    reason=f"test opening {opening_id} under seal {seal.sha256[:16]}",
                    manifest_hash=d.sha256,
                )
                for d in items
            ]
            doc = {
                "schema_version": OPENING_SCHEMA_VERSION,
                "campaign_id": self._campaign_id,
                "opening_record_id": opening_id,
                "freeze_record_sha256": seal.sha256,
                "datasets": [d.to_json() for d in items],
                "opened_at": stamp,
                "authorization": authorization,
            }
            payload = _pretty_json(doc)
            path = self._opening_dir() / f"{opening_id}.json"
            self._append_exposures(rows)
            _atomic_write(path, payload, create_only=True)
        return RecordFile(path, sha256_bytes(payload))

    def append_evaluation(
        self,
        freeze_hash: str,
        *,
        status: str,
        economic_decision: str,
        reason: str,
        opening_record_id: str | None = None,
        result_artifacts: Sequence[ArtifactSpec] = (),
        created_at: datetime | str | None = None,
    ) -> RecordFile:
        """Append an ``evaluation-record.json`` (schema 1) that cites the unchanged seal.

        ``NOT_EVALUATED`` never references an opening or results and is refused once a
        test was opened; ``COMPLETED``/``FAILED``/``INTERRUPTED`` must cite the recorded
        opening and registered result artifacts (partial ones for failures). A test outcome
        requires an opened test, and TEST_PASSED a completed one. The suite is evaluated
        once: after a ``COMPLETED`` or ``NOT_EVALUATED`` disposition nothing more is
        appended (a correction is rerun as a labelled diagnostic trial, never re-scored), while
        a ``FAILED`` or ``INTERRUPTED`` one may be followed by its continuation. ``created_at``
        is at or after the seal and the cited opening.
        """
        state = _member(EvaluationStatus, status, "status")
        decision = _member(EconomicOutcome, economic_decision, "economic_decision")
        _note("reason", reason)
        if reason is None:
            raise InvalidDataError("an evaluation disposition needs a reason")
        if opening_record_id is not None:
            _identifier("opening_record_id", opening_record_id, InvalidDataError)
        results = list(result_artifacts)
        with self._locked():
            current = self._test_state(freeze_hash)
            seal, opened = current.seal, current.opened
            refs: list[dict[str, Any]] = []
            if state is EvaluationStatus.NOT_EVALUATED:
                if opening_record_id is not None or results:
                    raise InvalidDataError("a no-test disposition cannot cite an opening or results")
                if opened:
                    raise InvalidDataError(
                        "the test was opened: record COMPLETED, FAILED or INTERRUPTED instead of NOT_EVALUATED"
                    )
                if decision in (EconomicOutcome.TEST_PASSED, EconomicOutcome.TEST_FAILED):
                    raise InvalidDataError("a test outcome requires an opened test")
            else:
                if opening_record_id is None or opening_record_id not in opened:
                    raise InvalidDataError(
                        "an opened evaluation must cite the recorded opening",
                        context={"opening_record_id": opening_record_id},
                    )
                if not results:
                    raise InvalidDataError("an opened evaluation needs result artifacts")
                if decision is EconomicOutcome.TEST_PASSED and state is not EvaluationStatus.COMPLETED:
                    raise InvalidDataError("TEST_PASSED requires a COMPLETED evaluation")
                registry = self._registry_state()[1]
                refs = [self._resolve(spec, registry).ref.to_json() for spec in results]
            if current.final:
                raise InvalidDataError(
                    "the evaluation already has its final disposition "
                    "(COMPLETED or NOT_EVALUATED); a correction is a labelled "
                    "diagnostic, not a new disposition",
                    context={"evaluations": len(current.dispositions)},
                )
            stamp = self._now() if created_at is None else _utc_text("created_at", created_at)
            _not_before("created_at", stamp, current.sealed_at, "sealed_at")
            if opening_record_id is not None:
                _not_before("created_at", stamp, current.opened_at[opening_record_id], "opened_at")
            doc = evaluation_record_template()
            doc.update(
                status=state.value,
                freeze_record_sha256=seal.sha256,
                opening_record_id=opening_record_id,
                result_artifacts=refs,
                economic_decision=decision.value,
                reason=reason,
                created_at=stamp,
            )
            payload = _pretty_json(doc)
            path = self._evaluation_dir() / f"evaluation-{len(self.evaluations()) + 1:04d}.json"
            _atomic_write(path, payload, create_only=True)
        return RecordFile(path, sha256_bytes(payload))

    # -- registry document --------------------------------------------------------

    def write_registry_document(
        self, *, protected_paths: Sequence[str] = (), delivery_profile: str = "research_run"
    ) -> RecordFile:
        """Render ``registry/artifact-registry.json`` (template schema 4) from the JSONL.

        A derived, deterministic view (no timestamps): rewriting it is safe. Paths are
        relative to the campaign root, which is recorded as ``"."``.
        """
        profile = _member(DeliveryProfile, delivery_profile, "delivery_profile")
        protected = [_text("protected_paths", p) for p in protected_paths]
        with self._locked():
            records = self._registry_state()[1]
            doc = {
                "schema_version": 4,
                "campaign_id": self._campaign_id,
                "record_format": "1",
                "implementation_root": ".",
                "protected_paths": protected,
                "artifacts": [r.to_json() for r in records],
                "artifact_record_fields": list(ARTIFACT_RECORD_FIELDS),
                "stage_records": [self._relative(p) for p in self.stage_records()],
                "exposure_ledger": self._relative(self.exposure_ledger_path),
                "trial_ledger": self._relative(self.trial_ledger_path),
                "delivery_profile": profile.value,
            }
            payload = _pretty_json(doc)
            path = self._dir / "registry" / "artifact-registry.json"
            _atomic_write(path, payload, create_only=False)
        return RecordFile(path, sha256_bytes(payload))


__all__ = [
    "ARTIFACT_RECORD_FIELDS",
    "DIAGNOSTIC_LABEL",
    "EVALUATION_RECORD_FIELDS",
    "EXPOSURE_LEDGER_HEADER",
    "FORBIDDEN_SEAL_KEYS",
    "FREEZE_FIELDS",
    "FREEZE_HASH_FIELDS",
    "GENESIS_SHA256",
    "LEDGER_FILES",
    "MANIFEST_EXCLUDED_NAMES",
    "REQUIRED_SEAL_HASHES",
    "STAGE_RECORD_FIELDS",
    "TRIAL_LEDGER_HEADER",
    "AdmissionDecision",
    "ArtifactRecord",
    "ArtifactRef",
    "ArtifactSpec",
    "Campaign",
    "DataRole",
    "DeliveryProfile",
    "EconomicOutcome",
    "EngineeringStatus",
    "EvaluationMode",
    "EvaluationStatus",
    "EvidenceScope",
    "ExecutionStatus",
    "ExposureKind",
    "ExposureRecord",
    "FreezeSeal",
    "ImmutableRecordError",
    "LedgerSnapshot",
    "OpenedDataset",
    "RecordFile",
    "ResearchTimingMode",
    "TrialCounts",
    "TrialRecord",
    "TrialStatus",
    "TrialValidity",
    "content_sha256",
    "evaluation_record_template",
    "freeze_template",
    "stage_record_template",
]
