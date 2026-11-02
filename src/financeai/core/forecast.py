"""Domain-neutral forecast contracts: targets, capabilities, queries, immutable tickets and outcomes.

The vocabulary follows the architecture contract. A :class:`TargetSpec` identifies *what* is forecast (domain and
schema, unit, horizon, labeler and availability rule). A :class:`ForecastQuery` freezes the issue-time features; an
adapter answers with an immutable :class:`ForecastTicket` that records exactly what was issued (a point mean plus an
optional variance and marginal quantiles, each flagged explicitly) together with the model and transform versions.
An :class:`Outcome` later reports the realised value and the clock at which it became available. Nothing here knows
about markets, money or a particular estimator.

Clocks are int64 nanoseconds; in JSON they travel as canonical decimal strings so values beyond 2**53 stay exact.
Issued arrays are copied into immutable buffers: neither the caller's source array nor the returned array can change
a ticket after issue. A point forecast carries no calibrated-confidence claim.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import (
    FinanceAIError,
    IncompatibleSchemaError,
    InvalidConfigError,
    InvalidDataError,
    UnsupportedCapabilityError,
)
from financeai.core.hashing import sha256_json

INT64_MAX: Final = 2**63 - 1
_NS_TEXT: Final = re.compile(r"0|[1-9][0-9]*")
_VERSION_TEXT: Final = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
_MAX_TEXT: Final = 256

ErrorType = type[FinanceAIError]


class Capability(StrEnum):
    """Forecast capability names fixed by the architecture contract."""

    MEAN = "mean"
    VARIANCE = "variance"
    MARGINAL_QUANTILES = "marginal_quantiles"
    CDF = "cdf"
    MARGINAL_SAMPLES = "marginal_samples"
    JOINT_SAMPLES = "joint_samples"
    ONLINE_UPDATE = "online_update"
    DELAYED_FEEDBACK = "delayed_feedback"


TICKET_CAPABILITIES: Final = frozenset({Capability.MEAN, Capability.VARIANCE, Capability.MARGINAL_QUANTILES})
"""Payloads a ticket can carry. The others describe adapters or need payloads this ticket schema does not have."""


# --------------------------------------------------------------------------- validation helpers


def _text(value: object, name: str, error: ErrorType) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > _MAX_TEXT
        or not value.isprintable()
    ):
        raise error(
            f"{name} must be a non-empty printable string without surrounding whitespace", context={"field": name}
        )
    return value


def _ns(value: object, name: str, error: ErrorType) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or not 0 <= int(value) <= INT64_MAX:
        raise error(f"{name} must be a non-negative int64 nanosecond count", context={"field": name, "got": value})
    return int(value)


def _real(value: object, name: str, error: ErrorType) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | float | np.integer | np.floating):
        raise error(f"{name} must be a real number", context={"field": name})
    try:
        out = float(value)
    except OverflowError:
        raise error(f"{name} must be finite", context={"field": name}) from None
    if not math.isfinite(out):
        raise error(f"{name} must be finite", context={"field": name})
    return out


def _frozen(array: NDArray[Any]) -> NDArray[Any]:
    """Copy into a bytes-backed buffer: unlike a flag flip, it can never be made writeable again."""
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _float_array(value: object, name: str, ndim: int, error: ErrorType) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != ndim:
        raise error(
            f"{name} must be a {ndim}-D real array", context={"field": name, "dtype": str(raw.dtype), "ndim": raw.ndim}
        )
    array = raw.astype(np.float64)
    if not bool(np.isfinite(array).all()):
        raise error(f"{name} must be finite", context={"field": name})
    return _frozen(array)


def _clock_array(value: object, name: str, error: ErrorType) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu" or raw.ndim != 1:
        raise error(f"{name} must be a 1-D integer nanosecond array", context={"field": name, "dtype": str(raw.dtype)})
    if raw.size and (int(raw.min()) < 0 or int(raw.max()) > INT64_MAX):
        raise error(f"{name} must hold non-negative int64 nanoseconds", context={"field": name})
    return _frozen(raw.astype(np.int64))


def _levels(value: object, error: ErrorType) -> tuple[float, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        raise error("quantile levels must be a sequence of numbers")
    levels = tuple(_real(v, "quantile level", error) for v in value)
    if any(not 0.0 < v < 1.0 for v in levels) or any(b <= a for a, b in itertools.pairwise(levels)):
        raise error("quantile levels must be strictly increasing inside (0, 1)", context={"levels": levels})
    return levels


def _names(value: object, error: ErrorType) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        raise error("feature_names must be a sequence of strings")
    names = tuple(_text(v, "feature name", error) for v in value)
    if len(set(names)) != len(names):
        raise error("feature names must be distinct", context={"names": names})
    return names


def require_keys(
    doc: object, keys: Iterable[str], *, where: str, error: ErrorType = InvalidDataError
) -> Mapping[str, Any]:
    """Return ``doc`` if it is a mapping with exactly ``keys`` (strict documents reject unknown fields)."""
    if not isinstance(doc, Mapping):
        raise error(f"{where} must be an object")
    expected = set(keys)
    missing, extra = expected - set(doc), set(doc) - expected
    if missing or extra:
        raise error(f"{where}: keys differ", context={"missing": sorted(missing), "extra": sorted(map(str, extra))})
    return doc


def require_schema(doc: object, *, schema: str, version: str, where: str) -> Mapping[str, Any]:
    """Return ``doc`` if it declares ``schema`` with the same major ``schema_version``; reject unknown majors."""
    if not isinstance(doc, Mapping):
        raise IncompatibleSchemaError(f"{where} must be an object")
    if doc.get("schema") != schema:
        raise IncompatibleSchemaError(f"{where} is not a {schema} document", context={"schema": doc.get("schema")})
    got = doc.get("schema_version")
    if not isinstance(got, str) or _VERSION_TEXT.fullmatch(got) is None or got.split(".")[0] != version.split(".")[0]:
        raise IncompatibleSchemaError(f"unsupported {where} schema version", context={"got": got, "supported": version})
    return doc


def encode_ns(ns: int) -> str:
    """Canonical decimal string of an int64 nanosecond clock (exact beyond 2**53)."""
    return str(_ns(ns, "clock", InvalidDataError))


def _decode_ns(text: object, name: str, error: ErrorType) -> int:
    if not isinstance(text, str) or _NS_TEXT.fullmatch(text) is None:
        raise error(f"{name} must be a canonical decimal string", context={"field": name, "got": repr(text)[:40]})
    return _ns(int(text), name, error)


def decode_ns(text: object) -> int:
    """Inverse of :func:`encode_ns`; rejects signs, exponents, padding, leading zeros and non-strings."""
    return _decode_ns(text, "clock", InvalidDataError)


def require_capability(capabilities: Iterable[Capability], capability: Capability, *, component: str) -> None:
    """Raise :class:`UnsupportedCapabilityError` unless ``capability`` is declared."""
    if capability not in frozenset(capabilities):
        raise UnsupportedCapabilityError(
            f"{component} does not provide {capability.value}",
            context={"component": component, "capability": capability.value},
        )


# --------------------------------------------------------------------------- targets


@dataclass(frozen=True, slots=True)
class AvailabilityRule:
    """Earliest legitimate release of a label: ``feature_cutoff + horizon_ns + min_delay_ns``.

    Outcomes carry their actual ``available_at_ns``; this rule is the lower bound every outcome and training label is
    checked against. It is part of the target identity: a changed reporting delay is a different target.
    """

    name: str
    min_delay_ns: int = 0

    def __post_init__(self) -> None:
        _text(self.name, "availability.name", InvalidConfigError)
        object.__setattr__(
            self, "min_delay_ns", _ns(self.min_delay_ns, "availability.min_delay_ns", InvalidConfigError)
        )

    def to_mapping(self) -> dict[str, Any]:
        return {"name": self.name, "min_delay_ns": encode_ns(self.min_delay_ns)}

    @classmethod
    def from_mapping(cls, doc: object) -> AvailabilityRule:
        d = require_keys(doc, ("name", "min_delay_ns"), where="availability", error=InvalidConfigError)
        return cls(d["name"], _decode_ns(d["min_delay_ns"], "availability.min_delay_ns", InvalidConfigError))


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Generic target identity: domain/schema, unit, horizon, labeler and availability rule.

    The labelled interval ends ``horizon_ns`` after the forecast origin (the ticket's feature cutoff). Domain
    extensions add their own fields elsewhere; a demand target never acquires venue or USD-face fields.
    """

    domain: str
    schema: str
    unit: str
    horizon_ns: int
    labeler: str
    availability: AvailabilityRule

    def __post_init__(self) -> None:
        for name in ("domain", "schema", "unit", "labeler"):
            _text(getattr(self, name), name, InvalidConfigError)
        object.__setattr__(self, "horizon_ns", _ns(self.horizon_ns, "horizon_ns", InvalidConfigError))
        if not isinstance(self.availability, AvailabilityRule):
            raise InvalidConfigError("availability must be an AvailabilityRule")

    @property
    def identity(self) -> str:
        """SHA-256 of the canonical mapping: every field (including the availability rule) changes it."""
        return sha256_json(self.to_mapping())

    def earliest_available_ns(self, feature_cutoff_ns: int) -> int:
        cutoff = _ns(feature_cutoff_ns, "feature_cutoff_ns", InvalidDataError)
        earliest = cutoff + self.horizon_ns + self.availability.min_delay_ns
        if earliest > INT64_MAX:
            raise InvalidDataError("label availability overflows the int64 clock", context={"cutoff_ns": cutoff})
        return earliest

    def to_mapping(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "schema": self.schema,
            "unit": self.unit,
            "horizon_ns": encode_ns(self.horizon_ns),
            "labeler": self.labeler,
            "availability": self.availability.to_mapping(),
        }

    @classmethod
    def from_mapping(cls, doc: object) -> TargetSpec:
        d = require_keys(
            doc,
            ("domain", "schema", "unit", "horizon_ns", "labeler", "availability"),
            where="target",
            error=InvalidConfigError,
        )
        return cls(
            domain=d["domain"],
            schema=d["schema"],
            unit=d["unit"],
            horizon_ns=_decode_ns(d["horizon_ns"], "horizon_ns", InvalidConfigError),
            labeler=d["labeler"],
            availability=AvailabilityRule.from_mapping(d["availability"]),
        )


# --------------------------------------------------------------------------- queries and tickets


def _empty_features() -> NDArray[np.float64]:
    return _frozen(np.zeros(0, dtype=np.float64))


@dataclass(frozen=True, slots=True, eq=False)
class ForecastQuery:
    """Issue-time request: the frozen feature snapshot plus the forecast payload the caller needs.

    ``query_id`` becomes the ticket ID. ``quantile_levels`` requests marginal quantiles; an adapter without
    ``marginal_quantiles`` must raise :class:`UnsupportedCapabilityError` rather than invent them.
    """

    query_id: str
    stream: str
    target: TargetSpec
    issued_at_ns: int
    feature_cutoff_ns: int
    features: NDArray[np.float64]
    feature_names: tuple[str, ...]
    transform_version: str
    quantile_levels: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in ("query_id", "stream", "transform_version"):
            _text(getattr(self, name), name, InvalidDataError)
        if not isinstance(self.target, TargetSpec):
            raise InvalidDataError("target must be a TargetSpec")
        issued = _ns(self.issued_at_ns, "issued_at_ns", InvalidDataError)
        cutoff = _ns(self.feature_cutoff_ns, "feature_cutoff_ns", InvalidDataError)
        if cutoff > issued:
            raise InvalidDataError("feature cutoff is later than the issue time", context={"query_id": self.query_id})
        names = _names(self.feature_names, InvalidDataError)
        features = _float_array(self.features, "features", 1, InvalidDataError)
        if features.shape[0] != len(names):
            raise InvalidDataError(
                "one feature value per feature name is required",
                context={"values": features.shape[0], "names": len(names)},
            )
        object.__setattr__(self, "issued_at_ns", issued)
        object.__setattr__(self, "feature_cutoff_ns", cutoff)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "quantile_levels", _levels(self.quantile_levels, InvalidDataError))

    def _key(self) -> tuple[object, ...]:
        return (
            self.query_id,
            self.stream,
            self.target,
            self.issued_at_ns,
            self.feature_cutoff_ns,
            self.features.tobytes(),
            self.feature_names,
            self.transform_version,
            self.quantile_levels,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ForecastQuery):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash((self.query_id, self.issued_at_ns))


@dataclass(frozen=True, slots=True, eq=False)
class ForecastTicket:
    """The prediction actually issued: immutable, versioned and scored once against its outcome.

    ``capabilities`` flags the payload explicitly: ``mean`` always, ``variance`` iff a variance is present and
    ``marginal_quantiles`` iff quantiles (non-decreasing, on ``quantile_levels``) are present. ``features`` is the
    issue-time feature snapshot, so delayed updates train on what was known at issue, not on a recomputation.
    """

    ticket_id: str
    stream: str
    target: TargetSpec
    issued_at_ns: int
    feature_cutoff_ns: int
    model_id: str
    model_version: str
    transform_version: str
    capabilities: frozenset[Capability]
    mean: float
    variance: float | None = None
    quantile_levels: tuple[float, ...] = ()
    quantiles: NDArray[np.float64] | None = None
    features: NDArray[np.float64] = field(default_factory=_empty_features)

    def __post_init__(self) -> None:
        for name in ("ticket_id", "stream", "model_id", "model_version", "transform_version"):
            _text(getattr(self, name), name, InvalidDataError)
        if not isinstance(self.target, TargetSpec):
            raise InvalidDataError("target must be a TargetSpec")
        issued = _ns(self.issued_at_ns, "issued_at_ns", InvalidDataError)
        cutoff = _ns(self.feature_cutoff_ns, "feature_cutoff_ns", InvalidDataError)
        if cutoff > issued:
            raise InvalidDataError("feature cutoff is later than the issue time", context={"ticket_id": self.ticket_id})
        caps = self.capabilities
        if not isinstance(caps, frozenset | set) or not all(isinstance(c, Capability) for c in caps):
            raise InvalidDataError("capabilities must be a set of Capability members")
        caps = frozenset(caps)
        if Capability.MEAN not in caps:
            raise InvalidDataError("every ticket carries a point mean", context={"ticket_id": self.ticket_id})
        if not caps <= TICKET_CAPABILITIES:
            raise InvalidDataError(
                "ticket flags a capability without a payload in this schema",
                context={"extra": sorted(c.value for c in caps - TICKET_CAPABILITIES)},
            )
        mean = _real(self.mean, "mean", InvalidDataError)
        variance: float | None = None
        if Capability.VARIANCE in caps:
            if self.variance is None:
                raise InvalidDataError("variance flagged but missing", context={"ticket_id": self.ticket_id})
            variance = _real(self.variance, "variance", InvalidDataError)
            if variance < 0:
                raise InvalidDataError("variance must be non-negative", context={"ticket_id": self.ticket_id})
        elif self.variance is not None:
            raise InvalidDataError("variance supplied without the variance flag", context={"ticket_id": self.ticket_id})
        levels = _levels(self.quantile_levels, InvalidDataError)
        quantiles: NDArray[np.float64] | None = None
        if Capability.MARGINAL_QUANTILES in caps:
            if self.quantiles is None or not levels:
                raise InvalidDataError("marginal quantiles flagged but missing", context={"ticket_id": self.ticket_id})
            quantiles = _float_array(self.quantiles, "quantiles", 1, InvalidDataError)
            if quantiles.shape[0] != len(levels):
                raise InvalidDataError("one quantile per issued level is required")
            if bool(np.any(np.diff(quantiles) < 0)):
                raise InvalidDataError(
                    "quantiles must be non-decreasing in the level", context={"ticket_id": self.ticket_id}
                )
        elif self.quantiles is not None or levels:
            raise InvalidDataError(
                "quantiles supplied without the marginal_quantiles flag", context={"ticket_id": self.ticket_id}
            )
        features = (
            _empty_features() if self.features is None else _float_array(self.features, "features", 1, InvalidDataError)
        )
        for name, value in (
            ("issued_at_ns", issued),
            ("feature_cutoff_ns", cutoff),
            ("capabilities", caps),
            ("mean", mean),
            ("variance", variance),
            ("quantile_levels", levels),
            ("quantiles", quantiles),
            ("features", features),
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def from_query(
        cls,
        query: ForecastQuery,
        *,
        model_id: str,
        model_version: str,
        mean: float,
        variance: float | None = None,
        quantiles: Sequence[float] | NDArray[np.float64] | None = None,
    ) -> ForecastTicket:
        """Issue a ticket that copies the query's identity, clocks, transform version and features."""
        caps = {Capability.MEAN}
        if variance is not None:
            caps.add(Capability.VARIANCE)
        if query.quantile_levels:
            if quantiles is None:
                raise InvalidDataError(
                    "the query requested quantiles the forecast does not provide", context={"query_id": query.query_id}
                )
            caps.add(Capability.MARGINAL_QUANTILES)
        elif quantiles is not None:
            raise InvalidDataError(
                "quantiles supplied but the query requested none", context={"query_id": query.query_id}
            )
        return cls(
            ticket_id=query.query_id,
            stream=query.stream,
            target=query.target,
            issued_at_ns=query.issued_at_ns,
            feature_cutoff_ns=query.feature_cutoff_ns,
            model_id=model_id,
            model_version=model_version,
            transform_version=query.transform_version,
            capabilities=frozenset(caps),
            mean=mean,
            variance=variance,
            quantile_levels=query.quantile_levels,
            quantiles=None if quantiles is None else np.asarray(quantiles),
            features=query.features,
        )

    @property
    def earliest_available_ns(self) -> int:
        """No outcome may become available before this clock (and never before the issue itself)."""
        return max(self.issued_at_ns, self.target.earliest_available_ns(self.feature_cutoff_ns))

    def require(self, capability: Capability) -> None:
        require_capability(self.capabilities, capability, component=f"ticket {self.ticket_id}")

    def quantile(self, level: float) -> float:
        """The issued quantile at exactly ``level``; a finite grid implies no interpolation."""
        self.require(Capability.MARGINAL_QUANTILES)
        assert self.quantiles is not None
        for issued_level, value in zip(self.quantile_levels, self.quantiles.tolist(), strict=True):
            if issued_level == level:
                return float(value)
        raise UnsupportedCapabilityError(
            "quantile level was not issued", context={"level": level, "issued": self.quantile_levels}
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "stream": self.stream,
            "target": self.target.to_mapping(),
            "issued_at_ns": encode_ns(self.issued_at_ns),
            "feature_cutoff_ns": encode_ns(self.feature_cutoff_ns),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "transform_version": self.transform_version,
            "capabilities": sorted(c.value for c in self.capabilities),
            "mean": self.mean,
            "variance": self.variance,
            "quantile_levels": list(self.quantile_levels),
            "quantiles": None if self.quantiles is None else self.quantiles.tolist(),
            "features": self.features.tolist(),
        }

    @classmethod
    def from_mapping(cls, doc: object) -> ForecastTicket:
        d = require_keys(
            doc,
            (
                "ticket_id",
                "stream",
                "target",
                "issued_at_ns",
                "feature_cutoff_ns",
                "model_id",
                "model_version",
                "transform_version",
                "capabilities",
                "mean",
                "variance",
                "quantile_levels",
                "quantiles",
                "features",
            ),
            where="ticket",
        )
        names = d["capabilities"]
        if not isinstance(names, list) or len(set(map(str, names))) != len(names):
            raise InvalidDataError("ticket capabilities must be a list of distinct names")
        if not isinstance(d["quantile_levels"], list) or not isinstance(d["features"], list):
            raise InvalidDataError("ticket quantile_levels and features must be lists")
        try:
            caps = frozenset(Capability(n) for n in names)
        except ValueError as exc:
            raise InvalidDataError("unknown capability name", context={"names": names}) from exc
        quantiles = d["quantiles"]
        return cls(
            ticket_id=d["ticket_id"],
            stream=d["stream"],
            target=TargetSpec.from_mapping(d["target"]),
            issued_at_ns=_decode_ns(d["issued_at_ns"], "issued_at_ns", InvalidDataError),
            feature_cutoff_ns=_decode_ns(d["feature_cutoff_ns"], "feature_cutoff_ns", InvalidDataError),
            model_id=d["model_id"],
            model_version=d["model_version"],
            transform_version=d["transform_version"],
            capabilities=caps,
            mean=d["mean"],
            variance=d["variance"],
            quantile_levels=tuple(d["quantile_levels"]),
            quantiles=None if quantiles is None else np.asarray(quantiles),
            features=np.asarray(d["features"]),
        )

    def _key(self) -> tuple[object, ...]:
        return (
            self.ticket_id,
            self.stream,
            self.target,
            self.issued_at_ns,
            self.feature_cutoff_ns,
            self.model_id,
            self.model_version,
            self.transform_version,
            self.capabilities,
            self.mean,
            self.variance,
            self.quantile_levels,
            None if self.quantiles is None else self.quantiles.tobytes(),
            self.features.tobytes(),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ForecastTicket):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash((self.ticket_id, self.issued_at_ns))


@dataclass(frozen=True, slots=True)
class Outcome:
    """The realised value of a ticket's target and the clock at which it became visible.

    ``version`` identifies the outcome message; together with ``ticket_id`` it makes scoring idempotent.
    """

    ticket_id: str
    value: float
    available_at_ns: int
    version: int = 0

    def __post_init__(self) -> None:
        _text(self.ticket_id, "ticket_id", InvalidDataError)
        object.__setattr__(self, "value", _real(self.value, "value", InvalidDataError))
        object.__setattr__(self, "available_at_ns", _ns(self.available_at_ns, "available_at_ns", InvalidDataError))
        object.__setattr__(self, "version", _ns(self.version, "version", InvalidDataError))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "value": self.value,
            "available_at_ns": encode_ns(self.available_at_ns),
            "version": self.version,
        }

    @classmethod
    def from_mapping(cls, doc: object) -> Outcome:
        d = require_keys(doc, ("ticket_id", "value", "available_at_ns", "version"), where="outcome")
        return cls(
            d["ticket_id"],
            d["value"],
            _decode_ns(d["available_at_ns"], "available_at_ns", InvalidDataError),
            d["version"],
        )


@dataclass(frozen=True, slots=True, eq=False)
class MatureBatch:
    """Training rows whose labels were all released by ``as_of_ns`` (the input of ``fit``).

    Rows are chronological by feature cutoff and carry the issue-time features. Every label must respect the target's
    availability rule *and* be available at ``as_of_ns``; an immature label is rejected, never silently used.
    """

    target: TargetSpec
    feature_names: tuple[str, ...]
    transform_version: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    feature_cutoff_ns: NDArray[np.int64]
    available_at_ns: NDArray[np.int64]
    as_of_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.target, TargetSpec):
            raise InvalidDataError("target must be a TargetSpec")
        names = _names(self.feature_names, InvalidDataError)
        _text(self.transform_version, "transform_version", InvalidDataError)
        x = _float_array(self.x, "x", 2, InvalidDataError)
        if x.shape[1] != len(names):
            raise InvalidDataError("x needs one column per feature name", context={"columns": x.shape[1]})
        n = x.shape[0]
        y = _float_array(self.y, "y", 1, InvalidDataError)
        cutoff = _clock_array(self.feature_cutoff_ns, "feature_cutoff_ns", InvalidDataError)
        available = _clock_array(self.available_at_ns, "available_at_ns", InvalidDataError)
        if y.shape != (n,) or cutoff.shape != (n,) or available.shape != (n,):
            raise InvalidDataError("batch columns must have one entry per row", context={"rows": n})
        as_of = _ns(self.as_of_ns, "as_of_ns", InvalidDataError)
        if n:
            if bool(np.any(np.diff(cutoff) < 0)):
                raise InvalidDataError("batch rows must be chronological by feature cutoff")
            lag = self.target.horizon_ns + self.target.availability.min_delay_ns
            if int(cutoff.max()) + lag > INT64_MAX:
                raise InvalidDataError("label availability overflows the int64 clock")
            early = int(np.count_nonzero(available < cutoff + lag))
            if early:
                raise InvalidDataError(
                    "label released earlier than the target's availability rule allows", context={"rows": early}
                )
            immature = int(np.count_nonzero(available > as_of))
            if immature:
                raise InvalidDataError(
                    "label is not available at the batch as-of time", context={"rows": immature, "as_of_ns": as_of}
                )
        for name, value in (
            ("feature_names", names),
            ("x", x),
            ("y", y),
            ("feature_cutoff_ns", cutoff),
            ("available_at_ns", available),
            ("as_of_ns", as_of),
        ):
            object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return int(self.y.shape[0])


# --------------------------------------------------------------------------- adapters


@runtime_checkable
class ForecastAdapter(Protocol):
    """A model behind the generic forecasting path (no market, money or execution knowledge).

    ``fit`` sees only a :class:`MatureBatch`. ``predict`` answers a query with an immutable ticket and must not
    mutate the adapter. ``update`` consumes one matured ``(ticket, outcome)`` pair and exists only for adapters that
    declare ``online_update``; adapters declaring ``delayed_feedback`` accept such updates while later tickets are
    still pending. ``get_state``/``set_state`` exchange an exact JSON-compatible snapshot for checkpoints.
    Unsupported operations raise :class:`UnsupportedCapabilityError`; they never return a fabricated value.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def capabilities(self) -> frozenset[Capability]: ...

    def fit(self, batch: MatureBatch) -> None: ...

    def predict(self, query: ForecastQuery) -> ForecastTicket: ...

    def update(self, ticket: ForecastTicket, outcome: Outcome) -> None: ...

    def get_state(self) -> dict[str, Any]: ...

    def set_state(self, state: Mapping[str, Any]) -> None: ...


# --------------------------------------------------------------------------- mixtures


def mixture_moments(
    weights: Sequence[float] | NDArray[np.float64],
    means: Sequence[float] | NDArray[np.float64],
    variances: Sequence[float] | NDArray[np.float64],
) -> tuple[float, float]:
    """Mean and total variance of a mixture of distributions *of the same target*.

    Total variance is ``sum_k w_k (v_k + mu_k**2) - (sum_k w_k mu_k)**2`` (operating contract), evaluated in the
    algebraically identical centred form ``sum_k w_k (v_k + (mu_k - mean)**2)``, which cannot cancel
    catastrophically. Weights are normalised. This is the spread of one target, not the risk of an averaged
    portfolio: that needs the joint law of the combined actions.
    """
    w = _float_array(weights, "weights", 1, InvalidDataError)
    mu = _float_array(means, "means", 1, InvalidDataError)
    v = _float_array(variances, "variances", 1, InvalidDataError)
    if not w.shape == mu.shape == v.shape or w.size == 0:
        raise InvalidDataError("a mixture needs one weight, mean and variance per component")
    if bool(np.any(w < 0)) or float(w.max()) <= 0:
        raise InvalidDataError("mixture weights must be non-negative with a positive total")
    if bool(np.any(v < 0)):
        raise InvalidDataError("component variances must be non-negative")
    scaled = w / float(w.max())  # scaling first keeps very large finite weights from overflowing the total
    normalised = scaled / math.fsum(scaled.tolist())
    mean = math.fsum((normalised * mu).tolist())
    total = math.fsum((normalised * (v + (mu - mean) ** 2)).tolist())
    return mean, total
