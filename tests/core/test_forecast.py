"""Domain-neutral forecast contracts: target identity, capabilities, immutable tickets, maturity bounds, mixtures."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from financeai.core.errors import (
    IncompatibleSchemaError,
    InvalidConfigError,
    InvalidDataError,
    UnsupportedCapabilityError,
)
from financeai.core.forecast import (
    TICKET_CAPABILITIES,
    AvailabilityRule,
    Capability,
    ForecastAdapter,
    ForecastQuery,
    ForecastTicket,
    MatureBatch,
    Outcome,
    TargetSpec,
    decode_ns,
    encode_ns,
    mixture_moments,
    require_capability,
    require_keys,
    require_schema,
)
from financeai.core.serialization import canonical_json, loads_strict

HOUR = 3_600 * 10**9
T0 = 1_704_067_200 * 10**9  # 2024-01-01T00:00:00Z
ALL_PAYLOAD = frozenset({Capability.MEAN, Capability.VARIANCE, Capability.MARGINAL_QUANTILES})


def _target(**overrides: Any) -> TargetSpec:
    fields: dict[str, Any] = {
        "domain": "demand",
        "schema": "demand.period_total/1",
        "unit": "units",
        "horizon_ns": HOUR,
        "labeler": "demand.period_total@1",
        "availability": AvailabilityRule("reported_after_period_end", min_delay_ns=3 * HOUR),
    }
    fields.update(overrides)
    return TargetSpec(**fields)


def _query(features: Any = None, levels: tuple[float, ...] = ()) -> ForecastQuery:
    return ForecastQuery(
        query_id="q-1",
        stream="sensor-0",
        target=_target(),
        issued_at_ns=T0,
        feature_cutoff_ns=T0,
        features=np.array([1.0, 2.0, 3.0]) if features is None else features,
        feature_names=("a", "b", "c"),
        transform_version="tv1",
        quantile_levels=levels,
    )


def _ticket(**overrides: Any) -> ForecastTicket:
    fields: dict[str, Any] = {
        "ticket_id": "t-1",
        "stream": "sensor-0",
        "target": _target(),
        "issued_at_ns": T0,
        "feature_cutoff_ns": T0 - HOUR,
        "model_id": "m",
        "model_version": "1",
        "transform_version": "tv1",
        "capabilities": ALL_PAYLOAD,
        "mean": 5.0,
        "variance": 2.0,
        "quantile_levels": (0.1, 0.5, 0.9),
        "quantiles": np.array([3.0, 5.0, 7.0]),
        "features": np.array([1.0, 2.0]),
    }
    fields.update(overrides)
    return ForecastTicket(**fields)


# --------------------------------------------------------------------------- capabilities and targets


def test_capability_names_match_the_architecture_contract() -> None:
    assert [c.value for c in Capability] == [
        "mean",
        "variance",
        "marginal_quantiles",
        "cdf",
        "marginal_samples",
        "joint_samples",
        "online_update",
        "delayed_feedback",
    ]
    assert frozenset({Capability.MEAN, Capability.VARIANCE, Capability.MARGINAL_QUANTILES}) == TICKET_CAPABILITIES


def test_target_identity_is_stable_and_covers_every_field() -> None:
    base = _target()
    assert base.identity == _target().identity
    assert len(base.identity) == 64
    variants = [
        _target(domain="sensor"),
        _target(schema="demand.period_total/2"),
        _target(unit="kWh"),
        _target(horizon_ns=2 * HOUR),
        _target(labeler="demand.period_total@2"),
        _target(availability=AvailabilityRule("other", min_delay_ns=3 * HOUR)),
        _target(availability=AvailabilityRule("reported_after_period_end", min_delay_ns=2 * HOUR)),
    ]
    assert len({v.identity for v in variants} | {base.identity}) == len(variants) + 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"domain": ""},
        {"domain": " demand"},
        {"schema": ""},
        {"unit": ""},
        {"labeler": "x\n"},
        {"horizon_ns": -1},
        {"horizon_ns": True},
        {"horizon_ns": 3600.0},
        {"horizon_ns": 2**63},
        {"availability": "after"},
    ],
)
def test_target_spec_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(InvalidConfigError):
        _target(**overrides)


@pytest.mark.parametrize(("name", "delay"), [("x", -1), ("", 0), ("x", True), ("x", 1.0)])
def test_availability_rule_rejects_invalid_values(name: str, delay: object) -> None:
    with pytest.raises(InvalidConfigError):
        AvailabilityRule(name, min_delay_ns=delay)  # type: ignore[arg-type]


def test_target_mapping_round_trip_is_strict() -> None:
    spec = _target()
    doc = loads_strict(canonical_json(spec.to_mapping()))
    assert TargetSpec.from_mapping(doc) == spec
    assert doc["horizon_ns"] == str(HOUR)  # ns clocks travel as decimal strings
    with pytest.raises(InvalidConfigError):
        TargetSpec.from_mapping({**doc, "extra": 1})
    with pytest.raises(InvalidConfigError):
        TargetSpec.from_mapping({k: v for k, v in doc.items() if k != "unit"})
    with pytest.raises(InvalidConfigError):
        TargetSpec.from_mapping({**doc, "horizon_ns": HOUR})  # a JSON number is not an exact clock string


def test_earliest_availability_is_cutoff_plus_horizon_plus_delay() -> None:
    spec = _target()
    assert spec.earliest_available_ns(T0) == T0 + 4 * HOUR
    with pytest.raises(InvalidDataError):
        spec.earliest_available_ns(2**63 - HOUR)


@pytest.mark.parametrize("value", [0, 1, 2**53 + 1, 2**63 - 1])
def test_ns_clock_strings_round_trip_exactly(value: int) -> None:
    assert decode_ns(encode_ns(value)) == value


@pytest.mark.parametrize("text", ["1e9", "+1", "-1", " 1", "01", "", "1.0", str(2**63), 5, True, None])
def test_ns_clock_strings_reject_lossy_or_ambiguous_spellings(text: object) -> None:
    with pytest.raises(InvalidDataError):
        decode_ns(text)


@pytest.mark.parametrize("value", [-1, 2**63, True, 1.0])
def test_encode_ns_rejects_values_outside_int64(value: object) -> None:
    with pytest.raises(InvalidDataError):
        encode_ns(value)  # type: ignore[arg-type]


def test_require_schema_accepts_same_major_and_rejects_others() -> None:
    doc = {"schema": "financeai.x", "schema_version": "1.3"}
    assert require_schema(doc, schema="financeai.x", version="1.0", where="x") is doc
    for bad in (
        {"schema": "financeai.y", "schema_version": "1.0"},
        {"schema": "financeai.x", "schema_version": "2.0"},
        {"schema": "financeai.x", "schema_version": "1"},
        {"schema": "financeai.x", "schema_version": 1.0},
        {"schema": "financeai.x"},
        ["financeai.x"],
    ):
        with pytest.raises(IncompatibleSchemaError):
            require_schema(bad, schema="financeai.x", version="1.0", where="x")


def test_require_keys_is_exact() -> None:
    assert require_keys({"a": 1}, ("a",), where="doc") == {"a": 1}
    with pytest.raises(InvalidDataError):
        require_keys({"a": 1, "b": 2}, ("a",), where="doc")
    with pytest.raises(InvalidConfigError):
        require_keys({}, ("a",), where="doc", error=InvalidConfigError)
    with pytest.raises(InvalidDataError):
        require_keys([("a", 1)], ("a",), where="doc")


# --------------------------------------------------------------------------- queries


def test_query_copies_and_freezes_features() -> None:
    source = np.array([1.0, 2.0, 3.0])
    query = _query(source)
    source[0] = 99.0
    assert query.features[0] == 1.0
    assert not query.features.flags.writeable
    with pytest.raises(ValueError):
        query.features[0] = 5.0
    with pytest.raises(ValueError):
        query.features.setflags(write=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"features": np.array([1.0, np.nan, 3.0])},
        {"features": np.array([1.0, 2.0])},
        {"features": np.array([True, False, True])},
        {"features": np.ones((3, 1))},
        {"feature_names": ("a", "a", "c")},
        {"feature_cutoff_ns": T0 + 1},
        {"quantile_levels": (0.5, 0.1)},
        {"quantile_levels": (0.0, 0.5)},
        {"quantile_levels": (0.5, 1.0)},
        {"quantile_levels": (0.5, 0.5)},
        {"query_id": ""},
    ],
)
def test_query_validation(kwargs: dict[str, Any]) -> None:
    fields: dict[str, Any] = {
        "query_id": "q",
        "stream": "s",
        "target": _target(),
        "issued_at_ns": T0,
        "feature_cutoff_ns": T0,
        "features": np.array([1.0, 2.0, 3.0]),
        "feature_names": ("a", "b", "c"),
        "transform_version": "tv1",
    }
    fields.update(kwargs)
    with pytest.raises(InvalidDataError):
        ForecastQuery(**fields)


# --------------------------------------------------------------------------- tickets


def test_ticket_is_immutable_and_never_aliases_caller_arrays() -> None:
    quantiles = np.array([3.0, 5.0, 7.0])
    features = np.array([1.0, 2.0])
    ticket = _ticket(quantiles=quantiles, features=features)
    quantiles[:] = 0.0
    features[:] = 0.0
    assert ticket.quantiles is not None
    assert ticket.quantiles.tolist() == [3.0, 5.0, 7.0]
    assert ticket.features.tolist() == [1.0, 2.0]
    for array in (ticket.quantiles, ticket.features):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array[0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        ticket.mean = 1.0  # type: ignore[misc]


def test_ticket_from_query_does_not_share_memory_with_the_query() -> None:
    query = _query(levels=(0.1, 0.5, 0.9))
    ticket = ForecastTicket.from_query(
        query, model_id="m", model_version="1", mean=4.0, variance=1.0, quantiles=[3.0, 4.0, 5.0]
    )
    assert ticket.capabilities == ALL_PAYLOAD
    assert ticket.ticket_id == query.query_id
    assert (ticket.stream, ticket.target, ticket.issued_at_ns, ticket.feature_cutoff_ns, ticket.transform_version) == (
        query.stream,
        query.target,
        query.issued_at_ns,
        query.feature_cutoff_ns,
        query.transform_version,
    )
    assert ticket.quantile_levels == (0.1, 0.5, 0.9)
    assert ticket.features.tolist() == query.features.tolist()
    assert not np.shares_memory(ticket.features, query.features)
    point = ForecastTicket.from_query(_query(), model_id="m", model_version="1", mean=4.0)
    assert point.capabilities == frozenset({Capability.MEAN})
    assert point.variance is None and point.quantiles is None and point.quantile_levels == ()


def test_ticket_from_query_must_honour_the_requested_quantiles() -> None:
    with pytest.raises(InvalidDataError):
        ForecastTicket.from_query(_query(levels=(0.5,)), model_id="m", model_version="1", mean=1.0)
    with pytest.raises(InvalidDataError):
        ForecastTicket.from_query(_query(), model_id="m", model_version="1", mean=1.0, quantiles=[1.0])


@pytest.mark.parametrize(
    "overrides",
    [
        {"capabilities": frozenset({Capability.MEAN, Capability.MARGINAL_QUANTILES})},  # variance without flag
        {"variance": None},  # flag without variance
        {"capabilities": frozenset({Capability.MEAN, Capability.VARIANCE})},  # quantiles without flag
        {"quantiles": None, "quantile_levels": ()},  # flag without quantiles
        {"capabilities": frozenset({Capability.VARIANCE, Capability.MARGINAL_QUANTILES})},  # no point mean
        {"capabilities": ALL_PAYLOAD | {Capability.CDF}},  # no cdf payload exists
        {"capabilities": ALL_PAYLOAD | {Capability.ONLINE_UPDATE}},  # adapter capability, not a payload
        {"capabilities": frozenset({"mean", "variance", "marginal_quantiles"})},  # plain strings
        {"variance": -1.0},
        {"variance": float("inf")},
        {"quantiles": np.array([7.0, 5.0, 3.0])},
        {"quantiles": np.array([3.0, 5.0])},
        {"quantile_levels": (0.9, 0.5, 0.1)},
        {"mean": float("nan")},
        {"mean": True},
        {"feature_cutoff_ns": T0 + 1},
        {"issued_at_ns": -1},
        {"model_version": ""},
    ],
)
def test_ticket_capability_flags_and_values_are_validated(overrides: dict[str, Any]) -> None:
    with pytest.raises(InvalidDataError):
        _ticket(**overrides)


def test_ticket_capability_access_raises_typed_errors() -> None:
    ticket = _ticket()
    assert ticket.quantile(0.5) == 5.0
    with pytest.raises(UnsupportedCapabilityError):
        ticket.quantile(0.25)  # off the issued grid: no interpolation is implied
    point = _ticket(capabilities=frozenset({Capability.MEAN}), variance=None, quantile_levels=(), quantiles=None)
    with pytest.raises(UnsupportedCapabilityError) as info:
        point.quantile(0.5)
    assert info.value.code == "unsupported_capability"
    with pytest.raises(UnsupportedCapabilityError):
        point.require(Capability.VARIANCE)
    point.require(Capability.MEAN)


def test_ticket_mapping_round_trip_is_exact() -> None:
    for ticket in (
        _ticket(mean=0.1 + 0.2, variance=1e-300, quantiles=np.array([-1e300, 0.3, 1e300])),
        _ticket(
            capabilities=frozenset({Capability.MEAN}), variance=None, quantile_levels=(), quantiles=None, features=None
        ),
    ):
        doc = loads_strict(canonical_json(ticket.to_mapping()))
        restored = ForecastTicket.from_mapping(doc)
        assert restored == ticket
        assert restored.to_mapping() == ticket.to_mapping()
    doc = _ticket().to_mapping()
    with pytest.raises(InvalidDataError):
        ForecastTicket.from_mapping({**doc, "extra": 1})
    with pytest.raises(InvalidDataError):
        ForecastTicket.from_mapping({**doc, "capabilities": ["mean", "variance", "marginal_quantiles", "magic"]})
    for field, value in (
        ("quantile_levels", 5),
        ("quantile_levels", "0.5"),
        ("features", {"a": 1.0}),
        ("quantiles", [3.0, None, 7.0]),
        ("capabilities", "mean"),
    ):
        with pytest.raises(InvalidDataError):
            ForecastTicket.from_mapping({**doc, field: value})


def test_ticket_earliest_availability_never_precedes_issue() -> None:
    instant = _target(horizon_ns=0, availability=AvailabilityRule("immediate"))
    ticket = _ticket(target=instant)
    assert ticket.earliest_available_ns == T0  # the cutoff is earlier, but no outcome may precede the issue
    assert _ticket().earliest_available_ns == T0 - HOUR + 4 * HOUR


def test_unequal_tickets_compare_unequal() -> None:
    assert _ticket() == _ticket()
    assert _ticket() != _ticket(mean=5.5)
    assert _ticket() != _ticket(features=np.array([1.0, 2.5]))
    assert hash(_ticket()) == hash(_ticket())


# --------------------------------------------------------------------------- outcomes and mature batches


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": True},
        {"available_at_ns": -1},
        {"available_at_ns": 1.5},
        {"version": -1},
        {"version": True},
        {"ticket_id": ""},
    ],
)
def test_outcome_validation(kwargs: dict[str, Any]) -> None:
    fields: dict[str, Any] = {"ticket_id": "t-1", "value": 3.0, "available_at_ns": T0, "version": 0}
    fields.update(kwargs)
    with pytest.raises(InvalidDataError):
        Outcome(**fields)


def test_outcome_mapping_round_trip() -> None:
    outcome = Outcome("t-1", 2.5, 2**62 + 7, version=3)
    assert Outcome.from_mapping(loads_strict(canonical_json(outcome.to_mapping()))) == outcome
    with pytest.raises(InvalidDataError):
        Outcome.from_mapping({**outcome.to_mapping(), "value": None})


def _batch(**overrides: Any) -> MatureBatch:
    cutoffs = np.array([T0, T0 + HOUR, T0 + 2 * HOUR], dtype=np.int64)
    fields: dict[str, Any] = {
        "target": _target(),
        "feature_names": ("a", "b"),
        "transform_version": "tv1",
        "x": np.arange(6, dtype=np.float64).reshape(3, 2),
        "y": np.array([1.0, 2.0, 3.0]),
        "feature_cutoff_ns": cutoffs,
        "available_at_ns": cutoffs + 4 * HOUR,
        "as_of_ns": T0 + 6 * HOUR,
    }
    fields.update(overrides)
    return MatureBatch(**fields)


def test_mature_batch_copies_and_freezes_its_rows() -> None:
    x = np.arange(6, dtype=np.float64).reshape(3, 2)
    batch = _batch(x=x)
    x[:] = -1.0
    assert batch.x[0].tolist() == [0.0, 1.0]
    assert len(batch) == 3
    for array in (batch.x, batch.y, batch.feature_cutoff_ns, batch.available_at_ns):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_mature_batch_rejects_labels_that_are_not_yet_available() -> None:
    with pytest.raises(InvalidDataError, match="not available"):
        _batch(as_of_ns=T0 + 5 * HOUR)  # the third label is released at T0 + 6h


@pytest.mark.parametrize(
    "overrides",
    [
        {"available_at_ns": np.array([T0, T0 + HOUR, T0 + 2 * HOUR]) + 4 * HOUR - 1},  # earlier than the rule allows
        {"x": np.ones((3, 3))},
        {"y": np.array([1.0, np.nan, 3.0])},
        {"y": np.array([1.0, 2.0])},
        {"feature_cutoff_ns": np.array([T0 + HOUR, T0, T0 + 2 * HOUR])},  # not chronological
        {"feature_cutoff_ns": np.array([T0, T0 + HOUR, T0 + 2 * HOUR], dtype=np.float64)},
        {"feature_names": ("a",)},
    ],
)
def test_mature_batch_validation(overrides: dict[str, Any]) -> None:
    with pytest.raises(InvalidDataError):
        _batch(**overrides)


# --------------------------------------------------------------------------- adapter protocol


class _Constant:
    model_id = "constant"
    model_version = "1"
    capabilities = frozenset({Capability.MEAN})

    def fit(self, batch: MatureBatch) -> None:
        return None

    def predict(self, query: ForecastQuery) -> ForecastTicket:
        require_capability(
            self.capabilities,
            Capability.MARGINAL_QUANTILES if query.quantile_levels else Capability.MEAN,
            component=self.model_id,
        )
        return ForecastTicket.from_query(query, model_id=self.model_id, model_version=self.model_version, mean=1.0)

    def update(self, ticket: ForecastTicket, outcome: Outcome) -> None:
        require_capability(self.capabilities, Capability.ONLINE_UPDATE, component=self.model_id)

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: Mapping[str, Any]) -> None:
        return None


def test_adapters_conform_structurally_and_raise_capability_errors() -> None:
    adapter = _Constant()
    assert isinstance(adapter, ForecastAdapter)
    assert not isinstance(object(), ForecastAdapter)
    assert adapter.predict(_query()).mean == 1.0
    with pytest.raises(UnsupportedCapabilityError) as info:
        adapter.predict(_query(levels=(0.5,)))
    assert info.value.context["capability"] == "marginal_quantiles"
    assert info.value.context["component"] == "constant"
    with pytest.raises(UnsupportedCapabilityError):
        adapter.update(_ticket(), Outcome("t-1", 1.0, T0 + 4 * HOUR))


# --------------------------------------------------------------------------- mixtures


def _oracle(weights: list[float], means: list[float], variances: list[float]) -> tuple[Fraction, Fraction]:
    w = [Fraction(x) for x in weights]
    total = sum(w, Fraction(0))
    w = [x / total for x in w]
    mu = [Fraction(x) for x in means]
    v = [Fraction(x) for x in variances]
    mean = sum((a * b for a, b in zip(w, mu, strict=True)), Fraction(0))
    second = sum((a * (c + b * b) for a, b, c in zip(w, mu, v, strict=True)), Fraction(0))
    return mean, second - mean * mean  # operating contract: sum w (v + mu^2) - (sum w mu)^2


def test_mixture_total_variance_matches_the_operating_contract_formula() -> None:
    mean, variance = mixture_moments([0.2, 0.3, 0.5], [1.0, -2.0, 4.0], [0.5, 1.5, 2.0])
    exact_mean, exact_variance = _oracle([0.2, 0.3, 0.5], [1.0, -2.0, 4.0], [0.5, 1.5, 2.0])
    assert mean == pytest.approx(float(exact_mean), rel=1e-15)
    assert variance == pytest.approx(float(exact_variance), rel=1e-14)
    assert mixture_moments([3.0], [2.0], [0.25]) == (2.0, 0.25)
    # equal means: the mixture variance is the weighted within-component variance only
    assert mixture_moments([1.0, 3.0], [5.0, 5.0], [1.0, 2.0]) == pytest.approx((5.0, 1.75))


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.tuples(st.floats(1e-3, 1e3), st.floats(-1e3, 1e3), st.floats(0.0, 1e3)), min_size=1, max_size=6),
    st.floats(1e-3, 1e3),
)
def test_mixture_moments_property(components: list[tuple[float, float, float]], scale: float) -> None:
    weights, means, variances = (list(c) for c in zip(*components, strict=True))
    mean, variance = mixture_moments(weights, means, variances)
    exact_mean, exact_variance = _oracle(weights, means, variances)
    assert mean == pytest.approx(float(exact_mean), rel=1e-9, abs=1e-9)
    assert variance == pytest.approx(float(exact_variance), rel=1e-9, abs=1e-6)
    within = float(
        sum(Fraction(w) * Fraction(v) for w, v in zip(weights, variances, strict=True))
        / sum(Fraction(w) for w in weights)
    )
    assert variance >= within * (1 - 1e-12) - 1e-12  # between-component spread is never negative
    scaled = mixture_moments([w * scale for w in weights], means, variances)
    assert scaled == pytest.approx((mean, variance), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize(
    "args",
    [
        ([1.0, -0.5], [1.0, 2.0], [1.0, 1.0]),
        ([0.0, 0.0], [1.0, 2.0], [1.0, 1.0]),
        ([1.0, float("nan")], [1.0, 2.0], [1.0, 1.0]),
        ([1.0], [1.0, 2.0], [1.0, 1.0]),
        ([1.0, 1.0], [1.0, 2.0], [1.0, -1.0]),
        ([1.0, 1.0], [1.0, float("inf")], [1.0, 1.0]),
        ([], [], []),
    ],
)
def test_mixture_rejects_invalid_components(args: tuple[list[float], list[float], list[float]]) -> None:
    with pytest.raises(InvalidDataError):
        mixture_moments(*args)
