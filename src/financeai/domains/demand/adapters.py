"""Baseline demand forecast adapters: seasonal naive, per-phase seasonal EWMA and ridge on causal lags.

Each implements :class:`financeai.core.forecast.ForecastAdapter`. ``fit`` sees only a mature batch; ``update``
consumes one matured ``(ticket, outcome)`` pair and learns from the *issue-time* features stored in the ticket and
the residual of the *issued* mean, never from a recomputed forecast. Where offered, marginal quantiles are the issued
mean plus empirical quantiles of the most recent matured residuals: a descriptive interval, not a calibrated
guarantee. The ridge adapter is point-only and raises :class:`UnsupportedCapabilityError` for quantile requests.

Learning modes (evolution contract): seasonal naive keeps its point rule FROZEN and BOUNDED-updates its residual
window; seasonal EWMA is a BOUNDED_UPDATE recursion; ridge is a FULL_REFIT every ``refit_every`` matured outcomes over
a bounded window. All three checkpoint exactly through ``get_state``/``set_state`` (JSON-compatible).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar, Final, Self

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import (
    IncompatibleSchemaError,
    InvalidConfigError,
    InvalidDataError,
    UnsupportedCapabilityError,
)
from financeai.core.forecast import (
    Capability,
    ForecastAdapter,
    ForecastQuery,
    ForecastTicket,
    MatureBatch,
    Outcome,
    require_capability,
    require_keys,
)
from financeai.models.ridge import RidgeModel

_STATE_KEYS: Final = ("kind", "config", "fits", "updates", "feature_names", "transform_version", "model")


def _int(value: object, name: str, minimum: int, maximum: int | None = None) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise InvalidConfigError(
            f"{name} must be an integer >= {minimum}" + ("" if maximum is None else f" and <= {maximum}"),
            context={"got": value},
        )
    return value


def _rate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 < float(value) <= 1.0:
        raise InvalidConfigError(f"{name} must be a number in (0, 1]", context={"got": value})
    return float(value)


def _reals(value: object, name: str) -> list[float]:
    if not isinstance(value, list) or not all(
        isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in value
    ):
        raise InvalidDataError(f"{name} must be a list of finite numbers")
    return [float(v) for v in value]


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidDataError(f"{name} must be a non-negative integer", context={"got": value})
    return value


def _residual_quantiles(mean: float, residuals: Sequence[float], levels: tuple[float, ...]) -> NDArray[np.float64]:
    return mean + np.quantile(np.asarray(residuals, dtype=np.float64), levels)


class DemandAdapter(ABC):
    """Shared plumbing: feature binding, capability checks, versioning and exact state exchange."""

    kind: ClassVar[str]
    CAPABILITIES: ClassVar[frozenset[Capability]]
    PARAMS: ClassVar[tuple[str, ...]]

    def __init__(self) -> None:
        self._names: tuple[str, ...] | None = None
        self._transform: str | None = None
        self._fits = 0
        self._updates = 0

    @property
    def model_id(self) -> str:
        return self.kind

    @property
    def model_version(self) -> str:
        """Fitted-state generation: ``f<fits>.u<matured updates>`` (tickets record which state issued them)."""
        return f"f{self._fits}.u{self._updates}"

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self.CAPABILITIES

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> Self:
        if not isinstance(params, Mapping):
            raise InvalidConfigError("adapter params must be an object")
        unknown = sorted(set(params) - set(cls.PARAMS))
        if unknown:
            raise InvalidConfigError(
                f"unknown {cls.kind} parameters", context={"unknown": unknown, "allowed": cls.PARAMS}
            )
        return cls(**params)

    @abstractmethod
    def config(self) -> dict[str, Any]:
        """JSON-native hyperparameters (part of the checkpoint identity)."""

    @abstractmethod
    def fit(self, batch: MatureBatch) -> None: ...

    @abstractmethod
    def predict(self, query: ForecastQuery) -> ForecastTicket: ...

    @abstractmethod
    def update(self, ticket: ForecastTicket, outcome: Outcome) -> None: ...

    @abstractmethod
    def _model_state(self) -> dict[str, Any]: ...

    @abstractmethod
    def _load_model_state(self, state: object, names: tuple[str, ...] | None) -> None: ...

    def _index(self, names: tuple[str, ...], feature: str) -> int:
        try:
            return names.index(feature)
        except ValueError:
            raise InvalidConfigError(f"{self.kind} needs feature {feature!r}", context={"available": names}) from None

    def _check_batch(self, batch: MatureBatch) -> None:
        if not isinstance(batch, MatureBatch):
            raise InvalidDataError("fit needs a MatureBatch")

    def _check_query(self, query: ForecastQuery) -> None:
        if self._names is None:
            raise UnsupportedCapabilityError(f"{self.kind} is not fitted")
        if query.transform_version != self._transform or query.feature_names != self._names:
            raise InvalidDataError(
                "query features differ from the fitted transform",
                context={"fitted": self._transform, "query": query.transform_version},
            )
        if query.quantile_levels:
            require_capability(self.CAPABILITIES, Capability.MARGINAL_QUANTILES, component=self.kind)

    def _check_feedback(self, ticket: ForecastTicket, outcome: Outcome) -> None:
        require_capability(self.CAPABILITIES, Capability.ONLINE_UPDATE, component=self.kind)
        if self._names is None:
            raise UnsupportedCapabilityError(f"{self.kind} is not fitted")
        if outcome.ticket_id != ticket.ticket_id:
            raise InvalidDataError("outcome belongs to another ticket", context={"ticket_id": ticket.ticket_id})
        if ticket.transform_version != self._transform:
            raise InvalidDataError(
                "ticket was issued under another feature transform", context={"ticket_id": ticket.ticket_id}
            )

    def _ticket(
        self,
        query: ForecastQuery,
        mean: float,
        *,
        variance: float | None = None,
        quantiles: NDArray[np.float64] | None = None,
    ) -> ForecastTicket:
        return ForecastTicket.from_query(
            query,
            model_id=self.kind,
            model_version=self.model_version,
            mean=mean,
            variance=variance,
            quantiles=quantiles,
        )

    def get_state(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "config": self.config(),
            "fits": self._fits,
            "updates": self._updates,
            "feature_names": None if self._names is None else list(self._names),
            "transform_version": self._transform,
            "model": self._model_state(),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        d = require_keys(state, _STATE_KEYS, where=f"{self.kind} state")
        if d["kind"] != self.kind or d["config"] != self.config():
            raise IncompatibleSchemaError(
                "adapter state belongs to another kind or configuration",
                context={"kind": d["kind"], "config": d["config"]},
            )
        fits, updates = _count(d["fits"], "fits"), _count(d["updates"], "updates")
        raw_names, transform = d["feature_names"], d["transform_version"]
        if (raw_names is None) != (transform is None) or (raw_names is None) != (fits == 0):
            raise InvalidDataError("adapter state mixes fitted and unfitted fields")
        if raw_names is not None and (
            not isinstance(raw_names, list)
            or not isinstance(transform, str)
            or not all(isinstance(n, str) for n in raw_names)
        ):
            raise InvalidDataError("adapter state feature binding is malformed")
        names = None if raw_names is None else tuple(raw_names)
        self._load_model_state(d["model"], names)
        self._names, self._transform, self._fits, self._updates = names, transform, fits, updates


class SeasonalNaiveAdapter(DemandAdapter):
    """Forecast = value one season earlier (``lag_<season_lag>``); quantiles from matured residuals."""

    kind = "seasonal_naive"
    CAPABILITIES = frozenset(
        {Capability.MEAN, Capability.MARGINAL_QUANTILES, Capability.ONLINE_UPDATE, Capability.DELAYED_FEEDBACK}
    )
    PARAMS = ("season_lag", "residual_window", "min_residuals")

    def __init__(self, season_lag: int = 168, residual_window: int = 336, min_residuals: int = 24) -> None:
        super().__init__()
        self.season_lag = _int(season_lag, "season_lag", 1)
        self.residual_window = _int(residual_window, "residual_window", 1)
        self.min_residuals = _int(min_residuals, "min_residuals", 1, self.residual_window)
        self._col = -1
        self._residuals: deque[float] = deque(maxlen=self.residual_window)

    def config(self) -> dict[str, Any]:
        return {
            "season_lag": self.season_lag,
            "residual_window": self.residual_window,
            "min_residuals": self.min_residuals,
        }

    def fit(self, batch: MatureBatch) -> None:
        self._check_batch(batch)
        col = self._index(batch.feature_names, f"lag_{self.season_lag}")
        residuals = (batch.y - batch.x[:, col]).tolist()[-self.residual_window :]  # out of sample: no parameters
        if len(residuals) < self.min_residuals:
            raise InvalidDataError(
                "too few matured rows for the residual quantiles",
                context={"rows": len(residuals), "min_residuals": self.min_residuals},
            )
        self._col, self._residuals = col, deque(residuals, maxlen=self.residual_window)
        self._names, self._transform = batch.feature_names, batch.transform_version
        self._fits += 1

    def predict(self, query: ForecastQuery) -> ForecastTicket:
        self._check_query(query)
        mean = float(query.features[self._col])
        quantiles = _residual_quantiles(mean, self._residuals, query.quantile_levels) if query.quantile_levels else None
        return self._ticket(query, mean, quantiles=quantiles)

    def update(self, ticket: ForecastTicket, outcome: Outcome) -> None:
        self._check_feedback(ticket, outcome)
        self._residuals.append(outcome.value - ticket.mean)
        self._updates += 1

    def _model_state(self) -> dict[str, Any]:
        return {"residuals": list(self._residuals)}

    def _load_model_state(self, state: object, names: tuple[str, ...] | None) -> None:
        d = require_keys(state, ("residuals",), where="seasonal_naive model")
        residuals = _reals(d["residuals"], "residuals")
        if len(residuals) > self.residual_window or (names is not None and len(residuals) < self.min_residuals):
            raise InvalidDataError("residual buffer size is inconsistent with the configuration")
        self._col = -1 if names is None else self._index(names, f"lag_{self.season_lag}")
        self._residuals = deque(residuals, maxlen=self.residual_window)


class SeasonalEWMAAdapter(DemandAdapter):
    """One exponentially weighted level per hour-of-week phase, updated only with matured outcomes."""

    kind = "seasonal_ewma"
    CAPABILITIES = frozenset(
        {
            Capability.MEAN,
            Capability.VARIANCE,
            Capability.MARGINAL_QUANTILES,
            Capability.ONLINE_UPDATE,
            Capability.DELAYED_FEEDBACK,
        }
    )
    PARAMS = ("alpha", "variance_alpha", "n_phases", "residual_window", "min_residuals")

    def __init__(
        self,
        alpha: float = 0.2,
        variance_alpha: float = 0.05,
        n_phases: int = 168,
        residual_window: int = 336,
        min_residuals: int = 24,
    ) -> None:
        super().__init__()
        self.alpha = _rate(alpha, "alpha")
        self.variance_alpha = _rate(variance_alpha, "variance_alpha")
        self.n_phases = _int(n_phases, "n_phases", 1)
        self.residual_window = _int(residual_window, "residual_window", 1)
        self.min_residuals = _int(min_residuals, "min_residuals", 1, self.residual_window)
        self._phase_col = -1
        self._levels: list[float] = []
        self._variance = 0.0
        self._residuals: deque[float] = deque(maxlen=self.residual_window)

    def config(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "variance_alpha": self.variance_alpha,
            "n_phases": self.n_phases,
            "residual_window": self.residual_window,
            "min_residuals": self.min_residuals,
        }

    def _phase(self, value: float) -> int:
        phase = int(value)
        if phase != value or not 0 <= phase < self.n_phases:
            raise InvalidDataError(
                "phase feature must be an integer in [0, n_phases)", context={"phase": value, "n_phases": self.n_phases}
            )
        return phase

    def fit(self, batch: MatureBatch) -> None:
        self._check_batch(batch)
        col = self._index(batch.feature_names, "phase")
        levels: list[float | None] = [None] * self.n_phases
        residuals: deque[float] = deque(maxlen=self.residual_window)
        variance: float | None = None
        n_residuals = 0
        for raw_phase, y in zip(batch.x[:, col].tolist(), batch.y.tolist(), strict=True):
            p = self._phase(raw_phase)
            level = levels[p]
            if level is None:
                levels[p] = y
                continue
            r = y - level  # one-step-ahead residual of the level as it stood before this row
            residuals.append(r)
            n_residuals += 1
            variance = r * r if variance is None else (1 - self.variance_alpha) * variance + self.variance_alpha * r * r
            levels[p] = level + self.alpha * (y - level)
        missing = [p for p, level in enumerate(levels) if level is None]
        if missing:
            raise InvalidDataError("training rows do not cover every phase", context={"missing": len(missing)})
        if variance is None or n_residuals < self.min_residuals:
            raise InvalidDataError(
                "too few matured rows for the residual quantiles",
                context={"residuals": n_residuals, "min_residuals": self.min_residuals},
            )
        self._phase_col, self._variance, self._residuals = col, variance, residuals
        self._levels = [float(level) for level in levels if level is not None]
        self._names, self._transform = batch.feature_names, batch.transform_version
        self._fits += 1

    def predict(self, query: ForecastQuery) -> ForecastTicket:
        self._check_query(query)
        mean = self._levels[self._phase(float(query.features[self._phase_col]))]
        quantiles = _residual_quantiles(mean, self._residuals, query.quantile_levels) if query.quantile_levels else None
        return self._ticket(query, mean, variance=self._variance, quantiles=quantiles)

    def update(self, ticket: ForecastTicket, outcome: Outcome) -> None:
        self._check_feedback(ticket, outcome)
        p = self._phase(float(ticket.features[self._phase_col]))
        r = outcome.value - ticket.mean  # residual of the issued forecast
        self._residuals.append(r)
        self._variance = (1 - self.variance_alpha) * self._variance + self.variance_alpha * r * r
        level = self._levels[p]
        self._levels[p] = level + self.alpha * (outcome.value - level)
        self._updates += 1

    def _model_state(self) -> dict[str, Any]:
        return {"levels": list(self._levels), "variance": self._variance, "residuals": list(self._residuals)}

    def _load_model_state(self, state: object, names: tuple[str, ...] | None) -> None:
        d = require_keys(state, ("levels", "variance", "residuals"), where="seasonal_ewma model")
        levels, residuals = _reals(d["levels"], "levels"), _reals(d["residuals"], "residuals")
        variance = _reals([d["variance"]], "variance")[0]
        fitted = names is not None
        if (
            len(levels) != (self.n_phases if fitted else 0)
            or variance < 0
            or len(residuals) > self.residual_window
            or (fitted and len(residuals) < self.min_residuals)
        ):
            raise InvalidDataError("seasonal_ewma state is inconsistent with the configuration")
        self._phase_col = -1 if names is None else self._index(names, "phase")
        self._levels, self._variance = levels, variance
        self._residuals = deque(residuals, maxlen=self.residual_window)


class RidgeLagAdapter(DemandAdapter):
    """Ridge regression (train-only standardisation) on causal lag/mean/calendar features, refitted on matured data."""

    kind = "ridge_lags"
    CAPABILITIES = frozenset({Capability.MEAN, Capability.ONLINE_UPDATE, Capability.DELAYED_FEEDBACK})
    PARAMS = ("alpha", "features", "window", "refit_every")
    EXCLUDED: ClassVar[frozenset[str]] = frozenset({"phase"})
    """Default selection drops the hour-of-week index, which is categorical, not a linear quantity."""

    def __init__(
        self, alpha: float = 1.0, features: Sequence[str] | None = None, window: int = 672, refit_every: int = 24
    ) -> None:
        super().__init__()
        if isinstance(alpha, bool) or not isinstance(alpha, int | float) or not math.isfinite(alpha) or alpha < 0:
            raise InvalidConfigError("alpha must be a finite non-negative number", context={"got": alpha})
        self.alpha = float(alpha)
        if features is not None:
            if isinstance(features, str) or not isinstance(features, Sequence) or not features:
                raise InvalidConfigError("features must be a non-empty list of feature names")
            if not all(isinstance(f, str) for f in features) or len(set(features)) != len(features):
                raise InvalidConfigError("features must be distinct names", context={"features": list(features)})
        self.features: tuple[str, ...] | None = None if features is None else tuple(features)
        self.window = _int(window, "window", 2)
        self.refit_every = _int(refit_every, "refit_every", 1)
        self._cols: list[int] = []
        self._selected: tuple[str, ...] = ()
        self._model: RidgeModel | None = None
        self._buffer_x: deque[tuple[float, ...]] = deque(maxlen=self.window)
        self._buffer_y: deque[float] = deque(maxlen=self.window)
        self._since_refit = 0

    def config(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "features": None if self.features is None else list(self.features),
            "window": self.window,
            "refit_every": self.refit_every,
        }

    def _selection(self, names: tuple[str, ...]) -> tuple[tuple[str, ...], list[int]]:
        selected = self.features if self.features is not None else tuple(n for n in names if n not in self.EXCLUDED)
        return selected, [self._index(names, n) for n in selected]

    def _refit(self) -> None:
        self._model = RidgeModel(alpha=self.alpha).fit(
            np.array(self._buffer_x, dtype=np.float64), np.array(self._buffer_y, dtype=np.float64)
        )
        self._since_refit = 0
        self._fits += 1

    def fit(self, batch: MatureBatch) -> None:
        self._check_batch(batch)
        selected, cols = self._selection(batch.feature_names)
        x, y = batch.x[-self.window :, cols], batch.y[-self.window :]
        model = RidgeModel(alpha=self.alpha).fit(x, y)
        self._selected, self._cols, self._model, self._since_refit = selected, cols, model, 0
        self._buffer_x = deque((tuple(row) for row in x.tolist()), maxlen=self.window)
        self._buffer_y = deque(y.tolist(), maxlen=self.window)
        self._names, self._transform = batch.feature_names, batch.transform_version
        self._fits += 1

    def predict(self, query: ForecastQuery) -> ForecastTicket:
        self._check_query(query)
        assert self._model is not None
        mean = float(self._model.predict(query.features[self._cols][None, :])[0])
        return self._ticket(query, mean)

    def update(self, ticket: ForecastTicket, outcome: Outcome) -> None:
        self._check_feedback(ticket, outcome)
        self._buffer_x.append(tuple(ticket.features[self._cols].tolist()))  # issue-time features, matured label
        self._buffer_y.append(outcome.value)
        self._since_refit += 1
        self._updates += 1
        if self._since_refit >= self.refit_every:
            self._refit()

    def _model_state(self) -> dict[str, Any]:
        model = self._model
        return {
            "features": list(self._selected),
            "coef": None if model is None or model.coef_ is None else model.coef_.tolist(),
            "intercept": None if model is None else model.intercept_,
            "mean": None if model is None or model.mean_ is None else model.mean_.tolist(),
            "scale": None if model is None or model.scale_ is None else model.scale_.tolist(),
            "n_train": None if model is None else model.n_train_,
            "buffer_x": [list(row) for row in self._buffer_x],
            "buffer_y": list(self._buffer_y),
            "since_refit": self._since_refit,
        }

    def _load_model_state(self, state: object, names: tuple[str, ...] | None) -> None:
        d = require_keys(
            state,
            ("features", "coef", "intercept", "mean", "scale", "n_train", "buffer_x", "buffer_y", "since_refit"),
            where="ridge_lags model",
        )
        since = _count(d["since_refit"], "since_refit")
        if names is None:
            if d["coef"] is not None or d["buffer_x"] or d["buffer_y"] or d["features"] or since:
                raise InvalidDataError("unfitted ridge state carries model fields")
            self._selected, self._cols, self._model, self._since_refit = (), [], None, 0
            self._buffer_x, self._buffer_y = deque(maxlen=self.window), deque(maxlen=self.window)
            return
        selected, cols = self._selection(names)
        if d["features"] != list(selected):
            raise InvalidDataError("ridge state selects other features than its configuration")
        coef, mean, scale = _reals(d["coef"], "coef"), _reals(d["mean"], "mean"), _reals(d["scale"], "scale")
        intercept = _reals([d["intercept"]], "intercept")[0]
        n_train = _count(d["n_train"], "n_train")
        rows = d["buffer_x"]
        if not isinstance(rows, list):
            raise InvalidDataError("buffer_x must be a list of rows")
        buffer_x = [tuple(_reals(row, "buffer_x row")) for row in rows]
        buffer_y = _reals(d["buffer_y"], "buffer_y")
        k = len(cols)
        if (
            len(coef) != k
            or len(mean) != k
            or len(scale) != k
            or len(buffer_x) != len(buffer_y)
            or len(buffer_y) > self.window
            or any(len(row) != k for row in buffer_x)
            or since >= self.refit_every
        ):
            raise InvalidDataError("ridge state is inconsistent with its configuration")
        model = RidgeModel(alpha=self.alpha)
        model.coef_, model.mean_, model.scale_ = (np.asarray(v, dtype=np.float64) for v in (coef, mean, scale))
        model.intercept_, model.n_train_ = intercept, n_train
        self._selected, self._cols, self._model, self._since_refit = selected, cols, model, since
        self._buffer_x = deque(buffer_x, maxlen=self.window)
        self._buffer_y = deque(buffer_y, maxlen=self.window)


ADAPTER_KINDS: Final[Mapping[str, type[DemandAdapter]]] = MappingProxyType(
    {
        SeasonalNaiveAdapter.kind: SeasonalNaiveAdapter,
        SeasonalEWMAAdapter.kind: SeasonalEWMAAdapter,
        RidgeLagAdapter.kind: RidgeLagAdapter,
    }
)
"""Allowlisted adapter kinds: configurations select a registered kind, never a module path or code."""


def build_adapter(kind: str, params: Mapping[str, Any] | None = None) -> ForecastAdapter:
    """Construct a registered demand adapter from JSON-compatible parameters."""
    if not isinstance(kind, str) or kind not in ADAPTER_KINDS:
        raise InvalidConfigError("unknown demand adapter kind", context={"kind": kind, "known": sorted(ADAPTER_KINDS)})
    adapter: ForecastAdapter = ADAPTER_KINDS[kind].from_params({} if params is None else params)
    return adapter
