"""Ridge regression with train-only standardisation and an unpenalised intercept.

A deliberately small, domain-neutral forecast adapter: ``fit`` on finite training rows,
``predict`` point forecasts. It makes no calibration or uncertainty claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from financeai.core.errors import InvalidConfigError, InvalidDataError, UnsupportedCapabilityError


@dataclass
class RidgeModel:
    alpha: float
    mean_: NDArray[np.float64] | None = field(default=None, init=False)
    scale_: NDArray[np.float64] | None = field(default=None, init=False)
    coef_: NDArray[np.float64] | None = field(default=None, init=False)
    """Coefficients in *original* feature units."""
    intercept_: float = field(default=0.0, init=False)
    n_train_: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise InvalidConfigError("alpha must be finite and non-negative")

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> RidgeModel:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] < 2:
            raise InvalidDataError("ridge needs X of shape (n, k) and y of shape (n,) with n >= 2")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise InvalidDataError("ridge training data must be finite")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        z = (x - mean) / scale
        y_mean = float(y.mean())
        gram = z.T @ z + self.alpha * np.eye(z.shape[1])
        beta = np.linalg.solve(gram, z.T @ (y - y_mean))
        self.mean_, self.scale_ = mean, scale
        self.coef_ = beta / scale
        self.intercept_ = y_mean - float(self.coef_ @ mean)
        self.n_train_ = int(x.shape[0])
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.coef_ is None:
            raise UnsupportedCapabilityError("model is not fitted")
        return np.asarray(x, dtype=np.float64) @ self.coef_ + self.intercept_
