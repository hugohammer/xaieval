"""Explanation-derived predictors: the core of the framework.

Every explanation method induces a function that can be evaluated at a new
point.  The framework measures how well that induced function reproduces

* ``y``       -- the original target, and
* ``f(x)``    -- the black box's own score (a global, out-of-sample fidelity
                 measure).

Three constructions are implemented:

``AdditiveCurveSurrogate``
    ``g(x) = a0 + sum_j a_j curve_j(x_j)``.  Applies to all four methods once
    their output is in curve form, which is what makes the comparison fair.
    The ``unit`` variant fixes every ``a_j = 1``; under feature independence
    that is the L2-optimal additive projection of the black box, not an
    arbitrary constraint, and it is the variant the theory speaks about.

``LocalModelIDW``
    LIME's native construction: an inverse-distance-weighted average of the
    training-point local linear models evaluated at the new point.  Legitimate
    because a LIME local model *is* a function over feature space.

``AttributionSumIDW``
    The thesis's SHAP construction, retained solely to demonstrate that it is
    algebraically identical to inverse-distance kNN on the black box's
    predictions and therefore carries no SHAP information.  Never report it as
    a SHAP result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

from .curves import CurveSet
from .explainers import LocalModelSet


class Predictor:
    """Minimal interface: ``fit(X, target)`` then ``predict(X)``."""

    name: str = "predictor"
    family: str = ""
    variant: str = ""

    def fit(self, X: np.ndarray, target: np.ndarray) -> "Predictor":  # pragma: no cover
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------
# Additive curve surrogate
# --------------------------------------------------------------------------


@dataclass
class AdditiveCurveSurrogate(Predictor):
    """``a0 + sum_j a_j curve_j(x_j)``, fitted by OLS, ridge, or held at unity."""

    curveset: CurveSet
    mode: str = "unit"  # "unit" | "ols" | "ridge"
    family: str = ""
    ridge_alphas: tuple = (1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 100.0)

    coef_: np.ndarray | None = None
    intercept_: float = 0.0

    def __post_init__(self) -> None:
        self.family = self.family or self.curveset.method
        self.variant = self.mode
        self.name = f"{self.family}-curve-{self.mode}"

    def fit(self, X: np.ndarray, target: np.ndarray) -> "AdditiveCurveSurrogate":
        Z = self.curveset.transform(X)
        target = np.asarray(target, dtype=float).ravel()

        if self.mode == "unit":
            # Coefficients fixed at 1; only the level is free.  This is the
            # additive projection, shifted to the target's mean.
            self.coef_ = np.ones(Z.shape[1], dtype=float)
            self.intercept_ = float(np.mean(target) - np.mean(Z.sum(axis=1)))
        elif self.mode == "ols":
            # lstsq (not the normal equations) so a rank-deficient Z yields the
            # minimum-norm solution instead of blowing up.
            model = LinearRegression()
            model.fit(Z, target)
            self.coef_, self.intercept_ = model.coef_.astype(float), float(model.intercept_)
        elif self.mode == "ridge":
            if Z.shape[1] == 0:
                self.coef_, self.intercept_ = np.zeros(0), float(np.mean(target))
            else:
                model = RidgeCV(alphas=np.asarray(self.ridge_alphas))
                try:
                    model.fit(Z, target)
                    self.coef_, self.intercept_ = model.coef_.astype(float), float(model.intercept_)
                    self.alpha_ = float(model.alpha_)
                except Exception:
                    fallback = Ridge(alpha=1.0).fit(Z, target)
                    self.coef_, self.intercept_ = fallback.coef_.astype(float), float(fallback.intercept_)
                    self.alpha_ = 1.0
        else:
            raise ValueError(f"unknown curve fit mode {self.mode!r}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self.curveset.transform(X)
        return self.intercept_ + Z @ self.coef_

    def coefficient_table(self) -> dict[str, float]:
        return dict(zip(self.curveset.names, np.asarray(self.coef_, dtype=float)))


# --------------------------------------------------------------------------
# Inverse-distance weighting shared by the two local constructions
# --------------------------------------------------------------------------


def _idw_weights(
    X_anchor: np.ndarray,
    X_query: np.ndarray,
    power: float,
    tau: float | None,
    eps: float,
) -> np.ndarray:
    """Row-normalised weights, shape (n_query, n_anchor).

    ``d`` is the squared Euclidean distance, matching the thesis definition;
    weights are ``d ** (-power/2)`` so that ``power=2`` reproduces the thesis's
    inverse-squared-distance scheme.
    """
    X_anchor = np.atleast_2d(np.asarray(X_anchor, dtype=float))
    X_query = np.atleast_2d(np.asarray(X_query, dtype=float))

    d2 = (
        np.sum(X_query**2, axis=1)[:, None]
        + np.sum(X_anchor**2, axis=1)[None, :]
        - 2.0 * X_query @ X_anchor.T
    )
    np.maximum(d2, 0.0, out=d2)

    exact = d2 <= eps
    d2 = np.maximum(d2, eps)
    W = d2 ** (-power / 2.0)

    if tau is not None:
        W = np.where(d2 < tau, W, 0.0)

    # A query point coinciding with anchors takes their mean, not an inf.
    hit_rows = exact.any(axis=1)
    if hit_rows.any():
        W[hit_rows] = exact[hit_rows].astype(float)

    row = W.sum(axis=1, keepdims=True)
    row[row == 0.0] = 1.0
    return W / row


@dataclass
class LocalModelIDW(Predictor):
    """Inverse-distance-weighted average of local linear models."""

    local: LocalModelSet
    power: float = 2.0
    tau: float | None = None
    eps: float = 1e-12
    family: str = "LIME"
    variant: str = "idw"

    def __post_init__(self) -> None:
        self.name = f"{self.family}-local-idw"
        self.offset_ = 0.0

    def fit(self, X: np.ndarray, target: np.ndarray) -> "LocalModelIDW":
        # Parameter-free apart from a level correction, so that a systematic
        # offset between the local models and the target does not masquerade
        # as poor explanation quality.
        raw = self._raw_predict(X)
        self.offset_ = float(np.mean(np.asarray(target, dtype=float)) - np.mean(raw))
        return self

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        W = _idw_weights(self.local.X, X, self.power, self.tau, self.eps)
        preds = self.local.evaluate_all(X)  # (m_anchor, n_query)
        return np.einsum("qm,mq->q", W, preds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._raw_predict(X) + self.offset_


@dataclass
class AttributionSumIDW(Predictor):
    """Degenerate construction, kept only as a demonstration.

    ``yhat(x0) = base + sum_i w_i S_i`` where ``S_i = sum_j phi_j(x_i)``.  When
    the attributions satisfy local accuracy this equals ``sum_i w_i f(x_i)``.
    """

    X_anchor: np.ndarray
    sums: np.ndarray
    baseline: float
    power: float = 2.0
    tau: float | None = None
    eps: float = 1e-12
    family: str = "SHAP"
    variant: str = "sum-idw-degenerate"

    def __post_init__(self) -> None:
        self.name = f"{self.family}-sum-idw"

    def fit(self, X: np.ndarray, target: np.ndarray) -> "AttributionSumIDW":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        W = _idw_weights(self.X_anchor, X, self.power, self.tau, self.eps)
        return self.baseline + W @ np.asarray(self.sums, dtype=float)


@dataclass
class BlackBoxKNN(Predictor):
    """Inverse-distance kNN on the black box's own scores.

    Included so the paper can print ``max |AttributionSumIDW - BlackBoxKNN|``
    and show it is zero to machine precision.
    """

    X_anchor: np.ndarray
    scores: np.ndarray
    power: float = 2.0
    tau: float | None = None
    eps: float = 1e-12
    family: str = "control"
    variant: str = "blackbox-knn"

    def __post_init__(self) -> None:
        self.name = "blackbox-knn"

    def fit(self, X: np.ndarray, target: np.ndarray) -> "BlackBoxKNN":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        W = _idw_weights(self.X_anchor, X, self.power, self.tau, self.eps)
        return W @ np.asarray(self.scores, dtype=float)
