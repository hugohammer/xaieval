"""Reference points that bracket every explanation-derived predictor.

Without these, "all four methods reach AUC 0.90" is uninterpretable -- it may
simply be what any model reaches on the dataset.  The bracket is:

* ``intercept``   -- predict the mean.  The floor; R2 = 0 by construction.
* ``linear``      -- linear/logistic regression on the raw encoded features.
  If a curve surrogate cannot beat this, the explanation transform added
  nothing.  This is the empirical counterpart of the proposition: a curve-based
  additive surrogate is, in the independent-feature linear case, exactly as
  good as an additive model fitted directly to the data.
* ``spline_gam``  -- additive model on spline bases of the raw features, fitted
  directly to the target.  The best *additive* fit obtainable without any
  explanation at all, so it upper-bounds what any additive curve surrogate can
  achieve, and the gap to it measures how much the explanation loses.
* ``blackbox``    -- the model itself.  The ceiling for the ``y`` target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import SplineTransformer

from .predictors import Predictor


@dataclass
class InterceptOnly(Predictor):
    family: str = "baseline"
    variant: str = "intercept"

    def __post_init__(self) -> None:
        self.name = "intercept"
        self.mu_ = 0.0

    def fit(self, X, target):
        self.mu_ = float(np.mean(np.asarray(target, dtype=float)))
        return self

    def predict(self, X):
        return np.full(np.atleast_2d(X).shape[0], self.mu_)


@dataclass
class LinearRaw(Predictor):
    """Least squares on the raw encoded features."""

    family: str = "baseline"
    variant: str = "linear-raw"

    def __post_init__(self) -> None:
        self.name = "linear-raw"
        self.model_ = LinearRegression()

    def fit(self, X, target):
        self.model_.fit(np.atleast_2d(X), np.asarray(target, dtype=float).ravel())
        return self

    def predict(self, X):
        return self.model_.predict(np.atleast_2d(X))


@dataclass
class SplineGAM(Predictor):
    """Additive spline model fitted directly to the target.

    Ridge-regularised because a spline basis on many features is wide; the
    penalty is chosen by leave-one-out CV on the training split.
    """

    n_knots: int = 6
    degree: int = 3
    continuous_idx: np.ndarray | None = None
    family: str = "baseline"
    variant: str = "spline-gam"

    def __post_init__(self) -> None:
        self.name = "spline-gam"
        self.model_ = RidgeCV(alphas=np.logspace(-3, 3, 13))
        self.spline_ = None
        self.cont_ = None

    def _basis(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if fit:
            self.cont_ = (
                np.asarray(self.continuous_idx, dtype=int)
                if self.continuous_idx is not None
                else np.array([j for j in range(X.shape[1]) if len(np.unique(X[:, j])) > 2], dtype=int)
            )
        cont = self.cont_
        rest = np.array([j for j in range(X.shape[1]) if j not in set(cont.tolist())], dtype=int)
        blocks = [X[:, rest]] if rest.size else []
        if cont.size:
            if fit:
                self.spline_ = SplineTransformer(
                    n_knots=self.n_knots, degree=self.degree, include_bias=False
                )
                blocks.append(self.spline_.fit_transform(X[:, cont]))
            else:
                blocks.append(self.spline_.transform(X[:, cont]))
        return np.column_stack(blocks) if blocks else np.zeros((X.shape[0], 0))

    def fit(self, X, target):
        B = self._basis(X, fit=True)
        self.model_.fit(B, np.asarray(target, dtype=float).ravel())
        return self

    def predict(self, X):
        return self.model_.predict(self._basis(X, fit=False))


@dataclass
class BlackBoxReference(Predictor):
    """The black box itself, as a predictor of whatever target is in play.

    Not used in the default run: against ``fhat`` it is trivially $R^2 = 1$, and
    against ``y`` the runner already records it on the probability scale (a raw
    log-odds score compared to a 0/1 outcome would give a meaningless $R^2$).
    Kept because it is the natural reference when adding a new target.
    """

    score_fn: object = None
    family: str = "baseline"
    variant: str = "blackbox"

    def __post_init__(self) -> None:
        self.name = "blackbox"

    def fit(self, X, target):
        return self

    def predict(self, X):
        return np.asarray(self.score_fn(np.atleast_2d(X)), dtype=float).ravel()
