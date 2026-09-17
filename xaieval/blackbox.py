"""Black-box models, wrapped behind a single scalar score function.

Every explanation method and every explanation-derived predictor in this project
talks to the black box through :meth:`BlackBox.score`, which returns one real
number per row:

* regression      -> the predicted value;
* classification  -> the predicted probability of the positive class, or its
  log-odds, depending on ``output_scale``.

**Which scale to explain on is a real decision, not a formality.**

The argument for log-odds: additivity of a classifier is a property of its
*score*, and probabilities are a squashed version of that score, so a model that
is additive in log-odds looks non-additive on the probability scale.  Judged on
probabilities alone, the additivity ceiling understates how additive the model
really is.

The argument against, which is why ``probability`` is now the default: a tree
ensemble grows pure leaves and emits probabilities of exactly 0 and 1.  Mapping
those to log-odds requires a clip, the clip value is then the single largest
determinant of the score's variance, and on a near-separable problem the
majority of predictions pin to it -- at which point the log-odds "score" is
essentially a two-valued spike and every downstream quantity built on it is
meaningless.  We hit exactly this: with a 1e-6 clip, 60% of breast-cancer
predictions saturated and the additivity ceiling came out at -2.11.
:func:`resolution_clip` fixes the clip itself, and
:meth:`BlackBox.saturated_fraction` makes the residual risk visible, but the
probability scale simply cannot fail this way.

Both scales are supported and the paper should report the comparison; run
``--output-scale logit`` as a sensitivity analysis and check the reported
saturation fraction before trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

#: Fallback probability clip for models whose output is genuinely continuous
#: (neural nets, Platt-scaled SVMs).  These rarely reach exactly 0 or 1, so the
#: clip almost never binds and its precise value does not matter.
DEFAULT_LOGIT_CLIP = 1e-4


def resolution_clip(estimator) -> float:
    """Smallest probability the model can actually resolve.

    This matters more than it looks.  A random forest grows pure leaves, so it
    reports probabilities of *exactly* 0 and 1 -- and a naive clip at 1e-6 then
    maps those to log-odds of +/-13.8, a precision the model does not possess.
    On a near-separable problem the majority of predictions saturate and the
    log-odds score degenerates into a two-valued spike at the clip bounds; the
    partial dependence curves built from it then sum to something with far more
    variance than the function itself, and the additivity ceiling goes
    *negative*.  That is not a property of the model or of PDP -- it is the clip
    talking.

    An ensemble of ``T`` trees resolves probabilities no finer than ``1/T``, so
    clipping at ``1/(2T)`` is the natural floor: it keeps saturated predictions
    at the edge of what the model can express (+/-6.4 log-odds at T = 300)
    instead of inventing an extreme.
    """
    n_trees = getattr(estimator, "n_estimators", None)
    if n_trees:
        return 1.0 / (2.0 * float(n_trees))
    return DEFAULT_LOGIT_CLIP


def _logit(p: np.ndarray, clip: float = DEFAULT_LOGIT_CLIP) -> np.ndarray:
    clip = float(min(max(clip, 1e-12), 0.49))
    p = np.clip(np.asarray(p, dtype=float), clip, 1.0 - clip)
    return np.log(p / (1.0 - p))


def expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -700, 700)))


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------

_TREE_MODELS = {"random_forest", "extra_trees", "gradient_boosting"}


def _estimator(name: str, task: str, random_state: int):
    clf = task == "classification"
    if name == "random_forest":
        cls = RandomForestClassifier if clf else RandomForestRegressor
        return cls(n_estimators=300, random_state=random_state, n_jobs=1)
    if name == "extra_trees":
        cls = ExtraTreesClassifier if clf else ExtraTreesRegressor
        return cls(n_estimators=300, random_state=random_state, n_jobs=1)
    if name == "gradient_boosting":
        cls = GradientBoostingClassifier if clf else GradientBoostingRegressor
        return cls(random_state=random_state)
    if name == "mlp":
        cls = MLPClassifier if clf else MLPRegressor
        return cls(
            hidden_layer_sizes=(64, 32),
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=random_state,
        )
    if name == "svm_rbf":
        if clf:
            return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state)
        return SVR(kernel="rbf", C=1.0, gamma="scale")
    raise ValueError(f"unknown black-box model {name!r}")


def _search_space(name: str) -> dict[str, list]:
    if name in ("random_forest", "extra_trees"):
        return {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
        }
    if name == "gradient_boosting":
        return {
            "n_estimators": [100, 200, 400],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 5],
            "subsample": [0.7, 1.0],
            "min_samples_leaf": [1, 5, 10],
        }
    if name == "mlp":
        return {
            "hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32), (128, 64)],
            "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
            "learning_rate_init": [1e-3, 1e-2],
        }
    if name == "svm_rbf":
        return {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1.0]}
    return {}


# --------------------------------------------------------------------------
# Wrapper
# --------------------------------------------------------------------------


@dataclass
class BlackBox:
    """A fitted model plus the scalar score function used throughout."""

    name: str
    estimator: Any
    task: str
    output_scale: str = "probability"  # "logit" | "probability"; ignored for regression
    best_params: dict | None = None
    #: Probability clip used when mapping to log-odds.  Set from the model's own
    #: resolution by :func:`fit_blackbox`; see :func:`resolution_clip`.
    logit_clip: float = DEFAULT_LOGIT_CLIP

    # -- scores ------------------------------------------------------------

    def score(self, X: np.ndarray) -> np.ndarray:
        """The quantity that explanations describe and surrogates reproduce."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.task == "regression":
            return np.asarray(self.estimator.predict(X), dtype=float).ravel()
        p = self.proba(X)
        return _logit(p, self.logit_clip) if self.output_scale == "logit" else p

    def proba(self, X: np.ndarray) -> np.ndarray:
        """Positive-class probability (classification only)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        proba = self.estimator.predict_proba(X)
        return np.asarray(proba[:, 1], dtype=float).ravel()

    def score_to_proba(self, s: np.ndarray) -> np.ndarray:
        """Map a score back to a probability, for scoring surrogate output."""
        s = np.asarray(s, dtype=float)
        if self.task == "regression":
            return s
        return expit(s) if self.output_scale == "logit" else np.clip(s, 0.0, 1.0)

    def saturated_fraction(self, X: np.ndarray) -> float:
        """Share of predictions pinned at the clip bounds.

        Reported so that a degenerate log-odds scale is visible in the results
        rather than inferred later from an impossible ceiling.
        """
        if self.task != "classification":
            return 0.0
        p = self.proba(X)
        return float(np.mean((p <= self.logit_clip) | (p >= 1.0 - self.logit_clip)))

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        return (self.proba(X) >= 0.5).astype(int)

    @property
    def score_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.score

    @property
    def is_tree(self) -> bool:
        return self.name in _TREE_MODELS


def fit_blackbox(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    *,
    output_scale: str = "probability",
    tuning: str = "per_dataset",
    n_search_iter: int = 25,
    cv_folds: int = 5,
    random_state: int = 0,
    fixed_params: dict | None = None,
    n_jobs: int = 1,
) -> BlackBox:
    """Fit (and optionally tune) one black box on an already-encoded matrix.

    ``fixed_params`` short-circuits the search -- used to reuse a per-dataset
    tuning result across repeats.
    """
    est = _estimator(name, task, random_state)
    best_params: dict | None = None

    if fixed_params:
        est.set_params(**fixed_params)
        best_params = dict(fixed_params)
    elif tuning != "none":
        space = _search_space(name)
        if space:
            cv = (
                StratifiedKFold(cv_folds, shuffle=True, random_state=random_state)
                if task == "classification"
                else KFold(cv_folds, shuffle=True, random_state=random_state)
            )
            scoring = "roc_auc" if task == "classification" else "r2"
            search = RandomizedSearchCV(
                est,
                space,
                n_iter=n_search_iter,
                scoring=scoring,
                cv=cv,
                random_state=random_state,
                n_jobs=n_jobs,
                refit=True,
                error_score="raise",
            )
            search.fit(X, y)
            est = search.best_estimator_
            best_params = dict(search.best_params_)
            return BlackBox(name, est, task, output_scale, best_params,
                            logit_clip=resolution_clip(est))

    est.fit(X, y)
    return BlackBox(name, est, task, output_scale, best_params,
                    logit_clip=resolution_clip(est))
