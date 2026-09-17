"""The four explanation methods, and the conversion of each into curve form.

PDP and ALE are implemented here rather than taken from a library so that the
grid, the background sample, the binary-feature handling and the centring
convention are identical across methods -- otherwise a comparison between them
partly measures a difference between library defaults.

SHAP and LIME are computed with the reference libraries and then converted to
curves by :func:`dependence_curves`.

.. warning::

   A note on the SHAP construction, because it is the single most important
   correction relative to the thesis.  SHAP satisfies local accuracy,

       sum_j phi_j(x) = f(x) - E[f(X)],

   *exactly*.  So any predictor built from the per-instance **sums** S_i, such
   as the inverse-distance-weighted estimate

       yhat(x0) = E[f] + sum_i w_i S_i    with  sum_i w_i = 1,

   collapses algebraically to ``sum_i w_i f(x_i)`` -- plain inverse-distance
   kNN regression on the black box's own predictions.  The individual
   attributions cancel out; the predictor contains no SHAP information
   whatsoever, and its perfect fit at training points is a tautology (zero
   distance gives infinite weight).  :class:`~xaieval.predictors.AttributionSumIDW`
   implements that construction only so the paper can demonstrate the
   degeneracy numerically; the construction actually evaluated is the additive
   dependence-curve surrogate below, which uses the per-feature attributions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .blackbox import BlackBox
from .config import ExplainerConfig
from .curves import Curve1D, CurveSet, smooth_dependence
from .preprocessing import FeatureSpace


# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------


def _feature_grid(x: np.ndarray, is_binary: bool, grid_size: int) -> np.ndarray:
    levels = np.unique(x)
    if is_binary or levels.size <= grid_size:
        return levels
    qs = np.linspace(0.0, 1.0, grid_size)
    return np.unique(np.quantile(x, qs))


# --------------------------------------------------------------------------
# Partial dependence
# --------------------------------------------------------------------------


def compute_pdp(
    bb: BlackBox,
    X: np.ndarray,
    space: FeatureSpace,
    cfg: ExplainerConfig,
    rng: np.random.Generator,
) -> CurveSet:
    """Empirical partial dependence for every encoded feature.

    ``PD_j(v) = mean_i f(v, x_{-j}^{(i)})`` over a background subsample.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    bg_size = min(cfg.pdp_background_size, n)
    bg_idx = rng.choice(n, size=bg_size, replace=False) if bg_size < n else np.arange(n)
    background = X[bg_idx]

    curves = []
    for j in range(space.p):
        grid = _feature_grid(X[:, j], bool(space.is_binary[j]), cfg.pdp_grid_size)
        vals = np.empty(grid.size, dtype=float)
        Xb = np.repeat(background, 1, axis=0)
        for k, v in enumerate(grid):
            Xv = Xb.copy()
            Xv[:, j] = v
            vals[k] = float(np.mean(bb.score(Xv)))
        curves.append(
            Curve1D(grid, vals, name=space.names[j], is_binary=bool(space.is_binary[j]))
        )

    baseline = float(np.mean(bb.score(X)))
    return CurveSet(curves, baseline=baseline, method="PDP").centred(X, baseline)


# --------------------------------------------------------------------------
# Accumulated local effects
# --------------------------------------------------------------------------


def compute_ale(
    bb: BlackBox,
    X: np.ndarray,
    space: FeatureSpace,
    cfg: ExplainerConfig,
) -> CurveSet:
    """First-order ALE (Apley & Zhu, 2020) for every encoded feature.

    Continuous features: quantile bins, mean within-bin prediction difference
    between the bin edges, accumulated.  Binary features: the mean prediction
    difference between the two levels, evaluated at the observed rows.
    """
    X = np.asarray(X, dtype=float)
    curves = []

    for j in range(space.p):
        xj = X[:, j]
        levels = np.unique(xj)

        if space.is_binary[j] or levels.size <= 2:
            if levels.size == 1:
                curves.append(Curve1D(levels, np.zeros(1), name=space.names[j], is_binary=True))
                continue
            lo, hi = float(levels[0]), float(levels[-1])
            X_lo, X_hi = X.copy(), X.copy()
            X_lo[:, j], X_hi[:, j] = lo, hi
            delta = float(np.mean(bb.score(X_hi) - bb.score(X_lo)))
            curves.append(
                Curve1D(np.array([lo, hi]), np.array([0.0, delta]), name=space.names[j], is_binary=True)
            )
            continue

        n_bins = max(2, min(cfg.ale_n_bins, levels.size - 1))
        edges = np.unique(np.quantile(xj, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 3:
            curves.append(Curve1D(np.array([xj.mean()]), np.zeros(1), name=space.names[j]))
            continue

        # Bin membership: bin k holds edges[k-1] < x <= edges[k], k = 1..K.
        idx = np.clip(np.searchsorted(edges, xj, side="left"), 1, edges.size - 1)

        deltas = np.zeros(edges.size - 1, dtype=float)
        support = np.zeros(edges.size - 1, dtype=float)
        for k in range(1, edges.size):
            m = idx == k
            support[k - 1] = float(m.sum())
            if not m.any():
                continue  # empty bin contributes no increment
            Xk_lo, Xk_hi = X[m].copy(), X[m].copy()
            Xk_lo[:, j], Xk_hi[:, j] = edges[k - 1], edges[k]
            deltas[k - 1] = float(np.mean(bb.score(Xk_hi) - bb.score(Xk_lo)))

        accumulated = np.concatenate([[0.0], np.cumsum(deltas)])
        curves.append(
            Curve1D(
                edges,
                accumulated,
                name=space.names[j],
                is_binary=False,
                support=np.concatenate([[0.0], support]),
            )
        )

    baseline = float(np.mean(bb.score(X)))
    return CurveSet(curves, baseline=baseline, method="ALE").centred(X, baseline)


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------


@dataclass
class AttributionSet:
    """Per-instance attributions on the rows they were computed for."""

    X: np.ndarray  # (m, p) rows explained
    A: np.ndarray  # (m, p) attributions
    baseline: float  # E[f(X)] on the model's score scale
    method: str = ""

    @property
    def sums(self) -> np.ndarray:
        return self.A.sum(axis=1)


#: Largest local-accuracy residual, relative to the spread of the score, that we
#: accept before falling back to a model-agnostic explainer.
LOCAL_ACCURACY_TOL = 1e-6


def compute_shap(
    bb: BlackBox,
    X: np.ndarray,
    space: FeatureSpace,
    cfg: ExplainerConfig,
    rng: np.random.Generator,
    tree_perturbation: str | None = None,
) -> AttributionSet:
    """SHAP attributions on the same score scale as PDP/ALE.

    Choosing the explainer needs care.  TreeSHAP is fast, but for an sklearn
    tree *classifier* its ``raw`` output is the class probability, not the
    log-odds -- so on the log-odds scale its attributions do not sum to
    ``score(x) - E[score]`` and local accuracy silently fails.  Since every
    downstream construction assumes the attributions decompose the same
    quantity the PDP and ALE curves describe, the fast path is used only when
    the scales genuinely match, and the result is verified rather than trusted:
    if the local-accuracy residual exceeds ``LOCAL_ACCURACY_TOL`` relative to
    the spread of the score, we fall back to a model-agnostic explainer applied
    directly to ``bb.score``.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    m = min(cfg.shap_max_points, n)
    idx = rng.choice(n, size=m, replace=False) if m < n else np.arange(n)
    X_expl = X[idx]

    target = bb.score(X_expl)
    spread = float(np.std(target)) + 1e-12

    A = base = None
    tree_scale_matches = bb.is_tree and (
        bb.task == "regression" or bb.output_scale == "probability"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if tree_scale_matches:
            try:
                import shap

                # Be explicit: leaving `data` unset resolves to
                # `tree_path_dependent` (conditional), which silently differs
                # from the marginal convention used everywhere else.
                mode = tree_perturbation or cfg.shap_tree_perturbation
                if mode == "interventional":
                    k = min(cfg.shap_kernel_background, X.shape[0])
                    bg = X[rng.choice(X.shape[0], size=k, replace=False)] if k < X.shape[0] else X
                    expl = shap.TreeExplainer(
                        bb.estimator, data=bg, feature_perturbation="interventional"
                    )
                else:
                    expl = shap.TreeExplainer(
                        bb.estimator, feature_perturbation="tree_path_dependent"
                    )
                A, base = _shap_values(expl, X_expl, positive_class=bb.task == "classification")
                resid = np.max(np.abs(base + np.asarray(A).sum(axis=1) - target)) / spread
                if not np.isfinite(resid) or resid > LOCAL_ACCURACY_TOL:
                    A = base = None  # scale mismatch after all; fall through
            except Exception:
                A = base = None

        if A is None:
            _expl, A, base = _fallback_shap(bb, X, X_expl, cfg, rng)

    A = np.asarray(A, dtype=float)
    if A.shape != X_expl.shape:
        raise RuntimeError(f"SHAP returned shape {A.shape}, expected {X_expl.shape}")

    resid = float(np.max(np.abs(base + A.sum(axis=1) - target)) / spread)
    if resid > 1e-3:
        warnings.warn(
            f"SHAP local accuracy violated (relative residual {resid:.2e}). Constructions "
            f"that rely on the attributions decomposing the score are not valid here."
        )

    return AttributionSet(X=X_expl, A=A, baseline=float(base), method="SHAP")


def _shap_values(explainer, X_expl, positive_class: bool):
    vals = explainer.shap_values(X_expl, check_additivity=False)
    base = explainer.expected_value
    if isinstance(vals, list):  # older API: one array per class
        k = 1 if (positive_class and len(vals) > 1) else 0
        vals = vals[k]
        base = base[k] if np.ndim(base) > 0 else base
    else:
        vals = np.asarray(vals)
        if vals.ndim == 3:  # (n, p, n_classes)
            k = 1 if (positive_class and vals.shape[2] > 1) else 0
            vals = vals[:, :, k]
            base = np.atleast_1d(base)[k] if np.ndim(base) > 0 else base
    return np.asarray(vals, dtype=float), float(np.atleast_1d(base)[0])


def _fallback_shap(bb, X, X_expl, cfg: ExplainerConfig, rng):
    """Model-agnostic SHAP applied directly to ``bb.score``.

    Permutation SHAP is preferred: it satisfies local accuracy exactly by
    construction (the antithetic permutation sum telescopes to
    ``f(x) - E[f]``), which is precisely the property the framework depends on.
    KernelSHAP only approximates it, so it is a last resort.
    """
    import shap

    n = X.shape[0]
    k = min(cfg.shap_kernel_background, n)
    bg_idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
    background = X[bg_idx]
    p = X.shape[1]

    def _clean(vals, expected):
        vals = np.asarray(vals, dtype=float)
        if vals.ndim == 3:
            vals = vals[:, :, -1]
        return vals, float(np.atleast_1d(expected).ravel()[-1])

    # 2p+1 is the minimum for one antithetic permutation pair, which is all
    # local accuracy requires; the multiplier buys variance reduction in the
    # individual attributions, which matter here because they feed the
    # dependence curves.
    max_evals = max(2 * p + 1, cfg.shap_agnostic_evals_mult * p)
    est_calls = X_expl.shape[0] * max_evals * k
    if est_calls > cfg.shap_cost_warn_threshold:
        warnings.warn(
            f"Model-agnostic SHAP will make roughly {est_calls/1e6:.0f}M model calls "
            f"({X_expl.shape[0]} rows x {max_evals} evals x {k} background). This is the "
            f"path taken when the explainer's native scale does not match the scale being "
            f"explained -- a tree classifier on the log-odds scale, for example. To cut it: "
            f"lower explainer.shap_max_points, lower shap_agnostic_evals_mult, lower "
            f"shap_kernel_background, or run with --output-scale probability so TreeSHAP "
            f"applies."
        )

    try:
        masker = shap.maskers.Independent(background, max_samples=k)
        explainer = shap.PermutationExplainer(bb.score, masker, seed=int(rng.integers(0, 2**31 - 1)))
        out = explainer(X_expl, max_evals=max_evals, silent=True)
        vals, base = _clean(out.values, out.base_values)
        return explainer, vals, base
    except Exception:
        pass

    explainer = shap.KernelExplainer(bb.score, background, link="identity")
    vals = explainer.shap_values(
        X_expl, nsamples=cfg.shap_kernel_nsamples, silent=True, l1_reg=f"num_features({p})"
    )
    vals, base = _clean(vals, explainer.expected_value)
    return explainer, vals, base


# --------------------------------------------------------------------------
# LIME
# --------------------------------------------------------------------------


@dataclass
class LocalModelSet:
    """One local linear model per anchor point: ``L_i(x) = b_i + beta_i . x``."""

    X: np.ndarray  # (m, p) anchor points
    intercepts: np.ndarray  # (m,)
    coefs: np.ndarray  # (m, p)
    baseline: float = 0.0
    method: str = "LIME"

    def evaluate(self, x0: np.ndarray) -> np.ndarray:
        """Every local model evaluated at a single point -> shape (m,)."""
        return self.intercepts + self.coefs @ np.asarray(x0, dtype=float).ravel()

    def evaluate_all(self, X0: np.ndarray) -> np.ndarray:
        """Shape (m, n): local model i evaluated at test point k."""
        X0 = np.atleast_2d(np.asarray(X0, dtype=float))
        return self.intercepts[:, None] + self.coefs @ X0.T

    def attributions(self, centre: np.ndarray) -> np.ndarray:
        """``beta_ij * (x_ij - centre_j)`` -- LIME's attribution analogue.

        Subtracting a reference point makes the per-feature contributions
        comparable to SHAP values, which are also deviations from a baseline.
        """
        return self.coefs * (self.X - np.asarray(centre, dtype=float).ravel()[None, :])


def _lime_scalar(value, label: int) -> float:
    if isinstance(value, dict):
        return float(value.get(label, next(iter(value.values()))))
    if np.ndim(value):
        arr = np.asarray(value).ravel()
        return float(arr[label] if label < arr.size else arr[0])
    return float(value)


def _lime_coefficients(exp, x_scaled: np.ndarray, p: int) -> tuple[float, np.ndarray, float]:
    """Extract the local linear model, choosing the label key that is correct.

    This needs care.  In regression mode ``lime`` files the fitted ridge
    coefficients under label 1 and a **sign-flipped display copy** under label
    0::

        ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
        ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

    Reading label 0 -- the first key, and the obvious choice -- therefore yields
    a local model with every coefficient negated, which silently inverts every
    LIME result downstream.  Rather than hard-coding a key and hoping the
    convention holds across versions, we reconstruct the prediction at the
    explained point under each available label and keep the one that reproduces
    LIME's own ``local_pred``.  The residual is returned so the caller can
    verify rather than assume.
    """
    local_exp = getattr(exp, "local_exp", {}) or {0: []}
    target = float(np.ravel(exp.local_pred)[0]) if getattr(exp, "local_pred", None) is not None else np.nan

    best = None
    for key in local_exp:
        coefs = np.zeros(p, dtype=float)
        for j, w in local_exp[key]:
            coefs[int(j)] = float(w)
        intercept = _lime_scalar(exp.intercept, key)
        resid = abs(intercept + float(coefs @ x_scaled) - target) if np.isfinite(target) else 0.0
        if best is None or resid < best[2]:
            best = (intercept, coefs, resid)

    return best  # (intercept, coefs, residual)


def compute_lime(
    bb: BlackBox,
    X: np.ndarray,
    space: FeatureSpace,
    cfg: ExplainerConfig,
    rng: np.random.Generator,
) -> LocalModelSet:
    """Fit a LIME local linear model around a sample of training points."""
    from lime.lime_tabular import LimeTabularExplainer

    X = np.asarray(X, dtype=float)
    n, p = X.shape
    m = min(cfg.lime_max_points, n)
    idx = rng.choice(n, size=m, replace=False) if m < n else np.arange(n)
    X_anchor = X[idx]

    categorical_features = list(np.flatnonzero(space.is_binary))
    seed = int(rng.integers(0, 2**31 - 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = LimeTabularExplainer(
            training_data=X,
            mode="regression",
            feature_names=space.names,
            categorical_features=categorical_features if cfg.lime_discretize else [],
            discretize_continuous=cfg.lime_discretize,
            kernel_width=cfg.lime_kernel_width,
            sample_around_instance=True,
            random_state=seed,
        )

        # LIME fits its ridge on standardised perturbations, so the coefficients
        # come back on the scaled space; we need the mean/scale to reconstruct
        # and then to convert to the original encoded features.
        scaler = getattr(explainer, "scaler", None)
        mu = np.asarray(getattr(scaler, "mean_", np.zeros(p)), dtype=float)
        s = np.asarray(getattr(scaler, "scale_", np.ones(p)), dtype=float)
        s = np.where(s == 0, 1.0, s)

        intercepts = np.zeros(m, dtype=float)
        coefs = np.zeros((m, p), dtype=float)
        residuals = np.zeros(m, dtype=float)
        for i in range(m):
            exp = explainer.explain_instance(
                X_anchor[i],
                bb.score,
                num_features=p,
                num_samples=cfg.lime_n_samples,
                model_regressor=None,
            )
            x_scaled = (X_anchor[i] - mu) / s
            intercepts[i], coefs[i], residuals[i] = _lime_coefficients(exp, x_scaled, p)

    worst = float(np.nanmax(residuals)) if residuals.size else 0.0
    if worst > 1e-6:
        warnings.warn(
            f"LIME local models could not be reconstructed exactly (max residual {worst:.2e} "
            f"against lime's own local_pred); the extracted coefficients may not match the "
            f"fitted local model."
        )

    # Convert from the standardised space to the original encoded features:
    # L(x) = b + c . (x - mu)/s  =  [b - (c/s).mu] + (c/s).x
    intercepts = intercepts - (coefs / s) @ mu
    coefs = coefs / s

    return LocalModelSet(
        X=X_anchor,
        intercepts=intercepts,
        coefs=coefs,
        baseline=float(np.mean(bb.score(X))),
        method="LIME",
    )


# --------------------------------------------------------------------------
# Attribution -> curve conversion (the common construction)
# --------------------------------------------------------------------------


def dependence_curves(
    X_expl: np.ndarray,
    A: np.ndarray,
    space: FeatureSpace,
    baseline: float,
    cfg: ExplainerConfig,
    method: str,
) -> CurveSet:
    """Smooth per-instance attributions into one curve per feature.

    This is the construction that lets a *local* explanation predict at a new
    point without recomputing the explanation: the attribution of feature ``j``
    is modelled as a function of ``x_j`` alone, estimated from the training
    attributions.  Applied to SHAP this is the SHAP-dependence-plot curve;
    applied to LIME it is the same object built from ``beta_ij (x_ij - xbar_j)``.
    """
    X_expl = np.atleast_2d(np.asarray(X_expl, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    curves = [
        smooth_dependence(
            X_expl[:, j],
            A[:, j],
            name=space.names[j],
            is_binary=bool(space.is_binary[j]),
            n_bins=cfg.dependence_n_bins,
            min_bin=cfg.dependence_min_bin,
        )
        for j in range(space.p)
    ]
    return CurveSet(curves, baseline=float(baseline), method=method).centred(X_expl, baseline)


def curve_attributions(curveset: CurveSet, X: np.ndarray) -> np.ndarray:
    """Per-instance attributions implied by a curve set.

    Needed so that the established metrics (infidelity, sensitivity, ...),
    which are defined for attribution vectors, can also be applied to PDP and
    ALE.  For a centred curve set the attribution of feature j at x is simply
    ``curve_j(x_j)``.
    """
    return curveset.transform(X)
