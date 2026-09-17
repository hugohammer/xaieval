"""Sanity tests.

The first two are not routine unit tests -- they are the formal claims the
paper makes, checked numerically so that a reviewer's "are you sure?" has an
answer that runs.  Run with::

    python -m pytest Code/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xaieval.anova import additivity_ceiling, r2  # noqa: E402
from xaieval.blackbox import BlackBox, fit_blackbox  # noqa: E402
from xaieval.config import ExplainerConfig  # noqa: E402
from xaieval.controls import noisy_curves, permute_curves  # noqa: E402
from xaieval.curves import Curve1D, CurveSet, smooth_dependence  # noqa: E402
from xaieval.explainers import compute_ale, compute_pdp  # noqa: E402
from xaieval.predictors import (  # noqa: E402
    AdditiveCurveSurrogate,
    AttributionSumIDW,
    BlackBoxKNN,
)
from xaieval.preprocessing import FeatureSpace  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def toy():
    """An additive-plus-interaction regression problem with known structure."""
    rng = np.random.default_rng(0)
    n, p = 600, 4
    X = rng.standard_normal((n, p))
    f = 1.5 * X[:, 0] + np.sin(2 * X[:, 1]) + 0.7 * (X[:, 2] ** 2 - 1)
    y = f + 0.8 * X[:, 0] * X[:, 3] + rng.normal(0, 0.2, n)
    space = FeatureSpace(
        names=[f"x{j}" for j in range(p)],
        is_binary=np.zeros(p, dtype=bool),
        source=[f"x{j}" for j in range(p)],
    )
    bb = fit_blackbox("random_forest", X, y, "regression", tuning="none", random_state=0)
    return X, y, space, bb


# --------------------------------------------------------------------------
# Claim 1: the SHAP-sum IDW construction is kNN on the black box
# --------------------------------------------------------------------------


def test_shap_sum_idw_is_blackbox_knn():
    """The thesis's SHAP predictor contains no SHAP information.

    With local accuracy, S_i = f(x_i) - E[f], and the inverse-distance weights
    sum to one, so E[f] + sum_i w_i S_i == sum_i w_i f(x_i) identically -- no
    matter what the individual attributions are.
    """
    rng = np.random.default_rng(1)
    n_anchor, n_query, p = 120, 40, 6
    X_anchor = rng.standard_normal((n_anchor, p))
    X_query = rng.standard_normal((n_query, p))

    f_anchor = rng.standard_normal(n_anchor) * 2.0 + 0.5
    baseline = float(f_anchor.mean())

    # Any attribution matrix whose rows sum to f(x_i) - E[f] satisfies local
    # accuracy.  Deliberately use *random* per-feature splits to show that the
    # predictor cannot depend on them.
    raw = rng.standard_normal((n_anchor, p))
    A = raw - raw.mean(axis=1, keepdims=True) + ((f_anchor - baseline) / p)[:, None]
    assert np.allclose(A.sum(axis=1), f_anchor - baseline)

    shap_pred = AttributionSumIDW(X_anchor=X_anchor, sums=A.sum(axis=1), baseline=baseline)
    knn = BlackBoxKNN(X_anchor=X_anchor, scores=f_anchor)

    assert np.max(np.abs(shap_pred.predict(X_query) - knn.predict(X_query))) < 1e-9

    # And it is invariant to redistributing attributions across features.
    raw2 = rng.standard_normal((n_anchor, p))
    A2 = raw2 - raw2.mean(axis=1, keepdims=True) + ((f_anchor - baseline) / p)[:, None]
    shap_pred2 = AttributionSumIDW(X_anchor=X_anchor, sums=A2.sum(axis=1), baseline=baseline)
    assert np.max(np.abs(shap_pred.predict(X_query) - shap_pred2.predict(X_query))) < 1e-9


def test_shap_sum_idw_train_fit_is_tautological():
    """Training R^2 of 1.0 against the black box is forced by the construction."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((80, 3))
    f = rng.standard_normal(80)
    pred = AttributionSumIDW(X_anchor=X, sums=f - f.mean(), baseline=float(f.mean()))
    assert r2(f, pred.predict(X)) > 1.0 - 1e-9


# --------------------------------------------------------------------------
# Claim 2: unit coefficients are optimal, not arbitrary
# --------------------------------------------------------------------------


def test_ols_never_worse_than_unit_on_training_data():
    """The internal inconsistency in the thesis tables cannot recur here.

    The unit-coefficient vector lies inside the OLS parameter space, so OLS
    must attain at least its training R^2.  Table 4.5/4.6 of the thesis
    reported the opposite, which is impossible on a shared design matrix.
    """
    rng = np.random.default_rng(3)
    n, p = 300, 5
    X = rng.standard_normal((n, p))
    curves = [
        Curve1D(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30) * (0.5 + j), name=f"x{j}")
        for j in range(p)
    ]
    cs = CurveSet(curves, baseline=0.0, method="test")
    target = cs.transform(X).sum(axis=1) + rng.normal(0, 0.5, n)

    unit = AdditiveCurveSurrogate(cs, mode="unit").fit(X, target)
    ols = AdditiveCurveSurrogate(cs, mode="ols").fit(X, target)

    assert r2(target, ols.predict(X)) >= r2(target, unit.predict(X)) - 1e-9


def test_unit_surrogate_recovers_a_purely_additive_model(toy):
    """When the model is additive, the additive projection reproduces it."""
    rng = np.random.default_rng(4)
    n, p = 800, 4
    X = rng.standard_normal((n, p))
    coefs = np.array([1.5, -0.8, 0.6, 0.3])

    class Additive:
        def predict(self, Z):
            Z = np.atleast_2d(Z)
            return Z @ coefs

    bb = BlackBox("linear", Additive(), "regression")
    space = FeatureSpace([f"x{j}" for j in range(p)], np.zeros(p, bool), [f"x{j}" for j in range(p)])
    cfg = ExplainerConfig(pdp_grid_size=25, pdp_background_size=200)
    cs = compute_pdp(bb, X, space, cfg, rng)

    ceiling = additivity_ceiling(bb, X, cs)
    assert ceiling["additivity_r2"] > 0.99


def test_additivity_ceiling_falls_with_interaction():
    """The ceiling is a property of the model and must track interaction strength."""
    rng = np.random.default_rng(5)
    n, p = 800, 4
    X = rng.standard_normal((n, p))
    space = FeatureSpace([f"x{j}" for j in range(p)], np.zeros(p, bool), [f"x{j}" for j in range(p)])
    cfg = ExplainerConfig(pdp_grid_size=20, pdp_background_size=200)

    ceilings = []
    for rho in (0.0, 0.5, 0.9):

        class F:
            def __init__(self, rho):
                self.rho = rho

            def predict(self, Z):
                Z = np.atleast_2d(Z)
                a = Z[:, 0] + Z[:, 1]
                i = Z[:, 0] * Z[:, 1]
                return np.sqrt(1 - self.rho) * a + np.sqrt(self.rho) * i

        bb = BlackBox(f"f{rho}", F(rho), "regression")
        cs = compute_pdp(bb, X, space, cfg, np.random.default_rng(0))
        ceilings.append(additivity_ceiling(bb, X, cs)["additivity_r2"])

    assert ceilings[0] > ceilings[1] > ceilings[2]
    assert ceilings[0] > 0.98
    assert ceilings[2] < 0.35


# --------------------------------------------------------------------------
# Claim 3: the transform is a no-op on binary features
# --------------------------------------------------------------------------


def test_curve_transform_is_affine_on_binary_features():
    """A curve of a two-valued feature adds nothing a linear model lacked.

    This is why a dataset dominated by one-hot dummies cannot discriminate
    between explanation methods -- the point the thesis's heart-disease setup
    missed, with 14 of its 18 encoded columns binary.
    """
    rng = np.random.default_rng(6)
    x = rng.integers(0, 2, size=400).astype(float)
    a = 2.3 * x + rng.normal(0, 0.1, 400)
    curve = smooth_dependence(x, a, is_binary=True)
    z = curve(x)

    # z must be an exact affine function of x: fitting x on z leaves no residual.
    A = np.column_stack([np.ones_like(z), z])
    beta, *_ = np.linalg.lstsq(A, x, rcond=None)
    assert np.max(np.abs(A @ beta - x)) < 1e-9


# --------------------------------------------------------------------------
# Controls behave as intended
# --------------------------------------------------------------------------


def test_permuted_curves_destroy_predictive_content():
    rng = np.random.default_rng(7)
    n, p = 400, 4
    X = rng.standard_normal((n, p))
    grid = np.linspace(-3, 3, 40)
    cs = CurveSet([Curve1D(grid, np.sin(1.5 * grid) * (j + 1), name=f"x{j}") for j in range(p)])
    target = cs.transform(X).sum(axis=1)

    intact = AdditiveCurveSurrogate(cs, mode="unit").fit(X, target)
    null = AdditiveCurveSurrogate(permute_curves(cs, rng), mode="unit").fit(X, target)

    assert r2(target, intact.predict(X)) > 0.99
    assert r2(target, null.predict(X)) < 0.5


def test_noise_sweep_is_monotone_on_average():
    rng = np.random.default_rng(8)
    n, p = 500, 4
    X = rng.standard_normal((n, p))
    grid = np.linspace(-3, 3, 40)
    cs = CurveSet([Curve1D(grid, np.sin(1.5 * grid) * (j + 1), name=f"x{j}") for j in range(p)])
    target = cs.transform(X).sum(axis=1)

    scores = []
    for sigma in (0.0, 0.5, 1.0, 2.0, 4.0):
        draws = [
            r2(target, AdditiveCurveSurrogate(noisy_curves(cs, sigma, rng), mode="unit")
               .fit(X, target).predict(X))
            for _ in range(8)
        ]
        scores.append(float(np.mean(draws)))
    assert all(scores[i] >= scores[i + 1] - 0.02 for i in range(len(scores) - 1)), scores


# --------------------------------------------------------------------------
# Explainer mechanics
# --------------------------------------------------------------------------


def test_pdp_and_ale_agree_under_independence(toy):
    """Under independent features ALE reduces to PDP up to a constant.

    Their agreement here is a correctness check on the two implementations; any
    *disagreement* on real data is then attributable to feature correlation
    rather than to an implementation difference.
    """
    X, y, space, bb = toy
    cfg = ExplainerConfig(pdp_grid_size=20, pdp_background_size=250, ale_n_bins=20)
    pdp = compute_pdp(bb, X, space, cfg, np.random.default_rng(0))
    ale = compute_ale(bb, X, space, cfg)

    for j in range(space.p):
        a, b = pdp.curves[j](X[:, j]), ale.curves[j](X[:, j])
        if np.std(a) < 1e-8:
            continue
        assert np.corrcoef(a, b)[0, 1] > 0.9, f"feature {j} PDP/ALE disagree under independence"


def test_curves_are_centred(toy):
    X, y, space, bb = toy
    cfg = ExplainerConfig(pdp_grid_size=15, pdp_background_size=200)
    for cs in (compute_pdp(bb, X, space, cfg, np.random.default_rng(0)),
               compute_ale(bb, X, space, cfg)):
        Z = cs.transform(X)
        assert np.max(np.abs(Z.mean(axis=0))) < 1e-8


def test_curve_extrapolation_is_clamped():
    c = Curve1D(np.array([0.0, 1.0, 2.0]), np.array([10.0, 20.0, 30.0]))
    assert c(np.array([-5.0]))[0] == 10.0
    assert c(np.array([99.0]))[0] == 30.0


# --------------------------------------------------------------------------
# Library traps that silently invert or rescale results
# --------------------------------------------------------------------------


def test_lime_coefficients_have_the_right_sign():
    """Guard against LIME's regression-mode sign flip.

    ``lime`` stores the fitted ridge coefficients under label 1 and a
    sign-negated display copy under label 0.  Reading the first key -- the
    obvious choice -- silently inverts every local model, which turned a
    positive fidelity into a negative one before this was caught.  Here the
    black box is monotone increasing in x0, so a correctly extracted local
    model must have a positive x0 coefficient.
    """
    from xaieval.config import ExplainerConfig
    from xaieval.explainers import compute_lime

    rng = np.random.default_rng(11)
    n, p = 300, 4
    X = rng.standard_normal((n, p))
    coefs_true = np.array([2.0, -1.0, 0.5, 0.0])

    class Linear:
        def predict(self, Z):
            return np.atleast_2d(Z) @ coefs_true

    bb = BlackBox("linear", Linear(), "regression")
    space = FeatureSpace([f"x{j}" for j in range(p)], np.zeros(p, bool), [f"x{j}" for j in range(p)])
    cfg = ExplainerConfig(lime_max_points=25, lime_n_samples=800)

    lm = compute_lime(bb, X, space, cfg, np.random.default_rng(0))
    mean_coefs = lm.coefs.mean(axis=0)

    # On an exactly linear black box the local models must recover the truth.
    assert np.allclose(mean_coefs, coefs_true, atol=0.2), mean_coefs
    assert mean_coefs[0] > 0 and mean_coefs[1] < 0

    # And each local model must reproduce the black box at its own anchor.
    own = np.array([lm.intercepts[i] + lm.coefs[i] @ lm.X[i] for i in range(len(lm.X))])
    assert r2(bb.score(lm.X), own) > 0.95


def test_shap_local_accuracy_holds_on_the_explained_scale():
    """SHAP attributions must decompose the score the curves describe.

    TreeSHAP's ``raw`` output for an sklearn tree classifier is the probability,
    not the log-odds, so on the log-odds scale it violates local accuracy.  Every
    construction here assumes the decomposition holds, so the explainer choice
    must be verified against the scale actually in use.
    """
    from xaieval.config import ExplainerConfig
    from xaieval.explainers import compute_shap

    rng = np.random.default_rng(12)
    n, p = 250, 5
    X = rng.standard_normal((n, p))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, n) > 0).astype(int)
    space = FeatureSpace([f"x{j}" for j in range(p)], np.zeros(p, bool), [f"x{j}" for j in range(p)])
    cfg = ExplainerConfig(shap_max_points=60, shap_kernel_background=40)

    for scale in ("probability", "logit"):
        bb = fit_blackbox("random_forest", X, y, "classification",
                          output_scale=scale, tuning="none", random_state=0)
        s = compute_shap(bb, X, space, cfg, np.random.default_rng(0))
        target = bb.score(s.X)
        resid = np.max(np.abs(s.baseline + s.sums - target)) / (np.std(target) + 1e-12)
        assert resid < 1e-6, f"local accuracy violated on the {scale} scale: {resid:.2e}"


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


def test_one_hot_drops_reference_level_so_design_is_full_rank():
    """The thesis's 'effective rank 16 of 18' was an encoding artefact."""
    import pandas as pd

    from xaieval.datasets import Dataset
    from xaieval.preprocessing import design_matrix_diagnostics, fit_transform

    rng = np.random.default_rng(9)
    n = 300
    df = pd.DataFrame({
        "cont": rng.standard_normal(n),
        "cat": rng.choice(["a", "b", "c", "d"], size=n),
        "bin": rng.integers(0, 2, n),
    })
    ds = Dataset(
        name="t", X=df, y=rng.integers(0, 2, n), task="classification",
        numeric=["cont"], binary=["bin"], categorical=["cat"],
    )
    Z_tr, Z_te, space, _ = fit_transform(ds, df.iloc[:250], df.iloc[250:])
    diag = design_matrix_diagnostics(Z_tr)
    assert diag["rank_deficiency"] == 0, diag
    # 1 continuous + 1 binary dummy + 3 category dummies (4 levels minus one).
    assert space.p == 5
