"""The additivity ceiling: how much of a black box any additive explanation can convey.

This is the generalisation of the thesis's Proposition 3.1, and the quantity
that makes the empirical results interpretable.

**Under feature independence**, the L2-optimal additive approximation of a model
``f`` is exactly the sum of its centred partial dependence functions,

    f_add(x) = E[f(X)] + sum_j ( PD_j(x_j) - E[f(X)] ),

i.e. the first-order term of the functional-ANOVA decomposition.  Its quality

    R2_add = 1 - Var(f - f_add) / Var(f)

is then not a property of the PDP *estimate* -- it is a property of the *model*:
the fraction of the model's variance that is additive.  Two consequences the
paper should state:

1.  A curve-based explanation-derived predictor cannot exceed ``R2_add``.  It is
    a ceiling, not a target, and it differs from dataset to dataset.
2.  Reporting raw fidelity therefore conflates "the explanation is good" with
    "the model happens to be additive".  The informative quantity is the
    *attainment ratio* ``R2_achieved / R2_add`` -- how much of the available
    additive structure the explanation actually recovered.

The thesis's finding that a unit-coefficient PDP surrogate reproduces the random
forest with R2 = 0.97 is, read this way, a statement that the forest was 97%
additive on that dataset, not that PDP was an excellent explanation.

.. warning::

   **The independence assumption is load-bearing, and it fails on most real
   data.**  When features are dependent, the sum of marginal partial dependence
   functions is *not* the L2-optimal additive approximation -- that is precisely
   the gap Hooker's generalized functional ANOVA was introduced to close.  Two
   things follow, and both matter for how the number is described in the paper:

   * ``R2_add`` computed here is **not an upper bound** under dependence.  It is
     the reconstruction quality of one particular additive approximation, and a
     genuinely optimal additive fit (backfitting on the joint distribution) can
     beat it.
   * It is **not bounded below by zero** either.  The PDP sum can be worse than
     predicting the mean, giving a negative value.  We observed exactly this on
     three real datasets, though there the dominant cause was a saturated
     log-odds scale rather than dependence alone -- see
     ``blackbox.resolution_clip``.

   So call this quantity "the additive PDP reconstruction R2", and describe it
   as the L2-optimal additive projection *only under feature independence*.  The
   synthetic correlation sweep (``syn_corr30/60/85``) is the instrument for
   quantifying how far the two drift apart; report that drift rather than
   asserting the bound.
"""

from __future__ import annotations

import numpy as np

from .blackbox import BlackBox
from .curves import CurveSet


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def additivity_ceiling(
    bb: BlackBox,
    X: np.ndarray,
    pdp_curves: CurveSet,
) -> dict[str, float]:
    """Fraction of the black box's score variance captured additively.

    ``pdp_curves`` must be centred over the *training* data; ``X`` is the split
    on which the ceiling is evaluated (report the test-split value).

    A negative ``additivity_r2`` is not a bug: it means the PDP sum predicts the
    model worse than the model's own mean does, which can happen under feature
    dependence or on a saturated score scale.  Treat it as a signal that the
    setup is degenerate, not as a small ceiling -- see the module docstring.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    f = bb.score(X)
    f_add = pdp_curves.additive_score(X)
    resid = f - f_add
    return {
        "additivity_r2": r2(f, f_add),
        "interaction_var_frac": float(np.var(resid) / np.var(f)) if np.var(f) > 0 else np.nan,
        "score_var": float(np.var(f)),
        "residual_var": float(np.var(resid)),
    }


def attainment_ratio(achieved_r2: float, ceiling_r2: float) -> float:
    """``achieved / ceiling``, clipped at 0, NaN when the ceiling is degenerate.

    Values near 1 mean the explanation recovered essentially all of the additive
    structure that was there to recover.  Values above 1 are possible for the
    ``y`` target (where the ceiling does not apply) and for fitted variants that
    rescale the curves -- flag rather than clip those.
    """
    if not np.isfinite(ceiling_r2) or ceiling_r2 <= 1e-6:
        return np.nan
    return float(max(achieved_r2, 0.0) / ceiling_r2)


def interaction_strength_h(
    bb: BlackBox,
    X: np.ndarray,
    pdp_curves: CurveSet,
) -> float:
    """Friedman-style overall interaction statistic, ``sqrt(1 - R2_add)``.

    Provided because some readers will expect the H-statistic vocabulary; it is
    a monotone transform of the ceiling and carries no extra information.
    """
    ceiling = additivity_ceiling(bb, X, pdp_curves)["additivity_r2"]
    return float(np.sqrt(max(0.0, 1.0 - ceiling)))
