"""Established explanation-quality metrics, for comparison with the framework.

A paper proposing a new evaluation measure has to answer "what does this add?".
The answer is a rank correlation: compute these metrics on the same
explanations, on the same splits, and report how the rankings relate.  High
correlation means the new measure is redundant; low correlation means it is
capturing a different axis, which is the claim worth making.

All four take an attribution matrix, so PDP and ALE participate too via
:func:`xaieval.explainers.curve_attributions`.

References
----------
Yeh et al. (2019), *On the (In)fidelity and Sensitivity of Explanations* --
infidelity, max-sensitivity.
Bhatt et al. (2020), *Evaluating and Aggregating Feature-based Model
Explanations* -- faithfulness correlation, complexity.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def infidelity(
    score_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    A: np.ndarray,
    *,
    n_perturb: int = 50,
    sigma: float = 0.2,
    rng: np.random.Generator | None = None,
) -> float:
    """Yeh et al. (2019) infidelity with Gaussian perturbations.

    ``E_I[ (I . phi(x) - (f(x) - f(x - I)))^2 ]``, averaged over rows.  Lower is
    better.  Reported unnormalised, on the score scale.
    """
    rng = rng or np.random.default_rng(0)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    fx = score_fn(X)
    scale = sigma * (X.std(axis=0, keepdims=True) + 1e-12)

    total = 0.0
    for _ in range(n_perturb):
        I = rng.normal(0.0, 1.0, size=X.shape) * scale
        f_pert = score_fn(X - I)
        predicted = np.einsum("ij,ij->i", I, A)
        actual = fx - f_pert
        total += float(np.mean((predicted - actual) ** 2))
    return total / max(n_perturb, 1)


def faithfulness_correlation(
    score_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    A: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
    n_subsets: int = 50,
    subset_frac: float = 0.3,
    rng: np.random.Generator | None = None,
) -> float:
    """Bhatt et al. (2020) faithfulness correlation.

    Correlation, per instance, between the summed attribution of a random
    feature subset and the drop in the model score when that subset is replaced
    by a baseline value.  Higher is better.  Averaged over instances.
    """
    rng = rng or np.random.default_rng(0)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n, p = X.shape
    baseline = X.mean(axis=0) if baseline is None else np.asarray(baseline, dtype=float)
    k = max(1, int(round(subset_frac * p)))

    fx = score_fn(X)
    attr_sums = np.zeros((n, n_subsets))
    drops = np.zeros((n, n_subsets))

    for s in range(n_subsets):
        S = rng.choice(p, size=k, replace=False)
        Xp = X.copy()
        Xp[:, S] = baseline[S]
        attr_sums[:, s] = A[:, S].sum(axis=1)
        drops[:, s] = fx - score_fn(Xp)

    corrs = []
    for i in range(n):
        a, d = attr_sums[i], drops[i]
        if np.std(a) < 1e-12 or np.std(d) < 1e-12:
            continue
        corrs.append(float(np.corrcoef(a, d)[0, 1]))
    return float(np.mean(corrs)) if corrs else np.nan


def max_sensitivity(
    explain_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    A: np.ndarray,
    *,
    n_perturb: int = 20,
    radius: float = 0.1,
    rng: np.random.Generator | None = None,
) -> float:
    """Yeh et al. (2019) max-sensitivity.

    Largest relative change in the attribution vector under a small input
    perturbation.  Lower is better.  ``explain_fn`` must return attributions
    for arbitrary rows -- for curve-based methods that is just re-evaluating
    the curves, which is cheap; for SHAP/LIME the caller supplies the
    curve-based surrogate of the explanation, which is what the framework
    actually uses to predict.
    """
    rng = rng or np.random.default_rng(0)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    A = np.atleast_2d(np.asarray(A, dtype=float))
    scale = radius * (X.std(axis=0, keepdims=True) + 1e-12)
    denom = np.linalg.norm(A, axis=1) + 1e-12

    worst = np.zeros(X.shape[0])
    for _ in range(n_perturb):
        Xp = X + rng.normal(0.0, 1.0, size=X.shape) * scale
        Ap = np.atleast_2d(np.asarray(explain_fn(Xp), dtype=float))
        rel = np.linalg.norm(Ap - A, axis=1) / denom
        worst = np.maximum(worst, rel)
    return float(np.mean(worst))


def complexity_entropy(A: np.ndarray) -> float:
    """Bhatt et al. (2020) complexity: entropy of normalised |attributions|.

    Lower means the explanation concentrates on few features.  Averaged over
    instances and reported in nats.
    """
    A = np.abs(np.atleast_2d(np.asarray(A, dtype=float)))
    tot = A.sum(axis=1, keepdims=True)
    tot = np.where(tot > 0, tot, 1.0)
    P = A / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.sum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
    return float(np.mean(H))


def sparseness_gini(A: np.ndarray) -> float:
    """Chalasani et al. (2020) sparseness: Gini index of |attributions|.

    Higher means more concentrated.  Complements the entropy measure.
    """
    A = np.abs(np.atleast_2d(np.asarray(A, dtype=float)))
    out = []
    for row in A:
        v = np.sort(row)
        n = v.size
        s = v.sum()
        if s <= 0 or n == 0:
            continue
        idx = np.arange(1, n + 1)
        out.append(float((2.0 * np.sum(idx * v)) / (n * s) - (n + 1.0) / n))
    return float(np.mean(out)) if out else np.nan
