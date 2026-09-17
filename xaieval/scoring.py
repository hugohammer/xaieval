"""Scoring a predictor against a target, and aggregating over repeated splits.

Two targets are scored throughout:

``y``     the original outcome.  Answers "how much of the task does the
          explanation retain?".
``fhat``  the black box's own score.  Answers "how faithfully does the
          explanation reproduce the model?", out of sample and globally.

For classification the surrogate is a real-valued function, so R2 is reported
against the score scale (for the ``fhat`` target) or against the 0/1 outcome
(for the ``y`` target, where it is the Brier-type skill score).  Ranking metrics
-- ROC-AUC and Spearman -- are reported alongside because they are invariant to
the monotone rescaling that separates the ``unit`` and ``ols`` variants, and so
isolate shape agreement from scale agreement.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _safe_auc(y_true: np.ndarray, s: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, np.asarray(s, dtype=float).ravel()))
    except ValueError:
        return np.nan


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(stats.spearmanr(a, b).statistic)


def score_against_y(
    y_true: np.ndarray, pred: np.ndarray, task: str, threshold: float | None = None
) -> dict[str, float]:
    """Metrics for a real-valued prediction of the original target."""
    y_true = np.asarray(y_true).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    out = {"r2": r2_score_manual(y_true, pred), "spearman": spearman(y_true, pred)}
    if task == "classification":
        out["auc"] = _safe_auc(y_true, pred)
        thr = float(np.mean(y_true)) if threshold is None else threshold
        # Threshold at the base rate rather than 0.5: the surrogate output is
        # not calibrated as a probability, and 0.5 would penalise a perfectly
        # good ranking for an arbitrary offset.
        cut = float(np.quantile(pred, 1.0 - thr)) if 0 < thr < 1 else float(np.median(pred))
        lab = (pred >= cut).astype(int)
        out["accuracy"] = float(accuracy_score(y_true, lab))
        out["f1"] = float(f1_score(y_true, lab, zero_division=0))
    else:
        out["rmse"] = float(np.sqrt(np.mean((y_true - pred) ** 2)))
    return out


def score_against_blackbox(f_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Fidelity metrics: agreement with the black box's own score."""
    f_true = np.asarray(f_true, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    return {
        "r2": r2_score_manual(f_true, pred),
        "spearman": spearman(f_true, pred),
        "rmse": float(np.sqrt(np.mean((f_true - pred) ** 2))),
        "mae": float(np.mean(np.abs(f_true - pred))),
    }


# --------------------------------------------------------------------------
# Aggregation over repeated splits
# --------------------------------------------------------------------------


def mean_ci(values, alpha: float = 0.05) -> tuple[float, float, float, int]:
    """Mean and a two-sided t confidence interval over repeated splits.

    A t interval (not a bootstrap) because the unit of replication is the
    split, the number of splits is small and chosen by us, and the quantity
    averaged is already a mean over test points.
    """
    v = np.asarray([x for x in np.asarray(values, dtype=float).ravel() if np.isfinite(x)])
    n = v.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(v.mean())
    if n == 1:
        return m, np.nan, np.nan, 1
    se = float(v.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1) * se)
    return m, m - half, m + half, n


def paired_difference(a, b, alpha: float = 0.05) -> dict[str, float]:
    """Paired comparison of two methods across the same splits.

    Splits are shared between methods, so the paired test is both valid and far
    more powerful than comparing two independent means -- which matters when
    the differences of interest are a few hundredths of an AUC.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 2:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "n": int(a.size)}
    d = a - b
    mean, lo, hi, n = mean_ci(d, alpha)
    try:
        p = float(stats.ttest_rel(a, b).pvalue)
    except Exception:
        p = np.nan
    return {"diff": mean, "lo": lo, "hi": hi, "p": p, "n": n}


def dataset_level_comparison(per_cell, a: str, b: str) -> dict[str, float]:
    """Paired comparison with the **dataset** as the unit of replication.

    Cell-level tests over (dataset, model, repeat) are pseudoreplicated: the 20
    repeats of one dataset share most of their training data, and the model
    families share the data entirely.  Treating 1840 cells as independent
    produces p-values in the 1e-100 range that no reviewer should believe, and
    -- more importantly -- it can declare a difference significant that vanishes
    once between-dataset variation is in the error term.  On our data the
    PDP-vs-ALE difference does exactly that (cell-level p = 1e-18, dataset-level
    p = 0.19).

    ``per_cell`` must be indexed by (dataset, model, repeat) with one column per
    method.  We average to one value per dataset and test over those.
    """
    import numpy as _np
    from scipy import stats as _st

    ds = per_cell.groupby(level="dataset").mean()
    x, y = ds[a].to_numpy(float), ds[b].to_numpy(float)
    m = _np.isfinite(x) & _np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return {"diff": _np.nan, "lo": _np.nan, "hi": _np.nan,
                "p_t": _np.nan, "p_wilcoxon": _np.nan, "n_datasets": int(x.size),
                "n_favouring_a": int(_np.sum(x > y))}
    diff, lo, hi, _ = mean_ci(x - y)
    try:
        p_w = float(_st.wilcoxon(x, y).pvalue)
    except Exception:
        p_w = float("nan")
    return {"diff": diff, "lo": lo, "hi": hi,
            "p_t": float(_st.ttest_rel(x, y).pvalue), "p_wilcoxon": p_w,
            "n_datasets": int(x.size), "n_favouring_a": int(_np.sum(x > y))}


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05,
                    return_adjusted: bool = False) -> dict[str, bool] | dict[str, float]:
    """Holm step-down correction.

    Returns which comparisons survive at ``alpha``, or -- with
    ``return_adjusted`` -- the adjusted $p$-values themselves, which can be
    reported directly and compared against any threshold the reader prefers.
    """
    items = [(k, v) for k, v in pvalues.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)

    if return_adjusted:
        adj: dict[str, float] = {k: float("nan") for k in pvalues}
        running = 0.0
        for i, (k, p) in enumerate(items):
            # Holm-adjusted p is the running maximum of (m - i) * p, clipped
            # at 1, so the adjusted values stay monotone in the raw ordering.
            running = max(running, min(1.0, (m - i) * p))
            adj[k] = running
        return adj

    out: dict[str, bool] = {k: False for k in pvalues}
    for i, (k, p) in enumerate(items):
        if p <= alpha / (m - i):
            out[k] = True
        else:
            break
    return out
