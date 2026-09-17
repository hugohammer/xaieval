"""One-dimensional effect curves and the design matrices built from them.

A :class:`Curve1D` is a function from one feature's value to a scalar effect.
PDP and ALE produce such curves natively; SHAP and LIME do not, but a curve can
be *estimated* from their per-instance attributions (see
:func:`xaieval.explainers.dependence_curves`).  Putting all four methods behind
the same object is what makes them comparable through a single construction.

Centring convention: a curve is centred so that its mean over the training data
is zero.  With that convention the sum of the curves plus the mean black-box
score is exactly the first-order functional-ANOVA (additive) projection of the
black box under feature independence -- which is the object the theory in the
paper is about.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class Curve1D:
    """Effect of one feature, tabulated on a grid.

    Values between grid points are linearly interpolated; values outside the
    training range are clamped to the nearest endpoint (extrapolating a curve
    that was never estimated out there would invent structure).
    """

    grid: np.ndarray
    values: np.ndarray
    name: str = ""
    is_binary: bool = False
    #: How many training rows supported each grid point; used to weight
    #: centring and to flag grid points estimated from very little data.
    support: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.grid = np.asarray(self.grid, dtype=float).ravel()
        self.values = np.asarray(self.values, dtype=float).ravel()
        if self.grid.shape != self.values.shape:
            raise ValueError(f"curve {self.name!r}: grid {self.grid.shape} vs values {self.values.shape}")
        order = np.argsort(self.grid, kind="stable")
        self.grid, self.values = self.grid[order], self.values[order]
        if self.support is not None:
            self.support = np.asarray(self.support, dtype=float).ravel()[order]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        if self.grid.size == 1:
            return np.full_like(x, self.values[0])
        return np.interp(x, self.grid, self.values, left=self.values[0], right=self.values[-1])

    # -- transformations ---------------------------------------------------

    def centred(self, x_train: np.ndarray) -> "Curve1D":
        """Shift so the curve averages to zero over the observed feature values."""
        offset = float(np.mean(self(x_train)))
        return replace(self, values=self.values - offset)

    def with_values(self, values: np.ndarray) -> "Curve1D":
        return replace(self, values=np.asarray(values, dtype=float))

    @property
    def amplitude(self) -> float:
        """Spread of the curve; the natural scale for noise-based corruption."""
        return float(np.std(self.values)) if self.values.size > 1 else 0.0


@dataclass
class CurveSet:
    """One :class:`Curve1D` per encoded feature, plus the constant baseline."""

    curves: list[Curve1D]
    baseline: float = 0.0
    method: str = ""

    def __len__(self) -> int:
        return len(self.curves)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.curves]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Design matrix ``Z`` with ``Z[i, j] = curve_j(X[i, j])``."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != len(self.curves):
            raise ValueError(f"X has {X.shape[1]} columns, curve set has {len(self.curves)}")
        return np.column_stack([c(X[:, j]) for j, c in enumerate(self.curves)])

    def additive_score(self, X: np.ndarray) -> np.ndarray:
        """``baseline + sum_j curve_j(x_j)`` -- the additive projection."""
        return self.baseline + self.transform(X).sum(axis=1)

    def centred(self, X_train: np.ndarray, baseline: float) -> "CurveSet":
        X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
        return CurveSet(
            curves=[c.centred(X_train[:, j]) for j, c in enumerate(self.curves)],
            baseline=float(baseline),
            method=self.method,
        )

    def map_values(self, fn) -> "CurveSet":
        """Apply ``fn(values, index) -> values`` to every curve (used by controls)."""
        return CurveSet(
            curves=[c.with_values(fn(c.values, j)) for j, c in enumerate(self.curves)],
            baseline=self.baseline,
            method=self.method,
        )

    @property
    def amplitudes(self) -> np.ndarray:
        return np.array([c.amplitude for c in self.curves])


# --------------------------------------------------------------------------
# Estimating a curve from a scatter of per-instance attributions
# --------------------------------------------------------------------------


def smooth_dependence(
    x: np.ndarray,
    a: np.ndarray,
    *,
    name: str = "",
    is_binary: bool = False,
    n_bins: int = 20,
    min_bin: int = 10,
) -> Curve1D:
    """Fit a monotone-free 1-D curve to an attribution-vs-value scatter.

    This is the construction that turns SHAP or LIME output into something that
    can be evaluated at a *new* point.  It is deliberately the same machinery
    for both, so any SHAP/LIME difference the experiment reports is a property
    of the attributions rather than of two different aggregation schemes.

    Binary features get the per-level mean.  Continuous features get quantile
    binned means at the within-bin mean feature value, with bins merged until
    each holds at least ``min_bin`` rows.
    """
    x = np.asarray(x, dtype=float).ravel()
    a = np.asarray(a, dtype=float).ravel()
    finite = np.isfinite(x) & np.isfinite(a)
    x, a = x[finite], a[finite]

    if x.size == 0:
        return Curve1D(np.array([0.0]), np.array([0.0]), name=name, is_binary=is_binary)

    levels = np.unique(x)
    if is_binary or levels.size <= 2:
        vals = np.array([a[x == lv].mean() for lv in levels])
        sup = np.array([float((x == lv).sum()) for lv in levels])
        return Curve1D(levels, vals, name=name, is_binary=True, support=sup)

    # Quantile edges, deduplicated (ties collapse bins, which is correct).
    n_bins = max(2, min(n_bins, max(2, x.size // max(min_bin, 1))))
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        return Curve1D(np.array([x.mean()]), np.array([a.mean()]), name=name, support=np.array([float(x.size)]))

    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, edges.size - 2)

    centres, vals, sup = [], [], []
    carry_x: list[np.ndarray] = []
    carry_a: list[np.ndarray] = []
    for b in range(edges.size - 1):
        m = idx == b
        cx = np.concatenate(carry_x + [x[m]]) if carry_x else x[m]
        ca = np.concatenate(carry_a + [a[m]]) if carry_a else a[m]
        if cx.size == 0:
            continue
        if cx.size < min_bin and b < edges.size - 2:
            carry_x, carry_a = [cx], [ca]  # merge forward into the next bin
            continue
        centres.append(float(cx.mean()))
        vals.append(float(ca.mean()))
        sup.append(float(cx.size))
        carry_x, carry_a = [], []
    if carry_x:  # tail that never reached min_bin
        cx, ca = carry_x[0], carry_a[0]
        if centres:
            # fold into the last accepted bin
            w_old, w_new = sup[-1], float(cx.size)
            vals[-1] = (vals[-1] * w_old + float(ca.sum())) / (w_old + w_new)
            centres[-1] = (centres[-1] * w_old + float(cx.sum())) / (w_old + w_new)
            sup[-1] = w_old + w_new
        else:
            centres, vals, sup = [float(cx.mean())], [float(ca.mean())], [float(cx.size)]

    if len(centres) < 2:
        return Curve1D(np.array([x.mean()]), np.array([a.mean()]), name=name, support=np.array([float(x.size)]))

    return Curve1D(np.array(centres), np.array(vals), name=name, is_binary=False, support=np.array(sup))
