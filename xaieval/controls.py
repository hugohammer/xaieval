"""Null explanations and the corruption sweep.

A new evaluation metric has to be shown to *measure something*.  Two checks:

**Null explanations.**  Destroy the information in an explanation while keeping
its superficial statistics, and the measure must collapse.  Three nulls:

* ``permuted``  -- shuffle each curve's values across its grid.  The marginal
  distribution of effect sizes is untouched; only the mapping from feature
  value to effect is destroyed.  This is the sharp null: any predictor that
  still performs well was not using the explanation's shape.
* ``randomised`` -- replace each curve with Gaussian noise of matched amplitude.
* ``row_shuffled`` -- for attribution-based methods, permute attribution rows
  across instances before the curves are estimated.

**Corruption sweep.**  Add noise of increasing amplitude and check that the
measure degrades monotonically.  A metric that is flat under corruption cannot
rank explanations; one that collapses immediately is measuring noise.  The
resulting curve is the headline validity evidence for the paper.
"""

from __future__ import annotations

import numpy as np

from .curves import CurveSet


def permute_curves(cs: CurveSet, rng: np.random.Generator) -> CurveSet:
    """Shuffle each curve's values across its grid points."""

    def _fn(values, j):
        if values.size < 2:
            return values
        return rng.permutation(values)

    out = cs.map_values(_fn)
    out.method = f"{cs.method}-permuted"
    return out


def randomise_curves(cs: CurveSet, rng: np.random.Generator) -> CurveSet:
    """Replace each curve with amplitude-matched Gaussian noise."""

    def _fn(values, j):
        amp = float(np.std(values)) if values.size > 1 else 0.0
        draw = rng.normal(0.0, amp if amp > 0 else 1e-8, size=values.shape)
        return draw - draw.mean()

    out = cs.map_values(_fn)
    out.method = f"{cs.method}-randomised"
    return out


def noisy_curves(cs: CurveSet, sigma: float, rng: np.random.Generator) -> CurveSet:
    """Add ``sigma`` standard deviations of noise to each curve's values.

    Noise is scaled per curve by that curve's own amplitude, so a feature with
    a large effect and one with a small effect are corrupted comparably.
    """
    if sigma <= 0:
        return cs

    def _fn(values, j):
        amp = float(np.std(values)) if values.size > 1 else 0.0
        if amp <= 0:
            return values
        return values + rng.normal(0.0, sigma * amp, size=values.shape)

    out = cs.map_values(_fn)
    out.method = f"{cs.method}-noise{sigma:g}"
    return out


def shuffle_attribution_rows(A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute attribution rows, breaking the instance/attribution pairing."""
    A = np.asarray(A, dtype=float)
    return A[rng.permutation(A.shape[0])]


def noisy_attributions(A: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add per-feature-scaled Gaussian noise to an attribution matrix."""
    A = np.asarray(A, dtype=float)
    if sigma <= 0:
        return A
    scale = A.std(axis=0, keepdims=True)
    scale = np.where(scale > 0, scale, 1e-12)
    return A + rng.normal(0.0, 1.0, size=A.shape) * sigma * scale
