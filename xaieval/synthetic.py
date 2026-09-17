"""Synthetic data with a known additive/interaction split.

These are the datasets that let the paper make a *validated* claim rather than
an observed one.  On real data the additivity ceiling has to be estimated; here
it is known in closed form, so the measured ceiling can be checked against the
truth and the framework can be shown to recover a quantity it is supposed to
recover.

Each generator produces

    f(x) = sum_j g_j(x_j)  +  rho * h(x)          (the "signal")
    y    = f(x) + noise    or   Bernoulli(sigmoid(f(x)))

where ``h`` is a pure interaction term orthogonal to every additive function of
a single feature.  The population additive fraction is then

    R2_add = Var(sum_j g_j) / Var(f)

and sweeping ``rho`` from 0 to 1 traces the ceiling from 1 down to whatever the
interaction strength dictates.  ``interaction_strength`` in the emitted sidecar
records the *design* value; ``true_additive_r2`` records the Monte-Carlo value
on the emitted sample.

Correlation between features is controlled separately by ``rho_corr`` because it
is the axis on which PDP and ALE are supposed to differ.  A PDP/ALE comparison
run only on independent features cannot detect any difference and should not be
reported as evidence that none exists -- which is the trap the thesis's
"correlation effects are limited" conclusion fell into.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _gaussian_copula(n: int, p: int, rho_corr: float, rng: np.random.Generator) -> np.ndarray:
    """Correlated standard-normal features with equicorrelation ``rho_corr``."""
    if abs(rho_corr) < 1e-12:
        return rng.standard_normal((n, p))
    Sigma = np.full((p, p), rho_corr, dtype=float)
    np.fill_diagonal(Sigma, 1.0)
    # Nearest-PSD guard for large p and rho.
    w, V = np.linalg.eigh(Sigma)
    w = np.clip(w, 1e-8, None)
    L = V @ np.diag(np.sqrt(w))
    return rng.standard_normal((n, p)) @ L.T


def _additive_part(X: np.ndarray) -> np.ndarray:
    """A deliberately non-linear additive signal, so that a *linear* surrogate
    on the raw features is not already optimal and the curve transform has
    something to contribute."""
    x = X
    parts = [
        1.2 * x[:, 0],
        0.9 * np.sin(1.5 * x[:, 1]),
        0.8 * (x[:, 2] ** 2 - 1.0),
        1.0 * np.tanh(2.0 * x[:, 3]),
    ]
    for j in range(4, x.shape[1]):
        parts.append((0.6 / np.sqrt(j)) * x[:, j])
    return np.sum(parts, axis=0)


def _interaction_part(X: np.ndarray) -> np.ndarray:
    """Pure pairwise interactions, mean-zero in each argument.

    Each term has zero conditional mean given either single feature, so it
    contributes nothing to any partial dependence function -- exactly the
    component an additive explanation must miss.
    """
    x = X
    h = x[:, 0] * x[:, 1]
    if x.shape[1] > 3:
        h = h + x[:, 2] * x[:, 3]
    if x.shape[1] > 5:
        h = h + 0.8 * x[:, 4] * x[:, 5]
    return h


def make_synthetic(
    name: str,
    *,
    n: int = 2000,
    p: int = 8,
    interaction_strength: float = 0.0,
    rho_corr: float = 0.0,
    task: str = "classification",
    noise_sd: float = 0.3,
    n_binary: int = 0,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Generate one synthetic dataset and its sidecar."""
    rng = np.random.default_rng(seed)
    X = _gaussian_copula(n, p, rho_corr, rng)

    # Optionally dichotomise a few columns, to study the binary-feature
    # degeneracy on purpose rather than by accident.
    for j in range(min(n_binary, p)):
        X[:, p - 1 - j] = (X[:, p - 1 - j] > 0).astype(float)

    add = _additive_part(X)
    inter = _interaction_part(X)

    # Standardise both parts so that ``interaction_strength`` is interpretable
    # as a variance share rather than an arbitrary coefficient.
    add = (add - add.mean()) / (add.std() + 1e-12)
    inter = (inter - inter.mean()) / (inter.std() + 1e-12)

    rho = float(np.clip(interaction_strength, 0.0, 1.0))

    # Build the two orthogonal components on the final scale, then sum them, so
    # that the reported additive share is computed from the exact components
    # that went into f rather than reverse-engineered from it.
    raw = np.sqrt(1.0 - rho) * add + np.sqrt(rho) * inter
    k = 2.5 / (raw.std() + 1e-12)  # scale to a usable log-odds range
    comp_add = k * np.sqrt(1.0 - rho) * add
    comp_int = k * np.sqrt(rho) * inter
    f = comp_add + comp_int

    # ``add`` and ``inter`` are orthogonal by construction (each interaction
    # term has zero conditional mean given any single feature), so the additive
    # share of Var(f) is the ratio of the component variances up to Monte-Carlo
    # error.  Computed empirically here rather than assumed.
    true_additive_r2 = float(np.var(comp_add) / (np.var(f) + 1e-12))

    if task == "classification":
        prob = 1.0 / (1.0 + np.exp(-f))
        y = rng.binomial(1, prob)
        target_name = "y"
    else:
        y = f + rng.normal(0.0, noise_sd * f.std(), size=n)
        target_name = "y"

    cols = {f"x{j+1}": X[:, j] for j in range(p)}
    df = pd.DataFrame(cols)
    df[target_name] = y

    binary_cols = [f"x{p-j}" for j in range(min(n_binary, p))]
    spec = {
        "name": name,
        "target": target_name,
        "task": task,
        "numeric": [c for c in df.columns if c != target_name and c not in binary_cols],
        "binary": binary_cols,
        "source": "synthetic",
        "citation": "generated by Code/xaieval/synthetic.py",
        "synthetic": {
            "n": n,
            "p": p,
            "interaction_strength": rho,
            "feature_correlation": rho_corr,
            "n_binary": n_binary,
            "noise_sd": noise_sd,
            "seed": seed,
            "true_additive_r2": round(true_additive_r2, 6),
            "signal_var": float(np.var(f)),
        },
    }
    if task == "classification":
        spec["positive_if"] = ">0"
    return df, spec


#: The *designs*.  Each is instantiated as ``N_REPLICATES`` independent draws
#: of the data-generating process; see :data:`SYNTHETIC_SUITE`.
SYNTHETIC_DESIGNS = [
    # Interaction sweep: the additivity ceiling should fall as tau rises.
    dict(design="syn_inter00", interaction_strength=0.00, rho_corr=0.0),
    dict(design="syn_inter10", interaction_strength=0.10, rho_corr=0.0),
    dict(design="syn_inter25", interaction_strength=0.25, rho_corr=0.0),
    dict(design="syn_inter50", interaction_strength=0.50, rho_corr=0.0),
    dict(design="syn_inter75", interaction_strength=0.75, rho_corr=0.0),
    # Correlation sweep at fixed interaction: the axis on which PDP and ALE
    # are supposed to diverge.
    dict(design="syn_corr30", interaction_strength=0.20, rho_corr=0.3),
    dict(design="syn_corr60", interaction_strength=0.20, rho_corr=0.6),
    dict(design="syn_corr85", interaction_strength=0.20, rho_corr=0.85),
    # A regression variant, so the framework is not classification-only.
    dict(design="syn_regress", interaction_strength=0.25, rho_corr=0.2, task="regression"),
]

#: Independent draws of the DGP per design.
#:
#: With a single draw per design, every repeated split resamples the *same*
#: 2000 rows, so all the uncertainty reported for a synthetic design is
#: split-to-split noise and none of it is draw-to-draw noise.  Any claim about
#: how a quantity varies *across designs* -- the refitting-gain monotonicity in
#: particular -- is then confounded with the single draw each design happened
#: to get, which is what made that gain look non-monotone.  Replicating the
#: draw makes the design the unit of replication and the draw the source of
#: error, which is the correct arrangement.
N_REPLICATES = 10


def _replicate_name(design: str, rep: int) -> str:
    return f"{design}_r{rep:02d}"


def build_suite(n_replicates: int = N_REPLICATES) -> list[dict]:
    """Expand the designs into one entry per independent draw."""
    suite = []
    for d_idx, base in enumerate(SYNTHETIC_DESIGNS):
        for rep in range(1, n_replicates + 1):
            cfg = dict(base)
            design = cfg.pop("design")
            # Seeds are a deterministic function of (design index, replicate)
            # so the suite is reproducible and no two draws collide.
            cfg["seed"] = 100_000 + 1000 * d_idx + rep
            cfg["name"] = _replicate_name(design, rep)
            cfg["design"] = design
            cfg["replicate"] = rep
            suite.append(cfg)
    return suite


#: The suite written by ``make_datasets.py --synthetic``.
SYNTHETIC_SUITE = build_suite()


def write_synthetic_suite(data_dir: Path, n: int = 2000, p: int = 8,
                          n_replicates: int = N_REPLICATES) -> list[Path]:
    """Write the full synthetic suite into ``data_dir`` as CSV + JSON pairs."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for cfg in build_suite(n_replicates):
        cfg = dict(cfg)
        name = cfg.pop("name")
        design = cfg.pop("design")
        replicate = cfg.pop("replicate")
        df, spec = make_synthetic(name, n=n, p=p, **cfg)
        spec["synthetic"]["design"] = design
        spec["synthetic"]["replicate"] = replicate
        csv_path = data_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        (data_dir / f"{name}.json").write_text(json.dumps(spec, indent=2))
        written.append(csv_path)
    return written
