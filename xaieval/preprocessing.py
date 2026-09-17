"""Preprocessing into the numeric design matrix that everything downstream uses.

Two deliberate departures from the thesis pipeline:

1.  **One-hot encoding drops a reference level.**  Encoding all levels of a
    categorical makes the dummy block sum to a constant, which makes the design
    matrix exactly rank deficient.  In the thesis this surfaced as an
    "effective rank 16 out of 18" mystery and destabilised the OLS
    coefficients; it is an artefact of the encoding, not a property of the
    explanation transform.

2.  **Continuous features stay continuous.**  Quantile-binning a continuous
    variable turns it into a set of binary dummies, and a curve-based
    explanation of a binary dummy is an affine function of that dummy -- i.e.
    the explanation transform becomes a no-op.  Binning therefore removes
    exactly the signal the experiment is meant to measure.

The preprocessor is fitted on the training split only and applied to the test
split, so no test information reaches the imputation medians, the category
vocabulary, or the scaler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .datasets import Dataset


@dataclass
class FeatureSpace:
    """Column-level metadata for the encoded design matrix."""

    names: list[str]
    #: True where the encoded column takes at most two distinct training values.
    is_binary: np.ndarray
    #: Name of the raw variable each encoded column came from.
    source: list[str]

    @property
    def p(self) -> int:
        return len(self.names)

    @property
    def binary_fraction(self) -> float:
        return float(np.mean(self.is_binary)) if self.p else 0.0

    @property
    def continuous_idx(self) -> np.ndarray:
        return np.flatnonzero(~self.is_binary)

    @property
    def binary_idx(self) -> np.ndarray:
        return np.flatnonzero(self.is_binary)

    def diagnostics(self) -> dict:
        return {
            "p_encoded": self.p,
            "n_binary_columns": int(self.is_binary.sum()),
            "n_continuous_columns": int((~self.is_binary).sum()),
            "binary_fraction": round(self.binary_fraction, 4),
        }


def _make_encoder() -> OneHotEncoder:
    """OneHotEncoder with a dropped reference level, across sklearn versions."""
    kwargs = dict(drop="first", handle_unknown="infrequent_if_exist")
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(sparse=False, **kwargs)


def build_preprocessor(ds: Dataset) -> ColumnTransformer:
    """Median-impute + scale numerics; mode-impute + drop-first one-hot the rest."""
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    # Binary columns are imputed and passed through as 0/1 -- scaling them would
    # only rename the two values they take.
    binary_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", _make_encoder()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", _make_encoder()),
        ]
    )

    blocks = []
    if ds.numeric:
        blocks.append(("num", numeric_pipe, ds.numeric))
    if ds.binary:
        blocks.append(("bin", binary_pipe, ds.binary))
    if ds.categorical:
        blocks.append(("cat", categorical_pipe, ds.categorical))

    return ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=False)


def _clean_names(raw_names) -> list[str]:
    out = []
    for nm in raw_names:
        nm = str(nm)
        nm = nm.split("__", 1)[-1]  # strip the ColumnTransformer block prefix
        out.append(nm)
    return out


def fit_transform(
    ds: Dataset, X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, FeatureSpace, ColumnTransformer]:
    """Fit on ``X_train``, transform both splits, and describe the columns."""
    pre = build_preprocessor(ds)
    Z_train = np.asarray(pre.fit_transform(X_train), dtype=float)
    Z_test = np.asarray(pre.transform(X_test), dtype=float)

    names = _clean_names(pre.get_feature_names_out())

    # Which raw variable produced each encoded column?
    source: list[str] = []
    for block, _, cols in pre.transformers_:
        if block == "num":
            source.extend(cols)
        elif block in ("bin", "cat"):
            enc = pre.named_transformers_[block].named_steps["encode"]
            for col, cats in zip(cols, enc.categories_):
                n_kept = len(cats) - 1 if enc.drop is not None else len(cats)
                source.extend([col] * max(n_kept, 0))
    if len(source) != len(names):  # defensive: fall back to the encoded name
        source = list(names)

    is_binary = np.array(
        [len(np.unique(Z_train[:, j])) <= 2 for j in range(Z_train.shape[1])], dtype=bool
    )

    space = FeatureSpace(names=names, is_binary=is_binary, source=source)
    return Z_train, Z_test, space, pre


def design_matrix_diagnostics(Z: np.ndarray, tol: float = 1e-10) -> dict:
    """Rank and collinearity of an encoded design matrix.

    Reported so that a rank-deficiency claim in the paper is backed by a number
    that a reader can reproduce, and so that the *expected* rank (full, after
    dropping reference levels) can be checked rather than assumed.
    """
    Zc = Z - Z.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Zc, compute_uv=False)
    rank = int(np.sum(sv > max(Zc.shape) * np.finfo(float).eps * sv[0])) if sv.size else 0
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(Zc, rowvar=False)
    C = np.nan_to_num(C)
    off = C[~np.eye(C.shape[0], dtype=bool)] if C.ndim == 2 and C.shape[0] > 1 else np.array([0.0])
    return {
        "n_columns": int(Z.shape[1]),
        "effective_rank": rank,
        "rank_deficiency": int(Z.shape[1] - rank),
        "condition_number": float(sv[0] / sv[-1]) if sv.size and sv[-1] > tol else np.inf,
        "mean_abs_corr": float(np.mean(np.abs(off))),
        "max_abs_corr": float(np.max(np.abs(off))),
    }
