"""Dataset discovery and loading.

Drop one CSV per dataset into ``Data/``.  Optionally add a JSON sidecar with the
same stem (``Data/heart_disease.csv`` + ``Data/heart_disease.json``) describing
the target and the column roles.  Without a sidecar the loader infers roles,
which is fine for well-formed files but you should write the sidecar for
anything with an ID column, a multi-class target to binarise, or sentinel
values encoded as zeros.

Sidecar schema (all keys optional except ``target``)::

    {
      "name":            "heart_disease",
      "target":          "num",
      "task":            "classification",       // or "regression"
      "positive_if":     ">0",                   // rule to binarise the target
      "drop":            ["id", "dataset"],      // columns to discard outright
      "categorical":     ["cp", "restecg"],      // >2 levels, one-hot encoded
      "binary":          ["sex", "fbs", "exang"],
      "numeric":         ["age", "chol"],        // anything not listed is inferred
      "zero_is_missing": ["chol", "trestbps"],   // 0 recoded to NaN before impute
      "na_values":       ["?", "", "NA"],
      "max_missing_frac": 0.3,                   // drop columns above this
      "source":          "https://...",
      "citation":        "Detrano et al. (1989)"
    }
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_NA_VALUES = ["?", "", " ", "NA", "N/A", "na", "nan", "NaN", "null", "None", "-"]

#: A *non-numeric* column with more than this many distinct values is
#: high-cardinality and gets flagged; it is still one-hot encoded, which may not
#: be what you want.
CATEGORICAL_MAX_LEVELS = 12

#: A numeric column with at most this many distinct values is *flagged* as
#: possibly nominal, but is still treated as numeric unless the sidecar says
#: otherwise.  See :func:`_infer_roles` for why the default leans this way.
LOW_CARDINALITY_NUMERIC = 12


@dataclass
class Dataset:
    """A loaded, cleaned but *not yet encoded* dataset."""

    name: str
    X: pd.DataFrame
    y: np.ndarray
    task: str  # "classification" | "regression"
    numeric: list[str]
    binary: list[str]
    categorical: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.X)

    @property
    def p_raw(self) -> int:
        return self.X.shape[1]

    @property
    def positive_rate(self) -> float | None:
        if self.task != "classification":
            return None
        return float(np.mean(self.y))

    def summary_row(self) -> dict[str, Any]:
        return {
            "dataset": self.name,
            "n": self.n,
            "p_raw": self.p_raw,
            "n_numeric": len(self.numeric),
            "n_binary": len(self.binary),
            "n_categorical": len(self.categorical),
            "task": self.task,
            "positive_rate": self.positive_rate,
            "dropped_missing": ", ".join(self.meta.get("dropped_high_missing", [])) or "--",
            "source": self.meta.get("source", ""),
        }


# --------------------------------------------------------------------------
# Target binarisation rules
# --------------------------------------------------------------------------


def _apply_positive_rule(values: pd.Series, rule: str | None) -> np.ndarray:
    """Turn a raw target column into a 0/1 array using ``positive_if``.

    Supported rules: ``">0"``, ``">=2"``, ``"==1"``, ``"!=0"``, ``"in:a,b"``,
    or ``None`` (already binary / two-valued).
    """
    if rule is None:
        uniq = pd.unique(values.dropna())
        if len(uniq) != 2:
            raise ValueError(
                f"target has {len(uniq)} distinct values; supply 'positive_if' "
                f"in the sidecar to binarise it (values seen: {uniq[:10]})"
            )
        positive = sorted(uniq, key=str)[-1]
        return (values == positive).to_numpy().astype(int)

    rule = rule.strip()
    if rule.startswith("in:"):
        wanted = {s.strip() for s in rule[3:].split(",")}
        return values.astype(str).isin(wanted).to_numpy().astype(int)

    m = re.fullmatch(r"(>=|<=|==|!=|>|<)\s*(-?[\d.]+)", rule)
    if not m:
        raise ValueError(f"cannot parse positive_if rule {rule!r}")
    op, num = m.group(1), float(m.group(2))
    v = pd.to_numeric(values, errors="coerce")
    table = {
        ">": v > num,
        "<": v < num,
        ">=": v >= num,
        "<=": v <= num,
        "==": v == num,
        "!=": v != num,
    }
    return table[op].to_numpy().astype(int)


# --------------------------------------------------------------------------
# Role inference
# --------------------------------------------------------------------------


def _infer_roles(X: pd.DataFrame, spec: dict) -> tuple[list[str], list[str], list[str]]:
    """Split columns into numeric / binary / categorical.

    Explicit sidecar lists win; anything unlisted is inferred from dtype.

    **A numeric column stays numeric unless it takes exactly two values.**  This
    matters more than it looks, and the rule deliberately leans one way.

    A curve-based explanation of a two-valued feature is an *affine* function of
    that feature, so one-hot encoding a variable destroys precisely the
    structure this framework measures.  An earlier heuristic here treated any
    numeric column with few distinct values as categorical, and on the
    Wisconsin breast-cancer data -- whose measurements are integer 1-10 scales
    -- that expanded 9 variables into 80 dummies and rendered the dataset
    useless for the comparison, silently.

    The two failure directions are not symmetric:

    * treating a genuine ordinal/continuous variable as nominal *destroys the
      measurement*, and is easy to miss;
    * treating a genuine nominal code as numeric merely imposes a spurious
      ordering -- a modelling imperfection, not an invalidation.

    So the default takes the second risk, and low-cardinality numeric columns
    are *flagged* by ``make_datasets.py --inspect`` for the user to declare as
    ``categorical`` in the sidecar when they really are nominal.
    """
    declared_num = list(spec.get("numeric", []))
    declared_bin = list(spec.get("binary", []))
    declared_cat = list(spec.get("categorical", []))
    declared = set(declared_num) | set(declared_bin) | set(declared_cat)

    numeric, binary, categorical = list(declared_num), list(declared_bin), list(declared_cat)

    for col in X.columns:
        if col in declared:
            continue
        s = X[col].dropna()
        n_levels = s.nunique()
        if n_levels <= 1:
            continue  # constant column; dropped by the caller
        if n_levels == 2:
            binary.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        else:
            categorical.append(col)

    keep = set(X.columns)
    return (
        [c for c in numeric if c in keep],
        [c for c in binary if c in keep],
        [c for c in categorical if c in keep],
    )


def low_cardinality_numeric(ds: "Dataset") -> list[tuple[str, int]]:
    """Numeric columns with few distinct values -- candidates for ``categorical``.

    Reported by ``--inspect`` so the inference rule above can be overridden
    deliberately rather than discovered by accident.
    """
    out = []
    for col in ds.numeric:
        k = int(ds.X[col].dropna().nunique())
        if k <= LOW_CARDINALITY_NUMERIC:
            out.append((col, k))
    return sorted(out, key=lambda kv: kv[1])


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_dataset(csv_path: Path) -> Dataset:
    """Load one dataset from ``csv_path`` plus its optional JSON sidecar."""
    csv_path = Path(csv_path)
    spec_path = csv_path.with_suffix(".json")
    spec: dict = json.loads(spec_path.read_text()) if spec_path.exists() else {}

    name = spec.get("name", csv_path.stem)
    na_values = spec.get("na_values", DEFAULT_NA_VALUES)
    df = pd.read_csv(csv_path, na_values=na_values, keep_default_na=True)
    df.columns = [str(c).strip() for c in df.columns]

    # -- target ----------------------------------------------------------
    target = spec.get("target")
    if target is None:
        # Convention: last column is the target.
        target = df.columns[-1]
        warnings.warn(f"[{name}] no 'target' in sidecar; using last column {target!r}")
    if target not in df.columns:
        raise KeyError(f"[{name}] target column {target!r} not found in {list(df.columns)}")

    task = spec.get("task")
    if task is None:
        tvals = df[target].dropna()
        task = (
            "regression"
            if pd.api.types.is_numeric_dtype(tvals) and tvals.nunique() > CATEGORICAL_MAX_LEVELS
            else "classification"
        )

    if task == "classification":
        y = _apply_positive_rule(df[target], spec.get("positive_if"))
    else:
        y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)

    X = df.drop(columns=[target])

    # -- explicit drops ---------------------------------------------------
    drops = [c for c in spec.get("drop", []) if c in X.columns]
    X = X.drop(columns=drops)

    meta: dict[str, Any] = {
        "source": spec.get("source", ""),
        "citation": spec.get("citation", ""),
        "target_column": target,
        "dropped_explicit": drops,
    }

    # -- sentinel zeros ---------------------------------------------------
    zero_missing = [c for c in spec.get("zero_is_missing", []) if c in X.columns]
    for col in zero_missing:
        X[col] = pd.to_numeric(X[col], errors="coerce").replace(0.0, np.nan)
    meta["zero_is_missing"] = zero_missing

    # -- rows with a missing target --------------------------------------
    ok = ~pd.isna(y) if task == "regression" else np.ones(len(y), dtype=bool)
    if not ok.all():
        X, y = X.loc[ok].reset_index(drop=True), y[ok]

    # -- high-missingness columns ----------------------------------------
    max_missing = float(spec.get("max_missing_frac", 0.3))
    frac = X.isna().mean()
    high = sorted(frac[frac > max_missing].index.tolist())
    if high:
        X = X.drop(columns=high)
    meta["dropped_high_missing"] = high
    meta["missing_fraction"] = {c: round(float(frac[c]), 4) for c in high}

    # -- constant / all-missing columns ----------------------------------
    constant = [c for c in X.columns if X[c].dropna().nunique() <= 1]
    if constant:
        X = X.drop(columns=constant)
    meta["dropped_constant"] = constant

    # -- duplicate rows ---------------------------------------------------
    dup = X.duplicated()
    meta["n_duplicate_rows"] = int(dup.sum())

    numeric, binary, categorical = _infer_roles(X, spec)
    X = X[numeric + binary + categorical]

    if len(X.columns) == 0:
        raise ValueError(f"[{name}] no usable feature columns remain after cleaning")

    return Dataset(
        name=name,
        X=X.reset_index(drop=True),
        y=np.asarray(y),
        task=task,
        numeric=numeric,
        binary=binary,
        categorical=categorical,
        meta=meta,
    )


#: Datasets present in ``Data/`` but deliberately kept out of the study.
#:
#: ``magic_telescope`` -- strong feature dependence drives the PDP
#:   reconstruction to -1.33 on the RBF SVM, i.e.\ far worse than predicting the
#:   model's mean.  Every pooled number it enters is dominated by that one cell,
#:   so it distorts far more than it informs.
#: ``credit_g``, ``syn_binary6`` -- more than 60% of their encoded columns are
#:   binary.  On a two-valued column a curve is an affine function of the raw
#:   dummy, so these datasets cannot discriminate between explanation methods
#:   at all; they were previously reported as a separate stratum, which gave a
#:   two-dataset "stratum" that could carry no test.
#:
#: The files stay on disk; only the analysis excludes them.
EXCLUDED_DATASETS = ("magic_telescope", "credit_g", "syn_binary6")


def discover_datasets(data_dir: Path, only: list[str] | None = None,
                      exclude: tuple[str, ...] = EXCLUDED_DATASETS) -> list[Path]:
    """Return the CSV paths in ``data_dir``, optionally filtered by stem."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"data directory {data_dir} does not exist. Put one CSV per dataset in it, "
            f"or run `python make_datasets.py --synthetic` to generate the synthetic suite."
        )
    paths = sorted(p for p in data_dir.glob("*.csv") if not p.name.startswith("_"))
    if exclude and not only:
        drop = set(exclude)
        paths = [p for p in paths if p.stem not in drop]
    if only:
        wanted = set(only)
        paths = [p for p in paths if p.stem in wanted or json_name(p) in wanted]
        missing = wanted - {p.stem for p in paths} - {json_name(p) for p in paths}
        if missing:
            raise FileNotFoundError(f"requested datasets not found in {data_dir}: {sorted(missing)}")
    return paths


def json_name(csv_path: Path) -> str:
    spec_path = csv_path.with_suffix(".json")
    if spec_path.exists():
        try:
            return json.loads(spec_path.read_text()).get("name", csv_path.stem)
        except json.JSONDecodeError:
            pass
    return csv_path.stem
