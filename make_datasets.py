#!/usr/bin/env python3
"""Populate ``Data/`` with datasets.

Three sources:

``--synthetic``
    Write the synthetic suite: an interaction sweep (known additivity ceiling),
    a feature-correlation sweep (the axis on which PDP and ALE should differ),
    a mostly-binary dataset (where the curve transform is provably degenerate),
    and a regression variant.  No download required, and it is what validates
    the measure against ground truth.

``--openml NAME:ID [...]``
    Fetch tabular benchmarks from OpenML and write CSV + sidecar.  Requires
    network access.  A curated default list is provided by ``--openml-suite``.

``--template PATH``
    Write a sidecar JSON template next to an existing CSV, with the roles
    inferred, so it can be edited by hand.

Every real dataset needs a sidecar declaring its target and column roles -- see
``xaieval/datasets.py`` for the schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from xaieval.datasets import CATEGORICAL_MAX_LEVELS  # noqa: E402
from xaieval.synthetic import write_synthetic_suite  # noqa: E402


#: Tabular benchmarks chosen for a high share of *continuous* features, which is
#: what the framework needs in order to have anything to measure: on a binary
#: column the curve transform is affine and therefore a no-op.  Every entry here
#: was checked with ``--inspect``; the binary fraction each one produces is noted.
#: Each entry is ``name: (openml_data_id, task)``.
OPENML_SUITE: dict[str, tuple[int, str]] = {
    # -- classification, continuous-dominated -----------------------------
    "diabetes_pima": (37, "classification"),        # 768 x 8,    binary frac 0.00
    "breast_cancer_wisc": (15, "classification"),   # 699 x 9,    0.00
    "phoneme": (1489, "classification"),            # 5404 x 5,   0.00
    "spambase": (44, "classification"),             # 4601 x 57,  0.00
    "wine_quality_red": (40691, "classification"),  # 1599 x 11,  0.00
    "wine_quality_white": (40498, "classification"),  # 4898 x 11, 0.00
    "magic_telescope": (1120, "classification"),    # 19020 x 10, 0.00
    "qsar_biodeg": (1494, "classification"),        # 1055 x 41,  0.07
    "heart_statlog": (53, "classification"),        # 270 x 13,   0.23 - small n,
                                                    # kept as the closest analogue
                                                    # to the thesis's dataset
    # -- deliberately included negative control ---------------------------
    # 0.88 binary fraction: mostly genuine nominal attributes.  Kept so the
    # paper can show, on real data, that the measure cannot discriminate
    # between explanation methods when the design is dominated by dummies.
    # Report it as such; do not average it into the main comparison.
    "credit_g": (31, "classification"),             # 1000 x 20,  0.88
    # -- regression --------------------------------------------------------
    "concrete": (4353, "regression"),               # 1030 x 8,   0.00
    "cpu_activity": (197, "regression"),            # 8192 x 21,  0.00
    "wind_speed": (503, "regression"),              # 6574 x 14,  0.00
    "abalone": (183, "regression"),                 # 4177 x 8,   0.22
    "boston_housing": (531, "regression"),          # 506 x 13,   0.45
}

#: Considered and rejected, with the reason -- recorded so the choice is
#: reproducible and so nobody re-adds them by accident:
#:
#: ``bank_marketing`` (1461)   45k rows and 0.83 binary fraction: slow *and*
#:                             uninformative.
#: ``energy_efficiency`` (1472) stored on OpenML with its numeric predictors as
#:                             nominal strings, giving 45 encoded columns from 9
#:                             variables; needs a hand-written sidecar.
#: ``climate_crashes`` (1467)  positive rate 0.915 -- too imbalanced to read.
#: ``steel_plates`` (1504)     one exactly collinear column (rank deficiency 1).
OPENML_REJECTED = {
    "bank_marketing": (1461, "45k rows, binary fraction 0.83"),
    "energy_efficiency": (1472, "numeric predictors stored as nominal"),
    "climate_crashes": (1467, "positive rate 0.915"),
    "steel_plates": (1504, "rank-deficient design matrix"),
}


def infer_spec(df: pd.DataFrame, target: str, task: str, name: str) -> dict:
    """Guess the column roles for a sidecar.

    Deliberately the same rule as ``xaieval.datasets._infer_roles``: a numeric
    column stays numeric unless it takes exactly two values.  One-hot encoding a
    numeric measurement destroys the continuous structure the framework
    measures, so the inference errs towards keeping columns numeric and leaves
    genuine nominal codes for the user to declare.
    """
    feats = [c for c in df.columns if c != target]
    numeric, binary, categorical = [], [], []
    for c in feats:
        s = df[c].dropna()
        k = s.nunique()
        if k <= 1:
            continue
        if k == 2:
            binary.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(c)
        else:
            categorical.append(c)
    spec = {
        "name": name,
        "target": target,
        "task": task,
        "numeric": numeric,
        "binary": binary,
        "categorical": categorical,
        "max_missing_frac": 0.3,
    }
    if task == "classification":
        vals = pd.unique(df[target].dropna())
        if len(vals) > 2:
            spec["positive_if"] = ">0"
            spec["_TODO"] = (
                f"target has {len(vals)} levels {list(vals)[:8]}; "
                "edit 'positive_if' to define the positive class"
            )
    return spec


def fetch_openml(name: str, data_id: int, task: str, out_dir: Path) -> Path | None:
    from sklearn.datasets import fetch_openml

    print(f"  fetching {name} (OpenML id {data_id}) ...", end=" ", flush=True)
    try:
        bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    except Exception as exc:
        print(f"FAILED: {exc}")
        return None

    df = bunch.frame.copy()
    target = bunch.target.name if bunch.target is not None else df.columns[-1]
    if target not in df.columns:
        df[target] = bunch.target

    # Binarise a multi-class or non-numeric classification target up front, so
    # the sidecar does not need a hand-written rule.
    if task == "classification":
        y = df[target]
        vals = pd.unique(y.dropna())
        if len(vals) == 2:
            positive = sorted(vals, key=str)[-1]
            df[target] = (y == positive).astype(int)
        else:
            num = pd.to_numeric(y, errors="coerce")
            if num.notna().all():
                df[target] = (num > num.median()).astype(int)
            else:
                counts = y.value_counts()
                df[target] = (y == counts.index[0]).astype(int)
    else:
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df = df[df[target].notna()]

    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    spec = infer_spec(df, target, task, name)
    spec["source"] = f"https://www.openml.org/d/{data_id}"
    spec["citation"] = f"OpenML dataset {data_id}"
    if task == "classification":
        spec.pop("positive_if", None)
        spec.pop("_TODO", None)
    (out_dir / f"{name}.json").write_text(json.dumps(spec, indent=2))
    print(f"ok  ({len(df)} x {df.shape[1] - 1})")
    return csv_path


def inspect_datasets(data_dir: Path) -> int:
    """Pre-flight check: load every dataset and report what the run will see.

    Worth running before committing to a long experiment.  Two columns decide
    whether a dataset is usable at all:

    ``binary_frac``
        Share of *encoded* columns taking at most two values.  A curve-based
        explanation of a two-valued feature is an affine function of that
        feature, so the explanation transform is a no-op on those columns and
        the framework has nothing to measure there.  Above roughly 0.6 the
        dataset cannot meaningfully discriminate between explanation methods.

    ``pos_rate``
        A rate near 0 or 1 means the auto-binarisation of a multi-class target
        probably went wrong; check the sidecar's ``positive_if``.
    """
    from xaieval.datasets import discover_datasets, load_dataset, low_cardinality_numeric
    from xaieval.preprocessing import design_matrix_diagnostics, fit_transform

    paths = discover_datasets(data_dir)
    if not paths:
        print(f"No CSV files in {data_dir}.")
        return 1

    hdr = (f"{'dataset':24s} {'n':>6s} {'raw':>4s} {'enc':>4s} {'bin':>4s} "
           f"{'binary_frac':>11s} {'rankdef':>7s} {'task':>4s} {'pos_rate':>8s}  verdict")
    print(hdr)
    print("-" * len(hdr))

    usable = 0
    suspects: list[str] = []
    for path in paths:
        try:
            ds = load_dataset(path)
            Z, _, space, _ = fit_transform(ds, ds.X, ds.X.iloc[:1])
            diag = design_matrix_diagnostics(Z)
        except Exception as exc:
            print(f"{path.stem:24s} {'':>6s} {'':>4s} {'':>4s} {'':>4s} "
                  f"{'':>11s} {'':>7s} {'':>4s} {'':>8s}  LOAD FAILED: {exc}")
            continue

        bf = space.binary_fraction
        pos = ds.positive_rate
        notes = []
        if bf > 0.6:
            notes.append("mostly binary - cannot discriminate")
        if pos is not None and (pos < 0.05 or pos > 0.95):
            notes.append("target near-degenerate - check positive_if")
        if ds.n < 300:
            notes.append("small n - wide intervals")
        if diag["rank_deficiency"] > 0:
            notes.append("rank deficient - unexpected")
        if ds.n > 20000:
            notes.append("large n - slow (LIME cost is linear in anchors)")
        verdict = "; ".join(notes) if notes else "ok"
        if not notes or all("slow" in x or "small n" in x for x in notes):
            usable += 1

        print(f"{ds.name:24s} {ds.n:6d} {ds.p_raw:4d} {space.p:4d} "
              f"{int(space.is_binary.sum()):4d} {bf:11.2f} {diag['rank_deficiency']:7d} "
              f"{'clf' if ds.task == 'classification' else 'reg':>4s} "
              f"{('--' if pos is None else f'{pos:.3f}'):>8s}  {verdict}")

        lc = low_cardinality_numeric(ds)
        if lc:
            cols = ", ".join(f"{c} ({k})" for c, k in lc[:6])
            more = f", +{len(lc) - 6} more" if len(lc) > 6 else ""
            suspects.append(f"  {ds.name}: {cols}{more}")

    print(f"\n{usable} of {len(paths)} dataset(s) look usable for the main comparison.")

    if suspects:
        print("\nLow-cardinality numeric columns (name and distinct-value count). These are being\n"
              "treated as numeric. If any is genuinely nominal rather than ordinal, list it under\n"
              "'categorical' in that dataset's sidecar -- but note that doing so converts it to\n"
              "binary dummies, on which the explanation transform is a no-op:")
        for s in suspects:
            print(s)

    print("\nSidecars are inferred, not authoritative: open the JSON for anything flagged above,\n"
          "and check 'target' and 'positive_if' even for the ones that are not.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=HERE.parent / "Data")
    p.add_argument("--inspect", action="store_true",
                   help="load every dataset in --data-dir and report the diagnostics "
                        "that decide whether it is usable; runs no experiment")
    p.add_argument("--synthetic", action="store_true", help="write the synthetic suite")
    p.add_argument("--synthetic-n", type=int, default=2000)
    p.add_argument("--synthetic-p", type=int, default=8)
    p.add_argument("--openml-suite", action="store_true", help="fetch the curated OpenML list")
    p.add_argument("--openml", nargs="*", default=None, metavar="NAME:ID[:TASK]",
                   help="fetch specific OpenML datasets")
    p.add_argument("--template", type=Path, default=None,
                   help="write an inferred sidecar next to this CSV")
    p.add_argument("--target", type=str, default=None, help="target column for --template")
    args = p.parse_args(argv)

    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.inspect:
        return inspect_datasets(args.data_dir)

    if args.template:
        df = pd.read_csv(args.template)
        target = args.target or df.columns[-1]
        tvals = df[target].dropna()
        task = ("regression" if pd.api.types.is_numeric_dtype(tvals)
                and tvals.nunique() > CATEGORICAL_MAX_LEVELS else "classification")
        spec = infer_spec(df, target, task, args.template.stem)
        out = args.template.with_suffix(".json")
        out.write_text(json.dumps(spec, indent=2))
        print(f"Wrote sidecar template {out}\n  review 'task', 'positive_if' and the role lists by hand.")
        return 0

    if args.synthetic:
        print(f"Writing synthetic suite to {args.data_dir} "
              f"(n={args.synthetic_n}, p={args.synthetic_p})")
        written = write_synthetic_suite(args.data_dir, n=args.synthetic_n, p=args.synthetic_p)
        for w in written:
            spec = json.loads(w.with_suffix(".json").read_text())["synthetic"]
            print(f"  {w.name:22s} interaction={spec['interaction_strength']:.2f} "
                  f"corr={spec['feature_correlation']:.2f} "
                  f"true_additive_r2={spec['true_additive_r2']:.3f}")

    todo: list[tuple[str, int, str]] = []
    if args.openml_suite:
        todo += [(n, i, t) for n, (i, t) in OPENML_SUITE.items()]
    for entry in args.openml or []:
        parts = entry.split(":")
        name, data_id = parts[0], int(parts[1])
        task = parts[2] if len(parts) > 2 else "classification"
        todo.append((name, data_id, task))

    if todo:
        print(f"\nFetching {len(todo)} dataset(s) from OpenML into {args.data_dir}")
        ok = 0
        for name, data_id, task in todo:
            if fetch_openml(name, data_id, task, args.data_dir):
                ok += 1
        print(f"  {ok}/{len(todo)} fetched")

    if not (args.synthetic or todo):
        p.print_help()
        print("\nNothing to do. Pass --synthetic and/or --openml-suite.")

    csvs = sorted(args.data_dir.glob("*.csv"))
    print(f"\n{args.data_dir} now holds {len(csvs)} dataset(s):")
    for c in csvs:
        print(f"  {c.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
