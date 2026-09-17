"""Turn the raw result table into paper-ready LaTeX tables and figures.

Reads only ``Results/raw/*.csv``, so output can be regenerated without
re-running any experiment.  Tables are ``booktabs`` fragments meant to be
``\\input`` from the manuscript; figures are written as both PNG (300 dpi, for
drafts and slides) and PDF (vector, for the manuscript).

Presentation choices that differ from the thesis, and why:

* **Every number carries a confidence interval.**  The thesis reported single
  split point estimates and then discussed differences of 0.005 in AUC.  With
  repeated splits those differences are visibly inside the noise.
* **Fidelity is reported next to the additivity ceiling and the attainment
  ratio.**  A raw fidelity number cannot be interpreted without knowing how
  much additive structure was available to capture.
* **Null controls sit in the same table as the methods.**  If a null explanation
  scores close to a real one, that is the headline, and it should not be
  hidden in an appendix.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .config import (  # noqa: E402
    GREY_DARK,
    GREY_LIGHT,
    GREY_MID,
    INK,
    METHOD_COLOURS,
    METHOD_LINESTYLES,
    METHOD_MARKERS,
)
from .scoring import dataset_level_comparison, mean_ci, paired_difference  # noqa: E402

METHODS = ["PDP", "ALE", "SHAP", "LIME"]

#: How each model family is written wherever a reader sees it.  The raw keys in
#: the results table are lower-case with underscores; these are the forms used
#: in the manuscript, so the two must not drift apart.
MODEL_LABEL = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "mlp": "MLP",
    "svm_rbf": "SVM RBF",
}


def model_label(name: str) -> str:
    """Display name for a model family."""
    return MODEL_LABEL.get(str(name), str(name).replace("_", " "))


#: Datasets whose encoded design is more than this fraction binary are reported
#: in a separate stratum rather than pooled into the headline result.
#:
#: This is not a quality filter and nothing is discarded.  On a two-valued
#: column the curve transform is affine, so a categorical-dominated dataset
#: cannot distinguish between explanation methods -- pooling it into the main
#: average dilutes a real effect with cells that are incapable of showing one.
#: Reporting the two strata side by side is also the cleaner claim: the second
#: stratum is the empirical demonstration that the degeneracy is real.
BINARY_FRACTION_LIMIT = 0.6

PLOT_STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": GREY_MID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": GREY_DARK,
    "ytick.color": GREY_DARK,
    "axes.grid": True,
    "grid.color": GREY_LIGHT,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "legend.frameon": False,
}


# --------------------------------------------------------------------------
# LaTeX helpers
# --------------------------------------------------------------------------


def tex_escape(s: str) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("^", r"\^{}"),
                 ("~", r"\textasciitilde{}")]:
        s = s.replace(a, b)
    return s


def fmt(v, nd: int = 3, dash: str = "--") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    return f"{v:.{nd}f}"


def fmt_ci(mean, lo, hi, nd: int = 3) -> str:
    """``0.912 (0.905, 0.919)`` -- point estimate with its interval."""
    if mean is None or not np.isfinite(mean):
        return "--"
    if lo is None or not np.isfinite(lo):
        return fmt(mean, nd)
    return f"{mean:.{nd}f} \\,({lo:.{nd}f}, {hi:.{nd}f})"


def _latex_safe(s: str) -> str:
    """Escape a string unless it is deliberately carrying LaTeX.

    Dataset and model names routinely contain underscores (``random_forest``,
    ``heart_disease``); emitted raw, those are a ``Missing $ inserted`` error
    that breaks the manuscript build.  Cells we author ourselves contain maths
    (``$R^2$``) or spacing macros (``\\,``) and must pass through untouched, so
    the presence of ``$`` or a backslash is taken as "author-controlled".
    """
    s = str(s)
    if "$" in s or "\\" in s:
        return s
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                 ("^", r"\^{}"), ("~", r"\textasciitilde{}")):
        s = s.replace(a, b)
    return s


def _cell(v) -> str:
    """Render one cell.  Floats get three decimals, missing values an en-dash."""
    if v is None:
        return "--"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "--"
        return f"{v:.3f}"
    if isinstance(v, (np.floating,)):
        return "--" if not np.isfinite(float(v)) else f"{float(v):.3f}"
    if isinstance(v, (np.integer,)):
        return str(int(v))
    s = str(v)
    return "--" if s in ("nan", "None", "<NA>", "") else _latex_safe(s)


def _tabular(df: pd.DataFrame, colfmt: str, rule_after: Sequence[int] | None = None) -> str:
    """Emit a booktabs tabular directly.

    Written by hand rather than via ``DataFrame.to_latex`` so that the package
    has no jinja2 dependency and so that the maths in the headers survives
    untouched.

    ``rule_after`` gives zero-based row positions after which a ``\\midrule`` is
    inserted, used to separate blocks of rows that belong together.
    """
    cuts = set(rule_after or ())
    header = " & ".join(_latex_safe(c) for c in df.columns) + " \\\\"
    body = []
    for i, row in enumerate(df.itertuples(index=False, name=None)):
        body.append(" & ".join(_cell(v) for v in row) + " \\\\")
        if i in cuts:
            body.append("\\midrule")
    return "\n".join(
        [f"\\begin{{tabular}}{{{colfmt}}}", "\\toprule", header, "\\midrule", *body,
         "\\bottomrule", "\\end{tabular}"]
    )


#: Beyond this many body rows a table is emitted as a page-breaking longtable
#: rather than a float.  The text block is ~697pt and a \small row ~12pt, so a
#: float much past 45 rows cannot fit on one page.
LONGTABLE_ROWS = 45


def _longtable(df: pd.DataFrame, colfmt: str, caption: str, label: str,
               rule_after: Sequence[int] | None = None) -> str:
    """Emit a booktabs longtable, repeating the header on each page."""
    cuts = set(rule_after or ())
    header = " & ".join(_latex_safe(c) for c in df.columns) + " \\\\"
    ncol = df.shape[1]
    body = []
    for i, row in enumerate(df.itertuples(index=False, name=None)):
        body.append(" & ".join(_cell(v) for v in row) + " \\\\")
        if i in cuts:
            body.append("\\midrule")
    return "\n".join([
        f"\\begin{{longtable}}{{{colfmt}}}",
        f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
        "\\toprule", header, "\\midrule", "\\endfirsthead",
        f"\\multicolumn{{{ncol}}}{{l}}{{\\footnotesize\\itshape "
        f"Table \\ref{{{label}}} continued from previous page}}\\\\",
        "\\toprule", header, "\\midrule", "\\endhead",
        f"\\midrule\\multicolumn{{{ncol}}}{{r}}{{\\footnotesize\\itshape "
        f"continued on next page}}\\\\", "\\endfoot",
        "\\bottomrule", "\\endlastfoot",
        *body,
        "\\end{longtable}",
    ])


def write_table(
    path: Path,
    df: pd.DataFrame,
    caption: str,
    label: str,
    *,
    column_format: str | None = None,
    note: str | None = None,
    escape: bool = True,
    rule_after: Sequence[int] | None = None,
    tabcolsep: str | None = None,
    longtable: bool | None = None,
) -> None:
    """Write a booktabs table fragment.

    ``longtable`` emits a page-breaking table instead of a float; left at None
    it switches on automatically past ``LONGTABLE_ROWS``, because a float
    taller than the text block silently overflows the page bottom.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = df.copy()
    if escape:
        body.columns = [tex_escape(c) for c in body.columns]
        for c in body.columns:
            if body[c].dtype == object:
                body[c] = body[c].map(lambda x: tex_escape(x) if isinstance(x, str) else x)
    else:
        body.columns = [str(c) for c in body.columns]

    ncol = body.shape[1]
    colfmt = column_format or ("l" + "r" * (ncol - 1))
    if longtable is None:
        longtable = len(body) > LONGTABLE_ROWS

    head = ["% Generated by Code/xaieval/report.py -- do not edit by hand."]
    if longtable:
        # A longtable is not a float, so it breaks across pages instead of
        # overflowing the bottom of one.  Grouped so \small and \tabcolsep
        # do not leak into the surrounding text.
        head += ["{\\small"]
        if tabcolsep:
            head.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
        lines = head + [
            _longtable(body, colfmt, caption, label, rule_after),
        ]
        if note:
            lines.append(f"\\noindent\\begin{{minipage}}{{\\linewidth}}\\footnotesize"
                         f"\\vspace{{2pt}}{note}\\end{{minipage}}")
        lines.append("}")
    else:
        head += ["\\begin{table}[htbp]", "\\centering", "\\small"]
        if tabcolsep:
            head.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
        lines = head + [
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            _tabular(body, colfmt, rule_after),
        ]
        if note:
            lines.append(f"\\begin{{minipage}}{{\\linewidth}}\\footnotesize\\vspace{{2pt}}{note}\\end{{minipage}}")
        lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Aggregation helpers
# --------------------------------------------------------------------------


def split_strata(diagnostics: pd.DataFrame, res: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition datasets into (continuous-dominated, categorical-dominated).

    Falls back to putting everything in the primary stratum when the
    diagnostics file is missing, so an old ``Results/raw`` still reports.
    """
    all_names = sorted(res["dataset"].astype(str).unique()) if "dataset" in res else []
    if diagnostics is None or diagnostics.empty or "binary_fraction" not in diagnostics.columns:
        return all_names, []
    dm = diagnostics[diagnostics.get("kind") == "design_matrix"].drop_duplicates("dataset")
    frac = dict(zip(dm["dataset"].astype(str), dm["binary_fraction"].astype(float)))
    primary = [d for d in all_names if frac.get(d, 0.0) <= BINARY_FRACTION_LIMIT]
    flagged = [d for d in all_names if d not in set(primary)]
    return primary, flagged


def _stratum_note(flagged: list[str]) -> str:
    if not flagged:
        return ""
    return (
        f" Datasets whose encoded design is more than "
        f"{int(BINARY_FRACTION_LIMIT * 100)}\\% binary are excluded here and reported "
        f"separately ({', '.join(tex_escape(d) for d in flagged)}): on a two-valued column "
        f"the curve transform is affine, so those datasets cannot discriminate between "
        f"explanation methods and pooling them would dilute the comparison."
    )


def _sel(df: pd.DataFrame, **kw) -> pd.DataFrame:
    out = df
    for k, v in kw.items():
        if k not in out.columns:
            return out.iloc[0:0]
        out = out[out[k] == v] if not isinstance(v, (list, tuple, set)) else out[out[k].isin(list(v))]
    return out


def _per_repeat(df: pd.DataFrame, group: list[str], metric: str) -> pd.DataFrame:
    """Collapse to one value per (group..., repeat), averaging over control draws."""
    d = df[df["metric"] == metric]
    keys = group + ["repeat"]
    keys = [k for k in keys if k in d.columns]
    if d.empty:
        return d
    return d.groupby(keys, dropna=False, observed=True)["value"].mean().reset_index()


def _ci_by(df: pd.DataFrame, group: list[str], metric: str, alpha: float = 0.05) -> pd.DataFrame:
    """Mean and t-CI across repeats, for each group."""
    per = _per_repeat(df, group, metric)
    if per.empty:
        return pd.DataFrame(columns=group + ["mean", "lo", "hi", "n"])
    if not group:
        # Pandas refuses to group by nothing; a single all-rows group is meant.
        m, lo, hi, n = mean_ci(per["value"].to_numpy(), alpha)
        return pd.DataFrame([{"mean": m, "lo": lo, "hi": hi, "n": n}])
    rows = []
    for keys, g in per.groupby(group, dropna=False, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        m, lo, hi, n = mean_ci(g["value"].to_numpy(), alpha)
        rows.append({**dict(zip(group, keys)), "mean": m, "lo": lo, "hi": hi, "n": n})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def table_datasets(datasets: pd.DataFrame, diagnostics: pd.DataFrame, out: Path) -> None:
    d = datasets.copy()
    if diagnostics is not None and not diagnostics.empty:
        dm = diagnostics[diagnostics.get("kind") == "design_matrix"]
        cols = ["dataset", "p_encoded", "n_binary_columns", "binary_fraction",
                "effective_rank", "rank_deficiency", "max_abs_corr"]
        dm = dm[[c for c in cols if c in dm.columns]].drop_duplicates("dataset")
        d = d.merge(dm, on="dataset", how="left")

    # Real datasets only.  The synthetic rows carried no information -- every
    # design has the same n, p and binary fraction by construction -- and the
    # naming scheme is introduced in the text where each design is used.
    d = d.copy()
    d = d[~d["dataset"].astype(str).str.startswith("syn")]
    d = d.sort_values("dataset").reset_index(drop=True)
    rule_after = None

    # Kept deliberately narrow: "Binary cols" is recoverable from the fraction
    # and the encoded width, and "Rank def." is zero on every row, so both live
    # in the caption instead of costing text width.
    show = pd.DataFrame({
        "Dataset": d["dataset"],
        "$n$": d["n"],
        "$p_{\\mathrm{raw}}$": d["p_raw"],
        "$p$": d.get("p_encoded"),
        "Bin.\\ frac.": d.get("binary_fraction").map(lambda v: fmt(v, 2)) if "binary_fraction" in d else None,
        "Task": d["task"].map({"classification": "clf", "regression": "reg"}),
        "Pos.\\ rate": d["positive_rate"].map(lambda v: fmt(v, 3)),
    })
    show = show.dropna(axis=1, how="all")

    write_table(
        out / "tab_datasets.tex",
        show,
        caption=(
            "The real datasets used in the study. $n$ is the number of rows, "
            "$p_{\\mathrm{raw}}$ the number of variables before encoding, $p$ the width of "
            "the one-hot encoded design matrix, and \\emph{Bin.\\ frac.} the share of "
            "encoded columns taking at most two values. \\emph{Task} is clf for "
            "classification and reg for regression, and \\emph{Pos.\\ rate} the marginal "
            "rate of the positive class, shown as a dash for regression."
        ),
        label="tab:datasets",
        escape=False,
        rule_after=rule_after,
    )


def table_blackbox(res: pd.DataFrame, out: Path) -> None:
    metric = "auc" if (res["metric"] == "auc").any() else "r2"
    bb = _sel(res, family="blackbox", target="y", split="test")
    ci = _ci_by(bb, ["dataset", "model"], metric)
    if ci.empty:
        return
    ci = ci.assign(model=ci["model"].map(model_label))
    piv = ci.pivot(index="dataset", columns="model", values="mean")
    lo = ci.pivot(index="dataset", columns="model", values="lo")
    hi = ci.pivot(index="dataset", columns="model", values="hi")
    show = pd.DataFrame(index=piv.index)
    for c in piv.columns:
        show[c] = [fmt_ci(piv.loc[i, c], lo.loc[i, c], hi.loc[i, c]) for i in piv.index]
    show = show.reset_index().rename(columns={"dataset": "Dataset"})

    write_table(
        out / "tab_blackbox.tex",
        show,
        caption=(
            f"Held-out black-box performance (test {'ROC-AUC' if metric=='auc' else '$R^2$'}), "
            "mean over repeated stratified splits with a 95\\% $t$ interval."
        ),
        label="tab:blackbox",
        escape=False,
    )


def table_fidelity(res: pd.DataFrame, out: Path, target: str = "fhat",
                   datasets: list[str] | None = None, suffix: str = "",
                   stratum_note: str = "") -> None:
    """Main results: explanation-derived predictor performance, pooled over datasets.

    ``datasets`` restricts the pool to one stratum; see :data:`BINARY_FRACTION_LIMIT`.
    """
    sub = _sel(res, target=target, split="test")
    if datasets is not None:
        sub = sub[sub["dataset"].isin(datasets)]
    sub = sub[sub["family"].isin(METHODS)]
    sub = sub[sub["variant"].str.startswith("curve-") | (sub["variant"] == "local-idw")]
    if sub.empty:
        return

    # Against the black box, R^2 is the natural headline: both sides live on the
    # same scale.  Against a 0/1 outcome it is not, because the unit-coefficient
    # construction deliberately does not rescale the curves -- so lead with the
    # rank metrics, which are invariant to that and remain comparable.
    if target == "fhat":
        primary, primary_label = "r2", "$R^2$ vs.\\ $\\hat f$"
    else:
        primary = "auc" if (sub["metric"] == "auc").any() else "r2"
        primary_label = "ROC-AUC" if primary == "auc" else "$R^2$ vs.\\ $y$"

    keys = ["family", "variant", "model"]
    ci = _ci_by(sub, keys, primary)
    sp = _ci_by(sub, keys, "spearman")
    if ci.empty:
        return
    ci = ci.merge(sp, on=keys, how="left", suffixes=("", "_sp"))

    ci["Construction"] = ci["variant"].str.replace("curve-", "additive, ", regex=False)
    ci = ci.sort_values(["model", "family", "Construction"])

    show = pd.DataFrame({
        "Model": ci["model"].map(model_label),
        "Method": ci["family"],
        "Construction": ci["Construction"],
        primary_label: [fmt_ci(m, l, h) for m, l, h in zip(ci["mean"], ci["lo"], ci["hi"])],
        "Spearman": [fmt_ci(m, l, h) for m, l, h in zip(ci["mean_sp"], ci["lo_sp"], ci["hi_sp"])],
        "$k$": ci["n"],
    })

    if target == "fhat":
        tail = (
            "\\emph{additive, unit} fixes all coefficients at 1 and is the $L^2$-optimal "
            "additive projection under feature independence; \\emph{additive, ols} and "
            "\\emph{additive, ridge} refit them. Spearman is reported alongside $R^2$ because "
            "it is invariant to the monotone rescaling that separates those variants."
        )
    else:
        tail = (
            "Rank metrics lead here: the unit-coefficient construction deliberately leaves the "
            "curves on the black box's own scale, so an $R^2$ against a 0/1 outcome would "
            "penalise it for an offset rather than for a loss of information. The fitted "
            "variants are free to rescale and are directly comparable on either metric."
        )

    tgt = "the black box's own score" if target == "fhat" else "the original target $y$"
    write_table(
        out / f"tab_fidelity_{target}{suffix}.tex",
        show,
        caption=(
            f"Out-of-sample performance of each explanation-derived predictor against {tgt}, "
            "pooled over datasets. Mean over repeated splits with a 95\\% $t$ interval; "
            f"$k$ is the number of (dataset, split) units. {tail}{stratum_note}"
        ),
        label=f"tab:fidelity-{target}{suffix.replace('_', '-')}",
        escape=False,
    )


#: Below this reconstruction quality the attainment ratio is not reported.
#: The ratio has the reconstruction R^2 in its denominator, so once that
#: approaches zero the quotient is dominated by noise and is not interpretable.
ATTAINMENT_MIN_DENOM = 0.15


def table_fidelity_compact(res: pd.DataFrame, out: Path, datasets: list[str] | None = None,
                           stratum_note: str = "") -> None:
    """Main-text fidelity table: one row per construction, one column per method.

    Averaged over the four model families.  The per-model breakdown is four
    times the size and readers have to do the averaging in their heads to reach
    the comparison the paper actually makes; the long form is kept as
    supplementary material for anyone who wants it.
    """
    sub = _sel(res, target="fhat", split="test")
    if datasets is not None:
        sub = sub[sub["dataset"].isin(datasets)]
    sub = sub[sub["family"].isin(METHODS)]
    sub = sub[sub["variant"].str.startswith("curve-") | (sub["variant"] == "local-idw")]
    if sub.empty:
        return

    ci = _ci_by(sub, ["family", "variant"], "r2")
    if ci.empty:
        return
    ci["cell"] = [
        "--" if not np.isfinite(m) else
        (f"{m:.3f}" if not np.isfinite(l) else f"{m:.3f} $\\pm$ {max(m - l, 0):.3f}")
        for m, l in zip(ci["mean"], ci["lo"])
    ]
    piv = ci.pivot_table(index="variant", columns="family", values="cell", aggfunc="first")

    label = {"curve-unit": "additive, unit", "curve-ols": "additive, ols",
             "curve-ridge": "additive, ridge", "local-idw": "local-IDW"}
    methods = [m for m in METHODS if m in piv.columns]

    rows = []
    for v in ("curve-unit", "curve-ols", "curve-ridge", "local-idw"):
        if v not in piv.index:
            continue
        r = {"Construction": label[v]}
        for m in methods:
            val = piv.loc[v, m]
            r[m] = "--" if (val is None or (isinstance(val, float) and not np.isfinite(val))) else val
        rows.append(r)

    write_table(
        out / "tab_fidelity_compact.tex",
        pd.DataFrame(rows),
        caption=(
            "Test $R^2$ of each explanation-derived predictor against the black box's own "
            "predictions. Rows are the constructions of Eqs.~\\ref{eq:unit}--\\ref{eq:ridge} "
            "--- $g_{\\mathbf 1}$, $g_{\\hat\\alpha}$ and $g_{\\hat\\alpha_\\lambda}$ --- plus "
            "LIME's native local-IDW variant; "
            "columns are the explanation methods. Entries are test $R^2$, averaged over "
            "datasets and over the four model families, with the half-width of a 95\\% $t$ "
            "interval. A dash marks a construction a method does not have."
            + stratum_note
        ),
        column_format="l" + "r" * len(methods),
        label="tab:fidelity-compact",
        escape=False,
    )


def table_aggregator_control(res: pd.DataFrame, out: Path, datasets: list[str] | None = None) -> None:
    """Is the ranking a property of the explanations, or of the aggregator?

    ``CondMean`` feeds the black box's *own predictions* through the identical
    binning used for SHAP and LIME, with no explanation involved.  If it matches
    an explanation-derived predictor, that predictor's score is attributable to
    the aggregation step rather than to the explanation, and the shared-
    construction fairness argument fails.  ``SHAP-cond`` is conditional
    (tree_path_dependent) TreeSHAP against the marginal default, isolating the
    estimator choice on tree models.
    """
    fams = ["PDP", "ALE", "SHAP", "SHAP-cond", "CondMean"]
    # Reported under the *refitted* construction.  Under unit coefficients the
    # control sums p conditional means that each carry the whole marginal
    # signal, so it over-counts by roughly a factor of p and lands near -11 --
    # a scaling artefact that tells us nothing about whether the aggregator
    # explains the ranking.  Refitting removes the over-counting and makes the
    # comparison the one worth reporting.
    sub = _sel(res, target="fhat", split="test", variant="curve-ols")
    if datasets is not None:
        sub = sub[sub["dataset"].isin(datasets)]
    sub = sub[sub["family"].isin(fams) & (sub["metric"] == "r2")]
    if sub.empty:
        return
    ci = _ci_by(sub, ["family", "model"], "r2")
    if ci.empty:
        return
    piv = ci.pivot_table(index="family", columns="model", values="mean")
    rows = []
    label = {"CondMean": "E[f | x_j], binned (no explanation)",
             "SHAP-cond": "SHAP, conditional TreeSHAP"}
    for f in fams:
        if f not in piv.index:
            continue
        r = {"Curve source": label.get(f, f)}
        for m in piv.columns:
            v = piv.loc[f, m]
            r[model_label(m)] = fmt(v, 3) if np.isfinite(v) else "--"
        rows.append(r)
    write_table(
        out / "tab_aggregator_control.tex",
        pd.DataFrame(rows),
        caption=(
            "Test $R^2$ against the black box under the refitted construction "
            "$g_{\\hat\\alpha}$ of "
            "Eq.~\\ref{eq:ols}, averaged over the real datasets. Rows are curve sources, "
            "columns model families. \\emph{SHAP, conditional TreeSHAP} replaces the marginal "
            "\\texttt{interventional} estimator with \\texttt{tree\\_path\\_dependent} and is "
            "defined only on the two tree families. \\emph{E[f\\,$\\vert$\\,$x_j$], binned} "
            "applies the binning of Eq.~\\ref{eq:binned} to the black box's own predictions, "
            "with no explanation in the pipeline. Dashes mark undefined combinations."
        ),
        label="tab:aggregator-control",
        escape=False,
    )


def table_ceiling(res: pd.DataFrame, out: Path,
                  datasets: list[str] | None = None) -> None:
    """Additive PDP reconstruction, and how much of it each method recovers.

    The ratio is computed as a **ratio of means** -- mean achieved $R^2$ over
    mean reconstruction $R^2$, both averaged across repeats -- not as the mean
    of per-repeat ratios.  The latter is what the runner records per repeat, and
    averaging it is badly behaved: the denominator is a noisy quantity that can
    approach zero, so individual ratios blow up (we measured a cell whose mean
    was 14.3 against a median of 1.96) and repeats where the denominator hit the
    guard were silently dropped, biasing the average over a non-random subset.

    PDP is excluded from the method columns: the reconstruction is *defined* as
    the PDP unit-coefficient sum, so its ratio is 1.000 by construction and
    carries no information.
    """
    if datasets is not None:
        res = res[res["dataset"].isin(datasets)]
    ceil = _sel(res, family="ceiling", target="fhat", split="test")
    c_ci = _ci_by(ceil, ["dataset", "model"], "additivity_r2")
    if c_ci.empty:
        return

    ach = _sel(res, target="fhat", split="test", variant="curve-unit")
    ach = ach[ach["family"].isin(METHODS)]
    a_ci = _ci_by(ach, ["dataset", "model", "family"], "r2")
    if a_ci.empty:
        return

    piv = a_ci.pivot_table(index=["dataset", "model"], columns="family", values="mean")
    base = c_ci.set_index(["dataset", "model"])
    shown = [m for m in METHODS if m != "PDP"]

    rows, n_suppressed, n_above = [], 0, 0
    for idx in base.index:
        denom = base.loc[idx, "mean"]
        denom_lo = base.loc[idx, "lo"]
        r = {"Dataset": idx[0], "Model": model_label(idx[1]),
             "$R^2_{\\mathrm{add}}$": fmt_ci(denom, denom_lo, base.loc[idx, "hi"])}
        # Gate on the *lower confidence bound*, not the point estimate: the
        # ratio is trustworthy only when the denominator is confidently bounded
        # away from zero.  A cell can have a comfortable mean and an interval
        # that nearly touches zero (abalone x mlp: 0.226 with lo = 0.029), and
        # the resulting ratio is then driven by whichever repeats happened to
        # land near the bottom of that interval.
        usable = (
            np.isfinite(denom) and np.isfinite(denom_lo) and denom_lo >= ATTAINMENT_MIN_DENOM
        )
        for m in shown:
            val = piv.loc[idx, m] if (idx in piv.index and m in piv.columns) else np.nan
            if not usable or not np.isfinite(val):
                r[m] = "--"
            else:
                ratio = val / denom
                r[m] = fmt(ratio, 2)
                n_above += ratio > 1.0
        if not usable:
            n_suppressed += 1
        rows.append(r)

    total = len(base.index) * len(shown)
    note = (
        f"Ratios are suppressed (--) where the lower confidence bound on "
        f"$R^2_{{\\mathrm{{add}}}}$ falls below {ATTAINMENT_MIN_DENOM}, since the quotient is "
        f"then driven by noise in its denominator ({n_suppressed} of {len(base.index)} "
        f"model--dataset cells). Of the ratios shown, {n_above} of "
        f"{total - n_suppressed * len(shown)} exceed 1: the additive PDP reconstruction is not "
        f"an upper bound once features are dependent."
    )

    write_table(
        out / "tab_ceiling.tex",
        pd.DataFrame(rows),
        caption=(
            "One row per (dataset, model) cell. $R^2_{\\mathrm{add}}$ is the share of the "
            "black box's score variance recovered by the sum of its centred partial "
            "dependence functions, with a 95\\% interval. The ALE, SHAP and LIME columns give "
            "the ratio of mean achieved $R^2$ to mean $R^2_{\\mathrm{add}}$ under the "
            "unit-coefficient construction $g_{\\mathbf 1}$. PDP has no column because the "
            "reconstruction is "
            "defined as its unit-coefficient sum, so its ratio is $1.000$ identically. A dash "
            "marks a cell whose $R^2_{\\mathrm{add}}$ is not confidently bounded away from "
            "zero, where the ratio would be numerically unstable."
        ),
        column_format="llrrrr",
        label="tab:ceiling",
        note=note,
        escape=False,
    )


def table_controls(res: pd.DataFrame, out: Path, datasets: list[str] | None = None,
                   stratum_note: str = "") -> None:
    """Do null explanations collapse?  If not, the measure is not measuring."""
    sub = _sel(res, target="fhat", split="test")
    if datasets is not None:
        sub = sub[sub["dataset"].isin(datasets)]
    sub = sub[sub["family"].isin(METHODS)]
    keep = ["curve-unit", "null-permuted", "null-randomised"]
    sub = sub[sub["variant"].isin(keep)]
    if sub.empty:
        return

    ci = _ci_by(sub, ["family", "variant"], "r2")
    piv = ci.pivot(index="family", columns="variant", values="mean")
    lo = ci.pivot(index="family", columns="variant", values="lo")
    hi = ci.pivot(index="family", columns="variant", values="hi")

    labels = {"curve-unit": "Intact", "null-permuted": "Values permuted", "null-randomised": "Randomised"}
    rows = []
    for m in [m for m in METHODS if m in piv.index]:
        r = {"Method": m}
        for v in keep:
            r[labels[v]] = fmt_ci(piv.loc[m, v], lo.loc[m, v], hi.loc[m, v]) if v in piv.columns else "--"
        if "curve-unit" in piv.columns and "null-permuted" in piv.columns:
            r["Collapse"] = fmt(piv.loc[m, "curve-unit"] - piv.loc[m, "null-permuted"], 3)
        rows.append(r)

    write_table(
        out / "tab_controls.tex",
        pd.DataFrame(rows),
        caption=(
            "Test $R^2$ against the black box under the unit-coefficient construction "
            "$g_{\\mathbf 1}$ of Eq.~\\ref{eq:unit}, "
            "averaged over datasets and model families. \\emph{Intact} is the unmodified "
            "explanation, \\emph{Values permuted} the null of Eq.~\\ref{eq:nullperm}, and "
            "\\emph{Randomised} the null of Eq.~\\ref{eq:nullrand}. \\emph{Collapse} is the "
            "difference between the intact and permuted columns. Parenthesised ranges are "
            "95\\% intervals." + stratum_note
        ),
        label="tab:controls",
        escape=False,
    )


def table_baselines(res: pd.DataFrame, out: Path,
                    datasets: list[str] | None = None) -> None:
    """Where the explanation-derived predictors sit between floor and ceiling.

    Averaged over the four model families and ranked best to worst, because the
    question the table answers -- how much of the task does an explanation
    retain, relative to using no explanation at all -- is a question about the
    ordering, not about any individual model family.
    """
    if datasets is not None:
        res = res[res["dataset"].isin(datasets)]
    sub = _sel(res, family="baseline", target="y", split="test")
    metric = "auc" if (sub["metric"] == "auc").any() else "r2"
    ci = _ci_by(sub, ["variant"], metric)
    bb = _ci_by(_sel(res, family="blackbox", target="y", split="test"), [], metric)
    best = _sel(res, target="y", split="test", variant="curve-unit")
    best = best[best["family"].isin(METHODS)]
    b_ci = _ci_by(best, ["family"], metric)
    if ci.empty:
        return

    labels = {"intercept": "Intercept only", "linear-raw": "Linear on raw features",
              "spline-gam": "Additive spline model (fitted to $y$)"}
    rows = []
    for _, r in ci.iterrows():
        if r["variant"] not in labels:
            continue
        rows.append({"Predictor": labels[r["variant"]], "_m": r["mean"],
                     "Score": fmt_ci(r["mean"], r["lo"], r["hi"])})
    for _, r in b_ci.iterrows():
        rows.append({"Predictor": f"{r['family']} (additive, unit)", "_m": r["mean"],
                     "Score": fmt_ci(r["mean"], r["lo"], r["hi"])})
    for _, r in bb.iterrows():
        rows.append({"Predictor": "Black box", "_m": r["mean"],
                     "Score": fmt_ci(r["mean"], r["lo"], r["hi"])})

    tbl = pd.DataFrame(rows).sort_values("_m", ascending=False).drop(columns="_m")

    write_table(
        out / "tab_baselines.tex",
        tbl,
        caption=(
            f"Predictors scored against the original target $y$ (test "
            f"{'ROC-AUC' if metric == 'auc' else '$R^2$'}), averaged over datasets and over "
            "the four model families, with a 95\\% interval. Rows are ordered best to worst. "
            "Explanation rows use the unit-coefficient construction $g_{\\mathbf 1}$ of "
            "Eq.~\\ref{eq:unit}; "
            "\\emph{Additive spline model} and \\emph{Linear on raw features} are fitted "
            "directly to $y$ with no explanation involved."
        ),
        column_format="lr",
        label="tab:baselines",
        escape=False,
    )


def table_shap_degeneracy(res: pd.DataFrame, out: Path) -> None:
    """The algebraic identity, measured."""
    gap = _sel(res, family="control", variant="shap-knn-identity", target="fhat", split="test")
    if gap.empty:
        return
    g = _ci_by(gap, ["dataset", "model"], "max_abs_gap")
    rel = _ci_by(gap, ["dataset", "model"], "rel_gap")

    anch = _ci_by(gap, ["model"], "r2_at_anchors")
    knn = _ci_by(_sel(res, family="control", variant="blackbox-knn", target="fhat", split="test"), ["model"], "r2")
    shp = _ci_by(_sel(res, family="SHAP", variant="sum-idw-degenerate", target="fhat", split="test"), ["model"], "r2")

    rows = []
    for _, r in g.iterrows():
        rr = rel[(rel["dataset"] == r["dataset"]) & (rel["model"] == r["model"])]
        rows.append({
            "Dataset": r["dataset"],
            "Model": model_label(r["model"]),
            "$\\max|\\hat y_{\\mathrm{SHAP\\text{-}sum}} - \\hat y_{k\\mathrm{NN}}|$": f"{r['mean']:.2e}",
            "relative to $\\mathrm{sd}(\\hat f)$": f"{rr['mean'].iloc[0]:.2e}" if len(rr) else "--",
        })

    def _list(frame):
        return ", ".join(f"{r['mean']:.3f}" for _, r in frame.iterrows()) or "--"

    note = (
        "At the anchor points the construction fits by tautology -- the zero-distance weight "
        "dominates -- giving $R^2 = " + _list(anch) + "$; out of sample it collapses to $R^2 = "
        + _list(shp) + "$, which equals the pure $k$NN control's $" + _list(knn)
        + "$. A large train--test gap for this construction is therefore an artefact of the "
        "construction, not a property of SHAP."
    )

    write_table(
        out / "tab_shap_degeneracy.tex",
        pd.DataFrame(rows),
        caption=(
            "The inverse-distance-weighted SHAP-sum construction is not a SHAP result. "
            "Because SHAP satisfies local accuracy, $\\sum_j \\phi_j(x^{(i)}) = \\hat f(x^{(i)}) - "
            "\\mathbb{E}[\\hat f]$ exactly, so the predictor "
            "$\\mathbb{E}[\\hat f] + \\sum_i w_i S_i$ with $\\sum_i w_i = 1$ collapses "
            "algebraically to $\\sum_i w_i \\hat f(x^{(i)})$ -- inverse-distance $k$NN on the "
            "black box's own predictions, containing no per-feature attribution information. "
            "The table reports the numerically measured gap between the two, which is zero to "
            "machine precision."
        ),
        label="tab:shap-degeneracy",
        note=note,
        escape=False,
    )


def table_traintest_gap(res: pd.DataFrame, out: Path, datasets: list[str] | None = None,
                        stratum_note: str = "") -> None:
    """Train-test gap per method -- the thesis's headline claim, done with CIs."""
    if datasets is not None:
        res = res[res["dataset"].isin(datasets)]
    sub = res[res["family"].isin(METHODS) & (res["target"] == "fhat") & (res["metric"] == "r2")]
    sub = sub[sub["variant"].isin(["curve-unit", "curve-ols", "local-idw", "sum-idw-degenerate"])]
    if sub.empty:
        return

    keys = ["dataset", "model", "family", "variant", "repeat"]
    piv = sub.pivot_table(index=keys, columns="split", values="value").reset_index()
    if "train" not in piv or "test" not in piv:
        return
    piv["gap"] = piv["train"] - piv["test"]

    rows = []
    for (fam, var), g in piv.groupby(["family", "variant"]):
        m_tr, lo_tr, hi_tr, _ = mean_ci(g["train"].to_numpy())
        m_te, lo_te, hi_te, _ = mean_ci(g["test"].to_numpy())
        m_gp, lo_gp, hi_gp, n = mean_ci(g["gap"].to_numpy())
        rows.append({
            "Method": fam,
            "Construction": var.replace("curve-", "additive, "),
            "Train $R^2$": fmt_ci(m_tr, lo_tr, hi_tr),
            "Test $R^2$": fmt_ci(m_te, lo_te, hi_te),
            "$\\Delta$": fmt_ci(m_gp, lo_gp, hi_gp),
            "_k": n,
        })

    tbl = pd.DataFrame(rows).sort_values(["Method", "Construction"])
    # The cell count is constant across rows, so it belongs in the caption
    # rather than in a column that pushes the table past the text width.
    ks = sorted(set(tbl["_k"]))
    kdesc = (
        f"Each row averages over $k = {ks[0]}$ (dataset, model, repeat) cells. "
        if len(ks) == 1
        else f"Rows average over between {ks[0]} and {ks[-1]} (dataset, model, repeat) cells. "
    )
    tbl = tbl.drop(columns="_k")

    write_table(
        out / "tab_traintest_gap.tex",
        tbl,
        caption=(
            "Train and test $R^2$ against the black box, and their difference "
            "$\\Delta$, for each (method, construction) pair. " + kdesc +
            "Parenthesised ranges are 95\\% intervals." + stratum_note
        ),
        column_format="llrrr",
        label="tab:traintest-gap",
        escape=False,
        tabcolsep="4pt",
    )


_DESIGN_RE = re.compile(r"^(?P<design>.+)_r\d{2}$")


def design_of(name: str) -> str:
    """Collapse a replicated draw back to its design (``syn_corr60_r07`` -> ``syn_corr60``)."""
    m = _DESIGN_RE.match(str(name))
    return m.group("design") if m else str(name)


def collapse_replicates(res: pd.DataFrame) -> pd.DataFrame:
    """Rename replicated draws to their design, keeping each draw a separate unit.

    Every statistic in the paper that treats the dataset as the unit of
    replication has to be computed on *designs*, not on draws.  Nine synthetic
    designs drawn ten times each would otherwise cast ninety votes against the
    thirteen real datasets, and a pooled mean would become a statement about
    the synthetic generator.  Renaming the dataset to its design fixes the
    weighting; re-indexing the repeat so that each (draw, split) pair stays
    distinct keeps the within-design variability that the intervals need.
    """
    if "dataset" not in res.columns:
        return res
    out = res.copy()
    names = out["dataset"].astype(str)
    draw = names.str.extract(r"_r(\d{2})$", expand=False)
    out["dataset"] = names.map(design_of)
    if "repeat" in out.columns:
        d = pd.to_numeric(draw, errors="coerce").fillna(0).astype(int)
        out["repeat"] = d * 1000 + pd.to_numeric(out["repeat"], errors="coerce").fillna(0).astype(int)
    return out


def table_refit_gain(res: pd.DataFrame, out: Path) -> None:
    """The refitting gain by design, as a relative increase in $R^2$.

    Reported as $(R^2(g_{\\hat\\alpha}) - R^2(g_{\\mathbf 1})) / R^2(g_{\\mathbf 1})$
    rather than as a difference: the absolute gap is hard to read against
    designs whose overall fidelity differs, and the question is how much
    refitting buys, which is a proportion.

    The unit of replication is the independent draw of the design, so the
    interval is a draw-to-draw interval and not a split-to-split one.
    """
    sub = _sel(res, target="fhat", split="test", family="PDP")
    sub = sub[(sub["metric"] == "r2") & sub["variant"].isin(["curve-unit", "curve-ols"])]
    sub = sub[sub["dataset"].astype(str).str.startswith("syn_")]
    if sub.empty:
        return
    sub = sub.copy()
    sub["design"] = sub["dataset"].map(design_of)

    # One value per (design, draw, model, variant), averaging over splits.
    per = (sub.groupby(["design", "dataset", "model", "variant"], observed=True)["value"]
              .mean().reset_index())
    wide = per.pivot_table(index=["design", "dataset", "model"],
                           columns="variant", values="value").reset_index()
    if "curve-unit" not in wide or "curve-ols" not in wide:
        return
    denom = wide["curve-unit"].abs()
    wide = wide[denom > 0.05]          # a near-zero baseline makes a ratio meaningless
    if wide.empty:
        return
    wide["rel"] = (wide["curve-ols"] - wide["curve-unit"]) / wide["curve-unit"]

    truth_corr = {"syn_inter00": 0.0, "syn_inter10": 0.0, "syn_inter25": 0.0,
                  "syn_inter50": 0.0, "syn_inter75": 0.0,
                  "syn_corr30": 0.3, "syn_corr60": 0.6, "syn_corr85": 0.85,
                  "syn_regress": 0.2}
    order = [d for d in truth_corr if d in set(wide["design"])]
    models = sorted(wide["model"].unique())

    rows, cut = [], None
    for i, d in enumerate(order):
        g = wide[wide["design"] == d]
        r = {"Design": tex_escape(d), "$\\rho$": f"{truth_corr[d]:.2f}"}
        for m in models:
            gg = g[g["model"] == m]
            r[model_label(m)] = (
                f"{100 * gg['rel'].mean():+.1f}" if len(gg) else "--"
            )
        rows.append(r)
        if truth_corr[d] == 0.0:
            cut = i

    rule_after = [cut] if cut is not None and cut + 1 < len(rows) else None
    write_table(
        out / "tab_refit_gain.tex",
        pd.DataFrame(rows),
        caption=(
            "Relative refitting gain for PDP, $(R^2(g_{\\hat\\alpha}) - "
            "R^2(g_{\\mathbf 1}))/R^2(g_{\\mathbf 1})$ of Eq.~\\ref{eq:gain}, in per cent. "
            "Rows are the synthetic designs, those with independent features "
            "($\\rho = 0$) above the rule and those with correlated features below; $\\rho$ "
            "is the designed feature correlation of Eq.~\\ref{eq:dgpX}. Columns are the four "
            "model families. Each entry averages the independent draws of that design."
        ),
        column_format="ll" + "r" * len(models),
        label="tab:refit-gain",
        escape=False,
        rule_after=rule_after,
    )


#: Which way each evaluation metric is supposed to point, per its own authors.
#: ``+1`` means a larger value is meant to indicate a better explanation.
#: Complexity and sparseness measure how *concentrated* an attribution vector
#: is rather than how faithful it is; they are included because they are
#: routinely reported as explanation-quality metrics, and the question of
#: whether concentration tracks quality is exactly what is being tested.
DISCRIM_DIRECTION = {
    "proposed_r2": +1,
    "faithfulness_corr": +1,
    "infidelity": -1,
    "max_sensitivity": -1,
    "complexity": -1,
    "sparseness": +1,
}

#: Plain-text names for figure legends.  The LaTeX labels below cannot simply
#: have their backslashes stripped -- that turns ``\hat f`` into ``hatf``.
DISCRIM_LABEL_PLAIN = {
    "proposed_r2": "Proposed ($R^2$ vs. $\\hat{f}$)",
    "faithfulness_corr": "Faithfulness correlation",
    "infidelity": "Infidelity",
    "max_sensitivity": "Max-sensitivity",
    "complexity": "Complexity",
    "sparseness": "Sparseness",
}

DISCRIM_LABEL = {
    "proposed_r2": "Proposed ($R^2$ vs.\\ $\\hat f$)",
    "faithfulness_corr": "Faithfulness correlation",
    "infidelity": "Infidelity",
    "max_sensitivity": "Max-sensitivity",
    "complexity": "Complexity",
    "sparseness": "Sparseness",
}


def _discrimination_accuracy(res: pd.DataFrame,
                             datasets: list[str] | None = None) -> pd.DataFrame:
    """Per-(metric, rung) accuracy at ranking the intact explanation first.

    Returns one row per (metric, rung, dataset): the fraction of
    (model, repeat, family) groups in that dataset for which the metric put the
    intact explanation ahead of the degraded one.  Keeping the dataset as the
    unit lets the caller aggregate with the same replication convention used
    everywhere else in the paper.
    """
    sub = res[(res.get("target") == "discrimination")] if "target" in res.columns else res.iloc[0:0]
    if sub.empty:
        return pd.DataFrame()
    if datasets is not None:
        sub = sub[sub["dataset"].isin(datasets)]
    # Only genuine explanation methods -- the CondMean aggregator control is not
    # an explanation and its R2 is on a wildly different scale.
    sub = sub[sub["family"].isin(METHODS)]
    sub = sub[sub["metric"].isin(DISCRIM_DIRECTION)]
    if sub.empty:
        return pd.DataFrame()

    keys = ["dataset", "model", "repeat", "family", "metric", "rung"]
    per = sub.groupby(keys, dropna=False, observed=True)["value"].mean().reset_index()
    wide = per.pivot_table(index=["dataset", "model", "repeat", "family", "metric"],
                           columns="rung", values="value").reset_index()
    rungs = sorted(c for c in wide.columns if isinstance(c, (int, float)))
    if 0.0 not in rungs:
        return pd.DataFrame()

    out_rows = []
    for metric, g in wide.groupby("metric", observed=True):
        direction = DISCRIM_DIRECTION[metric]
        for r in rungs:
            if r == 0.0:
                continue
            good, bad = g[0.0].to_numpy(float), g[r].to_numpy(float)
            ok = np.isfinite(good) & np.isfinite(bad)
            if not ok.any():
                continue
            correct = (direction * (good - bad) > 0).astype(float)
            # Exact ties are chance, not a success.
            correct[direction * (good - bad) == 0] = 0.5
            tmp = pd.DataFrame({"dataset": g["dataset"].to_numpy()[ok],
                                "correct": correct[ok]})
            for ds, gg in tmp.groupby("dataset", observed=True):
                out_rows.append({"metric": metric, "rung": r, "dataset": ds,
                                 "accuracy": float(gg["correct"].mean()),
                                 "n": int(len(gg))})
    return pd.DataFrame(out_rows)


def _rung_labels(res: pd.DataFrame) -> dict[float, str]:
    """Map the numeric rung index back to its human label."""
    sub = res[res.get("target") == "discrimination"] if "target" in res.columns else res.iloc[0:0]
    out: dict[float, str] = {}
    for rung, g in sub.groupby("rung", dropna=True, observed=True):
        v = str(g["variant"].iloc[0]).replace("discrim-", "")
        out[float(rung)] = {"intact": "intact",
                            "permuted": "permuted"}.get(v, v.replace("noise", "$\\sigma = ") + "$")
    return out


def table_discrimination(res: pd.DataFrame, out: Path,
                         datasets: list[str] | None = None) -> None:
    """Which evaluation metric actually notices that an explanation is worse?"""
    acc = _discrimination_accuracy(res, datasets)
    if acc.empty:
        return
    labels = _rung_labels(res)
    rungs = sorted(acc["rung"].unique())

    rows = []
    for metric in DISCRIM_DIRECTION:
        g = acc[acc["metric"] == metric]
        if g.empty:
            continue
        r = {"Evaluation metric": DISCRIM_LABEL[metric]}
        overall = []
        for rung in rungs:
            gg = g[g["rung"] == rung]
            if gg.empty:
                r[labels.get(rung, str(rung))] = "--"
                continue
            m, lo, hi, _ = mean_ci(gg["accuracy"].to_numpy())
            r[labels.get(rung, str(rung))] = f"{m:.3f}"
            overall.append(m)
        r["Mean"] = f"{np.mean(overall):.3f}" if overall else "--"
        r["_sort"] = float(np.mean(overall)) if overall else -np.inf
        rows.append(r)

    n_ds = int(acc["dataset"].nunique())
    tbl = (pd.DataFrame(rows).sort_values("_sort", ascending=False)
             .drop(columns="_sort").reset_index(drop=True))
    write_table(
        out / "tab_discrimination.tex",
        tbl,
        caption=(
            "One row per evaluation metric. Columns are the rungs of the degradation "
            "ladder of Eq.~\\ref{eq:ladder}: $\\sigma$ is Gaussian noise added to the "
            "explanation's curves in units of each curve's own amplitude, and "
            "\\emph{permuted} is the null of Eq.~\\ref{eq:nullperm}. Each entry is the "
            "fraction of (model, split, explanation method) cells in which the metric ranked "
            "the intact explanation above that rung, averaged over the "
            f"{n_ds} datasets; \\emph{{Mean}} averages across rungs. Each metric is read in "
            "the direction its originating paper specifies, so that a larger entry always "
            "means better agreement with the known ordering. Rows are ordered by "
            "\\emph{Mean}, best first."
        ),
        column_format="l" + "r" * (len(rungs) + 1),
        label="tab:discrimination",
        escape=False,
    )


def figure_discrimination(res: pd.DataFrame, out: Path,
                          datasets: list[str] | None = None) -> None:
    """Accuracy against degradation severity, one line per evaluation metric."""
    acc = _discrimination_accuracy(res, datasets)
    if acc.empty:
        return
    labels = _rung_labels(res)
    rungs = sorted(acc["rung"].unique())

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    palette = [METHOD_COLOURS["PDP"], METHOD_COLOURS["ALE"], METHOD_COLOURS["SHAP"],
               METHOD_COLOURS["LIME"], GREY_DARK, GREY_MID]
    markers = ["o", "s", "^", "D", "v", "P"]

    for i, metric in enumerate(DISCRIM_DIRECTION):
        g = acc[acc["metric"] == metric]
        if g.empty:
            continue
        xs, ms, los, his = [], [], [], []
        for j, rung in enumerate(rungs):
            gg = g[g["rung"] == rung]
            if gg.empty:
                continue
            m, lo, hi, _ = mean_ci(gg["accuracy"].to_numpy())
            xs.append(j); ms.append(m); los.append(lo); his.append(hi)
        lw = 2.4 if metric == "proposed_r2" else 1.3
        ax.plot(xs, ms, marker=markers[i % len(markers)], color=palette[i % len(palette)],
                linewidth=lw, markersize=5, label=DISCRIM_LABEL_PLAIN[metric])
        ax.fill_between(xs, los, his, color=palette[i % len(palette)], alpha=0.10, linewidth=0)

    ax.axhline(0.5, color=INK, linewidth=0.9, linestyle=":", zorder=1)
    # The data occupy the top and bottom of the panel and the chance line runs
    # through the middle, so an in-axes legend has nowhere to sit that does not
    # collide with something.  Put it under the axes.
    ax.text(0.995, 0.505, "chance", transform=ax.get_yaxis_transform(),
            fontsize=8, color=INK, va="bottom", ha="right")
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels([labels.get(r, str(r)).replace("$", "").replace("\\sigma", "σ")
                        for r in rungs])
    ax.set_xlabel("Degradation applied to the explanation")
    ax.set_ylabel("P(ranks intact above degraded)")
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out, "fig_discrimination")


def table_refmetrics(res: pd.DataFrame, out: Path, datasets: list[str] | None = None,
                     stratum_note: str = "") -> None:
    """Does the proposed measure add anything over established metrics?"""
    if datasets is not None:
        res = res[res["dataset"].isin(datasets)]
    ref = _sel(res, target="explanation", split="test")
    prop = _sel(res, target="fhat", split="test", variant="curve-unit")
    prop = prop[(prop["metric"] == "r2") & prop["family"].isin(METHODS)]
    if ref.empty or prop.empty:
        return

    keys = ["dataset", "model", "family", "repeat"]
    p = prop.groupby(keys, observed=True)["value"].mean().rename("proposed").reset_index()

    rows, n_cells = [], 0
    for metric in ["infidelity", "faithfulness_corr", "max_sensitivity", "complexity", "sparseness"]:
        r = ref[ref["metric"] == metric].groupby(keys, observed=True)["value"].mean().rename(metric).reset_index()
        merged = p.merge(r, on=keys, how="inner").dropna()
        if len(merged) < 5:
            continue
        from scipy import stats as _st

        rho = _st.spearmanr(merged["proposed"], merged[metric])
        # Also rank agreement at the level that matters: the ordering of the
        # four methods within each (dataset, model, repeat) cell.
        within = []
        for _, g in merged.groupby(["dataset", "model", "repeat"], observed=True):
            if g["family"].nunique() < 3:
                continue
            s = _st.spearmanr(g["proposed"], g[metric]).statistic
            if np.isfinite(s):
                within.append(s)
        m_w, lo_w, hi_w, n_w = mean_ci(np.asarray(within)) if within else (np.nan,) * 3 + (0,)

        rows.append({
            "Established metric": metric.replace("_", " "),
            "Pooled $\\rho$": fmt(rho.statistic, 3),
            "$p$": f"{rho.pvalue:.1e}",
            "Within-cell $\\rho$": fmt_ci(m_w, lo_w, hi_w),
        })
        n_cells = n_w

    if not rows:
        return
    write_table(
        out / "tab_refmetrics.tex",
        pd.DataFrame(rows),
        caption=(
            "Spearman rank correlations between the proposed metric (test $R^2$ against the "
            "black box, unit-coefficient construction) and five established metrics computed "
            "on the same explanations and splits, one row per established metric. "
            "\\emph{Pooled $\\rho$} ranks over all (dataset, model, method, split) cells at "
            "once, with its $p$-value alongside. \\emph{Within-cell $\\rho$} ranks the four "
            "explanation methods against each other inside a single (dataset, model, split) "
            f"cell and averages over the {n_cells} such cells, with a 95\\% interval."
            + stratum_note
        ),
        label="tab:refmetrics",
        escape=False,
    )


def table_lime_kernel(res: pd.DataFrame, out: Path) -> None:
    """How much of the LIME result is the library's default kernel width?"""
    sub = res[(res["family"] == "LIME") & res["variant"].str.startswith("kernel-")]
    if sub.empty:
        return
    sub = sub.copy()
    sub["kw"] = sub["variant"].str.replace("kernel-", "", regex=False).astype(float)

    fid = _ci_by(sub[(sub["target"] == "fhat") & (sub["split"] == "test")], ["kw"], "r2")
    anc = _ci_by(sub[(sub["target"] == "fhat") & (sub["split"] == "train")], ["kw"], "anchor_fit_r2")
    if fid.empty:
        return
    m = fid.merge(anc, on="kw", how="left", suffixes=("", "_a")).sort_values("kw")

    show = pd.DataFrame({
        "Kernel width": m["kw"].map(lambda v: fmt(v, 2)),
        "Local fit at own anchor $R^2$": [fmt_ci(a, b, c) for a, b, c in
                                          zip(m.get("mean_a"), m.get("lo_a"), m.get("hi_a"))],
        "Test $R^2$ vs.\\ $\\hat f$": [fmt_ci(a, b, c) for a, b, c in zip(m["mean"], m["lo"], m["hi"])],
    })

    write_table(
        out / "tab_lime_kernel.tex",
        show,
        caption=(
            "LIME sensitivity to its kernel width. \\emph{Local fit at own anchor} is how well "
            "each local linear model reproduces the black box at the very point it was fitted "
            "around -- a lower bound on how local the approximation is. The library default, "
            "$0.75\\sqrt{p}$, combined with perturbations drawn at the full training standard "
            "deviation, produces a neighbourhood wide enough that the local model is closer to "
            "a globally weighted linear fit. Any single-width LIME result is therefore a "
            "statement about the default rather than about the method."
        ),
        label="tab:lime-kernel",
        escape=False,
    )


def fig_lime_kernel(res: pd.DataFrame, out: Path) -> None:
    sub = res[(res["family"] == "LIME") & res["variant"].str.startswith("kernel-")]
    if sub.empty:
        return
    sub = sub.copy()
    sub["kw"] = sub["variant"].str.replace("kernel-", "", regex=False).astype(float)
    fid = _ci_by(sub[(sub["target"] == "fhat") & (sub["split"] == "test")], ["kw"], "r2").sort_values("kw")
    anc = _ci_by(sub[(sub["target"] == "fhat") & (sub["split"] == "train")], ["kw"], "anchor_fit_r2").sort_values("kw")
    if fid.empty:
        return

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        c = METHOD_COLOURS["LIME"]
        ax.plot(fid["kw"], fid["mean"], marker="D", color=c, linewidth=1.6, markersize=5,
                markeredgecolor="white", markeredgewidth=0.6, label="test $R^2$ vs. $\\hat f$")
        ax.fill_between(fid["kw"], fid["lo"], fid["hi"], color=c, alpha=0.15, linewidth=0)
        if not anc.empty:
            ax.plot(anc["kw"], anc["mean"], marker="o", color=GREY_DARK, linewidth=1.4,
                    linestyle="--", markersize=4.5, markeredgecolor="white", markeredgewidth=0.6,
                    label="local fit at own anchor")
        ax.set_xscale("log")
        # Label only the widths actually swept: the default log locator emits
        # overlapping minor ticks at this range.
        widths = sorted(fid["kw"].unique())
        ax.set_xticks(widths)
        ax.set_xticklabels([f"{w:g}" for w in widths])
        ax.minorticks_off()
        ax.set_xlabel("LIME kernel width (log scale)")
        ax.set_ylabel("$R^2$")
        ax.legend(loc="center right")
        _save(fig, out, "fig_lime_kernel")


def table_synthetic(res: pd.DataFrame, truth: pd.DataFrame, out: Path) -> None:
    """Does the measured ceiling recover the designed one?"""
    if truth is None or truth.empty:
        return
    ceil = _sel(res, family="ceiling", target="fhat", split="test")
    ci = _ci_by(ceil, ["dataset", "model"], "additivity_r2")
    if ci.empty:
        return
    m = ci.merge(truth, on="dataset", how="inner")
    if m.empty:
        return

    # Collapse the independent draws of each design, then block by model so that
    # a reader can run an eye down one model's rows and watch the measured
    # ceiling track the designed one.
    m = m.copy()
    m["design"] = m["dataset"].map(design_of)
    per_design = []
    for (model, design), g in m.groupby(["model", "design"], observed=True):
        mm, lo, hi, _ = mean_ci(g["mean"].to_numpy())
        per_design.append({
            "model": model, "design": design,
            "tau": float(g["interaction_strength"].iloc[0]),
            "rho": float(g["feature_correlation"].iloc[0]),
            "true": float(g["true_additive_r2"].mean()),
            "mean": mm, "lo": lo, "hi": hi, "draws": int(g["dataset"].nunique()),
        })
    d = pd.DataFrame(per_design).sort_values(["model", "rho", "tau", "design"])

    show = pd.DataFrame({
        "Model": d["model"].map(model_label),
        "Design": d["design"].map(tex_escape),
        # tau is the interaction weight, rho the feature correlation; the two
        # must not both be called rho.
        "$\\tau$": d["tau"].map(lambda v: fmt(v, 2)),
        "$\\rho$": d["rho"].map(lambda v: fmt(v, 2)),
        "True $R^2_{\\mathrm{add}}$": d["true"].map(lambda v: fmt(v, 3)),
        "Measured $R^2_{\\mathrm{add}}$": [fmt_ci(a, b, c) for a, b, c in zip(d["mean"], d["lo"], d["hi"])],
    })

    # A rule between model blocks.
    cuts, seen = [], None
    for i, mod in enumerate(d["model"].tolist()):
        if seen is not None and mod != seen:
            cuts.append(i - 1)
        seen = mod

    n_draws = int(d["draws"].max()) if len(d) else 0
    write_table(
        out / "tab_synthetic.tex",
        show,
        caption=(
            "One row per (model family, synthetic design), blocked by model family and "
            "ordered within a block by $\\rho$ and then $\\tau$. $\\tau$ is the designed "
            "interaction weight of Eq.~\\ref{eq:dgpf} and $\\rho$ the designed feature "
            "correlation of Eq.~\\ref{eq:dgpX}. \\emph{True} $R^2_{\\mathrm{add}}$ is the "
            "additive variance share the design fixes by construction; \\emph{Measured} "
            f"$R^2_{{\\mathrm{{add}}}}$ is the quantity estimated from the fitted black box, "
            f"averaged over {n_draws} independent draws of the design with a 95\\% interval "
            "over draws."
        ),
        column_format="llrrrr",
        label="tab:synthetic",
        escape=False,
        rule_after=cuts or None,
    )


def table_pairwise(res: pd.DataFrame, out: Path, datasets: list[str] | None = None,
                   stratum_note: str = "", suffix: str = "") -> None:
    """Paired method-vs-method comparisons, laid out as a correlation-style matrix.

    Every pair appears once, in the upper triangle: the cell in row $a$ and
    column $b$ is $\\Delta R^2 = R^2(a) - R^2(b)$ with the Holm-corrected paired
    $p$ beneath it.  The lower triangle would only repeat the same numbers with
    the sign flipped.
    """
    sub = _sel(res, target="fhat", split="test", variant="curve-unit")
    if datasets is not None:
        sub = sub[sub["dataset"].isin(datasets)]
    sub = sub[(sub["metric"] == "r2") & sub["family"].isin(METHODS)]
    if sub.empty:
        return
    keys = ["dataset", "model", "repeat"]
    piv = sub.pivot_table(index=keys, columns="family", values="value")
    present = [m for m in METHODS if m in piv.columns]

    stats, pvals = {}, {}
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            dsl = dataset_level_comparison(piv, a, b)
            stats[(a, b)] = dsl
            pvals[f"{a} vs {b}"] = dsl["p_t"]
    if not stats:
        return

    from .scoring import holm_bonferroni

    adj = holm_bonferroni(pvals, return_adjusted=True)

    n_ds = max((s["n_datasets"] for s in stats.values()), default=0)
    rows = []
    for i, a in enumerate(present):
        r = {"": a}
        for j, b in enumerate(present):
            if j <= i:
                r[b] = ""
                continue
            s = stats[(a, b)]
            p = adj.get(f"{a} vs {b}", np.nan)
            if not np.isfinite(s["diff"]):
                r[b] = "--"
                continue
            ptxt = "--" if not np.isfinite(p) else (f"{p:.3f}" if p >= 1e-3 else f"{p:.0e}")
            r[b] = f"{s['diff']:+.3f} \\, ({ptxt})"
        rows.append(r)

    write_table(
        out / f"tab_pairwise{suffix}.tex",
        pd.DataFrame(rows),
        caption=(
            "Paired differences in test $R^2$ against the black box between explanation "
            "methods, under the unit-coefficient construction $g_{\\mathbf 1}$ of "
            "Eq.~\\ref{eq:unit}. The entry "
            "in row $a$, column $b$ is $\\Delta R^2 = R^2(a) - R^2(b)$, followed in parentheses "
            "by the Holm-corrected $p$-value of a paired $t$-test over datasets. A positive "
            f"entry favours the row method. The unit of replication is the dataset ($n = {n_ds}$): "
            "differences are averaged to one value per dataset before testing. Holm correction "
            f"is over the {len(stats)} comparisons. The lower triangle is left empty because it "
            "carries the same numbers with the sign reversed."
            + stratum_note
        ),
        column_format="l" + "r" * len(present),
        label=f"tab:pairwise{suffix.replace('_', '-')}",
        escape=False,
    )


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _method_style(m: str) -> dict:
    return {
        "color": METHOD_COLOURS.get(m, GREY_DARK),
        "marker": METHOD_MARKERS.get(m, "o"),
        "linestyle": METHOD_LINESTYLES.get(m, "-"),
    }


def _save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png")
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)


def fig_corruption(res: pd.DataFrame, out: Path,
                   datasets: list[str] | None = None) -> None:
    """Does the measure degrade as the explanation is corrupted?"""
    if datasets is not None:
        res = res[res["dataset"].isin(datasets)]
    sub = res[res["family"].isin(METHODS) & (res["target"] == "fhat")
              & (res["split"] == "test") & (res["metric"] == "r2")
              & res["variant"].str.startswith("noise-")]
    if sub.empty:
        return
    sub = sub.copy()
    sub["sigma"] = sub["variant"].str.replace("noise-", "", regex=False).astype(float)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        for m in METHODS:
            g = sub[sub["family"] == m]
            if g.empty:
                continue
            agg = g.groupby("sigma")["value"].agg(["mean", "count", "std"]).reset_index()
            se = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
            st = _method_style(m)
            ax.plot(agg["sigma"], agg["mean"], label=m, linewidth=1.6, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.6, **st)
            ax.fill_between(agg["sigma"], agg["mean"] - 1.96 * se, agg["mean"] + 1.96 * se,
                            color=st["color"], alpha=0.15, linewidth=0)
        ax.axhline(0.0, color=GREY_MID, linewidth=1.0, linestyle=":", zorder=0)
        ax.set_xlabel("noise added to explanation (multiples of curve amplitude)")
        ax.set_ylabel("test $R^2$ against the black box")
        ax.legend(ncol=2, loc="upper right")
        _save(fig, out, "fig_corruption")


def fig_ceiling_attainment(res: pd.DataFrame, out: Path,
                           datasets: list[str] | None = None) -> None:
    """Achieved fidelity against the additivity ceiling, per dataset/model."""
    if datasets is not None:
        res = res[res["dataset"].isin(datasets)]
    ceil = _sel(res, family="ceiling", target="fhat", split="test")
    c = _per_repeat(ceil, ["dataset", "model"], "additivity_r2").rename(columns={"value": "ceiling"})
    ach = _sel(res, target="fhat", split="test", variant="curve-unit")
    ach = ach[ach["family"].isin(METHODS)]
    a = _per_repeat(ach, ["dataset", "model", "family"], "r2").rename(columns={"value": "achieved"})
    m = a.merge(c, on=["dataset", "model", "repeat"], how="inner")
    if m.empty:
        return
    agg = m.groupby(["dataset", "model", "family"])[["ceiling", "achieved"]].mean().reset_index()

    # How often does a method beat the PDP reconstruction?  Under feature
    # dependence the reconstruction is not an upper bound, and saying so on the
    # figure is more honest than drawing a line labelled "ceiling" that
    # a large share of the points sit above.
    other = agg[agg["family"] != "PDP"]
    n_above = int((other["achieved"] > other["ceiling"]).sum())
    n_other = int(len(other))

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
        lim = [min(0.0, agg[["ceiling", "achieved"]].min().min()) - 0.05, 1.05]
        # The diagonal carries no inline label: rotated text on it either clips
        # at the corner or collides with the point cloud, and the line is
        # self-explanatory once the annotation names it.
        ax.plot(lim, lim, color=GREY_MID, linewidth=1.0, linestyle="--", zorder=0,
                label="equality")
        for meth in METHODS:
            g = agg[agg["family"] == meth]
            if g.empty:
                continue
            st = _method_style(meth)
            ax.scatter(g["ceiling"], g["achieved"], s=34, label=meth, alpha=0.85,
                       edgecolor="white", linewidth=0.6,
                       color=st["color"], marker=st["marker"], zorder=3)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("additive PDP reconstruction $R^2_{\\mathrm{add}}$ (a property of the model)")
        ax.set_ylabel("achieved test $R^2$")
        if n_other:
            ax.text(0.03, 0.97,
                    f"{n_above}/{n_other} non-PDP points ({n_above/n_other:.0%})\n"
                    f"lie above it: under feature\ndependence the reconstruction\n"
                    f"is not an upper bound",
                    transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=GREY_DARK)
        ax.legend(ncol=2, loc="lower right")
        ax.set_aspect("equal", adjustable="box")
        _save(fig, out, "fig_ceiling_attainment")


def fig_synthetic_recovery(res: pd.DataFrame, truth: pd.DataFrame, out: Path) -> None:
    if truth is None or truth.empty:
        return
    ceil = _sel(res, family="ceiling", target="fhat", split="test")
    ci = _ci_by(ceil, ["dataset", "model"], "additivity_r2")
    m = ci.merge(truth, on="dataset", how="inner")
    if m.empty:
        return
    m = m[m["feature_correlation"] <= 1e-9] if "feature_correlation" in m else m
    if m.empty:
        return

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.plot([0, 1], [0, 1], color=GREY_MID, linestyle="--", linewidth=1.0, zorder=0,
                label="perfect recovery")
        models = sorted(m["model"].unique())
        greys = [INK, GREY_DARK, GREY_MID, GREY_LIGHT]
        markers = ["o", "s", "^", "D"]
        for i, mod in enumerate(models):
            g = m[m["model"] == mod].sort_values("true_additive_r2")
            err = np.vstack([g["mean"] - g["lo"], g["hi"] - g["mean"]])
            ax.errorbar(g["true_additive_r2"], g["mean"], yerr=err, fmt=markers[i % 4],
                        color=greys[i % 4], markersize=5, linewidth=1.2, capsize=2,
                        markeredgecolor="white", markeredgewidth=0.5, label=mod)
        ax.set_xlabel("designed additive share of $\\mathrm{Var}(f)$")
        ax.set_ylabel("measured $R^2_{\\mathrm{add}}$")
        ax.legend(loc="upper left")
        _save(fig, out, "fig_synthetic_recovery")


def fig_fidelity_forest(res: pd.DataFrame, out: Path,
                        datasets: list[str] | None = None) -> None:
    """Per-dataset fidelity with intervals -- shows where methods actually differ.

    Synthetic designs are grouped first and real datasets second, so that the
    block where feature independence holds by construction can be read as a
    block.  Replicated draws of a design are collapsed onto one row, with the
    interval taken over draws.
    """
    if datasets is not None:
        res = res[res["dataset"].isin(datasets)]
    sub = _sel(res, target="fhat", split="test", variant="curve-unit")
    sub = sub[sub["family"].isin(METHODS)]
    if sub.empty:
        return
    sub = sub.copy()
    sub["dataset"] = sub["dataset"].map(design_of)
    ci = _ci_by(sub, ["dataset", "family"], "r2")
    if ci.empty:
        return

    names = list(ci["dataset"].unique())
    synth = sorted(d for d in names if str(d).startswith("syn_"))
    real = sorted(d for d in names if not str(d).startswith("syn_"))
    datasets = synth + real

    with plt.rc_context(PLOT_STYLE):
        h = max(3.0, 0.42 * len(datasets) + 1.2)
        fig, ax = plt.subplots(figsize=(5.4, h))
        offsets = np.linspace(-0.26, 0.26, len(METHODS))
        for k, meth in enumerate(METHODS):
            g = ci[ci["family"] == meth].set_index("dataset")
            ys, xs, los, his = [], [], [], []
            for i, d in enumerate(datasets):
                if d not in g.index:
                    continue
                ys.append(i + offsets[k])
                xs.append(g.loc[d, "mean"])
                los.append(max(g.loc[d, "mean"] - g.loc[d, "lo"], 0.0))
                his.append(max(g.loc[d, "hi"] - g.loc[d, "mean"], 0.0))
            if not ys:
                continue
            st = _method_style(meth)
            ax.errorbar(xs, ys, xerr=np.vstack([los, his]), fmt=st["marker"],
                        color=st["color"], markersize=4.5, linewidth=1.2, capsize=1.8,
                        markeredgecolor="white", markeredgewidth=0.5, label=meth)
        if synth and real:
            ax.axhline(len(synth) - 0.5, color=GREY_MID, linewidth=0.8, linestyle="--")
            ax.text(0.995, (len(synth) - 1) / 2, "synthetic", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=7, color=GREY_DARK, rotation=90)
            ax.text(0.995, len(synth) + (len(real) - 1) / 2, "real",
                    transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=7, color=GREY_DARK, rotation=90)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets)
        ax.invert_yaxis()
        ax.set_xlabel("test $R^2$ against the black box")
        ax.grid(axis="y", visible=False)
        ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.04))
        _save(fig, out, "fig_fidelity_forest")


def fig_traintest(res: pd.DataFrame, out: Path,
                  datasets: list[str] | None = None) -> None:
    """Train vs test, the corrected version of the thesis's headline figure."""
    if datasets is not None:
        res = res[res["dataset"].isin(datasets)]
    sub = res[res["family"].isin(METHODS + ["control"]) & (res["target"] == "fhat")
              & (res["metric"] == "r2")]
    sub = sub[sub["variant"].isin(["curve-unit", "local-idw", "sum-idw-degenerate", "blackbox-knn"])]
    if sub.empty:
        return
    keys = ["dataset", "model", "family", "variant", "repeat"]
    piv = sub.pivot_table(index=keys, columns="split", values="value").reset_index()
    if "train" not in piv or "test" not in piv:
        return
    piv["label"] = piv["family"] + "\n" + piv["variant"].str.replace("curve-", "additive ", regex=False)
    order = [l for l in [
        "PDP\nadditive unit", "ALE\nadditive unit", "SHAP\nadditive unit", "LIME\nadditive unit",
        "LIME\nlocal-idw", "SHAP\nsum-idw-degenerate", "control\nblackbox-knn",
    ] if l in set(piv["label"])]
    if not order:
        return

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        x = np.arange(len(order))
        w = 0.36
        tr = [piv[piv["label"] == l]["train"].mean() for l in order]
        te = [piv[piv["label"] == l]["test"].mean() for l in order]
        tr_e = [1.96 * piv[piv["label"] == l]["train"].sem() for l in order]
        te_e = [1.96 * piv[piv["label"] == l]["test"].sem() for l in order]
        ax.bar(x - w / 2, tr, w - 0.03, yerr=tr_e, label="train", color=GREY_MID,
               edgecolor="white", linewidth=1.0, capsize=2)
        ax.bar(x + w / 2, te, w - 0.03, yerr=te_e, label="test", color="#2a78d6",
               edgecolor="white", linewidth=1.0, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=7)
        ax.set_ylabel("$R^2$ against the black box")
        ax.legend(ncol=2, loc="upper left")
        ax.grid(axis="x", visible=False)
        _save(fig, out, "fig_traintest")


def fig_example_curves(curve_payload: dict, out: Path) -> None:
    """Side-by-side PDP / ALE / SHAP-dependence / LIME-dependence for one feature."""
    if not curve_payload:
        return
    feats = curve_payload.get("features", [])
    if not feats:
        return
    n = min(3, len(feats))

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, n, figsize=(2.5 * n + 0.8, 2.9), sharey=False)
        axes = np.atleast_1d(axes)
        for k in range(n):
            ax = axes[k]
            f = feats[k]
            for meth in METHODS:
                cur = f.get(meth)
                if not cur:
                    continue
                st = _method_style(meth)
                ax.plot(cur["grid"], cur["values"], linewidth=1.6,
                        color=st["color"], linestyle=st["linestyle"],
                        label=meth if k == 0 else None)
            ax.axhline(0.0, color=GREY_LIGHT, linewidth=0.8, zorder=0)
            ax.set_title(f["name"], fontsize=9)
            ax.set_xlabel("feature value")
            if k == 0:
                ax.set_ylabel("centred effect on score")
        handles = [Line2D([0], [0], color=METHOD_COLOURS[m], linestyle=METHOD_LINESTYLES[m],
                          linewidth=1.6, label=m) for m in METHODS]
        fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.06))
        _save(fig, out, "fig_example_curves")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_report(results_dir: Path) -> None:
    """Regenerate every table and figure from ``Results/raw``."""
    results_dir = Path(results_dir)
    raw = results_dir / "raw"
    tables = results_dir / "tables"
    figures = results_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    res = pd.read_csv(raw / "results.csv")
    for col in ("family", "variant", "target", "split", "metric", "dataset", "model"):
        if col in res.columns:
            res[col] = res[col].astype(str)

    datasets = pd.read_csv(raw / "datasets.csv") if (raw / "datasets.csv").exists() else pd.DataFrame()
    diagnostics = pd.read_csv(raw / "diagnostics.csv") if (raw / "diagnostics.csv").exists() else pd.DataFrame()
    truth = pd.read_csv(raw / "synthetic_truth.csv") if (raw / "synthetic_truth.csv").exists() else pd.DataFrame()
    payload_path = raw / "example_curves.json"
    payload = {}
    if payload_path.exists():
        import json

        payload = json.loads(payload_path.read_text())

    # Pooled tables are computed on the continuous-dominated stratum only.
    # Categorical-dominated datasets are not dropped -- they get their own
    # companion tables, because a dataset on which the transform is provably a
    # no-op cannot contribute to a comparison between methods, and averaging it
    # in would understate whatever difference exists.
    # Statistics whose unit of replication is the dataset must treat a
    # synthetic *design* as one unit, not its ten independent draws.  ``res``
    # keeps the draws (the refit-gain, synthetic-recovery and ceiling tables
    # need them); ``resd`` collapses them and is what everything else uses.
    resd = collapse_replicates(res)

    primary, flagged = split_strata(diagnostics, resd)
    note = _stratum_note(flagged)

    # Experiments that a real dataset can carry are run on real data only.
    # Synthetic designs are reserved for the questions real data cannot answer:
    # validating the estimator against a known additive share, and testing the
    # dependence diagnostic at a known feature correlation.  Mixing the two in
    # one average would report a number that is partly a property of our own
    # generator.
    real = [d for d in primary if not str(d).startswith("syn_")]
    if not real:
        real = primary
    print(f"Pooled tables use the {len(real)} real dataset(s); synthetic designs "
          f"are reported separately.")
    if flagged:
        print(f"Strata: {len(primary)} continuous-dominated, {len(flagged)} "
              f"categorical-dominated (>{BINARY_FRACTION_LIMIT:.0%} binary): {', '.join(flagged)}")
        print("  Pooled tables use the first stratum; the second gets *_categorical companions.")

    if not datasets.empty:
        table_datasets(datasets, diagnostics, tables)
    table_blackbox(resd, tables)
    table_fidelity_compact(resd, tables, datasets=real, stratum_note=note)
    table_aggregator_control(resd, tables, datasets=real)
    for target in ("fhat", "y"):
        table_fidelity(resd, tables, target=target, datasets=real, stratum_note=note)
        if flagged:
            table_fidelity(
                res, tables, target=target, datasets=flagged, suffix="_categorical",
                stratum_note=(
                    " This table covers only the categorical-dominated datasets "
                    f"({', '.join(tex_escape(d) for d in flagged)}), where more than "
                    f"{int(BINARY_FRACTION_LIMIT * 100)}\\% of encoded columns are binary and the "
                    "curve transform is therefore affine. Methods are expected to be "
                    "indistinguishable here; that they are is the point."
                ),
            )
    table_ceiling(res, tables, datasets=real)
    table_controls(resd, tables, datasets=real, stratum_note=note)
    table_baselines(resd, tables, datasets=real)
    table_shap_degeneracy(res, tables)
    table_traintest_gap(resd, tables, datasets=real, stratum_note=note)
    table_refmetrics(resd, tables, datasets=real, stratum_note=note)
    table_pairwise(resd, tables, datasets=real, stratum_note=note)
    if flagged:
        table_pairwise(res, tables, datasets=flagged, suffix="_categorical", stratum_note=(
            " Restricted to the categorical-dominated stratum, where more than "
            f"{int(BINARY_FRACTION_LIMIT * 100)}\\% of encoded columns are binary and the curve "
            "transform is affine. Differences here should be near zero."))
    table_lime_kernel(res, tables)
    table_synthetic(res, truth, tables)
    table_discrimination(resd, tables, datasets=real)
    table_refit_gain(res, tables)

    write_prose_numbers(resd, diagnostics, tables, raw_res=res, real=real)

    figure_discrimination(resd, figures, datasets=real)

    fig_corruption(resd, figures, datasets=real)
    fig_ceiling_attainment(resd, figures, datasets=real)
    fig_synthetic_recovery(res, truth, figures)
    fig_fidelity_forest(res, figures, datasets=real)
    fig_traintest(resd, figures, datasets=real)
    fig_lime_kernel(res, figures)
    fig_example_curves(payload, figures)

    made_t = sorted(p.name for p in tables.glob("*.tex"))
    made_f = sorted(p.name for p in figures.glob("*.png"))
    print(f"\nTables ({len(made_t)}) -> {tables}")
    for t in made_t:
        print(f"  {t}")
    print(f"\nFigures ({len(made_f)}) -> {figures}")
    for f in made_f:
        print(f"  {f}")


# --------------------------------------------------------------------------
# Prose numbers
# --------------------------------------------------------------------------

#: Datasets whose feature design is independent by construction, used to
#: stratify the headline comparison.  The correlation sweep and the real data
#: are dependent; the interaction sweep is not.
INDEP_SYNTH = ["syn_inter00", "syn_inter10", "syn_inter25", "syn_inter50", "syn_inter75"]
CORR_SYNTH = ["syn_corr30", "syn_corr60", "syn_corr85"]


def write_prose_numbers(res: pd.DataFrame, diagnostics: pd.DataFrame, out: Path,
                        raw_res: pd.DataFrame | None = None,
                        real: list[str] | None = None) -> None:
    """Emit every number the manuscript quotes in prose as a LaTeX macro.

    The tables in this paper are generated, so they cannot drift from the
    experiment.  The *prose* could, and did: after one run was replaced by
    another, sentences in four sections still carried the old run's values while
    the tables beside them carried the new one.  Hand-checking is what failed,
    so it is not the fix.  Every quantity quoted in the text is computed here
    and written as ``\\newcommand``; the manuscript refers to the macro and can
    no longer disagree with the tables.

    Adding a number to the prose means adding it here first.  That friction is
    the point.
    """
    from scipy import stats as _st

    macros: dict[str, str] = {}

    def put(name: str, value, nd: int = 3, pct: bool = False, sci: bool = False):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            macros[name] = "??"
        elif pct:
            macros[name] = f"{value:.0%}".replace("%", "\\%")
        elif sci:
            macros[name] = f"{value:.1e}".replace("e-0", "\\times 10^{-").replace("e-", "\\times 10^{-") + "}"
        else:
            macros[name] = f"{value:.{nd}f}"

    primary, _flagged = split_strata(diagnostics, res)
    # ``sel`` drives every pooled statistic and is restricted to real data, so
    # that no headline number is part-generator.  ``sel_all`` keeps the
    # synthetic designs and is used only where the point *is* the contrast
    # between independent-by-construction and real data.
    sel_all = res[res["dataset"].isin(primary)]
    pool = real if real else primary
    sel = res[res["dataset"].isin(pool)]
    macros["numNPrimary"] = str(len(primary))
    macros["numNFlagged"] = str(len(_flagged))
    macros["numNAll"] = str(len(primary) + len(_flagged))

    # Synthetic bookkeeping: designs, draws per design, and the resulting count.
    src = raw_res if raw_res is not None else res
    all_names = sorted(set(src["dataset"].astype(str)))
    synth = [d for d in all_names if d.startswith("syn_")]
    designs = sorted({design_of(d) for d in synth})
    macros["numNSynth"] = str(len(synth))
    macros["numNSynthDesign"] = str(len(designs))
    if designs:
        per = [sum(1 for d in synth if design_of(d) == g) for g in designs]
        macros["numNDraws"] = str(max(set(per), key=per.count))
    # Share of non-PDP points that exceed the additive PDP reconstruction.
    ceil = _sel(sel, family="ceiling", target="fhat", split="test")
    c = _per_repeat(ceil, ["dataset", "model"], "additivity_r2").rename(columns={"value": "ceiling"})
    ach = _sel(sel, target="fhat", split="test", variant="curve-unit")
    ach = ach[ach["family"].isin(METHODS)]
    aa = _per_repeat(ach, ["dataset", "model", "family"], "r2").rename(columns={"value": "achieved"})
    mm = aa.merge(c, on=["dataset", "model", "repeat"], how="inner")
    if not mm.empty:
        agg = mm.groupby(["dataset", "model", "family"])[["ceiling", "achieved"]].mean().reset_index()
        other = agg[agg["family"] != "PDP"]
        if len(other):
            put("numAttainAbove", 100.0 * float((other["achieved"] > other["ceiling"]).mean()),
                nd=0, pct=False)
            macros["numAttainAbove"] = macros["numAttainAbove"] + "\\%"

    # Discrimination experiment: headline numbers for the prose.
    acc = _discrimination_accuracy(sel)
    if not acc.empty:
        rungs = sorted(acc["rung"].unique())
        last = rungs[-1] if rungs else None
        for metric, tag in [("proposed_r2", "Proposed"), ("faithfulness_corr", "Faith"),
                            ("infidelity", "Infid"), ("max_sensitivity", "Sens"),
                            ("complexity", "Cplx"), ("sparseness", "Spars")]:
            g = acc[acc["metric"] == metric]
            if g.empty:
                continue
            per_rung = [float(g[g["rung"] == r]["accuracy"].mean()) for r in rungs
                        if len(g[g["rung"] == r])]
            if per_rung:
                put(f"numDisc{tag}Mean", float(np.mean(per_rung)))
            if last is not None and len(g[g["rung"] == last]):
                put(f"numDisc{tag}Perm", float(g[g["rung"] == last]["accuracy"].mean()))
        # best established alternative, by mean across rungs
        best, best_v = None, -np.inf
        for metric in DISCRIM_DIRECTION:
            if metric == "proposed_r2":
                continue
            g = acc[acc["metric"] == metric]
            if g.empty:
                continue
            v = float(np.mean([g[g["rung"] == r]["accuracy"].mean() for r in rungs
                               if len(g[g["rung"] == r])]))
            if v > best_v:
                best, best_v = metric, v
        if best is not None:
            put("numDiscBestOtherMean", best_v)
            macros["numDiscBestOtherName"] = DISCRIM_LABEL[best]
            # Can the proposed measure and the best alternative actually be
            # separated?  Pair them by dataset, averaging rungs within a
            # dataset first so the unit of replication stays the dataset.
            pr = (acc[acc["metric"] == "proposed_r2"]
                  .groupby("dataset", observed=True)["accuracy"].mean())
            bo = (acc[acc["metric"] == best]
                  .groupby("dataset", observed=True)["accuracy"].mean())
            common = pr.index.intersection(bo.index)
            if len(common) > 2:
                a, b = pr.loc[common].to_numpy(), bo.loc[common].to_numpy()
                put("numDiscVsBestDiff", float(np.mean(a - b)))
                put("numDiscVsBestP", float(_st.ttest_rel(a, b).pvalue), nd=3)
                macros["numDiscVsBestWins"] = f"{int((a > b).sum())}/{len(common)}"
                macros["numDiscNDatasets"] = str(len(common))

    # Baseline scores against the original outcome, so the discussion cannot
    # quote a stale AUC.
    bl = sel[(sel["target"] == "y") & (sel["split"] == "test")]
    metric_y = "auc" if (bl["metric"] == "auc").any() else "r2"
    bl = bl[bl["metric"] == metric_y]
    for fam, var, tag in [("blackbox", None, "BlackBox"),
                          ("baseline", "spline-gam", "Spline"),
                          ("baseline", "linear-raw", "Linear"),
                          ("baseline", "intercept", "Intercept")]:
        g = bl[bl["family"] == fam]
        if var is not None:
            g = g[g["variant"] == var]
        if len(g):
            put(f"numY{tag}", float(g["value"].mean()))
    for fam in METHODS:
        g = bl[(bl["family"] == fam) & (bl["variant"] == "curve-unit")]
        if len(g):
            put(f"numY{fam}", float(g["value"].mean()))

    reps = src[src["dataset"].astype(str).str.startswith("syn_")]
    if not reps.empty and "repeat" in reps.columns:
        macros["numRepeatsSynth"] = str(int(reps["repeat"].nunique()))
    real_names = [d for d in all_names if not d.startswith("syn_")]
    macros["numNRealAll"] = str(len(real_names))

    def cell_table(variant, families, metric="r2", target="fhat", source=None):
        src = sel if source is None else source
        s = src[(src["target"] == target) & (src["split"] == "test")
                & (src["metric"] == metric) & (src["variant"] == variant)
                & (src["family"].isin(families))]
        if s.empty:
            return None
        return s.pivot_table(index=["dataset", "model", "repeat"],
                             columns="family", values="value")

    # -- 1. null controls ---------------------------------------------------
    ctl = sel[(sel["target"] == "fhat") & (sel["split"] == "test") & (sel["metric"] == "r2")]
    for fam in ("PDP", "SHAP"):
        for var, tag in (("curve-unit", "Intact"), ("null-permuted", "Perm")):
            v = ctl[(ctl["family"] == fam) & (ctl["variant"] == var)]["value"]
            put(f"num{fam.title()}{tag}", float(v.mean()) if len(v) else np.nan)

    # -- 2. headline comparison, dataset as the unit ------------------------
    piv = cell_table("curve-unit", METHODS)
    if piv is not None:
        ds = piv.groupby(level="dataset").mean()
        for a, b, tag in [("SHAP", "PDP", "ShapPdp"), ("SHAP", "ALE", "ShapAle"),
                          ("ALE", "PDP", "AlePdp")]:
            if a in ds and b in ds:
                d = ds[a] - ds[b]
                put(f"num{tag}Diff", float(d.mean()))
                put(f"num{tag}P", float(_st.ttest_rel(ds[a], ds[b]).pvalue), nd=3)
                macros[f"num{tag}Wins"] = f"{int((d > 0).sum())}/{len(d)}"
                # the pseudoreplicated version, quoted to show the inflation
                put(f"num{tag}PCell", float(_st.ttest_rel(piv[a], piv[b]).pvalue), sci=True)

    # -- 3. stratified by feature dependence (the mechanism) ----------------
    # This is the one comparison that needs the synthetic designs: independent
    # features cannot be obtained from real data, so the contrast is only
    # available by construction.
    piv_all = cell_table("curve-unit", METHODS, source=sel_all)
    if piv_all is not None:
        ds = piv_all.groupby(level="dataset").mean()
        real_ds = [d for d in ds.index if not str(d).startswith("syn_")]
        for tag, grp in [("Indep", INDEP_SYNTH), ("Corr", CORR_SYNTH), ("Real", real_ds)]:
            g = ds.loc[[x for x in grp if x in ds.index]]
            if len(g) < 2:
                continue
            macros[f"numN{tag}"] = str(len(g))
            for a, b, nm in [("SHAP", "PDP", "ShapPdp"), ("SHAP", "ALE", "ShapAle")]:
                if a not in g or b not in g:
                    continue
                d = g[a] - g[b]
                put(f"num{nm}{tag}", float(d.mean()))
                macros[f"num{nm}{tag}Wins"] = f"{int((d > 0).sum())}/{len(g)}"
                if len(g) > 2:
                    put(f"num{nm}{tag}P", float(_st.ttest_rel(g[a], g[b]).pvalue), nd=3)

    # -- 4. aggregator control ---------------------------------------------
    pols = cell_table("curve-ols", METHODS + ["CondMean"])
    if pols is not None:
        ds = pols.groupby(level="dataset").mean()
        for f in ["PDP", "ALE", "SHAP", "CondMean"]:
            if f in ds:
                put(f"numOls{f.replace('-', '')}", float(ds[f].mean()))
        real_ds = [d for d in ds.index if not str(d).startswith("syn_")]
        for f in ["PDP", "ALE", "SHAP"]:
            if f in ds and "CondMean" in ds:
                d = ds[f] - ds["CondMean"]
                put(f"num{f}VsCond", float(d.mean()))
                put(f"num{f}VsCondP", float(_st.ttest_rel(ds[f], ds["CondMean"]).pvalue), nd=4)
                macros[f"num{f}VsCondWins"] = f"{int((d > 0).sum())}/{len(d)}"
        if "CondMean" in ds and "PDP" in ds:
            # ``ds`` is already restricted to real data, so this is all of it.
            g = ds
            put("numCondVsPdpReal", float((g["CondMean"] - g["PDP"]).mean()))
            put("numCondVsPdpRealP", float(_st.ttest_rel(g["CondMean"], g["PDP"]).pvalue), nd=4)
            macros["numCondVsPdpRealWins"] = f"{int((g['CondMean'] > g['PDP']).sum())}/{len(g)}"
            put("numShapVsCondReal", float((g["SHAP"] - g["CondMean"]).mean()))
            put("numShapVsCondRealP", float(_st.ttest_rel(g["SHAP"], g["CondMean"]).pvalue), nd=4)
            macros["numShapVsCondRealWins"] = f"{int((g['SHAP'] > g['CondMean']).sum())}/{len(g)}"
    pu = cell_table("curve-unit", ["CondMean"])
    if pu is not None and "CondMean" in pu:
        put("numCondMeanUnit", float(pu["CondMean"].mean()), nd=1)

    # -- 5. marginal vs conditional TreeSHAP, tree models only -------------
    for var, tag in (("curve-unit", "Unit"), ("curve-ols", "Ols")):
        p2 = cell_table(var, ["SHAP", "SHAP-cond"])
        if p2 is None or "SHAP-cond" not in p2:
            continue
        p2 = p2.dropna()
        ds2 = p2.groupby(level="dataset").mean().dropna()
        if len(ds2) < 3:
            continue
        put(f"numTreeShapDiff{tag}", float((ds2["SHAP-cond"] - ds2["SHAP"]).mean()), nd=4)
        put(f"numTreeShapP{tag}", float(_st.ttest_rel(ds2["SHAP-cond"], ds2["SHAP"]).pvalue), nd=2)

    # -- 6. refitting gain, dataset as the unit -----------------------------
    pg = cell_table("curve-ols", ["PDP"])
    pgu = cell_table("curve-unit", ["PDP"])
    if pg is not None and pgu is not None:
        gap = (pg["PDP"] - pgu["PDP"]).groupby(level="dataset").mean()
        gi = gap.loc[[x for x in INDEP_SYNTH if x in gap.index]]
        gc = gap.loc[[x for x in CORR_SYNTH if x in gap.index]]
        put("numGainIndep", float(gi.mean()), nd=4)
        put("numGainCorr", float(gc.mean()), nd=4)
        if len(gi) > 1 and len(gc) > 1:
            put("numGainP", float(_st.ttest_ind(gc, gi, equal_var=False).pvalue), sci=True)

    # -- 7. degeneracy identity --------------------------------------------
    idg = sel[(sel["variant"] == "shap-knn-identity") & (sel["metric"] == "rel_gap")]["value"]
    if len(idg):
        macros["numIdentityMedian"] = f"{float(idg.median()):.0e}".replace("e-", "\\times 10^{-") + "}"
        macros["numIdentityMax"] = f"{float(idg.max()):.0e}".replace("e-", "\\times 10^{-") + "}"
    for var, tag, sp in [("sum-idw-degenerate", "DegenTrain", "train"),
                         ("sum-idw-degenerate", "DegenTest", "test"),
                         ("blackbox-knn", "Knn", "test")]:
        v = sel[(sel["variant"] == var) & (sel["split"] == sp) & (sel["metric"] == "r2")
                & (sel["target"] == "fhat")]["value"]
        put(f"num{tag}", float(v.mean()) if len(v) else np.nan)

    # -- 8. established metrics --------------------------------------------
    ref = sel[(sel["target"] == "explanation") & (sel["split"] == "test")]
    prop = sel[(sel["target"] == "fhat") & (sel["split"] == "test")
               & (sel["variant"] == "curve-unit") & (sel["metric"] == "r2")
               & sel["family"].isin(METHODS)]
    keys = ["dataset", "model", "family", "repeat"]
    if not ref.empty and not prop.empty:
        p = prop.groupby(keys, observed=True)["value"].mean().rename("proposed").reset_index()
        for metric, tag in [("infidelity", "Infid"), ("faithfulness_corr", "Faith"),
                            ("max_sensitivity", "Sens"), ("complexity", "Cplx"),
                            ("sparseness", "Spars")]:
            r = ref[ref["metric"] == metric].groupby(keys, observed=True)["value"].mean()
            m = p.merge(r.rename(metric).reset_index(), on=keys, how="inner").dropna()
            if len(m) < 5:
                continue
            put(f"numRho{tag}", float(_st.spearmanr(m["proposed"], m[metric]).statistic))
            within = []
            for _, g in m.groupby(["dataset", "model", "repeat"], observed=True):
                if g["family"].nunique() >= 3:
                    v = _st.spearmanr(g["proposed"], g[metric]).statistic
                    if np.isfinite(v):
                        within.append(v)
            if within:
                mn, lo, hi, _ = mean_ci(np.asarray(within))
                put(f"numRhoWithin{tag}", float(mn))
                put(f"numRhoWithin{tag}Lo", float(lo))
                put(f"numRhoWithin{tag}Hi", float(hi))

    # -- 9. magic_telescope, per model (never pooled) -----------------------
    mt = res[(res["dataset"] == "magic_telescope") & (res["family"] == "ceiling")
             & (res["split"] == "test") & (res["metric"] == "additivity_r2")]
    for m_, tag in [("random_forest", "Rf"), ("svm_rbf", "Svm")]:
        v = mt[mt["model"] == m_]["value"]
        put(f"numMagicPdp{tag}", float(v.mean()) if len(v) else np.nan)
    mtc = res[(res["dataset"] == "magic_telescope") & (res["target"] == "fhat")
              & (res["split"] == "test") & (res["metric"] == "r2")
              & (res["variant"] == "curve-unit") & (res["model"] == "random_forest")]
    for fam, tag in [("ALE", "Ale"), ("SHAP", "Shap")]:
        v = mtc[mtc["family"] == fam]["value"]
        put(f"numMagic{tag}Rf", float(v.mean()) if len(v) else np.nan)

    out.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by Code/xaieval/report.py -- do not edit by hand.",
             "% Every number quoted in the manuscript prose is defined here.",
             "% Adding a number to the text means adding it here first.", ""]
    for k in sorted(macros):
        lines.append(f"\\newcommand{{\\{k}}}{{{macros[k]}}}")
    (out / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {len(macros)} prose macros -> {out/'numbers.tex'}")
