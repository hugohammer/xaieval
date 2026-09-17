"""Experiment orchestration.

One repeat = one stratified train/test split.  Within a repeat, for each black
box:

1.  fit the model on the encoded training split;
2.  compute all four explanations on the training split only;
3.  build every explanation-derived predictor, fit it on the training split,
    score it on both splits against both targets;
4.  score the baselines and the null/corruption controls the same way;
5.  measure the additivity ceiling and the established reference metrics.

Everything lands in one long-format table (``Results/raw/results.csv``) with the
columns ``dataset, model, repeat, family, variant, target, split, metric,
value``.  Reporting reads only that file, so tables and figures can be
regenerated without re-running anything.
"""

from __future__ import annotations

import json
import re
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from . import anova, controls, refmetrics, scoring
from .baselines import InterceptOnly, LinearRaw, SplineGAM
from .blackbox import fit_blackbox
from .config import ExperimentConfig
from .curves import CurveSet
from .datasets import Dataset, discover_datasets, load_dataset
from .explainers import (
    compute_ale,
    compute_lime,
    compute_pdp,
    compute_shap,
    curve_attributions,
    dependence_curves,
)
from .predictors import (
    AdditiveCurveSurrogate,
    AttributionSumIDW,
    BlackBoxKNN,
    LocalModelIDW,
)
from .preprocessing import design_matrix_diagnostics, fit_transform

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------
# Row accumulation
# --------------------------------------------------------------------------


@dataclass
class Rows:
    """Accumulates long-format result rows."""

    items: list[dict] = field(default_factory=list)

    def add(self, base: dict, metrics: dict) -> None:
        for metric, value in metrics.items():
            if value is None:
                continue
            self.items.append({**base, "metric": metric, "value": float(value)})

    def extend(self, other: "Rows") -> None:
        self.items.extend(other.items)


def _score_predictor(
    rows: Rows,
    base: dict,
    pred,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    f_tr: np.ndarray,
    f_te: np.ndarray,
    task: str,
    verbose: bool = False,
) -> dict[str, float]:
    """Fit against both targets and record train/test metrics for each.

    Returns the test metrics against the black box, which the caller uses for
    the attainment ratio.
    """
    out: dict[str, float] = {}

    # Target 1: the original outcome.
    try:
        pred.fit(X_tr, y_tr)
        rows.add({**base, "target": "y", "split": "train"},
                 scoring.score_against_y(y_tr, pred.predict(X_tr), task))
        rows.add({**base, "target": "y", "split": "test"},
                 scoring.score_against_y(y_te, pred.predict(X_te), task))
    except Exception as exc:  # a failed variant must not kill the repeat
        rows.add({**base, "target": "y", "split": "test"}, {"failed": 1.0})
        if verbose:
            print(f"    [warn] {base.get('family')}/{base.get('variant')} y-target failed: {exc}")

    # Target 2: the black box's own score (out-of-sample global fidelity).
    try:
        pred.fit(X_tr, f_tr)
        rows.add({**base, "target": "fhat", "split": "train"},
                 scoring.score_against_blackbox(f_tr, pred.predict(X_tr)))
        te = scoring.score_against_blackbox(f_te, pred.predict(X_te))
        rows.add({**base, "target": "fhat", "split": "test"}, te)
        out = te
    except Exception as exc:
        rows.add({**base, "target": "fhat", "split": "test"}, {"failed": 1.0})
        if verbose:
            print(f"    [warn] {base.get('family')}/{base.get('variant')} fhat-target failed: {exc}")

    return out


# --------------------------------------------------------------------------
# One (dataset, repeat) unit of work
# --------------------------------------------------------------------------


def run_repeat(
    ds: Dataset,
    repeat: int,
    cfg: ExperimentConfig,
    tuned_params: dict[str, dict],
    export_curves_to: Path | None = None,
) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """Run every black box and every predictor on one split.

    Returns ``(result_rows, diagnostic_rows, newly_tuned_params)``.  When
    ``export_curves_to`` is given, the first model's curve set is dumped there
    for the illustrative curve-comparison figure.
    """
    rows = Rows()
    diagnostics: list[dict] = []
    newly_tuned: dict[str, dict] = {}

    seed = cfg.random_state + 1000 * repeat
    rng = np.random.default_rng(seed)

    stratify = ds.y if ds.task == "classification" else None
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=cfg.test_size, random_state=seed, stratify=stratify
    )
    Z_tr, Z_te, space, _pre = fit_transform(ds, X_tr_raw, X_te_raw)

    if repeat == 0:
        diagnostics.append(
            {
                "dataset": ds.name,
                "kind": "design_matrix",
                **space.diagnostics(),
                **design_matrix_diagnostics(Z_tr),
            }
        )

    for model_name in cfg.blackbox.models:
        t0 = time.time()
        key = f"{ds.name}|{model_name}"
        fixed = tuned_params.get(key)
        try:
            bb = fit_blackbox(
                model_name,
                Z_tr,
                y_tr,
                ds.task,
                output_scale=cfg.output_scale,
                tuning=cfg.blackbox.tuning if cfg.blackbox.tuning != "per_dataset" or fixed is None else "none",
                n_search_iter=cfg.blackbox.n_search_iter,
                cv_folds=cfg.blackbox.cv_folds,
                random_state=seed,
                fixed_params=fixed,
                n_jobs=1,
            )
        except Exception as exc:
            print(f"  [error] {ds.name}/{model_name} rep{repeat}: black box failed: {exc}")
            continue

        if cfg.blackbox.tuning == "per_dataset" and fixed is None and bb.best_params:
            newly_tuned[key] = bb.best_params

        base_ctx = {"dataset": ds.name, "model": model_name, "repeat": repeat}
        f_tr, f_te = bb.score(Z_tr), bb.score(Z_te)

        # Scale health.  On the log-odds scale a tree ensemble's exact 0/1
        # probabilities pin to the clip; when that happens to a large share of
        # predictions the score degenerates and every downstream quantity built
        # on it is an artefact.  Recorded so it is visible in the results rather
        # than diagnosed later from an impossible ceiling.
        rows.add(
            {**base_ctx, "family": "scale", "variant": "diagnostic",
             "target": "fhat", "split": "train"},
            {
                "saturated_fraction": bb.saturated_fraction(Z_tr),
                "score_sd": float(np.std(f_tr)),
                "logit_clip": float(bb.logit_clip),
            },
        )

        # -- black-box performance in its own right ------------------------
        # Scored on the probability scale: on the log-odds scale an R^2 against
        # a 0/1 outcome is meaningless (the ranges differ by an order of
        # magnitude), even though AUC would be unaffected.
        rows.add(
            {**base_ctx, "family": "blackbox", "variant": "model", "target": "y", "split": "test"},
            scoring.score_against_y(y_te, bb.score_to_proba(f_te), ds.task),
        )
        rows.add(
            {**base_ctx, "family": "blackbox", "variant": "model", "target": "y", "split": "train"},
            scoring.score_against_y(y_tr, bb.score_to_proba(f_tr), ds.task),
        )

        export_here = export_curves_to if (export_curves_to and model_name == cfg.blackbox.models[0]) else None
        try:
            rows_r, diag_r = _run_explanations(
                rows, base_ctx, bb, ds, space, Z_tr, Z_te, y_tr, y_te, f_tr, f_te, cfg, rng,
                export_curves_to=export_here,
            )
            diagnostics.extend(diag_r)
        except Exception:
            print(f"  [error] {ds.name}/{model_name} rep{repeat} explanations:\n{traceback.format_exc()}")

        if cfg.verbose:
            print(f"    {ds.name}/{model_name} rep{repeat} done in {time.time()-t0:.1f}s", flush=True)

    return rows.items, diagnostics, newly_tuned


def _run_explanations(
    rows: Rows,
    base_ctx: dict,
    bb,
    ds: Dataset,
    space,
    Z_tr,
    Z_te,
    y_tr,
    y_te,
    f_tr,
    f_te,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
    export_curves_to: Path | None = None,
) -> tuple[Rows, list[dict]]:
    diagnostics: list[dict] = []
    ecfg, pcfg = cfg.explainer, cfg.predictor

    # ---- compute the four explanations on the training split only --------
    # Each explainer is isolated: a library failure on one method must not
    # silently discard the other three.
    curvesets: dict[str, CurveSet] = {}
    shap_set = None
    lime_set = None

    def _try(label, fn):
        try:
            return fn()
        except Exception as exc:
            print(f"  [warn] {base_ctx['dataset']}/{base_ctx['model']} rep{base_ctx['repeat']}: "
                  f"{label} failed: {type(exc).__name__}: {exc}")
            if cfg.verbose > 1:
                traceback.print_exc()
            return None

    cs_pdp = _try("PDP", lambda: compute_pdp(bb, Z_tr, space, ecfg, rng))
    if cs_pdp is not None:
        curvesets["PDP"] = cs_pdp

    cs_ale = _try("ALE", lambda: compute_ale(bb, Z_tr, space, ecfg))
    if cs_ale is not None:
        curvesets["ALE"] = cs_ale

    shap_set = _try("SHAP", lambda: compute_shap(bb, Z_tr, space, ecfg, rng))
    if shap_set is not None:
        curvesets["SHAP"] = dependence_curves(
            shap_set.X, shap_set.A, space, shap_set.baseline, ecfg, "SHAP"
        )

    # Conditional (tree_path_dependent) SHAP, for tree models only, recorded as a
    # separate family.  Marginal vs conditional is a confound for any comparison
    # against PDP -- a conditional estimator can track E[f | X_j] under feature
    # dependence, which a marginal one cannot -- so both are measured rather than
    # one being chosen silently.
    if ecfg.shap_also_conditional and bb.is_tree:
        sc = _try("SHAP-cond", lambda: compute_shap(
            bb, Z_tr, space, ecfg, rng, tree_perturbation="tree_path_dependent"))
        if sc is not None:
            curvesets["SHAP-cond"] = dependence_curves(
                sc.X, sc.A, space, sc.baseline, ecfg, "SHAP-cond"
            )

    # THE control for the shared-aggregator fairness claim.  Feed the *black
    # box's own predictions* through the identical binning used for SHAP and
    # LIME: curve_j = binned mean of f(x) against x_j, i.e. an estimate of
    # E[f | X_j].  No explanation is involved at any point.  If this matches or
    # beats an explanation-derived predictor, that predictor's score is
    # attributable to the aggregator rather than to the explanation, and the
    # "shared construction" fairness argument fails.
    def _condmean():
        f_all = bb.score(Z_tr)
        A = np.repeat(f_all[:, None], space.p, axis=1)
        return dependence_curves(Z_tr, A, space, float(np.mean(f_all)), ecfg, "CondMean")

    cs_cm = _try("CondMean", _condmean)
    if cs_cm is not None:
        curvesets["CondMean"] = cs_cm

    lime_set = _try("LIME", lambda: compute_lime(bb, Z_tr, space, ecfg, rng))
    if lime_set is not None:
        curvesets["LIME"] = dependence_curves(
            lime_set.X, lime_set.attributions(centre=Z_tr.mean(axis=0)),
            space, lime_set.baseline, ecfg, "LIME",
        )

    if not curvesets:
        return rows, diagnostics

    if export_curves_to is not None:
        _export_curves(curvesets, space, export_curves_to)

    # ---- additivity ceiling ---------------------------------------------
    # Defined by the PDP curves, since the additive functional-ANOVA projection
    # is built from partial dependence by definition.
    ceiling_r2 = np.nan
    if "PDP" in curvesets:
        ceiling_tr = anova.additivity_ceiling(bb, Z_tr, curvesets["PDP"])
        ceiling_te = anova.additivity_ceiling(bb, Z_te, curvesets["PDP"])
        rows.add(
            {**base_ctx, "family": "ceiling", "variant": "additivity", "target": "fhat", "split": "test"},
            ceiling_te,
        )
        rows.add(
            {**base_ctx, "family": "ceiling", "variant": "additivity", "target": "fhat", "split": "train"},
            ceiling_tr,
        )
        ceiling_r2 = ceiling_te["additivity_r2"]

    # ---- the common construction: additive curve surrogate ---------------
    for family, cs in curvesets.items():
        for mode in pcfg.curve_fit_modes:
            base = {**base_ctx, "family": family, "variant": f"curve-{mode}"}
            te = _score_predictor(
                rows, base, AdditiveCurveSurrogate(cs, mode=mode, family=family),
                Z_tr, Z_te, y_tr, y_te, f_tr, f_te, ds.task, verbose=cfg.verbose > 1,
            )
            if te:
                rows.add(
                    {**base_ctx, "family": family, "variant": f"curve-{mode}",
                     "target": "fhat", "split": "test"},
                    {"attainment": anova.attainment_ratio(te.get("r2", np.nan), ceiling_r2)},
                )

    # ---- LIME's native construction --------------------------------------
    if lime_set is not None:
        _score_predictor(
            rows,
            {**base_ctx, "family": "LIME", "variant": "local-idw"},
            LocalModelIDW(lime_set, power=pcfg.idw_power, tau=pcfg.idw_tau, eps=pcfg.idw_eps),
            Z_tr, Z_te, y_tr, y_te, f_tr, f_te, ds.task, verbose=cfg.verbose > 1,
        )

    # ---- the degenerate SHAP construction, and its proof of degeneracy ---
    if shap_set is not None:
        shap_sum_pred = AttributionSumIDW(
            X_anchor=shap_set.X,
            sums=shap_set.sums,
            baseline=shap_set.baseline,
            power=pcfg.idw_power,
            tau=pcfg.idw_tau,
            eps=pcfg.idw_eps,
        )
        _score_predictor(
            rows,
            {**base_ctx, "family": "SHAP", "variant": "sum-idw-degenerate"},
            shap_sum_pred, Z_tr, Z_te, y_tr, y_te, f_tr, f_te, ds.task, verbose=cfg.verbose > 1,
        )
        knn = BlackBoxKNN(
            X_anchor=shap_set.X,
            scores=bb.score(shap_set.X),
            power=pcfg.idw_power,
            tau=pcfg.idw_tau,
            eps=pcfg.idw_eps,
        )
        _score_predictor(
            rows,
            {**base_ctx, "family": "control", "variant": "blackbox-knn"},
            knn, Z_tr, Z_te, y_tr, y_te, f_tr, f_te, ds.task, verbose=cfg.verbose > 1,
        )
        # The identity claim, measured rather than asserted.
        gap = float(np.max(np.abs(shap_sum_pred.predict(Z_te) - knn.predict(Z_te))))
        rel = gap / (float(np.std(f_te)) + 1e-12)
        # The tautology is a statement about the *anchor* points: there the
        # zero-distance weight dominates and the fit is exact by construction.
        # Measured on the anchors themselves, not on the whole training split,
        # which may contain rows that were not used as anchors.
        f_anchor = bb.score(shap_set.X)
        rows.add(
            {**base_ctx, "family": "control", "variant": "shap-knn-identity",
             "target": "fhat", "split": "test"},
            {
                "max_abs_gap": gap,
                "rel_gap": rel,
                "r2_at_anchors": scoring.r2_score_manual(f_anchor, shap_sum_pred.predict(shap_set.X)),
                "anchor_fraction": float(len(shap_set.X)) / float(len(Z_tr)),
            },
        )

    # ---- LIME kernel-width sensitivity -----------------------------------
    # Reported separately because the default width is an arbitrary library
    # constant (0.75*sqrt(p)) that materially changes how local the local model
    # actually is.
    for kw in ecfg.lime_kernel_width_sweep:
        swept = dict(ecfg.__dict__)
        swept["lime_kernel_width"] = float(kw)
        sweep_cfg = type(ecfg)(**swept)
        lm = _try(f"LIME(kw={kw})", lambda c=sweep_cfg: compute_lime(bb, Z_tr, space, c, rng))
        if lm is None:
            continue
        cs_kw = dependence_curves(
            lm.X, lm.attributions(centre=Z_tr.mean(axis=0)), space, lm.baseline, ecfg, "LIME"
        )
        # How well does each local model reproduce the black box at its own
        # anchor?  A low value means the "local" approximation is not local.
        own = lm.intercepts + np.einsum("ij,ij->i", lm.coefs, lm.X)
        rows.add(
            {**base_ctx, "family": "LIME", "variant": f"kernel-{kw:g}",
             "target": "fhat", "split": "train"},
            {"anchor_fit_r2": scoring.r2_score_manual(bb.score(lm.X), own)},
        )
        _score_predictor(
            rows,
            {**base_ctx, "family": "LIME", "variant": f"kernel-{kw:g}"},
            AdditiveCurveSurrogate(cs_kw, mode="unit", family="LIME"),
            Z_tr, Z_te, y_tr, y_te, f_tr, f_te, ds.task, verbose=cfg.verbose > 1,
        )

    # ---- baselines --------------------------------------------------------
    # The black box itself is not in this loop: against ``fhat`` it is trivially
    # R^2 = 1, and against ``y`` it is already recorded above on the correct
    # (probability) scale.
    for pred in (
        InterceptOnly(),
        LinearRaw(),
        SplineGAM(continuous_idx=space.continuous_idx),
    ):
        _score_predictor(
            rows,
            {**base_ctx, "family": "baseline", "variant": pred.variant},
            pred, Z_tr, Z_te, y_tr, y_te, f_tr, f_te, ds.task, verbose=cfg.verbose > 1,
        )

    # ---- null explanations and corruption sweep ---------------------------
    if cfg.control.run_controls:
        _run_controls(rows, base_ctx, curvesets, Z_tr, Z_te, y_tr, y_te, f_tr, f_te, ds.task, cfg, rng)

    # ---- established metrics ---------------------------------------------
    if cfg.refmetric.run_ref_metrics:
        diagnostics.extend(
            _run_refmetrics(rows, base_ctx, bb, curvesets, Z_te, cfg, rng)
        )

    # ---- evaluation metrics against each other ----------------------------
    if cfg.discrimination.run_discrimination:
        _run_discrimination(rows, base_ctx, bb, curvesets, Z_tr, Z_te,
                            y_tr, y_te, f_tr, f_te, ds.task, cfg, rng)

    return rows, diagnostics


def _export_curves(curvesets: dict[str, CurveSet], space, path: Path, n_features: int = 3) -> None:
    """Dump a few continuous features' curves for the illustrative figure.

    Continuous features only: for a binary column every method produces a
    two-point curve, which shows nothing.
    """
    cont = [j for j in range(space.p) if not space.is_binary[j]]
    if not cont:
        cont = list(range(min(n_features, space.p)))
    # Pick the features with the largest PDP amplitude -- the ones worth showing.
    pdp = curvesets.get("PDP")
    if pdp is not None:
        cont = sorted(cont, key=lambda j: -pdp.curves[j].amplitude)
    chosen = cont[:n_features]

    payload = {"features": []}
    for j in chosen:
        entry = {"name": space.names[j]}
        for meth, cs in curvesets.items():
            c = cs.curves[j]
            entry[meth] = {"grid": c.grid.tolist(), "values": c.values.tolist()}
        payload["features"].append(entry)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _run_controls(rows, base_ctx, curvesets, Z_tr, Z_te, y_tr, y_te, f_tr, f_te, task, cfg, rng):
    """Null explanations plus the noise sweep, averaged over draws."""
    ccfg = cfg.control
    for family, cs in curvesets.items():
        # Sharp nulls.
        for label, maker in (
            ("null-permuted", controls.permute_curves),
            ("null-randomised", controls.randomise_curves),
        ):
            for draw in range(ccfg.n_control_draws):
                cs_null = maker(cs, rng)
                _score_predictor(
                    rows,
                    {**base_ctx, "family": family, "variant": label, "draw": draw},
                    AdditiveCurveSurrogate(cs_null, mode="unit", family=family),
                    Z_tr, Z_te, y_tr, y_te, f_tr, f_te, task,
                )

        # Graded corruption.
        for sigma in ccfg.noise_levels:
            for draw in range(ccfg.n_control_draws if sigma > 0 else 1):
                cs_noisy = controls.noisy_curves(cs, sigma, rng)
                _score_predictor(
                    rows,
                    {**base_ctx, "family": family, "variant": f"noise-{sigma:g}",
                     "noise_sigma": sigma, "draw": draw},
                    AdditiveCurveSurrogate(cs_noisy, mode="unit", family=family),
                    Z_tr, Z_te, y_tr, y_te, f_tr, f_te, task,
                )


def _run_discrimination(rows, base_ctx, bb, curvesets, Z_tr, Z_te, y_tr, y_te,
                        f_tr, f_te, task, cfg, rng):
    """Score every evaluation metric on a quality ladder of known ordering.

    For each explanation family we build rungs of decreasing quality -- intact,
    increasingly noisy, then permuted -- and record *all* the evaluation
    metrics on each rung: the five established ones plus our own test $R^2$.
    Because the ordering of the rungs is fixed by construction, the report stage
    can ask of each metric how often it ranks a better rung above a worse one,
    which is the comparison between evaluation metrics that the metrics
    literature does not usually run.
    """
    dcfg = cfg.discrimination
    rcfg = cfg.refmetric
    n = min(dcfg.n_points, Z_te.shape[0])
    idx = rng.choice(Z_te.shape[0], size=n, replace=False) if n < Z_te.shape[0] else np.arange(n)
    Xs = Z_te[idx]

    for family, cs in curvesets.items():
        for level in dcfg.levels:
            # An intact explanation is deterministic, so one draw is all there is.
            n_draws = 1 if level == 0.0 else dcfg.n_draws
            for draw in range(n_draws):
                if level is None:
                    cs_l = controls.permute_curves(cs, rng)
                    rung, label = len(dcfg.levels) - 1, "permuted"
                elif level == 0.0:
                    cs_l, rung, label = cs, 0, "intact"
                else:
                    cs_l = controls.noisy_curves(cs, level, rng)
                    rung, label = dcfg.levels.index(level), f"noise{level:g}"

                base = {**base_ctx, "family": family, "variant": f"discrim-{label}",
                        "rung": rung, "draw": draw, "split": "test"}

                # Our measure on this rung: the unit-coefficient additive
                # surrogate, scored out of sample against the black box.
                try:
                    te = _score_predictor(
                        rows, {**base_ctx, "family": family,
                               "variant": f"discrimfit-{label}", "rung": rung, "draw": draw},
                        AdditiveCurveSurrogate(cs_l, mode="unit", family=family),
                        Z_tr, Z_te, y_tr, y_te, f_tr, f_te, task,
                    )
                    if "r2" in te:
                        rows.add({**base, "target": "discrimination"},
                                 {"proposed_r2": te["r2"]})
                except Exception as exc:
                    if cfg.verbose > 1:
                        print(f"    [warn] discrim fit {family}/{label} failed: {exc}")

                # The established metrics on the same rung.
                A = curve_attributions(cs_l, Xs)
                try:
                    rows.add({**base, "target": "discrimination"}, {
                        "infidelity": refmetrics.infidelity(
                            bb.score, Xs, A,
                            n_perturb=rcfg.infidelity_n_perturb,
                            sigma=rcfg.infidelity_sigma, rng=rng,
                        ),
                        "faithfulness_corr": refmetrics.faithfulness_correlation(
                            bb.score, Xs, A,
                            n_subsets=rcfg.faithfulness_n_subsets,
                            subset_frac=rcfg.faithfulness_subset_frac, rng=rng,
                        ),
                        "max_sensitivity": refmetrics.max_sensitivity(
                            lambda XX, _cs=cs_l: curve_attributions(_cs, XX), Xs, A,
                            n_perturb=rcfg.sensitivity_n_perturb,
                            radius=rcfg.sensitivity_radius, rng=rng,
                        ),
                        "complexity": refmetrics.complexity_entropy(A),
                        "sparseness": refmetrics.sparseness_gini(A),
                    })
                except Exception as exc:
                    if cfg.verbose > 1:
                        print(f"    [warn] discrim metrics {family}/{label} failed: {exc}")


def _run_refmetrics(rows, base_ctx, bb, curvesets, Z_te, cfg, rng) -> list[dict]:
    """Infidelity / faithfulness / sensitivity / complexity on the same explanations."""
    rcfg = cfg.refmetric
    n = min(rcfg.n_ref_points, Z_te.shape[0])
    idx = rng.choice(Z_te.shape[0], size=n, replace=False) if n < Z_te.shape[0] else np.arange(n)
    Xs = Z_te[idx]

    for family, cs in curvesets.items():
        A = curve_attributions(cs, Xs)
        base = {**base_ctx, "family": family, "variant": "refmetric", "target": "explanation", "split": "test"}
        try:
            rows.add(base, {
                "infidelity": refmetrics.infidelity(
                    bb.score, Xs, A,
                    n_perturb=rcfg.infidelity_n_perturb, sigma=rcfg.infidelity_sigma, rng=rng,
                ),
                "faithfulness_corr": refmetrics.faithfulness_correlation(
                    bb.score, Xs, A,
                    n_subsets=rcfg.faithfulness_n_subsets,
                    subset_frac=rcfg.faithfulness_subset_frac, rng=rng,
                ),
                "max_sensitivity": refmetrics.max_sensitivity(
                    lambda XX, _cs=cs: curve_attributions(_cs, XX), Xs, A,
                    n_perturb=rcfg.sensitivity_n_perturb, radius=rcfg.sensitivity_radius, rng=rng,
                ),
                "complexity": refmetrics.complexity_entropy(A),
                "sparseness": refmetrics.sparseness_gini(A),
            })
        except Exception as exc:
            if cfg.verbose > 1:
                print(f"    [warn] refmetrics {family} failed: {exc}")
    return []


# --------------------------------------------------------------------------
# Top-level driver
# --------------------------------------------------------------------------


def effective_workers(n_jobs: int) -> int:
    """How many worker processes joblib will actually start.

    Reported at startup so an under-utilised run is visible immediately rather
    than inferred from a wall-clock that feels too long.
    """
    import os

    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # not POSIX
        available = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 0:
        return 1
    return available + 1 + n_jobs if n_jobs < 0 else min(n_jobs, available)


def plan_summary(n_datasets: int, cfg: ExperimentConfig) -> str:
    """One-line description of the work and how well it will fill the machine."""
    workers = effective_workers(cfg.n_jobs)
    units = n_datasets * cfg.n_repeats * max(1, len(cfg.blackbox.models))
    phase2 = n_datasets * max(0, cfg.n_repeats - 1)
    # ``units`` above ignores the per-dataset repeat override, so it overstates
    # the work whenever replicated synthetic draws are in the suite; the caller
    # passes the true count when it knows it.
    note = ""
    if phase2 and phase2 < workers:
        note = (f"  [only {phase2} parallel job(s) in the main phase -- {workers - phase2} "
                f"worker(s) will idle; raise --repeats or add datasets to fill the machine]")
    elif n_datasets < workers and cfg.n_repeats <= 1:
        note = "  [single repeat: the run is effectively serial]"
    return (f"  workers   : {workers}\n"
            f"  work      : {n_datasets} dataset(s) x {cfg.n_repeats} repeat(s) "
            f"x {len(cfg.blackbox.models)} model(s) = {units} model-fit units{note}")


#: A replicated synthetic draw, e.g. ``syn_corr60_r07``.
_REPLICATE_RE = re.compile(r"^(?P<design>.+)_r(?P<rep>\d{2})$")


def replicate_of(name: str) -> tuple[str, int] | None:
    """``('syn_corr60', 7)`` for a replicated draw, else ``None``."""
    m = _REPLICATE_RE.match(str(name))
    return (m.group("design"), int(m.group("rep"))) if m else None


def repeats_for(name: str, cfg: ExperimentConfig) -> int:
    """Split repeats for one dataset (fewer for replicated synthetic draws)."""
    if cfg.n_repeats_replicated and replicate_of(name):
        return max(1, min(cfg.n_repeats, cfg.n_repeats_replicated))
    return cfg.n_repeats


def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    """Run every dataset x repeat and write the raw result table."""
    raw_dir = cfg.results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(cfg.results_dir / "config.json")

    paths = discover_datasets(cfg.data_dir, list(cfg.datasets) if cfg.datasets else None)
    if not paths:
        raise FileNotFoundError(f"no CSV files found in {cfg.data_dir}")

    all_rows: list[dict] = []
    all_diag: list[dict] = []

    datasets = [load_dataset(p) for p in paths]
    dataset_summaries = [ds.summary_row() for ds in datasets]
    for ds in datasets:
        print(f"  {ds.name}: n={ds.n}, p_raw={ds.p_raw}, task={ds.task}", flush=True)

    n_workers = effective_workers(cfg.n_jobs)

    # Parallelism runs across (dataset, repeat) pairs, not repeats within a
    # dataset.  Two phases, because repeat 0 of each dataset establishes the
    # per-dataset hyper-parameters that later repeats reuse:
    #
    #   Phase 1: repeat 0 of every dataset          -> n_datasets jobs
    #   Phase 2: every remaining (dataset, repeat)  -> n_datasets*(repeats-1)
    #
    # Nesting the parallel call inside a per-dataset loop instead would cap the
    # width at ``n_repeats - 1`` and leave repeat 0 of each dataset running
    # alone on one core -- with a per-dataset hyper-parameter search inside it.
    phase1 = [(ds, (raw_dir / "example_curves.json") if i == 0 else None)
              for i, ds in enumerate(datasets)]
    print(f"\nPhase 1/2: repeat 0 of {len(phase1)} dataset(s) on {n_workers} worker(s)"
          f"{' [tuning]' if cfg.blackbox.tuning == 'per_dataset' else ''}", flush=True)

    phase1_out = Parallel(n_jobs=cfg.n_jobs, verbose=max(0, cfg.verbose * 5))(
        delayed(run_repeat)(ds, 0, cfg, {}, export) for ds, export in phase1
    )

    tuned_by_dataset: dict[str, dict[str, dict]] = {}
    for ds, (r0, d0, tuned_new) in zip(datasets, phase1_out):
        all_rows.extend(r0)
        all_diag.extend(d0)
        tuned_by_dataset[ds.name] = tuned_new if cfg.blackbox.tuning == "per_dataset" else {}

    jobs = [(ds, rep) for ds in datasets for rep in range(1, repeats_for(ds.name, cfg))]
    if jobs:
        width = min(n_workers, len(jobs))
        print(f"Phase 2/2: {len(jobs)} (dataset, repeat) job(s) on {width} worker(s)", flush=True)
        phase2_out = Parallel(n_jobs=cfg.n_jobs, verbose=max(0, cfg.verbose * 5))(
            delayed(run_repeat)(ds, rep, cfg, tuned_by_dataset.get(ds.name, {}))
            for ds, rep in jobs
        )
        for r, d, _ in phase2_out:
            all_rows.extend(r)
            all_diag.extend(d)

    df = pd.DataFrame(all_rows)
    df = df.drop(columns=[c for c in ("_verbose",) if c in df.columns])
    df.to_csv(raw_dir / "results.csv", index=False)

    pd.DataFrame(dataset_summaries).to_csv(raw_dir / "datasets.csv", index=False)
    if all_diag:
        pd.DataFrame(all_diag).to_csv(raw_dir / "diagnostics.csv", index=False)

    # Record the synthetic ground truth alongside, for the validation figure.
    truth = []
    for path in paths:
        spec_path = path.with_suffix(".json")
        if spec_path.exists():
            spec = json.loads(spec_path.read_text())
            if "synthetic" in spec:
                truth.append({"dataset": spec.get("name", path.stem), **spec["synthetic"]})
    if truth:
        pd.DataFrame(truth).to_csv(raw_dir / "synthetic_truth.csv", index=False)

    print(f"\nWrote {len(df)} result rows to {raw_dir/'results.csv'}")
    return df
