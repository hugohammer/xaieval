"""Configuration objects for the explanation-derived-prediction experiments.

Everything that a reviewer might reasonably ask "what value did you use for X?"
lives here, so the answer is one file rather than scattered literals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence


# --------------------------------------------------------------------------
# Colour / style constants (validated CVD-safe categorical set, all-pairs)
# --------------------------------------------------------------------------

#: One hue per explanation family.  These four pass the all-pairs colour-vision
#: separation checks on a light (print) surface; a fifth categorical hue does
#: not, which is why baselines and reference lines are drawn in neutral grey
#: rather than a fifth colour.  Marker shape and line style duplicate the
#: colour encoding so the figures survive greyscale printing.
METHOD_COLOURS = {
    "PDP": "#2a78d6",
    "ALE": "#eb6834",
    "SHAP": "#1baf7a",
    "LIME": "#4a3aa7",
}
METHOD_MARKERS = {"PDP": "o", "ALE": "s", "SHAP": "^", "LIME": "D"}
METHOD_LINESTYLES = {"PDP": "-", "ALE": "--", "SHAP": "-.", "LIME": (0, (3, 1, 1, 1))}

GREY_DARK = "#52514e"
GREY_MID = "#9a9892"
GREY_LIGHT = "#c9c7c1"
INK = "#0b0b0b"


# --------------------------------------------------------------------------
# Experiment configuration
# --------------------------------------------------------------------------


@dataclass
class BlackBoxConfig:
    """Which black-box models to run, and how hard to tune them."""

    models: Sequence[str] = ("random_forest", "gradient_boosting", "mlp", "svm_rbf")

    #: ``"none"``    - use the documented default hyper-parameters, no search.
    #: ``"per_dataset"`` - one randomised search per (dataset, model), on the
    #:                 training split of repeat 0, reused for every repeat.
    #: ``"per_repeat"``  - a fresh search inside every repeat (correct but slow;
    #:                 use for the camera-ready run).
    tuning: str = "per_dataset"
    n_search_iter: int = 25
    cv_folds: int = 5
    random_state: int = 20260804


@dataclass
class ExplainerConfig:
    """Budgets for the four explanation methods."""

    # PDP / ALE ------------------------------------------------------------
    pdp_grid_size: int = 25
    #: Number of training rows used as the marginalisation background for PDP.
    #: PDP costs ``grid_size x background_size`` model calls per feature.
    pdp_background_size: int = 300
    ale_n_bins: int = 20

    # SHAP -----------------------------------------------------------------
    #: Rows explained by SHAP.  TreeSHAP is cheap; the model-agnostic path is
    #: not, and it is the one used whenever the explainer's native output scale
    #: does not match the scale being explained (notably a tree *classifier* on
    #: the log-odds scale -- see ``explainers.compute_shap``).
    shap_max_points: int = 800
    #: Background rows the masker averages over.  Cost is linear in this.
    shap_kernel_background: int = 50
    shap_kernel_nsamples: str | int = "auto"
    #: How TreeSHAP handles absent features.
    #:
    #: ``shap.TreeExplainer(model)`` with no ``data`` silently resolves to
    #: ``tree_path_dependent``, which uses *conditional* expectations along tree
    #: paths.  The model-agnostic fallback used for non-tree models uses an
    #: Independent masker, i.e. *marginal* expectations.  Mixing the two means
    #: SHAP is estimating a different quantity for tree and non-tree models, and
    #: -- worse for this study -- a conditional estimator can track
    #: E[f | X_j] under feature dependence in a way the marginal PDP provably
    #: cannot, so any SHAP advantage on dependent data is confounded with the
    #: estimator.  ``interventional`` (the default here) matches the marginal
    #: convention of PDP and of the fallback path.
    shap_tree_perturbation: str = "interventional"
    #: Additionally compute conditional (``tree_path_dependent``) SHAP for tree
    #: models, recorded as the separate family ``SHAP-cond``.  This is what makes
    #: the marginal-vs-conditional confound measurable inside one run.
    shap_also_conditional: bool = True

    #: Permutation-SHAP evaluations per explained row, as a multiple of ``p``.
    #: Local accuracy holds exactly from ``2p + 1`` (one antithetic permutation
    #: pair); larger values only reduce the variance of the individual
    #: attributions.  Since those attributions feed the dependence curves, some
    #: averaging is worth paying for -- but the cost is
    #: ``shap_max_points x mult x p x shap_kernel_background`` model calls, so
    #: this is the knob to turn first on a wide dataset.
    shap_agnostic_evals_mult: int = 6
    #: Warn when the model-agnostic path is estimated to exceed this many model
    #: evaluations, so a multi-hour stage is announced rather than discovered.
    shap_cost_warn_threshold: int = 20_000_000

    # LIME -----------------------------------------------------------------
    #: Number of training points around which a local model is fitted.  This is
    #: the dominant cost of the whole pipeline.
    lime_max_points: int = 300
    lime_n_samples: int = 1000
    #: ``None`` reproduces the LIME default of ``0.75 * sqrt(p)``.
    lime_kernel_width: float | None = None
    #: Optional sensitivity sweep.  The LIME default neighbourhood is wide --
    #: perturbations are drawn with the full training standard deviation around
    #: the instance -- so the "local" model is close to a globally weighted
    #: linear fit and often fails to reproduce the black box even at the point
    #: it explains.  Any single-width result is therefore a statement about the
    #: library default rather than about LIME, and the paper should show the
    #: sweep.  Empty tuple disables it.
    lime_kernel_width_sweep: tuple[float, ...] = ()
    #: The thesis used ``discretize_continuous=False`` so that each local model
    #: is a smooth function of the raw features.  Keep it.
    lime_discretize: bool = False

    # Dependence-curve smoother (SHAP / LIME additive construction) ---------
    #: Number of quantile bins used to smooth an attribution-vs-feature-value
    #: scatter into a one-dimensional curve.
    dependence_n_bins: int = 20
    #: Minimum rows per bin before neighbouring bins are merged.
    dependence_min_bin: int = 10


@dataclass
class PredictorConfig:
    """How explanation output is turned into a predictor."""

    #: Fitting modes for the additive curve surrogate.
    #: ``unit``  - all coefficients fixed at 1, intercept only.  Under feature
    #:             independence this is the L2-optimal additive projection of
    #:             the black box, so it is the theoretically motivated default.
    #: ``ols``   - unconstrained least squares on the transformed design matrix.
    #: ``ridge`` - ditto with leave-one-out-CV-selected L2 penalty.
    curve_fit_modes: Sequence[str] = ("unit", "ols", "ridge")

    #: Inverse-distance weighting for the LIME native construction.
    #: ``weight_i ∝ dist(x_i, x0) ** -idw_power``.
    idw_power: float = 2.0
    #: Distance beyond which a training anchor gets zero weight.  ``None`` means
    #: no truncation (the thesis used tau = infinity throughout).
    idw_tau: float | None = None
    #: Numerical floor so that a test point coinciding with a training point
    #: does not produce an infinite weight.
    idw_eps: float = 1e-12


@dataclass
class ControlConfig:
    """Null explanations and the corruption sweep that validate the measure."""

    run_controls: bool = True
    #: Noise levels, in units of the standard deviation of the quantity being
    #: corrupted (curve values or attributions).
    noise_levels: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    n_control_draws: int = 5


@dataclass
class RefMetricConfig:
    """Budgets for the established metrics we benchmark against."""

    run_ref_metrics: bool = True
    n_ref_points: int = 200
    infidelity_n_perturb: int = 50
    infidelity_sigma: float = 0.2
    faithfulness_n_subsets: int = 50
    faithfulness_subset_frac: float = 0.3
    sensitivity_n_perturb: int = 20
    sensitivity_radius: float = 0.1


@dataclass
class DiscriminationConfig:
    """The head-to-head test of *evaluation* metrics against each other.

    Every metric in the literature claims to say whether an explanation is any
    good.  That claim is testable: degrade an explanation by a known amount and
    see which metric notices.  We build a ladder of explanation quality whose
    ordering is exact by construction -- intact, then progressively noisier
    curves, then the sharp permuted null -- and score each metric by how often
    it ranks the better rung above the worse one.

    Difficulty comes from the rungs being close together, not from noising the
    data.  Noising the data is the other obvious route, and we do not take it:
    it changes what the *true* explanation is, so the ground-truth ordering
    that the whole comparison rests on would itself become an estimate.
    """

    run_discrimination: bool = True
    #: Corruption levels forming the quality ladder, best first.  ``0.0`` is the
    #: intact explanation; ``None`` marks the permuted sharp null.
    levels: Sequence[float | None] = (0.0, 0.5, 1.0, 2.0, None)
    #: Independent corruption draws per level.  The permuted null and the noisy
    #: rungs are random, so a single draw would confound metric quality with
    #: which draw was taken.
    n_draws: int = 3
    #: Test rows the reference metrics are evaluated on.  Smaller than
    #: ``RefMetricConfig.n_ref_points`` because this stage runs at every rung of
    #: the ladder and the cost is linear in it.
    n_points: int = 120


@dataclass
class ExperimentConfig:
    """Top-level configuration."""

    data_dir: Path = Path("Data")
    results_dir: Path = Path("Results")
    datasets: Sequence[str] | None = None  # None = every dataset in data_dir

    n_repeats: int = 20
    #: Repeats for a *replicated* synthetic design (a dataset whose name ends in
    #: ``_rNN``).  Those designs are drawn independently ``N_REPLICATES`` times,
    #: so draw-to-draw variation already supplies the error term and there is
    #: little left for extra splits of the same draw to buy.  ``None`` uses
    #: ``n_repeats`` for everything.
    n_repeats_replicated: int | None = 3
    test_size: float = 0.25
    random_state: int = 20260804

    #: For classification, the scale on which the black box is explained and
    #: reproduced.
    #:
    #: ``"probability"`` (default) is bounded in [0, 1] and cannot degenerate.
    #: ``"logit"`` is the more principled scale for additivity in the abstract --
    #: a model additive in log-odds looks non-additive in probability -- but it
    #: requires clipping a tree ensemble's exact 0/1 outputs, and on a
    #: near-separable problem most predictions pin to the clip and every
    #: downstream quantity becomes an artefact of it (see
    #: ``blackbox.resolution_clip``).  It is also ~30x more expensive, because
    #: TreeSHAP cannot be used.
    #:
    #: Run ``logit`` as a sensitivity analysis, and check the reported
    #: ``saturated_fraction`` before believing the numbers.
    output_scale: str = "probability"

    n_jobs: int = -1
    verbose: int = 1

    blackbox: BlackBoxConfig = field(default_factory=BlackBoxConfig)
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    refmetric: RefMetricConfig = field(default_factory=RefMetricConfig)
    discrimination: DiscriminationConfig = field(default_factory=DiscriminationConfig)

    # -- convenience -------------------------------------------------------

    def quick(self) -> "ExperimentConfig":
        """Shrink every budget for a smoke test.  Not for reported results."""
        self.n_repeats = 2
        self.blackbox.models = ("random_forest",)
        self.blackbox.tuning = "none"
        self.explainer.pdp_grid_size = 10
        self.explainer.pdp_background_size = 100
        self.explainer.ale_n_bins = 10
        self.explainer.shap_max_points = 200
        self.explainer.lime_max_points = 60
        self.explainer.lime_n_samples = 200
        self.control.noise_levels = (0.0, 0.5, 2.0)
        self.control.n_control_draws = 2
        self.refmetric.n_ref_points = 50
        self.refmetric.infidelity_n_perturb = 10
        self.refmetric.faithfulness_n_subsets = 10
        self.refmetric.sensitivity_n_perturb = 5
        self.discrimination.levels = (0.0, 1.0, None)
        self.discrimination.n_draws = 1
        self.discrimination.n_points = 40
        return self

    def to_json(self, path: Path) -> None:
        def _default(o):
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, tuple):
                return list(o)
            return str(o)

        path.write_text(json.dumps(asdict(self), indent=2, default=_default))
