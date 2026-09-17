# Evaluating XAI methods by predicting from explanations

Experiment code for the paper. Every explanation method induces a predictor. This code builds that predictor
for PDP, ALE, SHAP and LIME, and measures how well it predicts on held-out data
against two targets: the original outcome `y`, and the black box's own score
`f(x)` (an out-of-sample, global fidelity measure).

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 make_datasets.py --synthetic            # writes Data/syn_*.csv
python3 -m pytest tests -q                      # 14 tests, ~20 s
python3 run_experiments.py --quick              # smoke test, a few minutes
python3 run_experiments.py                      # the real run
```

Output:

```
Results/raw/results.csv     long format: dataset, model, repeat, family,
                            variant, target, split, metric, value
Results/tables/*.tex        booktabs fragments to \input from the manuscript
Results/figures/*.png,.pdf  figures
```

`run_experiments.py --report-only` rebuilds every table and figure from
`Results/raw` without re-running anything.

---

## What changed relative to the thesis, and why

These are the corrections the code exists to implement. Each one changes a
result, so each is worth stating in the paper.

### 1. The SHAP construction was measuring nothing about SHAP

SHAP satisfies local accuracy exactly:

```
sum_j phi_j(x_i) = f(x_i) - E[f(X)]
```

So the thesis's predictor

```
yhat(x0) = E[f] + sum_i w_i S_i        with sum_i w_i = 1
```

collapses algebraically to `sum_i w_i f(x_i)` — inverse-distance kNN
regression on the black box's own predictions. The per-feature attributions
cancel; the predictor is invariant to how the attribution mass is distributed
across features. Its perfect training fit (`R^2 = 1.0`) is a tautology: at an
anchor point the zero-distance weight dominates.

The reported "SHAP overfits (ΔR² = 0.35–0.56), LIME does not" contrast was
therefore a comparison between a kNN smoother and a locally-linear smoother,
not between two explanation methods.

This code:

* builds the SHAP predictor from the **per-feature attributions**, by fitting a
  one-dimensional dependence curve `x_j -> phi_j` and summing
  (`explainers.dependence_curves`);
* retains the degenerate construction as `SHAP/sum-idw-degenerate` **solely to
  demonstrate** the identity, printing `max|yhat_SHAPsum - yhat_kNN|`, which
  comes out at ~1e-15 (`tab_shap_degeneracy.tex`);
* proves the identity as a unit test (`test_shap_sum_idw_is_blackbox_knn`).

### 2. Tables 4.5 and 4.6 of the thesis are mutually inconsistent

The unit-coefficient vector lies inside the OLS parameter space, so OLS cannot
have a *higher* training RSS on the same design matrix and target. The thesis
reported PDP(OLS) train `R² = 0.419` against `f̂` alongside a unit-coefficient
model claimed to be consistently better across train and test at `R² ≈ 0.97`.
That is impossible. `test_ols_never_worse_than_unit_on_training_data` makes the
condition a test rather than an assumption.

### 3. Two encoding choices had silently gutted the experiment

* **All levels of every categorical were one-hot encoded.** The dummy block then
  sums to a constant, so the design matrix is exactly rank deficient — the
  "effective rank 16 of 18" the thesis treated as a property of the PDP
  transform. Here reference levels are dropped and the rank deficiency is zero
  by construction (`test_one_hot_drops_reference_level_so_design_is_full_rank`).
* **Age was quantile-binned into dummies.** For a two-valued feature, a curve
  is an *affine* function of that feature, so the explanation transform is a
  no-op — a linear model on the raw dummies spans exactly the same column
  space. With 14 of 18 encoded columns binary, the thesis's PDP surrogate was
  effectively a linear probability model plus four transformed columns, which
  explains why PDP ≈ ALE and why everything landed at AUC ≈ 0.90. Binning is
  gone, and `tab_datasets.tex` reports the binary fraction of every dataset so
  the reader can see where the measure has room to discriminate.

### 4. The additivity ceiling: what the numbers actually mean

Under feature independence the L²-optimal additive approximation to `f` is the
sum of its centred partial dependence functions — the first-order
functional-ANOVA projection. So

```
R2_add = 1 - Var(f - f_add) / Var(f)
```

is a property of the **model**, not of any explanation: the fraction of the
model's variance that is additive. Under independence it is an upper bound on
what any additive explanation-derived predictor can reach — but that condition
is load-bearing and usually fails on real data; see §7.

This reframes the thesis's headline. "A unit-coefficient PDP surrogate
reproduces the random forest with R² = 0.97" is a statement that *the forest
was 97% additive on that dataset*, not that PDP was an excellent explanation.

The code therefore reports fidelity next to the reconstruction and next to the
**attainment ratio** `R²/R²_add` — how much of the recoverable structure the
explanation actually recovered (`anova.py`, `tab_ceiling.tex`). The synthetic
suite validates the estimator against a known ground truth
(`tab_synthetic.tex`, `fig_synthetic_recovery.png`).

Two reporting details that matter, both learned from the full run:

* The ratio is computed as a **ratio of means**, not a mean of per-repeat
  ratios. The denominator is noisy and can approach zero, so per-repeat ratios
  blow up — one cell averaged 14.3 against a median of 1.96 — and repeats where
  the denominator hit the guard were dropped, biasing the average. Ratios are
  suppressed entirely below `ATTAINMENT_MIN_DENOM`.
* **PDP is omitted from the ratio columns.** The reconstruction is defined as
  the PDP unit-coefficient sum, so PDP's ratio is 1.000 by construction.

### 5. Two library traps that silently invert or rescale results

Both were found while building this, both are guarded by tests, and both are
worth a sentence in the paper's reproducibility section.

* **LIME negates its coefficients in regression mode.** `lime` stores the fitted
  ridge coefficients under label `1` and a sign-flipped display copy under
  label `0`:

  ```python
  ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
  ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]
  ```

  Reading the first key — the obvious choice — inverts every local model. In
  our runs this turned a fidelity of **+0.51 into −0.90**. The extraction now
  reconstructs the prediction under each label and keeps the one that
  reproduces LIME's own `local_pred` (`test_lime_coefficients_have_the_right_sign`).

* **TreeSHAP's `raw` output for an sklearn tree classifier is the probability,
  not the log-odds.** On the log-odds scale its attributions do not sum to
  `score(x) − E[score]`, so local accuracy silently fails and every construction
  that assumes the decomposition becomes invalid. The explainer choice is now
  verified against the scale in use, falling back to permutation SHAP on
  `bb.score` when they do not match
  (`test_shap_local_accuracy_holds_on_the_explained_scale`).

### 6. The output scale can silently destroy the results

Found by reviewing a real run, not by reasoning ahead of time — worth a
paragraph in the paper's methods section.

A random forest grows pure leaves and so emits probabilities of *exactly* 0 and
1. Converting those to log-odds needs a clip, and an aggressive clip (the
original 1e-6) maps them to ±13.8 — a precision the model does not possess. On a
near-separable problem the majority of predictions pin to that bound, the
log-odds "score" becomes a two-valued spike, and the PDP sum built from it has
far more variance than the function itself. The additivity reconstruction then
comes out **negative**:

| Dataset | saturated | ceiling @1e-6 | ceiling @1/(2T) | ceiling, probability |
|---|---|---|---|---|
| breast_cancer_wisc | 59.7% | −2.11 | −0.20 | **+0.93** |
| spambase | 22.3% | −0.59 | −0.04 | **+0.81** |
| qsar_biodeg | 7.0% | −0.90 | +0.16 | **+0.72** |

Two changes followed:

* `blackbox.resolution_clip` sets the clip from the model's own resolution —
  `1/(2T)` for a `T`-tree ensemble (±6.4 log-odds at `T = 300`) instead of a
  fixed 1e-6. This removes most of the damage but **cannot remove all of it**:
  at 60% saturation the model has no resolution to recover.
* **`probability` is now the default scale.** It is bounded in [0, 1] and cannot
  fail this way. Use `--output-scale logit` as a sensitivity analysis, and check
  the `saturated_fraction` diagnostic (now recorded per dataset/model/split
  under `family="scale"`) before believing it.

### 7. The additivity ceiling is only a ceiling under independence

Also a correction to an earlier claim in this codebase, and one that would
otherwise have propagated into the paper.

`R²_add` equals the *L²-optimal* additive projection **only when features are
independent**. Under dependence the sum of marginal PDPs is not the optimal
additive approximation — that is exactly the gap Hooker's generalized functional
ANOVA addresses — so it is neither an upper bound nor bounded below by zero.

Describe it as "the additive PDP reconstruction R²", and state the independence
condition when calling it a ceiling. The `syn_corr30/60/85` sweep is the
instrument for quantifying the drift: report it rather than asserting the bound.

### 8. Everything else

* **Repeated splits with confidence intervals.** 20 stratified splits by
  default. The thesis discussed AUC differences of 0.005 from a single split of
  184 test points; those are well inside the noise. Method comparisons are
  *paired* on the split and Holm-corrected (`tab_pairwise.tex`).
* **Baselines.** Intercept-only, linear on raw features, and an additive spline
  model fitted directly to `y`. The last bounds what any additive
  explanation-derived predictor can achieve without an explanation at all.
* **Null controls and a corruption sweep.** Curve values permuted across the
  grid (destroying the value→effect mapping while preserving the distribution
  of effect sizes), amplitude-matched random curves, and a graded noise sweep.
  A measure that does not collapse under these is not reading the explanation
  (`tab_controls.tex`, `fig_corruption.png`).
* **Comparison with established metrics.** Infidelity, faithfulness
  correlation, max-sensitivity, complexity and sparseness on the same
  explanations and splits, with rank correlations against the proposed measure
  (`tab_refmetrics.tex`). This is what answers "what does this add?".
* **Probability by default for classifiers.** Log-odds is the more principled
  scale for additivity in the abstract, but it is not safe by default (§6) and
  costs ~30x more. Both are supported and the paper should report the
  comparison; run `--output-scale logit` as a sensitivity analysis and check the
  saturation diagnostic.

---

## Adding datasets

Drop one CSV per dataset into `Data/`, plus a JSON sidecar with the same stem:

```json
{
  "name": "heart_disease",
  "target": "num",
  "task": "classification",
  "positive_if": ">0",
  "drop": ["id", "dataset"],
  "categorical": ["cp", "restecg"],
  "binary": ["sex", "fbs", "exang"],
  "zero_is_missing": ["chol", "trestbps"],
  "max_missing_frac": 0.3,
  "source": "https://archive.ics.uci.edu/dataset/45/heart+disease",
  "citation": "Detrano et al. (1989)"
}
```

Every key except `target` is optional; roles are inferred otherwise. Write the
sidecar by hand for anything with an ID column, a multi-class target, or
sentinel values stored as zeros. `make_datasets.py --template Data/foo.csv`
writes an inferred sidecar to edit.

Helpers:

```bash
python make_datasets.py --synthetic       # the synthetic suite (no network)
python make_datasets.py --openml-suite    # 15 curated tabular benchmarks
python make_datasets.py --openml mydata:1590:classification
python make_datasets.py --inspect         # pre-flight: is each dataset usable?
```

**Always run `--inspect` before a long run.** It loads every dataset through the
real pipeline and prints what the experiment will actually see — encoded width,
binary fraction, rank deficiency, class balance — plus a list of low-cardinality
numeric columns you may want to declare as `categorical`. It runs no experiment
and takes seconds. It exists because the failure it catches is silent: a
dataset can look fine and still be incapable of measuring anything.

### What `--inspect` verdicts do and do not do

`run_experiments.py` does **not** read the verdicts — it runs every CSV in
`Data/`. The verdicts are for you, and there are two different kinds:

| Verdict | What to do |
|---|---|
| `LOAD FAILED`, `target near-degenerate`, `rank deficient` | Fix the sidecar or remove the file. These are broken, not merely awkward. |
| `mostly binary` | **Leave it in.** Handled automatically by stratified reporting, below. |
| `small n`, `large n` | Leave it in; they are cost/precision notes, not defects. |

To restrict a run explicitly, use `--datasets a b c` rather than deleting
files — it keeps `Data/` reproducible and the choice visible in
`Results/config.json`.

### Stratified reporting

Pooled tables are computed on the **continuous-dominated** stratum
(binary fraction ≤ `BINARY_FRACTION_LIMIT`, default 0.6). Categorical-dominated
datasets are not discarded: they get `*_categorical` companion tables, and the
split is stated in every caption.

This is not fastidiousness. On a two-valued column the curve transform is
affine, so those datasets *cannot* distinguish between explanation methods;
averaging them into the headline would dilute a real effect with cells that are
incapable of showing one. Reported separately they become evidence instead —
the companion table is where you demonstrate the degeneracy on real data, and
`credit_g` is in the curated suite precisely for that.

**Choose datasets with continuous features.** A dataset that is mostly
categorical cannot discriminate between explanation methods, for the reason in
§3 above. `tab_datasets.tex` reports the binary fraction; treat anything above
roughly 0.6 as uninformative for the main comparison, and say so in the paper
rather than averaging it in silently. The curated suite deliberately includes
one such dataset (`credit_g`, binary fraction 0.88) as a negative control, and
`make_datasets.py` records which candidates were rejected and why.

**Numeric columns stay numeric.** A numeric column is only treated as
categorical if you declare it in the sidecar. This is deliberate: an earlier
rule that auto-categorised low-cardinality numeric columns expanded the
Wisconsin breast-cancer data (integer 1–10 measurement scales) from 9 variables
into 80 dummies, silently making it useless for the comparison. Imposing a
spurious ordering on a nominal code is the milder error, so that is the way the
default fails; `--inspect` lists the candidates for you to override.

If you use the UCI heart-disease data, cite the original donors (Detrano et
al., 1989) rather than only the Kaggle mirror.

### The synthetic suite

`make_datasets.py --synthetic` writes ten datasets where the ground truth is
known in closed form:

| Group | Datasets | Purpose |
|---|---|---|
| Interaction sweep | `syn_inter00 … syn_inter75` | additivity ceiling known exactly; validates the estimator |
| Correlation sweep | `syn_corr30/60/85` | the axis on which PDP and ALE are *supposed* to differ |
| Binary degeneracy | `syn_binary6` | 6 of 8 features binary; the transform is provably a no-op |
| Regression | `syn_regress` | the framework is not classification-only |

The correlation sweep matters for a specific reason: the thesis concluded that
"correlation effects are limited on this dataset" from PDP ≈ ALE, but on a
mostly-binary design PDP and ALE *cannot* differ. Without a correlated design
that conclusion is untestable.

---

## Layout

```
Code/
  run_experiments.py     CLI: run + report
  make_datasets.py       CLI: synthetic suite, OpenML fetch, sidecar templates
  xaieval/
    config.py            every budget and constant in one place
    datasets.py          loading, sidecars, cleaning
    preprocessing.py     encoding (drop-first), the FeatureSpace metadata
    blackbox.py          model zoo behind a single scalar score function
    curves.py            Curve1D / CurveSet; attribution -> curve smoother
    explainers.py        PDP, ALE, SHAP, LIME + conversion to curve form
    predictors.py        the explanation-derived predictors
    baselines.py         intercept / linear / spline-GAM / black box
    controls.py          null explanations and the corruption sweep
    refmetrics.py        infidelity, faithfulness, sensitivity, complexity
    anova.py             the additivity ceiling and attainment ratio
    scoring.py           metrics, t intervals, paired tests, Holm
    synthetic.py         ground-truth data generators
    runner.py            orchestration
    report.py            LaTeX tables and figures
  tests/test_sanity.py   14 tests; the first few are the paper's claims
```

## What a validation run looks like

From a low-budget shakedown (10 synthetic datasets × {random forest, gradient
boosting} × 3 splits, probability scale) — not results for the paper, but a
check that each piece measures what it claims to:

* **The SHAP degeneracy identity holds numerically.**
  `max|ŷ_SHAPsum − ŷ_kNN|` came out at 4e-16 to 3e-15, and the degenerate
  construction's out-of-sample R² equalled the pure kNN control's to three
  decimals.
* **The ceiling estimator tracks known ground truth.** Across the interaction
  sweep, designed additive share 1.00 / 0.90 / 0.76 / 0.50 / 0.25 gave measured
  ceilings of 0.84 / 0.81 / 0.80 / 0.66 / 0.48. Monotone, with the expected
  compression at both ends: the black box neither reproduces a purely additive
  signal exactly, nor learns a strongly interactive one fully — so what it
  learned is more additive than the process that generated the data. That gap
  is a property of the fitted model and should be reported, not tuned away.
* **The nulls collapse.** Intact curves reached test R² ≈ 0.75; permuting the
  curve values across the grid dropped it to ≈ −0.7.
* **The measure is not a restatement of an existing one.** Pooled Spearman
  against established metrics was near zero for infidelity (−0.00),
  max-sensitivity (0.06), complexity (0.05) and sparseness (−0.04), and 0.35
  for faithfulness correlation. Ranking the four methods *within* a single
  (dataset, model, split) cell, the correlations were moderate for infidelity
  (0.43), faithfulness (0.50) and sensitivity (0.40) but ≈ 0 for the two
  complexity measures — i.e. partially overlapping with the faithfulness family,
  orthogonal to the complexity family.
* **LIME's locality is a hyperparameter, not a property.** Sweeping the kernel
  width: at width 0.25 the local models reproduce the black box at their own
  anchor perfectly (R² = 1.00) but carry no out-of-sample predictive content
  (R² = −0.00); at width 4 the anchor fit falls to 0.44 while out-of-sample
  fidelity rises to 0.40. Locality and transferable predictive content trade off
  directly, and the library default sits at one arbitrary point on that curve.

## Cost

The dominant cost is the **output scale**, and by a wide margin.

On the probability scale a tree model gets TreeSHAP, which is nearly free. On
the log-odds scale TreeSHAP's output is the wrong quantity (see §5), so the code
falls back to permutation SHAP applied to `bb.score` — correct, but it costs

```
shap_max_points  ×  (shap_agnostic_evals_mult × p)  ×  shap_kernel_background
```

model calls per (dataset, model, split). At defaults with `p = 8` that is ~1.4M;
at `p = 57` (spambase) it is ~10M, and the stage dominates everything else. The
code warns when the estimate exceeds `shap_cost_warn_threshold`.

Measured per (dataset, model, split), explainer stages only
(random forest; these were taken at `evals_mult=10, background=100`, so the
current defaults of 6 and 50 make the log-odds SHAP figure roughly 3× cheaper):

| Dataset | Scale | fit | PDP | ALE | SHAP | LIME | total |
|---|---|---|---|---|---|---|---|
| diabetes_pima (n=768, p=8) | probability | 0.3 | 2.5 | 4.3 | **2.5** | 7.3 | **17 s** |
| diabetes_pima (n=768, p=8) | log-odds | 0.3 | 2.6 | 3.2 | **76.6** | 9.8 | **92 s** |
| wine_quality_red (n=1599, p=11) | probability | 0.5 | 4.4 | 5.5 | **3.1** | 7.4 | **21 s** |

The scale changes the SHAP stage by 30× on the *narrowest* dataset in the suite,
and the gap widens with `p`. Add roughly 20 s for the controls and reference
metrics.

So on the probability scale, budget ~40 s per (dataset, model, split): a
15-dataset × 4-model × 20-repeat run is on the order of 3–4 hours on 8 cores.
On the log-odds scale the same run is a day or more, and wide datasets
(spambase at `p = 57`, ~10M model calls per unit) are impractical without
cutting the budgets above.

**Recommended design:** run the probability scale over the full dataset suite,
and the log-odds scale as a robustness check on the narrow datasets only
(`p ≲ 15`). Report the first as the main result and the second as a scale
sensitivity analysis — the additivity ceiling is the quantity most affected by
the choice, so the comparison is worth having.

Other knobs: `--repeats`, `--models`, `--no-controls`, `--no-refmetrics`, and
`explainer.lime_max_points` in `config.py`.

### Cores

All available cores are used by default (`n_jobs = -1`); nothing to set.
`--n-jobs N` caps it. The startup banner prints the worker count and the total
work, and warns when the machine will not be filled.

Parallelism runs across **(dataset, repeat) pairs**, in two phases: repeat 0 of
every dataset first (this is where per-dataset tuning happens), then every
remaining pair in one flat batch. So the width of the main phase is
`n_datasets × (n_repeats − 1)` — for the recommended run, 15 × 19 = 285 jobs,
which saturates any reasonable machine.

The consequence to know: **a run with few datasets and few repeats will not use
your cores**, because there is not enough independent work. `--quick --repeats
2` on one dataset is essentially serial, by construction rather than by
oversight. Scale `--repeats` up before concluding the code is slow.

Inner estimator parallelism is deliberately pinned to `n_jobs=1`
(`blackbox.py`) so that scikit-learn does not oversubscribe cores inside each
joblib worker; joblib's loky backend likewise caps BLAS threads per worker.
Parallelism is at the outer level only, which is the right level here because
the units are independent and roughly equal in cost.

The `--lime-kernel-sweep` flag multiplies LIME cost by the number of widths. It
is off by default but worth one run: LIME's default neighbourhood is wide
enough that its local models often fail to reproduce the black box even at the
point they explain, so any single-width result is a statement about the library
default rather than about LIME (`tab_lime_kernel.tex`).

## Reproducibility

Fixed seeds throughout (`ExperimentConfig.random_state`); repeat `k` uses seed
`random_state + 1000k`. The resolved configuration is written to
`Results/config.json` on every run. Hyperparameter tuning defaults to
`per_dataset` — one randomised search on repeat 0's training split, reused
across repeats. That is a mild optimism which does not bias the *comparison
between explanation methods*, but for the camera-ready run use
`--tuning per_repeat` and say which was used.

---

## Citation

This is the code accompanying:

> Selbæk, J. & Hammer, H. L. *Evaluating Explanation Methods by the Predictors
> They Induce* (under review, 2026).

## License

Released under the MIT License; see [`LICENSE`](LICENSE).
