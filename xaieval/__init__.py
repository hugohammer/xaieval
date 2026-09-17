"""Evaluating explainable-AI methods by predicting from explanations.

Rewritten from scratch for the paper.  Three things this package does that the
original thesis code did not:

* it builds the SHAP predictor from the per-feature attributions rather than
  their sum (the sum construction is algebraically kNN on the black box's own
  predictions and contains no SHAP information -- see
  :mod:`xaieval.explainers`);
* it reports every result over repeated splits with confidence intervals,
  alongside baselines and null-explanation controls; and
* it reports the additivity ceiling, so that a fidelity number can be read as
  "how much of the recoverable structure the explanation recovered" rather
  than "how additive the model happened to be".
"""

__version__ = "1.0.0"

from .config import ExperimentConfig  # noqa: F401
