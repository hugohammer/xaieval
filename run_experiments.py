#!/usr/bin/env python3
"""Run the explanation-derived-prediction experiments and build the report.

Typical use::

    python run_experiments.py --quick                 # 5-minute smoke test
    python run_experiments.py                         # default run
    python run_experiments.py --repeats 30 --tuning per_repeat   # camera-ready
    python run_experiments.py --report-only           # rebuild tables/figures

Results land in ``Results/``:

    Results/raw/results.csv       one row per (dataset, model, repeat, method,
                                  variant, target, split, metric)
    Results/tables/*.tex          booktabs fragments to \\input
    Results/figures/*.{png,pdf}   figures
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from xaieval.config import ExperimentConfig  # noqa: E402
from xaieval.report import build_report  # noqa: E402
from xaieval.runner import plan_summary, run_experiment  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=HERE.parent / "Data")
    p.add_argument("--results-dir", type=Path, default=HERE.parent / "Results")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="dataset names to run (default: every CSV in --data-dir)")
    p.add_argument("--models", nargs="*", default=None,
                   help="black boxes: random_forest gradient_boosting mlp svm_rbf extra_trees")
    p.add_argument("--repeats", type=int, default=None, help="number of train/test splits")
    p.add_argument("--test-size", type=float, default=None)
    p.add_argument("--tuning", choices=["none", "per_dataset", "per_repeat"], default=None)
    p.add_argument("--output-scale", choices=["logit", "probability"], default=None,
                   help="scale on which classifiers are explained and reproduced")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="tiny budgets; smoke test only")
    p.add_argument("--no-controls", action="store_true", help="skip null/corruption controls")
    p.add_argument("--no-refmetrics", action="store_true", help="skip established metrics")
    p.add_argument("--lime-kernel-sweep", nargs="*", type=float, default=None,
                   metavar="W", help="LIME kernel widths to sweep (default set used if flag given "
                                     "with no values); expensive but shows how much of a LIME "
                                     "result is the library default")
    p.add_argument("--report-only", action="store_true",
                   help="rebuild tables and figures from an existing Results/raw")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(data_dir=args.data_dir, results_dir=args.results_dir)
    if args.quick:
        cfg.quick()
    # Explicit flags win over --quick.
    if args.datasets:
        cfg.datasets = args.datasets
    if args.models:
        cfg.blackbox.models = tuple(args.models)
    if args.repeats is not None:
        cfg.n_repeats = args.repeats
    if args.test_size is not None:
        cfg.test_size = args.test_size
    if args.tuning is not None:
        cfg.blackbox.tuning = args.tuning
    if args.output_scale is not None:
        cfg.output_scale = args.output_scale
    if args.seed is not None:
        cfg.random_state = args.seed
    if args.n_jobs is not None:
        cfg.n_jobs = args.n_jobs
    if args.no_controls:
        cfg.control.run_controls = False
    if args.no_refmetrics:
        cfg.refmetric.run_ref_metrics = False
    if args.lime_kernel_sweep is not None:
        cfg.explainer.lime_kernel_width_sweep = tuple(
            args.lime_kernel_sweep or (0.25, 0.5, 1.0, 2.0, 4.0)
        )
    cfg.verbose = args.verbose
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)

    if not args.report_only:
        from xaieval.datasets import discover_datasets

        n_datasets = len(discover_datasets(cfg.data_dir, list(cfg.datasets) if cfg.datasets else None))
        print("Configuration")
        print(f"  data      : {cfg.data_dir}")
        print(f"  results   : {cfg.results_dir}")
        print(f"  models    : {', '.join(cfg.blackbox.models)}")
        print(f"  repeats   : {cfg.n_repeats}   test size: {cfg.test_size}")
        print(f"  tuning    : {cfg.blackbox.tuning}   score scale: {cfg.output_scale}")
        print(f"  controls  : {cfg.control.run_controls}   ref metrics: {cfg.refmetric.run_ref_metrics}")
        print(plan_summary(n_datasets, cfg))
        t0 = time.time()
        run_experiment(cfg)
        print(f"\nExperiments finished in {(time.time() - t0) / 60:.1f} min")

    build_report(cfg.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
