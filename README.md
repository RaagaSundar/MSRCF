# MSRCF

[![CI](https://github.com/RaagaSundar/MSRCF/actions/workflows/ci.yml/badge.svg)](https://github.com/RaagaSundar/MSRCF/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/types-mypy-blue.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

A Python project for tracking aerospace composite panels from manufacture through their service life using a single risk number.

The idea is simple. Factory engineers use one set of models to flag risky parts at build time. Maintenance engineers use a different set to track damage once the part is flying. Nobody connects them. This project does. It computes a Risk Continuity Score (RCS) for every component that starts at "as-built" and updates each inspection cycle, so there's always one number telling you how worried to be about each part.

## What it does

Three layers, in order:

1. Scores each component's manufacturing complexity (ply count, void probability, fastener density, zone complexity, thickness variation) on a 1-5 risk matrix.
2. Trains five classifiers (SVM, Random Forest, Logistic Regression, KNN, XGBoost) to predict what damage mode each component is likely to develop.
3. Combines those two signals with an in-service degradation curve to produce the RCS over time, and flags every component as GREEN, YELLOW, or RED.

On top of that you also get:

- Per-damage-mode RCS so you know *what* to inspect for, not just *whether*.
- Monte Carlo uncertainty bands around each trajectory.
- Remaining Useful Life forecasts (cycles-to-YELLOW, cycles-to-RED).
- SHAP feature importance and ROC / calibration plots for the classifier.
- Sobol sensitivity on the Phi weight vector.
- Isolation Forest anomaly flagging for components that look strange.
- Conformal prediction sets with a guaranteed coverage level, so you can say "the true damage mode is in this set at least 90% of the time" instead of just trusting the single best guess.

## Setup

Python 3.10 or newer. Install dependencies:

```
pip install -r requirements.txt
```

## Run it

From the project root:

```
python src/main.py
```

That generates the dataset, trains the models, builds the RCS, and writes all figures and CSVs. Takes about a minute on a laptop.

If you want to tune the best classifier:

```
python src/main.py --tune-best
```

Other useful flags:

- `--n-components 1000` -- bigger fleet
- `--seed 7` -- different RNG seed
- `--mc-samples 500` -- tighter Monte Carlo bands
- `--skip-shap` -- skip SHAP (saves about 5 seconds)
- `--skip-sobol` -- skip Sobol sensitivity
- `--skip-conformal` -- skip conformal prediction sets

## Statistical evaluation

A single run on one seed isn't enough to claim one model beats another. Two
extra scripts do the rigorous version:

```
python src/experiments.py --n-seeds 15   # every metric as mean +/- 95% CI
python src/ablation.py    --n-seeds 8    # prove each feature/design choice earns its place
```

`experiments.py` re-runs the whole benchmark across many seeds, reports each
metric as a mean with a 95% confidence interval, then runs a Friedman test
plus a Nemenyi post-hoc (the standard way to compare classifiers over many
runs) and draws a critical-difference diagram. `ablation.py` drops each
feature one at a time and toggles the two RCS design choices, so you can see
exactly how much each piece contributes. Both write figures and CSVs into
`results/`.

The maths behind all of this -- the RCS bounds, the Bayesian update written
as a log-odds recursion, and the significance tests -- is derived in the
appendices of the technical report.

## Run the tests

```
python -m pytest          # fast unit tests (~13 s)
python -m pytest -m ""     # include the slow training-heavy integration tests
```

The fast suite has 24 tests covering dataset physics, the risk-matrix bounds,
the Bayesian update, RUL ordering, the anomaly detector, and the statistics
helpers (the confidence-interval and Nemenyi math are checked against known
values). The training-heavy multi-seed and ablation tests are marked `slow`
and skipped by default.

## Make targets

If you have `make`:

```
make install     # runtime + dev dependencies
make run          # full pipeline
make experiments  # multi-seed benchmark + significance tests
make ablation     # ablation study
make all          # lint + type-check + test
```

## What you get out

After a run you'll have:

`data/`

- `msrcf_synthetic_dataset.csv` -- raw 500-component dataset
- `msrcf_scored_dataset.csv` -- with Phi_composite scores added
- `msrcf_rcs_trajectories.csv` -- long-format RCS over 11 cycles
- `msrcf_rul_forecast.csv` -- cycles-to-yellow and cycles-to-red per component
- `msrcf_anomaly_flags.csv` -- anomaly scores

`results/`

- `confusion_matrices.png` -- one per classifier
- `model_comparison.png` -- accuracy / precision / recall / F1 bar chart
- `rcs_trajectories.png` -- 5 sample components over time
- `rcs_trajectories_mc.png` -- same, with uncertainty bands
- `rcs_per_class.png` -- RCS broken out by damage mode
- `rul_histogram.png` -- cycles-to-yellow and cycles-to-red distributions
- `risk_dashboard.png` -- fleet-level summary (4 panels)
- `anomaly_scatter.png` -- anomaly scores vs Phi
- `feature_attribution.png` -- SHAP heatmap
- `roc_calibration.png` -- ROC and reliability curves
- `sobol_sensitivity.png` -- Phi weight sensitivity
- `conformal_coverage.png` -- conformal coverage + set-size curves (with `conformal_metrics.csv` and `conformal_per_class.csv`)
- plus CSV versions of the metrics

From `experiments.py` and `ablation.py`:

- `multiseed_forest.png` -- per-model metrics with 95% CI over the seed ensemble
- `critical_difference.png` -- Friedman/Nemenyi critical-difference diagram
- `ablation.png` -- feature importance + RCS design-choice ablation
- plus `multiseed_summary.csv`, `multiseed_per_seed.csv`, `ablation_features.csv`, `ablation_design.csv`

## Project layout

```
src/        source modules
tests/      pytest suite
data/       inputs and outputs (regenerated each run)
results/    figures (regenerated each run)
report/     technical report (markdown)
```

Source files:

- `config.py` -- all hyperparameters in one place
- `data_generator.py` -- synthetic dataset
- `risk_matrix.py` -- Pillar 1 (Phi_composite scorer)
- `damage_predictor.py` -- Pillar 2 (the five classifiers)
- `rcs_engine.py` -- Pillar 3 (RCS, per-class, Monte Carlo)
- `rul_predictor.py` -- Remaining Useful Life
- `anomaly_detector.py` -- Isolation Forest
- `explainability.py` -- SHAP, ROC, calibration
- `sensitivity.py` -- Sobol sensitivity
- `conformal.py` -- conformal prediction sets (LAC + APS)
- `experiments.py` -- multi-seed benchmark + Friedman/Nemenyi
- `ablation.py` -- feature + design-choice ablation
- `dashboard.py` -- plots
- `main.py` -- runs everything in order

Plus `pyproject.toml` (packaging + ruff/mypy/pytest config), a `Makefile`,
and a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs lint,
type-check, and tests on Python 3.10/3.11/3.12.

## A few things to know

- The dataset is synthetic. The distributions and rules are anchored in published composites literature, but using this on a real fleet would mean back-testing against actual inspection records first.
- All seeds default to 42, so runs are reproducible. Pass `--seed` to change.
- No web frameworks. Just matplotlib for plotting and pytest for testing.

## Read more

Full methodology and results are in `report/MSRCF_Technical_Report.md`.

## License

MIT. See `LICENSE`.
