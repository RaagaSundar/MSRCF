# MSRCF — Manufacture-to-Service Risk Continuity Framework

A research-grade Python implementation of the **MSRCF** for aerospace
CFRP (carbon fiber reinforced polymer) structural panels. MSRCF is the
first framework, to the authors' knowledge, that unifies upstream
**manufacturing-complexity risk scoring** with downstream **in-service
damage-mode prediction** under a single time-evolving metric, the
**Risk Continuity Score (RCS)**.

This v0.2 release extends the original three-pillar architecture with:

- **Per-damage-mode RCS_k(t) decomposition** — see *why* a component is RED.
- **Monte Carlo uncertainty bands** on every RCS trajectory.
- **Remaining Useful Life (RUL)** forecasting out to user-configurable horizons.
- **GridSearchCV hyperparameter tuning** for the best baseline classifier (opt-in).
- **SHAP feature attribution + per-class ROC/calibration** for the production classifier.
- **Sobol' global sensitivity analysis** on the Φ_composite weight vector.
- **IsolationForest anomaly detection** for out-of-distribution components.
- **18-test pytest suite** that pins the core invariants of every pillar.
- A clean **CLI** so seed / sample size / tuning / MC count can be set per-run.

---

## Architecture at a glance

```
                 +-----------------------------+
 Manufacturing   |  Pillar 1: Phi_composite    |
   complexity ----> generalized risk matrix    |---+
   features      |  (risk_matrix.py)           |   |
                 +--------------+--------------+   |
                                |                  |
                                v                  |
                 +-----------------------------+   |  Sobol sensitivity
                 |  Pillar 2: Damage classifier|   |  (sensitivity.py)
                 |  SVM / RF / LR / KNN / XGB  |   |
                 |  + opt-in GridSearchCV      |---+
                 |  (damage_predictor.py)      |   |  SHAP + ROC + calib.
                 +--------------+--------------+   |  (explainability.py)
                                |                  |
                                v                  |
                 +-----------------------------+   |  IsolationForest
                 |  Pillar 3: Risk Continuity  |---+  (anomaly_detector.py)
                 |  Score (RCS) over cycles    |
                 |  + per-class RCS_k(t)       |
                 |  + Monte Carlo CI band      |
                 |  (rcs_engine.py)            |
                 +--------------+--------------+
                                |
                                v
                 +-----------------------------+
                 |  RUL forecast cycles_to_RED |
                 |  (rul_predictor.py)         |
                 +--------------+--------------+
                                |
                                v
                       Dashboard + figures
                       (dashboard.py)
```

## Project layout

```
MSRCF/
├── src/
│   ├── config.py             # single source of truth for hyperparameters
│   ├── data_generator.py     # synthetic CFRP dataset (500 components)
│   ├── risk_matrix.py        # Pillar 1: Phi_composite scorer
│   ├── damage_predictor.py   # Pillar 2: 5-classifier benchmark + GridSearchCV
│   ├── rcs_engine.py         # Pillar 3: time-evolving RCS + per-class + MC band
│   ├── rul_predictor.py      # Remaining Useful Life extrapolator
│   ├── anomaly_detector.py   # IsolationForest OOD detector
│   ├── explainability.py     # SHAP + ROC + calibration
│   ├── sensitivity.py        # Sobol' weight sensitivity
│   ├── dashboard.py          # matplotlib visualisation layer
│   └── main.py               # end-to-end CLI orchestrator
├── tests/                    # pytest suite (18 tests, ~5 s)
├── data/                     # CSV outputs (regenerated on every run)
├── results/                  # PNG + CSV outputs (regenerated on every run)
├── report/
│   └── MSRCF_Technical_Report.md
├── requirements.txt
└── README.md
```

## Setup

Python 3.10+ recommended. Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the pipeline

From the project root:

```bash
python src/main.py                           # baseline run, 500 components
python src/main.py --tune-best                # add a GridSearchCV refinement
python src/main.py --n-components 1000        # bigger fleet
python src/main.py --skip-shap --skip-sobol   # faster smoke test
python src/main.py --mc-samples 500           # tighter MC bands
```

CLI reference:

| Flag                | Default | What it does                                            |
|---------------------|---------|---------------------------------------------------------|
| `--n-components N`  | 500     | Fleet size to synthesise.                               |
| `--seed S`          | 42      | RNG seed for full reproducibility.                       |
| `--tune-best`       | off     | GridSearchCV on the winning baseline model.             |
| `--mc-samples M`    | 200     | MC replicates for the RCS uncertainty band.             |
| `--skip-shap`       | off     | Skip SHAP feature attribution (saves ~5 s).             |
| `--skip-sobol`      | off     | Skip Sobol' weight sensitivity (saves ~3 s).            |

## Running the tests

```bash
python -m pytest tests/ -v
```

18 tests cover dataset physics, risk-matrix invariants, the Bayesian
update + RCS shapes, the MC band monotonicity, RUL ordering, and the
anomaly-detector contamination rate.

## Outputs

After a successful baseline run you should see:

```
data/
  msrcf_synthetic_dataset.csv     # raw features + ground truth labels
  msrcf_scored_dataset.csv        # adds per-feature bin scores + Phi
  msrcf_rcs_trajectories.csv      # long-format RCS (with MC band, per-class)
  msrcf_rul_forecast.csv          # cycles-to-yellow / cycles-to-red per component
  msrcf_anomaly_flags.csv         # IsolationForest output
results/
  confusion_matrices.png          # one per model
  model_comparison.png + .csv     # classifier metrics
  rcs_trajectories.png            # 5 representative components
  rcs_trajectories_mc.png         # same + Monte Carlo 5-95 % band
  rcs_per_class.png               # per-damage-mode RCS_k(t) panels
  rul_histogram.png               # cycles-to-yellow + cycles-to-red distributions
  risk_dashboard.png              # fleet-level summary dashboard
  anomaly_scatter.png             # anomaly score vs Phi
  feature_attribution.png + .csv  # SHAP heatmap (or permutation fallback)
  roc_calibration.png + .csv      # one-vs-rest ROC + reliability curves
  sobol_sensitivity.png + sobol_indices.csv
```

## Reading the technical report

The methodology, results, and discussion are written up in academic
style in [`report/MSRCF_Technical_Report.md`](report/MSRCF_Technical_Report.md).

## Notes

- Dependencies are intentionally limited to widely-available scientific
  Python libraries; no web framework is required.
- The dataset is fully synthetic and uses a documented physics-informed
  ground-truth function. The repository contains no proprietary
  aerospace data.
- All random seeds default to 42 for full reproducibility; pass
  `--seed` to vary.
