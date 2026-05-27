"""
config.py
=========

Centralised configuration for the Manufacture-to-Service Risk
Continuity Framework. All hyperparameters that affect the
physics, the classifier benchmark, or the RCS engine live here so
they can be audited in one place and overridden from the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Global reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Pillar 1 - Phi_composite risk matrix
# ---------------------------------------------------------------------------
PHI_FEATURE_ORDER: tuple[str, ...] = (
    "ply_count",
    "void_probability",
    "fastener_density",
    "zone_complexity",
    "thickness_variation",
)

# Weights are locked by the project brief; void probability dominates per
# Mesogitis et al. (2014). The sensitivity module re-validates them.
PHI_WEIGHTS: tuple[float, ...] = (0.25, 0.30, 0.15, 0.20, 0.10)

# 20/40/60/80 percentile bin edges for per-feature 1-5 score binning.
PHI_PERCENTILES: tuple[float, ...] = (20.0, 40.0, 60.0, 80.0)


# ---------------------------------------------------------------------------
# Pillar 2 - damage classifier benchmark
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20
CV_FOLDS = 5

# Damage class labels (also defined in damage_predictor.DAMAGE_CLASS_NAMES,
# duplicated here so config is the single source of truth).
DAMAGE_CLASS_NAMES = {
    0: "No damage",
    1: "Matrix cracking",
    2: "Delamination",
    3: "Fiber breakage",
    4: "Fatigue crack",
}

# When --tune-best is passed on the CLI, this small grid is searched
# over the best baseline model. Kept compact to keep total wall time
# under a minute on a laptop.
XGB_TUNING_GRID: dict[str, list] = {
    "clf__n_estimators": [200, 400, 600],
    "clf__max_depth": [3, 4, 6],
    "clf__learning_rate": [0.05, 0.1, 0.15],
}
RF_TUNING_GRID: dict[str, list] = {
    "clf__n_estimators": [200, 400, 600],
    "clf__max_depth": [None, 6, 10],
    "clf__min_samples_split": [2, 4],
}


# ---------------------------------------------------------------------------
# Pillar 3 - RCS engine
# ---------------------------------------------------------------------------
ALPHA = 0.4
BETA = 0.4
GAMMA = 0.2
LAMBDA = 0.15
DEFAULT_CYCLES = tuple(range(0, 11))

INFO_WEIGHT = 0.15
DRIFT = 0.10
PRIOR_BLEND = 0.55
PHI_MIN = 5.0
PHI_MAX = 25.0

RCS_FLAG_THRESHOLDS = {"YELLOW": 40.0, "RED": 70.0}


# ---------------------------------------------------------------------------
# Monte Carlo uncertainty
# ---------------------------------------------------------------------------
MC_SAMPLES = 200
# Gaussian noise sigma in *logit* space, applied independently to the
# per-class classifier probabilities to seed each Monte Carlo replicate.
# Calibrated to represent inspector-to-inspector reliability scatter
# (Talreja & Singh, 2012, Table 9.3): sigma=1.2 in logit space
# corresponds to roughly +/- 30 percentage points around a 50 %
# probability and +/- 5 points around a 95 % probability - a credible
# upper bound on per-inspection classifier disagreement.
MC_NOISE_SIGMA = 1.2
MC_CI_LOW = 5.0         # percentile bounds for the trajectory ribbon
MC_CI_HIGH = 95.0


# ---------------------------------------------------------------------------
# RUL extrapolation horizon
# ---------------------------------------------------------------------------
RUL_MAX_CYCLE = 60


# ---------------------------------------------------------------------------
# Sobol sensitivity
# ---------------------------------------------------------------------------
SOBOL_N_BASE = 256
SOBOL_WEIGHT_SCALE = 0.15  # max relative perturbation around nominal


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
ISO_CONTAMINATION = 0.05


@dataclass
class RunConfig:
    """Per-run options that may be overridden by argparse."""

    n_components: int = 500
    seed: int = RANDOM_SEED
    tune_best: bool = False
    mc_samples: int = MC_SAMPLES
    skip_shap: bool = False
    skip_sobol: bool = False
    output_dir: str = ""        # populated by main.py from project root
    data_dir: str = ""
    report_dir: str = ""
