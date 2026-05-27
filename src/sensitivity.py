"""
sensitivity.py
==============

Sobol global sensitivity analysis for the Phi_composite scorer.

We perturb each of the five Phi_composite weights independently within
+/- SOBOL_WEIGHT_SCALE of its nominal value, re-normalise the weight
vector to sum to 1, and compute the resulting fleet-mean Phi_composite.
The first-order Sobol indices (S1) tell us which weight has the largest
marginal effect on the fleet mean; the total-order indices (ST) include
interaction effects.

This is exactly the audit a certification authority would request:
"how brittle is your risk score with respect to the chosen weight
vector?" If the dominant index is dominated by a single weight, the
score is sensitive (bad). If indices are evenly distributed and
small, the score is robust (good).
"""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110


# ---------------------------------------------------------------------------
def sobol_phi_weight_sensitivity(
    score_matrix: np.ndarray,
    feature_names: list[str],
    nominal_weights: np.ndarray,
    output_path: str,
    n_base: int = 256,
    weight_scale: float = 0.15,
    seed: int = 42,
) -> tuple[str, pd.DataFrame]:
    """
    Run a Sobol' sensitivity analysis on the fleet-mean Phi_composite
    with respect to small relative perturbations of the five weights.

    Parameters
    ----------
    score_matrix : np.ndarray, shape (n_components, n_features)
        Per-component, per-feature 1-5 bin scores. Phi_composite is
        Phi = 5 * (score_matrix @ weights).
    feature_names : list[str]
        Feature names aligned with the columns of score_matrix.
    nominal_weights : np.ndarray, shape (n_features,)
        The locked nominal weight vector (sums to 1).
    n_base : int
        Sobol' base sample size. Total samples used by SALib is
        n_base * (2*D + 2) where D = number of features.
    weight_scale : float
        Maximum relative perturbation. e.g. 0.15 means each weight is
        sampled within +/- 15 % of its nominal value, then the vector
        is re-normalised to sum to 1.

    Returns
    -------
    output_path, indices_df
        Path to the rendered bar chart and a DataFrame with first-order
        and total-order Sobol' indices per feature weight.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    D = len(nominal_weights)
    bounds = [
        [
            float(nominal_weights[i] * (1 - weight_scale)),
            float(nominal_weights[i] * (1 + weight_scale)),
        ]
        for i in range(D)
    ]
    problem = {
        "num_vars": D,
        "names": [f"w_{n}" for n in feature_names],
        "bounds": bounds,
    }
    samples = sobol_sample.sample(problem, n_base, calc_second_order=False, seed=seed)
    # Re-normalise each sample so the weight vector sums to 1 (the
    # constraint of the original Phi formula).
    samples_norm = samples / np.clip(samples.sum(axis=1, keepdims=True), 1e-9, None)

    # Vectorised fleet-mean Phi for each parameter draw.
    fleet_phi = 5.0 * (samples_norm @ score_matrix.T).mean(axis=1)

    Si = sobol_analyze.analyze(
        problem, fleet_phi, calc_second_order=False, print_to_console=False
    )
    indices_df = pd.DataFrame(
        {
            "weight": problem["names"],
            "S1": Si["S1"],
            "ST": Si["ST"],
            "S1_conf": Si["S1_conf"],
            "ST_conf": Si["ST_conf"],
        }
    )

    # ----------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(D)
    width = 0.4
    ax.bar(
        x - width / 2,
        indices_df["S1"],
        width,
        yerr=indices_df["S1_conf"],
        label="First-order S1",
        color="#1f77b4",
        capsize=4,
    )
    ax.bar(
        x + width / 2,
        indices_df["ST"],
        width,
        yerr=indices_df["ST_conf"],
        label="Total-order ST",
        color="#d62728",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(indices_df["weight"], rotation=15)
    ax.set_ylabel("Sobol index")
    ax.set_title(
        f"Phi_composite weight sensitivity (Sobol', +/- {int(weight_scale * 100)} %)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path, indices_df


__all__ = ["sobol_phi_weight_sensitivity"]
