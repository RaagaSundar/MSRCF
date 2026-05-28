"""Smoke + sanity tests for the ablation study (ablation.py).

Kept deliberately small (few seeds, small fleet) so CI stays fast; the
full-scale numbers are produced by `python src/ablation.py`.
"""

import numpy as np
import pytest

import ablation

pytestmark = pytest.mark.slow


def test_feature_ablation_covers_all_features():
    df_feat = ablation.run_feature_ablation(seeds=[42, 43], n_components=80)
    # One row per feature, all five present.
    assert set(df_feat["removed_feature"]) == set(ablation.FEATURES)
    # Deltas and CIs must be finite numbers.
    assert np.isfinite(df_feat["delta_f1_mean"].to_numpy()).all()
    assert np.isfinite(df_feat["delta_f1_ci95"].to_numpy()).all()
    # The full-model F1 baseline is attached and in [0, 1].
    assert 0.0 <= df_feat.attrs["full_f1_mean"] <= 1.0


def test_design_ablation_phi_prior_widens_yellow_band():
    """
    The headline claim of the Phi-blended prior is that it keeps the
    YELLOW band from collapsing. The 'full' config (blend=0.55) should
    therefore have a YELLOW mean no smaller than the classifier-only
    'no_phi_prior' config (blend=1.0).
    """
    df_design = ablation.run_design_ablation(seeds=[42, 43], n_components=120)
    df_design = df_design.set_index("config")
    full_yellow = df_design.loc["full", "YELLOW_mean"]
    no_prior_yellow = df_design.loc["no_phi_prior", "YELLOW_mean"]
    assert full_yellow >= no_prior_yellow
