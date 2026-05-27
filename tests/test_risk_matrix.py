"""Tests for the Phi_composite risk matrix scorer."""

import numpy as np


def test_phi_within_expected_range(scored_dataset):
    phi = scored_dataset["phi_composite"].to_numpy()
    # Phi = 5 * (weighted sum of 1-5 scores) with weights summing to 1,
    # so Phi must live in [5, 25].
    assert phi.min() >= 5.0 - 1e-9
    assert phi.max() <= 25.0 + 1e-9


def test_score_bins_are_integers_1_to_5(scored_dataset):
    for feat in [
        "ply_count",
        "void_probability",
        "fastener_density",
        "zone_complexity",
        "thickness_variation",
    ]:
        col = scored_dataset[f"score_{feat}"]
        assert col.between(1, 5).all()
        assert (col.astype(int) == col).all()


def test_weights_sum_to_one():
    from risk_matrix import WEIGHTS

    assert np.isclose(WEIGHTS.sum(), 1.0)


def test_risk_tiers_cover_population(scored_dataset):
    found = set(scored_dataset["risk_tier"].astype(str).unique())
    # The 120-component sample is large enough to cover at least three
    # tiers reliably.
    assert len(found.intersection({"Low", "Moderate", "High", "Critical"})) >= 3
