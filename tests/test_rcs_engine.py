"""Tests for the RCS engine (Pillar 3)."""

import numpy as np

import rcs_engine


def test_degradation_factor_monotone_and_bounded():
    cycles = np.arange(0, 30)
    d = rcs_engine.degradation_factor(cycles)
    assert d[0] == 0.0
    assert (np.diff(d) >= 0).all()
    assert (d >= 0).all() and (d <= 1).all()


def test_bayesian_update_bounded_between_clipped_limits():
    prior = np.array([0.01, 0.5, 0.99, 0.2])
    post = rcs_engine.bayesian_update(prior, np.array([0.6, 0.6, 0.6, 0.4]), cycle=3)
    assert (post >= 0.01).all() and (post <= 0.99).all()


def test_bayesian_update_increases_under_positive_evidence():
    prior = np.array([0.3, 0.5, 0.7])
    pos = rcs_engine.bayesian_update(prior, np.array([0.9, 0.9, 0.9]), cycle=2)
    assert (pos >= prior).all()


def test_compute_rcs_trajectory_shapes(scored_dataset):
    ids = scored_dataset["component_id"].tolist()
    phi = scored_dataset["phi_composite"].to_numpy()
    # Build a synthetic per-class probability matrix that includes
    # nominal (class 0) and four damaged classes.
    n = len(ids)
    rng = np.random.default_rng(0)
    raw = rng.random((n, 5))
    probs = raw / raw.sum(axis=1, keepdims=True)
    traj = rcs_engine.compute_rcs_trajectory(
        component_ids=ids,
        phi_composite=phi,
        initial_class_probabilities=probs,
        class_labels=[0, 1, 2, 3, 4],
    )
    assert traj.rcs_normalised.shape == (11, n)
    assert traj.rcs_per_class.shape == (11, n, 4)
    assert ((traj.rcs_normalised >= 0) & (traj.rcs_normalised <= 100)).all()


def test_flag_thresholds_consistent():
    assert rcs_engine.flag_for_score(20.0) == "GREEN"
    assert rcs_engine.flag_for_score(50.0) == "YELLOW"
    assert rcs_engine.flag_for_score(80.0) == "RED"
    # Edge cases
    assert rcs_engine.flag_for_score(39.99) == "GREEN"
    assert rcs_engine.flag_for_score(40.0) == "YELLOW"
    assert rcs_engine.flag_for_score(70.0) == "RED"


def test_mc_band_lower_le_median_le_upper(scored_dataset):
    ids = scored_dataset["component_id"].tolist()
    phi = scored_dataset["phi_composite"].to_numpy()
    rng = np.random.default_rng(0)
    raw = rng.random((len(ids), 5))
    probs = raw / raw.sum(axis=1, keepdims=True)
    median, lower, upper = rcs_engine.compute_rcs_mc_band(
        component_ids=ids,
        phi_composite=phi,
        initial_class_probabilities=probs,
        class_labels=[0, 1, 2, 3, 4],
        n_samples=20,
        noise_sigma=0.8,
        seed=0,
    )
    assert (lower <= median + 1e-9).all()
    assert (median <= upper + 1e-9).all()
