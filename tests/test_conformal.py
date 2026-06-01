"""
Tests for the split-conformal prediction module.

The pure-function tests (quantile, score functions, set construction,
coverage helpers) are fast and deterministic; they pin down the maths
that makes the >= 1 - alpha guarantee exact. The end-to-end test fits a
model and is marked `slow`.
"""

import numpy as np
import pytest

import conformal


# ---------------------------------------------------------------------------
# Finite-sample conformal quantile
# ---------------------------------------------------------------------------
def test_conformal_quantile_closed_form():
    scores = np.arange(1, 11) / 10.0  # [0.1 .. 1.0], n = 10
    # k = ceil((n+1)(1-alpha)); threshold = k-th smallest score.
    assert conformal.conformal_quantile(scores, 0.10) == pytest.approx(1.0)  # k=10
    assert conformal.conformal_quantile(scores, 0.20) == pytest.approx(0.9)  # k=9
    assert conformal.conformal_quantile(scores, 0.50) == pytest.approx(0.6)  # k=6
    # alpha so small the rank exceeds n -> full-label-space threshold.
    assert conformal.conformal_quantile(scores, 0.01) == float("inf")  # k=11 > 10
    # empty calibration set -> infinite threshold.
    assert conformal.conformal_quantile(np.array([]), 0.10) == float("inf")


# ---------------------------------------------------------------------------
# LAC score + set construction
# ---------------------------------------------------------------------------
def test_lac_scores_and_sets():
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    y_idx = np.array([0, 1])
    np.testing.assert_allclose(conformal.lac_scores(proba, y_idx), [0.3, 0.4])

    # qhat = 0.45 -> include classes with p >= 0.55.
    sets = conformal.lac_sets(proba, 0.45)
    np.testing.assert_array_equal(sets, [[True, False, False], [False, True, False]])
    assert conformal.empirical_coverage(sets, y_idx) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# APS score + set construction
# ---------------------------------------------------------------------------
def test_aps_scores_cumulative_mass():
    proba = np.array([[0.5, 0.3, 0.2]] * 3)
    y_idx = np.array([0, 1, 2])  # top, middle, bottom class
    # Cumulative mass up to and including the true class.
    np.testing.assert_allclose(conformal.aps_scores(proba, y_idx), [0.5, 0.8, 1.0])

    # Order-invariance: an unsorted row must score by rank, not column.
    unsorted = np.array([[0.2, 0.5, 0.3]])
    assert conformal.aps_scores(unsorted, np.array([0]))[0] == pytest.approx(1.0)  # weakest
    assert conformal.aps_scores(unsorted, np.array([1]))[0] == pytest.approx(0.5)  # strongest


def test_aps_sets_threshold():
    proba = np.array([[0.5, 0.3, 0.2]])
    # qhat = 0.6 -> smallest top-set reaching 0.6 is {class0, class1}.
    np.testing.assert_array_equal(conformal.aps_sets(proba, 0.6), [[True, True, False]])
    # qhat = 0.4 -> just the argmax.
    np.testing.assert_array_equal(conformal.aps_sets(proba, 0.4), [[True, False, False]])
    # Infinite threshold -> full label space.
    np.testing.assert_array_equal(conformal.aps_sets(proba, float("inf")), [[True, True, True]])


# ---------------------------------------------------------------------------
# Coverage / efficiency helpers
# ---------------------------------------------------------------------------
def test_coverage_and_size_helpers():
    sets = np.array([[True, False, False], [True, True, False], [False, False, True]])
    assert conformal.empirical_coverage(sets, np.array([0, 1, 2])) == pytest.approx(1.0)
    assert conformal.empirical_coverage(sets, np.array([1, 0, 0])) == pytest.approx(1.0 / 3.0)
    assert conformal.mean_set_size(sets) == pytest.approx(4.0 / 3.0)


# ---------------------------------------------------------------------------
# The headline property: marginal coverage >= 1 - alpha under exchangeability
# ---------------------------------------------------------------------------
def test_split_conformal_marginal_coverage_synthetic():
    """
    With perfectly calibrated probabilities (labels drawn from the same
    distribution used as the score), split conformal must attain *at
    least* 1-alpha marginal coverage for both LAC and APS -- that is the
    theorem. LAC tracks the target tightly; the non-randomised APS used
    here is deliberately conservative and over-covers, so we only assert
    the lower-bound guarantee plus non-trivial set sizes.
    """
    rng = np.random.default_rng(0)
    n, k, alpha = 4000, 5, 0.10
    logits = rng.normal(size=(n, k))
    proba = np.exp(logits)
    proba /= proba.sum(axis=1, keepdims=True)
    # Sample labels from each row's own categorical -> calibrated + exchangeable.
    cum = np.cumsum(proba, axis=1)
    y = (rng.random((n, 1)) < cum).argmax(axis=1)

    half = n // 2
    p_cal, y_cal = proba[:half], y[:half]
    p_test, y_test = proba[half:], y[half:]

    for score_fn, set_fn in (
        (conformal.lac_scores, conformal.lac_sets),
        (conformal.aps_scores, conformal.aps_sets),
    ):
        qhat = conformal.conformal_quantile(score_fn(p_cal, y_cal), alpha)
        sets = set_fn(p_test, qhat)
        cov = conformal.empirical_coverage(sets, y_test)
        size = conformal.mean_set_size(sets)
        assert cov >= 0.87  # >= 1 - alpha guarantee (minus finite-sample slack)
        assert cov <= 1.0
        assert 0.0 < size < k  # non-trivial: not empty, not the whole label space


# ---------------------------------------------------------------------------
# End-to-end with a real fitted model
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_run_conformal_end_to_end(tmp_path):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    import damage_predictor as dp
    import data_generator

    df = data_generator.generate_dataset(n_components=240, seed=42)
    estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=1)),
        ]
    )
    out_png = tmp_path / "conformal.png"
    res = conformal.run_conformal_analysis(
        df=df,
        estimator=estimator,
        feature_columns=list(dp.FEATURE_COLUMNS),
        target_column=dp.TARGET_COLUMN,
        class_names=dp.DAMAGE_CLASS_NAMES,
        alpha=0.10,
        seed=42,
        output_path=str(out_png),
    )

    assert out_png.exists()
    n_classes = len(res.class_labels)

    # Sweep default has 5 distinct alphas x 2 methods = 10 rows.
    assert len(res.metrics_df) == 10
    assert set(res.metrics_df["method"]) == {"LAC", "APS"}
    assert res.metrics_df["empirical_coverage"].between(0.0, 1.0).all()
    assert res.metrics_df["mean_set_size"].between(0.0, n_classes + 1e-9).all()

    head = res.headline()
    assert len(head) == 2
    # Conformal coverage is model-agnostic; even on small data it should
    # be comfortably above this floor at the 90 % target.
    assert (head["empirical_coverage"] >= 0.6).all()

    # Per-class table covers every class for both methods.
    assert len(res.per_class_df) == 2 * n_classes
    finite = res.per_class_df["coverage"].dropna()
    assert finite.between(0.0, 1.0).all()
