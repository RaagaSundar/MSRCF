"""Tests for the statistical evaluation harness (experiments.py).

These validate the *statistics*, not just that the code runs: the
confidence-interval half-width is checked against scipy's t-interval, and
the Nemenyi critical difference is checked against its closed form and a
hand-computed value.
"""

import numpy as np
import pytest
from scipy import stats

import experiments as ex


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------
def test_mean_ci_matches_scipy_t_interval():
    rng = np.random.default_rng(0)
    x = rng.normal(0.8, 0.05, size=25)
    mean, half, std = ex.mean_ci(x, confidence=0.95)

    # Reference: scipy t-interval half-width = t * s / sqrt(n).
    n = len(x)
    s = np.std(x, ddof=1)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    expected_half = tcrit * s / np.sqrt(n)

    assert mean == pytest.approx(float(np.mean(x)))
    assert std == pytest.approx(float(s))
    assert half == pytest.approx(float(expected_half), rel=1e-9)


def test_mean_ci_single_value_has_zero_width():
    mean, half, std = ex.mean_ci(np.array([0.5]))
    assert mean == 0.5
    assert half == 0.0
    assert std == 0.0


def test_mean_ci_widens_with_lower_n():
    rng = np.random.default_rng(1)
    big = rng.normal(0, 1, size=200)
    small = big[:5]
    _, half_big, _ = ex.mean_ci(big)
    _, half_small, _ = ex.mean_ci(small)
    # Fewer samples -> wider interval (all else equal-ish).
    assert half_small > half_big


# ---------------------------------------------------------------------------
# Nemenyi critical difference
# ---------------------------------------------------------------------------
def test_nemenyi_cd_closed_form():
    # CD = q_alpha * sqrt( k(k+1) / (6N) ). For k=5, N=10, q=2.728:
    k, n = 5, 10
    expected = 2.728 * np.sqrt(k * (k + 1) / (6 * n))
    got = ex.nemenyi_critical_difference(k, n, alpha=0.05)
    assert got == pytest.approx(expected, rel=1e-9)


def test_nemenyi_cd_shrinks_with_more_blocks():
    cd_small = ex.nemenyi_critical_difference(5, 5)
    cd_large = ex.nemenyi_critical_difference(5, 50)
    # More replication -> tighter CD -> easier to detect differences.
    assert cd_large < cd_small


def test_nemenyi_unsupported_k_raises():
    with pytest.raises(ValueError):
        ex.nemenyi_critical_difference(k=99, n_blocks=10)


# ---------------------------------------------------------------------------
# End-to-end multi-seed harness (tiny config for speed)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_multiseed_smoke():
    result = ex.run_multiseed_benchmark(seeds=[42, 43, 44], n_components=120)
    # Five models, three seeds.
    assert set(result.f1_matrix.columns) >= {
        "XGBoost",
        "RandomForest",
        "SVM",
        "KNN",
        "LogisticRegression",
    }
    assert result.f1_matrix.shape[0] == 3
    # Average ranks must be a permutation-consistent set in [1, k].
    assert result.average_ranks.min() >= 1.0
    assert result.average_ranks.max() <= 5.0
    # Friedman p-value is a valid probability.
    assert 0.0 <= result.friedman_p <= 1.0
    # Summary table sorted by macro-F1 mean descending.
    f1_means = result.summary["f1_macro_mean"].to_numpy()
    assert np.all(np.diff(f1_means) <= 1e-9)
