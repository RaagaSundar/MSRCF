"""
conformal.py
============

Distribution-free uncertainty quantification for the Pillar-2 damage
classifier via *split (inductive) conformal prediction*.

A point prediction ("this panel will delaminate") hides how sure the
model is. Conformal prediction instead returns a *set* of damage modes
with a finite-sample, distribution-free guarantee: for a user-chosen
miscoverage rate alpha, the true damage mode is contained in the
predicted set with probability at least 1 - alpha, regardless of the
classifier, the feature distribution, or the sample size. The only
assumption is exchangeability of the calibration and test data.

That guarantee is exactly the language a certification authority speaks:
not "the model is 92 % accurate on a test set" but "with 90 % confidence
the true failure mode is one of {delamination, matrix cracking}". A
singleton set is a confident call; a large set is the model honestly
saying "inspect for several modes".

We implement two standard non-conformity scores and report the trade-off
between them:

    LAC  (Least Ambiguous set-valued Classifier; Sadinle et al., 2019):
         score s = 1 - p_hat(y | x). Produces the smallest sets that
         still attain marginal coverage, but can under-cover hard
         classes.
    APS  (Adaptive Prediction Sets; Romano, Sesia & Candes, 2020):
         the cumulative sorted-probability mass up to the true class.
         Slightly larger sets, but adapts set size to per-instance
         difficulty and gives better class-conditional coverage.

The calibration quantile uses the finite-sample correction
q_hat = ceil((n+1)(1-alpha)) / n empirical quantile of the calibration
scores, which is what makes the >= 1 - alpha guarantee exact rather than
asymptotic (Vovk et al., 2005; Angelopoulos & Bates, 2023).

To keep the procedure statistically valid we never calibrate on data the
model was fitted on: a fresh clone of the production estimator is trained
on a disjoint fit split, calibrated on a held-out calibration split, and
evaluated on a test split.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110


# ---------------------------------------------------------------------------
# Core conformal primitives (pure functions over probability matrices)
# ---------------------------------------------------------------------------
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Finite-sample conformal quantile of calibration `scores`.

    Returns the ceil((n+1)(1-alpha))-th smallest score, which is the
    threshold that guarantees marginal coverage >= 1 - alpha on an
    exchangeable test point. When the requested rank exceeds n (alpha so
    small the guarantee needs more calibration points than we have) the
    threshold is +inf, i.e. the prediction set is the full label space.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    k = max(k, 1)
    return float(np.sort(scores)[k - 1])


def lac_scores(proba: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    """LAC non-conformity score for the true class: s = 1 - p_hat(y)."""
    n = proba.shape[0]
    return 1.0 - proba[np.arange(n), y_idx]


def lac_sets(proba: np.ndarray, qhat: float) -> np.ndarray:
    """
    LAC prediction sets: include every class whose probability is high
    enough that its score 1 - p clears the calibrated threshold, i.e.
    p_hat(k) >= 1 - qhat. Returns a boolean (n_samples, n_classes) mask.
    """
    return proba >= (1.0 - qhat)


def aps_scores(proba: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    """
    APS non-conformity score: the cumulative probability mass of all
    classes ranked at least as likely as the true class (the true class
    included). Uses the non-randomised variant for reproducibility,
    which is marginally conservative.
    """
    n, _ = proba.shape
    order = np.argsort(-proba, axis=1, kind="stable")  # descending probability
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    # Position of each class within the descending order.
    ranks = np.empty_like(order)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(proba.shape[1])[None, :]
    pos = ranks[np.arange(n), y_idx]
    return cum[np.arange(n), pos]


def aps_sets(proba: np.ndarray, qhat: float) -> np.ndarray:
    """
    APS prediction sets: walking classes in descending probability,
    include the smallest top-set whose cumulative mass reaches `qhat`
    (the class that tips the sum over the threshold is included).
    Returns a boolean (n_samples, n_classes) mask.
    """
    n, k = proba.shape
    order = np.argsort(-proba, axis=1, kind="stable")
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    prefix_before = cum - sorted_p  # cumulative mass *excluding* the current class
    keep_sorted = prefix_before < qhat
    mask = np.zeros((n, k), dtype=bool)
    rows = np.arange(n)[:, None]
    mask[rows, order] = keep_sorted
    return mask


def empirical_coverage(sets_mask: np.ndarray, y_idx: np.ndarray) -> float:
    """Fraction of test points whose prediction set contains the truth."""
    n = sets_mask.shape[0]
    if n == 0:
        return float("nan")
    return float(sets_mask[np.arange(n), y_idx].mean())


def mean_set_size(sets_mask: np.ndarray) -> float:
    """Average prediction-set cardinality (the efficiency of the method)."""
    if sets_mask.shape[0] == 0:
        return float("nan")
    return float(sets_mask.sum(axis=1).mean())


# Registry of (calibration-score, set-construction) function pairs.
_METHODS: dict[str, tuple] = {
    "LAC": (lac_scores, lac_sets),
    "APS": (aps_scores, aps_sets),
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ConformalResult:
    """Outputs of a conformal analysis kept around for reporting."""

    output_path: str
    metrics_df: pd.DataFrame  # one row per (method, alpha)
    per_class_df: pd.DataFrame  # class-conditional coverage at headline alpha
    alpha: float
    class_labels: list[int]

    def headline(self) -> pd.DataFrame:
        """Metrics rows at the headline alpha, one per method."""
        return self.metrics_df[np.isclose(self.metrics_df["alpha"], self.alpha)].copy()


# ---------------------------------------------------------------------------
# Probability alignment
# ---------------------------------------------------------------------------
def _predict_proba_aligned(estimator, X: np.ndarray, class_labels: list[int]) -> np.ndarray:
    """
    Predict class probabilities re-ordered to `class_labels`, tolerating
    either a Pipeline (with a named 'clf' step) or a bare estimator.
    Rows are renormalised defensively in case a label was unseen in the
    fit split.
    """
    proba = estimator.predict_proba(X)
    if hasattr(estimator, "named_steps") and "clf" in estimator.named_steps:
        fitted_classes = list(estimator.named_steps["clf"].classes_)
    else:
        fitted_classes = list(estimator.classes_)
    out = np.zeros((proba.shape[0], len(class_labels)), dtype=float)
    for j, c in enumerate(class_labels):
        if c in fitted_classes:
            out[:, j] = proba[:, fitted_classes.index(c)]
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _plot_conformal(metrics_df: pd.DataFrame, alpha: float, output_path: str) -> None:
    palette = {"LAC": "#1f77b4", "APS": "#d62728"}
    markers = {"LAC": "s", "APS": "o"}
    methods = list(metrics_df["method"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel 1: empirical coverage vs target (the guarantee, visualised).
    ax = axes[0]
    lo = float(min(metrics_df["target_coverage"].min(), metrics_df["empirical_coverage"].min()))
    lims = [max(0.0, lo - 0.03), 1.01]
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1.0, label="ideal (y = x)")
    for m in methods:
        d = metrics_df[metrics_df["method"] == m].sort_values("target_coverage")
        ax.plot(
            d["target_coverage"],
            d["empirical_coverage"],
            marker=markers.get(m, "o"),
            color=palette.get(m, None),
            label=m,
        )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("target coverage  (1 - alpha)")
    ax.set_ylabel("empirical coverage on test")
    ax.set_title("Coverage calibration (split conformal)")
    ax.legend(loc="lower right")

    # Panel 2: efficiency (mean set size) vs target coverage.
    ax2 = axes[1]
    for m in methods:
        d = metrics_df[metrics_df["method"] == m].sort_values("target_coverage")
        ax2.plot(
            d["target_coverage"],
            d["mean_set_size"],
            marker=markers.get(m, "o"),
            color=palette.get(m, None),
            label=m,
        )
    ax2.axvline(
        1.0 - alpha, ls=":", color="black", lw=1.0, label=f"headline 1 - alpha = {1 - alpha:.2f}"
    )
    ax2.set_xlabel("target coverage  (1 - alpha)")
    ax2.set_ylabel("mean prediction-set size (classes)")
    ax2.set_title("Efficiency vs coverage")
    ax2.legend(loc="upper left")

    fig.suptitle(
        "Conformal prediction: distribution-free coverage for the damage classifier",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# End-to-end analysis
# ---------------------------------------------------------------------------
def run_conformal_analysis(
    df: pd.DataFrame,
    estimator,
    feature_columns: list[str],
    target_column: str,
    class_labels: list[int] | None = None,
    class_names: dict[int, str] | None = None,
    alpha: float = 0.10,
    alpha_sweep: tuple[float, ...] = (0.01, 0.05, 0.10, 0.15, 0.20),
    calib_fraction: float = 0.30,
    test_fraction: float = 0.20,
    seed: int = 42,
    output_path: str = "results/conformal_coverage.png",
) -> ConformalResult:
    """
    Run split conformal prediction on `df` using a fresh clone of
    `estimator`, sweeping miscoverage rates in `alpha_sweep` for both the
    LAC and APS scores.

    The data is split three ways (stratified, disjoint): a fit split to
    train the cloned model, a calibration split to compute the conformal
    threshold, and a test split to measure realised coverage and set
    size. This disjointness is what keeps the >= 1 - alpha guarantee
    valid.

    Returns
    -------
    ConformalResult
        Holds the per-(method, alpha) metrics table, the class-conditional
        coverage table at the headline `alpha`, and the figure path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    feature_columns = list(feature_columns)
    X = df[feature_columns].to_numpy(dtype=float)
    y = df[target_column].to_numpy(dtype=int)

    if class_labels is None:
        class_labels = sorted(np.unique(y).tolist())
    class_labels = list(class_labels)
    label_to_idx = {c: i for i, c in enumerate(class_labels)}

    # Three-way stratified split: fit / calibrate / test.
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y
    )
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_tr, y_tr, test_size=calib_fraction, random_state=seed, stratify=y_tr
    )

    model = clone(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_fit, y_fit)

    p_cal = _predict_proba_aligned(model, X_cal, class_labels)
    p_test = _predict_proba_aligned(model, X_test, class_labels)
    y_cal_idx = np.array([label_to_idx[int(c)] for c in y_cal])
    y_test_idx = np.array([label_to_idx[int(c)] for c in y_test])

    # Always include the headline alpha in the sweep; descending target.
    sweep = sorted({float(a) for a in alpha_sweep} | {float(alpha)}, reverse=True)

    rows: list[dict] = []
    for a in sweep:
        for method, (score_fn, set_fn) in _METHODS.items():
            qhat = conformal_quantile(score_fn(p_cal, y_cal_idx), a)
            sets = set_fn(p_test, qhat)
            sizes = sets.sum(axis=1)
            rows.append(
                {
                    "method": method,
                    "alpha": a,
                    "target_coverage": 1.0 - a,
                    "empirical_coverage": empirical_coverage(sets, y_test_idx),
                    "mean_set_size": mean_set_size(sets),
                    "singleton_rate": float((sizes == 1).mean()),
                    "empty_rate": float((sizes == 0).mean()),
                    "qhat": qhat,
                }
            )
    metrics_df = pd.DataFrame(rows).sort_values(["method", "alpha"]).reset_index(drop=True)

    # Class-conditional coverage at the headline alpha.
    per_rows: list[dict] = []
    for method, (score_fn, set_fn) in _METHODS.items():
        qhat = conformal_quantile(score_fn(p_cal, y_cal_idx), alpha)
        sets = set_fn(p_test, qhat)
        hit = sets[np.arange(len(y_test_idx)), y_test_idx]
        for c in class_labels:
            mask = y_test_idx == label_to_idx[c]
            support = int(mask.sum())
            per_rows.append(
                {
                    "method": method,
                    "class_id": c,
                    "class_name": (class_names or {}).get(c, str(c)),
                    "support": support,
                    "coverage": float(hit[mask].mean()) if support else float("nan"),
                }
            )
    per_class_df = pd.DataFrame(per_rows)

    _plot_conformal(metrics_df, alpha, output_path)

    return ConformalResult(
        output_path=output_path,
        metrics_df=metrics_df,
        per_class_df=per_class_df,
        alpha=float(alpha),
        class_labels=class_labels,
    )


__all__ = [
    "ConformalResult",
    "aps_scores",
    "aps_sets",
    "conformal_quantile",
    "empirical_coverage",
    "lac_scores",
    "lac_sets",
    "mean_set_size",
    "run_conformal_analysis",
]
