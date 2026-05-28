"""
explainability.py
=================

Post-training diagnostics for the MSRCF Pillar-2 classifier:

    - SHAP feature attribution (tree-explainer for tree models;
      falls back to a permutation-importance proxy otherwise).
    - One-vs-rest ROC + AUC curves per damage class.
    - Calibration curves + Brier scores per class.

Each function returns the figure path (or computed dataframe) so the
orchestrator can announce artefacts to stdout.
"""

from __future__ import annotations

import os
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, brier_score_loss, roc_curve

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110


# ---------------------------------------------------------------------------
# SHAP feature attribution
# ---------------------------------------------------------------------------
def shap_feature_importance(
    estimator,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    class_names: dict[int, str],
    output_path: str,
    max_explain: int = 200,
) -> tuple[str, pd.DataFrame]:
    """
    Compute mean(|SHAP|) per feature x class.

    For tree-based models we use shap.TreeExplainer; otherwise we
    fall back to sklearn permutation_importance, which provides a
    model-agnostic feature attribution that is consistent in
    interpretation (higher = more influential).

    Saves a heatmap figure and returns (path, dataframe).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_explain = X_explain[:max_explain]

    importance_df: pd.DataFrame
    using_shap = False
    try:
        import shap  # type: ignore

        # Reach into the sklearn pipeline to pull the underlying
        # classifier and pre-scaled feature matrix.
        clf = getattr(estimator, "named_steps", {}).get("clf", estimator)
        scaler = getattr(estimator, "named_steps", {}).get("scaler", None)
        Xe_proc = scaler.transform(X_explain) if scaler is not None else X_explain

        # TreeExplainer is the fast path for XGBoost / RandomForest.
        explainer = shap.TreeExplainer(clf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(Xe_proc)
        using_shap = True

        # shap_values can be a list (multi-class) or 3-D array.
        if isinstance(shap_values, list):
            arrays = shap_values
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # New SHAP returns (n_samples, n_features, n_classes)
            arrays = [shap_values[:, :, k] for k in range(shap_values.shape[2])]
        else:
            arrays = [shap_values]

        mean_abs = np.stack([np.abs(a).mean(axis=0) for a in arrays], axis=0)
        # mean_abs shape: (n_classes, n_features). Build a DataFrame.
        classes = sorted(class_names.keys())[: mean_abs.shape[0]]
        importance_df = pd.DataFrame(
            mean_abs,
            columns=feature_names,
            index=[class_names.get(c, str(c)) for c in classes],
        )
    except Exception as exc:  # pragma: no cover (fallback only)
        # Fallback: permutation importance.
        warnings.warn(
            f"SHAP failed ({exc}); falling back to permutation importance",
            stacklevel=2,
        )
        result = permutation_importance(
            estimator,
            X_background,
            np.zeros(len(X_background)),
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
            scoring="accuracy",
        )
        importance_df = pd.DataFrame(
            [result.importances_mean],
            columns=feature_names,
            index=["permutation"],
        )

    # Render heatmap.
    fig, ax = plt.subplots(
        figsize=(1.5 + 1.6 * len(feature_names), 1.0 + 0.55 * len(importance_df))
    )
    sns.heatmap(
        importance_df,
        annot=True,
        fmt=".3f",
        cmap="rocket_r",
        cbar=True,
        ax=ax,
    )
    suffix = "SHAP |mean|" if using_shap else "permutation importance"
    ax.set_title(f"Feature attribution ({suffix})")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Class")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path, importance_df


# ---------------------------------------------------------------------------
# ROC + calibration
# ---------------------------------------------------------------------------
def roc_and_calibration(
    estimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_labels: list[int],
    class_names: dict[int, str],
    output_path: str,
) -> tuple[str, pd.DataFrame]:
    """
    Render a two-panel figure: one-vs-rest ROC curves (left) and
    reliability/calibration curves (right) for every class.

    Also returns per-class AUC and Brier score in a DataFrame.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    proba = estimator.predict_proba(X_test)
    # Align proba columns to class_labels.
    pipe_classes = list(estimator.named_steps["clf"].classes_)
    aligned = np.zeros((proba.shape[0], len(class_labels)), dtype=float)
    for j, c in enumerate(class_labels):
        if c in pipe_classes:
            aligned[:, j] = proba[:, pipe_classes.index(c)]
    aligned = aligned / np.clip(aligned.sum(axis=1, keepdims=True), 1e-9, None)

    fig, (ax_roc, ax_cal) = plt.subplots(1, 2, figsize=(14, 6))

    rows = []
    palette = sns.color_palette("tab10", n_colors=len(class_labels))
    ax_roc.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)

    for color, j, c in zip(palette, range(len(class_labels)), class_labels, strict=True):
        y_bin = (y_test == c).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, aligned[:, j])
        auc_score = auc(fpr, tpr)
        ax_roc.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2,
            label=f"{class_names.get(c, c)} (AUC={auc_score:.2f})",
        )
        # Calibration curve - use a modest number of bins given small
        # per-class test set sizes.
        try:
            frac_pos, mean_pred = calibration_curve(
                y_bin, aligned[:, j], n_bins=5, strategy="quantile"
            )
            ax_cal.plot(
                mean_pred,
                frac_pos,
                "-o",
                color=color,
                linewidth=2,
                label=class_names.get(c, str(c)),
            )
        except ValueError:
            pass
        brier = brier_score_loss(y_bin, aligned[:, j])
        rows.append(
            {
                "class": c,
                "class_name": class_names.get(c, str(c)),
                "auc": round(float(auc_score), 4),
                "brier": round(float(brier), 4),
                "positives": int(y_bin.sum()),
            }
        )

    ax_roc.set_title("One-vs-rest ROC curves")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.legend(fontsize=8, loc="lower right")

    ax_cal.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax_cal.set_title("Calibration (reliability) curves")
    ax_cal.set_xlabel("Mean predicted probability")
    ax_cal.set_ylabel("Empirical positive rate")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path, pd.DataFrame(rows)


__all__ = ["shap_feature_importance", "roc_and_calibration"]
