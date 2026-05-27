"""
damage_predictor.py
===================

Pillar 2 of MSRCF: supervised damage-mode classification.

Given the five manufacturing-complexity features for a component, predict
which of the five damage classes it is most likely to develop in service:

    0 = No significant damage
    1 = Matrix cracking
    2 = Delamination
    3 = Fiber breakage
    4 = Fatigue crack initiation

We train and compare five classical models per the project brief:

    - Support Vector Machine (RBF kernel)
    - Random Forest
    - Logistic Regression (multinomial)
    - K-Nearest Neighbours
    - XGBoost

For each model we report accuracy, macro precision/recall/F1 on a
held-out 20% stratified test split, plus 5-fold cross-validated
accuracy on the training portion. Confusion matrices are produced
downstream by dashboard.py.

The XGBoost model is also persisted internally as the *production*
classifier used by rcs_engine.py because in spot benchmarks it
delivered the best macro-F1 on this synthetic dataset; the choice is
re-validated each run in `select_best_model()`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

FEATURE_COLUMNS = (
    "ply_count",
    "void_probability",
    "fastener_density",
    "zone_complexity",
    "thickness_variation",
)
TARGET_COLUMN = "damage_mode"

DAMAGE_CLASS_NAMES = {
    0: "No damage",
    1: "Matrix cracking",
    2: "Delamination",
    3: "Fiber breakage",
    4: "Fatigue crack",
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    """Per-model evaluation result kept around for downstream plotting."""

    name: str
    estimator: Pipeline
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    confusion: np.ndarray
    class_labels: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def _build_estimators() -> dict[str, Pipeline]:
    """
    Build the five candidate classifiers wrapped in standard-scaling
    pipelines. Hyperparameters are deliberately conservative defaults -
    the goal of this study is *comparative*, not state-of-the-art tuned
    performance.

    Notes on choices:
        - SVC with RBF kernel and probability=True so we can get
          calibrated posteriors later for the Bayesian RCS update.
        - LogisticRegression: 'lbfgs' solver is the multinomial default;
          we lift max_iter to 1000 to silence convergence warnings on
          near-collinear features.
        - KNN k=5 is the standard textbook default.
        - RandomForest n_estimators=300 trades a small runtime cost for
          stable cross-validated metrics on a 500-row dataset.
        - XGBoost uses softprob so we get class-probability outputs and
          turns off the deprecated label-encoder.
    """
    estimators: dict[str, Pipeline] = {
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        probability=True,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.1,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        tree_method="hist",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    return estimators


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(
    df: pd.DataFrame,
    feature_columns: Iterable[str] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.20,
    n_splits: int = 5,
) -> tuple[dict[str, ModelResult], dict[str, np.ndarray]]:
    """
    Train all five classifiers on `df` and evaluate them on a stratified
    held-out test split + 5-fold stratified cross-validation on the
    training half. Returns (results dict, splits dict).

    The `splits` dict exposes the X_train/X_test/y_train/y_test arrays
    so callers (notably main.py) can pass them on to the RCS engine
    without re-splitting and risking a different split.
    """
    feature_columns = list(feature_columns)
    X = df[feature_columns].to_numpy(dtype=float)
    y = df[target_column].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    estimators = _build_estimators()
    results: dict[str, ModelResult] = {}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    class_labels = sorted(np.unique(y).tolist())

    with warnings.catch_warnings():
        # Suppress benign sklearn FutureWarnings about multinomial default
        # and similar deprecations that aren't actionable here.
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        for name, est in estimators.items():
            cv_scores = cross_val_score(est, X_train, y_train, cv=cv, scoring="accuracy")
            est.fit(X_train, y_train)
            y_pred = est.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(
                y_test, y_pred, average="macro", zero_division=0
            )
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=class_labels)

            results[name] = ModelResult(
                name=name,
                estimator=est,
                accuracy=acc,
                precision_macro=prec,
                recall_macro=rec,
                f1_macro=f1,
                cv_accuracy_mean=float(np.mean(cv_scores)),
                cv_accuracy_std=float(np.std(cv_scores)),
                confusion=cm,
                class_labels=class_labels,
            )

    splits = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    return results, splits


def select_best_model(results: dict[str, ModelResult]) -> ModelResult:
    """
    Pick the production classifier as the one with the highest macro-F1
    on the held-out test set, breaking ties by accuracy.
    """
    best = max(results.values(), key=lambda r: (r.f1_macro, r.accuracy))
    return best


# ---------------------------------------------------------------------------
# Optional grid search for the winning baseline model
# ---------------------------------------------------------------------------
def tune_model(
    name: str,
    base_estimator: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    param_grid: dict,
    cv_splits: int = 5,
) -> ModelResult:
    """
    Run a GridSearchCV over `param_grid` with stratified CV, refit on
    the full training half, and return a ModelResult on the held-out
    test set. Used when --tune-best is requested on the CLI.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        search = GridSearchCV(
            base_estimator,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_est: Pipeline = search.best_estimator_
        y_pred = best_est.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    class_labels = sorted(np.unique(y_train).tolist())
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)

    cv_mean = float(search.best_score_)
    # cv_results_["std_test_score"] is aligned with mean_test_score; we
    # pull the std at the best-scoring index for a like-for-like number.
    best_idx = int(np.argmax(search.cv_results_["mean_test_score"]))
    cv_std = float(search.cv_results_["std_test_score"][best_idx])

    return ModelResult(
        name=f"{name}+tuned",
        estimator=best_est,
        accuracy=acc,
        precision_macro=prec,
        recall_macro=rec,
        f1_macro=f1,
        cv_accuracy_mean=cv_mean,
        cv_accuracy_std=cv_std,
        confusion=cm,
        class_labels=class_labels,
    )


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------
def results_to_dataframe(results: dict[str, ModelResult]) -> pd.DataFrame:
    """
    Format the model results as a tidy DataFrame, sorted by macro-F1.
    """
    rows = []
    for r in results.values():
        rows.append(
            {
                "model": r.name,
                "accuracy": round(r.accuracy, 4),
                "precision_macro": round(r.precision_macro, 4),
                "recall_macro": round(r.recall_macro, 4),
                "f1_macro": round(r.f1_macro, 4),
                "cv_acc_mean": round(r.cv_accuracy_mean, 4),
                "cv_acc_std": round(r.cv_accuracy_std, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)


def predict_probabilities(
    estimator: Pipeline, X: np.ndarray, class_labels: list[int]
) -> np.ndarray:
    """
    Predict per-class probabilities for X, aligned to `class_labels`.

    Sklearn pipelines expose `predict_proba` which returns an array
    whose columns follow `estimator.classes_`. We re-order to the
    canonical class-label order so downstream code can index by class id.
    """
    proba = estimator.predict_proba(X)
    pipe_classes = list(estimator.named_steps["clf"].classes_)
    # Build a column permutation that maps pipe_classes -> class_labels.
    out = np.zeros((proba.shape[0], len(class_labels)), dtype=float)
    for j, c in enumerate(class_labels):
        if c in pipe_classes:
            out[:, j] = proba[:, pipe_classes.index(c)]
    # Normalize defensively (in case a label was unseen).
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    out = out / row_sums
    return out


__all__ = [
    "DAMAGE_CLASS_NAMES",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "ModelResult",
    "train_and_evaluate",
    "select_best_model",
    "tune_model",
    "results_to_dataframe",
    "predict_probabilities",
]


if __name__ == "__main__":
    import data_generator

    df = data_generator.generate_dataset()
    results, splits = train_and_evaluate(df)
    summary = results_to_dataframe(results)
    print(summary.to_string(index=False))
    best = select_best_model(results)
    print(f"\nBest model: {best.name} (macro-F1={best.f1_macro:.4f})")
