"""
anomaly_detector.py
===================

Out-of-distribution flagging for incoming components.

Even if the Pillar-2 classifier returns a confident class prediction, a
component whose manufacturing features lie far outside the training
distribution is a candidate for manual review. We fit an
IsolationForest on the five Phi features and emit:

    - is_anomaly  : boolean per component (1 = anomaly, 0 = nominal)
    - anomaly_score : the IsolationForest decision_function (higher
                      means MORE anomalous, after sign-flipping the
                      raw sklearn output)

These augment the RCS dashboard: an anomalous component automatically
gets escalated regardless of its RCS flag, because the model's
confidence on it is suspect.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def fit_anomaly_detector(
    df: pd.DataFrame,
    feature_columns: list[str],
    contamination: float = 0.05,
    seed: int = 42,
) -> tuple[IsolationForest, StandardScaler, pd.DataFrame]:
    """
    Fit an IsolationForest on the listed feature columns and return
    (model, scaler, scored_df). The scored DataFrame adds two columns:

        - anomaly_score : higher = more anomalous (sign-flipped
                          decision_function)
        - is_anomaly    : 1 if predicted anomalous, 0 otherwise
    """
    X = df[feature_columns].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(Xs)
    raw = model.decision_function(Xs)  # higher = MORE normal
    is_anom = model.predict(Xs) == -1

    out = df.copy()
    # Flip sign so that "higher == more anomalous", which is the
    # intuitive direction for an operator.
    out["anomaly_score"] = -raw
    out["is_anomaly"] = is_anom.astype(int)
    return model, scaler, out


__all__ = ["fit_anomaly_detector"]
