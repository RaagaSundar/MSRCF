"""Tests for the IsolationForest anomaly layer."""

import anomaly_detector


def test_contamination_rate_matches_request(scored_dataset):
    feature_cols = [
        "ply_count",
        "void_probability",
        "fastener_density",
        "zone_complexity",
        "thickness_variation",
    ]
    _, _, out = anomaly_detector.fit_anomaly_detector(
        scored_dataset, feature_cols, contamination=0.10, seed=42
    )
    n = len(out)
    flagged = int(out["is_anomaly"].sum())
    # IsolationForest's contamination parameter is approximate; allow a
    # generous tolerance.
    assert abs(flagged / n - 0.10) < 0.05


def test_anomaly_score_alignment(scored_dataset):
    feature_cols = [
        "ply_count",
        "void_probability",
        "fastener_density",
        "zone_complexity",
        "thickness_variation",
    ]
    _, _, out = anomaly_detector.fit_anomaly_detector(
        scored_dataset, feature_cols, contamination=0.05, seed=42
    )
    # Anomalous components should have higher anomaly_score on average.
    anom_mean = out.loc[out["is_anomaly"] == 1, "anomaly_score"].mean()
    norm_mean = out.loc[out["is_anomaly"] == 0, "anomaly_score"].mean()
    assert anom_mean > norm_mean
