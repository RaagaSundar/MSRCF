"""Tests for the RUL extrapolator."""

import numpy as np

import rul_predictor


def test_rul_monotone_or_naninf(scored_dataset):
    n = len(scored_dataset)
    phi = scored_dataset["phi_composite"].to_numpy()
    last_p = np.full(n, 0.4)
    classifier = np.full(n, 0.6)
    result = rul_predictor.forecast_rul(
        component_ids=scored_dataset["component_id"].tolist(),
        phi_composite=phi,
        last_p_damage=last_p,
        classifier_damage=classifier,
        last_cycle=10,
        horizon_cycle=40,
    )
    # cycles_to_yellow <= cycles_to_red where both are finite.
    both = ~np.isnan(result.cycles_to_yellow) & ~np.isnan(result.cycles_to_red)
    assert (
        result.cycles_to_yellow[both] <= result.cycles_to_red[both] + 1e-9
    ).all()
    assert result.rcs_extended.shape == (41, n)


def test_rul_dataframe_shape():
    n = 5
    phi = np.array([10.0, 15.0, 20.0, 22.0, 8.0])
    last_p = np.array([0.05, 0.30, 0.60, 0.80, 0.02])
    classifier = np.array([0.05, 0.40, 0.70, 0.90, 0.02])
    result = rul_predictor.forecast_rul(
        component_ids=[f"X{i}" for i in range(n)],
        phi_composite=phi,
        last_p_damage=last_p,
        classifier_damage=classifier,
        last_cycle=10,
        horizon_cycle=30,
    )
    df = result.to_dataframe()
    assert list(df.columns) == [
        "component_id",
        "cycles_to_yellow",
        "cycles_to_red",
        "rcs_final_extrapolated",
    ]
    assert len(df) == n
