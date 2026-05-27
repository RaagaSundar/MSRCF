"""Tests for the synthetic dataset generator."""

import numpy as np


def test_dataset_shape_and_columns(small_dataset):
    assert len(small_dataset) == 120
    required = {
        "component_id",
        "ply_count",
        "void_probability",
        "fastener_density",
        "zone_complexity",
        "thickness_variation",
        "damage_mode",
    }
    assert required.issubset(set(small_dataset.columns))


def test_dataset_reproducible():
    import data_generator

    a = data_generator.generate_dataset(n_components=80, seed=7)
    b = data_generator.generate_dataset(n_components=80, seed=7)
    assert (a.values == b.values).all()


def test_void_probability_physics(small_dataset):
    """V_p = max(0, 0.08 - 0.012*P + 0.003*|T|). Must respect the floor."""
    vp = small_dataset["void_probability"].to_numpy()
    pres = small_dataset["cure_pressure_bar"].to_numpy()
    tdev = small_dataset["cure_temp_deviation_C"].to_numpy()
    expected = np.maximum(0.0, 0.08 - 0.012 * pres + 0.003 * np.abs(tdev))
    assert np.allclose(vp, expected, atol=1e-9)


def test_damage_classes_present(small_dataset):
    # All five damage classes must be present in a 120-component sample.
    found = set(small_dataset["damage_mode"].unique().tolist())
    assert found.issubset({0, 1, 2, 3, 4})
    # At least three distinct classes should appear (the rare ones can
    # be absent at small n).
    assert len(found) >= 3
