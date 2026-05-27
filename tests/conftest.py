"""
conftest.py
===========

Test-suite-wide pytest fixtures. Adds the src/ directory to sys.path so
test modules can `import data_generator` directly.
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(HERE), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture(scope="session")
def small_dataset():
    """A 120-component synthetic dataset, small enough for fast tests."""
    import data_generator

    return data_generator.generate_dataset(n_components=120, seed=42)


@pytest.fixture(scope="session")
def scored_dataset(small_dataset):
    """A small dataset with risk_matrix scoring applied."""
    import risk_matrix

    _, scored = risk_matrix.fit_and_score(small_dataset)
    return scored
