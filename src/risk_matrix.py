"""
risk_matrix.py
==============

Pillar 1 of the Manufacture-to-Service Risk Continuity Framework: the
generalized composite-panel risk matrix.

The original Phi(Z, H, N) formulation from Paper 2 was developed for
electrical wiring harness complexity scoring, where:
    Z = number of zones
    H = harness routes
    N = number of conductors

This module generalizes that idea to aerospace CFRP structural panels.
Five manufacturing-complexity features are scored individually on a 1-5
"very low" to "very high" scale using *percentile-based* thresholds
computed from the dataset (so the scorer adapts to whatever population
of components is being analysed), then combined with a fixed weight
vector justified by composites-failure literature:

    w_void  = 0.30   (highest - voids are the dominant delamination
                      precursor; see Mesogitis et al., 2014 and
                      Talreja & Singh, 2012, Ch. 4)
    w_plies = 0.25   (laminate thickness drives interlaminar shear and
                      delamination resistance; Liu et al., 2006)
    w_zones = 0.20   (geometric complexity correlates with cure
                      non-uniformity; Potter, 2009)
    w_fast  = 0.15   (fastener density mostly contributes to local
                      matrix cracking at hole boundaries)
    w_thick = 0.10   (ply thickness variation is a secondary driver
                      relative to bulk void content)

The output is Phi_composite, a continuous score on roughly [5, 25].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature ordering and weights (locked - do not reorder without updating the
# weight vector and the percentile threshold layout).
# ---------------------------------------------------------------------------
FEATURE_ORDER: tuple[str, ...] = (
    "ply_count",
    "void_probability",
    "fastener_density",
    "zone_complexity",
    "thickness_variation",
)

# Brief mandates weights [0.25, 0.30, 0.15, 0.20, 0.10] in the
# (ply_count, void_probability, fastener_density, zone_complexity,
#  thickness_variation) order.
WEIGHTS: np.ndarray = np.array([0.25, 0.30, 0.15, 0.20, 0.10])

# Percentile boundaries used to map a raw feature value to a 1-5 bin.
# The five bins (Very Low ... Very High) are defined as:
#     bin 1: value <  P20
#     bin 2: P20 <= value < P40
#     bin 3: P40 <= value < P60
#     bin 4: P60 <= value < P80
#     bin 5: value >= P80
# This guarantees an approximately uniform-by-rank score regardless of
# the underlying feature distribution.
PERCENTILES = (20.0, 40.0, 60.0, 80.0)


@dataclass
class RiskMatrixModel:
    """
    Fitted risk-matrix model. Holds the per-feature percentile thresholds
    used for binning. Use `RiskMatrixModel.fit(df)` to learn the thresholds
    from a training population, then call `model.score_dataframe(df)` on
    any subsequent dataframe to get Phi_composite values that are
    directly comparable across runs.
    """

    thresholds: dict[str, np.ndarray]

    # -----------------------------------------------------------------
    # Fitting and scoring
    # -----------------------------------------------------------------
    @classmethod
    def fit(cls, df: pd.DataFrame) -> RiskMatrixModel:
        """
        Learn percentile-based bin edges from a population of components.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least the columns in FEATURE_ORDER.
        """
        thresholds: dict[str, np.ndarray] = {}
        for feat in FEATURE_ORDER:
            thresholds[feat] = np.percentile(df[feat].to_numpy(), PERCENTILES)
        return cls(thresholds=thresholds)

    def score_feature(self, feature_name: str, values: np.ndarray | pd.Series) -> np.ndarray:
        """
        Score a single feature into the 1-5 bin scale using the fitted
        percentile thresholds.

        Returns an int array with values in {1, 2, 3, 4, 5}.
        """
        if feature_name not in self.thresholds:
            raise KeyError(f"Feature {feature_name!r} was not fitted.")
        edges = self.thresholds[feature_name]
        arr = np.asarray(values, dtype=float)
        # np.digitize returns 0..len(edges) inclusive; we want 1..5.
        bins = np.digitize(arr, edges, right=False) + 1
        return np.clip(bins, 1, 5)

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the per-feature 1-5 score, the weighted Phi_composite,
        and a categorical risk tier for each row in `df`.

        Returns a new DataFrame with the original columns plus:
            score_<feature> : int 1-5 per feature
            phi_composite   : weighted sum (continuous, roughly 5-25)
            risk_tier       : one of {Low, Moderate, High, Critical}
        """
        out = df.copy()
        score_columns: list[str] = []
        for feat in FEATURE_ORDER:
            col = f"score_{feat}"
            out[col] = self.score_feature(feat, df[feat])
            score_columns.append(col)

        score_matrix = out[score_columns].to_numpy(dtype=float)
        # Phi_composite as specified in the brief: weighted sum of the
        # per-feature 1-5 bin scores. The output lives on roughly [5, 25]
        # (sum of weights = 1; values weighted by 1..5).
        out["phi_composite"] = score_matrix @ WEIGHTS * 5.0

        out["risk_tier"] = pd.cut(
            out["phi_composite"],
            bins=[-np.inf, 10.0, 15.0, 20.0, np.inf],
            labels=["Low", "Moderate", "High", "Critical"],
        )
        return out


# ---------------------------------------------------------------------------
# Convenience top-level functions
# ---------------------------------------------------------------------------
def fit_and_score(df: pd.DataFrame) -> tuple[RiskMatrixModel, pd.DataFrame]:
    """
    Fit a RiskMatrixModel on `df` and immediately score the same frame.
    Most callers from main.py go through this entry point.
    """
    model = RiskMatrixModel.fit(df)
    scored = model.score_dataframe(df)
    return model, scored


def get_feature_weights() -> dict[str, float]:
    """Return the fixed feature weight vector as a name->weight mapping."""
    return dict(zip(FEATURE_ORDER, WEIGHTS.tolist(), strict=True))


__all__ = [
    "FEATURE_ORDER",
    "WEIGHTS",
    "PERCENTILES",
    "RiskMatrixModel",
    "fit_and_score",
    "get_feature_weights",
]


if __name__ == "__main__":
    import data_generator

    df = data_generator.generate_dataset(n_components=500)
    model, scored = fit_and_score(df)
    print(scored[["component_id", "phi_composite", "risk_tier"]].head())
    print("\nRisk tier distribution:")
    print(scored["risk_tier"].value_counts())
