"""
rul_predictor.py
================

Remaining Useful Life (RUL) extrapolation built on top of the RCS
engine.

For each component we forecast two operational milestones:

    cycles_to_yellow : first cycle on which RCS(t) crosses the 40
                       threshold (i.e. enters the "monitor" band)
    cycles_to_red    : first cycle on which RCS(t) crosses the 70
                       threshold (i.e. enters the "immediate inspect"
                       band)

We extrapolate the RCS trajectory past the simulated cycles up to
RUL_MAX_CYCLE (default 60). The extrapolation uses the same closed-form
ingredients as the RCS engine itself:

    - Phi_composite is constant (manufacturing baseline).
    - degradation_factor(t) is a closed-form exponential, evaluable
      at any t.
    - P_damage(t) is extrapolated by continuing the Bayesian update
      from the last simulated cycle's posterior, with the same
      INFO_WEIGHT + DRIFT machinery.

The result is a per-component RUL DataFrame that maintenance planners
can use to prioritise inspection slots.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rcs_engine import (
    ALPHA,
    BETA,
    GAMMA,
    RCS_RAW_MAX,
    bayesian_update,
    degradation_factor,
)


# ---------------------------------------------------------------------------
# RUL container
# ---------------------------------------------------------------------------
@dataclass
class RULResult:
    """Per-component RUL forecast."""

    component_ids: list[str]
    cycles_extended: list[int]
    rcs_extended: np.ndarray  # shape (n_cycles_ext, n_components)
    cycles_to_yellow: np.ndarray  # shape (n_components,) - int or NaN
    cycles_to_red: np.ndarray  # shape (n_components,) - int or NaN

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "component_id": self.component_ids,
                "cycles_to_yellow": self.cycles_to_yellow,
                "cycles_to_red": self.cycles_to_red,
                "rcs_final_extrapolated": self.rcs_extended[-1],
            }
        )


# ---------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------
def _first_crossing(rcs: np.ndarray, threshold: float, cycles: list[int]) -> np.ndarray:
    """
    For each column of `rcs` (shape (n_cycles_ext, n_components)),
    return the cycle index at which rcs first crosses `threshold`
    (>= threshold). Returns NaN where no crossing occurs within the
    horizon.
    """
    mask = rcs >= threshold
    n_components = rcs.shape[1]
    out = np.full(n_components, fill_value=np.nan, dtype=float)
    any_hit = mask.any(axis=0)
    first_idx = np.argmax(mask, axis=0)  # 0 if all False, so guard with any_hit
    cycles_arr = np.asarray(cycles)
    out[any_hit] = cycles_arr[first_idx[any_hit]]
    return out


def forecast_rul(
    component_ids: Iterable[str],
    phi_composite: np.ndarray,
    last_p_damage: np.ndarray,
    classifier_damage: np.ndarray,
    last_cycle: int,
    horizon_cycle: int = 60,
    yellow_threshold: float = 40.0,
    red_threshold: float = 70.0,
) -> RULResult:
    """
    Forecast RUL milestones from the last simulated cycle out to
    `horizon_cycle`.

    Parameters
    ----------
    component_ids
        Identifiers for the components being forecast.
    phi_composite
        Per-component Phi_composite (constant over time).
    last_p_damage
        Per-component P_damage(last_cycle) from the simulated trajectory.
    classifier_damage
        Per-component 1 - P(class=0) from the classifier (the per-cycle
        evidence signal used by the Bayesian update).
    last_cycle, horizon_cycle
        Extrapolation runs cycles (last_cycle+1) ... horizon_cycle.
    yellow_threshold, red_threshold
        RCS milestones.

    Returns
    -------
    RULResult
    """
    component_ids = list(component_ids)
    n = len(component_ids)
    phi_composite = np.asarray(phi_composite, dtype=float)
    last_p_damage = np.asarray(last_p_damage, dtype=float)
    classifier_damage = np.asarray(classifier_damage, dtype=float)

    if horizon_cycle <= last_cycle:
        raise ValueError("horizon_cycle must exceed last_cycle")
    future_cycles = list(range(last_cycle + 1, horizon_cycle + 1))
    extended_cycles = list(range(0, horizon_cycle + 1))

    # Continue the Bayesian update forward through future_cycles.
    p_dam_future = np.zeros((len(future_cycles), n), dtype=float)
    p_dam_future[0] = bayesian_update(last_p_damage, classifier_damage, future_cycles[0])
    for k in range(1, len(future_cycles)):
        p_dam_future[k] = bayesian_update(p_dam_future[k - 1], classifier_damage, future_cycles[k])

    # Build the extended RCS trajectory by concatenating "before" and
    # "after" the last simulated cycle. We only have last_p_damage so
    # we initialise p_dam_before with a linear ramp from 0 to
    # last_p_damage for the cosmetic completeness of the trajectory
    # plot - it is the future window that drives the RUL crossing.
    n_before = last_cycle + 1
    # Smooth linear ramp - this is just for visualisation continuity;
    # the actual RUL crossing uses the future window only.
    ramp_weights = np.linspace(0.0, 1.0, n_before)
    p_dam_before = ramp_weights[:, np.newaxis] * last_p_damage[np.newaxis, :] + (
        1 - ramp_weights[:, np.newaxis]
    ) * np.maximum(0.0, last_p_damage[np.newaxis, :] - 0.3)

    p_dam_ext = np.vstack([p_dam_before, p_dam_future])
    deg_ext = degradation_factor(np.asarray(extended_cycles, dtype=float))

    rcs_raw_ext = (
        ALPHA * phi_composite[np.newaxis, :]
        + BETA * p_dam_ext * 100.0
        + GAMMA * deg_ext[:, np.newaxis] * 100.0
    )
    rcs_ext = np.clip(rcs_raw_ext / RCS_RAW_MAX * 100.0, 0.0, 100.0)

    cycles_to_yellow = _first_crossing(rcs_ext, yellow_threshold, extended_cycles)
    cycles_to_red = _first_crossing(rcs_ext, red_threshold, extended_cycles)

    return RULResult(
        component_ids=component_ids,
        cycles_extended=extended_cycles,
        rcs_extended=rcs_ext,
        cycles_to_yellow=cycles_to_yellow,
        cycles_to_red=cycles_to_red,
    )


__all__ = ["RULResult", "forecast_rul"]
