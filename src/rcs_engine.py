"""
rcs_engine.py
=============

Pillar 3 of MSRCF: the Risk Continuity Score (RCS).

RCS is the central novel contribution of this framework. It folds together:

    1. Phi_composite                  - manufacturing risk baseline
                                         (Pillar 1)
    2. P_damage(t)                    - classifier-predicted probability
                                         of being in an actively damaging
                                         state at cycle t, updated each
                                         cycle by a Bayesian filter
                                         (Pillar 2)
    3. degradation_factor(t)          - monotone in-service wear-out term
                                         (1 - exp(-lambda * t)), lambda=0.15

Combined:

    RCS_raw(t) = alpha * Phi_composite + beta * P_damage(t)
               + gamma * degradation_factor(t) * 100

The 'degradation_factor * 100' lift puts the three terms on roughly the
same scale (Phi_composite ~ 5-25, P_damage ~ 0-100 once scaled,
degradation_factor ~ 0-1 -> 0-100 once scaled).

    alpha = 0.4   (manufacturing baseline)
    beta  = 0.4   (current classifier risk)
    gamma = 0.2   (service degradation)

We then min-max normalise RCS_raw to [0, 100] using the maximum
theoretically attainable score so the colour-flag thresholds remain
stable across runs:

    RCS > 70  -> RED    (immediate inspection)
    40-70     -> YELLOW (monitor)
    < 40      -> GREEN  (nominal)

Bayesian update of P_damage(t)
------------------------------
At t = 0 the prior P_damage(0) is the *classifier-supplied* probability
that the component is in any non-nominal class (i.e. P(class != 0)).
At every subsequent inspection cycle we model a fresh inspection event
and apply the classical update

    P(D | E_t) = P(E_t | D) * P(D) / P(E_t)

The per-cycle evidence E_t is deliberately *softened* toward 0.5 by an
information-content factor INFO_WEIGHT, with an additive drift driven
by the service-degradation factor:

    E_t = 0.5 + INFO_WEIGHT * (classifier_posterior - 0.5)
        + DRIFT * degradation_factor(t)

This formulation has three desirable properties:

    1. Each cycle's inspection carries finite (not infinite) information,
       so iterated updates do not collapse the posterior to {0, 1} - a
       well-known numerical pathology of iterated Bayesian filters when
       fed constant evidence.
    2. The mild upward drift means components do exhibit slowly rising
       P_damage trajectories over service life (matching field
       experience), but the rise is gradual rather than step-like.
    3. Components plateau at intermediate posteriors, populating the
       YELLOW band of the RCS flag scale, which is the operationally
       important band for "monitor closely" inspections.

We also floor the prior at 0.01 and ceiling at 0.99 to prevent the
posterior locking exactly at 0 or 1.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# RCS coefficients - locked per project brief
# ---------------------------------------------------------------------------
ALPHA = 0.4  # manufacturing baseline weight
BETA = 0.4  # current damage probability weight
GAMMA = 0.2  # service degradation weight
LAMBDA = 0.15  # exponential degradation rate

# Cycle range used for trajectory plots and the cumulative RCS.
DEFAULT_CYCLES: tuple[int, ...] = tuple(range(0, 11))

# Information-content factor on the per-cycle classifier evidence.
# Conceptually: how informative is a single fresh NDI inspection relative
# to a fully-informative test (1.0)? Tuned to 0.15 so iterated updates
# over 10 cycles do not saturate the posterior at {0, 1}, while still
# letting the classifier signal influence the trajectory.
INFO_WEIGHT = 0.15
# Per-cycle upward evidence drift driven by service degradation - the
# longer the component has been in service, the slightly more reason
# the inspection has to look damaged.
DRIFT = 0.10
# Blend weight for the initial damage prior P_damage(0):
#   P_damage(0) = PRIOR_BLEND * (1 - P(class 0))
#               + (1 - PRIOR_BLEND) * normalised_phi
# Pure-classifier priors saturate at 0 or 1 because a strong classifier
# is bimodal, which collapses the YELLOW band. Blending with the smooth
# Phi-derived signal restores a meaningful middle distribution that
# matches operator intuition: even a component the classifier calls
# "nominal" but with high Phi_composite warrants monitoring.
PRIOR_BLEND = 0.55
# Range used to min-max scale Phi_composite into a [0, 1] prior. The
# theoretical Phi_composite range is [5, 25] (sum of weights = 1, scores
# 1..5, then multiplied by 5).
PHI_MIN = 5.0
PHI_MAX = 25.0

# Theoretical maximum used to normalise RCS_raw onto [0, 100]:
#   max(Phi_composite)         = 25     (all bins = 5)
#   max(P_damage * 100)        = 100
#   max(degradation_factor*100)= 100
# so RCS_raw_max = 0.4*25 + 0.4*100 + 0.2*100 = 10 + 40 + 20 = 70.
# We divide by 70 and multiply by 100 to project to [0, 100].
RCS_RAW_MAX = ALPHA * 25.0 + BETA * 100.0 + GAMMA * 100.0


# ---------------------------------------------------------------------------
# Flag thresholds
# ---------------------------------------------------------------------------
FLAG_THRESHOLDS = {"GREEN": (-np.inf, 40.0), "YELLOW": (40.0, 70.0), "RED": (70.0, np.inf)}


def flag_for_score(rcs: float) -> str:
    """Return RED / YELLOW / GREEN given a normalised RCS in [0, 100]."""
    if rcs >= 70.0:
        return "RED"
    if rcs >= 40.0:
        return "YELLOW"
    return "GREEN"


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------
def degradation_factor(t: int | np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    """Exponential service-degradation factor: 1 - exp(-lambda * t)."""
    return 1.0 - np.exp(-lam * np.asarray(t, dtype=float))


def bayesian_update(prior: np.ndarray, classifier_prob: np.ndarray, cycle: int) -> np.ndarray:
    """
    Single per-cycle Bayesian step.

        E_t       = 0.5 + INFO_WEIGHT * (classifier_prob - 0.5)
                  + DRIFT * degradation_factor(t)
        P(D|E_t)  = E_t * P(D) / [E_t*P(D) + (1-E_t)*(1-P(D))]

    Conceptually, each cycle is a fresh inspection whose information
    content is INFO_WEIGHT * (classifier_prob - 0.5) (signed - positive
    when the classifier suggests damage), plus a slowly growing
    service-degradation drift.
    """
    prior = np.clip(prior, 0.01, 0.99)
    deg = float(degradation_factor(np.asarray(cycle)))
    e_t = 0.5 + INFO_WEIGHT * (classifier_prob - 0.5) + DRIFT * deg
    e_t = np.clip(e_t, 0.05, 0.95)

    p_e = e_t * prior + (1.0 - e_t) * (1.0 - prior)
    posterior = (e_t * prior) / np.maximum(p_e, 1e-9)
    return np.clip(posterior, 0.01, 0.99)


# ---------------------------------------------------------------------------
# RCS trajectory
# ---------------------------------------------------------------------------
@dataclass
class RCSTrajectory:
    """All time-evolving RCS quantities for one or more components."""

    component_ids: list[str]
    cycles: list[int]
    phi_composite: np.ndarray  # shape (n_components,)
    p_damage: np.ndarray  # shape (n_cycles, n_components)
    degradation: np.ndarray  # shape (n_cycles,)
    rcs_raw: np.ndarray  # shape (n_cycles, n_components)
    rcs_normalised: np.ndarray  # shape (n_cycles, n_components)
    flag: np.ndarray  # shape (n_cycles, n_components) dtype=object
    # Per-damage-mode RCS contributions, shape
    # (n_cycles, n_components, n_damage_modes). Index k is in
    # `damage_class_labels`; ordering excludes the nominal class.
    rcs_per_class: np.ndarray | None = None
    damage_class_labels: list[int] = field(default_factory=list)
    # Monte Carlo uncertainty band (if computed):
    # shape (n_cycles, n_components) each.
    rcs_lower: np.ndarray | None = None
    rcs_upper: np.ndarray | None = None

    def latest_flag(self) -> pd.Series:
        """Return the final-cycle flag for each component."""
        return pd.Series(self.flag[-1], index=self.component_ids, name="flag")

    def to_long_dataframe(self) -> pd.DataFrame:
        """Reshape into a long ('tidy') DataFrame suitable for seaborn."""
        rows: list[dict] = []
        for i, cid in enumerate(self.component_ids):
            for k, t in enumerate(self.cycles):
                row = {
                    "component_id": cid,
                    "cycle": t,
                    "phi_composite": float(self.phi_composite[i]),
                    "p_damage": float(self.p_damage[k, i]),
                    "degradation": float(self.degradation[k]),
                    "rcs_raw": float(self.rcs_raw[k, i]),
                    "rcs": float(self.rcs_normalised[k, i]),
                    "flag": str(self.flag[k, i]),
                }
                if self.rcs_lower is not None and self.rcs_upper is not None:
                    row["rcs_lower"] = float(self.rcs_lower[k, i])
                    row["rcs_upper"] = float(self.rcs_upper[k, i])
                if self.rcs_per_class is not None:
                    for j, cls in enumerate(self.damage_class_labels):
                        row[f"rcs_class_{cls}"] = float(self.rcs_per_class[k, i, j])
                rows.append(row)
        return pd.DataFrame(rows)


def compute_rcs_trajectory(
    component_ids: Iterable[str],
    phi_composite: np.ndarray,
    initial_class_probabilities: np.ndarray,
    class_labels: list[int],
    cycles: Iterable[int] = DEFAULT_CYCLES,
    nominal_class: int = 0,
) -> RCSTrajectory:
    """
    Run the full RCS pipeline over a range of inspection cycles.

    Parameters
    ----------
    component_ids : iterable of str
        Component identifiers to label rows in the resulting trajectory.
    phi_composite : array-like, shape (n_components,)
        The Pillar 1 manufacturing complexity score for each component.
    initial_class_probabilities : array-like, shape (n_components, n_classes)
        The classifier's posterior probabilities at t=0, in the same
        column order as `class_labels`.
    class_labels : list of int
        Mapping from column index in `initial_class_probabilities` to
        the integer class id. Must include `nominal_class`.
    cycles : iterable of int
        Inspection cycles to simulate.
    nominal_class : int
        The class label for "no significant damage" (i.e. the
        complement of "damage").

    Returns
    -------
    RCSTrajectory
        Container of all derived per-cycle quantities.
    """
    component_ids = list(component_ids)
    cycles = list(cycles)
    n_components = len(component_ids)
    n_cycles = len(cycles)

    phi_composite = np.asarray(phi_composite, dtype=float)
    init_probs = np.asarray(initial_class_probabilities, dtype=float)
    if init_probs.shape[0] != n_components:
        raise ValueError("init_probs rows must match number of components")

    if nominal_class not in class_labels:
        raise ValueError(f"nominal_class {nominal_class} not found in class_labels {class_labels}")
    nominal_idx = class_labels.index(nominal_class)

    # Raw classifier signal: probability the component is in ANY damaged
    # class.
    classifier_damage = 1.0 - init_probs[:, nominal_idx]
    # Phi-derived prior: normalised Phi_composite into [0, 1].
    phi_norm = np.clip((phi_composite - PHI_MIN) / (PHI_MAX - PHI_MIN), 0.0, 1.0)
    # Blend the two signals to form the initial P_damage. The pure
    # classifier signal is highly bimodal on an accurate classifier,
    # which collapses the YELLOW operational band. Blending with the
    # smooth Phi prior recovers a meaningful middle.
    p0 = PRIOR_BLEND * classifier_damage + (1.0 - PRIOR_BLEND) * phi_norm

    p_dam = np.zeros((n_cycles, n_components), dtype=float)
    p_dam[0] = np.clip(p0, 0.01, 0.99)
    # We apply the Bayesian update once per cycle using the same
    # softened classifier evidence (the classifier itself does not
    # change over the simulated time horizon).
    for k in range(1, n_cycles):
        p_dam[k] = bayesian_update(p_dam[k - 1], classifier_damage, cycles[k])

    # Degradation factor (same for every component at a given cycle).
    deg = degradation_factor(np.asarray(cycles, dtype=float), LAMBDA)

    # Assemble raw RCS:
    #   alpha * Phi_composite + beta * P_damage * 100 + gamma * deg * 100
    rcs_raw = (
        ALPHA * phi_composite[np.newaxis, :]
        + BETA * p_dam * 100.0
        + GAMMA * deg[:, np.newaxis] * 100.0
    )

    rcs_norm = np.clip(rcs_raw / RCS_RAW_MAX * 100.0, 0.0, 100.0)

    flag = np.empty_like(rcs_norm, dtype=object)
    flag[rcs_norm >= 70.0] = "RED"
    flag[(rcs_norm >= 40.0) & (rcs_norm < 70.0)] = "YELLOW"
    flag[rcs_norm < 40.0] = "GREEN"

    # ---------------------------------------------------------------
    # Per-class RCS decomposition
    #
    # We compute an RCS contribution per damage class k (excluding the
    # nominal class) by taking the per-class classifier probability
    # P(class=k) as its own running posterior and combining with the
    # same Phi / degradation contributions.
    # ---------------------------------------------------------------
    damage_class_indices = [j for j, c in enumerate(class_labels) if c != nominal_class]
    damage_class_labels = [class_labels[j] for j in damage_class_indices]

    rcs_per_class = np.zeros((n_cycles, n_components, len(damage_class_indices)), dtype=float)
    for j_idx, j in enumerate(damage_class_indices):
        per_cls_signal = init_probs[:, j]
        per_cls_phi_prior = PRIOR_BLEND * per_cls_signal + (1 - PRIOR_BLEND) * phi_norm
        p_k = np.zeros((n_cycles, n_components), dtype=float)
        p_k[0] = np.clip(per_cls_phi_prior, 0.01, 0.99)
        for k in range(1, n_cycles):
            p_k[k] = bayesian_update(p_k[k - 1], per_cls_signal, cycles[k])
        raw = (
            ALPHA * phi_composite[np.newaxis, :]
            + BETA * p_k * 100.0
            + GAMMA * deg[:, np.newaxis] * 100.0
        )
        rcs_per_class[:, :, j_idx] = np.clip(raw / RCS_RAW_MAX * 100.0, 0.0, 100.0)

    return RCSTrajectory(
        component_ids=component_ids,
        cycles=cycles,
        phi_composite=phi_composite,
        p_damage=p_dam,
        degradation=deg,
        rcs_raw=rcs_raw,
        rcs_normalised=rcs_norm,
        flag=flag,
        rcs_per_class=rcs_per_class,
        damage_class_labels=damage_class_labels,
    )


# ---------------------------------------------------------------------------
# Monte Carlo uncertainty bands
# ---------------------------------------------------------------------------
def compute_rcs_mc_band(
    component_ids: Iterable[str],
    phi_composite: np.ndarray,
    initial_class_probabilities: np.ndarray,
    class_labels: list[int],
    n_samples: int = 200,
    noise_sigma: float = 0.05,
    ci_low: float = 5.0,
    ci_high: float = 95.0,
    cycles: Iterable[int] = DEFAULT_CYCLES,
    nominal_class: int = 0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate Monte Carlo uncertainty bands on the RCS trajectory by
    perturbing the per-component initial classifier probabilities with
    additive Gaussian noise on the logit, re-normalising, and
    re-running the full per-cycle update.

    Returns
    -------
    median, lower, upper : np.ndarray, each shape (n_cycles, n_components)
        Median and percentile bands (default 5/95) of the simulated
        RCS distribution.
    """
    rng = np.random.default_rng(seed)
    component_ids = list(component_ids)
    cycles = list(cycles)
    init_probs = np.asarray(initial_class_probabilities, dtype=float)
    n_components = init_probs.shape[0]
    n_cycles = len(cycles)

    samples = np.empty((n_samples, n_cycles, n_components), dtype=float)
    # We perturb in logit space to keep probabilities on (0, 1) without
    # post-hoc clipping artefacts. We deliberately do NOT renormalise
    # the perturbed rows to sum to 1: doing so collapses most of the
    # injected noise back to the original probability vector. The RCS
    # engine only consumes the per-class probabilities independently
    # (one for the "damage" complement, optionally one per class for
    # the per-class decomposition) so leaving them un-normalised is
    # safe and preserves the intended MC variance.
    eps = 1e-6
    base_logit = np.log(np.clip(init_probs, eps, 1 - eps) / np.clip(1 - init_probs, eps, 1 - eps))

    for s in range(n_samples):
        noise = rng.normal(0.0, noise_sigma, size=base_logit.shape)
        perturbed_logit = base_logit + noise
        perturbed = 1.0 / (1.0 + np.exp(-perturbed_logit))
        traj = compute_rcs_trajectory(
            component_ids=component_ids,
            phi_composite=phi_composite,
            initial_class_probabilities=perturbed,
            class_labels=class_labels,
            cycles=cycles,
            nominal_class=nominal_class,
        )
        samples[s] = traj.rcs_normalised

    median = np.percentile(samples, 50.0, axis=0)
    lower = np.percentile(samples, ci_low, axis=0)
    upper = np.percentile(samples, ci_high, axis=0)
    return median, lower, upper


__all__ = [
    "ALPHA",
    "BETA",
    "GAMMA",
    "LAMBDA",
    "RCS_RAW_MAX",
    "DEFAULT_CYCLES",
    "FLAG_THRESHOLDS",
    "RCSTrajectory",
    "flag_for_score",
    "degradation_factor",
    "bayesian_update",
    "compute_rcs_trajectory",
    "compute_rcs_mc_band",
]
