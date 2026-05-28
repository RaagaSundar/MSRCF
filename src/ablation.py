"""
ablation.py
===========

Ablation study for MSRCF. An ablation answers the question a sceptical
reviewer always asks: "did every part of your design actually earn its
place, or are some pieces decoration?"

We run two complementary ablations, each averaged over multiple seeds so
the reported deltas carry confidence intervals rather than single-run
noise.

(A) Feature ablation - Pillar 2
    Drop each of the five manufacturing features in turn, retrain the
    production model (XGBoost), and measure the change in macro-F1. A
    feature that matters will cause a large drop when removed.

(B) Design-choice ablation - Pillar 3
    Toggle the two non-obvious RCS design decisions and measure their
    effect on the operational flag distribution:
        - PRIOR_BLEND  : classifier-only prior (blend = 1.0) vs the
                         Phi-blended prior (blend = 0.55).
        - DRIFT        : degradation-driven evidence drift on vs off.
    The headline metric here is the size of the YELLOW band, because the
    whole point of the Phi-blended prior was to keep YELLOW from
    collapsing.

Everything is deterministic given the seed list.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
import damage_predictor as dp
import data_generator
import rcs_engine
import risk_matrix
from experiments import mean_ci

FEATURES = list(dp.FEATURE_COLUMNS)


@dataclass
class AblationResult:
    feature_ablation: pd.DataFrame  # per-feature delta macro-F1 + CI
    design_ablation: pd.DataFrame  # per-config flag distribution + CI
    baseline_f1_mean: float
    seeds: list[int]


# ---------------------------------------------------------------------------
# (A) Feature ablation
# ---------------------------------------------------------------------------
def _f1_for_feature_set(df: pd.DataFrame, feature_cols: list[str], seed: int) -> float:
    """Train XGBoost on the given feature subset; return test macro-F1."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        results, _ = dp.train_and_evaluate(df, feature_columns=feature_cols, seed=seed)
    return results["XGBoost"].f1_macro


def run_feature_ablation(seeds: list[int], n_components: int = 500) -> pd.DataFrame:
    """
    For each feature, compute mean(macro-F1) with all features vs mean
    macro-F1 with that feature removed, over the seed ensemble. The
    reported delta is (full - ablated): larger positive delta means the
    feature is more important.
    """
    full_scores: list[float] = []
    ablated: dict[str, list[float]] = {f: [] for f in FEATURES}

    for seed in seeds:
        df = data_generator.generate_dataset(n_components=n_components, seed=seed)
        full_scores.append(_f1_for_feature_set(df, FEATURES, seed))
        for f in FEATURES:
            subset = [c for c in FEATURES if c != f]
            ablated[f].append(_f1_for_feature_set(df, subset, seed))

    full_mean, full_ci, _ = mean_ci(np.array(full_scores))

    rows = []
    for f in FEATURES:
        deltas = np.array(full_scores) - np.array(ablated[f])
        d_mean, d_ci, _ = mean_ci(deltas)
        abl_mean, abl_ci, _ = mean_ci(np.array(ablated[f]))
        rows.append(
            {
                "removed_feature": f,
                "ablated_f1_mean": round(abl_mean, 4),
                "ablated_f1_ci95": round(abl_ci, 4),
                "delta_f1_mean": round(d_mean, 4),
                "delta_f1_ci95": round(d_ci, 4),
            }
        )
    out = pd.DataFrame(rows).sort_values("delta_f1_mean", ascending=False)
    out.attrs["full_f1_mean"] = round(full_mean, 4)
    out.attrs["full_f1_ci95"] = round(full_ci, 4)
    return out


# ---------------------------------------------------------------------------
# (B) Design-choice ablation
# ---------------------------------------------------------------------------
def _flag_distribution(
    df: pd.DataFrame,
    scored_df: pd.DataFrame,
    prior_blend: float,
    use_drift: bool,
    seed: int,
) -> dict[str, int]:
    """
    Build a one-off RCS trajectory under temporary overrides of
    PRIOR_BLEND and DRIFT, and return the final-cycle flag counts.

    We patch the module-level constants in rcs_engine around the call,
    then restore them, so the override is local and side-effect free.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        results, _ = dp.train_and_evaluate(df, seed=seed)
    best = dp.select_best_model(results)
    X_full = df[FEATURES].to_numpy(dtype=float)
    init_probs = dp.predict_probabilities(best.estimator, X_full, best.class_labels)

    saved_blend = rcs_engine.PRIOR_BLEND
    saved_drift = rcs_engine.DRIFT
    try:
        rcs_engine.PRIOR_BLEND = prior_blend
        rcs_engine.DRIFT = saved_drift if use_drift else 0.0
        traj = rcs_engine.compute_rcs_trajectory(
            component_ids=df["component_id"].tolist(),
            phi_composite=scored_df["phi_composite"].to_numpy(),
            initial_class_probabilities=init_probs,
            class_labels=best.class_labels,
        )
    finally:
        rcs_engine.PRIOR_BLEND = saved_blend
        rcs_engine.DRIFT = saved_drift

    final_flags = pd.Series(traj.flag[-1])
    counts = final_flags.value_counts().to_dict()
    return {f: int(counts.get(f, 0)) for f in ["GREEN", "YELLOW", "RED"]}


def run_design_ablation(seeds: list[int], n_components: int = 500) -> pd.DataFrame:
    """
    Compare four RCS configurations over the seed ensemble and report the
    mean GREEN/YELLOW/RED counts with 95 % CI:

        - full          : PRIOR_BLEND=0.55, drift on   (the shipped config)
        - no_phi_prior  : PRIOR_BLEND=1.00, drift on   (classifier-only prior)
        - no_drift      : PRIOR_BLEND=0.55, drift off
        - minimal       : PRIOR_BLEND=1.00, drift off
    """
    configs = {
        "full": (cfg.PRIOR_BLEND, True),
        "no_phi_prior": (1.0, True),
        "no_drift": (cfg.PRIOR_BLEND, False),
        "minimal": (1.0, False),
    }
    accum: dict[str, dict[str, list[int]]] = {
        name: {"GREEN": [], "YELLOW": [], "RED": []} for name in configs
    }

    for seed in seeds:
        df = data_generator.generate_dataset(n_components=n_components, seed=seed)
        _, scored_df = risk_matrix.fit_and_score(df)
        for name, (blend, drift) in configs.items():
            dist = _flag_distribution(df, scored_df, blend, drift, seed)
            for flag in ["GREEN", "YELLOW", "RED"]:
                accum[name][flag].append(dist[flag])

    rows = []
    for name in configs:
        rec: dict = {"config": name}
        for flag in ["GREEN", "YELLOW", "RED"]:
            m, ci, _ = mean_ci(np.array(accum[name][flag]))
            rec[f"{flag}_mean"] = round(m, 1)
            rec[f"{flag}_ci95"] = round(ci, 1)
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_ablation(result: AblationResult, output_path: str) -> str:
    """Two-panel ablation figure: feature importance + design-choice YELLOW band."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax_feat, ax_design) = plt.subplots(1, 2, figsize=(14, 6))

    fa = result.feature_ablation
    ax_feat.barh(
        fa["removed_feature"],
        fa["delta_f1_mean"],
        xerr=fa["delta_f1_ci95"],
        color="#1f77b4",
        capsize=4,
    )
    ax_feat.set_xlabel("Drop in macro-F1 when feature removed")
    ax_feat.set_title(
        f"Feature ablation (XGBoost, full F1="
        f"{fa.attrs['full_f1_mean']:.3f}±{fa.attrs['full_f1_ci95']:.3f})"
    )
    ax_feat.axvline(0, color="black", linewidth=0.8)
    ax_feat.invert_yaxis()

    da = result.design_ablation.set_index("config")
    configs = da.index.tolist()
    x = np.arange(len(configs))
    width = 0.25
    for i, (flag, color) in enumerate(
        [("GREEN", "#4caf50"), ("YELLOW", "#fbc02d"), ("RED", "#e53935")]
    ):
        ax_design.bar(
            x + (i - 1) * width,
            da[f"{flag}_mean"],
            width,
            yerr=da[f"{flag}_ci95"],
            label=flag,
            color=color,
            capsize=3,
        )
    ax_design.set_xticks(x)
    ax_design.set_xticklabels(configs, rotation=15)
    ax_design.set_ylabel("Components (mean over seeds)")
    ax_design.set_title("RCS design ablation: flag distribution")
    ax_design.legend()

    fig.suptitle(f"MSRCF ablation study ({len(result.seeds)} seeds)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_ablation(seeds: list[int], n_components: int = 500) -> AblationResult:
    feat = run_feature_ablation(seeds, n_components)
    design = run_design_ablation(seeds, n_components)
    return AblationResult(
        feature_ablation=feat,
        design_ablation=design,
        baseline_f1_mean=float(feat.attrs["full_f1_mean"]),
        seeds=list(seeds),
    )


def main(argv: list[str] | None = None) -> AblationResult:
    import argparse

    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    results_dir = os.path.join(project_root, "results")

    parser = argparse.ArgumentParser(description="MSRCF ablation study.")
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--n-components", type=int, default=500)
    parser.add_argument("--seed0", type=int, default=cfg.RANDOM_SEED)
    args = parser.parse_args(argv)

    seeds = list(range(args.seed0, args.seed0 + args.n_seeds))
    print(f"Running ablation over {len(seeds)} seeds ({seeds[0]}..{seeds[-1]}) ...")
    result = run_ablation(seeds, n_components=args.n_components)

    print("\n=== Feature ablation (drop in macro-F1 when removed) ===")
    print(result.feature_ablation.to_string(index=False))
    print("\n=== Design-choice ablation (final-cycle flag counts) ===")
    print(result.design_ablation.to_string(index=False))

    fig_path = plot_ablation(result, os.path.join(results_dir, "ablation.png"))
    result.feature_ablation.to_csv(os.path.join(results_dir, "ablation_features.csv"), index=False)
    result.design_ablation.to_csv(os.path.join(results_dir, "ablation_design.csv"), index=False)
    print(f"\nSaved: {fig_path}")
    print("Saved: results/ablation_features.csv, results/ablation_design.csv")
    return result


if __name__ == "__main__":
    import sys

    HERE = os.path.dirname(os.path.abspath(__file__))
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    main()
