"""
dashboard.py
============

Visualization layer for MSRCF. All figures are produced with matplotlib
+ seaborn so the project remains a self-contained Python package - no
Streamlit, no Dash, no web servers.

The four artefacts this module produces (all PNG, saved into /results):

    1. Per-model confusion matrices            (one figure with subplots)
    2. Model-comparison bar chart              (accuracy / precision /
                                                recall / F1 grouped)
    3. RCS trajectory plot for 5 components    (multi-panel)
    4. Fleet risk dashboard                    (4-panel summary)

Every function writes to disk and returns the saved file path so main.py
can summarise outputs to stdout.
"""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # ensure headless / non-interactive rendering

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from damage_predictor import DAMAGE_CLASS_NAMES, ModelResult
from rcs_engine import RCSTrajectory


sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110


# ---------------------------------------------------------------------------
# 1. Confusion matrices
# ---------------------------------------------------------------------------
def plot_confusion_matrices(
    results: dict[str, ModelResult], output_path: str
) -> str:
    """
    Produce a single figure of confusion matrices, one subplot per model.

    The 5 classifiers are laid out in a 2x3 grid (the last cell is left
    blank). Counts are annotated inside each cell.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    names = list(results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    class_labels = next(iter(results.values())).class_labels
    tick_labels = [f"{c}\n{DAMAGE_CLASS_NAMES.get(c, str(c))}" for c in class_labels]

    for ax, name in zip(axes, names):
        r = results[name]
        sns.heatmap(
            r.confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=tick_labels,
            yticklabels=tick_labels,
        )
        ax.set_title(f"{name}  (F1={r.f1_macro:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", labelrotation=0, labelsize=8)
        ax.tick_params(axis="y", labelrotation=0, labelsize=8)

    # Hide unused subplot slot(s).
    for ax in axes[len(names):]:
        ax.axis("off")

    fig.suptitle("Damage-mode classifier confusion matrices", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 2. Model comparison bar chart
# ---------------------------------------------------------------------------
def plot_model_comparison(
    results: dict[str, ModelResult], output_path: str
) -> str:
    """
    Grouped bar chart comparing accuracy / macro-precision / macro-recall
    / macro-F1 across the five models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for r in results.values():
        rows.extend(
            [
                {"model": r.name, "metric": "Accuracy", "value": r.accuracy},
                {"model": r.name, "metric": "Precision", "value": r.precision_macro},
                {"model": r.name, "metric": "Recall", "value": r.recall_macro},
                {"model": r.name, "metric": "F1", "value": r.f1_macro},
            ]
        )
    long_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=long_df, x="model", y="value", hue="metric", ax=ax, palette="viridis"
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Classifier performance comparison (test set, macro-averaged)")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(title="", loc="upper right", ncols=4)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 3. RCS trajectory plot
# ---------------------------------------------------------------------------
def plot_rcs_trajectories(
    trajectory: RCSTrajectory,
    output_path: str,
    sample_ids: Iterable[str] | None = None,
) -> str:
    """
    Plot the RCS evolution over inspection cycles for a small sample
    of components. If sample_ids is omitted, picks 5 representative
    components spanning the Phi_composite range.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if sample_ids is None:
        sample_ids = _select_diverse_sample(trajectory, k=5)
    sample_ids = list(sample_ids)

    df = trajectory.to_long_dataframe()
    df = df[df["component_id"].isin(sample_ids)]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    palette = sns.color_palette("husl", n_colors=len(sample_ids))

    # Top: RCS trajectory with flag bands.
    ax_top.axhspan(70, 100, color="#ff8a80", alpha=0.18, label="RED zone")
    ax_top.axhspan(40, 70, color="#ffe082", alpha=0.25, label="YELLOW zone")
    ax_top.axhspan(0, 40, color="#a5d6a7", alpha=0.20, label="GREEN zone")
    for color, cid in zip(palette, sample_ids):
        sub = df[df["component_id"] == cid].sort_values("cycle")
        ax_top.plot(
            sub["cycle"],
            sub["rcs"],
            "-o",
            color=color,
            linewidth=2,
            markersize=5,
            label=cid,
        )
    ax_top.set_ylabel("RCS (0-100)")
    ax_top.set_title("Risk Continuity Score over inspection cycles")
    ax_top.set_ylim(0, 100)
    ax_top.legend(loc="upper left", ncols=2, fontsize=8)

    # Bottom: P_damage and degradation factor for the same set.
    for color, cid in zip(palette, sample_ids):
        sub = df[df["component_id"] == cid].sort_values("cycle")
        ax_bot.plot(
            sub["cycle"],
            sub["p_damage"],
            "-s",
            color=color,
            label=f"{cid} P(damage)",
            alpha=0.8,
        )
    # Single shared degradation curve overlay.
    cycles = sorted(df["cycle"].unique())
    deg = df.groupby("cycle")["degradation"].first().reindex(cycles).values
    ax_bot.plot(
        cycles,
        deg,
        "--",
        color="black",
        linewidth=2,
        label="Degradation factor (shared)",
    )
    ax_bot.set_xlabel("Inspection cycle")
    ax_bot.set_ylabel("Probability / factor")
    ax_bot.set_ylim(0, 1.05)
    ax_bot.legend(loc="upper left", fontsize=7, ncols=2)
    ax_bot.set_title("Bayesian P(damage) and service degradation per cycle")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _select_diverse_sample(trajectory: RCSTrajectory, k: int = 5) -> list[str]:
    """
    Pick `k` components with broad coverage of the RCS flag space:
    one GREEN, two YELLOW (one at each edge if possible), one RED, plus
    one extra picked to maximise visual spread of trajectories. Falls
    back to evenly-spaced quantiles if a flag zone is empty.
    """
    final = trajectory.rcs_normalised[-1]
    final_flag = trajectory.flag[-1]
    n = len(final)
    if n <= k:
        return list(trajectory.component_ids)

    chosen: list[int] = []

    def _pick_closest_to(target: float, exclude: set[int]) -> int | None:
        order = np.argsort(np.abs(final - target))
        for idx in order:
            if int(idx) not in exclude:
                return int(idx)
        return None

    # Target one component near the middle of each flag band, where
    # available.
    band_targets = [
        ("GREEN", 25.0),
        ("YELLOW", 50.0),
        ("RED", 80.0),
    ]
    for flag_name, tgt in band_targets:
        mask = (final_flag == flag_name)
        if mask.sum() == 0:
            continue
        candidate_indices = np.where(mask)[0]
        order = np.argsort(np.abs(final[candidate_indices] - tgt))
        for j in order:
            idx = int(candidate_indices[j])
            if idx not in chosen:
                chosen.append(idx)
                break

    # Top up with two more components at the extremes for visual range.
    extras = [
        _pick_closest_to(float(np.quantile(final, 0.05)), set(chosen)),
        _pick_closest_to(float(np.quantile(final, 0.95)), set(chosen)),
    ]
    for idx in extras:
        if idx is not None and idx not in chosen:
            chosen.append(idx)

    # If we still have fewer than k (because a flag band was empty),
    # fill from evenly-spaced quantiles.
    while len(chosen) < k:
        q = (len(chosen) + 1) / (k + 1)
        idx = _pick_closest_to(float(np.quantile(final, q)), set(chosen))
        if idx is None:
            break
        chosen.append(idx)

    return [trajectory.component_ids[i] for i in chosen[:k]]


# ---------------------------------------------------------------------------
# 4. Fleet risk dashboard
# ---------------------------------------------------------------------------
def plot_risk_dashboard(
    scored_df: pd.DataFrame,
    trajectory: RCSTrajectory,
    output_path: str,
) -> str:
    """
    Four-panel fleet-level dashboard:

        (top-left)    Phi_composite distribution by risk tier
        (top-right)   Damage mode (ground truth) counts
        (bottom-left) Final-cycle RCS histogram with flag thresholds
        (bottom-right) Flag breakdown by manufacturing risk tier
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    final_rcs = trajectory.rcs_normalised[-1]
    final_flag = pd.Series(trajectory.flag[-1], index=trajectory.component_ids)
    # Align to scored_df component order.
    final_rcs_series = pd.Series(final_rcs, index=trajectory.component_ids)
    panel_df = scored_df.set_index("component_id").copy()
    panel_df["final_rcs"] = final_rcs_series
    panel_df["final_flag"] = final_flag

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_phi, ax_dmg, ax_rcs, ax_flag = axes.flatten()

    # (1) Phi_composite distribution by risk tier
    tier_order = ["Low", "Moderate", "High", "Critical"]
    sns.histplot(
        data=panel_df,
        x="phi_composite",
        hue="risk_tier",
        hue_order=tier_order,
        multiple="stack",
        bins=20,
        palette="rocket_r",
        ax=ax_phi,
    )
    ax_phi.set_title("Manufacturing Phi_composite distribution")
    ax_phi.set_xlabel("Phi_composite")

    # (2) Damage mode distribution (ground truth)
    dmg_counts = (
        panel_df["damage_mode"].value_counts().sort_index().rename("count").reset_index()
    )
    dmg_counts["label"] = dmg_counts["damage_mode"].map(DAMAGE_CLASS_NAMES)
    sns.barplot(
        data=dmg_counts,
        x="label",
        y="count",
        hue="label",
        palette="mako",
        ax=ax_dmg,
        legend=False,
    )
    ax_dmg.set_title("Ground-truth damage mode counts (full dataset)")
    ax_dmg.set_xlabel("")
    ax_dmg.set_ylabel("Components")
    ax_dmg.tick_params(axis="x", labelrotation=20)

    # (3) Final-cycle RCS histogram with flag thresholds
    ax_rcs.axvspan(70, 100, color="#ff8a80", alpha=0.18)
    ax_rcs.axvspan(40, 70, color="#ffe082", alpha=0.25)
    ax_rcs.axvspan(0, 40, color="#a5d6a7", alpha=0.20)
    sns.histplot(panel_df["final_rcs"], bins=25, color="#1f77b4", ax=ax_rcs)
    ax_rcs.axvline(40, color="black", linestyle="--", linewidth=1)
    ax_rcs.axvline(70, color="black", linestyle="--", linewidth=1)
    ax_rcs.set_title(f"Final-cycle (t={trajectory.cycles[-1]}) RCS distribution")
    ax_rcs.set_xlabel("RCS")

    # (4) Flag breakdown by risk tier
    pivot = (
        panel_df.groupby(["risk_tier", "final_flag"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    pivot = pivot.reindex(tier_order)
    # Make sure all three flags are present even if a column is empty.
    for f in ["GREEN", "YELLOW", "RED"]:
        if f not in pivot.columns:
            pivot[f] = 0
    pivot = pivot[["GREEN", "YELLOW", "RED"]]
    pivot.plot(
        kind="bar",
        stacked=True,
        color=["#4caf50", "#fbc02d", "#e53935"],
        ax=ax_flag,
        edgecolor="white",
    )
    ax_flag.set_title("RCS flag breakdown by manufacturing risk tier")
    ax_flag.set_xlabel("Manufacturing risk tier")
    ax_flag.set_ylabel("Components")
    ax_flag.tick_params(axis="x", labelrotation=0)
    ax_flag.legend(title="Flag")

    fig.suptitle("MSRCF fleet risk dashboard", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 5. RCS trajectories with Monte Carlo uncertainty band
# ---------------------------------------------------------------------------
def plot_rcs_trajectories_with_uncertainty(
    trajectory: RCSTrajectory,
    output_path: str,
    sample_ids: Iterable[str] | None = None,
) -> str:
    """
    Like plot_rcs_trajectories, but with a 5/95 percentile MC band drawn
    around each component's median RCS line. Requires that
    `trajectory.rcs_lower` and `trajectory.rcs_upper` are populated.
    """
    if trajectory.rcs_lower is None or trajectory.rcs_upper is None:
        raise ValueError(
            "trajectory has no MC band; populate rcs_lower/rcs_upper first"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if sample_ids is None:
        sample_ids = _select_diverse_sample(trajectory, k=5)
    sample_ids = list(sample_ids)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axhspan(70, 100, color="#ff8a80", alpha=0.18, label="RED zone")
    ax.axhspan(40, 70, color="#ffe082", alpha=0.25, label="YELLOW zone")
    ax.axhspan(0, 40, color="#a5d6a7", alpha=0.20, label="GREEN zone")

    palette = sns.color_palette("husl", n_colors=len(sample_ids))
    id_to_idx = {cid: i for i, cid in enumerate(trajectory.component_ids)}
    cycles = trajectory.cycles
    for color, cid in zip(palette, sample_ids):
        i = id_to_idx[cid]
        med = trajectory.rcs_normalised[:, i]
        lo = trajectory.rcs_lower[:, i]
        hi = trajectory.rcs_upper[:, i]
        ax.fill_between(cycles, lo, hi, color=color, alpha=0.18)
        ax.plot(cycles, med, "-o", color=color, linewidth=2, markersize=5, label=cid)

    ax.set_xlabel("Inspection cycle")
    ax.set_ylabel("RCS (0-100)")
    ax.set_title("RCS trajectory with Monte Carlo 5-95 % uncertainty band")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 6. Per-class RCS trajectory plot
# ---------------------------------------------------------------------------
def plot_rcs_per_class(
    trajectory: RCSTrajectory,
    class_names: dict[int, str],
    output_path: str,
    sample_ids: Iterable[str] | None = None,
) -> str:
    """
    For each of up to four sample components, draw the per-damage-mode
    RCS_k(t) curves on a small-multiples grid (one panel per component).
    """
    if trajectory.rcs_per_class is None:
        raise ValueError("trajectory has no per-class decomposition")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if sample_ids is None:
        sample_ids = _select_diverse_sample(trajectory, k=4)
    sample_ids = list(sample_ids)[:4]
    id_to_idx = {cid: i for i, cid in enumerate(trajectory.component_ids)}

    n_panels = len(sample_ids)
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9), sharex=True, sharey=True
    )
    axes = axes.flatten()
    cycles = trajectory.cycles
    palette = sns.color_palette("Set1", n_colors=len(trajectory.damage_class_labels))

    for ax, cid in zip(axes, sample_ids):
        i = id_to_idx[cid]
        ax.axhspan(70, 100, color="#ff8a80", alpha=0.10)
        ax.axhspan(40, 70, color="#ffe082", alpha=0.15)
        ax.axhspan(0, 40, color="#a5d6a7", alpha=0.10)
        for color, j, cls in zip(
            palette, range(len(trajectory.damage_class_labels)),
            trajectory.damage_class_labels,
        ):
            ax.plot(
                cycles,
                trajectory.rcs_per_class[:, i, j],
                "-o",
                color=color,
                linewidth=2,
                markersize=4,
                label=class_names.get(cls, str(cls)),
            )
        # Overlay the aggregate RCS in black.
        ax.plot(
            cycles,
            trajectory.rcs_normalised[:, i],
            "--",
            color="black",
            linewidth=2,
            label="Aggregate RCS",
        )
        ax.set_title(cid)
        ax.set_xlabel("Inspection cycle")
        ax.set_ylabel("RCS (0-100)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, loc="upper left")

    for ax in axes[n_panels:]:
        ax.axis("off")
    fig.suptitle("Per-damage-mode RCS decomposition", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 7. RUL histogram
# ---------------------------------------------------------------------------
def plot_rul_histogram(rul_df: pd.DataFrame, output_path: str) -> str:
    """
    Two-panel histogram of cycles-to-yellow and cycles-to-red.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax_y, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

    def _hist(ax, series, color, title):
        finite = series.dropna()
        if len(finite) == 0:
            ax.text(0.5, 0.5, "No crossings within horizon", ha="center")
            ax.set_title(title)
            return
        sns.histplot(finite, bins=18, color=color, ax=ax, kde=True)
        ax.axvline(
            float(finite.median()),
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"median={float(finite.median()):.1f}",
        )
        ax.set_title(title)
        ax.set_xlabel("Cycle")
        ax.legend()

    _hist(
        ax_y,
        rul_df["cycles_to_yellow"],
        "#fbc02d",
        f"Cycles-to-YELLOW (n={int(rul_df['cycles_to_yellow'].notna().sum())})",
    )
    _hist(
        ax_r,
        rul_df["cycles_to_red"],
        "#e53935",
        f"Cycles-to-RED (n={int(rul_df['cycles_to_red'].notna().sum())})",
    )
    fig.suptitle("Remaining Useful Life forecast distribution", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 8. Anomaly score scatter
# ---------------------------------------------------------------------------
def plot_anomaly_scatter(
    anomaly_df: pd.DataFrame,
    output_path: str,
    final_rcs: pd.Series | None = None,
) -> str:
    """
    Anomaly score vs. Phi_composite, coloured by anomaly flag. If
    final_rcs is provided, the size of each marker scales with RCS.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    merged = anomaly_df.copy()
    if final_rcs is not None:
        merged = merged.merge(
            final_rcs.rename("final_rcs").reset_index().rename(
                columns={"index": "component_id"}
            ),
            on="component_id",
            how="left",
        )
        sizes = 20 + (merged["final_rcs"].fillna(0) / 100.0) * 150
    else:
        sizes = np.full(len(merged), 50.0)

    # Plot the two anomaly classes separately to avoid the seaborn
    # legend pathway that breaks with mixed kwargs on Python 3.14.
    for label, color in [(0, "#4caf50"), (1, "#e53935")]:
        mask = merged["is_anomaly"] == label
        ax.scatter(
            merged.loc[mask, "phi_composite"],
            merged.loc[mask, "anomaly_score"],
            s=sizes[mask] if hasattr(sizes, "loc") else sizes[mask.to_numpy()],
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            label=("anomaly" if label == 1 else "nominal"),
        )
    ax.legend(loc="upper left", title="Isolation Forest")
    ax.set_title("Anomaly score vs Phi_composite (marker size ~ final RCS)")
    ax.set_xlabel("Phi_composite")
    ax.set_ylabel("Anomaly score (higher = more anomalous)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


__all__ = [
    "plot_confusion_matrices",
    "plot_model_comparison",
    "plot_rcs_trajectories",
    "plot_risk_dashboard",
    "plot_rcs_trajectories_with_uncertainty",
    "plot_rcs_per_class",
    "plot_rul_histogram",
    "plot_anomaly_scatter",
]
