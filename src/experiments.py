"""
experiments.py
==============

Statistical evaluation harness for the MSRCF damage-mode classifiers.

A single train/test run on one seed tells you almost nothing about which
model is *really* best - the ranking can flip with the random split. This
module runs the full five-classifier benchmark across many independent
seeds and reports results the way a referee would expect:

    1. Per-model metrics as mean +/- 95 % confidence interval (Student-t)
       over the seeds.
    2. A Friedman omnibus test on the per-seed macro-F1 ranks - the
       non-parametric, repeated-measures test recommended by Demsar (2006)
       for comparing classifiers across multiple datasets/runs.
    3. If Friedman rejects, a Nemenyi post-hoc test with a critical
       difference (CD) and a CD diagram that visually groups models that
       are statistically indistinguishable.

Reference:
    Demsar, J. (2006). "Statistical Comparisons of Classifiers over
    Multiple Data Sets." Journal of Machine Learning Research, 7, 1-30.

The whole analysis is deterministic given the seed list, so it is fully
reproducible.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from scipy.stats import t as student_t

import config as cfg
import damage_predictor as dp
import data_generator

# ---------------------------------------------------------------------------
# Nemenyi critical values q_alpha (two-tailed), already divided by sqrt(2),
# indexed by the number of models k. Source: Demsar (2006), Table 5.
# These are the studentized-range based critical values used to form the
# Nemenyi critical difference.
# ---------------------------------------------------------------------------
NEMENYI_Q_ALPHA_005 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}
NEMENYI_Q_ALPHA_010 = {
    2: 1.645,
    3: 2.052,
    4: 2.291,
    5: 2.459,
    6: 2.589,
    7: 2.693,
    8: 2.780,
    9: 2.855,
    10: 2.920,
}


@dataclass
class MultiSeedResult:
    """Container for the multi-seed benchmark + significance analysis."""

    seeds: list[int]
    per_seed_metrics: pd.DataFrame  # long: (seed, model, metric, value)
    summary: pd.DataFrame  # per-model mean / std / 95% CI
    f1_matrix: pd.DataFrame  # wide: index=seed, cols=model -> macro-F1
    average_ranks: pd.Series  # per-model mean rank (lower = better)
    friedman_stat: float
    friedman_p: float
    nemenyi_cd: float
    significant_pairs: list[tuple[str, str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------
def mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Return (mean, half_width, std) for a Student-t confidence interval.

    half_width = t_{(1+confidence)/2, n-1} * s / sqrt(n)

    For n == 1 the half-width is 0 (no dispersion estimable).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, 0.0, 0.0
    std = float(np.std(values, ddof=1))
    tcrit = float(student_t.ppf((1 + confidence) / 2.0, df=n - 1))
    half = tcrit * std / np.sqrt(n)
    return mean, half, std


# ---------------------------------------------------------------------------
# Nemenyi critical difference
# ---------------------------------------------------------------------------
def nemenyi_critical_difference(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """
    Compute the Nemenyi critical difference for comparing k classifiers
    over n_blocks paired observations (here: seeds):

        CD = q_alpha * sqrt( k*(k+1) / (6 * n_blocks) )

    Two classifiers differ significantly if the absolute difference of
    their average ranks exceeds CD.
    """
    table = NEMENYI_Q_ALPHA_005 if alpha == 0.05 else NEMENYI_Q_ALPHA_010
    if k not in table:
        raise ValueError(f"No tabulated Nemenyi q for k={k} (supported 2..10)")
    q = table[k]
    return q * np.sqrt(k * (k + 1) / (6.0 * n_blocks))


# ---------------------------------------------------------------------------
# Multi-seed benchmark
# ---------------------------------------------------------------------------
def run_multiseed_benchmark(
    seeds: list[int],
    n_components: int = 500,
) -> MultiSeedResult:
    """
    Train + evaluate all five classifiers once per seed and aggregate.

    For each seed we (re)generate the dataset *and* re-split *and*
    re-seed the models, so each seed is a genuinely independent
    realisation of the entire experiment - the correct unit of
    replication for a Friedman/Nemenyi analysis.
    """
    rows: list[dict] = []
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        for seed in seeds:
            df = data_generator.generate_dataset(n_components=n_components, seed=seed)
            results, _ = dp.train_and_evaluate(df, seed=seed)
            for name, r in results.items():
                rows.append(
                    {
                        "seed": seed,
                        "model": name,
                        "accuracy": r.accuracy,
                        "precision_macro": r.precision_macro,
                        "recall_macro": r.recall_macro,
                        "f1_macro": r.f1_macro,
                    }
                )

    per_seed = pd.DataFrame(rows)

    # ---- Per-model mean +/- 95% CI table ----
    summary_rows = []
    for name, grp in per_seed.groupby("model"):
        rec: dict = {"model": name}
        for metric in metric_names:
            mean, half, std = mean_ci(grp[metric].to_numpy())
            rec[f"{metric}_mean"] = round(mean, 4)
            rec[f"{metric}_ci95"] = round(half, 4)
            rec[f"{metric}_std"] = round(std, 4)
        summary_rows.append(rec)
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("f1_macro_mean", ascending=False)
        .reset_index(drop=True)
    )

    # ---- Friedman + Nemenyi on macro-F1 ----
    f1_matrix = per_seed.pivot(index="seed", columns="model", values="f1_macro")
    model_order = list(f1_matrix.columns)
    # Rank per seed: highest F1 -> rank 1 (best). 'min' method handles ties.
    ranks = f1_matrix.rank(axis=1, ascending=False, method="average")
    average_ranks = ranks.mean(axis=0).sort_values()

    # Friedman omnibus across the model columns.
    f1_cols = [f1_matrix[m].to_numpy() for m in model_order]
    friedman_stat, friedman_p = friedmanchisquare(*f1_cols)

    k = len(model_order)
    n_blocks = f1_matrix.shape[0]
    cd = nemenyi_critical_difference(k, n_blocks, alpha=0.05)

    # Identify significantly different pairs (|rank_i - rank_j| > CD).
    significant: list[tuple[str, str, float]] = []
    names = list(average_ranks.index)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = abs(average_ranks[names[i]] - average_ranks[names[j]])
            if diff > cd:
                significant.append((names[i], names[j], round(float(diff), 3)))

    return MultiSeedResult(
        seeds=list(seeds),
        per_seed_metrics=per_seed,
        summary=summary,
        f1_matrix=f1_matrix,
        average_ranks=average_ranks,
        friedman_stat=float(friedman_stat),
        friedman_p=float(friedman_p),
        nemenyi_cd=float(cd),
        significant_pairs=significant,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_metric_forest(result: MultiSeedResult, output_path: str) -> str:
    """
    Forest plot: per-model macro-F1 and accuracy mean with 95 % CI bars,
    over the seed ensemble.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary = result.summary
    models = summary["model"].tolist()
    y = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(13, 0.7 * len(models) + 2.5), sharey=True)
    for ax, metric, title in zip(
        axes, ["f1_macro", "accuracy"], ["Macro-F1", "Accuracy"], strict=True
    ):
        means = summary[f"{metric}_mean"].to_numpy()
        cis = summary[f"{metric}_ci95"].to_numpy()
        ax.errorbar(
            means,
            y,
            xerr=cis,
            fmt="o",
            color="#1f77b4",
            capsize=5,
            markersize=7,
            linewidth=2,
        )
        for yi, m, c in zip(y, means, cis, strict=True):
            ax.annotate(
                f"{m:.3f} ± {c:.3f}",
                (m, yi),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.set_title(f"{title} (mean ± 95% CI, {len(result.seeds)} seeds)")
        ax.set_xlabel(title)
        ax.grid(True, axis="x", alpha=0.3)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_critical_difference(result: MultiSeedResult, output_path: str) -> str:
    """
    Draw a Demsar-style critical-difference diagram. Models are placed on
    a horizontal axis by average rank (best = left). A CD bar shows the
    Nemenyi critical difference; horizontal "clique" bars connect groups
    of models that are NOT significantly different.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ranks = result.average_ranks.sort_values()
    names = list(ranks.index)
    values = ranks.to_numpy()
    k = len(names)
    cd = result.nemenyi_cd

    lo = int(np.floor(values.min()))
    hi = int(np.ceil(values.max()))
    lo = min(lo, 1)
    hi = max(hi, k)

    fig, ax = plt.subplots(figsize=(11, 0.5 * k + 3))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, k + 2)
    ax.invert_xaxis()  # rank 1 (best) on the right reads naturally; flip below
    ax.invert_xaxis()  # keep best on the LEFT

    # Top axis line with rank ticks.
    axis_y = k + 1
    ax.hlines(axis_y, lo, hi, color="black", linewidth=1.2)
    for r in range(lo, hi + 1):
        ax.vlines(r, axis_y - 0.12, axis_y + 0.12, color="black", linewidth=1.2)
        ax.text(r, axis_y + 0.28, str(r), ha="center", va="bottom", fontsize=9)
    ax.text((lo + hi) / 2, axis_y + 0.75, "Average rank (lower = better)", ha="center", fontsize=10)

    # Each model: a vertical stub down to a label row.
    for idx, (name, rank) in enumerate(zip(names, values, strict=True)):
        row_y = axis_y - 1.0 - idx * 0.7
        ax.plot([rank, rank], [axis_y, row_y], color="#444", linewidth=1.0)
        ax.plot([rank, lo - 0.3], [row_y, row_y], color="#444", linewidth=1.0)
        ax.text(lo - 0.35, row_y, f"{name}  (r={rank:.2f})", ha="right", va="center", fontsize=9)

    # CD ruler.
    ruler_y = axis_y - 0.55
    ax.hlines(ruler_y, lo, lo + cd, color="#d62728", linewidth=3)
    ax.text(
        lo + cd / 2, ruler_y + 0.18, f"CD = {cd:.2f}", ha="center", color="#d62728", fontsize=10
    )

    # Clique bars: connect consecutive models whose rank gap <= CD.
    clique_y = axis_y - 0.30
    i = 0
    drawn = 0
    while i < k:
        j = i
        while j + 1 < k and (values[j + 1] - values[i]) <= cd:
            j += 1
        if j > i:
            ax.hlines(
                clique_y - drawn * 0.16,
                values[i] - 0.03,
                values[j] + 0.03,
                color="#2ca02c",
                linewidth=4,
            )
            drawn += 1
        i = j + 1 if j > i else i + 1

    ax.axis("off")
    ax.set_title(
        f"Critical-difference diagram (Nemenyi, α=0.05, {len(result.seeds)} seeds)\n"
        f"Friedman χ²={result.friedman_stat:.2f}, p={result.friedman_p:.2e}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> MultiSeedResult:
    import argparse

    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    results_dir = os.path.join(project_root, "results")

    parser = argparse.ArgumentParser(description="Multi-seed statistical benchmark.")
    parser.add_argument(
        "--n-seeds", type=int, default=15, help="Number of independent seeds (default 15)."
    )
    parser.add_argument("--n-components", type=int, default=500)
    parser.add_argument(
        "--seed0",
        type=int,
        default=cfg.RANDOM_SEED,
        help="First seed; the harness uses seed0..seed0+n_seeds-1.",
    )
    args = parser.parse_args(argv)

    seeds = list(range(args.seed0, args.seed0 + args.n_seeds))
    print(
        f"Running multi-seed benchmark over {len(seeds)} seeds "
        f"({seeds[0]}..{seeds[-1]}), n_components={args.n_components} ..."
    )
    result = run_multiseed_benchmark(seeds, n_components=args.n_components)

    print("\n=== Per-model metrics (mean +/- 95% CI over seeds) ===")
    show_cols = ["model", "f1_macro_mean", "f1_macro_ci95", "accuracy_mean", "accuracy_ci95"]
    print(result.summary[show_cols].to_string(index=False))

    print("\n=== Average macro-F1 rank (lower = better) ===")
    print(result.average_ranks.round(3).to_string())

    print(f"\nFriedman chi2 = {result.friedman_stat:.3f},  p = {result.friedman_p:.3e}")
    print(f"Nemenyi CD (alpha=0.05) = {result.nemenyi_cd:.3f}")
    if result.friedman_p < 0.05:
        print("Friedman REJECTS H0: at least one model differs significantly.")
    else:
        print("Friedman does not reject H0 at alpha=0.05.")
    if result.significant_pairs:
        print("Significantly different pairs (rank-gap > CD):")
        for a, b, d in result.significant_pairs:
            print(f"  {a:<18} vs {b:<18} rank-gap={d}")
    else:
        print("No pairwise differences exceed the CD.")

    forest = plot_metric_forest(result, os.path.join(results_dir, "multiseed_forest.png"))
    cd_path = plot_critical_difference(result, os.path.join(results_dir, "critical_difference.png"))
    result.summary.to_csv(os.path.join(results_dir, "multiseed_summary.csv"), index=False)
    result.per_seed_metrics.to_csv(os.path.join(results_dir, "multiseed_per_seed.csv"), index=False)
    print(f"\nSaved: {forest}")
    print(f"Saved: {cd_path}")
    print("Saved: results/multiseed_summary.csv, results/multiseed_per_seed.csv")
    return result


if __name__ == "__main__":
    import sys

    HERE = os.path.dirname(os.path.abspath(__file__))
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    main()
