"""Generate the MSRCF showcase / hero banner (2:1) from real pipeline results.

Outputs results/showcase_hero.png  (2400x1200 px, 2:1 -> GitHub social preview
+ LinkedIn hero). Pure matplotlib, reads the CSVs the main pipeline already
writes. Re-run any time with:  python make_showcase_hero.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
DATA = BASE / "data"
OUT = BASE / "assets" / "showcase_hero.png"

# ---- palette -------------------------------------------------------------
NAVY = "#0B2545"
ACCENT = "#2563EB"
ACCENT_L = "#9DC3FF"
GREEN = "#2E9E5B"
AMBER = "#E2A40A"
RED = "#D1342F"
ORANGE = "#E0822F"
GRAYBAR = "#C2CCD6"
INK = "#1F2933"
MUTE = "#6B7480"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#C9D2DC",
        "axes.linewidth": 0.9,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTE,
        "ytick.color": MUTE,
        "axes.labelcolor": MUTE,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

fig = plt.figure(figsize=(16, 8), dpi=150)
fig.patch.set_facecolor("white")

# ---- banner --------------------------------------------------------------
fig.add_artist(
    Rectangle((0, 0.84), 1, 0.16, transform=fig.transFigure, facecolor=NAVY, zorder=-10, clip_on=False)
)
fig.add_artist(
    Rectangle((0, 0.835), 1, 0.006, transform=fig.transFigure, facecolor=ACCENT, zorder=-9, clip_on=False)
)

fig.text(0.045, 0.945, "MSRCF", fontsize=37, fontweight="bold", color="white", va="center")
fig.text(0.175, 0.955, "Manufacture-to-Service Risk Continuity Framework", fontsize=15, color="white", va="center")
fig.text(0.175, 0.920, "for aerospace carbon-fibre composite structures", fontsize=11, color=ACCENT_L, va="center")
fig.text(
    0.045,
    0.873,
    "One risk number for a composite part — from the factory floor to the flight line.",
    fontsize=12,
    color="#D8E2F0",
    style="italic",
    va="center",
)
fig.text(0.965, 0.945, "Raaga Sundar", fontsize=15.5, fontweight="bold", color="white", ha="right", va="center")
fig.text(0.965, 0.912, "github.com/RaagaSundar/MSRCF", fontsize=10.5, color=ACCENT_L, ha="right", va="center")
fig.text(
    0.965,
    0.878,
    "~92% accuracy   ·   Friedman p < 1e-9   ·   distribution-free coverage",
    fontsize=9.5,
    color="#D8E2F0",
    ha="right",
    va="center",
)

# ---- panels --------------------------------------------------------------
gs = fig.add_gridspec(1, 3, left=0.05, right=0.965, top=0.74, bottom=0.155, wspace=0.235)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
for ax in (ax1, ax2, ax3):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# === Panel 1: RCS trajectories with Monte Carlo bands =====================
traj = pd.read_csv(DATA / "msrcf_rcs_trajectories.csv")
finals = traj.sort_values("cycle").groupby("component_id").tail(1).sort_values("rcs")
idxs = np.linspace(0, len(finals) - 1, 5).round().astype(int)
chosen = finals.iloc[idxs]["component_id"].tolist()
flag_color = {"GREEN": GREEN, "YELLOW": AMBER, "RED": RED}

ymax = max(traj[traj.component_id == c]["rcs_upper"].max() for c in chosen)
ymin = min(traj[traj.component_id == c]["rcs_lower"].min() for c in chosen)
top = max(ymax * 1.05, 74)
bottom = max(0, ymin * 0.9)

ax1.axhspan(0, 40, color=GREEN, alpha=0.05, zorder=0)
ax1.axhspan(40, 70, color=AMBER, alpha=0.05, zorder=0)
ax1.axhspan(70, top, color=RED, alpha=0.05, zorder=0)
ax1.axhline(40, color=AMBER, ls="--", lw=1, alpha=0.6, zorder=1)
ax1.axhline(70, color=RED, ls="--", lw=1, alpha=0.6, zorder=1)
bbox = dict(fc="white", ec="none", alpha=0.75, pad=0.6)
ax1.text(0.15, 41, "YELLOW", fontsize=7.5, color=AMBER, va="bottom", ha="left", bbox=bbox, zorder=4)
ax1.text(0.15, 71, "RED", fontsize=7.5, color=RED, va="bottom", ha="left", bbox=bbox, zorder=4)

for cid in chosen:
    d = traj[traj.component_id == cid].sort_values("cycle")
    c = flag_color.get(d["flag"].iloc[-1], ACCENT)
    ax1.fill_between(d["cycle"], d["rcs_lower"], d["rcs_upper"], color=c, alpha=0.13, zorder=2)
    ax1.plot(d["cycle"], d["rcs"], color=c, lw=2.1, zorder=3)
    ax1.scatter(d["cycle"].iloc[-1], d["rcs"].iloc[-1], color=c, s=20, zorder=4)

ax1.set_xlim(0, 10)
ax1.set_ylim(bottom, top)
ax1.set_xlabel("Inspection cycle")
ax1.set_ylabel("Risk Continuity Score")
ax1.set_title("Risk evolves over service life", loc="left", pad=17)
ax1.text(0.0, 1.012, "per-component RCS with Monte-Carlo 90% bands", transform=ax1.transAxes, fontsize=8.8, color=MUTE)

# === Panel 2: classifier benchmark ========================================
ms = pd.read_csv(RESULTS / "multiseed_summary.csv").sort_values("accuracy_mean")
disp = {"LogisticRegression": "Logistic Reg.", "RandomForest": "Random Forest"}
names = [disp.get(m, m) for m in ms["model"]]
acc = ms["accuracy_mean"].to_numpy()
err = ms["accuracy_ci95"].to_numpy()
top2 = set(ms.sort_values("accuracy_mean", ascending=False)["model"].head(2))
colors = [ACCENT if m in top2 else GRAYBAR for m in ms["model"]]
ypos = np.arange(len(names))
ax2.barh(ypos, acc, xerr=err, color=colors, height=0.62, error_kw=dict(ecolor=MUTE, elinewidth=1.1, capsize=3), zorder=3)
ax2.set_yticks(ypos)
ax2.set_yticklabels(names)
for i, (a, e) in enumerate(zip(acc, err)):
    ax2.text(a + e + 0.015, i, f"{a:.3f}", va="center", ha="left", fontsize=9, color=INK)
ax2.set_xlim(0, 1.0)
ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.set_xlabel("Accuracy (mean ± 95% CI, 15 seeds)")
ax2.set_title("5-classifier damage-mode benchmark", loc="left", pad=17)
ax2.text(0.0, 1.012, "XGBoost & Random Forest tied; both beat the rest (p<1e-9)", transform=ax2.transAxes, fontsize=8.8, color=MUTE)
ax2.xaxis.grid(True, color="#E9EDF2", lw=0.8, zorder=0)
ax2.set_axisbelow(True)

# === Panel 3: conformal per-class coverage ================================
cp = pd.read_csv(RESULTS / "conformal_per_class.csv")
order = [0, 1, 2, 3, 4]
short = {0: "None", 1: "Matrix", 2: "Delam", 3: "Fiber", 4: "Fatigue"}
lac = cp[cp.method == "LAC"].set_index("class_id")["coverage"].reindex(order)
aps = cp[cp.method == "APS"].set_index("class_id")["coverage"].reindex(order)
x = np.arange(len(order))
w = 0.38
ax3.bar(x - w / 2, lac.to_numpy(), width=w, color=ORANGE, zorder=3)
ax3.bar(x + w / 2, aps.to_numpy(), width=w, color=ACCENT, zorder=3)
ax3.axhline(0.90, color=INK, ls="--", lw=1.2, zorder=4)
ax3.text(4.45, 0.905, "90% target", fontsize=8, color=INK, va="bottom", ha="right", bbox=bbox, zorder=5)
# inline legend above the first group
ax3.text(-w / 2, lac.iloc[0] + 0.015, "LAC", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=ORANGE)
ax3.text(w / 2, aps.iloc[0] + 0.015, "APS", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=ACCENT)
# flag the LAC under-coverage
for xi, v in zip(x, lac.to_numpy()):
    if v < 0.85:
        ax3.text(xi - w / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=RED)
ax3.set_xticks(x)
ax3.set_xticklabels([short[c] for c in order])
ax3.set_ylim(0, 1.10)
ax3.set_ylabel("Per-class coverage")
ax3.set_title("Conformal coverage by damage mode", loc="left", pad=17)
ax3.text(0.0, 1.012, "naive (LAC) under-covers rare modes — adaptive (APS) restores it", transform=ax3.transAxes, fontsize=8.5, color=MUTE)
ax3.yaxis.grid(True, color="#E9EDF2", lw=0.8, zorder=0)
ax3.set_axisbelow(True)

# ---- footer --------------------------------------------------------------
fig.add_artist(Rectangle((0.05, 0.086), 0.915, 0.0016, transform=fig.transFigure, facecolor="#E0E6EC", zorder=-5))
fig.text(
    0.5,
    0.05,
    "Python  ·  scikit-learn  ·  XGBoost  ·  SHAP  ·  SALib  ·  matplotlib"
    "        |        reproducible (fixed seeds)  ·  CI-tested on 3.10–3.12  ·  MIT licensed",
    ha="center",
    fontsize=9.5,
    color=MUTE,
)

OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor="white")
print(f"saved {OUT}  ({int(16 * 150)}x{int(8 * 150)} px, 2:1)")
