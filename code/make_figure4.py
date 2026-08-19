#!/usr/bin/env python3
"""
P4-1 · Figure 4 — controls and mechanism. Arial, bold A-D, real radiographs with Grad-CAM.

The claim: the model does look at the bone (A, B). Yet half the performance survives an input with the bone erased (Fig 3A).
             And that result is not an artefact of undertraining (C); it is stable across seeds (D).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_export import save_panels

BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
R = BASE / "results"

matplotlib.rcParams.update({
    "font.family": "Arial", "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#3a3a38", "axes.labelcolor": "#111111",
    "xtick.color": "#3a3a38", "ytick.color": "#3a3a38",
    "axes.grid": True, "grid.color": "#e8e8e4", "grid.linewidth": 0.55,
    "axes.axisbelow": True, "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_NULL, C_GEOM, C_BASE, C_CNN = "#9a9a95", "#1baf7a", "#eb6834", "#2a78d6"
INK, INK2 = "#111111", "#52514e"
ROM = {"Type 1": "I", "Type 2": "II", "Type 3": "III", "Type 4": "IV", "Type 5": "V", "Type 6": "VI"}


PANEL_TEXTS = []


def panel(ax, letter, dx=-0.055, dy=1.06):
    t = ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=13, fontweight="bold",
                color=INK, ha="left", va="top")
    PANEL_TEXTS.append(t)
    return t


def drop_letters():
    """Panel-level files carry no A/B/C letter (letters are added when assembling)."""
    global PANEL_TEXTS
    for t in PANEL_TEXTS:
        t.remove()
    PANEL_TEXTS = []


cam = np.load(R / "exp12_cams.npz")
gc = pd.read_csv(R / "exp12_gradcam_stats.csv")
picks = json.load(open(R / "fig4_cam_picks.json"))

fig = plt.figure(figsize=(7.4, 5.3))
gs = fig.add_gridspec(2, 4, height_ratios=[0.42, 1.25], hspace=0.46, wspace=0.58,
                      left=0.095, right=0.965, top=0.885, bottom=0.105)

# ── A: Grad-CAM overlays ────────────────────────────────────────────────────
axes_a = [fig.add_subplot(gs[0, i]) for i in range(4)]
for ax, p in zip(axes_a, picks):
    img = np.load(R / "campack" / f"{p['key']}.npy").astype(float) / 255.
    c = cam[p["key"]].astype(float)
    c = np.kron(c, np.ones((4, 4)))[:448, :448]
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.imshow(c, cmap="inferno", alpha=0.42, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#3a3a38"); sp.set_linewidth(0.6)
    ok = "correct" if p["correct"] else "misclassified"
    ax.set_title(f"Schatzker {ROM[p['true']]}\n{ok}", fontsize=7.2, color=INK, pad=3)
panel(axes_a[0], "A", dx=-0.24, dy=1.30)
axes_a[1].text(1.29, 1.52, "Grad-CAM on the mask-ROI model: attention sits on the bone, not the periphery",
               transform=axes_a[1].transAxes, fontsize=8.8, color=INK, ha="center", va="top")

# ── B: CAM mass in the outer margin ─────────────────────────────────────────
axB = fig.add_subplot(gs[1, :2])
order = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6"]
g = gc.groupby("true")["cam_edge_fraction"]
mu = g.mean().reindex(order); sd = g.std().reindex(order); nn = g.count().reindex(order)
axB.bar(range(6), mu.values, yerr=sd.values, color=C_CNN, width=0.6, linewidth=0,
        error_kw=dict(lw=0.9, capsize=2.5, ecolor=INK2))
unif = gc.edge_area_fraction.iloc[0]
axB.axhline(unif, ls=(0, (4, 3)), lw=1.1, color=C_BASE)
axB.text(5.45, unif + .012, "uniform attention", ha="right",
         va="bottom", fontsize=7.2, color=C_BASE)
for i, (v, n_) in enumerate(zip(mu.values, nn.values)):
    axB.text(i, 0.012, f"{int(n_)} img", ha="center", fontsize=6.2, color="#ffffff")
axB.set_xticks(range(6)); axB.set_xticklabels([ROM[c] for c in order], fontsize=7.8)
axB.set_ylabel("fraction of Grad-CAM mass\nin the outer 15% margin", fontsize=8)
axB.set_ylim(0, 0.62); axB.set_xlabel("Schatzker class", fontsize=8)
pm = gc.groupby("pid")["cam_edge_fraction"].mean()
t, pv = stats.ttest_1samp(pm.values, unif)
axB.set_title(f"Attention is central, not peripheral\n"
              f"{pm.mean():.3f} per patient vs {unif:.2f} expected (n = {len(pm)})",
              fontsize=8.5, color=INK, loc="center", pad=6)
panel(axB, "B", dx=-0.135, dy=1.10)

# ── C: training curves, true vs permuted labels ─────────────────────────────
axC = fig.add_subplot(gs[1, 2])
for f, lab, col in [("exp08_G6_roi_resnet50_prep2_image_curves.csv", "true labels", C_CNN),
                    ("exp08_G6_shuffle_resnet50_prep2_image_curves.csv", "permuted labels", C_NULL)]:
    d = pd.read_csv(R / f).groupby("epoch")["train_acc"].agg(["mean", "std"])
    axC.plot(d.index, d["mean"], lw=1.8, color=col, label=lab)
    axC.fill_between(d.index, d["mean"] - d["std"], d["mean"] + d["std"], color=col, alpha=.18, lw=0)
axC.set_xlabel("epoch", fontsize=8); axC.set_ylabel("training accuracy", fontsize=8)
axC.set_ylim(0, 1.12); axC.legend(fontsize=7, frameon=False, loc="lower right",
                                  handlelength=1.2, borderpad=0.2)
axC.set_title("Convergence\n(six-class, mask ROI)", fontsize=8.5, color=INK, loc="center", pad=6)
panel(axC, "C", dx=-0.32, dy=1.10)

# ── D: seed-to-seed stability ───────────────────────────────────────────────
axD = fig.add_subplot(gs[1, 3])
s9 = json.load(open(R / "exp09_stats.json"))
CFG = [("G6_roi", "6-class ROI", C_CNN), ("G6_bg", "tibia erased", C_GEOM),
       ("G6_center", "centre", C_GEOM), ("G6_shuffle", "permuted", C_NULL)]
for i, (k, lab, col) in enumerate(CFG):
    v = np.array(s9[k]["seed_vals"])
    axD.scatter(np.full(5, i) + np.linspace(-.14, .14, 5), v, s=20, color=col,
                edgecolors="#ffffff", linewidths=.8, zorder=3)
    axD.plot([i - .22, i + .22], [v.mean()] * 2, lw=1.6, color=INK)
    axD.text(i, v.max() + .012, f"SD\n{v.std(ddof=1):.3f}", ha="center", fontsize=6.5,
             color=INK2, linespacing=1.2)
axD.axhline(1 / 6, ls=(0, (4, 3)), lw=.9, color=INK2)
axD.set_xticks(range(4)); axD.set_xticklabels([l for _, l, _ in CFG], fontsize=6.8, rotation=30, ha="right")
axD.set_ylabel("balanced accuracy", fontsize=8); axD.set_ylim(0.10, 0.46)
axD.set_title("Seed-to-seed spread\n(six-class)", fontsize=8.5, color=INK, loc="center", pad=6)
panel(axD, "D", dx=-0.36, dy=1.10)

fig.savefig(R / "Figure4.png", dpi=400)
fig.savefig(R / "Figure4.pdf")
drop_letters()
save_panels(fig, {"Figure4A": axes_a, "Figure4B": axB, "Figure4C": axC, "Figure4D": axD})
print("Figure 4 saved")
print(f"  CAM edge mass {gc.cam_edge_fraction.mean():.3f} vs uniform {unif:.3f}, p={pv:.2e}")
