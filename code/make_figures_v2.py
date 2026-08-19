#!/usr/bin/env python3
"""
P4-1 · Manuscript figures v2 — real radiographs, Arial, bold panel labels, four or more panels.

Problem with v1: an imaging paper with no knee radiograph in it, DejaVu Sans, and panels that were too plain.

Figure 1  The dataset and a confound in how it was assembled   (A–D)
Figure 2  What the model is shown, and how it scores            (A–D)
Figure 3  Where it fails, and why that is the coronal plane     (A–D)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_export import save_panels

BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
R = BASE / "results"
IMG = R / "figpack"

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
ROM = {"Type 1": "I", "Type 2": "II", "Type 3": "III", "Type 4": "IV",
       "Type 5": "V", "Type 6": "VI", "NTPF": "NTPF"}
ORD = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6", "NTPF"]

stats9 = json.load(open(R / "exp09_stats.json"))
base10 = json.load(open(R / "exp10_samesplit_baseline.json"))
keys = json.load(open(R / "fig_example_keys.json"))
sv = lambda k: np.array(stats9[k]["seed_vals"])
# exp17: the tibia-only (inverse ablation) run postdates exp09, so it is read from its own file
_ab17 = json.load(open(R / "exp17_ablation_pair.json"))
sv17 = lambda k: np.array(_ab17["sixclass_bacc"][k]["per_seed"])
pre = {r["task"]: r for r in base10 if r.get("prespecified")}
geo = {r["task"]: r for r in base10 if r["features"] == "mask geometry" and r["model"] == "logistic regression"}
BIN, SIX = "Binary: NTPF vs fracture", "Six-class (fracture patients)"

scan = [json.loads(l) for l in open(R / "eda05_fullscan.jsonl")]
meta = pd.read_excel(BASE / "data/platif/Tibial Plateau Fracture Metadata.xlsx")
SCH = "Fracture Type of  Schatzker Classification"
meta["has_ct"] = meta["Patient ID"].map({r["pid"]: r["has_ct"] for r in scan})
meta["cls"] = meta[SCH].replace("Normal", "NTPF")
imgdf = pd.read_csv(R / "eda06_aggregate.csv")


PANEL_TEXTS = []


def panel(fig, ax, letter, dx=-0.055, dy=1.055):
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


def show(ax, arr, title=None, ts=7.5):
    ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#3a3a38"); sp.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=ts, color=INK, pad=3)


def ladder(ax, task, chance, top, show_legend=False, ylab=True):
    ser = [("label-permuted", sv({"bin": "C2_shuffle", "six": "G6_shuffle"}[task]), C_NULL),
           ("mask geometry", np.array(geo[BIN if task == "bin" else SIX]["seed_vals"]), C_GEOM),
           ("non-imaging\nbaseline", np.array(pre[BIN if task == "bin" else SIX]["seed_vals"]), C_BASE),
           ("image model", sv({"bin": "C2_roi", "six": "G6_roi"}[task]), C_CNN)]
    for i, (lab, v, col) in enumerate(ser):
        ax.bar(i, v.mean(), color=col, width=0.6, linewidth=0, label=lab if show_legend else None)
        ax.scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, s=11,
                   color="#ffffff", edgecolors=INK, linewidths=.6, zorder=4)
        ax.text(i, max(v.max(), chance) + top * .030, f"{v.mean():.3f}", ha="center",
            fontsize=7.2, color=INK,
            bbox=dict(boxstyle="square,pad=0.06", fc="#ffffff", ec="none"))
    ax.axhline(chance, ls=(0, (4, 3)), lw=.9, color=INK2)
    ax.text(0.5, chance + top * .008, "chance", ha="center", va="bottom", fontsize=7, color=INK2)
    d = ser[3][1] - ser[2][1]
    _, pv = stats.ttest_rel(ser[3][1], ser[2][1])
    y = max(ser[3][1].max(), ser[2][1].max()) + top * .13
    ax.plot([2, 2, 3, 3], [y - top * .02, y, y, y - top * .02], lw=.85, color=INK2)
    ax.text(2.5, y + top * .012, f"Δ {d.mean():+.3f}\np = {pv:.3f}", ha="center",
            fontsize=7.2, color=INK, linespacing=1.3)
    ax.set_xticks(range(4)); ax.set_xticklabels(["null", "geom", "base", "image"], fontsize=7.5)
    ax.set_ylim(0, top)
    if ylab:
        ax.set_ylabel("balanced accuracy", fontsize=8)


# ═══════════════════════════ FIGURE 1 ═══════════════════════════
fig = plt.figure(figsize=(7.4, 6.3))
gs = fig.add_gridspec(3, 7, height_ratios=[0.52, 1.0, 0.86], hspace=0.62, wspace=1.05,
                      left=0.095, right=0.985, top=0.955, bottom=0.075)

# A — representative image per class with the tibial mask outline
axes_a = [fig.add_subplot(gs[0, i]) for i in range(7)]
for ax, c in zip(axes_a, ORD):
    a = np.load(IMG / "roi" / f"{keys[c]}.npy")
    show(ax, a, f"Schatzker {ROM[c]}" if c != "NTPF" else "NTPF", ts=7)
panel(fig, axes_a[0], "A", dx=-0.30, dy=1.30)
axes_a[3].text(0.5, 1.62, "One representative anteroposterior radiograph per class",
               transform=axes_a[3].transAxes, fontsize=9, color=INK, ha="center", va="top")
axes_a[3].text(0.5, -0.13, "NTPF = no Schatzker-classifiable tibial plateau fracture; all 186 patients "
               "carry a fracture diagnosis", transform=axes_a[3].transAxes, fontsize=7,
               color=INK2, ha="center", va="top")

# D — CT availability (cited last in the text)
axB = fig.add_subplot(gs[2, :])
g = meta.groupby("cls")["has_ct"].agg(["sum", "count"]).reindex(ORD)
rate = 100 * g["sum"] / g["count"]
axB.bar(range(7), rate, color=[C_CNN] * 6 + [C_BASE], width=0.66, linewidth=0)
for i, (v, s_, c_) in enumerate(zip(rate, g["sum"], g["count"])):
    axB.text(i, v + 2.5, f"{v:.0f}", ha="center", fontsize=7, color=INK)
    axB.text(i, 5, f"{int(s_)}/{int(c_)}", ha="center", fontsize=6.2, color="#ffffff")
axB.set_xticks(range(7)); axB.set_xticklabels([ROM[c] for c in ORD], fontsize=7.5)
axB.set_ylabel("patients with a\ncoronal CT (%)", fontsize=8); axB.set_ylim(0, 122)
is_n = (meta.cls == "NTPF").values; hc = meta.has_ct.astype(int).values
t2 = [[int(((~is_n) & (hc == 0)).sum()), int(((~is_n) & (hc == 1)).sum())],
      [int((is_n & (hc == 0)).sum()), int((is_n & (hc == 1)).sum())]]
odds, pv = stats.fisher_exact(t2)
axB.set_title(f"Coronal CT availability tracks the label\n"
              f"Fisher exact p = {pv:.1e},  odds ratio = {1/odds:.0f}",
              fontsize=8.5, color=INK, loc="center", pad=6)
panel(fig, axB, "D", dx=-0.065)

# B — class distribution
axC = fig.add_subplot(gs[1, :3])
cnt = meta.cls.value_counts().reindex(ORD)
axC.barh(range(7)[::-1], cnt.values, color=[C_CNN] * 6 + [C_BASE], height=0.62, linewidth=0)
for i, v in enumerate(cnt.values):
    axC.text(v + 1.5, 6 - i, str(v), va="center", fontsize=7, color=INK)
axC.set_yticks(range(7)[::-1]); axC.set_yticklabels([ROM[c] for c in ORD], fontsize=7.5)
axC.set_xlabel("patients", fontsize=8); axC.set_xlim(0, 72)
axC.set_title("Class distribution (n = 186)", fontsize=8.5, color=INK, loc="center", pad=6)
panel(fig, axC, "B", dx=-0.21)

# C — acquisition heterogeneity
axD = fig.add_subplot(gs[1, 3:])
sub = imgdf.copy()
sub["is_n"] = sub.cls == "NTPF"
axD.scatter(sub.loc[~sub.is_n, "w"], sub.loc[~sub.is_n, "h"], s=11, alpha=.55,
            color=C_CNN, linewidths=0, label="Schatzker I–VI")
axD.scatter(sub.loc[sub.is_n, "w"], sub.loc[sub.is_n, "h"], s=13, alpha=.75,
            color=C_BASE, linewidths=0, label="NTPF")
axD.axhline(4000, ls=(0, (4, 3)), lw=.9, color=INK2)
axD.text(1000, 4110, "height > 4000 px", fontsize=6.8, color=INK2, ha="left")
axD.set_ylim(1200, 5350)
axD.set_xlabel("image width (px)", fontsize=8); axD.set_ylabel("image height (px)", fontsize=8)
axD.legend(fontsize=7, frameon=False, loc="lower right")
axD.set_title("Acquisition is not uniform\n1467–4892 × 920–4892 px, aspect ratio 0.75–3.07",
              fontsize=8.5, color=INK, loc="center", pad=6)
panel(fig, axD, "C", dx=-0.115)

fig.savefig(R / "Figure1.png", dpi=400)
fig.savefig(R / "Figure1.pdf")
drop_letters()
save_panels(fig, {"Figure1A": axes_a, "Figure1B": axC, "Figure1C": axD, "Figure1D": axB})
print("Figure 1 saved")

# ═══════════════════════════ FIGURE 2 ═══════════════════════════
fig = plt.figure(figsize=(7.4, 6.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.42,
                      left=0.09, right=0.985, top=0.90, bottom=0.085)

# A — the four input variants (same patient)
kA = keys["Type 6"]
variants = [("full radiograph", "full"), ("mask ROI\n(annotator-provided)", "roi"),
            ("centre crop\n(no mask used)", "center"), ("tibia erased\n(femur, fibula kept)", "bg")]
axes_a = [fig.add_subplot(gs[0, i]) for i in range(4)]
for ax, (t, d) in zip(axes_a, variants):
    show(ax, np.load(IMG / d / f"{kA}.npy"), t, ts=7.2)
panel(fig, axes_a[0], "A", dx=-0.22, dy=1.34)
axes_a[1].text(1.21, 1.56, "Four of the five inputs compared, shown for one patient (Schatzker VI)",
               transform=axes_a[1].transAxes, fontsize=9, color=INK, ha="center", va="top")

# B, C — performance ladders
axB = fig.add_subplot(gs[1, :2]); ladder(axB, "bin", 0.5, 1.22, show_legend=True)
axB.set_title("Binary: NTPF vs fracture (n = 186)", fontsize=8.5, color=INK, loc="center", pad=6)
axB.legend(fontsize=6.8, frameon=False, loc="upper left", handlelength=1.1)
panel(fig, axB, "B", dx=-0.135)
axC = fig.add_subplot(gs[1, 2:]); ladder(axC, "six", 1 / 6, 0.60, ylab=False)
axC.set_title("Six-class Schatzker, fracture patients (n = 128)", fontsize=8.5,
              color=INK, loc="center", pad=6)
panel(fig, axC, "C", dx=-0.115)

fig.text(0.09, 0.022, "Every image model is compared with a pre-specified non-imaging baseline "
         "fitted on the identical patient partitions; dots are the five seeds.",
         fontsize=7.5, color=INK2)
fig.savefig(R / "Figure2.png", dpi=400)
fig.savefig(R / "Figure2.pdf")
drop_letters()
save_panels(fig, {"Figure2A": axes_a, "Figure2B": axB, "Figure2C": axC})
print("Figure 2 saved")

# ═══════════════════════════ FIGURE 3 ═══════════════════════════
d = pd.read_csv(R / "exp08_G6_roi_resnet50_prep2_image_logits.csv")
cols = [c for c in d.columns if c.startswith("logit_")]
classes = [c[6:] for c in cols]
gg = d.groupby(["seed", "pid"])
pred = gg[cols].mean().values.argmax(1)
true = np.array([classes.index(v) for v in gg["true"].first().values])

fig = plt.figure(figsize=(7.4, 6.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.18], hspace=0.30, wspace=0.30,
                      left=0.115, right=0.975, top=0.915, bottom=0.075)

# A — ablation
axA = fig.add_subplot(gs[0, 0])
sil = json.load(open(R / "exp13_mask_silhouette.json"))
ABL = [("permuted", sv("G6_shuffle"), C_NULL),
       ("mask\nsilhouette", np.array(sil["16"]), C_GEOM),
       ("tibia\nerased", sv("G6_bg"), C_CNN),
       ("tibia\nonly", sv17("tib_tibia_only"), C_CNN),
       ("centre", sv("G6_center"), C_CNN),
       ("mask ROI", sv("G6_roi"), C_CNN)]
for i, (lab, v, col) in enumerate(ABL):
    axA.bar(i, v.mean(), color=col, width=0.6, linewidth=0)
    axA.scatter(np.full(len(v), i) + np.linspace(-.10, .10, len(v)), v, s=9,
                color="#ffffff", edgecolors=INK, linewidths=.55, zorder=4)
    axA.text(i, v.max() + .012, f"{v.mean():.3f}", ha="center", fontsize=6.8, color=INK)
bl = np.array(pre[SIX]["seed_vals"]).mean()
axA.axhline(1 / 6, ls=(0, (4, 3)), lw=.9, color=INK2)
# mark that the erasure (2) and its complement (3) are a pair — the key contrast of this figure
_yb = 0.435
axA.plot([2, 2, 3, 3], [_yb - .012, _yb, _yb, _yb - .012], lw=.85, color=INK2, clip_on=False)
axA.text(2.5, _yb + .006, "complementary pair\n(partition the radiograph)", ha="center",
         va="bottom", fontsize=6.0, color=INK2, linespacing=1.25)

axA.set_xticks(range(len(ABL))); axA.set_xticklabels([l for l, _, _ in ABL], fontsize=6.3)
axA.set_ylabel("balanced accuracy", fontsize=8); axA.set_ylim(0, 0.52)
axA.set_title(f"Erasure and its complement\n"
              f"dashed line: chance 0.167\n≈ baseline {bl:.3f}",
              fontsize=8.5, color=INK, loc="center", pad=6)
panel(fig, axA, "A", dx=-0.155)

# B — per-class recall
axB = fig.add_subplot(gs[0, 1])
rows = []
for i, c in enumerate(classes):
    m = true == i
    k, n = int((pred[m] == i).sum()), int(m.sum())
    lo = 0. if k == 0 else stats.beta.ppf(.025, k, n - k + 1)
    hi = 1. if k == n else stats.beta.ppf(.975, k + 1, n - k)
    rows.append((ROM[c], k, n, k / n, lo, hi))
rows.sort(key=lambda r: r[3])
for j, (c, k, n, r_, lo, hi) in enumerate(rows):
    axB.plot([lo, hi], [j, j], lw=2.2, color=C_CNN, alpha=.5, solid_capstyle="butt")
    axB.scatter([r_], [j], s=34, color=C_CNN, zorder=4, edgecolors="#ffffff", linewidths=1.1)
    axB.text(0.985, j, f"{k}/{n}", va="center", ha="right", fontsize=7, color=INK2)
axB.axvline(1 / 6, ls=(0, (4, 3)), lw=.9, color=INK2)
axB.text(1 / 6, -0.78, "chance", fontsize=7, color=INK2, ha="center", va="bottom")
axB.set_yticks(range(len(rows))); axB.set_yticklabels([f"Schatzker {r[0]}" for r in rows], fontsize=7.8)
axB.set_xlim(-0.02, 1.0); axB.set_ylim(-1.05, len(rows) - 0.4)
axB.set_xlabel("per-class recall (95% Clopper–Pearson CI)", fontsize=8)
axB.set_title("Recall is concentrated in Schatzker VI", fontsize=8.5, color=INK, loc="center", pad=6)
panel(fig, axB, "B", dx=-0.235)

# C — confusion matrix
axC = fig.add_subplot(gs[1, 0])
cm = np.zeros((6, 6), int)
for t_, p_ in zip(true, pred):
    cm[t_, p_] += 1
cmn = cm / cm.sum(1, keepdims=True)
im = axC.imshow(cmn, cmap="Blues", vmin=0, vmax=0.75)
for i in range(6):
    for j in range(6):
        axC.text(j, i, f"{cmn[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=6.6,
                 color="#ffffff" if cmn[i, j] > .42 else INK)
axC.set_xticks(range(6)); axC.set_xticklabels([ROM[c] for c in classes], fontsize=7.5)
axC.set_yticks(range(6)); axC.set_yticklabels([ROM[c] for c in classes], fontsize=7.5)
axC.set_xlabel("predicted", fontsize=8); axC.set_ylabel("true", fontsize=8)
axC.grid(False)
axC.set_title("Row-normalised confusion (5 seeds pooled)", fontsize=8.5, color=INK, loc="center", pad=6)
panel(fig, axC, "C", dx=-0.215)

# D — real images of the failure modes
axD_gs = gs[1, 1].subgridspec(1, 2, wspace=0.08)
axD_list = []
for i, c in enumerate(["Type 4", "Type 5"]):
    axd = fig.add_subplot(axD_gs[0, i]); axD_list.append(axd)
    show(axd, np.load(IMG / "roi" / f"{keys[c]}.npy"),
         f"Schatzker {ROM[c]}\nrecall {[r for r in rows if r[0]==ROM[c]][0][3]:.2f}", ts=7.2)
    axd.set_xlabel({"Type 4": "medial condyle", "Type 5": "bicondylar"}[c],
                   fontsize=7.2, color=INK2, labelpad=3)
    if i == 0:
        panel(fig, axd, "D", dx=-0.16, dy=1.72)
        axd.text(1.04, 1.50, "The two classes with the\nlowest recall",
                 transform=axd.transAxes, fontsize=8.5, color=INK, ha="center", va="top",
                 linespacing=1.45)

fig.savefig(R / "Figure3.png", dpi=400)
fig.savefig(R / "Figure3.pdf")
drop_letters()
save_panels(fig, {"Figure3A": axA, "Figure3B": axB, "Figure3C": axC, "Figure3D": axD_list})
print("Figure 3 saved")
for c, k, n, r_, lo, hi in rows:
    print(f"   Schatzker {c:<4s} {k:3d}/{n:3d} = {r_:.3f} [{lo:.3f}, {hi:.3f}]")
