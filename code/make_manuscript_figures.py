#!/usr/bin/env python3
"""
P4-1 · Three manuscript figures (English). Uses the validated palette.

Fig 1  Care-pathway confounding — CT availability by class
Fig 2  Benchmark ladder — every image model against a same-split non-imaging baseline
Fig 3  Where the signal is — input ablation + per-class recall with exact CIs

Colour roles (passes validate_palette.js: contrast WARN resolved by printing the value on every bar)
  null(shuffle)      #9a9a95  neutral grey — a floor, not data
  mask geometry      #1baf7a  aqua
  non-imaging base   #eb6834  orange
  image model (CNN)  #2a78d6  blue
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
R = BASE / "results"
OUT = R

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e6e2", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
})
C_NULL, C_GEOM, C_BASE, C_CNN = "#9a9a95", "#1baf7a", "#eb6834", "#2a78d6"
INK, INK2 = "#0b0b0b", "#52514e"

stats9 = json.load(open(R / "exp09_stats.json"))
base10 = json.load(open(R / "exp10_samesplit_baseline.json"))
sv = lambda k: np.array(stats9[k]["seed_vals"])
pre = {r["task"]: r for r in base10 if r.get("prespecified")}
geom = {r["task"]: r for r in base10 if r["features"] == "mask geometry" and r["model"] == "logistic regression"}
BIN, SIX = "Binary: NTPF vs fracture", "Six-class (fracture patients)"


def valuelabel(ax, x, y, txt, dy=0.012, size=7.5):
    ax.text(x, y + dy, txt, ha="center", va="bottom", fontsize=size, color=INK)


# ══════════════════════════════════ Figure 1 ══════════════════════════════════
scan = [json.loads(l) for l in open(R / "eda05_fullscan.jsonl")]
meta = pd.read_excel(BASE / "data/platif/Tibial Plateau Fracture Metadata.xlsx")
SCH = "Fracture Type of  Schatzker Classification"
ctmap = {r["pid"]: r["has_ct"] for r in scan}
meta["has_ct"] = meta["Patient ID"].map(ctmap)
meta["cls"] = meta[SCH].replace("Normal", "NTPF")
ORD = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6", "NTPF"]
LBL = ["I", "II", "III", "IV", "V", "VI", "NTPF"]
g = meta.groupby("cls")["has_ct"].agg(["sum", "count"]).reindex(ORD)
rate = 100 * g["sum"] / g["count"]

fig, ax = plt.subplots(figsize=(7.2, 3.7))
cols = [C_CNN] * 6 + [C_BASE]
b = ax.bar(range(7), rate, color=cols, width=0.66, linewidth=0)
for i, (v, s, c) in enumerate(zip(rate, g["sum"], g["count"])):
    valuelabel(ax, i, v, f"{v:.0f}%", dy=1.5)
    ax.text(i, 4, f"{int(s)}/{int(c)}", ha="center", fontsize=7, color="#ffffff")
ax.set_xticks(range(7)); ax.set_xticklabels(LBL)
ax.set_xlabel("Schatzker class (NTPF = no tibial plateau fracture)")
ax.set_ylabel("patients with a coronal CT section (%)")
ax.set_ylim(0, 118)
is_ntpf = (meta.cls == "NTPF").values
has_ct = meta.has_ct.astype(int).values
# [[fracture without CT, fracture with CT], [NTPF without CT, NTPF with CT]]
t2 = [[int(((~is_ntpf) & (has_ct == 0)).sum()), int(((~is_ntpf) & (has_ct == 1)).sum())],
      [int((is_ntpf & (has_ct == 0)).sum()),    int((is_ntpf & (has_ct == 1)).sum())]]
odds, p = stats.fisher_exact(t2)
print("  2x2 =", t2)
ax.set_title("Coronal CT availability tracks the label, not the anatomy", loc="left",
             fontsize=10.5, color=INK, pad=34)
ax.text(0, 1.005, f"Fisher exact p = {p:.1e} · odds ratio = {1/odds:.0f}\n"
        f"The descriptor states a coronal CT was acquired for every patient; 150/186 have one.",
        transform=ax.transAxes, fontsize=8, color=INK2, va="bottom", linespacing=1.6)
fig.tight_layout()
fig.savefig(OUT / "figure1_confounding.png", dpi=300)
print("Figure 1 saved · Fisher p =", f"{p:.3e}", "OR =", f"{1/odds:.1f}")

# ══════════════════════════════════ Figure 2 ══════════════════════════════════
TASKS = [
    {"name": "Binary\nNTPF vs fracture", "n": 186, "chance": 0.500,
     "null": sv("C2_shuffle"), "geom": geom[BIN], "base": pre[BIN], "cnn": sv("C2_roi")},
    {"name": "Six-class Schatzker\n(fracture patients)", "n": 128, "chance": 1 / 6,
     "null": sv("G6_shuffle"), "geom": geom[SIX], "base": pre[SIX], "cnn": sv("G6_roi")},
]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
for ax, t in zip(axes, TASKS):
    top = 1.20 if t["chance"] > 0.3 else 0.58
    series = [("label-permutation null", t["null"], C_NULL),
              ("mask geometry only", np.array(t["geom"]["seed_vals"]), C_GEOM),
              ("non-imaging baseline\n(pre-specified)", np.array(t["base"]["seed_vals"]), C_BASE),
              ("image model (ResNet-50)", t["cnn"], C_CNN)]
    for i, (lab, vals, col) in enumerate(series):
        ax.bar(i, vals.mean(), color=col, width=0.62, linewidth=0,
               label=lab if ax is axes[0] else None)
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.13, 0.13, len(vals)), vals,
                   s=13, color="#ffffff", edgecolors=INK, linewidths=0.7, zorder=4)
        valuelabel(ax, i, vals.max(), f"{vals.mean():.3f}", dy=top * 0.022)
    ax.axhline(t["chance"], ls=(0, (4, 3)), lw=1, color=INK2)
    ax.text(0.5, t["chance"] + top * .008, "chance", ha="center", va="bottom",
            fontsize=7.5, color=INK2)
    ax.set_xticks(range(4)); ax.set_xticklabels(["null", "geometry", "baseline", "image"])
    ax.set_ylim(0, top)
    ax.set_title(f"{t['name']}   (n = {t['n']} patients)", fontsize=9.5, color=INK, loc="left")
    ax.set_ylabel("balanced accuracy" if ax is axes[0] else "")
    ax.set_yticks(np.arange(0, (1.01 if t["chance"] > .3 else .451), .2 if t["chance"] > .3 else .1))
    # significance annotation — pinned clearly above bars, points and value labels
    d = t["cnn"] - np.array(t["base"]["seed_vals"])
    _, pv = stats.ttest_rel(t["cnn"], np.array(t["base"]["seed_vals"]))
    hi = max(t["cnn"].max(), np.array(t["base"]["seed_vals"]).max())
    y = hi + top * 0.14
    ax.plot([2, 2, 3, 3], [y - top * .02, y, y, y - top * .02], lw=0.9, color=INK2)
    ax.text(2.5, y + top * .012, f"Δ {d.mean():+.3f},  p = {pv:.3f}", ha="center",
            fontsize=8, color=INK)
axes[0].legend(fontsize=7.5, frameon=False, loc="upper left", ncol=1)
fig.suptitle("Every image model is compared against a non-imaging baseline fitted on the identical patient partitions",
             fontsize=10.5, color=INK, x=0.012, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "figure2_ladder.png", dpi=300)
print("Figure 2 saved")

# ══════════════════════════════════ Figure 3 ══════════════════════════════════
fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                               gridspec_kw={"width_ratios": [1, 1.25]})
# (A) input ablation
ABL = [("label-\npermuted", sv("G6_shuffle"), C_NULL),
       ("background\nonly", sv("G6_bg"), C_GEOM),
       ("centre crop\n(no mask)", sv("G6_center"), C_GEOM),
       ("mask ROI\n(annotator)", sv("G6_roi"), C_CNN)]
for i, (lab, vals, col) in enumerate(ABL):
    axA.bar(i, vals.mean(), color=col, width=0.62, linewidth=0)
    axA.scatter(np.full(len(vals), i) + np.linspace(-0.13, 0.13, len(vals)), vals,
                s=13, color="#ffffff", edgecolors=INK, linewidths=0.7, zorder=4)
    valuelabel(axA, i, vals.max(), f"{vals.mean():.3f}", dy=0.011)
# chance (0.167) and the pre-specified non-imaging baseline (0.165) essentially coincide — drawn as one line and described as such
bl = np.array(pre[SIX]["seed_vals"]).mean()
axA.axhline(1 / 6, ls=(0, (4, 3)), lw=1, color=INK2)
axA.text(3.46, 1 / 6 + .006, f"chance (0.167) = non-imaging baseline ({bl:.3f})",
         ha="right", va="bottom", fontsize=7.5, color=INK2)
axA.set_xticks(range(4))
axA.set_xticklabels([l for l, _, _ in ABL], fontsize=8)
axA.set_ylabel("balanced accuracy"); axA.set_ylim(0, 0.48)
axA.set_title("A \u00b7 Erasing the bone removes only half the signal", fontsize=9.5,
              color=INK, loc="left")

# (B) per-class recall with Clopper-Pearson intervals
d = pd.read_csv(R / "exp08_G6_roi_resnet50_prep2_image_logits.csv")
cols = [c for c in d.columns if c.startswith("logit_")]
classes = [c[6:] for c in cols]
gg = d.groupby(["seed", "pid"])
pred = gg[cols].mean().values.argmax(1)
true = np.array([classes.index(v) for v in gg["true"].first().values])
rows = []
for i, c in enumerate(classes):
    m = true == i
    k, n = int((pred[m] == i).sum()), int(m.sum())
    lo = 0.0 if k == 0 else stats.beta.ppf(.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(.975, k + 1, n - k)
    ROM = {"1":"I","2":"II","3":"III","4":"IV","5":"V","6":"VI"}
    rows.append((ROM[c.replace("Type ", "")], k, n, k / n, lo, hi))
rows.sort(key=lambda r: r[3])
ys = np.arange(len(rows))
for j, (c, k, n, r, lo, hi) in enumerate(rows):
    axB.plot([lo, hi], [j, j], lw=2, color=C_CNN, solid_capstyle="butt", alpha=.55)
    axB.scatter([r], [j], s=42, color=C_CNN, zorder=4, edgecolors="#ffffff", linewidths=1.2)
    axB.text(0.965, j, f"{k}/{n}", va="center", ha="right", fontsize=7.5, color=INK2)
axB.axvline(1 / 6, ls=(0, (4, 3)), lw=1, color=INK2)
axB.text(1 / 6, -0.72, "chance", fontsize=7.5, color=INK2, va="bottom", ha="center")
axB.set_yticks(ys); axB.set_yticklabels([f"Schatzker {r[0]}" for r in rows])
axB.set_xlim(-0.02, 1.0); axB.set_ylim(-1.0, len(rows) - 0.4); axB.set_xlabel("per-class recall (95% Clopper–Pearson CI)")
axB.set_title("B · Only Schatzker VI is identified; IV and V collapse", fontsize=9.5,
              color=INK, loc="left")
fig.suptitle("Six-class Schatzker typing among fracture patients (n = 128, five seeds)",
             fontsize=10.5, color=INK, x=0.012, ha="left", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT / "figure3_ablation_perclass.png", dpi=300)
print("Figure 3 saved")
for c, k, n, r, lo, hi in rows:
    print(f"   Schatzker {c:<4s} {k:3d}/{n:3d} = {r:.3f} [{lo:.3f}, {hi:.3f}]")
