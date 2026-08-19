#!/usr/bin/env python3
"""
P4-1 · Figure 2 candidate — performance ladder by task.

What the single panel is meant to say:
  not "how much did the CNN reach", but **"how much of that is available without looking at the knee"**.

Left: the ladder for three tasks (chance -> permuted null -> mask geometry -> metadata -> CNN)
Right: the share of the margin above chance explained by the non-anatomical baseline — 104% / 36% / 13%
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "Apple SD Gothic Neo"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
R = BASE / "results"


def jl(p):
    return {r["config"]: r for r in (json.loads(l) for l in open(p) if l.strip())} if p.exists() else {}


cnn3, cnn6 = jl(R / "exp03_cnn.jsonl"), jl(R / "exp06_primary6.jsonl")
e1 = pd.read_csv(R / "exp01_shortcut_baselines.csv")
e5 = pd.read_csv(R / "exp05_mask_geometry.csv")
e7 = pd.read_csv(R / "exp07_baseline6.csv")

def bestv(df, col, val):
    d = df[df[col] == val]
    return float(d.balanced_acc.max()) if len(d) else np.nan

TASKS = [
    {"name": "Binary\nNTPF vs fracture", "chance": 0.500, "n": 186,
     "null": cnn3.get("F_7cls_roi_shuffle"),  # the binary shuffle was never run — handled as None below
     "null_val": np.nan,
     "geom": bestv(e5, "task", "Binary NTPF vs fracture — mask geometry only"),
     "meta": bestv(e1, "target", "Binary: NTPF vs tibial plateau fracture"),
     "cnn": cnn3.get("C_bin_roi", {}).get("balanced_acc", np.nan),
     "cnn_folds": [0.877, 0.753, 0.776, 0.818, 0.696],
     "verdict": "baseline not exceeded\n(p=0.670)", "color": "#c8553d"},
    {"name": "Seven-class\n(NTPF included)", "chance": 1/7, "n": 186,
     "null_val": cnn3.get("F_7cls_roi_shuffle", {}).get("balanced_acc", np.nan),
     "geom": bestv(e5, "task", "Seven-class — mask geometry only"),
     "meta": bestv(e1, "target", "Seven-class: Schatzker Type 1-6 plus NTPF"),
     "cnn": cnn3.get("A_7cls_roi", {}).get("balanced_acc", np.nan),
     "cnn_folds": [0.263, 0.348, 0.324, 0.439, 0.520],
     "verdict": "baseline exceeded", "color": "#c9a227"},
    {"name": "Six-class (fracture only)\nprimary analysis", "chance": 1/6, "n": 128,
     "null_val": cnn6.get("I_6cls_roi_shuffle", {}).get("balanced_acc", np.nan),
     "geom": bestv(e7, "features", "mask geometry"),
     "meta": bestv(e7, "features", "metadata (non-anatomical)"),
     "cnn": cnn6.get("G_6cls_roi", {}).get("balanced_acc", np.nan),
     "cnn_folds": [0.477, 0.294, 0.429, 0.349, 0.233],
     "verdict": "baseline exceeded\n(p=0.020)", "color": "#3d7a5a"},
]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6.2),
                              gridspec_kw={"width_ratios": [1.75, 1]})

LAYERS = [("chance", "chance", "#bbbbbb"), ("permuted null", "null_val", "#9aa7b1"),
          ("mask geometry", "geom", "#7f9ec4"), ("non-anatomical metadata", "meta", "#e0a458"),
          ("CNN (image)", "cnn", None)]
W = 0.15
x = np.arange(len(TASKS))
for i, (lbl, key, col) in enumerate(LAYERS):
    vals = [t.get(key, np.nan) for t in TASKS]
    cols = [col if col else t["color"] for t in TASKS]
    off = (i - 2) * W
    b = ax.bar(x + off, vals, W, label=lbl, color=cols,
               edgecolor="white", linewidth=0.6)
    for xi, v in zip(x + off, vals):
        if not np.isnan(v):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5, rotation=90)

# CNN fold scatter
for j, t in enumerate(TASKS):
    xs = np.full(5, j + 2 * W) + np.random.default_rng(j).uniform(-0.035, 0.035, 5)
    ax.scatter(xs, t["cnn_folds"], s=14, color="black", zorder=5, alpha=0.75,
               label="CNN per fold" if j == 0 else None)

ax.set_xticks(x)
ax.set_xticklabels([f"{t['name']}\n({t['n']} patients)" for t in TASKS], fontsize=10)
ax.set_ylabel("balanced accuracy (patient-level evaluation)")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=8.5, ncol=3, loc="upper left")
ax.set_title("Performance ladder by task — how far do we get without pixels?", fontsize=12)
ax.grid(axis="y", alpha=0.25, lw=0.5)

# right: decomposition of the margin above chance
share, names, cols = [], [], []
for t in TASKS:
    dc, dm = t["cnn"] - t["chance"], t["meta"] - t["chance"]
    share.append(100 * dm / dc if dc > 0 else np.nan)
    names.append(t["name"].replace("\n", " ")); cols.append(t["color"])
b = ax2.barh(range(len(TASKS)), share, color=cols, height=0.55)
ax2.axvline(100, ls="--", lw=1, color="#c8553d")
ax2.text(100, -0.62, "100% = the image\nadds nothing", fontsize=8, color="#c8553d", ha="center")
for i, s in enumerate(share):
    ax2.text(s + 2.5, i, f"{s:.0f}%", va="center", fontsize=11, weight="bold")
ax2.set_yticks(range(len(TASKS))); ax2.set_yticklabels(names, fontsize=10)
ax2.invert_yaxis()
ax2.set_xlabel("share of the CNN margin above chance\nexplained by the non-anatomical baseline (%)")
ax2.set_xlim(0, 125)
ax2.set_title("Narrowing the task reduces the contamination", fontsize=12)
ax2.grid(axis="x", alpha=0.25, lw=0.5)

fig.suptitle("PlaTiF: one task in this dataset is contaminated and the other is clean",
             fontsize=13.5, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.945])
p = R / "fig02_performance_ladder.png"
fig.savefig(p, dpi=140)
print(f"written: {p}")
for t, s in zip(TASKS, share):
    print(f"  {t['name'].replace(chr(10),' '):<34s} chance {t['chance']:.3f} · metadata {t['meta']:.3f} · "
          f"CNN {t['cnn']:.3f} -> explained {s:.0f}%")
