#!/usr/bin/env python3
"""
P4-1 · Supplementary Figures S1–S4.

S1  Dataset integrity: the two unusable masks               — show the defect
S2  Input-variant gallery across all six Schatzker classes  — full disclosure of model inputs
S3  Fold-level results for every configuration and seed     — the variation hidden by aggregates
S4  Per-class recall vs published human agreement           — report the **negative result** honestly
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
IMG = R / "figpack"
OUT = R

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
ORD6 = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6"]
keys = json.load(open(R / "fig_example_keys.json"))


def panel(ax, letter, dx=-0.055, dy=1.06, size=12):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size, fontweight="bold",
            color=INK, ha="left", va="top")


def show(ax, a, title=None, ts=7):
    ax.imshow(a, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#3a3a38"); sp.set_linewidth(0.5)
    if title:
        ax.set_title(title, fontsize=ts, color=INK, pad=2.5)


# ═════════════════════ S2 · input-variant gallery ═════════════════════
VAR = [("full radiograph", "full"), ("mask ROI", "roi"),
       ("centre crop", "center"), ("tibia erased", "bg"), ("tibia only", "tib")]
fig, axes = plt.subplots(5, 6, figsize=(7.4, 6.7))
for r, (vname, vdir) in enumerate(VAR):
    for c, cls in enumerate(ORD6):
        show(axes[r, c], np.load(IMG / vdir / f"{keys[cls]}.npy"),
             f"Schatzker {ROM[cls]}" if r == 0 else None)
    axes[r, 0].set_ylabel(vname, fontsize=7.5, color=INK)
fig.suptitle("Figure S2. The five inputs compared, shown for one representative patient per class",
             fontsize=9.5, color=INK, ha="center", y=0.985)
fig.text(0.012, 0.012, "The centre crop uses no mask; across the 419 images with an applicable mask it "
         "captures a median 0.447 of the tibial bounding box and none of it in 22 images (Table S2).\n"
         "Erasing the tibia leaves the femur, fibula, joint space and soft tissue intact, and the erased "
         "region retains the outline of the annotator's mask. The tibia-only input is the exact "
         "complement of the tibia-erased one: the two share a field of view and partition the radiograph.",
         fontsize=6.8, color=INK2, linespacing=1.35)
fig.tight_layout(rect=[0, 0.055, 1, 0.955])
fig.savefig(OUT / "FigureS2.png", dpi=400); fig.savefig(OUT / "FigureS2.pdf")
print("Figure S2 saved (input gallery)")

# ═════════════════════ S3 · fold-level detail ═════════════════════
rows = []
for l in open(R / "exp08_summary.jsonl"):
    r = json.loads(l)
    for k, v in enumerate(r["fold_baccs"], 1):
        rows.append({"config": r["config"], "seed": r["seed"], "fold": k, "bacc": v,
                     "chance": r["chance"]})
d = pd.DataFrame(rows)
CFGS = ["C2_roi", "C2_shuffle", "A7_roi", "G6_roi", "G6_bg", "G6_center", "G6_shuffle"]
NICE = {"C2_roi": "Binary, ROI", "C2_shuffle": "Binary, permuted", "A7_roi": "7-class, ROI",
        "G6_roi": "6-class, ROI", "G6_bg": "6-class, tibia erased", "G6_center": "6-class, centre",
        "G6_shuffle": "6-class, permuted"}
fig, axes = plt.subplots(1, 7, figsize=(7.4, 3.1), sharey=False)
for ax, cfg in zip(axes, CFGS):
    s = d[d.config == cfg]
    ch = s.chance.iloc[0]
    for sd_ in sorted(s.seed.unique()):
        ss = s[s.seed == sd_]
        ax.plot(ss.fold, ss.bacc, marker="o", ms=3, lw=0.9, alpha=.75,
                color=C_CNN if "shuffle" not in cfg else C_NULL)
    ax.axhline(ch, ls=(0, (4, 3)), lw=.9, color=INK2)
    ax.set_title(NICE[cfg], fontsize=6.8, color=INK)
    ax.set_xticks([1, 3, 5]); ax.tick_params(labelsize=6.5)
    ax.set_xlabel("fold", fontsize=7)
    ax.set_ylim(0, 1.0 if cfg.startswith("C2") else 0.62)
axes[0].set_ylabel("balanced accuracy", fontsize=8)
fig.suptitle("Figure S3. Fold-level balanced accuracy for every configuration and seed "
             "(five seeds, five folds); dashed line is chance",
             fontsize=9, color=INK, ha="center", y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT / "FigureS3.png", dpi=400); fig.savefig(OUT / "FigureS3.pdf")
print("Figure S3 saved (fold-level)")

# ═════════════════════ S1 · dataset integrity ═════════════════════
S2t = pd.read_csv(R / "supplementary/Table_S2.csv")
bad = S2t[~S2t.mask_applicable]
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1, 1, 1.5]})
scan = {r["pid"]: r for r in (json.loads(l) for l in open(R / "eda05_fullscan.jsonl"))}
for ax, (_, r) in zip(axes[:2], bad.iterrows()):
    im = next(i for i in scan[r.patient_id]["images"] if i["name"] == r.image_index)
    ax.axis("off")
    ax.text(0.5, 0.92, f"Patient {r.patient_id}, {r.image_index}", ha="center", fontsize=8.5,
            color=INK, transform=ax.transAxes, fontweight="bold")
    ax.text(0.5, 0.62, f"radiograph\n{r.height_px} × {r.width_px} px", ha="center", fontsize=8,
            color=INK, transform=ax.transAxes, linespacing=1.5)
    ax.text(0.5, 0.30, f"segmentation mask\n{im['orig']['shape'][0] if False else ''}",
            ha="center", fontsize=8, color=C_BASE, transform=ax.transAxes, linespacing=1.5)
    ax.add_patch(plt.Rectangle((0.08, 0.05), 0.84, 0.90, fill=False, lw=1,
                               edgecolor="#cccccc", transform=ax.transAxes))
# actual dimensions come from shapecheck.json
sc = json.load(open(R / "shapecheck.json"))
badsc = [x for x in sc if not x["ok_bw"]]
for ax, x in zip(axes[:2], badsc):
    ax.texts[2].set_text(f"segmentation mask\n{x['bw'][0]} × {x['bw'][1]} px")
axes[2].axis("off")
axes[2].text(0, 0.95, "Integrity findings (Table S6)", fontsize=9, color=INK,
             fontweight="bold", va="top")
S6t = pd.read_csv(R / "supplementary/Table_S6.csv")
txt = S6t.groupby("finding").size().sort_values(ascending=False)
lines = [f"• {k}\n   {v} occurrence(s)" for k, v in txt.items()]
axes[2].text(0, 0.83, "\n\n".join(lines), fontsize=7.2, color=INK2, va="top", linespacing=1.45)
fig.suptitle("Figure S1. Two of 421 images carry a segmentation mask whose dimensions differ from "
             "the radiograph", fontsize=9, color=INK, ha="center", y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(OUT / "FigureS1.png", dpi=400); fig.savefig(OUT / "FigureS1.pdf")
print("Figure S1 saved (integrity)")

# ═════════════════════ S4 · human-agreement comparison (negative result) ═════════════════════
KAP = {"I": 0.189, "II": 0.406, "III": 0.284, "IV": 0.393, "V": 0.237, "VI": 0.624}
dd = pd.read_csv(R / "exp08_G6_roi_resnet50_prep2_image_logits.csv")
cols = [c for c in dd.columns if c.startswith("logit_")]
classes = [c[6:] for c in cols]
gg = dd.groupby(["seed", "pid"])
pred = gg[cols].mean().values.argmax(1)
true = np.array([classes.index(v) for v in gg["true"].first().values])
rec = {}
for i, c in enumerate(classes):
    m = true == i
    rec[ROM[c]] = (pred[m] == i).mean()
fig, ax = plt.subplots(figsize=(5.2, 3.9))
for k in KAP:
    ax.scatter(KAP[k], rec[k], s=60, color=C_CNN, edgecolors="#ffffff", linewidths=1.2, zorder=3)
    ax.annotate(k, (KAP[k], rec[k]), textcoords="offset points", xytext=(7, 4),
                fontsize=8, color=INK)
sr = stats.spearmanr([KAP[k] for k in KAP], [rec[k] for k in KAP])
ax.set_xlabel("published human inter-observer agreement,\nAP radiograph alone (Fleiss κ)", fontsize=8)
ax.set_ylabel("our per-class recall", fontsize=8)
ax.set_xlim(0.12, 0.70); ax.set_ylim(0, 0.82)
ax.set_title(f"Figure S4. No association detected between where humans disagree\nand where the model fails "
             f"(Spearman ρ = {sr.statistic:.2f}, p = {sr.pvalue:.2f}, n = 6)",
             fontsize=8.5, color=INK, loc="center", pad=8)
fig.text(0.02, 0.015, "Human κ values are per-type, AP-only, from Masouros et al.,\n"
         "Cureus 2022;14:e22227. With six points this test has little power;\n"
         "we report it as a null, not as evidence of independence.",
         fontsize=6.8, color=INK2, linespacing=1.4)
fig.tight_layout(rect=[0, 0.115, 1, 1])
fig.savefig(OUT / "FigureS4.png", dpi=400); fig.savefig(OUT / "FigureS4.pdf")
print(f"Figure S4 saved  (Spearman rho={sr.statistic:.3f}, p={sr.pvalue:.3f})")
