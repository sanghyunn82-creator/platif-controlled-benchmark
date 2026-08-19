#!/usr/bin/env python3
"""
P4-1 · The non-anatomical signal revealed by the full scan, in one figure — a candidate for Figure 1.

Each panel shows how tightly an **acquisition or administrative by-product**, rather than a pixel, tracks the class.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency

matplotlib.rcParams["font.family"] = "Apple SD Gothic Neo"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
pat = pd.read_csv(BASE / "results/eda06_patients.csv")
img = pd.read_csv(BASE / "results/eda06_aggregate.csv")
ORDER = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6", "NTPF"]
C_NTPF, C_FX = "#c8553d", "#4a7ba7"
col = [C_FX] * 6 + [C_NTPF]

fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.2))

# (a) CT availability
g = pat.groupby("cls")["has_ct"] if "has_ct" in pat else None
ct = pat.assign(has_ct=pat.pid.map(dict(zip(img.pid, [1]*len(img))))) if g is None else None
scan = [__import__("json").loads(l) for l in open(BASE / "results/eda05_fullscan.jsonl")]
ctmap = {r["pid"]: r["has_ct"] for r in scan}
pat["has_ct"] = pat.pid.map(ctmap).astype(int)
r = pat.groupby("cls")["has_ct"].agg(["sum", "count"]).reindex(ORDER)
rate = 100 * r["sum"] / r["count"]
a = ax[0, 0]
a.bar(ORDER, rate, color=col)
for i, (v, s, c) in enumerate(zip(rate, r["sum"], r["count"])):
    a.text(i, v + 2, f"{v:.0f}%\n({int(s)}/{int(c)})", ha="center", fontsize=8)
chi2, p, _, _ = chi2_contingency(pd.crosstab(pat.cls, pat.has_ct))
a.set_title(f"(a) Coronal CT availability — chi-square p={p:.1e}\n"
            f"the descriptor states it was \"acquired for each patient\"", fontsize=10)
a.set_ylabel("patients with a CT (%)"); a.set_ylim(0, 118)
a.tick_params(axis="x", rotation=30)

# (b) laterality skew
ctab = pd.crosstab(pat.cls, pat.lat).reindex(ORDER).fillna(0)
a = ax[0, 1]
bot = np.zeros(len(ORDER))
for c, cc in [("L", "#7ba7c8"), ("R", "#c8a54a"), ("R and L", "#8a8a8a")]:
    if c in ctab:
        a.bar(ORDER, ctab[c], bottom=bot, label=c, color=cc)
        bot += ctab[c].values
for i, cls in enumerate(ORDER):
    n = ctab.loc[cls].sum()
    rr = ctab.loc[cls].get("R", 0) / n * 100 if n else 0
    a.text(i, n + 0.8, f"R {rr:.0f}%", ha="center", fontsize=8)
a.set_title("(b) laterality — L/R is burnt into the corner\nNTPF is 67% right, Type 4 is 10% right", fontsize=10)
a.set_ylabel("patients"); a.legend(fontsize=8); a.tick_params(axis="x", rotation=30)

# (c) age
a = ax[0, 2]
data = [pat.loc[pat.cls == c, "age"].values for c in ORDER]
bp = a.boxplot(data, labels=ORDER, patch_artist=True, widths=0.6)
for patch, c in zip(bp["boxes"], col):
    patch.set_facecolor(c); patch.set_alpha(0.65)
a.set_title("(c) age — NTPF patients are significantly younger\n(NTPF 38.7 y vs fracture types 43.5-47.9 y)", fontsize=10)
a.set_ylabel("age"); a.tick_params(axis="x", rotation=30)

# (d) field of view
img["fullscan"] = ((img.h > 4000) | (img.w > 3000)).astype(int)
r2 = img.groupby("cls")["fullscan"].agg(["sum", "count"]).reindex(ORDER)
a = ax[1, 0]
a.bar(ORDER, 100 * r2["sum"] / r2["count"], color=col)
for i, (s, c) in enumerate(zip(r2["sum"], r2["count"])):
    a.text(i, 100 * s / c + 1, f"{int(s)}/{int(c)}", ha="center", fontsize=8)
a.set_title("(d) share of images above 4000 px (whole-leg)\n31 of the 51 are concentrated in NTPF", fontsize=10)
a.set_ylabel("share of images (%)"); a.tick_params(axis="x", rotation=30)

# (e) images per patient
a = ax[1, 1]
m = pat.groupby("cls")["n_im"].mean().reindex(ORDER)
sd = pat.groupby("cls")["n_im"].std().reindex(ORDER)
a.bar(ORDER, m, yerr=sd, color=col, capsize=3)
a.set_title("(e) images per patient\nlabels are per patient — the 421 images are not independent", fontsize=10)
a.set_ylabel("images (mean +/- SD)"); a.tick_params(axis="x", rotation=30)

# (f) resolution scatter
a = ax[1, 2]
for c, cc in zip(ORDER, col):
    s = img[img.cls == c]
    a.scatter(s.w, s.h, s=16, alpha=0.6, label=c,
              color=cc, edgecolors="none" if c != "NTPF" else "k", linewidths=0.4)
a.axhline(4000, ls="--", lw=0.8, c="gray"); a.axvline(3000, ls="--", lw=0.8, c="gray")
a.set_title("(f) resolution scatter — the protocol is not uniform\n1467-4892 x 920-4892, aspect ratio 0.75-3.07", fontsize=10)
a.set_xlabel("width (px)"); a.set_ylabel("height (px)"); a.legend(fontsize=7, ncol=2)

fig.suptitle("PlaTiF carries signal unrelated to anatomy that correlates with the label (all 186 patients, 421 images)",
             fontsize=13, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.955])
p = BASE / "results/fig01_nonanatomical_signal.png"
fig.savefig(p, dpi=140)
print(f"written: {p}")
