#!/usr/bin/env python3
"""
P4-1 · Aggregation of the full 186-patient scan — answers the five open questions.

The input is an already-summarised JSONL (256 KB), so this runs locally.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
SCAN = BASE / "results/eda05_fullscan.jsonl"
XLSX = BASE / "data/platif/Tibial Plateau Fracture Metadata.xlsx"
SCH = "Fracture Type of  Schatzker Classification"
LAT = "Fracture of the Right (R) or Left (L) Tibia"
NAME = {1: "Type 1", 2: "Type 2", 3: "Type 3", 4: "Type 4",
        5: "Type 5", 6: "Type 6", 7: "NTPF"}

recs = [json.loads(l) for l in open(SCAN)]
meta = pd.read_excel(XLSX)
mlab = dict(zip(meta["Patient ID"], meta[SCH]))
mage = dict(zip(meta["Patient ID"], meta["Age"]))
mlat = dict(zip(meta["Patient ID"], meta[LAT]))

print("=" * 78)
print(f"full-scan aggregation — patients {len(recs)} · errors {sum('error' in r for r in recs)}")
print("=" * 78)

# ── Q1. label constancy within a patient ───────────────────────────────────
print("\n> Q1. is the label constant within a patient? (the descriptor Usage Notes claim left and right may differ)")
nonconst = [r for r in recs if not r.get("label_constant", True)]
print(f"   patients with a constant label: {sum(r.get('label_constant', False) for r in recs)}/{len(recs)}")
print(f"   patients without             : {len(nonconst)}")
for r in nonconst:
    print(f"     WARNING ID {r['pid']}: labels {r['labels']} · xlsx '{mlab.get(r['pid'])}' · {mlat.get(r['pid'])} side")
if not nonconst:
    print("   -> constant for all 186. **The Usage Notes statement does not match the public data.**")

# .mat label vs xlsx
mism = [(r["pid"], r["labels"], mlab.get(r["pid"]))
        for r in recs if NAME.get(r["labels"][0]) != str(mlab.get(r["pid"])).replace("Normal", "NTPF")]
print(f"   .mat label vs xlsx mismatches: {len(mism)}" + (f" {mism[:5]}" if mism else " (all agree)"))

# ── Q2. images per patient, and the image-weighted age hypothesis ──────────
print("\n> Q2. images per patient -> which denominator gives the reported age 45.88+/-17.54?")
nim = np.array([r["n_images"] for r in recs])
print(f"   total images {nim.sum()} (descriptor: 421) · per patient {nim.min()}-{nim.max()}, "
      f"median {int(np.median(nim))}")
print(f"   distribution: {dict(sorted(Counter(nim.tolist()).items()))}")

ages = np.array([mage[r["pid"]] for r in recs], dtype=float)
labs = np.array([r["labels"][0] for r in recs])
w = nim.astype(float)
def ms(a, wt=None):
    if wt is None:
        return a.mean(), a.std(ddof=1)
    m = np.average(a, weights=wt)
    v = np.average((a - m) ** 2, weights=wt) * wt.sum() / (wt.sum() - 1)
    return m, np.sqrt(v)
print(f"   {'value reported in descriptor':34s} mean=45.88  SD=17.54")
for lbl, a, wt in [("plain mean over 186 patients", ages, None),
                   ("image-weighted mean over 421", ages, w),
                   ("128 patients excluding NTPF", ages[labs != 7], None),
                   ("image-weighted, excluding NTPF", ages[labs != 7], w[labs != 7])]:
    m, s = ms(a, wt)
    hit = "  <- match" if abs(m - 45.88) < 0.05 and abs(s - 17.54) < 0.05 else ""
    print(f"   {lbl:28s} mean={m:6.2f}  SD={s:5.2f}{hit}")

# ── Q3. coronal CT availability ────────────────────────────────────────────
print("\n> Q3. coronal CT availability (the descriptor claims both universal availability and absence)")
ct = np.array([r["has_ct"] for r in recs])
print(f"   available for {ct.sum()}/{len(recs)} ({100*ct.mean():.1f}%)")
tab = pd.DataFrame({"cls": [NAME[l] for l in labs], "ct": ct})
g = tab.groupby("cls")["ct"].agg(["sum", "count"])
g["availability%"] = (100 * g["sum"] / g["count"]).round(1)
print(g.to_string())
if ct.sum() < len(recs):
    from scipy.stats import chi2_contingency
    obs = pd.crosstab(tab["cls"], tab["ct"])
    if obs.shape[1] > 1:
        chi2, p, _, _ = chi2_contingency(obs)
        print(f"   class x CT availability, chi-square p={p:.4f} "
              f"{'<- class-dependent; the missingness leaks label information' if p < 0.05 else '<- independent of class'}")

# ── Q4/Q5. resolution, field of view, non-AP views ─────────────────────────
print("\n> Q4/Q5. resolution · field of view (whole-leg acquisitions) · mask encoding")
rows = []
for r in recs:
    for im in r["images"]:
        rows.append({"pid": r["pid"], "cls": NAME[im["label"]], "h": im["h"], "w": im["w"],
                     "aspect": im["aspect"], "mask_area": im["mask_area_frac"],
                     "vext": im["mask_vertical_extent"],
                     "bw_unique": tuple(im["bw_unique"]) if im["bw_unique"] else None,
                     "bw_dtype": im["bw_dtype"],
                     "omin": im["orig"].get("min"), "omax": im["orig"].get("max"),
                     "odtype": im["orig"]["dtype"], "onuniq": im["orig"].get("n_unique")})
df = pd.DataFrame(rows)
print(f"   images {len(df)}")
print(f"   height {df.h.min()}-{df.h.max()} (median {int(df.h.median())}) · "
      f"width {df.w.min()}-{df.w.max()} (median {int(df.w.median())})")
print(f"   aspect ratio h/w {df.aspect.min():.2f}-{df.aspect.max():.2f} (median {df.aspect.median():.2f})")
print(f"   original dtype: {dict(df.odtype.value_counts())} · value range min={df.omin.min()} max={df.omax.max()}")
print(f"   mask dtype: {dict(df.bw_dtype.value_counts())} · unique values: {dict(Counter(df.bw_unique))}")
print(f"   mask area fraction {df.mask_area.min():.3f}-{df.mask_area.max():.3f} (median {df.mask_area.median():.3f})")
big = df[(df.h > 4000) | (df.w > 3000)]
print(f"   above 4000 px (suspected whole-leg): {len(big)} images / {big.pid.nunique()} patients")
print(f"     by class: {dict(big.cls.value_counts())}")
wide = df[df.aspect < 1.0]
print(f"   wider than tall (suspected atypical view): {len(wide)} images / {wide.pid.nunique()} patients")

# ── quantifying the shortcut risk ──────────────────────────────────────────
print("\n> shortcut risk — laterality skew and image-count skew")
pat = pd.DataFrame({"pid": [r["pid"] for r in recs], "cls": [NAME[l] for l in labs],
                    "n_im": nim, "lat": [mlat.get(r["pid"]) for r in recs], "age": ages})
print(pd.crosstab(pat.cls, pat.lat).to_string())
print("\n   images per patient and age, by class")
print(pat.groupby("cls").agg(n=("pid", "count"), mean_images=("n_im", "mean"),
                             mean_age=("age", "mean")).round(2).to_string())

out = BASE / "results/eda06_aggregate.csv"
df.to_csv(out, index=False)
pat.to_csv(BASE / "results/eda06_patients.csv", index=False)
print(f"\nwritten: {out} · eda06_patients.csv")
