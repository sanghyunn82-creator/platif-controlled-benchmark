#!/usr/bin/env python3
"""
P4-1 · Experiment 9 — **the statistics the audit demanded.** All fold-level t-tests are discarded.

What changed:
  - fold-value t-test (ignores fold correlation, Dietterich 1998) -> **patient-level stratified bootstrap, 20,000 resamples**
  - across-configuration comparison -> **paired comparison on the same seeds and splits** (seed level)
  - single-seed point estimate -> **mean over 5 seeds +/- between-seed SD**
  - per-class recall with **exact Clopper-Pearson intervals**
  - **Benjamini-Hochberg correction** reported alongside

Input is the per-image logit CSV written by exp08 (no retraining).
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
R = BASE / "results"
rng = np.random.default_rng(20260813)
NB = 20000


def load(cfg):
    f = R / f"exp08_{cfg}_resnet50_prep2_image_logits.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    cols = [c for c in d.columns if c.startswith("logit_")]
    classes = [c[6:] for c in cols]
    # aggregate to patients: mean logit per patient -> one vote per patient
    out = {}
    for seed, ds in d.groupby("seed"):
        g = ds.groupby("pid")
        pred = g[cols].mean().values.argmax(1)
        true = np.array([classes.index(v) for v in g["true"].first().values])
        out[seed] = {"pid": g["true"].first().index.values, "true": true, "pred": pred}
    return classes, out


def boot_bacc(true, pred, n=NB):
    """Patient-level stratified bootstrap — resample with replacement within each class."""
    vals = []
    idx_by_c = {c: np.where(true == c)[0] for c in np.unique(true)}
    for _ in range(n):
        take = np.concatenate([rng.choice(v, len(v), replace=True) for v in idx_by_c.values()])
        try:
            vals.append(balanced_accuracy_score(true[take], pred[take]))
        except Exception:
            pass
    v = np.array(vals)
    return v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5), v


def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def seedvals(out):
    return np.array([balanced_accuracy_score(v["true"], v["pred"]) for v in out.values()])


print("=" * 88)
print("■ 1. performance by configuration — 5-seed mean +/- between-seed SD, patient-level bootstrap CI")
print("=" * 88)
CFG = ["G6_roi", "G6_bg", "G6_center", "G6_shuffle", "C2_roi", "C2_shuffle", "A7_roi"]
CH = {"G6": 1/6, "C2": 0.5, "A7": 1/7}
store = {}
print(f"{'config':<12s} {'seed mean':>9s} {'seed SD':>8s} {'boot mean':>9s} {'95% CI':>18s} {'chance':>7s}")
print("-" * 88)
for c in CFG:
    r = load(c)
    if not r:
        print(f"{c:<12s} (missing)"); continue
    classes, out = r
    sv = seedvals(out)
    # bootstrap over all seeds pooled (seed x patient)
    T = np.concatenate([v["true"] for v in out.values()])
    P = np.concatenate([v["pred"] for v in out.values()])
    m, lo, hi, dist = boot_bacc(T, P)
    ch = CH[c[:2]]
    store[c] = {"classes": classes, "out": out, "seed_vals": sv, "boot": dist,
                "true": T, "pred": P, "chance": ch}
    print(f"{c:<12s} {sv.mean():9.4f} {sv.std(ddof=1):8.4f} {m:9.4f} "
          f"[{lo:.3f}, {hi:.3f}]".rjust(0).rjust(18) + f" {ch:7.3f}")

print()
print("=" * 88)
print("■ 2. paired comparisons — same seeds and splits, so the tests are paired")
print("=" * 88)
PAIRS = [("G6_roi", "G6_bg", "ROI - background only : does anatomy add beyond the background?"),
         ("G6_roi", "G6_center", "ROI - centre crop : contribution of localisation"),
         ("G6_roi", "G6_shuffle", "ROI - permuted null"),
         ("G6_bg", "G6_shuffle", "background only - permuted null : is there signal in the background?"),
         ("C2_roi", "C2_shuffle", "binary ROI - permuted null")]
pvals = []
for a, b, desc in PAIRS:
    if a not in store or b not in store:
        continue
    d = store[a]["seed_vals"] - store[b]["seed_vals"]
    t = stats.ttest_rel(store[a]["seed_vals"], store[b]["seed_vals"])
    se = d.std(ddof=1) / np.sqrt(len(d))
    ci = stats.t.ppf(0.975, len(d) - 1) * se
    pvals.append(t.pvalue)
    print(f"  {desc}")
    print(f"    diff {d.mean():+.4f} +/- {d.std(ddof=1):.4f} · 95% CI [{d.mean()-ci:+.4f}, {d.mean()+ci:+.4f}] "
          f"· p={t.pvalue:.4f}")
# Benjamini-Hochberg correction
if pvals:
    order = np.argsort(pvals); adj = np.empty(len(pvals))
    for rank, i in enumerate(order, 1):
        adj[i] = min(1, pvals[i] * len(pvals) / rank)
    print(f"\n  BH-adjusted q values: {[f'{x:.4f}' for x in adj]}")

print()
print("=" * 88)
print("■ 3. decomposition of the background contribution — the audit blocker")
print("=" * 88)
if "G6_roi" in store and "G6_bg" in store and "G6_shuffle" in store:
    ch = 1/6
    roi, bg, nul = (store[k]["seed_vals"].mean() for k in ["G6_roi", "G6_bg", "G6_shuffle"])
    print(f"  chance {ch:.3f} · permuted null {nul:.3f} · background only {bg:.3f} · ROI {roi:.3f}")
    print(f"  margin above chance: ROI +{roi-ch:.3f} · background only +{bg-ch:.3f}")
    print(f"  -> **the background alone recovers {100*(bg-ch)/(roi-ch):.0f}% of the ROI margin**")
    print(f"  relative to the null: ROI +{roi-nul:.3f} · background only +{bg-nul:.3f} "
          f"→ {100*(bg-nul)/(roi-nul):.0f}%")

print()
print("=" * 88)
print("■ 4. Type 6 decomposition — what the six-class score is made of")
print("=" * 88)
for cfg in ["G6_roi", "G6_bg"]:
    if cfg not in store: continue
    s = store[cfg]; classes = s["classes"]; T, P = s["true"], s["pred"]
    print(f"\n  [{cfg}] per-class recall (5 seeds pooled, Clopper-Pearson 95% CI)")
    for i, c in enumerate(classes):
        m = T == i
        k, n = int((P[m] == i).sum()), int(m.sum())
        lo, hi = clopper(k, n)
        print(f"    {c:<8s} {k:3d}/{n:3d} = {k/n:.3f}  [{lo:.3f}, {hi:.3f}]")
    # Type 6 versus the rest
    i6 = classes.index("Type 6")
    t6, p6 = (T == i6).astype(int), (P == i6).astype(int)
    m, lo, hi, _ = boot_bacc(t6, p6)
    print(f"    -> Type6 vs rest bacc {m:.3f} [{lo:.3f}, {hi:.3f}]")
    # five classes excluding Type 6
    keep = T != i6
    if keep.sum():
        m5, lo5, hi5, _ = boot_bacc(T[keep], P[keep])
        print(f"    -> five classes excluding Type6, bacc {m5:.3f} [{lo5:.3f}, {hi5:.3f}] (chance 0.200)")

print()
print("=" * 88)
print("■ 5. binary task — does it beat the metadata baseline (0.798)?")
print("=" * 88)
if "C2_roi" in store:
    sv = store["C2_roi"]["seed_vals"]
    for base, name in [(0.798, "full metadata"), (0.785, "CT availability only"), (0.593, "mask geometry")]:
        t = stats.ttest_1samp(sv, base)
        se = sv.std(ddof=1)/np.sqrt(len(sv)); ci = stats.t.ppf(0.975, len(sv)-1)*se
        verdict = "exceeds" if (sv.mean()-ci) > base else ("indistinguishable" if sv.mean()+ci > base else "does not exceed")
        print(f"  vs {name:<22s} {base:.3f} -> seed mean {sv.mean():.4f} "
              f"[{sv.mean()-ci:.4f}, {sv.mean()+ci:.4f}] · p={t.pvalue:.4f} · **{verdict}**")
    print(f"\n  WARNING: the baseline used 5-fold x 20 repeats while the CNN used 5-fold x 5 seeds.")
    print(f"    The schemes differ, so this comparison is indicative only; the manuscript must use")

json.dump({k: {"seed_vals": v["seed_vals"].tolist(),
               "boot_mean": float(v["boot"].mean()),
               "boot_ci": [float(np.percentile(v["boot"], 2.5)), float(np.percentile(v["boot"], 97.5))]}
           for k, v in store.items()},
          open(R / "exp09_stats.json", "w"), indent=2, ensure_ascii=False)
print(f"\nwritten: {R/'exp09_stats.json'}")
