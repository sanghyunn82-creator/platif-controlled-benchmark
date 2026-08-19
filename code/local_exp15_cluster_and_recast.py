#!/usr/bin/env python3
"""P4-1 · exp15 — (a) recompute every interval with a patient-cluster bootstrap,
                  (b) attach the ablation and baseline controls to the VI-vs-rest recast. CPU, local.

Review points:
 1. The earlier bootstrap resampled (seed, pid) rows as independent units. One patient appears five
    times, so the effective sample was inflated and the intervals were too narrow. The manuscript
    noted this dependence for Clopper-Pearson but stayed silent about the bootstrap -> replaced by
    patient-level cluster resampling.
 2. The VI-vs-rest figure of 0.793 carried none of the two controls applied to the six-class numbers
    (tibia erasure, non-imaging baseline).

Output: results/exp15_cluster_bootstrap.json
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import zoom as ndzoom

BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
R = BASE / "results"
NB = 20000
rng = np.random.default_rng(20260818)


def frame(cfg):
    d = pd.read_csv(R / f"exp08_{cfg}_resnet50_prep2_image_logits.csv")
    cols = [c for c in d.columns if c.startswith("logit_")]
    cls = [c[6:] for c in cols]
    g = d.groupby(["seed", "pid"])
    P = g[cols].mean().values.argmax(1)
    T = np.array([cls.index(v) for v in g["true"].first().values])
    idx = g["true"].first().index
    return (np.array([s for s, _ in idx]), np.array([p for _, p in idx]), T, P, cls)


def cluster_boot(T, P, pids, n=NB):
    """Resample patients with class stratification, carrying all five evaluations of each."""
    upid = np.unique(pids)
    lab = {p: T[pids == p][0] for p in upid}
    rows = {p: np.where(pids == p)[0] for p in upid}
    byc = {}
    for p in upid:
        byc.setdefault(lab[p], []).append(p)
    byc = {k: np.array(v) for k, v in byc.items()}
    out = []
    for _ in range(n):
        take = np.concatenate([rng.choice(v, len(v), replace=True) for v in byc.values()])
        idx = np.concatenate([rows[p] for p in take])
        out.append(balanced_accuracy_score(T[idx], P[idx]))
    o = np.array(out)
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]


def cluster_boot_diff(T, Pa, Pb, pids, n=NB):
    upid = np.unique(pids)
    lab = {p: T[pids == p][0] for p in upid}
    rows = {p: np.where(pids == p)[0] for p in upid}
    byc = {}
    for p in upid:
        byc.setdefault(lab[p], []).append(p)
    byc = {k: np.array(v) for k, v in byc.items()}
    out = []
    for _ in range(n):
        take = np.concatenate([rng.choice(v, len(v), replace=True) for v in byc.values()])
        idx = np.concatenate([rows[p] for p in take])
        out.append(balanced_accuracy_score(T[idx], Pa[idx]) - balanced_accuracy_score(T[idx], Pb[idx]))
    o = np.array(out)
    return [float(o.mean()), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]


res = {"cluster_bootstrap_ci": {}, "vi_vs_rest": {}, "five_class_excl_VI": {}}

# ── (a) cluster bootstrap for each CNN configuration ───────────────────────
for cfg in ["G6_roi", "G6_bg", "G6_center", "G6_shuffle", "C2_roi", "C2_shuffle", "A7_roi"]:
    seeds, pids, T, P, cls = frame(cfg)
    res["cluster_bootstrap_ci"][cfg] = cluster_boot(T, P, pids)
    print(f"  {cfg:11s} cluster 95% CI {res['cluster_bootstrap_ci'][cfg]}", flush=True)

# ── (b) VI-vs-rest recast plus controls ────────────────────────────────────
sil = np.load(R / "mask_silhouettes.npz", allow_pickle=True)
kmap = {k: i for i, k in enumerate(sil["keys"])}
X16 = np.stack([ndzoom(g.astype(np.float32), (16 / 48, 16 / 48), order=1)
                for g in sil["grids"]]).reshape(len(sil["keys"]), -1)
dref = pd.read_csv(R / "exp08_G6_roi_resnet50_prep2_image_logits.csv")

for cfg in ["G6_roi", "G6_bg", "G6_center", "G6_shuffle"]:
    seeds, pids, T, P, cls = frame(cfg)
    vi = cls.index("Type 6")
    v = [balanced_accuracy_score((T[seeds == s] == vi).astype(int),
                                 (P[seeds == s] == vi).astype(int)) for s in sorted(set(seeds))]
    v = np.array(v)
    ci = cluster_boot((T == vi).astype(int), (P == vi).astype(int), pids)
    res["vi_vs_rest"][cfg] = {"mean": round(float(v.mean()), 4),
                             "sd": round(float(v.std(ddof=1)), 4), "cluster_ci": ci}
    print(f"  VI-vs-rest {cfg:11s} {v.mean():.4f} ± {v.std(ddof=1):.4f}  CI {ci}", flush=True)

# VI-vs-rest from the silhouette alone
outv = []
for s in sorted(dref.seed.unique()):
    rows = dref[dref.seed == s][["fold", "key", "pid", "true"]].drop_duplicates()
    idx = np.array([kmap[k] for k in rows.key]); Xs = X16[idx]
    y = (rows["true"].values == "Type 6").astype(int)
    pr = np.zeros(len(y))
    for f in sorted(rows.fold.unique()):
        te = (rows.fold == f).values
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        m.fit(Xs[~te], y[~te]); pr[te] = m.predict_proba(Xs[te])[:, 1]
    dd = pd.DataFrame({"pid": rows.pid.values, "y": y, "p": pr}).groupby("pid").mean()
    outv.append(balanced_accuracy_score(dd.y.round().astype(int), (dd.p > 0.5).astype(int)))
outv = np.array(outv)
res["vi_vs_rest"]["mask_silhouette_16"] = {"mean": round(float(outv.mean()), 4),
                                          "sd": round(float(outv.std(ddof=1)), 4)}
print(f"  VI-vs-rest silhouette16 {outv.mean():.4f} ± {outv.std(ddof=1):.4f}", flush=True)

# ── (c) mean recall over the five non-VI classes — true labels vs permuted null ──
for cfg in ["G6_roi", "G6_bg", "G6_shuffle"]:
    seeds, pids, T, P, cls = frame(cfg)
    vi = cls.index("Type 6")
    o = []
    for s in sorted(set(seeds)):
        m = seeds == s
        o.append(np.mean([(P[m][T[m] == i] == i).mean() for i in range(6) if i != vi]))
    o = np.array(o)
    res["five_class_excl_VI"][cfg] = {"mean": round(float(o.mean()), 4),
                                     "sd": round(float(o.std(ddof=1)), 4)}
    print(f"  5cls-excl-VI {cfg:11s} {o.mean():.4f} ± {o.std(ddof=1):.4f}", flush=True)
res["five_class_excl_VI"]["note"] = ("Mean recall of the six-class model over the five classes other than VI. "
                                    "No five-class classifier was trained. A uniform predictor expects 1/6 = 0.1667.")

# ── (d) cluster bootstrap for the two headline paired differences ──────────
seeds, pids, T, P6, cls = frame("G6_roi")
_, _, _, Pbg, _ = frame("G6_bg")
res["paired_cluster_boot"] = {"G6_roi_minus_G6_bg": cluster_boot_diff(T, P6, Pbg, pids)}
print("  ROI − tibia-erased (cluster):", res["paired_cluster_boot"]["G6_roi_minus_G6_bg"])

json.dump(res, open(R / "exp15_cluster_bootstrap.json", "w"), indent=1)
print("\nwritten: results/exp15_cluster_bootstrap.json")
