#!/usr/bin/env python3
"""
P4-1 · Experiment 7: **non-anatomical baseline for six-class typing among 128 fracture patients** — the direct comparator for exp06 (CNN).

exp04 was based on the seven-class task (NTPF included). For the six-class task that exp06 treats as
primary (NTPF excluded), "how far can we get without looking at a pixel" must be measured on the same
scale before the CNN contribution can be isolated.
Three baselines are produced.
  1. non-imaging metadata (age, laterality, image count, CT availability, field of view)
  2. mask geometry (area, bbox, position)
  3. both
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
rng = np.random.default_rng(20260812)
BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
SCH = "Fracture Type of  Schatzker Classification"

META = ["age", "lat_R", "lat_L", "lat_both", "n_images", "has_ct",
        "max_h", "max_w", "max_dim", "any_fullscan", "any_landscape", "med_mask_area"]
GEOM = ["mask_area_frac", "bbox_h_rel", "bbox_w_rel", "bbox_aspect",
        "bbox_top_rel", "bbox_left_rel", "vext", "fill_ratio"]


def build():
    feat = pd.read_csv(BASE / "results/exp01_features.csv")
    recs = [json.loads(l) for l in open(BASE / "results/eda05_fullscan.jsonl")]
    rows = []
    for r in recs:
        for im in r["images"]:
            br, bc = im.get("mask_bbox_rows"), im.get("mask_bbox_cols")
            if not br or not bc:
                continue
            h, w = im["h"], im["w"]
            bh, bw_ = br[1] - br[0] + 1, bc[1] - bc[0] + 1
            rows.append({"pid": r["pid"], "mask_area_frac": im["mask_area_frac"],
                         "bbox_h_rel": bh / h, "bbox_w_rel": bw_ / w,
                         "bbox_aspect": bh / max(bw_, 1), "bbox_top_rel": br[0] / h,
                         "bbox_left_rel": bc[0] / w, "vext": im["mask_vertical_extent"],
                         "fill_ratio": im["mask_area_frac"] * h * w / max(bh * bw_, 1)})
    geom = pd.DataFrame(rows).groupby("pid").mean().reset_index()
    df = feat.merge(geom, on="pid", how="inner")
    return df[df.y != "NTPF"].reset_index(drop=True)   # fracture patients only


def evaluate(X, y, model, n_rep=20, seed=0):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
    ba, f1 = [], []
    for tr, te in cv.split(X, y):
        m = model(); m.fit(X[tr], y[tr]); p = m.predict(X[te])
        ba.append(balanced_accuracy_score(y[te], p))
        f1.append(f1_score(y[te], p, average="macro"))
    return float(np.mean(ba)), float(np.mean(f1))


def main():
    df = build()
    y = df.y.values
    cls, cnt = np.unique(y, return_counts=True)
    print(f"fracture patients {len(df)} · six-class · chance {1/len(cls):.3f}")
    print(f"distribution {dict(zip(cls, cnt))}\n")

    mk = lambda: make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
    mk_rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                          class_weight="balanced", random_state=0, n_jobs=-1)
    print(f"{'feature set':<24s} {'model':<10s} {'balanced acc':>13s} {'macro-F1':>10s} "
          f"{'perm null':>10s} {'p':>8s}")
    print("-" * 80)
    out = []
    for fname, feats in [("metadata (non-anatomical)", META), ("mask geometry", GEOM),
                         ("both", META + GEOM)]:
        X = df[feats].values.astype(float)
        for mname, model in [("logistic regression", mk), ("RF", mk_rf)]:
            ba, f1 = evaluate(X, y, model)
            null = np.array([evaluate(X, rng.permutation(y), model, n_rep=1, seed=i)[0]
                             for i in range(200)])
            p = float((null >= ba).sum() + 1) / 201
            flag = " ***" if p < 0.01 else (" *" if p < 0.05 else "")
            print(f"{fname:<24s} {mname:<8s} {ba:13.3f} {f1:10.3f} {null.mean():10.3f} {p:8.4f}{flag}")
            out.append({"features": fname, "model": mname, "n": len(y),
                        "balanced_acc": round(ba, 4), "macro_f1": round(f1, 4),
                        "null_mean": round(float(null.mean()), 4), "p": round(p, 4)})
    pd.DataFrame(out).to_csv(BASE / "results/exp07_baseline6.csv", index=False)
    best = max(out, key=lambda r: r["balanced_acc"])
    print(f"\nbest baseline: {best['features']} / {best['model']} -> {best['balanced_acc']:.3f} (p={best['p']})")
    print("-> the exp06 CNN must exceed this before any anatomical contribution can be claimed.")
    print(f"\nwritten: {BASE/'results/exp07_baseline6.csv'}")


if __name__ == "__main__":
    main()
