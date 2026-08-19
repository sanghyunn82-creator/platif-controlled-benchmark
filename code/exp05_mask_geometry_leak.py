#!/usr/bin/env python3
"""
P4-1 · Experiment 5: **does the roi crop itself leak the label?** A self-check.

Motivation: in exp03 condition C (binary, roi) the first fold reached 0.877, well above the metadata
      baseline of 0.798. That is encouraging, but the roi crop is taken from the bbox of the `BW` mask
      shipped with the dataset, and **that mask was drawn by an annotator who had seen the fracture.**
      If mask geometry differs systematically between fractured and non-fractured knees, **the crop
      geometry alone could recover the label**, and part of the 0.877 would be annotator information.

Method: no pixels at all, **mask geometry only** — area fraction, bbox height/width/aspect ratio,
      relative position in the frame, vertical occupancy. Averaged per patient, stratified 5-fold x 20 repeats, 200 permuted nulls.

If this clearly exceeds chance, the roi results need a caveat and a comparison against a crop that
does not depend on the mask (a fixed centre crop, for instance).
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
rng = np.random.default_rng(20260812)
BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
SCH = "Fracture Type of  Schatzker Classification"


def build():
    recs = [json.loads(l) for l in open(BASE / "results/eda05_fullscan.jsonl")]
    meta = pd.read_excel(BASE / "data/platif/Tibial Plateau Fracture Metadata.xlsx").set_index("Patient ID")
    rows = []
    for r in recs:
        for im in r["images"]:
            br, bc = im.get("mask_bbox_rows"), im.get("mask_bbox_cols")
            if not br or not bc:
                continue
            h, w = im["h"], im["w"]
            bh, bw_ = br[1] - br[0] + 1, bc[1] - bc[0] + 1
            rows.append({
                "pid": r["pid"],
                "mask_area_frac": im["mask_area_frac"],
                "bbox_h_rel": bh / h,               # bbox height / image height
                "bbox_w_rel": bw_ / w,              # bbox width / image width
                "bbox_aspect": bh / max(bw_, 1),    # bbox aspect ratio
                "bbox_top_rel": br[0] / h,          # relative position in the frame
                "bbox_left_rel": bc[0] / w,
                "vext": im["mask_vertical_extent"],
                "fill_ratio": im["mask_area_frac"] * h * w / max(bh * bw_, 1),  # fill relative to the bbox
            })
    df = pd.DataFrame(rows).groupby("pid").mean().reset_index()
    df["y"] = [str(meta.loc[p, SCH]).replace("Normal", "NTPF") for p in df.pid]
    df["y_bin"] = np.where(df.y == "NTPF", "NTPF", "Fracture")
    return df


FEATS = ["mask_area_frac", "bbox_h_rel", "bbox_w_rel", "bbox_aspect",
         "bbox_top_rel", "bbox_left_rel", "vext", "fill_ratio"]


def evaluate(X, y, model, n_rep=20, seed=0):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
    return float(np.mean([balanced_accuracy_score(y[te], model().fit(X[tr], y[tr]).predict(X[te]))
                          for tr, te in cv.split(X, y)]))


def block(df, y, title, n_perm=200):
    print(f"\n{'='*74}\n■ {title}\n{'='*74}")
    cls, cnt = np.unique(y, return_counts=True)
    print(f"n={len(y)} · distribution {dict(zip(cls, cnt))} · chance {1/len(cls):.3f}\n")
    mk = lambda: make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
    mk_rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                          class_weight="balanced", random_state=0, n_jobs=-1)
    X = df[FEATS].values.astype(float)
    out = []
    print(f"{'model':<20s} {'balanced acc':>13s} {'perm null':>10s} {'p':>8s}")
    print("-" * 74)
    for name, model in [("logistic regression", mk), ("RF", mk_rf)]:
        ba = evaluate(X, y, model)
        null = np.array([evaluate(X, rng.permutation(y), model, n_rep=1, seed=i) for i in range(n_perm)])
        p = float((null >= ba).sum() + 1) / (n_perm + 1)
        flag = " ***" if p < 0.01 else (" *" if p < 0.05 else "")
        print(f"{name:<10s} {ba:13.3f} {null.mean():10.3f} {p:8.4f}{flag}")
        out.append({"task": title, "model": name, "balanced_acc": round(ba, 4),
                    "null_mean": round(float(null.mean()), 4), "p": round(p, 4)})
    return out


def main():
    df = build()
    print(f"patients {len(df)} · mask-geometry features {len(FEATS)} (no pixels used)")
    res = []
    res += block(df, df.y_bin.values, "Binary NTPF vs fracture — mask geometry only")
    res += block(df, df.y.values, "Seven-class — mask geometry only")
    pd.DataFrame(res).to_csv(BASE / "results/exp05_mask_geometry.csv", index=False)

    print(f"\n{'='*74}\n■ mask geometry by class (means)\n{'='*74}")
    print(df.groupby("y")[FEATS].mean().round(4).to_string())
    print(f"\nwritten: {BASE/'results/exp05_mask_geometry.csv'}")
    print("\n■ how to read this")
    print("  If this clearly exceeds chance, the roi results (exp03 A/C/E) need a caveat.")
    print("  The mask was drawn by an annotator who saw the fracture, so crop geometry may be annotation, not anatomy.")


if __name__ == "__main__":
    main()
