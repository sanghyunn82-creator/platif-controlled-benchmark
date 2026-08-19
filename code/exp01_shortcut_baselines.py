#!/usr/bin/env python3
"""
P4-1 · Experiment 1: shortcut baselines — **not a single pixel is used.**

Hypothesis: PlaTiF carries acquisition and administrative by-products that correlate with the label
      but have nothing to do with anatomy, so metadata alone should beat chance by a wide margin.
      If so, we cannot tell how much of a pixel model's accuracy comes from fracture morphology.

Features used (all non-anatomical):
  - laterality (R/L/bilateral)   <- also burnt into the image corners
  - age                          <- administrative
  - images per patient           <- number of acquisitions
  - coronal CT availability      <- a by-product of the care pathway; strongly class-correlated in the full scan
  - maximum resolution / whole-leg flag <- acquisition protocol

Evaluation: rows are patients, so there is no leakage. Stratified 5-fold x 20 repeats.
Control: permuted-label null distribution, 200 draws — chance is estimated from the data itself.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
rng = np.random.default_rng(20260812)

BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
SCAN = BASE / "results/eda05_fullscan.jsonl"
XLSX = BASE / "data/platif/Tibial Plateau Fracture Metadata.xlsx"
SCH = "Fracture Type of  Schatzker Classification"
LAT = "Fracture of the Right (R) or Left (L) Tibia"
NAME = {1: "Type 1", 2: "Type 2", 3: "Type 3", 4: "Type 4",
        5: "Type 5", 6: "Type 6", 7: "NTPF"}


def build():
    recs = [json.loads(l) for l in open(SCAN)]
    meta = pd.read_excel(XLSX).set_index("Patient ID")
    rows = []
    for r in recs:
        pid = r["pid"]
        ims = r["images"]
        # the xlsx labels (the ones matching descriptor Table 1) are treated as of record.
        lab = str(meta.loc[pid, SCH]).replace("Normal", "NTPF")
        side = str(meta.loc[pid, LAT])
        rows.append({
            "pid": pid,
            "y": lab,
            "age": float(meta.loc[pid, "Age"]),
            "lat_R": int(side == "R"), "lat_L": int(side == "L"), "lat_both": int(side == "R and L"),
            "n_images": r["n_images"],
            "has_ct": int(r["has_ct"]),
            "max_h": max(i["h"] for i in ims),
            "max_w": max(i["w"] for i in ims),
            "max_dim": max(max(i["h"], i["w"]) for i in ims),
            "any_fullscan": int(any(i["h"] > 4000 or i["w"] > 3000 for i in ims)),
            "any_landscape": int(any(i["aspect"] < 1.0 for i in ims)),
            "med_mask_area": float(np.median([i["mask_area_frac"] for i in ims])),
        })
    return pd.DataFrame(rows)


FEATSETS = {
    "laterality only":        ["lat_R", "lat_L", "lat_both"],
    "age only":               ["age"],
    "image count only":       ["n_images"],
    "CT availability only":   ["has_ct"],
    "field of view only":     ["max_dim", "any_fullscan", "any_landscape"],
    "all (non-anatomical)":   ["age", "lat_R", "lat_L", "lat_both", "n_images", "has_ct",
                           "max_h", "max_w", "max_dim", "any_fullscan", "any_landscape",
                           "med_mask_area"],
}


def evaluate(X, y, model, n_rep=20, seed=0):
    """Stratified 5-fold x n_rep repeats. Rows are patients, so there is no leakage."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
    ba, f1 = [], []
    for tr, te in cv.split(X, y):
        m = model()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        ba.append(balanced_accuracy_score(y[te], p))
        f1.append(f1_score(y[te], p, average="macro"))
    return float(np.mean(ba)), float(np.mean(f1))


def run(df, target, title):
    print("\n" + "=" * 78)
    print(f"■ {title}")
    print("=" * 78)
    y = df[target].values
    classes, counts = np.unique(y, return_counts=True)
    print(f"classes {len(classes)} · n={len(y)} · distribution {dict(zip(classes, counts))}")
    print(f"chance level (balanced acc) = {1/len(classes):.3f}\n")

    mk = lambda: make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
    mk_rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                          class_weight="balanced", random_state=0, n_jobs=-1)

    out = []
    print(f"{'feature set':<24s} {'model':<20s} {'balanced acc':>13s} {'macro-F1':>10s} "
          f"{'perm null mean':>15s} {'p':>8s}")
    print("-" * 78)
    for fname, feats in FEATSETS.items():
        X = df[feats].values.astype(float)
        for mname, model in [("logistic regression", mk), ("RF", mk_rf)]:
            ba, f1 = evaluate(X, y, model)
            # permuted-label null distribution
            null = []
            for i in range(200):
                ys = rng.permutation(y)
                nb, _ = evaluate(X, ys, model, n_rep=1, seed=i)
                null.append(nb)
            null = np.array(null)
            p = float((null >= ba).sum() + 1) / (len(null) + 1)
            flag = " ***" if p < 0.01 else (" *" if p < 0.05 else "")
            print(f"{fname:<20s} {mname:<8s} {ba:13.3f} {f1:10.3f} "
                  f"{null.mean():13.3f} {p:8.4f}{flag}")
            out.append({"target": title, "features": fname, "model": mname,
                        "balanced_acc": round(ba, 4), "macro_f1": round(f1, 4),
                        "null_mean": round(float(null.mean()), 4),
                        "null_sd": round(float(null.std()), 4), "p": round(p, 4)})
    return out


def main():
    df = build()
    df["y_bin"] = np.where(df.y == "NTPF", "NTPF", "Fracture")
    print(f"patients {len(df)} · features {len(FEATSETS['all (non-anatomical)'])} (no pixels used)")

    res = []
    res += run(df, "y_bin", "Binary: NTPF vs tibial plateau fracture")
    res += run(df, "y", "Seven-class: Schatzker Type 1-6 plus NTPF")

    o = BASE / "results/exp01_shortcut_baselines.csv"
    pd.DataFrame(res).to_csv(o, index=False)
    df.to_csv(BASE / "results/exp01_features.csv", index=False)
    print(f"\nwritten: {o}")

    print("\n" + "=" * 78)
    print("■ interpretation")
    print("=" * 78)
    best = max((r for r in res if r["target"].startswith("Binary")), key=lambda r: r["balanced_acc"])
    print(f"best binary: {best['features']} / {best['model']} -> balanced acc {best['balanced_acc']:.3f} "
          f"(permuted null {best['null_mean']:.3f}, p={best['p']})")
    best7 = max((r for r in res if r["target"].startswith("Seven")), key=lambda r: r["balanced_acc"])
    print(f"best seven-class: {best7['features']} / {best7['model']} -> balanced acc {best7['balanced_acc']:.3f} "
          f"(permuted null {best7['null_mean']:.3f}, p={best7['p']})")
    print("\nIf this much is achievable without any pixels, a pixel model's accuracy cannot be read")
    print("as 'it read the fracture'. Only the margin above this baseline is an anatomical contribution.")


if __name__ == "__main__":
    main()
