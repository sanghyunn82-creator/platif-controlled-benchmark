#!/usr/bin/env python3
"""
P4-1 · Experiment 4: measure what remains **after the shortcut is removed.**

In experiment 1 the single variable "coronal CT availability" reached balanced acc 0.785 on the binary task.
The question that follows is: **what is left once that variable cannot be used?**

Method: keep only the 150 patients who have a CT (24 NTPF + 126 fracture). Within that subset the CT
      variable is **constant** and cannot act as a shortcut. What remains is the only baseline we can defend.

Secondary question: of the 58 NTPF patients, the 34 without a CT may have been labelled **without CT
          confirmation**, given the descriptor states fracture types were validated using CT. That would
          mean the labelling procedure differed by class. We compare the two subgroups directly.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
rng = np.random.default_rng(20260812)
BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")

FEATS_NO_CT = ["age", "lat_R", "lat_L", "lat_both", "n_images",
               "max_h", "max_w", "max_dim", "any_fullscan", "any_landscape", "med_mask_area"]


def evaluate(X, y, model, n_rep=20, seed=0):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
    ba = []
    for tr, te in cv.split(X, y):
        m = model(); m.fit(X[tr], y[tr])
        ba.append(balanced_accuracy_score(y[te], m.predict(X[te])))
    return float(np.mean(ba))


def block(df, y, title, feats, n_perm=200):
    print(f"\n{'='*74}\n■ {title}\n{'='*74}")
    cls, cnt = np.unique(y, return_counts=True)
    print(f"n={len(y)} · distribution {dict(zip(cls, cnt))} · chance {1/len(cls):.3f}\n")
    mk = lambda: make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
    mk_rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                          class_weight="balanced", random_state=0, n_jobs=-1)
    print(f"{'model':<20s} {'balanced acc':>13s} {'perm null':>10s} {'p':>8s}")
    print("-" * 74)
    out = []
    X = df[feats].values.astype(float)
    for name, model in [("logistic regression", mk), ("RF", mk_rf)]:
        ba = evaluate(X, y, model)
        null = np.array([evaluate(X, rng.permutation(y), model, n_rep=1, seed=i)
                         for i in range(n_perm)])
        p = float((null >= ba).sum() + 1) / (n_perm + 1)
        flag = " ***" if p < 0.01 else (" *" if p < 0.05 else "")
        print(f"{name:<10s} {ba:13.3f} {null.mean():10.3f} {p:8.4f}{flag}")
        out.append({"subset": title, "model": name, "n": len(y),
                    "balanced_acc": round(ba, 4), "null_mean": round(float(null.mean()), 4),
                    "p": round(p, 4)})
    return out


def main():
    df = pd.read_csv(BASE / "results/exp01_features.csv")
    df["y_bin"] = np.where(df.y == "NTPF", "NTPF", "Fracture")
    print(f"total {len(df)} patients · with CT {int(df.has_ct.sum())}")

    res = []
    # 1) everyone (reference; CT variable dropped)
    res += block(df, df.y_bin.values, "all 186 patients, CT variable dropped", FEATS_NO_CT)
    # 2) CT holders only — shortcut removed
    sub = df[df.has_ct == 1].reset_index(drop=True)
    res += block(sub, sub.y_bin.values, "150 CT holders only, shortcut removed", FEATS_NO_CT)
    # 3) same condition, seven-class
    res += block(sub, sub.y.values, "150 CT holders only, seven-class", FEATS_NO_CT)

    # 4) within NTPF: does CT availability split the group?
    print(f"\n{'='*74}\n■ NTPF, 58 patients: 24 with CT vs 34 without — did the labelling procedure differ?\n{'='*74}")
    n = df[df.y == "NTPF"]
    a, b = n[n.has_ct == 1], n[n.has_ct == 0]
    print(f"{'variable':<20s} {'with CT (24)':>13s} {'without CT (34)':>16s} {'test':>10s} {'p':>8s}")
    print("-" * 74)
    for c in ["age", "n_images", "max_dim", "any_fullscan", "med_mask_area"]:
        u, p = mannwhitneyu(a[c], b[c])
        print(f"{c:<16s} {a[c].mean():12.2f} {b[c].mean():13.2f} {'MWU':>10s} {p:8.4f}"
              + (" *" if p < 0.05 else ""))
    ct = pd.crosstab(n.has_ct, n.lat_R)
    if ct.shape == (2, 2):
        chi2, p, _, _ = chi2_contingency(ct)
        print(f"{'right-side share':<20s} {100*a.lat_R.mean():12.1f}% {100*b.lat_R.mean():15.1f}% "
              f"{'chi2':>10s} {p:8.4f}" + (" *" if p < 0.05 else ""))

    pd.DataFrame(res).to_csv(BASE / "results/exp04_ct_subset.csv", index=False)
    print(f"\nwritten: {BASE/'results/exp04_ct_subset.csv'}")
    print("\n■ how to read this")
    print("  The figure from the 150 CT holders is **the baseline with the shortcut removed**.")
    print("  A pixel model must exceed it before any anatomical contribution can be claimed.")


if __name__ == "__main__":
    main()
