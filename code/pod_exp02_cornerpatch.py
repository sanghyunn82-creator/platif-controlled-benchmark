#!/usr/bin/env python3
"""
P4-1 · Experiment 2: can the corner patches alone classify? — **run on the GPU host.**

Background: PlaTiF images carry burnt-in L/R markers in the corners, and the left/right distribution
      differs sharply by class (NTPF R39/L17 vs Type4 L9/R1). If reading the marker and the collimation
      traces alone recovers the label, a pixel model's accuracy cannot be read as "it read the fracture".

Design:
  - crop only the **four corner patches** (18% of each side) and resize to 32x32 -> 4x32x32 = 4096 features
  - comparator 1: the image **outside the bone mask** only (anatomy erased, background/markers/hardware kept), resized to 64x64
  - comparator 2: the whole image at 64x64 (an upper reference)
  - patient-level aggregation (per-image predictions averaged) -> patient-stratified 5-fold x 10 repeats
  - permuted-label null distribution, 100 draws

Note: split at patient level first, then use that patient's images. Shuffling per image would leak.
"""
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.ndimage import zoom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path("/workspace/platif_p4")
MATDIR = ROOT / "mat"
OUT = ROOT / "results"
CORNER_FRAC = 0.18
CS = 32   # corner patch resize
FS = 64   # whole/background image resize
rng = np.random.default_rng(20260812)


def shrink(a, n):
    a = np.asarray(a, dtype=np.float32)
    if a.shape[0] < n or a.shape[1] < n:
        a = np.pad(a, ((0, max(0, n - a.shape[0])), (0, max(0, n - a.shape[1]))))
    f = (n / a.shape[0], n / a.shape[1])
    return zoom(a, f, order=1)[:n, :n]


def extract():
    """Per-patient .mat -> three feature vectors. Arrays are discarded immediately."""
    rows, skipped = [], []
    files = sorted(MATDIR.rglob("*.mat"))
    for k, p in enumerate(files, 1):
        pid = int(re.search(r"(\d+)", p.stem).group(1))
        md = sio.loadmat(p, struct_as_record=False, squeeze_me=True)
        top = next(v for kk, v in md.items() if not kk.startswith("__"))
        for f in top._fieldnames:
            if not f.startswith("im"):
                continue
            im = getattr(top, f)
            g = np.asarray(im.OriginalImage, dtype=np.float32)
            bw = np.asarray(im.BW).astype(bool)
            # Data defect: 2 of 421 images (ID 47 im1, ID 81 im1) have BW/maskedImage dimensions that
            # differ from OriginalImage — the mask cannot be applied. They are excluded and recorded.
            if bw.shape != g.shape:
                skipped.append({"pid": pid, "im": f,
                                "orig": list(g.shape), "bw": list(bw.shape)})
                del g, bw, im
                continue
            h, w = g.shape
            ch, cw = max(8, int(h * CORNER_FRAC)), max(8, int(w * CORNER_FRAC))
            corners = np.concatenate([
                shrink(g[:ch, :cw], CS).ravel(), shrink(g[:ch, -cw:], CS).ravel(),
                shrink(g[-ch:, :cw], CS).ravel(), shrink(g[-ch:, -cw:], CS).ravel()])
            outside = g.copy()
            outside[bw] = 0.0                      # erase the bone
            rows.append({
                "pid": pid, "im": f, "label": int(np.asarray(im.label).ravel()[0]),
                "corner": corners.astype(np.float32),
                "outside": shrink(outside, FS).ravel().astype(np.float32),
                "full": shrink(g, FS).ravel().astype(np.float32),
            })
            del g, bw, outside, im
        del md, top
        if k % 25 == 0:
            print(f"  extracted {k}/{len(files)}", flush=True)
    if skipped:
        print(f"  images skipped for mask/image size mismatch: {len(skipped)}", flush=True)
        for s in skipped:
            print(f"     ID {s['pid']} {s['im']} orig={s['orig']} bw={s['bw']}", flush=True)
        json.dump(skipped, open(OUT / "exp02_skipped.json", "w"))
    return rows


def patient_matrix(rows, key, y_by_pid):
    """Aggregate per-image features to the patient (needed for patient-level splits)."""
    df = {}
    for r in rows:
        df.setdefault(r["pid"], []).append(r[key])
    pids = sorted(df)
    X = np.stack([np.mean(df[p], axis=0) for p in pids])
    y = np.array([y_by_pid[p] for p in pids])
    return np.array(pids), X, y


def evaluate(X, y, n_rep=10, seed=0, C=0.05):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
    ba, f1 = [], []
    for tr, te in cv.split(X, y):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, C=C, class_weight="balanced"))
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        ba.append(balanced_accuracy_score(y[te], p))
        f1.append(f1_score(y[te], p, average="macro"))
    return float(np.mean(ba)), float(np.mean(f1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_excel(ROOT / "data/Tibial Plateau Fracture Metadata.xlsx").set_index("Patient ID")
    SCH = "Fracture Type of  Schatzker Classification"
    y_full = {int(p): str(meta.loc[p, SCH]).replace("Normal", "NTPF") for p in meta.index}
    y_bin = {p: ("NTPF" if v == "NTPF" else "Fracture") for p, v in y_full.items()}

    print("feature extraction start (each patient opened once)", flush=True)
    rows = extract()
    print(f"extracted {len(rows)} images", flush=True)

    res = []
    for tname, ymap in [("Binary: NTPF vs fracture", y_bin), ("Seven-class", y_full)]:
        print(f"\n{'='*72}\n■ {tname}\n{'='*72}", flush=True)
        nc = len(set(ymap.values()))
        print(f"{'input':<28s} {'balanced acc':>13s} {'macro-F1':>10s} {'perm null':>10s} {'p':>8s}")
        print("-" * 72)
        for key, desc in [("corner", "four corner patches only (32x32x4)"),
                          ("outside", "background with bone erased (64x64)"),
                          ("full", "whole image (64x64) [reference]")]:
            pids, X, y = patient_matrix(rows, key, ymap)
            ba, f1 = evaluate(X, y)
            null = []
            for i in range(100):
                ys = rng.permutation(y)
                nb, _ = evaluate(X, ys, n_rep=1, seed=i)
                null.append(nb)
            null = np.array(null)
            p = float((null >= ba).sum() + 1) / (len(null) + 1)
            flag = " ***" if p < 0.01 else (" *" if p < 0.05 else "")
            print(f"{desc:<28s} {ba:13.3f} {f1:10.3f} {null.mean():10.3f} {p:8.4f}{flag}", flush=True)
            res.append({"target": tname, "input": desc, "n_classes": nc,
                        "chance": round(1 / nc, 3), "balanced_acc": round(ba, 4),
                        "macro_f1": round(f1, 4), "null_mean": round(float(null.mean()), 4),
                        "null_sd": round(float(null.std()), 4), "p": round(p, 4)})

    pd.DataFrame(res).to_csv(OUT / "exp02_cornerpatch.csv", index=False)
    print(f"\nwritten: {OUT/'exp02_cornerpatch.csv'}", flush=True)


if __name__ == "__main__":
    main()
