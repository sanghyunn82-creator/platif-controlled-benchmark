#!/usr/bin/env python3
"""
P4-1 · Experiment 10 — **re-measure the baselines on the exact split the CNN used.** CPU, GPU host.

Problem: the CNN used 5-fold x 5-seed StratifiedGroupKFold while the baselines used 5-fold x 20
      repeats of RepeatedStratifiedKFold. **Different schemes cannot be compared directly**
      (audit point 16). For the manuscript they must be measured on the same partitions.

Method: the logit CSVs written by exp08 carry seed, fold and pid, so the **patient split the CNN
      actually used** can be reconstructed and the baselines fitted on it, giving seed-paired comparisons.

Three feature sets: non-imaging metadata / mask geometry / both.
Two models: logistic regression / random forest. **The primary comparator is pre-specified as
              metadata + logistic regression**; the rest are reported for completeness only
              (audit point 11 — no picking the maximum).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/workspace/platif_p4")
R = ROOT / "results"
SCH = "Fracture Type of  Schatzker Classification"
LAT = "Fracture of the Right (R) or Left (L) Tibia"
LBL = {1: "Type 1", 2: "Type 2", 3: "Type 3", 4: "Type 4", 5: "Type 5", 6: "Type 6", 7: "NTPF"}

META = ["age", "lat_R", "lat_L", "lat_both", "n_images", "has_ct",
        "max_h", "max_w", "max_dim", "any_fullscan", "any_landscape", "med_mask_area"]
GEOM = ["mask_area_frac", "bbox_h_rel", "bbox_w_rel", "bbox_aspect",
        "bbox_top_rel", "bbox_left_rel", "vext", "fill_ratio"]


def build_features():
    idx = json.load(open(R / "prep2_index.json"))
    scan = [json.loads(l) for l in open(R / "eda05_fullscan.jsonl")]
    meta = pd.read_excel(ROOT / "data/Tibial Plateau Fracture Metadata.xlsx").set_index("Patient ID")
    geo = {}
    for r in scan:
        rows = []
        for im in r["images"]:
            br, bc = im.get("mask_bbox_rows"), im.get("mask_bbox_cols")
            if not br or not bc:
                continue
            h, w = im["h"], im["w"]
            bh, bw_ = br[1] - br[0] + 1, bc[1] - bc[0] + 1
            rows.append({"mask_area_frac": im["mask_area_frac"], "bbox_h_rel": bh / h,
                         "bbox_w_rel": bw_ / w, "bbox_aspect": bh / max(bw_, 1),
                         "bbox_top_rel": br[0] / h, "bbox_left_rel": bc[0] / w,
                         "vext": im["mask_vertical_extent"],
                         "fill_ratio": im["mask_area_frac"] * h * w / max(bh * bw_, 1)})
        if rows:
            geo[r["pid"]] = pd.DataFrame(rows).mean().to_dict()
    byp = {}
    for r in idx:
        byp.setdefault(r["pid"], []).append(r)
    out = []
    for pid, ims in byp.items():
        side = str(meta.loc[pid, LAT])
        d = {"pid": pid,
             "age": float(meta.loc[pid, "Age"]),
             "lat_R": int(side == "R"), "lat_L": int(side == "L"), "lat_both": int(side == "R and L"),
             "n_images": len(ims), "has_ct": int(any(True for _ in [1]) and pid in geo),
             "max_h": max(i["h"] for i in ims), "max_w": max(i["w"] for i in ims),
             "max_dim": max(max(i["h"], i["w"]) for i in ims),
             "any_fullscan": int(any(i["h"] > 4000 or i["w"] > 3000 for i in ims)),
             "any_landscape": int(any(i["h"] / i["w"] < 1.0 for i in ims)),
             "pat_label": str(meta.loc[pid, SCH]).replace("Normal", "NTPF")}
        d.update(geo.get(pid, {k: np.nan for k in GEOM}))
        d["med_mask_area"] = d.get("mask_area_frac", np.nan)
        out.append(d)
    df = pd.DataFrame(out)
    # has_ct comes straight from the full-scan results
    ct = {r["pid"]: int(r["has_ct"]) for r in scan}
    df["has_ct"] = df.pid.map(ct)
    return df.dropna(subset=GEOM).reset_index(drop=True)


def splits_from_cnn(cfg):
    """Reconstruct the (seed, fold) -> test patient set that the CNN actually used."""
    f = R / f"exp08_{cfg}_resnet50_prep2_image_logits.csv"
    d = pd.read_csv(f)
    out = {}
    for (s, k), g in d.groupby(["seed", "fold"]):
        out.setdefault(s, {})[k] = sorted(set(g.pid.tolist()))
    return out


def run(df, target_col, classes, cfg, name):
    sp = splits_from_cnn(cfg)
    mk = {"logistic regression": lambda: make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=2000, class_weight="balanced")),
          "RF": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                              class_weight="balanced", random_state=0, n_jobs=-1)}
    sets = {"non-imaging metadata": META, "mask geometry": GEOM, "both": META + GEOM}
    print(f"\n{'='*84}\n■ {name} — same split as the CNN\n{'='*84}")
    print(f"{'feature set':<22s} {'model':<20s} {'seed mean':>9s} {'seed SD':>8s}  per-seed values")
    print("-" * 84)
    res = []
    for fname, feats in sets.items():
        for mname, mkf in mk.items():
            sv = []
            for seed, folds in sorted(sp.items()):
                T, P = [], []
                for k, te_pids in sorted(folds.items()):
                    te = df[df.pid.isin(te_pids)]
                    tr = df[~df.pid.isin(te_pids)]
                    if len(te) == 0 or tr[target_col].nunique() < 2:
                        continue
                    m = mkf()
                    m.fit(tr[feats].values.astype(float),
                          [classes.index(v) for v in tr[target_col]])
                    P += list(m.predict(te[feats].values.astype(float)))
                    T += [classes.index(v) for v in te[target_col]]
                sv.append(balanced_accuracy_score(T, P))
            sv = np.array(sv)
            star = " *pre-specified primary comparator" if (fname == "non-imaging metadata" and mname == "logistic regression") else ""
            print(f"{fname:<12s} {mname:<8s} {sv.mean():9.4f} {sv.std(ddof=1):8.4f}  "
                  f"{[round(x,3) for x in sv]}{star}")
            res.append({"task": name, "features": fname, "model": mname,
                        "mean": round(float(sv.mean()), 4), "sd": round(float(sv.std(ddof=1)), 4),
                        "seed_vals": [round(float(x), 4) for x in sv],
                        "prespecified": bool(fname == "non-imaging metadata" and mname == "logistic regression")})
    return res


def main():
    df = build_features()
    print(f"patients {len(df)} · features: {len(META)} metadata + {len(GEOM)} geometry")
    res = []
    # binary task: all 186 patients
    d2 = df.copy()
    d2["y"] = np.where(d2.pat_label == "NTPF", "NTPF", "Fracture")
    res += run(d2, "y", ["Fracture", "NTPF"], "C2_roi", "Binary: NTPF vs fracture")
    # six-class task: 128 fracture patients
    d6 = df[df.pat_label != "NTPF"].reset_index(drop=True)
    CLS6 = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6"]
    d6 = d6.rename(columns={"pat_label": "y"})
    res += run(d6, "y", CLS6, "G6_roi", "Six-class (fracture patients)")
    json.dump(res, open(R / "exp10_samesplit_baseline.json", "w"), ensure_ascii=False, indent=2)
    print(f"\nwritten: {R/'exp10_samesplit_baseline.json'}")


if __name__ == "__main__":
    main()
