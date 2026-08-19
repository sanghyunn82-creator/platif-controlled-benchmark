#!/usr/bin/env python3
"""P4-1 · exp14 — fill in the macro-F1 and patient-level bootstrap CIs missing for the baselines. CPU, local.

Review point (R2, lens C): Methods promises macro-F1 and patient-level bootstrap intervals, but Table 2
fills them in only for the seven CNN configurations; the pre-specified baseline, mask-geometry and
silhouette rows are empty. The two paired differences the paper rests on (+0.019, +0.180) also lack patient-resampling intervals.

Here the baseline predictions are rebuilt on the **same folds with the same models** as exp10, and the
**same bootstrap procedure** as exp09 is applied (patient-level, class-stratified, 20,000 resamples).
Output: results/exp14_baseline_extras.json
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import zoom as ndzoom

BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
R = BASE / "results"
SCH = "Fracture Type of  Schatzker Classification"
LAT = "Fracture of the Right (R) or Left (L) Tibia"
NB, rng = 20000, np.random.default_rng(0)
META = ["age", "lat_R", "lat_L", "lat_both", "n_images", "has_ct",
        "max_h", "max_w", "max_dim", "any_fullscan", "any_landscape", "med_mask_area"]
GEOM = ["mask_area_frac", "bbox_h_rel", "bbox_w_rel", "bbox_aspect",
        "bbox_top_rel", "bbox_left_rel", "vext", "fill_ratio"]


def build_features():
    idx = json.load(open(R / "prep2_index.json"))
    scan = [json.loads(l) for l in open(R / "eda05_fullscan.jsonl")]
    meta = pd.read_excel(BASE / "data/platif/Tibial Plateau Fracture Metadata.xlsx").set_index("Patient ID")
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
        d = {"pid": pid, "age": float(meta.loc[pid, "Age"]),
             "lat_R": int(side == "R"), "lat_L": int(side == "L"), "lat_both": int(side == "R and L"),
             "n_images": len(ims),
             "max_h": max(i["h"] for i in ims), "max_w": max(i["w"] for i in ims),
             "max_dim": max(max(i["h"], i["w"]) for i in ims),
             "any_fullscan": int(any(i["h"] > 4000 or i["w"] > 3000 for i in ims)),
             "any_landscape": int(any(i["h"] / i["w"] < 1.0 for i in ims)),
             "pat_label": str(meta.loc[pid, SCH]).replace("Normal", "NTPF")}
        d.update(geo.get(pid, {k: np.nan for k in GEOM}))
        d["med_mask_area"] = d.get("mask_area_frac", np.nan)
        out.append(d)
    df = pd.DataFrame(out)
    df["has_ct"] = df.pid.map({r["pid"]: int(r["has_ct"]) for r in scan})
    return df.dropna(subset=GEOM).reset_index(drop=True)


def cnn_frame(cfg):
    d = pd.read_csv(R / f"exp08_{cfg}_resnet50_prep2_image_logits.csv")
    cols = [c for c in d.columns if c.startswith("logit_")]
    classes = [c[6:] for c in cols]
    g = d.groupby(["seed", "fold", "pid"])
    out = g[cols].mean().reset_index()
    out["pred"] = out[cols].values.argmax(1)
    out["true"] = [classes.index(v) for v in g["true"].first().values]
    return out[["seed", "fold", "pid", "true", "pred"]], classes


def boot_ci(true, pred, n=NB):
    idx_by_c = {c: np.where(true == c)[0] for c in np.unique(true)}
    v = []
    for _ in range(n):
        take = np.concatenate([rng.choice(x, len(x), replace=True) for x in idx_by_c.values()])
        v.append(balanced_accuracy_score(true[take], pred[take]))
    v = np.array(v)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_diff(true, pa, pb, n=NB):
    """Balanced-accuracy difference between two predictions on the same patient sample (paired bootstrap)."""
    idx_by_c = {c: np.where(true == c)[0] for c in np.unique(true)}
    v = []
    for _ in range(n):
        take = np.concatenate([rng.choice(x, len(x), replace=True) for x in idx_by_c.values()])
        v.append(balanced_accuracy_score(true[take], pa[take])
                 - balanced_accuracy_score(true[take], pb[take]))
    v = np.array(v)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def fit_baseline(df, ycol, classes, frame, feats):
    """Rebuild per-patient baseline predictions on the exact (seed, fold) split the CNN used."""
    pred = np.full(len(frame), -1)
    for (s, k), g in frame.groupby(["seed", "fold"]):
        te_pids = set(g.pid)
        te, tr = df[df.pid.isin(te_pids)], df[~df.pid.isin(te_pids)]
        if len(te) == 0 or tr[ycol].nunique() < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        m.fit(tr[feats].values.astype(float), [classes.index(v) for v in tr[ycol]])
        p = dict(zip(te.pid, m.predict(te[feats].values.astype(float))))
        sel = frame.index[(frame.seed == s) & (frame.fold == k)]
        pred[sel] = [p.get(q, -1) for q in frame.loc[sel, "pid"]]
    return pred


def silhouette_pred(frame, classes, grid=16):
    z = np.load(R / "mask_silhouettes.npz", allow_pickle=True)
    kmap = {k: i for i, k in enumerate(z["keys"])}
    X = np.stack([ndzoom(g.astype(np.float32), (grid / 48, grid / 48), order=1)
                  for g in z["grids"]]).reshape(len(z["keys"]), -1)
    d = pd.read_csv(R / "exp08_G6_roi_resnet50_prep2_image_logits.csv")
    pred = np.full(len(frame), -1)
    for s in sorted(frame.seed.unique()):
        rows = d[d.seed == s][["fold", "key", "pid", "true"]].drop_duplicates()
        idx = np.array([kmap[k] for k in rows.key]); Xs = X[idx]
        y = np.array([classes.index(v) for v in rows.true])
        proba = np.zeros((len(y), len(classes)))
        for f in sorted(rows.fold.unique()):
            te = (rows.fold == f).values
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            m.fit(Xs[~te], y[~te]); proba[te] = m.predict_proba(Xs[te])
        pr = pd.DataFrame(proba); pr["pid"] = rows.pid.values
        pp = pr.groupby("pid").mean(); pp["pred"] = pp.values.argmax(1)
        mp = pp["pred"].to_dict()
        sel = frame.index[frame.seed == s]
        pred[sel] = [mp.get(q, -1) for q in frame.loc[sel, "pid"]]
    return pred


df = build_features()
res = {}
for task, cfg, ycol, classes in [
        ("binary", "C2_roi", "y2", ["Fracture", "NTPF"]),
        ("sixclass", "G6_roi", "y6", ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6"])]:
    frame, cls = cnn_frame(cfg)
    dd = df.copy()
    if task == "binary":
        dd["y2"] = np.where(dd.pat_label == "NTPF", "NTPF", "Fracture")
    else:
        keep = set(frame.pid)
        dd = dd[dd.pid.isin(keep)].copy()
        lab = frame.groupby("pid")["true"].first().to_dict()
        dd["y6"] = [cls[lab[p]] for p in dd.pid]
    true = frame["true"].values
    img = frame["pred"].values
    base = fit_baseline(dd, ycol, cls, frame, META)
    geom = fit_baseline(dd, ycol, cls, frame, GEOM)
    ent = {"n_patient_evaluations": int(len(frame)),
           "image_bacc": float(balanced_accuracy_score(true, img)),
           "baseline_bacc": float(balanced_accuracy_score(true, base)),
           "baseline_macro_f1_by_seed": [], "geom_macro_f1_by_seed": []}
    for s in sorted(frame.seed.unique()):
        m = (frame.seed == s).values
        ent["baseline_macro_f1_by_seed"].append(round(float(f1_score(true[m], base[m], average="macro")), 4))
        ent["geom_macro_f1_by_seed"].append(round(float(f1_score(true[m], geom[m], average="macro")), 4))
    ent["baseline_boot_ci"] = boot_ci(true, base)
    ent["geom_boot_ci"] = boot_ci(true, geom)
    ent["paired_diff_image_minus_baseline_boot"] = boot_diff(true, img, base)
    if task == "sixclass":
        sil = silhouette_pred(frame, cls)
        ent["silhouette_bacc"] = float(balanced_accuracy_score(true, sil))
        ent["silhouette_boot_ci"] = boot_ci(true, sil)
        ent["silhouette_macro_f1_by_seed"] = [
            round(float(f1_score(true[(frame.seed == s).values], sil[(frame.seed == s).values],
                                 average="macro")), 4) for s in sorted(frame.seed.unique())]
    res[task] = ent
    print(f"■ {task}")
    for k, v in ent.items():
        print(f"   {k}: {v}")
json.dump(res, open(R / "exp14_baseline_extras.json", "w"), indent=1)
print("\nwritten: results/exp14_baseline_extras.json")
