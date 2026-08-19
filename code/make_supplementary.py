#!/usr/bin/env python3
"""
P4-1 · Supplementary Tables S1-S6 — **submit the processed data itself.**

A reviewer must be able to reproduce the analysis, and a reader must be able to check our claims
(especially the confounding and the label defects) directly.
S1  Per-patient processed dataset (186 rows) — every derived variable we built; this is the baseline input.
S2  Per-image inventory (421 rows)           — resolution, mask, label, field of view. The raw material of the audit.
S3  All model results (config x seed)        — all five seeds, fold values included.
S4  Same-split non-imaging baselines         — feature set x model x seed.
S5  Per-class performance and confusion      — with Clopper-Pearson intervals.
S6  Dataset integrity findings               — 4 label conflicts, 2 mask mismatches, 4 bilateral acquisitions.

Output: supplementary/Table_S*.csv plus a single combined xlsx.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
R = BASE / "results"
OUT = R / "supplementary"
OUT.mkdir(exist_ok=True)
SCH = "Fracture Type of  Schatzker Classification"
LAT = "Fracture of the Right (R) or Left (L) Tibia"
LBL = {1: "Type 1", 2: "Type 2", 3: "Type 3", 4: "Type 4", 5: "Type 5", 6: "Type 6", 7: "NTPF"}
ROM = {"Type 1": "I", "Type 2": "II", "Type 3": "III", "Type 4": "IV",
       "Type 5": "V", "Type 6": "VI", "NTPF": "NTPF"}

scan = [json.loads(l) for l in open(R / "eda05_fullscan.jsonl")]
idx = json.load(open(R / "prep2_index.json"))
meta = pd.read_excel(BASE / "data/platif/Tibial Plateau Fracture Metadata.xlsx")
tabs = {}

# ── S2: per-image inventory ─────────────────────────────────────────────────
rows = []
scanmap = {r["pid"]: r for r in scan}
for r in idx:
    s = scanmap[r["pid"]]
    im = next((i for i in s["images"] if i["name"] == r["im"]), None)
    br = im.get("mask_bbox_rows") if im else None
    bc = im.get("mask_bbox_cols") if im else None
    rows.append({
        "image_key": r["key"], "patient_id": r["pid"], "image_index": r["im"],
        "height_px": r["h"], "width_px": r["w"],
        "aspect_ratio_h_over_w": round(r["h"] / r["w"], 4),
        "image_level_label": ROM[LBL[r["label"]]],
        "mask_applicable": bool(r["has_roi"]),
        "mask_area_fraction": round(im["mask_area_frac"], 5) if im else None,
        "mask_bbox_row0": br[0] if br else None, "mask_bbox_row1": br[1] if br else None,
        "mask_bbox_col0": bc[0] if bc else None, "mask_bbox_col1": bc[1] if bc else None,
        "whole_leg_view_gt4000px": int(r["h"] > 4000 or r["w"] > 3000),
        "centre_crop_iou_with_mask_bbox": (round(r["center_iou"], 4)
                                          if r.get("center_iou") is not None else None),
        "centre_crop_coverage_of_mask_bbox": (round(r["center_coverage"], 4)
                                              if r.get("center_coverage") is not None else None),
    })
S2 = pd.DataFrame(rows).sort_values(["patient_id", "image_index"])
tabs["S2_per_image_inventory"] = S2

# ── S1: per-patient processed dataset ───────────────────────────────────────
feat = pd.read_csv(R / "exp01_features.csv") if (R / "exp01_features.csv").exists() else None
g = S2.groupby("patient_id")
S1 = pd.DataFrame({
    "patient_id": g.size().index,
    "n_images": g.size().values,
    "image_level_labels": g["image_level_label"].apply(lambda s: "|".join(sorted(set(s)))).values,
    "label_constant_within_patient": g["image_level_label"].nunique().values == 1,
    "max_height_px": g["height_px"].max().values, "max_width_px": g["width_px"].max().values,
    "any_whole_leg_view": g["whole_leg_view_gt4000px"].max().values,
    "median_mask_area_fraction": g["mask_area_fraction"].median().round(5).values,
})
mm = meta.set_index("Patient ID")
S1["metadata_label"] = [ROM[str(mm.loc[p, SCH]).replace("Normal", "NTPF")] for p in S1.patient_id]
S1["age_years"] = [int(mm.loc[p, "Age"]) for p in S1.patient_id]
S1["sex"] = [mm.loc[p, "Gender"] for p in S1.patient_id]
S1["fractured_side"] = [mm.loc[p, LAT] for p in S1.patient_id]
S1["has_coronal_ct"] = [int(scanmap[p]["has_ct"]) for p in S1.patient_id]
S1["diagnosis_title"] = [str(mm.loc[p, "Diagnosis Title"]).replace("\n", " / ") for p in S1.patient_id]
# A conflict means the metadata label is **absent** from that patient's set of image-level labels.
# (Two labels arising from a bilateral acquisition is not a conflict — the descriptor Usage Notes allow it.)
S1["label_conflict_mat_vs_metadata"] = [
    int(ROM[str(mm.loc[p, SCH]).replace("Normal", "NTPF")] not in
        set(S1.loc[S1.patient_id == p, "image_level_labels"].iloc[0].split("|")))
    for p in S1.patient_id]
S1 = S1[["patient_id", "metadata_label", "image_level_labels", "label_constant_within_patient",
         "label_conflict_mat_vs_metadata", "n_images", "has_coronal_ct", "age_years", "sex",
         "fractured_side", "max_height_px", "max_width_px", "any_whole_leg_view",
         "median_mask_area_fraction", "diagnosis_title"]]
tabs["S1_per_patient_processed"] = S1

# ── S3: all model results ───────────────────────────────────────────────────
rows = []
for l in open(R / "exp08_summary.jsonl"):
    r = json.loads(l)
    rows.append({"config": r["config"], "seed": r["seed"], "architecture": r["arch"],
                 "preprocessing": r["prep"], "label_source": r["label_source"],
                 "n_patients": r["n_patients"], "chance": r["chance"],
                 "balanced_accuracy": r["balanced_acc"], "macro_f1": r["macro_f1"],
                 "accuracy": r["accuracy"], "final_train_accuracy": r["final_train_acc"],
                 "fold_balanced_accuracies": "; ".join(f"{x:.4f}" for x in r["fold_baccs"])})
S3 = pd.DataFrame(rows).sort_values(["config", "seed"])
tabs["S3_model_results_by_seed"] = S3

# ── S4: same-split baselines ────────────────────────────────────────────────
rows = []
for r in json.load(open(R / "exp10_samesplit_baseline.json")):
    for i, v in enumerate(r["seed_vals"]):
        rows.append({"task": r["task"],
                     "feature_set": r["features"],
                     "model": r["model"],
                     "pre_specified_primary_comparator": bool(r.get("prespecified")),
                     "seed": i, "balanced_accuracy": v})
# mask silhouette alone (exp13) — a control that uses no radiograph pixels
sil = json.load(open(R / "exp13_mask_silhouette.json"))
silp = json.load(open(R / "exp13_mask_silhouette_perm.json"))
for grid, vals in sil.items():
    for i, v in enumerate(vals):
        rows.append({"task": "Six-class (fracture patients)",
                     "feature_set": f"annotator mask silhouette {grid}x{grid} (no radiograph pixels)",
                     "model": "logistic regression", "pre_specified_primary_comparator": False,
                     "seed": i, "balanced_accuracy": round(v, 4)})
for i, v in enumerate(silp["grid16_perm"]):
    rows.append({"task": "Six-class (fracture patients)",
                 "feature_set": "annotator mask silhouette 16x16, labels permuted",
                 "model": "logistic regression", "pre_specified_primary_comparator": False,
                 "seed": i, "balanced_accuracy": round(v, 4)})
S4 = pd.DataFrame(rows)
tabs["S4_nonimaging_baselines"] = S4

# ── S5: per-class performance ───────────────────────────────────────────────
d = pd.read_csv(R / "exp08_G6_roi_resnet50_prep2_image_logits.csv")
cols = [c for c in d.columns if c.startswith("logit_")]
classes = [c[6:] for c in cols]
gg = d.groupby(["seed", "pid"])
pred = gg[cols].mean().values.argmax(1)
true = np.array([classes.index(v) for v in gg["true"].first().values])
rows = []
for i, c in enumerate(classes):
    m_ = true == i
    k, n = int((pred[m_] == i).sum()), int(m_.sum())
    lo = 0. if k == 0 else stats.beta.ppf(.025, k, n - k + 1)
    hi = 1. if k == n else stats.beta.ppf(.975, k + 1, n - k)
    prec_d = int((pred == i).sum())
    rows.append({"class": ROM[c], "n_patient_evaluations_5seeds": n, "correct": k,
                 "recall": round(k / n, 4), "recall_ci_low": round(lo, 4), "recall_ci_high": round(hi, 4),
                 "precision": round(k / prec_d, 4) if prec_d else None,
                 **{f"predicted_as_{ROM[cc]}": int(((true == i) & (pred == j)).sum())
                    for j, cc in enumerate(classes)}})
S5 = pd.DataFrame(rows)
tabs["S5_per_class_and_confusion"] = S5

# ── S6: integrity findings ──────────────────────────────────────────────────
rows = []
for p in S1.loc[S1.label_conflict_mat_vs_metadata == 1, "patient_id"]:
    r = S1[S1.patient_id == p].iloc[0]
    rows.append({"finding": "label conflict (.mat vs metadata spreadsheet)", "patient_id": int(p),
                 "detail": f"image labels {r.image_level_labels} vs metadata {r.metadata_label}",
                 "action_taken": "image-level .mat label used; both reported"})
for p in S1.loc[~S1.label_constant_within_patient, "patient_id"]:
    r = S1[S1.patient_id == p].iloc[0]
    rows.append({"finding": "labels differ between images of one patient (bilateral acquisition)",
                 "patient_id": int(p), "detail": f"labels {r.image_level_labels}",
                 "action_taken": "image-level labels used; NTPF images excluded from the six-class task"})
for r in S2.loc[~S2.mask_applicable].itertuples():
    rows.append({"finding": "segmentation mask dimensions differ from the radiograph",
                 "patient_id": int(r.patient_id),
                 "detail": f"{r.image_key}: image {r.height_px}x{r.width_px}, mask not applicable",
                 "action_taken": "excluded from mask-dependent inputs (ROI, background)"})
rows.append({"finding": "coronal CT documented inconsistently between descriptor and deposit",
             "patient_id": None,
             "detail": "data descriptor lists a coronal CT for every image; the deposit marks the field optional; 150/186 present",
             "action_taken": "reported"})
rows.append({"finding": "reported cohort age not reproducible from the released metadata",
             "patient_id": None,
             "detail": f"descriptor 45.88 +/- 17.54 (n=186); recomputed "
                       f"{meta['Age'].mean():.2f} +/- {meta['Age'].std(ddof=1):.2f}; "
                       f"sex counts and class percentages match exactly",
             "action_taken": "reported"})
S6 = pd.DataFrame(rows)
tabs["S6_dataset_integrity"] = S6

# ── write ──────────────────────────────────────────────────────────────────
names = {"S1_per_patient_processed": "Table S1. Per-patient processed dataset",
         "S2_per_image_inventory": "Table S2. Per-image inventory",
         "S3_model_results_by_seed": "Table S3. Model results by configuration and seed",
         "S4_nonimaging_baselines": "Table S4. Non-imaging baselines on identical partitions",
         "S5_per_class_and_confusion": "Table S5. Per-class performance and confusion",
         "S6_dataset_integrity": "Table S6. Dataset integrity findings"}
with pd.ExcelWriter(OUT / "Supplementary_Tables.xlsx", engine="openpyxl") as xl:
    for k, df in tabs.items():
        df.to_csv(OUT / f"Table_{k.split('_')[0]}.csv", index=False)
        df.to_excel(xl, sheet_name=k.split("_")[0], index=False)
        print(f"  {names[k]:58s} {len(df):5d} rows x {len(df.columns):2d} cols")
print(f"\nwritten: {OUT}")
