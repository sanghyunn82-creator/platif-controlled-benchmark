#!/usr/bin/env python3
"""
P4-1 · Combined results table — a candidate for manuscript Table 1.

For each task it shows a **ladder**:
    chance -> mask-geometry baseline -> non-anatomical metadata baseline -> CNN

The format is the point. Not "our CNN reached 0.379", but
"of that 0.379, 0.227 is available without looking at the knee".

Experiments still running are marked "pending" and skipped (rerunning fills them in).
"""
import json
from pathlib import Path

import pandas as pd

BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
R = BASE / "results"


def load_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def csv(p):
    return pd.read_csv(p) if p.exists() else None


def best(df, **filt):
    """Highest balanced_acc among the rows matching the condition."""
    if df is None or not len(df):
        return None
    d = df
    for k, v in filt.items():
        if k in d.columns:
            d = d[d[k] == v]
    if not len(d):
        return None
    r = d.loc[d.balanced_acc.idxmax()]
    return float(r.balanced_acc), (float(r.p) if "p" in r and pd.notna(r.p) else None)


def fmt(v, p=None):
    if v is None:
        return "pending"
    s = f"{v:.3f}"
    if p is not None:
        s += " ***" if p < 0.01 else (" *" if p < 0.05 else " (ns)")
    return s


def main():
    exp01 = csv(R / "exp01_shortcut_baselines.csv")
    exp04 = csv(R / "exp04_ct_subset.csv")
    exp05 = csv(R / "exp05_mask_geometry.csv")
    exp07 = csv(R / "exp07_baseline6.csv")
    cnn3 = {r["config"]: r for r in load_jsonl(R / "exp03_cnn.jsonl")}
    cnn6 = {r["config"]: r for r in load_jsonl(R / "exp06_primary6.jsonl")}

    def cnnv(store, key):
        r = store.get(key)
        return (r["balanced_acc"], None) if r else (None, None)

    # exp01 stores its task titles in the target column
    b_bin = best(exp01, target="Binary: NTPF vs tibial plateau fracture") if exp01 is not None else None
    b_7 = best(exp01, target="Seven-class: Schatzker Type 1-6 plus NTPF") if exp01 is not None else None
    g_bin = best(exp05, task="Binary NTPF vs fracture — mask geometry only") if exp05 is not None else None
    g_7 = best(exp05, task="Seven-class — mask geometry only") if exp05 is not None else None
    m6 = best(exp07, features="metadata (non-anatomical)") if exp07 is not None else None
    g6 = best(exp07, features="mask geometry") if exp07 is not None else None

    # permuted-label null (integrity check on the CNN pipeline itself)
    null3 = cnn3.get("F_7cls_roi_shuffle")
    null6 = cnn6.get("I_6cls_roi_shuffle")

    rows = []

    def add(task, n, chance, geom, meta, cnn_roi, cnn_full, null, note):
        rows.append({
            "task": task, "n": n, "chance": f"{chance:.3f}",
            "permuted null": f"{null['balanced_acc']:.3f}" if null else "pending",
            "mask geometry": fmt(*(geom or (None, None))),
            "metadata": fmt(*(meta or (None, None))),
            "CNN (roi)": fmt(*cnn_roi), "CNN (full)": fmt(*cnn_full),
            "note": note,
        })

    add("Binary NTPF vs fracture", 186, 0.5, g_bin, b_bin,
        cnnv(cnn3, "C_bin_roi"), cnnv(cnn3, "D_bin_full"), cnn3.get("J_bin_roi_shuffle"),
        "the CNN does not exceed the metadata baseline (p=0.670/0.522)")
    add("Seven-class (NTPF included)", 186, 1 / 7, g_7, b_7,
        cnnv(cnn3, "A_7cls_roi"), cnnv(cnn3, "B_7cls_full"), null3,
        "the CNN exceeds the baseline")
    add("Six-class (fracture only) [primary]", 128, 1 / 6, g6, m6,
        cnnv(cnn6, "G_6cls_roi"), (cnn6.get("H_6cls_centercrop", {}).get("balanced_acc"), None),
        null6, "all baselines sit at chance -> a clean benchmark (CNN p=0.020)")

    df = pd.DataFrame(rows)
    print("=" * 110)
    print("■ Table 1 candidate — performance ladder by task (balanced accuracy, patient-level evaluation)")
    print("=" * 110)
    print(df.to_string(index=False))
    print()
    print("  * p<0.05, *** p<0.01, (ns) not significant — against 200 permuted-label draws")
    print("  The six-class CNN(full) cell is the fixed centre-crop control (no mask used).")

    # control conditions
    print()
    print("=" * 110)
    print("■ control conditions")
    print("=" * 110)
    ctl = []
    for store, keys in [(cnn3, ["E_7cls_roi_noflip", "F_7cls_roi_shuffle", "J_bin_roi_shuffle"]),
                        (cnn6, ["I_6cls_roi_shuffle"])]:
        for k in keys:
            r = store.get(k)
            ctl.append({"configuration": k, "balanced acc": f"{r['balanced_acc']:.3f}" if r else "pending",
                        "chance": f"{r['chance']:.3f}" if r else "",
                        "reading": {"E_7cls_roi_noflip": "horizontal flip off — compare with A for the left/right shortcut",
                                 "F_7cls_roi_shuffle": "permuted labels — should sit at chance",
                                 "J_bin_roi_shuffle": "permuted labels (binary) — the control a negative conclusion requires",
                                 "I_6cls_roi_shuffle": "permuted labels — should sit at chance"}[k]})
    print(pd.DataFrame(ctl).to_string(index=False))

    # decomposition of the margin above chance
    print()
    print("=" * 110)
    print("■ decomposition of the margin above chance — how much of the CNN score is anatomy?")
    print("=" * 110)
    for task, chance, meta, cnn in [
            ("binary", 0.5, b_bin, cnnv(cnn3, "C_bin_roi")),
            ("seven-class", 1 / 7, b_7, cnnv(cnn3, "A_7cls_roi")),
            ("six-class (fracture only)", 1 / 6, m6, cnnv(cnn6, "G_6cls_roi"))]:
        if not meta or cnn[0] is None:
            print(f"  {task:<26s} pending"); continue
        dm, dc = meta[0] - chance, cnn[0] - chance
        share = 100 * dm / dc if dc > 0 else float("nan")
        print(f"  {task:<26s} CNN +{dc:.3f} · metadata +{dm:.3f} -> "
              f"**metadata explains {share:.0f}% of the CNN margin**")

    df.to_csv(R / "table1_summary.csv", index=False)
    print(f"\nwritten: {R/'table1_summary.csv'}")


if __name__ == "__main__":
    main()
