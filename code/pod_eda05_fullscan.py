#!/usr/bin/env python3
"""
P4-1 · Full scan of all 186 PlaTiF .mat files — **run on the GPU host** (not locally).

This single scan settles five open questions at once.
  1. Label constancy within a patient — constant in a 12-patient sample, but the descriptor Usage Notes claim the opposite
  2. Images per patient -> decides immediately whether the reported age 45.88+/-17.54 is an image-weighted mean over 421 images
  3. Actual coronal CT availability and its correlation with class (only a 10-of-12 sample observation so far)
  4. Resolution, aspect ratio and field-of-view distribution (quantifying whole-leg acquisitions)
  5. Whether non-AP views are mixed in — the descriptor says "only AP" while Elnakib et al. excluded 3 as non-AP

Output: results/eda05_fullscan.jsonl (one line per patient). Aggregation happens in a separate script.
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path("/workspace/platif_p4")
MATDIR = ROOT / "mat"
OUT = ROOT / "results"


def arr_summary(a):
    a = np.asarray(a)
    s = {"shape": [int(x) for x in a.shape], "dtype": str(a.dtype)}
    if a.dtype.kind in "biuf" and a.size:
        f = a.ravel()
        samp = f if f.size <= 2_000_000 else f[:: max(1, f.size // 2_000_000)]
        s["min"] = float(samp.min())
        s["max"] = float(samp.max())
        s["mean"] = float(samp.mean())
        u = np.unique(samp)
        s["n_unique"] = int(u.size)
        if u.size <= 8:
            s["unique"] = [float(x) for x in u]
    return s


def scan_one(path):
    pid = int(re.search(r"(\d+)", Path(path).stem).group(1))
    rec = {"pid": pid, "file": Path(path).name,
           "file_mb": round(os.path.getsize(path) / 1024 / 1024, 2)}
    try:
        md = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
        keys = [k for k in md if not k.startswith("__")]
        rec["top_keys"] = keys
        top = md[keys[0]]
        fields = list(top._fieldnames)
        rec["fields"] = fields

        ims = []
        for f in fields:
            if not f.startswith("im"):
                continue
            im = getattr(top, f)
            orig = np.asarray(im.OriginalImage)
            bw = np.asarray(im.BW)
            h, w = orig.shape[:2]
            mask_frac = float(bw.astype(bool).mean())
            # vertical extent of the mask = how much of the frame the tibia occupies vertically.
            # Small for a knee AP, large for a whole-leg acquisition.
            rows = np.where(bw.astype(bool).any(axis=1))[0]
            cols = np.where(bw.astype(bool).any(axis=0))[0]
            bbox = ([int(rows.min()), int(rows.max())] if rows.size else None,
                    [int(cols.min()), int(cols.max())] if cols.size else None)
            v_extent = (float((rows.max() - rows.min() + 1) / h) if rows.size else 0.0)
            entry = {
                "name": f,
                "label": int(np.asarray(im.label).ravel()[0]),
                "h": int(h), "w": int(w),
                "aspect": round(float(h) / float(w), 4),
                "orig": arr_summary(orig),
                "bw_unique": arr_summary(bw).get("unique"),
                "bw_dtype": str(bw.dtype),
                "mask_area_frac": round(mask_frac, 5),
                "mask_vertical_extent": round(v_extent, 4),
                "mask_bbox_rows": bbox[0], "mask_bbox_cols": bbox[1],
                "has_maskedImage": hasattr(im, "maskedImage"),
            }
            ims.append(entry)
            del orig, bw, im
        rec["n_images"] = len(ims)
        rec["images"] = ims
        rec["labels"] = sorted({e["label"] for e in ims})
        rec["label_constant"] = len(rec["labels"]) == 1

        ct = getattr(top, "Coronal_CT", None)
        rec["has_ct"] = ct is not None
        if ct is not None:
            rec["ct"] = arr_summary(np.asarray(ct))
        del md, top
    except Exception:
        rec["error"] = traceback.format_exc(limit=3)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=str(OUT / "eda05_fullscan.jsonl"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    files = sorted(str(p) for p in MATDIR.rglob("*.mat"))
    print(f".mat files {len(files)} · workers {args.workers}", flush=True)
    if not files:
        sys.exit(f"ERROR: no .mat found in {MATDIR}")

    done = 0
    with open(args.out, "w") as fh, mp.Pool(args.workers, maxtasksperchild=4) as pool:
        for rec in pool.imap_unordered(scan_one, files, chunksize=1):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done += 1
            if done % 20 == 0 or done == len(files):
                print(f"  {done}/{len(files)}", flush=True)
    print(f"done: {args.out}", flush=True)


if __name__ == "__main__":
    main()
