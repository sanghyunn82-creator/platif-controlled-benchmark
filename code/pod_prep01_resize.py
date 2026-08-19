#!/usr/bin/env python3
"""
P4-1 · Preprocessing: bake the 421 radiographs into uint8 arrays once — **run on the GPU host.**

Originals are float64 in [0,1], 30-150 MB each (5.8 GB total). Reopening the .mat files on every run is wasteful, so
we pre-render 448x448 uint8. Two variants are produced.

  full : aspect-preserving pad to 448x448   — field of view, collimation and markers all remain
  roi  : bone-mask bbox + 12% margin crop   — focuses on anatomy, removes much of the background

The gap between the two variants is direct evidence of how much the background cues contribute.

Note: two images whose mask and radiograph dimensions disagree (ID 47 im1, ID 81 im1) cannot have a roi and get full only.
"""
import json
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom

ROOT = Path("/workspace/platif_p4")
MATDIR = ROOT / "mat"
OUT = ROOT / "prep"
S = 448
MARGIN = 0.12


def fit(a, s=S):
    """Aspect-preserving downscale with centre padding."""
    h, w = a.shape
    f = min(s / h, s / w)
    r = zoom(a, (f, f), order=1)
    r = r[:s, :s]
    out = np.zeros((s, s), dtype=np.float32)
    y0, x0 = (s - r.shape[0]) // 2, (s - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def to_u8(a):
    a = np.clip(a, 0, 1)
    return (a * 255.0 + 0.5).astype(np.uint8)


def one(p):
    pid = int(re.search(r"(\d+)", Path(p).stem).group(1))
    md = sio.loadmat(p, struct_as_record=False, squeeze_me=True)
    top = next(v for k, v in md.items() if not k.startswith("__"))
    recs = []
    for f in top._fieldnames:
        if not f.startswith("im"):
            continue
        im = getattr(top, f)
        g = np.asarray(im.OriginalImage, dtype=np.float32)
        bw = np.asarray(im.BW).astype(bool)
        lab = int(np.asarray(im.label).ravel()[0])
        key = f"{pid:03d}_{f}"
        np.save(OUT / "full" / f"{key}.npy", to_u8(fit(g)))
        ok_roi = bw.shape == g.shape
        if ok_roi:
            rows = np.where(bw.any(axis=1))[0]
            cols = np.where(bw.any(axis=0))[0]
            if rows.size and cols.size:
                r0, r1 = rows.min(), rows.max()
                c0, c1 = cols.min(), cols.max()
                mh, mw = int((r1 - r0) * MARGIN), int((c1 - c0) * MARGIN)
                r0, r1 = max(0, r0 - mh), min(g.shape[0], r1 + mh)
                c0, c1 = max(0, c0 - mw), min(g.shape[1], c1 + mw)
                np.save(OUT / "roi" / f"{key}.npy", to_u8(fit(g[r0:r1, c0:c1])))
            else:
                ok_roi = False
        recs.append({"key": key, "pid": pid, "im": f, "label": lab,
                     "h": int(g.shape[0]), "w": int(g.shape[1]), "has_roi": bool(ok_roi)})
        del g, bw, im
    del md, top
    return recs


def main():
    for d in ("full", "roi"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    files = sorted(str(x) for x in MATDIR.rglob("*.mat"))
    with Pool(16) as pool:
        out = [r for rs in pool.map(one, files) for r in rs]
    json.dump(out, open(ROOT / "results/prep_index.json", "w"))
    n_roi = sum(r["has_roi"] for r in out)
    print(f"images {len(out)} · full {len(out)} · roi {n_roi} "
          f"({len(out)-n_roi} without roi: mask/image size mismatch)")


if __name__ == "__main__":
    main()
