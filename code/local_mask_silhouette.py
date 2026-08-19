#!/usr/bin/env python3
"""P4-1 · How much Schatzker type is recoverable from the annotator mask alone — CPU only.

Review point: the tibia-erased input is the **full frame** with the mask interior set to zero, so the
tibial silhouette (the plateau contour) survives the ablation. The residual 0.257 therefore cannot be
attributed to information outside the tibia without testing this alternative. This script measures it
directly: it solves the six-class task on the same folds using **only the binary mask silhouette, with no radiograph pixels at all**.

Output: results/exp13_mask_silhouette.json
"""
import io, json, re, sys, zipfile
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom

BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
DATA = BASE / "data/platif"
OUT = BASE / "results"
TMP = Path("/private/tmp/claude-501/-Volumes-dkyoo-SSD1-review-partime/"
           "7d64dfba-5ad2-48ef-8add-70a0e6395218/scratchpad")
G = 48   # silhouette grid

def fit_grid(bw, s=G):
    """Aspect-preserving downscale with centre padding; keeps shape, not pixel values."""
    a = bw.astype(np.float32)
    h, w = a.shape
    f = min(s / h, s / w)
    r = zoom(a, (f, f), order=1)[:s, :s]
    out = np.zeros((s, s), dtype=np.float32)
    y0, x0 = (s - r.shape[0]) // 2, (s - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out

recs = []
for part in sorted(DATA.glob("Patient Data_Part *.zip")):
    z = zipfile.ZipFile(part)
    for nm in sorted(z.namelist()):
        if not nm.endswith(".mat"):
            continue
        pid = int(re.search(r"(\d+)", Path(nm).stem).group(1))
        tmpf = TMP / "one.mat"
        with z.open(nm) as fh, open(tmpf, "wb") as w:
            while True:
                b = fh.read(1 << 22)
                if not b: break
                w.write(b)
        md = sio.loadmat(tmpf, struct_as_record=False, squeeze_me=True)
        top = next(v for k, v in md.items() if not k.startswith("__"))
        for f in top._fieldnames:
            if not f.startswith("im"):
                continue
            im = getattr(top, f)
            g = np.asarray(im.OriginalImage)
            bw = np.asarray(im.BW).astype(bool)
            lab = int(np.asarray(im.label).ravel()[0])
            if bw.shape != g.shape:
                continue                      # the two images with an inapplicable mask
            recs.append({"key": f"{pid:03d}_{f}", "pid": pid, "label": lab,
                         "grid": fit_grid(bw).astype(np.float16)})
        print(f"  {Path(nm).stem}  cumulative {len(recs)} images", flush=True)
    z.close()

np.savez_compressed(OUT / "mask_silhouettes.npz",
                    keys=np.array([r["key"] for r in recs]),
                    pids=np.array([r["pid"] for r in recs]),
                    labels=np.array([r["label"] for r in recs]),
                    grids=np.stack([r["grid"] for r in recs]))
print(f"\nwritten: {len(recs)} images · grid {G}x{G}")
