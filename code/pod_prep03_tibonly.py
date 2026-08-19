#!/usr/bin/env python3
"""P4-1 · Inverse-ablation input: keep the tibia only, zero everything else.

Exact complement of the `bg` input built by pod_prep02_antialias.py:
    bg  : g[bw] = 0      (the tibia is erased)
    tib : g[~bw] = 0     (only the tibia is kept)
Both use the same fit()/to_u8(), so field of view, resolution and preprocessing match bg exactly, and the two together partition the radiograph.
"""
import json, re
from multiprocessing import Pool
from pathlib import Path
import numpy as np, scipy.io as sio
from scipy.ndimage import gaussian_filter, zoom

ROOT = Path("/workspace/platif_p4"); MATDIR = ROOT / "mat"; OUT = ROOT / "prep2"
S = 448

def fit(a, s=S):
    a = np.asarray(a, dtype=np.float32); h, w = a.shape
    f = min(s / h, s / w)
    if f < 1.0:
        sigma = ((1.0 / f) - 1.0) / 2.0
        if sigma > 0.05: a = gaussian_filter(a, sigma=sigma, mode="nearest")
    r = zoom(a, (f, f), order=1)[:s, :s]
    out = np.zeros((s, s), dtype=np.float32)
    y0, x0 = (s - r.shape[0]) // 2, (s - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out

def to_u8(a): return (np.clip(a, 0, 1) * 255.0 + 0.5).astype(np.uint8)

def one(p):
    pid = int(re.search(r"(\d+)", Path(p).stem).group(1))
    md = sio.loadmat(p, struct_as_record=False, squeeze_me=True)
    top = next(v for k, v in md.items() if not k.startswith("__"))
    n = 0
    for f in top._fieldnames:
        if not f.startswith("im"): continue
        im = getattr(top, f)
        g = np.asarray(im.OriginalImage, dtype=np.float32)
        bw = np.asarray(im.BW).astype(bool)
        if bw.shape != g.shape: continue
        if not (bw.any(axis=1).any() and bw.any(axis=0).any()): continue
        tib = g.copy(); tib[~bw] = 0.0          # exact complement of bg
        np.save(OUT / "tib" / f"{pid:03d}_{f}.npy", to_u8(fit(tib)))
        n += 1
    return n

if __name__ == "__main__":
    (OUT / "tib").mkdir(parents=True, exist_ok=True)
    files = sorted(str(x) for x in MATDIR.rglob("*.mat"))
    with Pool(8) as pool: tot = sum(pool.map(one, files))
    print(f"tib images written: {tot}")
