#!/usr/bin/env python3
"""
P4-1 · Preprocessing v2 — **anti-aliasing fix.** Run on the GPU host.

Bug in v1: `scipy.ndimage.zoom(a, f, order=1)` applies no low-pass prefilter when downscaling.
At a median factor of 6.3 (up to 10.9) that is effectively point sampling, and **high-frequency
structure such as fracture lines is destroyed before it reaches the model.** Worse, the aliasing
pattern depends on the original resolution, which is itself a **label-correlated variable** present
in our metadata baseline: the resize may have injected a new non-anatomical cue. That is a bug, not an ablation.

Fix: apply a Gaussian prefilter (sigma = (1/f - 1)/2) before downscaling, respecting Nyquist.
The v1 outputs stay in `prep/` and v2 is written to `prep2/`, so the two can be compared.

Also: the image-level label (`label`) is carried into the index so the training scripts can use it.
      (v1 used only the patient-level xlsx label, which trained 5 contralateral normal knees as fractures.)
"""
import json
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter, zoom

ROOT = Path("/workspace/platif_p4")
MATDIR = ROOT / "mat"
OUT = ROOT / "prep2"
S = 448
MARGIN = 0.12


def fit(a, s=S):
    """Anti-aliased downscale with aspect-preserving centre padding."""
    a = np.asarray(a, dtype=np.float32)
    h, w = a.shape
    f = min(s / h, s / w)
    if f < 1.0:
        # Gaussian prefilter matched to the 1/f downscale factor; protects Nyquist.
        sigma = ((1.0 / f) - 1.0) / 2.0
        if sigma > 0.05:
            a = gaussian_filter(a, sigma=sigma, mode="nearest")
    r = zoom(a, (f, f), order=1)[:s, :s]
    out = np.zeros((s, s), dtype=np.float32)
    y0, x0 = (s - r.shape[0]) // 2, (s - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def to_u8(a):
    return (np.clip(a, 0, 1) * 255.0 + 0.5).astype(np.uint8)


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
                # background only (bone erased) — controls for hardware/background cues. Built at 448.
                bg = g.copy(); bg[bw] = 0.0
                np.save(OUT / "bg" / f"{key}.npy", to_u8(fit(bg)))
                # centre-crop control: a bbox of the **same size** as the ROI placed at the image centre
                # (v1's H varied mask presence, field of view and resolution at once — a triple confound)
                bh, bw_ = r1 - r0, c1 - c0
                cy, cx = g.shape[0] // 2, g.shape[1] // 2
                y0, y1 = max(0, cy - bh // 2), min(g.shape[0], cy + bh // 2)
                x0, x1 = max(0, cx - bw_ // 2), min(g.shape[1], cx + bw_ // 2)
                np.save(OUT / "center" / f"{key}.npy", to_u8(fit(g[y0:y1, x0:x1])))
                # audit: how much of the true mask bbox does the centre bbox contain?
                iy0, iy1 = max(r0, y0), min(r1, y1)
                ix0, ix1 = max(c0, x0), min(c1, x1)
                inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
                union = bh * bw_ + (y1 - y0) * (x1 - x0) - inter
                iou = inter / union if union else 0.0
                cover = inter / (bh * bw_) if bh * bw_ else 0.0
            else:
                ok_roi = False; iou = cover = None
        else:
            iou = cover = None
        recs.append({"key": key, "pid": pid, "im": f, "label": lab,
                     "h": int(g.shape[0]), "w": int(g.shape[1]), "has_roi": bool(ok_roi),
                     "center_iou": iou, "center_coverage": cover})
        del g, bw, im
    del md, top
    return recs


def main():
    for d in ("full", "roi", "bg", "center"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    files = sorted(str(x) for x in MATDIR.rglob("*.mat"))
    with Pool(16) as pool:
        out = [r for rs in pool.map(one, files) for r in rs]
    json.dump(out, open(ROOT / "results/prep2_index.json", "w"))
    n_roi = sum(r["has_roi"] for r in out)
    cov = [r["center_coverage"] for r in out if r["center_coverage"] is not None]
    print(f"images {len(out)} · roi {n_roi}")
    print(f"centre crop coverage of the mask bbox: median {np.median(cov):.3f} · "
          f"mean {np.mean(cov):.3f} · below 50% in {sum(c < 0.5 for c in cov)} images")
    # report patients whose image-level labels disagree with the patient-level label
    from collections import defaultdict
    byp = defaultdict(set)
    for r in out:
        byp[r["pid"]].add(r["label"])
    mixed = {p: sorted(v) for p, v in byp.items() if len(v) > 1}
    print(f"patients with mixed image-level labels: {len(mixed)} -> {mixed}")


if __name__ == "__main__":
    main()
