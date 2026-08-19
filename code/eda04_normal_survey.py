#!/usr/bin/env python3
"""
PlaTiF exploratory pass 4: survey the Normal class more widely and check for burnt-in acquisition marks.

Two things stood out in pass 3.
  (1) some Normal patients show **surgical hardware (screws, plates)** (ID 55).
  (2) every image carries **burnt-in L/R laterality markers** in a corner.
      Normal is heavily skewed, R 39 / L 17, while Type 4 is R 2 / L 9.
      -> a model could shortcut by **reading the corner character** rather than the fracture.

This script checks both on a larger sample. The judgement is human: here we lay the images out and
quantify only whether a marker is present, via corner brightness statistics.
"""
import argparse
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

matplotlib.rcParams["font.family"] = "Apple SD Gothic Neo"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
DATA = BASE / "data/platif"
OUT = BASE / "results"
SCRATCH = DATA / ".scratch"
SCH = "Fracture Type of  Schatzker Classification"
LAT = "Fracture of the Right (R) or Left (L) Tibia"


def member_index():
    idx = {}
    for z in sorted(DATA.glob("Patient Data_Part *.zip")):
        with zipfile.ZipFile(z) as zf:
            for i in zf.infolist():
                if i.filename.lower().endswith(".mat"):
                    pid = int(re.search(r"(\d+)", Path(i.filename).name).group(1))
                    idx[pid] = (z, i.filename)
    return idx


def first_image(z, member, max_side=640):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = SCRATCH / Path(member).name
    with zipfile.ZipFile(z) as zf, zf.open(member) as src, open(tmp, "wb") as dst:
        while chunk := src.read(1 << 20):
            dst.write(chunk)
    try:
        md = sio.loadmat(tmp, struct_as_record=False, squeeze_me=True)
        top = next(v for k, v in md.items() if not k.startswith("__"))
        fields = [f for f in top._fieldnames if f.startswith("im")]
        im = getattr(top, fields[0])
        orig = np.asarray(im.OriginalImage, dtype=np.float32)
        h, w = orig.shape
        # brightness of the four corners (to detect burnt-in markers), measured at native resolution.
        cs = max(40, int(0.10 * min(h, w)))
        corners = {
            "top-left": float(orig[:cs, :cs].max()), "top-right": float(orig[:cs, -cs:].max()),
            "bottom-left": float(orig[-cs:, :cs].max()), "bottom-right": float(orig[-cs:, -cs:].max()),
        }
        # bright pixels outside the bone (mask == 0) are candidate markers or hardware
        bw = np.asarray(im.BW, dtype=bool)
        outside = orig[~bw]
        frac_bright_outside = float((outside > 0.75).mean()) if outside.size else 0.0
        step = max(1, int(np.ceil(max(h, w) / max_side)))
        small = orig[::step, ::step].copy()
        n_im = len(fields)
        del orig, bw, outside, md, top, im
        return small, (h, w), corners, frac_bright_outside, n_im
    finally:
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cls", default="Normal")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    meta = pd.read_excel(DATA / "Tibial Plateau Fracture Metadata.xlsx")
    idx = member_index()
    pids = [p for p in meta.loc[meta[SCH] == args.cls, "Patient ID"].tolist() if p in idx][: args.n]

    ncol = 4
    nrow = int(np.ceil(len(pids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 4.2 * nrow))
    axes = np.atleast_2d(axes)

    print(f"{'patient':>8s} {'side':>7s} {'imgs':>5s} {'resolution':>12s} "
          f"{'bright px outside bone %':>26s}  corner max brightness (TL/TR/BL/BR)")
    print("-" * 96)
    rows = []
    for k, pid in enumerate(pids):
        z, member = idx[pid]
        img, shape, corners, frac, n_im = first_image(z, member)
        side = meta.loc[meta["Patient ID"] == pid, LAT].iloc[0]
        cstr = " / ".join(f"{corners[c]:.2f}" for c in ["top-left", "top-right", "bottom-left", "bottom-right"])
        print(f"{pid:5d} {str(side):>7s} {n_im:4d} {shape[0]:5d}x{shape[1]:<6d} "
              f"{100*frac:12.2f}%  {cstr}")
        rows.append({"pid": pid, "side": side, "n_im": n_im, "h": shape[0], "w": shape[1],
                     "bright_outside_pct": round(100 * frac, 2), **corners})
        ax = axes[k // ncol, k % ncol]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"ID {pid} · {side} side · {n_im} images\n{shape[0]}x{shape[1]}", fontsize=8)
        ax.axis("off")
        del img
    for k in range(len(pids), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")

    fig.suptitle(f"PlaTiF '{args.cls}', {len(pids)} patients — hardware and corner-marker check\n"
                 f"the corner L/R characters are burnt in, and {args.cls} has a skewed laterality "
                 f"distribution, creating a shortcut-learning risk", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUT / f"eda04_{args.cls.replace(' ', '')}_survey.png"
    fig.savefig(p, dpi=110)
    pd.DataFrame(rows).to_csv(OUT / f"eda04_{args.cls.replace(' ', '')}_corners.csv", index=False)
    print(f"\nwritten: {p}")
    try:
        SCRATCH.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
