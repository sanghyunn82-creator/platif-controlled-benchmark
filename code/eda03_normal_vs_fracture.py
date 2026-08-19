#!/usr/bin/env python3
"""
PlaTiF exploratory pass 3: look at what the "Normal" class actually is.

Motivation: in the metadata all 58 Normal patients **carry a fracture diagnosis**
      (57 "Unspecified fracture of upper end of tibia", 1 ankle).
      They may therefore not be a healthy-knee control group. That changes the task definition itself,
      so the images must be inspected, not just the numbers.

Memory discipline: open one patient at a time, downsample immediately, discard the original array.
             float64 originals are 30-150 MB each and must never accumulate.
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

BASE = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed")
DATA = BASE / "data/platif"
OUT = BASE / "results"
SCRATCH = DATA / ".scratch"
LABEL_NAME = {1: "Type 1", 2: "Type 2", 3: "Type 3", 4: "Type 4",
              5: "Type 5", 6: "Type 6", 7: "Normal"}


def member_index():
    idx = {}
    for z in sorted(DATA.glob("Patient Data_Part *.zip")):
        with zipfile.ZipFile(z) as zf:
            for i in zf.infolist():
                if i.filename.lower().endswith(".mat"):
                    pid = int(re.search(r"(\d+)", Path(i.filename).name).group(1))
                    idx[pid] = (z, i.filename)
    return idx


def load_patient(z, member, max_side=700):
    """Return one patient's images, downsampled. Originals are discarded immediately."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = SCRATCH / Path(member).name
    with zipfile.ZipFile(z) as zf, zf.open(member) as src, open(tmp, "wb") as dst:
        while chunk := src.read(1 << 20):
            dst.write(chunk)
    try:
        md = sio.loadmat(tmp, struct_as_record=False, squeeze_me=True)
        top = next(v for k, v in md.items() if not k.startswith("__"))
        out = []
        for f in top._fieldnames:
            if not f.startswith("im"):
                continue
            im = getattr(top, f)
            orig = np.asarray(im.OriginalImage, dtype=np.float32)
            bw = np.asarray(im.BW, dtype=np.uint8)
            step = max(1, int(np.ceil(max(orig.shape) / max_side)))
            small = orig[::step, ::step].copy()
            smallbw = bw[::step, ::step].copy()
            out.append({"name": f, "img": small, "bw": smallbw,
                        "label": int(np.asarray(im.label).ravel()[0]),
                        "shape": orig.shape})
            del orig, bw, im
        del md, top
        return out
    finally:
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    meta = pd.read_excel(DATA / "Tibial Plateau Fracture Metadata.xlsx")
    sch = "Fracture Type of  Schatzker Classification"
    lat = "Fracture of the Right (R) or Left (L) Tibia"
    idx = member_index()

    # place Normal next to representative fracture types
    picks = []
    for cls in ["Normal", "Type 2", "Type 6"]:
        sub = meta.loc[meta[sch] == cls, "Patient ID"].tolist()
        sub = [p for p in sub if p in idx][: args.per_class]
        picks += [(cls, p) for p in sub]

    ncol = args.per_class
    nrow = len(picks) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.6 * nrow))
    axes = np.atleast_2d(axes)

    print(f"{'patient':>8s} {'meta':>8s} {'side':>7s} {'.mat label':>11s} {'imgs':>5s}  first-image resolution")
    print("-" * 66)
    for k, (cls, pid) in enumerate(picks):
        z, member = idx[pid]
        ims = load_patient(z, member)
        first = ims[0]
        lab = LABEL_NAME.get(first["label"], first["label"])
        side = meta.loc[meta["Patient ID"] == pid, lat].iloc[0]
        print(f"{pid:5d} {cls:>8s} {str(side):>7s} {str(lab):>9s} {len(ims):4d}  "
              f"{first['shape'][0]}x{first['shape'][1]}")

        ax = axes[k // ncol, k % ncol]
        ax.imshow(first["img"], cmap="gray")
    # draw only a thin mask outline so the image itself stays visible
        ax.contour(first["bw"], levels=[0.5], colors="#e05252", linewidths=0.8)
        ax.set_title(f"ID {pid} · xlsx={cls} · .mat={lab}\n{side} side · {len(ims)} images · "
                     f"{first['shape'][0]}×{first['shape'][1]}", fontsize=9)
        ax.axis("off")
        del ims, first

    fig.suptitle("PlaTiF: is a fracture actually visible in the radiographs of 'Normal' patients?\n"
                 "(red outline = the tibial mask shipped with the dataset)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / "eda03_normal_vs_fracture.png"
    fig.savefig(p, dpi=110)
    print(f"\nwritten: {p}")
    try:
        SCRATCH.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
