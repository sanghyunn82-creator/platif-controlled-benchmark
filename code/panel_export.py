#!/usr/bin/env python3
"""P4-1 · Export figure panels as individual files.

Alongside the assembled figures we emit **one file per panel**, so that the panels can be laid out
by hand (for example in PowerPoint). A panel is one axes or a group of axes (Figure 3D is two).

How it works: for the given axes, the tight bounding boxes (title, axis labels and ticks included)
are unioned and only that region is written out. Panel letters (A, B, C) are drawn in axes
coordinates and therefore **come along**; footnotes drawn in figure coordinates (fig.text) do not.
To export without the letters, disable the panel(...) calls in the figure script and re-run.

Output: manuscript/figures/panels/<Figure><panel>.png / .pdf   (e.g. Figure3A.png)
"""
from pathlib import Path

from matplotlib.transforms import Bbox

OUTDIR = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed/manuscript/figures/panels")


def save_panel(fig, axs, name, dpi=400, pad=0.10, outdir=None):
    """Crop the region enclosing axs (one axes or several) to name.png / name.pdf."""
    outdir = Path(outdir) if outdir else OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    if not isinstance(axs, (list, tuple)):
        axs = [axs]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bb = Bbox.union([a.get_tightbbox(rend) for a in axs])
    bb = bb.transformed(fig.dpi_scale_trans.inverted())
    # widen slightly so nothing is clipped (annotations and brackets outside the axes)
    bb = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
    fig.savefig(outdir / f"{name}.png", dpi=dpi, bbox_inches=bb)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches=bb)
    return outdir / f"{name}.png"


def save_panels(fig, mapping, dpi=400, pad=0.10, outdir=None):
    """mapping: {"Figure3A": axA, "Figure3D": [ax1, ax2], ...}"""
    out = []
    for name, axs in mapping.items():
        out.append(save_panel(fig, axs, name, dpi=dpi, pad=pad, outdir=outdir))
    print(f"   {len(out)} panels written -> {(Path(outdir) if outdir else OUTDIR)}")
    return out
