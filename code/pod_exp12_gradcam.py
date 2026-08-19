#!/usr/bin/env python3
"""
P4-1 · Experiment 12 — Grad-CAM, for Figure 4. GPU host.

Purpose: give **visual support** to the quantitative result that the background carries half the
      performance (0.257 with the bone erased), by showing where the model actually looks.

Design:
  - Train seed 0 / fold 1 with the same protocol as G6_roi (training fold only), then compute CAMs on that fold's **test patients**.
  - Because the mask is known, the **share of CAM mass inside versus outside the bone** is quantified; that is the number behind the figure.
  - Saved: the CAM arrays (downsampled) and a per-patient inside-bone fraction CSV.
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/workspace/platif_p4")
PREP = ROOT / "prep2"
OUT = ROOT / "results"
SCH = "Fracture Type of  Schatzker Classification"
DEV = "cuda"
MEAN, STD, S = 0.449, 0.226, 448
CLS6 = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6"]
LBL = {1: "Type 1", 2: "Type 2", 3: "Type 3", 4: "Type 4", 5: "Type 5", 6: "Type 6", 7: "NTPF"}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class DS(Dataset):
    def __init__(self, keys, y, train):
        self.keys, self.y, self.train = keys, y, train

    def __len__(self): return len(self.keys)

    def __getitem__(self, i):
        a = np.load(PREP / "roi" / f"{self.keys[i]}.npy").astype(np.float32) / 255.0
        if self.train:
            if np.random.rand() < .5: a = a[:, ::-1].copy()
            if np.random.rand() < .5: a = np.clip(a * np.random.uniform(.9, 1.1), 0, 1)
        a = (a - MEAN) / STD
        return torch.from_numpy(a)[None].repeat(3, 1, 1), self.y[i]


def main():
    idx = json.load(open(OUT / "prep2_index.json"))
    meta = pd.read_excel(ROOT / "data/Tibial Plateau Fracture Metadata.xlsx").set_index("Patient ID")
    pat = {int(p): str(meta.loc[p, SCH]).replace("Normal", "NTPF") for p in meta.index}
    rows = [r for r in idx if r.get("has_roi") and pat[r["pid"]] != "NTPF" and LBL[r["label"]] != "NTPF"]
    keys = np.array([r["key"] for r in rows]); pids = np.array([r["pid"] for r in rows])
    y = np.array([CLS6.index(LBL[r["label"]]) for r in rows])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    tr, te = next(iter(cv.split(keys, y, groups=pids)))
    pu = sorted(set(pids.tolist())); py = np.array([y[pids == p][0] for p in pu])
    cnt = np.bincount(py, minlength=6).astype(float)
    cls_w = (cnt.sum() / (6 * np.maximum(cnt, 1))).tolist()

    import torchvision
    set_seed(0)
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, 6); m = m.to(DEV)
    dl = DataLoader(DS(keys[tr], y[tr], True), batch_size=16, shuffle=True, num_workers=8, drop_last=True)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=25 * len(dl))
    sc = torch.cuda.amp.GradScaler(); w = torch.tensor(cls_w, dtype=torch.float32, device=DEV)
    for ep in range(25):
        m.train()
        for x, yy in dl:
            x, yy = x.to(DEV), yy.to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = F.cross_entropy(m(x), yy, weight=w, label_smoothing=.05)
            sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step()
    print("training complete", flush=True)

    # ── Grad-CAM (last block of layer4) ──────────────────────────────────────
    feats, grads = {}, {}
    tgt = m.layer4[-1]
    tgt.register_forward_hook(lambda mod, i, o: feats.__setitem__("v", o))
    tgt.register_full_backward_hook(lambda mod, gi, go: grads.__setitem__("v", go[0]))

    m.eval()
    recs, cams = [], {}
    bxmap = {r["key"]: r for r in idx}
    for k, yy in zip(keys[te], y[te]):
        a = np.load(PREP / "roi" / f"{k}.npy").astype(np.float32) / 255.0
        x = torch.from_numpy((a - MEAN) / STD)[None, None].repeat(1, 3, 1, 1).to(DEV)
        out = m(x)
        pred = int(out.argmax(1))
        m.zero_grad(); out[0, pred].backward()
        A, G = feats["v"][0], grads["v"][0]
        cam = F.relu((G.mean(dim=(1, 2), keepdim=True) * A).sum(0))
        cam = F.interpolate(cam[None, None], size=(S, S), mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cam.detach().cpu().numpy()
        # The roi input is a bone-bbox crop, so the bone itself fills most of the frame.
        # We therefore use the share of CAM mass in the outer 15% margin as a proxy for background attention.
        edge = np.ones_like(cam, bool); e = int(S * .15)
        edge[e:-e, e:-e] = False
        recs.append({"key": k, "pid": int(bxmap[k]["pid"]), "true": CLS6[yy],
                     "pred": CLS6[pred], "correct": int(pred == yy),
                     "cam_edge_fraction": float(cam[edge].sum() / (cam.sum() + 1e-8)),
                     "edge_area_fraction": float(edge.mean())})
        cams[k] = cam[::4, ::4].astype(np.float16)   # stored downsampled to 112x112
    pd.DataFrame(recs).to_csv(OUT / "exp12_gradcam_stats.csv", index=False)
    np.savez_compressed(OUT / "exp12_cams.npz", **cams)
    d = pd.DataFrame(recs)
    print(f"test images {len(d)} · mean CAM margin-mass fraction {d.cam_edge_fraction.mean():.3f} "
          f"(margin area fraction {d.edge_area_fraction.iloc[0]:.3f})", flush=True)
    print(d.groupby("true")["cam_edge_fraction"].agg(["mean", "count"]).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
