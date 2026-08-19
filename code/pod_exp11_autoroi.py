#!/usr/bin/env python3
"""
P4-1 · Experiment 11 — **automatic ROI.** Removes the dependence on the oracle mask. GPU host.

Problem (audit blocker 1): the six-class primary result crops the ROI using the **annotator ground-truth
mask** at inference time. We disqualified the NTPF task because CT is unavailable at deployment, yet the
annotator mask is equally unavailable at deployment. Disqualifying only one side is inconsistent.

Fix: fit a tibial bbox regressor **using only the training patients of each outer fold**, and crop the
      test patients with the **predicted bbox**. The mask is used as a training signal only, never at inference.
      (Nested design — the test patients' masks are never seen at any stage.)

Output: six-class performance from predicted ROIs. Compared with the oracle (G6_roi 0.345), the centre crop
      (0.260) and background-only (0.257), it shows how much of the mask's localisation is automatically recoverable.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/workspace/platif_p4")
PREP = ROOT / "prep2"
OUT = ROOT / "results"
SCH = "Fracture Type of  Schatzker Classification"
DEV = "cuda"
MEAN, STD = 0.449, 0.226
CLS6 = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6"]
LBL = {1: "Type 1", 2: "Type 2", 3: "Type 3", 4: "Type 4", 5: "Type 5", 6: "Type 6", 7: "NTPF"}
S = 448


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


# ── stage 1: bbox regressor ─────────────────────────────────────────────────
class BoxDS(Dataset):
    """Input is full (the whole image); the target is the mask bbox normalised to 448 coordinates (4 values)."""
    def __init__(self, keys, boxes, train):
        self.keys, self.boxes, self.train = keys, boxes, train

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        a = np.load(PREP / "full" / f"{self.keys[i]}.npy").astype(np.float32) / 255.0
        b = self.boxes[i].copy()
        if self.train and np.random.rand() < 0.5:
            a = a[:, ::-1].copy(); b = np.array([b[0], b[1], 1.0 - b[3], 1.0 - b[2]])
        a = (a - MEAN) / STD
        return torch.from_numpy(a)[None].repeat(3, 1, 1), torch.tensor(b, dtype=torch.float32)


def boxes_from_index(idx):
    """Convert the mask bbox to fractions of the padded 448 coordinate frame."""
    out = {}
    for r in idx:
        if not r.get("has_roi"):
            continue
        h, w = r["h"], r["w"]
        f = min(S / h, S / w)
        nh, nw = h * f, w * f
        oy, ox = (S - nh) / 2, (S - nw) / 2
        bb = r.get("bbox")
        if bb is None:
            continue
        r0, r1, c0, c1 = bb
        out[r["key"]] = np.array([(oy + r0 * f) / S, (oy + r1 * f) / S,
                                  (ox + c0 * f) / S, (ox + c1 * f) / S], dtype=np.float32)
    return out


def train_boxreg(keys, boxes, seed, fold, epochs=15, bs=16, lr=3e-4):
    import torchvision
    set_seed(seed * 100 + fold + 7000)
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 4); m = m.to(DEV)
    dl = DataLoader(BoxDS(keys, boxes, True), batch_size=bs, shuffle=True,
                    num_workers=8, pin_memory=True, drop_last=len(keys) > bs)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, len(dl)))
    sc = torch.cuda.amp.GradScaler()
    for _ in range(epochs):
        m.train()
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = F.l1_loss(m(x).sigmoid(), y)
            sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step()
    m.eval()
    return m


@torch.no_grad()
def predict_boxes(m, keys, bs=32):
    dl = DataLoader(BoxDS(keys, np.zeros((len(keys), 4), np.float32), False),
                    batch_size=bs, num_workers=8)
    out = []
    with torch.cuda.amp.autocast():
        for x, _ in dl:
            out.append(m(x.to(DEV)).sigmoid().float().cpu().numpy())
    return np.concatenate(out)


# ── stage 2: classify from the predicted ROI ────────────────────────────────
class ClsDS(Dataset):
    def __init__(self, keys, labels, boxes, train):
        self.keys, self.labels, self.boxes, self.train = keys, labels, boxes, train

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        a = np.load(PREP / "full" / f"{self.keys[i]}.npy").astype(np.float32) / 255.0
        r0, r1, c0, c1 = (np.clip(self.boxes[i], 0, 1) * S).astype(int)
        r0, r1 = max(0, min(r0, S - 8)), max(8, min(r1, S))
        c0, c1 = max(0, min(c0, S - 8)), max(8, min(c1, S))
        if r1 - r0 < 8: r1 = min(S, r0 + 8)
        if c1 - c0 < 8: c1 = min(S, c0 + 8)
        crop = a[r0:r1, c0:c1]
        t = torch.from_numpy(crop)[None, None]
        t = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)[0, 0].numpy()
        if self.train:
            if np.random.rand() < 0.5: t = t[:, ::-1].copy()
            if np.random.rand() < 0.5: t = np.clip(t * np.random.uniform(0.9, 1.1), 0, 1)
        t = (t - MEAN) / STD
        return torch.from_numpy(t)[None].repeat(3, 1, 1), self.labels[i]


def train_cls(keys, y, boxes, cls_w, seed, fold, epochs=25, bs=16, lr=3e-4):
    import torchvision
    set_seed(seed * 100 + fold)
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, 6); m = m.to(DEV)
    dl = DataLoader(ClsDS(keys, y, boxes, True), batch_size=bs, shuffle=True,
                    num_workers=8, pin_memory=True, drop_last=len(keys) > bs)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, len(dl)))
    sc = torch.cuda.amp.GradScaler()
    w = torch.tensor(cls_w, dtype=torch.float32, device=DEV)
    for _ in range(epochs):
        m.train()
        for x, yy in dl:
            x, yy = x.to(DEV), yy.to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = F.cross_entropy(m(x), yy, weight=w, label_smoothing=0.05)
            sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step()
    m.eval()
    return m


@torch.no_grad()
def predict_cls(m, keys, y, boxes, bs=32):
    dl = DataLoader(ClsDS(keys, y, boxes, False), batch_size=bs, num_workers=8)
    out = []
    with torch.cuda.amp.autocast():
        for x, _ in dl:
            out.append(m(x.to(DEV)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    idx = json.load(open(OUT / "prep2_index.json"))
    meta = pd.read_excel(ROOT / "data/Tibial Plateau Fracture Metadata.xlsx").set_index("Patient ID")
    pat = {int(p): str(meta.loc[p, SCH]).replace("Normal", "NTPF") for p in meta.index}
    bx = boxes_from_index(idx)

    rows = [r for r in idx if r.get("has_roi") and pat[r["pid"]] != "NTPF"
            and LBL[r["label"]] != "NTPF" and r["key"] in bx]
    keys = np.array([r["key"] for r in rows])
    pids = np.array([r["pid"] for r in rows])
    y = np.array([CLS6.index(LBL[r["label"]]) for r in rows])
    boxes = np.stack([bx[k] for k in keys])
    print(f"[auto ROI] patients {len(set(pids.tolist()))} · images {len(keys)}", flush=True)

    allrows, summary = [], []
    for seed in args.seeds:
        t0 = time.time()
        pu = sorted(set(pids.tolist()))
        py = np.array([y[pids == p][0] for p in pu])
        cnt = np.bincount(py, minlength=6).astype(float)
        cls_w = (cnt.sum() / (6 * np.maximum(cnt, 1))).tolist()
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        ious = []
        for k, (tr, te) in enumerate(cv.split(keys, y, groups=pids), 1):
            # 1) bbox regressor — training fold only
            reg = train_boxreg(keys[tr], boxes[tr], seed, k)
            pb_te = predict_boxes(reg, keys[te])
            pb_tr = predict_boxes(reg, keys[tr])
            del reg; torch.cuda.empty_cache()
            # record IoU
            for a, b in zip(pb_te, boxes[te]):
                iy0, iy1 = max(a[0], b[0]), min(a[1], b[1])
                ix0, ix1 = max(a[2], b[2]), min(a[3], b[3])
                inter = max(0, iy1-iy0) * max(0, ix1-ix0)
                ua = (a[1]-a[0])*(a[3]-a[2]); ub = (b[1]-b[0])*(b[3]-b[2])
                ious.append(inter / (ua+ub-inter) if (ua+ub-inter) > 0 else 0)
            # 2) classifier — trained on predicted bboxes too, to match deployment conditions
            clf = train_cls(keys[tr], y[tr], pb_tr, cls_w, seed, k, args.epochs)
            lg = predict_cls(clf, keys[te], y[te], pb_te)
            del clf; torch.cuda.empty_cache()
            for j, gi in enumerate(te):
                allrows.append({"seed": seed, "fold": k, "key": keys[gi], "pid": int(pids[gi]),
                                "true": CLS6[y[gi]],
                                **{f"logit_{c}": float(lg[j, ci]) for ci, c in enumerate(CLS6)}})
            d = pd.DataFrame(lg); d["pid"] = pids[te]; d["y"] = y[te]
            g = d.groupby("pid")
            fb = balanced_accuracy_score(g["y"].first().values,
                                         g[list(range(6))].mean().values.argmax(1))
            print(f"  [autoROI] seed {seed} fold {k}/5 bacc={fb:.3f}", flush=True)
        df = pd.DataFrame([r for r in allrows if r["seed"] == seed])
        g = df.groupby("pid")
        pl = g[[f"logit_{c}" for c in CLS6]].mean().values.argmax(1)
        pt = np.array([CLS6.index(v) for v in g["true"].first().values])
        summary.append({"config": "K_6cls_autoroi", "seed": seed,
                        "balanced_acc": round(float(balanced_accuracy_score(pt, pl)), 4),
                        "macro_f1": round(float(f1_score(pt, pl, average="macro")), 4),
                        "accuracy": round(float((pt == pl).mean()), 4),
                        "mean_box_iou": round(float(np.mean(ious)), 4),
                        "chance": round(1/6, 4), "classes": CLS6,
                        "confusion": confusion_matrix(pt, pl, labels=list(range(6))).tolist(),
                        "minutes": round((time.time()-t0)/60, 1)})
        print(f"■ [autoROI] seed {seed}: bacc {summary[-1]['balanced_acc']:.3f} · "
              f"box IoU {summary[-1]['mean_box_iou']:.3f} · {summary[-1]['minutes']} min", flush=True)

    pd.DataFrame(allrows).to_csv(OUT / "exp11_autoroi_logits.csv", index=False)
    with open(OUT / "exp11_autoroi.jsonl", "w") as fh:
        for s in summary:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    b = [s["balanced_acc"] for s in summary]
    print(f"\n■ auto-ROI six-class: {len(b)} seeds · mean {np.mean(b):.4f} · "
          f"SD {np.std(b, ddof=1) if len(b)>1 else 0:.4f} · values {b}", flush=True)
    print(f"   (oracle ROI 0.345 · centre crop 0.260 · background only 0.257 · chance 0.167)", flush=True)


if __name__ == "__main__":
    main()
