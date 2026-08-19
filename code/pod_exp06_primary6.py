#!/usr/bin/env python3
"""
P4-1 · Experiment 6: **six-class typing among the 128 fracture patients** — the primary analysis recommended by design v2. GPU host.

Why this is the primary analysis (established by the earlier experiments):
  - non-imaging metadata alone reaches 0.798 on the binary task (NTPF vs fracture) -> that boundary is contaminated
  - but once the shortcut is removed, discriminating fracture types gives 0.165 (chance 0.143, p=0.214) -> **no leakage**
  - so excluding NTPF leaves a clean benchmark

Two controls are run alongside it.
  G: roi crop (mask bbox) — the primary result
  H: **fixed centre crop** (no mask used) — tests whether the roi advantage comes from annotator information.
     exp05 showed mask geometry did not leak the seven-class label (0.137/0.144), but whether the crop
     helps because it frames anatomy better or because annotation leaks is a separate question. This is the answer.
  I: label-permuted null

**Per-patient predictions are saved** so that post-hoc analyses (NTPF stratification and so on) need no retraining.
(This fixes the design flaw of exp03.)
"""
import argparse
import json
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
PREP = ROOT / "prep"
OUT = ROOT / "results"
SCH = "Fracture Type of  Schatzker Classification"
DEV = "cuda"
MEAN, STD = 0.449, 0.226
CLS6 = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6"]


class DS(Dataset):
    def __init__(self, keys, labels, variant, train, center_crop=False):
        self.keys, self.labels, self.variant = keys, labels, variant
        self.train, self.center_crop = train, center_crop

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        a = np.load(PREP / self.variant / f"{self.keys[i]}.npy").astype(np.float32) / 255.0
        if self.center_crop:
            # fixed centre crop that uses no mask (take the middle 60% and resize back to 448)
            s = a.shape[0]
            c = int(s * 0.20)
            a = a[c:s - c, c:s - c]
            a = np.array(torch.nn.functional.interpolate(
                torch.from_numpy(a)[None, None], size=(448, 448),
                mode="bilinear", align_corners=False)[0, 0])
        if self.train:
            if np.random.rand() < 0.5:
                a = a[:, ::-1].copy()
            if np.random.rand() < 0.5:
                a = np.clip(a * np.random.uniform(0.9, 1.1), 0, 1)
            if np.random.rand() < 0.3:
                a = np.roll(a, np.random.randint(-12, 13), axis=np.random.randint(2))
        a = (a - MEAN) / STD
        return torch.from_numpy(a)[None].repeat(3, 1, 1), self.labels[i]


def make_model(n_cls):
    import torchvision
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, n_cls)
    return m.to(DEV)


def run_fold(trk, try_, tek, tey, tepid, n_cls, variant, epochs, cls_w, center_crop, bs=16, lr=3e-4):
    tr = DataLoader(DS(trk, try_, variant, True, center_crop), batch_size=bs, shuffle=True,
                    num_workers=8, pin_memory=True, drop_last=len(trk) > bs)
    te = DataLoader(DS(tek, tey, variant, False, center_crop), batch_size=bs, shuffle=False,
                    num_workers=8, pin_memory=True)
    m = make_model(n_cls)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, len(tr)))
    w = torch.tensor(cls_w, dtype=torch.float32, device=DEV)
    scaler = torch.cuda.amp.GradScaler()
    for _ in range(epochs):
        m.train()
        for x, y in tr:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = F.cross_entropy(m(x), y, weight=w, label_smoothing=0.05)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
    m.eval()
    lg = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x, _ in te:
            lg.append(m(x.to(DEV)).float().cpu())
    lg = torch.cat(lg).numpy()
    del m; torch.cuda.empty_cache()
    df = pd.DataFrame(lg); df["pid"] = tepid; df["y"] = tey
    g = df.groupby("pid")
    logits = g[list(range(n_cls))].mean()
    return logits.index.values, logits.values, g["y"].first().values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    idx = json.load(open(OUT / "prep_index.json"))
    meta = pd.read_excel(ROOT / "data/Tibial Plateau Fracture Metadata.xlsx").set_index("Patient ID")
    ylab = {int(p): str(meta.loc[p, SCH]).replace("Normal", "NTPF") for p in meta.index}

    CONFIGS = [
        ("G_6cls_roi",          "roi",  False, False),
        ("H_6cls_centercrop",   "full", True,  False),
        ("I_6cls_roi_shuffle",  "roi",  False, True),
    ]

    for name, variant, center_crop, shuffle in CONFIGS:
        t0 = time.time()
        # fracture patients only — NTPF excluded
        rows = [r for r in idx if (variant == "full" or r["has_roi"]) and ylab[r["pid"]] != "NTPF"]
        pids = np.array([r["pid"] for r in rows])
        keys = np.array([r["key"] for r in rows])
        plab = {p: ylab[p] for p in set(pids.tolist())}
        if shuffle:
            ps = sorted(plab); vals = [plab[p] for p in ps]
            np.random.default_rng(0).shuffle(vals); plab = dict(zip(ps, vals))
        y = np.array([CLS6.index(plab[p]) for p in pids])
        py = np.array([CLS6.index(plab[p]) for p in sorted(set(pids.tolist()))])
        cnt = np.bincount(py, minlength=6).astype(float)
        cls_w = (cnt.sum() / (6 * np.maximum(cnt, 1))).tolist()
        print(f"[{name}] patients {len(set(pids.tolist()))} · images {len(keys)} · "
              f"distribution {dict(zip(CLS6, cnt.astype(int)))}", flush=True)

        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        allp, alll, ally, foldid = [], [], [], []
        for k, (tr, te) in enumerate(cv.split(keys, y, groups=pids), 1):
            p, lg, t = run_fold(keys[tr], y[tr], keys[te], y[te], pids[te],
                                6, variant, args.epochs, cls_w, center_crop)
            allp.append(p); alll.append(lg); ally.append(t); foldid += [k] * len(p)
            print(f"  [{name}] fold {k}/5 n={len(t)} bacc={balanced_accuracy_score(t, lg.argmax(1)):.3f}",
                  flush=True)
        P = np.concatenate(allp); L = np.concatenate(alll); T = np.concatenate(ally)
        pred = L.argmax(1)
        # save per-patient predictions for post-hoc analysis
        pd.DataFrame({"pid": P, "fold": foldid, "true": T, "pred": pred,
                      **{f"logit_{c}": L[:, i] for i, c in enumerate(CLS6)}}) \
          .to_csv(OUT / f"exp06_{name}_predictions.csv", index=False)
        rec = {"config": name, "variant": variant, "center_crop": center_crop, "shuffled": shuffle,
               "n_patients": int(len(T)), "chance": round(1 / 6, 4),
               "balanced_acc": round(float(balanced_accuracy_score(T, pred)), 4),
               "macro_f1": round(float(f1_score(T, pred, average="macro")), 4),
               "accuracy": round(float((T == pred).mean()), 4),
               "classes": CLS6,
               "confusion": confusion_matrix(T, pred, labels=list(range(6))).tolist(),
               "per_class_f1": [round(float(x), 4) for x in
                                f1_score(T, pred, average=None, labels=list(range(6)))],
               "fold_baccs": [round(float(balanced_accuracy_score(T[np.array(foldid) == k],
                                                                 pred[np.array(foldid) == k])), 4)
                              for k in range(1, 6)],
               "minutes": round((time.time() - t0) / 60, 1)}
        with open(OUT / "exp06_primary6.jsonl", "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"■ {name}: bacc {rec['balanced_acc']:.3f} · macro-F1 {rec['macro_f1']:.3f} · "
              f"acc {rec['accuracy']:.3f} (chance 0.167) · {rec['minutes']} min", flush=True)
    print("written", flush=True)


if __name__ == "__main__":
    main()
