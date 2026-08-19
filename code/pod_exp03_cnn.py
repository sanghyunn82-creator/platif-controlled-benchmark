#!/usr/bin/env python3
"""
P4-1 · Experiment 3: transfer-learning CNN baseline — **run serially on the GPU host.**

The aim is not to maximise performance but to **measure honestly what this task actually supports**,
so control conditions are run alongside.

  A. seven-class · roi (bone bbox crop)        <- primary result
  B. seven-class · full (whole image)          <- how much do background cues add?
  C. binary NTPF vs fracture · roi
  D. binary · full
  E. seven-class · roi · **horizontal flip off** <- compared with A, exposes any left/right shortcut
  F. seven-class · roi · **permuted labels**     <- null level; if this is not ~0.14 (=1/7) the pipeline leaks

Discipline:
  - **patient-level splits** (StratifiedGroupKFold); labels from the xlsx of record (matching descriptor Table 1).
  - evaluation is also **patient level**: the logits of a patient's images are averaged into one vote.
  - class imbalance is handled by loss weighting; we read **balanced accuracy and macro-F1**, not accuracy.
  - results are written to file after every fold, so nothing is lost if the run dies.
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
MEAN, STD = 0.449, 0.226   # ImageNet grey-scale equivalent


class DS(Dataset):
    def __init__(self, keys, labels, variant, train, hflip):
        self.keys, self.labels, self.variant = keys, labels, variant
        self.train, self.hflip = train, hflip

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        a = np.load(PREP / self.variant / f"{self.keys[i]}.npy").astype(np.float32) / 255.0
        if self.train:
            if self.hflip and np.random.rand() < 0.5:
                a = a[:, ::-1].copy()
            # light augmentation only: strong augmentation hurts at this sample size
            if np.random.rand() < 0.5:
                s = np.random.uniform(0.9, 1.1)
                a = np.clip(a * s, 0, 1)
            if np.random.rand() < 0.3:
                k = np.random.randint(-12, 13)
                a = np.roll(a, k, axis=np.random.randint(2))
        a = (a - MEAN) / STD
        return torch.from_numpy(a)[None].repeat(3, 1, 1), self.labels[i]


def make_model(n_cls):
    import torchvision
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, n_cls)
    return m.to(DEV)


def run_fold(tr_keys, tr_y, te_keys, te_y, te_pid, n_cls, variant, hflip,
             epochs, cls_w, bs=16, lr=3e-4):
    tr = DataLoader(DS(tr_keys, tr_y, variant, True, hflip), batch_size=bs, shuffle=True,
                    num_workers=8, pin_memory=True, drop_last=len(tr_keys) > bs)
    te = DataLoader(DS(te_keys, te_y, variant, False, hflip), batch_size=bs, shuffle=False,
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
    logits = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x, _ in te:
            logits.append(m(x.to(DEV)).float().cpu())
    logits = torch.cat(logits).numpy()
    del m; torch.cuda.empty_cache()
    # average logits within a patient -> one vote per patient
    df = pd.DataFrame(logits)
    df["pid"] = te_pid; df["y"] = te_y
    g = df.groupby("pid")
    pred = g[list(range(n_cls))].mean().values.argmax(1)
    true = g["y"].first().values
    return true, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    idx = json.load(open(OUT / "prep_index.json"))
    meta = pd.read_excel(ROOT / "data/Tibial Plateau Fracture Metadata.xlsx").set_index("Patient ID")
    ylab = {int(p): str(meta.loc[p, SCH]).replace("Normal", "NTPF") for p in meta.index}
    CLS7 = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6", "NTPF"]
    CLS2 = ["Fracture", "NTPF"]

    CONFIGS = [
        ("A_7cls_roi",        "roi",  7, True,  False),
        ("B_7cls_full",       "full", 7, True,  False),
        ("C_bin_roi",         "roi",  2, True,  False),
        ("D_bin_full",        "full", 2, True,  False),
        ("E_7cls_roi_noflip", "roi",  7, False, False),
        ("F_7cls_roi_shuffle","roi",  7, True,  True),
        ("J_bin_roi_shuffle", "roi",  2, True,  True),
    ]
    if args.only:
        CONFIGS = [c for c in CONFIGS if c[0] in args.only.split(",")]

    resfile = OUT / "exp03_cnn.jsonl"
    for name, variant, n_cls, hflip, shuffle in CONFIGS:
        t0 = time.time()
        classes = CLS7 if n_cls == 7 else CLS2
        rows = [r for r in idx if (variant == "full" or r["has_roi"])]
        pids = np.array([r["pid"] for r in rows])
        keys = np.array([r["key"] for r in rows])
        plab = {p: ylab[p] for p in set(pids.tolist())}
        if shuffle:
            ps = sorted(plab); vals = [plab[p] for p in ps]
            rng = np.random.default_rng(0); rng.shuffle(vals)
            plab = dict(zip(ps, vals))
        if n_cls == 2:
            plab = {p: ("NTPF" if v == "NTPF" else "Fracture") for p, v in plab.items()}
        y = np.array([classes.index(plab[p]) for p in pids])
        py = np.array([classes.index(plab[p]) for p in sorted(set(pids.tolist()))])
        cnt = np.bincount(py, minlength=n_cls).astype(float)
        cls_w = (cnt.sum() / (n_cls * np.maximum(cnt, 1))).tolist()

        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        T, P = [], []
        for k, (tr, te) in enumerate(cv.split(keys, y, groups=pids), 1):
            t, p = run_fold(keys[tr], y[tr], keys[te], y[te], pids[te],
                            n_cls, variant, hflip, args.epochs, cls_w)
            T.append(t); P.append(p)
            print(f"  [{name}] fold {k}/5  n_test={len(t)}  "
                  f"bacc={balanced_accuracy_score(t, p):.3f}", flush=True)
        T, P = np.concatenate(T), np.concatenate(P)
        rec = {
            "config": name, "variant": variant, "n_classes": n_cls, "hflip": hflip,
            "shuffled": shuffle, "epochs": args.epochs, "n_patients": int(len(T)),
            "chance": round(1 / n_cls, 4),
            "balanced_acc": round(float(balanced_accuracy_score(T, P)), 4),
            "macro_f1": round(float(f1_score(T, P, average="macro")), 4),
            "accuracy": round(float((T == P).mean()), 4),
            "classes": classes,
            "confusion": confusion_matrix(T, P, labels=list(range(n_cls))).tolist(),
            "per_class_f1": [round(float(x), 4) for x in
                             f1_score(T, P, average=None, labels=list(range(n_cls)))],
            "minutes": round((time.time() - t0) / 60, 1),
        }
        with open(resfile, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"■ {name}: balanced acc {rec['balanced_acc']:.3f} · macro-F1 {rec['macro_f1']:.3f} "
              f"· acc {rec['accuracy']:.3f} (chance {rec['chance']}) · {rec['minutes']} min", flush=True)
    print(f"written: {resfile}", flush=True)


if __name__ == "__main__":
    main()
