#!/usr/bin/env python3
"""
P4-1 · Experiment 8 — **redesigned after the adversarial audit.** Run on the GPU host.

Everything the audit flagged in v1 (exp03/exp06) is fixed here.

  1. **No random seeding** -> set_seed() fixes random/np/torch/cuda/cudnn.
     --seeds runs several seeds and the between-seed SD is reported.
  2. **Fold values and per-patient predictions were not saved** -> fold values, **per-image logits**
     and per-patient predictions are all written to CSV, so bootstrap, McNemar and stratified
  3. **No training curves** -> train loss/acc are logged every epoch, so convergence at 350 steps
     and whether the shuffled run memorises the training set (i.e. capacity is sufficient) are visible.
  4. **5 contralateral normal knees were trained as fractures** -> --label-source image uses the
     image-level label; comparing with `patient` (the v1 behaviour) gives a sensitivity check.
  5. **np.roll augmentation** was circular, not a translation, and fabricated structure at the
     opposite edge -> replaced by a true zero-filled shift.
  6. Preprocessing defaults to **prep2 (anti-aliased)**.

Example:
  pod_exp08_seeded.py --config G6_roi --seeds 0 1 2 3 4
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
OUT = ROOT / "results"
SCH = "Fracture Type of  Schatzker Classification"
DEV = "cuda"
MEAN, STD = 0.449, 0.226
CLS6 = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6"]
CLS7 = CLS6 + ["NTPF"]
CLS2 = ["Fracture", "NTPF"]
LBL = {1: "Type 1", 2: "Type 2", 3: "Type 3", 4: "Type 4", 5: "Type 5", 6: "Type 6", 7: "NTPF"}


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def shift(a, dy, dx):
    """True zero-filled translation. np.roll is circular and fabricates edge structure."""
    out = np.zeros_like(a)
    ys, ye = max(0, dy), min(a.shape[0], a.shape[0] + dy)
    xs, xe = max(0, dx), min(a.shape[1], a.shape[1] + dx)
    out[ys:ye, xs:xe] = a[ys - dy:ye - dy, xs - dx:xe - dx]
    return out


class DS(Dataset):
    def __init__(self, keys, labels, prep, variant, train, hflip):
        self.keys, self.labels = keys, labels
        self.prep, self.variant, self.train, self.hflip = prep, variant, train, hflip

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        a = np.load(self.prep / self.variant / f"{self.keys[i]}.npy").astype(np.float32) / 255.0
        if self.train:
            if self.hflip and np.random.rand() < 0.5:
                a = a[:, ::-1].copy()
            if np.random.rand() < 0.5:
                a = np.clip(a * np.random.uniform(0.9, 1.1), 0, 1)
            if np.random.rand() < 0.3:
                a = shift(a, np.random.randint(-12, 13), np.random.randint(-12, 13))
        a = (a - MEAN) / STD
        return torch.from_numpy(a)[None].repeat(3, 1, 1), self.labels[i]


def make_model(n_cls, arch="resnet50"):
    import torchvision
    if arch == "resnet50":
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, n_cls)
    elif arch == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(
            weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
    else:
        raise ValueError(arch)
    return m.to(DEV)


def run_fold(trk, try_, tek, tey, tepid, n_cls, prep, variant, hflip, epochs,
             cls_w, seed, fold, arch, bs=16, lr=3e-4):
    set_seed(seed * 100 + fold)
    g = torch.Generator(); g.manual_seed(seed * 100 + fold)
    tr = DataLoader(DS(trk, try_, prep, variant, True, hflip), batch_size=bs, shuffle=True,
                    num_workers=8, pin_memory=True, drop_last=len(trk) > bs, generator=g,
                    worker_init_fn=lambda w: np.random.seed(seed * 1000 + fold * 10 + w))
    te = DataLoader(DS(tek, tey, prep, variant, False, hflip), batch_size=bs, shuffle=False,
                    num_workers=8, pin_memory=True)
    m = make_model(n_cls, arch)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, len(tr)))
    w = torch.tensor(cls_w, dtype=torch.float32, device=DEV)
    scaler = torch.cuda.amp.GradScaler()
    curve = []
    for ep in range(epochs):
        m.train(); tot = n = corr = 0
        for x, y in tr:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                o = m(x); loss = F.cross_entropy(o, y, weight=w, label_smoothing=0.05)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y); corr += (o.argmax(1) == y).sum().item()
        curve.append({"epoch": ep + 1, "train_loss": tot / n, "train_acc": corr / n})
    m.eval(); lg = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x, _ in te:
            lg.append(m(x.to(DEV)).float().cpu())
    lg = torch.cat(lg).numpy()
    del m; torch.cuda.empty_cache()
    return lg, curve


CONFIGS = {
    # name: (n_cls, variant, subset, hflip, shuffle)
    "G6_roi":       (6, "roi",    "fracture", True,  False),
    "G6_center":    (6, "center", "fracture", True,  False),
    "G6_bg":        (6, "bg",     "fracture", True,  False),
    "G6_shuffle":   (6, "roi",    "fracture", True,  True),
    "C2_roi":       (2, "roi",    "all",      True,  False),
    "C2_shuffle":   (2, "roi",    "all",      True,  True),
    "A7_roi":       (7, "roi",    "all",      True,  False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--prep", default="prep2")
    ap.add_argument("--label-source", choices=["image", "patient"], default="image")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prep = ROOT / args.prep
    idxf = OUT / ("prep2_index.json" if args.prep == "prep2" else "prep_index.json")
    idx = json.load(open(idxf))

    n_cls, variant, subset, hflip, shuffle = CONFIGS[args.config]
    classes = {2: CLS2, 6: CLS6, 7: CLS7}[n_cls]
    meta = pd.read_excel(ROOT / "data/Tibial Plateau Fracture Metadata.xlsx").set_index("Patient ID")
    pat_lab = {int(p): str(meta.loc[p, SCH]).replace("Normal", "NTPF") for p in meta.index}

    rows = [r for r in idx if (variant == "full" or r["has_roi"])]
    # image-level label vs patient-level label
    def lab_of(r):
        return LBL[r["label"]] if args.label_source == "image" else pat_lab[r["pid"]]
    if subset == "fracture":
        rows = [r for r in rows if pat_lab[r["pid"]] != "NTPF"]
        if args.label_source == "image":
            rows = [r for r in rows if lab_of(r) != "NTPF"]   # drop contralateral normal knees
    if n_cls == 2:
        conv = lambda v: ("NTPF" if v == "NTPF" else "Fracture")
    else:
        conv = lambda v: v

    pids = np.array([r["pid"] for r in rows])
    keys = np.array([r["key"] for r in rows])
    ilabs = np.array([conv(lab_of(r)) for r in rows])
    print(f"[{args.config}] patients {len(set(pids.tolist()))} · images {len(keys)} · "
          f"label source {args.label_source} · preprocessing {args.prep} · distribution "
          f"{dict(zip(*np.unique(ilabs, return_counts=True)))}", flush=True)

    all_rows, all_curves, summary = [], [], []
    for seed in args.seeds:
        lab = ilabs.copy()
        if shuffle:
            # shuffle at patient level (per-image shuffling breaks within-patient consistency)
            ps = sorted(set(pids.tolist()))
            vals = [lab[pids == p][0] for p in ps]
            rng = np.random.default_rng(seed); rng.shuffle(vals)
            mp = dict(zip(ps, vals)); lab = np.array([mp[p] for p in pids])
        y = np.array([classes.index(v) for v in lab])
        # labels used for patient-level stratification
        pu = sorted(set(pids.tolist()))
        py = np.array([y[pids == p][0] for p in pu])
        cnt = np.bincount(py, minlength=n_cls).astype(float)
        cls_w = (cnt.sum() / (n_cls * np.maximum(cnt, 1))).tolist()

        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_bacc = []
        for k, (tr, te) in enumerate(cv.split(keys, y, groups=pids), 1):
            lg, curve = run_fold(keys[tr], y[tr], keys[te], y[te], pids[te], n_cls,
                                 prep, variant, hflip, args.epochs, cls_w, seed, k, args.arch)
            for j, gi in enumerate(te):
                all_rows.append({"seed": seed, "fold": k, "key": keys[gi], "pid": int(pids[gi]),
                                 "true": classes[y[gi]],
                                 **{f"logit_{c}": float(lg[j, ci]) for ci, c in enumerate(classes)}})
            for c in curve:
                all_curves.append({"seed": seed, "fold": k, **c})
            # aggregate to patients within the fold
            d = pd.DataFrame(lg); d["pid"] = pids[te]; d["y"] = y[te]
            gg = d.groupby("pid")
            fb = balanced_accuracy_score(gg["y"].first().values,
                                         gg[list(range(n_cls))].mean().values.argmax(1))
            fold_bacc.append(round(float(fb), 4))
            print(f"  [{args.config}] seed {seed} fold {k}/5 bacc={fb:.3f} "
                  f"(final train_acc {curve[-1]['train_acc']:.3f})", flush=True)
        # aggregate over the seed
        df = pd.DataFrame([r for r in all_rows if r["seed"] == seed])
        gg = df.groupby("pid")
        pl = gg[[f"logit_{c}" for c in classes]].mean().values.argmax(1)
        pt = np.array([classes.index(v) for v in gg["true"].first().values])
        summary.append({"config": args.config, "seed": seed, "arch": args.arch,
                        "prep": args.prep, "label_source": args.label_source,
                        "n_patients": int(len(pt)), "chance": round(1 / n_cls, 4),
                        "balanced_acc": round(float(balanced_accuracy_score(pt, pl)), 4),
                        "macro_f1": round(float(f1_score(pt, pl, average="macro")), 4),
                        "accuracy": round(float((pt == pl).mean()), 4),
                        "fold_baccs": fold_bacc,
                        "final_train_acc": round(float(np.mean(
                            [c["train_acc"] for c in all_curves
                             if c["seed"] == seed and c["epoch"] == args.epochs])), 4),
                        "confusion": confusion_matrix(pt, pl, labels=list(range(n_cls))).tolist(),
                        "classes": classes})
        print(f"■ [{args.config}] seed {seed}: bacc {summary[-1]['balanced_acc']:.3f} · "
              f"train_acc {summary[-1]['final_train_acc']:.3f}", flush=True)

    tag = f"{args.config}_{args.arch}_{args.prep}_{args.label_source}"
    pd.DataFrame(all_rows).to_csv(OUT / f"exp08_{tag}_logits.csv", index=False)
    pd.DataFrame(all_curves).to_csv(OUT / f"exp08_{tag}_curves.csv", index=False)
    with open(OUT / "exp08_summary.jsonl", "a") as fh:
        for s in summary:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    b = [s["balanced_acc"] for s in summary]
    print(f"\n■ {tag}: {len(b)} seeds · mean bacc {np.mean(b):.4f} · "
          f"between-seed SD {np.std(b, ddof=1) if len(b) > 1 else 0:.4f} · values {b}", flush=True)


if __name__ == "__main__":
    main()
