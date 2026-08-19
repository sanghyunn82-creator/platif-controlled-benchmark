#!/usr/bin/env python3
"""P4-1 · Six-class training with GPU-side augmentation.

On the shared GPU host another job saturated all 128 cores, so the DataLoader could not keep the GPU fed (GPU at 0%, ten minutes per fold).
Here every augmentation runs on the GPU and the DataLoader is removed. **The protocol is unchanged** —
split, epochs, batch size, optimiser, schedule, loss, and the probabilities and ranges of every
augmentation are identical to the original; only the device the operations run on differs.
"""
import argparse, json, os, random, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
torch.set_num_threads(2)

ROOT = Path("/workspace/platif_p4"); PREP = ROOT / "prep2"; OUT = ROOT / "results"
SCH = "Fracture Type of  Schatzker Classification"
LBL = {1: "Type 1", 2: "Type 2", 3: "Type 3", 4: "Type 4", 5: "Type 5", 6: "Type 6", 7: "NTPF"}
CLS6 = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6"]
DEV, MEAN, STD = "cuda", 0.449, 0.226


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def augment(x, hflip, gen):
    """x: (B,448,448) float [0,1] on GPU. Same probabilities and ranges as the original."""
    B = x.shape[0]
    if hflip:
        m = torch.rand(B, device=DEV, generator=gen) < 0.5
        if m.any(): x[m] = torch.flip(x[m], dims=[2])
    m = torch.rand(B, device=DEV, generator=gen) < 0.5
    if m.any():
        s = torch.empty(int(m.sum()), 1, 1, device=DEV).uniform_(0.9, 1.1, generator=gen)
        x[m] = (x[m] * s).clamp(0, 1)
    m = torch.rand(B, device=DEV, generator=gen) < 0.3
    idx = torch.nonzero(m).flatten()
    for i in idx.tolist():                      # +/-12 px shift with zero padding
        dy, dx = (torch.randint(-12, 13, (2,), device=DEV, generator=gen)).tolist()
        x[i] = torch.roll(x[i], shifts=(dy, dx), dims=(0, 1))
        if dy > 0: x[i, :dy] = 0
        elif dy < 0: x[i, dy:] = 0
        if dx > 0: x[i, :, :dx] = 0
        elif dx < 0: x[i, :, dx:] = 0
    return x


def make_model(n_cls):
    import torchvision
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, n_cls); return m.to(DEV)


def run_fold(Xtr, ytr, Xte, yte, n_cls, hflip, epochs, cls_w, seed, fold, bs=16, lr=3e-4):
    set_seed(seed * 100 + fold)
    gen = torch.Generator(device=DEV); gen.manual_seed(seed * 100 + fold)
    m = make_model(n_cls)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    nstep = max(1, len(Xtr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * nstep)
    w = torch.tensor(cls_w, dtype=torch.float32, device=DEV)
    scaler = torch.amp.GradScaler("cuda")
    curve = []
    for ep in range(1, epochs + 1):
        m.train(); perm = torch.randperm(len(Xtr), device=DEV, generator=gen)
        corr = tot = 0
        for b in range(nstep):
            sel = perm[b * bs:(b + 1) * bs]
            x = Xtr[sel].float() / 255.0
            x = augment(x, hflip, gen)
            x = ((x - MEAN) / STD).unsqueeze(1).repeat(1, 3, 1, 1)
            y = ytr[sel]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                o = m(x); loss = F.cross_entropy(o, y, weight=w, label_smoothing=0.05)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            corr += (o.argmax(1) == y).sum().item(); tot += len(y)
        curve.append({"seed": seed, "fold": fold, "epoch": ep, "train_acc": corr / max(tot, 1)})
    m.eval(); lg = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for b in range(0, len(Xte), bs):
            x = Xte[b:b + bs].float() / 255.0
            x = ((x - MEAN) / STD).unsqueeze(1).repeat(1, 3, 1, 1)
            lg.append(m(x).float().cpu())
    del m; torch.cuda.empty_cache()
    return torch.cat(lg).numpy(), curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hflip", type=int, default=0)
    ap.add_argument("--tag", default="G6_roi_noflip")
    a = ap.parse_args()
    idx = json.load(open(OUT / "prep2_index.json"))
    meta = pd.read_excel(ROOT / "data/Tibial Plateau Fracture Metadata.xlsx").set_index("Patient ID")
    pat = {int(p): str(meta.loc[p, SCH]).replace("Normal", "NTPF") for p in meta.index}
    rows = [r for r in idx if r["has_roi"] and pat[r["pid"]] != "NTPF" and LBL[r["label"]] != "NTPF"]
    pids = np.array([r["pid"] for r in rows]); keys = [r["key"] for r in rows]
    y = np.array([CLS6.index(LBL[r["label"]]) for r in rows])
    print(f"[{a.tag}] patients {len(set(pids.tolist()))} · images {len(keys)} · hflip={bool(a.hflip)}", flush=True)
    X = torch.from_numpy(np.stack([np.load(PREP / "roi" / f"{k}.npy") for k in keys])).to(DEV)
    Y = torch.from_numpy(y).long().to(DEV)
    allrows, allcur, summ = [], [], []
    for seed in a.seeds:
        t0 = time.time()
        pu = sorted(set(pids.tolist())); py = np.array([y[pids == p][0] for p in pu])
        cnt = np.bincount(py, minlength=6).astype(float)
        cls_w = (cnt.sum() / (6 * np.maximum(cnt, 1))).tolist()
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        fb = []
        for k, (tr, te) in enumerate(cv.split(np.zeros(len(keys)), y, groups=pids), 1):
            lg, cur = run_fold(X[tr], Y[tr], X[te], Y[te], 6, bool(a.hflip), a.epochs, cls_w, seed, k)
            for j, gi in enumerate(te):
                allrows.append({"seed": seed, "fold": k, "key": keys[gi], "pid": int(pids[gi]),
                                "true": CLS6[y[gi]],
                                **{f"logit_{c}": float(lg[j, ci]) for ci, c in enumerate(CLS6)}})
            allcur += cur
            d = pd.DataFrame(lg); d["pid"] = pids[te]; d["y"] = y[te]; g = d.groupby("pid")
            v = balanced_accuracy_score(g["y"].first().values, g[list(range(6))].mean().values.argmax(1))
            fb.append(round(float(v), 4))
            print(f"  [{a.tag}] seed {seed} fold {k}/5 bacc={v:.3f} ({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame([r for r in allrows if r["seed"] == seed]); g = df.groupby("pid")
        pl = g[[f"logit_{c}" for c in CLS6]].mean().values.argmax(1)
        pt = np.array([CLS6.index(v) for v in g["true"].first().values])
        summ.append({"config": a.tag, "seed": seed, "arch": "resnet50", "prep": "prep2",
                     "label_source": "image", "n_patients": int(len(pt)), "chance": 0.1667,
                     "balanced_acc": round(float(balanced_accuracy_score(pt, pl)), 4),
                     "macro_f1": round(float(f1_score(pt, pl, average="macro")), 4),
                     "accuracy": round(float((pt == pl).mean()), 4), "fold_baccs": fb,
                     "final_train_acc": round(float(np.mean([c["train_acc"] for c in allcur
                                              if c["seed"] == seed and c["epoch"] == a.epochs])), 4),
                     "confusion": confusion_matrix(pt, pl, labels=list(range(6))).tolist(),
                     "classes": CLS6})
        print(f"■ [{a.tag}] seed {seed}: bacc {summ[-1]['balanced_acc']:.3f} "
              f"· train_acc {summ[-1]['final_train_acc']:.3f} · {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(allrows).to_csv(OUT / f"exp08_{a.tag}_resnet50_prep2_image_logits.csv", index=False)
        pd.DataFrame(allcur).to_csv(OUT / f"exp08_{a.tag}_resnet50_prep2_image_curves.csv", index=False)
        with open(OUT / f"exp08_{a.tag}_summary.jsonl", "a") as fh:
            fh.write(json.dumps(summ[-1], ensure_ascii=False) + "\n")
    b = [s["balanced_acc"] for s in summ]
    print(f"\n■ {a.tag}: {len(b)} seeds · mean {np.mean(b):.4f} · SD "
          f"{np.std(b, ddof=1) if len(b) > 1 else 0:.4f} · {b}", flush=True)


if __name__ == "__main__":
    main()
