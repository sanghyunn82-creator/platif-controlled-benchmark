#!/usr/bin/env python3
"""P4-1 · exp17 — the ablation pair: the tibia erased, and the tibia alone.

Review point: erasing the bone and erasing the outline look arbitrary, and their results look
predictable. Until now the design had only `bg` (tibia erased) and not its **complement**. Saying
"half survives erasure" also says "the other half is in the bone", and that was never shown directly.

  tib : g[~bw] = 0   only the tibia is kept
  bg  : g[bw]  = 0   the tibia is erased
Both go through the same fit() and share a field of view (full frame fitted to 448), so they are an
**exact complement** whose sum is the full image; tib vs bg is a single-variable contrast.
(roi is a mask-bbox crop and therefore differs in field of view — reported for reference only.)

How to read it:
  tib ~ roi  -> the signal is fully present inside the bone; the residual of bg is **redundancy**, not a substitute.
  tib << roi -> the signal was never in the plateau to begin with.

Output: results/exp17_ablation_pair.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score

R = Path("/workspace/platif_p4/results")
CLS6 = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6"]
LOGITS = [f"logit_{c}" for c in CLS6]
ROM = dict(zip(CLS6, ["I", "II", "III", "IV", "V", "VI"]))

ARMS = {
    "roi_maskbbox":  R / "ORIG_G6_roi_logits.csv",     # mask-bbox crop (different field of view)
    "center_crop":   R / "ORIG_G6_center_logits.csv",  # no mask used (different field of view)
    "bg_tibia_erased": R / "ORIG_G6_bg_logits.csv",    # full frame, tibia erased
    "tib_tibia_only":  R / "exp08_G6_tib_resnet50_prep2_image_logits.csv",  # full frame, tibia only
}


def load(p):
    d = pd.read_csv(p).sort_values(["seed", "key"]).reset_index(drop=True)
    g = d.groupby(["seed", "pid"], sort=True)
    P = g[LOGITS].mean().values.argmax(1)
    T = np.array([CLS6.index(v) for v in g["true"].first().values])
    S = np.array([s for s, _ in g["true"].first().index])
    return d, S, T, P


arms, raw = {}, {}
for k, p in ARMS.items():
    if not p.exists():
        print(f"[skipped] {k}: {p.name} not found", file=sys.stderr); continue
    raw[k], S, T, P = load(p)
    arms[k] = (S, T, P)
    print(f"[loaded] {k}: {p.name}  patient evaluations: {len(T)}")

keys = list(arms); base = raw[keys[0]]
for k in keys[1:]:
    assert base[["seed", "key", "fold"]].equals(raw[k][["seed", "key", "fold"]]) \
        and base["true"].equals(raw[k]["true"]), f"split mismatch: {k}"
print("split identity: all arms agree")

seeds = sorted(set(arms[keys[0]][0].tolist()))
out = {"arms": list(arms), "seeds": seeds,
       "note": "tib and bg are exact complements sharing a field of view; roi/center differ in field of view and are for reference."}


def seedwise(S, T, P, fn):
    return np.array([fn(T[S == s], P[S == s]) for s in seeds])


def paired(a, b, label):
    d = b - a; t, p = stats.ttest_rel(b, a)
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    print(f"  {label}: {a.mean():.4f} -> {b.mean():.4f}  diff {d.mean():+.4f} "
          f"95%CI [{ci[0]:+.4f},{ci[1]:+.4f}] t={t:.3f} p={p:.4f}")
    return {"from": round(float(a.mean()), 4), "to": round(float(b.mean()), 4),
            "diff": round(float(d.mean()), 4),
            "ci": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
            "t": round(float(t), 3), "p": round(float(p), 4)}


print("\n── six-class balanced accuracy ──")
bacc = {k: seedwise(*arms[k], balanced_accuracy_score) for k in arms}
for k, v in bacc.items():
    print(f"  {k:18s} {v.mean():.4f} ± {v.std(ddof=1):.4f}   {np.round(v,3).tolist()}")
out["sixclass_bacc"] = {k: {"mean": round(float(v.mean()), 4),
                           "sd": round(float(v.std(ddof=1)), 4),
                           "per_seed": [round(float(x), 4) for x in v]} for k, v in bacc.items()}

print("\n── the ablation pair (same field of view, exact complement) ──")
out["pair"] = {}
if "bg_tibia_erased" in bacc and "tib_tibia_only" in bacc:
    out["pair"]["erased_to_only"] = paired(bacc["bg_tibia_erased"], bacc["tib_tibia_only"],
                                           "tibia erased -> tibia only")
print("\n── reference (different field of view) ──")
if "roi_maskbbox" in bacc and "tib_tibia_only" in bacc:
    out["pair"]["roi_to_only"] = paired(bacc["roi_maskbbox"], bacc["tib_tibia_only"],
                                        "mask ROI -> tibia only")

print("\n── per-class recall (5 seeds pooled) ──")
hdr = "class   n  " + "".join(f"{k:>20s}" for k in arms)
print(hdr)
out["per_class"] = {}
for i, c in enumerate(CLS6):
    _, T0, _ = arms[keys[0]]; n = int((T0 == i).sum()); row, cells = {}, ""
    for k in arms:
        _, T, P = arms[k]
        kk = int((P[T == i] == i).sum())
        row[k] = {"k": kk, "recall": round(kk / n, 4)}
        cells += f"{kk:8d}/{n:3d}={kk/n:.3f}"
    out["per_class"][ROM[c]] = {"n": n, **row}
    print(f"{ROM[c]:5s} {n:3d} {cells}")

print("\n── VI-vs-rest ──")
i6 = CLS6.index("Type 6")
vi = {k: seedwise(*arms[k], lambda T, P: balanced_accuracy_score(
        (T == i6).astype(int), (P == i6).astype(int))) for k in arms}
for k, v in vi.items():
    print(f"  {k:18s} {v.mean():.4f} ± {v.std(ddof=1):.4f}")
out["vi_vs_rest"] = {k: {"mean": round(float(v.mean()), 4),
                        "sd": round(float(v.std(ddof=1)), 4)} for k, v in vi.items()}
if "bg_tibia_erased" in vi and "tib_tibia_only" in vi:
    out["vi_vs_rest_pair"] = paired(vi["bg_tibia_erased"], vi["tib_tibia_only"],
                                    "VI-vs-rest erased -> tibia only")

json.dump(out, open(R / "exp17_ablation_pair.json", "w"), indent=1, ensure_ascii=False)
print(f"\nwritten: {R / 'exp17_ablation_pair.json'}")
