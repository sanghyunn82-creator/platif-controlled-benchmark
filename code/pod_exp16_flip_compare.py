#!/usr/bin/env python3
"""P4-1 · exp16 — the decisive experiment for the horizontal-flip augmentation.

Question: is the collapse of Schatzker IV (recall 0.04) caused by the mirror invariance we
           introduced, or by the limits of the AP projection?

Three arms are used. The no-flip run was carried out with the GPU-augmentation script
(pod_exp08_gpuaug.py), whereas the existing flip-on run used the CPU path (pod_exp08_seeded.py).
Comparing those two directly would change two things at once: the flip flag and the CPU->GPU
augmentation rewrite. A flip-on control is therefore rerun through the same GPU script.

  A = ORIG_G6_roi_logits.csv        flip ON,  CPU augmentation (the 0.345 reported in the paper)
  B = exp08_G6_roi_flip_*           flip ON,  GPU augmentation (the control)
  C = SAFE_gpuaug_noflip_logits.csv flip OFF, GPU augmentation

  B vs C = the decisive contrast (only the flip differs)
  A vs B = port validation (these should agree)

All three arms are asserted to share identical (seed, key, fold, true) rows.
Output: results/exp16_flip_compare.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score

R = Path("/workspace/platif_p4/results")
ROM = {"Type 1": "I", "Type 2": "II", "Type 3": "III",
       "Type 4": "IV", "Type 5": "V", "Type 6": "VI"}
CLS6 = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6"]
LOGITS = [f"logit_{c}" for c in CLS6]

ARMS = {
    "A_flip_cpu":   R / "ORIG_G6_roi_logits.csv",
    "B_flip_gpu":   R / "exp08_G6_roi_flip_resnet50_prep2_image_logits.csv",
    "C_noflip_gpu": R / "SAFE_gpuaug_noflip_logits.csv",
}


def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def load(p):
    """Image logits -> patient-level prediction: argmax of the mean logit per (seed, pid)."""
    d = pd.read_csv(p).sort_values(["seed", "key"]).reset_index(drop=True)
    g = d.groupby(["seed", "pid"], sort=True)
    P = g[LOGITS].mean().values.argmax(1)
    T = np.array([CLS6.index(v) for v in g["true"].first().values])
    idx = g["true"].first().index
    S = np.array([s for s, _ in idx])
    return d, S, T, P


arms, raw = {}, {}
for k, p in ARMS.items():
    if not p.exists():
        print(f"[skipped] {k}: {p.name} not found", file=sys.stderr)
        continue
    raw[k], S, T, P = load(p)
    arms[k] = (S, T, P)
    print(f"[loaded] {k}: {p.name}  patient evaluations: {len(T)}")

# ── split identity across arms (the premise of a paired comparison) ─────────
keys = list(arms)
base = raw[keys[0]]
split_ok = {}
for k in keys[1:]:
    d = raw[k]
    same = (base[["seed", "key", "fold"]].equals(d[["seed", "key", "fold"]])
            and base["true"].equals(d["true"]))
    split_ok[f"{keys[0]}_vs_{k}"] = bool(same)
    assert same, f"split mismatch: {keys[0]} vs {k} — paired comparison invalid"
    assert np.array_equal(arms[keys[0]][1], arms[k][1]), f"label mismatch: {k}"
print("split identity:", split_ok)

out = {"arms": list(arms), "split_identical": split_ok,
       "n_patient_evaluations": int(len(arms[keys[0]][1]))}
seeds = sorted(set(arms[keys[0]][0].tolist()))
out["seeds"] = seeds
i4, i6 = CLS6.index("Type 4"), CLS6.index("Type 6")


def seedwise(S, T, P, fn):
    return np.array([fn(T[S == s], P[S == s]) for s in seeds])


def paired(a, b, label):
    """Seed-paired comparison of b - a."""
    d = b - a
    r = {"mean_a": round(float(a.mean()), 4), "sd_a": round(float(a.std(ddof=1)), 4),
         "mean_b": round(float(b.mean()), 4), "sd_b": round(float(b.std(ddof=1)), 4),
         "diff": round(float(d.mean()), 4)}
    if np.allclose(d, 0):
        r["note"] = "difference is exactly 0 — no test possible"
        return r
    t, p = stats.ttest_rel(b, a)
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    r.update({"ci": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
              "t": round(float(t), 3), "df": len(d) - 1, "p": round(float(p), 4)})
    print(f"  {label}: {a.mean():.4f} -> {b.mean():.4f}  diff {d.mean():+.4f} "
          f"95%CI [{ci[0]:+.4f},{ci[1]:+.4f}] t={t:.3f} p={p:.4f}")
    return r


# ── 1. six-class balanced accuracy ─────────────────────────────────────────
print("\n── six-class balanced accuracy (seed-paired) ──")
bacc = {k: seedwise(*arms[k], balanced_accuracy_score) for k in arms}
for k, v in bacc.items():
    print(f"  {k:14s} {v.mean():.4f} ± {v.std(ddof=1):.4f}   {np.round(v,3).tolist()}")
out["sixclass_bacc"] = {k: {"mean": round(float(v.mean()), 4),
                           "sd": round(float(v.std(ddof=1)), 4),
                           "per_seed": [round(float(x), 4) for x in v]}
                        for k, v in bacc.items()}
out["sixclass_paired"] = {}
if "B_flip_gpu" in bacc and "C_noflip_gpu" in bacc:
    out["sixclass_paired"]["decisive_B_to_C"] = paired(
        bacc["B_flip_gpu"], bacc["C_noflip_gpu"], "decisive B(flip)->C(noflip)")
if "A_flip_cpu" in bacc and "B_flip_gpu" in bacc:
    out["sixclass_paired"]["port_A_to_B"] = paired(
        bacc["A_flip_cpu"], bacc["B_flip_gpu"], "port A(cpu)->B(gpu)")

# ── 2. per-class recall (pooled over five seeds) ───────────────────────────
print("\n── per-class recall (5 seeds pooled, Clopper-Pearson) ──")
hdr = "class    n  " + "".join(f"{k:>26s}" for k in arms)
print(hdr)
out["per_class"] = {}
for i, c in enumerate(CLS6):
    S0, T0, _ = arms[keys[0]]
    m = T0 == i
    n = int(m.sum())
    row, cells = {}, ""
    for k in arms:
        _, T, P = arms[k]
        kk = int((P[T == i] == i).sum())
        lo, hi = clopper(kk, n)
        row[k] = {"k": kk, "recall": round(kk / n, 4),
                  "ci": [round(lo, 4), round(hi, 4)]}
        cells += f"  {kk:3d}/{n:3d}={kk/n:.3f}[{lo:.2f},{hi:.2f}]"
    out["per_class"][ROM[c]] = {"n": n, **row}
    print(f"{ROM[c]:6s} {n:3d}{cells}")

# ── 3. type IV: seed-paired recall (the headline) ──────────────────────────
print("\n── type IV recall (seed-paired) ──")
rec4 = {k: seedwise(*arms[k], lambda T, P: float((P[T == i4] == i4).mean())) for k in arms}
for k, v in rec4.items():
    print(f"  {k:14s} {v.mean():.4f} ± {v.std(ddof=1):.4f}   {np.round(v,3).tolist()}")
out["type4_recall"] = {k: {"mean": round(float(v.mean()), 4),
                          "sd": round(float(v.std(ddof=1)), 4),
                          "per_seed": [round(float(x), 4) for x in v]}
                       for k, v in rec4.items()}
out["type4_paired"] = {}
if "B_flip_gpu" in rec4 and "C_noflip_gpu" in rec4:
    out["type4_paired"]["decisive_B_to_C"] = paired(
        rec4["B_flip_gpu"], rec4["C_noflip_gpu"], "decisive B(flip)->C(noflip)")
if "A_flip_cpu" in rec4 and "B_flip_gpu" in rec4:
    out["type4_paired"]["port_A_to_B"] = paired(
        rec4["A_flip_cpu"], rec4["B_flip_gpu"], "port A(cpu)->B(gpu)")

# ── 4. direction of type IV errors (the direct prediction of the mirror hypothesis) ──
print("\n── type IV error direction: share assigned to a lateral class (I/II/III) ──")
lat = [CLS6.index(c) for c in ["Type 1", "Type 2", "Type 3"]]
out["type4_direction"] = {}
for k in arms:
    _, T, P = arms[k]
    m = T == i4
    n = int(m.sum()); kk = int(np.isin(P[m], lat).sum())
    lo, hi = clopper(kk, n)
    dist = {ROM[c]: int((P[m] == j).sum()) for j, c in enumerate(CLS6)}
    out["type4_direction"][k] = {"lateral": kk, "n": n,
                                 "frac": round(kk / n, 4),
                                 "ci": [round(lo, 4), round(hi, 4)], "row": dist}
    print(f"  {k:14s} lateral {kk:3d}/{n:3d} = {kk/n:.3f} [{lo:.3f},{hi:.3f}]  row {dist}")

# ── 5. VI-vs-rest recast ───────────────────────────────────────────────────
print("\n── VI-vs-rest balanced accuracy (seed-paired) ──")
vi = {k: seedwise(*arms[k], lambda T, P: balanced_accuracy_score(
        (T == i6).astype(int), (P == i6).astype(int))) for k in arms}
for k, v in vi.items():
    print(f"  {k:14s} {v.mean():.4f} ± {v.std(ddof=1):.4f}")
out["vi_vs_rest"] = {k: {"mean": round(float(v.mean()), 4),
                        "sd": round(float(v.std(ddof=1)), 4)} for k, v in vi.items()}
out["vi_vs_rest_paired"] = {}
if "B_flip_gpu" in vi and "C_noflip_gpu" in vi:
    out["vi_vs_rest_paired"]["decisive_B_to_C"] = paired(
        vi["B_flip_gpu"], vi["C_noflip_gpu"], "decisive B(flip)->C(noflip)")

# ── 6. macro F1 (secondary) ────────────────────────────────────────────────
f1 = {k: seedwise(*arms[k], lambda T, P: f1_score(T, P, average="macro")) for k in arms}
out["macro_f1"] = {k: {"mean": round(float(v.mean()), 4),
                      "sd": round(float(v.std(ddof=1)), 4)} for k, v in f1.items()}

json.dump(out, open(R / "exp16_flip_compare.json", "w"), indent=1, ensure_ascii=False)
print(f"\nwritten: {R / 'exp16_flip_compare.json'}")
