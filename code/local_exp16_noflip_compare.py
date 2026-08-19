#!/usr/bin/env python3
"""P4-1 · exp16 — horizontal flip on (G6_roi) versus off (G6_roi_noflip).

Question: is the collapse of Schatzker IV (recall 0.04) caused by the mirror invariance we
           introduced, or by the limits of the AP projection?

Both runs use the same (seed, fold) patient split (StratifiedGroupKFold(random_state=seed), same
rows, same labels), so a seed-paired comparison is valid.

Output: results/exp16_noflip_compare.json
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score

R = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed/results")
ROM = {"Type 1": "I", "Type 2": "II", "Type 3": "III",
       "Type 4": "IV", "Type 5": "V", "Type 6": "VI"}
rng = np.random.default_rng(20260818)


def load(cfg):
    d = pd.read_csv(R / f"exp08_{cfg}_resnet50_prep2_image_logits.csv")
    cols = [c for c in d.columns if c.startswith("logit_")]
    cls = [c[6:] for c in cols]
    g = d.groupby(["seed", "pid"])
    P = g[cols].mean().values.argmax(1)
    T = np.array([cls.index(v) for v in g["true"].first().values])
    idx = g["true"].first().index
    return (np.array([s for s, _ in idx]), np.array([p for _, p in idx]), T, P, cls)


def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


sF, pF, TF, PF, cls = load("G6_roi")
sN, pN, TN, PN, _ = load("G6_roi_noflip")
assert cls == _ and np.array_equal(sF, sN) and np.array_equal(pF, pN) and np.array_equal(TF, TN), \
    "the two runs have different (seed, pid) splits — paired comparison invalid"
seeds = sorted(set(sF.tolist()))
out = {"n_patient_evaluations": int(len(TF)), "seeds": seeds}

print("── per-class recall (5 seeds pooled) ──")
print(f"{'class':6s} {'flip on':>22s} {'flip off':>22s} {'diff':>9s}")
out["per_class"] = {}
for i, c in enumerate(cls):
    m = TF == i
    kF, kN, n = int((PF[m] == i).sum()), int((PN[m] == i).sum()), int(m.sum())
    lF, hF = clopper(kF, n); lN, hN = clopper(kN, n)
    out["per_class"][ROM[c]] = {"n": n, "flip_on": [kF, round(kF / n, 4), round(lF, 4), round(hF, 4)],
                                "flip_off": [kN, round(kN / n, 4), round(lN, 4), round(hN, 4)]}
    print(f"{ROM[c]:6s} {kF:3d}/{n:3d}={kF/n:.3f} [{lF:.3f},{hF:.3f}] "
          f"{kN:3d}/{n:3d}={kN/n:.3f} [{lN:.3f},{hN:.3f}] {kN/n - kF/n:+9.3f}")

print("\n── type IV: seed-paired recall ──")
i4 = cls.index("Type 4")
rF = [(PF[(sF == s) & (TF == i4)] == i4).mean() for s in seeds]
rN = [(PN[(sN == s) & (TN == i4)] == i4).mean() for s in seeds]
d = np.array(rN) - np.array(rF)
print(f"  flip on  {np.round(rF,3)}  mean {np.mean(rF):.4f}")
print(f"  flip off {np.round(rN,3)}  mean {np.mean(rN):.4f}")
if np.allclose(d, 0):
    print("  difference is 0 — paired t-test not possible")
    out["type4_paired"] = {"flip_on": [round(float(x),4) for x in rF],
                           "flip_off": [round(float(x),4) for x in rN], "diff_mean": 0.0}
else:
    t, p = stats.ttest_rel(rN, rF)
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    print(f"  diff {d.mean():+.4f} 95%CI [{ci[0]:+.4f},{ci[1]:+.4f}] t={t:.3f} df={len(d)-1} p={p:.4f}")
    out["type4_paired"] = {"flip_on": [round(float(x),4) for x in rF],
                           "flip_off": [round(float(x),4) for x in rN],
                           "diff_mean": round(float(d.mean()), 4),
                           "ci": [round(float(ci[0]),4), round(float(ci[1]),4)],
                           "t": round(float(t),3), "df": len(d)-1, "p": round(float(p),4)}

print("\n── type IV error direction (share assigned to lateral classes I/II/III) ──")
lat = [cls.index(c) for c in ["Type 1", "Type 2", "Type 3"]]
for nm, P_ in (("flip on", PF), ("flip off", PN)):
    m = TF == i4
    k = int(np.isin(P_[m], lat).sum()); n = int(m.sum())
    row = {ROM[c]: int((P_[m] == j).sum()) for j, c in enumerate(cls)}
    print(f"  {nm:9s} lateral {k}/{n} = {k/n:.3f}   row {row}")
    out.setdefault("type4_direction", {})[nm] = {"lateral": k, "n": n, "row": row}

print("\n── six-class overall (seed-paired) ──")
bF = [balanced_accuracy_score(TF[sF == s], PF[sF == s]) for s in seeds]
bN = [balanced_accuracy_score(TN[sN == s], PN[sN == s]) for s in seeds]
d = np.array(bN) - np.array(bF)
t, p = stats.ttest_rel(bN, bF)
ci = stats.t.interval(0.95, 4, loc=d.mean(), scale=stats.sem(d))
print(f"  flip on  {np.mean(bF):.4f} ± {np.std(bF,ddof=1):.4f}")
print(f"  flip off {np.mean(bN):.4f} ± {np.std(bN,ddof=1):.4f}")
print(f"  diff {d.mean():+.4f} 95%CI [{ci[0]:+.4f},{ci[1]:+.4f}] t={t:.3f} df=4 p={p:.4f}")
out["sixclass_paired"] = {"flip_on_mean": round(float(np.mean(bF)),4),
                          "flip_on_sd": round(float(np.std(bF,ddof=1)),4),
                          "flip_off_mean": round(float(np.mean(bN)),4),
                          "flip_off_sd": round(float(np.std(bN,ddof=1)),4),
                          "diff": round(float(d.mean()),4),
                          "ci": [round(float(ci[0]),4), round(float(ci[1]),4)],
                          "t": round(float(t),3), "p": round(float(p),4)}

print("\n── VI-vs-rest recast ──")
i6 = cls.index("Type 6")
for nm, P_ in (("flip on", PF), ("flip off", PN)):
    v = [balanced_accuracy_score((TF[sF == s] == i6).astype(int), (P_[sF == s] == i6).astype(int))
         for s in seeds]
    print(f"  {nm:9s} {np.mean(v):.4f} ± {np.std(v,ddof=1):.4f}")
    out.setdefault("vi_vs_rest", {})[nm] = {"mean": round(float(np.mean(v)),4),
                                            "sd": round(float(np.std(v,ddof=1)),4)}

json.dump(out, open(R / "exp16_noflip_compare.json", "w"), indent=1, ensure_ascii=False)
print(f"\nwritten: {R / 'exp16_noflip_compare.json'}")
