#!/usr/bin/env python3
"""
PlaTiF exploratory pass 2: measure the internal structure of the .mat files.

The aim is to settle the open items in the study design directly.
  1. actual image resolution and bit depth
  2. mask pixel encoding (0/1 vs 0/255)
  3. images per patient, and how X-rays are distinguished from CT sections
  4. image-level class distribution — the descriptor reports patient level only

Memory discipline:
  - extract one .mat at a time to a temporary file, read it, delete it immediately.
  - never accumulate arrays; compute statistics and discard.
  - check peak RSS after every file and abort as soon as --max-rss-mb is exceeded.
  - keep temporary files on the external SSD (an internal /tmp cleanup once destroyed a run).

Usage:
  eda02_matprobe.py --limit 5     # quick reconnaissance (default)
  eda02_matprobe.py --limit 0     # full survey
"""
import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed/data/platif")
OUT = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed/results")
SCRATCH = DATA / ".scratch"


def rss_mb():
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def mat_version(path):
    with open(path, "rb") as f:
        head = f.read(128)
    txt = head.decode("latin-1", "ignore")
    if "MATLAB 7.3" in txt:
        return "v7.3"
    if "MATLAB 5.0" in txt:
        return "v5/v7"
    return "unknown"


def arr_stats(a, name=""):
    """Summarise one array. The array itself is not returned."""
    if not isinstance(a, np.ndarray):
        return {"kind": type(a).__name__, "value": str(a)[:120]}
    s = {"kind": "ndarray", "shape": list(a.shape), "dtype": str(a.dtype),
         "nbytes_mb": round(a.nbytes / 1024 / 1024, 2)}
    if a.dtype.kind in "biufc" and a.size:
        flat = a.ravel()
        # for large arrays compute statistics from a sample (saves memory and time)
        samp = flat if flat.size <= 4_000_000 else flat[:: max(1, flat.size // 4_000_000)]
        try:
            s["min"] = float(np.min(samp))
            s["max"] = float(np.max(samp))
            s["mean"] = round(float(np.mean(samp)), 3)
            u = np.unique(samp)
            s["n_unique"] = int(u.size)
            if u.size <= 12:
                s["unique_values"] = [float(x) for x in u]
        except Exception as e:  # complex types and similar
            s["stats_error"] = str(e)
    del a
    return s


def walk_scipy(obj, depth=0, maxdepth=4):
    """Recursively summarise the result of scipy.io.loadmat(struct_as_record=False, squeeze_me=True)."""
    import scipy.io as sio
    if depth > maxdepth:
        return {"truncated": True}
    if isinstance(obj, sio.matlab.mat_struct):
        return {f: walk_scipy(getattr(obj, f), depth + 1, maxdepth)
                for f in obj._fieldnames}
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        out = {"kind": "object_array", "shape": list(obj.shape),
               "items": []}
        for i, el in enumerate(obj.ravel()[:8]):
            out["items"].append(walk_scipy(el, depth + 1, maxdepth))
        if obj.size > 8:
            out["items_truncated_from"] = int(obj.size)
        return out
    if isinstance(obj, np.ndarray):
        return arr_stats(obj)
    if isinstance(obj, (str, bytes, int, float, np.generic)):
        return {"kind": type(obj).__name__, "value": str(obj)[:120]}
    return {"kind": type(obj).__name__}


def walk_h5(g, depth=0, maxdepth=4):
    import h5py
    if depth > maxdepth:
        return {"truncated": True}
    out = {}
    for k in list(g.keys())[:40]:
        v = g[k]
        if isinstance(v, h5py.Group):
            out[k] = walk_h5(v, depth + 1, maxdepth)
        else:
            try:
                out[k] = arr_stats(np.asarray(v[()]))
            except Exception as e:
                out[k] = {"read_error": str(e)}
    return out


def probe_one(mat_path):
    ver = mat_version(mat_path)
    rec = {"mat_version": ver, "file_mb": round(mat_path.stat().st_size / 1024 / 1024, 2)}
    if ver == "v7.3":
        import h5py
        with h5py.File(mat_path, "r") as f:
            rec["top_level_keys"] = [k for k in f.keys() if not k.startswith("#")]
            rec["structure"] = walk_h5(f)
    else:
        import scipy.io as sio
        md = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        rec["top_level_keys"] = [k for k in md if not k.startswith("__")]
        rec["structure"] = {k: walk_scipy(v) for k, v in md.items()
                            if not k.startswith("__")}
        del md
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5,
                    help="number of .mat files to inspect; 0 means all")
    ap.add_argument("--max-rss-mb", type=float, default=1200,
                    help="peak RSS ceiling; abort immediately if exceeded")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    zips = sorted(DATA.glob("Patient Data_Part *.zip"))
    if not zips:
        sys.exit(f"ERROR: no zip found: {DATA}")

    # member listing (nothing extracted)
    todo = []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for i in zf.infolist():
                if not i.is_dir() and i.filename.lower().endswith(".mat"):
                    todo.append((z, i.filename, i.file_size))
    todo.sort(key=lambda t: t[1])
    total_available = len(todo)
    if args.limit:
        # sampling only the first files would bias the survey, so spread the draws evenly
        step = max(1, len(todo) // args.limit)
        todo = todo[::step][: args.limit]

    print(f"inspecting {len(todo)} of {total_available} .mat files "
          f"(RSS ceiling {args.max_rss_mb} MB)")
    print("=" * 72)

    results = {}
    aborted = None
    for n, (z, member, fsize) in enumerate(todo, 1):
        tmp = SCRATCH / Path(member).name
        try:
            with zipfile.ZipFile(z) as zf, zf.open(member) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            rec = probe_one(tmp)
            rec["zip"] = z.name
            rec["member"] = member
            results[member] = rec
            print(f"[{n}/{len(todo)}] {member}  {rec['mat_version']}  "
                  f"{rec['file_mb']} MB  keys={rec['top_level_keys']}  "
                  f"RSS={rss_mb():.0f}MB")
        except Exception as e:
            results[member] = {"error": repr(e), "zip": z.name}
            print(f"[{n}/{len(todo)}] {member}  ERROR {e!r}")
        finally:
            tmp.unlink(missing_ok=True)

        if rss_mb() > args.max_rss_mb:
            aborted = f"RSS {rss_mb():.0f} MB > ceiling {args.max_rss_mb} MB — aborted"
            print(f"\n🔴 {aborted}")
            print("   From this point on the work must move to the GPU host.")
            break

    payload = {
        "n_mat_total": total_available,
        "n_probed": len(results),
        "peak_rss_mb": round(rss_mb(), 1),
        "aborted": aborted,
        "records": results,
    }
    with open(OUT / "eda02_matprobe.json", "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    print()
    print("=" * 72)
    print(f"peak RSS {rss_mb():.0f} MB · written {OUT / 'eda02_matprobe.json'}")
    try:
        SCRATCH.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
