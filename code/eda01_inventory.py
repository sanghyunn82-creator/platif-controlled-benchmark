#!/usr/bin/env python3
"""
PlaTiF exploratory passes 0 and 1: survey the zip contents and read the metadata.

Nothing is extracted. Only the zip central directory is read, so memory and time are negligible.
This stays well within the local compute budget.

What we want to establish:
  - the real uncompressed size (checking the "2.3 TB" claim on the Zenodo page)
  - how patients are distributed across the five zips
  - the actual columns, class distribution and bilateral-fracture count in the metadata xlsx

Output: results/eda01_inventory.json plus a human-readable summary on stdout.
"""
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed/data/platif")
OUT = Path("/Volumes/dkyoo_SSD1/review_partime/orthopedic_premed/results")
XLSX = DATA / "Tibial Plateau Fracture Metadata.xlsx"

# Expected values from the data descriptor (Sci Data 2026, PMC12905147), for comparison with the EDA.
PAPER_PCT = {"I": 14.51, "II": 18.27, "III": 6.45, "IV": 5.91,
             "V": 6.45, "VI": 17.20, "Normal": 31.18}
PAPER_N_PATIENTS = 186
PAPER_N_IMAGES = 421


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024


def scan_zips():
    """Read the zip central directory only — nothing is extracted."""
    zips = sorted(DATA.glob("Patient Data_Part *.zip"))
    if not zips:
        sys.exit(f"ERROR: no zip files found: {DATA}")

    per_zip = {}
    all_members = []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
        comp = sum(i.compress_size for i in infos)
        uncomp = sum(i.file_size for i in infos)
        per_zip[z.name] = {
            "n_members": len(infos),
            "compressed_bytes": comp,
            "uncompressed_bytes": uncomp,
            "ratio": round(uncomp / comp, 2) if comp else None,
            "members": [i.filename for i in infos],
        }
        for i in infos:
            all_members.append((z.name, i.filename, i.file_size, i.compress_size))
    return per_zip, all_members


def patient_id(member_name):
    """Extract a patient identifier from a member path. The rule is unknown, so keep it adjustable."""
    stem = Path(member_name).name
    m = re.search(r"(\d+)", stem)
    return m.group(1) if m else stem


def read_metadata():
    if not XLSX.exists():
        return {"error": f"metadata file not found: {XLSX}"}
    import pandas as pd

    xl = pd.ExcelFile(XLSX)
    out = {"sheets": xl.sheet_names, "per_sheet": {}}
    for sh in xl.sheet_names:
        df = xl.parse(sh)
        info = {
            "n_rows": int(len(df)),
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "n_missing": {str(c): int(df[c].isna().sum()) for c in df.columns},
        }
        # for low-cardinality columns, carry the distribution as is (class, sex, laterality, ...)
        info["value_counts"] = {}
        for c in df.columns:
            nun = df[c].nunique(dropna=True)
            if nun <= 25:
                vc = df[c].value_counts(dropna=False)
                info["value_counts"][str(c)] = {str(k): int(v) for k, v in vc.items()}
        # numeric summary
        info["numeric_summary"] = {}
        for c in df.select_dtypes("number").columns:
            s = df[c].dropna()
            if len(s):
                info["numeric_summary"][str(c)] = {
                    "min": float(s.min()), "max": float(s.max()),
                    "mean": round(float(s.mean()), 2), "median": float(s.median()),
                }
        out["per_sheet"][sh] = info
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}

    print("=" * 72)
    print("PlaTiF exploratory pass 0: zip listing (nothing extracted)")
    print("=" * 72)
    per_zip, members = scan_zips()
    result["zips"] = {k: {kk: vv for kk, vv in v.items() if kk != "members"}
                      for k, v in per_zip.items()}

    tot_c = sum(v["compressed_bytes"] for v in per_zip.values())
    tot_u = sum(v["uncompressed_bytes"] for v in per_zip.values())
    for name, v in per_zip.items():
        print(f"  {name:28s} members {v['n_members']:4d}  "
              f"compressed {human(v['compressed_bytes']):>9s} -> uncompressed {human(v['uncompressed_bytes']):>9s} "
              f"(x{v['ratio']})")
    print(f"  {'total':28s} members {len(members):4d}  "
          f"compressed {human(tot_c):>9s} -> uncompressed {human(tot_u):>9s} (x{tot_u/tot_c:.2f})")
    result["total_compressed_bytes"] = tot_c
    result["total_uncompressed_bytes"] = tot_u

    print()
    print("  > checking the Zenodo claim that it expands to 2.3 TB:")
    if tot_u > 1.5 * 1024**4:
        print(f"    measured {human(tot_u)} — consistent with the claim. The extraction strategy needs rethinking.")
    else:
        print(f"    measured {human(tot_u)} — inconsistent with the claim. Free disk (1.2 TB) is sufficient.")
    result["zenodo_2_3tb_claim_supported"] = bool(tot_u > 1.5 * 1024**4)

    # ---- patient distribution ----
    print()
    print("-" * 72)
    print("patients -> zip distribution")
    print("-" * 72)
    by_zip = defaultdict(set)
    ext_counter = Counter()
    for zname, mname, fsize, csize in members:
        by_zip[zname].add(patient_id(mname))
        ext_counter[Path(mname).suffix.lower()] += 1
    for zname in sorted(by_zip):
        print(f"  {zname:28s} unique identifiers {len(by_zip[zname]):4d}")
    all_ids = set().union(*by_zip.values()) if by_zip else set()
    overlap = sum(len(a & b) for i, a in enumerate(by_zip.values())
                  for b in list(by_zip.values())[i + 1:])
    print(f"  unique identifiers overall {len(all_ids)} (patients reported in the descriptor: {PAPER_N_PATIENTS})")
    print(f"  identifiers appearing in more than one zip: {overlap} " +
          ("<- patients span zips; be careful when splitting" if overlap else "<- no overlap"))
    print(f"  extension distribution: {dict(ext_counter)}")
    result["patients_per_zip"] = {k: len(v) for k, v in by_zip.items()}
    result["n_unique_ids"] = len(all_ids)
    result["cross_zip_id_overlap"] = overlap
    result["extensions"] = dict(ext_counter)
    result["members_sample"] = [m[1] for m in members[:15]]

    print()
    print("  example member names:")
    for m in members[:10]:
        print(f"    {m[1]}  ({human(m[2])})")

    # ---- metadata ----
    print()
    print("-" * 72)
    print("pass 1: metadata xlsx")
    print("-" * 72)
    meta = read_metadata()
    result["metadata"] = meta
    if "error" in meta:
        print(f"  {meta['error']}")
    else:
        for sh, info in meta["per_sheet"].items():
            print(f"  [sheet {sh}] {info['n_rows']} rows x {len(info['columns'])} columns")
            print(f"    columns: {info['columns']}")
            miss = {k: v for k, v in info["n_missing"].items() if v}
            if miss:
                print(f"    missing: {miss}")
            for c, vc in info["value_counts"].items():
                print(f"    [{c}] {vc}")
            for c, ns in info["numeric_summary"].items():
                print(f"    [{c}] min={ns['min']} max={ns['max']} "
                      f"mean={ns['mean']} median={ns['median']}")

        # compare with the percentages in the descriptor
        print()
        print("  > comparison with the class distribution in the descriptor:")
        found = False
        for sh, info in meta["per_sheet"].items():
            for c, vc in info["value_counts"].items():
                keys = {str(k).strip() for k in vc}
                if len({"I", "II", "III", "IV", "V", "VI"} & keys) >= 4:
                    found = True
                    tot = sum(vc.values())
                    print(f"    (sheet {sh}, column {c}, total {tot})")
                    for cls, pct in PAPER_PCT.items():
                        n = vc.get(cls, 0)
                        got = 100 * n / tot if tot else 0
                        flag = "OK" if abs(got - pct) < 0.6 else "MISMATCH"
                        print(f"      {cls:7s} measured {n:3d} ({got:5.2f}%) / descriptor {pct:5.2f}%  {flag}")
        if not found:
            print("    Could not locate the Schatzker column automatically; check the column list above by hand.")

    with open(OUT / "eda01_inventory.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print()
    print(f"written: {OUT / 'eda01_inventory.json'}")


if __name__ == "__main__":
    main()
