"""
09_quality_report.py — Independent data-quality audit.

Answers, with measurements rather than assertions:
  1. Do charts carry more than one finding at once?
  2. Are there duplicate or near-duplicate images?
  3. Is there enough visual and content diversity?
"""
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"


def dhash(p, s=16):
    """Perceptual difference hash; robust to tiny rendering jitter."""
    a = np.asarray(Image.open(p).convert("L").resize((s + 1, s), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).tobytes()


def phash_report():
    files = sorted((DATA / "images").rglob("*.png"))
    print(f"hashing {len(files):,} images (perceptual dHash 16x16) ...")
    with ProcessPoolExecutor(max_workers=9) as ex:
        hs = list(ex.map(dhash, files, chunksize=64))

    c = collections.Counter(hs)
    dups = {k: v for k, v in c.items() if v > 1}
    n_extra = sum(v - 1 for v in dups.values())

    print("\n=== Q2: DUPLICATE / NEAR-DUPLICATE IMAGES ===")
    print(f"  total images             : {len(files):,}")
    print(f"  unique perceptual hashes : {len(c):,}")
    print(f"  visually identical extras: {n_extra}  ({100 * n_extra / len(files):.3f}%)")

    if dups:
        h2f = collections.defaultdict(list)
        for f, h in zip(files, hs):
            if h in dups:
                h2f[h].append(f)
        cross = same = 0
        for h, fs in h2f.items():
            labs = {f.parent.name for f in fs}
            if len(labs) > 1:
                cross += 1
            else:
                same += 1
        print(f"  duplicate groups         : {len(h2f)}")
        print(f"    within one class       : {same}  (harmless: same finding, same look)")
        print(f"    ACROSS classes         : {cross}  (would be a real label conflict)")
        for h, fs in list(h2f.items())[:5]:
            print(f"      {[f'{f.parent.name}/{f.name}' for f in fs][:3]}")
    return len(files), n_extra


def multi_finding_report():
    sys.path.insert(0, str(ROOT / "build"))
    import taxonomy as tax

    multi = []
    orig = tax._resolve

    def spy(matches):
        if len(matches) > 1:
            multi.append(tuple(sorted(matches)))
        return orig(matches)

    tax._resolve = spy
    ver = {"univariate": lambda d: tax.verify_univariate(d["x"]),
           "bivariate": lambda d: tax.verify_bivariate(d["x"], d["y"]),
           "timeseries": lambda d: tax.verify_timeseries(d["t"], d["y"]),
           "grouped": lambda d: tax.verify_grouped(d["value"], d["group"]),
           "composition": lambda d: tax.verify_composition(d["category"])}

    m = pd.read_parquet(DATA / "tables" / "metadata.parquet")
    for _, r in m.iterrows():
        ver[r.arity](json.loads(r.data_table_json))
    tax._resolve = orig

    print("\n=== Q1: CHARTS CARRYING MORE THAN ONE FINDING ===")
    print(f"  samples checked          : {len(m):,}")
    print(f"  matching >1 class        : {len(multi)}  ({100 * len(multi) / len(m):.2f}%)")
    if multi:
        for k, v in collections.Counter(multi).most_common(8):
            print("     ", k, v)
    else:
        print("  every shipped sample satisfies exactly one class guard, because")
        print("  ambiguous candidates were rejected during mining rather than labelled.")
    return len(multi)


def diversity_report():
    m = pd.read_parquet(DATA / "tables" / "metadata.parquet")
    print("\n=== Q3: DIVERSITY ===")
    print(f"  chart types              : {m.chart_type.nunique()}")
    print(f"  distinct real source tables: {m[m.provenance=='real'].source_id.nunique():,}")
    print(f"  distinct column names used : {pd.unique(m.columns_.explode() if False else m['columns']).size:,}")
    print(f"  underlying data points     : {int(m.n_points.sum()):,}")
    print(f"  n_points  min/median/max   : {int(m.n_points.min())} / "
          f"{int(m.n_points.median())} / {int(m.n_points.max())}")

    # visual style variety
    recs = {}
    with open(DATA / "records.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            recs.setdefault(r["render"]["style"], 0)
            recs[r["render"]["style"]] += 1
    print(f"  render styles              : {len(recs)}  {dict(list(recs.items())[:4])}...")

    # how varied is the content within a single class?
    print("\n  per-class distinct source tables (real portion):")
    g = (m[m.provenance == "real"].groupby("label").source_id.nunique()
         .sort_values())
    print(f"    min {g.min()} ({g.idxmin()})  median {int(g.median())}  max {g.max()} ({g.idxmax()})")


if __name__ == "__main__":
    n_multi = multi_finding_report()
    n_img, n_dup = phash_report()
    diversity_report()

    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    print(f"  multi-finding leakage : {n_multi} samples")
    print(f"  duplicate images      : {n_dup} of {n_img:,} ({100*n_dup/n_img:.3f}%)")
    print("  See DATASHEET.md section 9 for the limitations that are real.")
