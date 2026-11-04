"""
14_near_duplicates.py — Near-duplicate data across splits.

    python build/14_near_duplicates.py      # -> results/near_duplicates.json, .csv

The integrity audit removes exact duplicates (data fingerprints) and keeps source tables
split-disjoint, but the same series can be published under two sources (e.g. GDP in both
World Bank and Our World in Data). This compares every test and validation chart with
every chart a model can train on (train split + supplementary real pool) of the same
arity, using a fixed-length signature of the plotted data:

    time series    y resampled to 64 points
    univariate     64 quantiles
    bivariate      32 quantiles of x + 32 of y
    grouped        per-group mean, sd and size, groups sorted by mean
    composition    sorted category shares

Signatures that correlate above 0.99 are candidates. A candidate is a near-duplicate when
the data itself matches:

    time series     means within 5% of a standard deviation, sds within 5%
    bivariate       at least half of the (x, y) pairs identical (6 significant figures)
    univariate      at least 90% of values shared counting multiplicity, >= 15 distinct values
    grouped         as univariate, and the same number of groups
    composition     same categories, every share within 1 point

Only real charts are compared. Synthetic charts are independent random draws (one seed per
sample), so two of them can share a shape but not their data; comparing them on shape
flags thousands of independent samples from the same generator.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "dataset", ROOT / "results"
R_THRESH, SCALE_TOL, OVERLAP, SHARE_TOL, MULTISET, MIN_DISTINCT = 0.99, 0.05, 0.5, 0.01, 0.9, 15
Q64 = np.linspace(0.01, 0.99, 64)


def signature(arity, d):
    """(shape vector, location, scale) of the plotted data."""
    if arity == "timeseries":
        y = np.asarray(d["y"], float)
        v = np.interp(np.linspace(0, len(y) - 1, 64), np.arange(len(y)), y)
        return v, y.mean(), y.std()
    if arity == "univariate":
        x = np.asarray(d["x"], float)
        return np.quantile(x, Q64), x.mean(), x.std()
    if arity == "bivariate":
        x, y = np.asarray(d["x"], float), np.asarray(d["y"], float)
        q = np.linspace(0.02, 0.98, 32)
        zx = (np.quantile(x, q) - x.mean()) / (x.std() or 1)
        zy = (np.quantile(y, q) - y.mean()) / (y.std() or 1)
        # location/scale of both axes folded into one pair; corr of x with y must match too
        return np.r_[zx, zy, 10 * np.corrcoef(x, y)[0, 1]], x.mean() + y.mean(), x.std() + y.std()
    if arity == "grouped":
        df = pd.DataFrame({"g": [str(g) for g in d["group"]], "v": np.asarray(d["value"], float)})
        s = df.groupby("g").v.agg(["mean", "std", "size"]).fillna(0).sort_values("mean")
        v = np.r_[s["mean"].values, s["std"].values, s["size"].values / max(s["size"].sum(), 1)]
        return np.resize(v, 64) if len(v) else np.zeros(64), df.v.mean(), df.v.std()
    vc = pd.Series([str(c) for c in d["category"]]).value_counts(normalize=True).values
    v = np.zeros(32); v[:min(32, len(vc))] = vc[:32]
    return v, 0.0, 1.0


def _r(v):
    return [float(f"{x:.6g}") for x in np.asarray(v, float)]


def multiset_overlap(a, b):
    """Share of a's values (with multiplicity) also present in b."""
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values()) / max(len(a), 1)


def same_data(arity, a, b, sa, sb):
    if arity == "timeseries":
        pooled = max(sa[2], sb[2], 1e-12)
        return abs(sa[1] - sb[1]) / pooled < SCALE_TOL and abs(sa[2] - sb[2]) / pooled < SCALE_TOL
    if arity == "composition":
        ca = pd.Series([str(c) for c in a["category"]]).value_counts(normalize=True)
        cb = pd.Series([str(c) for c in b["category"]]).value_counts(normalize=True)
        return set(ca.index) == set(cb.index) and (ca - cb.reindex(ca.index)).abs().max() < SHARE_TOL
    if arity == "bivariate":   # rows, not values: (x, y) pairs are far more specific
        pa, pb = set(zip(_r(a["x"]), _r(a["y"]))), set(zip(_r(b["x"]), _r(b["y"])))
        return len(pa & pb) / max(len(pa), 1) >= OVERLAP
    key = "x" if arity == "univariate" else "value"
    va, vb = _r(a[key]), _r(b[key])
    if len(set(va)) < MIN_DISTINCT:   # coarse integer data shares values by chance
        return False
    if arity == "grouped" and len(set(map(str, a["group"]))) != len(set(map(str, b["group"]))):
        return False
    return multiset_overlap(va, vb) >= MULTISET


def load():
    meta = pd.read_parquet(DATA / "tables" / "metadata.parquet")
    rows = [dict(sample_id=r.sample_id, split=r.split, arity=r.arity, label=r.label,
                 provenance=r.provenance, source=r.source, d=json.loads(r.data_table_json))
            for r in meta.itertuples()]
    extra = ROOT / "cache" / "records_extra.jsonl"
    if extra.exists():
        for line in open(extra):
            r = json.loads(line)
            rows.append(dict(sample_id=r["sample_id"], split="train_extra", arity=r["arity"], label=r["label"],
                             provenance="real", source=r["source"], d=r["ground_truth"]["data_table"]))
    return rows


def main():
    rows = [r for r in load() if r["provenance"] == "real"]
    found, candidates = [], 0
    for arity in sorted({r["arity"] for r in rows}):
        rs = [r for r in rows if r["arity"] == arity]
        sig = [signature(arity, r["d"]) for r in rs]
        V = np.array([x[0] for x in sig], float)
        V = V - V.mean(1, keepdims=True)
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        split = np.array([r["split"] for r in rs])
        trainable = np.where(np.isin(split, ["train", "train_extra"]))[0]
        for held in ("val", "test"):
            qi = np.where(split == held)[0]
            ti = np.r_[trainable, np.where(split == "val")[0]] if held == "test" else trainable
            C = V[qi] @ V[ti].T
            for a, b in zip(*np.where(C > R_THRESH)):
                i, j = qi[a], ti[b]
                candidates += 1
                if same_data(arity, rs[i]["d"], rs[j]["d"], sig[i], sig[j]):
                    found.append({"held_out": rs[i]["sample_id"], "held_split": held, "held_label": rs[i]["label"],
                                  "held_source": rs[i]["source"], "match": rs[j]["sample_id"],
                                  "match_split": rs[j]["split"], "match_label": rs[j]["label"],
                                  "match_source": rs[j]["source"], "arity": arity, "corr": round(float(C[a, b]), 5)})
        print(f"{arity:12s} {len(rs):>6,} real charts checked", flush=True)
    df = pd.DataFrame(found)
    RES.mkdir(exist_ok=True)
    df.to_csv(RES / "near_duplicates.csv", index=False)
    held = df.drop_duplicates("held_out") if len(df) else df
    summary = {"criterion": f"signature corr > {R_THRESH}, then same data (see module docstring)",
               "real_charts_compared": len(rows), "shape_candidates": candidates,
               "pairs": int(len(df)),
               "test_charts_with_a_near_duplicate": int((held.held_split == "test").sum()) if len(df) else 0,
               "val_charts_with_a_near_duplicate": int((held.held_split == "val").sum()) if len(df) else 0,
               "same_label_pairs": int((df.held_label == df.match_label).sum()) if len(df) else 0}
    if len(df):
        summary["by_arity"] = held.groupby("arity").size().to_dict()
        summary["test_ids"] = sorted(held[held.held_split == "test"].held_out.tolist())
    (RES / "near_duplicates.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "test_ids"}, indent=2))


if __name__ == "__main__":
    main()
