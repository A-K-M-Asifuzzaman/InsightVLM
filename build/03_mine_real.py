"""
03_mine_real.py — Mine verified candidates out of the harvested real tables.

For every source table we enumerate plottable views (single numeric column,
numeric pairs, entity time series, numeric-by-category groupings, categorical
compositions), subsample each view to its final plotted size FIRST, then run the
taxonomy battery on exactly those values. A candidate is kept only if a named
statistical test confirms a single class.

Subsampling before verification is what guarantees the ground truth in the record
describes the array that gets drawn -- not a parent dataframe.

Output: cache/candidates_real.jsonl  (one verified candidate per line, data inline)
"""
import json
import os
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as tax  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_sources"
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)
OUT = CACHE / "candidates_real.jsonl"

MAX_POINTS = 600          # final plotted size; verification runs on exactly this
MAX_PER_TABLE_CLASS = 4   # source diversity: cap one table's share of any class
# Entity time series are exempt from the tight cap: each country/entity in a
# World Bank or OWID table is an independent real-world series, not a redundant
# view of the same numbers, so capping at 4 would discard genuine diversity.
MAX_PER_TABLE_CLASS_TS = 90
MAX_NUM_COLS = 12
MAX_PAIRS = 40
MAX_ENTITIES = 220

ID_PAT = ("id", "index", "idx", "key", "code", "uuid", "guid", "_no", "rowid")
TIME_PAT = ("year", "date", "time", "month", "day", "period", "quarter", "week", "yr")

# Columns that carry no meaning on their own: V1, A4, x2, f7, bare letters, or
# generic words. ~16% of mined real axis labels looked like this, which makes the
# rendered chart unreadable and gives a VLM nothing to ground on.
ANON_RE = re.compile(r"^(v|a|x|y|f|c|col|attr|att|feat|var)_?\d+$|^[a-z]$|^col\d*$", re.I)
GENERIC = {"value", "val", "class", "target", "label", "unnamed: 0", "unnamed", "y", "x"}


def humanise(col, table, source):
    """Readable axis label. Anonymous columns get their dataset as context."""
    raw = str(col).strip()
    pretty = raw.replace("_", " ").strip()
    if pretty.islower():
        pretty = pretty[:1].upper() + pretty[1:]
    if ANON_RE.match(raw) or raw.lower() in GENERIC:
        ctx = str(table).replace("_", " ").strip()
        # openml tables are stored as "<did>_<name>"; drop the numeric id
        if source == "openml" and "_" in str(table):
            ctx = str(table).split("_", 1)[1].replace("_", " ").replace("-", " ")
        ctx = ctx[:38].strip()
        return f"{pretty} [{ctx}]" if ctx else pretty
    return pretty


def _is_identifier(name, s):
    nm = str(name).lower()
    if s.nunique() == len(s) and len(s) > 20:
        if any(p in nm for p in ID_PAT):
            return True
        if pd.api.types.is_numeric_dtype(s):
            v = np.sort(s.dropna().values)
            if len(v) > 2:
                d = np.diff(v)
                if len(d) and np.allclose(d, d[0]) and d[0] > 0:
                    return True
    return False


def _looks_temporal(name, s):
    nm = str(name).lower()
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if any(p in nm for p in TIME_PAT):
        if pd.api.types.is_numeric_dtype(s):
            v = s.dropna()
            if len(v) and 1500 <= v.min() and v.max() <= 2200:
                return True
            return True
        return True
    return False


def profile(df):
    """Split columns into roles used to enumerate plottable views."""
    num, cat, tim, ident = [], [], [], []
    for c in df.columns:
        s = df[c]
        try:
            if _is_identifier(c, s):
                ident.append(c)
                continue
            if _looks_temporal(c, s):
                tim.append(c)
                continue
            if pd.api.types.is_numeric_dtype(s):
                nun = s.nunique(dropna=True)
                if nun > 15:
                    num.append(c)
                elif 2 <= nun <= 12:
                    cat.append(c)
            else:
                nun = s.nunique(dropna=True)
                if 2 <= nun <= 12:
                    cat.append(c)
        except Exception:
            continue
    return num[:MAX_NUM_COLS], cat[:8], tim[:3], ident


def _sub(a, seed, cap=MAX_POINTS):
    a = np.asarray(a)
    if len(a) <= cap:
        return a, None
    idx = np.sort(np.random.default_rng(seed).choice(len(a), cap, replace=False))
    return a[idx], idx


def mine_table(rec):
    """Return verified candidates for one source table."""
    out = []
    per_class = Counter()

    def emit(cand):
        k = cand["label"]
        cap = MAX_PER_TABLE_CLASS_TS if cand["arity"] == "timeseries" else MAX_PER_TABLE_CLASS
        if per_class[k] >= cap:
            return
        per_class[k] += 1
        out.append(cand)

    try:
        df = pd.read_csv(ROOT / rec["path"], low_memory=False)
    except Exception:
        return out
    if df.empty:
        return out
    df = df.loc[:, ~df.columns.duplicated()]
    num, cat, tim, _ident = profile(df)
    sid = rec["source_id"]
    seed = abs(hash(sid)) % (2 ** 31)

    HL = lambda c: humanise(c, rec["name"], rec["source"])

    base = dict(provenance="real", source_id=sid, source=rec["source"],
                source_url=rec.get("url"), license=rec.get("license"),
                table=rec["name"])

    # ---- univariate ------------------------------------------------------
    for c in num:
        v = df[c].dropna().astype(float).values
        v = v[np.isfinite(v)]
        if len(v) < tax.MIN_N_UNI:
            continue
        v, _ = _sub(v, seed + hash(str(c)) % 1000)
        lab, st, _why = tax.verify_univariate(v)
        if lab:
            emit(dict(base, arity="univariate", label=lab, stats=st,
                      columns=[str(c)], data={"x": [float(z) for z in v]},
                      axis_labels={"x": HL(c), "y": "Frequency"}))

    # ---- bivariate -------------------------------------------------------
    pairs = [(a, b) for i, a in enumerate(num) for b in num[i + 1:]][:MAX_PAIRS]
    for a, b in pairs:
        d = df[[a, b]].dropna()
        if len(d) < tax.MIN_N_BI:
            continue
        d = d.astype(float)
        keep = np.isfinite(d[a].values) & np.isfinite(d[b].values)
        d = d[keep]
        if len(d) < tax.MIN_N_BI:
            continue
        if len(d) > MAX_POINTS:
            idx = np.sort(np.random.default_rng(seed + 7).choice(len(d), MAX_POINTS, replace=False))
            d = d.iloc[idx]
        x, y = d[a].values, d[b].values
        lab, st, _why = tax.verify_bivariate(x, y)
        if lab:
            emit(dict(base, arity="bivariate", label=lab, stats=st,
                      columns=[str(a), str(b)],
                      data={"x": [float(z) for z in x], "y": [float(z) for z in y]},
                      axis_labels={"x": HL(a), "y": HL(b)}))

    # ---- time series -----------------------------------------------------
    # long entity/time/value layout (World Bank, OWID) -> one series per entity
    # pandas 3 gives string columns a dedicated `str` dtype, not `object`, so an
    # `== object` test silently matches nothing and every entity-long table
    # (World Bank, OWID) collapses into one duplicate-timestamp series.
    ent_cols = [c for c in df.columns
                if c not in tim
                and not pd.api.types.is_numeric_dtype(df[c])
                and pd.api.types.is_string_dtype(df[c])
                and 3 <= df[c].nunique() <= 400]
    for tcol in tim[:1]:
        tvals = pd.to_numeric(df[tcol], errors="coerce")
        if tvals.notna().sum() < tax.MIN_N_TS:
            tvals = pd.to_datetime(df[tcol], errors="coerce").astype("int64", errors="ignore")
            tvals = pd.to_numeric(tvals, errors="coerce")
        if tvals.notna().sum() < tax.MIN_N_TS:
            continue

        if ent_cols:
            ecol = ent_cols[0]
            ents = df[ecol].value_counts()
            ents = ents[ents >= tax.MIN_N_TS].index[:MAX_ENTITIES]
            for vcol in num[:4]:
                for e in ents:
                    sel = df[ecol] == e
                    tt = tvals[sel].values
                    yy = pd.to_numeric(df.loc[sel, vcol], errors="coerce").values
                    ok = np.isfinite(tt) & np.isfinite(yy)
                    if ok.sum() < tax.MIN_N_TS:
                        continue
                    lab, st, _why = tax.verify_timeseries(tt[ok], yy[ok])
                    if lab:
                        o = np.argsort(tt[ok])
                        emit(dict(base, arity="timeseries", label=lab, stats=st,
                                  columns=[str(tcol), str(vcol)], entity=str(e),
                                  data={"t": [float(z) for z in tt[ok][o]],
                                        "y": [float(z) for z in yy[ok][o]]},
                                  axis_labels={"x": HL(tcol), "y": f"{HL(vcol)} - {e}"}))
        else:
            for vcol in num[:6]:
                tt = tvals.values
                yy = pd.to_numeric(df[vcol], errors="coerce").values
                ok = np.isfinite(tt) & np.isfinite(yy)
                if ok.sum() < tax.MIN_N_TS:
                    continue
                lab, st, _why = tax.verify_timeseries(tt[ok], yy[ok])
                if lab:
                    o = np.argsort(tt[ok])
                    emit(dict(base, arity="timeseries", label=lab, stats=st,
                              columns=[str(tcol), str(vcol)],
                              data={"t": [float(z) for z in tt[ok][o]],
                                    "y": [float(z) for z in yy[ok][o]]},
                              axis_labels={"x": HL(tcol), "y": HL(vcol)}))

    # ---- grouped ---------------------------------------------------------
    for g in cat[:5]:
        for v in num[:6]:
            d = df[[g, v]].dropna()
            if len(d) < tax.MIN_N_GRP:
                continue
            if len(d) > MAX_POINTS:
                idx = np.sort(np.random.default_rng(seed + 13).choice(len(d), MAX_POINTS, replace=False))
                d = d.iloc[idx]
            vals = pd.to_numeric(d[v], errors="coerce").values
            grps = d[g].astype(str).values
            lab, st, _why = tax.verify_grouped(vals, grps)
            if lab:
                ok = np.isfinite(vals)
                emit(dict(base, arity="grouped", label=lab, stats=st,
                          columns=[str(g), str(v)],
                          data={"group": [str(z) for z in grps[ok]],
                                "value": [float(z) for z in vals[ok]]},
                          axis_labels={"x": HL(g), "y": HL(v)}))

    # ---- composition -----------------------------------------------------
    for c in cat[:6]:
        labs = df[c].dropna().astype(str).values
        if len(labs) < tax.MIN_N_GRP:
            continue
        lab, st, _why = tax.verify_composition(labs)
        if lab:
            emit(dict(base, arity="composition", label=lab, stats=st,
                      columns=[str(c)], data={"category": [str(z) for z in labs]},
                      axis_labels={"x": HL(c), "y": "Count"}))

    return out


def main():
    manifest = json.loads((RAW / "manifest.json").read_text())
    print(f"mining {len(manifest)} real source tables "
          f"(cap {MAX_PER_TABLE_CLASS}/class/table for source diversity)")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    counts = Counter()
    by_source = defaultdict(Counter)
    n_written = 0

    with open(OUT, "w") as fh, ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        futs = {ex.submit(mine_table, rec): rec["source_id"] for rec in manifest}
        bar = tqdm(as_completed(futs), total=len(futs), unit="table",
                   desc="mining real tables", ncols=100)
        for fu in bar:
            try:
                cands = fu.result()
            except Exception:
                continue
            for c in cands:
                fh.write(json.dumps(c) + "\n")
                counts[c["label"]] += 1
                by_source[c["source"]][c["label"]] += 1
                n_written += 1
            bar.set_postfix(found=n_written, classes=len(counts))

    print(f"\nwrote {n_written:,} verified real candidates -> {OUT}")
    print(f"{'class':32s} {'count':>8s}   {'quota gap to 1000':>18s}")
    print("-" * 64)
    for c in tax.CLASS_NAMES:
        n = counts[c]
        gap = max(0, 1000 - n)
        flag = "" if n >= 1000 else ("  NEEDS SYNTH" if n < 1000 else "")
        print(f"{c:32s} {n:>8,}   {gap:>18,}{flag}")
    print("-" * 64)
    print(f"{'TOTAL':32s} {n_written:>8,}")
    (CACHE / "real_class_counts.json").write_text(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
