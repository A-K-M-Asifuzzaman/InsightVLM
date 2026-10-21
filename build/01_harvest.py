"""
01_harvest.py — Collect authentic public tabular + time-series data.

Every source table is saved as CSV under raw_sources/<source>/<name>.csv with a
provenance record (url, license, shape) in raw_sources/manifest.json.

Sources (all public, redistributable or linkable):
  seaborn-data   BSD-3      github.com/mwaskom/seaborn-data
  vega-datasets  BSD-3      github.com/vega/vega-datasets
  openml         per-set    openml.org  (license recorded per dataset)
  worldbank      CC-BY-4.0  api.worldbank.org
  owid           CC-BY-4.0  ourworldindata.org
"""
import io
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_sources"
CACHE = ROOT / "cache"
RAW.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "InsightVLM-research-dataset-builder/1.0 (academic use)"}
TIMEOUT = 45

# Bounds that make a table useful for chart mining
MIN_ROWS, MAX_ROWS = 40, 60_000
MIN_COLS, MAX_COLS = 2, 80

manifest = []


def _get(url, timeout=TIMEOUT):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _save(source, name, df, url, license_, extra=None):
    """Persist one source table + its provenance row."""
    if df is None or df.empty:
        return False
    df = df.loc[:, ~df.columns.duplicated()]
    if not (MIN_ROWS <= len(df) <= MAX_ROWS):
        return False
    if not (MIN_COLS <= df.shape[1] <= MAX_COLS):
        return False

    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))[:90]
    out_dir = RAW / source
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{safe}.csv"
    try:
        df.to_csv(path, index=False)
    except Exception:
        return False

    rec = {
        "source_id": f"{source}/{safe}",
        "source": source,
        "name": str(name),
        "path": str(path.relative_to(ROOT)),
        "url": url,
        "license": license_,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns][:80],
    }
    if extra:
        rec.update(extra)
    manifest.append(rec)
    return True


# --------------------------------------------------------------------------
# seaborn-data
# --------------------------------------------------------------------------
SEABORN = [
    "anagrams", "anscombe", "attention", "brain_networks", "car_crashes",
    "diamonds", "dots", "dowjones", "exercise", "flights", "fmri", "geyser",
    "glue", "healthexp", "iris", "mpg", "penguins", "planets", "seaice",
    "taxis", "tips", "titanic",
]


def harvest_seaborn():
    base = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/{}.csv"
    ok = 0
    for name in SEABORN:
        url = base.format(name)
        try:
            df = pd.read_csv(io.BytesIO(_get(url)))
            ok += _save("seaborn", name, df, url, "BSD-3-Clause")
        except Exception as e:
            print(f"  seaborn/{name}: {e}", file=sys.stderr)
    print(f"seaborn-data: {ok} tables")


# --------------------------------------------------------------------------
# vega-datasets
# --------------------------------------------------------------------------
def harvest_vega():
    idx_url = "https://api.github.com/repos/vega/vega-datasets/contents/data"
    try:
        files = json.loads(_get(idx_url))
    except Exception as e:
        print(f"  vega index failed: {e}", file=sys.stderr)
        return
    ok = 0
    for f in files:
        name = f["name"]
        if not name.endswith((".csv", ".json")):
            continue
        # topology / graph files are not tabular
        if any(k in name for k in ("london", "us-10m", "world-110m", "flare", "miserables", "airports")):
            continue
        url = f["download_url"]
        try:
            raw = _get(url)
            if name.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(raw))
            else:
                obj = json.loads(raw)
                if not isinstance(obj, list) or not obj or not isinstance(obj[0], dict):
                    continue
                df = pd.DataFrame(obj)
            ok += _save("vega", name.rsplit(".", 1)[0], df, url, "BSD-3-Clause")
        except Exception:
            continue
    print(f"vega-datasets: {ok} tables")


# --------------------------------------------------------------------------
# OpenML — filtered list, parquet download
# --------------------------------------------------------------------------
def _openml_pq_url(did):
    return f"https://data.openml.org/datasets/{did // 10000:04d}/{did:04d}/dataset_{did}.pq"


def harvest_openml(target=600):
    list_url = (
        "https://www.openml.org/api/v1/json/data/list"
        f"/number_instances/{MIN_ROWS}..{MAX_ROWS}"
        "/number_features/4..60"
        "/status/active/limit/4000"
    )
    try:
        payload = json.loads(_get(list_url, timeout=120))
        entries = payload["data"]["dataset"]
    except Exception as e:
        print(f"  openml list failed: {e}", file=sys.stderr)
        return
    print(f"  openml: {len(entries)} candidate datasets listed")

    # Prefer datasets with a decent numeric-feature count; dedupe by name (keep lowest did)
    seen, cand = set(), []
    for e in entries:
        nm = e.get("name", "")
        base = nm.lower().split("_")[0]
        if base in seen:
            continue
        seen.add(base)
        q = {d["name"]: d["value"] for d in e.get("quality", [])}
        try:
            n_num = float(q.get("NumberOfNumericFeatures", 0))
            n_inst = float(q.get("NumberOfInstances", 0))
        except (TypeError, ValueError):
            continue
        if n_num < 2 or n_inst < MIN_ROWS:
            continue
        cand.append((int(e["did"]), nm, n_num, n_inst))

    cand.sort(key=lambda t: (-min(t[2], 12), t[0]))
    cand = cand[: target * 2]
    print(f"  openml: {len(cand)} after filtering, downloading up to {target}")

    got = [0]

    def fetch(item):
        did, nm, _, _ = item
        if got[0] >= target:
            return None
        try:
            raw = _get(_openml_pq_url(did), timeout=90)
            df = pd.read_parquet(io.BytesIO(raw))
            return (did, nm, df)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(fetch, c) for c in cand]
        for fu in as_completed(futs):
            r = fu.result()
            if r is None:
                continue
            did, nm, df = r
            if got[0] >= target:
                continue
            if _save("openml", f"{did}_{nm}", df, f"https://www.openml.org/d/{did}",
                     "see openml.org/d/%d" % did, {"openml_did": did}):
                got[0] += 1
    print(f"openml: {got[0]} tables")


# --------------------------------------------------------------------------
# World Bank — authentic long time series (country x year per indicator)
# --------------------------------------------------------------------------
WB_INDICATORS = [
    "NY.GDP.PCAP.CD", "SP.DYN.LE00.IN", "SP.POP.TOTL", "FP.CPI.TOTL.ZG",
    "SL.UEM.TOTL.ZS", "EN.GHG.CO2.PC.CE.AR5", "SE.PRM.ENRR", "SP.DYN.TFRT.IN",
    "IT.NET.USER.ZS", "NE.EXP.GNFS.ZS", "NE.IMP.GNFS.ZS", "SH.XPD.CHEX.GD.ZS",
    "AG.LND.FRST.ZS", "EG.USE.ELEC.KH.PC", "SP.URB.TOTL.IN.ZS", "SP.DYN.IMRT.IN",
    "NY.GDP.MKTP.KD.ZG", "GC.DOD.TOTL.GD.ZS", "SI.POV.GINI", "MS.MIL.XPND.GD.ZS",
    "TX.VAL.TECH.MF.ZS", "SH.STA.SUIC.P5", "ER.H2O.FWTL.ZS", "EG.FEC.RNEW.ZS",
]


def harvest_worldbank():
    ok = 0
    for ind in WB_INDICATORS:
        url = (f"https://api.worldbank.org/v2/country/all/indicator/{ind}"
               "?format=json&per_page=20000&date=1970:2023")
        try:
            payload = json.loads(_get(url, timeout=90))
            rows = payload[1] if len(payload) > 1 and payload[1] else []
        except Exception:
            continue
        # Keep the human-readable indicator name and use it as the value column.
        # Harvesting it as a bare "value" column produced axis labels like
        # "value (Ethiopia)", which tells a reader (or a VLM) nothing.
        ind_name = ""
        recs = []
        for r in rows:
            v = r.get("value")
            if v is None:
                continue
            ind_name = ind_name or (r.get("indicator", {}) or {}).get("value", "") or ind
            recs.append({
                "country": r["country"]["value"],
                "country_code": r.get("countryiso3code") or "",
                "year": int(r["date"]),
                "value": float(v),
            })
        if not recs:
            continue
        df = pd.DataFrame(recs).rename(columns={"value": ind_name[:90] or ind})
        # drop aggregates (they have blank iso3 or are known regions)
        df = df[df["country_code"].str.len() == 3]
        ok += _save("worldbank", ind, df,
                    f"https://data.worldbank.org/indicator/{ind}",
                    "CC-BY-4.0", {"indicator": ind, "indicator_name": ind_name,
                                  "is_timeseries": True})
    print(f"worldbank: {ok} indicator tables")


# --------------------------------------------------------------------------
# Our World in Data — grapher CSV endpoints
# --------------------------------------------------------------------------
OWID_SLUGS = [
    "co2-emissions-per-capita", "life-expectancy", "share-of-population-in-extreme-poverty",
    "annual-co2-emissions-per-country", "population-density", "gdp-per-capita-worldbank",
    "child-mortality", "human-development-index", "share-of-individuals-using-the-internet",
    "meat-production-tonnes", "cereal-yields", "renewable-share-energy",
    "primary-energy-cons", "electricity-generation", "annual-number-of-deaths-by-cause",
    "sea-surface-temperature-anomaly", "ice-sheet-mass-balance", "global-precipitation-anomaly",
    "number-of-natural-disaster-events", "urban-and-rural-population",
    "average-years-of-schooling", "daily-caloric-supply-per-person",
    "prevalence-of-undernourishment", "military-expenditure-share-gdp",
]


def harvest_owid():
    ok = 0
    for slug in OWID_SLUGS:
        url = f"https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=true"
        try:
            df = pd.read_csv(io.BytesIO(_get(url, timeout=90)))
            ok += _save("owid", slug, df, f"https://ourworldindata.org/grapher/{slug}",
                        "CC-BY-4.0", {"is_timeseries": True})
        except Exception:
            continue
    print(f"owid: {ok} tables")


# --------------------------------------------------------------------------
# UCI classics not always well-formed on OpenML
# --------------------------------------------------------------------------
UCI_DIRECT = {
    "wine_quality_red": ("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", ";"),
    "wine_quality_white": ("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", ";"),
    "abalone": ("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", ","),
    "airfoil_self_noise": ("https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat", "\t"),
    "concrete_slump": ("https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data", ","),
    "yacht_hydrodynamics": ("https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data", r"\s+"),
}


def harvest_uci():
    ok = 0
    for name, (url, sep) in UCI_DIRECT.items():
        try:
            raw = _get(url, timeout=60)
            df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")
            if df.shape[1] < 3:
                continue
            ok += _save("uci", name, df, url, "CC-BY-4.0 (UCI ML Repository)")
        except Exception:
            continue
    print(f"uci: {ok} tables")


if __name__ == "__main__":
    t0 = time.time()
    harvest_seaborn()
    harvest_vega()
    harvest_uci()
    harvest_worldbank()
    harvest_owid()
    harvest_openml(target=600)

    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    by_src = {}
    total_rows = 0
    for m in manifest:
        by_src[m["source"]] = by_src.get(m["source"], 0) + 1
        total_rows += m["n_rows"]
    print("\n" + "=" * 60)
    print(f"HARVEST COMPLETE in {time.time() - t0:.0f}s")
    print(f"  tables : {len(manifest)}")
    print(f"  rows   : {total_rows:,}")
    for k, v in sorted(by_src.items()):
        print(f"  {k:12s} {v}")
    print(f"  manifest -> {RAW / 'manifest.json'}")
