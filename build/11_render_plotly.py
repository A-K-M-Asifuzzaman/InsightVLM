"""
11_render_plotly.py — Re-draw the real test charts with a second rendering library.

    python build/11_render_plotly.py                          # real test charts: the eval set
    python build/11_render_plotly.py --split train --workers 6  # renderer augmentation
    python build/11_render_plotly.py --records cache/records_extra.jsonl --workers 6

Every training image comes from matplotlib/seaborn, so a model could be learning
matplotlib's look instead of the statistics. This renders the SAME data tables
(the ones the labels were verified on) with Plotly: different fonts, gridlines,
colours, marker shapes, bin rules and layout. Scoring on these is the
cross-renderer generalisation test.

Chart types map to Plotly's nearest native form. Where Plotly has no equivalent
(boxen, hexbin, pointplot) the substitute is one the taxonomy already lists as
valid for every class that chart type can carry, so no label becomes unreadable.
No label-dependent marks are drawn (see 06_render.py --clean).
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
OUT = DATA / "images_plotly"
W, H = 768, 576
# no "seaborn" template: it imitates the training renderer
TEMPLATES = ["plotly", "plotly_white", "ggplot2", "simple_white", "presentation"]


def _t(v):
    t = np.asarray(v, float)
    # World Bank / OWID dates are stored as epoch microseconds
    return pd.to_datetime(t, unit="us") if t.size and np.nanmin(t) > 1e14 else t


def figure(r, axis):
    d = json.loads(r.data_table_json)
    # the same axis labels the matplotlib renderer used
    xl, yl = axis.get("x") or "x", axis.get("y") or "y"
    ct, a = r.chart_type, r.arity

    if a == "univariate":
        x = np.asarray(d["x"], float)
        if ct in ("boxplot",):
            f = px.box(y=x, points="outliers", labels={"y": xl})
        elif ct == "violin":
            f = px.violin(y=x, box=True, labels={"y": xl})
        elif ct == "ecdf":
            f = px.ecdf(x=x, labels={"x": xl})
        elif ct == "strip":
            f = px.strip(y=x, labels={"y": xl})
        elif ct == "kde":
            from scipy.stats import gaussian_kde
            g = np.linspace(x.min(), x.max(), 300)
            f = px.area(x=g, y=gaussian_kde(x)(g), labels={"x": xl, "y": "density"})
        elif ct == "qq":
            from scipy import stats
            (osm, osr), (sl, ic, _) = stats.probplot(x, dist="norm")
            f = px.scatter(x=osm, y=osr, labels={"x": "theoretical quantile", "y": "sample quantile"})
            f.add_trace(go.Scatter(x=osm, y=sl * osm + ic, mode="lines", showlegend=False))
        else:  # histogram, and boxen (histogram is valid for every distribution class)
            f = px.histogram(x=x, labels={"x": xl}, marginal="box" if ct == "boxen" else None)
        title = f"{xl}"
    elif a == "bivariate":
        x, y = np.asarray(d["x"], float), np.asarray(d["y"], float)
        lab = {"x": xl, "y": yl}
        if ct == "hexbin":
            f = px.density_heatmap(x=x, y=y, nbinsx=30, nbinsy=30, labels=lab)
        elif ct == "jointplot":
            f = px.scatter(x=x, y=y, marginal_x="histogram", marginal_y="histogram", labels=lab)
        elif ct == "regplot":
            f = px.scatter(x=x, y=y, trendline="ols", labels=lab)
        elif ct == "bubble":
            s = np.random.default_rng(zlib.crc32(r.sample_id.encode())).uniform(4, 20, len(x))
            f = px.scatter(x=x, y=y, size=s, labels=lab)
        else:
            f = px.scatter(x=x, y=y, labels=lab)
        title = f"{yl} and {xl}"
    elif a == "timeseries":
        t, y = _t(d["t"]), np.asarray(d["y"], float)
        lab = {"x": xl, "y": yl}
        if ct == "area":
            f = px.area(x=t, y=y, labels=lab)
        elif ct == "step":
            f = px.line(x=t, y=y, line_shape="hv", labels=lab)
        else:
            f = px.line(x=t, y=y, markers=ct == "line_marker", labels=lab)
        title = f"{yl}"
    elif a == "grouped":
        df = pd.DataFrame({"group": [str(g) for g in d["group"]], "value": np.asarray(d["value"], float)})
        df = df.sort_values("group")
        lab = {"group": xl, "value": yl}
        if ct == "grouped_violin":
            f = px.violin(df, x="group", y="value", color="group", labels=lab)
        elif ct == "grouped_strip":
            f = px.strip(df, x="group", y="value", color="group", labels=lab)
        elif ct in ("grouped_bar", "pointplot"):
            s = df.groupby("group").value.agg(["mean", "std"]).reset_index()
            lab = {"group": xl, "mean": yl}
            f = (px.bar(s, x="group", y="mean", error_y="std", color="group", labels=lab)
                 if ct == "grouped_bar" else
                 px.line(s, x="group", y="mean", error_y="std", markers=True, labels=lab))
        else:
            f = px.box(df, x="group", y="value", color="group", labels=lab)
        f.update_layout(showlegend=False)
        title = f"{yl} by {xl}"
    else:  # composition
        vc = pd.Series([str(c) for c in d["category"]]).value_counts()
        if ct == "pie":
            f = px.pie(names=vc.index, values=vc.values)
        elif ct == "barh":
            f = px.bar(x=vc.values, y=vc.index, orientation="h", labels={"x": "count", "y": xl})
        else:
            f = px.bar(x=vc.index, y=vc.values, labels={"x": xl, "y": "count"})
        title = f"{xl}"
    tpl = TEMPLATES[zlib.crc32(r.sample_id.encode()) % len(TEMPLATES)]
    f.update_layout(template=tpl, title=title[:70], margin=dict(l=70, r=30, t=60, b=60))
    return f


def _jobs(split, records):
    """Rows to render as (namespace-like row, axis labels, output path)."""
    from types import SimpleNamespace as NS
    if records:   # e.g. cache/records_extra.jsonl (supplementary real training pool)
        out = []
        for line in open(records):
            r = json.loads(line)
            row = NS(sample_id=r["sample_id"], label=r["label"], arity=r["arity"],
                     chart_type=r["chart_type"], data_table_json=json.dumps(r["ground_truth"]["data_table"]))
            out.append((row, r.get("axis_labels") or {}))
        return out
    meta = pd.read_parquet(DATA / "tables" / "metadata.parquet")
    df = meta[meta.split == split]
    if split == "test":   # the evaluation set is the real test charts only
        df = df[df.provenance == "real"]
    axes = {}
    with open(DATA / "splits" / f"{split}.jsonl") as fh:
        for line in fh:
            rec = json.loads(line)
            axes[rec["sample_id"]] = rec.get("axis_labels") or {}
    cols = ["sample_id", "label", "arity", "chart_type", "data_table_json"]
    # plain namespaces: pandas' itertuples rows can't be pickled to worker processes
    return [(NS(**dict(zip(cols, v))), axes.get(v[0], {})) for v in df[cols].itertuples(index=False)]


def _work(chunk):
    """Render one chunk with its own persistent browser. Returns failures."""
    import signal
    import kaleido

    def _timeout(*_):
        raise TimeoutError("render hung")
    # a hung headless browser froze every worker of a 14k-chart run at 0% CPU;
    # the watchdog turns a hang into one failed chart and a fresh browser
    signal.signal(signal.SIGALRM, _timeout)
    fails = []
    kaleido.start_sync_server()
    try:
        for row, axis in chunk:
            p = OUT / row.label / f"{row.sample_id}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.png")
            signal.alarm(60)
            try:
                pio.write_image(figure(row, axis), tmp, width=W, height=H)
                tmp.rename(p)
            except Exception as e:   # one bad chart must not stop the rest
                fails.append((row.sample_id, f"{type(e).__name__}: {e}"))
                tmp.unlink(missing_ok=True)
                if isinstance(e, TimeoutError):
                    try:
                        kaleido.stop_sync_server()
                    except Exception:
                        pass
                    kaleido.start_sync_server()
            finally:
                signal.alarm(0)
    finally:
        kaleido.stop_sync_server()
    return fails


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="test (real only, the eval set) or train")
    ap.add_argument("--records", help="render these records instead (e.g. cache/records_extra.jsonl)")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    jobs = [j for j in _jobs(a.split, a.records) if not (OUT / j[0].label / f"{j[0].sample_id}.png").exists()]
    print(f"rendering {len(jobs):,} charts with {a.workers} worker(s)", flush=True)
    # one persistent browser per worker, one chart at a time: batch write_images died
    # silently mid-run and left empty files; this way a crash loses one chart and a
    # rerun resumes where it stopped
    chunks = [jobs[i::a.workers] for i in range(a.workers)]
    with ProcessPoolExecutor(a.workers) as ex:
        fails = [f for fs in ex.map(_work, chunks) for f in fs]
    for sid, err in fails:
        print("  FAIL", sid, err)
    print(f"{len(jobs) - len(fails):,} rendered -> {OUT}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
