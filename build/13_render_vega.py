"""
13_render_vega.py — Held-out third renderer: the real test charts drawn with Vega-Lite.

    python build/13_render_vega.py --workers 6     # -> dataset/images_vega/<label>/<id>.png

Training uses matplotlib (and, with renderer augmentation, Plotly), so this is the
one renderer no model ever sees: its accuracy is the unseen-tool generalisation
number. Same data tables as every other render, no label-dependent marks. Themes
that imitate a training renderer (ggplot2, fivethirtyeight: both are matplotlib
styles in training) and dark themes are excluded.
"""
import argparse
import importlib
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
P = importlib.import_module("11_render_plotly")   # shared job list + date handling

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dataset" / "images_vega"
THEMES = ["default", "carbonwhite", "excel", "googlecharts", "latimes", "powerbi",
          "quartz", "urbaninstitute", "vox"]
W, H = 620, 430   # plot area; the saved PNG adds axes and title around it


def chart(r, axis):
    d = json.loads(r.data_table_json)
    xl, yl = axis.get("x") or "x", axis.get("y") or "y"
    ct, a = r.chart_type, r.arity
    rng = np.random.default_rng(zlib.crc32(r.sample_id.encode()))
    if a == "univariate":
        df = pd.DataFrame({"x": np.asarray(d["x"], float)})
        base = alt.Chart(df)
        X = alt.X("x:Q", title=xl)
        if ct == "boxplot":
            c = base.mark_boxplot(extent=1.5).encode(y=alt.Y("x:Q", title=xl))
        elif ct == "ecdf":
            c = base.transform_window(cum="count()", sort=[{"field": "x"}]) \
                .transform_joinaggregate(n="count()").transform_calculate(p="datum.cum / datum.n") \
                .mark_line(interpolate="step-after").encode(X, y=alt.Y("p:Q", title="cumulative share"))
        elif ct == "strip":
            df["j"] = rng.uniform(-1, 1, len(df))
            c = alt.Chart(df).mark_circle(size=18, opacity=0.5).encode(
                x=alt.X("j:Q", axis=None, scale=alt.Scale(domain=[-3, 3])), y=alt.Y("x:Q", title=xl))
        elif ct in ("kde", "violin"):
            c = base.transform_density("x", as_=["x", "density"]).mark_area(opacity=0.7).encode(
                X, y=alt.Y("density:Q", title="density"))
        elif ct == "qq":
            from scipy import stats
            (osm, osr), (sl, ic, _) = stats.probplot(df.x.values, dist="norm")
            q = pd.DataFrame({"t": osm, "s": osr, "fit": sl * osm + ic})
            pts = alt.Chart(q).mark_point(size=20).encode(
                x=alt.X("t:Q", title="theoretical quantile"), y=alt.Y("s:Q", title="sample quantile"))
            c = pts + alt.Chart(q).mark_line(color="gray").encode(x="t:Q", y="fit:Q")
        else:   # histogram, and boxen (histogram is valid for every distribution class)
            c = base.mark_bar().encode(alt.X("x:Q", bin=alt.Bin(maxbins=30), title=xl),
                                       y=alt.Y("count():Q", title="count"))
        title = xl
    elif a == "bivariate":
        df = pd.DataFrame({"x": np.asarray(d["x"], float), "y": np.asarray(d["y"], float)})
        X, Y = alt.X("x:Q", title=xl), alt.Y("y:Q", title=yl)
        if ct == "hexbin":
            c = alt.Chart(df).mark_rect().encode(alt.X("x:Q", bin=alt.Bin(maxbins=30), title=xl),
                                                 alt.Y("y:Q", bin=alt.Bin(maxbins=30), title=yl),
                                                 color=alt.Color("count():Q", title="count"))
        elif ct == "bubble":
            df["s"] = rng.uniform(10, 120, len(df))
            c = alt.Chart(df).mark_circle(opacity=0.5).encode(X, Y, size=alt.Size("s:Q", legend=None))
        else:
            c = alt.Chart(df).mark_circle(size=22, opacity=0.6).encode(X, Y)
            if ct == "regplot":
                c = c + c.transform_regression("x", "y").mark_line(color="gray")
        title = f"{yl} and {xl}"
    elif a == "timeseries":
        t = P._t(d["t"])
        df = pd.DataFrame({"t": t, "y": np.asarray(d["y"], float)})
        X = alt.X("t:T" if np.issubdtype(df.t.dtype, np.datetime64) else "t:Q", title=xl)
        Y = alt.Y("y:Q", title=yl)
        if ct == "area":
            c = alt.Chart(df).mark_area(opacity=0.6, line=True).encode(X, Y)
        elif ct == "step":
            c = alt.Chart(df).mark_line(interpolate="step-after").encode(X, Y)
        else:
            c = alt.Chart(df).mark_line(point=ct == "line_marker").encode(X, Y)
        title = yl
    elif a == "grouped":
        df = pd.DataFrame({"g": [str(v) for v in d["group"]], "v": np.asarray(d["value"], float)})
        G, V = alt.X("g:N", title=xl), alt.Y("v:Q", title=yl)
        if ct in ("grouped_box", "grouped_violin"):   # box is valid for every group class
            c = alt.Chart(df).mark_boxplot().encode(G, V, color=alt.Color("g:N", legend=None))
        elif ct == "grouped_strip":
            df["j"] = rng.uniform(-0.3, 0.3, len(df))
            c = alt.Chart(df).mark_circle(size=16, opacity=0.5).encode(
                G, V, xOffset="j:Q", color=alt.Color("g:N", legend=None))
        elif ct == "grouped_bar":
            c = alt.Chart(df).mark_bar().encode(G, alt.Y("mean(v):Q", title=yl),
                                                color=alt.Color("g:N", legend=None)) + \
                alt.Chart(df).mark_errorbar(extent="stdev").encode(G, V)
        else:   # pointplot
            c = alt.Chart(df).mark_line(point=True).encode(G, alt.Y("mean(v):Q", title=yl)) + \
                alt.Chart(df).mark_errorbar(extent="stdev").encode(G, V)
        title = f"{yl} by {xl}"
    else:   # composition
        vc = pd.Series([str(v) for v in d["category"]]).value_counts().reset_index()
        vc.columns = ["cat", "n"]
        if ct == "pie":
            c = alt.Chart(vc).mark_arc().encode(theta="n:Q", color=alt.Color("cat:N", title=xl))
        elif ct == "barh":
            c = alt.Chart(vc).mark_bar().encode(y=alt.Y("cat:N", sort="-x", title=xl), x=alt.X("n:Q", title="count"))
        else:
            c = alt.Chart(vc).mark_bar().encode(x=alt.X("cat:N", sort="-y", title=xl), y=alt.Y("n:Q", title="count"))
        title = xl
    return c.properties(width=W, height=H, title=title[:70])


def _work(chunk):
    import vl_convert as vlc
    alt.data_transformers.disable_max_rows()
    fails = []
    for row, axis in chunk:
        p = OUT / row.label / f"{row.sample_id}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            spec = chart(row, axis).to_dict()
            theme = THEMES[zlib.crc32(row.sample_id.encode()) % len(THEMES)]
            png = vlc.vegalite_to_png(spec, scale=1, theme=None if theme == "default" else theme)
            tmp = p.with_suffix(".tmp.png")
            tmp.write_bytes(png)
            tmp.rename(p)
        except Exception as e:   # one bad chart must not stop the rest
            fails.append((row.sample_id, f"{type(e).__name__}: {str(e)[:200]}"))
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="render only the first N (smoke test)")
    a = ap.parse_args()
    jobs = [j for j in P._jobs("test", None) if not (OUT / j[0].label / f"{j[0].sample_id}.png").exists()]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"rendering {len(jobs):,} charts with {a.workers} worker(s)", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        fails = [f for fs in ex.map(_work, [jobs[i::a.workers] for i in range(a.workers)]) for f in fs]
    for sid, err in fails:
        print("  FAIL", sid, err)
    print(f"{len(jobs) - len(fails):,} rendered -> {OUT}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
