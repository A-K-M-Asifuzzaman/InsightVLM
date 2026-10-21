"""
06_render.py — Render each record to a chart image with pixel-valid grounding boxes.

Two rendering rules that the notebook prototype got wrong, and which matter:

  1. NO bbox_inches="tight". Tight-cropping recomputes the canvas *after* the
     axes transform has been read, so every box drawn from ax.transData is
     shifted relative to the file that actually gets saved. The canvas here is
     fixed, so display coordinates and image pixels are the same thing.
  2. Boxes are emitted in image pixel space with the y axis flipped, because
     matplotlib's origin is bottom-left and an image's is top-left.

Each image carries evidence boxes marking the pixels that support the finding:
the outlier band, the fitted trend, the changepoint, the anomalous points, the
dominant wedge. That grounding is what separates this from chart-QA datasets.
"""
import argparse
import json
import os
import sys
import warnings
import zlib
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from tqdm import tqdm                    # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
DATA = ROOT / "dataset"
# env, not globals set in main(): spawned worker processes re-import this module
IMG = Path(os.environ.get("RENDER_OUT", DATA / "images"))

W_PX, H_PX, DPI = 768, 576, 100
FIGSIZE = (W_PX / DPI, H_PX / DPI)

STYLES = [
    ("seaborn-v0_8-whitegrid", "tab10"), ("seaborn-v0_8-darkgrid", "Set2"),
    ("ggplot", "tab10"), ("bmh", "Dark2"), ("fivethirtyeight", "tab10"),
    ("seaborn-v0_8-ticks", "colorblind"), ("default", "tab10"),
    ("seaborn-v0_8-muted", "Set1"),
]


# --clean: draw no label-dependent marks. The trend line (only on increasing/
# decreasing), changepoint line and anomaly circles each reveal the class, so a
# model can read the annotation instead of the data. Boxes are still emitted.
CLEAN = os.environ.get("RENDER_CLEAN") == "1"
# --randstyle: training-only domain randomisation (fonts, line/marker sizes, colours,
# grids, spines, backgrounds, titles). The data and the chart type are untouched, so
# the label is too. Its own seed, so the same chart gets a different look.
RANDSTYLE = os.environ.get("RENDER_RANDSTYLE") == "1"
RAND_CMAPS = ["tab10", "Set1", "Set2", "Dark2", "Paired", "viridis", "plasma", "cividis", "Accent"]


def _rand_rc(rr):
    import colorsys
    tint = lambda: colorsys.hsv_to_rgb(rr.uniform(0, 1), rr.uniform(0, 0.12), rr.uniform(0.93, 1.0))  # noqa: E731
    return {
        "font.family": str(rr.choice(["DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono", "STIXGeneral"])),
        "font.size": float(rr.uniform(7, 13)), "lines.linewidth": float(rr.uniform(0.7, 3.2)),
        "lines.markersize": float(rr.uniform(2, 8)), "axes.linewidth": float(rr.uniform(0.4, 2.0)),
        "axes.grid": bool(rr.random() < 0.5), "grid.alpha": float(rr.uniform(0.2, 0.9)),
        "grid.linestyle": str(rr.choice(["-", "--", ":"])), "axes.facecolor": tint(),
        "figure.facecolor": tint() if rr.random() < 0.3 else "white",
        "axes.spines.top": bool(rr.random() < 0.5), "axes.spines.right": bool(rr.random() < 0.5),
        "xtick.direction": str(rr.choice(["in", "out"])), "ytick.direction": str(rr.choice(["in", "out"])),
    }


def _seed(sid):
    # crc32, not hash(): str hashing is salted per process, so styles were not reproducible
    return zlib.crc32(sid.encode()) % (2 ** 31)


def _px(fig, ax, xd, yd):
    """Data coords -> image pixel coords (origin top-left)."""
    pts = ax.transData.transform(np.column_stack([np.atleast_1d(xd), np.atleast_1d(yd)]))
    return pts[:, 0], H_PX - pts[:, 1]


def _box_from_data(fig, ax, x0, y0, x1, y1, label):
    xs, ys = _px(fig, ax, [x0, x1], [y0, y1])
    a, b = sorted([float(xs[0]), float(xs[1])])
    c, d = sorted([float(ys[0]), float(ys[1])])
    return {"label": label,
            "bbox": [round(max(0, a), 1), round(max(0, c), 1),
                     round(min(W_PX, b), 1), round(min(H_PX, d), 1)]}


def _box_from_artist(fig, artist, label):
    try:
        e = artist.get_window_extent(fig.canvas.get_renderer())
    except Exception:
        return None
    return {"label": label,
            "bbox": [round(max(0, float(e.x0)), 1), round(max(0, H_PX - float(e.y1)), 1),
                     round(min(W_PX, float(e.x1)), 1), round(min(H_PX, H_PX - float(e.y0)), 1)]}


# --------------------------------------------------------------------------
# per-arity drawing. Each returns a list of (kind, args) evidence requests
# resolved to pixel boxes after fig.canvas.draw().
# --------------------------------------------------------------------------
def draw_univariate(ax, rec, r, colour):
    st = rec["ground_truth"]["statistics"]
    x = np.asarray(rec["ground_truth"]["data_table"]["x"], float)
    ct = rec["chart_type"]
    ev = []

    if ct == "histogram":
        nb = int(np.clip(np.sqrt(len(x)) * r.uniform(0.8, 1.4), 12, 45))
        cnt, edges, _ = ax.hist(x, bins=nb, color=colour, edgecolor="white", alpha=0.88)
        i = int(np.argmax(cnt))
        ev.append(("data", edges[i], 0, edges[i + 1], cnt[i], "modal_bin"))
        ax.set_ylabel("Frequency")
    elif ct == "kde":
        from scipy.stats import gaussian_kde
        gr = np.linspace(x.min(), x.max(), 240)
        dens = gaussian_kde(x)(gr)
        ax.fill_between(gr, dens, color=colour, alpha=0.45)
        ax.plot(gr, dens, color=colour, lw=2)
        ev.append(("data", gr[int(np.argmax(dens))], 0, gr[int(np.argmax(dens))], dens.max(), "peak"))
        ax.set_ylabel("Density")
    elif ct == "boxplot":
        bp = ax.boxplot(x, vert=True, patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor(colour)
        bp["boxes"][0].set_alpha(0.8)
        ev.append(("data", 0.75, st["q1"], 1.25, st["q3"], "iqr_box"))
        ev.append(("data", 0.6, st["median"], 1.4, st["median"], "median_line"))
        ax.set_ylabel(rec["axis_labels"]["x"])
        ax.set_xticks([])
    elif ct == "violin":
        vp = ax.violinplot(x, showmedians=True, widths=0.7)
        for b in vp["bodies"]:
            b.set_facecolor(colour)
            b.set_alpha(0.7)
        ev.append(("data", 0.7, st["q1"], 1.3, st["q3"], "iqr_region"))
        ax.set_ylabel(rec["axis_labels"]["x"])
        ax.set_xticks([])
    elif ct == "ecdf":
        s = np.sort(x)
        ax.step(s, np.arange(1, len(s) + 1) / len(s), color=colour, lw=2)
        ev.append(("data", st["median"], 0, st["median"], 0.5, "median_crossing"))
        ax.set_ylabel("Cumulative proportion")
    elif ct == "qq":
        from scipy import stats as sps
        (osm, osr), (sl, ic, _) = sps.probplot(x, dist="norm")
        ax.scatter(osm, osr, s=14, color=colour, alpha=0.75)
        ax.plot(osm, sl * osm + ic, "r--", lw=1.8)
        ev.append(("data", osm.min(), (sl * osm + ic).min(), osm.max(), (sl * osm + ic).max(),
                   "normal_reference_line"))
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles")
    elif ct == "boxen":
        import seaborn as sns
        sns.boxenplot(y=x, color=colour, ax=ax)
        ev.append(("data", -0.4, st["q1"], 0.4, st["q3"], "iqr_box"))
        ax.set_ylabel(rec["axis_labels"]["x"])
    else:  # strip
        jitter = r.normal(0, 0.045, len(x))
        ax.scatter(jitter, x, s=15, color=colour, alpha=0.55)
        ax.set_xlim(-0.35, 0.35)
        ev.append(("data", -0.3, st["q1"], 0.3, st["q3"], "iqr_region"))
        ax.set_ylabel(rec["axis_labels"]["x"])
        ax.set_xticks([])

    if ct in ("histogram", "kde", "ecdf"):
        ax.set_xlabel(rec["axis_labels"]["x"])
    lo, hi = st.get("iqr_lower_fence"), st.get("iqr_upper_fence")
    if lo is not None and st.get("outlier_count", 0) > 0:
        if ct in ("histogram", "kde", "ecdf"):
            if x.max() > hi:
                ev.append(("data", hi, ax.get_ylim()[0], x.max(), ax.get_ylim()[1], "upper_outliers"))
            if x.min() < lo:
                ev.append(("data", x.min(), ax.get_ylim()[0], lo, ax.get_ylim()[1], "lower_outliers"))
    return ev


def draw_bivariate(ax, rec, r, colour):
    st = rec["ground_truth"]["statistics"]
    d = rec["ground_truth"]["data_table"]
    x, y = np.asarray(d["x"], float), np.asarray(d["y"], float)
    ct = rec["chart_type"]
    ev = []

    if ct == "hexbin":
        ax.hexbin(x, y, gridsize=int(r.integers(16, 28)), cmap="viridis", mincnt=1)
    elif ct == "bubble":
        sz = np.abs(r.normal(45, 18, len(x))).clip(8, 130)
        ax.scatter(x, y, s=sz, color=colour, alpha=0.5, edgecolor="white", lw=0.5)
    else:
        ax.scatter(x, y, s=int(r.integers(14, 30)), color=colour, alpha=0.65,
                   edgecolor="white", lw=0.4)

    if ct == "regplot" or (ct == "scatter" and r.random() < 0.45):
        b1, b0 = st.get("linear_slope"), st.get("linear_intercept")
        if b1 is not None:
            gx = np.linspace(x.min(), x.max(), 60)
            ax.plot(gx, b1 * gx + b0, "r-", lw=2, alpha=0.9)
            ev.append(("data", gx[0], b1 * gx[0] + b0, gx[-1], b1 * gx[-1] + b0, "fitted_trend"))

    cls = rec["label"]
    if cls == "rel_heteroscedastic":
        q = np.quantile(x, [2 / 3, 1.0])
        ev.append(("data", q[0], y.min(), q[1], y.max(), "high_variance_region"))
        q2 = np.quantile(x, [0.0, 1 / 3])
        ev.append(("data", q2[0], y.min(), q2[1], y.max(), "low_variance_region"))
    elif cls == "rel_nonlinear_nonmonotonic":
        c2 = st.get("curvature_coef") or 0
        xs = (x - x.mean()) / (x.std() + 1e-12)
        vi = int(np.argmin(xs ** 2))
        ev.append(("data", x[vi] - 0.1 * x.std(), y.min(), x[vi] + 0.1 * x.std(), y.max(),
                   "curvature_vertex" if c2 else "vertex"))

    ax.set_xlabel(rec["axis_labels"]["x"])
    ax.set_ylabel(rec["axis_labels"]["y"])
    return ev


def draw_timeseries(ax, rec, r, colour):
    st = rec["ground_truth"]["statistics"]
    d = rec["ground_truth"]["data_table"]
    t, y = np.asarray(d["t"], float), np.asarray(d["y"], float)
    ct = rec["chart_type"]
    ev = []
    # OpenML/vega dates are stored as epoch microseconds; plotted raw they give a
    # "1.585 1e15" axis. Plot as dates. Everything below (fits, boxes) reads this
    # same t, so evidence boxes stay in the axis's own coordinates.
    is_date = t.size and np.nanmin(t) > 1e14
    if is_date:
        import matplotlib.dates as mdates
        t = mdates.date2num(t.astype("datetime64[us]"))

    if ct == "area":
        ax.fill_between(t, y, np.min(y), color=colour, alpha=0.45)
        ax.plot(t, y, color=colour, lw=1.8)
    elif ct == "step":
        ax.step(t, y, where="mid", color=colour, lw=1.8)
    elif ct == "line_marker":
        ax.plot(t, y, color=colour, lw=1.5, marker="o", ms=3.5, alpha=0.9)
    else:
        ax.plot(t, y, color=colour, lw=1.9)

    cls = rec["label"]
    if cls in ("ts_increasing", "ts_decreasing"):
        b1, b0 = np.polyfit(np.arange(len(y)), y, 1)
        fit = b1 * np.arange(len(y)) + b0
        if not CLEAN:
            ax.plot(t, fit, "r--", lw=1.6, alpha=0.85)
        ev.append(("data", t[0], fit[0], t[-1], fit[-1], "fitted_trend"))
    elif cls == "ts_changepoint":
        ci = st.get("changepoint_index")
        if ci is not None and 0 < ci < len(t):
            if not CLEAN:
                ax.axvline(t[ci], color="red", ls="--", lw=1.5, alpha=0.8)
            span = max(1, len(t) // 25)
            lo = t[max(0, ci - span)]
            hi = t[min(len(t) - 1, ci + span)]
            ev.append(("data", lo, y.min(), hi, y.max(), "changepoint"))
            ev.append(("data", t[0], y[:ci].mean(), t[ci], y[:ci].mean(), "level_before"))
            ev.append(("data", t[ci], y[ci:].mean(), t[-1], y[ci:].mean(), "level_after"))
    elif cls == "ts_anomaly_spike":
        for i in (st.get("anomaly_indices") or [])[:6]:
            if 0 <= i < len(t):
                if not CLEAN:
                    ax.scatter([t[i]], [y[i]], s=110, facecolor="none", edgecolor="red", lw=2,
                               zorder=5)
                w = (t[-1] - t[0]) / max(len(t), 1) * 2.5
                h = (y.max() - y.min()) * 0.06
                ev.append(("data", t[i] - w, y[i] - h, t[i] + w, y[i] + h, "anomaly_point"))
    elif cls == "ts_seasonal":
        p = st.get("seasonal_period")
        if p and p > 1:
            step = (t[-1] - t[0]) / max(len(t) - 1, 1)
            ev.append(("data", t[0], y.min(), t[0] + p * step, y.max(), "one_seasonal_cycle"))

    if is_date:
        loc = mdates.AutoDateLocator(maxticks=7)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.set_xlabel(rec["axis_labels"]["x"])
    ax.set_ylabel(rec["axis_labels"]["y"])
    return ev


def draw_grouped(ax, rec, r, palette):
    import seaborn as sns
    st = rec["ground_truth"]["statistics"]
    d = rec["ground_truth"]["data_table"]
    g = np.asarray(d["group"])
    v = np.asarray(d["value"], float)
    order = sorted(set(g.tolist()))
    ct = rec["chart_type"]
    ev = []

    if ct == "grouped_violin":
        sns.violinplot(x=g, y=v, order=order, hue=g, hue_order=order, legend=False,
                       palette=palette, ax=ax)
    elif ct == "grouped_bar":
        sns.barplot(x=g, y=v, order=order, hue=g, hue_order=order, legend=False,
                    palette=palette, errorbar="sd", ax=ax)
    elif ct == "pointplot":
        sns.pointplot(x=g, y=v, order=order, color="tab:blue", errorbar="sd", ax=ax)
    elif ct == "grouped_strip":
        sns.stripplot(x=g, y=v, order=order, hue=g, hue_order=order, legend=False,
                      palette=palette, alpha=0.6, ax=ax)
    else:
        sns.boxplot(x=g, y=v, order=order, hue=g, hue_order=order, legend=False,
                    palette=palette, ax=ax)

    meds = st.get("group_medians") or {}
    for i, lv in enumerate(order):
        sub = v[g == lv]
        if len(sub):
            ev.append(("data", i - 0.4, np.percentile(sub, 25), i + 0.4, np.percentile(sub, 75),
                       f"group:{lv}"))
    hi, lo = st.get("highest_group"), st.get("lowest_group")
    for nm, tag in ((hi, "highest_median_group"), (lo, "lowest_median_group")):
        if nm in order:
            i = order.index(nm)
            ev.append(("data", i - 0.45, v.min(), i + 0.45, v.max(), tag))

    ax.set_xlabel(rec["axis_labels"]["x"])
    ax.set_ylabel(rec["axis_labels"]["y"])
    if max(len(str(o)) for o in order) > 8:
        ax.tick_params(axis="x", rotation=20)
    return ev


def draw_composition(ax, rec, r, palette):
    st = rec["ground_truth"]["statistics"]
    labs = np.asarray(rec["ground_truth"]["data_table"]["category"])
    counts = st.get("counts") or {}
    cats = list(counts)
    vals = [counts[c] for c in cats]
    ct = rec["chart_type"]
    ev = []
    cols = plt.get_cmap(palette if palette in plt.colormaps() else "tab10")(
        np.linspace(0, 0.9, len(cats)))

    if ct == "pie":
        wedges, *_ = ax.pie(vals, labels=cats, autopct="%1.1f%%", colors=cols,
                            startangle=int(r.integers(0, 90)))
        ax.set_ylabel("")
        return [("artist", wedges[0], "dominant_wedge")] + \
               [("artist", w, f"wedge:{c}") for w, c in list(zip(wedges, cats))[1:4]]
    if ct == "barh":
        bars = ax.barh(cats, vals, color=cols)
        ax.set_xlabel("Count")
        ax.set_ylabel(rec["axis_labels"]["x"])
    else:
        bars = ax.bar(cats, vals, color=cols)
        ax.set_ylabel("Count")
        ax.set_xlabel(rec["axis_labels"]["x"])
        if max(len(str(c)) for c in cats) > 8:
            ax.tick_params(axis="x", rotation=20)

    out = [("artist", bars[0], "dominant_category_bar")]
    out += [("artist", b, f"bar:{c}") for b, c in list(zip(bars, cats))[1:5]]
    return out


DRAW = {"univariate": draw_univariate, "bivariate": draw_bivariate,
        "timeseries": draw_timeseries, "grouped": draw_grouped,
        "composition": draw_composition}


def render_one(rec):
    sid = rec["sample_id"]
    r = np.random.default_rng(_seed(sid))
    style, palette = STYLES[_seed(sid) % len(STYLES)]
    out = IMG / rec["label"] / f"{sid}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    rr = np.random.default_rng(_seed(sid + "|rand")) if RANDSTYLE else None
    rc = _rand_rc(rr) if RANDSTYLE else {}
    if RANDSTYLE:
        palette = str(rr.choice(RAND_CMAPS))

    try:
        with plt.style.context(style), plt.rc_context(rc):
            fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
            if rec["chart_type"] == "jointplot":
                # built by hand rather than seaborn.jointplot, which owns its own
                # figure and would break the fixed canvas the boxes depend on
                gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)
                ax = fig.add_subplot(gs[1:, :3])
                axt = fig.add_subplot(gs[0, :3], sharex=ax)
                axr = fig.add_subplot(gs[1:, 3], sharey=ax)
                d = rec["ground_truth"]["data_table"]
                axt.hist(np.asarray(d["x"], float), bins=28, color="tab:blue", alpha=0.7)
                axr.hist(np.asarray(d["y"], float), bins=28, orientation="horizontal",
                         color="tab:blue", alpha=0.7)
                axt.axis("off")
                axr.axis("off")
            else:
                ax = fig.add_subplot(111)

            cmap = plt.get_cmap(palette if palette in plt.colormaps() else "tab10")
            colour = cmap(int(r.integers(0, 8)) % cmap.N)
            if RANDSTYLE:
                import colorsys
                colour = colorsys.hsv_to_rgb(rr.uniform(0, 1), rr.uniform(0.35, 1.0), rr.uniform(0.3, 0.85))
            fn = DRAW[rec["arity"]]
            reqs = fn(ax, rec, r, palette if rec["arity"] in ("grouped", "composition") else colour)

            title = rec["axis_labels"].get("y") or rec["axis_labels"].get("x")
            if rec["arity"] == "univariate":
                title = f"Distribution of {rec['axis_labels']['x']}"
            elif rec["arity"] == "bivariate":
                title = f"{rec['axis_labels']['y']} vs {rec['axis_labels']['x']}"
            elif rec["arity"] == "timeseries":
                title = f"{rec['axis_labels']['y']} over {rec['axis_labels']['x']}"
            elif rec["arity"] == "grouped":
                title = f"{rec['axis_labels']['y']} by {rec['axis_labels']['x']}"
            else:
                title = f"Composition of {rec['axis_labels']['x']}"
            if not RANDSTYLE:
                ax.set_title(title[:78], fontsize=11, fontweight="bold")
            elif rr.random() > 0.3:
                ax.set_title(title[:78], fontsize=float(rr.uniform(8, 14)),
                             fontweight=str(rr.choice(["bold", "normal"])), loc=str(rr.choice(["center", "left"])))

            fig.subplots_adjust(left=0.13, right=0.965, top=0.90, bottom=0.13)
            fig.canvas.draw()

            boxes = [{"label": "plot_area",
                      "bbox": [round(float(ax.bbox.x0), 1), round(float(H_PX - ax.bbox.y1), 1),
                               round(float(ax.bbox.x1), 1), round(float(H_PX - ax.bbox.y0), 1)]}]
            for req in reqs or []:
                if req[0] == "data":
                    _, x0, y0, x1, y1, lab = req
                    try:
                        boxes.append(_box_from_data(fig, ax, x0, y0, x1, y1, lab))
                    except Exception:
                        pass
                else:
                    b = _box_from_artist(fig, req[1], req[2])
                    if b:
                        boxes.append(b)

            fig.savefig(out, dpi=DPI, facecolor="white")
            plt.close(fig)
    except Exception as e:
        plt.close("all")
        return sid, None, f"{type(e).__name__}: {e}"

    return sid, {"image_size": [W_PX, H_PX], "style": style,
                 "grounding_boxes": boxes[:14]}, None


def _work(chunk):
    return [render_one(r) for r in chunk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="no label-dependent marks")
    ap.add_argument("--randstyle", action="store_true", help="training-only style randomisation")
    ap.add_argument("--split", help="render only this split")
    ap.add_argument("--arity", help="render only this arity (e.g. timeseries)")
    ap.add_argument("--records", default=str(CACHE / "records.jsonl"),
                    help="records to render (default cache/records.jsonl)")
    ap.add_argument("--out", help="image dir relative to dataset/ (default images); "
                                  "records then go to <out>/records.jsonl")
    a = ap.parse_args()
    global IMG, CLEAN
    if a.out:
        os.environ["RENDER_OUT"] = str(DATA / a.out)
        IMG = DATA / a.out
    if a.clean:
        os.environ["RENDER_CLEAN"] = "1"
        CLEAN = True
    if a.randstyle:
        os.environ["RENDER_RANDSTYLE"] = "1"

    recs = [json.loads(l) for l in open(a.records)]
    recs = [r for r in recs if (not a.split or r["split"] == a.split)
            and (not a.arity or r["arity"] == a.arity)]
    print(f"rendering {len(recs):,} charts at {W_PX}x{H_PX} "
          f"({len(set(r['chart_type'] for r in recs))} chart types) clean={CLEAN} -> {IMG}")
    IMG.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    chunks = [recs[i:i + 40] for i in range(0, len(recs), 40)]
    got, errs = {}, Counter()
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        futs = [ex.submit(_work, c) for c in chunks]
        bar = tqdm(as_completed(futs), total=len(futs), unit="batch",
                   desc="rendering charts", ncols=100)
        done = 0
        for fu in bar:
            for sid, meta, err in fu.result():
                done += 1
                if meta:
                    got[sid] = meta
                else:
                    errs[err.split(":")[0]] += 1
            bar.set_postfix(ok=len(got), failed=sum(errs.values()))

    out = IMG / "records.jsonl" if a.out else DATA / "records.jsonl"
    kept = 0
    with open(out, "w") as fh:
        for r in recs:
            m = got.get(r["sample_id"])
            if not m:
                continue
            r["render"] = m
            fh.write(json.dumps(r) + "\n")
            kept += 1

    print(f"\nrendered {kept:,}/{len(recs):,} charts -> {IMG}")
    if errs:
        print("failures by type:")
        for k, v in errs.most_common(8):
            print(f"  {k}: {v}")
    print(f"records with render metadata -> {out}")


if __name__ == "__main__":
    main()
