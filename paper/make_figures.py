"""
make_figures.py — Every figure in the Scientific Data manuscript, from saved results.

    python paper/make_figures.py        # -> paper/figures/*.pdf (+ .png previews)

Figures whose inputs don't exist yet are skipped with a note, so this can be
re-run as results arrive.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
from PIL import Image            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "dataset", ROOT / "results"
OUT = ROOT / "paper" / "figures"

# reference palette, light mode: categorical slots in fixed order + text inks
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})

NAMES = {"ts_increasing": "Increasing", "ts_decreasing": "Decreasing", "ts_seasonal": "Seasonal",
         "ts_changepoint": "Change-point", "ts_anomaly_spike": "Anomaly spike"}
SHORT = {"ts_increasing": "Increasing", "ts_decreasing": "Decreasing", "ts_seasonal": "Seasonal",
         "ts_changepoint": "Change-\npoint", "ts_anomaly_spike": "Anomaly\nspike"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=160)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def fig_marks_examples():
    """Same data, v1 render (label-dependent mark) vs v2 clean render."""
    sids = [("ts_changepoint", "ts_changepoint__0812"), ("ts_anomaly_spike", None)]
    meta = pd.read_parquet(DATA / "tables" / "metadata.parquet")
    if sids[1][1] is None:
        te = meta[(meta.split == "test") & (meta.label == "ts_anomaly_spike") & (meta.provenance == "real")]
        sids[1] = ("ts_anomaly_spike", te.sample_id.iloc[0])
    paths = [(DATA / "ablate_marked" / l / f"{s}.png", DATA / "ablate_clean" / l / f"{s}.png") for l, s in sids]
    if not all(a.exists() and b.exists() for a, b in paths):
        print("skip fig_marks_examples: run 06_render.py --out ablate_marked/ablate_clean first"); return
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    for r, ((l, _), (a, b)) in enumerate(zip(sids, paths)):
        for c, (p, t) in enumerate(((a, "v1: label-dependent mark"), (b, "v2: clean render"))):
            ax = axes[r, c]
            ax.imshow(Image.open(p)); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(GRID)
            ax.set_title(f"{NAMES[l]} · {t}", fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    save(fig, "fig_marks_examples")


def fig_marks_ablation():
    f = RES / "ablate_marks.csv"
    if not f.exists():
        print("skip fig_marks_ablation: run train/ablate_marks.py"); return
    d = pd.read_csv(f)
    d = d[d.model == "resnet101_nb1"]
    cls = list(NAMES)
    cols = ["real", "synthetic"] + cls
    marked, clean = d[d.marks].iloc[0], d[~d.marks].iloc[0]
    labels = ["All\nreal", "All\nsynthetic"] + [SHORT[c] for c in cls]
    x = np.arange(len(cols)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    b1 = ax.bar(x - w / 2 - 0.01, [marked[c] for c in cols], w, color=S1, label="With marks (v1 render)")
    b2 = ax.bar(x + w / 2 + 0.01, [clean[c] for c in cols], w, color=S2, label="Marks removed (v2 render)")
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color=INK2)
    ax.set_xticks(x, labels, rotation=0, fontsize=7.5)
    ax.set_ylim(0, 1.1); ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_ylabel("Accuracy")
    ax.axvline(1.5, color=GRID, lw=1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=7.5)
    ax.set_title("ResNet-101 trained on v1, time-series test charts", fontsize=8.5, color=INK, loc="left", pad=22)
    save(fig, "fig_marks_ablation")


def fig_results():
    f = RES / "stats_report.json"
    if not f.exists():
        print("skip fig_results: run train/stats_report.py"); return
    rep = pd.DataFrame(json.loads(f.read_text()))
    mp = rep[rep.view == "matplotlib"].sort_values("accuracy")
    ours = rep[rep.system.str.contains("ours")].set_index("view")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.7, 1]})
    names = [s.replace(" (zero-shot)", ", zero-shot").replace("-Instruct", "").replace("InsightVLM ensemble (ours)", "Ours (CNN ensemble)")
             for s in mp.system]
    cols = [S1 if "ours" in s else S2 if "Human" in s else INK2 for s in mp.system]
    lo = [a - c[0] for a, c in zip(mp.accuracy, mp.ci95)]
    hi = [c[1] - a for a, c in zip(mp.accuracy, mp.ci95)]
    y = np.arange(len(mp))
    a1.barh(y, mp.accuracy, color=cols, height=0.6, xerr=[lo, hi], error_kw={"ecolor": INK, "lw": 0.8, "capsize": 2})
    for yi, v, c in zip(y, mp.accuracy, mp.ci95):
        a1.text(c[1] + 0.02, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
    a1.axvline(0.288, color=INK2, ls="--", lw=0.8)
    a1.text(0.292, len(mp) - 0.45, "chart type alone", fontsize=6.5, color=INK2)
    a1.set_yticks(y, names, fontsize=7.5); a1.set_xlim(0, 1.05); a1.set_xlabel("Accuracy, real test charts")
    a1.grid(axis="y", visible=False)
    views = [v for v in ("matplotlib", "plotly", "vega") if v in ours.index]
    labels = {"matplotlib": "matplotlib\n(train)", "plotly": "Plotly\n(aug.)", "vega": "Vega-Lite\n(unseen)"}
    x = np.arange(len(views))
    a2.bar(x, [ours.loc[v, "accuracy"] for v in views], color=[S1, S3, S2][:len(views)], width=0.6,
           yerr=[[ours.loc[v, "accuracy"] - ours.loc[v, "ci95"][0] for v in views],
                 [ours.loc[v, "ci95"][1] - ours.loc[v, "accuracy"] for v in views]],
           error_kw={"ecolor": INK, "lw": 0.8, "capsize": 2})
    for xi, v in zip(x, views):
        a2.text(xi, ours.loc[v, "ci95"][1] + 0.02, f"{ours.loc[v, 'accuracy']:.3f}", ha="center", fontsize=7.5)
    a2.set_xticks(x, [labels[v] for v in views], fontsize=7); a2.set_ylim(0, 1.05)
    a2.set_ylabel("Accuracy, same real charts"); a2.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig_results")


SHORTNAME = {
    "dist_normal": "normal", "dist_right_skew": "right skew", "dist_left_skew": "left skew",
    "dist_bimodal": "bimodal", "dist_heavy_tail": "heavy tail", "dist_uniform": "uniform",
    "rel_strong_positive": "strong +", "rel_strong_negative": "strong −", "rel_none": "none",
    "rel_nonlinear_monotonic": "curved monotone", "rel_nonlinear_nonmonotonic": "U-shape",
    "rel_heteroscedastic": "fan shape", "ts_increasing": "increasing", "ts_decreasing": "decreasing",
    "ts_seasonal": "seasonal", "ts_changepoint": "level shift", "ts_anomaly_spike": "anomaly",
    "grp_significant_diff": "groups differ", "grp_no_diff": "no difference",
    "grp_dominant_category": "dominant cat."}


def _ensemble_real_preds():
    cfg = json.loads((RES / "final_ensemble.json").read_text())
    cl = cfg["classes"]
    meta = pd.read_parquet(DATA / "tables" / "metadata.parquet")
    te = meta[meta.split == "test"].reset_index(drop=True)
    p = sum(np.load(RES / "probs" / f"{m['name']}.npz")["test"] for m in cfg["members"])
    rm = (te.provenance == "real").values
    return cl, te.label.values[rm], np.array([cl[i] for i in p.argmax(1)])[rm]


def fig_confusion():
    if not (RES / "final_ensemble.json").exists():
        print("skip fig_confusion"); return
    cl, y, p = _ensemble_real_preds()
    cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(p, name="pred")).reindex(index=cl, columns=cl, fill_value=0)
    cmn = cm.div(cm.sum(1).replace(0, 1), axis=0).values
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cl)), [SHORTNAME[c] for c in cl], rotation=90, fontsize=7)
    ax.set_yticks(range(len(cl)), [f"{SHORTNAME[c]} ({cm.loc[c].sum()})" for c in cl], fontsize=7)
    ax.grid(False)
    for i in range(len(cl)):
        for j in range(len(cl)):
            if cmn[i, j] >= 0.05:
                ax.text(j, i, f"{cmn[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.5,
                        color="white" if cmn[i, j] > 0.55 else INK)
    for k in (6, 12, 17):   # family boundaries
        ax.axhline(k - 0.5, color=INK2, lw=0.6); ax.axvline(k - 0.5, color=INK2, lw=0.6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (number of real test charts)")
    fig.colorbar(im, shrink=0.7, label="Share of the true class")
    fig.tight_layout()
    save(fig, "fig_confusion")


def fig_per_class():
    if not (RES / "final_ensemble.json").exists():
        print("skip fig_per_class"); return
    cl, y, p = _ensemble_real_preds()
    d = pd.DataFrame({"y": y, "ok": y == p}).groupby("y").ok.agg(["mean", "count"]).reindex(cl)
    d = d.sort_values("mean")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    fam_col = {"dist": S1, "rel": S2, "ts": S3, "grp": "#eda100"}
    cols = [fam_col[c.split("_")[0]] for c in d.index]
    ax.barh(range(len(d)), d["mean"], color=cols, height=0.65)
    for i, (m, n) in enumerate(zip(d["mean"], d["count"])):
        ax.text(m + 0.01, i, f"{m:.2f}  (n={n})", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(d)), [SHORTNAME[c] for c in d.index], fontsize=7)
    ax.set_xlim(0, 1.15); ax.set_xlabel("Accuracy, real test charts"); ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v, label=k) for k, v in
                       zip(("Distribution", "Relationship", "Temporal", "Group"), fam_col.values())],
              loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_per_class")


if __name__ == "__main__":
    which = sys.argv[1:] or [n for n in dir() if n.startswith("fig_")]
    for n in which:
        globals()[n]()
