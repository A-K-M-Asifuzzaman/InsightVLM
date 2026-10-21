"""
stats_report.py — Headline table with uncertainty, on the REAL test charts only.

    python train/stats_report.py

Every system is scored on the same real test charts (matplotlib v2 render, and the
Plotly re-render where available). For each: accuracy with a bootstrap 95% CI,
macro-F1, and an exact McNemar test against the best system (paired, same charts).
Writes results/stats_report.json and results/stats_table.tex.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble import DATA, RES  # noqa: E402

B = 4000


def boot_ci(ok, seed=0):
    ok = np.asarray(ok, float)
    idx = np.random.default_rng(seed).integers(0, len(ok), (B, len(ok)))
    return [float(v) for v in np.percentile(ok[idx].mean(1), [2.5, 97.5])]


def macro_f1(y, p):
    fs = []
    for c in np.unique(y):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        fs.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(fs))


def mcnemar(a, b):
    """Exact McNemar: only discordant pairs carry information."""
    n01 = int(np.sum(a & ~b)); n10 = int(np.sum(~a & b))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, float(p)


def systems():
    """name -> DataFrame[sample_id, pred] over real test charts, per view."""
    meta = pd.read_parquet(DATA / "tables" / "metadata.parquet")
    te = meta[meta.split == "test"].reset_index(drop=True)
    ci = json.loads((DATA / "class_index.json").read_text())["label_to_id"]
    classes = sorted(ci, key=ci.get)
    out = {}
    fe = RES / "final_ensemble.json"
    if fe.exists():
        cfg = json.loads(fe.read_text())
        for view, key in (("matplotlib", "test"), ("plotly", "test_plotly"), ("vega", "test_vega")):
            parts = [(m["weight"], RES / "probs" / f"{m['name']}.npz") for m in cfg["members"]]
            if not all(f.exists() and key in np.load(f).files for _, f in parts):
                continue
            p = sum(w * np.load(f)[key] for w, f in parts) * np.exp(cfg["alpha"] * np.array(cfg["log_prior_shift"]))
            ids = te.sample_id if key == "test" else pd.read_csv(RES / "probs" / f"{view}_ids.csv").sample_id
            out[("InsightVLM ensemble (ours)", view)] = pd.DataFrame(
                {"sample_id": ids, "pred": [classes[i] for i in p.argmax(1)]})
    # v2_zs_*: NB6 answers on charts identical in v1/v2, NB6b answers on the clean time series
    for f in sorted((RES / "nb6").glob("v2_zs_*.csv")) if (RES / "nb6").exists() else []:
        d = pd.read_csv(f)
        out[(f.stem[6:] + " (zero-shot)", "matplotlib")] = d[["sample_id", "pred"]]
    hs = RES / "human_study" / "answers.csv"
    if hs.exists():
        d = pd.read_csv(hs)   # sample_id, annotator, pred
        for a, g in d.groupby("annotator"):
            out[(f"Human {a}", "matplotlib")] = g[["sample_id", "pred"]]
    return meta, out


def main():
    meta, sy = systems()
    truth = meta.set_index("sample_id").label
    real = set(meta[(meta.split == "test") & (meta.provenance == "real")].sample_id)
    rows = {}
    for (name, view), d in sy.items():
        d = d[d.sample_id.isin(real)].copy()
        d["ok"] = d.pred.values == truth.loc[d.sample_id].values
        rows[(name, view)] = d.set_index("sample_id")[["pred", "ok"]]
    report = []
    for view in ("matplotlib", "plotly", "vega"):
        cur = {n: s for (n, v), s in rows.items() if v == view}
        if not cur:
            continue
        best = max(cur, key=lambda n: cur[n].ok.mean())
        for n, d in sorted(cur.items(), key=lambda kv: -kv[1].ok.mean()):
            common = d.index.intersection(cur[best].index)
            n01, n10, p = (mcnemar(cur[best].ok.loc[common].values, d.ok.loc[common].values)
                           if n != best else (0, 0, 1.0))
            y = truth.loc[d.index].values
            report.append({"system": n, "view": view, "n": int(len(d)), "accuracy": float(d.ok.mean()),
                           "ci95": boot_ci(d.ok.values), "macro_f1": macro_f1(y, d.pred.fillna("none").values),
                           "vs_best": best, "best_only_right": n01, "this_only_right": n10,
                           "mcnemar_p": p})
    df = pd.DataFrame(report)
    print(df.drop(columns="vs_best").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    (RES / "stats_report.json").write_text(json.dumps(report, indent=2))
    tex = df.assign(CI=df.ci95.map(lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"),
                    p=df.mcnemar_p.map(lambda v: "--" if v == 1.0 else f"{v:.2g}")) \
        [["system", "view", "n", "accuracy", "CI", "macro_f1", "p"]] \
        .to_latex(index=False, float_format="%.3f",
                  caption="Accuracy on real test charts. 95\\% bootstrap CIs; $p$ is an exact "
                          "McNemar test against the best system on the same charts.",
                  label="tab:real")
    (RES / "stats_table.tex").write_text(tex)
    print(f"\nwrote {RES / 'stats_report.json'} and stats_table.tex")


if __name__ == "__main__":
    main()
