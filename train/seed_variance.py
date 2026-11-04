"""
seed_variance.py — Training variance of the deployed recipe across 3 seeds.

    python train/cache_probs.py --view plotly <seed members>; ... --view vega <seed members>
    python train/seed_variance.py           # -> results/seed_variance.json

Seed 1 is the deployed pair (r101_384_real50_xp + d121_384_real50_xp); seeds 2 and 3
retrain both with only the global seed changed. For each seed the two members are
ensembled exactly as deployed (equal weights). Reported: each seed's accuracy and the
mean and standard deviation across seeds, per renderer.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble import DATA, RES  # noqa: E402

SEEDS = {1: ("r101_384_real50_xp", "d121_384_real50_xp"),
         2: ("r101_384_real50_xp_s2", "d121_384_real50_xp_s2"),
         3: ("r101_384_real50_xp_s3", "d121_384_real50_xp_s3")}


def main():
    cfg = json.loads((RES / "final_ensemble.json").read_text())
    ci = {c: i for i, c in enumerate(cfg["classes"])}
    meta = pd.read_parquet(DATA / "tables" / "metadata.parquet")
    te = meta[meta.split == "test"].reset_index(drop=True)
    y, real = te.label.map(ci).values, (te.provenance == "real").values
    lab = te.set_index("sample_id").label.map(ci)
    views = {v: lab.loc[pd.read_csv(RES / "probs" / f"{v}_ids.csv").sample_id].values for v in ("plotly", "vega")}

    def score(names):
        z = [np.load(RES / "probs" / f"{n}.npz") for n in names]
        p = sum(x["test"] for x in z) / len(z)
        r = {"real_matplotlib": float((p.argmax(1) == y)[real].mean()),
             "synthetic": float((p.argmax(1) == y)[~real].mean())}
        for v, yv in views.items():
            if all(f"test_{v}" in x.files for x in z):
                r[f"real_{v}"] = float((sum(x[f"test_{v}"] for x in z).argmax(1) == yv).mean())
        return r

    out = {"per_seed": {}, "members": {}}
    for s, pair in SEEDS.items():
        if all((RES / "probs" / f"{n}.npz").exists() for n in pair):
            out["per_seed"][s] = score(pair)
            for n in pair:
                out["members"][n] = score([n])
    keys = sorted({k for r in out["per_seed"].values() for k in r})
    out["ensemble_mean_sd"] = {k: [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None]
                               for k in keys for v in [[r[k] for r in out["per_seed"].values() if k in r]]}
    out["n_seeds"] = len(out["per_seed"])
    (RES / "seed_variance.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["ensemble_mean_sd"], indent=1))


if __name__ == "__main__":
    main()
