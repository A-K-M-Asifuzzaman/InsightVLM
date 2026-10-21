"""
cache_probs.py — Softmax outputs of every checkpoint, saved once.

    python train/cache_probs.py [member ...]            # val + test (matplotlib render)
    python train/cache_probs.py --view plotly [member ...]   # adds test_plotly (real test, Plotly render)
    python train/cache_probs.py --view vega [member ...]     # adds test_vega (held-out renderer)

Writes results/probs/<member>.npz (rows aligned to metadata order; test_<view> rows
aligned to results/probs/<view>_ids.csv). Everything downstream (ensemble weighting,
prior correction, stats) reads these, so no model is run twice. Tuning uses val
only; test is scored once at the end.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble import DATA, MEMBERS, RES, probs_for  # noqa: E402

OUT = RES / "probs"


def load_split(split):
    meta = pd.read_parquet(DATA / "tables" / "metadata.parquet")
    df = meta[meta.split == split].reset_index(drop=True)
    boxes = {}
    for sid, g in zip(df.sample_id, df.grounding_boxes_json):
        b = next((x["bbox"] for x in json.loads(g) if x["label"] == "plot_area"), None)
        if b:
            boxes[sid] = b
    return df, boxes


def view_split(view):
    """Real test charts as drawn by another renderer (dataset/images_<view>/)."""
    te, _ = load_split("test")
    te = te[te.provenance == "real"]
    te = te.assign(image=[f"images_{view}/{l}/{s}.png" for l, s in zip(te.label, te.sample_id)])
    te = te[[(DATA / p).exists() for p in te.image]].reset_index(drop=True)
    te[["sample_id"]].to_csv(OUT / f"{view}_ids.csv", index=False)
    return te, {}   # no plot-area boxes for other renderers


def main():
    OUT.mkdir(exist_ok=True)
    argv = sys.argv[1:]
    view = argv[argv.index("--view") + 1] if "--view" in argv else None
    args = [a for a in argv if a not in ("--view", view)]
    ci = json.loads((DATA / "class_index.json").read_text())["label_to_id"]
    classes = sorted(ci, key=ci.get)
    splits = {f"test_{view}": view_split(view)} if view else {s: load_split(s) for s in ("val", "test")}
    for name, ck, arch, crop, size in MEMBERS:
        if (args and name not in args) or not ck.exists() or (view and crop):
            continue
        t0 = time.time()
        f = OUT / f"{name}.npz"
        arrs = dict(np.load(f)) if f.exists() else {}
        arrs.update({s: probs_for(ck, arch, crop, df, classes, bx, size) for s, (df, bx) in splits.items()})
        np.savez(f, **arrs)
        print(f"{name:20s} {sorted(splits)} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
