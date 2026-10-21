"""
ablate_marks.py — Do the models read the data, or the label-dependent annotations?

    python build/06_render.py --split test --arity timeseries --out ablate_marked
    python build/06_render.py --split test --arity timeseries --clean --out ablate_clean
    python train/ablate_marks.py

Same test time series, same style seed, rendered twice: once with the original
label-dependent marks (trend line / changepoint line / anomaly circles) and once
without. Only the marks differ, so any accuracy gap is reliance on them.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble import DATA, MEMBERS, RES, probs_for  # noqa: E402

MODELS = ("resnet101_nb1", "resnet50_sweep")


def main():
    meta = pd.read_parquet(DATA / "tables" / "metadata.parquet")
    te = meta[(meta.split == "test") & (meta.arity == "timeseries")].reset_index(drop=True)
    ci = json.loads((DATA / "class_index.json").read_text())["label_to_id"]
    classes = sorted(ci, key=ci.get)
    y = te.label.map(ci).values
    real = (te.provenance == "real").values
    rows = []
    for name, ck, arch, crop, size in MEMBERS:
        if name not in MODELS:
            continue
        for view in ("ablate_marked", "ablate_clean"):
            df = te.assign(image=[f"{view}/{l}/{s}.png" for l, s in zip(te.label, te.sample_id)])
            ok = probs_for(ck, arch, False, df, classes, {}, size).argmax(1) == y
            r = {"model": name, "marks": view == "ablate_marked", "all": ok.mean(),
                 "real": ok[real].mean(), "synthetic": ok[~real].mean()}
            r.update({c: ok[te.label.values == c].mean() for c in sorted(te.label.unique())})
            rows.append(r)
            print(pd.DataFrame(rows).round(3).to_string(), flush=True)
    pd.DataFrame(rows).to_csv(RES / "ablate_marks.csv", index=False)


if __name__ == "__main__":
    main()
