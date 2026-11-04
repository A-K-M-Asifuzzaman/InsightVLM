"""
human_study_eval.py — Score the two human studies.

    python train/human_study_eval.py <export_dir>

<export_dir> holds the annotation pages' exported documents
(labels/<annotator>/charts/*.json for the baseline study, labels/<annotator>/items/*.json
for the label check). Answer keys stay in results/human_study/ and are never published.

Exclusion rule, fixed before any data was seen: an annotator is excluded if they accept
more than 40% of the label-check decoys, or if their median time per chart is under 1.5 s.

Reports
  label check   share of real labels a person confirms ("yes"), rejects, or cannot tell;
                decoy rejection rate; Fleiss' kappa across annotators
  baseline      human accuracy on 200 real test charts, and the model on the same charts
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HS = ROOT / "results" / "human_study"
MAX_DECOY_ACCEPT, MIN_MEDIAN_MS = 0.40, 1500


def read(export, sub):
    rows = []
    for f in Path(export).glob(f"labels/*/{sub}/*.json"):
        d = json.loads(f.read_text())
        rows.append({"annotator": f.parts[-3], "item": f.stem, **d.get("data", d)})
    return pd.DataFrame(rows)


def fleiss(table):
    """Fleiss' kappa for an items x categories count table (equal raters per item)."""
    n = table.sum(1)[0]
    p = table.sum(0) / table.sum()
    P = ((table ** 2).sum(1) - n) / (n * (n - 1))
    return float((P.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum()))


def main(export):
    out = {}
    lc = read(export, "items")
    if len(lc):
        key = pd.read_csv(HS / "label_check_key.csv")
        lc = lc.merge(key, on="item")
        per = lc.groupby("annotator").apply(lambda g: pd.Series({
            "decoy_accept": (g[g.kind == "decoy"].answer == "yes").mean(),
            "median_ms": g.ms.median(), "n": len(g)}))
        keep = per[(per.decoy_accept <= MAX_DECOY_ACCEPT) & (per.median_ms >= MIN_MEDIAN_MS)].index
        v = lc[lc.annotator.isin(keep) & (lc.kind == "label")]
        out["label_check"] = {
            "annotators": int(len(per)), "kept": int(len(keep)), "excluded": per.drop(keep).round(3).to_dict("index"),
            "confirmed": float((v.answer == "yes").mean()) if len(v) else None,
            "rejected": float((v.answer == "no").mean()) if len(v) else None,
            "cannot_tell": float((v.answer == "unsure").mean()) if len(v) else None,
            "decoys_rejected": float((lc[lc.annotator.isin(keep) & (lc.kind == "decoy")].answer == "no").mean()) if len(keep) else None,
            "by_class_confirmed": v.groupby("true_label").answer.apply(lambda a: round(float((a == "yes").mean()), 3)).to_dict()}
        full = v.groupby("item").filter(lambda g: g.annotator.nunique() == len(keep))
        if len(keep) >= 2 and len(full):
            t = pd.crosstab(full.item, full.answer).reindex(columns=["yes", "no", "unsure"], fill_value=0).values
            out["label_check"]["fleiss_kappa"] = fleiss(t)
    bl = read(export, "charts")
    if len(bl):
        key = pd.read_csv(HS / "key.csv").rename(columns={"chart": "item"})
        bl = bl.merge(key, on="item", suffixes=("", "_true"))
        fast = bl.groupby("annotator").ms.median() < MIN_MEDIAN_MS
        bl = bl[~bl.annotator.isin(fast[fast].index)]
        bl["ok"] = bl.label == bl.label_true
        acc = bl.groupby("annotator").ok.mean()
        out["baseline"] = {"annotators": acc.round(4).to_dict(), "excluded_too_fast": fast[fast].index.tolist(),
                           "mean": float(acc.mean()) if len(acc) else None}
    print(json.dumps(out, indent=2))
    (HS / "human_study_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HS / "export")
