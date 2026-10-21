"""
12_extra_real.py — Supplementary real training pool from candidates the balanced set left out.

    python build/12_extra_real.py
    python build/06_render.py --clean --records cache/records_extra.jsonl --out images_extra

05_assemble caps real data at 500 per class so every class stays balanced, which
leaves verified real candidates unused. This keeps the ones whose SOURCE TABLE is
assigned to train by the same assign_split() hash, so no val or test source can
enter (source-disjointness is exactly preserved), drops exact duplicates of any
dataset sample, and caps each class so a few abundant classes don't swamp training.

Output: cache/records_extra.jsonl (split "train_extra", same schema as records.jsonl)
"""
import importlib
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
A = importlib.import_module("05_assemble")
import taxonomy as tax  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE, DATA = ROOT / "cache", ROOT / "dataset"
CAP = 400


def main():
    used = {json.loads(l)["fingerprint"] for l in open(DATA / "records.jsonl")}
    pools, seen = defaultdict(list), set()
    for l in open(CACHE / "candidates_real.jsonl"):
        c = json.loads(l)
        fp = A.data_fingerprint(c)
        if fp in used or fp in seen or A.assign_split(c["source_id"]) != "train":
            continue
        seen.add(fp)
        c["fingerprint"] = fp
        pools[c["label"]].append(c)

    rng = np.random.default_rng(A.SEED)
    out = []
    for cls in tax.CLASS_NAMES:
        chosen = pools.get(cls, [])
        rng.shuffle(chosen)
        chosen = chosen[:CAP]
        if not chosen:
            continue
        for c in chosen:
            c["split"] = "train_extra"
        diffs = A.assign_difficulty(chosen, cls)
        charts = tax.CLASSES[cls]["charts"]
        for i, c in enumerate(chosen):
            st = c["stats"]
            sid = f"{cls}__x{i:04d}"
            steps, insight = A.reasoning_chain(cls, st, c["columns"], c["axis_labels"])
            out.append({
                "sample_id": sid, "image": f"images_extra/{cls}/{sid}.png", "split": "train_extra",
                "label": cls, "family": tax.CLASSES[cls]["family"], "arity": c["arity"],
                "chart_type": charts[zlib.crc32(sid.encode()) % len(charts)],
                "difficulty": diffs[i], "evidence_margin": round(A.evidence_margin(cls, st), 6),
                "provenance": "real", "domain": c.get("domain"), "source_id": c["source_id"],
                "source": c["source"], "source_url": c.get("source_url"), "license": c.get("license"),
                "table": c.get("table"), "columns": c["columns"], "axis_labels": c["axis_labels"],
                "fingerprint": c["fingerprint"],
                "ground_truth": {"class": cls, "class_description": tax.CLASSES[cls]["desc"],
                                 "verification_test": tax.CLASSES[cls]["test"], "statistics": st,
                                 "data_table": c["data"], "n_points": st.get("n"),
                                 "reasoning_chain": steps, "insight": insight},
                "qa": A.build_qa(cls, st, c["columns"], c["axis_labels"], insight),
            })
        print(f"  {cls:28s} {len(chosen):>4d}")

    # the guarantee this file exists to keep
    test_val_sources = {json.loads(l)["source_id"] for l in open(DATA / "records.jsonl")
                        if json.loads(l)["split"] in ("val", "test")}
    assert not test_val_sources & {r["source_id"] for r in out}, "extra pool shares a val/test source"
    assert not used & {r["fingerprint"] for r in out}, "extra pool duplicates a dataset sample"
    with open(CACHE / "records_extra.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(out):,} extra real training records -> {CACHE / 'records_extra.jsonl'}")


if __name__ == "__main__":
    main()
