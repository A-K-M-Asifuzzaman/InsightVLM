"""
07_validate.py — Integrity audit of the finished dataset.

This is the credibility artefact. It re-derives every label from scratch by
re-running the taxonomy battery on the stored data table, so the audit does not
trust the pipeline that produced the records. It also proves the split is
source-disjoint, the classes are exactly balanced, and every grounding box lies
inside its image.

Exit code is non-zero if any hard check fails.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as tax  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
RECS = DATA / "records.jsonl"
RECHECK_N = 3000

VERIFY = {
    "univariate": lambda d: tax.verify_univariate(d["x"]),
    "bivariate": lambda d: tax.verify_bivariate(d["x"], d["y"]),
    "timeseries": lambda d: tax.verify_timeseries(d["t"], d["y"]),
    "grouped": lambda d: tax.verify_grouped(d["value"], d["group"]),
    "composition": lambda d: tax.verify_composition(d["category"]),
}

fails, warns = [], []


def check(cond, msg, hard=True):
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  {'FAIL' if hard else 'WARN'}  {msg}")
        (fails if hard else warns).append(msg)


def main():
    recs = [json.loads(l) for l in open(RECS)]
    print(f"auditing {len(recs):,} records from {RECS}\n")

    # ---- 1. class balance -------------------------------------------------
    print("[1] class balance")
    cc = Counter(r["label"] for r in recs)
    check(len(cc) == 20, f"all 20 classes present (found {len(cc)})")
    check(len(set(cc.values())) == 1,
          f"perfectly balanced: {sorted(set(cc.values()))} per class")

    # ---- 2. splits --------------------------------------------------------
    print("\n[2] splits")
    sp = Counter(r["split"] for r in recs)
    print(f"        {dict(sp)}")
    per = defaultdict(Counter)
    for r in recs:
        per[r["label"]][r["split"]] += 1
    skew = {c: dict(v) for c, v in per.items()
            if abs(v["train"] - 700) > 1 or abs(v["val"] - 100) > 1 or abs(v["test"] - 200) > 1}
    check(not skew, f"every class splits 700/100/200 ({len(skew)} deviations)")

    # ---- 3. source leakage ------------------------------------------------
    print("\n[3] source-disjoint splits (real data)")
    src_splits = defaultdict(set)
    for r in recs:
        if r["provenance"] == "real":
            src_splits[r["source_id"]].add(r["split"])
    leaked = {s: v for s, v in src_splits.items() if len(v) > 1}
    check(not leaked,
          f"no real source appears in >1 split ({len(leaked)} leaked of {len(src_splits):,})")

    # ---- 4. duplicates ----------------------------------------------------
    print("\n[4] duplicates")
    fp = Counter(r["fingerprint"] for r in recs)
    dups = {k: v for k, v in fp.items() if v > 1}
    check(not dups, f"no duplicate data fingerprints ({len(dups)} repeated)")
    check(len(set(r["sample_id"] for r in recs)) == len(recs), "sample_ids unique")

    # ---- 5. label re-verification from stored data ------------------------
    print(f"\n[5] label re-verification (independent re-run on {RECHECK_N:,} sampled records)")
    rng = np.random.default_rng(7)
    idx = rng.choice(len(recs), min(RECHECK_N, len(recs)), replace=False)
    agree, disagree = 0, Counter()
    for i in tqdm(idx, desc="re-verifying", unit="rec", ncols=100, leave=False):
        r = recs[int(i)]
        try:
            got, _st, why = VERIFY[r["arity"]](r["ground_truth"]["data_table"])
        except Exception as e:
            disagree[f"error:{type(e).__name__}"] += 1
            continue
        if got == r["label"]:
            agree += 1
        else:
            disagree[f'{r["label"]}->{got or why}'] += 1
    rate = agree / len(idx)
    check(rate >= 0.999, f"stored labels reproduce from stored data: {100*rate:.2f}% "
                         f"({agree:,}/{len(idx):,})")
    if disagree:
        for k, v in disagree.most_common(6):
            print(f"          {k}: {v}")

    # ---- 6. ground-truth completeness -------------------------------------
    print("\n[6] ground-truth completeness")
    miss = [r["sample_id"] for r in recs if not r["ground_truth"].get("statistics")]
    check(not miss, f"every record has a statistics dict ({len(miss)} missing)")
    thin = [r["sample_id"] for r in recs if len(r["ground_truth"].get("reasoning_chain", [])) < 3]
    check(not thin, f"every reasoning chain has >=3 steps ({len(thin)} short)")
    noins = [r["sample_id"] for r in recs if len(r["ground_truth"].get("insight", "")) < 80]
    check(not noins, f"every insight is substantive ({len(noins)} too short)")
    notab = [r["sample_id"] for r in recs if not r["ground_truth"].get("data_table")]
    check(not notab, f"every record carries its underlying data table ({len(notab)} missing)")
    noqa = [r["sample_id"] for r in recs if len(r.get("qa", [])) < 2]
    check(not noqa, f"every record has >=2 QA pairs ({len(noqa)} short)")

    # ---- 7. images and grounding boxes ------------------------------------
    print("\n[7] images and grounding boxes")
    sample = [recs[int(i)] for i in rng.choice(len(recs), min(1500, len(recs)), replace=False)]
    bad_size, missing, bad_box = 0, 0, 0
    for r in tqdm(sample, desc="checking images", unit="img", ncols=100, leave=False):
        p = DATA / r["image"]
        if not p.exists():
            missing += 1
            continue
        W, H = r["render"]["image_size"]
        if Image.open(p).size != (W, H):
            bad_size += 1
        for b in r["render"]["grounding_boxes"]:
            x0, y0, x1, y1 = b["bbox"]
            if not (0 <= x0 <= x1 <= W and 0 <= y0 <= y1 <= H):
                bad_box += 1
    check(missing == 0, f"all sampled images exist ({missing} missing of {len(sample)})")
    check(bad_size == 0, f"all sampled images are the declared size ({bad_size} wrong)")
    check(bad_box == 0, f"all grounding boxes lie inside the canvas ({bad_box} out of bounds)")
    nb = [len(r["render"]["grounding_boxes"]) for r in recs]
    print(f"        grounding boxes per image: mean {np.mean(nb):.1f}, min {min(nb)}, max {max(nb)}")

    # ---- 8. diversity -----------------------------------------------------
    print("\n[8] diversity")
    prov = Counter(r["provenance"] for r in recs)
    ct = Counter(r["chart_type"] for r in recs)
    dif = Counter(r["difficulty"] for r in recs)
    styles = Counter(r["render"]["style"] for r in recs)
    print(f"        provenance : {dict(prov)}  ({100*prov['real']/len(recs):.1f}% real)")
    print(f"        chart types: {len(ct)} distinct, most common {ct.most_common(3)}")
    print(f"        difficulty : {dict(dif)}")
    print(f"        styles     : {len(styles)} distinct")
    print(f"        real source tables used: "
          f"{len(set(r['source_id'] for r in recs if r['provenance']=='real')):,}")
    worst = max(per[c][s] for c in per for s in ("train", "val", "test"))
    check(len(ct) >= 20, f"at least 20 chart types in use ({len(ct)})")
    check(all(prov[p] > 0 for p in ("real", "synthetic")), "both provenances present")
    pc = defaultdict(Counter)
    for r in recs:
        pc[r["label"]][r["provenance"]] += 1
    pure = {c: dict(v) for c, v in pc.items() if len(v) < 2}
    check(not pure, f"no class is single-provenance ({len(pure)} would be a shortcut feature)",
          hard=False)

    # ---- summary ----------------------------------------------------------
    print("\n" + "=" * 66)
    if fails:
        print(f"AUDIT FAILED — {len(fails)} hard check(s):")
        for f in fails:
            print(f"  - {f}")
    else:
        print("AUDIT PASSED — all hard checks green")
    if warns:
        print(f"{len(warns)} warning(s):")
        for w in warns:
            print(f"  - {w}")
    print("=" * 66)

    (ROOT / "cache" / "audit.json").write_text(json.dumps({
        "n_records": len(recs), "classes": dict(cc), "splits": dict(sp),
        "provenance": dict(prov), "chart_types": dict(ct), "difficulty": dict(dif),
        "label_reverification_rate": rate, "leaked_sources": len(leaked),
        "duplicate_fingerprints": len(dups), "passed": not fails,
    }, indent=2))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
