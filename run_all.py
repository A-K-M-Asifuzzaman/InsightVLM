#!/usr/bin/env python3
"""
run_all.py — Build InsightVLM-20K end to end, with live progress in the terminal.

    python run_all.py                # full build
    python run_all.py --from 3       # resume from stage 3 (skip the network harvest)
    python run_all.py --only 6       # re-run one stage
    python run_all.py --list         # show stages

Each stage streams its own tqdm bar. An overall banner shows where you are.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"

STAGES = [
    (1, "01_harvest.py",   "Harvest real public data",        "network, ~4 min"),
    (2, "taxonomy.py",     "Taxonomy self-test",              "~3 min"),
    (3, "03_mine_real.py", "Mine real tables -> candidates",  "~10 min"),
    (4, "04_synth.py",     "Controlled synthesis",            "~8 min"),
    (5, "05_assemble.py",  "Balance, split, ground truth",    "~2 min"),
    (6, "06_render.py",    "Render charts + grounding boxes", "~12 min"),
    (7, "07_validate.py",  "Integrity audit",                 "~5 min"),
    (8, "08_export.py",    "Export splits + datasheet",       "~1 min"),
]

C = {"b": "\033[1m", "d": "\033[2m", "g": "\033[32m", "r": "\033[31m",
     "y": "\033[33m", "c": "\033[36m", "x": "\033[0m"}


def banner(i, total, name, desc, est):
    bar = "#" * i + "." * (total - i)
    print(f"\n{C['c']}{'=' * 78}{C['x']}")
    print(f"{C['b']}[{bar}] stage {i}/{total} — {desc}{C['x']}  {C['d']}({name}, {est}){C['x']}")
    print(f"{C['c']}{'=' * 78}{C['x']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--only", type=int)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for n, f, d, e in STAGES:
            print(f"  {n}  {d:34s} {C['d']}{f}  ({e}){C['x']}")
        return 0

    todo = [s for s in STAGES if (s[0] == a.only if a.only else s[0] >= a.start)]
    t0 = time.time()
    # Stamp the build start so watch_progress.py can tell this build's outputs
    # from a previous build's leftovers. Without it the watcher walks stale
    # artifacts in increasing-mtime order and reports COMPLETE immediately.
    (ROOT / "cache").mkdir(exist_ok=True)
    (ROOT / "cache" / "build_start.json").write_text(json.dumps(
        {"started": t0, "from_stage": todo[0][0] if todo else None,
         "stages": [s[0] for s in todo]}))
    print(f"{C['b']}InsightVLM-20K build{C['x']} — {len(todo)} stage(s)")

    for n, fname, desc, est in todo:
        banner(n, len(STAGES), fname, desc, est)
        st = time.time()
        rc = subprocess.call([sys.executable, str(BUILD / fname)], cwd=str(ROOT))
        el = time.time() - st
        if rc != 0:
            print(f"\n{C['r']}stage {n} ({fname}) failed with exit code {rc} "
                  f"after {el:.0f}s{C['x']}")
            return rc
        print(f"{C['g']}stage {n} done in {el:.0f}s{C['x']}", flush=True)

    total = time.time() - t0
    print(f"\n{C['c']}{'=' * 78}{C['x']}")
    print(f"{C['g']}{C['b']}BUILD COMPLETE in {total/60:.1f} min{C['x']}")
    print(f"{C['c']}{'=' * 78}{C['x']}")
    d = ROOT / "dataset"
    print(f"  images     {d / 'images'}")
    print(f"  records    {d / 'records.jsonl'}")
    print(f"  splits     {d / 'splits'}/(train|val|test).jsonl")
    print(f"  instruct   {d / 'instruct'}/(train|val|test).jsonl")
    print(f"  parquet    {d / 'tables' / 'metadata.parquet'}")
    print(f"  datasheet  {d / 'DATASHEET.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
