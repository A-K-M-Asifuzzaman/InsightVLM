#!/usr/bin/env python3
"""
watch_progress.py — Live build dashboard. Run in a second terminal:

    python watch_progress.py            # refresh every 2s
    python watch_progress.py -i 5       # every 5s
    python watch_progress.py --once     # print once and exit

Progress is derived from files on disk, not from the build process, so this
attaches to a build that is already running (or shows the state of a finished
one). No dependencies beyond the standard library.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
DATA = ROOT / "dataset"
RAW = ROOT / "raw_sources"

C = {"b": "\033[1m", "d": "\033[2m", "g": "\033[32m", "r": "\033[31m",
     "y": "\033[33m", "c": "\033[36m", "m": "\033[35m", "x": "\033[0m"}

N_CLASSES, PER_CLASS = 20, 1000
N_TARGET = N_CLASSES * PER_CLASS

# stage -> (label, output marker used to decide "done", rough weight in seconds)
STAGES = [
    ("Harvest real data",      RAW / "manifest.json",           240),
    ("Taxonomy self-test",     CACHE / "selftest.marker",       170),
    ("Mine real tables",       CACHE / "real_class_counts.json", 70),
    ("Controlled synthesis",   CACHE / "synth_class_counts.json", 160),
    ("Balance + ground truth", CACHE / "records.jsonl",         110),
    ("Render charts",          DATA / "records.jsonl",          700),
    ("Integrity audit",        CACHE / "audit.json",            260),
    ("Export + datasheet",     DATA / "DATASHEET.md",            40),
]


def mtime(p):
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def count_lines(p):
    """Fast newline count; returns None if unreadable."""
    try:
        n, buf = 0, bytearray(1 << 20)
        with open(p, "rb") as f:
            while True:
                k = f.readinto(buf)
                if not k:
                    break
                n += buf[:k].count(b"\n")
        return n
    except OSError:
        return None


def count_pngs(root):
    n = 0
    try:
        for d in os.scandir(root):
            if d.is_dir():
                with os.scandir(d.path) as it:
                    n += sum(1 for e in it if e.name.endswith(".png"))
    except OSError:
        return 0
    return n


def fmt_dur(s):
    if s is None or s < 0 or s != s:
        return "--:--"
    s = int(s)
    if s >= 3600:
        return f"{s//3600}h{(s%3600)//60:02d}m"
    return f"{s//60:02d}:{s%60:02d}"


def bar(frac, width, fill="#", empty="."):
    frac = max(0.0, min(1.0, frac))
    k = int(round(frac * width))
    return fill * k + empty * (width - k)


def build_start():
    """When the current build began, so leftovers from an earlier build are not
    mistaken for progress. Written by run_all.py."""
    try:
        return float(json.loads((CACHE / "build_start.json").read_text())["started"])
    except Exception:
        return None


def stage_states():
    """Ordered stage completion: a stage counts as done when its marker exists,
    is not older than the previous stage's marker, and postdates the build
    start (otherwise a stale artefact from a previous build reads as done)."""
    t_start = build_start()
    states, prev_t = [], (t_start - 1.0 if t_start else 0.0)
    for name, marker, weight in STAGES:
        t = mtime(marker)
        done = t is not None and t >= prev_t - 1.0
        if done:
            prev_t = max(prev_t, t)
        states.append({"name": name, "done": done, "t": t, "weight": weight})
    # A stage that never ran but has a completed successor was skipped
    # (e.g. `run_all.py --from 3`), not stalled -- otherwise the dashboard would
    # sit on it forever showing the build as stuck.
    last_done = max((i for i, s in enumerate(states) if s["done"]), default=-1)
    for i, s in enumerate(states):
        if not s["done"] and i < last_done:
            s["skipped"] = True

    active = next((i for i, s in enumerate(states)
                   if not s["done"] and not s.get("skipped")), None)
    return states, active


def active_detail(idx):
    """(done, total, unit) for the stage currently running, or None."""
    if idx == 2:
        n = count_lines(CACHE / "candidates_real.jsonl")
        return (n, None, "candidates") if n else None
    if idx == 3:
        n = count_lines(CACHE / "candidates_synth.jsonl")
        return (n, None, "candidates") if n else None
    if idx == 5:
        return (count_pngs(DATA / "images"), N_TARGET, "charts")
    if idx == 6:
        return (None, None, "re-verifying labels")
    return None


def render(hist, t0, once):
    states, active = stage_states()
    w = min(shutil.get_terminal_size((100, 30)).columns, 100)
    out = []

    done_w = sum(s["weight"] for s in states if s["done"] or s.get("skipped"))
    tot_w = sum(s["weight"] for s in states)
    det = active_detail(active) if active is not None else None
    if det and det[1]:
        f = (det[0] or 0) / det[1]
        done_w += states[active]["weight"] * f

    frac = done_w / tot_w
    elapsed = time.time() - t0

    out.append(f"{C['b']}{C['c']}InsightVLM-20K build{C['x']}   "
               f"{C['d']}{time.strftime('%H:%M:%S')}   watching {ROOT.name}/{C['x']}")
    out.append("")
    out.append(f"  {C['b']}[{bar(frac, w - 26)}] {100*frac:5.1f}%{C['x']}")

    eta = (tot_w - done_w) / (done_w / elapsed) if done_w > 0 and elapsed > 3 else None
    if active is None:
        out.append(f"  {C['g']}{C['b']}BUILD COMPLETE{C['x']}   elapsed {fmt_dur(elapsed)}")
    else:
        out.append(f"  elapsed {fmt_dur(elapsed)}   "
                   f"est. remaining {C['y']}{fmt_dur(eta)}{C['x']}   "
                   f"(watching for {fmt_dur(elapsed)})")
    out.append("")

    for i, s in enumerate(states):
        if s["done"]:
            icon, col, note = "DONE", C["g"], ""
        elif s.get("skipped"):
            icon, col, note = "SKIP", C["d"], "not run this build"
        elif i == active:
            icon, col = "RUN ", C["y"]
            note = ""
            if det:
                d, tot, unit = det
                if tot:
                    r = hist.get(i)
                    rate = None
                    if r and time.time() - r[0] > 2 and d > r[1]:
                        rate = (d - r[1]) / (time.time() - r[0])
                    hist.setdefault(i, (time.time(), d))
                    eta_s = (tot - d) / rate if rate and rate > 0 else None
                    note = (f"{d:,}/{tot:,} {unit}  {bar(d/tot, 18)} {100*d/tot:4.1f}%"
                            + (f"  {rate:.0f}/s  ETA {fmt_dur(eta_s)}" if rate else ""))
                elif d:
                    note = f"{d:,} {unit} so far"
                else:
                    note = unit
        else:
            icon, col, note = "    ", C["d"], ""
        out.append(f"  {col}{icon}{C['x']}  {s['name']:<24s} {C['d']}{note}{C['x']}")

    # ---- artefact tallies -------------------------------------------------
    out.append("")
    out.append(f"  {C['d']}{'-' * (w - 4)}{C['x']}")
    # Artefacts older than the build start belong to a previous run. Showing
    # "final records 20,000 / audit PASSED" while this build is still at stage 4
    # reads as if it had already finished.
    t_start = build_start()

    def stale(p):
        t = mtime(p)
        return t_start is not None and t is not None and t < t_start

    def tag(p):
        return f"  {C['d']}(previous build){C['x']}" if stale(p) else ""

    rows = []
    m = RAW / "manifest.json"
    if m.exists():
        try:
            rows.append(("real source tables", f"{len(json.loads(m.read_text())):,}"))
        except Exception:
            pass
    for lbl, p in (("real candidates", CACHE / "candidates_real.jsonl"),
                   ("synthetic candidates", CACHE / "candidates_synth.jsonl")):
        if p.exists():
            n = count_lines(p)
            rows.append((lbl, (f"{n:,}" if n else "-") + tag(p)))
    npng = count_pngs(DATA / "images")
    if npng:
        rows.append(("chart images", f"{npng:,} / {N_TARGET:,}"))
    fr = DATA / "records.jsonl"
    if fr.exists():
        rows.append(("final records", f"{count_lines(fr):,}" + tag(fr)))
    a = CACHE / "audit.json"
    if a.exists():
        try:
            j = json.loads(a.read_text())
            ok = j.get("passed")
            rows.append(("audit", (f"{C['g']}PASSED{C['x']}" if ok else f"{C['r']}FAILED{C['x']}")
                         + f"  labels re-verified {100*j.get('label_reverification_rate',0):.2f}%"
                         + tag(a)))
        except Exception:
            pass
    for k, v in rows:
        out.append(f"  {k:<22s} {C['b']}{v}{C['x']}")

    du = 0
    try:
        du = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file())
    except Exception:
        pass
    if du:
        out.append(f"  {'dataset size':<22s} {C['b']}{du/1e9:.2f} GB{C['x']}")

    if not once:
        out.append("")
        out.append(f"  {C['d']}Ctrl-C to stop watching (the build keeps running){C['x']}")

    sys.stdout.write(("\033[2J\033[H" if not once else "") + "\n".join(out) + "\n")
    sys.stdout.flush()
    return active is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    hist = {}
    if a.once:
        render(hist, t0, True)
        return 0
    try:
        while True:
            if render(hist, t0, False):
                return 0
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\nstopped watching (build continues)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
