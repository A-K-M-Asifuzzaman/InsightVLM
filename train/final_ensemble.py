"""
final_ensemble.py — Pick the deployed ensemble on VAL, score it once on TEST.

    python train/cache_probs.py        # once, writes results/probs/*.npz
    python train/final_ensemble.py

Deployed policy: average the probabilities of every FULL-recipe member (50% real
batches + supplementary real pool + renderer augmentation) with EQUAL
weights, then apply a label-shift correction (Saerens et al. 2002): multiply by (p_real(y) / p_train(y))^alpha, using train-split class
frequencies only; alpha (0 = none) is the one thing tuned, on val.

Also reported: greedy forward selection with replacement on val (Caruana et al.
2004). With 553 real val charts it chases 1-2 point differences that are mostly
noise, which is why it is not the deployed policy.

Only v2-trained members (build_ensemble.V2) are candidates: v1 models learned the
label-dependent time-series marks. Crop-view members are excluded too: they need
the plot-area box, which an uploaded chart does not have. So the reported number
is exactly what app.py computes.

Writes results/final_ensemble.json, which app.py loads.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble import DATA, MEMBERS, RES, V2  # noqa: E402

ROOT = DATA.parent

ROUNDS = 12
ALPHAS = [0, 0.25, 0.5, 0.75, 1.0]
# The deployed members are the FULL recipe: 50% real batches + supplementary real pool
# + renderer augmentation (tag suffix _xp). Earlier members are ablation steps. The
# augmentation step was added for cross-renderer robustness before its results existed.
FULL_RECIPE_SUFFIX = "_xp"


def acc(P, y, mask=None):
    ok = P.argmax(1) == y
    return float(ok[mask].mean()) if mask is not None else float(ok.mean())


def main():
    meta = pd.read_parquet(DATA / "tables" / "metadata.parquet")
    ci = json.loads((DATA / "class_index.json").read_text())["label_to_id"]
    classes = sorted(ci, key=ci.get)
    split = {s: meta[meta.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    y = {s: split[s].label.map(ci).values for s in ("val", "test")}
    real = {s: (split[s].provenance == "real").values for s in ("val", "test")}

    probs, spec = {}, {}
    for name, ck, arch, crop, size in MEMBERS:
        f = RES / "probs" / f"{name}.npz"
        if crop or name not in V2 or not f.exists():
            continue
        z = np.load(f)
        probs[name] = {"val": z["val"], "test": z["test"]}
        spec[name] = {"ckpt": str(ck.relative_to(ROOT)), "arch": arch, "size": size}
    names = list(probs)
    MAIN_RECIPE = {n for n in names if n.endswith(FULL_RECIPE_SUFFIX)}
    ALL_CLEAN = {n for n in names if not n.endswith("_natural")}
    print(f"candidates: {names}\n")

    for n in names:
        print(f"  {n:20s} val_real {acc(probs[n]['val'], y['val'], real['val']):.4f}  "
              f"test_real {acc(probs[n]['test'], y['test'], real['test']):.4f}  "
              f"test_all {acc(probs[n]['test'], y['test']):.4f}")

    # train-split class priors: all data vs real only (smoothed, no val/test counts)
    tr = split["train"]
    p_all = tr.label.map(ci).value_counts().reindex(range(len(classes)), fill_value=0).values + 1.0
    p_real = tr[tr.provenance == "real"].label.map(ci).value_counts() \
        .reindex(range(len(classes)), fill_value=0).values + 1.0
    log_shift = np.log(p_real / p_real.sum()) - np.log(p_all / p_all.sum())

    def mix(counts, s, alpha):
        w = np.array([counts[n] for n in names], float)
        p = sum(wi * probs[n][s] for wi, n in zip(w, names) if wi) / w.sum()
        return p * np.exp(alpha * log_shift)

    def score(counts, alpha):
        p = mix(counts, "val", alpha)
        return (acc(p, y["val"], real["val"]), acc(p, y["val"]))  # real first, mixed breaks ties

    # Deployed policy: equal weights over the full-recipe members; only the label-shift
    # alpha is tuned on val. The all-members average and greedy selection are reported
    # as alternatives (greedy chases 1-2 point val differences that are mostly noise).
    main = {n: 1 if n in MAIN_RECIPE else 0 for n in names}
    eq_alpha = max(ALPHAS, key=lambda a: score(main, a))
    eq_val = score(main, eq_alpha)
    print(f"  equal weights over {[n for n in names if main[n]]}, alpha {eq_alpha}: "
          f"val_real {eq_val[0]:.4f} val_all {eq_val[1]:.4f}")

    best = None
    for alpha in ALPHAS:
        counts = dict.fromkeys(names, 0)
        cur = (-1, -1)
        for _ in range(ROUNDS):
            cand = max(names, key=lambda n: score({**counts, n: counts[n] + 1}, alpha))
            s = score({**counts, cand: counts[cand] + 1}, alpha)
            if s <= cur and sum(counts.values()):
                break
            counts[cand] += 1
            cur = s
        print(f"  alpha {alpha:<4} val_real {cur[0]:.4f} val_all {cur[1]:.4f}  "
              f"{ {k: v for k, v in counts.items() if v} }")
        if best is None or cur > best[0]:
            best = (cur, alpha, counts)

    def test_scores(counts, alpha):
        pt = mix(counts, "test", alpha)
        return {"real_only": acc(pt, y["test"], real["test"]),
                "synthetic_only": acc(pt, y["test"], ~real["test"]), "accuracy": acc(pt, y["test"])}

    (gvr, gva), galpha, gcounts = best
    counts, alpha = main, eq_alpha
    out = {
        "classes": classes,
        "policy": "equal weights over full-recipe members (+pool, +renderer aug); alpha tuned on val",
        "members": [{"name": n, **spec[n], "weight": counts[n]} for n in names if counts[n]],
        "alpha": alpha,
        "log_prior_shift": log_shift.round(6).tolist(),
        "val": {"real_only": eq_val[0], "accuracy": eq_val[1]},
        "test": test_scores(counts, alpha),
        "alternative_all_members": {"members": sorted(ALL_CLEAN), "test": test_scores(
            {n: int(n in ALL_CLEAN) for n in names}, max(ALPHAS, key=lambda a: score(
                {n: int(n in ALL_CLEAN) for n in names}, a)))},
        "alternative_greedy": {"members": {n: c for n, c in gcounts.items() if c}, "alpha": galpha,
                               "val": {"real_only": gvr, "accuracy": gva},
                               "test": test_scores(gcounts, galpha)},
    }
    g = out["alternative_greedy"]["test"]
    print(f"\n(alternative, greedy-on-val: real {g['real_only']:.4f}  synth {g['synthetic_only']:.4f})")
    print("\n" + "=" * 70)
    print(f"FINAL  members { {m['name']: m['weight'] for m in out['members']} }  alpha {alpha}")
    print(f"TEST   real {out['test']['real_only']:.4f}  synth {out['test']['synthetic_only']:.4f}  "
          f"all {out['test']['accuracy']:.4f}")
    print("=" * 70)
    (RES / "final_ensemble.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {RES / 'final_ensemble.json'}")


if __name__ == "__main__":
    main()
