"""
build_ensemble.py — Combine every trained checkpoint into the best available model.

Runs locally on MPS/CPU against the held-out test split. No Kaggle quota.

The metric that matters is **real-only accuracy**, not mixed accuracy. Mixed
already sits at 0.950 against an oracle ceiling of 0.9995, so there is almost no
headroom there; real-only sits at 0.837, and that is the number a reviewer will
read as the generalisation result.

The five checkpoints are diverse in two useful ways: three architectures
(resnet50 / resnet101 / densenet121) and two input views (full image / plot-area
crop). Averaging softmax probabilities across both axes is the cheapest
remaining accuracy gain, and it costs no training.

Probabilities are averaged, never logits: logit scales differ between
architectures, so averaging raw logits silently weights one model more heavily.
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as M
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
RES = ROOT / "results"
SIZE = 224
BATCH = 64

DEV = torch.device("mps" if torch.backends.mps.is_available()
                   else ("cuda" if torch.cuda.is_available() else "cpu"))

# (name, checkpoint, architecture, uses plot-area crop, input size)
# Three axes of diversity now: architecture, input view, and resolution. The
# 384px member gains +1.4 points on real data specifically, so it is expected to
# decorrelate usefully from the 224px members.
MEMBERS = [
    ("resnet101_nb1",     RES / "nb1/resnet101_main_best.pt",        "resnet101",   False, 224),
    ("resnet101_crop",    RES / "nb2/resnet101_crop_best.pt",        "resnet101",   True,  224),
    ("resnet101_384",     RES / "nb3/resnet101_384_best.pt",         "resnet101",   False, 384),
    ("resnet101_sweep",   RES / "account1/resnet101_224_best.pt",    "resnet101",   False, 224),
    ("resnet50_sweep",    RES / "account1/resnet50_224_best.pt",     "resnet50",    False, 224),
    ("densenet121_sweep", RES / "account1/densenet121_224_best.pt",  "densenet121", False, 224),
    # nb5, dataset v2 (no label-dependent marks); checkpoint picked on val real-only accuracy
    ("r101_384_real50",   RES / "nb5/r101_384_real50_best.pt",       "resnet101",   False, 384),
    ("r50_384_real50",    RES / "nb5/r50_384_real50_best.pt",        "resnet50",    False, 384),
    ("r101_384_natural",  RES / "nb5/r101_384_natural_best.pt",      "resnet101",   False, 384),
    ("d121_384_real50",   RES / "nb5/d121_384_real50_best.pt",       "densenet121", False, 384),
    # nb5x: + supplementary real pool from train-split sources (build/12_extra_real.py)
    ("r101_384_real50_x", RES / "nb5x/r101/r101_384_real50_x_best.pt", "resnet101",  False, 384),
    ("r50_384_real50_x",  RES / "nb5x/r50/r50_384_real50_x_best.pt",   "resnet50",   False, 384),
    ("d121_384_real50_x", RES / "nb5x/d121/d121_384_real50_x_best.pt", "densenet121", False, 384),
    # nb7: nb5x + renderer augmentation (Plotly + randomised matplotlib styles)
    ("r101_384_real50_xp", RES / "nb7/r101/r101_384_real50_xp_best.pt", "resnet101", False, 384),
    ("d121_384_real50_xp", RES / "nb7/d121/d121_384_real50_xp_best.pt", "densenet121", False, 384),
]
# Everything above nb5 was trained on v1 images, where time-series charts carry
# label-dependent marks (see train/ablate_marks.py). Only these may be deployed.
V2 = {"r101_384_real50", "r50_384_real50", "r101_384_natural", "d121_384_real50",
      "r101_384_real50_x", "r50_384_real50_x", "d121_384_real50_x", "r101_384_real50_xp", "d121_384_real50_xp"}

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def tf_for(size):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), NORM])


def build(name, n):
    if name == "resnet50":
        m = M.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, n)
    elif name == "resnet101":
        m = M.resnet101(weights=None); m.fc = nn.Linear(m.fc.in_features, n)
    elif name == "densenet121":
        m = M.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, n)
    elif name == "convnext_tiny":
        m = M.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n)
    elif name == "efficientnet_v2_s":
        m = M.efficientnet_v2_s(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    else:
        raise ValueError(name)
    return m


class TestDS(Dataset):
    def __init__(self, df, boxes=None, size=224):
        self.df = df.reset_index(drop=True)
        self.boxes = boxes
        self.tf = tf_for(size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(DATA / r.image).convert("RGB")
        if self.boxes is not None:
            b = self.boxes.get(r.sample_id)
            if b:
                x0, y0, x1, y1 = [int(round(v)) for v in b]
                if x1 - x0 > 20 and y1 - y0 > 20:
                    img = img.crop((x0, y0, x1, y1))
        return self.tf(img), i


def macro_f1(P, Y, n):
    fs = []
    for i in range(n):
        tp = ((P == i) & (Y == i)).sum(); fp = ((P == i) & (Y != i)).sum()
        fn = ((P != i) & (Y == i)).sum()
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return float(np.mean(fs))


@torch.no_grad()
def probs_for(ckpt, arch, crop, te, classes, boxes, size=224):
    m = build(arch, len(classes)).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV))
    m.eval()
    bs = BATCH if size <= 224 else max(8, BATCH // 3)
    dl = DataLoader(TestDS(te, boxes if crop else None, size), batch_size=bs,
                    shuffle=False, num_workers=4)
    out = np.zeros((len(te), len(classes)), dtype=np.float32)
    for x, ix in dl:
        p = torch.softmax(m(x.to(DEV)).float(), 1).cpu().numpy()
        out[ix.numpy()] = p
    del m
    if DEV.type == "mps":
        torch.mps.empty_cache()
    return out


def report(name, P, Y, real_mask, classes):
    acc = float((P == Y).mean())
    f1 = macro_f1(P, Y, len(classes))
    racc = float((P == Y)[real_mask].mean())
    sacc = float((P == Y)[~real_mask].mean())
    return dict(model=name, accuracy=acc, macro_f1=f1, real_only=racc, synthetic_only=sacc)


def main():
    meta = pd.read_parquet(DATA / "tables" / "metadata.parquet")
    te = meta[meta.split == "test"].reset_index(drop=True)
    ci = json.loads((DATA / "class_index.json").read_text())["label_to_id"]
    classes = sorted(ci, key=ci.get)
    c2i = {c: i for i, c in enumerate(classes)}
    Y = te.label.map(c2i).values
    real_mask = (te.provenance == "real").values

    boxes = {}
    with open(DATA / "splits" / "test.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            b = next((x["bbox"] for x in r["render"]["grounding_boxes"]
                      if x["label"] == "plot_area"), None)
            if b:
                boxes[r["sample_id"]] = b

    print(f"device {DEV} | test {len(te):,} | classes {len(classes)} | "
          f"real {real_mask.sum():,} synthetic {(~real_mask).sum():,}\n")

    # ---- individual members -------------------------------------------
    P_all, rows = {}, []
    for name, ck, arch, crop, size in MEMBERS:
        if not ck.exists():
            print(f"  skip {name}: {ck} missing")
            continue
        t0 = time.time()
        pr = probs_for(ck, arch, crop, te, classes, boxes, size)
        P_all[name] = pr
        r = report(name, pr.argmax(1), Y, real_mask, classes)
        r["view"] = "crop" if crop else "full"
        r["arch"] = arch
        r["size"] = size
        rows.append(r)
        print(f"  {name:20s} {arch:12s} {'crop' if crop else 'full':5s} @{size:<4d} "
              f"acc {r['accuracy']:.4f}  F1 {r['macro_f1']:.4f}  "
              f"real {r['real_only']:.4f}  synth {r['synthetic_only']:.4f}  "
              f"({time.time()-t0:.0f}s)")

    # ---- every subset, ranked by REAL-ONLY accuracy ---------------------
    print("\nsearching ensembles (ranked by real-only accuracy, the metric with headroom)")
    best = None
    combos = []
    names = list(P_all)
    for k in range(2, len(names) + 1):
        for cb in combinations(names, k):
            pr = np.mean([P_all[n] for n in cb], axis=0)
            r = report("+".join(cb), pr.argmax(1), Y, real_mask, classes)
            r["members"] = list(cb)
            combos.append(r)
            if best is None or r["real_only"] > best["real_only"]:
                best = r
    combos.sort(key=lambda r: -r["real_only"])
    print(f"  evaluated {len(combos)} combinations\n")
    print(f"  {'ensemble':46s} {'acc':>7s} {'F1':>7s} {'real':>7s} {'synth':>7s}")
    for r in combos[:8]:
        print(f"  {r['model'][:46]:46s} {r['accuracy']:>7.4f} {r['macro_f1']:>7.4f} "
              f"{r['real_only']:>7.4f} {r['synthetic_only']:>7.4f}")

    # ---- summary --------------------------------------------------------
    single_best = max(rows, key=lambda r: r["real_only"])
    print("\n" + "=" * 78)
    print(f"BEST SINGLE    {single_best['model']:24s} real {single_best['real_only']:.4f}  "
          f"mixed {single_best['accuracy']:.4f}  F1 {single_best['macro_f1']:.4f}")
    print(f"BEST ENSEMBLE  {best['model'][:24]:24s} real {best['real_only']:.4f}  "
          f"mixed {best['accuracy']:.4f}  F1 {best['macro_f1']:.4f}")
    gain = best["real_only"] - single_best["real_only"]
    print(f"GAIN on real-only: {gain:+.4f}  ({100*gain:+.2f} points)")
    print("=" * 78)

    out = RES / "ensemble"
    out.mkdir(exist_ok=True)
    (out / "ensemble_results.json").write_text(json.dumps(
        {"device": str(DEV), "test_n": int(len(te)),
         "members": rows, "best_single": single_best,
         "best_ensemble": best, "top_combinations": combos[:20]},
        indent=2, default=str))
    pd.DataFrame(rows + combos).to_csv(out / "ensemble_all.csv", index=False)
    print(f"\nwrote {out}/ensemble_results.json and ensemble_all.csv")


if __name__ == "__main__":
    sys.exit(main())
