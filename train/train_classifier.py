"""
train_classifier.py — Image-only baseline: predict the statistical finding from pixels.

    python train/train_classifier.py --smoke       # 2 min, verifies the whole loop
    python train/train_classifier.py --epochs 15   # full run

This is the honest "is the pattern visible in the image at all" baseline, and the
number a VLM has to beat. It deliberately gets no access to the data table.

WARNING ON AUGMENTATION
    Standard image pipelines start with RandomHorizontalFlip. That is silently
    label-destroying here: mirroring a scatter turns a positive correlation into
    a negative one, and mirroring a time series turns an increasing trend into a
    decreasing one -- both are separate classes in this taxonomy. Vertical flips
    break the distribution classes the same way. Only label-preserving transforms
    (mild crop/scale, colour jitter, JPEG-ish noise) are used below.
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_records(split=None, require_image=True):
    p = DATA / "records.jsonl"
    if not p.exists():
        p = ROOT / "cache" / "records.jsonl"
    recs = []
    with open(p) as fh:
        for line in fh:
            r = json.loads(line)
            if split and r["split"] != split:
                continue
            if require_image and not (DATA / r["image"]).exists():
                continue
            recs.append(r)
    return recs


class ChartDataset(Dataset):
    def __init__(self, recs, classes, size=224, train=False):
        from torchvision import transforms as T
        self.recs = recs
        self.cls = {c: i for i, c in enumerate(classes)}
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if train:
            # every transform here preserves the label -- see module docstring
            self.tf = T.Compose([
                T.Resize((size, size)),
                T.RandomApply([T.ColorJitter(0.25, 0.25, 0.2, 0.03)], p=0.7),
                T.RandomApply([T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05))], p=0.5),
                T.RandomApply([T.GaussianBlur(3, (0.1, 1.2))], p=0.2),
                T.ToTensor(), norm,
                T.RandomErasing(p=0.15, scale=(0.01, 0.04)),
            ])
        else:
            self.tf = T.Compose([T.Resize((size, size)), T.ToTensor(), norm])

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(DATA / r["image"]).convert("RGB")
        return self.tf(img), self.cls[r["label"]], i


def build_model(name, n_cls):
    import torchvision.models as M
    if name == "resnet18":
        m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, n_cls)
    elif name == "resnet50":
        m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, n_cls)
    elif name == "efficientnet_b0":
        m = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_cls)
    elif name == "vit_b_16":
        m = M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, n_cls)
    else:
        raise ValueError(name)
    return m


@torch.no_grad()
def evaluate(model, loader, device, recs, classes):
    model.eval()
    P, Y, IX = [], [], []
    for x, y, ix in loader:
        out = model(x.to(device, non_blocking=True))
        P.append(out.argmax(1).cpu())
        Y.append(y)
        IX.append(ix)
    P = torch.cat(P).numpy()
    Y = torch.cat(Y).numpy()
    IX = torch.cat(IX).numpy()

    acc = float((P == Y).mean())
    f1s = []
    per_class = {}
    for i, c in enumerate(classes):
        tp = int(((P == i) & (Y == i)).sum())
        fp = int(((P == i) & (Y != i)).sum())
        fn = int(((P != i) & (Y == i)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "n": int((Y == i).sum())}

    def slice_acc(key):
        d = defaultdict(lambda: [0, 0])
        for p, y, ix in zip(P, Y, IX):
            k = recs[ix].get(key)
            d[k][0] += int(p == y)
            d[k][1] += 1
        return {str(k): {"acc": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(d.items())}

    return {"accuracy": round(acc, 4), "macro_f1": round(float(np.mean(f1s)), 4),
            "per_class": per_class, "by_difficulty": slice_acc("difficulty"),
            "by_provenance": slice_acc("provenance"), "by_chart_type": slice_acc("chart_type"),
            "confusion": [[int(((Y == i) & (P == j)).sum()) for j in range(len(classes))]
                          for i in range(len(classes))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset + 1 epoch: proves the loop runs end to end")
    ap.add_argument("--out", default="train/runs")
    a = ap.parse_args()

    dev = pick_device()
    classes = json.loads((DATA / "class_index.json").read_text())["label_to_id"] \
        if (DATA / "class_index.json").exists() else None
    if classes:
        classes = sorted(classes, key=classes.get)
    else:
        sys.path.insert(0, str(ROOT / "build"))
        import taxonomy as tax
        classes = tax.CLASS_NAMES

    tr = load_records("train")
    va = load_records("val")
    te = load_records("test")
    if a.smoke:
        rng = np.random.default_rng(0)
        pick = lambda R, k: [R[i] for i in rng.choice(len(R), min(k, len(R)), replace=False)]  # noqa: E731
        tr, va, te = pick(tr, 600), pick(va, 200), pick(te, 200)
        a.epochs, a.size = 1, 160
    print(f"device={dev}  model={a.model}  train={len(tr):,} val={len(va):,} test={len(te):,}"
          f"  classes={len(classes)}")
    if not tr:
        print("no training records with images on disk yet -- is the render stage done?")
        return 1

    pin = dev.type == "cuda"
    dl = lambda ds, sh: DataLoader(ds, batch_size=a.bs, shuffle=sh, num_workers=a.workers,  # noqa: E731
                                   pin_memory=pin, drop_last=False, persistent_workers=a.workers > 0)
    dtr = dl(ChartDataset(tr, classes, a.size, train=True), True)
    dva = dl(ChartDataset(va, classes, a.size), False)
    dte = dl(ChartDataset(te, classes, a.size), False)

    model = build_model(a.model, len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, epochs=a.epochs,
                                                steps_per_epoch=max(1, len(dtr)), pct_start=0.25)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    outdir = ROOT / a.out
    outdir.mkdir(parents=True, exist_ok=True)
    best, hist = -1.0, []
    t0 = time.time()

    for ep in range(1, a.epochs + 1):
        model.train()
        tot = n = corr = 0
        te0 = time.time()
        for bi, (x, y, _) in enumerate(dtr):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.detach().item() * y.size(0)
            corr += int((out.detach().argmax(1) == y).sum())
            n += y.size(0)
            if bi % 20 == 0:
                print(f"\r  ep{ep} [{bi+1}/{len(dtr)}] loss {tot/max(n,1):.3f} "
                      f"acc {corr/max(n,1):.3f}", end="", flush=True)
        vm = evaluate(model, dva, dev, va, classes)
        print(f"\r  ep{ep}  train_loss {tot/max(n,1):.3f}  train_acc {corr/max(n,1):.3f}"
              f"  val_acc {vm['accuracy']:.4f}  val_macroF1 {vm['macro_f1']:.4f}"
              f"  ({time.time()-te0:.0f}s)")
        hist.append({"epoch": ep, "train_loss": tot / max(n, 1),
                     "train_acc": corr / max(n, 1), "val_acc": vm["accuracy"],
                     "val_macro_f1": vm["macro_f1"]})
        if vm["macro_f1"] > best:
            best = vm["macro_f1"]
            torch.save({"model": model.state_dict(), "classes": classes,
                        "args": vars(a)}, outdir / f"{a.model}_best.pt")

    print("\nevaluating on test ...")
    tm = evaluate(model, dte, dev, te, classes)
    probe = [r for r in te if r["provenance"] == "real"]
    pm = None
    if probe:
        dpr = dl(ChartDataset(probe, classes, a.size), False)
        pm = evaluate(model, dpr, dev, probe, classes)

    res = {"args": vars(a), "device": str(dev), "history": hist,
           "best_val_macro_f1": best, "test": tm,
           "test_real_only_probe": pm, "classes": classes,
           "wall_seconds": round(time.time() - t0, 1)}
    tag = "smoke" if a.smoke else a.model
    (outdir / f"metrics_{tag}.json").write_text(json.dumps(res, indent=2))

    print(f"\n{'='*64}")
    print(f"TEST  accuracy {tm['accuracy']:.4f}   macro-F1 {tm['macro_f1']:.4f}")
    if pm:
        print(f"REAL-ONLY PROBE  accuracy {pm['accuracy']:.4f}  "
              f"macro-F1 {pm['macro_f1']:.4f}  (n={len(probe)})")
    by_diff = {k: v["acc"] for k, v in tm["by_difficulty"].items()}
    by_prov = {k: v["acc"] for k, v in tm["by_provenance"].items()}
    print(f"by difficulty : {by_diff}")
    print(f"by provenance : {by_prov}")
    print(f"{'='*64}")
    print(f"metrics -> {outdir / f'metrics_{tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
