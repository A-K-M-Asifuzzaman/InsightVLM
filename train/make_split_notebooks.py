"""
make_split_notebooks.py — Emit four small, independent Kaggle notebooks.

One 48-cell notebook running six jobs for six hours is all-or-nothing: any
failure loses everything after it, and across five runs four failures came from
optimisations layered onto that single pipeline (an oversized memmap cache, a
DataLoader worker leak, channels_last, a partially-indented guard).

Each notebook here does ONE job, holds no state from any other, and writes a
JSON result file. A failure costs that notebook only.

  nb1_main.ipynb        baselines + main model + full evaluation + figures   ~1.2 h
  nb2_crop.ipynb        text-shortcut ablation (plot-area crop)              ~0.6 h
  nb3_resolution.ipynb  resolution ablation at 384 px                        ~1.0 h
  nb4_vlm.ipynb         Qwen2-VL-2B QLoRA                                    ~2.3 h

NB1 alone produces every headline number the paper needs.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"

DATA_ROOT = "/kaggle/input/datasets/asifxzaman/xxxyyy/dataset/dataset"
TAX_PATH = "/kaggle/input/datasets/asifxzaman/xxxyyy/taxonomy.py"


def nb(cells):
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"},
                         "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


def md(s):
    return {"cell_type": "markdown", "metadata": {},
            "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}


# v2 = v1 with the 5,000 time-series charts re-rendered without label marks. Kaggle
# would not accept a new dataset, so they are rendered in-notebook from the v1 records
# with build/06_render.py itself (embedded verbatim, same crc32 style seeds).
RENDER_V2TS = "RENDER_SRC = " + repr((ROOT / "build" / "06_render.py").read_text()) + "\n" + '''
import subprocess, sys
_w = Path("/tmp/v2render"); (_w / "build").mkdir(parents=True, exist_ok=True); (_w / "cache").mkdir(exist_ok=True)
(_w / "build" / "06_render.py").write_text(RENDER_SRC)
with open(DATA / "records.jsonl") as _f, open(_w / "cache" / "records.jsonl", "w") as _g:
    for _l in _f:
        if '"arity": "timeseries"' in _l: _g.write(_l)
OV = _w / "dataset" / "images_v2ts"
if not OV.exists():
    subprocess.run([sys.executable, str(_w / "build" / "06_render.py"), "--arity", "timeseries",
                    "--clean", "--out", "images_v2ts"], check=True)
'''


# ==========================================================================
# shared preamble — identical in every notebook, no cross-notebook state
# ==========================================================================
PREAMBLE = f'''
import os, sys, json, time, random, warnings, gc
from pathlib import Path
from collections import Counter, defaultdict
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision.models as M

SEED = 20260815
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| device", DEVICE,
      "|", torch.cuda.device_count(), "GPU(s)")

DATA_ROOT = Path("{DATA_ROOT}")
def find_data():
    if (DATA_ROOT / "records.jsonl").exists(): return DATA_ROOT
    # the dataset root is the records.jsonl next to tables/metadata.parquet: other attached
    # datasets carry their own records.jsonl, and Kaggle's mount paths change between runs
    c = [p.parent for p in Path("/kaggle/input").glob("**/records.jsonl")
         if (p.parent / "tables" / "metadata.parquet").exists()]
    if not c: raise FileNotFoundError("dataset root (records.jsonl + tables/) not found under /kaggle/input")
    return c[0]
DATA = find_data(); print("dataset:", DATA)

meta = pd.read_parquet(DATA / "tables" / "metadata.parquet")
_ci = json.loads((DATA / "class_index.json").read_text())["label_to_id"]
CLASSES = sorted(_ci, key=_ci.get); NC = len(CLASSES); C2I = {{c: i for i, c in enumerate(CLASSES)}}
tr = meta[meta.split == "train"].reset_index(drop=True)
va = meta[meta.split == "val"].reset_index(drop=True)
te = meta[meta.split == "test"].reset_index(drop=True)
print(f"{{len(meta):,}} samples | {{NC}} classes | train {{len(tr):,}} val {{len(va):,}} test {{len(te):,}}")

# ---- data pipeline -------------------------------------------------------
# No memmap cache. It was the source of two separate failures (a 24.6 GB file
# that was disk-bound, then a 21 GB one that was OOM-killed) and reading PNGs
# directly costs ~90 img/s, which is fine when the GPU is the bottleneck anyway.
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def tf_train(size):
    # Label-preserving only. NO horizontal/vertical flip: mirroring turns
    # rel_strong_positive into rel_strong_negative and ts_increasing into
    # ts_decreasing, which are separate classes here.
    return T.Compose([
        T.Resize((size, size)),
        T.RandomApply([T.ColorJitter(0.25, 0.25, 0.2, 0.03)], p=0.7),
        T.RandomApply([T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05))], p=0.5),
        T.RandomApply([T.GaussianBlur(3, (0.1, 1.2))], p=0.2),
        T.ToTensor(), NORM, T.RandomErasing(p=0.15, scale=(0.01, 0.04))])

def tf_eval(size):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), NORM])

class ChartDS(Dataset):
    def __init__(self, df, size, train=False, crop_plot=False):
        self.df = df.reset_index(drop=True)
        self.tf = tf_train(size) if train else tf_eval(size)
        self.crop = crop_plot
        self.boxes = {{}}
        if crop_plot:
            for sp in ("train", "val", "test"):
                with open(DATA / "splits" / f"{{sp}}.jsonl") as fh:
                    for line in fh:
                        r = json.loads(line)
                        b = next((x["bbox"] for x in r["render"]["grounding_boxes"]
                                  if x["label"] == "plot_area"), None)
                        if b: self.boxes[r["sample_id"]] = b
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(DATA / r.image).convert("RGB")
        if self.crop:
            b = self.boxes.get(r.sample_id)
            if b:
                x0, y0, x1, y1 = [int(round(v)) for v in b]
                if x1 - x0 > 20 and y1 - y0 > 20: img = img.crop((x0, y0, x1, y1))
        return self.tf(img), C2I[r.label], i

def loader(df, size, train, bs=48, crop=False):
    # modest workers, no pinned memory, no persistent workers: the combination
    # of persistent_workers + pin_memory leaked ~5 GB per model
    return DataLoader(ChartDS(df, size, train, crop), batch_size=bs, shuffle=train,
                      num_workers=2, pin_memory=False, prefetch_factor=2,
                      persistent_workers=False)

def build(name, n=NC):
    if name == "resnet50":
        m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2); m.fc = nn.Linear(m.fc.in_features, n)
    elif name == "resnet101":
        m = M.resnet101(weights=M.ResNet101_Weights.IMAGENET1K_V2); m.fc = nn.Linear(m.fc.in_features, n)
    elif name == "densenet121":
        m = M.densenet121(weights=M.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, n)
    elif name == "resnet18":
        m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Linear(m.fc.in_features, n)
    elif name == "convnext_tiny":
        m = M.convnext_tiny(weights=M.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n)
    elif name == "efficientnet_v2_s":
        m = M.efficientnet_v2_s(weights=M.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    else: raise ValueError(name)
    return m

def macro_f1(P, Y, n=NC):
    fs = []
    for i in range(n):
        tp = ((P==i)&(Y==i)).sum(); fp = ((P==i)&(Y!=i)).sum(); fn = ((P!=i)&(Y==i)).sum()
        pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
        fs.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return float(np.mean(fs))

@torch.no_grad()
def predict(model, dl):
    model.eval(); P=[]; Y=[]; IX=[]
    for x, y, ix in dl:
        with torch.autocast("cuda", enabled=DEVICE.type=="cuda"):
            o = model(x.to(DEVICE, non_blocking=True))
        P.append(o.float().argmax(1).cpu()); Y.append(y); IX.append(ix)
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(IX).numpy()

def train_model(name, size, epochs, bs=48, crop=False, patience=4, tag=""):
    dl_tr, dl_va = loader(tr, size, True, bs, crop), loader(va, size, False, bs, crop)
    model = build(name).to(DEVICE)
    if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, epochs=epochs,
                                                steps_per_epoch=len(dl_tr), pct_start=0.25)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type=="cuda")
    ckpt = f"{{name}}{{tag}}_best.pt"; best = -1.0; stale = 0; hist = []
    for ep in range(1, epochs+1):
        model.train(); t0 = time.time(); tot=n=corr=0
        for bi, (x, y, _) in enumerate(dl_tr):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=DEVICE.type=="cuda"):
                out = model(x); loss = crit(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.detach().item()*y.size(0); corr += (out.detach().argmax(1)==y).sum().item(); n += y.size(0)
            if bi % 60 == 0:
                print(f"\\r  ep{{ep}} [{{bi}}/{{len(dl_tr)}}] loss {{tot/max(n,1):.3f}} acc {{corr/max(n,1):.3f}}", end="", flush=True)
        P, Y, _ = predict(model, dl_va); vf1 = macro_f1(P, Y)
        hist.append(dict(epoch=ep, loss=tot/n, train_acc=corr/n, val_acc=float((P==Y).mean()), val_f1=vf1))
        print(f"\\r  ep{{ep}}  loss {{tot/n:.3f}}  train {{corr/n:.3f}}  val_acc {{(P==Y).mean():.4f}}  val_F1 {{vf1:.4f}}  ({{time.time()-t0:.0f}}s)")
        if vf1 > best:
            best, stale = vf1, 0
            torch.save((model.module if isinstance(model, nn.DataParallel) else model).state_dict(), ckpt)
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop: val macro-F1 flat for {{patience}} epochs"); break
    clean = build(name).to(DEVICE); clean.load_state_dict(torch.load(ckpt)); clean.eval()
    del model, dl_tr, dl_va, opt, sched, scaler; gc.collect(); torch.cuda.empty_cache()
    return clean, hist, best, ckpt

def save(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
    print("wrote", path)
print("preamble ready")
'''

# ==========================================================================
# NB1 — baselines + main model + full evaluation
# ==========================================================================
nb1 = [
    md("""
# InsightVLM — NB1: Baselines, Main Model, Full Evaluation

**This notebook alone produces every headline number the paper needs.**
Independent of NB2/NB3/NB4 — it holds no state from them and they hold none from it.

| Stage | Time |
|---|---|
| Statistical oracle (ceiling) | 2 min |
| Trivial floors (chart-type prior) | instant |
| ResNet-50 @224, 12 epochs | ~45 min |
| Evaluation, real-only probe, figures, LaTeX | ~15 min |

Output: `nb1_results.json`, `fig1_confusion.png`, `fig2_per_class_f1.png`,
`fig3_difficulty_provenance.png`, `results_table.tex`
"""),
    code(PREAMBLE),
    md("## 1. Statistical oracle — the ceiling"),
    code(f'''
!pip -q install diptest pymannkendall ruptures
'''),
    code(f'''
import importlib.util
missing = [m for m in ("diptest","pymannkendall","ruptures","statsmodels")
           if importlib.util.find_spec(m) is None]
if missing:
    raise ImportError(f"taxonomy.py needs {{missing}}; it has SILENT fallbacks, so the "
                      f"oracle would report a wrong number instead of failing. "
                      f"Run the pip cell, then restart the kernel.")

TAX = None
for c in [Path("{TAX_PATH}")] + list(Path("/kaggle/input").glob("**/taxonomy.py")):
    if not c.exists(): continue
    sys.path.insert(0, str(c.parent)); import taxonomy as TAX; print("taxonomy:", c); break
if TAX is None: raise FileNotFoundError("taxonomy.py not attached")

_r = np.random.default_rng(0)
_x = np.r_[_r.normal(-3,.7,200), _r.normal(3,.7,200)]
assert TAX.verify_univariate(_x)[0] == "dist_bimodal", "diptest missing -- restart the kernel"
print("taxonomy self-check passed")

VERIFY = {{"univariate": lambda d: TAX.verify_univariate(d["x"]),
          "bivariate":  lambda d: TAX.verify_bivariate(d["x"], d["y"]),
          "timeseries": lambda d: TAX.verify_timeseries(d["t"], d["y"]),
          "grouped":    lambda d: TAX.verify_grouped(d["value"], d["group"]),
          "composition":lambda d: TAX.verify_composition(d["category"])}}
ok = sum(VERIFY[r.arity](json.loads(r.data_table_json))[0] == r.label for _, r in te.iterrows())
ORACLE = ok/len(te)
print(f"ORACLE accuracy: {{ORACLE:.4f}} ({{ok}}/{{len(te)}}) -- the ceiling")
'''),
    md("""
## 2. Trivial floors

The 5% majority baseline **overstates** results on this task: chart type
correlates with the finding by construction (a pie chart can only be a
composition class). The chart-type prior uses no pixels and is the honest floor.
"""),
    code('''
floors = {"majority": float((te.label == tr.label.value_counts().idxmax()).mean())}
prior = tr.groupby("chart_type")["label"].agg(lambda s: s.value_counts().idxmax())
pred_ct = te.chart_type.map(prior).fillna(tr.label.value_counts().idxmax())
floors["chart_type_prior"] = float((pred_ct.values == te.label.values).mean())
for k, v in floors.items(): print(f"  {k:20s} {v:.4f}")
print(f"  {'oracle (ceiling)':20s} {ORACLE:.4f}")
print("\\nA vision model is only interesting between the prior and the oracle.")
'''),
    md("## 3. Main model — ResNet-50 @ 224"),
    code('''
# Account 1 already trained resnet50 / resnet101 / densenet121 to convergence
# before its quota ran out. If you attach those checkpoints as a Kaggle dataset,
# this cell skips ~45 min of retraining and goes straight to the evaluation that
# account 1 never reached. resnet101 scored best there (macro-F1 0.9539), though
# it is tied with resnet50 to within noise.
MODEL, SIZE, EPOCHS = "resnet101", 224, 12
CKPT_HINTS = [f"{MODEL}_224_best.pt", f"{MODEL}_224_main_best.pt", f"{MODEL}_main_best.pt"]

found = None
for pat in CKPT_HINTS:
    hits = list(Path("/kaggle/input").glob(f"**/{pat}")) if Path("/kaggle/input").exists() else []
    if hits: found = hits[0]; break

if found:
    print(f"loading checkpoint: {found}")
    cnn = build(MODEL).to(DEVICE)
    cnn.load_state_dict(torch.load(found, map_location=DEVICE)); cnn.eval()
    hist, best_val = [], float("nan")
    print("skipped training -- evaluating the account-1 checkpoint")
else:
    print(f"no checkpoint found for {MODEL}; training from scratch (~45-60 min)")
    print("  (to skip this, upload results/account1/*.pt as a Kaggle dataset and attach it)")
    t0 = time.time()
    cnn, hist, best_val, _ = train_model(MODEL, SIZE, EPOCHS, tag="_main")
    print(f"\\ntrained in {(time.time()-t0)/60:.1f} min | best val macro-F1 {best_val:.4f}")

P, Y, IX = predict(cnn, loader(te, SIZE, False))
ACC, F1 = float((P==Y).mean()), macro_f1(P, Y)
print(f"TEST accuracy {ACC:.4f}  macro-F1 {F1:.4f}")
'''),
    md("## 4. Evaluation — per-class, per-difficulty, real-only probe"),
    code('''
res = pd.DataFrame({
    "true":[CLASSES[i] for i in Y], "pred":[CLASSES[i] for i in P],
    "difficulty":te.difficulty.values[IX], "provenance":te.provenance.values[IX],
    "family":te.family.values[IX], "chart_type":te.chart_type.values[IX]})
res["ok"] = res.true == res.pred

def boot_ci(mask, n=4000):
    a = np.asarray(mask, float); rng = np.random.default_rng(0)
    return tuple(np.percentile(a[rng.integers(0,len(a),(n,len(a)))].mean(1), [2.5,97.5]))

lo, hi = boot_ci(res.ok)
print(f"TEST accuracy {res.ok.mean():.4f}  [95% CI {lo:.4f}, {hi:.4f}]  macro-F1 {F1:.4f}\\n")
for k in ("family","difficulty","provenance"):
    g = res.groupby(k).ok.agg(["mean","count"]).round(4); g.columns=["accuracy","n"]
    print(f"--- by {k} ---"); display(g)

probe = res[res.provenance=="real"]
plo, phi = boot_ci(probe.ok)
print(f"\\nREAL-ONLY PROBE  n={len(probe)}  accuracy {probe.ok.mean():.4f} [95% CI {plo:.4f}, {phi:.4f}]")
print(f"synthetic-only   accuracy {res[res.provenance=='synthetic'].ok.mean():.4f}")
print("\\nA large gap means the model learned the generator, not the statistics.")
'''),
    md("## 5. Figures and LaTeX table"),
    code('''
cm = pd.crosstab(res.true, res.pred).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
cmn = cm.div(cm.sum(1).replace(0,1), axis=0)
fig, ax = plt.subplots(figsize=(11,9))
im = ax.imshow(cmn.values, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(NC)); ax.set_xticklabels(CLASSES, rotation=90, fontsize=7)
ax.set_yticks(range(NC)); ax.set_yticklabels(CLASSES, fontsize=7)
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{MODEL} @{SIZE}px")
for i in range(NC):
    for j in range(NC):
        v = cmn.values[i,j]
        if v > 0.02: ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=5.5,
                             color="white" if v>0.5 else "black")
fig.colorbar(im, shrink=0.8); plt.tight_layout()
plt.savefig("fig1_confusion.png", dpi=200, bbox_inches="tight"); plt.show()

print("most confused pairs:")
off = cmn.copy(); np.fill_diagonal(off.values, 0)
for (t,p), v in off.stack().sort_values(ascending=False).head(8).items():
    print(f"  {t:30s} -> {p:30s} {v:.3f}")
'''),
    code('''
f1_by = {}
for i,c in enumerate(CLASSES):
    tp=((P==i)&(Y==i)).sum(); fp=((P==i)&(Y!=i)).sum(); fn=((P!=i)&(Y==i)).sum()
    pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
    f1_by[c]=2*pr*rc/(pr+rc) if pr+rc else 0
s = pd.Series(f1_by).sort_values()
fig, ax = plt.subplots(figsize=(7,6))
ax.barh(range(len(s)), s.values,
        color=["#c0392b" if v<.5 else "#e67e22" if v<.75 else "#27ae60" for v in s.values])
ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=7)
ax.axvline(s.mean(), ls="--", c="k", lw=1, label=f"macro-F1 {s.mean():.3f}")
ax.set_xlabel("F1"); ax.legend(); plt.tight_layout()
plt.savefig("fig2_per_class_f1.png", dpi=200, bbox_inches="tight"); plt.show()

fig, axes = plt.subplots(1,2, figsize=(11,3.6))
d = res.groupby("difficulty").ok.mean().reindex(["easy","medium","hard"])
axes[0].plot(d.index, d.values, "o-", lw=2); axes[0].set_ylim(0,1)
axes[0].set_ylabel("accuracy"); axes[0].set_title("Accuracy vs difficulty")
pv = res.groupby("provenance").ok.mean()
axes[1].bar(pv.index, pv.values, color=["#2980b9","#95a5a6"]); axes[1].set_ylim(0,1)
axes[1].set_title("Accuracy by provenance")
for i,v in enumerate(pv.values): axes[1].text(i, v+.02, f"{v:.3f}", ha="center")
plt.tight_layout(); plt.savefig("fig3_difficulty_provenance.png", dpi=200, bbox_inches="tight"); plt.show()
'''),
    code('''
summary = pd.DataFrame([
    {"Model":"Majority class","Accuracy":floors["majority"],"Macro-F1":np.nan},
    {"Model":"Chart-type prior (no pixels)","Accuracy":floors["chart_type_prior"],"Macro-F1":np.nan},
    {"Model":f"{MODEL} @{SIZE}px","Accuracy":ACC,"Macro-F1":F1},
    {"Model":"Statistical oracle (ceiling)","Accuracy":ORACLE,"Macro-F1":np.nan},
]).round(4)
display(summary)
tex = summary.to_latex(index=False, na_rep="--", float_format="%.4f",
        caption="Baselines on InsightVLM-20K (test split, 4{,}000 samples).",
        label="tab:baselines")
Path("results_table.tex").write_text(tex); print(tex)

save({"oracle":ORACLE, "floors":floors,
      "model":{"name":MODEL,"size":SIZE,"epochs":EPOCHS,"accuracy":ACC,"macro_f1":F1,
               "ci95":[lo,hi],"best_val_f1":best_val,"history":hist},
      "by_difficulty":res.groupby("difficulty").ok.mean().round(4).to_dict(),
      "by_provenance":res.groupby("provenance").ok.mean().round(4).to_dict(),
      "by_family":res.groupby("family").ok.mean().round(4).to_dict(),
      "per_class_f1":{k:round(v,4) for k,v in f1_by.items()},
      "real_only_probe":{"n":int(len(probe)),"accuracy":float(probe.ok.mean()),"ci95":[plo,phi]}},
     "nb1_results.json")
res.to_csv("nb1_test_predictions.csv", index=False)
print("\\nNB1 complete. Every headline number for the paper is in nb1_results.json.")
'''),
]

# ==========================================================================
# NB2 — text-shortcut ablation
# ==========================================================================
nb2 = [
    md("""
# InsightVLM — NB2: Text-Shortcut Ablation

Chart titles read `"Distribution of X"`, `"Y vs X"`, `"Y over X"` — each reveals
the **arity**, which narrows the candidate classes sharply. A model reading text
gets part of the task for free.

The `plot_area` grounding box excludes the title and axis labels. Training on
that crop answers it directly: **if accuracy holds, the model reads data; if it
collapses, it was reading captions.**

Same model, same size, same epochs as NB1 — only the crop differs, so the gap is
purely the text effect. ~35 min. Output: `nb2_results.json`
"""),
    code(PREAMBLE),
    code('''
# Same backbone as NB1: the ablation isolates the crop, nothing else.
MODEL, SIZE, EPOCHS = "resnet101", 224, 12
print("=== plot-area crop (title and axis labels removed) ===")
t0 = time.time()
cnn, hist, best_val, _ = train_model(MODEL, SIZE, EPOCHS, crop=True, tag="_crop")
P, Y, IX = predict(cnn, loader(te, SIZE, False, crop=True))
acc, f1 = float((P==Y).mean()), macro_f1(P, Y)
prov = te.provenance.values[IX]
print(f"\\ncropped: accuracy {acc:.4f}  macro-F1 {f1:.4f}  ({(time.time()-t0)/60:.1f} min)")
print(f"  real-only      {float((P==Y)[prov=='real'].mean()):.4f}")
print(f"  synthetic-only {float((P==Y)[prov=='synthetic'].mean()):.4f}")
print("\\nCompare against NB1's full-image number: the difference is the text shortcut.")
save({"model":MODEL,"size":SIZE,"epochs":EPOCHS,"crop":"plot_area",
      "accuracy":acc,"macro_f1":f1,"best_val_f1":best_val,
      "real_only":float((P==Y)[prov=='real'].mean()),
      "synthetic_only":float((P==Y)[prov=='synthetic'].mean()),
      "history":hist}, "nb2_results.json")
'''),
]

# ==========================================================================
# NB3 — resolution ablation
# ==========================================================================
nb3 = [
    md("""
# InsightVLM — NB3: Resolution Ablation @ 384 px

Charts encode meaning in thin lines, small markers and axis text. 224 px is the
ImageNet convention, not necessarily right here.

Same model and epoch budget as NB1 — **only the input resolution differs**, so
the comparison is clean. Batch is reduced to keep activations inside 16 GB.
~60 min. Output: `nb3_results.json`
"""),
    code(PREAMBLE),
    code('''
# Same backbone and epoch budget as NB1: only resolution differs.
MODEL, SIZE, EPOCHS, BS = "resnet101", 384, 12, 12
print(f"=== {MODEL} @ {SIZE}px, batch {BS} ===")
t0 = time.time()
cnn, hist, best_val, _ = train_model(MODEL, SIZE, EPOCHS, bs=BS, tag="_384")
P, Y, IX = predict(cnn, loader(te, SIZE, False, bs=BS))
acc, f1 = float((P==Y).mean()), macro_f1(P, Y)
prov = te.provenance.values[IX]
print(f"\\n@{SIZE}px: accuracy {acc:.4f}  macro-F1 {f1:.4f}  ({(time.time()-t0)/60:.1f} min)")
print(f"  real-only      {float((P==Y)[prov=='real'].mean()):.4f}")
print(f"  synthetic-only {float((P==Y)[prov=='synthetic'].mean()):.4f}")
print("\\nCompare against NB1 @224px: the difference is the resolution effect.")
save({"model":MODEL,"size":SIZE,"epochs":EPOCHS,"batch":BS,
      "accuracy":acc,"macro_f1":f1,"best_val_f1":best_val,
      "real_only":float((P==Y)[prov=='real'].mean()),
      "synthetic_only":float((P==Y)[prov=='synthetic'].mean()),
      "history":hist}, "nb3_results.json")
'''),
]

# ==========================================================================
# NB4 — VLM
# ==========================================================================
nb4 = [
    md("""
# InsightVLM — NB4: Qwen2-VL-2B + QLoRA

The generative model: produces the finding, the reasoning chain and the insight,
not just a class id. **Completely independent of NB1–NB3** — run it on a separate
account if GPU quota is tight. ~2.3 h. Output: `nb4_results.json`

**After the install cell, restart the kernel**, then run the rest. `transformers`
is preinstalled on Kaggle; upgrading it in-session leaves the old module loaded
and `Qwen2VLForConditionalGeneration` may not resolve until a restart.
"""),
    code('''
!pip -q install -U "transformers>=4.45" accelerate peft trl bitsandbytes qwen-vl-utils
print("INSTALL DONE -- now RESTART THE KERNEL, then run the cells below.")
'''),
    code(f'''
import os, sys, json, time, re, gc
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset as TorchDS

DATA_ROOT = Path("{DATA_ROOT}")
DATA = DATA_ROOT if (DATA_ROOT / "records.jsonl").exists() else \\
       list(Path("/kaggle/input").glob("**/records.jsonl"))[0].parent
print("dataset:", DATA)
meta = pd.read_parquet(DATA / "tables" / "metadata.parquet")
te = meta[meta.split == "test"].reset_index(drop=True)
_ci = json.loads((DATA / "class_index.json").read_text())["label_to_id"]
CLASSES = sorted(_ci, key=_ci.get)
print(f"{{len(meta):,}} samples | {{len(CLASSES)}} classes | test {{len(te):,}}")
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
'''),
    code('''
from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                          BitsAndBytesConfig, TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
# 1500 keeps the run near 1.5 h on a T4. Raise it if quota allows -- the
# size is printed and stored in nb4_results.json, so report whatever you use.
N_TRAIN = 1500
rows = [json.loads(l) for l in open(DATA / "instruct" / "train.jsonl")][:N_TRAIN]
for r in rows: r["image"] = str(DATA / r["image"])
print(len(rows), "training samples (subset -- report this size in the paper)")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.float16)   # T4 is fp16, not bf16
processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=768*28*28)
vlm = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
vlm = prepare_model_for_kbit_training(vlm); vlm.config.use_cache = False
vlm = get_peft_model(vlm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
vlm.print_trainable_parameters()
'''),
    code('''
class VLMDataset(TorchDS):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def collate(batch):
    texts, images = [], []
    for b in batch:
        msgs = [{"role":"user","content":[{"type":"image","image":b["image"]},
                 {"type":"text","text":b["messages"][0]["content"].replace("<image>\\n","")}]},
                {"role":"assistant","content":[{"type":"text","text":b["messages"][1]["content"]}]}]
        texts.append(processor.apply_chat_template(msgs, tokenize=False))
        imgs, _ = process_vision_info(msgs); images.append(imgs)
    enc = processor(text=texts, images=images, return_tensors="pt",
                    padding=True, truncation=True, max_length=1024)
    labels = enc["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    for tok in ("<|image_pad|>","<|vision_start|>","<|vision_end|>"):
        tid = processor.tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid >= 0: labels[labels == tid] = -100
    enc["labels"] = labels
    return enc

# TrainingArguments' signature changes between transformers releases -- on the
# Kaggle image `warmup_ratio` raised TypeError. Rather than pin a version, keep
# only the kwargs this build actually accepts and report what was dropped.
import inspect
_want = dict(output_dir="insightvlm-lora", num_train_epochs=1,
             per_device_train_batch_size=1, gradient_accumulation_steps=8,
             learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
             logging_steps=10, save_strategy="epoch", fp16=True,
             gradient_checkpointing=True, optim="paged_adamw_8bit",
             report_to="none", remove_unused_columns=False, dataloader_num_workers=2)
_sig = inspect.signature(TrainingArguments.__init__).parameters
_ok = {k: v for k, v in _want.items() if k in _sig}
_dropped = sorted(set(_want) - set(_ok))
print("transformers", __import__("transformers").__version__)
if _dropped:
    print("  unsupported in this build, dropped:", _dropped)
targs = TrainingArguments(**_ok)

# Trainer picks NotebookProgressCallback in a notebook, which renders an IPython
# widget. Widgets never reach Kaggle's log stream, so a batch run shows no
# progress at all. Force plain text instead.
from transformers import TrainerCallback
class PlainProgress(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            el = time.time() - _t0
            done = state.global_step / max(state.max_steps, 1)
            eta = el / done - el if done > 0 else 0
            print(f"  step {state.global_step}/{state.max_steps} "
                  f"({100*done:.0f}%)  " +
                  "  ".join(f"{k}={v:.4g}" for k, v in logs.items() if isinstance(v, (int, float))) +
                  f"  elapsed {el/60:.0f}m  eta {eta/60:.0f}m", flush=True)

_t0 = time.time()
trainer = Trainer(model=vlm, args=targs, train_dataset=VLMDataset(rows),
                  data_collator=collate, callbacks=[PlainProgress()])
trainer.remove_callback(type(trainer.callback_handler.callbacks[-2]))  # drop the widget one
print(f"training {len(rows)} samples -> {targs.max_steps if targs.max_steps>0 else len(rows)//8} steps")
trainer.train()
vlm.save_pretrained("insightvlm-lora/final")
print("saved -> insightvlm-lora/final")
'''),
    code('''
vlm.eval()
PROMPT = ("You are a data analyst. Examine this chart and report the statistical "
          "findings it supports. State the dominant pattern, cite the numbers that "
          "justify it, and note any caveat a reader should know.")

def generate(path, max_new_tokens=320):
    msgs = [{"role":"user","content":[{"type":"image","image":path},{"type":"text","text":PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs)
    inp = processor(text=[text], images=[imgs], return_tensors="pt").to(vlm.device)
    with torch.no_grad():
        out = vlm.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]

CLS = json.loads((DATA/"class_index.json").read_text())["classes"]
DESC = {CLS[c]["desc"]: c for c in CLASSES}
def parse_class(t):
    for d,c in DESC.items():
        if d[:40].lower() in t.lower(): return c
    for c in CLASSES:
        if c.replace("_"," ") in t.lower(): return c
    return None

# CAUTION: the SFT target embeds the class description verbatim, so the model can
# score well by reciting a memorised phrase. The real-vs-synthetic gap is the check.
ev = te.sample(300, random_state=0); out_rows = []
for k,(_,r) in enumerate(ev.iterrows()):
    txt = generate(str(DATA / r.image))
    gt = set(re.findall(r"-?\\d+\\.\\d{2}", str(json.loads(r.statistics_json))))
    pr = set(re.findall(r"-?\\d+\\.\\d{2}", txt))
    out_rows.append({"true":r.label,"pred":parse_class(txt),"provenance":r.provenance,
                     "numeric_fidelity":len(gt&pr)/max(len(pr),1),"text":txt})
    if k % 25 == 0: print(f"\\r  {k}/{len(ev)}", end="", flush=True)

v = pd.DataFrame(out_rows); v["ok"] = v.true == v.pred
print(f"\\n\\nVLM class accuracy   {v.ok.mean():.4f}")
print(f"VLM numeric fidelity {v.numeric_fidelity.mean():.4f}")
display(v.groupby("provenance").ok.agg(["mean","count"]).round(4))
print("\\nA large real-vs-synthetic gap means template recitation, not chart reading.")
v.to_csv("nb4_vlm_predictions.csv", index=False)
Path("nb4_results.json").write_text(json.dumps({
    "n_train":len(rows), "n_eval":len(v), "class_accuracy":float(v.ok.mean()),
    "numeric_fidelity":float(v.numeric_fidelity.mean()),
    "by_provenance":v.groupby("provenance").ok.mean().round(4).to_dict()}, indent=2))
print("wrote nb4_results.json")
'''),
]

# ==========================================================================
# NB5 — real-data generalisation
# ==========================================================================
nb5 = [
    md("""
# InsightVLM — NB5: Closing the Real-vs-Synthetic Gap

NB1–NB3 models reach ~0.998 on synthetic but ~0.85 on real charts. Two causes
are fixable without new data:

1. **Sampling** — real is 31% of train, so the loss is dominated by synthetic
   samples the model already gets right. A weighted sampler makes each batch ~50% real.
2. **Model selection** — checkpoints were picked on mixed val F1, which is
   saturated. Here the checkpoint is picked on **val real-only accuracy**.

Runs on **dataset v2**, where time-series charts carry no label-dependent marks
(v1 drew a trend line / changepoint line / anomaly circles only on the matching
class, and models learned to read them: real time-series accuracy fell from 0.96
to 0.31 when the marks were removed).

Four runs at 384 px: ResNet-101, ResNet-50 and DenseNet-121 with ~50% real batches, plus ResNet-101 with plain
shuffling as the sampling ablation. Test is touched once per run, at the end.

Output: `<tag>_best.pt`, `<tag>_probs.npz` (val/test softmax, metadata order),
`nb5_results.json`. ~4.5 h on 2xT4.
"""),
    code(PREAMBLE),
    code(RENDER_V2TS + '''
from torch.utils.data import WeightedRandomSampler
for _df in (tr, va, te):
    _m = _df.arity == "timeseries"
    _df.loc[_m, "image"] = [str(OV / p.split("/", 1)[1]) for p in _df.loc[_m, "image"]]
_ts = pd.concat([tr, va, te]).query("arity == 'timeseries'").image
assert len(_ts) == 5000 and all(Path(p).exists() for p in _ts), "overlay incomplete"
print(f"v2 overlay: {len(_ts):,} clean time-series images from {OV}")
# The metadata frame carries every chart's full data table (up to 58k points each).
# DataLoader workers fork the process and slowly copy it; NB5 v1 and v3 both died of
# host OOM early in epoch 2. Training needs four columns.
_KEEP = ["sample_id", "image", "label", "provenance"]
tr, va, te = (d[_KEEP].copy() for d in (tr, va, te))
del meta, _ts; gc.collect()
import psutil
TRAIN_DS = globals().get("TRAIN_DS", ChartDS)
is_real = (tr.provenance == "real").values
va_real = (va.provenance == "real").values
te_real = (te.provenance == "real").values
print(f"train real {is_real.sum():,} synth {(~is_real).sum():,}")

@torch.no_grad()
def probs(model, df, size, bs):
    model.eval(); out = np.zeros((len(df), NC), np.float32)
    for x, _, ix in loader(df, size, False, bs):
        with torch.autocast("cuda", enabled=DEVICE.type=="cuda"):
            o = model(x.to(DEVICE, non_blocking=True))
        out[ix.numpy()] = torch.softmax(o.float(), 1).cpu().numpy()
    return out

def train_real(name, tag, size=384, epochs=12, bs=24, lr=3e-4, real_share=0.5, patience=4):
    # real_share=None: plain shuffle (natural ~31% real), the sampling ablation
    ds = TRAIN_DS(tr, size, train=True)
    if real_share is None:
        samp = dict(shuffle=True)
    else:
        w = np.where(is_real, real_share / is_real.sum(), (1 - real_share) / (~is_real).sum())
        samp = dict(sampler=WeightedRandomSampler(torch.tensor(w), len(tr), replacement=True))
    # 2 workers, as in NB1-3: 4 workers + ConvNeXt killed the NB5 v1 kernel
    dl_tr = DataLoader(ds, batch_size=bs, num_workers=2, prefetch_factor=2, **samp)
    model = build(name).to(DEVICE)
    if torch.cuda.device_count() > 1: model = nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=len(dl_tr), pct_start=0.15)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type=="cuda")
    ckpt = f"{tag}_best.pt"; best = (-1, -1); stale = 0; hist = []
    for ep in range(1, epochs+1):
        model.train(); t0 = time.time(); tot=n=corr=0
        for bi, (x, y, _) in enumerate(dl_tr):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=DEVICE.type=="cuda"):
                out = model(x); loss = crit(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()*y.size(0); corr += (out.argmax(1)==y).sum().item(); n += y.size(0)
            if bi % 60 == 0:
                print(f"  ep{ep} [{bi}/{len(dl_tr)}] loss {tot/n:.3f} acc {corr/n:.3f}", flush=True)
        pv = probs(model, va, size, bs).argmax(1); ok = pv == va.label.map(C2I).values
        vr, vm = float(ok[va_real].mean()), float(ok.mean())
        hist.append(dict(epoch=ep, loss=tot/n, train_acc=corr/n, val_real=vr, val_acc=vm))
        print(f"  ep{ep} loss {tot/n:.3f} train {corr/n:.3f} val_real {vr:.4f} val_acc {vm:.4f} "
              f"({time.time()-t0:.0f}s, RAM {psutil.virtual_memory().percent:.0f}%)", flush=True)
        if (vr, vm) > best:   # real-only first, mixed breaks ties
            best, stale = (vr, vm), 0
            torch.save((model.module if isinstance(model, nn.DataParallel) else model).state_dict(), ckpt)
        else:
            stale += 1
            if stale >= patience: print("  early stop"); break
    del model, opt, dl_tr; gc.collect(); torch.cuda.empty_cache()
    m = build(name).to(DEVICE); m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    if torch.cuda.device_count() > 1: m = nn.DataParallel(m)
    pv_, pt_ = probs(m, va, size, bs), probs(m, te, size, bs)
    np.savez(f"{tag}_probs.npz", val=pv_, test=pt_)
    okt = pt_.argmax(1) == te.label.map(C2I).values
    r = dict(model=name, tag=tag, size=size, real_share=real_share, best_val_real=best[0], best_val_acc=best[1],
             test_acc=float(okt.mean()), test_real=float(okt[te_real].mean()),
             test_synth=float(okt[~te_real].mean()), history=hist)
    print(f"== {tag}: TEST acc {r['test_acc']:.4f}  real {r['test_real']:.4f}  synth {r['test_synth']:.4f}")
    del m; gc.collect(); torch.cuda.empty_cache()
    return r
'''),
    code('''
RUNS = [("resnet101", "r101_384_real50", 0.5),    # main model
        ("resnet50",  "r50_384_real50",  0.5),    # second ensemble member
        ("densenet121", "d121_384_real50", 0.5),  # third, different architecture
        ("resnet101", "r101_384_natural", None)]  # ablation: same, without real oversampling
RESULTS = {}
for name, tag, share in RUNS:
    t0 = time.time()
    RESULTS[tag] = train_real(name, tag, real_share=share)
    RESULTS[tag]["minutes"] = round((time.time()-t0)/60, 1)
    save(RESULTS, "nb5_results.json")   # after each run, so a timeout keeps finished ones
'''),
]

# ==========================================================================
# NB6 — zero-shot open VLM baselines
# ==========================================================================
nb6 = [
    md("""
# InsightVLM — NB6: Zero-Shot Open VLM Baselines

Can a general-purpose VLM name the statistical finding with no task training?
Each model sees the chart plus all 20 findings (name + one-line definition) and
must answer with one label. Greedy decoding, same prompt for every model.

Evaluated on **all 1,192 real test charts** plus a fixed 400-chart synthetic sample.
Each model is saved as soon as it finishes, so a timeout keeps completed ones.
Output: `zs_<model>.csv`, `nb6_results.json`.
"""),
    code('''
!pip -q install -U transformers accelerate qwen-vl-utils
'''),
    code(f'''
import os, json, time, re, gc
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image

DATA_ROOT = Path("{DATA_ROOT}")
DATA = DATA_ROOT if (DATA_ROOT / "records.jsonl").exists() else \\
       list(Path("/kaggle/input").glob("**/records.jsonl"))[0].parent
meta = pd.read_parquet(DATA / "tables" / "metadata.parquet")
te = meta[meta.split == "test"]
EVAL = pd.concat([te[te.provenance == "real"],
                  te[te.provenance == "synthetic"].sample(400, random_state=0)]).reset_index(drop=True)
INFO = json.loads((DATA / "class_index.json").read_text())["classes"]
CLASSES = list(INFO)
print(f"eval {{len(EVAL)}} charts ({{(EVAL.provenance=='real').sum()}} real)")
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

PROMPT = ("You are a statistician. Look at the chart and decide which ONE statistical "
          "finding it shows. Options:\\n" +
          "\\n".join(f"- {{c}}: {{INFO[c]['desc']}}" for c in CLASSES) +
          "\\nAnswer with exactly one option name from the list (e.g. dist_normal) and nothing else.")

def parse(txt):
    t = txt.strip().lower()
    hits = [c for c in CLASSES if c in t]
    if hits: return min(hits, key=t.index)          # first-mentioned label
    for c in CLASSES:                                # tolerate spaces instead of underscores
        if c.replace("_", " ") in t: return c
    return None
'''),
    code('''
from transformers import AutoProcessor, AutoModelForImageTextToText

MODELS = ["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"]
RESULTS = {}

def run(model_id, dtype=torch.float16):
    # 768x576 charts are already inside the default pixel budget, no resize knobs needed
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto").eval()
    rows, t0 = [], time.time()
    for k, r in EVAL.iterrows():
        msgs = [{"role": "user", "content": [{"type": "image", "image": str(DATA / r.image)},
                                             {"type": "text", "text": PROMPT}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=16, do_sample=False)
        txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        rows.append({"sample_id": r.sample_id, "true": r.label, "pred": parse(txt),
                     "provenance": r.provenance, "raw": txt})
        if k == 9 and sum(x["pred"] is None for x in rows) >= 8 and dtype == torch.float16:
            # T4 has no bf16; some Qwen builds overflow in fp16 and emit garbage
            print(f"  fp16 output unparseable ({rows[-1]['raw']!r}), retrying in bf16", flush=True)
            del model; gc.collect(); torch.cuda.empty_cache()
            return run(model_id, torch.bfloat16)
        if k % 100 == 0:
            el = time.time() - t0
            print(f"  {model_id} {k}/{len(EVAL)}  {el/60:.0f}m  eta {el/(k+1)*(len(EVAL)-k-1)/60:.0f}m", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    return pd.DataFrame(rows)

for mid in MODELS:
    try:
        d = run(mid)
    except Exception as e:                      # a missing/unsupported model must not sink the rest
        print(f"SKIP {mid}: {type(e).__name__}: {e}"); continue
    tag = mid.split("/")[-1]
    d.to_csv(f"zs_{tag}.csv", index=False)
    d["ok"] = d.true == d.pred
    RESULTS[tag] = {"real": float(d[d.provenance == "real"].ok.mean()),
                    "synthetic": float(d[d.provenance == "synthetic"].ok.mean()),
                    "unparsed": float(d.pred.isna().mean()), "n": len(d)}
    print(tag, RESULTS[tag], flush=True)
    Path("nb6_results.json").write_text(json.dumps(RESULTS, indent=2))
'''),
]

# NB5x — NB5 plus the supplementary real training pool (build/12_extra_real.py), one
# model per notebook. NB5 v4 and NB5b both died of host OOM on their third model:
# RAM grows ~2-4% per epoch and is not returned between models in one kernel, so
# each model now gets a fresh kernel.
EXTRA_SNIPPET = '''
# + supplementary real pool: verified real candidates the balanced set left out,
# from TRAIN-split sources only (build/12_extra_real.py asserts no val/test source)
_ex = [p for p in Path("/kaggle/input").glob("**/extra_real.csv")]
assert _ex, "attach dataset asifxzaman/insightvlm-extra-real"
_x = pd.read_csv(_ex[0])
# Kaggle nests an uploaded zip one level deeper (images_extra/images_extra/...), so
# locate the folder holding the class directories from one known file
_hit = next(Path("/kaggle/input").glob(f"**/{_x.label.iloc[0]}/{_x.sample_id.iloc[0]}.png"))
_x["image"] = [str(_hit.parent.parent / l / f"{s}.png") for l, s in zip(_x.label, _x.sample_id)]
assert all(Path(p).exists() for p in _x.image), "extra images missing"
tr = pd.concat([tr, _x[_KEEP]], ignore_index=True)
print(f"+ {len(_x):,} extra real training charts -> train {len(tr):,}")
'''


def nb5x(runs, title):
    import copy
    cells = copy.deepcopy(nb5)
    src = lambda c: "".join(c["source"])  # noqa: E731
    i = next(k for k, c in enumerate(cells) if "import psutil" in src(c))
    cells[i] = code(src(cells[i]).replace("import psutil\n", "import psutil\n" + EXTRA_SNIPPET, 1))
    j = next(k for k, c in enumerate(cells) if "RUNS = [" in src(c))
    s = src(cells[j])
    a, b = s.index("RUNS = ["), s.index("RESULTS = {}")
    cells[j] = code(s[:a] + "RUNS = " + repr(runs) + "\n" + s[b:])
    cells[0] = md(f"# InsightVLM — {title}\n\nNB5 recipe (v2 clean renders, ~50% real batches, "
                  "checkpoint on val real accuracy) plus the supplementary real training pool. "
                  f"One model: `{runs[0][1]}`.")
    return cells


NB5X = {
    "nb5x_r101": nb5x([("resnet101", "r101_384_real50_x", 0.5)], "NB5x: ResNet-101 + extra real"),
    "nb5x_r50": nb5x([("resnet50", "r50_384_real50_x", 0.5)], "NB5x: ResNet-50 + extra real"),
    "nb5x_d121": nb5x([("densenet121", "d121_384_real50_x", 0.5)], "NB5x: DenseNet-121 + extra real"),
}

# NB7 — NB5x plus renderer augmentation: each training chart is shown as its
# matplotlib, Plotly or randomised-style render, uniformly, per sample per epoch. Test renders are
# never uploaded; the Vega-Lite test set stays a renderer no model has seen.
AUG_SNIPPET = '''
# + renderer augmentation: every TRAINING chart (train + extra) also exists as a Plotly
# render and a randomised-style matplotlib render. Views share sample ids, so each is
# located under its own folder name.
_ac = [p for p in Path("/kaggle/input").glob("**/aug_train.csv")]
assert _ac, "attach dataset asifxzaman/insightvlm-aug-train"
_a = pd.read_csv(_ac[0])
VIEWS = ["plotly", "mplrand"]
for _v in VIEWS:
    _g = _a[_a.view == _v]
    _hit = next(Path("/kaggle/input").glob(f"**/images_{_v}_train/{_g.label.iloc[0]}/{_g.sample_id.iloc[0]}.png"))
    _m = {s: str(_hit.parent.parent / l / f"{s}.png") for l, s in zip(_g.label, _g.sample_id)}
    tr["img_" + _v] = tr.sample_id.map(_m)
    assert tr["img_" + _v].notna().mean() > 0.99, f"{_v}: only {tr['img_' + _v].notna().mean():.1%} covered"
    assert Path(tr["img_" + _v].dropna().iloc[-1]).exists()
assert not set(_a.sample_id) & (set(te.sample_id) | set(va.sample_id)), "augmentation touches val/test"

class RendererAugDS(ChartDS):
    # original matplotlib render or one of the alternative views, uniformly
    def __getitem__(self, i):
        r = self.df.iloc[i]
        opts = [DATA / r.image] + [r["img_" + v] for v in VIEWS if isinstance(r["img_" + v], str)]
        img = Image.open(opts[random.randrange(len(opts))]).convert("RGB")
        return self.tf(img), C2I[r.label], i
TRAIN_DS = RendererAugDS
print(f"renderer augmentation: {len(tr):,} train charts x (matplotlib + {' + '.join(VIEWS)})")
'''


def nb7(runs, title):
    cells = nb5x(runs, title)
    src = lambda c: "".join(c["source"])  # noqa: E731
    i = next(k for k, c in enumerate(cells) if "extra_real.csv" in src(c))
    s = src(cells[i])
    at = s.index("TRAIN_DS = globals()")
    cells[i] = code(s[:at] + AUG_SNIPPET + "\n" + s[at:])
    cells[0] = md(f"# InsightVLM — {title}\n\nNB5x recipe (clean v2 renders, extra real pool, ~50% "
                  "real batches, checkpoint on val real accuracy) plus renderer augmentation: each "
                  "training chart is shown as its matplotlib, Plotly or randomised-style render at "
                  "random. Vega-Lite stays unseen. One model: "
                  f"`{runs[0][1]}`.")
    return cells


NB7 = {
    "nb7_r101": nb7([("resnet101", "r101_384_real50_xp", 0.5)], "NB7: ResNet-101 + extra real + renderer aug"),
    "nb7_d121": nb7([("densenet121", "d121_384_real50_xp", 0.5)], "NB7: DenseNet-121 + extra real + renderer aug"),
}

# NB6b — the same zero-shot run on the time-series part of NB6's eval set only,
# with the clean v2 images. Every other chart is byte-identical between v1 and v2,
# so NB6's predictions on them stand and only these need re-running.
_src = lambda c: "".join(c["source"])  # noqa: E731
nb6b = [
    md("""
# InsightVLM — NB6b: Zero-Shot VLMs on Clean Time-Series Charts (v2)

NB6 ran on v1 images, where time-series charts carried label-dependent marks.
This reruns the same models, prompt and parser on the time-series charts of
NB6's eval set using the clean v2 renders. Output: `zsts_<model>.csv`.
"""),
    nb6[1],
    code(_src(nb6[2]) + "\n" + RENDER_V2TS + '''

EVAL = EVAL[EVAL.arity == "timeseries"].reset_index(drop=True)
EVAL["image"] = [str(OV / p.split("/", 1)[1]) for p in EVAL.image]
assert all(Path(p).exists() for p in EVAL.image)
print(f"v2 time-series eval: {len(EVAL)} charts ({(EVAL.provenance=='real').sum()} real)")
'''),
    code(_src(nb6[3]).replace('f"zs_{tag}.csv"', 'f"zsts_{tag}.csv"')
         .replace('"nb6_results.json"', '"nb6b_results.json"')),
]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cells in [("nb1_main", nb1), ("nb2_crop", nb2),
                        ("nb3_resolution", nb3), ("nb4_vlm", nb4), ("nb5_real", nb5),
                        ("nb6_zeroshot", nb6), ("nb6b_zeroshot_ts", nb6b)] + list(NB5X.items()) + list(NB7.items()):
        p = OUT / f"{name}.ipynb"
        p.write_text(json.dumps(nb(cells), indent=1))
        print(f"{p.name:24s} {len(cells):2d} cells  {p.stat().st_size/1024:5.0f} KB")
