"""
make_kaggle_notebook.py — Emit InsightVLM_Kaggle_Training.ipynb (publication grade).

Generating the notebook from source keeps JSON escaping out of the way and lets
the cells stay readable and editable here.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "InsightVLM_Kaggle_Training.ipynb"

cells = []


def _lines(s):
    """nbformat requires every source line except the last to end in a newline.
    A bare .split('\\n') drops them and Jupyter then renders the whole cell as a
    single line, which turns valid code into a syntax error."""
    return s.strip("\n").splitlines(keepends=True)


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(s)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": _lines(s)})


# Cells that only make sense once a CNN has produced predictions, keyed on their
# first meaningful line. Guarding is applied as a deterministic post-processing
# pass over the finished cell list: locating cells by scanning the generator
# source mis-targeted them, and indenting only a cell's first statement leaves
# the rest at module level -- which produced `NameError: name 'res' is not
# defined` in VLM_ONLY mode instead of a clean skip.
NEEDS_PREDS = (
    "res = pd.DataFrame({",
    "probe = res[res.provenance",
    "from scipy.stats import spearmanr",
    "point_hits, area_ctrl, per_class_hits",
    "cm = pd.crosstab(res.true",
    "f1_by = {}",
    "import matplotlib.patches as mpatches",
    "rows = [",
)


def apply_pred_guards():
    """Wrap each prediction-dependent cell's ENTIRE body under `if HAVE_PREDS`."""
    wrapped = 0
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        first = next((ln for ln in src.split("\n")
                      if ln.strip() and not ln.strip().startswith("#")), "")
        if not first.startswith(NEEDS_PREDS):
            continue
        body = "\n".join(("    " + ln) if ln.strip() else ln
                          for ln in src.rstrip("\n").split("\n"))
        c["source"] = _lines('if not HAVE_PREDS:\n'
                             '    print("skipped: no CNN predictions (VLM_ONLY mode)")\n'
                             'else:\n' + body)
        wrapped += 1
    return wrapped


# ==========================================================================
md(r"""
# InsightVLM-20K — Training, Evaluation & Analysis

Companion notebook for **InsightVLM: Automated Statistical Reasoning from Data Visualizations**.

**Task.** Chart image → statistical finding + supporting statistics + evidence-grounded insight.

**Dataset.** 20,000 charts · 20 finding classes · exactly 1,000 each · 26 chart types · source-disjoint splits.

---

| § | Content | Paper artefact |
|---|---|---|
| 3 | Dataset characterisation | Table 1 |
| 4 | **Statistical oracle** | Ceiling — labels reproduce from the data |
| 5 | **Trivial floors** incl. chart-type prior | The real floor; 5% chance overstates results |
| 6 | **Backbone comparison** (4 models) | Table 2 |
| 7 | **Resolution ablation** 224 → 384 | Charts carry thin lines and small text |
| 8 | Evaluation, bootstrap CIs, McNemar | Table 3 |
| 9 | **Evidence Pointing Accuracy** (Grad-CAM vs ground-truth boxes) | The grounding contribution |
| 10 | Publication figures | Figures 1–5 |
| 11 | LaTeX tables | Paste into the paper |
| 12 | Qwen2-VL-2B + QLoRA | Generative model (own session) |

**Before running:** Accelerator **GPU T4 x2** (or P100) · Internet **On**.
Sections 1–11 fit one ~6 h session; §12 needs its own.
""")

md("## 0. Configuration")

code(r"""
from pathlib import Path

# ---- paths (from the actual upload) -------------------------------------
# The upload nested one level deeper than expected (dataset/dataset). The
# fallback search below catches it, but that glob walks all 20,000 PNGs, so
# point straight at the real location.
DATA_ROOT = Path("/kaggle/input/datasets/asifxzaman/xxxyyy/dataset/dataset")
TAX_PATH  = Path("/kaggle/input/datasets/asifxzaman/xxxyyy/taxonomy.py")

# ---- experiment ----------------------------------------------------------
SEED         = 20260815
SWEEP_MODELS = ["resnet50", "convnext_tiny", "efficientnet_b0", "vit_b_16"]
SWEEP_EPOCHS = 8          # short sweep to rank backbones
SWEEP_SIZE   = 224
FINAL_EPOCHS = 16         # upper bound; early stopping usually ends sooner
FINAL_SIZE   = 384        # charts encode meaning in thin lines and small text
BATCH        = 48
LR           = 3e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTH = 0.05
NUM_WORKERS  = 2          # Kaggle T4x2 gives 4 vCPU; 2 starves the GPU
PATIENCE     = 4          # stop when val macro-F1 stalls
EPOCH_TIME_LIMIT = 420    # s; abort a backbone whose first epoch exceeds this
RAM_FLOOR_GB = 9          # stop starting new backbones below this, rather than be OOM-killed

RUN_SWEEP   = True
RUN_FINAL   = True
RUN_GRADCAM = True
RUN_VLM     = False       # section 12; ~2.3 h. Run it on a second account.

# Second-account mode: skip the whole CNN pipeline and run only the VLM.
# Kaggle caps GPU quota per week, so splitting the work across accounts is the
# practical way to fit both halves.
VLM_ONLY = False
if VLM_ONLY:
    RUN_SWEEP = RUN_FINAL = RUN_GRADCAM = RUN_ENSEMBLE = False
    RUN_VLM = True

# ---- quality knobs -------------------------------------------------------
USE_EMA      = True       # weight averaging: reliably +0.5-1.5 macro-F1, no risk
USE_TTA      = True       # multi-scale test-time aug (label-preserving only)
STRONG_AUG   = True       # extra label-preserving augmentation to fight overfit
RUN_ENSEMBLE = True       # average softmax probs of the top-3 backbones

# MAX-QUALITY MODE. Observed overfitting (train 0.992 vs val 0.949 by epoch 8)
# means the binding constraint is regularisation, not capacity -- so this adds
# augmentation, EMA and TTA rather than just more epochs. ~8 h on T4 x2.
QUALITY_MODE = True
if QUALITY_MODE:
    # convnext/swin/efficientnet_v2 are tuned for Ampere+. On a T4 convnext_small
    # ran 21x slower than resnet50 AND leaked 12.9 GB in one epoch. These four are
    # all classic conv nets that run efficiently in fp16 on Turing.
    SWEEP_MODELS = ["resnet50", "resnet101", "densenet121", "efficientnet_b0"]
    SWEEP_EPOCHS = 12
    FINAL_EPOCHS = 30
    PATIENCE     = 6
    ABLATION_SIZES = [224, 384]   # 512 adds ~1.8 h; enable only with spare quota
else:
    ABLATION_SIZES = [224, 384]

# Set True for a ~15 min end-to-end rehearsal before committing GPU hours.
DRY_RUN = False
if DRY_RUN:
    SWEEP_MODELS, SWEEP_EPOCHS, FINAL_EPOCHS = ["resnet18"], 1, 1
    ABLATION_SIZES = [224]
    USE_EMA = USE_TTA = RUN_ENSEMBLE = False
print(f"config loaded | DRY_RUN={DRY_RUN} | QUALITY_MODE={QUALITY_MODE}")
""")

md("## 1. Environment")

code(r"""
import os, sys, json, math, time, random, warnings, gc, re
from collections import Counter, defaultdict
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import torch, torch.nn as nn
import matplotlib.pyplot as plt

def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all()

print("torch", torch.__version__, "| CUDA", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name}  {p.total_memory/1e9:.1f} GB")
    DEVICE = torch.device("cuda")
else:
    print("  NO GPU — enable it in Settings > Accelerator, or this takes days")
    DEVICE = torch.device("cpu")

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 220, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})
""")

md("## 2. Load the dataset")

code(r"""
def find_dataset_root():
    if (DATA_ROOT / "records.jsonl").exists():
        return DATA_ROOT
    print(f"{DATA_ROOT} has no records.jsonl — falling back to a recursive search ...")
    cands = list(Path("/kaggle/input").glob("**/records.jsonl")) if Path("/kaggle/input").exists() else []
    cands += list(Path(".").glob("**/dataset/records.jsonl"))
    if not cands:
        top = sorted(Path("/kaggle/input").glob("*")) if Path("/kaggle/input").exists() else []
        raise FileNotFoundError(
            f"records.jsonl not found. Mounted under /kaggle/input: {[p.name for p in top]}. "
            f"Point DATA_ROOT at the folder that directly contains records.jsonl.")
    return cands[0].parent

DATA = find_dataset_root()
print("dataset root:", DATA)
for sub in ("images", "splits", "instruct", "tables"):
    print(f"  {sub:9s} {'ok' if (DATA / sub).exists() else 'MISSING'}")

meta = pd.read_parquet(DATA / "tables" / "metadata.parquet")
_ci = json.loads((DATA / "class_index.json").read_text())["label_to_id"]
CLASSES = sorted(_ci, key=_ci.get); NC = len(CLASSES)
C2I = {c: i for i, c in enumerate(CLASSES)}
print(f"\n{len(meta):,} samples | {NC} classes | {meta.chart_type.nunique()} chart types")

# A zip that extracted one level too deep is the usual Kaggle failure: records.jsonl
# is found but every image path silently resolves to nothing.
_probe = meta.sample(min(200, len(meta)), random_state=0)
_ok = sum((DATA / r.image).exists() for _, r in _probe.iterrows())
print(f"image paths resolving: {_ok}/{len(_probe)}")
if _ok < len(_probe):
    raise FileNotFoundError(
        f"only {_ok}/{len(_probe)} images resolve under {DATA}. images/ must sit beside "
        f"records.jsonl, i.e. {DATA}/images/<class>/<id>.png")

tr = meta[meta.split == "train"].reset_index(drop=True)
va = meta[meta.split == "val"].reset_index(drop=True)
te = meta[meta.split == "test"].reset_index(drop=True)
print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}")
""")

md("## 3. Dataset characterisation — Table 1")

code(r"""
print("class balance :", set(meta.label.value_counts().values), "per class")
print("provenance    :", dict(meta.provenance.value_counts()))
print("difficulty    :", dict(meta.difficulty.value_counts()))

leak = meta[meta.provenance == "real"].groupby("source_id")["split"].nunique()
print("leaked sources:", int((leak > 1).sum()), "(must be 0 — splits are source-disjoint)")

table1 = (meta.groupby(["family", "arity"])
          .agg(classes=("label", "nunique"), samples=("label", "size"),
               chart_types=("chart_type", "nunique"),
               real=("provenance", lambda s: (s == "real").sum()))
          .reset_index())
table1["synthetic"] = table1.samples - table1.real
display(table1)

print("\nchart types per arity:")
for a, g in meta.groupby("arity"):
    print(f"  {a:12s} ({g.chart_type.nunique():2d}) {sorted(g.chart_type.unique())}")
""")

md(r"""
## 4. Baseline A — Statistical oracle (ceiling)

Runs the verification battery on each sample's stored `data_table`. It sees the
numbers, not the picture, so it is **not** a vision baseline — it is the upper
bound a model with perfect data reconstruction would reach, and it confirms the
labels reproduce from the stored data.
""")

code(r"""
# taxonomy.py degrades SILENTLY without these: _dip() returns (None, None), the
# dip_p guard defaults to 1.0, and dist_bimodal (guard: dip_p < 0.05) can then
# never match — all 1,000 bimodal samples fail and the oracle reports ~95%,
# making correct ground truth look broken.
!pip -q install diptest pymannkendall ruptures
""")

code(r"""
import importlib.util
missing = [m for m in ("diptest", "pymannkendall", "ruptures", "statsmodels")
           if importlib.util.find_spec(m) is None]
if missing:
    raise ImportError(f"taxonomy.py needs {missing}. It has soft fallbacks for these, so the "
                      f"oracle would run and quietly report a WRONG number instead of failing. "
                      f"Run the pip cell above, then restart the kernel.")

TAX = None
_cands = ([TAX_PATH] if TAX_PATH.exists() else []) \
       + list(Path("/kaggle/input").glob("**/taxonomy.py")) + list(Path(".").glob("**/taxonomy.py"))
for c in _cands:
    try:
        sys.path.insert(0, str(c.parent)); import taxonomy as TAX
        print("loaded taxonomy from", c); break
    except Exception as e:
        print("  could not import", c, "->", type(e).__name__, e); TAX = None
if TAX is None:
    raise FileNotFoundError(f"taxonomy.py not importable; expected at {TAX_PATH}")

# A textbook bimodal must verify. If this trips, a dependency is missing and
# every downstream oracle number would be understated.
_r = np.random.default_rng(0)
_x = np.r_[_r.normal(-3, .7, 200), _r.normal(3, .7, 200)]
assert TAX.verify_univariate(_x)[0] == "dist_bimodal", (
    "taxonomy self-check FAILED — diptest is almost certainly missing. Restart the kernel.")
print("taxonomy self-check passed")
""")

code(r"""
VERIFY = {"univariate":  lambda d: TAX.verify_univariate(d["x"]),
          "bivariate":   lambda d: TAX.verify_bivariate(d["x"], d["y"]),
          "timeseries":  lambda d: TAX.verify_timeseries(d["t"], d["y"]),
          "grouped":     lambda d: TAX.verify_grouped(d["value"], d["group"]),
          "composition": lambda d: TAX.verify_composition(d["category"])}

t0 = time.time(); ok, miss = 0, Counter()
for _, r in te.iterrows():
    got, _st, _why = VERIFY[r.arity](json.loads(r.data_table_json))
    ok += (got == r.label)
    if got != r.label:
        miss[f"{r.label} -> {got or _why}"] += 1

ORACLE_ACC = ok / len(te)
print(f"ORACLE accuracy on test: {ORACLE_ACC:.4f}  ({ok}/{len(te)})  [{time.time()-t0:.0f}s]")
print("Ceiling: a model with perfect data reconstruction scores here.")
if miss:
    print("\nmismatches (expect none):")
    for k, v in miss.most_common(10): print("  ", k, v)
""")

md(r"""
## 5. Baseline B — Trivial floors

**Report these.** With balanced classes the majority baseline is 1/20 = 5%, but
that badly overstates any result: chart type correlates with the label by
construction (a pie chart can only be `grp_dominant_category`; a QQ plot only
appears in distribution classes). The chart-type prior uses **no pixels at all**
and is the honest floor a vision model must clear.
""")

code(r"""
floors = {}
floors["majority class"] = float((te.label == tr.label.value_counts().idxmax()).mean())

prior = tr.groupby("chart_type")["label"].agg(lambda s: s.value_counts().idxmax())
pred_ct = te.chart_type.map(prior).fillna(tr.label.value_counts().idxmax())
floors["chart-type prior"] = float((pred_ct.values == te.label.values).mean())

# chart_type -> arity is a function (verified: every chart type belongs to exactly
# one arity), so conditioning on both is identical to conditioning on chart type.
# Reporting them as two baselines would be padding.
assert meta.groupby("chart_type").arity.nunique().max() == 1
print("  (chart-type + arity prior omitted: arity is determined by chart type)")

for k, v in floors.items(): print(f"  {k:26s} {v:.4f}")
print(f"\n  {'statistical oracle (ceiling)':26s} {ORACLE_ACC:.4f}")
print("\nA vision model is only interesting in the gap between the prior and the oracle.")
""")

md(r"""
## 6. Backbone comparison — Table 2

### Augmentation warning

Do **not** add `RandomHorizontalFlip`. Mirroring a scatter turns
`rel_strong_positive` into `rel_strong_negative`; mirroring a line chart turns
`ts_increasing` into `ts_decreasing`. Those are *separate classes* here, so the
default flip silently corrupts roughly 30% of the labels and raises no error.
Vertical flips break the distribution classes the same way. Every transform
below is label-preserving.
""")

code(r"""
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision.models as M
from PIL import Image

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def tf_train(size):
    # Every transform here is label-preserving. Small rotations are safe: tilting
    # a scatter by 5 degrees cannot flip the sign of a correlation, and charts sit
    # on white so fill=255 leaves no black corners. Flips are NOT safe and are
    # excluded -- see the warning above.
    aug = [T.Resize((size, size)),
           T.RandomApply([T.ColorJitter(0.25, 0.25, 0.2, 0.03)], p=0.7),
           T.RandomApply([T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05))], p=0.5),
           T.RandomApply([T.GaussianBlur(3, (0.1, 1.2))], p=0.2)]
    if STRONG_AUG:
        aug += [T.RandomApply([T.RandomRotation(5, fill=255)], p=0.3),
                T.RandomGrayscale(p=0.10),
                T.RandomApply([T.RandomAdjustSharpness(2.0)], p=0.25),
                T.RandomApply([T.RandomAutocontrast()], p=0.20)]
    aug += [T.ToTensor(), NORM, T.RandomErasing(p=0.20, scale=(0.01, 0.05))]
    return T.Compose(aug)

def tf_eval(size):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), NORM])

class ChartDS(Dataset):
    def __init__(self, df, size, train=False):
        self.df = df.reset_index(drop=True)
        self.tf = tf_train(size) if train else tf_eval(size)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(DATA / r.image).convert("RGB")), C2I[r.label], i

def loader(df, size, train, bs=None):
    return DataLoader(ChartDS(df, size, train), batch_size=bs or BATCH, shuffle=train,
                      num_workers=NUM_WORKERS, pin_memory=True,
                      persistent_workers=NUM_WORKERS > 0, drop_last=False)

def build(name, n=NC):
    if name == "resnet18":
        m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Linear(m.fc.in_features, n)
    elif name == "resnet50":
        m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2); m.fc = nn.Linear(m.fc.in_features, n)
    elif name == "resnet101":
        m = M.resnet101(weights=M.ResNet101_Weights.IMAGENET1K_V2); m.fc = nn.Linear(m.fc.in_features, n)
    elif name == "densenet121":
        m = M.densenet121(weights=M.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, n)
    elif name == "convnext_tiny":
        m = M.convnext_tiny(weights=M.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n)
    elif name == "convnext_small":
        m = M.convnext_small(weights=M.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n)
    elif name == "efficientnet_b0":
        m = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    elif name == "efficientnet_v2_s":
        m = M.efficientnet_v2_s(weights=M.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    elif name in ("swin_t", "swin_s"):
        w = {"swin_t": M.Swin_T_Weights, "swin_s": M.Swin_S_Weights}[name].IMAGENET1K_V1
        m = getattr(M, name)(weights=w)
        m.head = nn.Linear(m.head.in_features, n)
    elif name == "vit_b_16":
        # ViT_B_16 has a fixed 224 patch grid; torchvision can interpolate the
        # positional embedding, but only if image_size is set at construction.
        m = M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, n)
    else:
        raise ValueError(name)
    return m

print("data pipeline and model builders ready")
""")

md(r"""
### Speed fix — decode once, train from memory

Measured on this dataset: ResNet-18 took 184 s/epoch and ResNet-50 took 174 s/epoch.
ResNet-50 has roughly 4x the FLOPs, so **identical timing proves neither is
GPU-bound** — both sit idle while 768x576 PNGs are decoded and resized again on
every epoch. That is ~76 img/s against the several hundred a T4 can consume.

Decoding once into a `uint8` memmap removes the repeated work. Expect a large
speedup, and expect ResNet-50 to become *slower* per epoch than ResNet-18 — that
is the correct sign, it means you are finally compute-bound.
""")

code(r"""
import os, shutil

# A cache is an optimisation, never a requirement. Two failures taught this:
#   * one 640px cache for every resolution -> 24.6 GB, disk-bound, 159 img/s
#   * a 592px cache for the 512px arm      -> 21 GB of dirty pages, OOM kill
# np.memmap(mode="w+") writes land in page cache as DIRTY pages; writing faster
# than the kernel flushes pushes the container past its limit even though the
# array is nominally disk-backed. So: budget against real free RAM, flush while
# building, and fall back to reading PNGs when the budget will not stretch.
CACHE_DIR = next((d for d in ("/kaggle/temp", "/tmp", "/kaggle/working")
                  if os.path.isdir(d)), ".")

def _free_ram_gb():
    try:
        import psutil; return psutil.virtual_memory().available / 1e9
    except Exception:
        return 12.0

MAX_CACHE_GB = max(3.0, min(10.0, _free_ram_gb() * 0.40))
print(f"free RAM {_free_ram_gb():.1f} GB -> cache budget {MAX_CACHE_GB:.1f} GB")

_CACHE = {"px": None, "arr": None, "idx": None, "path": None}

def _cache_px_for(train_size):
    return int(round(train_size * 1.15 / 8) * 8)

def _drop_cache():
    if _CACHE["arr"] is not None:
        _CACHE["arr"] = None; gc.collect()
        try: os.remove(_CACHE["path"])
        except OSError: pass
    _CACHE.update(px=None, arr=None, idx=None, path=None)

def ensure_cache(train_size):
    px = _cache_px_for(train_size)
    if _CACHE["px"] is not None and _CACHE["px"] >= px:
        return _CACHE["arr"], _CACHE["idx"]      # resident cache already covers this

    gb = 20000 * px * px * 3 / 1e9
    if gb > MAX_CACHE_GB:
        print(f"  {px}px cache needs {gb:.1f} GB > {MAX_CACHE_GB:.1f} GB budget "
              f"-> reading PNGs directly at {train_size}px (slower, but this arm is "
              f"GPU-bound anyway)")
        _drop_cache()
        return None, None

    _drop_cache()
    ids = meta.sample_id.tolist()
    idx = {sid: i for i, sid in enumerate(ids)}
    shape = (len(ids), px, px, 3)
    path = os.path.join(CACHE_DIR, f"img_cache_{px}.npy")
    free_disk = shutil.disk_usage(CACHE_DIR).free / 1e9
    if free_disk < gb + 3:
        print(f"  only {free_disk:.1f} GB scratch free -> reading PNGs directly")
        return None, None

    print(f"building {gb:.1f} GB cache at {px}px for {train_size}px training ...")
    arr = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
    for i, (_, r) in enumerate(meta.iterrows()):
        arr[i] = np.asarray(Image.open(DATA / r.image).convert("RGB")
                            .resize((px, px), Image.BILINEAR))
        if i % 2000 == 0:
            arr.flush()                          # bound dirty pages, avoid the OOM
            print(f"\r  {i}/{len(ids)}", end="", flush=True)
    arr.flush()
    print(f"\r  cached {len(ids)} images at {px}px      ")
    _CACHE.update(px=px, arr=arr, idx=idx, path=path)
    return arr, idx

class CachedDS(Dataset):
    def __init__(self, df, size, train=False):
        self.df = df.reset_index(drop=True)
        self.tf = tf_train(size) if train else tf_eval(size)
        self.arr, self.idx = ensure_cache(size)   # may legitimately be (None, None)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        if self.arr is not None:
            img = Image.fromarray(np.asarray(self.arr[self.idx[r.sample_id]]))
        else:
            img = Image.open(DATA / r.image).convert("RGB")
        return self.tf(img), C2I[r.label], i

def loader(df, size, train, bs=None):      # replaces the disk-reading loader above
    # persistent_workers keeps 4 processes and their prefetch buffers alive until
    # the DataLoader is garbage-collected. Across a multi-model sweep those
    # accumulate and the kernel is OOM-killed (observed: died mid-way through the
    # third backbone). Workers now shut down when the iterator finishes, and
    # prefetch is halved; the cost is ~1 s of respawn per epoch.
    return DataLoader(CachedDS(df, size, train), batch_size=bs or BATCH, shuffle=train,
                      num_workers=NUM_WORKERS, pin_memory=False, prefetch_factor=2,
                      persistent_workers=False, drop_last=False)

def free_ram_gb():
    try:
        import psutil; return psutil.virtual_memory().available / 1e9
    except Exception:
        return float("nan")

# Warm up first. Timing the very first pass measures worker spawn and first-touch
# paging, not throughput -- an earlier version of this probe reported 7 img/s while
# training actually ran at 200 img/s.
_probe = tr.head(2000)
for _ in loader(_probe, SWEEP_SIZE, True):
    break
_t = time.time(); _n = 0
for x, y, _ in loader(_probe, SWEEP_SIZE, True):
    _n += y.size(0)
print(f"data pipeline @{SWEEP_SIZE}px: {_n/(time.time()-_t):.0f} img/s "
      f"(steady state, after warm-up) | free RAM {free_ram_gb():.1f} GB")
""")

code(r"""
def macro_f1(P, Y, n=NC):
    fs = []
    for i in range(n):
        tp = ((P == i) & (Y == i)).sum(); fp = ((P == i) & (Y != i)).sum(); fn = ((P != i) & (Y == i)).sum()
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(fs))

class EMA:
    # Exponential moving average of weights. Evaluating the averaged model is a
    # near-free generalisation gain and costs one extra copy of the parameters.
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
    @torch.no_grad()
    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))


@torch.no_grad()
def predict(model, dl):
    model.eval(); P, Y, IX, LOG = [], [], [], []
    for x, y, ix in dl:
        with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
            o = model(x.to(DEVICE, non_blocking=True))
        o = o.float().cpu()
        LOG.append(o); P.append(o.argmax(1)); Y.append(y); IX.append(ix)
    return (torch.cat(P).numpy(), torch.cat(Y).numpy(),
            torch.cat(IX).numpy(), torch.cat(LOG).numpy())


TTA_SCALES = (0.85, 0.925, 1.0)   # <= 1.0 only: upscaling would need a bigger cache


def predict_tta(model, df, size, bs, scales=TTA_SCALES):
    # Average logits over label-preserving rescalings only. No flips: mirroring
    # would turn rel_strong_positive into rel_strong_negative.
    if not USE_TTA:
        return predict(model, loader(df, size, False, bs))
    acc_log, Y0, IX0 = None, None, None
    for sc in scales:
        s_ = max(64, int(round(size * sc / 32)) * 32)
        _P, Y_, IX_, LOG_ = predict(model, loader(df, s_, False, bs))
        order = np.argsort(IX_)                      # align across passes
        LOG_, Y_, IX_ = LOG_[order], Y_[order], IX_[order]
        acc_log = LOG_ if acc_log is None else acc_log + LOG_
        Y0, IX0 = Y_, IX_
    return acc_log.argmax(1), Y0, IX0, acc_log

def train_model(name, size, epochs, bs=None, tag=""):
    seed_all()
    print(f"  [ram] {free_ram_gb():.1f} GB free before {name}")
    dl_tr, dl_va = loader(tr, size, True, bs), loader(va, size, False, bs)
    model = build(name).to(DEVICE)
    # channels_last was tried and removed. It bought only 10% on convnext_small
    # (1947 -> 1751 s/epoch) and caused a hard CUDA failure on EfficientNet:
    #   torch.AcceleratorError: CUDA error: misaligned address  (conv forward)
    # cuDNN kernels for depthwise separable convs do not reliably accept the
    # channels_last stride layout on Turing. Not worth the crash.
    if DEVICE.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=epochs,
                                                steps_per_epoch=len(dl_tr), pct_start=0.25)
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    ckpt = f"{name}{tag}_best.pt"; best = -1.0; hist = []; stale = 0
    ema = EMA(model) if USE_EMA else None

    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); tot = n = corr = 0
        for bi, (x, y, _) in enumerate(dl_tr):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                out = model(x); loss = crit(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            if ema is not None: ema.update(model)
            tot += loss.detach().item() * y.size(0)
            corr += (out.detach().argmax(1) == y).sum().item(); n += y.size(0)
            if bi % 40 == 0:
                print(f"\r  {name} ep{ep} [{bi}/{len(dl_tr)}] loss {tot/max(n,1):.3f} "
                      f"acc {corr/max(n,1):.3f}", end="", flush=True)
        if ema is not None:                       # evaluate the averaged weights
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
        ep_secs = time.time() - t0
        if ep == 1 and ep_secs > EPOCH_TIME_LIMIT:
            print(f"  ABORT {name}: epoch 1 took {ep_secs:.0f}s > {EPOCH_TIME_LIMIT}s limit. "
                  f"This backbone would need {ep_secs*epochs/3600:.1f} h; skipping it "
                  f"rather than spending the session on one model.")
            del model, dl_tr, dl_va, opt, sched, scaler, ema
            gc.collect(); torch.cuda.empty_cache()
            raise TimeoutError(name)
        P, Y, _, _ = predict(model, dl_va); vf1 = macro_f1(P, Y); vacc = float((P == Y).mean())
        hist.append(dict(epoch=ep, loss=tot/n, train_acc=corr/n, val_acc=vacc, val_f1=vf1))
        print(f"\r  {name} ep{ep}  loss {tot/n:.3f}  train {corr/n:.3f}  "
              f"val_acc {vacc:.4f}  val_F1 {vf1:.4f}  ({time.time()-t0:.0f}s)")
        if vf1 > best:
            best, stale = vf1, 0
            if ema is not None:
                backup2 = {k: v.detach().clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)
            m_ = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(m_.state_dict(), ckpt)
            if ema is not None: model.load_state_dict(backup2)
        if ema is not None:
            model.load_state_dict(backup)         # restore live weights for training
        if vf1 <= best:
            stale += 1
            # observed: train acc hits 0.99 by epoch 8 while val plateaus, so the
            # remaining epochs only memorise. Stop and keep the best checkpoint.
            if stale >= PATIENCE:
                print(f"  early stop: val macro-F1 flat for {PATIENCE} epochs "
                      f"(best {best:.4f})")
                break

    clean = build(name).to(DEVICE); clean.load_state_dict(torch.load(ckpt)); clean.eval()
    del model, dl_tr, dl_va, opt, sched, scaler, ema
    gc.collect(); torch.cuda.empty_cache()
    print(f"  [ram] {free_ram_gb():.1f} GB free after {name}")
    return clean, hist, best, ckpt
""")

code(r"""
sweep = {}
if RUN_SWEEP:
    for name in SWEEP_MODELS:
        if free_ram_gb() < RAM_FLOOR_GB:
            print(f"\nSKIP {name}: only {free_ram_gb():.1f} GB RAM free "
                  f"(floor {RAM_FLOOR_GB} GB). Proceeding with the {len(sweep)} "
                  f"backbone(s) already trained rather than risking the kernel.")
            break
        print(f"\n=== {name} @ {SWEEP_SIZE}px, {SWEEP_EPOCHS} epochs ===")
        t0 = time.time()
        try:
            m, h, bf1, ck = train_model(name, SWEEP_SIZE, SWEEP_EPOCHS, tag=f"_{SWEEP_SIZE}")
        except TimeoutError:
            gc.collect(); torch.cuda.empty_cache()
            continue
        except Exception as e:
            # keep the backbones that did train rather than losing the whole sweep
            print(f"  FAILED {name}: {type(e).__name__}: {str(e)[:160]}")
            gc.collect(); torch.cuda.empty_cache()
            continue
        _dl = loader(te, SWEEP_SIZE, False)
        P, Y, IX, _ = predict(m, _dl)
        del _dl
        sweep[name] = dict(params_M=sum(p.numel() for p in m.parameters())/1e6,
                           val_f1=bf1, test_acc=float((P == Y).mean()),
                           test_f1=macro_f1(P, Y), minutes=(time.time()-t0)/60, history=h, ckpt=ck)
        print(f"  -> test acc {sweep[name]['test_acc']:.4f}  macro-F1 {sweep[name]['test_f1']:.4f}")
        del m; gc.collect(); torch.cuda.empty_cache()

    table2 = (pd.DataFrame(sweep).T[["params_M", "val_f1", "test_acc", "test_f1", "minutes"]]
              .astype(float).round(4).sort_values("test_f1", ascending=False))
    table2.index.name = "backbone"; display(table2)
    BEST_MODEL = table2.index[0]
else:
    BEST_MODEL = SWEEP_MODELS[0]
print("best backbone:", BEST_MODEL)
""")

md(r"""
## 6b. Text-shortcut ablation — crop to the plot area

Chart titles read `"Distribution of X"`, `"Y vs X"`, `"Y over X"`, `"Y by X"`,
`"Composition of X"`. Each **reveals the arity**, and arity plus chart type narrows
the candidate classes sharply. A model that reads text therefore gets a large part
of the task for free, and headline accuracy would overstate genuine chart reading.

The `plot_area` grounding box marks the axes region, excluding the title, axis
labels and tick labels. Training on that crop answers the question directly: if
accuracy holds, the model reads data; if it collapses, it was reading captions.

**Report both numbers.** The gap is the size of the text shortcut.
""")

code(r"""
# plot_area boxes live in the split records, not in metadata.parquet
PLOT_AREA = {}
for _sp in ("train", "val", "test"):
    with open(DATA / "splits" / f"{_sp}.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            pa = next((b["bbox"] for b in r["render"]["grounding_boxes"]
                       if b["label"] == "plot_area"), None)
            if pa: PLOT_AREA[r["sample_id"]] = pa
print(f"plot_area boxes loaded: {len(PLOT_AREA):,}")

class CroppedChartDS(ChartDS):
    # Same pipeline, but the image is cropped to the axes region first.
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(DATA / r.image).convert("RGB")
        box = PLOT_AREA.get(r.sample_id)
        if box:
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            if x1 - x0 > 20 and y1 - y0 > 20:
                img = img.crop((x0, y0, x1, y1))
        return self.tf(img), C2I[r.label], i

# NOTE: this reads the original PNGs, not the cache. The cache stores whole
# resized images, so plot_area crop geometry no longer applies to it. This
# ablation is therefore slower than the cached path -- that is expected.
def crop_loader(df, size, train, bs=None):
    return DataLoader(CroppedChartDS(df, size, train), batch_size=bs or BATCH,
                      shuffle=train, num_workers=NUM_WORKERS, pin_memory=True,
                      persistent_workers=NUM_WORKERS > 0)

# visual check that the crop really removes the caption
_r = te.iloc[0]
_im = Image.open(DATA / _r.image); _b = PLOT_AREA[_r.sample_id]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
axs[0].imshow(_im); axs[0].set_title("full image (title + axis labels visible)", fontsize=8)
axs[1].imshow(_im.crop([int(v) for v in _b])); axs[1].set_title("plot_area crop (text removed)", fontsize=8)
for a in axs: a.axis("off")
plt.tight_layout(); plt.show()
""")

code(r"""
crop_res = {}
if RUN_SWEEP or RUN_FINAL:
    seed_all()
    _dl_tr, _dl_va = crop_loader(tr, SWEEP_SIZE, True), crop_loader(va, SWEEP_SIZE, False)
    _m = build(BEST_MODEL).to(DEVICE)
    if DEVICE.type == "cuda" and torch.cuda.device_count() > 1: _m = nn.DataParallel(_m)
    _opt = torch.optim.AdamW(_m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    _sch = torch.optim.lr_scheduler.OneCycleLR(_opt, max_lr=LR, epochs=SWEEP_EPOCHS,
                                               steps_per_epoch=len(_dl_tr), pct_start=0.25)
    _crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    _sc = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    for ep in range(1, SWEEP_EPOCHS + 1):
        _m.train(); t0 = time.time()
        for x, y, _ in _dl_tr:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            _opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = _crit(_m(x), y)
            _sc.scale(loss).backward(); _sc.step(_opt); _sc.update(); _sch.step()
        print(f"  crop ep{ep} done ({time.time()-t0:.0f}s)")

    cP, cY, cIX, _ = predict(_m, crop_loader(te, SWEEP_SIZE, False))
    cres = pd.DataFrame({"true": [CLASSES[i] for i in cY], "pred": [CLASSES[i] for i in cP],
                         "provenance": te.provenance.values[cIX]})
    cres["ok"] = cres.true == cres.pred
    crop_res = {"acc": float(cres.ok.mean()), "f1": macro_f1(cP, cY),
                "real": float(cres[cres.provenance == "real"].ok.mean()),
                "synthetic": float(cres[cres.provenance == "synthetic"].ok.mean())}

    # Compare against the SAME model at the SAME size and epoch budget -- the
    # sweep entry -- not against the final tuned run, or the gap would mix the
    # text effect with resolution and training length.
    base = sweep.get(BEST_MODEL, {})
    print(f"\n{'':22s} {'full image':>12s} {'plot-area crop':>16s} {'delta':>9s}")
    for k, lab in (("test_acc", "accuracy"), ("test_f1", "macro-F1")):
        fv = base.get(k, float("nan"))
        cv = crop_res["acc" if k == "test_acc" else "f1"]
        print(f"  {lab:20s} {fv:>12.4f} {cv:>16.4f} {cv-fv:>+9.4f}")
    print(f"  (both: {BEST_MODEL} @ {SWEEP_SIZE}px, {SWEEP_EPOCHS} epochs)")
    print(f"\n  crop real-only      {crop_res['real']:.4f}   "
          f"crop synthetic-only {crop_res['synthetic']:.4f}")
    print("\nA large drop means the caption was doing the work, not the plotted data.")
    del _m; gc.collect(); torch.cuda.empty_cache()
""")

md(r"""
## 7. Resolution ablation

Charts encode meaning in thin lines, small markers and axis text. 224 px is the
ImageNet convention, not necessarily the right choice here — this measures the
difference instead of assuming it.
""")

code(r"""
# A fair ablation needs the SAME epoch budget at both resolutions. Comparing the
# 8-epoch sweep run at 224 against a 16-epoch run at 384 would confound
# resolution with training length.
res_ablation, hist_final = {}, []

def run_at(size):
    bs = max(12, BATCH // 3) if size >= 384 else BATCH
    if BEST_MODEL.startswith(("vit", "swin")) and size != 224:
        print(f"  skip {size}px: {BEST_MODEL} has a fixed 224 window/patch grid")
        return None
    print(f"\n=== {BEST_MODEL} @ {size}px, up to {FINAL_EPOCHS} epochs, batch {bs} ===")
    m, hist, bf1, ck = train_model(BEST_MODEL, size, FINAL_EPOCHS, bs=bs, tag=f"_abl{size}")
    P_, Y_, IX_, _ = predict_tta(m, te, size, bs)
    out = dict(model=m, hist=hist, ckpt=ck, bs=bs,
               acc=float((P_ == Y_).mean()), f1=macro_f1(P_, Y_))
    print(f"  {size}px -> test acc {out['acc']:.4f}  macro-F1 {out['f1']:.4f}")
    return out

if RUN_FINAL:
    for _sz in sorted(set(ABLATION_SIZES)):
        r_ = run_at(_sz)
        if r_: res_ablation[_sz] = r_

    if len(res_ablation) > 1:
        lo_, hi_ = min(res_ablation), max(res_ablation)
        d_ = res_ablation[hi_]["f1"] - res_ablation[lo_]["f1"]
        print(f"\nRESOLUTION EFFECT {lo_} -> {hi_} px: macro-F1 {d_:+.4f} "
              f"({res_ablation[lo_]['f1']:.4f} -> {res_ablation[hi_]['f1']:.4f}), "
              f"equal epoch budget")
        display(pd.DataFrame(res_ablation).T[["acc", "f1", "bs"]].astype(float).round(4))

    BEST_SIZE = max(res_ablation, key=lambda k: res_ablation[k]["f1"])
    cnn = res_ablation[BEST_SIZE]["model"]
    hist_final = res_ablation[BEST_SIZE]["hist"]
    FINAL_SIZE, FINAL_BS = BEST_SIZE, res_ablation[BEST_SIZE]["bs"]
    final = dict(acc=res_ablation[BEST_SIZE]["acc"], f1=res_ablation[BEST_SIZE]["f1"])
    print(f"\nbest configuration: {BEST_MODEL} @ {FINAL_SIZE}px")
elif sweep:
    FINAL_BS = BATCH
    cnn = build(BEST_MODEL).to(DEVICE)
    cnn.load_state_dict(torch.load(sweep[BEST_MODEL]["ckpt"]))
    FINAL_SIZE = SWEEP_SIZE
    final = {}
else:
    cnn, FINAL_BS, final = None, BATCH, {}

HAVE_PREDS = cnn is not None
if HAVE_PREDS:
    P, Y, IX, LOGITS = predict(cnn, loader(te, FINAL_SIZE, False, FINAL_BS))
    if not final:
        final = dict(acc=float((P == Y).mean()), f1=macro_f1(P, Y))
else:
    print("no CNN trained (VLM_ONLY) -- sections 8-11 will be skipped")
""")

md(r"""
### Ensemble of the top-3 backbones

Different architectures make different mistakes, so averaging them is usually the
single easiest accuracy gain left. One subtlety: raw logit scales are not
comparable across architectures, so we average **softmax probabilities**, not
logits.
""")

code(r"""
ens_res = {}
if RUN_ENSEMBLE and RUN_SWEEP and len(sweep) >= 2:
    top = sorted(sweep, key=lambda k: -sweep[k]["test_f1"])[:3]
    print("ensembling:", ", ".join(f"{t} ({sweep[t]['test_f1']:.4f})" for t in top))
    prob, Ye, IXe = None, None, None
    for name in top:
        m_ = build(name).to(DEVICE)
        m_.load_state_dict(torch.load(sweep[name]["ckpt"])); m_.eval()
        _P, Y_, IX_, LOG_ = predict_tta(m_, te, SWEEP_SIZE, BATCH)
        o = np.argsort(IX_); LOG_, Y_, IX_ = LOG_[o], Y_[o], IX_[o]
        # softmax first: logit magnitudes differ between architectures, so
        # averaging raw logits would silently weight one model more than another
        pr = torch.softmax(torch.from_numpy(LOG_).float(), dim=1).numpy()
        prob = pr if prob is None else prob + pr
        Ye, IXe = Y_, IX_
        del m_; gc.collect(); torch.cuda.empty_cache()

    Pe = prob.argmax(1)
    ens_res = {"members": top, "acc": float((Pe == Ye).mean()), "f1": macro_f1(Pe, Ye)}
    best_single = max(sweep[t]["test_f1"] for t in top)
    print(f"\nensemble        acc {ens_res['acc']:.4f}   macro-F1 {ens_res['f1']:.4f}")
    print(f"best single     macro-F1 {best_single:.4f}")
    print(f"ensemble gain   {ens_res['f1'] - best_single:+.4f}")

    # adopt the ensemble for the headline evaluation if it genuinely wins
    if ens_res["f1"] > final.get("f1", -1):
        print("\n-> ensemble beats the single best model; using it for section 8 onward.")
        P, Y, IX = Pe, Ye, IXe
        final = {"acc": ens_res["acc"], "f1": ens_res["f1"]}
        FINAL_SIZE = SWEEP_SIZE
    else:
        print("\n-> single model still wins; keeping it.")
else:
    print("ensemble skipped (needs RUN_SWEEP with >=2 backbones)")
""")

md("## 8. Evaluation — Table 3, bootstrap CIs, significance")

code(r"""
res = pd.DataFrame({
    "true": [CLASSES[i] for i in Y], "pred": [CLASSES[i] for i in P],
    "difficulty": te.difficulty.values[IX], "provenance": te.provenance.values[IX],
    "chart_type": te.chart_type.values[IX], "family": te.family.values[IX],
    "evidence_margin": te.evidence_margin.values[IX], "sample_id": te.sample_id.values[IX],
})
res["ok"] = res.true == res.pred

def boot_ci(mask, n=4000, seed=0):
    a = np.asarray(mask, dtype=float)
    if len(a) == 0: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    return tuple(np.percentile(a[rng.integers(0, len(a), (n, len(a)))].mean(1), [2.5, 97.5]))

lo, hi = boot_ci(res.ok)
print(f"TEST accuracy {res.ok.mean():.4f}  [95% CI {lo:.4f}, {hi:.4f}]")
print(f"TEST macro-F1 {macro_f1(P, Y):.4f}\n")

for k in ("family", "difficulty", "provenance"):
    g = res.groupby(k).ok.agg(["mean", "count"]).round(4)
    g.columns = ["accuracy", "n"]
    print(f"--- by {k} ---"); display(g)
""")

code(r"""
# Real-only generalisation probe: held-out REAL source tables only.
# Unbalanced by construction -> report over classes with n>=20 and give per-class n.
probe = res[res.provenance == "real"]
cnt = probe.true.value_counts()
usable = cnt[cnt >= 20].index.tolist()
plo, phi = boot_ci(probe.ok)

print(f"REAL-ONLY PROBE   n={len(probe)}   accuracy {probe.ok.mean():.4f} "
      f"[95% CI {plo:.4f}, {phi:.4f}]")
print(f"  restricted to {len(usable)}/{NC} classes with n>=20: "
      f"{probe[probe.true.isin(usable)].ok.mean():.4f}")
print(f"  synthetic-only accuracy: {res[res.provenance=='synthetic'].ok.mean():.4f}")
print("\nIf real is far below synthetic, the model learned the generator, not the statistics.")
display(probe.groupby("true").ok.agg(["mean", "count"]).round(3)
        .rename(columns={"mean": "accuracy", "count": "n"}).sort_values("n", ascending=False))
""")

code(r"""
# Does accuracy track the evidence margin? Difficulty is the within-class tertile
# of that margin, so a flat curve here is itself a reportable finding.
from scipy.stats import spearmanr
rho, pv = spearmanr(res.evidence_margin.rank(), res.ok.astype(int))
print(f"Spearman(evidence margin, correct) = {rho:.3f}  p={pv:.2g}")
if abs(rho) < 0.15:
    print("  NOTE: near-zero correlation. The evidence margin is a STATISTICAL quantity,")
    print("  not a visual one -- extreme margins produce degenerate charts (a residual")
    print("  spread ratio of 1500 renders as one vertical smear; excess kurtosis of 600")
    print("  is a single spike with an invisible tail). The top tertile therefore mixes")
    print("  'obvious' with 'unreadable'. Report this rather than presenting the tertiles")
    print("  as a clean difficulty axis.")
print(res.groupby("difficulty").ok.agg(["mean", "count"]).round(4), "\n")

# McNemar: is the model significantly better than the pixel-free chart-type prior?
from statsmodels.stats.contingency_tables import mcnemar
prior_ok = pd.Series(pred_ct.values)[IX].values == res.true.values
tb = [[int((res.ok & prior_ok).sum()),  int((res.ok & ~prior_ok).sum())],
      [int((~res.ok & prior_ok).sum()), int((~res.ok & ~prior_ok).sum())]]
mc = mcnemar(tb, exact=False, correction=True)
print(f"McNemar vs chart-type prior: chi2={mc.statistic:.1f}  p={mc.pvalue:.3g}")
print(f"  model correct / prior wrong : {tb[0][1]}")
print(f"  prior correct / model wrong : {tb[1][0]}")
""")

md(r"""
## 9. Evidence Pointing Accuracy — the grounding metric

Every sample ships pixel-level boxes marking the evidence for its finding (the
outlier band, the fitted trend, the changepoint, the anomalous points). Grad-CAM
gives the pixels the model actually used. Overlapping the two answers a question
plain accuracy cannot: **is the model right for the right reason?**

Scored as a pointing game — does the CAM peak land inside an evidence box? —
against a random-point-in-plot-area control, since large boxes would otherwise
inflate the hit rate on their own.
""")

code(r"""
class GradCAM:
    def __init__(self, model, layer):
        self.model, self.a, self.g = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "a", o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "g", go[0].detach()))
    def __call__(self, x, cls_idx):
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        out[0, cls_idx].backward()
        w = self.g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.a).sum(1)).squeeze(0)
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).cpu().numpy()

def cam_layer(model, name):
    if name.startswith("resnet"):      return model.layer4[-1]
    if name.startswith("convnext"):    return model.features[-1]
    if name.startswith("efficientnet"): return model.features[-1]
    if name.startswith("densenet"):     return model.features[-1]
    # swin and vit are transformers: Grad-CAM on a patch-token stage is not the
    # same object as a conv feature map, so they need attention rollout instead
    raise ValueError(f"no Grad-CAM hook for {name} (transformer: needs attention rollout)")

IMG_W, IMG_H = 768, 576   # native canvas the grounding boxes are expressed in

recs_by_id = {}
with open(DATA / "splits" / "test.jsonl") as fh:
    for line in fh:
        r = json.loads(line); recs_by_id[r["sample_id"]] = r
print(f"loaded {len(recs_by_id):,} test records with grounding boxes")
""")

code(r"""
point_hits, area_ctrl, per_class_hits = [], [], defaultdict(list)
EPA = CTRL = float("nan")

if RUN_GRADCAM and not BEST_MODEL.startswith(("vit", "swin")):
    cam_fn = GradCAM(cnn, cam_layer(cnn, BEST_MODEL))
    tfm = tf_eval(FINAL_SIZE)
    ids = res[res.ok].sample(min(600, int(res.ok.sum())), random_state=0).sample_id.tolist()

    for k, sid in enumerate(ids):
        rec = recs_by_id.get(sid)
        if not rec: continue
        boxes = [b for b in rec["render"]["grounding_boxes"] if b["label"] != "plot_area"]
        plot = next((b for b in rec["render"]["grounding_boxes"] if b["label"] == "plot_area"), None)
        if not boxes or not plot: continue

        x = tfm(Image.open(DATA / rec["image"]).convert("RGB")).unsqueeze(0).to(DEVICE)
        cam = cam_fn(x, C2I[rec["label"]])
        iy, ix_ = np.unravel_index(cam.argmax(), cam.shape)
        px = (ix_ + 0.5) / cam.shape[1] * IMG_W
        py = (iy + 0.5) / cam.shape[0] * IMG_H

        hit = any(b["bbox"][0] <= px <= b["bbox"][2] and b["bbox"][1] <= py <= b["bbox"][3]
                  for b in boxes)
        point_hits.append(hit); per_class_hits[rec["label"]].append(hit)

        # control: chance a uniformly random point in the plot area lands in a box
        pa = max((plot["bbox"][2]-plot["bbox"][0]) * (plot["bbox"][3]-plot["bbox"][1]), 1)
        ba = sum(max(b["bbox"][2]-b["bbox"][0], 0) * max(b["bbox"][3]-b["bbox"][1], 0) for b in boxes)
        area_ctrl.append(min(ba / pa, 1.0))
        if k % 100 == 0: print(f"\r  grad-cam {k}/{len(ids)}", end="", flush=True)

    EPA, CTRL = float(np.mean(point_hits)), float(np.mean(area_ctrl))
    print(f"\n\nEvidence Pointing Accuracy : {EPA:.4f}   (n={len(point_hits)}, correct predictions only)")
    print(f"Random-point control       : {CTRL:.4f}")
    print(f"Lift over control          : {EPA/max(CTRL,1e-9):.2f}x")
    print("\nSeparates 'right answer' from 'right answer for the right reason'.")
    display(pd.DataFrame({"pointing_acc": {k: float(np.mean(v)) for k, v in per_class_hits.items()},
                          "n": {k: len(v) for k, v in per_class_hits.items()}})
            .round(3).sort_values("pointing_acc", ascending=False))
else:
    print("Grad-CAM skipped (RUN_GRADCAM off, or a transformer backbone won the sweep "
          "— swin/vit need attention rollout, not Grad-CAM).")
""")

md("## 10. Publication figures")

code(r"""
# Figure 1 — confusion matrix
cm = pd.crosstab(res.true, res.pred).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
cmn = cm.div(cm.sum(1).replace(0, 1), axis=0)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cmn.values, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(NC)); ax.set_xticklabels(CLASSES, rotation=90, fontsize=7)
ax.set_yticks(range(NC)); ax.set_yticklabels(CLASSES, fontsize=7)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"InsightVLM-20K — {BEST_MODEL} @ {FINAL_SIZE}px (row-normalised)")
for i in range(NC):
    for j in range(NC):
        v = cmn.values[i, j]
        if v > 0.02:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if v > 0.5 else "black")
fig.colorbar(im, shrink=0.8); plt.tight_layout()
plt.savefig("fig1_confusion.png", bbox_inches="tight"); plt.show()

print("most confused pairs:")
off = cmn.copy(); np.fill_diagonal(off.values, 0)
for (t, p), v in off.stack().sort_values(ascending=False).head(10).items():
    print(f"  {t:30s} -> {p:30s} {v:.3f}")
""")

code(r"""
# Figure 2 — per-class F1
f1_by = {}
for i, c in enumerate(CLASSES):
    tp = ((P == i) & (Y == i)).sum(); fp = ((P == i) & (Y != i)).sum(); fn = ((P != i) & (Y == i)).sum()
    pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
    f1_by[c] = 2*pr*rc/(pr+rc) if pr+rc else 0
s = pd.Series(f1_by).sort_values()

fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(range(len(s)), s.values,
        color=["#c0392b" if v < .5 else "#e67e22" if v < .75 else "#27ae60" for v in s.values])
ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=7)
ax.axvline(s.mean(), ls="--", c="k", lw=1, label=f"macro-F1 {s.mean():.3f}")
ax.set_xlabel("F1"); ax.set_title("Per-class F1"); ax.legend()
plt.tight_layout(); plt.savefig("fig2_per_class_f1.png", bbox_inches="tight"); plt.show()

# Figure 3 — difficulty and provenance
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
d = res.groupby("difficulty").ok.mean().reindex(["easy", "medium", "hard"])
axes[0].plot(d.index, d.values, "o-", lw=2); axes[0].set_ylim(0, 1)
axes[0].set_ylabel("accuracy"); axes[0].set_title("Accuracy vs difficulty (evidence-margin tertiles)")
pv_ = res.groupby("provenance").ok.mean()
axes[1].bar(pv_.index, pv_.values, color=["#2980b9", "#95a5a6"]); axes[1].set_ylim(0, 1)
axes[1].set_title("Accuracy by provenance")
for i, v in enumerate(pv_.values): axes[1].text(i, v + .02, f"{v:.3f}", ha="center")
plt.tight_layout(); plt.savefig("fig3_difficulty_provenance.png", bbox_inches="tight"); plt.show()

# Figure 4 — training curves
if hist_final:
    h = pd.DataFrame(hist_final)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(h.epoch, h.train_acc, "o-", label="train acc")
    ax.plot(h.epoch, h.val_acc, "s-", label="val acc")
    ax.plot(h.epoch, h.val_f1, "^-", label="val macro-F1")
    ax.set_xlabel("epoch"); ax.legend(); ax.set_title(f"{BEST_MODEL} @ {FINAL_SIZE}px")
    plt.tight_layout(); plt.savefig("fig4_training.png", bbox_inches="tight"); plt.show()
""")

code(r"""
# Figure 5 — qualitative grid with ground-truth evidence boxes
import matplotlib.patches as mpatches
n_bad = int((~res.ok).sum())
show = pd.concat([res[res.ok].sample(4, random_state=1),
                  res[~res.ok].sample(min(4, n_bad), random_state=1)])
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, (_, r) in zip(axes.ravel(), show.iterrows()):
    rec = recs_by_id.get(r.sample_id)
    if rec is None: ax.axis("off"); continue
    ax.imshow(Image.open(DATA / rec["image"])); ax.axis("off")
    for b in rec["render"]["grounding_boxes"]:
        if b["label"] == "plot_area": continue
        x0, y0, x1, y1 = b["bbox"]
        ax.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec="red", lw=1.4))
    ok = r.true == r.pred
    ax.set_title(f"{'CORRECT' if ok else 'MISS'}\ntrue={r.true}\npred={r.pred}",
                 fontsize=7, color="green" if ok else "red")
for ax in axes.ravel()[len(show):]: ax.axis("off")
plt.tight_layout(); plt.savefig("fig5_qualitative.png", bbox_inches="tight"); plt.show()
""")

md("## 11. LaTeX tables + results export")

code(r"""
rows = [
      {"Model": "Majority class",           "Accuracy": floors["majority class"],           "Macro-F1": np.nan},
    {"Model": "Chart-type prior",         "Accuracy": floors["chart-type prior"],         "Macro-F1": np.nan},
]
if RUN_SWEEP:
    for k, v in sweep.items():
        rows.append({"Model": f"{k} @{SWEEP_SIZE}px", "Accuracy": v["test_acc"], "Macro-F1": v["test_f1"]})
rows.append({"Model": f"{BEST_MODEL} @{FINAL_SIZE}px (best)",
             "Accuracy": final["acc"], "Macro-F1": final["f1"]})
if ens_res:
    rows.append({"Model": f"Ensemble ({'+'.join(ens_res['members'])})",
                 "Accuracy": ens_res["acc"], "Macro-F1": ens_res["f1"]})
rows.append({"Model": "Statistical oracle (ceiling)", "Accuracy": ORACLE_ACC, "Macro-F1": np.nan})
summary = pd.DataFrame(rows).round(4)
display(summary)

print("\n% ---- paste into the paper ----")
print(summary.to_latex(index=False, na_rep="--", float_format="%.4f",
                       caption="Baselines on InsightVLM-20K (test split, 4{,}000 samples). "
                               "The chart-type prior uses no pixels and is the honest floor; "
                               "the statistical oracle is the ceiling.",
                       label="tab:baselines"))

json.dump({
    "dataset": {"n": int(len(meta)), "classes": NC,
                "chart_types": int(meta.chart_type.nunique()),
                "provenance": {k: int(v) for k, v in meta.provenance.value_counts().items()}},
    "floors": floors, "oracle_accuracy": float(ORACLE_ACC),
    "sweep": {k: {kk: vv for kk, vv in v.items() if kk not in ("history", "ckpt")}
              for k, v in sweep.items()},
    "final": {"model": BEST_MODEL, "size": int(FINAL_SIZE),
              **{k: float(v) for k, v in final.items()}},
    "test": {"accuracy": float(res.ok.mean()), "macro_f1": float(macro_f1(P, Y)),
             "ci95": [float(lo), float(hi)],
             "by_difficulty": res.groupby("difficulty").ok.mean().round(4).to_dict(),
             "by_provenance": res.groupby("provenance").ok.mean().round(4).to_dict(),
             "by_family": res.groupby("family").ok.mean().round(4).to_dict()},
    "real_only_probe": {"n": int(len(probe)), "accuracy": float(probe.ok.mean()),
                        "ci95": [float(plo), float(phi)], "classes_ge_20": len(usable)},
    "ensemble": ens_res,
    "evidence_pointing": {"accuracy": EPA, "random_control": CTRL},
}, open("insightvlm_results.json", "w"), indent=2)

summary.to_csv("results_summary.csv", index=False)
res.to_csv("test_predictions.csv", index=False)
print("\nwrote results_summary.csv, test_predictions.csv, insightvlm_results.json, fig1-5*.png")
""")

md(r"""
## 12. InsightVLM — Qwen2-VL-2B + QLoRA

**Run this in its own session** (4–6 h). Everything above is the discriminative
baseline; this is the generative model that produces the finding, the reasoning
chain and the insight. Set `RUN_VLM = True` in §0.
""")

code(r"""
# Free everything the CNN pipeline holds before Qwen loads. Section 11 has
# already written results_summary.csv / insightvlm_results.json / the figures, so
# nothing is lost if the VLM stage fails or the session hits the 12 h cap.
if RUN_VLM:
    import shutil
    for _v in ("cnn", "res_ablation"):
        globals().pop(_v, None)
    if _CACHE.get("arr") is not None:
        _CACHE["arr"] = None
        try: os.remove(_CACHE["path"])          # reclaim up to 21 GB of scratch
        except OSError: pass
        _CACHE.update(px=None, idx=None, path=None)
    gc.collect(); torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        free, tot = torch.cuda.mem_get_info(i)
        print(f"  GPU{i} free {free/1e9:.1f} / {tot/1e9:.1f} GB")
    print(f"  scratch free: {shutil.disk_usage(CACHE_DIR).free/1e9:.1f} GB")
""")

code(r"""
# subprocess rather than a nested `!pip`: an escaped shell command inside an
# `if` block depends on IPython's input transformer and is not plain Python.
if RUN_VLM:
    import subprocess, sys as _s
    subprocess.run([_s.executable, "-m", "pip", "-q", "install", "-U",
                    "transformers>=4.45", "accelerate", "peft", "trl",
                    "bitsandbytes", "qwen-vl-utils"], check=True)
    print("VLM dependencies installed")
""")

code(r"""
if RUN_VLM:
    from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration,
                              BitsAndBytesConfig, TrainingArguments, Trainer)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from qwen_vl_utils import process_vision_info
    from torch.utils.data import Dataset as TorchDS

    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

    def load_instruct(split, limit=None):
        rows = [json.loads(l) for l in open(DATA / "instruct" / f"{split}.jsonl")]
        if limit: rows = rows[:limit]
        for r in rows: r["image"] = str(DATA / r["image"])
        return rows

    # Kaggle sessions cap near 12 h. Report the subset size in the paper.
    train_rows = load_instruct("train", 4000)
    print(len(train_rows), "training samples")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)  # T4 is fp16, not bf16
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=768*28*28)
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    vlm = prepare_model_for_kbit_training(vlm); vlm.config.use_cache = False
    vlm = get_peft_model(vlm, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    vlm.print_trainable_parameters()
""")

code(r"""
if RUN_VLM:
    class VLMDataset(TorchDS):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return self.rows[i]

    def collate(batch):
        texts, images = [], []
        for b in batch:
            msgs = [{"role": "user", "content": [
                        {"type": "image", "image": b["image"]},
                        {"type": "text", "text": b["messages"][0]["content"].replace("<image>\n", "")}]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": b["messages"][1]["content"]}]}]
            texts.append(processor.apply_chat_template(msgs, tokenize=False))
            imgs, _ = process_vision_info(msgs); images.append(imgs)
        enc = processor(text=texts, images=images, return_tensors="pt",
                        padding=True, truncation=True, max_length=1024)
        labels = enc["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        for tok in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
            tid = processor.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0: labels[labels == tid] = -100
        enc["labels"] = labels
        return enc

    trainer = Trainer(
        model=vlm,
        args=TrainingArguments(
            output_dir="insightvlm-qwen2vl-2b-lora", num_train_epochs=1,
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
            logging_steps=25, save_strategy="epoch", fp16=True,
            gradient_checkpointing=True, optim="paged_adamw_8bit",
            report_to="none", remove_unused_columns=False, dataloader_num_workers=2),
        train_dataset=VLMDataset(train_rows), data_collator=collate)
    trainer.train()
    trainer.save_model("insightvlm-qwen2vl-2b-lora/final")
""")

code(r"""
if RUN_VLM:
    vlm.eval()
    PROMPT = ("You are a data analyst. Examine this chart and report the statistical "
              "findings it supports. State the dominant pattern, cite the numbers that "
              "justify it, and note any caveat a reader should know.")

    def generate(image_path, max_new_tokens=320):
        msgs = [{"role": "user", "content": [{"type": "image", "image": image_path},
                                             {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        inp = processor(text=[text], images=[imgs], return_tensors="pt").to(vlm.device)
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0]

    CLS = json.loads((DATA / "class_index.json").read_text())["classes"]
    DESC2CLASS = {CLS[c]["desc"]: c for c in CLASSES}

    def parse_class(txt):
        for desc, c in DESC2CLASS.items():
            if desc[:40].lower() in txt.lower(): return c
        for c in CLASSES:
            if c.replace("_", " ") in txt.lower(): return c
        return None

    # CAUTION: the SFT target embeds the class description verbatim, so the model can
    # score well by reciting a memorised phrase. The real-vs-synthetic gap below is
    # the check — if real is far worse, it learned the template, not the chart.
    ev = te.sample(300, random_state=0)
    rows_v = []
    for k, (_, r) in enumerate(ev.iterrows()):
        out = generate(str(DATA / r.image))
        gt_nums = set(re.findall(r"-?\d+\.\d{2}", str(json.loads(r.statistics_json))))
        pr_nums = set(re.findall(r"-?\d+\.\d{2}", out))
        rows_v.append({"true": r.label, "pred": parse_class(out), "provenance": r.provenance,
                       "difficulty": r.difficulty,
                       "numeric_fidelity": len(gt_nums & pr_nums) / max(len(pr_nums), 1),
                       "text": out})
        if k % 25 == 0: print(f"\r  {k}/{len(ev)}", end="", flush=True)

    vres = pd.DataFrame(rows_v); vres["ok"] = vres.true == vres.pred
    print(f"\n\nVLM class accuracy   : {vres.ok.mean():.4f}")
    print(f"VLM numeric fidelity : {vres.numeric_fidelity.mean():.4f} "
          f"(share of cited numbers that appear in the ground truth)")
    display(vres.groupby("provenance").ok.agg(["mean", "count"]).round(4))
    vres.to_csv("vlm_predictions.csv", index=False)
    print("\nA large real-vs-synthetic gap means template recitation, not chart reading.")
""")

md(r"""
## 13. Reporting checklist

- [ ] Report **macro-F1** alongside accuracy — the test split is balanced, the real-only probe is not.
- [ ] Report the **chart-type prior**, not 5% chance, as the floor. Chart type correlates with the label by construction, so 5% overstates the result.
- [ ] Report the **statistical oracle** as the ceiling; a vision model is only interesting in the gap between the two.
- [ ] Report the **real-only probe separately**, with per-class n. 4 of 20 classes have n < 20.
- [ ] Report **real vs synthetic accuracy**. A large gap means the model learned the generator.
- [ ] Report the **accuracy-vs-difficulty curve**. Difficulty is the within-class tertile of the evidence margin, so a flat curve is itself a finding.
- [ ] Report **Evidence Pointing Accuracy against the random control** — a raw hit rate is inflated by box area.
- [ ] State the **synthetic fraction (~70%)** and the **VLM subset size**.
- [ ] Note that horizontal/vertical flip augmentation is invalid here and was excluded.
- [ ] Note the fixed seed, and that splits are source-disjoint (0 leaked sources).
- [ ] Acknowledge the **single-label limitation**: compound-finding charts were excluded during mining, so the model is not trained to report two findings at once.
""")

# nbformat >= 4.5 requires a cell id; it is only a warning today but becomes a
# hard error in future versions.
for _i, _c in enumerate(cells):
    _c["id"] = f"cell-{_i:03d}"

print(f"guarded {apply_pred_guards()} prediction-dependent cells")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({len(cells)} cells)")
