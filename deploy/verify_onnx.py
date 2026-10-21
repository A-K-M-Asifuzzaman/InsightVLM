"""verify_onnx.py — The ONNX ensemble must reproduce the evaluated predictions."""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_ensemble import DATA, tf_for  # noqa: E402

cfg = json.loads((ROOT / "results" / "final_ensemble.json").read_text())
sess = ort.InferenceSession(str(ROOT / "deploy/web/insightvlm_ensemble.onnx"))
te = pd.read_parquet(DATA / "tables/metadata.parquet").query("split == 'test'").reset_index(drop=True)
ref = sum(np.load(ROOT / f"results/probs/{m['name']}.npz")["test"] for m in cfg["members"]) / len(cfg["members"])
tf = tf_for(384)
pred = np.zeros(len(te), int)
for i, p in enumerate(te.image):
    x = tf(Image.open(DATA / p).convert("RGB")).unsqueeze(0).numpy()
    pred[i] = sess.run(None, {"image": x})[0].mean(0).argmax()
y = te.label.map({c: i for i, c in enumerate(cfg["classes"])}).values
real = (te.provenance == "real").values
print(f"agreement with evaluated predictions: {(pred == ref.argmax(1)).mean():.4f}")
print(f"ONNX real {np.mean(pred[real] == y[real]):.4f}  synthetic {np.mean(pred[~real] == y[~real]):.4f}")
