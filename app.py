"""
app.py — Interactive tester for the InsightVLM chart classifier.

    python train/final_ensemble.py   # once: picks the ensemble, writes results/final_ensemble.json
    python app.py                    # opens http://127.0.0.1:7860

Upload any chart, or draw a random held-out test chart (real or synthetic) to see
the prediction next to the verified ground truth.
"""
import json
import random
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "train"))
from build_ensemble import DEV, build, tf_for  # noqa: E402

CFG = json.loads((ROOT / "results" / "final_ensemble.json").read_text())
CLASSES = CFG["classes"]
INFO = json.loads((ROOT / "dataset" / "class_index.json").read_text())["classes"]
SHIFT = np.exp(CFG["alpha"] * np.array(CFG["log_prior_shift"]))

NAMES = {
    "dist_normal": "Normal distribution", "dist_right_skew": "Right-skewed distribution",
    "dist_left_skew": "Left-skewed distribution", "dist_bimodal": "Bimodal distribution",
    "dist_heavy_tail": "Heavy-tailed distribution", "dist_uniform": "Uniform distribution",
    "rel_strong_positive": "Strong positive correlation", "rel_strong_negative": "Strong negative correlation",
    "rel_none": "No relationship", "rel_nonlinear_monotonic": "Non-linear monotonic relationship",
    "rel_nonlinear_nonmonotonic": "U-shaped / inverted-U relationship",
    "rel_heteroscedastic": "Heteroscedastic (fan-shaped) relationship",
    "ts_increasing": "Increasing trend", "ts_decreasing": "Decreasing trend",
    "ts_seasonal": "Seasonal pattern", "ts_changepoint": "Level shift (change-point)",
    "ts_anomaly_spike": "Anomaly spike", "grp_significant_diff": "Groups differ significantly",
    "grp_no_diff": "No difference between groups", "grp_dominant_category": "One dominant category",
}

MODELS = []
for m in CFG["members"]:
    net = build(m["arch"], len(CLASSES))
    net.load_state_dict(torch.load(ROOT / m["ckpt"], map_location="cpu"))
    MODELS.append((m["name"], net.to(DEV).eval(), tf_for(m["size"]), m["weight"]))

meta = pd.read_parquet(ROOT / "dataset" / "tables" / "metadata.parquet")
TEST = meta[meta.split == "test"].reset_index(drop=True)


def to_rgb(img):
    # transparent PNGs would otherwise turn black where alpha is 0
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, "white")
    return Image.alpha_composite(bg, img).convert("RGB")


@torch.no_grad()
def classify(img):
    img = to_rgb(img)
    per_model, total = {}, 0
    for name, net, tf, w in MODELS:
        p = torch.softmax(net(tf(img).unsqueeze(0).to(DEV)).float(), 1)[0].cpu().numpy()
        per_model[name] = int(p.argmax())
        total = total + w * p
    p = total / sum(w for *_, w in MODELS) * SHIFT
    return p / p.sum(), per_model


def card(p, per_model, truth=None):
    k = int(p.argmax())
    c = CLASSES[k]
    agree = sum(v == k for v in per_model.values())
    lines = [f"## {NAMES[c]}",
             f"**Confidence** {p[k]:.1%} &nbsp;·&nbsp; **Family** {INFO[c]['family']} "
             f"&nbsp;·&nbsp; **Models agreeing** {agree}/{len(per_model)}",
             "", f"> {INFO[c]['desc']}", "",
             f"**Statistical test behind this label:** {INFO[c]['test']}"]
    if p[k] < 0.5:
        lines += ["", "⚠️ Low confidence: the chart may show a mix of patterns, or one "
                  "outside the 20 classes the model knows."]
    if truth is not None:
        ok = truth.label == c
        lines += ["", "---", f"### {'✅ Correct' if ok else '❌ Wrong'}",
                  f"**Ground truth** {NAMES[truth.label]} &nbsp;·&nbsp; "
                  f"**Data** {truth.provenance} ({truth.source}) &nbsp;·&nbsp; "
                  f"**Chart** {truth.chart_type} &nbsp;·&nbsp; **Difficulty** {truth.difficulty}",
                  "", f"*Verified insight:* {truth.insight}"]
    return "\n".join(lines)


def run_upload(img):
    if img is None:
        return None, ""
    p, per_model = classify(img)
    return {NAMES[c]: float(v) for c, v in zip(CLASSES, p)}, card(p, per_model)


def run_random(provenance):
    r = TEST[TEST.provenance == provenance].iloc[random.randrange((TEST.provenance == provenance).sum())]
    img = Image.open(ROOT / "dataset" / r.image)
    p, per_model = classify(img)
    return img, {NAMES[c]: float(v) for c, v in zip(CLASSES, p)}, card(p, per_model, r)


def perf_md():
    t = CFG["test"]
    members = ", ".join(f"{m['name']} ×{m['weight']}" for m in CFG["members"])
    rows = [f"| **This app's ensemble** | **{t['real_only']:.1%}** | {t['synthetic_only']:.1%} |"]
    sr = ROOT / "results" / "stats_report.json"
    for r in json.loads(sr.read_text()) if sr.exists() else []:
        if r["view"] == "matplotlib" and "zero-shot" in r["system"]:
            rows.append(f"| {r['system']} | {r['accuracy']:.1%} | – |")
    rows.append("| Chart type alone, no pixels (floor) | 28.8% (all test charts) | |")
    table = "\n".join(rows)
    return f"""
Held-out **test split**, source-disjoint from training. *Real* charts plot real public data
(OpenML, World Bank, OWID, …): the harder and more honest number (n = 1,192). Synthetic n = 2,808.

| System | Real charts | Synthetic |
|---|---|---|
{table}

**Members (weight):** {members}
**Label-shift correction α:** {CFG['alpha']}, chosen on the validation split. Test was scored once.
"""


CSS = """
.gradio-container {max-width: 1180px !important; margin: auto}
#hero h1 {font-size: 2rem; margin-bottom: .2rem}
#hero p {opacity: .75; margin-top: 0}
#result {min-height: 220px}
"""

with gr.Blocks(title="InsightVLM Chart Tester") as demo:
    gr.Markdown("# InsightVLM Chart Tester\nUpload a chart and the model names the "
                "statistical finding it shows, across 20 findings in 4 families.",
                elem_id="hero")
    with gr.Tab("Test the model"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                image = gr.Image(type="pil", label="Chart", sources=["upload", "clipboard"],
                                 height=420)
                with gr.Row():
                    b_real = gr.Button("🎲 Random real test chart", variant="primary")
                    b_syn = gr.Button("🎲 Random synthetic test chart")
            with gr.Column(scale=4):
                result = gr.Markdown(elem_id="result",
                                     value="*Upload a chart or draw a random test chart.*")
                top = gr.Label(num_top_classes=5, label="Top-5 probabilities")
    with gr.Tab("Model performance"):
        gr.Markdown(perf_md())
        with gr.Accordion("The 20 findings", open=False):
            gr.Markdown("\n".join(f"- **{NAMES[c]}** ({INFO[c]['family']}): {INFO[c]['desc']}"
                                  for c in CLASSES))

    image.upload(run_upload, image, [top, result])
    b_real.click(lambda: run_random("real"), None, [image, top, result], api_name="random_real")
    b_syn.click(lambda: run_random("synthetic"), None, [image, top, result],
                api_name="random_synthetic")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="indigo", font=gr.themes.GoogleFont("Inter")),
                css=CSS)
