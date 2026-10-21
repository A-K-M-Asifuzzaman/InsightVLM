# Methods

## Rendering

Charts are drawn by `build/06_render.py` with matplotlib and seaborn on a 768×576 canvas,
26 chart types and 8 styles. Version 2 renders with `--clean` (no label-dependent marks).
Tight bounding-box cropping is disabled because it resizes the canvas after the axes
transform is read, which would shift every evidence box.

| Render set | Script | Used for |
|---|---|---|
| matplotlib, clean | `06_render.py --clean` | training and the main test set |
| matplotlib, randomised styles | `06_render.py --clean --randstyle` | training augmentation only |
| Plotly (5 templates) | `11_render_plotly.py` | real test set; training augmentation |
| Vega-Lite (9 themes) | `13_render_vega.py` | real test set only — never seen in training |

Randomised styles vary fonts, font sizes, line and marker sizes, colours, background tints,
gridlines, spines and title placement; data and chart type are unchanged. Templates that
imitate a training renderer (Plotly's seaborn template, ggplot2 and fivethirtyeight in
Vega-Lite) are excluded from the test renders.

## Models

ImageNet-initialised ResNet-101, ResNet-50 and DenseNet-121, fine-tuned at 384×384:

- AdamW (lr 3×10⁻⁴, weight decay 0.05), one-cycle schedule, 12 epochs, label smoothing 0.1,
  mixed precision, 2× T4 (Kaggle)
- **About half of every batch is real** (weighted sampler; real is 31% of the balanced split)
- **Checkpoint chosen on real validation accuracy** (553 charts), mixed accuracy as tie-break
- Label-preserving augmentation only: mild crop/scale, colour jitter, blur, small erasures.
  **No flips** — mirroring turns a positive correlation into a negative one and an
  increasing series into a decreasing one.

Each step is an ablation of the next: balanced split → + supplementary real pool → +
renderer augmentation (each epoch shows each training chart as one of its three renders).
One model per Kaggle kernel: host memory grows across epochs and is not returned between
models.

**Deployed ensemble.** Equal-weight average of the class probabilities of the two
full-recipe models (ResNet-101 and DenseNet-121 with pool and renderer augmentation). A
label-shift correction towards real class frequencies is available with its strength tuned
on validation; the tuned strength is zero. Alternatives reported: the average of all seven
trained models (86.2%) and greedy selection on validation (86.5%).

**Zero-shot baselines.** Qwen2.5-VL-3B/7B and Qwen3-VL-8B receive each chart with the 20
class names and one-line definitions and answer with one name (greedy decoding).

**Generative model.** Qwen2-VL-2B with QLoRA (r = 16, 4-bit NF4), one epoch on 4,000 samples,
trained to write the reasoning chain and insight. It was trained on version-1 renders.

## Evaluation protocol

- **Headline:** accuracy on the 1,192 real test charts, for the matplotlib, Plotly and
  Vega-Lite renders. Mixed accuracy is reported but is dominated by synthetic charts.
- **Floor and ceiling:** the chart-type rule (28.8%) and the statistical oracle on the data
  table (99.95%), not the 5% chance level.
- **Uncertainty:** 95% bootstrap confidence intervals (4,000 resamples); paired exact McNemar
  tests between systems on the same charts.
- **Selection on validation only.** The test split is scored once.

## Grounding analysis

For each real test chart with at least one evidence box, the Grad-CAM peak of each deployed
member (predicted class, last convolutional block) is mapped back to image pixels. A hit is
a peak inside an evidence box; the control is a uniformly random point in the plot area,
whose expected hit rate equals the boxes' share of the plot area (`train/grounding_eval.py`).
