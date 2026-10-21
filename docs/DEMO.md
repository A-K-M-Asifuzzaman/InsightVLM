# Demo: InsightVLM Chart Tester

**Live:** <https://huggingface.co/spaces/AsifZaman1912/InsightVLM>

![Demo](assets/demo.png)

## Using it

- **Upload a chart** — click the panel, drag an image onto it, or paste from the clipboard.
- **Random real / synthetic test chart** — draws one of 400 held-out test charts and shows the
  prediction next to the verified answer, its source, chart type and verified insight.

For every chart the page shows the finding, its confidence, how many ensemble members agree,
the definition of the finding and the statistical test behind it, and the five most likely
findings. Below 50% confidence it warns that the chart may show mixed patterns or one outside
the 20 classes.

## How it works

The deployed ensemble (ResNet-101 + DenseNet-121) is exported to a single ONNX graph
(`deploy/export_onnx.py`) and run in the browser by ONNX Runtime Web (WebGPU when available,
otherwise WebAssembly). Nothing is uploaded. The model (198 MB) downloads once and is cached by
the browser.

Preprocessing reproduces evaluation exactly: the page implements PIL's antialiased bilinear
resize to 384×384 in JavaScript, followed by ImageNet normalisation. `deploy/verify_onnx.py`
confirms the ONNX model agrees with the evaluated ensemble on all 4,000 test charts, and the
page agreed with it on every chart we tested in a headless browser.

## Running it locally

```bash
git clone https://huggingface.co/spaces/AsifZaman1912/InsightVLM insightvlm-demo   # needs git-lfs
cd insightvlm-demo && python -m http.server 8000
```

Or the Gradio version: `pip install -r requirements-app.txt && python app.py` (needs the
trained checkpoints and `dataset/`).

## Limits

The tester only knows the 20 findings and reports one per chart. Charts drawn by tools unlike
the training renderers are less reliable (71.6% on an unseen plotting library), and the weakest
classes are heavy tails, left skew and seasonality.
