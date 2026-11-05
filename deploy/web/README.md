---
title: InsightVLM Chart Tester
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Name the statistical finding a chart shows (20 findings)
---

# InsightVLM Chart Tester

Upload a chart image and the model names the statistical finding it shows (one of 20
findings across distributions, relationships, time series and group comparisons), with
its confidence, model agreement and the statistical test that defines the finding. Or
draw a random held-out test chart to compare the prediction with the verified answer.

The model (an equal-weight ensemble of ResNet-101 and DenseNet-121 at 384×384) runs in
the browser with ONNX Runtime Web; images never leave the visitor's computer.
Test accuracy: 86.7% on 1,192 real charts, 99.9% on synthetic charts.

Code and documentation: https://github.com/A-K-M-Asifuzzman/InsightVLM
