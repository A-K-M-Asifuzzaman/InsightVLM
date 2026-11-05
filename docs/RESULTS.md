# Results

All numbers are on the held-out test split, scored once. Files: `results/stats_report.json`,
`results/final_ensemble.json`, `results/grounding.json`, `results/ablate_marks.csv`.

## Main results

| System | Real · matplotlib | Real · Plotly | Real · Vega-Lite | Synthetic |
|---|---|---|---|---|
| Chart type alone (no pixels) | 0.288 | – | – | – |
| Qwen2.5-VL-3B, zero-shot | 0.306 | – | – | 0.303¹ |
| Qwen2.5-VL-7B, zero-shot | 0.411 | – | – | 0.498¹ |
| Qwen3-VL-8B, zero-shot | 0.468 | – | – | 0.648¹ |
| ResNet-101, balanced split | 0.841 | 0.747 | 0.607 | 0.997 |
| ResNet-50, balanced split | 0.828 | 0.722 | 0.608 | 0.998 |
| ResNet-101, + supplementary pool | 0.840 | 0.735 | 0.602 | 0.999 |
| ResNet-50, + supplementary pool | 0.845 | 0.760 | 0.599 | 0.998 |
| DenseNet-121, + supplementary pool | 0.861 | 0.771 | 0.638 | 0.997 |
| ResNet-101, + pool + renderer augmentation | 0.859 | 0.854 | 0.650 | 0.998 |
| DenseNet-121, + pool + renderer augmentation | 0.855 | 0.852 | 0.712 | 0.998 |
| **Ensemble (deployed)** | **0.867** [0.846, 0.886] | **0.857** [0.836, 0.877] | **0.716** [0.690, 0.741] | **0.999** |
| Statistical oracle on the data table | 0.9995 | – | – | 0.9995 |

¹ Fixed 400-chart synthetic sample. Brackets: 95% bootstrap CI. Over all 4,000 test charts
the ensemble scores 0.959; macro-F1 on real charts is 0.776. Against Qwen3-VL-8B the ensemble
is right on 511 charts where it is wrong and wrong on 36 where it is right (exact McNemar
p < 10⁻¹⁰⁰).

**Seed variance** (`train/seed_variance.py`): retraining both members with two more seeds
gives real-chart accuracy 0.867 / 0.863 / 0.858, mean **0.863 ± 0.004** (s.d., 3 seeds);
Plotly 0.857 ± 0.006, Vega-Lite 0.713 ± 0.012, synthetic 0.999 ± 0.000. The deployed run is
the best of the three on matplotlib charts, within one s.d. of the mean.

**Renderer augmentation** is the step that matters for other plotting libraries: Plotly is
seen in augmentation, so its gap closes; Vega-Lite is never seen, and accuracy on it still
rises from 0.60–0.64 to 0.65–0.71 per model without costing matplotlib accuracy.

## Where the model fails

![Confusion matrix](figures/fig_confusion.png)

Distribution shape is the weakest family (0.763 on real charts; 0.886–0.930 for the others).
The commonest errors: bimodal read as right-skewed (19 of 100), heavy-tailed read as
right-skewed (13 of 26), significant group difference read as none (12 of 100). Accuracy is
lowest where real examples are rare: heavy tails 0.385 (n = 26), seasonality 0.462 (n = 13),
left skew 0.474 (n = 19). These per-class numbers are indicative only: with n = 13–26 a class's
95% interval spans roughly ±19 to ±27 points.

![Per-class accuracy](figures/fig_per_class.png)

## Near-duplicates across splits

`build/14_near_duplicates.py` compares every real validation/test chart with every real chart a
model can train on (same arity). Signatures correlating above 0.99 are near-duplicates only when
the data matches: time series within 5% in mean and spread; ≥ 50% identical (x, y) pairs or ≥ 90%
identical values for sampled data; same categories and shares for compositions.

| | Charts |
|---|---|
| Real test charts flagged | 85 of 1,192 (7.1%) |
| Real validation charts flagged | 34 |

Most are republished tables (World Bank vs Our World in Data internet use; repeated OpenML
uploads). Ensemble accuracy on real test charts: **0.867 with them, 0.863 [0.843, 0.883] without**;
0.918 on the flagged charts alone. IDs: `results/near_duplicates.json`.

## Human label check

Two annotators (one an author) judged 188 real charts shown with their labelled finding, plus
25 decoys with a wrong finding, from the chart alone. Exclusion rule (fixed in advance): more
than 40% of decoys accepted; neither met it.

| | |
|---|---|
| Labels confirmed / rejected / cannot tell | **0.737** [0.676, 0.790] / 0.168 / 0.096 |
| Decoys rejected / accepted | 0.78 / 0.04 |
| Agreement between annotators | 0.867 raw, Fleiss κ 0.68 |

Confirmation is 0.95–1.00 for clear-cut findings but 0.25 for bimodal and heteroscedastic and
0.40 for monotonic non-linear and anomaly-spike charts: statistically correct, visually subtle.
Scoring: `train/human_study_eval.py`.

## Grounding

Grad-CAM pointing game on 1,023 real test charts with evidence boxes:

| Charts | Pointing | Random control | Gain [95% CI] |
|---|---|---|---|
| Time series (305) | 0.649 | 0.495 | +0.154 [+0.139, +0.168] |
| Groups (258) | 0.599 | 0.489 | +0.110 [+0.064, +0.153] |
| Relationships (173) | 0.642 | 0.569 | +0.073 [+0.029, +0.115] |
| Distributions (287) | 0.253 | 0.254 | −0.002 [−0.035, +0.033] |
| **All** | **0.524** | **0.439** | **+0.085** [+0.068, +0.102] |

The models attend to the verified evidence beyond chance for three of four families, not for
distributions — whose shape is a property of the whole plot rather than one region.

## Generative model

Qwen2-VL-2B + QLoRA (300-chart subset, version-2 renders): names the correct finding 78.7%
[74.0, 83.3] of the time (65.9% real, 83.5% synthetic) but only 13.9% of the numbers it cites match the
ground-truth statistics. The deterministic pipeline on the data table is right 99.95% of
the time and exact. Trained on version-1 renders the same model scored 80.0%, with 100% on
time series against 94.7% on version 2 and identical results elsewhere: the leak inflated its
time-series accuracy too; numeric fidelity did not change (13.8%).

## The rendering leak

Version 1 drew marks on time-series charts only for the matching class (trend line, change-
point line, anomaly circles). Rendering the same 1,000 test series with and without the
marks, identical otherwise, measures how much version-1 models relied on them:

| Test charts | ResNet-101, with marks | without | ResNet-50, with marks | without |
|---|---|---|---|---|
| Real time series (305) | 0.957 | **0.311** | 0.944 | **0.407** |
| Synthetic time series (695) | 1.000 | 0.801 | 1.000 | 0.911 |
| level shift | 1.000 | 0.355 | 1.000 | 0.705 |
| seasonal (never marked) | 0.970 | 0.970 | 0.990 | 0.990 |

![Marks ablation](figures/fig_marks_ablation.png)

The seasonal class, which never carried a mark, did not move — isolating the cause. The
eleven-test shortcut audit had passed, because it examines data and pixel statistics, not
what a trained model uses. Every result above uses version 2 and models trained on it. An
earlier ensemble had also been selected on test accuracy (0.865); selected on validation, the
same models give 0.845.
