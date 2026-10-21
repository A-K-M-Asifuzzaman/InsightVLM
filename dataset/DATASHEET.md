# InsightVLM-20K — Datasheet

A balanced multimodal benchmark for **automated statistical reasoning from
chart images**. Each sample pairs a rendered chart with the data behind it, a
verified statistical finding, the test that licenses that finding, a
step-by-step reasoning chain, and pixel-level grounding boxes.

## 1. At a glance

| | |
|---|---|
| Samples | **20,000** |
| Classes | **20**, exactly 1,000 each |
| Families | distribution (6,000), group (3,000), relationship (6,000), temporal (5,000) |
| Chart types | 26 |
| Splits | train 14,000 / val 2,000 / test 4,000 |
| Provenance | 30.4% real-seeded, 69.6% controlled-synthetic |
| Real source tables | 654 |
| Image size | 768 x 576 PNG |

## 2. Task

**Input:** a chart image.  
**Output:** the reconstructed data table, the statistical finding with its
supporting statistics, and an evidence-grounded insight.

## 3. Label taxonomy

| Class | Family | Verification test |
|---|---|---|
| `dist_normal` | distribution | Shapiro-Wilk + moment bounds + Hartigan dip |
| `dist_right_skew` | distribution | Fisher-Pearson skewness + Shapiro-Wilk |
| `dist_left_skew` | distribution | Fisher-Pearson skewness + Shapiro-Wilk |
| `dist_bimodal` | distribution | Hartigan dip test + 2-component GMM BIC |
| `dist_heavy_tail` | distribution | Excess kurtosis + IQR outlier fraction |
| `dist_uniform` | distribution | KS test against Uniform + platykurtic bound |
| `rel_strong_positive` | relationship | Pearson r with Spearman agreement |
| `rel_strong_negative` | relationship | Pearson r with Spearman agreement |
| `rel_none` | relationship | Pearson + Spearman + quadratic R^2 all near zero |
| `rel_nonlinear_monotonic` | relationship | Spearman >> Pearson divergence |
| `rel_nonlinear_nonmonotonic` | relationship | Quadratic R^2 with significant curvature, linear r ~ 0 |
| `rel_heteroscedastic` | relationship | Breusch-Pagan + residual-spread ratio across x terciles |
| `ts_increasing` | temporal | Mann-Kendall tau (non-parametric monotonic trend) |
| `ts_decreasing` | temporal | Mann-Kendall tau (non-parametric monotonic trend) |
| `ts_seasonal` | temporal | Periodogram dominant power + ACF at the implied lag |
| `ts_changepoint` | temporal | Binary-segmentation level shift vs residual scale |
| `ts_anomaly_spike` | temporal | Median/MAD robust z-score point anomalies |
| `grp_significant_diff` | group | Kruskal-Wallis H + epsilon-squared effect size |
| `grp_no_diff` | group | Kruskal-Wallis H + epsilon-squared effect size |
| `grp_dominant_category` | group | Top-category share + normalised Shannon entropy |

## 4. How labels were assigned

Labels are **never** assigned by assumption. Both construction tracks pass
through one shared gate (`build/taxonomy.py`), which runs a battery of named
statistical tests on **exactly the array that gets plotted** — data is
subsampled to its final plotted size *before* verification, so the ground
truth always describes the rendered image rather than a parent dataframe.

A candidate is kept only if **exactly one** class's guard set passes.
Multi-match candidates are discarded as ambiguous, which keeps the primary
label single-label and clean for a confusion matrix.

Guards are mutually exclusive by construction. Some examples of the care this
required:

- `dist_heavy_tail` is gated on **Bowley (quartile) skewness**, not moment
  skewness. Moment skewness is driven by single extreme points, so symmetric
  heavy-tailed samples were being mislabelled as skewed 12.5% of the time.
- `dist_bimodal` additionally requires a KS test against the uniform to reject.
  Hartigan's dip test is calibrated against the uniform, so a genuinely flat
  distribution trips `dip_p < 0.05` about 5% of the time.
- `ts_changepoint` requires a two-level step model to beat a linear fit **and**
  an abruptness ratio > 0.55. A pure ramp also produces a high-SNR
  'changepoint', and a smooth rise-then-fall defeats a monotonic residual test.

Generator-to-class fidelity over 10 seeds x 20 classes is **95.5%**; every
remaining miss is a conservative rejection, never a wrong label.

## 5. Provenance

Real data cannot balance these classes: public corpora hold thousands of
right-skewed columns and almost no clean inverted-U relationships. A
1000/class balance from real data alone is not achievable. The dataset is
therefore a declared hybrid.

**Track R (real-seeded).** Public tables mined for every plottable view —
single columns, column pairs, per-entity time series, numeric-by-category
groupings, categorical compositions.

| Source | Samples | Licence |
|---|---|---|
| OpenML | 4,686 | per-dataset, recorded per row |
| World Bank | 710 | CC-BY-4.0 |
| Our World in Data | 415 | CC-BY-4.0 |
| vega-datasets | 195 | BSD-3-Clause |
| seaborn-data | 47 | BSD-3-Clause |
| UCI ML Repository | 30 | CC-BY-4.0 |

**Track S (controlled synthesis).** Parametric generators where the property
holds by construction, drawn from a domain vocabulary (health, economics,
environment, engineering, education, retail) with plausible names, units and
magnitudes. Every synthetic sample still has to pass the same gate.

Provenance is balanced **within** each class rather than real-first. Real-first
would leave some classes ~100% real and others ~100% synthetic, letting a model
key on provenance artefacts instead of the statistical pattern:

| Class | Real | Synthetic |
|---|---|---|
| `dist_normal` | 163 | 837 |
| `dist_right_skew` | 500 | 500 |
| `dist_left_skew` | 75 | 925 |
| `dist_bimodal` | 500 | 500 |
| `dist_heavy_tail` | 163 | 837 |
| `dist_uniform` | 114 | 886 |
| `rel_strong_positive` | 366 | 634 |
| `rel_strong_negative` | 68 | 932 |
| `rel_none` | 500 | 500 |
| `rel_nonlinear_monotonic` | 189 | 811 |
| `rel_nonlinear_nonmonotonic` | 9 | 991 |
| `rel_heteroscedastic` | 500 | 500 |
| `ts_increasing` | 500 | 500 |
| `ts_decreasing` | 458 | 542 |
| `ts_seasonal` | 64 | 936 |
| `ts_changepoint` | 175 | 825 |
| `ts_anomaly_spike` | 421 | 579 |
| `grp_significant_diff` | 500 | 500 |
| `grp_no_diff` | 500 | 500 |
| `grp_dominant_category` | 318 | 682 |

## 6. Splits

70 / 10 / 20 train-val-test, exactly 700/100/200 per class.

Splits are **source-disjoint** for real data: a source table is assigned
wholly to one split, so no chart derived from `openml/61_iris` can appear in
both train and test. Synthetic samples each carry a unique source id and
cannot leak by construction.

### Real-only generalisation probe

`splits/test_real_only.jsonl` — the 1,192 test samples drawn purely
from held-out **real** source tables. Reporting only mixed-test accuracy would
leave open whether a model learned the statistics or the generator, so this
probe is the honest generalisation number.

It is **unbalanced by construction** — public data cannot supply every class
(4 of 20 classes fall under n=20). Report macro-F1
over classes with n>=20 and give per-class n alongside; per-class counts are in
`splits/test_real_only_counts.json`.

## 7. What each record contains

```
sample_id, image, split, label, family, arity, chart_type, difficulty,
provenance, source_id, source_url, license, columns, axis_labels,
ground_truth: {
    class, class_description, verification_test,
    statistics      <- the full statistic dictionary (the ground truth)
    data_table      <- the exact values plotted (reconstruction target)
    reasoning_chain <- observation -> statistic -> test -> conclusion
    insight         <- prose, every claim citing a computed number
},
qa: [{q, a}, ...],
render: { image_size, style, grounding_boxes: [{label, bbox}] }
```

`bbox` is `[x0, y0, x1, y1]` in **image pixels, origin top-left**. Charts are
rendered on a fixed canvas with `bbox_inches="tight"` deliberately disabled —
tight cropping resizes the canvas after the axes transform is read, which
silently invalidates every box.

## 8. Difficulty

Each sample records an `evidence_margin` — the statistic that carries its
class (Pearson |r|, Mann-Kendall |tau|, changepoint SNR, epsilon-squared, ...).
Difficulty is the **within-class tertile** of that margin, so every class is
one third easy / medium / hard and `hard` always means *near this class's own
decision boundary*.

Absolute thresholds were tried first and do not work: the verification gate
already refuses weak evidence — a bimodal too subtle to trip Hartigan's dip
test is never admitted — so classes like `ts_seasonal` and `rel_heteroscedastic`
have no attainable 'hard' region on an absolute scale and graded 100% easy.
Tertiles are self-calibrating and introduce no arbitrary constants.

{'hard': 6680, 'easy': 6658, 'medium': 6662}

## 8b. Bias and shortcut audit

Eleven measured failure modes, each with a named statistic and an explicit
threshold. A *shortcut* is any feature that predicts the label without
containing the statistical evidence. Reproduce with
`python build/10_bias_audit.py`.

| Test | Statistic | Value | Threshold |
|---|---|---|---|
| Class balance | imbalance ratio | 1.000 | = 1.000 |
| Split balance | max s.d. of proportions | 0.0000 | < 0.01 |
| Chart type -> label | pixel-free prior accuracy | 0.2883 | < 0.40 |
| Render style -> label | Cramer's V | 0.0000 | < 0.05 |
| Provenance detectable | ROC AUC | 0.548 | < 0.75 |
| Axis-text shortcut | macro-F1 | 0.3688 | < 0.55 |
| Pixel-stat shortcut | macro-F1 | 0.4078 | < 0.55 |
| Source concentration | max top-source share | 17.2% | < 35% |
| Length confound (within family) | max/min median n | 2.04 | < 4.0 |
| Subject-domain balance | max domain share | 22.8% | < 45% |
| Category-vocabulary spread | max vocabulary share | 13.0% | < 35% |

**Provenance detectability is the result that matters most.** With ~70% of
samples synthetic, the obvious objection is that a model learns to spot
generated charts rather than statistical patterns. A classifier trained on
low-level pixel statistics separates them at AUC 0.548
— essentially chance. Real and synthetic charts are not visually distinguishable.

**Chart type is associated with the label by construction and is disclosed
rather than removed.** A pie chart can only carry a composition finding; a QQ
plot only a distribution finding. The pixel-free chart-type prior
(0.2883) is therefore the
evaluation floor. Reporting against the 5% majority baseline instead would
overstate every result by roughly 24 points.

Two flags raised by an earlier run of this audit were traced to defects in
our own tooling, not to the data: the `domain` field was being dropped during
assembly, and the length confound was being measured across families (where
the difference is structural) instead of within them. Both are fixed.

Full report: `dataset/BIAS_AUDIT.md`.

## 9. Known limitations

1. **70% of samples are synthetic.** Balance
   forced this. Report real and synthetic performance separately; the real
   subset is the harder and more meaningful evaluation.
2. **Single-label by design.** Real charts often show several findings at once
   (a trend *and* an anomaly). Ambiguous candidates were discarded rather than
   multi-labelled, so the dataset under-represents compound findings.
3. **Thresholds are conventions.** `|r| > 0.7` as 'strong' and eta-squared >
   0.14 as 'large' follow common practice but remain choices; they are recorded
   explicitly in `taxonomy.py` so results can be re-derived under other cuts.
4. **Rendering is matplotlib/seaborn only.** No Excel, Plotly, D3 or hand-drawn
   charts, so cross-renderer generalisation is untested.
5. **Class priors are artificial.** Perfect balance is a training convenience;
   real-world finding frequencies are heavily skewed.

## 10. Reproduction

```bash
python run_all.py            # full rebuild, ~40 min on 11 cores
python build/taxonomy.py     # taxonomy self-test (generator -> class fidelity)
python build/07_validate.py  # integrity audit
```

## 11. Audit result

```
{
  "n_records": 20000,
  "label_reverification_rate": 1.0,
  "leaked_sources": 0,
  "duplicate_fingerprints": 0,
  "passed": true
}
```

Labels re-verified by re-running the full battery on the stored data tables,
independent of the pipeline that produced them.

### Numerical reproducibility

The oracle reproduces ~99.95% of labels, not 100%, when run against a
different scientific-Python build (measured on Kaggle vs the build machine's
statsmodels 0.14.6 / scipy 1.18.0). The cause is boundary proximity, not data
error: `rel_heteroscedastic` carries the tightest guards in the taxonomy and
its closest sample sits at `resid_spread_ratio` = 2.5054 against a 2.5 floor —
a 0.2% margin. A ~1e-4 difference in the Breusch-Pagan statistic tips it.

Such samples fail to **`no_class_matched`**, i.e. they are dropped, never
assigned a wrong class. Report the oracle as the measured value rather than
assuming 1.00, and pin `statsmodels`/`scipy` when exact reproduction matters.
