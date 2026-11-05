# InsightVLM-20K

**A balanced benchmark for turning chart images into verified statistical findings.**

Give a model a chart. It should recover the data, name the statistical pattern, cite the numbers that prove it, and point to where in the image the evidence is. InsightVLM-20K supplies the chart and all of those answers, and every answer has been checked by a statistical test run on the plotted data.

| | |
|---|---|
| **Samples** | 20,000 charts (768 × 576 PNG) |
| **Classes** | 20, exactly 1,000 each |
| **Families** | distribution 6,000 · relationship 6,000 · temporal 5,000 · group 3,000 |
| **Chart types** | 26 |
| **Splits** | train 14,000 · val 2,000 · test 4,000 (700 / 100 / 200 per class) |
| **Provenance** | 30.4% real-seeded (6,083) · 69.6% controlled-synthetic (13,917) |
| **Real source tables** | 654 |
| **Label audit** | 100% re-verified, 0 leaked sources, 0 duplicates |

> **Read this before reporting results.** About 70% of the data is synthetic. Always report the **real-only test probe** (Section 6) next to the full test score, and compare against the **chart-type prior of 0.288**, not the 5% majority baseline (Section 8).

---

## 1. What a model is asked to do

**Input:** one chart image.

**Output:**
1. the reconstructed data table,
2. the statistical finding (class) with its supporting statistics,
3. an evidence-grounded insight in prose,
4. optionally, grounding boxes for the evidence.

Because every record includes the ground-truth table, statistics, reasoning chain, QA pairs and boxes, the same data supports classification, table extraction, chart QA, reasoning-chain generation and grounding.

## 2. Label taxonomy (20 classes)

| Family | Class | Verification test |
|---|---|---|
| **Distribution** | `dist_normal` | Shapiro-Wilk + moment bounds + Hartigan dip |
| | `dist_right_skew` | Fisher-Pearson skewness + Shapiro-Wilk |
| | `dist_left_skew` | Fisher-Pearson skewness + Shapiro-Wilk |
| | `dist_bimodal` | Hartigan dip + 2-component GMM BIC (+ KS vs uniform must reject) |
| | `dist_heavy_tail` | Excess kurtosis + IQR outlier fraction (gated on Bowley skewness) |
| | `dist_uniform` | KS vs Uniform + platykurtic bound |
| **Relationship** | `rel_strong_positive` | Pearson r with Spearman agreement |
| | `rel_strong_negative` | Pearson r with Spearman agreement |
| | `rel_none` | Pearson, Spearman and quadratic R² all near zero |
| | `rel_nonlinear_monotonic` | Spearman >> Pearson divergence |
| | `rel_nonlinear_nonmonotonic` | Quadratic R² with significant curvature, linear r ≈ 0 |
| | `rel_heteroscedastic` | Breusch-Pagan + residual-spread ratio across x terciles |
| **Temporal** | `ts_increasing` | Mann-Kendall tau |
| | `ts_decreasing` | Mann-Kendall tau |
| | `ts_seasonal` | Periodogram dominant power + ACF at implied lag |
| | `ts_changepoint` | Binary-segmentation level shift vs residual scale |
| | `ts_anomaly_spike` | Median/MAD robust z-score |
| **Group** | `grp_significant_diff` | Kruskal-Wallis H + epsilon-squared |
| | `grp_no_diff` | Kruskal-Wallis H + epsilon-squared |
| | `grp_dominant_category` | Top-category share + normalised Shannon entropy |

## 3. How labels were assigned

Labels are never assumed from how a chart was generated. Both construction tracks go through one shared gate (`build/taxonomy.py`) that runs named statistical tests on **exactly the array that gets plotted**. Data is subsampled to its final plotted size *before* verification, so the ground truth describes the image, not a parent dataframe.

- **Exactly one class must pass.** A candidate is kept only if one class's guard set passes. Multi-match candidates are discarded as ambiguous, keeping the label single and the confusion matrix clean.
- **Guards are mutually exclusive by design.** Examples:
  - `dist_heavy_tail` uses **Bowley (quartile) skewness**, not moment skewness. Moment skewness is driven by single extreme points and mislabelled symmetric heavy-tailed samples as skewed 12.5% of the time.
  - `dist_bimodal` also requires a KS test against uniform to reject. Hartigan's dip is calibrated against uniform, so a flat distribution trips `dip_p < 0.05` about 5% of the time.
  - `ts_changepoint` requires a two-level step model to beat a linear fit **and** an abruptness ratio > 0.55, because a pure ramp also produces a high-SNR "changepoint".
- **Generator-to-class fidelity: 95.5%** (10 seeds × 20 classes). Every miss is a conservative rejection, never a wrong label.

## 4. Where the data comes from

Real data cannot balance these classes: public corpora hold thousands of right-skewed columns and almost no clean inverted-U relationships. The dataset is therefore a **declared hybrid** of two tracks.

**Track R, real-seeded.** Public tables mined for every plottable view: single columns, column pairs, per-entity time series, numeric-by-category groupings, categorical compositions.

| Source | Samples | Licence |
|---|---:|---|
| OpenML | 4,686 | per-dataset, recorded per row |
| World Bank | 710 | CC-BY-4.0 |
| Our World in Data | 415 | CC-BY-4.0 |
| vega-datasets | 195 | BSD-3-Clause |
| seaborn-data | 47 | BSD-3-Clause |
| UCI ML Repository | 30 | CC-BY-4.0 |

**Track S, controlled synthesis.** Parametric generators where the property holds by construction, using realistic names, units and magnitudes from six domains (health, economics, environment, engineering, education, retail). Synthetic samples go through the same gate.

### Real vs synthetic mix per class

Every class mixes both tracks, so no class is purely real or purely synthetic and provenance alone cannot predict the label.

| Class | Real | Synthetic | | Class | Real | Synthetic |
|---|---:|---:|---|---|---:|---:|
| `dist_normal` | 163 | 837 | | `rel_heteroscedastic` | 500 | 500 |
| `dist_right_skew` | 500 | 500 | | `ts_increasing` | 500 | 500 |
| `dist_left_skew` | 75 | 925 | | `ts_decreasing` | 458 | 542 |
| `dist_bimodal` | 500 | 500 | | `ts_seasonal` | 64 | 936 |
| `dist_heavy_tail` | 163 | 837 | | `ts_changepoint` | 175 | 825 |
| `dist_uniform` | 114 | 886 | | `ts_anomaly_spike` | 421 | 579 |
| `rel_strong_positive` | 366 | 634 | | `grp_significant_diff` | 500 | 500 |
| `rel_strong_negative` | 68 | 932 | | `grp_no_diff` | 500 | 500 |
| `rel_none` | 500 | 500 | | `grp_dominant_category` | 318 | 682 |
| `rel_nonlinear_monotonic` | 189 | 811 | | `rel_nonlinear_nonmonotonic` | **9** | **991** |

## 5. Record format

```
sample_id, image, split, label, family, arity, chart_type, difficulty,
provenance, source_id, source_url, license, columns, axis_labels,
ground_truth: {
    class, class_description, verification_test,
    statistics       # full statistic dictionary (the ground truth)
    data_table       # exact values plotted (reconstruction target)
    reasoning_chain  # observation -> statistic -> test -> conclusion
    insight          # prose; every claim cites a computed number
},
qa: [{q, a}, ...],
render: { image_size, style, grounding_boxes: [{label, bbox}] }
```

`bbox` is `[x0, y0, x1, y1]` in **image pixels, origin top-left**. Charts use a fixed canvas with `bbox_inches="tight"` deliberately disabled, because tight cropping resizes the canvas after the axes transform is read and silently invalidates every box.

## 6. Splits and evaluation protocol

Splits are 70 / 10 / 20 train / val / test, exactly 700 / 100 / 200 per class.

- **Real data is source-disjoint.** A source table goes wholly to one split, so nothing derived from `openml/61_iris` appears in both train and test.
- **Synthetic data cannot leak.** Each synthetic sample has a unique source id.

### Recommended reporting

| Report | Why |
|---|---|
| Full test (4,000), accuracy and macro-F1 | Headline comparison |
| **Real-only probe** (`splits/test_real_only.jsonl`, 1,192 samples) | The honest generalisation number: did the model learn the statistics or the generator? |
| Real vs synthetic, separately | The real subset is the harder, more meaningful evaluation |
| Per-class n alongside every real-only number | Real data cannot supply every class |
| Comparison to the **0.2883** chart-type prior | The true floor (Section 8) |

The real-only probe is **unbalanced by construction**, with 4 of 20 classes under n = 20. Report macro-F1 over classes with n ≥ 20 only. Per-class counts: `splits/test_real_only_counts.json`.

## 7. Difficulty

Each sample records an `evidence_margin`, the statistic that carries its class (Pearson |r|, Mann-Kendall |tau|, changepoint SNR, epsilon-squared, ...). Difficulty is the **within-class tertile** of that margin, so every class is one third easy, medium and hard, and `hard` always means *near this class's own decision boundary*.

| easy | medium | hard |
|---:|---:|---:|
| 6,658 | 6,662 | 6,680 |

Absolute thresholds were tried first and failed: the gate already refuses weak evidence, so classes like `ts_seasonal` and `rel_heteroscedastic` had no attainable "hard" region and graded 100% easy. Tertiles are self-calibrating and add no arbitrary constants.

## 8. Bias and shortcut audit

A *shortcut* is any feature that predicts the label without containing the statistical evidence. Eleven failure modes were measured, each with a named statistic and threshold. All pass. Reproduce with `python build/10_bias_audit.py`; full report in `dataset/BIAS_AUDIT.md`.

| Test | Statistic | Value | Threshold | Pass |
|---|---|---:|---:|:-:|
| Class balance | imbalance ratio | 1.000 | = 1.000 | ✓ |
| Split balance | max s.d. of proportions | 0.0000 | < 0.01 | ✓ |
| Chart type → label | pixel-free prior accuracy | 0.2883 | < 0.40 | ✓ |
| Render style → label | Cramér's V | 0.0000 | < 0.05 | ✓ |
| Provenance detectable | ROC AUC | 0.548 | < 0.75 | ✓ |
| Axis-text shortcut | macro-F1 | 0.3688 | < 0.55 | ✓ |
| Pixel-stat shortcut | macro-F1 | 0.4078 | < 0.55 | ✓ |
| Source concentration | max top-source share | 17.2% | < 35% | ✓ |
| Length confound (within family) | max/min median n | 2.04 | < 4.0 | ✓ |
| Subject-domain balance | max domain share | 22.8% | < 45% | ✓ |
| Category-vocabulary spread | max vocabulary share | 13.0% | < 35% | ✓ |

**Can a model just spot synthetic charts?** A classifier on low-level pixel statistics separates real from synthetic at AUC 0.548, close to chance (0.5), so the two are not easily distinguished at the pixel level.

**Chart type is tied to the label by design, and disclosed.** A pie chart can only carry a composition finding; a QQ plot only a distribution finding. The chart-type prior (**0.2883**) is therefore the evaluation floor. Comparing against the 5% majority baseline would overstate every result by about 24 points.

*Audit history:* two flags in an earlier run were defects in our tooling, not the data. The `domain` field was dropped during assembly, and length was measured across families instead of within them. Both are fixed.

## 9. Intended use and limitations

**Intended for:** training and benchmarking chart-to-insight models, statistical-reasoning evaluation, table reconstruction, and grounded chart QA.

**Not suited for:** estimating how often each finding occurs in the real world, or claims about performance on charts from other renderers.

1. **70% synthetic.** Balance forced this. Report real and synthetic performance separately.
2. **Single-label by design.** Real charts often show several findings at once (a trend *and* an anomaly). Ambiguous candidates were discarded, so compound findings are under-represented.
3. **Thresholds are conventions.** `|r| > 0.7` as "strong" and eta-squared > 0.14 as "large" follow common practice but are choices. They are recorded in `taxonomy.py` so results can be re-derived under other cuts.
4. **One renderer.** matplotlib/seaborn only. No Excel, Plotly, D3 or hand-drawn charts, so cross-renderer generalisation is untested.
5. **Artificial class priors.** Perfect balance is a training convenience; real finding frequencies are heavily skewed.
6. **Very few real examples for some classes**, e.g. `rel_nonlinear_nonmonotonic` (9 real) and `ts_seasonal` (64 real).

## 10. Integrity and reproducibility

```bash
python run_all.py            # full rebuild, ~40 min on 11 cores
python build/taxonomy.py     # taxonomy self-test (generator -> class fidelity)
python build/07_validate.py  # integrity audit
```

```json
{
  "n_records": 20000,
  "label_reverification_rate": 1.0,
  "leaked_sources": 0,
  "duplicate_fingerprints": 0,
  "passed": true
}
```

Labels were re-verified by re-running the full test battery on the stored data tables, independently of the build pipeline.

**Numerical reproducibility.** On a different scientific-Python build (measured on Kaggle vs the build machine's statsmodels 0.14.6 / scipy 1.18.0), the oracle reproduces about **99.95%** of labels, not 100%. The cause is boundary proximity, not data error: `rel_heteroscedastic` has the tightest guards, and its closest sample has `resid_spread_ratio` = 2.5054 against a 2.5 floor (a 0.2% margin), which a ~1e-4 change in the Breusch-Pagan statistic can tip. Such samples fail to `no_class_matched`, meaning they are dropped, never given a wrong class. Report the measured value and pin `statsmodels` / `scipy` for exact reproduction.

## 11. Citation, licence, contact

**Citation**

> Asifuzzaman, A. K. M., Jisan, R. H., Alif, F. K., Iftee, I. K., & Arfat, M. (2026). *InsightVLM-20K: a chart corpus with statistically verified findings* (Version 2) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23968282

```bibtex
@dataset{asifuzzaman_2026_insightvlm20k,
  author    = {Asifuzzaman, A. K. M. and Jisan, R. H. and Alif, F. K. and Iftee, I. K. and Arfat, M.},
  title     = {InsightVLM-20K: a chart corpus with statistically verified findings},
  year      = {2026},
  version   = {2},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23968282},
  url       = {https://doi.org/10.5281/zenodo.23968282}
}
```

