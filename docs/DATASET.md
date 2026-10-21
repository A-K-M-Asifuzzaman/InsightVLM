# InsightVLM-20K: the dataset

20,000 chart images, 26 chart types, 20 statistical finding classes (exactly 1,000 each).
Version 2 is the current release; see [What changed in version 2](#what-changed-in-version-2).

## The 20 findings

| Family | Class | Verification guard |
|---|---|---|
| Distribution | `dist_normal` | Shapiro–Wilk p > 0.05, \|skew\| < 0.35, dip-test p > 0.15 |
| | `dist_right_skew` | skew > 1.0 and Bowley skew > 0.05 |
| | `dist_left_skew` | skew < −1.0 and Bowley skew < −0.05 |
| | `dist_bimodal` | Hartigan dip p < 0.05, 2-component GMM BIC gain > 10, KS-vs-uniform p < 0.05 |
| | `dist_heavy_tail` | excess kurtosis > 2.0, IQR outliers > 2.5%, \|Bowley\| < 0.10 |
| | `dist_uniform` | KS vs uniform p > 0.10, excess kurtosis < −0.9 |
| Relationship | `rel_strong_positive` | Pearson r > 0.70, \|r − ρ\| < 0.09 |
| | `rel_strong_negative` | r < −0.70, \|r − ρ\| < 0.09 |
| | `rel_none` | \|r\| < 0.12, \|ρ\| < 0.12, quadratic R² < 0.05 |
| | `rel_nonlinear_monotonic` | \|ρ\| > 0.80, \|ρ\| − \|r\| > 0.10, linear R² < 0.72 |
| | `rel_nonlinear_nonmonotonic` | \|r\| < 0.25, quadratic R² > 0.45, \|t_curv\| > 4 |
| | `rel_heteroscedastic` | Breusch–Pagan p < 0.001, residual-spread ratio > 2.5 |
| Time series | `ts_increasing` | Mann–Kendall p < 0.01, τ > 0.35 |
| | `ts_decreasing` | Mann–Kendall p < 0.01, τ < −0.35 |
| | `ts_seasonal` | periodogram power > 0.30, ACF > 0.35, \|τ\| < 0.25 |
| | `ts_changepoint` | step-model SNR > 2.5, step beats line, abruptness > 0.55 |
| | `ts_anomaly_spike` | robust \|z\| > 6, no trend after removal |
| Group | `grp_significant_diff` | Kruskal–Wallis p < 0.001, ε² > 0.14 |
| | `grp_no_diff` | Kruskal–Wallis p > 0.30, ε² < 0.02 |
| | `grp_dominant_category` | top share > 0.55, normalised entropy < 0.75 |

All thresholds live in `build/taxonomy.py`; `dataset/class_index.json` lists each class's
description, test and the chart types it may be drawn as.

## How a label is issued

1. **Subsample before verifying.** Data is cut to its final plotted size before any test
   runs, so the label describes what the image shows.
2. **Exactly one class must match.** A candidate that passes two classes' guards is
   discarded, not multi-labelled.
3. **Reject rather than mislabel.** Every guard is conservative; more candidates can be
   mined, a wrong label cannot be undone.

Robust statistics are used where naive ones fail measurably: moment skewness alone called
12.5% of symmetric Student-t(4) samples skewed, so skew classes also need Bowley skewness;
the dip test is calibrated against the uniform, so bimodality also needs a KS test against
the uniform; and a linear ramp looks like a change point, so level shifts also need a step
model to beat a line. Re-running the battery on the stored tables reproduces 3,998 of 4,000
test labels; the two exceptions sit within 1% of a threshold and fall to *no class*.

## Sources and licences

| Source | Real charts | Source tables | Licence |
|---|---|---|---|
| OpenML | 4,686 | 562 | per dataset (recorded per sample) |
| World Bank | 710 | 22 | CC BY 4.0 |
| Our World in Data | 415 | 20 | CC BY 4.0 |
| vega-datasets | 195 | 30 | BSD-3-Clause |
| seaborn-data | 47 | 14 | BSD-3-Clause |
| UCI ML Repository | 30 | 6 | CC BY 4.0 |
| Synthetic | 13,917 | – | CC0 (generated) |

Real data cannot fill every class (719 public tables contain only ten clean inverted-U
relationships), so the corpus is a declared hybrid. Provenance is balanced *within* each
class so no class is mostly real or mostly synthetic.

## Splits

| Split | All | Real | Per class |
|---|---|---|---|
| train | 14,000 | 4,338 | 700 |
| val | 2,000 | 553 | 100 |
| test | 4,000 | 1,192 | 200 |

Splits are **source-disjoint**: a hash of the source-table id assigns every chart from one
table to one split (0 of 654 sources leak). `splits/test_real_only.jsonl` is the headline
evaluation set.

**Supplementary real training pool.** Balance caps real data at 500 per class, which left
verified real candidates unused. `build/12_extra_real.py` keeps those whose source table
hashes to *train* (at most 400 per class, exact duplicates removed) and asserts that no
validation or test source contributes: 2,545 extra real training charts.

## Files

| Path | Contents |
|---|---|
| `images/<class>/<id>.png` | 768×576 chart images |
| `records.jsonl` | one record per sample: label, family, chart type, split, provenance, source and licence, data table, statistics, test, reasoning chain, insight, render metadata, evidence boxes |
| `tables/metadata.parquet` | the same, flattened |
| `splits/{train,val,test}.jsonl`, `splits/test_real_only.jsonl` | per-split records |
| `instruct/{split}.jsonl` | image/prompt/response triples for instruction tuning |
| `class_index.json` | label ↔ id map, descriptions, tests, valid chart types |
| `DATASHEET.md`, `BIAS_AUDIT.md` | datasheet and shortcut-audit report |

Evidence boxes are in image pixel coordinates (origin top-left) and include the plot area
plus the evidence for the finding (modal bin, outlier band, change point, anomalous points,
dominant wedge, …).

## Shortcut audit (version 2)

All eleven checks pass (`build/10_bias_audit.py`): perfect class and split balance; style
carries no label information (Cramér's V 0.000); real and synthetic charts are not
separable from pixel statistics (AUC 0.551); axis text alone reaches macro-F1 0.369 and
pixel statistics 0.404; no source dominates a class (max 17.2%). Chart type *is* associated
with the label by construction (a pie chart can only show a composition), so the
chart-type rule (28.8%) is reported as the evaluation floor.

## What changed in version 2

- **No label-dependent marks.** Version 1 drew a trend line only on increasing/decreasing
  series, a vertical line only at change points and circles only around anomalies. Models
  learned to read them (see [RESULTS.md](RESULTS.md#the-rendering-leak)). Version 2 draws
  none; the evidence remains in the metadata boxes.
- **Reproducible styles.** Style is chosen from a CRC32 hash of the sample id (version 1
  used Python's salted `hash()`).
- **Dates on time axes.** Epoch-microsecond timestamps are drawn as calendar dates.
- **Extra renders.** Plotly and Vega-Lite re-renders of the real test charts, and training
  renders (Plotly, randomised matplotlib styles) for augmentation.
