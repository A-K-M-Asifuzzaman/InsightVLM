# InsightVLM-20K

**A chart corpus with statistically verified findings.** Version 2.

20,000 chart images · 26 chart types · 20 finding classes (exactly 1,000 each). Every label is issued by a named statistical test run on the data that is actually plotted, and every sample ships with its data table, statistics, reasoning chain, insight and evidence boxes.

| | |
|---|---|
| **Images** | 20,000 PNG, 768 × 576 |
| **Classes** | 20 across 4 families (distribution, relationship, time series, group) |
| **Splits** | train 14,000 · val 2,000 · test 4,000 (700 / 100 / 200 per class) |
| **Provenance** | 6,083 real-seeded · 13,917 synthetic |
| **Real source tables** | 654, none shared across splits |
| **Label check** | re-running the full test battery reproduces 3,998 of 4,000 test labels (99.95%) |

> **Start here:** evaluate on `splits/test_real_only.jsonl` (the headline set) and compare against the **28.8% chart-type floor**, not the 5% majority baseline. Details in [Splits](#splits) and [Shortcut audit](#shortcut-audit-version-2).

**Cite as:** Asifuzzaman, A. K. M., Jisan, R. H., Alif, F. K., Iftee, I. K., & Arfat, M. (2026). *InsightVLM-20K: a chart corpus with statistically verified findings* (Version 2) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23968282

---

## Quick start

```python
import json

with open("splits/test_real_only.jsonl") as f:
    records = [json.loads(line) for line in f]

r = records[0]
print(r["label"], r["family"], r["chart_type"], r["provenance"])
print(r["image"])                          # images/<class>/<id>.png
print(r["ground_truth"]["data_table"])     # reconstruction target
print(r["ground_truth"]["insight"])        # evidence-grounded prose
```

For instruction tuning, use `instruct/{split}.jsonl` (image / prompt / response triples).

## The 20 findings

All thresholds live in `build/taxonomy.py`. `class_index.json` lists each class's description, test and valid chart types.

| Family | Class | Verification guard |
|---|---|---|
| **Distribution** | `dist_normal` | Shapiro–Wilk p > 0.05, \|skew\| < 0.35, dip-test p > 0.15 |
| | `dist_right_skew` | skew > 1.0 and Bowley skew > 0.05 |
| | `dist_left_skew` | skew < −1.0 and Bowley skew < −0.05 |
| | `dist_bimodal` | Hartigan dip p < 0.05, 2-component GMM BIC gain > 10, KS-vs-uniform p < 0.05 |
| | `dist_heavy_tail` | excess kurtosis > 2.0, IQR outliers > 2.5%, \|Bowley\| < 0.10 |
| | `dist_uniform` | KS vs uniform p > 0.10, excess kurtosis < −0.9 |
| **Relationship** | `rel_strong_positive` | Pearson r > 0.70, \|r − ρ\| < 0.09 |
| | `rel_strong_negative` | r < −0.70, \|r − ρ\| < 0.09 |
| | `rel_none` | \|r\| < 0.12, \|ρ\| < 0.12, quadratic R² < 0.05 |
| | `rel_nonlinear_monotonic` | \|ρ\| > 0.80, \|ρ\| − \|r\| > 0.10, linear R² < 0.72 |
| | `rel_nonlinear_nonmonotonic` | \|r\| < 0.25, quadratic R² > 0.45, \|t_curv\| > 4 |
| | `rel_heteroscedastic` | Breusch–Pagan p < 0.001, residual-spread ratio > 2.5 |
| **Time series** | `ts_increasing` | Mann–Kendall p < 0.01, τ > 0.35 |
| | `ts_decreasing` | Mann–Kendall p < 0.01, τ < −0.35 |
| | `ts_seasonal` | periodogram power > 0.30, ACF > 0.35, \|τ\| < 0.25 |
| | `ts_changepoint` | step-model SNR > 2.5, step beats line, abruptness > 0.55 |
| | `ts_anomaly_spike` | robust \|z\| > 6, no trend after removal |
| **Group** | `grp_significant_diff` | Kruskal–Wallis p < 0.001, ε² > 0.14 |
| | `grp_no_diff` | Kruskal–Wallis p > 0.30, ε² < 0.02 |
| | `grp_dominant_category` | top share > 0.55, normalised entropy < 0.75 |

## How a label is issued

1. **Subsample before verifying.** Data is cut to its final plotted size before any test runs, so the label describes what the image shows.
2. **Exactly one class must match.** A candidate that passes two classes' guards is discarded, not multi-labelled.
3. **Reject rather than mislabel.** Every guard is conservative. More candidates can be mined; a wrong label cannot be undone.

Robust statistics are used wherever the naive ones failed measurably:

- **Skewness:** moment skewness alone called 12.5% of symmetric Student-t(4) samples skewed, so the skew classes also require Bowley skewness.
- **Bimodality:** the dip test is calibrated against the uniform, so bimodality also requires a KS test against the uniform.
- **Change points:** a linear ramp looks like a change point, so level shifts also require a step model to beat a line.

Re-running the battery on the stored tables reproduces 3,998 of 4,000 test labels. The two exceptions sit within 1% of a threshold and fall to *no class*, so they are dropped, not mislabelled.

## Sources and licences

| Source | Real charts | Source tables | Licence |
|---|---:|---:|---|
| OpenML | 4,686 | 562 | per dataset (recorded per sample) |
| World Bank | 710 | 22 | CC BY 4.0 |
| Our World in Data | 415 | 20 | CC BY 4.0 |
| vega-datasets | 195 | 30 | BSD-3-Clause |
| seaborn-data | 47 | 14 | BSD-3-Clause |
| UCI ML Repository | 30 | 6 | CC BY 4.0 |
| Synthetic | 13,917 | – | CC0 (generated) |
| **Total** | **6,083 real + 13,917 synthetic** | **654** | |

Real data cannot fill every class: 719 public tables contained only ten clean inverted-U relationships. The corpus is therefore a **declared hybrid**. Every class mixes real and synthetic charts, in different proportions, so provenance is only a weak predictor of the label (a real-vs-synthetic rule alone reaches about 7.5% accuracy, against 5% chance). Per-class real/synthetic counts are in `DATASHEET.md`.

## Splits

| Split | All | Real | Per class |
|---|---:|---:|---:|
| train | 14,000 | 4,338 | 700 |
| val | 2,000 | 553 | 100 |
| test | 4,000 | 1,192 | 200 |

**Source-disjoint.** A hash of the source-table id assigns every chart from one table to one split (0 of 654 sources leak).

**Headline evaluation set.** `splits/test_real_only.jsonl` (1,192 real test charts).

**Near-duplicates across splits.** The same data can appear under two source tables, for example in both World Bank and Our World in Data. `build/14_near_duplicates.py` flags **85 real test charts** with near-duplicate data in training. Report results on the deduplicated set as well; see [RESULTS.md](RESULTS.md#near-duplicates-across-splits).

**Supplementary real training pool.** Balance caps real data at 500 per class, which left verified real candidates unused. `build/12_extra_real.py` keeps those whose source table hashes to *train* (at most 400 per class, exact duplicates removed) and asserts that no validation or test source contributes. This gives **2,545 extra real training charts**. These 2,545 charts are **in addition to** the 20,000-chart corpus and are not part of any split file.

## Files

| Path | Contents |
|---|---|
| `images/<class>/<id>.png` | 768 × 576 chart images |
| `records.jsonl` | one record per sample: label, family, chart type, split, provenance, source and licence, data table, statistics, test, reasoning chain, insight, render metadata, evidence boxes |
| `tables/metadata.parquet` | the same, flattened |
| `splits/{train,val,test}.jsonl` | per-split records |
| `splits/test_real_only.jsonl` | real-only test set (headline) |
| `instruct/{split}.jsonl` | image / prompt / response triples for instruction tuning |
| `class_index.json` | label ↔ id map, descriptions, tests, valid chart types |
| `DATASHEET.md`, `BIAS_AUDIT.md` | datasheet and shortcut-audit report |

**Evidence boxes** are in image pixel coordinates (origin top-left). They cover the plot area plus the evidence for the finding: modal bin, outlier band, change point, anomalous points, dominant wedge, and so on.

## Shortcut audit (version 2)

A *shortcut* is any feature that predicts the label without containing the statistical evidence. All eleven checks pass (`build/10_bias_audit.py`).

| Check | Result |
|---|---|
| Class and split balance | perfect |
| Render style → label | Cramér's V = 0.000 |
| Real vs synthetic from pixel statistics | AUC 0.551 (close to chance) |
| Axis text alone | macro-F1 0.369 |
| Pixel statistics alone | macro-F1 0.404 |
| Source concentration | no source above 17.2% of a class |
| **Chart-type rule (evaluation floor)** | **28.8%** |

Chart type *is* associated with the label by construction (a pie chart can only show a composition), so the chart-type rule is reported as the evaluation floor. See `BIAS_AUDIT.md` for the full report.

## What changed in version 2

| Change | Why |
|---|---|
| **No label-dependent marks** | Version 1 drew a trend line only on increasing/decreasing series, a vertical line only at change points, and circles only around anomalies. Models learned to read them (see [RESULTS.md](RESULTS.md#the-rendering-leak)). Version 2 draws none; the evidence stays in the metadata boxes. |
| **Reproducible styles** | Style is chosen from a CRC32 hash of the sample id. Version 1 used Python's salted `hash()`, which changes between runs. |
| **Dates on time axes** | Epoch-microsecond timestamps are now drawn as calendar dates. |
| **Extra renders** | Plotly and Vega-Lite re-renders of the real test charts, plus training renders (Plotly, randomised matplotlib styles) for augmentation. |