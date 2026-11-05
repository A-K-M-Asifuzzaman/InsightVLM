# InsightVLM-20K: Bias and Shortcut Audit

**Version 2 · 20,000 samples · 11 checks, all pass.**

A *shortcut* is any feature that predicts the label without containing the statistical evidence. If a model can score well from such a feature, a benchmark result says little about statistical reasoning. This audit measures eleven ways that could happen. Each has a named statistic, an explicit threshold and a pass/fail result. Reproduce with `python build/10_bias_audit.py`.

**Citation:** Asifuzzaman, A. K. M., Jisan, R. H., Alif, F. K., Iftee, I. K., & Arfat, M. (2026). *InsightVLM-20K: a chart corpus with statistically verified findings* (Version 2) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23968282

## Summary

| Group | Checks | Verdict |
|---|---|---|
| Balance (what the data contains) | 1, 2, 8, 10, 11 | All pass, none close to the limit |
| Label-leaking features (what a model could exploit) | 3, 4, 5, 6, 7, 9 | All pass; chart type carries real signal by design and is handled as the evaluation floor |

## Results

Chance level for a 20-class balanced problem is **0.05**.

| # | Test | Statistic | Value | Threshold | Margin | Result |
|---|---|---|---:|---:|---|:-:|
| 1 | Class balance | imbalance ratio | 1.000 | = 1.000 | exact | PASS |
| 2 | Split balance | max s.d. of class proportions across splits | 0.0000 | < 0.01 | exact | PASS |
| 3 | Chart type → label | pixel-free prior accuracy | 0.2883 | < 0.40 | 0.112 below | PASS |
| 3 | Chart type → label | Cramér's V (reported, no threshold) | 0.471 | none | n/a | disclosed |
| 4 | Render style → label | Cramér's V | 0.0000 | < 0.05 | 0.05 below | PASS |
| 5 | Provenance detectable from pixels | ROC AUC | 0.551 | < 0.75 | 0.199 below | PASS |
| 6 | Axis-text shortcut | macro-F1 | 0.3688 | < 0.55 | 0.181 below | PASS |
| 7 | Pixel-statistics shortcut | macro-F1 | 0.4038 | < 0.55 | 0.146 below | PASS |
| 8 | Source concentration | max top-source share of a class | 17.2% | < 35% | 17.8 pts below | PASS |
| 9 | Length confound (within family) | max/min median n | 2.04 | < 4.0 | 1.96 below | PASS |
| 10 | Subject-domain balance | max domain share | 22.8% | < 45% | 22.2 pts below | PASS |
| 11 | Category-vocabulary spread | max vocabulary share | 13.0% | < 35% | 22.0 pts below | PASS |

## What each result shows

### Balance and structure (checks 1, 2, 8, 10, 11)

- **Classes and splits are exactly balanced.** Every class has 1,000 samples, and every split holds exactly 700 / 100 / 200 per class.
- **No single source dominates.** The largest top-source share in any class is 17.2%, roughly half the 35% limit. Splits are also source-disjoint: 0 of 654 real sources appear in more than one split.
- **Subject matter and vocabulary are spread out.** No domain exceeds 22.8% and no category word list exceeds 13.0%, so a model cannot lean on topic or wording to guess the class.

### Chart type is tied to the label by design (check 3)

A pie chart can only show a composition finding, and a QQ plot only a distribution finding, so this link is structural. We disclose it instead of removing it. A rule that uses **only the chart type, with no pixels**, reaches **0.2883** accuracy, and the association strength is Cramér's V = 0.471.

That rule is the **evaluation floor**. Comparing against the 5% majority baseline would overstate every result by about 24 points (0.2883 − 0.05 = 0.2383). Any model result should be read as its gain over 0.2883.

### Render style carries no label information (check 4)

Cramér's V = 0.0000 between style and label. Styles are drawn from a CRC32 hash of the sample id, so they are reproducible and independent of the label. Version 1 used Python's salted `hash()`, which was not reproducible between runs.

### Real vs synthetic charts (check 5, plus a provenance-prior check)

About 70% of samples are synthetic, so the main objection is that a model might learn to spot generated charts instead of statistical patterns. Two measurements address this:

1. **Pixel detectability:** a classifier on low-level pixel statistics separates real from synthetic charts at ROC AUC **0.551**, where 0.5 is chance and our limit is 0.75.
2. **Provenance as a label predictor:** provenance is *not* perfectly balanced inside each class. The real share ranges from 9 of 1,000 (`rel_nonlinear_nonmonotonic`) to 500 of 1,000 (several classes), as the per-class table in `DATASHEET.md` shows. Using those counts, a rule that knows only whether a chart is real or synthetic and always guesses that group's most common class would be correct on **1,491 of 20,000 samples (0.0746)**: 500 of 6,083 real samples and 991 of 13,917 synthetic ones. That is only 2.5 points above the 0.05 chance level, so provenance is a very weak label predictor.

Because the real/synthetic mix still differs by class, the real-only test set is reported separately as the headline generalisation number.

### Text and pixel shortcuts (checks 6, 7)

- Axis text alone reaches macro-F1 **0.3688**, and pixel statistics alone reach **0.4038**. Both are under the 0.55 limit, but both are well above the 0.05 chance level, so they show that some label information is available without reading the evidence.
- These scores are best read together with the chart-type floor (0.2883): axis labels and pixel statistics partly reflect chart type, so part of this signal comes from the same structural link described above.

### Length confound (check 9)

Within each family, the median number of data points differs by at most 2.04× between classes (limit 4.0×). Length is compared within families because across families the difference is structural; an earlier run measured across families and raised a false alarm.

## Corrections made during development

Two flags in an earlier run were traced to defects in our own tooling, not to the data. Both are fixed:

1. The `domain` field was dropped during assembly.
2. Length was measured across families instead of within them.

## Residual risks we do not claim to have solved

1. **Single-label design.** Charts showing two findings were excluded, so compound findings are under-represented.
2. **Renderer monoculture.** The main set is matplotlib/seaborn. Version 2 adds Plotly and Vega-Lite re-renders of the real test charts, but cross-renderer generalisation has not yet been measured.
3. **Artificial class priors.** Perfect balance is a training convenience; real-world finding frequencies are heavily skewed.
4. **Near-duplicates across splits.** The same data can appear under two sources (for example World Bank and Our World in Data), so source-disjoint splits do not fully prevent overlap. 85 real test charts have near-duplicate data in training; report results on the deduplicated set too.
5. **Thresholds are conventions.** The pass limits above (for example 0.55 for macro-F1) are chosen cut-offs, not statistical tests.

Reproduce with `python build/10_bias_audit.py`.