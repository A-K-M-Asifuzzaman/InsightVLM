# InsightVLM-20K — Bias and Shortcut Audit

Nine measured failure modes, each with a named statistic and an explicit
threshold. A *shortcut* is any feature that predicts the label without
containing the statistical evidence.

| # | Test | Statistic | Value | Threshold | Result |
|---|---|---|---|---|---|
| 1 | Class balance | imbalance ratio | 1.000 | = 1.000 | PASS |
| 2 | Split balance | max s.d. of proportions | 0.0000 | < 0.01 | PASS |
| 3 | Chart type -> label | Cramer's V / prior acc | 0.471 / 0.2883 | acc < 0.40 | PASS |
| 4 | Render style -> label | Cramer's V | 0.0000 | < 0.05 | PASS |
| 5 | Provenance detectable | ROC AUC | 0.551 | < 0.75 | PASS |
| 6 | Axis-text shortcut | macro-F1 | 0.3688 | < 0.55 | PASS |
| 7 | Pixel-stat shortcut | macro-F1 | 0.4038 | < 0.55 | PASS |
| 8 | Source concentration | max top-source share | 17.2% | < 35% | PASS |
| 9 | Length confound (within family) | max/min median n | 2.04 | < 4.0 | PASS |
| 10 | Subject-domain balance | max domain share | 22.8% | < 45% | PASS |
| 11 | Category-vocabulary spread | max vocabulary share | 13.0% | < 35% | PASS |

## Interpretation

**Chart type is associated with the label by construction and this is not a
defect.** A pie chart can only carry a composition finding; a QQ plot only a
distribution finding. The association is disclosed and handled by reporting
the pixel-free chart-type prior (0.2883) as the evaluation floor rather
than the 5% majority baseline, which would overstate every result by roughly
24 points.

**Provenance is the bias that mattered most**, because ~70% of samples are
synthetic. Two safeguards: provenance is balanced *within* every class, so
'looks synthetic' cannot act as a proxy for any label; and a held-out
real-only probe is reported separately as the headline generalisation number.

## Residual risks we do not claim to have solved

1. Single-label design: charts showing two findings were excluded, so the
   dataset under-represents compound findings.
2. Renderer monoculture: matplotlib/seaborn only, so a model may learn
   renderer-specific conventions. Cross-renderer generalisation is untested.
3. Class priors are artificial: perfect balance is a training convenience;
   real-world finding frequencies are heavily skewed.

Reproduce with `python build/10_bias_audit.py`.
