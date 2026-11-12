# Beauty human alignment: experimental v2

Completed 2026-11-12 with `dev-gpt-6-luna`, judging eight Beauty dimensions
independently. Human scores stay on 1-10; judge scores use
`1 + 9 * score / 100` (0 -> 1, 50 -> 5.5, 100 -> 10).

## Scope and results

Training: HS001-HS100. Mini-validation: a fixed 20-image subset of full validation,
HS101-HS200. HS201-HS300 were not scored or used for selection. All comparisons use
the same corrected human labels and frozen local prompt versions.

| Subset | Version | Images | RMSE | MAE | Pooled Pearson |
|---|---|---:|---:|---:|---:|
| Mini-validation | Baseline | 20 | 2.1976 | 1.6121 | 0.1813 |
| Mini-validation | Current prompts (v1) | 20 | 2.7124 | 1.8766 | 0.1812 |
| Mini-validation | Experimental v2 | 20 | **1.8605** | **1.3091** | **0.2870** |
| Full validation | Baseline | 100 | 2.5508 | 1.8261 | 0.1921 |
| Full validation | Experimental v2 | 100 | **2.3160** | **1.5418** | **0.3204** |

V1 failed screening and did not receive full validation. V2 completed 1,600 calls
(100 training + 100 validation images), reusing its mini-validation scores.
Metrics pool image/dimension pairs; full-validation RMSE improved **9.2%**.

## Five v2 changes

- Centering: normalize opposite-margin imbalance by canvas width/height.
- Space Use: distinguish intentional grouped symbols from accidental crowding.
- Contrast: count clear outlines as well as fills when assessing box visibility.
- Style Uniformity: compare semantically equivalent elements and connector roles.
- Color Consistency: allow coherent palettes with more than three hue families.

Full-validation RMSE improved most for Centering (3.4990 -> 2.8699) and Contrast
(2.4740 -> 1.9193). Style Uniformity slightly worsened (2.4693 -> 2.5882).
Alignment, Resolution, and Canvas Fill prompts were unchanged; all dimensions
were re-judged, so score differences also include judge variability.

## Uncertainty and limitations

For full validation, v2-minus-baseline RMSE change was **-0.2348**, with a paired
diagram-bootstrap 95% interval of **[-0.3668, -0.1079]**. The two-sided paired
loss-change permutation p-value was **0.000200**, or **0.001000** after a
five-candidate Bonferroni adjustment (10,000 resamples). This is not a p-value
for judge-human correlation.

These are exploratory validation results, not held-out test results. Resampling
keeps all eight dimensions of a diagram together, but does not estimate human-label
uncertainty or judge rerun variability. V2 remains experimental; the source
evaluator was not replaced. Private labels, images, backend, and raw run outputs
remain local. See [the workflow README](README.md) for the runner and dashboard.
