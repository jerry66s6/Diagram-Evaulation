# Human alignment

Do the judge's scores agree with the human study? All 300 human-study diagrams were rated by
people (`human_ratings.json`: 13 dimensions, 1-10, one anonymized rater per diagram). HS001-HS100
are used to tune the correctness prompt; HS101-HS300 are held out for one final test.
`report.html` shows every version tried on HS001-HS100.

```bash
python3 run.py score --first 100 --wording d2      # gpt-6-luna only; resumable; raw outputs in results/
python3 run.py compare --first 100 --wording d2    # summary_d2.json, judge_scores_d2.json
python3 run.py tune --wordings v1,d1,d2            # versions side by side
python3 run.py report                              # report.html
```

Judge scores are put on 0-10 (formula scores x10, Beauty /10); human scores stay as rated.
Metrics per dimension: MAE, RMSE, share within 1 point, Spearman, Kendall, Pearson, AUC for
clean vs broken diagrams, inserted errors caught (dimension below 8), and false alarms (a clean
diagram humans rated 9+ that the judge scored below 8).

## Versions (correctness)

| Version | What it is |
|---|---|
| v1 | The evaluator as it was in git: caption sketch, item-by-item check (3 samples), formula score |
| v1r | v1 again, for run-to-run noise |
| v2 | v1 + four general prompt sentences |
| d1 | Direct: each dimension asked once with image and caption, the model gives 1-10; no sketch, no formula |
| d1r | d1 again, for run-to-run noise |
| **d2** | d1 + four general sentences from the d1 mistakes (chosen; now `src/diagram_correctness/direct.py`) |
| d3 | d2 + the caption sketch as a checklist |

## Results on HS001-HS100 (overall correctness)

| | v1 | v2 | d1 | d2 | d3 |
|---|---:|---:|---:|---:|---:|
| Spearman | 0.51 | 0.56 | 0.74 | **0.76** | 0.76 |
| Kendall | 0.36 | 0.40 | 0.57 | **0.59** | 0.58 |
| MAE (0-10) | 1.19 | 1.09 | 0.77 | **0.73** | 0.75 |
| Clean vs broken AUC (humans 0.91) | 0.73 | 0.74 | 0.81 | 0.84 | 0.85 |
| False alarms on clean diagrams | 51 | 51 | 24 | 24 | 29 |

1. **Direct scoring beats sketch + formula on all five dimensions**, with less than half the
   input tokens and a fifth of the output tokens.
2. **The formula was the main problem, not the model.** v1 found most inserted errors, but one
   detected error usually cost 2.5 points while humans often drop a dimension to 4-6; and clean
   figures lost points when the caption sketch did not match how they were drawn (exclusion boxes
   chained into the main flow in 21 of 100 sketches; one arrow serving several boxes).
3. **d2 improved on d1 beyond run-to-run noise** on Details and Connectivity, the dimensions its
   sentences targeted. d1 and d1r agree with humans equally (Spearman 0.74 both), and averaging the
   two runs gives only 0.75, so one call per dimension is enough.
4. **Adding the sketch back (d3) did not help**: the same overall, Connectivity worse, about 1.6x the cost.
5. **Beauty v6 was not tuned.** Centering and Space Use do not track the human ratings (Spearman
   about 0); Color Consistency and Style Uniformity do best (0.51, 0.42).

Caveats: the prompt was tuned on these 100 diagrams, so the result has to be confirmed on
HS101-HS300. Each diagram has one rater; humans often lowered every dimension of a diagram with
any error, which the judge, scoring each dimension on its own, does not copy.
