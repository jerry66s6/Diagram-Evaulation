# Beauty prompt tuning against the human ratings

The four Beauty dimensions that agreed least with humans on HS201-HS300 (Centering, Space Use, Alignment, Canvas Fill)
were re-prompted and compared with the current prompts in `src/diagram_correctness/beauty.py`.

```bash
python3 experiments/beauty_tuning/run.py score --versions current,rater --start 1 --end 200   # gpt-6-luna; resumable
python3 experiments/beauty_tuning/run.py compare --start 101 --end 200
python3 experiments/beauty_tuning/run.py test --choice centering=rater,space_use=rater,alignment=rater,canvas_fill=rater
```

`results.json` has the validation and test tables. Raw outputs are in `results/` (git-ignored).

## Setup

- **Splits:** HS001-HS100 training, HS101-HS200 validation, HS201-HS300 test. For each dimension the version with the
  higher validation Spearman is chosen, and only that choice is scored on the test set, once.
- **current:** the prompts in `src` (0-100, divided by 10). Its test scores are the pipeline's (`../pipeline/results`),
  made with the same prompts.
- **rater:** the question and the two end labels the human raters saw in the study (`user-study/lib/metrics.ts`),
  scored 1-10 as they did, with none of the current prompts' isolation rules ("if the flaws belong to another
  dimension, score high").
- gpt-6-luna, one call per diagram and dimension: 1,600 calls for HS001-HS200, 400 for the test.

## Results

Validation (HS101-HS200) chose `rater` for all four dimensions. Spearman with the human scores:

| Dimension | Validation, current | Validation, rater | Test, current | Test, rater | Test change (95% interval) |
|---|---:|---:|---:|---:|---:|
| Centering | 0.09 | 0.54 | 0.16 | 0.24 | +0.08 (-0.19 to +0.34) |
| Space use | 0.27 | 0.56 | 0.13 | **0.42** | **+0.29 (+0.07 to +0.50)** |
| Alignment | 0.40 | 0.59 | 0.21 | **0.43** | **+0.22 (+0.01 to +0.42)** |
| Canvas fill | 0.17 | 0.30 | 0.19 | 0.32 | +0.13 (-0.12 to +0.28) |
| Beauty overall (mean of 8) | | | 0.53 | 0.56 | +0.03 (-0.10 to +0.16) |

Test MAE: Centering 2.19 -> 1.76, Space use 2.29 -> 1.68, Alignment 1.82 -> 1.50, Canvas fill 1.14 -> 1.41. The interval
is a paired bootstrap over the 100 test diagrams.

1. **Space Use and Alignment improve clearly on the test set.** Centering and Canvas Fill improve less, and their
   intervals include zero. Centering's large validation gain (0.09 -> 0.54) did not hold.
2. **Canvas Fill now varies:** the current prompt scored 99 of 100 test diagrams at 10; the rater prompt spreads its
   scores much like humans do (SD 1.49 vs human 2.11).
3. **The rater prompt is harsh:** it scores 0.8 to 1.2 points below humans on the test set, where the current prompts
   were mostly lenient (Centering +2.0, Canvas Fill +1.1).
4. **Part of the gain comes from marking down broken diagrams, as humans do.** The current prompts tell the judge to
   ignore flaws that belong to other dimensions; the rater prompt does not. On the 47 unmodified test diagrams alone the
   rater prompt still agrees better (Space Use 0.46 vs -0.02, Alignment 0.46 vs 0.35, Centering 0.29 vs 0.17).

## Effect on the pipeline

`python3 experiments/pipeline/run.py classify --beauty-tuned` swaps the four tuned dimensions into the pipeline:

| HS201-HS300 | Current prompts | Tuned prompts |
|---|---:|---:|
| Gradient boosting on judge scores: AUC | 0.842 | 0.849 |
| Gradient boosting on judge scores: accuracy | 0.75 | 0.74 |
| Mean of the 13 judge scores: AUC | 0.903 | 0.925 |

The classifier, trained on human scores, barely changes. Better Beauty scores help the plain mean, but not a model whose
rules were learned from human scores.

## Limits

- One version tried against the current prompts; one call per diagram and dimension, so rerun noise is not measured.
- HS201-HS300 has now been used once for this test and once for the pipeline; it should not be used to choose further prompts.
- `src` is unchanged: the tuned prompts live in this experiment only.
