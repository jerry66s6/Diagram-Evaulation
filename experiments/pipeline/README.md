# VLM pipeline: two judges, then gradient boosting

The full pipeline on HS201-HS300: the correctness judge and the Beauty judge in `src/` score each diagram on the
13 dimensions, the gradient boosting classifier from `../classification` calls it good or bad, and the call is compared
with the true label (`positive` / `negative` in `human_ratings.json`, not the human scores).

```bash
python3 experiments/pipeline/run.py score                      # gpt-6-luna; resumable; raw outputs in results/<id>/ (git-ignored)
python3 experiments/pipeline/run.py score --only HS201,HS202   # a quick trial
python3 experiments/pipeline/run.py classify                   # results.json, predictions.csv
python3 experiments/pipeline/run.py classify --beauty-tuned    # with the four tuned Beauty prompts
python3 experiments/pipeline/run.py score --start 1 --end 200 --dimensions color_consistency,contrast,style_uniformity,resolution
python3 experiments/pipeline/run.py train-on-judges [--beauty-tuned]   # gradient boosting trained on judge scores
```

`predictions.csv` keeps each diagram's 13 judge scores, its P(bad), and the call.

## Setup

- **Judges:** `judge_correctness_dimension` (`src/diagram_correctness/direct.py`) for Layout, Connectivity, Presence,
  Details, Legibility, and `judge_beauty_dimension` (`src/diagram_correctness/beauty.py`, the prompts merged in PR #13)
  for the 8 Beauty dimensions. One gpt-6-luna call per dimension: 1,300 calls, about 2.1M input and 0.68M output tokens.
- **Scale:** Beauty scores (0-100) are divided by 10 to match the 1-10 scale of the human scores.
- **Classifier:** gradient boosting exactly as in `../classification`: trained on the human scores of HS001-HS200
  (learning rate 0.03, depth 2, 200 trees), with the good/bad cut (P(bad) > 0.433) chosen there on val. Here it gets
  the judges' scores instead of human scores.
- **Test:** HS201-HS300, 53 bad and 47 good.

## Results

| | Test AUC (95% interval) | Accuracy (95% interval) | F1 | Bad caught | Clean flagged |
|---|---:|---:|---:|---:|---:|
| **Pipeline: gradient boosting on judge scores** | **0.842** (0.764-0.912) | **0.75** (0.67-0.83) | 0.766 | 41 of 53 | 13 of 47 |
| Reference: the same classifier on human scores | 0.975 (0.949-0.993) | 0.90 (0.84-0.95) | 0.909 | 50 of 53 | 7 of 47 |
| Reference: mean of the 13 judge scores (no training) | 0.903 (0.841-0.955) | - | - | - | - |

1. **The pipeline catches 41 of 53 bad diagrams and wrongly flags 13 of 47 good ones** (AUC 0.842, accuracy 0.75).
2. **It is clearly below the same classifier on human scores:** AUC -0.133 (-0.207 to -0.069), accuracy -0.15
   (-0.24 to -0.06), on the same bootstrap draws.
3. **Even the plain mean of the 13 judge scores ranks better** than the classifier on them: AUC -0.061 (-0.122 to
   -0.005) for the classifier. The classifier learned how humans use the 13 scores, and the judges use them
   differently, so the rules do not carry over.
4. **Where the judges differ from humans** (same 100 diagrams, `per_dimension` in `results.json`): the judges rate
   Centering 9.63 vs humans 7.61, Resolution 9.02 vs 7.30, and Canvas Fill 9.98 vs 8.84, and rank Centering, Space Use,
   Canvas Fill, and Alignment almost unlike humans (Spearman 0.13-0.21). Style uniformity, the classifier's strongest
   input, separates good from bad less well with judge scores (AUC alone 0.717 vs 0.802 with human scores).
5. **Mistakes fall most on hard diagrams:** 6 of the 12 missed bad diagrams and 7 of the 13 wrongly flagged good ones.

## With the tuned Beauty prompts

`classify --beauty-tuned` swaps in the four Beauty dimensions tuned in `../beauty_tuning` (Centering, Space Use,
Alignment, Canvas Fill): AUC 0.849 (0.775-0.916), accuracy 0.74, 40 of 53 bad caught, 13 of 47 good flagged
(`results_beauty_tuned.json`). The mean of the 13 judge scores rises to AUC 0.925; the classifier trained on human
scores barely moves.

## Gradient boosting trained on judge scores

`train-on-judges [--beauty-tuned]` trains gradient boosting on the judges' scores of HS001-HS200 instead of the human
scores, with the same split, folds, setting grid, and cut rule as `../classification`, and tests it once on HS201-HS300.
Judge scores for HS001-HS200: Correctness from the study's d17 outputs (`../human_alignment/results`; the prompt is
identical to the one in `src`), Color Consistency, Contrast, Style Uniformity, and Resolution scored here with the
prompts in `src` (800 calls), and the four tuned dimensions from `../beauty_tuning` (current or tuned prompts).

| HS201-HS300 | Beauty prompts in `src`: AUC / accuracy | Four tuned Beauty prompts: AUC / accuracy | Bad caught / good flagged (tuned) |
|---|---:|---:|---:|
| **Gradient boosting trained on judge scores** | 0.879 / 0.79 | **0.911 / 0.82** | 43 of 53 / **8 of 47** |
| Gradient boosting trained on human scores, given judge scores | 0.842 / 0.75 | 0.849 / 0.74 | 40 of 53 / 13 of 47 |
| Mean of the 13 judge scores (no training) | 0.903 / 0.78 | 0.925 / 0.78 | 49 of 53 / 18 of 47 |
| Reference: gradient boosting trained and tested on human scores | 0.975 / 0.90 | 0.975 / 0.90 | 50 of 53 / 7 of 47 |

Intervals are in `results_trained_on_judges.json` and `results_trained_on_judges_beauty_tuned.json`.

1. **Training on judge scores helps.** With the tuned Beauty prompts, test AUC rises from 0.849 to 0.911 (+0.062, 95%
   interval +0.005 to +0.126) and accuracy from 0.74 to 0.82 (+0.08, -0.01 to +0.17), with good diagrams wrongly flagged
   falling from 13 to 8 of 47. With the prompts in `src` the gain is smaller and not clear (AUC +0.037, -0.025 to +0.104).
2. **The best pipeline is the tuned Beauty prompts with gradient boosting trained on judge scores:** AUC 0.911
   (0.853-0.958), accuracy 0.82 (0.74-0.89). It also has the higher val AUC of the two (0.952 vs 0.948).
3. **It is still below human scores:** AUC -0.064 (-0.120 to -0.016), accuracy -0.08 (-0.16 to +0.00).
4. **The plain mean of the judge scores ranks about as well** (AUC 0.925; difference -0.013, -0.055 to +0.029), but at
   its val-chosen cut it flags 18 of 47 good diagrams, against 8 for the trained model.

The val AUC of the two rows trained on human scores is measured on human scores.

## Limits

- One call per dimension, so run-to-run noise is not measured.
- HS201-HS300 has now served as the test set for the first pipeline, the Beauty tuning, and this comparison. Both
  versions here were planned before the test run and both are reported; nothing was chosen on the test set.
- The test set has 100 diagrams; one diagram moves accuracy by 1 point.
