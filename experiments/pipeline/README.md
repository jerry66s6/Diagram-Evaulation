# VLM pipeline: two judges, then gradient boosting

The full pipeline on HS201-HS300: the correctness judge and the Beauty judge in `src/` score each diagram on the
13 dimensions, the gradient boosting classifier from `../classification` calls it good or bad, and the call is compared
with the true label (`positive` / `negative` in `human_ratings.json`, not the human scores).

```bash
python3 experiments/pipeline/run.py score                      # gpt-6-luna; resumable; raw outputs in results/<id>/ (git-ignored)
python3 experiments/pipeline/run.py score --only HS201,HS202   # a quick trial
python3 experiments/pipeline/run.py classify                   # results.json, predictions.csv
python3 experiments/pipeline/run.py classify --beauty-tuned    # with the four tuned Beauty prompts
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

## Limits

- One call per dimension, so run-to-run noise is not measured.
- The classifier never saw judge scores. Training it on judge scores of HS001-HS200 is a separate run: the d17
  correctness scores for those exist in `../human_alignment/results/`; Beauty scores there exist for only the four
  dimensions in `../beauty_tuning/results/`.
- The test set has 100 diagrams; one diagram moves accuracy by 1 point.
