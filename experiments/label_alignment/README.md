# Are the 300 labels right? Re-checking them, then comparing the two classifiers

Positive means the original published figure; negative means the figure with one to three errors inserted on purpose.
Two ways that can be wrong: an inserted error that cannot be seen (a negative that is really good), or an original
figure that already has a clear error (a positive that is really bad). This experiment looks for both, aligns the
labels, and compares the classifiers on the original and the corrected labels. Two label sets that follow the
human scores (the classifiers' consensus, and human mean below 8) were also tried.

```bash
python3 experiments/label_alignment/run.py   # results.json; no API calls
```

## How the labels were re-checked

1. **Suspects (101 of 300).** Negatives where the humans or the judges did not lower any targeted dimension below 8,
   or that either classifier (10-fold out-of-fold, human or judge scores) called good: 38. Positives where the humans
   or the judges gave any dimension 5 or less, or that either classifier called bad: 63. The rule treats human and
   judge signals the same way.
2. **Visual check.** Each suspect was looked at again by Claude (Opus 5.5, five parallel passes), from the image, the
   caption and, for negatives, the inserted errors with their pixel boxes, without seeing any human or judge score.
   One standard for both labels: a diagram is bad if it has at least one clear error of the kinds that were inserted
   (overlap, arrow missing, reversed or through a box, truncated or wrong label, text too small or faint to read,
   mixed line styles, ...). Small but readable text, mild blur and dense content do not count.
3. **Aligned label:** negative if the check found a clear error, positive otherwise; diagrams not reviewed keep their
   label. Every verdict and what was seen are in `review.csv`. HS186 and HS262 were checked a second time.

## What changed

- **No negative was wrong.** All 38 suspicious negatives have at least one inserted error that can be seen or that
  contradicts the caption, including those humans scored high (HS187, HS061, HS101, HS293).
- **11 positives are really bad** (7% of the 150 originals), 6 of them in the test set:

| Id | Test? | What is wrong with the original figure |
|---|---|---|
| HS029 | | branch labels a few pixels tall, unreadable |
| HS095 | | one arrow reversed; exclusion lists about 4 px tall |
| HS142 | | arrows in three unrelated styles |
| HS185 | | three panels squeezed into 761x289 px, lists unreadable |
| HS186 | | exclusion arrows point into the main flow |
| HS234 | test | a count label is wrong (judgement call: layout is fine) |
| HS250 | test | formula with a missing term; arrow stops short of its box |
| HS252 | test | 27 tables in 600x707 px, digits unreadable |
| HS261 | test | a connector runs through a heading and two labels |
| HS262 | test | the split bar never connects to the two result boxes |
| HS295 | test | arrow from "excluded" into "included"; blank stage bars |

## The classifiers on the original and the corrected labels

Test set HS201-HS300. For each label set every classifier is retrained on the labels of HS001-HS200 with fixed
settings, so only the labels change: the settings 5-fold val chose on the original labels (200 trees, learning rate
0.03; depth 2 on human scores, depth 3 on judge scores). The good/bad cut is chosen on val as in `../classification`.
Judge scores are the saved ones (four tuned Beauty prompts); no judge was re-run.

| Classifier | Original labels: AUC / accuracy | Corrected labels: AUC (95% interval) | Corrected labels: accuracy (95% interval) | Bad caught / good flagged (corrected) |
|---|---:|---:|---:|---:|
| **Trained on human scores, given judge scores (the pipeline)** | 0.849 / 0.74 | **0.901** (0.836-0.952) | **0.82** (0.74-0.89) | 53 of 59 / 12 of 41 |
| Trained on judge scores, given judge scores | 0.911 / 0.82 | 0.889 (0.814-0.948) | 0.83 (0.75-0.90) | 47 of 59 / 5 of 41 |
| Trained on human scores, given human scores | 0.975 / 0.90 | 0.976 (0.946-0.996) | 0.92 (0.86-0.97) | 56 of 59 / 5 of 41 |

1. **The pipeline gains most:** accuracy 0.74 -> 0.82, AUC 0.849 -> 0.901. Of the 6 test originals that are really
   bad it calls 5 bad, so part of what counted as good diagrams wrongly flagged were bad diagrams.
2. **The classifier trained on judge scores barely moves** (accuracy 0.82 -> 0.83, AUC 0.911 -> 0.889): it calls only
   2 of the 6 bad originals bad.
3. **Human scores stay the ceiling:** 0.92 accuracy on the corrected labels.

## Other label versions tried

Two relabelings that follow the human scores, with the same fixed settings:

- **Both classifiers:** flip a label when both classifiers disagree with it (10-fold out-of-fold calls over all 300,
  P(bad) > 0.5): 10 negatives become positive, 2 positives negative; 3 changes in the test set.
- **Human mean < 8:** bad when the mean of the 13 human scores is below 8: 46 negatives become positive, 10 positives
  negative; 19 changes in the test set (44 bad, 56 good).

| Labels | Pipeline: AUC / accuracy | Trained on judge scores | Trained on human scores |
|---|---:|---:|---:|
| original | 0.849 / 0.74 | 0.911 / 0.82 | 0.975 / 0.90 |
| corrected (visual check) | **0.901 / 0.82** | 0.889 / 0.83 | 0.976 / 0.92 |
| both classifiers | 0.868 / 0.77 | 0.926 / 0.86 | 0.981 / 0.92 |
| human mean < 8 | 0.848 / 0.83 | 0.867 / 0.77 | 0.971 / 0.91 |

The visual check is the label set kept: fewest changes, each with a visible error, made without seeing any score,
and one standard for both labels. The other two are computed from the human scores or from the classifiers' own calls,
so the classifiers reading those scores are partly graded on their own output. With human mean < 8 the pipeline's
accuracy rises (0.83) but its AUC does not (0.848): the ranking is no better, the cut fits the new base rate.

## Limits

- The 11 changes were made by a model pass; a human look at them is still to do.
- The threshold 8 is one choice; other thresholds were not tried.
- Only the 101 suspects were re-checked; an original with a flaw that neither humans nor judges noticed and both
  classifiers call good keeps its positive label.
- The check is one model pass per diagram, not human annotators; HS234 is a judgement call (a wrong count, not a
  visual error). Without it the corrected test set has 5 changes instead of 6.
- 100 test diagrams; one diagram moves accuracy by 1 point.
