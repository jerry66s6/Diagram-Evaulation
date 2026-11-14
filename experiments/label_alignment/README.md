# Are the 300 labels right? Re-checking them, then comparing the two classifiers

Positive means the original published figure; negative means the figure with one to three errors inserted on purpose.
Two ways that can be wrong: an inserted error that cannot be seen (a negative that is really good), or an original
figure that already has a clear error (a positive that is really bad). This experiment looks for both, aligns the
labels, and compares the two gradient boosting classifiers on the original and the aligned labels. It then tries two
label sets that follow the human scores (the classifiers' consensus, and human mean below 8).

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

## The two classifiers on the original and the aligned labels

Test set HS201-HS300; the same split, folds, setting grid and cut rule as `../classification`, retrained for each
label set. Judge scores use the four tuned Beauty prompts (the best pipeline in `../pipeline`).

| Labels | Classifier | Test AUC (95% interval) | Accuracy (95% interval) | Bad caught | Good flagged |
|---|---|---:|---:|---:|---:|
| original | trained on human scores | 0.975 (0.949-0.993) | 0.90 (0.84-0.95) | 50 of 53 | 7 of 47 |
| original | trained on judge scores | 0.911 (0.853-0.958) | 0.82 (0.74-0.89) | 43 of 53 | 8 of 47 |
| aligned | trained on human scores | 0.972 (0.940-0.995) | **0.93** (0.88-0.97) | 55 of 59 | 3 of 41 |
| aligned | trained on judge scores | 0.864 (0.780-0.933) | **0.79** (0.71-0.87) | 48 of 59 | 10 of 41 |

Human minus judge: original labels AUC +0.064 (+0.016 to +0.120), accuracy +0.08 (+0.00 to +0.16); aligned labels
AUC +0.107 (+0.041 to +0.187), accuracy +0.14 (+0.07 to +0.22).

1. **Wrong labels are few and do not explain the pipeline's lower accuracy.** Aligning them widens the gap between
   the two classifiers instead of closing it: the classifier on human scores goes up (0.90 -> 0.93), the one on judge
   scores goes down (0.82 -> 0.79, AUC 0.911 -> 0.864).
2. **Humans see the flaws of the bad originals; the judges mostly do not.** 9 of the 11 bad originals have a human
   score of 5 or less, 7 a judge score of 5 or less, but the judges' low scores move the classifier less. Of the 6 in the test set, the classifier on human scores calls 3
   bad, the one on judge scores 2. On the other 94 test diagrams, aligned training labels raise the human classifier's
   accuracy from 0.93 to 0.96 and leave the judge classifier at 0.82-0.83.
3. **The label definition holds:** inserted errors are reliably visible, and "original figure" means good for 93% of
   the originals. The gap is in the judge scores (see `../pipeline`, "Why the accuracy is lower").

## Labels that follow the human scores

The classifier is meant to call a diagram bad when people find it bad, so two more label sets follow the human
scores instead of how the diagram was made. Both classifiers are retrained on each (same split, folds and grid).

- **Both classifiers:** flip a label when both classifiers disagree with it (10-fold out-of-fold calls over all 300,
  P(bad) > 0.5). 10 negatives become positive (HS035, HS047, HS054, HS061, HS094, HS101, HS135, HS187, HS256, HS293)
  and 2 positives become negative (HS029, HS241); 3 changes are in the test set.
- **Human mean < 8:** bad when the mean of the 13 human scores is below 8. 46 negatives become positive and 10
  positives negative; the test set has 44 bad and 56 good.

| Labels | Changed (in test) | Classifier | Test AUC (95% interval) | Accuracy (95% interval) | Bad caught | Good flagged |
|---|---:|---|---:|---:|---:|---:|
| original | 0 | trained on human scores | 0.975 (0.949-0.993) | 0.90 (0.84-0.95) | 50 of 53 | 7 of 47 |
| original | 0 | trained on judge scores | 0.911 (0.853-0.958) | **0.82** (0.74-0.89) | 43 of 53 | 8 of 47 |
| visual check | 11 (6) | trained on human scores | 0.972 (0.940-0.995) | 0.93 (0.88-0.97) | 55 of 59 | 3 of 41 |
| visual check | 11 (6) | trained on judge scores | 0.864 (0.780-0.933) | **0.79** (0.71-0.87) | 48 of 59 | 10 of 41 |
| both classifiers | 12 (3) | trained on human scores | 0.972 (0.943-0.992) | 0.93 (0.88-0.97) | 49 of 52 | 4 of 48 |
| both classifiers | 12 (3) | trained on judge scores | 0.939 (0.887-0.980) | **0.84** (0.76-0.91) | 43 of 52 | 7 of 48 |
| human mean < 8 | 56 (19) | trained on human scores | 0.980 (0.958-0.996) | 0.90 (0.84-0.95) | 36 of 44 | 2 of 56 |
| human mean < 8 | 56 (19) | trained on judge scores | 0.874 (0.794-0.940) | **0.78** (0.70-0.85) | 30 of 44 | 8 of 56 |

Human minus judge, accuracy: original +0.08 (+0.00 to +0.16), visual check +0.14 (+0.07 to +0.22), both classifiers
+0.09 (+0.01 to +0.17), human mean < 8 +0.12 (+0.04 to +0.20).

1. **Labels set by the human scores do not help the judge pipeline.** With "human mean < 8" the classifier on judge
   scores gets accuracy 0.78 and AUC 0.874, below the original labels (0.82, 0.911). The judges' 13 scores follow the
   inserted errors better than they follow people's overall verdict: of the 19 test diagrams whose label changes, it
   gets 53% right (human-score classifier 74%).
2. **Only the "both classifiers" labels raise it** (0.84, AUC 0.939), and those labels were partly made by the judge
   classifier itself, so part of the gain is built in. The change is 3 test diagrams; the interval overlaps the
   original result.
3. **The human-score rows of the last two label sets are circular:** those labels are computed from the same human
   scores the classifier reads. With "human mean < 8" the plain rule "mean of the 13 human scores below 8" is right by
   definition (accuracy 1.00); gradient boosting gets 0.90 only because shallow trees on 200 diagrams approximate a mean
   poorly.

## Limits

- The threshold 8 is one choice; other thresholds were not tried.
- Only the 101 suspects were re-checked; an original with a flaw that neither humans nor judges noticed and both
  classifiers call good keeps its positive label.
- The check is one model pass per diagram, not human annotators; HS234 is a judgement call (a wrong count, not a
  visual error). Without it the aligned test set has 5 changes instead of 6.
- 100 test diagrams; one diagram moves accuracy by 1 point.
