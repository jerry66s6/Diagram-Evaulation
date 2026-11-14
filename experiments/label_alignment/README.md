# Are the 300 labels right? Re-checking them, then comparing the two classifiers

Positive means the original published figure; negative means the figure with one to three errors inserted on purpose.
Two ways that can be wrong: an inserted error that cannot be seen (a negative that is really good), or an original
figure that already has a clear error (a positive that is really bad). This experiment looks for both, aligns the
labels, and compares the two gradient boosting classifiers on the original and the aligned labels.

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

## Limits

- Only the 101 suspects were re-checked; an original with a flaw that neither humans nor judges noticed and both
  classifiers call good keeps its positive label.
- The check is one model pass per diagram, not human annotators; HS234 is a judgement call (a wrong count, not a
  visual error). Without it the aligned test set has 5 changes instead of 6.
- 100 test diagrams; one diagram moves accuracy by 1 point.
