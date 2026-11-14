# Classification: good vs bad from the human scores

Can the 13 human scores separate good diagrams from bad ones, and which method and dimensions do it best?

```bash
python3 experiments/classification/run.py      # under a minute; needs numpy and scikit-learn
python3 experiments/classification/report.py   # report.html, from results.json
```

`run.py` writes `results.json` (including `test_ids`) and `predictions.csv` (each diagram's val or test score and call).
`report.html` is one self-contained page for sharing.

**Boundary:** this answers the question for human scores only. It uses no images, no judge scores, and no API calls,
and changes nothing outside this folder.

## Setup

- **Input:** the 13 human scores in `../human_alignment/human_ratings.json` (5 Correctness, 8 Beauty).
- **Label:** `label` in the same file, 150 of each. `negative` means the diagram was deliberately made bad with inserted
  errors; `positive` is good. The label is not derived from the human scores.
- **Test:** HS201-HS300, scored once at the end: 53 bad and 47 good; 23 easy, 41 medium, 36 hard.
- **Train/val:** 5 folds of HS001-HS200 (160/40), balanced on label and difficulty. All methods see the same folds.
- **Choices on val only:** each method's settings (highest mean val AUC) and its cut between good and bad (highest
  accuracy on the out-of-fold val scores). The method is then refit on all 200 and tested.
- **Methods:** decision tree, logistic regression, random forest, gradient boosting, and an RBF SVM. Two rules with
  no training are the comparison: the mean of the 13 scores and the lowest of the 13.

## Results

Test intervals are 95% bootstrap intervals over the 100 test diagrams. All methods use the same bootstrap draws, so the
difference between two methods is paired (`pairwise` in `results.json`).

| Method | Val AUC | Val accuracy | Test AUC (interval) | Test accuracy (interval) | Bad caught of 53 | Clean flagged of 47 |
|---|---:|---:|---:|---:|---:|---:|
| gradient_boosting | **0.961** | 0.900 | 0.975 (0.949-0.993) | 0.90 (0.84-0.95) | 50 | 7 |
| svm_rbf | 0.960 | **0.910** | 0.964 (0.931-0.990) | 0.91 (0.85-0.96) | 49 | **5** |
| random_forest | 0.954 | 0.905 | 0.974 (0.947-0.992) | 0.91 (0.85-0.96) | 49 | **5** |
| logistic_regression | 0.953 | 0.905 | **0.976** (0.947-0.997) | **0.92** (0.86-0.97) | **52** | 7 |
| decision_tree | 0.932 | 0.870 | 0.934 (0.879-0.978) | 0.86 (0.79-0.92) | 46 | 7 |
| mean_of_13 (no training) | 0.917 | 0.860 | 0.936 (0.888-0.975) | 0.85 (0.78-0.92) | 46 | 8 |
| lowest_of_13 (no training) | 0.917 | 0.840 | 0.914 (0.855-0.966) | 0.87 (0.80-0.93) | 47 | 7 |

1. **The 13 human scores separate good from bad well: AUC about 0.96-0.97, accuracy about 0.90.**
2. **Combining the scores beats averaging them.** All four of these methods are above the mean of the 13 on test: logistic
   regression +0.040 (interval +0.012 to +0.074), gradient boosting +0.039 (+0.002 to +0.080), random forest +0.037 (+0.002
   to +0.079), SVM +0.028 (+0.005 to +0.058).
3. **Those four cannot be separated, and gradient boosting is the one carried forward.** It has the best val AUC, by 0.001
   over SVM, which is how the method was to be picked; logistic regression has the best test AUC, by 0.001 over gradient
   boosting. Compared directly on the same bootstrap draws, no two of the four differ. Gradient boosting's test AUC is clearly
   above the decision tree (+0.041, +0.003 to +0.085), the mean of the 13 (+0.039, +0.002 to +0.080), and the lowest of the 13
   (+0.061, +0.013 to +0.115). On accuracy only logistic regression vs the mean of the 13 differs clearly (+0.07, +0.01 to
   +0.14). These are 21 comparisons with no correction for that, so a borderline one can be chance.
4. **The single decision tree is no better than the plain mean** (test AUC 0.934 vs 0.936). Its first split is
   `style_uniformity <= 8.5` (`decision_tree` in `results.json`).
5. **The test scores may run a little high.** Most methods score above their val AUC on test, including the plain mean (0.917 ->
   0.936), although the test set has fewer easy diagrams. Val AUC of 0.95-0.96 is the more cautious figure.

### Which inputs matter

Val AUC / test AUC:

| Input | logistic_regression | random_forest | svm_rbf |
|---|---:|---:|---:|
| All 13 | 0.953 / 0.976 | 0.954 / 0.974 | 0.960 / 0.964 |
| Correctness 5 | 0.902 / 0.948 | 0.903 / 0.953 | 0.913 / 0.938 |
| Beauty 8 | 0.880 / 0.829 | 0.926 / 0.888 | 0.919 / 0.894 |
| Without style_uniformity | 0.929 / 0.962 | 0.941 / 0.980 | 0.946 / 0.956 |

All 13 have the best val AUC for all three methods. Shuffling one score at a time on the test set hurts most for style
uniformity, presence, and details, and barely matters for contrast, alignment, and centering (`dimensions` in `results.json`).
Humans lowered dimensions together, so these importances share credit among correlated scores.

**Style uniformity is mostly a real signal.** Its human mean is 9.37 on the 150 positives, 6.13 on the 75 negatives with a
Beauty error, 5.05 on the 20 negatives with no Beauty error but an enlarged or shrunk label (which does break uniformity),
and 8.13 on the other 55 negatives. SVM's val AUC only falls from 0.960 to 0.946 without it.

### By group

Gradient boosting, the method with the best val AUC. On the test set it makes 10 mistakes:

| Difficulty | Bad caught | Clean flagged |
|---|---:|---:|
| Easy | 12 of 12 | 0 of 11 |
| Medium | 22 of 23 | 4 of 18 |
| Hard | 16 of 18 | 3 of 18 |

Each inserted error appears too few times to compare (`val_by_inserted_error` in `results.json`).

## Checks

Test and train/val do not overlap, and val covers HS001-HS200 once. Across 5 shuffles of the labels, the test AUC falls to
0.51-0.53 (logistic regression, SVM, random forest), so the pipeline does not leak the label.

## Limits

- **Human scores must exist.** A new diagram has none, so these models cannot score it as they are. Repeating this on judge
  scores is a separate experiment: it needs the judge's Beauty scores for HS146-HS300 first, and HS201-HS300 suits it as a
  test set because the correctness prompt was not tuned on them.
- **Only these error types were tested.** The negatives are bad because of 23 kinds of inserted error. Diagrams that are
  bad in other ways, such as new problems in LLM-generated figures, have not been tried.
- **The test set has 100 diagrams.** One diagram moves accuracy by 1 point, so small gaps between methods mean little.
- **The test set is not entirely fresh.** These diagrams were also in earlier exploratory runs of the same methods. The methods
  and setting grids did not change after those runs.
