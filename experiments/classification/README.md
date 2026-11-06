# Classification: good vs bad from the human scores

Can the 13 human scores separate good diagrams from bad ones, and which method and dimensions do it best?

```bash
python3 experiments/classification/run.py   # under a minute; needs numpy and scikit-learn
```

Writes `results.json` (including `test_ids`) and `predictions.csv` (each diagram's val or test score and call).

**Boundary:** this answers the question for human scores only. It uses no images, no judge scores, and no API calls,
and changes nothing outside this folder.

## Setup

- **Input:** the 13 human scores in `../human_alignment/human_ratings.json` (5 Correctness, 8 Beauty).
- **Label:** `label` in the same file, 150 of each. `negative` means the diagram was deliberately made bad with inserted
  errors; `positive` is good. The label is not derived from the human scores.
- **Test:** a fixed set of 60, with 10 per label and difficulty, scored once at the end.
- **Train/val:** 5 folds of the other 240 (192/48), balanced the same way. All methods see the same folds.
- **Choices on val only:** each method's settings (highest mean val AUC) and its cut between good and bad (highest
  accuracy on the out-of-fold val scores). The method is then refit on all 240 and tested.
- **Methods:** decision tree, logistic regression, random forest, gradient boosting, and an RBF SVM. Two rules with
  no training are the comparison: the mean of the 13 scores and the lowest of the 13.

## Results

Test intervals are 95% bootstrap intervals over the 60 test diagrams.

| Method | Val AUC | Val accuracy | Test AUC (interval) | Test accuracy | Bad caught of 30 | Clean flagged of 30 |
|---|---:|---:|---:|---:|---:|---:|
| svm_rbf | **0.963** | 0.912 | **0.996** (0.983-1.000) | 0.950 | 28 | 1 |
| random_forest | 0.961 | **0.917** | 0.983 (0.952-1.000) | 0.917 | 29 | 4 |
| gradient_boosting | 0.958 | 0.908 | 0.969 (0.926-0.996) | 0.900 | 28 | 4 |
| logistic_regression | 0.956 | 0.908 | 0.979 (0.939-1.000) | **0.967** | 29 | 1 |
| decision_tree | 0.944 | 0.896 | 0.903 (0.817-0.969) | 0.833 | 26 | 6 |
| mean_of_13 (no training) | 0.921 | 0.846 | 0.938 (0.870-0.988) | 0.900 | 25 | 1 |
| lowest_of_13 (no training) | 0.902 | 0.838 | 0.956 (0.895-0.993) | 0.900 | 27 | 3 |

1. **This test set looks easier than average, so 0.996 is too high to expect in general.** Every method except the
   decision tree scored higher on test than on val, including both rules with no training. An earlier version of this
   experiment rotated the test set over all 300 diagrams on 5 shuffles and gives a steadier estimate: SVM 0.962, random
   forest 0.961, gradient boosting 0.957, logistic regression 0.955, decision tree 0.910, mean of the 13 0.926, lowest of
   the 13 0.916. The test set stays as drawn; it is not redrawn to change the result.
2. **Combining the scores beats averaging them, by about 0.03-0.04 AUC.** On this test set the gap to the mean of the 13
   is clear for SVM (+0.058, interval +0.011 to +0.118) and logistic regression (+0.041, +0.007 to +0.083), and borderline
   for random forest (+0.045, -0.001 to +0.099). Gradient boosting's interval includes zero (+0.031, -0.023 to +0.092).
3. **SVM has the best val AUC and is the one to carry forward, but only by 0.002.** SVM, random forest, gradient boosting,
   and logistic regression cannot be separated on val. Logistic regression is the simple alternative, and its weights can be read directly.
4. **The single decision tree is the weakest on test.** It is readable, though. The tested tree reduces to: `style_uniformity
   <= 8.5` -> bad; otherwise `presence <= 9.5` and `details <= 7.5` -> bad; otherwise good.

### Which inputs matter

Val AUC / test AUC:

| Input | logistic_regression | random_forest | svm_rbf |
|---|---:|---:|---:|
| All 13 | 0.956 / 0.979 | 0.961 / 0.983 | 0.963 / 0.996 |
| Correctness 5 | 0.907 / 0.969 | 0.893 / 0.966 | 0.908 / 0.971 |
| Beauty 8 | 0.875 / 0.808 | 0.916 / 0.923 | 0.919 / 0.903 |
| Without style_uniformity | 0.931 / 0.977 | 0.942 / 0.990 | 0.953 / 0.994 |

All 13 have the best val AUC for all three methods. Shuffling one score at a time on the test set hurts most for presence,
style uniformity, and layout, and barely matters for contrast, canvas fill, resolution, alignment, and centering
(`dimensions` in `results.json`). Humans lowered dimensions together, so these importances share credit among correlated scores.

**Style uniformity is mostly a real signal.** Its human mean is 9.37 on the 150 positives, 6.13 on the 75 negatives with a
Beauty error, 5.05 on the 20 negatives with no Beauty error but an enlarged or shrunk label (which does break uniformity),
and 8.13 on the other 55 negatives. SVM's val AUC only falls from 0.963 to 0.953 without it.

### By group

SVM, the method with the best val AUC. On the test set it makes 3 mistakes:

| Difficulty | Bad caught | Clean flagged |
|---|---:|---:|
| Easy | 9 of 10 | 0 of 10 |
| Medium | 10 of 10 | 0 of 10 |
| Hard | 9 of 10 | 1 of 10 |

The test set has too few of each error type, so this uses the out-of-fold val calls on the 240: the errors missed most
often are `fade_label` (caught 10 of 14), `truncate_label` (15 of 20), and `erase_arrow` (6 of 8). A diagram can have
several errors, so a catch is not credit to any one of them.

## Checks

The test and train/val sets do not overlap, each test group has 10 diagrams, and each val fold has 8 per group. Across
5 shuffles of the labels, the test AUC falls to 0.48-0.50 (logistic regression, SVM, random forest), so the pipeline does
not leak the label.

## Limits

- **Human scores must exist.** A new diagram has none, so these models cannot score it as they are. Using them on judge
  scores is a separate experiment.
- **Only these error types were tested.** The negatives are bad because of 23 kinds of inserted error. Diagrams that are
  bad in other ways, such as new problems in LLM-generated figures, have not been tried.
- **The test set has 60 diagrams.** One diagram moves accuracy by 1.7 points, so small gaps between methods mean little.
- **The test set is not entirely fresh.** These 60 were also in the earlier rotating run. The methods and setting grids did
  not change after that run.
