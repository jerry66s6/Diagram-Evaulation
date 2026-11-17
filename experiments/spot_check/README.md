# Low-confidence calls on the 6,000 dataset diagrams: pick, group, spot-check

The labels of the 6,000 dataset diagrams are not reliable enough to grade the pipeline on all of them (in the 300
study diagrams 7% of the originals already had a clear error). So only the calls the classifier is unsure of are
analyzed, and a sample of them is checked by hand.

**Scope:** choosing the low-confidence diagrams, grouping them, and drawing the sample. The manual verdicts are not
part of this step.

```bash
python3 experiments/spot_check/run.py   # spot_check.csv, summary.json, spot_check.html; no API calls
```

## Low confidence

Every diagram has a P(bad) from the final classifier (`../pipeline/dataset_predictions.csv`); it is called bad above
0.361. A call is low confidence when P(bad) is within 0.1 of that cut (0.261 to 0.461): **691 of 6,000 diagrams
(12%)**. There the call matches the dataset label 60% of the time, against 86% for the other 5,309.

## Groups

| Call against the label | Diagrams |
|---|---:|
| Original, called bad | 192 |
| Errors inserted, called good | 86 |
| Errors inserted, called bad | 87 |
| Original, called good | 326 |

- **Hard diagrams are twice as often unsure:** 139 of 678 hard diagrams (21%) are low confidence, against 10% of easy
  (155 of 1,505) and medium (397 of 3,817).
- **Inserted errors called good** (86): the errors are mostly legibility (31) and arrow errors (31), then layout (23).
- **Judge scores:** 112 of the 192 originals called bad have no judge score of 5 or less; the call comes from several
  middling scores, not from one clear problem.

`summary.json` has the full counts (inserted error kinds and low judge dimensions per group).

## Spot check

25 diagrams drawn at random from each group (seed 0), 100 in all, in random order.

- `spot_check.html`: one card per diagram, with the image and the caption. A closed box under each card holds the
  label, the call, P(bad), the inserted errors and what the judges said about any dimension they scored 5 or less, so
  the diagram can be judged before seeing them. The images load from `diagrams/images` (not in git), so open the page
  from inside the project folder.
- `spot_check.csv`: all 691 low-confidence diagrams; the 100 in the sample have a `spot_check` number. Write good or
  bad in `verdict` (and anything else in `note`).
