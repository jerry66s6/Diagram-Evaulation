# The 6,000 dataset diagrams to label again by hand

The final pipeline gives every dataset diagram a P(bad). Two groups are labeled again by people in the user-study site
(`../../user-study`, "Review mode", deployed as a separate site): the diagrams the pipeline is unsure of, and as many
of the least confident high-confidence ones, to check whether high-confidence calls can be trusted. The dataset labels
are not used anywhere here: the band comes from the 100 test diagrams, and the reviewers check our call.

**Scope:** choosing the diagrams and preparing them for the site. The reviewing itself is not part of this step.

```bash
python3 experiments/spot_check/run.py   # spot_check.csv, band.json, review_manifest.csv (for the site, not in git); no API calls
```

## Low confidence: split conformal prediction on the 100 test diagrams

Standard split conformal prediction at 90% coverage, calibrated on HS201-HS300 with the corrected labels
(`../pipeline/predictions_final.csv`):

1. Score of each test diagram: 1 minus the classifier's probability of its true label (1 - P(bad) for a bad diagram,
   P(bad) for a good one).
2. qhat = the ceil(101 x 0.9) = 91st smallest score = **0.708**.
3. A diagram's prediction set holds every label whose score is at most qhat. It holds both good and bad, so the call is
   unsure, when **0.292 <= P(bad) <= 0.708**.

**1,135 of the 6,000 diagrams** are low confidence.

## High confidence, checked too

On the 100 test diagrams the high-confidence calls are not error-free: 77 diagrams, AUC 0.946, accuracy 0.88, 9 errors.
So high-confidence calls are checked as well. Their confidence is max(P(bad), 1 - P(bad)); the **1,135 least
confident** (as many as the low-confidence ones; ties broken by id, 3 tied diagrams left out) are reviewed:
P(bad) 0.146-0.291 (726 called good) and 0.713-0.854 (409 called bad).

| Part | Dataset diagrams | Reviewed | Test diagrams | Test errors |
|---|---:|---:|---:|---:|
| Low confidence | 1,135 | all | 23 | 9 |
| High confidence, least confident | 1,135 | all | 17 | 5 |
| High confidence, the rest | 3,730 | no | 60 | 4 |

**2,270 diagrams are reviewed**, in 76 sets of 29 or 30.

## Files

- `spot_check.csv`: the 2,270 reviewed diagrams with their part, P(bad), our call and the 13 judge scores.
- `band.json`: qhat, the band, the high-confidence limit, and the counts above.
- `review_manifest.csv` (git-ignored): the same diagrams with image path and caption, for the site's importer.

## Limits

- 100 test diagrams set the band; with 80% coverage instead of 90% it would shrink to 0.491-0.509 (53 diagrams), so
  the band is sensitive to the coverage chosen.
- The 90% guarantee assumes the test diagrams and the 6,000 come from the same distribution. They do not quite: the
  test set has a third of each difficulty and half bad diagrams; the 6,000 are 25% easy, 64% medium, 11% hard.
- The 3,730 most confident calls are not reviewed; on the test set that part still had 4 errors in 60.
