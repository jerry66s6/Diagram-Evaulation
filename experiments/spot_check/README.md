# Low-confidence calls on the 6,000 dataset diagrams, to label again by hand

The final pipeline gives every dataset diagram a P(bad). The diagrams it is unsure of are labeled again by people in
the user-study site (`../../user-study`, "Review mode"). The dataset labels are not used anywhere here: the band comes
from the 100 test diagrams, and the reviewers check our call.

**Scope:** choosing the low-confidence diagrams and preparing them for the site. The reviewing itself is not part of
this step.

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

**1,135 of the 6,000 diagrams** are in that band (836 called bad, 299 called good; easy 245, medium 668, hard 222).

## Files

- `spot_check.csv`: the 1,135 diagrams with P(bad), our call and the 13 judge scores.
- `band.json`: qhat, the band and the counts.
- `review_manifest.csv` (git-ignored): the same diagrams with image path and caption, for the site's importer.

## Limits

- 100 test diagrams set the band; with 80% coverage instead of 90% it would shrink to 0.491-0.509 (53 diagrams), so
  the band is sensitive to the coverage chosen.
- The 90% guarantee assumes the test diagrams and the 6,000 come from the same distribution. They do not quite: the
  test set has a third of each difficulty and half bad diagrams; the 6,000 are 25% easy, 64% medium, 11% hard.
