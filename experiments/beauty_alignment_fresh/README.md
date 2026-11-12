# Fresh beauty alignment and image review

See [the short v2 results summary](RESULTS.md) for the completed comparison.

This experiment does not import historical experiment results or tuned prompts.
Its baseline snapshots the current `BEAUTY_PROMPTS` from the canonical
[`beauty.py`](../../src/diagram_correctness/beauty.py) evaluator and uses
the existing local LLMAPI transport with `dev-gpt-6-luna`. There are eight
independent calls per image, so the fresh 200-image baseline requires 1,600 calls.
Raw scores, observations, rationales, and response/model metadata are saved after
each successful call. Resume uses only scores from this new run.

## Boundary and splits

- HS001-HS100: training. Rank images by their largest absolute dimension error.
- HS101-HS200: validation. A fixed random 20-image subset screens candidates.
- HS201-HS300: no scoring, review, tuning, or metric computation.
- Human scores already use 1-10. Convert only judge scores:
  `1 + 9 * score / 100`. RMSE, MAE, and Pearson are validation-only.
- Overall metrics pool all image/dimension pairs; per-dimension metrics are also
  reported. Pearson is `null` when either input is constant.

The original full labels file is preserved as a run snapshot. Corrections affect
only beauty labels for displayed review images in a separate corrected labels
file; correctness labels and testing labels are preserved without analysis.
The source labels file is never overwritten.

## Run locally

From the repository root in PowerShell:

```powershell
python experiments\beauty_alignment_fresh\run.py prepare `
  --labels human_ratings.json `
  --images 'C:\datasets\human_study\images' `
  --run experiments\beauty_alignment_fresh\outputs\fresh_001

python experiments\beauty_alignment_fresh\run.py score `
  --run experiments\beauty_alignment_fresh\outputs\fresh_001

python experiments\beauty_alignment_fresh\run.py serve `
  --run experiments\beauty_alignment_fresh\outputs\fresh_001
```

LLMAPI sign-in uses the existing local backend; credentials are not copied into
experiment outputs. Open <http://127.0.0.1:8766> for review. The dashboard needs
the local server to save corrections; opening its HTML file alone is insufficient.
Labels, images, the local backend, and run outputs are not included in the PR.
Exact reproduction of the reported results requires the original local run's
frozen baseline/current_001 prompts and corrected labels; preparing a new run
snapshots the evaluator in that checkout, which may have different prompts.

## Human review before tuning

Review the top 20 training discrepancies and the fixed mini-validation images.
While the baseline is running, fully scored training images appear in a
provisional ranking that refreshes automatically. Final batch approval stays
disabled until all 1,600 baseline scores are saved. The mini-validation images
are available immediately. Automatic refresh does not overwrite unsaved edits.
Every card displays the original image with editable human beauty scores, an
optional note, and an explicit verification checkbox. Save each image even when
no score changes are necessary, then approve the batch.
The save button advances to the next unreviewed image in the current batch.
Each review tab shows its remaining count and a next-unreviewed shortcut.

Training cards include judge observations and scores. Mini-validation cards show
only the image and human labels so validation errors do not become prompt-edit
evidence. Correcting training labels can change the top-20 ranking; newly surfaced
images must also be verified. The mini-validation IDs never change.

Batch approval records exact image IDs, not approval of future prompt edits.
Before each candidate, present the exact training images and five proposed edits
in chat and obtain approval. Do not edit prompts before that approval.

## Subsequent iterations

After review, use at most five candidate versions with five targeted edits each.
All edits must be motivated by verified training examples. Score the training set
and fixed mini-validation set for each candidate; only promising candidates
proceed to the full validation set. Select by full-validation pooled RMSE, reporting
MAE and Pearson as well. Stop after two consecutive candidates fail to improve
the best full-validation RMSE (a failed mini-validation screen counts as a
non-improvement). Repeated validation selection is tuning, not a held-out test
estimate.

The runner accepts version JSON files in the new run's `versions` directory and
`score --version NAME --subset training|mini|validation`. Each JSON uses the
baseline's structure and records all exact prompts, the five edits, and the
approved training evidence IDs. Keep every attempted version, including rejected
candidates. Run `report --run ...` to refresh saved metrics without model calls.
The dashboard displays every version and separate mini/full-validation metric
curves; a baseline-only chart does not imply any improvement.

### Evaluating the current user-edited prompts

The approved first candidate snapshots the current evaluator without additional
edits. Its three semantic changes (Centering, Space Use, and shared dimension
ownership text) are user changes, not a new five-edit tuning proposal.

```powershell
python experiments\beauty_alignment_fresh\run.py snapshot `
  --run experiments\beauty_alignment_fresh\outputs\fresh_001 `
  --version current_001 --description "Approved current user-edited prompts" `
  --edits "Whole-diagram-only Centering with margin formula" `
          "Crowding-only Space Use independent of canvas coverage" `
          "Consistent dimension ownership and exclusions across all judges"
```

Then score `--version current_001 --subset training` and `--subset mini`.
Proceed to `--subset validation` only if mini-validation pooled RMSE improves
over the best candidate (initially baseline). Do not use validation discrepancies
to design later prompts. Keep the saved candidate prompts unchanged when resuming.

Open the results tab directly with `/?tab=metrics`. It shows candidate
mini-validation metrics alongside the same-subset baseline, paired uncertainty,
per-dimension mini-validation results, and live training progress.
If an older review server is already running, serve the updated dashboard on a
separate port:

```powershell
python experiments\beauty_alignment_fresh\run.py serve `
  --run experiments\beauty_alignment_fresh\outputs\fresh_001 --port 8767
```

Then open <http://127.0.0.1:8767/?tab=metrics>.

The **Prompt changes** tab (`/?tab=prompts`) compares any two saved versions,
defaulting to baseline versus the latest candidate. Each dimension shows
side-by-side changed passages, word-level additions/removals, and line counts.
Unchanged passages are collapsed, and complete exact prompts are available within
each dimension. Recorded edits describe the selected After version, not an
automatically generated summary of an arbitrary version pair.

### Experimental v2

The approved v2 changes Centering to canvas-relative margin imbalance, distinguishes
intentional groups in Space Use, judges outline-or-fill Contrast, compares
semantic roles in Style Uniformity, and removes the three-hue ceiling from Color
Consistency. Alignment, Resolution, Canvas Fill, and the shared preambles remain
identical to `current_001`. The v2 builder itself does not modify the source
evaluator. The evaluated v2 prompts have now been promoted into
[`beauty.py`](../../src/diagram_correctness/beauty.py); all eight prompts
match the frozen v2 snapshot. The earlier `current_001` version remains a
historical comparison, not the current evaluator.

```powershell
python experiments\beauty_alignment_fresh\make_v2.py `
  --run experiments\beauty_alignment_fresh\outputs\fresh_001
python experiments\beauty_alignment_fresh\run.py score `
  --run experiments\beauty_alignment_fresh\outputs\fresh_001 --version v2 --subset mini
python experiments\beauty_alignment_fresh\run.py score `
  --run experiments\beauty_alignment_fresh\outputs\fresh_001 --version v2 --subset training
```

Only run `--subset validation` if v2 mini-validation pooled RMSE beats baseline.
Review v2 results at `/?tab=metrics` and its five exact changes relative to its
parent at `/?tab=prompts&before=current_001&after=v2`. Both version selectors remain
available for comparing with baseline. Scores always use `1 + 9 * score / 100`,
mapping raw judge scores 0, 50, 100 to human-scale scores 1, 5.5, 10.

V2 completed all 1,600 calls across HS001-HS200. Mini-validation RMSE improved
from 2.1976 to 1.8605; full-validation RMSE improved from 2.5508 to 2.3160
(9.2%). Full-validation MAE improved from 1.8261 to 1.5418, and pooled Pearson
from 0.1921 to 0.3204. The paired RMSE-change 95% interval is
[-0.3668, -0.1079]; the exploratory loss-change permutation p-value is 0.000200
(five-candidate-adjusted 0.001000), not a correlation-significance p-value.
Centering and Contrast improved most; Style Uniformity slightly worsened.
All eight dimensions were re-judged, so even unchanged prompts can produce small
score changes. The test set remains untouched. Promotion into the evaluator
does not make these exploratory validation results a held-out test estimate.

### Uncertainty and significance

`metrics.json` includes paired comparisons against baseline for each completely
scored mini/full-validation subset:

- Candidate-minus-baseline RMSE change and a paired diagram-bootstrap 95%
  percentile interval (10,000 resamples).
- A two-sided paired permutation p-value for mean squared-error change, swapping
  baseline/candidate predictions within diagrams (10,000 draws, plus-one estimate).
- A Bonferroni-adjusted p-value for the planned maximum of five candidates.

All eight dimensions stay together when resampling; the sample size is diagrams,
not image/dimension pairs. Negative changes favor the candidate. RMSE is still
the selection criterion; a small improvement need not be statistically significant.
Report intervals and p-values as exploratory: repeated validation selection is
tuning, and multiplicity correction does not undo adaptive tuning. The intervals
do not include model rerun variability or uncertainty in human labels.

The current export has one rating per diagram. It cannot measure inter-rater
agreement or a human noise ceiling; those require overlapping independent ratings.
HS201-HS300 remain unused, including for significance testing.
