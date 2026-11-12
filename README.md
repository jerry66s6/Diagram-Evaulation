# Diagram Correctness Evaluator

For the broader dataset-building mission, research scope, evaluation principles, and
roadmap, see [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

This package evaluates how well a rendered diagram aligns with a written description across five dimensions:

| Dimension | What it measures |
|---|---|
| **Layout** | One consistent reading direction (loops allowed), no unintended overlap, nothing cut off at the canvas edge, and arrows routed cleanly (not through boxes or text, no needless crossings or detours) |
| **Connectivity** | Every connection the description calls for is drawn, and every drawn arrow is correct: no reversed, wrong, duplicate, or dangling arrows |
| **Presence** | All components the description calls for are drawn (matched by meaning), and nothing out of place is added; extra detail that fits the diagram is fine |
| **Details** | Component labels and connection labels (such as YES/NO branches) are meaningful and agree with the description rather than missing, generic, or placeholder text |
| **Legibility** | Text fits its box, does not collide with other text, is readable at the size the diagram is displayed, and stands out from its background |

## How correctness is scored

Each dimension is scored on its own by gpt-6-luna: the model gets the image and the
caption, lists the problems it sees on that dimension, and gives an integer score from 1 to
10, the scale the human raters used. Overall correctness is the mean of the five scores. There
is no expectation sketch and no scoring formula; the prompt carries the scoring idea in words
([`direct.py`](src/diagram_correctness/direct.py)):

- the human-study question for the dimension and its criteria from [`config/rubric.json`](config/rubric.json);
- that the caption is a short summary rewritten from the figure, so Connectivity and Presence
  flag only clear contradictions, and a label worded differently from the caption is still accurate;
- how real figures draw things (one line can serve several boxes, an exclusion box may be a side
  branch or sit in the main chain, a box touching the image edge is not cut off unless something
  is visibly missing, text is judged at the size shown);
- what each of the other four dimensions covers, so a problem lowers only its own dimension;
- the scoring guide: 9-10 no real problem, 7-8 small problems that do not change the meaning,
  4-6 at least one clear error, 1-3 several errors or a misleading diagram. A single clear error
  leaves the top band, and errors that change the meaning cost more than cosmetic ones.

This is version d17 of the human alignment study
([`experiments/human_alignment`](experiments/human_alignment)), tuned on HS001-HS200. On
HS201-HS300, which were not used for tuning, it agrees with the human ratings of overall
correctness at Spearman 0.87 (Kendall 0.70, mean absolute error 0.80 points on 0-10), per
dimension 0.61-0.73, and separates clean from broken diagrams with AUC 0.90 (humans 0.95). The
first direct version (d2) reached Spearman 0.76 on HS001-HS100, the diagrams it was tuned on, against
0.51 for the sketch-and-formula method.

## Previous method: sketch and formula

Before, the model first read only the description and wrote a sketch of the components and
connections the diagram should have; it then checked the image against that sketch item by
item (three times, majority vote), and code computed each criterion as the share of items that
passed, with each error also multiplying the score by `error_factor`. The code is still here
(`CorrectnessPipeline`, `--method formula`) because the Transformer, degradation, and
correctness isolation experiments below use it and Beauty shares its model backend. Branch
`correctness-v-formula` has the repository as it was before the switch, including the full
description of that method.

## Transformer experiment

The repository contains a controlled example in [`examples/transformer`](examples/transformer):

- a written Transformer encoder specification;
- a strong candidate;
- a mixed candidate missing positional encoding and residual paths and using a placeholder label;
- a weak candidate with missing components, incorrect arrows, overlap, off-canvas content, placeholders, and tiny text.

Run the deterministic smoke test without an API key:

```bash
PYTHONPATH=src python3 scripts/run_transformer_examples.py --offline
```

Run the full experiment with one GPT expectation call and three GPT candidate calls:

```bash
export OPENAI_API_KEY='...'
PYTHONPATH=src python3 scripts/run_transformer_examples.py
```

Outputs are written to `examples/transformer/results/`:

- rendered PNGs;
- `description_expectations.json`, the countable specification derived only from the text;
- one complete JSON report per candidate;
- `summary.json` and `summary.md` with all five scores and sensemaking.

## Single-dimension degradation experiment

The [interactive experiment report](examples/transformer/ablation/index.html) compares
the original strong diagram with five actual SVG edits: swapped stages, removed
arrows, missing components, generic labels, and undersized text. Its matrix uses
interventions as rows and measured dimensions as columns, highlights the target
diagonal, and shows every change relative to the original evaluated in the same run.

Run the deterministic experiment with standard-library Python (no API key or
rendering dependency needed):

```bash
python3 scripts/run_dimension_ablation.py
```

The self-contained HTML includes plots, original/edited diagram comparisons,
criterion evidence, and data downloads. Generated SVGs, full JSON reports, a CSV,
the frozen inventory, source hashes, and the exact mutation manifest are saved in
`examples/transformer/ablation/`. The existing Transformer results stay unchanged.
The saved historical GPT+SVG baseline is shown separately and is not used to
calculate deterministic-run deltas.

To run all six candidates with the configured GPT judge plus deterministic checks:

```bash
PYTHONPATH=src python3 scripts/run_dimension_ablation.py --live
```

This requires the normal dependencies and `OPENAI_API_KEY`; results go to
`examples/transformer/ablation-live/`. Both modes reuse the existing description-only
inventory so the expectations stay fixed. This is one diagram with one intervention
per dimension and unequal degradation sizes, not a calibrated sensitivity benchmark.

Validate the experiment without third-party dependencies:

```bash
python3 -m unittest discover -s tests -p 'test_dimension_ablation.py'
```

## Beauty-axis isolation experiment

The **current v6 Beauty evaluator has eight dimensions**: Centering, Space Use,
Color Consistency, Contrast, Alignment, Style Uniformity, Resolution, and Canvas
Fill. [`judge_beauty_v6`](src/diagram_correctness/beauty_v6.py) makes eight
independent VLM calls with the same observation/score/reasoning response contract.
Centering compares left/right and top/bottom margins around the full diagram,
including its title and connectors. Each axis scores
`100 * (1 - abs(a - b) / canvas_dimension)`;
the final score is the rounded worse axis. Large equal margins are centered.
Label centering inside shapes and equal inter-node gaps are not part of this score.
Space Use judges only crowding and edge-to-edge gaps relative to neighboring
node/group size, treating intentionally layered symbols as a single footprint.
It does not score group span, outer margins, or canvas coverage;
adding padding or scaling nodes and gaps together should not affect it.
Canvas Fill estimates node silhouette area: 0 at 3% or below, 100 at 6% or
above, and linear interpolation between. Outline-only nodes count by their
full area; thin connectors, text, and empty loop interiors do not.
Spreading the same nodes farther apart can improve Space Use without changing
Canvas Fill; tiny, well-spaced nodes can score high on Space Use and low on Canvas Fill.
The human-study UI asks participants about the same eight Beauty dimensions.

The evaluator now includes the five human-alignment v2 prompt changes, also
covering outline-or-fill Contrast, semantic-role Style Uniformity, and coherent
palettes without a three-hue limit. With GPT-6 luna, the evaluated v2 snapshot
achieved RMSE **1.8605 on 20 mini-validation images** and **2.3160 on all 100
validation images**; these are not held-out test results. See
[the experiment results](experiments/beauty_alignment_fresh/RESULTS.md).

Generate v6 fixtures and compute the historical footprint diagnostic without model calls:

```bash
PYTHONPATH=src python3 scripts/run_beauty_v6_ablation.py
```

Add `--live` for repeated model scoring through the public OpenAI backend.
Offline footprint measurements are diagnostics, not the current v6 VLM scores.
They use the same 3%/6% scoring rule, but estimate footprint differently.
Older evaluators and saved results remain unchanged; the human-alignment experiment
is outside the scope of this change.

Score only Canvas Fill using the same prompt and calling code:

```powershell
python scripts\run_canvas_fill.py examples\canvas_fill_vlm\diagrams\baseline.png --output examples\canvas_fill_vlm\results.json
```

This uses the configured OpenAI model and `OPENAI_API_KEY`. To use an existing
local backend instead, add `--backend local\beauty_alignment_backend.py`.
The runner saves the prompt and per-image observations, scores, and reasoning.
Use `--model` to override the model (custom backends must accept `Backend(model=...)`).
See [the v6 results and known gaps](docs/beauty_axis_experiment.md#v4-v6-current-beauty-set).

### Historical five-dimension experiments

The [historical Beauty-axis dashboard](examples/beauty_ablation/index.html),
[three v3 iteration dashboards](examples/beauty_ablation/ceres/), and
[design notes](docs/beauty_axis_experiment.md) cover five visual-quality dimensions:
Aesthetic Craft, Palette, Ink Balance, Density, and Balance. The v3 prompt suite
assigns each judge one concrete observable, requires the observation before the
score, and reports sensitivity, specificity, and isolation. The experiment compares
a synthetic baseline flowchart with one targeted intervention per dimension and
reports off-diagonal score changes.

The portable runner excludes the prototype's internal API transport and uses this
repository's public OpenAI backend for optional live rescoring. Reproduce the
historical report without an API call:

```bash
PYTHONPATH=src python3 scripts/run_beauty_ablation.py
```

Or set `OPENAI_API_KEY` and pass `--live` to score the generated diagrams with the
model configured in `config/evaluator.json`.

The latest measured v3 iteration reached 89.2/100 overall isolation. Its remaining
Ink Balance-to-Density leak led to a fixture-only correction that has not yet been
rerun, so no corrected score is inferred.
## Revised degradation experiment (experimental v2)

The v2 experiment evaluates the original, the five existing partial edits, and
five severe edits. It presents **SVG**, **GPT**, and **Combined** as separate
views. Missing GPT judgments remain pending; they are never replaced by SVG
scores in the Combined view. The v1 script, rubric, and saved numerical results
remain available for reproducibility.

```bash
# SVG checks only; needs no API key.
.venv/bin/python scripts/run_ablation_v2.py --publish-current

# Enter the key locally at a hidden prompt, then evaluate all 11 images.
.venv/bin/python scripts/run_ablation_v2.py --prompt-key --publish-current
```

Alternatively, save `OPENAI_API_KEY="..."` in the project-local `.env` (excluded
from Git) or export the variable in your shell, then use `--live` instead of
`--prompt-key`. The v2 runner reads only this key from `.env`; it never executes
the file. A fresh full run uses 11 GPT image judgments, each covering all five
dimensions. It reuses the existing component inventory plus explicitly
transcribed caption-only spatial requirements; no new expectation API call is
made. Neither intervention names nor severity labels are sent to the judge.

Outputs are in `examples/transformer/ablation-v2/`. The `--publish-current` option
also updates the old local report's HTML and keeps its original HTML as
`examples/transformer/ablation/index.v1.html`. Refresh the current page after each
example finishes. The HTML embeds all 11 diagrams and all available results.
Completed model judgments are cached, and source/protocol hashes reject mixing
judgments from different inputs. Use a fresh `--output-dir` after metric changes.

V2 is explicitly opt-in: `SensitivityJudge` and `config/rubric_v2.json` check
pairwise agreement with stated stage sequences, spatial relations, and required
containment, with no default top-to-bottom assumption. Accidental enclosure is
not treated as intentional grouping. Arrow endpoints must form expected directed
pairs. Repeated or corrupted labels can be resolved from the graph only if the
complete visible directed graph has a unique correspondence with the expected
graph; SVG element IDs are not used as semantic labels.

Within each dimension, v2 takes the **minimum available criterion score** rather
than an arithmetic mean, so passing boundary checks cannot compensate for a
failed sequence or unreadable font. Between judges it still takes the median per
matching criterion, so a SVG/GPT disagreement remains visible in the separate
views. Overall weights are unchanged. The minimum rule is an experimental,
conservative design choice, not a human-calibrated scale or a guarantee that any
small edit must score zero. Severe edits may affect multiple dimensions.

## Diagram dataset

The diagram dataset lives in `diagrams/`, which is kept out of Git because of its size
(about 750 MB). It has 7,856 flowchart-like and other diagram figures from PubMed Central
Open Access articles, each with a caption and an easy, medium, or hard label for structural
complexity. 4,751 are positive (the original figure) and 3,105 negative (the figure with one
to three deliberate errors, such as a removed box, a reversed arrow, or faded text). `diagrams/README.md`
describes it, `diagrams/data.csv` lists it, and `diagrams/pipeline/` rebuilds it.

## Install

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
```

On macOS, CairoSVG also needs the native Cairo library:

```bash
brew install cairo
```

## Evaluate another diagram

```bash
diagram-correctness \
  --description intended-diagram.md \
  --candidate candidate.png \
  --output correctness-report.json
```

`--candidate-svg candidate.svg` works too; the SVG is rendered to a PNG next to the report. The
report has the five 1-10 scores, the problems the model listed, and the overall mean. Add
`--method formula` to run the previous sketch-and-formula pipeline (with an SVG it can also run
the deterministic geometry judge; `--skip-deterministic` turns that off).

## Configuration

- `config/rubric.json`: shared criteria used by GPT and the deterministic judge.
- `config/evaluator.json`: the single model ID (`OPENAI_MODEL` in the environment or `.env`, if set, overrides it). Direct scoring uses only the model; the other settings below are for `--method formula`: geometry tolerances, severity multipliers, and the following. `presence_policy` sets two shared scoring rules: whether an abbreviation such as "FFN" counts as a correct label (`accept_abbreviations`), and whether a box in the right position with a placeholder or missing label counts as drawn but mislabeled (`placeholder_counts_as_present`).
  - `judge_samples`: how many times the GPT judge answers each image (default 3).
  - `deterministic_judge`: `false` by default, so only the GPT judge scores. Set it to `true` to also run the deterministic SVG judge when `--candidate-svg` is given; the code is kept for that.
  - `weighting`: `equal` (the default for now: every dimension counts the same), `severity` (rubric severities), `history` (issue frequency times severity from `weight_history`), or `auto` (history when one is configured, otherwise severity).
  - `dimension_reducer`: `mean` or `minimum` for combining criteria within a dimension. The default is `minimum`, so the worst criterion sets the dimension score.
  - `error_factor`: each detected issue multiplies its criterion's score by this factor (default `0.75`: one error gives at most 0.75, two 0.56, three 0.42), so a single error in a large diagram still lowers the score visibly; the score is the smaller of this and the share of items that passed. Remove it to score by share passed only.
- `config/issue_history.json`: issue counts for `weighting: history`. The current file is placeholder data; switch to `history` only after replacing it with frequencies measured on a real calibration corpus.

## Test

```bash
python3 -m pytest -q
```
