# Diagram Correctness Evaluator

For the broader dataset-building mission, research scope, evaluation principles, and
roadmap, see [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

This package evaluates how well a rendered diagram aligns with a written description across five dimensions:

| Dimension | What it measures |
|---|---|
| **Layout** | One consistent reading direction (loops allowed), no unintended overlap, and nothing cut off at the canvas edge |
| **Connectivity** | Every expected connection is drawn, and every drawn arrow is correct: no reversed, extra, duplicate, or dangling arrows |
| **Presence** | All components expected from the description are drawn, and no invented or duplicated components are added |
| **Details** | Component labels and connection labels (such as YES/NO branches) are correct rather than missing, generic, or placeholder text |
| **Legibility** | Text fits its box, does not collide with other text, and is readable at the size the diagram is displayed |

The final formula is:

```text
Correctness =
    w_presence     * Presence
  + w_layout       * Layout
  + w_connectivity * Connectivity
  + w_details      * Details
  + w_legibility   * Legibility
```

Every criterion uses:

```text
(expected opportunities - detected issues) / expected opportunities
```

A criterion with zero opportunities is `N/A` and is excluded rather than treated as perfect. The opportunities always come from the frozen inventory and the per-item verdicts, never from a number the model picks.

Each failure is counted in one dimension only:

- a missing component costs Presence; its connections are not counted again under Connectivity, and its label is not counted again under Details;
- a label that is drawn but too small to read costs Legibility, not Details;
- a reversed arrow costs Connectivity precision (`connectivity.endpoints`), not recall (`connectivity.connections`).

## Description-first GPT design

The default configuration uses **only GPT-5.5** as the VLM source.

1. **Expectation stage:** GPT-5.5 receives only the written description—no reference or candidate image. It freezes a countable inventory for all five dimensions: the expected components with their exact labels, and the directed connections with any label the description gives them (for example a YES/NO branch). It also lists placeholder strings to reject.
2. **Scoring stage:** the same model receives one candidate image, the description, and the frozen inventory. It cannot redefine what should exist while scoring. It answers item by item:
   - **Components:** `present`, `mislabeled`, `absent`, or `uncertain` for every inventory component, with the quoted visible text, a bounding box, and whether the text is legible. `uncertain` counts as not drawn.
   - **Connections:** `present`, `reversed`, `absent`, or `uncertain` for every inventory connection, with the text written on the arrow.
   - **Extras:** drawn components and arrows that the inventory does not call for.
   - **Counted criteria** (order, canvas, overlap, overflow): the number of failures only. The evaluator supplies the number of opportunities from the item answers and caps the failures at that number.

   The evaluator, not the model, computes every score from these answers, and compares connection labels in code. The prompt tells the model to judge each part on its own, so a crossing arrow is not a missing connection and tiny text is not a wrong label. With `judge_samples` above 1, the same question is asked several times; the evaluator keeps the majority verdict per item and the median failure count per criterion, and records each sample's score in the metric notes.
3. **Geometry stage:** the deterministic judge inspects only the candidate SVG and checks it against the same inventory. It is not another language model. It returns the same per-component and per-connection verdicts, and the report metadata (`component_verdicts`, `component_disagreements`) lists where the two judges disagree on components. Its checks:
   - arrow direction comes from the arrowhead (`marker-start` or `marker-end`), and plain lines that touch no component, or start and end on the same component, are treated as decoration;
   - connection labels are the free text nearest each arrow;
   - unexpected components are labeled boxes that match no inventory component and do not enclose one;
   - the reading direction is inferred from the drawing, and connections that loop back to an earlier step are skipped;
   - each component counts once for canvas and overlap checks, and unlabeled polygons (arrowheads) are not overlap candidates;
   - font sizes are inherited from parent groups, resolved from `px`, `pt`, `em`, and `%`, and compared at the rendered size from the root `width`/`height` versus the `viewBox`;
   - text outside boxes must not collide with other text. Text widths are a conservative estimate because SVGs do not contain browser `getBBox()` results.
4. **Aggregation:** judges are combined per criterion with the median; `criterion_disagreements` in the report metadata lists criteria where judges differ by 0.25 or more, which a median of two would hide. Criteria are combined per dimension with `dimension_reducer` (`mean` or `minimum`). Dimension weights follow `weighting`; the default, `severity`, gives each dimension the multiplier of its most severe rubric criterion, normalized.

The deterministic judge needs an SVG. For raster candidates, such as figures extracted from papers, it runs only when a VFIG command is configured to convert the image; otherwise only the GPT judge scores the diagram.

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

The [historical Beauty-axis dashboard](examples/beauty_ablation/index.html) and
[design notes](docs/beauty_axis_experiment.md) cover five visual-quality dimensions:
Aesthetics, Palette, Ink Balance, Density, and Balance. The experiment compares a
synthetic baseline flowchart with one targeted intervention per dimension and
reports off-diagonal score changes.

The portable runner excludes the prototype's internal API transport and uses this
repository's public OpenAI backend for optional live rescoring. Reproduce the
historical report without an API call:

```bash
PYTHONPATH=src python3 scripts/run_beauty_ablation.py
```

Or set `OPENAI_API_KEY` and pass `--live` to score the generated diagrams with the
model configured in `config/evaluator.json`.

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

## DiagramGen 10-example pilot

The [self-contained pilot page](examples/diagramgen10/index.html) contains ten
original DiagramGen PNGs and their unmodified `expanded_query` descriptions.
Selection is fixed before scoring: 4 model-architecture-labeled examples,
4 flowcharts, and 2 directed graphs, ordered within category by
`SHA256("20260930:" + id)`. The downloaded 270-record generation index is hash
checked; the manifest records every selected ID, caption hash, and image hash.
The [dataset card](examples/diagramgen10/source/README.md) is retained with the data.

Prepare the inputs or rebuild the HTML without model calls:

```bash
python3 scripts/run_diagramgen10.py
```

Run or resume the existing GPT evaluator after configuring `OPENAI_API_KEY` locally:

```bash
.venv/bin/python scripts/run_diagramgen10.py --live
```

A fresh successful run uses 10 description-only expectation calls and 10 image
judgment calls; all five dimensions come from each image judgment. The original
dataset PNG is the candidate, not a newly generated diagram. This pilot uses the
GPT component only: LaTeX/DOT source is archived, but SVG geometry and VFIG are
omitted. Core evaluator code, prompts, rubric, and weights are unchanged.

Each example saves its frozen inventory, exact text prompts, structured model
responses, and complete report under `examples/diagramgen10/results/`. Completed
reports and inventories are reused when resuming. Configuration and evaluator
source hashes prevent mixing different protocols; use `--output-dir` for a new
protocol. Failed or blocked examples have no numeric score. `scores.csv`,
`summary.json`, and the standalone HTML update after each example. Copying just
`index.html` preserves all images, descriptions, and available results offline.

This is a small pilot, not a calibrated accuracy benchmark. Dataset category
labels are broad, some descriptions underspecify connections, and reference
images need not satisfy every expanded-description request. Review inventories
and evidence manually before drawing conclusions. No beauty judge is run here.

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

With an existing generated SVG:

```bash
diagram-correctness \
  --description intended-diagram.md \
  --candidate-svg candidate.svg \
  --output correctness-report.json
```

Those are the only two semantic inputs. The SVG is rendered internally to a temporary PNG solely because the GPT vision endpoint consumes an image; the evaluator does not generate or use a target/reference diagram.

For GPT-only scoring:

```bash
diagram-correctness \
  --description intended-diagram.md \
  --candidate candidate.png \
  --skip-deterministic
```

If SVGs are unavailable, install the official [VFIG repository](https://github.com/RAIVNLab/VFig) and set its inference command in `config/evaluator.json`.

## Configuration

- `config/rubric.json`: shared criteria used by GPT and the deterministic judge.
- `config/evaluator.json`: the single model ID, VFIG command, geometry tolerances, and severity multipliers. `presence_policy` sets two shared scoring rules: whether an abbreviation such as "FFN" counts as a correct label (`accept_abbreviations`), and whether a box in the right position with a placeholder or missing label counts as drawn but mislabeled (`placeholder_counts_as_present`).
  - `judge_samples`: how many times the GPT judge answers each image (default 3).
  - `weighting`: `severity` (rubric severities), `history` (issue frequency times severity from `weight_history`), `equal`, or `auto` (history when one is configured, otherwise severity).
  - `dimension_reducer`: `mean` or `minimum` for combining criteria within a dimension.
- `config/issue_history.json`: issue counts for `weighting: history`. The current file is placeholder data, which is why the default is `severity`; switch to `history` only after replacing it with frequencies measured on a real calibration corpus.

## Test

```bash
python3 -m pytest -q
```
