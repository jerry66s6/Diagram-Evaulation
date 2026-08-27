# Diagram Correctness Evaluator

This package evaluates how well a rendered diagram aligns with a written description across five dimensions:

| Dimension | What it measures |
|---|---|
| **Layout** | Sensible placement, no unintended overlap, and no off-canvas content |
| **Connectivity** | Every arrow joins its intended source and target in the correct direction |
| **Presence** | All diagram elements expected from the description are present |
| **Details** | Labels and content are meaningful rather than missing, generic, or placeholder text |
| **Legibility** | Text does not overflow and meets the minimum rendered font size |

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

A criterion with zero opportunities is `N/A` and is excluded rather than treated as perfect.

## Description-first GPT design

The default configuration uses **only GPT-5.5** as the VLM source.

1. **Expectation stage:** GPT-5.5 receives only the written description—no reference or candidate image. It freezes a countable inventory for all five dimensions. Presence explicitly lists expected components; Details explicitly lists meaningful labels and rejects placeholders.
2. **Scoring stage:** the same model receives one candidate image, the description, and the frozen inventory. It cannot redefine what should exist while scoring.
3. **Geometry stage:** the deterministic judge inspects only the candidate SVG and checks it against the structured component/connection specification extracted from the description. It is not another language model.
4. Criterion-aligned signals are aggregated into the five dimension scores. Dimension weights are historical issue frequency multiplied by severity, then normalized.

The deterministic legibility checks cover minimum font size and estimated text overflow. Its width estimate is conservative because SVGs do not contain browser `getBBox()` results.

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
- `config/evaluator.json`: the single model ID, VFIG command, geometry tolerances, and severity multipliers.
- `config/issue_history.json`: calibration-set issue counts used to derive dimension weights.

Replace the sample issue frequencies with frequencies measured on the real calibration corpus before reporting benchmark results.

## Test

```bash
python3 -m pytest -q
```
