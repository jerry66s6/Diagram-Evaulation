# Diagram Correctness Evaluator

This package implements **Syntactically Axis 1: Correctness** as a four-judge panel:

- GPT-5.5 vision judge
- GPT-5.6 vision judge
- Claude Opus 4.8 vision judge
- deterministic SVG geometry judge

It evaluates five dimensions: **presence, layout, connectivity, details, and legibility**. The original formula listed four terms but described five dimensions; this implementation includes legibility as the fifth term:

```text
Correctness =
    w_presence     * Presence
  + w_layout       * Layout
  + w_connectivity * Connectivity
  + w_details      * Details
  + w_legibility   * Legibility
```

Every criterion score is:

```text
(expected opportunities - detected issues) / expected opportunities
```

Counts are clamped to `[0, 1]`. A criterion with zero opportunities is `N/A` and is excluded rather than silently treated as perfect.

## How the judging pipeline works

1. A separate expectation VLM reads only the reference image and freezes a concrete inventory. This is the first VLM stage requested for presence/details.
2. GPT-5.5, GPT-5.6, and Claude Opus 4.8 independently compare the reference and candidate against that inventory.
3. VFIG converts both images to SVG, unless existing SVGs are supplied.
4. The deterministic judge parses shapes, paths, lines, connectors, and text and evaluates the **same criterion IDs** as the VLM panel.
5. Results are aggregated criterion-first. The panel median is used to reduce the effect of a single outlier judge.
6. Dimension weights are derived from historical issue frequency times a severity multiplier, then normalized.

The deterministic legibility checks cover both rubric conditions:

- rendered font size below the configured minimum;
- text geometry outside its containing shape.

Text width is conservatively estimated from the SVG text position and font size because VFIG output does not include browser `getBBox()` data. The report labels this approximation.

## Install

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
```

Set both API keys:

```bash
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
```

Install the official [VFIG repository](https://github.com/RAIVNLab/VFig), then replace `/absolute/path/to/VFig` in `config/evaluator.json` with its real location. VFIG can be skipped when ground-truth and candidate SVG files already exist.

## Run

With VFIG image-to-SVG conversion:

```bash
diagram-correctness \
  --reference reference.png \
  --candidate candidate.png \
  --output correctness-report.json
```

With existing SVGs:

```bash
diagram-correctness \
  --reference reference.png \
  --candidate candidate.png \
  --reference-svg reference.svg \
  --candidate-svg candidate.svg \
  --output correctness-report.json
```

For a VLM-only smoke test before VFIG is installed:

```bash
diagram-correctness \
  --reference reference.png \
  --candidate candidate.png \
  --skip-deterministic
```

The JSON output contains the final score, exact normalized formula, dimension scores, weights, every judge's criterion-level counts, evidence, and the frozen reference inventory.

## Configure the rubric and weights

- `config/rubric.json` is the single source of truth for criteria shared by all judges.
- `config/issue_history.json` contains issue counts observed on a calibration set.
- `config/evaluator.json` contains model IDs, VFIG command arguments, geometry tolerances, and severity multipliers.

An issue-history row contributes:

```text
historical count * severity multiplier
```

to its dimension. The dimension totals are normalized to sum to `1.0`. Replace the sample counts with frequencies measured on your evaluation/calibration corpus.

## Test

```bash
python3 -m pytest -q
```

