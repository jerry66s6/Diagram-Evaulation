# Project Context

## Mission

The main aim is to build a high-quality dataset of flowcharts and the evaluation
system needed to curate it reliably.

The intended dataset will contain two primary sources:

1. **Extracted flowcharts** from scientific papers available through public datasets
   and other appropriately licensed sources.
2. **Generated flowcharts** produced from textual descriptions or structured
   specifications by large language models.

Both sources can contain malformed, incomplete, misleading, illegible, or visually
poor diagrams. The project therefore also aims to build a robust judge LLM that can
identify high-quality flowcharts and explain why a candidate should be retained,
rejected, or sent for human review.

The end goal is not simply to collect as many diagrams as possible. It is to create
a traceable, diverse, and well-evaluated dataset suitable for research and model
development involving diagram understanding, generation, and evaluation.

## End-to-end vision

The broader system is expected to have four stages.

### 1. Source acquisition

Candidate flowcharts may come from:

- scientific-paper datasets with clear public access and licensing;
- structured extraction from paper figures and captions;
- LLM-generated diagrams based on paper text, synthetic prompts, or structured
  process descriptions; and
- controlled variants created specifically to test evaluator behavior.

Every candidate should retain provenance such as its source dataset, paper or prompt
identifier, extraction or generation method, applicable license, and processing
history. Once we have all the images/SVGs, we might do some
normalization/preprocessing to ensure a standardized dataset.

### 2. Defining the evaluation system

To begin with, we care about two top-level axes:

1. Correctness - Correctness is evaluated across the dimensions below. Each
   dimension receives its own score, and the dimension scores are combined using
   severity-based weights.
   - Layout - Whether boxes are placed sensibly, without overlap or off-canvas content
   - Connectivity - Whether every arrow correctly joins its intended source and target
   - Presence - Whether all diagram elements the LLM expects to see and extract are present
   - Details - Whether labels and content are meaningful rather than missing or placeholder text
2. Beauty - Beauty combines one subjective aesthetic assessment with four objective pixel-based guardrails.
   - Aesthetics - Whether the diagram has polished, professional, coherent, and visually sophisticated styling.
   - Palette - Whether the diagram uses a purposeful, coordinated middle range of perceptually distinct colors.
   - Ink balance - Whether the canvas contains a healthy amount of visual content without appearing empty or overloaded.
   - Density - Whether diagram elements are evenly and comfortably spaced rather than cramped or clustered.
   - Balance - Whether the overall composition and whitespace are centered rather than visually biased toward one side or corner.

At the judge LLM level, we ask the judge LLM to rate a given image across these dimensions.

Future work - Accessibility, maybe as part of beauty dimension.

### 3. Sanity checking the isolation of dimensions

In this step, we take an example flowchart and purposely mess it up along one of the
dimensions. We then see whether the modified image also scores low along any other
dimensions. This helps us catch correlations between any two dimensions and ensure
that our dimensions evaluate isolated aspects of a flowchart.

### 4. Human alignment study

Once we have isolated dimensions, we can start collecting human data and calculate something like RMSE with those scores (lower is better).
Then, we can modify the prompt to improve that RMSE using a validation set.
We can keep doing prompt iterations till we stop seeing meaningful improvement.

## Current status (as of 2026-10-06)

### 1. Source acquisition

- The first public source in use is the DiagramGen benchmark
  (DiagramAgent/DiagramGenBenchmark, Apache-2.0). A 10-example pilot of 4 model
  architecture diagrams, 4 flowcharts and 2 directed graphs is fixed in
  `examples/diagramgen10/`, with image and caption hashes recorded for provenance.
- Controlled variants exist for the isolation experiments: a Transformer encoder
  diagram and a synthetic five-step flowchart.
- Extracting flowcharts from paper figures and generating new flowcharts with LLMs
  have not started yet.

### 2. Evaluation system

- **Correctness** is implemented in `src/diagram_correctness/`. GPT-5.5 first turns
  the written description into a frozen inventory of expected components and
  connections, then judges the rendered image against that inventory. A
  deterministic judge separately checks the SVG's geometry, labels, connectors and
  font sizes. Dimension scores are combined with weights derived from issue
  frequency times severity. The frequencies in `config/issue_history.json` are
  placeholders until a calibration set exists.
- **Legibility is still a fifth correctness dimension in the code** (minimum font
  size and text overflow), and the user study asks about it too. The dimension list
  above omits it, so we need to decide whether to keep it under Correctness, move it
  to Beauty or Accessibility, or drop it.
- **Beauty** is implemented as five separate single-dimension GPT prompts. They run
  from the Beauty experiment script, not yet from the main evaluator CLI. The
  objective pixel-based guardrails for Palette, Ink balance, Density and Balance are
  not implemented yet, so all five are currently vision-model judgments.
- **DiagramGen pilot:** all 10 examples were scored with the GPT part of the
  correctness evaluator. The SVG judge was not used in this pilot.
  Overall scores range from 58 to 100, with a median of about 90. The reference
  images do not always satisfy every request in the dataset's expanded
  descriptions, so low scores need manual review before drawing conclusions.

### 3. Isolation of dimensions

- **Correctness** (Transformer encoder, `examples/transformer/ablation-v2/`): one
  partial and one severe edit per dimension, scored by the SVG judge, by GPT, and
  combined. Every edit lowered its target dimension, and the partial edits stayed
  close to isolated. In the GPT view, the severe Layout edit also lowered
  Connectivity from 90 to 50, and the severe Legibility edit also dropped Details
  from 100 to 0. The severe Presence edit removed connections as well, which is
  expected because missing components take their arrows with them.
- **Beauty** (synthetic five-step flowchart, `docs/beauty_axis_experiment.md`):
  every edit lowered its target dimension. The Ink balance edit also lowered
  Aesthetics and Density, and the Balance edit also lowered Density.
- Both experiments use a single diagram with one or two edit strengths per
  dimension. They are sanity checks, not a calibrated benchmark.

### 4. Human alignment study

- A rating interface is built in `user-study/`. Each participant rates 20 to 30
  diagrams (24 by default), balanced across easy, medium and hard and shown in
  random order. Each diagram is shown with its caption and rated on all ten
  dimensions on a 1 to 10 scale. Researchers export a CSV with one row per
  participant and diagram.
- It currently holds 24 demo diagrams with placeholder captions and provisional
  difficulty labels.

## Next steps

1. Assemble the study set: real flowcharts with their captions and provenance, plus
   an agreed rule for labeling easy, medium and hard. Load them with
   `npm run import:diagrams` in `user-study/`.
2. Decide where Legibility belongs.
3. Run the study. Human ratings use a 1 to 10 scale and judge scores use 0 to 100,
   so both need mapping to a common scale before computing RMSE.
4. Replace the placeholder issue frequencies with frequencies measured on a
   calibration set.
