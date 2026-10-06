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
   - Legibility - Whether text is readable, without tiny fonts or text overflowing its box
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

**UI.** To collect human data, we built a rating UI in `user-study/`. Each participant
rates 20-30 diagrams (24 by default), balanced across easy, medium and hard and shown
in random order. Each diagram is shown with its caption, and the participant scores it
from 1 to 10 on every correctness and beauty dimension. Researchers can export all
ratings as a CSV with one row per participant and diagram.

**Metrics.** Humans rate on a 1-10 scale and the judge on 0-100, so both are first put
on the same scale. Then, per dimension:

- RMSE and MAE between human and judge scores (lower is better).
- Spearman correlation, to check whether the judge ranks diagrams the same way humans do.
- Inter-rater agreement between humans, e.g. Krippendorff's alpha, to know how much
  agreement is realistically achievable.
- The same metrics broken down by difficulty (easy, medium, hard).
