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

The broader system is expected to have five stages.

### 1. Source acquisition

Candidate flowcharts may come from:

- scientific-paper datasets with clear public access and licensing;
- structured extraction from paper figures and captions;
- LLM-generated diagrams based on paper text, synthetic prompts, or structured
  process descriptions; and
- controlled variants created specifically to test evaluator behavior.

Every candidate should retain provenance such as its source dataset, paper or prompt
identifier, extraction or generation method, applicable license, and processing
history.

### 2. Normalization and preprocessing

Extracted and generated candidates should be normalized into consistent
representations where possible:

- the original image or vector artifact;
- a normalized raster rendering;
- an SVG or another structured diagram representation;
- the source or intended textual description;
- detected or expected components, labels, and directed connections; and
- metadata describing dimensions, language, domain, and transformation history.

Keeping both rendered and structured forms makes it possible to combine perceptual
vision-model judgments with deterministic geometry and graph checks. Preprocessing
must clean and standardize inputs without silently repairing the defects that the
judge is supposed to measure.

### 3. Defining the evaluation system

A robust evaluator should assess multiple independent aspects of quality rather than
produce an unexplained single score.

The proposed target taxonomy has four top-level axes:

| Axis | Primary question | Intended evaluation approach |
|---|---|---|
| Syntactic correctness | Are the diagram's structure, layout, and connections valid? | Vision-model judgments aligned with deterministic geometry checks |
| Semantic correctness | Does the diagram faithfully represent the source meaning or relevant knowledge? | Source-grounded component and relationship checks plus a semantic vision judge |
| Visual quality | Is the diagram readable, balanced, and visually appealing? | A dominant perceptual assessment with deterministic visual guardrails |
| Accessibility | Can users with accessibility needs effectively consume the diagram? | Deterministic accessibility checks plus an overall accessibility assessment |

This taxonomy is a target design rather than a claim that all four axes are already
implemented. It also separates two ideas that are easy to conflate:

- **syntactic correctness** concerns valid structure, especially layout and
  connectivity; and
- **semantic correctness** concerns whether the expected entities, details, and
  relationships communicate the right meaning.

#### Current correctness implementation

The correctness evaluator currently implemented in this repository measures:

- **Presence**: whether all expected components are drawn;
- **Layout**: whether placement, ordering, overlap, and canvas use are sensible;
- **Connectivity**: whether required arrows connect the correct endpoints in the
  correct direction;
- **Details**: whether labels and content are complete and meaningful; and
- **Legibility**: whether text is readable, contained, and sufficiently large.

The current design derives a frozen, countable inventory from the written
description before the judge sees the candidate. A vision-language model evaluates
the rendered diagram against that inventory, while a deterministic judge inspects
SVG geometry, text, and connectors. Criterion-aligned signals are then aggregated
instead of asking a model for an unsupported holistic score.

These five operational dimensions currently live under one correctness score. As
the broader four-axis system develops, their ownership may be refined: Layout and
Connectivity naturally support syntactic correctness; Presence and Details support
semantic correctness; and Legibility also contributes to visual quality and
accessibility. Shared evidence may inform multiple axes, but the same failure should
not be double-counted without an explicit scoring policy.

#### Proposed judge architecture

The intended architecture combines complementary evidence rather than relying on a
single unexplained model judgment:

- one or more vision-language judges inspect the rendered candidate;
- deterministic checks measure properties that are reliably observable in SVG;
- all participating judges evaluate aligned sub-criteria and denominators; and
- aggregation occurs criterion-first, followed by dimension and axis aggregation.

When only a raster image is available, VFIG can reconstruct an SVG for deterministic
measurement. This makes boxes, paths, lines, text positions, alignment, overlap,
connection endpoints, overflow, and font size testable. Reconstruction should
preserve visible defects rather than silently improve the candidate. If reliable
geometry is unavailable, the evaluator should record the missing deterministic
signal instead of treating it as a perfect score.

#### Visual quality

The proposed visual-quality score includes:

- **Aesthetics**: overall visual sophistication, coherence, and richness;
- **Palette**: whether perceptually distinct colors form a healthy, coordinated
  range;
- **Ink balance**: whether the canvas contains too little or too much visual ink;
- **Density**: whether content is appropriately distributed rather than forming
  cramped blobs;
- **Balance**: whether the composition is centered rather than lopsided; and
- **Legibility**: whether text fits its container and meets a minimum readable size.

The perceptual aesthetics judgment should remain the dominant signal. Deterministic
or pixel-based metrics act as guardrails: satisfying basic color, density, or
balance thresholds does not by itself make a diagram visually sophisticated. Final
weights and healthy ranges must be calibrated with representative diagrams and
human judgments.

These dimensions need careful isolation testing because real visual properties can
be correlated. For example, excessive ink may also reduce perceived aesthetics,
density, and legibility.

#### Accessibility

The proposed accessibility axis combines:

- text/background contrast;
- availability of text alternatives or directly exposed text;
- meaningful semantic roles and structure; and
- an overall assessment of accessibility and understandability.

Accessibility criteria must ultimately be tied to the artifact formats the dataset
will release. A raster image, SVG, and interactive document expose different
information to assistive technology.

#### Semantic correctness

The proposed semantic axis evaluates:

- whether required concepts and entities are present;
- whether important attributes and details are preserved;
- whether relationships and dependencies match the source meaning; and
- overall faithfulness to a source paragraph or, where explicitly appropriate,
  well-established world knowledge.

Source-grounded evaluation is preferred whenever a source passage or structured
specification exists. World-knowledge judgments should be identified separately
because they have different uncertainty and reproducibility characteristics.

### 4. Dataset curation

Judge outputs should support more than a binary keep/drop decision. A useful
curation record should include:

- top-level axis scores and their component dimension scores;
- dimension-level scores and the scoring version;
- the exact expected inventory and rubric;
- issue descriptions and localized evidence;
- model and deterministic-judge agreement or disagreement;
- confidence or uncertainty signals;
- the acceptance threshold or review rule applied; and
- any final human decision.

A comparable result record should expose both aggregate and component scores rather
than only a final ranking. At minimum it should include syntactic and semantic
correctness signals, visual-quality components, accessibility results where
available, judge participation, unavailable signals, and the evidence behind each
decision.

High-confidence candidates can be accepted or rejected automatically. Borderline
cases, judge disagreements, and underrepresented domains should be prioritized for
human review. Thresholds should be calibrated from human-labeled data rather than
selected only from a small synthetic experiment.

### 5. Validation and release

Before a dataset release, the project should measure:

- agreement with independent human raters;
- consistency across repeated model calls;
- robustness to controlled single-dimension degradations;
- precision and recall at proposed acceptance thresholds;
- performance across source, domain, language, and complexity slices;
- duplicate and near-duplicate rates;
- contamination or train/test leakage risks; and
- provenance and license completeness.

Dataset versions should pin evaluator configuration and retain enough metadata to
reproduce every inclusion decision.

## What this repository currently contains

The repository is currently focused on the evaluation and validation layer. It
includes:

- a Python package for description-based flowchart correctness evaluation;
- a description-first expectation extractor using structured model output;
- a vision-model judge and deterministic SVG judge;
- shared criteria for presence, layout, connectivity, details, and legibility;
- severity- and frequency-based dimension weighting;
- a command-line interface for evaluating candidate diagrams;
- controlled Transformer-flowchart examples with strong, mixed, and weak variants;
- single-dimension intervention experiments and self-contained result dashboards;
  and
- tests for scoring, deterministic geometry, atomic component judgments, pipeline
  behavior, and intervention invariants.

This is a research prototype, not yet a complete ingestion or dataset-production
system. Scientific-paper extraction, large-scale LLM generation, storage schemas,
deduplication, human annotation tooling, accessibility evaluation, a complete
semantic axis, and release automation remain future work.

## Evaluation principles

The following principles should guide future work:

1. **Description-first evaluation**

   Infer what should be present before viewing the candidate so the judge cannot
   silently redefine the task around what it sees.

2. **Atomic, countable decisions**

   Prefer per-component and per-connection verdicts over vague overall impressions.

3. **Aligned measurement contracts**

   Human raters, model judges, and deterministic checks should evaluate the same
   opportunities with the same definitions.

4. **Evidence over unsupported scores**

   Preserve visible text, bounding boxes, connector evidence, and issue descriptions
   behind each score.

5. **Separate dimensions where practical**

   Score independent properties independently, then explicitly study genuine
   correlations and evaluator leakage.

6. **Human-calibrated thresholds**

   Treat synthetic experiments as engineering checks, not substitutes for human
   agreement and real-corpus calibration.

7. **Reproducibility and provenance**

   Record source, license, transformations, prompts, model configuration, rubric
   version, and judge outputs for every candidate.

8. **Public-safe implementation**

   The repository should use public dependencies and documented configuration. It
   must not contain private service endpoints, credentials, proprietary datasets, or
   other internal-only infrastructure.

## Data and responsible-use requirements

- Use only sources whose terms permit the intended research and redistribution.
- Keep paper, figure, caption, and dataset identifiers needed for attribution.
- Do not treat public accessibility as proof of redistribution rights.
- Avoid storing credentials, private endpoints, personally identifiable information,
  or confidential document content.
- Document whether a diagram is extracted, generated, transformed, or manually
  authored.
- Detect duplicates before splitting the data so related diagrams do not leak across
  training, validation, and test sets.
- Preserve rejected examples and rejection reasons when licensing permits; hard
  negatives are valuable for judge training and evaluation.
- Review domain slices for systematic judge bias, especially diagrams with unusual
  notation, dense scientific content, or non-English labels.

## Near-term priorities

1. Finalize a shared scoring contract for humans, the vision judge, and deterministic
   checks.
2. Build a representative calibration set containing real extracted diagrams,
   generated diagrams, and controlled degradations.
3. Collect independent human labels and measure agreement, correlation, error, and
   threshold performance.
4. Calibrate dimension weights and acceptance policies from real issue frequencies
   and downstream impact.
5. Define the versioned dataset record and provenance schema.
6. Implement public-dataset ingestion and licensing checks.
7. Add reproducible LLM generation with prompt and model metadata.
8. Add deduplication, split construction, and review tooling.
9. Expand evaluation beyond correctness while preserving interpretable,
   dimension-specific evidence, beginning with the proposed semantic, visual-quality,
   and accessibility axes.

## Success criteria

The project will be successful when it can produce a dataset for which:

- every flowchart has clear provenance and an appropriate usage basis;
- every acceptance decision is traceable to versioned criteria and evidence;
- judge decisions agree with trained human reviewers at a measured, useful level;
- quality thresholds generalize across sources and domains;
- dataset splits are free from meaningful duplicate leakage; and
- the complete curation process can be reproduced without private infrastructure.
