# Description-Based SVG Diagram Correctness Evaluator

## Project Summary, Current Results, Limitations, and Next-Step Plan

## Executive Summary

This project implements **Syntactically Axis 1: Correctness** for generated diagrams. The evaluator measures how well a generated SVG matches its written description. It does not require or use a reference diagram, and it does not generate a diagram. Its two semantic inputs are:

1. the original written diagram description; and
2. the related generated SVG.

The current system evaluates five dimensions: **Presence, Layout, Connectivity, Details, and Legibility**. GPT-5.5 first converts the description into structured expectations, then evaluates the rendered SVG against those expectations. A deterministic judge separately parses the SVG and measures geometry, labels, connectors, canvas bounds, overlap, and font properties. Both sources contribute to the final score.

A controlled Transformer encoder experiment with strong, mixed, and weak candidates produced the expected ranking:

| Candidate | Overall | Layout | Connectivity | Presence | Details | Legibility |
|---|---:|---:|---:|---:|---:|---:|
| Strong | 0.972 | 1.000 | 0.950 | 0.962 | 1.000 | 1.000 |
| Mixed | 0.762 | 0.939 | 0.679 | 0.762 | 0.447 | 1.000 |
| Weak | 0.203 | 0.269 | 0.125 | 0.129 | 0.000 | 0.500 |

This is encouraging evidence that the evaluator responds in the intended direction. However, the current experiment is a prototype validation, not yet a human-agreement study. Pearson correlation and RMSE cannot be reported until paired human scores are collected. Before that study, the scoring opportunities and denominators should be standardized so humans, GPT, and the deterministic judge evaluate exactly the same checklist items.

## 1. Research Goal

The goal is to answer the following question:

> Given a written description and a generated diagram, how correctly does the diagram express the requested structure and content?

The evaluator is designed for situations where no gold/reference image exists. This is important because diagram generation usually starts from text, and there may be many valid visual layouts for the same description. The evaluator therefore checks semantic and structural alignment with the description instead of pixel similarity to a reference image.

The current scope is correctness only. It does not attempt to score visual aesthetics, creativity, or whether the diagram uses the same style as another image.

## 2. How the Design Evolved

The initial implementation assumed a reference image and multiple VLM judges. The project requirements were then clarified:

- there is no reference diagram;
- the evaluator receives only the previous description and generated SVG;
- the evaluator should use one GPT model as the VLM source;
- Presence and Details must first be inferred from the description and then compared with the candidate; and
- the deterministic judge must use criteria aligned with the VLM judge.

The implementation was refactored accordingly. All reference-image inputs, reference SVG fixtures, and reference-based comparison logic were removed. GPT-5.5 is now the only VLM. The evaluator supports description + SVG directly and renders the SVG internally only to create the image supplied to GPT vision.

## 3. Current Evaluation Pipeline

### Stage 1: Description-derived expectations

GPT-5.5 receives only the written description. It does not see the candidate at this stage. It extracts:

- expected components and their labels;
- expected directed connections;
- expected reading order and containment;
- meaningful label requirements;
- prohibited placeholders such as “Layer,” “TBD,” or “???”; and
- legibility expectations.

This structured output is saved as `description_expectations.json`. Freezing expectations before showing the candidate reduces the risk that the model changes its interpretation of what should exist after seeing the result.

### Stage 2: GPT visual evaluation

The SVG is rendered to a temporary PNG for GPT vision. GPT receives:

- the original description;
- the frozen description-derived expectations; and
- the rendered candidate image.

For every rubric criterion, GPT returns integer values for `expected_count` and `detected_issues`, plus issue descriptions and visible evidence. GPT does not directly produce the final correctness score.

### Stage 3: Deterministic SVG evaluation

The deterministic judge parses the candidate SVG directly. It checks:

- whether expected labeled components can be matched;
- relative top-to-bottom order;
- off-canvas content;
- unintended shape overlap;
- expected directed connections;
- whether connector endpoints attach to matched components;
- missing, incomplete, generic, or placeholder labels;
- estimated text overflow; and
- minimum font size.

When an SVG is already available, VFIG is not needed. VFIG remains an optional route for reconstructing SVG geometry from raster-only candidates.

### Stage 4: Aggregation

Every criterion uses the basic formula:

```text
criterion score = (expected opportunities - detected issues) / expected opportunities
```

Scores are clamped to the range 0–1. A criterion with zero opportunities is treated as not applicable rather than automatically perfect.

GPT and deterministic scores are combined criterion-first using the median. With the current two participating sources, the median is equivalent to their average. Criteria within a dimension are then averaged to produce the dimension score.

The current overall formula is:

```text
Correctness =
    0.20 × Presence
  + 0.10 × Layout
  + 0.40 × Connectivity
  + 0.10 × Details
  + 0.20 × Legibility
```

These weights currently come from sample issue-frequency data multiplied by severity. They should be recalibrated using a real evaluation corpus before being treated as final benchmark weights.

## 4. What Each Dimension Measures

| Dimension | Current meaning | Current sub-criteria |
|---|---|---|
| Presence | Whether all components, groups, inputs, outputs, repeated blocks, and operations expected from the description are visible | Expected elements present |
| Layout | Whether the diagram has a sensible reading order and avoids geometric problems | Description-implied order, canvas bounds, unintended overlap |
| Connectivity | Whether required directed links exist and arrows visibly attach to the intended components | Required source-to-target connections, endpoint attachment |
| Details | Whether labels and content are complete, accurate, meaningful, and non-placeholder | Meaningful labels |
| Legibility | Whether text can be read without clipping or undersized fonts | Text overflow, minimum font size of 12 px |

Presence and Details are related but intended to answer different questions. Presence asks whether the requested conceptual component exists. Details asks whether its visible label/content is accurate and meaningful. This distinction needs tighter operational rules because the deterministic matcher currently relies partly on labels, which can cause a mislabelled component to be penalized in both dimensions.

## 5. Implementation Completed

The following work has been completed:

- Built a Python package for description-based diagram correctness evaluation.
- Removed all required reference-image and reference-SVG inputs.
- Configured GPT-5.5 as the single VLM expectation extractor and visual judge.
- Implemented structured-output schemas for description expectations and criterion-level judgments.
- Implemented candidate-only deterministic SVG parsing and geometry checks.
- Added direct description + SVG command-line evaluation.
- Added automatic SVG-to-PNG rendering for GPT vision.
- Added macOS/Homebrew Cairo discovery for rendering.
- Defined a shared rubric with aligned criterion identifiers.
- Implemented severity/frequency-based dimension weights.
- Created three controlled Transformer encoder SVG examples.
- Added offline deterministic and live GPT + deterministic experiment modes.
- Added complete JSON reports, rendered PNGs, a summary table, and sensemaking output.
- Added unit and flow tests; the current suite passes all 7 tests.
- Committed and pushed the implementation to the GitHub repository.

The project repository is:

`https://github.com/jerry66s6/Diagram-Evaulation`

Important implementation commits include:

- `41d5bc7` — initial multi-judge correctness evaluator;
- `6ff8519` — reference-free description + SVG redesign; and
- `cf56111` — latest diagram evaluation update.

## 6. Controlled Transformer Experiment

The written specification describes a Transformer encoder containing Input Tokens, Token Embeddings, Positional Encoding, an Add operation, a repeated Transformer Encoder Block, attention, two Add & Norm stages, a feed-forward network, residual connections, and Contextual Embeddings.

Three candidates were constructed to test whether the score changes sensibly:

### Strong candidate

The strong candidate contains the requested components, detailed labels, main arrows, residual paths, readable text, and an organized top-to-bottom layout.

### Mixed candidate

The mixed candidate intentionally omits Positional Encoding and residual paths, shortens several labels, and uses the placeholder “Layer” for the second normalization stage. Its main structure remains readable and mostly well arranged.

### Weak candidate

The weak candidate intentionally contains missing components, incorrect arrows, generic and placeholder labels, overlapping boxes, off-canvas content, and undersized text.

The expectation was therefore:

```text
Strong > Mixed > Weak
```

The live experiment reproduced this ordering.

## 7. Live Results and Interpretation

| Candidate | Overall | Layout | Connectivity | Presence | Details | Legibility |
|---|---:|---:|---:|---:|---:|---:|
| Strong | 0.972 | 1.000 | 0.950 | 0.962 | 1.000 | 1.000 |
| Mixed | 0.762 | 0.939 | 0.679 | 0.762 | 0.447 | 1.000 |
| Weak | 0.203 | 0.269 | 0.125 | 0.129 | 0.000 | 0.500 |

### Strong candidate sensemaking

The deterministic judge found no issues. GPT identified ambiguity in the lower residual connection, interpreting it as continuing from the left skip path rather than visibly originating at the first Add & Norm component. This reduced Presence and Connectivity slightly. The result is useful because it shows that visually ambiguous arrow routing can affect the VLM even when the underlying SVG endpoints satisfy deterministic checks.

### Mixed candidate sensemaking

The mixed candidate remained strong in Layout and Legibility because it was organized, on-canvas, non-overlapping, and readable. Its Connectivity score fell because the Positional Encoding connection and both residual connections were missing. Presence fell because required components or semantic roles were absent or incorrectly represented. Details fell sharply because labels such as “Encoder Block,” “Self-Attention,” “FFN,” and “Layer” did not fully match the requested meaningful labels.

### Weak candidate sensemaking

The weak candidate scored near zero for Presence, Connectivity, and Details because most of the requested Transformer structure was missing or replaced by placeholders. Layout was low due to overlap and off-canvas content. Legibility was 0.500 because overflow checks passed while all measured text failed the minimum-font criterion.

### What the experiment demonstrates

The experiment provides an initial form of construct validity: intentionally degraded candidates receive lower scores, and the dimension-level scores explain why. It also shows the benefit of keeping both visual and deterministic signals. GPT notices semantic and perceptual ambiguity, while SVG parsing provides reproducible geometric evidence.

The experiment does not yet establish agreement with humans, generalization to other diagram types, or statistically reliable benchmark performance.

## 8. Why JSON Contains Long Decimal Values

GPT returns integer counts rather than the final number. For example, a score may be calculated as `8/13 = 0.615384615...`. That value is combined with deterministic fractions, criterion averages, and dimension weights. Python’s JSON serializer preserves the resulting floating-point precision, producing values such as:

```json
"correctness": 0.7624438742085802
```

This does not represent 16 digits of meaningful evaluation precision and is not a VLM confidence score. The recommended reporting format is three decimals (`0.762`) or one decimal as a percentage (`76.2%`). Full internal precision can remain available for reproducibility, but presentation fields should be rounded.

## 9. Current Methodological Limitations

### 9.1 Judges do not always use identical denominators

Although GPT and the deterministic judge use the same criterion identifiers, they currently choose different numbers of opportunities in some cases. For the mixed candidate, GPT counted 13 Presence opportunities while the deterministic judge counted 11. GPT treated canvas and overlap as high-level checks, while the deterministic judge counted individual elements or shapes.

This weakens the meaning of averaging their scores. Criterion alignment should mean not only the same general rubric but the same individual scoring opportunities.

### 9.2 Some failures can be counted twice

A component with the wrong label can appear missing under Presence and incorrect under Details. Placeholder labels can also add issues beyond the number of expected component labels. In the weak example, the deterministic Details judge detected 15 issues against 11 expected opportunities, causing the score to clamp to zero. A stricter issue-ID design is needed to prevent duplicate penalties within a criterion.

### 9.3 The positive-control explanation is currently hard-coded

The generated summary described the strong candidate as passing every check even though the live GPT score was 0.972. Sensemaking should be generated from the actual report rather than fixed example text.

### 9.4 The weight calibration is provisional

Connectivity currently receives 40% of the overall weight. The weighting mechanism is implemented, but its issue frequencies are sample data rather than frequencies measured from a real dataset.

### 9.5 The experiment is too small for human-agreement statistics

There are only three controlled candidates, and no independent human scores have been collected. Pearson correlation, Spearman correlation, RMSE, MAE, and calibration error therefore cannot yet be reported as human-agreement results.

With only three examples, even a very high correlation would be unstable and could mainly reflect the obvious Strong > Mixed > Weak ordering.

### 9.6 Deterministic matching is approximate

The deterministic judge uses approximate label similarity and estimated text geometry. It does not have full browser `getBBox()` rendering measurements. Complex paths, unusual SVG transforms, or ambiguous labels may require additional parsing support.

### 9.7 GPT evaluation can be perceptual and variable

The strong residual path illustrates that GPT may interpret visual topology differently from the SVG parser. Repeated runs are needed to measure run-to-run variance, especially for ambiguous connectors.

## 10. Recommended Next Step: Create One Shared Measurement Contract

The highest-priority next step is to replace judge-selected denominators with one centrally defined checklist derived from the description.

Each opportunity should receive a stable unique ID, for example:

```text
presence.component.input_tokens
presence.component.positional_encoding
connectivity.edge.add_to_attention
connectivity.edge.norm1_to_norm2_residual
details.label.attention
legibility.font.attention
legibility.overflow.attention
```

The same checklist should be used by:

- human raters;
- GPT-5.5; and
- the deterministic SVG judge.

Judges should return only:

- the IDs that failed;
- evidence for each failed ID; and
- `uncertain` when an opportunity cannot be judged reliably.

The evaluator—not the judge—should calculate the denominator and score. Unknown IDs, duplicate IDs, and incompatible opportunities should be rejected automatically.

This change would make comparisons interpretable, prevent denominator drift, and make the later Pearson/RMSE study valid.

## 11. Proposed Human Evaluation Study

### Step 1: Finalize scoring units

Define fixed opportunity IDs for all five dimensions. Decide explicitly:

- whether a visibly present but incorrectly labelled box counts as present;
- whether the same failure may affect both Presence and Details;
- how extra/unrequested components are penalized;
- how overlap opportunities are counted; and
- how uncertainty is handled.

### Step 2: Build a human scoring form

For each diagram, show the description and rendered candidate. For every opportunity ID, ask the rater to mark:

- Pass;
- Fail; or
- Uncertain.

Raters should also provide brief evidence for failed or uncertain cases. They should not see automated scores while rating.

### Step 3: Collect independent human ratings

Use at least two or three independent raters. First measure agreement among humans. If humans disagree substantially, the rubric or examples need clarification before the automated evaluator can be validated against them.

### Step 4: Expand the evaluation dataset

The three Transformer examples should remain as a smoke test, but the validation dataset should include substantially more diagrams and more diagram types. If feasible, a first study could use 30 or more candidates spanning:

- architecture diagrams;
- flowcharts;
- process diagrams;
- model diagrams;
- simple and dense layouts;
- strong, medium, and weak quality levels; and
- different kinds of controlled errors.

### Step 5: Calculate agreement metrics

For each candidate `i`, let `e_i` be the evaluator score and `h_i` the human consensus or mean score.

```text
RMSE = sqrt(mean((e_i - h_i)^2))
```

Report:

- Pearson correlation for linear association;
- Spearman correlation for ranking agreement;
- RMSE for the size of errors;
- MAE for easily interpreted average error;
- mean signed error for systematic over- or under-scoring;
- confidence intervals, preferably bootstrapped at the diagram level; and
- per-dimension results in addition to the overall score.

Human inter-rater reliability should also be reported before system-versus-human agreement. Appropriate choices depend on the final label format, such as an intraclass correlation for continuous scores or Krippendorff’s alpha for opportunity-level categorical decisions.

### Step 6: Test robustness

Run GPT multiple times on a subset of diagrams to measure score variability. Also test sensitivity to:

- dimension weights;
- label-similarity thresholds;
- endpoint tolerances;
- font thresholds; and
- diagrams with unusual SVG structure.

### Step 7: Calibrate weights from real data

Replace the sample issue-frequency file with observed frequencies from the validation corpus. The final weighting decision should be documented and fixed before reporting test-set performance.

## 12. Recommended Engineering Work Order

1. Introduce stable opportunity IDs and fixed denominators.
2. Change GPT output from self-selected counts to failed/uncertain opportunity IDs.
3. Change deterministic outputs to use the same IDs.
4. Add validation that rejects duplicate, unknown, or incompatible issue IDs.
5. Separate raw internal scores from rounded display scores.
6. Generate sensemaking dynamically from actual issues.
7. Add a human-rating CSV or spreadsheet template.
8. Add a script that calculates human agreement, Pearson, Spearman, RMSE, MAE, bias, and confidence intervals.
9. Expand the test corpus beyond the Transformer examples.
10. Run repeated-model and sensitivity experiments.
11. Calibrate final weights and freeze an evaluation version.

## 13. Suggested Result Table for the Human Study

| Candidate | Human overall | Evaluator overall | Error | Human Presence | Evaluator Presence | Human Connectivity | Evaluator Connectivity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Candidate 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Candidate 2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

The final report should include one table for aggregate scores, one per-dimension comparison, a human-versus-evaluator scatter plot, and a distribution of residual errors.

## 14. Current Commands and Result Locations

Run the deterministic smoke test without an API key:

```bash
PYTHONPATH=src python3 scripts/run_transformer_examples.py --offline
```

Run the live GPT-5.5 + deterministic evaluation:

```bash
export OPENAI_API_KEY='YOUR_KEY'
PYTHONPATH=src python3 scripts/run_transformer_examples.py
```

Results are written to:

```text
examples/transformer/results/
```

Important artifacts include:

- `summary.md` — readable table and sensemaking;
- `summary.json` — machine-readable summary scores;
- `description_expectations.json` — expectations extracted from the description;
- `candidate_strong.json` — complete strong-candidate report;
- `candidate_mixed.json` — complete mixed-candidate report;
- `candidate_weak.json` — complete weak-candidate report; and
- candidate PNG files — the images shown to GPT.

## 15. Current Status

The prototype is operational, tested, committed, and pushed. It accepts the intended inputs, produces interpretable dimension scores, preserves criterion-level evidence, and correctly orders the controlled candidates by quality.

The next milestone should not be adding more scoring complexity. It should be establishing a shared, fixed opportunity-level rubric across humans, GPT, and deterministic geometry. Once that measurement contract is implemented, the project will be ready for a credible human-agreement experiment and meaningful Pearson/RMSE reporting.
