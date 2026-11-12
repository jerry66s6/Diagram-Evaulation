# Beauty-axis judge isolation experiment

## Purpose

The current evaluator is **v6: eight Beauty dimensions**, implemented by
`judge_beauty` in the canonical [`beauty.py`](../src/diagram_correctness/beauty.py).
The human-study UI asks participants about the same eight dimensions. Older
evaluator modules, runners, and deterministic Canvas Fill have been removed.
Saved historical reports and frozen alignment snapshots remain available;
the historical sections below do not describe the current scoring code.

### Historical v3 experiment

The original experiment tests whether independent vision-model prompts can measure five
visual-quality dimensions without allowing a defect in one dimension to dominate
the others:

| Dimension | Measurement |
|---|---|
| Aesthetic craft | Execution quality of shapes, strokes, typography, connectors, and rendering consistency |
| Palette | Coordinated chromatic hues with stable semantic color roles |
| Ink balance | The fraction of the canvas covered by fills, strokes, text, texture, and decoration |
| Density | Local clearance between neighboring elements relative to their size |
| Balance | Foreground margins and visual-centroid placement on the canvas |

The experiment uses a synthetic five-node flowchart and five targeted variants.
Text and logical connections remain constant. Each judge call receives only one
dimension's rubric, the candidate image, and a strict score/reasoning schema.

The v3 prompt contract narrows each dimension to one concrete observable. Each
judge receives the complete ownership map, assumes the other four dimensions are
perfect, applies a dimension-specific mental mask, records its observation before
scoring, and then uses observable-anchored score bands. The public output key
remains `aesthetics` for compatibility, although its meaning is now specifically
Aesthetic Craft.

The experiment reports:

- **Sensitivity:** the target dimension's score drop from baseline.
- **Specificity:** target drop divided by all positive score drops.
- **Isolation:** specificity multiplied by `min(target drop / 15, 1)`, preventing
  an insensitive judge from appearing isolated merely because nothing changed.

## Portable implementation

The external version deliberately excludes the prototype's internal authentication,
endpoint, routing headers, deployment identifiers, and dependency lockfile.
Live runs instead use the repository's existing public OpenAI backend and
`OPENAI_API_KEY`. Historical provider and deployment metadata are omitted because
they are not necessary to understand the method or results.

The single current runner renders eight-dimension fixtures without inventing
offline judge scores:

```bash
PYTHONPATH=src python3 scripts/run_beauty_ablation.py
```

This generates PNG fixtures, geometry/color measurements, a JSON summary, and an
HTML report under `examples/beauty_ablation/v6/fixtures/`. Historical reports
remain in their original locations; the old five-dimension replay code is removed.

Re-score the generated diagrams with the model in `config/evaluator.json`:

```bash
export OPENAI_API_KEY='...'
PYTHONPATH=src python3 scripts/run_beauty_ablation.py --live
```

Live scoring averages three calls per image and dimension by default so model
variance is not mistaken for cross-dimension leakage. Use `--repeats N` to change
that count and `--model MODEL` to override the configured judge model. Live
outputs go to `examples/beauty_ablation/v6/live/` and are ignored by Git.

## Original imported results

The imported second run produced:

| Variant | Aesthetics | Palette | Ink balance | Density | Balance |
|---|---:|---:|---:|---:|---:|
| Baseline | 64 | 95 | 84 | 88 | 96 |
| Aesthetics degraded | **35** | 90 | 85 | 82 | 95 |
| Palette degraded | 55 | **72** | 88 | 88 | 82 |
| Ink balance degraded | 24 | 92 | **18** | 72 | 94 |
| Density degraded | 68 | 95 | 86 | **78** | 95 |
| Balance degraded | 62 | 95 | 78 | 68 | **45** |

Every intervention lowered its target dimension. The 15-point leakage check also
flags three off-diagonal drops:

- ink-balance degradation lowers aesthetics and density;
- balance degradation lowers density.

These may represent genuine visual correlations, prompt coupling, or both. The
matrix alone cannot distinguish those causes.

## V3 Ceres iterations

The final two source commits also preserved three complete v3 runs, each using
three repeats per image and dimension. Internal route identifiers are removed from
the portable dashboards:

| Iteration | Overall isolation | Main finding |
|---|---:|---|
| [1](../examples/beauty_ablation/ceres/iteration1.html) | 62.7 | Craft and Balance perturbations were too weak. |
| [2](../examples/beauty_ablation/ceres/iteration2.html) | 63.4 | Craft typography changed local clearance; Ink coverage remained in the healthy band. |
| [3](../examples/beauty_ablation/ceres/iteration3.html) | **89.2** | All target sensitivities passed; only Ink Balance leaked into Density. |

Iteration 3 produced:

| Perturbation | Target drop | Worst non-target leak | Isolation |
|---|---:|---:|---:|
| Aesthetic Craft | 39 | 2 (Balance) | 92.9 |
| Palette | 31 | 0 | 100.0 |
| Ink Balance | 56 | 22 (Density) | 64.4 |
| Density | 54 | 3 (Aesthetic Craft) | 91.5 |
| Balance | 64 | 1 (Aesthetic Craft) | 97.0 |

The remaining Ink Balance-to-Density leak exposed a fixture bug: extra border
boxes reduced box-to-box clearance. The latest generator removes those borders
and uses neutral full-canvas bands to increase only ink coverage. That corrected
fixture was not rerun because of the source environment's hourly quota, so this
repository does not claim a corrected score.

## V4-V6: current Beauty set

Content and structure are judged by correctness; these dimensions are purely visual.
Each version was tested two ways: an **isolation** run (one baseline plus single-variable
fixtures; each fixture should lower only its own dimension) and a **breaking** round
(objectively bad flowcharts that try to score high on every dimension).

### Dimensions (v6, final)

| Dimension | Judge | Measures | Added |
|---|---|---|---|
| Centering | Model | Full-diagram opposite-margin imbalance normalized by canvas width/height | v4; canvas-relative in human-alignment v2 |
| Space Use | Model | Peer-node/group clearance relative to size; intentional layered symbols count as one group | v4; grouped-symbol handling in human-alignment v2 |
| Color Consistency | Model | Coherent role/section mapping, without a three-hue limit | v4; expanded in human-alignment v2 |
| Contrast | Model | Box boundaries (outline and/or fill) and arrows stand out from the background | v4; expanded in human-alignment v2 |
| Alignment | Model | Deliberate geometry: no near-misses (almost aligned, almost level); arrows take the cleanest path for the layout (grid by default; rings, hubs, trees, loop-backs may use diagonal or curved arrows); consistent endpoints | v5, expanded v6 |
| Style Uniformity | Model | Semantically equivalent elements and connectors share shape, size, border, font, arrow style | v5; semantic roles in human-alignment v2 |
| Resolution | Model | No blur, pixelation, or compression artifacts | v6 |
| Canvas Fill | Model | Elements not so small that most of the canvas is empty (estimated footprint score: 0 at 3%, 100 at 6%+; large enclosed regions such as the inside of an arrow ring are not counted) | v5; VLM in current v6 |

All eight prompts and the main caller: `src/diagram_correctness/beauty.py`.
Canvas Fill-only caller: `scripts/run_canvas_fill.py`. Canvas Fill is model-judged;
the old deterministic implementation and its diagnostics have been removed.
The historical version results below the smoke check predate the Canvas Fill VLM
change and used deterministic Canvas Fill; they are not evidence of the new
prompt's performance. Historical diagnostics and the current VLM both use the
3%/6% rule, but estimate footprint differently.
Space Use now scores only peer-node/group clearance, treating intentional layered
symbols as one footprint, not group span or the amount of unused canvas.
Adding canvas padding or uniformly scaling nodes and gaps together
should preserve its score. Moving the same nodes farther apart can improve Space
Use while leaving Canvas Fill unchanged. Historical isolation results below used
the earlier span-and-gap Space Use prompt and have not been rerun with this change.
Centering now compares only the full diagram's opposite margins, including title
and connectors. Each axis scores `100 * (1 - abs(a - b) / canvas_dimension)`;
the final score is the rounded minimum of the two axes. This avoids large
penalties for tiny differences between nearly zero margins.
Internal label centering and gap equality no longer lower Centering. Their old
v6 fixture IDs are retained as placement controls. Historical Centering results
below used the old three-part prompt and have not been rerun with this change.

The current evaluator matches the evaluated human-alignment v2 snapshot across
all eight prompts. Its five changes also cover outline-or-fill Contrast,
semantic-role Style Uniformity, and coherent multi-hue Color Consistency.
GPT-6 luna validation RMSE was 1.8605 on the fixed 20-image mini-validation set
and 2.3160 on the full 100-image validation set, not the held-out test set.
See [the alignment results](../experiments/beauty_alignment_fresh/RESULTS.md).

### Clearance-only Space Use smoke check

Historical, before the human-alignment v2 edits: one live call per dimension and
fixture on `dev-gpt-56-ceres`:

| Fixture | Space Use | Canvas Fill |
|---|---:|---:|
| Baseline | 99 | 100 |
| Same nodes crowded together | 61 | 100 |
| Tiny nodes spread apart | 100 | 13 |
| Baseline centered on a canvas twice as wide and twice as tall | 98 | 0 |

Crowding lowered Space Use while preserving Canvas Fill. Adding padding left
Space Use within one point while lowering Canvas Fill, and tiny, well-spaced nodes
scored high only on Space Use. These are small, single-run separation checks,
not evidence of perfect isolation or repeatability across arbitrary diagrams.

### Canvas Fill VLM smoke check

Open the [small review dashboard](../examples/canvas_fill_vlm/index.html) for
fixture images, expected behaviors, scoring-formula checks, and estimation limits.

The current prompt was exercised through the existing local backend on
`dev-gpt-56-ceres`, with one call per fixture. Saved
[responses and prompt](../examples/canvas_fill_vlm/results.json) include the
observations and reasoning; [input diagrams](../examples/canvas_fill_vlm/diagrams/)
are retained alongside them.

| Fixture | VLM score (3%/6%) | VLM estimated footprint | Historical footprint |
|---|---:|---:|---:|
| Baseline | 100 | 9.8% | 10.51% |
| Tiny nodes spread across canvas | 13 | 3.4% | 4.08% |
| Pale nodes (contrast-only defect) | 100 | 9.5% | 10.14% |
| Straight-arrow cycle | 93 | 5.8% | 6.26% |

All four responses applied the stated scoring formula to their estimated areas.
The tiny-node fixture scored below 50, pale nodes matched baseline at 100,
and the cycle stayed near 100 with an estimated footprint consistent with excluding
its empty interior. VLM area estimates are approximate: tiny nodes scored 13
versus the historical diagnostic's 36 because their estimated footprints differ.
Blank, outline-only, and exact-boundary fixtures were not live-tested in this
four-image run. This is a smoke check, not an accuracy or human-alignment claim.

Reproduce with the existing local backend:

```powershell
python scripts\run_canvas_fill.py examples\canvas_fill_vlm\diagrams\baseline.png examples\canvas_fill_vlm\diagrams\degraded_canvas_fill.png examples\canvas_fill_vlm\diagrams\degraded_contrast.png examples\canvas_fill_vlm\diagrams\control_cycle_straight.png --backend local\beauty_alignment_backend.py --output examples\canvas_fill_vlm\results.json
```

### Results by version

| Version | Dimensions | Isolation | Mean off-target change | Bad diagrams passing every dimension |
|---|---:|---:|---:|---|
| [v4](../examples/beauty_ablation/v4/iteration2/index.html) | 4 | 97.5 | - | 5 of 7 |
| [v5](../examples/beauty_ablation/v5/iteration1/index.html) | 7 | 96.9 | 0.48 pts | 7 of 8 new |
| [v6](../examples/beauty_ablation/v6/iteration6/index.html) | 8 | 87.8 | 0.48 pts | 4 of 9 new |

v4 breaks that were judged subjective (dominant title, decoration, oversized, neon colors)
are excluded from later rounds.

### v6 isolation

Every fixture drops only its own dimension (largest off-target change: 2 points), except the
mixed-detached-arrows fixture, which Alignment does not reliably catch (see Known gaps).
Five valid non-grid designs (two cycles, a loop-back arrow, a tree, a hub) score 85+ on every
dimension.

| Fixture | Target: baseline -> fixture |
|---|---|
| Diagram off-center / labels in corners / uneven gaps | Centering 99 -> 26 / 42 / 52 |
| Nodes packed tightly | Space Use 98 -> 58 |
| Random box colors | Color Consistency 100 -> 30 |
| Pale boxes and arrows | Contrast 96 -> 35 |
| Nodes off-grid / tilted / diagonal arrows / mixed detached arrows | Alignment 99 -> 54 / 55 / 56 / **98** |
| Mixed shapes, fonts, borders | Style Uniformity 100 -> 35 |
| Tiny elements | Canvas Fill 100 -> 35 |
| Blur / pixelation | Resolution 99 -> 52 / 59 |

### v6 breaking round

| Group | Result |
|---|---|
| Confirmed defects from earlier rounds | 11 of 11 caught by their assigned dimension |
| Design-choice control (all arrows equally detached) | Correctly not penalized |
| New candidates caught | JPEG artifacts, noisy background (Resolution); rotated diagram (Alignment); hairline arrows (Contrast); mixed dashed/solid arrows (Style Uniformity) |
| New candidates passing every dimension | Ghosting, labels touching box edges, arrowheads buried in boxes, stray marks (correctness does flag stray marks) |

### Known gaps

- **Alignment endpoint consistency is unreliable.** Arrows that are detached inconsistently
  (some touch their boxes, some stop short) are caught in the breaking round (18px: 61;
  45px: 75, runs 60/67/98) but missed by the equivalent isolation fixture (98). Across
  Alignment wordings the result flips between runs. Treat this sub-check as best-effort; a
  deterministic gap measurement would be needed for a reliable signal.
- Alignment's valid-layout handling was tested on five designs. A purely general wording
  ("no near-misses", no default layout) let the judge rationalize defects, so the final
  wording keeps a grid default with named exceptions; unusual layouts may still be misjudged.
- One synthetic flowchart and a single model family; scores are not calibrated against
  human ratings.

```bash
PYTHONPATH=src python3 scripts/run_beauty_ablation.py          # fixtures only; no scores
PYTHONPATH=src python3 scripts/run_beauty_ablation.py --live   # 20 images x 8 model dimensions x 3 repeats
```

## Limitations and next steps

- One synthetic diagram and one intervention per dimension are not a calibrated
  benchmark.
- The corrected Ink Balance fixture still requires a full repeated live run.
- Intervention strengths differ, so score-drop magnitudes do not rank intrinsic
  judge sensitivity.
- A larger public calibration set needs human labels, repeated judge runs, and
  confidence intervals.
- Palette, ink balance, density, and balance may benefit from deterministic
  image/SVG measurements combined with a vision judge.
- Future experiments should use factorial or multi-dose interventions to separate
  real dimension correlations from prompt leakage.
