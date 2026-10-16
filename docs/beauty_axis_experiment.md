# Beauty-axis judge isolation experiment

## Purpose

This experiment tests whether independent vision-model prompts can measure five
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

The checked-in prompts and generated SVG fixtures reflect the latest source state.
The offline replay remains the original imported result set so that this public
port does not invent scores for a fixture that was not rerun.

Run the checked-in historical results without an API call:

```bash
PYTHONPATH=src python3 scripts/run_beauty_ablation.py
```

This generates equivalent SVG interventions, a CSV, JSON summary, and a portable
HTML report under `examples/beauty_ablation/reproduced/`.

Re-score the generated diagrams with the model in `config/evaluator.json`:

```bash
export OPENAI_API_KEY='...'
PYTHONPATH=src python3 scripts/run_beauty_ablation.py --live
```

Live scoring averages three calls per image and dimension by default so model
variance is not mistaken for cross-dimension leakage. Use `--repeats N` to change
that count and `--model MODEL` to override the configured judge model. Live
outputs go to `examples/beauty_ablation/live/` and are ignored by Git.

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
| Centering | Model | Whole diagram centered; labels centered in their shapes; even gaps between neighbors | v4, expanded v6 |
| Space Use | Model | Nodes not condensed while canvas space is empty | v4 |
| Color Consistency | Model | One coherent theme; each role keeps one hue | v4 |
| Contrast | Model | Boxes and arrows stand out from the background | v4 |
| Alignment | Model | Deliberate geometry: no near-misses (almost aligned, almost level); arrows take the cleanest path for the layout (grid by default; rings, hubs, trees, loop-backs may use diagonal or curved arrows); consistent endpoints | v5, expanded v6 |
| Style Uniformity | Model | Same-role elements share shape, size, border, font, arrow style | v5 |
| Resolution | Model | No blur, pixelation, or compression artifacts | v6 |
| Canvas Fill | Python | Elements not so small that most of the canvas is empty (footprint score: 0 at 3%, 100 at 6%+; large enclosed regions such as the inside of an arrow ring are not counted) | v5 |

Prompts: `src/diagram_correctness/beauty_v6.py`; Canvas Fill: `canvas_fill.py`.

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
PYTHONPATH=src python3 scripts/run_beauty_v6_ablation.py          # fixtures + Canvas Fill only
PYTHONPATH=src python3 scripts/run_beauty_v6_ablation.py --live   # 15 images x 7 model dimensions x 3 repeats
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
