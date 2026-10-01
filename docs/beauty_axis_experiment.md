# Beauty-axis judge isolation experiment

## Purpose

This experiment tests whether independent vision-model prompts can measure five
visual-quality dimensions without allowing a defect in one dimension to dominate
the others:

| Dimension | Measurement |
|---|---|
| Aesthetics | Overall visual sophistication, coherence, and polish |
| Palette | A moderate number of coordinated, purposeful colors |
| Ink balance | Neither an almost-empty canvas nor excessive visual ink |
| Density | Comfortable element spacing rather than a cramped cluster |
| Balance | A centered composition with balanced surrounding whitespace |

The experiment uses a synthetic five-node flowchart and five targeted variants.
Text and logical connections remain constant. Each judge call receives only one
dimension's rubric, the candidate image, and a strict score/reasoning schema.

## Portable implementation

The external version deliberately excludes the prototype's internal authentication,
endpoint, routing headers, deployment identifiers, and dependency lockfile.
Live runs instead use the repository's existing public OpenAI backend and
`OPENAI_API_KEY`. Historical provider and deployment metadata are omitted because
they are not necessary to understand the method or results.

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

Use `--model MODEL` to override the configured judge model. Live outputs go to
`examples/beauty_ablation/live/` and are ignored by Git.

## Historical results

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

## Limitations and next steps

- One synthetic diagram and one intervention per dimension are not a calibrated
  benchmark.
- Intervention strengths differ, so score-drop magnitudes do not rank intrinsic
  judge sensitivity.
- A larger public calibration set needs human labels, repeated judge runs, and
  confidence intervals.
- Palette, ink balance, density, and balance may benefit from deterministic
  image/SVG measurements combined with a vision judge.
- Future experiments should use factorial or multi-dose interventions to separate
  real dimension correlations from prompt leakage.
