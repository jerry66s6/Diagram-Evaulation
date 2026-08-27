# Transformer example results

Mode: **live gpt-5.5 + deterministic SVG**

| Candidate | Overall | Layout | Connectivity | Presence | Details | Legibility |
|---|---:|---:|---:|---:|---:|---:|
| candidate_strong | 0.972 | 1.000 | 0.950 | 0.962 | 1.000 | 1.000 |
| candidate_mixed | 0.762 | 0.939 | 0.679 | 0.762 | 0.447 | 1.000 |
| candidate_weak | 0.203 | 0.269 | 0.125 | 0.129 | 0.000 | 0.500 |

## Sensemaking

The ranking is **candidate_strong > candidate_mixed > candidate_weak**.

- **Strong (0.972)** is the positive control: every description-derived component, exact label, directed arrow, residual path, and legibility check passes.
- **Mixed (0.762)** keeps most of the layout (0.939) and remains readable (1.000), but missing positional/residual paths reduce connectivity to 0.679; abbreviated and placeholder labels reduce Details to 0.447.
- **Weak (0.203)** retains a few recognizable structures, so Layout is not zero (0.269), while missing components, wrong arrows, placeholder content, and small text drive Presence to 0.129, Connectivity to 0.125, Details to 0.000, and Legibility to 0.500.

Overall is a severity/frequency-weighted score, not a simple mean. In this sample, Connectivity has the largest weight (0.40), Presence and Legibility each have 0.20, and Layout and Details each have 0.10.

Each candidate JSON report contains every criterion-level score and issue from each participating judge.
