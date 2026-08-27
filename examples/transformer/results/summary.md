# Transformer example results

Mode: **offline deterministic smoke test**

| Candidate | Overall | Layout | Connectivity | Presence | Details | Legibility |
|---|---:|---:|---:|---:|---:|---:|
| candidate_strong | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| candidate_mixed | 0.813 | 0.952 | 0.750 | 0.909 | 0.364 | 1.000 |
| candidate_weak | 0.306 | 0.697 | 0.250 | 0.182 | 0.000 | 0.500 |

## Sensemaking

The ranking is **candidate_strong > candidate_mixed > candidate_weak**.

- **Strong (1.000)** is the positive control: every description-derived component, exact label, directed arrow, residual path, and legibility check passes.
- **Mixed (0.813)** keeps most of the layout (0.952) and remains readable (1.000), but missing positional/residual paths reduce connectivity to 0.750; abbreviated and placeholder labels reduce Details to 0.364.
- **Weak (0.306)** retains a few recognizable structures, so Layout is not zero (0.697), while missing components, wrong arrows, placeholder content, and small text drive Presence to 0.182, Connectivity to 0.250, Details to 0.000, and Legibility to 0.500.

Overall is a severity/frequency-weighted score, not a simple mean. In this sample, Connectivity has the largest weight (0.40), Presence and Legibility each have 0.20, and Layout and Details each have 0.10.

Each candidate JSON report contains every criterion-level score and issue from each participating judge.
