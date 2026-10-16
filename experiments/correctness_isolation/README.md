# Correctness isolation experiment

Does each kind of error lower only its own correctness dimension? This is the Beauty
isolation experiment (`docs/beauty_axis_experiment.md`) applied to Correctness, on real PMC
flowcharts instead of one synthetic diagram. Nothing outside this folder is changed:
candidate wording fixes are added to the judge prompt in memory only.

```bash
python3 run.py make                       # 30 diagrams (10 per difficulty) + single-error images + controls
python3 run.py score --per-level 4        # gpt-6-luna; baselines 3x, each error image once; resumable
python3 run.py report                     # report.html, summary.json, results.csv
python3 run.py score --per-level 4 --wording v2   # Layout wording fix (in memory)
python3 run.py score --wording v3 --only-kinds erase_arrow,detach_arrow,reverse_arrow
python3 run.py report --wording v2        # report_v2.html, ...
```

## Setup

- 30 positive PMC flowcharts not in the human study (10 easy, 10 medium, 10 hard). Each is
  scored as is (3 runs, averaged) and with exactly one added error, using the 16 error types
  that made the dataset's negatives, plus two harmless controls (white margin, grayscale).
- Scored: all error images of 12 diagrams (4 per difficulty) and the arrow errors of all 30;
  171 error images in total. Same caption and frozen expectation sketch for every version of
  a diagram. Cost about $2.
- Metrics as in the Beauty experiment: target drop (sensitivity), specificity, isolation =
  specificity x min(target drop / 15, 1), mean |off-target change|, and errors no dimension caught.

## Results (current wording)

Overall isolation **35.5** (Beauty v6: 92.9), mean |off-target change| 7.6 points, 49 of 171
errors not caught by any dimension (no drop of 15+). Baseline run-to-run SD: Layout 4.6,
Connectivity 3.6, others about 1. Controls stayed within noise (largest change 3.8 points).

| Error | Target | Caught | Isolation | Main other drop |
|---|---|---:|---:|---|
| shrink_label | legibility | 83% | 73.9 | - |
| fade_label | legibility | 75% | 70.8 | - |
| remove_box | presence | 92% | 60.7 | - |
| crop_edge | layout | 92% | 46.0 | Presence 31 (the cut-off box is gone) |
| enlarge_label | legibility | 75% | 40.4 | Layout 26 |
| reverse_arrow | connectivity | 62% | 31.6 | Layout 15 (reading order) |
| swap_labels | details | 27% | 10.1 | Connectivity 22, Layout 17 |
| erase_arrow | connectivity | 20% | 17.6 | - |
| detach_arrow | connectivity | 12% | 12.5 | - |
| low_resolution | legibility | 17% | 7.0 | Layout 25 |
| blank_label / truncate_label | details | 50% / 33% | 48 / 37 | - |

All 16 types are in `report.html`.

## Conclusions

1. **The dimensions do not conflict.** Presence and Legibility errors stay in their dimension.
2. **Layout double-counts two things**, both fixable with wording: text spilling out of its box
   is also counted as boxes overlapping, and blur at the image edge is read as "cut off".
3. **Some cross-dimension drops are correct, not leaks.** Swapping two labels makes the arrows
   contradict the described flow (Connectivity); cropping a box removes it (Presence); a
   reversed arrow also runs against the reading order (Layout order and Connectivity overlap).
4. **Sensitivity, not isolation, is the main limit.** gpt-6-luna misses many small errors:
   erased or detached arrows, emptied or shortened labels.
5. **The first arrow errors were mostly invisible** (a fixture bug like the Beauty Ink Balance
   one): they hit box borders or table dividers, or "reversed" arrows with no head. 21 of 25
   changed fewer than 150 pixels. `diagrams/pipeline/defects.py` now edits only real arrows
   (arrowhead and length checked) and `negatives.py` requires a visible change; the 967 dataset
   negatives with arrow errors (57 in the human study) were remade from re-downloaded originals.

## Wording tests (in memory)

| Test | Change | Effect |
|---|---|---|
| v2 | Layout overlap: text spilling out of its box is Legibility; count only colliding shapes | enlarge_label Layout leak 26 -> 15, isolation 40 -> 55; overlap_box isolation 19 -> 37 |
| v2 | Layout canvas: blur is not cut off | low_resolution Layout leak 25 -> 19 (boxes touching the edge still read as cut) |
| v3 | Connectivity connections: no visible line means not connected, even if aligned | erase_arrow caught 3 -> 5 of 15; reverse_arrow 5 -> 6 of 8 |
| v3 | Connectivity endpoints: trace each arrow; a gap is dangling | detach_arrow caught 1 -> 0 of 8 (no gain) |

No false alarms: originals' Connectivity 82.8 -> 82.9 (v3), controls unchanged (v2).
Single runs on small samples: these are directions, not precise effects.

**Recommended minimal change** to `config/rubric.json`: add the two v2 Layout sentences and
the v3 Connectivity-connections sentence; leave out the endpoints sentence. Not applied yet.
