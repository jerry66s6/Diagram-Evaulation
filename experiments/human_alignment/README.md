# Human alignment

Do the judge's scores agree with the human study? All 300 human-study diagrams were rated by
people (`human_ratings.json`: 13 dimensions, 1-10, one anonymized rater per diagram). HS001-HS100
are used to tune the correctness prompt; HS101-HS300 are held out for one final test.
`report.html` shows every version tried on HS001-HS100.

```bash
python3 run.py score --first 100 --wording d2      # gpt-6-luna only; resumable; raw outputs in results/
python3 run.py compare --first 100 --wording d2    # summary_d2.json, judge_scores_d2.json
python3 run.py tune --wordings v1,d1,d2,d6         # versions side by side
python3 run.py report                              # report.html
```

Judge scores are put on 0-10 (formula scores x10, Beauty /10); human scores stay as rated.
Metrics per dimension: MAE, RMSE, share within 1 point, Spearman, Kendall, Pearson, AUC for
clean vs broken diagrams, inserted errors caught (dimension below 8), and false alarms (a clean
diagram humans rated 9+ that the judge scored below 8).

## Versions (correctness)

| Version | What it is |
|---|---|
| v1 | The evaluator as it was in git: caption sketch, item-by-item check (3 samples), formula score |
| v1r | v1 again, for run-to-run noise |
| v2 | v1 + four general prompt sentences |
| d1 | Direct: each dimension asked once with image and caption, the model gives 1-10; no sketch, no formula |
| d1r | d1 again, for run-to-run noise |
| **d2** | d1 + four general sentences from the d1 mistakes (chosen; now `src/diagram_correctness/direct.py`) |
| d3 | d2 + the caption sketch as a checklist |
| d4 | d2 + one sentence per dimension from the d2 mistakes (tight crop is not cut off; exclusion box in the main chain is fine; wrong-box labels are Details; typos do not count; Legibility told the study's display size) |
| d5 | d4 without the wrong-box and typo sentences (they excused real errors), and a softer cost for small text |
| d6 | d5 with a milder Legibility wording: small text that still reads at the display size is fine |
| d7 | The d6 ideas as short general rules with no sizes, plus a Presence rule: a blank, cut-off, or hidden box does not show its component |
| **d8** | Best part of each: d7 Presence, d6 Connectivity, d2 Details, a Layout rule that names a border lying on the edge, and Legibility told why figures look smaller (fitted to a screen) without numbers (best general prompt) |
| d9 | d8 with Legibility judged as a person would, by the hardest text, naming weak contrast; and Details treating a label in the wrong place as a small problem (neither helped) |
| d10 | d8 with a size sentence in words (no number) for the 32 figures the study showed much smaller, and a Details rule: same thing in other words is accurate; a different thing, a contradiction, or a left-out item is wrong |
| d11 | d8 with only the first half of the d10 Details rule (same thing in other words is accurate; a different thing or a contradiction is wrong) |
| **d12** | d11 with Legibility back to the d2 wording (no screen-size sentence) and Details also calling a label wrong when it drops a count, value, or condition the caption gives (current candidate) |

| d13 | d12 with Connectivity checked connection by connection (line present, arrowhead direction), plain lines and arrows to a group or panel accepted (worse: Spearman 0.49, more false alarms; catches 5 of 6 reversed arrows) |
| d14 | d13 plus "a missing, cut-off, or faint box breaks the connections through it" (no change from d12) |
| d15 | d12 with a Layout severity rule (one obvious layout error belongs in 3-5) and wrap-around flows accepted (worse: it lowered clean figures too, false alarms 2 -> 10) |

d5-d15 re-ran only the dimensions whose prompt changed; the others reuse the result files of the version with the
same prompt (`COPIED` in `run.py`).

## Results on HS001-HS100 (overall correctness)

| | v1 | v2 | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | d11 | d12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spearman | 0.51 | 0.56 | 0.74 | 0.76 | 0.76 | 0.77 | 0.79 | 0.79 | 0.77 | 0.78 | 0.79 | 0.75 | 0.78 | 0.79 |
| Kendall | 0.36 | 0.40 | 0.57 | 0.59 | 0.58 | 0.61 | 0.62 | 0.63 | 0.59 | 0.61 | 0.61 | 0.57 | 0.61 | 0.61 |
| MAE (0-10) | 1.19 | 1.09 | 0.77 | 0.73 | 0.75 | 0.73 | 0.74 | 0.70 | 0.73 | 0.70 | 0.70 | 0.73 | 0.72 | 0.70 |
| Clean vs broken AUC (humans 0.91) | 0.73 | 0.74 | 0.81 | 0.84 | 0.85 | 0.87 | 0.86 | 0.87 | 0.83 | 0.85 | 0.83 | 0.85 | 0.85 | 0.85 |
| False alarms on clean diagrams | 51 | 51 | 24 | 24 | 29 | 23 | 27 | 23 | 29 | 14 | 16 | 15 | **13** | 15 |

These numbers use the study ratings as exported. Since then the author re-rated Legibility on the 45 HS001-HS100
cases where the study rating and the d2 judge differed by 2+ points (`legibility_review.csv`: study and new score)
and changed 7 of them (HS014, HS018, HS041, HS065, HS066, HS071, HS086); `human_ratings.json` now holds the new
scores, and git history has the export. With them, Legibility agreement rises for every wording without numbers
(d2/d12 Spearman 0.64 -> 0.74, MAE 1.43 -> 1.21) and d6's numbers fall behind (0.64); overall correctness for d12 is
Spearman 0.80, Kendall 0.63, MAE 0.66. All 7 changes moved toward the judge, as re-rating only disagreements must,
so these figures are optimistic; HS101-HS300 keep the study ratings.

Per dimension, d2 -> d8 (Spearman / MAE / false alarms): Layout 0.50 / 1.58 / 10 -> 0.61 / 1.19 / 2, Connectivity
0.57 / 1.23 / 6 -> 0.55 / 1.26 / 6, Presence 0.54 / 0.85 / 1 -> 0.73 / 0.76 / 0, Details 0.64 / 1.04 / 4 (same prompt
and run), Legibility 0.64 / 1.43 / 3 -> 0.66 / 1.39 / 2. Overall Spearman moves by about 0.01-0.02 between identical
runs, so the overall gain over d2 is small; the Layout and Presence gains and the fewer false alarms are clear.

1. **Direct scoring beats sketch + formula on all five dimensions**, with less than half the
   input tokens and a fifth of the output tokens.
2. **The formula was the main problem, not the model.** v1 found most inserted errors, but one
   detected error usually cost 2.5 points while humans often drop a dimension to 4-6; and clean
   figures lost points when the caption sketch did not match how they were drawn (exclusion boxes
   chained into the main flow in 21 of 100 sketches; one arrow serving several boxes).
3. **d2 improved on d1 beyond run-to-run noise** on Details and Connectivity, the dimensions its
   sentences targeted. d1 and d1r agree with humans equally (Spearman 0.74 both), and averaging the
   two runs gives only 0.75, so one call per dimension is enough.
4. **Adding the sketch back (d3) did not help**: the same overall, Connectivity worse, about 1.6x the cost.
5. **d4-d6: Layout gained most** (Spearman 0.50 -> 0.64, MAE 1.58 -> 1.28, AUC 0.76 -> 0.80). d2 still called tightly
   cropped figures cut off when a box border lay on the image edge; saying that only sliced text or an incomplete box
   is cut off lowered Layout false alarms on clean figures from 10 to 7 and still caught all three inserted crops.
6. **The study's display size explains much of the Legibility gap.** The study showed each figure fitted to a panel
   about 660x360 pixels, so tall or large figures looked far smaller to the raters than to the model: d2's judge-minus-human
   Legibility difference correlated -0.48 with that scale (figures shown under 40%: humans 5.7, d2 7.8). Telling the
   model the scale in numbers ranks Legibility better (0.64 -> 0.68) but overshoots the level (d4 bias -1.2, d6 -0.8,
   d2 +0.6). Wordings without numbers (d7: look at the whole figure as a reader; d8: figures are fitted to a screen;
   d9: judge as a person, by the hardest text, weak contrast counts) change nothing: at every display scale they
   score like d2, and on figures humans rated 3 (HS041, HS086) the judge still says the small text "remains readable".
   A size sentence in words for just the figures shown much smaller (d10: "this figure is large or tall, so they see
   it much smaller than its file") did not move them either (those 32 figures: humans 6.1, d8 7.7, d10 8.0). Only the
   number (d6) changed the judge's reading. The judge gets the full-resolution file and cannot
   un-see detail, so a prompt alone does not close this gap; showing the judge the figure at the size the raters saw
   would be an input change, not a prompt change.
7. **Presence: "visibly present" taken literally** (d7). Humans marked Presence down when a component was blank, cut
   off, or hidden (5-7) while the judge kept 10, since the rubric hands those to other dimensions. Saying such a box
   does not show its component raised Presence Spearman 0.59 -> 0.73.
8. **How a Layout rule is worded matters.** The short d7 rule ("content touching the edge is not cut off") lost the
   gain (false alarms 7 -> 14): the judge still read a border lying on the edge as cut off. Naming that case as a rule
   (d8) brought false alarms down to 2 and still caught every inserted crop.
9. **Sentences that excuse a kind of error backfire.** "Typos and abbreviations do not count" (Details) made the judge
   pass real wrong labels ("big prices" for "pig prices"), and "a label in the wrong box is a Details problem"
   (Connectivity) excused real arrow errors; both were dropped in d5. Likewise in d9, making a label in the wrong
   place a small problem lowered Details on every measure (Spearman 0.64 -> 0.60, AUC 0.76 -> 0.71): the judge then
   caught 2 of 8 label swaps against the humans' 3, and humans and judge disagree on which swaps they catch, not on
   how strict to be. "A label that names the same thing in other words is accurate" (d11) did what it says (clean
   figures with an abbreviation, a dropped word, or a typo went from 7-8 to 10, as humans rated them) but also passed
   some broken ones; Details ends level with d2 (Spearman 0.63 vs 0.64, MAE 1.03 vs 1.04, within 1 point 0.76 vs
   0.71). Adding "or leaves out what the caption asks it to state" (d10) made the judge flag omissions on clean figures.
   Details also moves a lot between identical runs (d1 vs d1r MAE 1.30 vs 1.16), so differences this small are noise.
   d12 keeps the d11 rule and adds "or drops a count, value, or condition the caption gives": the wording nitpicks
   stay at 10 and the dropped counts and conditions are caught again (Details AUC 0.77, the best so far; 5 of 8 label
   swaps caught). Aggregate Details agreement stays within noise (Spearman 0.60, MAE 1.09).
10. **Beauty v6 was not tuned.** Centering and Space Use do not track the human ratings (Spearman
   about 0); Color Consistency and Style Uniformity do best (0.51, 0.42).

Caveats: the prompt was tuned on these 100 diagrams, so the result has to be confirmed on
HS101-HS300. Each diagram has one rater; humans often lowered every dimension of a diagram with
any error, which the judge, scoring each dimension on its own, does not copy.
