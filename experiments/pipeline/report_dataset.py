"""One self-contained page: the final pipeline on the 6,000 dataset diagrams.

    python3 experiments/pipeline/report_dataset.py   # report_dataset.html; run `classify-dataset` first
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S = json.load(open(HERE / "dataset_results.json"))
STUDY = json.load(open(HERE / "results_final.json"))
pred = list(csv.DictReader(open(HERE / "dataset_predictions.csv")))
defects = {r["id"]: json.loads(r["defects"]) for r in csv.DictReader(open(ROOT / "diagrams" / "data.csv")) if r["label"] == "negative"}
# From experiments/pipeline/dataset_results/score.log (not in git): the two scoring runs and the re-run of refused calls.
RUN = {"calls": 78000, "hours": 20, "cost": 23.5, "refused": 71}

NAMES = {"detach_arrow": "Arrow detached from its box", "reverse_arrow": "Arrow reversed", "erase_arrow": "Arrow erased",
         "fade_label": "Label faded", "low_resolution": "Low resolution", "recolor_boxes": "Boxes recolored",
         "pixelated": "Pixelated", "faint_shapes": "Shapes faint", "duplicate_box": "Box duplicated",
         "shrink_label": "Label shrunk", "tilt_boxes": "Boxes tilted", "swap_adjacent_steps": "Two adjacent steps swapped",
         "enlarge_label": "Label enlarged", "off_center": "Drawing off center", "truncate_label": "Label truncated",
         "overlap_box": "Boxes overlap", "arrow_through_box": "Arrow runs through a box", "swap_labels": "Two labels swapped",
         "mixed_edges": "Mixed line styles", "remove_box": "Box removed", "crop_edge": "Edge cropped",
         "blank_label": "Label blanked", "shrink_drawing": "Drawing shrunk"}

bad = [p for p in pred if p["label"] == "negative"]
good = [p for p in pred if p["label"] == "positive"]
caught, flagged = S["bad_caught"], S["good_flagged"]
assert caught == sum(p["called_bad"] == "1" for p in bad) and flagged == sum(p["called_bad"] == "1" for p in good)

by_count = defaultdict(lambda: [0, 0])  # number of inserted errors -> [caught, diagrams]
by_type = defaultdict(lambda: [0, 0, 0, 0])  # error type -> [caught, diagrams with it, caught when it is the only error, diagrams with only it]
for p in bad:
    d, c = defects[p["id"]], int(p["called_bad"])
    by_count[len(d)][0] += c
    by_count[len(d)][1] += 1
    for t in {x["defect"] for x in d}:
        by_type[t][0] += c
        by_type[t][1] += 1
        if len(d) == 1:
            by_type[t][2] += c
            by_type[t][3] += 1


def share(a, b):
    return f"{a:,} of {b:,} ({a / b:.0%})" if b else "&ndash;"


def alone(t):
    return f"{by_type[t][2] / by_type[t][3]:.0%}"


def order(item):
    t, v = item
    return (v[2] / v[3] if v[3] else 2, v[0] / v[1])


count_rows = "".join(f"<tr><td>{n}</td><td>{share(*by_count[n])}</td></tr>" for n in sorted(by_count))
type_rows = "".join(f"<tr><td>{NAMES[t]} <code>{t}</code></td><td>{share(v[2], v[3])}</td><td>{share(v[0], v[1])}</td></tr>"
                    for t, v in sorted(by_type.items(), key=order))
level_rows = "".join(f"<tr><td>{level.capitalize()}</td><td>{v['diagrams']:,}</td><td>{v['accuracy']:.2f}</td>"
                     f"<td>{share(v['bad_caught'], v['bad'])}</td><td>{share(v['good_flagged'], v['good'])}</td></tr>"
                     for level, v in S["by_difficulty"].items())
missed_single = sum(v[3] - v[2] for v in by_type.values())
singles = sum(v[3] for v in by_type.values())
hard = S["by_difficulty"]["hard"]

CSS = """
:root { color-scheme: light; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#6f6e68; --grid:#e8e7e3; --rule:#d5d4cf; --wash:#f4f3f0; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7;
  --muted:#9c9b92; --grid:#2e2e2b; --rule:#444440; --wash:#242422; } }
:root[data-theme="dark"] { color-scheme: dark; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#9c9b92; --grid:#2e2e2b;
  --rule:#444440; --wash:#242422; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink); font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 800px; margin: 0 auto; padding: 48px 24px 64px; }
h1 { font-size: 28px; line-height: 1.2; margin: 0 0 6px; }
h2 { font-size: 18px; margin: 36px 0 10px; }
p { margin: 0 0 12px; }
.meta { color: var(--ink2); font-size: 14px; margin-bottom: 28px; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 8px 0 8px; }
.tile { background: var(--wash); border-radius: 8px; padding: 14px 16px; }
.tile .label { font-size: 13px; color: var(--ink2); }
.tile .value { font-size: 28px; font-weight: 600; line-height: 1.25; font-variant-numeric: tabular-nums; }
.tile .note { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 15px; margin: 8px 0 12px; width: 100%; }
th, td { padding: 8px 12px; border-bottom: 1px solid var(--grid); text-align: right; font-variant-numeric: tabular-nums; vertical-align: top; }
th { color: var(--ink2); font-weight: 600; border-bottom-color: var(--rule); }
th:first-child, td:first-child { text-align: left; padding-left: 0; }
td code { color: var(--muted); }
ol, ul { padding-left: 22px; margin: 0 0 12px; }
li { margin-bottom: 6px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }
footer { margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--grid); color: var(--ink2); font-size: 14px; }
@media (max-width: 640px) { main { padding: 32px 16px 48px; } .tiles { grid-template-columns: repeat(2, 1fr); } h1 { font-size: 24px; } }
"""

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset Pipeline Result</title><style>{CSS}</style></head><body><main>
<h1>The final VLM pipeline on the 6,000 dataset diagrams</h1>
<p class="meta">Diagram Evaluation &middot; <code>diagrams/data.csv</code>, the {S['diagrams']:,} diagrams outside the human study &middot; branch <code>vlm-pipeline</code></p>

<div class="tiles">
<div class="tile"><div class="label">Accuracy</div><div class="value">{S['accuracy']:.2f}</div><div class="note">95% interval {S['accuracy_interval'][0]:.2f} to {S['accuracy_interval'][1]:.2f}</div></div>
<div class="tile"><div class="label">AUC</div><div class="value">{S['test_auc']:.3f}</div><div class="note">95% interval {S['test_auc_interval'][0]:.3f} to {S['test_auc_interval'][1]:.3f}</div></div>
<div class="tile"><div class="label">Bad diagrams caught</div><div class="value">{caught / S['bad']:.0%}</div><div class="note">{caught:,} of {S['bad']:,}</div></div>
<div class="tile"><div class="label">Good diagrams flagged</div><div class="value">{flagged / S['good']:.0%}</div><div class="note">{flagged:,} of {S['good']:,}</div></div>
</div>
<p>The final pipeline, unchanged: two VLM judges (gpt-6-luna) score the 13 dimensions, and the classifier in
<code>classifier.pkl</code> (gradient boosting trained on the human scores of HS001 to HS200) calls each diagram good
or bad. The call is compared with the label in <code>data.csv</code>: negative (bad, one to four errors inserted on
purpose) or positive (good, the original figure).</p>

<h2>The same result as on the study test set</h2>
<div class="scroll"><table>
<thead><tr><th>Test diagrams</th><th>AUC</th><th>Accuracy</th><th>Bad caught</th><th>Good flagged</th></tr></thead>
<tbody>
<tr><td>HS201 to HS300 (100, corrected labels)</td><td>{STUDY['test_auc']:.3f}</td><td>{STUDY['accuracy']:.2f}</td>
<td>{share(STUDY['bad_caught'], STUDY['bad'])}</td><td>{share(STUDY['good_flagged'], STUDY['good'])}</td></tr>
<tr><td>The dataset ({S['diagrams']:,}, its own labels)</td><td>{S['test_auc']:.3f}</td><td>{S['accuracy']:.2f}</td>
<td>{share(caught, S['bad'])}</td><td>{share(flagged, S['good'])}</td></tr>
</tbody></table></div>
<p>None of these {S['diagrams']:,} diagrams was used to build the pipeline, so the result on the 100 test diagrams holds
on 60 times as many.</p>

<h2>Calls against the labels</h2>
<div class="scroll"><table>
<thead><tr><th></th><th>Called bad</th><th>Called good</th></tr></thead>
<tbody>
<tr><td>Bad ({S['bad']:,})</td><td>{caught:,}</td><td>{S['bad'] - caught:,}</td></tr>
<tr><td>Good ({S['good']:,})</td><td>{flagged:,}</td><td>{S['good'] - flagged:,}</td></tr>
</tbody></table></div>

<h2>By difficulty</h2>
<div class="scroll"><table>
<thead><tr><th>Difficulty</th><th>Diagrams</th><th>Accuracy</th><th>Bad caught</th><th>Good flagged</th></tr></thead>
<tbody>{level_rows}</tbody></table></div>
<p>Hard diagrams are where the pipeline loses: {hard['good_flagged'] / hard['good']:.0%} of the hard originals are
flagged bad, against about 15 to 18% for easy and medium.</p>

<h2>Which bad diagrams it misses</h2>
<p>The more errors were inserted, the more often the diagram is caught:</p>
<div class="scroll"><table>
<thead><tr><th>Errors inserted</th><th>Caught</th></tr></thead>
<tbody>{count_rows}</tbody></table></div>
<p>By the kind of error. &ldquo;Only error&rdquo; counts the diagrams where it is the single inserted error, the
cleanest view of what the judges see; &ldquo;any&rdquo; counts every diagram that has it, often with other errors.
Sorted by the first column, hardest first.</p>
<div class="scroll"><table>
<thead><tr><th>Error</th><th>Caught, only error</th><th>Caught, any</th></tr></thead>
<tbody>{type_rows}</tbody></table></div>
<ul>
<li><strong>Arrow errors alone are mostly missed:</strong> a detached, reversed or erased arrow as the only error is
caught in {by_type['detach_arrow'][2] + by_type['reverse_arrow'][2] + by_type['erase_arrow'][2]} of
{by_type['detach_arrow'][3] + by_type['reverse_arrow'][3] + by_type['erase_arrow'][3]} diagrams. A faded label or low
resolution alone is caught in {alone('fade_label')} and {alone('low_resolution')} of the diagrams.</li>
<li><strong>Missing or cut content is caught:</strong> a blanked label, a cropped edge or a removed box alone is caught
in {alone('blank_label')}, {alone('crop_edge')} and {alone('remove_box')} of the diagrams.</li>
<li>Of the {singles:,} diagrams with a single error, {missed_single:,} are missed.</li>
</ul>

<h2>The run</h2>
<p>{RUN['calls']:,} judge calls ({S['diagrams']:,} diagrams &times; 13 dimensions), 6 at a time, about {RUN['hours']} hours,
about ${RUN['cost']}. {RUN['refused']} calls were refused by OpenAI's prompt filter (a false flag); all went through
when re-run, so every diagram has all 13 scores.</p>

<h2>Limits</h2>
<ul>
<li>The labels are the original ones. In the 300 study diagrams 7% of the originals already had a clear error; if
that holds here, part of the {flagged:,} flagged originals are really bad, and the accuracy shown here is a little
lower than it would be on corrected labels.</li>
<li>Each diagram was scored once by each judge; how much the scores move between runs was not measured here.</li>
<li>Several error kinds never appear alone, so their &ldquo;only error&rdquo; column is empty; single-error counts are
small ({min(v[3] for v in by_type.values() if v[3])} to {max(v[3] for v in by_type.values())} diagrams per kind).</li>
</ul>

<footer>Code, per-diagram calls and scores (<code>dataset_predictions.csv</code>), the summary
(<code>dataset_results.json</code>) and every raw judge output (<code>dataset_judge_outputs.jsonl.gz</code>):
<code>experiments/pipeline</code> on branch <code>vlm-pipeline</code>. To rebuild this page:
<code>python3 experiments/pipeline/report_dataset.py</code></footer>
</main></body></html>
"""
(HERE / "report_dataset.html").write_text(page)
