"""One self-contained page: the label check and the classifiers on the corrected labels (HS201-HS300).

    python3 experiments/label_alignment/report.py   # report.html; run run.py first
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "results.json"))
review = list(csv.DictReader(open(HERE / "review.csv")))
changed = [r for r in review if r["aligned_label"] != r["label"]]
OLD, NEW = R["original"]["methods"], R["visual check"]["methods"]
P = NEW["pipeline"]

WHAT = {  # short form of the reason in review.csv
    "HS029": "Branch labels a few pixels tall, unreadable",
    "HS095": "One arrow reversed; exclusion lists about 4 px tall",
    "HS142": "Arrows drawn in three unrelated styles",
    "HS185": "Three panels squeezed into 761x289 px; lists unreadable",
    "HS186": "Exclusion arrows point into the main flow",
    "HS234": "A count label is wrong (layout is fine; a judgement call)",
    "HS250": "Formula with a missing term; arrow stops short of its box",
    "HS252": "27 tables in 600x707 px; digits unreadable",
    "HS261": "A connector runs through a heading and two labels",
    "HS262": "The split bar never connects to the two result boxes",
    "HS295": "Arrow from \"excluded\" into \"included\"; blank stage bars",
}
NAMES = {"pipeline": "Trained on human scores, given judge scores (the pipeline)",
         "judge": "Trained on judge scores, given judge scores",
         "human": "Trained on human scores, given human scores"}

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
td.text, th.text { text-align: left; }
tr.main td { font-weight: 600; }
ol, ul { padding-left: 22px; margin: 0 0 12px; }
li { margin-bottom: 6px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }
footer { margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--grid); color: var(--ink2); font-size: 14px; }
@media (max-width: 640px) { main { padding: 32px 16px 48px; } .tiles { grid-template-columns: repeat(2, 1fr); } h1 { font-size: 24px; } }
"""


def pair(m):
    return f"{m['test_auc']:.3f} / {m['accuracy']:.2f}"


changes = "".join(f"<tr><td>{r['id']}</td><td class=\"text\">{'test' if int(r['id'][2:]) >= 201 else 'train/val'}</td>"
                  f"<td class=\"text\">{WHAT[r['id']]}</td></tr>" for r in changed)
main_rows = "".join(
    f"<tr{' class=\"main\"' if k == 'pipeline' else ''}><td>{NAMES[k]}</td><td>{pair(OLD[k])}</td>"
    f"<td>{NEW[k]['test_auc']:.3f} ({NEW[k]['test_auc_interval'][0]:.3f} to {NEW[k]['test_auc_interval'][1]:.3f})</td>"
    f"<td>{NEW[k]['accuracy']:.2f} ({NEW[k]['accuracy_interval'][0]:.2f} to {NEW[k]['accuracy_interval'][1]:.2f})</td>"
    f"<td>{NEW[k]['bad_caught']} of {NEW[k]['bad']} / {NEW[k]['good_flagged']} of {NEW[k]['good']}</td></tr>" for k in NAMES)
other_rows = "".join(
    f"<tr><td>{label}</td>" + "".join(f"<td>{pair(R[key]['methods'][k])}</td>" for k in NAMES) + "</tr>"
    for label, key in (("Original", "original"), ("Corrected (kept)", "visual check"),
                       ("Both classifiers disagree with the label: flip it", "both classifiers"),
                       ("Mean of the 13 human scores below 8: bad", "human mean < 8")))
n_test = sum(int(r["id"][2:]) >= 201 for r in changed)
n_reviewed = sum(r["reviewed"] == "1" for r in review)
n_neg_reviewed = sum(r["reviewed"] == "1" and r["label"] == "negative" for r in review)

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Corrected Labels Result</title><style>{CSS}</style></head><body><main>
<h1>Corrected labels: the VLM pipeline on HS201 to HS300</h1>
<p class="meta">Diagram Evaluation &middot; test set HS201 to HS300 (100 diagrams) &middot; branch <code>vlm-pipeline</code></p>

<div class="tiles">
<div class="tile"><div class="label">Accuracy</div><div class="value">{P['accuracy']:.2f}</div><div class="note">was {OLD['pipeline']['accuracy']:.2f} on the original labels</div></div>
<div class="tile"><div class="label">Test AUC</div><div class="value">{P['test_auc']:.3f}</div><div class="note">was {OLD['pipeline']['test_auc']:.3f}</div></div>
<div class="tile"><div class="label">Bad diagrams caught</div><div class="value">{P['bad_caught']} of {P['bad']}</div><div class="note">{P['bad_caught'] / P['bad']:.0%}</div></div>
<div class="tile"><div class="label">Good diagrams flagged</div><div class="value">{P['good_flagged']} of {P['good']}</div><div class="note">{P['good_flagged'] / P['good']:.0%}</div></div>
</div>
<p>The pipeline: two VLM judges (gpt-6-luna) score the 13 dimensions, and gradient boosting trained on human scores
calls each diagram good or bad.</p>

<h2>What was wrong with the labels</h2>
<p>Positive meant the original published figure, negative the figure with one to three errors inserted on purpose.
{n_reviewed} diagrams that looked suspicious (by the human scores, the judge scores, or either classifier) were looked
at again, image and caption only, without seeing any score. The same standard was used for both labels: a diagram is
bad if it has a clear error of the kinds that were inserted.</p>
<ul>
<li><strong>No negative was wrong.</strong> In all {n_neg_reviewed} suspicious negatives at least one inserted error
can be seen or contradicts the caption.</li>
<li><strong>{len(changed)} original figures already have a clear error</strong> and are relabeled bad, {n_test} of them
in the test set.</li>
</ul>
<div class="scroll"><table>
<thead><tr><th>Diagram</th><th class="text">Set</th><th class="text">What is wrong with the original figure</th></tr></thead>
<tbody>{changes}</tbody></table></div>

<h2>Results on the corrected labels</h2>
<p>Every classifier is retrained on the corrected labels of HS001 to HS200 with the same settings as before (200 trees,
learning rate 0.03), so only the labels change. The judge scores are the saved ones; no judge was re-run.</p>
<div class="scroll"><table>
<thead><tr><th>Classifier</th><th>Original labels: AUC / accuracy</th><th>Corrected: AUC (95% interval)</th>
<th>Corrected: accuracy (95% interval)</th><th>Bad caught / good flagged</th></tr></thead>
<tbody>{main_rows}</tbody></table></div>
<ol>
<li><strong>The pipeline gains most:</strong> accuracy {OLD['pipeline']['accuracy']:.2f} to {P['accuracy']:.2f}, AUC
{OLD['pipeline']['test_auc']:.3f} to {P['test_auc']:.3f}. It calls 5 of the {n_test} bad test originals bad: part of
what counted as good diagrams wrongly flagged were bad diagrams.</li>
<li><strong>The classifier trained on judge scores barely moves</strong> (accuracy {OLD['judge']['accuracy']:.2f} to
{NEW['judge']['accuracy']:.2f}); it calls only 2 of the {n_test} bad originals bad.</li>
<li><strong>Human scores stay the ceiling</strong> at {NEW['human']['accuracy']:.2f} accuracy.</li>
</ol>

<h2>Other label versions tried</h2>
<p>AUC / accuracy on HS201 to HS300, same settings.</p>
<div class="scroll"><table>
<thead><tr><th>Labels</th><th>Pipeline</th><th>Trained on judge scores</th><th>Trained on human scores</th></tr></thead>
<tbody>{other_rows}</tbody></table></div>
<p>The corrected labels are kept: fewest changes, each with a visible error, made without seeing any score. The other
two are computed from the human scores or from the classifiers' own calls, so the classifiers that read those scores
are partly graded on their own output.</p>

<h2>Limits</h2>
<ul>
<li>Only the suspicious diagrams were re-checked, by a model (Claude), not by human annotators; a human look at the
{len(changed)} changes is still to do.</li>
<li>100 test diagrams: one diagram moves accuracy by 1 point.</li>
</ul>

<footer>Code, the per-diagram review (<code>review.csv</code>) and full results: <code>experiments/label_alignment</code>
on branch <code>vlm-pipeline</code>. To reproduce: <code>python3 experiments/label_alignment/run.py</code></footer>
</main></body></html>
"""
(HERE / "report.html").write_text(page)
