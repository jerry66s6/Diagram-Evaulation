"""One short, self-contained page with the pipeline's result on HS201-HS300.

    python3 experiments/pipeline/report.py                       # report.html: classifier trained on human scores
    python3 experiments/pipeline/report.py --trained-on-judges   # report_trained_on_judges.html: trained on judge scores

Run `classify --beauty-tuned` or `train-on-judges --beauty-tuned` first.
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JUDGES = "--trained-on-judges" in sys.argv
if JUDGES:
    R = json.load(open(HERE / "results_trained_on_judges_beauty_tuned.json"))
    M = R["methods"]["gradient boosting trained on judge scores"]
    rows = list(csv.DictReader(open(HERE / "predictions_trained_on_judges_beauty_tuned.csv")))
    OUT = HERE / "report_trained_on_judges.html"
    other = json.load(open(HERE / "results_beauty_tuned.json"))["methods"]["pipeline: gradient boosting on judge scores"]
    STEP2 = (f"Gradient boosting, trained on the judges' scores of HS001 to HS200, turns the 13 scores into a good or bad call. "
             f"Its settings ({M['settings']['n_estimators']} trees, depth {M['settings']['max_depth']}, learning rate "
             f"{M['settings']['learning_rate']}) and the good/bad cut were chosen by 5-fold train/val on those 200 diagrams, "
             f"then it was refit on all 200.")
    NOTE = (f"<p>Trained on the human scores of HS001 to HS200 instead, the same pipeline gets AUC {other['test_auc']:.3f} "
            f"and accuracy {other['accuracy']:.2f}.</p>")
else:
    R = json.load(open(HERE / "results_beauty_tuned.json"))
    M = R["methods"]["pipeline: gradient boosting on judge scores"]
    rows = list(csv.DictReader(open(HERE / "predictions_beauty_tuned.csv")))
    OUT = HERE / "report.html"
    STEP2 = "Gradient boosting, trained on the human scores of HS001 to HS200, turns the 13 scores into a good or bad call."
    NOTE = ""
bad = [r for r in rows if r["label"] == "negative"]
good = [r for r in rows if r["label"] == "positive"]
caught, missed = sum(r["called_bad"] == "1" for r in bad), sum(r["called_bad"] == "0" for r in bad)
flagged, kept = sum(r["called_bad"] == "1" for r in good), sum(r["called_bad"] == "0" for r in good)
assert caught == M["bad_caught"] and flagged == M["clean_flagged"]

CSS = """
:root { color-scheme: light; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#6f6e68; --grid:#e8e7e3; --rule:#d5d4cf; --wash:#f4f3f0; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7;
  --muted:#9c9b92; --grid:#2e2e2b; --rule:#444440; --wash:#242422; } }
:root[data-theme="dark"] { color-scheme: dark; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#9c9b92; --grid:#2e2e2b;
  --rule:#444440; --wash:#242422; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink); font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 48px 24px 64px; }
h1 { font-size: 28px; line-height: 1.2; margin: 0 0 6px; }
h2 { font-size: 18px; margin: 36px 0 10px; }
p { margin: 0 0 12px; }
.meta { color: var(--ink2); font-size: 14px; margin-bottom: 28px; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 8px 0 8px; }
.tile { background: var(--wash); border-radius: 8px; padding: 14px 16px; }
.tile .label { font-size: 13px; color: var(--ink2); }
.tile .value { font-size: 28px; font-weight: 600; line-height: 1.25; font-variant-numeric: tabular-nums; }
.tile .note { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
table { border-collapse: collapse; font-size: 15px; margin: 8px 0 12px; }
th, td { padding: 9px 16px; border-bottom: 1px solid var(--grid); text-align: right; font-variant-numeric: tabular-nums; }
th { color: var(--ink2); font-weight: 600; border-bottom-color: var(--rule); }
th:first-child, td:first-child { text-align: left; padding-left: 0; }
ol { padding-left: 22px; margin: 0 0 12px; }
li { margin-bottom: 6px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }
footer { margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--grid); color: var(--ink2); font-size: 14px; }
@media (max-width: 640px) { main { padding: 32px 16px 48px; } .tiles { grid-template-columns: repeat(2, 1fr); } h1 { font-size: 24px; } }
"""

low, high = M["test_auc_interval"]
alow, ahigh = M["accuracy_interval"]
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VLM Pipeline Result</title><style>{CSS}</style></head><body><main>
<h1>VLM pipeline: good vs bad diagrams</h1>
<p class="meta">Diagram Evaluation &middot; test set HS201 to HS300 (100 diagrams) &middot; branch <code>vlm-pipeline</code></p>

<div class="tiles">
<div class="tile"><div class="label">Test AUC</div><div class="value">{M["test_auc"]:.3f}</div><div class="note">95% interval {low:.3f} to {high:.3f}</div></div>
<div class="tile"><div class="label">Accuracy</div><div class="value">{M["accuracy"]:.2f}</div><div class="note">95% interval {alow:.2f} to {ahigh:.2f}</div></div>
<div class="tile"><div class="label">Bad diagrams caught</div><div class="value">{caught} of {len(bad)}</div><div class="note">{caught / len(bad):.0%}</div></div>
<div class="tile"><div class="label">Good diagrams flagged</div><div class="value">{flagged} of {len(good)}</div><div class="note">{flagged / len(good):.0%}</div></div>
</div>

<h2>Calls against the true labels</h2>
<table>
<thead><tr><th></th><th>Called bad</th><th>Called good</th></tr></thead>
<tbody>
<tr><td>Bad ({len(bad)})</td><td>{caught}</td><td>{missed}</td></tr>
<tr><td>Good ({len(good)})</td><td>{flagged}</td><td>{kept}</td></tr>
</tbody></table>

<h2>Pipeline</h2>
<ol>
<li>Two VLM judges (gpt-6-luna) score each diagram on 13 dimensions: the correctness judge on Layout, Connectivity,
Presence, Details, and Legibility, and the Beauty judge on its eight dimensions, with the four tuned prompts for
Centering, Space Use, Alignment, and Canvas Fill.</li>
<li>{STEP2}</li>
<li>The call is compared with each diagram's true label: positive (good) or negative (deliberately made bad with
inserted errors). The human scores of the test diagrams are not used.</li>
</ol>
{NOTE}
<footer>Code and full results: <code>experiments/pipeline</code> on branch <code>vlm-pipeline</code>.
To reproduce: <code>python3 experiments/pipeline/run.py {'train-on-judges' if JUDGES else 'classify'} --beauty-tuned</code></footer>
</main></body></html>
"""
OUT.write_text(page)
