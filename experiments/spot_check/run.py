"""Low-confidence calls of the final pipeline on the 6,000 dataset diagrams: pick them, group them, and draw a sample
for a manual spot check. No API calls; reads ../pipeline/dataset_predictions.csv and the raw judge outputs.

    python3 experiments/spot_check/run.py   # spot_check.csv, summary.json, spot_check.html

Low confidence: P(bad) within 0.1 of the cut (0.361), so 0.261 <= P(bad) < 0.461.
Groups: the call against the dataset label (four groups), difficulty, and for the negatives the kind of inserted error.
Sample: 25 random diagrams from each of the four groups (seed 0), shown in random order on the page.
"""
import csv
import gzip
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIPELINE = ROOT / "experiments" / "pipeline"
CUT = json.load(open(PIPELINE / "dataset_results.json"))["cut"]
BAND = 0.1
PER_GROUP = 25
LOW = 5  # a judge score of 5 or less is shown on the card

GROUPS = {  # (called bad, label) -> name
    (True, "positive"): "Original, called bad",
    (False, "negative"): "Errors inserted, called good",
    (True, "negative"): "Errors inserted, called bad",
    (False, "positive"): "Original, called good",
}
ERROR_GROUPS = {"layout": "Layout", "connectivity": "Arrows", "presence": "Missing or extra box", "details": "Labels",
                "legibility": "Legibility", "beauty": "Style"}

data = {r["id"]: r for r in csv.DictReader(open(ROOT / "diagrams" / "data.csv"))}
pred = list(csv.DictReader(open(PIPELINE / "dataset_predictions.csv")))
dims = [k[len("judge_"):] for k in pred[0] if k.startswith("judge_")]
for r in pred:
    r["p"] = float(r["p_bad"])
    r["bad_call"] = r["called_bad"] == "1"
    r["group"] = GROUPS[(r["bad_call"], r["label"])]
    r["errors"] = json.loads(data[r["id"]]["defects"]) if r["label"] == "negative" else []
low = [r for r in pred if abs(r["p"] - CUT) < BAND]
ids = {r["id"] for r in low}

notes = defaultdict(dict)  # id -> dimension -> what the judge wrote
with gzip.open(PIPELINE / "dataset_judge_outputs.jsonl.gz", "rt") as f:
    for line in f:
        o = json.loads(line)
        if o["id"] in ids:
            notes[o["id"]][o["dimension"]] = " ".join(o.get("problems") or []) or o.get("reasoning", "")


def judge_low(r):
    """Judge dimensions scored 5 or less (1-10 scale), lowest first."""
    scores = sorted((float(r[f"judge_{k}"]), k) for k in dims)
    return [(k, s) for s, k in scores if s <= LOW]


def agree(rows):
    return sum(r["bad_call"] == (r["label"] == "negative") for r in rows) / len(rows)


rng = random.Random(0)
sample = []
for name in GROUPS.values():
    sample += rng.sample([r for r in low if r["group"] == name], PER_GROUP)
rng.shuffle(sample)
order = {r["id"]: n for n, r in enumerate(sample, 1)}

with open(HERE / "spot_check.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "image", "difficulty", "label", "called_bad", "p_bad", "group", "inserted_errors", "judge_low",
                "spot_check", "verdict", "note"])
    for r in sorted(low, key=lambda r: r["p"]):
        w.writerow([r["id"], data[r["id"]]["image"], r["difficulty"], r["label"], int(r["bad_call"]), r["p_bad"], r["group"],
                    "; ".join(e["defect"] for e in r["errors"]), "; ".join(f"{k} {s:g}" for k, s in judge_low(r)),
                    order.get(r["id"], ""), "", ""])

outside = [r for r in pred if r["id"] not in ids]
summary = {
    "cut": CUT, "band": [CUT - BAND, CUT + BAND], "low_confidence": len(low), "all": len(pred),
    "agree_with_label": {"low_confidence": agree(low), "the rest": agree(outside)},
    "groups": {name: sum(r["group"] == name for r in low) for name in GROUPS.values()},
    "difficulty": {d: {"low_confidence": sum(r["difficulty"] == d for r in low), "all": sum(r["difficulty"] == d for r in pred)}
                   for d in ("easy", "medium", "hard")},
    "inserted_errors_by_group": {name: dict(Counter(ERROR_GROUPS[e["criterion"].split(".")[0]] for r in low if r["group"] == name
                                                    for e in r["errors"]).most_common())
                                 for name in GROUPS.values() if "inserted" in name},
    "judge_low_by_group": {name: dict(Counter(k for r in low if r["group"] == name for k, _ in judge_low(r)).most_common())
                           for name in GROUPS.values()},
    "no_judge_low_by_group": {name: sum(not judge_low(r) for r in low if r["group"] == name) for name in GROUPS.values()},
    "sample": {"per_group": PER_GROUP, "seed": 0, "ids": [r["id"] for r in sample]},
}
json.dump(summary, open(HERE / "summary.json", "w"), indent=1)

# ---- page ----------------------------------------------------------------------------------------------------------
e = html.escape
group_rows = "".join(f"<tr><td>{name}</td><td>{n}</td><td>{n / len(low):.0%}</td></tr>" for name, n in summary["groups"].items())
level_rows = "".join(f"<tr><td>{d.capitalize()}</td><td>{v['low_confidence']}</td><td>{v['all']:,}</td>"
                     f"<td>{v['low_confidence'] / v['all']:.0%}</td></tr>" for d, v in summary["difficulty"].items())
error_cols = list(ERROR_GROUPS.values())
error_rows = "".join(f"<tr><td>{name}</td>" + "".join(f"<td>{c.get(g, 0)}</td>" for g in error_cols) + "</tr>"
                     for name, c in summary["inserted_errors_by_group"].items())
judge_cols = [k for k, _ in Counter(k for r in low for k, _ in judge_low(r)).most_common()]
judge_rows = "".join(f"<tr><td>{name}</td>" + "".join(f"<td>{c.get(k, 0)}</td>" for k in judge_cols)
                     + f"<td>{summary['no_judge_low_by_group'][name]}</td></tr>" for name, c in summary["judge_low_by_group"].items())


def card(n, r):
    d = data[r["id"]]
    errs = "".join(f"<li>{e(x['defect'])} <span class=\"muted\">({e(x['criterion'])})</span></li>" for x in r["errors"])
    lows = "".join(f"<li><strong>{k.replace('_', ' ')} {s:g}</strong>: {e(notes[r['id']].get(k, ''))}</li>" for k, s in judge_low(r))
    return f"""<article id="{r['id']}"><h3>{n}. {r['id']}</h3>
<img loading="lazy" src="../../diagrams/{e(d['image'])}" alt="Diagram {r['id']}">
<p class="caption">{e(d['caption'])}</p>
<details><summary>Label, call and what the judges said</summary>
<p><strong>{r['group']}</strong> &middot; P(bad) {r['p']:.3f} &middot; {r['difficulty']}</p>
{f'<p>Inserted errors:</p><ul>{errs}</ul>' if errs else '<p>The original figure, no inserted error.</p>'}
{f'<p>Judge scores of {LOW} or less:</p><ul>{lows}</ul>' if lows else f'<p>No judge score of {LOW} or less.</p>'}
</details></article>"""


CSS = """
:root { color-scheme: light; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#6f6e68; --grid:#e8e7e3; --rule:#d5d4cf; --wash:#f4f3f0; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7;
  --muted:#9c9b92; --grid:#2e2e2b; --rule:#444440; --wash:#242422; } }
:root[data-theme="dark"] { color-scheme: dark; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#9c9b92; --grid:#2e2e2b;
  --rule:#444440; --wash:#242422; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink); font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 900px; margin: 0 auto; padding: 48px 24px 64px; }
h1 { font-size: 28px; line-height: 1.2; margin: 0 0 6px; }
h2 { font-size: 18px; margin: 36px 0 10px; }
h3 { font-size: 16px; margin: 0 0 8px; }
p { margin: 0 0 12px; }
.meta { color: var(--ink2); font-size: 14px; margin-bottom: 28px; }
.muted { color: var(--muted); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 15px; margin: 8px 0 12px; width: 100%; }
th, td { padding: 8px 10px; border-bottom: 1px solid var(--grid); text-align: right; font-variant-numeric: tabular-nums; vertical-align: top; }
th { color: var(--ink2); font-weight: 600; border-bottom-color: var(--rule); }
th:first-child, td:first-child { text-align: left; padding-left: 0; }
ul { padding-left: 22px; margin: 0 0 12px; }
li { margin-bottom: 4px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }
article { border-top: 1px solid var(--rule); padding: 20px 0; }
article img { display: block; max-width: 100%; max-height: 720px; margin: 0 0 10px; background: #fff; border: 1px solid var(--grid); }
.caption { font-size: 14px; color: var(--ink2); }
details { background: var(--wash); border-radius: 8px; padding: 10px 14px; font-size: 14px; }
summary { cursor: pointer; color: var(--ink2); }
details[open] summary { margin-bottom: 8px; }
footer { margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--grid); color: var(--ink2); font-size: 14px; }
@media (max-width: 640px) { main { padding: 32px 16px 48px; } h1 { font-size: 24px; } }
"""

a = summary["agree_with_label"]
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Low-Confidence Spot Check</title><style>{CSS}</style></head><body><main>
<h1>Low-confidence calls on the 6,000 dataset diagrams</h1>
<p class="meta">Final pipeline (<code>../pipeline/classifier.pkl</code>) &middot; {len(low)} low-confidence diagrams, {len(sample)} drawn for the spot check &middot; branch <code>vlm-pipeline</code></p>

<p>The classifier gives every diagram a P(bad) and calls it bad above {CUT:.3f}. A call is <strong>low confidence</strong>
when P(bad) is within {BAND} of that cut ({CUT - BAND:.3f} to {CUT + BAND:.3f}): {len(low)} of the {len(pred):,} diagrams
({len(low) / len(pred):.0%}). There the call matches the dataset label only {a['low_confidence']:.0%} of the time, against
{a['the rest']:.0%} for the other diagrams.</p>

<h2>Call against the label</h2>
<div class="scroll"><table>
<thead><tr><th>Group</th><th>Diagrams</th><th>Share</th></tr></thead>
<tbody>{group_rows}</tbody></table></div>

<h2>Difficulty</h2>
<div class="scroll"><table>
<thead><tr><th>Difficulty</th><th>Low confidence</th><th>All</th><th>Share</th></tr></thead>
<tbody>{level_rows}</tbody></table></div>

<h2>Kind of inserted error (negatives)</h2>
<p>Errors counted once each; a diagram can have several.</p>
<div class="scroll"><table>
<thead><tr><th>Group</th>{''.join(f'<th>{g}</th>' for g in error_cols)}</tr></thead>
<tbody>{error_rows}</tbody></table></div>

<h2>Judge scores of {LOW} or less</h2>
<p>Diagrams with each dimension scored {LOW} or less on the 1 to 10 scale; a diagram can count in several columns.</p>
<div class="scroll"><table>
<thead><tr><th>Group</th>{''.join(f'<th>{k.replace("_", " ")}</th>' for k in judge_cols)}<th>none</th></tr></thead>
<tbody>{judge_rows}</tbody></table></div>

<h2>Spot check: {len(sample)} diagrams</h2>
<p>{PER_GROUP} drawn at random from each of the four groups, in random order. Look at the image and the caption first
and decide good or bad; then open the box to see the label, the call and what the judges said. Verdicts go in the
<code>verdict</code> column of <code>spot_check.csv</code> (rows with a <code>spot_check</code> number).</p>
{''.join(card(n, r) for n, r in enumerate(sample, 1))}

<footer>Code and data: <code>experiments/spot_check</code> on branch <code>vlm-pipeline</code>. The images load from
<code>diagrams/images</code>, which is not in git; open this page inside the project folder. To rebuild:
<code>python3 experiments/spot_check/run.py</code></footer>
</main></body></html>
"""
(HERE / "spot_check.html").write_text(page)
print(json.dumps({k: summary[k] for k in ("low_confidence", "agree_with_label", "groups", "difficulty", "inserted_errors_by_group",
                                          "judge_low_by_group", "no_judge_low_by_group")}, indent=1))
