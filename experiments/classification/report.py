"""report.html: the results on one self-contained page, to share by email. Run run.py first.

    python3 experiments/classification/report.py
"""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "results.json"))
M = R["methods"]
LABEL = {"gradient_boosting": "Gradient boosting", "svm_rbf": "SVM (RBF)", "random_forest": "Random forest",
         "logistic_regression": "Logistic regression", "decision_tree": "Decision tree",
         "mean_of_13": "Mean of the 13 scores", "lowest_of_13": "Lowest of the 13 scores"}
TOP = sorted(["gradient_boosting", "svm_rbf", "random_forest", "logistic_regression"], key=lambda n: -M[n]["val_auc"])
CONTEXT = ["decision_tree", "mean_of_13", "lowest_of_13"]
FITTED = ["decision_tree", "logistic_regression", "random_forest", "gradient_boosting", "svm_rbf"]
AUC_LOW, AUC_HIGH = 0.84, 1.0
AUC_TICKS = [0.85, 0.90, 0.95, 1.00]
REPO = "https://github.com/jerry66s6/Diagram-Evaulation/tree/classification/experiments/classification"

# The tested tree with leaves of the same class merged (`decision_tree` in results.json has it in full).
TREE = """style_uniformity <= 8.5
    layout <= 7.5                                         bad
    otherwise, color_consistency <= 6.5                   bad
    otherwise, details <= 7.5                             bad
    otherwise, presence <= 9.5                            bad
    otherwise                                             good
style_uniformity > 8.5
    presence <= 8.5 and details <= 8.5                    bad
    presence > 8.5, details <= 9.5, connectivity <= 8.5   bad
    otherwise                                             good"""

CSS = """
:root { color-scheme: light; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#6f6e68; --grid:#e8e7e3; --rule:#d5d4cf;
  --accent:#2a78d6; --context:#8a8983; --ref:#52514e; --wash:#f4f3f0; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7;
  --muted:#9c9b92; --grid:#2e2e2b; --rule:#444440; --accent:#3987e5; --context:#8f8e86; --ref:#c3c2b7; --wash:#242422; } }
:root[data-theme="dark"] { color-scheme: dark; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#9c9b92; --grid:#2e2e2b; --rule:#444440;
  --accent:#3987e5; --context:#8f8e86; --ref:#c3c2b7; --wash:#242422; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink); font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 48px 24px 64px; }
h1 { font-size: 30px; line-height: 1.2; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 20px; line-height: 1.3; margin: 44px 0 10px; }
h3 { font-size: 16px; margin: 28px 0 6px; }
p { margin: 0 0 12px; }
ul, ol { margin: 0 0 12px; padding-left: 22px; }
li { margin-bottom: 8px; }
a { color: var(--accent); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }
pre { background: var(--wash); padding: 14px 16px; border-radius: 6px; overflow-x: auto; line-height: 1.6; margin: 8px 0 12px; }
.meta { color: var(--ink2); font-size: 14px; margin-bottom: 24px; }
.answer { background: var(--wash); border-radius: 6px; padding: 18px 20px; font-size: 17px; }
.answer p:last-child { margin: 0; }
.caption { color: var(--ink2); font-size: 14px; margin: 0 0 12px; }
dl { display: grid; grid-template-columns: 96px 1fr; gap: 8px 16px; margin: 0; }
dt { font-weight: 600; }
dd { margin: 0; }
.table { overflow-x: auto; margin: 12px 0 16px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { padding: 8px 10px; border-bottom: 1px solid var(--grid); text-align: right; vertical-align: top; white-space: nowrap; }
th { font-weight: 600; color: var(--ink2); border-bottom-color: var(--rule); vertical-align: bottom; white-space: normal; }
th:not(:first-child) { min-width: 58px; }
th:first-child, td:first-child { text-align: left; padding-left: 0; }
td { font-variant-numeric: tabular-nums; }
td small { display: block; color: var(--muted); font-size: 12px; }
tr.break td { border-bottom-color: var(--rule); }
td.sig { font-weight: 600; background: var(--wash); }
table.matrix { font-size: 13px; }
table.matrix th, table.matrix td { padding: 7px 6px; }
table.matrix td small { font-size: 11px; }
.chart { margin: 8px 0 16px; }
.row { display: grid; grid-template-columns: 170px 1fr; align-items: center; border-radius: 4px; outline-offset: 1px; }
.row:hover { background: var(--wash); }
.name { font-size: 14px; padding-right: 12px; }
.plot { position: relative; height: 32px; margin-right: 18px; }
.tick, .ref { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--grid); }
.ref { background: var(--ref); }
.whisker { position: absolute; top: 15px; height: 2px; border-radius: 1px; background: var(--c); }
.dot { position: absolute; top: 10px; width: 12px; height: 12px; margin-left: -6px; border-radius: 50%; background: var(--c); box-shadow: 0 0 0 2px var(--surface); }
.accent { --c: var(--accent); }
.context { --c: var(--context); }
.axis { height: 24px; font-size: 12px; color: var(--muted); }
.axis span { position: absolute; top: 4px; transform: translateX(-50%); font-variant-numeric: tabular-nums; }
.bars .plot { height: 26px; }
.bar { position: absolute; top: 6px; height: 14px; border-radius: 0 4px 4px 0; background: var(--accent); }
.value { position: absolute; top: 3px; font-size: 13px; color: var(--ink2); font-variant-numeric: tabular-nums; margin-left: 6px; }
#tip { position: fixed; z-index: 5; max-width: 280px; padding: 8px 10px; border-radius: 6px; background: var(--ink); color: var(--surface); font-size: 13px; line-height: 1.45; white-space: pre-line; pointer-events: none; }
footer { margin-top: 44px; padding-top: 16px; border-top: 1px solid var(--grid); color: var(--ink2); font-size: 14px; }
@media (max-width: 640px) {
  main { padding: 32px 16px 48px; }
  h1 { font-size: 25px; }
  dl { grid-template-columns: 1fr; gap: 2px; }
  dd { margin-bottom: 10px; }
  .row { grid-template-columns: 116px 1fr; }
  .name { font-size: 13px; }
}
@media print { #tip { display: none; } .row:hover { background: none; } }
"""

JS = """
const tip = document.getElementById('tip');
function show(el, x, y) {
  tip.textContent = el.dataset.tip;
  tip.hidden = false;
  const left = Math.min(x + 14, innerWidth - tip.offsetWidth - 8);
  const top = y + tip.offsetHeight + 22 > innerHeight ? y - tip.offsetHeight - 12 : y + 14;
  tip.style.left = Math.max(8, left) + 'px';
  tip.style.top = Math.max(8, top) + 'px';
}
document.querySelectorAll('[data-tip]').forEach(el => {
  el.addEventListener('pointermove', e => show(el, e.clientX, e.clientY));
  el.addEventListener('pointerleave', () => { tip.hidden = true; });
  el.addEventListener('focus', () => { const r = el.getBoundingClientRect(); show(el, r.left + 40, r.bottom - 6); });
  el.addEventListener('blur', () => { tip.hidden = true; });
});
"""


def f3(v):
    return f"{v:.3f}"


def signed(v):
    return f"{v:+.3f}".replace("-", "−")


def weight(dimensions, name):
    return f"{dimensions[name]['logistic_coefficient']:+.2f}".replace("-", "−")


def at(v):
    return (v - AUC_LOW) / (AUC_HIGH - AUC_LOW) * 100


def auc_row(name):
    m = M[name]
    low, high = m["test_auc_interval"]
    ticks = "".join(f'<i class="tick" style="left:{at(t):.2f}%"></i>' for t in AUC_TICKS)
    tip = f"{LABEL[name]}\nTest AUC {f3(m['test_auc'])}\n95% interval {f3(low)} to {f3(high)}"
    return (f'<div class="row {"accent" if name in TOP else "context"}" tabindex="0" data-tip="{html.escape(tip)}">'
            f'<div class="name">{LABEL[name]}</div><div class="plot">{ticks}<i class="ref" style="left:{at(M["mean_of_13"]["test_auc"]):.2f}%"></i>'
            f'<span class="whisker" style="left:{at(low):.2f}%;width:{at(high) - at(low):.2f}%"></span>'
            f'<span class="dot" style="left:{at(m["test_auc"]):.2f}%"></span></div></div>')


def auc_chart():
    axis = "".join(f'<span style="left:{at(t):.2f}%">{t:.2f}</span>' for t in AUC_TICKS)
    return ('<div class="chart">' + "".join(auc_row(n) for n in TOP + CONTEXT)
            + f'<div class="row"><div class="name"></div><div class="plot axis">{axis}</div></div></div>')


def method_row(name, last_in_group=False):
    m = M[name]
    gap = "<td>—</td>" if name == "mean_of_13" else f'<td>{signed(m["gap"])}<small>{signed(m["gap_interval"][0])} to {signed(m["gap_interval"][1])}</small></td>'
    return (f'<tr{" class=break" if last_in_group else ""}><td>{LABEL[name]}</td><td>{f3(m["val_auc"])}</td><td>{f3(m["val_accuracy"])}</td>'
            f'<td>{f3(m["test_auc"])}<small>{f3(m["test_auc_interval"][0])} to {f3(m["test_auc_interval"][1])}</small></td>{gap}'
            f'<td>{m["accuracy"]:.2f}<small>{m["accuracy_interval"][0]:.2f} to {m["accuracy_interval"][1]:.2f}</small></td>'
            f'<td>{f3(m["f1"])}</td><td>{m["bad_caught"]}</td><td>{m["clean_flagged"]}</td></tr>')


def is_clear(values):
    return values[0] > 0 or values[1] < 0


def pair(a, b):
    """a minus b, from whichever order results.json stores the pair in."""
    for p in R["pairwise"]:
        if (p["a"], p["b"]) == (a, b):
            return p["auc_difference"], p["auc_interval"], p["accuracy_difference"], p["accuracy_interval"]
        if (p["a"], p["b"]) == (b, a):
            flip = lambda v: sorted(-x for x in v)
            return -p["auc_difference"], flip(p["auc_interval"]), -p["accuracy_difference"], flip(p["accuracy_interval"])


def pairwise_table():
    """Upper triangle: test AUC of the row method minus the column method, with the paired 95% interval."""
    order = TOP + CONTEXT
    short = {"gradient_boosting": "Gradient boosting", "svm_rbf": "SVM", "random_forest": "Random forest", "logistic_regression": "Logistic regression",
             "decision_tree": "Decision tree", "mean_of_13": "Mean of 13", "lowest_of_13": "Lowest of 13"}
    def cell(a, b):
        diff, (low, high), _, _ = pair(a, b)
        return f"<td{' class=sig' if is_clear((low, high)) else ''}>{signed(diff)}<small>{signed(low)} to {signed(high)}</small></td>"
    head = "<tr><th>Row minus column</th>" + "".join(f"<th>{short[n]}</th>" for n in order[1:]) + "</tr>"
    body = "".join(f"<tr><td>{short[a]}</td>" + "<td></td>" * i + "".join(cell(a, b) for b in order[i + 1:]) + "</tr>"
                   for i, a in enumerate(order[:-1]))
    return f"<thead>{head}</thead><tbody>{body}</tbody>"


def dimension_chart():
    drop = {k: sum(d["test_auc_drop"][n] for n in FITTED) / len(FITTED) for k, d in R["dimensions"].items()}
    top = max(drop.values())
    rows = []
    for k in sorted(drop, key=lambda k: -drop[k]):
        width = max(drop[k], 0) / top * 86
        tip = k.replace("_", " ").capitalize() + "\n" + "\n".join(f"{LABEL[n]}: {signed(R['dimensions'][k]['test_auc_drop'][n])}" for n in FITTED)
        shown = f3(abs(drop[k])) if abs(drop[k]) < 0.0005 else f3(drop[k]).replace("-", "−")
        rows.append(f'<div class="row" tabindex="0" data-tip="{html.escape(tip)}"><div class="name">{k.replace("_", " ").capitalize()}</div><div class="plot">'
                    f'<span class="bar" style="width:{width:.2f}%"></span><span class="value" style="left:{width:.2f}%">{shown}</span></div></div>')
    return '<div class="chart bars">' + "".join(rows) + "</div>"


def input_rows():
    names = ["logistic_regression", "random_forest", "svm_rbf"]
    rows = ""
    for part, row in R["by_input"].items():
        rows += f"<tr><td>{part[0].upper() + part[1:]}</td>" + "".join(f'<td>{f3(row[n]["val_auc"])}</td><td>{f3(row[n]["test_auc"])}</td>' for n in names) + "</tr>"
    head = "".join(f'<th colspan="2">{LABEL[n]}</th>' for n in names)
    sub = "<th>Val</th><th>Test</th>" * len(names)
    return f"<thead><tr><th></th>{head}</tr><tr><th>Input</th>{sub}</tr></thead><tbody>{rows}</tbody>"


def difficulty_rows():
    return "".join(f'<tr><td>{k.capitalize()}</td><td>{v["bad_caught"]} of {v["bad"]}</td><td>{v["clean_flagged"]} of {v["clean"]}</td></tr>' for k, v in R["test_by_difficulty"].items())


def page():
    top_auc = [M[n]["test_auc"] for n in TOP]
    top_acc = [M[n]["accuracy"] for n in TOP]
    top_val = [M[n]["val_auc"] for n in TOP]
    gaps = [M[n]["gap"] for n in TOP]
    mean = M["mean_of_13"]
    tree, lr, gb, svm = M["decision_tree"], M["logistic_regression"], M["gradient_boosting"], M["svm_rbf"]
    d = R["dimensions"]
    s = R["style_uniformity_means"]
    no_style = R["by_input"]["without style_uniformity"]["svm_rbf"]["val_auc"]
    chosen = R["best_val_auc"]
    beats = ["the " + LABEL[n].lower() for n in TOP + CONTEXT if n != chosen and is_clear(pair(chosen, n)[1]) and pair(chosen, n)[0] > 0]
    beats = beats[0] if len(beats) == 1 else ", ".join(beats[:-1]) + ", and " + beats[-1]
    top_clear = [(a, b) for i, a in enumerate(TOP) for b in TOP[i + 1:] if is_clear(pair(a, b)[1])]
    accuracy_clear = []
    for i, a in enumerate(TOP + CONTEXT):
        for b in (TOP + CONTEXT)[i + 1:]:
            _, _, diff, ci = pair(a, b)
            if is_clear(ci):
                hi, lo = (a, b) if diff > 0 else (b, a)
                accuracy_clear.append(f"{LABEL[hi].lower()} is {abs(diff):.2f} above the {LABEL[lo].lower()} ({min(abs(x) for x in ci):.2f} to {max(abs(x) for x in ci):.2f})")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diagram Classification Results</title><style>{CSS}</style></head><body><main>

<h1>Classifying good and bad diagrams from human scores</h1>
<p class="meta">Diagram Evaluation &middot; November 13, 2026 &middot; branch <code>classification</code></p>

<div class="answer"><p><strong>The 13 human scores separate good diagrams from bad ones well.</strong> On 100 held-out test diagrams, logistic regression,
random forest, gradient boosting, and an SVM reach AUC {f3(min(top_auc))} to {f3(max(top_auc))} and accuracy {min(top_acc):.2f} to {max(top_acc):.2f}.
All four beat simply averaging the scores (AUC {f3(mean["test_auc"])}, accuracy {mean["accuracy"]:.2f}), and they cannot be told apart from each other.
<strong>{LABEL[chosen]} is the method carried forward:</strong> it has the best val AUC, which is how the method was to be picked, and its test AUC
({f3(M[chosen]["test_auc"])}) is clearly above {beats}.</p></div>

<h2>Setup</h2>
<dl>
<dt>Input</dt><dd>The 13 updated human scores for each diagram, 1 to 10: 5 Correctness and 8 Beauty. No images and no judge scores.</dd>
<dt>Label</dt><dd>150 positive (good) and 150 negative (deliberately made bad with inserted errors, from 23 error types). The label does not come from the human scores.</dd>
<dt>Test</dt><dd>HS201 to HS300, scored once at the end: 53 bad and 47 good.</dd>
<dt>Train/val</dt><dd>5-fold on HS001 to HS200 (160/40 in each fold), balanced on label and difficulty. Each method's settings and its good/bad cut are chosen on val, then it is refit on all 200.</dd>
<dt>Methods</dt><dd>Decision tree, logistic regression, random forest, gradient boosting, and an RBF SVM. Two rules with no training are the comparison: the mean and the lowest of the 13 scores.</dd>
</dl>

<h2>Results</h2>
<p class="caption">Test AUC with 95% bootstrap intervals. Blue marks the four methods above the mean of the 13 scores, which is the vertical line. The axis starts at {AUC_LOW:.2f}.</p>
{auc_chart()}
<div class="table"><table><thead><tr><th>Method</th><th>Val AUC</th><th>Val accuracy</th><th>Test AUC</th><th>Test AUC minus mean</th><th>Test accuracy</th><th>Test F1</th><th>Bad caught of 53</th><th>Clean flagged of 47</th></tr></thead>
<tbody>{"".join(method_row(n, n == TOP[-1] or n == "decision_tree") for n in TOP + CONTEXT)}</tbody></table></div>

<ol>
<li><strong>Combining the scores beats averaging them.</strong> The four methods are {f3(min(gaps))} to {f3(max(gaps))} above the mean of the 13 on test AUC, and each difference's interval excludes zero.</li>
<li><strong>Those four cannot be separated.</strong> Gradient boosting has the best val AUC ({f3(gb["val_auc"])}, SVM {f3(svm["val_auc"])}), and logistic regression the best test AUC ({f3(lr["test_auc"])}, gradient boosting {f3(gb["test_auc"])}). {"No two of them differ clearly in the direct comparisons below." if not top_clear else "Some of them differ in the direct comparisons below."}</li>
<li><strong>A single decision tree is no better than the plain mean</strong> (test AUC {f3(tree["test_auc"])} vs {f3(mean["test_auc"])}), though its rules are the easiest to read.</li>
<li><strong>The test scores may run a little high.</strong> Most methods score above their val AUC on test, including the plain mean ({f3(mean["val_auc"])} to {f3(mean["test_auc"])}), although the test set has fewer easy diagrams. Val AUC of {min(top_val):.2f} to {max(top_val):.2f} is the more cautious figure.</li>
</ol>

<h3>Comparing methods directly</h3>
<p class="caption">Test AUC of the row method minus the column method, with a 95% interval from the same bootstrap draws for both. Shaded: the interval
excludes zero. These are {len(R["pairwise"])} comparisons with no correction for testing that many, so a borderline one can be chance.</p>
<div class="table"><table class="matrix">{pairwise_table()}</table></div>
<p>On test accuracy, {"no pair differs clearly" if not accuracy_clear else ("only one pair differs clearly: " if len(accuracy_clear) == 1 else f"only {len(accuracy_clear)} pairs differ clearly: ") + "; ".join(accuracy_clear)}. With 100 test diagrams, accuracy gaps of a few points are within noise.</p>

<h2>Which inputs matter</h2>
<p class="caption">Drop in test AUC when one score is shuffled, averaged over the five trained methods. Hover over a row for each method.</p>
{dimension_chart()}
<p>Style uniformity, presence, and details matter most. Logistic regression agrees: its largest standardized weights are style uniformity ({weight(d, "style_uniformity")}), presence ({weight(d, "presence")}), and details ({weight(d, "details")}). Humans tended to lower several dimensions together, so these importances share credit among correlated scores.</p>
<p>Style uniformity mostly drops where style really changed. Its human mean is {s["positive"]["mean"]:.2f} on positives, {s["negative, Beauty error"]["mean"]:.2f} on negatives with a Beauty error, {s["negative, no Beauty error, label size changed"]["mean"]:.2f} on negatives with an enlarged or shrunk label, and {s["negative, no Beauty error, label size unchanged"]["mean"]:.2f} on the other negatives. The results do not rest on it alone: SVM val AUC moves from {f3(svm["val_auc"])} to {f3(no_style)} without it.</p>

<h3>Using part of the scores</h3>
<div class="table"><table>{input_rows()}</table></div>
<p>All 13 scores have the best val AUC for all three methods.</p>

<h2>Decision tree rules</h2>
<p class="caption">The tested tree, fitted on HS001 to HS200, with leaves of the same class merged.</p>
<pre>{html.escape(TREE)}</pre>

<h2>By difficulty</h2>
<p class="caption">{LABEL[chosen]}, the method carried forward, on the test set.</p>
<div class="table"><table style="max-width:420px"><thead><tr><th>Difficulty</th><th>Bad caught</th><th>Clean flagged</th></tr></thead><tbody>{difficulty_rows()}</tbody></table></div>

<h2>Limits</h2>
<ul>
<li><strong>Human scores must exist.</strong> A new diagram has none, so these models cannot score it as they are. Repeating this on judge scores is a separate experiment: it needs the judge's Beauty scores for HS146 to HS300 first, and HS201 to HS300 suits it as a test set because the correctness prompt was not tuned on them.</li>
<li><strong>Only these error types were tested.</strong> Diagrams that are bad in other ways, such as new problems in LLM-generated figures, have not been tried.</li>
<li><strong>The test set has 100 diagrams.</strong> One diagram moves accuracy by 1 point, so small gaps between methods mean little.</li>
<li><strong>The test set is not entirely fresh.</strong> These diagrams were also in earlier exploratory runs of the same methods. The methods and setting grids did not change after those runs.</li>
</ul>
<p>As a check on the pipeline, shuffling the labels drops test AUC to about 0.5.</p>

<footer>Code and full results: <a href="{REPO}">experiments/classification</a>. To reproduce: <code>python3 experiments/classification/run.py</code></footer>
</main><div id="tip" hidden></div><script>{JS}</script></body></html>
"""


(HERE / "report.html").write_text(page())
