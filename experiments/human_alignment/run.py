"""Human alignment: score the 300 human-study diagrams with the evaluator as it is in git
(correctness pipeline + Beauty v6, gpt-6-luna) and compare with the human ratings.

    python3 run.py score      # resumable; raw outputs in results/<id>/
    python3 run.py compare --first 100   # HS001-HS100 (prompt-tuning set); summary.json, judge_scores.json
    python3 run.py score --first 100 --wording v2     # correctness only, with the v2 prompt wording
    python3 run.py compare --first 100 --wording v2   # summary_v2.json
    python3 run.py tune --wordings v1,v1r,v2          # side by side, from the summaries
    python3 run.py report                             # report.html: every version on HS001-HS100
    python3 run.py compare --start 101 --first 200 --wording d12   # HS101-HS200; summary_d12_101-200.json

Nothing outside this folder is changed. Images come from diagrams/human_study/images (git-ignored).
"""
import argparse
import json
import math
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scipy.stats import kendalltau, pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = HERE / "results"
CORRECTNESS = ["layout", "connectivity", "presence", "details", "legibility"]
BEAUTY = ["centering", "space_use", "color_consistency", "contrast", "alignment", "style_uniformity", "resolution", "canvas_fill"]

# Correctness prompt wording tried on HS001-HS100. The script adds these sentences to the prompt when it runs, so
# src/ and config/ stay as they are until a version is chosen. v1 is the prompt in git; v1r repeats it to measure
# run-to-run noise. Each change is one general sentence, not tied to particular diagrams or error types.
WORDINGS = {
    "v1": {},
    "v1r": {},
    "v2": {
        "prompt": [  # (text in the judge prompt, sentence added right after it)
            ("in the checklist (extra detail between the two is fine).",  # Connectivity
             " One line or arrow that serves several components (a branching trunk line, a line that ends at a row "
             "of boxes, a large background arrow with the steps placed along it) connects each of them."),
            ("Set label_legible to false when the element is drawn but its text is too small, blurry, or clipped "
             "to read.",  # Legibility
             " Judge at the size the image is shown, without zooming in: text you would need to zoom in on or guess "
             "at is not legible."),
            ("but its label is missing, a placeholder, or names something else.",  # Details
             " A label that is cut off counts as wrong. Judge a label by the element's place in the flow as well as "
             "by its text."),
        ],
        "criteria": {"layout.order": " Steps appear along that direction in the order the description gives."},
        "rules": {"description_order": " Also count one issue per such connection whose two components appear "
                                       "along the reading direction in the opposite order to the description."},
    },
}
# v3 = v2 + one stage-1 sentence (the caption sketch). In 21 of the first 100 sketches an exclusion box was a step
# in the chain (exclusion -> next cohort), which real flow charts never draw, so clean figures lost Connectivity.
WORDINGS["v3"] = {**WORDINGS["v2"], "expectation": [
    ("and the directed connections of the intended flow.",
     " A side branch (such as excluded or removed items) leaves the main flow and ends there: connect the step "
     "before it to the side branch, and that step directly to the next main step."),
]}


# v4 = a stage-1 sketch written for captions that summarize an existing figure. On the 53 unmodified figures of
# HS001-HS100 the v1 sketch had a required connection the figure does not draw in 20 (38%), against 5 (9%) for
# components, and adding the v3 sketch to direct scoring (d3) doubled Connectivity false alarms on them (5 -> 10).
# It replaces v3's side-branch sentence, which made an exclusion drawn in the main chain a missing connection.
WORDINGS["v4"] = {**WORDINGS["v2"], "expectation": [
    ("and the directed connections of the intended flow.",
     " The description is a short caption rewritten from an existing figure, not a drawing specification: list what "
     "the caption says, not how you would draw it. Keep one component for each step, group, or outcome the caption "
     "names, and do not split one step into several (a step that acquires and merges a dataset is one component). "
     "Excluded or removed items are a component of their own, but the figure may draw them as a side branch or "
     "inside the main chain, so make connections into or past them typical, not required. Mark a connection "
     "required only when the caption states that one step leads to another or lists the steps in that order; "
     "connections you add to complete the picture are typical."),
]}
SKETCH_CHECK = (
    "The attached image is a published flowchart. Below are components and connections that someone expected it to "
    "contain, written from its caption alone. For each numbered item, say whether the figure shows it. A component "
    "(C) is shown when some box or node in the figure plays that role; the wording may differ, and one box may cover "
    "several items. A connection (E) is shown when the figure's arrows or lines lead from the first to the second, "
    "directly or through other drawn steps. Answer for every item.\n\n"
)
SKETCH_CHECK_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "string"}, "shown": {"type": "boolean"}},
        "required": ["id", "shown"], "additionalProperties": False}}},
    "required": ["items"],
    "additionalProperties": False,
}


def sketches(wording: str, workers: int, first: int | None = None, start: int = 1) -> None:
    """Make the stage-1 sketch of a wording for each diagram, then check its required items against each figure. On an
    unmodified figure (label positive) a required item the figure does not show is a sketch error; on a broken one it
    may be the inserted error (a removed box, an erased arrow)."""
    _load_env()
    from diagram_correctness.vlm import OpenAIBackend
    model = os.environ.get("OPENAI_MODEL") or "gpt-6-luna"
    if model != "gpt-6-luna":
        raise SystemExit(f"model is {model!r}; this experiment runs on gpt-6-luna only")
    pipeline, backends = _pipeline(model, wording)
    check_backend = OpenAIBackend(model)
    backends.append(check_backend)
    todo = [d for d in _humans() if _in_range(d, first, start)]

    def sketch(d):
        path = RESULTS / d["id"] / _inventory_file(wording)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(pipeline.config.expectation_extractor.extract(d["caption"], pipeline.config.criteria), indent=1))
        return json.loads(path.read_text())

    def check(d):
        path = RESULTS / d["id"] / f"sketchcheck_{wording}.json"
        if path.exists():
            return json.loads(path.read_text())
        inv = sketch(d)
        names = {c["id"]: c["label"] for c in inv["components"]}
        items = {f"C{n}": c["label"] for n, c in enumerate([c for c in inv["components"] if c["tier"] == "required"], 1)}
        edges = [e for e in inv["connections"] if e["tier"] == "required"]
        items |= {f"E{n}": f"{names.get(e['source'], e['source'])} -> {names.get(e['target'], e['target'])}"
                  for n, e in enumerate(edges, 1)}
        payload = check_backend.structured(SKETCH_CHECK + "\n".join(f"{k}. {v}" for k, v in items.items()),
                                           [ROOT / d["image"]], SKETCH_CHECK_SCHEMA, "sketch_check")
        shown = {x["id"]: x["shown"] for x in payload["items"]}
        result = {"items": {k: {"text": v, "shown": shown.get(k)} for k, v in items.items()}}
        path.write_text(json.dumps(result, indent=1))
        return result

    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(sketch, d): d for d in todo}
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as error:
                failed += 1
                print(f"failed sketch {futures[f]['id']}: {error}", flush=True)
    results = {}
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(check, d): d for d in todo}
        for f in as_completed(futures):
            try:
                results[futures[f]["id"]] = f.result()
            except Exception as error:
                failed += 1
                print(f"failed check {futures[f]['id']}: {error}", flush=True)
    by_id = {d["id"]: d for d in todo}
    positives = [i for i in results if by_id[i]["label"] == "positive"]
    print(f"sketch {wording}: {len(todo)} diagrams, checked on {len(results)} ({len(positives)} unmodified), {failed} failed")
    for kind, name in (("C", "components"), ("E", "connections")):
        flags = [[v["shown"] for k, v in results[i]["items"].items() if k[0] == kind] for i in positives]
        n = sum(len(f) for f in flags)
        print(f"  required {name}: {n / len(flags):.1f} per sketch, {sum(sum(map(bool, f)) for f in flags) / n:.0%} shown "
              f"in the unmodified figure; sketches with one or more not shown: {sum(not all(f) for f in flags)} of {len(flags)}")
    # broken figures: how often the check finds a missing component / connection, by inserted error
    missing = lambda i, kind: any(not v["shown"] for k, v in results[i]["items"].items() if k[0] == kind)
    groups = [("unmodified", lambda d: d["label"] == "positive"),
              ("remove_box", lambda d: any(e["defect"] == "remove_box" for e in d["defects"])),
              ("blank_label", lambda d: any(e["defect"] == "blank_label" for e in d["defects"])),
              ("erase/reverse/detach arrow", lambda d: any(e["defect"] in ("erase_arrow", "reverse_arrow", "detach_arrow")
                                                           for e in d["defects"])),
              ("other broken", lambda d: d["label"] == "negative" and not any(
                  e["defect"] in ("remove_box", "blank_label", "erase_arrow", "reverse_arrow", "detach_arrow")
                  for e in d["defects"]))]
    for name, test in groups:
        ids = [i for i in results if test(by_id[i])]
        if ids:
            print(f"  {name:28s} n={len(ids):3d}: a component not shown {sum(missing(i, 'C') for i in ids):3d}, "
                  f"a connection not shown {sum(missing(i, 'E') for i in ids):3d}")
    usage = {k: sum(b.usage[k] for b in backends) for k in backends[0].usage}
    print("api usage this run:", usage)


# Direct scoring: no formula and no sketch. Each correctness dimension is asked on its own (like Beauty v6),
# with the image and the caption, and the model gives a 1-10 score as the human raters did. The prompt carries
# the scoring idea in words: what the dimension covers, what belongs to the others, and how much a problem costs.
DIRECT_QUESTION = {  # the question the human raters saw (user-study/lib/metrics.ts)
    "layout": "Are elements placed in a sensible order, without unintended overlap, content outside the canvas, or "
              "arrows running through boxes or text?",
    "connectivity": "Are the required connections present, with arrows joining the right elements in the right "
                    "direction?",
    "presence": "Are all the components mentioned in the caption visibly present?",
    "details": "Are labels complete, accurate, and meaningful, rather than missing or placeholders?",
    "legibility": "Is the text comfortably readable, without tiny type, clipping, overflow, or low contrast against "
                  "its background?",
}
DIRECT_OWNERS = """\
  LAYOUT       - reading order, nothing cut off at the canvas edge, no unintended overlap, clean arrow routing.
  CONNECTIVITY - the connections the caption calls for are drawn between the right components, in the right
                 direction; no wrong or dangling arrows.
  PRESENCE     - every component the caption calls for is drawn, and nothing out of place is added.
  DETAILS      - the labels on components and arrows are complete, accurate, and meaningful.
  LEGIBILITY   - text is comfortably readable: large and sharp enough, not clipped or overflowing, enough contrast."""
DIRECT_CONVENTIONS = {  # how real figures draw things (general points from the v1 analysis)
    "layout": "Steps should appear along the reading direction in the order the caption gives.",
    "connectivity": "One line or arrow can serve several components (a branching trunk line, a line ending at a row "
                    "of boxes, a large background arrow with the steps placed along it) and connects each of them. A "
                    "side branch, such as excluded or removed items, leaves the main flow and ends there.",
    "presence": "",
    "details": "A label that is cut off, or that names the wrong step for its place in the flow, is not accurate.",
    "legibility": "Judge text at the size the image is shown, without zooming in: text you would need to zoom in on "
                  "or guess at is not comfortably readable.",
}
DIRECT_SCORING = (
    "How to score, as an integer from 1 to 10. Look over the whole diagram for problems on this dimension only, "
    "then let each problem lower the score by how much it hurts a reader:\n"
    "- 9-10: no real problem on this dimension.\n"
    "- 7-8: small problems that do not change what the diagram says.\n"
    "- 4-6: at least one clear error that a reader would notice, or that changes part of the meaning.\n"
    "- 1-3: several such errors, or the diagram is misleading or unusable on this dimension.\n"
    "Be sensitive: a single clear error is enough to leave the top band. Errors that change the meaning cost more "
    "than cosmetic ones, and several errors cost more than one. A figure that is organized or worded differently "
    "from what you would draw, but still shows what the caption describes, has no problem. Problems that belong to "
    "another dimension do not lower this score."
)
DIRECT_SCHEMA = {
    "type": "object",
    "properties": {"problems": {"type": "array", "items": {"type": "string"}},
                   "score": {"type": "integer", "minimum": 1, "maximum": 10},
                   "reasoning": {"type": "string"}},
    "required": ["problems", "score", "reasoning"],
    "additionalProperties": False,
}
# Direct-scoring versions: d1 is the prompt above, d1r repeats it (noise). d2 adds one general sentence for each
# kind of d1 mistake on HS001-HS100: clear problems filed as small, boxes touching the image edge called cut off,
# the judge's own idea of the process used against the arrows, and numbers recomputed across boxes.
DIRECT = {
    "d1": {},
    "d1r": {},
    "d2": {
        "scoring": " A problem is small only when a reader would barely notice it; one that makes part of the "
                   "diagram harder to read, follow, or trust is a clear error.",
        "conventions": {
            "layout": " A figure cropped close to its content, with boxes touching the image edge, is not cut off; "
                      "only content that visibly continues past the edge is.",
            "connectivity": " Judge the arrows against what the caption states, not against your own idea of how the "
                            "process should work.",
            "details": " Judge the labels themselves: do not recompute numbers across boxes or second-guess wording "
                       "that fits the caption.",
        },
    },
}
# d3 = d2 + the checklist (steps 1 and 2 of the original method): the caption sketch, made with the v3 side-branch
# sentence, is shown to each dimension, and the model goes through it item by item before scoring. No formula.
DIRECT["d3"] = {**DIRECT["d2"], "sketch": True}
# d4 = d2 with one change per dimension, from the d2 mistakes on HS001-HS100 (Presence is unchanged, as a noise check):
# Layout still called tightly cropped figures cut off; Connectivity flagged exclusion boxes drawn in the main chain
# (which d2's side-branch sentence invited) and labels in the wrong boxes; Details took points for typos and
# abbreviations; Legibility missed that the study showed each figure fitted to a small panel, so tall or large
# figures looked much smaller to the raters than to the model (judge minus human correlated -0.48 with that scale).
DIRECT["d4"] = {
    "scoring": DIRECT["d2"]["scoring"],
    "base": {
        "layout": "Steps should appear along the reading direction in the order the caption gives. Many figures are "
                  "cropped tight, so boxes, labels, or arrows touch the image edge, and a box border lying on the "
                  "edge can look thin or missing there; that is not cut off. Content is cut off only when something "
                  "is visibly incomplete: text sliced through, or a box clearly missing part of its contents "
                  "compared with similar boxes.",
        "connectivity": "One line or arrow can serve several components (a branching trunk line, a line ending at a "
                        "row of boxes, a large background arrow with the steps placed along it) and connects each of "
                        "them. Excluded or removed items are often a side branch that leaves the main flow and ends "
                        "there, but a figure may also put the exclusion box in the main chain, meaning the next step "
                        "is what remains; both are correct. Judge the arrows against what the caption states, not "
                        "against your own idea of how the process should work. If the arrows would be right with the "
                        "labels in their proper boxes, a label in the wrong box is a DETAILS problem, not a "
                        "connectivity one.",
        "presence": "",
        "details": DIRECT_CONVENTIONS["details"] + DIRECT["d2"]["conventions"]["details"] + " Spelling slips, "
                   "abbreviations, and short paraphrases that still name the right thing do not lower the score.",
        "legibility": "{display} Judge the text at that size, without zooming in: text you would need to zoom in on "
                      "or guess at is not comfortably readable.",
    },
}
# d5 = d4, keeping what helped beyond noise and undoing what hurt. Layout gained most (Spearman 0.50 -> 0.64) and is
# kept. The display size ranked Legibility a little better but pushed it 1.2 points below the humans (figures shown
# under 40% got 3.7, humans 5.7), so d5 keeps the size and softens what it costs. The Details sentence on typos made
# the judge pass real wrong labels, and the Connectivity sentence on labels in the wrong boxes excused real arrow
# errors; both are dropped (Details is d2 again). Layout and Presence have the d4 prompt and Details the d2 prompt,
# so their result files are copied, not re-run.
DIRECT["d5"] = {
    "scoring": DIRECT["d2"]["scoring"],
    "base": {
        **DIRECT["d4"]["base"],
        "connectivity": DIRECT["d4"]["base"]["connectivity"].split(" If the arrows would be right")[0],
        "details": DIRECT_CONVENTIONS["details"] + DIRECT["d2"]["conventions"]["details"],
        "legibility": "{display} Judge the text at that size, as a reader looking at the whole figure would: text "
                      "that is still easy to read is fine; text small or dense enough that the reader must look "
                      "closely is a small problem; text the reader would need to zoom in on or guess at is a clear "
                      "error.",
    },
}
# d6 = d5 with another Legibility wording. d5's softer cost hardly moved the level (still 1.1 points below the humans,
# and 10 clean figures under 8): told the figure is shown smaller, the judge marked down any small text, even at 60%
# where d2 matched the humans. d6 says small text that still reads at that size is fine.
DIRECT["d6"] = {**DIRECT["d5"], "base": {
    **DIRECT["d5"]["base"],
    "legibility": "{display} Judge the text at that size, as a reader looking at the whole figure would. Text in "
                  "figures is often small, and small text that still reads at that size is fine; lower the score "
                  "only for text the reader would have to strain, zoom in on, or guess at to read.",
}}
# d7 = the d6 ideas as short general rules, with no sizes: Layout and Connectivity say the same in fewer words;
# Legibility asks the model to look at the whole figure as a reader would and judge text small compared with the
# figure (the study showed figures fitted to a small panel, but the panel size is not a rule for the judge); Presence
# takes "visibly present" literally, as the humans did (blank, cut-off, or hidden boxes: humans 5-7, judge 6-10).
# Details stays d2.
DIRECT["d7"] = {"scoring": DIRECT["d2"]["scoring"], "base": {
    "layout": "Steps should appear along the reading direction in the order the caption gives. Content touching the "
              "image edge is not cut off; it is cut off only when part of it is visibly missing, such as text "
              "sliced through or a box missing part of its contents.",
    "connectivity": DIRECT_CONVENTIONS["connectivity"].split(" A side branch")[0] + " Excluded items may branch off "
                    "to the side or sit in the main chain (the next step is what remains); both are correct."
                    + DIRECT["d2"]["conventions"]["connectivity"],
    "presence": "Count a component as present only when a reader can see what it is: a box that is blank, cut off, "
                "or hidden under another does not show its component.",
    "details": DIRECT_CONVENTIONS["details"] + DIRECT["d2"]["conventions"]["details"],
    "legibility": "Look at the figure as a reader would, taking in the whole figure at once: text that is small "
                  "compared with the figure, so that the reader has to look closely to make it out, is not "
                  "comfortably readable.",
}}
# d8 = the best part of each so far. Presence keeps the d7 rule (Spearman 0.59 -> 0.73), Connectivity the d6 wording,
# Details d2. The short d7 Layout rule lost the d6 gain (false alarms 7 -> 14: the judge still saw a border on the
# edge as cut off), so d8 names that case in rule form. The d7 Legibility wording changed nothing (same as d2 at every
# display scale) and the d6 numbers overshot, so d8 states why figures look smaller without giving numbers.
DIRECT["d8"] = {"scoring": DIRECT["d2"]["scoring"], "base": {
    **DIRECT["d7"]["base"],
    "layout": "Steps should appear along the reading direction in the order the caption gives. A box or label that "
              "touches the image edge is not cut off, even where its border lies on the edge and looks thin or "
              "missing. Content is cut off only when part of it is visibly missing, such as text sliced through or a "
              "box missing part of its contents.",
    "connectivity": DIRECT["d6"]["base"]["connectivity"],
    "legibility": "Readers see the whole figure fitted to a screen, so a large or tall figure is shown smaller than "
                  "its file. Judge text at that size, without zooming in: text that is small compared with the whole "
                  "figure becomes hard to read, and text you would need to zoom in on or guess at is not comfortably "
                  "readable.",
}}
# d9 = d8 with Legibility and Details closer to how the humans rated. Legibility: when the d8 judge found hard text it
# still reasoned "most text is clear" and gave 6-8 where humans gave 3-5, and it reads fine print a person cannot; d9
# asks it to judge as a person, by the hardest text, with weak contrast named (no sizes, same image). Details: humans
# rarely marked down a label in the wrong place (label swaps: humans 7.6, judge 6.9; steps drawn out of order: humans
# 9.7, judge 7.0); d9 makes the first a small problem and the second Layout's.
DIRECT["d9"] = {"scoring": DIRECT["d2"]["scoring"], "base": {
    **DIRECT["d8"]["base"],
    "details": "A label that is cut off is not accurate. A label that names a real component from the caption but "
               "sits in the wrong place in the flow is a small problem; steps drawn in the wrong order are a LAYOUT "
               "problem. Judge the labels themselves: do not recompute numbers across boxes or second-guess wording "
               "that fits the caption.",
    "legibility": "Judge as a person looking at the whole figure on a screen would, not as a model that can decode "
                  "fine print: people cannot make out every small word. Text a person would find hard to read, "
                  "because it is small, thin, blurry, or pale, or because it contrasts weakly with its fill, is not "
                  "comfortably readable. A figure is as legible as its hardest text: hard-to-read labels lower the "
                  "score even when most text is clear.",
}}
# d10 = d8 with two changes. Legibility: the only thing that moved the judge so far was being told this figure is
# shown smaller (d6, with numbers, overshot); the gap is in figures shown much smaller (under 45% of their size in the
# study's panel: humans 6.1, d8 7.7; elsewhere within 0.4), so those get one size sentence in words, and every other
# figure keeps the d8 sentence. Details: on clean figures the judge went from 10 straight to 6-8 for wording that
# still names the same thing (an abbreviation, a dropped word, a typo), where humans gave 10; d10 says such a label is
# accurate and keeps naming a different thing or contradicting the caption as wrong (d4's broader "typos do not
# count" passed real wrong labels).
DIRECT["d10"] = {"scoring": DIRECT["d2"]["scoring"], "base": {
    **DIRECT["d8"]["base"],
    "details": DIRECT_CONVENTIONS["details"] + " A label that names the same thing as the caption is accurate even if "
               "it is worded differently, abbreviated, or slightly misspelled; a label that names a different thing, "
               "contradicts the caption, or leaves out what the caption asks it to state is wrong."
               + DIRECT["d2"]["conventions"]["details"],
    "legibility": "{size}" + DIRECT["d8"]["base"]["legibility"].split("its file.")[1],
}}


def _size(image):
    """d10: one sentence in words for figures the study showed much smaller than their file; the d8 sentence otherwise."""
    from PIL import Image
    w, h = Image.open(image).size
    if min(PANEL[0] / w, PANEL[1] / h) < 0.45:  # d10 result: no effect (large figures d8 7.7, d10 8.0, humans 6.1)
        return ("Readers see the whole figure fitted to a screen, and this figure is large or tall, so they see it "
                "much smaller than its file.")
    return "Readers see the whole figure fitted to a screen, so a large or tall figure is shown smaller than its file."


# d11 = d8 with half of the d10 Details sentence. Its first half did what it should (clean figures with an
# abbreviation, a dropped word, or a typo went from 7-8 to 10, as the humans rated them); its second half, "leaves out
# what the caption asks it to state", made the judge flag omissions on clean figures (10 -> 8), so d11 drops it.
DIRECT["d11"] = {"scoring": DIRECT["d2"]["scoring"], "base": {
    **DIRECT["d8"]["base"],
    "details": DIRECT_CONVENTIONS["details"] + " A label that names the same thing as the caption is accurate even if "
               "it is worded differently, abbreviated, or slightly misspelled; a label that names a different thing or "
               "contradicts the caption is wrong." + DIRECT["d2"]["conventions"]["details"],
}}
# d12 = d11 with Legibility back to the d2 wording (the "fitted to a screen" premise is dropped: readers can open the
# figure, and no wording without numbers moved the judge anyway) and Details a little stricter. d11 passed labels that
# drop a count or condition the caption gives (a cohort box without its n, a "Negative" result without its test),
# where d8 and the humans marked them down; d10's broader "leaves out what the caption asks" also flagged dropped
# words on clean figures. d12 names counts, values, and conditions only.
DIRECT["d12"] = {"scoring": DIRECT["d2"]["scoring"], "base": {
    **DIRECT["d11"]["base"],
    "details": DIRECT_CONVENTIONS["details"] + " A label that names the same thing as the caption is accurate even if "
               "it is worded differently, abbreviated, or slightly misspelled; a label that names a different thing, "
               "contradicts the caption, or drops a count, value, or condition the caption gives for that component "
               "is wrong." + DIRECT["d2"]["conventions"]["details"],
    "legibility": DIRECT_CONVENTIONS["legibility"],
}}
# d13 and d14 = d12 with Connectivity changed (Connectivity had not moved past d2: Spearman 0.55-0.57). The d12 judge
# missed inserted arrow errors (an erased and a reversed arrow both scored 10) and flagged how clean figures draw
# connections (plain lines without arrowheads, arrows ending at a panel border, sources stacked above their total). d13
# asks it to go through the caption's connections one by one and accepts those drawings. d14 adds what the humans did
# when a box is removed, cut off, or too faint (removed box: humans 5.0, judge 6.8): the flow through it is broken.
_CONNECTIVITY_D13 = (
    "One line or arrow can serve several components (a branching trunk line, a line ending at a row of boxes, a large "
    "background arrow with the steps placed along it, an arrow ending at the border of a group or panel) and connects "
    "each of them; a plain line without an arrowhead connects two components when the direction is clear from the "
    "layout. Excluded or removed items are often a side branch that leaves the main flow and ends there, but a figure "
    "may also put the exclusion box in the main chain, meaning the next step is what remains; both are correct. Judge "
    "the arrows against what the caption states, not against your own idea of how the process should work. Go through "
    "the connections the caption calls for one by one: check that a line joins the two components and that its "
    "arrowhead, if it has one, points the right way.")
DIRECT["d13"] = {"scoring": DIRECT["d2"]["scoring"], "base": {**DIRECT["d12"]["base"], "connectivity": _CONNECTIVITY_D13}}
DIRECT["d14"] = {"scoring": DIRECT["d2"]["scoring"], "base": {
    **DIRECT["d12"]["base"],
    "connectivity": _CONNECTIVITY_D13 + " When a component on the flow is missing, cut off, or too faint to see, the "
                    "reader cannot follow the flow through it, so the connections through it are not shown either.",
}}
# d15 = d12 with Layout changed. For a layout error a reader sees at a glance (cut off, overlapping, an arrow through a
# box, steps out of order) humans gave 2-4 and the d8 judge about 6 (humans 1-3: judge mean 5.4); and the judge
# marked down flows that wrap into a second row or column, which humans rated 10.
DIRECT["d15"] = {"scoring": DIRECT["d2"]["scoring"], "base": {
    **DIRECT["d12"]["base"],
    "layout": DIRECT["d8"]["base"]["layout"] + " A flow that wraps into a new row or column, or turns back in a U "
              "shape, still reads in one direction. A layout error a reader sees at a glance (content cut off, boxes "
              "overlapping or knocked out of line, an arrow running through a box, steps out of order) makes the "
              "figure look broken: one such error already belongs in 3-5, and several in 1-3.",
}}
# d16 = d12 with a full Connectivity section instead of a few sentences: how to check the connections step by step,
# what is not an error, and what each score band means for connections. Connectivity was the dimension where the
# judge disagreed most with itself (d1 vs d1r RMSE 1.51); the idea is that a vaguer task gives noisier scores.
# d16r repeats d16 and d12r repeats d12's Connectivity, to compare run-to-run noise.
DIRECT["d16"] = {"scoring": DIRECT["d2"]["scoring"], "base": {
    **DIRECT["d12"]["base"],
    "connectivity": (
        "\n\nHow to check, in order:\n"
        "1. From the caption, work out the flow: which step leads to which, including branches (yes/no, study arms, "
        "exclusions) and any loops back.\n"
        "2. In the figure, follow each of those connections from one component to the next. A connection may run "
        "through a shared trunk line, a bracket, a line ending at a row of boxes, an arrow to the border of a group or "
        "panel, or a large background arrow; all of these connect. A plain line without an arrowhead connects two "
        "components when the direction is clear from the layout.\n"
        "3. Check the direction of every arrowhead, including in faint or blurry parts of the figure: it points from "
        "the earlier step to the later one, or back to an earlier step only where the caption describes a loop.\n"
        "4. Check for arrows that should not be there: one that joins components the caption does not connect, or one "
        "that ends in empty space.\n\n"
        "Not an error: the same flow organized differently from what you would draw (branches in another order, an "
        "exclusion box in the main chain meaning the next step is what remains, sources stacked above the total they "
        "add up to); a connector that stops just short of a box when it is clear which box it reaches; extra "
        "connections that fit the caption. Judge against what the caption states, not your own idea of the process.\n\n"
        "What the bands mean for CONNECTIVITY:\n"
        "- 9-10: every connection the caption calls for can be followed, in the right direction, and no arrow is "
        "wrong.\n"
        "- 7-8: a connection is drawn loosely or ambiguously (an arrowhead hard to make out, a line that could be read "
        "two ways), but a reader can still work out the flow.\n"
        "- 4-6: one connection the caption calls for is missing, reversed, or goes to the wrong component, while the "
        "rest of the flow is right.\n"
        "- 1-3: several such errors, or one that breaks the main path so a reader cannot get from the start to the "
        "end."),
}}
DIRECT["d16r"] = DIRECT["d16"]
# d17 = d12 with Connectivity and Presence rewritten, now tuned on HS001-HS200. Most big disagreements in both were
# the judge reading the caption, a short summary rewritten from the figure, as an exact list: on unmodified figures it
# wanted an arrow for every step the caption mentions, treated a small box beside an arrow stating its condition
# ("ICC >= 0.75") as a step, wanted an arrow between two boxes that restate one stage, and called a component missing
# when its label said less than the caption. It also read tiny text as a blank box and called a stage shown in each
# study arm a duplicate.
DIRECT["d17"] = {"scoring": DIRECT["d2"]["scoring"], "base": {
    **DIRECT["d12"]["base"],
    "connectivity": (
        "The caption is a short summary rewritten from the figure: it may merge, split, or reorder steps, and it does "
        "not describe every line. Use it to understand the flow, and count a connection as wrong only when the "
        "figure clearly contradicts that flow: two steps that must follow each other have nothing joining them, an "
        "arrow points the wrong way, or an arrow leads to the wrong component. One line or arrow can serve several "
        "components (a branching trunk line, a line ending at a row of boxes, a large background arrow with the "
        "steps placed along it, an arrow ending at the border of a group or panel) and connects each of them. A small "
        "box beside an arrow that states its condition or a note (a threshold, a criterion, a count) labels that "
        "arrow and needs no connection of its own. Two boxes that restate the same step need no arrow between them. "
        "Excluded or removed items are often a side branch that leaves the main flow and ends there, but a figure may "
        "also put the exclusion box in the main chain, meaning the next step is what remains; both are correct. Judge "
        "the arrows against what the caption states, not against your own idea of how the process should work."),
    "presence": (
        "The caption is a short summary rewritten from the figure and may give more detail than the figure's labels "
        "(where the data come from, what each arm receives). Go through the components the caption names and find "
        "each in the figure: a component is present when the figure shows that step or group, even if its label says "
        "less than the caption (that is a DETAILS matter). Count a component as present only when a reader can see "
        "what it is: a box with no text, cut off, or hidden under another does not show its component; a box whose "
        "text is merely small or faint is present (that is a LEGIBILITY matter). Boxes that repeat a name in "
        "different arms or branches, or restate a stage with added detail, are not duplicates."),
}}
DIRECT["d12r"] = DIRECT["d12"]
# d18 = d17 with two more changes. Connectivity: d17's remaining false alarms on clean figures (8 of 18 cases the judge
# scored 3+ below the humans) still asked for an arrow at every step the caption mentions, and the rubric sentence
# still says "every connection the description calls for is drawn"; d18 rewords it to the flow the caption describes
# (config/rubric.json is unchanged). Presence: its caption false alarms are gone (0 on clean figures), and it is now
# too lenient: it missed removed boxes (4 of 8, humans 7); d18 adds that a step the flow jumps over is missing.
DIRECT["d18"] = {"scoring": DIRECT["d2"]["scoring"], "base": {
    **DIRECT["d17"]["base"],
    "presence": DIRECT["d17"]["base"]["presence"] + " Check that the flow passes through every step the caption names "
                "in order: when the caption names a step between two others and the figure goes straight from one to "
                "the other with no box for it anywhere, that step is missing.",
}, "rubric": {
    "connectivity.connections": "The figure's connections follow the flow the caption describes, directly or "
                                "through intermediate drawn steps, in the right direction. A connection to a component "
                                "that is not drawn is judged under Presence, not here.",
}}
# d19 = d17 + the v4 caption sketch as a checklist for Presence, Details, and Connectivity (Layout and Legibility as
# d17). Checked against the unmodified figures of HS001-HS200, v4 has a required component the figure does not show in
# 5 of 103 sketches (v1 9, v3 11) and a required connection in 25 (v1 and v3 37); on broken figures its components
# flag 5 of 8 removed boxes and 7 of 9 blank labels, its connections 8 of 22 arrow errors.
DIRECT["d19"] = {**DIRECT["d17"], "sketch": "v4", "sketch_dims": ["presence", "details", "connectivity"]}
COPIED = {"d5": {"layout": "d4", "presence": "d4", "details": "d2"},  # same prompt, so the same result file
          "d6": {"layout": "d4", "presence": "d4", "details": "d2", "connectivity": "d5"},
          "d7": {"details": "d2"},
          "d8": {"connectivity": "d5", "presence": "d7", "details": "d2"},
          "d9": {"layout": "d8", "connectivity": "d5", "presence": "d7"},
          "d10": {"layout": "d8", "connectivity": "d5", "presence": "d7", "legibility": "d8"},  # Legibility: figures not shown much smaller
          "d11": {"layout": "d8", "connectivity": "d5", "presence": "d7", "legibility": "d8"},
          "d12": {"layout": "d8", "connectivity": "d5", "presence": "d7", "legibility": "d2"},
          "d13": {"layout": "d8", "presence": "d7", "details": "d12", "legibility": "d2"},
          "d14": {"layout": "d8", "presence": "d7", "details": "d12", "legibility": "d2"},
          "d15": {"connectivity": "d5", "presence": "d7", "details": "d12", "legibility": "d2"},
          "d16": {"layout": "d8", "presence": "d7", "details": "d12", "legibility": "d2"},
          "d16r": {"layout": "d8", "presence": "d7", "details": "d12", "legibility": "d2"},
          "d12r": {"layout": "d8", "presence": "d7", "details": "d12", "legibility": "d2"},
          "d17": {"layout": "d12", "details": "d12", "legibility": "d12"},
          "d18": {"layout": "d12", "details": "d12", "legibility": "d12"},
          "d19": {"layout": "d17", "legibility": "d17"}}
PANEL = (660, 360)  # the study's diagram panel on a 1440-pixel-wide screen (user-study/app/globals.css)


def _display(image):
    """How the study showed the figure: scaled to fit PANEL (small figures are enlarged, large ones shrunk)."""
    from PIL import Image
    w, h = Image.open(image).size
    scale = min(PANEL[0] / w, PANEL[1] / h)
    size = "at about its full size" if 0.95 <= scale <= 1.05 else f"at about {round(100 * scale)}% of its size"
    return (f"Readers see this figure ({w}x{h} pixels) fitted to a panel about {PANEL[0]}x{PANEL[1]} pixels, so it "
            f"is shown {size}.")
SKETCH_NOTE = (
    "Checklist: before seeing the figure, another model read only the caption and listed what this diagram should "
    "contain; the items for this dimension are below. Go through the required items one by one and decide whether "
    "the figure shows each, matching by meaning, then look at the rest of the figure. Typical items are only "
    "guesses and never lower the score. The checklist is a sketch: a figure organized or worded differently that "
    "still shows what the caption describes is fine.\n"
)


def _sketch_lines(inventory, dimension):
    """The checklist items one dimension needs, as plain lines (Legibility needs none)."""
    names = {c["id"]: c["label"] for c in inventory.get("components", [])}
    arrow = lambda e: f"{names.get(e['source'], e['source'])} -> {names.get(e['target'], e['target'])}" + (
        f" (labeled '{e['label']}')" if e.get("label") else "")
    if dimension == "presence":
        return [f"- [{c['tier']}] {c['label']}" for c in inventory.get("components", [])]
    if dimension == "details":
        return [f"- [{c['tier']}] {c['label']}" for c in inventory.get("components", [])] + [
            f"- [{e['tier']}] arrow {arrow(e)}" for e in inventory.get("connections", []) if e.get("label")]
    if dimension in ("connectivity", "layout"):
        return [f"- [{e['tier']}] {arrow(e)}" for e in inventory.get("connections", [])]
    return []


def _direct_prompt(dimension, caption, wording="d1", inventory=None, image=None):
    change = DIRECT[wording]
    lines = _sketch_lines(inventory, dimension) if inventory and dimension in change.get("sketch_dims", CORRECTNESS) else []
    checklist = SKETCH_NOTE + "\n".join(lines) + "\n\n" if lines else ""
    criteria = [change.get("rubric", {}).get(c["id"], c["description"])  # a version may reword a rubric criterion
                for c in json.load(open(ROOT / "config" / "rubric.json"))["criteria"] if c["dimension"] == dimension]
    conventions = change.get("base", DIRECT_CONVENTIONS)[dimension]
    if "{display}" in conventions:
        conventions = conventions.format(display=_display(image))
    if "{size}" in conventions:
        conventions = conventions.format(size=_size(image))
    name = dimension.upper()
    return (
        f"You are rating one aspect of a flowchart: {name}. The attached image is the flowchart, and the caption "
        f"below describes what it should show.\n\n{name}: {DIRECT_QUESTION[dimension]}\nIt covers: "
        + " ".join(criteria) + (" " + conventions if conventions else "")
        + change.get("conventions", {}).get(dimension, "")
        + f"\n\nEach aspect has its own rater:\n{DIRECT_OWNERS}\nJudge only {name}.\n\n"
        + DIRECT_SCORING + change.get("scoring", "") + "\n\n" + checklist
        + "First list the problems you see on this dimension, each with where it is (an empty list if there are "
        "none), then give the score and a one- or two-sentence reason.\n\nCaption:\n" + caption
    )


def _correctness_file(wording):
    return "correctness.json" if wording == "v1" else f"correctness_{wording}.json"


def _inventory_file(wording):  # wordings that change stage 1 get their own sketch
    return f"inventory_{wording}.json" if WORDINGS[wording].get("expectation") else "inventory.json"


class _Patched:
    """A backend that adds sentences to the prompt before passing it on (used for the stage-1 sketch)."""

    def __init__(self, backend, edits):
        self.backend, self.edits, self.name = backend, edits, backend.name

    def structured(self, prompt, *args):
        for old, extra in self.edits:
            assert prompt.count(old) == 1, old
            prompt = prompt.replace(old, old + extra)
        return self.backend.structured(prompt, *args)


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _humans():
    return json.load(open(HERE / "human_ratings.json"))["diagrams"]


# ---------- score ----------

def _in_range(d, first, start=1):
    return start <= int(d["id"][2:]) <= (first or 10 ** 6)


def _pipeline(model, wording="v1"):
    from dataclasses import replace
    from diagram_correctness import vlm
    from diagram_correctness.models import PresencePolicy
    from diagram_correctness.pipeline import CorrectnessPipeline, PipelineConfig
    from diagram_correctness.rubric import load_config, load_rubric
    from diagram_correctness.vlm import ExpectationExtractor, OpenAIBackend, VLMJudge
    settings = load_config(ROOT / "config" / "evaluator.json")
    policy = PresencePolicy.from_dict(settings.get("presence_policy"))
    judge_backend, expectation_backend = OpenAIBackend(model), OpenAIBackend(model)
    change = WORDINGS[wording]
    criteria = [replace(c, description=c.description + change.get("criteria", {}).get(c.id, ""))
                for c in load_rubric(ROOT / "config" / "rubric.json")]
    for key, extra in change.get("rules", {}).items():
        vlm.COUNTING_RULES[key] += extra

    class Judge(VLMJudge):
        def prompt(self, *args, **kwargs):
            text = super().prompt(*args, **kwargs)
            for old, extra in change.get("prompt", []):
                assert text.count(old) == 1, old
                text = text.replace(old, old + extra)
            return text

    pipeline = CorrectnessPipeline(PipelineConfig(
        criteria=criteria,
        judges=[Judge(judge_backend, policy, samples=int(settings.get("judge_samples", 1)))],
        expectation_extractor=ExpectationExtractor(_Patched(expectation_backend, change.get("expectation", []))),
        weighting=settings.get("weighting", "auto"),
        dimension_reducer=settings.get("dimension_reducer", "mean"),
        error_factor=settings.get("error_factor"),
        severity_multipliers=settings.get("severity_multipliers"),
        fallback_frequencies=settings.get("fallback_frequencies"),
    ))
    return pipeline, [judge_backend, expectation_backend]


def score(workers: int, only: set[str] | None = None, first: int | None = None, wording: str = "v1",
          start: int = 1) -> None:
    _load_env()
    from diagram_correctness.beauty import judge_beauty_dimension
    from diagram_correctness.beauty_v6 import BEAUTY_V6_MODEL_DIMENSIONS, BEAUTY_V6_PROMPTS
    from diagram_correctness.canvas_fill import judge_canvas_fill
    from diagram_correctness.vlm import OpenAIBackend
    model = os.environ.get("OPENAI_MODEL") or "gpt-6-luna"
    if model != "gpt-6-luna":
        raise SystemExit(f"model is {model!r}; this experiment runs on gpt-6-luna only")
    pipeline, backends = _pipeline(model, "v1" if wording in DIRECT else wording)
    beauty_backend = OpenAIBackend(model)
    backends.append(beauty_backend)
    print(f"scoring with {model}, wording {wording}", flush=True)

    def correctness(d):
        folder = RESULTS / d["id"]
        path = folder / _inventory_file(wording)  # the expectation sketch from the caption, made once
        if not path.exists():
            inv = pipeline.config.expectation_extractor.extract(d["caption"], pipeline.config.criteria)
            path.write_text(json.dumps(inv, indent=1))
        report = pipeline.run(description=d["caption"], candidate_image=ROOT / d["image"],
                              inventory=json.loads(path.read_text()), run_deterministic=False).to_dict()
        return {"correctness": report["correctness"],
                "dimensions": {k: v["score"] for k, v in report["dimensions"].items()},
                "criteria": {k: v["criteria"] for k, v in report["dimensions"].items()},
                "issues": {m["criterion_id"]: [x["description"] for x in m["issues"]] for m in report["metrics"] if m.get("issues")}}

    def beauty(d, dimension):
        if dimension == "canvas_fill":  # computed in Python, no model call
            result = judge_canvas_fill(ROOT / d["image"])
        else:
            result = judge_beauty_dimension(beauty_backend, ROOT / d["image"], dimension, BEAUTY_V6_PROMPTS)
        return {"score": result.score, "observation": result.observation, "reasoning": result.reasoning}

    sketches = {}
    if DIRECT.get(wording, {}).get("sketch"):  # step 1 once per diagram, with the v3 side-branch sentence
        sketch_wording = DIRECT[wording]["sketch"] if isinstance(DIRECT[wording]["sketch"], str) else "v3"
        sketch_pipeline, sketch_backends = _pipeline(model, sketch_wording)
        backends += sketch_backends

        def sketch(d):
            path = RESULTS / d["id"] / _inventory_file(sketch_wording)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                inv = sketch_pipeline.config.expectation_extractor.extract(d["caption"], sketch_pipeline.config.criteria)
                path.write_text(json.dumps(inv, indent=1))
            return d["id"], json.loads(path.read_text())

        todo = [d for d in _humans() if (not only or d["id"] in only) and _in_range(d, first, start)]
        with ThreadPoolExecutor(workers) as pool:
            sketches = dict(pool.map(sketch, todo))
        print(f"{len(sketches)} sketches ready", flush=True)

    def direct(d, dimension):
        prompt = _direct_prompt(dimension, d["caption"], wording, sketches.get(d["id"]), ROOT / d["image"])
        payload = beauty_backend.structured(prompt, [ROOT / d["image"]],
                                            DIRECT_SCHEMA, f"diagram_correctness_{dimension}")
        if not isinstance(payload.get("score"), int) or not 1 <= payload["score"] <= 10:
            raise RuntimeError(f"invalid score for {dimension}: {payload.get('score')!r}")
        return payload

    jobs = []
    for d in _humans():
        if wording in DIRECT:
            if (not only or d["id"] in only) and _in_range(d, first, start):
                (RESULTS / d["id"]).mkdir(parents=True, exist_ok=True)
                jobs += [(d, f"direct_{wording}_{k}", lambda d=d, k=k: direct(d, k)) for k in CORRECTNESS]
            continue
        if (only and d["id"] not in only) or not _in_range(d, first, start):
            continue
        (RESULTS / d["id"]).mkdir(parents=True, exist_ok=True)
        jobs.append((d, _correctness_file(wording)[:-5], lambda d=d: correctness(d)))
        if wording != "v1":  # Beauty is not part of the correctness prompt; it is scored once
            continue
        for dim in (*BEAUTY_V6_MODEL_DIMENSIONS, "canvas_fill"):
            jobs.append((d, f"beauty_{dim}", lambda d=d, dim=dim: beauty(d, dim)))
    for d, name, _ in jobs:  # a prompt identical to another version's reuses that version's result
        dim = name.rsplit("_", 1)[1]
        source = COPIED.get(wording, {}).get(dim)
        target, origin = RESULTS / d["id"] / f"{name}.json", RESULTS / d["id"] / f"direct_{source}_{dim}.json"
        if (source and origin.exists() and not target.exists() and _direct_prompt(dim, d["caption"], wording, None, ROOT / d["image"])
                == _direct_prompt(dim, d["caption"], source, None, ROOT / d["image"])):
            target.write_text(origin.read_text())
    jobs = [j for j in jobs if not (RESULTS / j[0]["id"] / f"{j[1]}.json").exists()]
    print(f"{len(jobs)} jobs to run", flush=True)

    def run(job):
        d, name, fn = job
        (RESULTS / d["id"] / f"{name}.json").write_text(json.dumps(fn(), indent=1))

    done = failed = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(run, j): j for j in jobs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:  # a failed job is retried on the next `score` run
                failed += 1
                print(f"failed {futures[future][0]['id']} {futures[future][1]}: {error}", flush=True)
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} done, {failed} failed", flush=True)
    usage = {k: sum(b.usage[k] for b in backends) for k in backends[0].usage}
    print("api usage this run:", usage)


# ---------- compare ----------

def _judge(diagram_id, wording="v1"):
    folder = RESULTS / diagram_id
    if wording in DIRECT:  # 1-10 from the model, overall = mean of the five (as for the humans)
        paths = [folder / f"direct_{wording}_{k}.json" for k in CORRECTNESS]
        if not all(p.exists() for p in paths):
            return None
        scores = {k: json.load(open(p))["score"] for k, p in zip(CORRECTNESS, paths)}
        correctness = statistics.mean(scores.values())
    else:
        if not (folder / _correctness_file(wording)).exists():
            return None
        c = json.load(open(folder / _correctness_file(wording)))  # everything on 0-10: correctness is 0-1, beauty 0-100
        scores = {k: None if c["dimensions"].get(k) is None else 10 * c["dimensions"][k] for k in CORRECTNESS}
        correctness = 10 * c["correctness"]
    for dim in BEAUTY:
        path = folder / f"beauty_{dim}.json"
        scores[dim] = json.load(open(path))["score"] / 10 if path.exists() else None
    return {"correctness": correctness, "scores": scores}


def _stats(pairs):
    """pairs of (human 1-10, judge 0-10)."""
    pairs = [(h, j) for h, j in pairs if h is not None and j is not None]
    if len(pairs) < 3:
        return {"n": len(pairs)}
    h, j = zip(*pairs)
    diff = [b - a for a, b in pairs]
    out = {"n": len(pairs), "human_mean": statistics.mean(h), "judge_mean": statistics.mean(j),
           "mae": statistics.mean(abs(x) for x in diff), "rmse": math.sqrt(statistics.mean(x * x for x in diff)),
           "bias": statistics.mean(diff)}
    constant = len(set(h)) == 1 or len(set(j)) == 1
    out["spearman"] = None if constant else float(spearmanr(h, j).statistic)
    out["kendall"] = None if constant else float(kendalltau(h, j).statistic)  # tau-b, handles ties
    out["pearson"] = None if constant else float(pearsonr(h, j).statistic)
    out["within_1"] = sum(abs(x) <= 1 for x in diff) / len(diff)  # share of diagrams where the two differ by 1 point or less
    return out


def _auc(scores, labels):
    """How often a positive scores above a negative (0.5 = no separation, 1 = perfect)."""
    pos = [s for s, l in zip(scores, labels) if l == "positive" and s is not None]
    neg = [s for s, l in zip(scores, labels) if l == "negative" and s is not None]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg)) if pos and neg else None


def compare(first: int | None = None, wording: str = "v1", start: int = 1) -> None:
    humans = [d for d in _humans() if _in_range(d, first, start)]
    rows = []
    for d in humans:
        judge = _judge(d["id"], wording)
        if judge is None:
            continue
        human = dict(d["scores"])  # as rated, 1-10
        rows.append({"id": d["id"], "difficulty": d["difficulty"], "label": d["label"], "rater": d["rater"],
                     "defects": d["defects"],
                     "human": human, "judge": judge["scores"], "judge_correctness": judge["correctness"]})
    print(f"wording {wording}: {len(rows)} of {len(humans)} diagrams scored by the judge\n")

    def table(subset):
        return {k: _stats([(r["human"][k], r["judge"][k]) for r in subset]) for k in CORRECTNESS + BEAUTY}

    def mean(values):
        values = [v for v in values if v is not None]
        return statistics.mean(values) if values else None

    for r in rows:  # overall per axis: the human mean of the dimensions vs the judge's own overall score
        r["human"]["correctness_overall"] = mean([r["human"][k] for k in CORRECTNESS])
        r["judge"]["correctness_overall"] = r["judge_correctness"]
        r["human"]["beauty_overall"] = mean([r["human"][k] for k in BEAUTY])
        r["judge"]["beauty_overall"] = mean([r["judge"][k] for k in BEAUTY])
        for who in ("human", "judge"):  # overall: the two axes count the same
            r[who]["overall"] = mean([r[who]["correctness_overall"], r[who]["beauty_overall"]])
    overall = ["correctness_overall", "beauty_overall", "overall"]

    summary = {
        "n": len(rows),
        "all": table(rows) | {k: _stats([(r["human"][k], r["judge"][k]) for r in rows]) for k in overall},
        "by_difficulty": {lv: table([r for r in rows if r["difficulty"] == lv]) for lv in ("easy", "medium", "hard")},
        "by_rater": {rt: table([r for r in rows if r["rater"] == rt]) for rt in sorted({r["rater"] for r in rows})},
        "positive_vs_negative_auc": {
            who: {k: _auc([r[who][k] for r in rows], [r["label"] for r in rows]) for k in CORRECTNESS + BEAUTY + overall}
            for who in ("human", "judge")},
        "caught": _caught(rows),
        "false_alarms": {k: sum(r["label"] == "positive" and r["human"][k] >= 9 and r["judge"][k] is not None
                                and r["judge"][k] < CAUGHT for r in rows) for k in CORRECTNESS},
    }
    suffix = ("" if wording == "v1" else f"_{wording}") + (f"_{start:03d}-{first:03d}" if (start, first) != (1, 100) and first else "")
    (HERE / f"summary{suffix}.json").write_text(json.dumps(summary, indent=1))
    (HERE / f"judge_scores{suffix}.json").write_text(json.dumps(
        {"model": "gpt-6-luna", "scale": "0-10", "diagrams": {r["id"]: {"correctness": r["judge_correctness"], "scores": {
            k: r["judge"][k] for k in CORRECTNESS + BEAUTY}} for r in rows}}, indent=1))

    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    print("| Dimension | n | Human mean | Judge mean | MAE | RMSE | Within 1 | Spearman | Kendall | Pearson | AUC human | AUC judge |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    auc = summary["positive_vs_negative_auc"]
    for k in CORRECTNESS + BEAUTY + overall:
        s = summary["all"][k]
        print(f"| {k} | {s['n']} | {fmt(s.get('human_mean'))} | {fmt(s.get('judge_mean'))} | {fmt(s.get('mae'))} | "
              f"{fmt(s.get('rmse'))} | {fmt(s.get('within_1'))} | {fmt(s.get('spearman'))} | {fmt(s.get('kendall'))} | {fmt(s.get('pearson'))} | {fmt(auc['human'][k])} | {fmt(auc['judge'][k])} |")
    for group in ("by_difficulty", "by_rater"):
        keys = list(summary[group])
        print(f"\nSpearman / MAE {group.replace('_', ' ')}\n")
        print("| Dimension | " + " | ".join(keys) + " |")
        print("|---|" + "---:|" * len(keys))
        for k in CORRECTNESS + BEAUTY:
            cells = [f"{fmt(summary[group][g][k].get('spearman'))} / {fmt(summary[group][g][k].get('mae'))}" for g in keys]
            print(f"| {k} | " + " | ".join(cells) + " |")
    print("\nInserted correctness errors caught (target dimension below 8)\n")
    print("| Error (dimension) | n | Human | Judge |\n|---|---:|---:|---:|")
    for key, c in summary["caught"].items():
        print(f"| {key} | {c['n']} | {c['human']} | {c['judge']} |")
    print("\nFalse alarms (positives humans gave 9+ but the judge below 8):", summary["false_alarms"])


# An error counts as caught when its dimension falls below 8. Humans give whole numbers, so that is 7 or lower; with
# the formula one detected error gives 7.5, so a 7-or-lower cut would never credit a single detection.
CAUGHT = 8


def _caught(rows):
    out = {}
    for r in rows:
        for e in r["defects"]:
            dim = e["criterion"].split(".")[0]
            if r["label"] != "negative" or dim not in CORRECTNESS or r["judge"][dim] is None:
                continue
            c = out.setdefault((e["defect"], dim), {"n": 0, "human": 0, "judge": 0})
            c["n"] += 1
            c["human"] += r["human"][dim] < CAUGHT
            c["judge"] += r["judge"][dim] < CAUGHT
    return {f"{defect} ({dim})": c for (defect, dim), c in sorted(out.items(), key=lambda kv: (kv[0][1], kv[0][0]))}


def tune(wordings: list[str]) -> None:
    """Correctness metrics of several wordings side by side (run compare for each first)."""
    sums = {w: json.load(open(HERE / f"summary{'' if w == 'v1' else '_' + w}.json")) for w in wordings}
    for metric in ("mae", "within_1", "spearman", "kendall"):
        print(f"\n{metric}\n\n| Dimension | " + " | ".join(wordings) + " |\n|---|" + "---:|" * len(wordings))
        for k in CORRECTNESS + ["correctness_overall"]:
            print(f"| {k} | " + " | ".join(f"{sums[w]['all'][k][metric]:.2f}" for w in wordings) + " |")
    print("\nAUC (positive vs negative)\n\n| Dimension | human | " + " | ".join(wordings) + " |\n|---|---:|" + "---:|" * len(wordings))
    for k in CORRECTNESS + ["correctness_overall"]:
        h = sums[wordings[0]]["positive_vs_negative_auc"]["human"][k]
        print(f"| {k} | {h:.2f} | " + " | ".join(f"{sums[w]['positive_vs_negative_auc']['judge'][k]:.2f}" for w in wordings) + " |")
    print("\nFalse alarms on positives:", {w: sums[w]["false_alarms"] for w in wordings})
    caught = {w: {k: v["judge"] for k, v in sums[w]["caught"].items()} for w in wordings}
    print("\nErrors caught (human / judge per wording)")
    for k, v in sums[wordings[0]]["caught"].items():
        print(f"  {k}: human {v['human']}/{v['n']}, judge " + ", ".join(f"{w} {caught[w].get(k, '-')}" for w in wordings))


# ---------- report ----------

VERSIONS = [  # (wording, method, what it is)
    ("v1", "Sketch + formula", "The evaluator as it was in git: a sketch from the caption, an item-by-item check "
                               "(3 samples, majority vote), and code computing the score."),
    ("v1r", "Sketch + formula", "v1 run again unchanged, to measure run-to-run noise."),
    ("v2", "Sketch + formula", "v1 plus four general sentences (shared connectors, judge text at display size, cut-off "
                               "labels are wrong, steps in the described order), from the v1 mistakes."),
    ("d1", "Direct", "No sketch, no formula: each dimension asked once with the image and caption; the model gives "
                     "1-10, with the scoring idea (what costs how much) in the prompt."),
    ("d1r", "Direct", "d1 run again unchanged, to measure run-to-run noise."),
    ("d2", "Direct", "d1 plus four general sentences, from the d1 mistakes: a problem that makes the diagram harder "
                     "to read, follow, or trust is a clear error; boxes touching the image edge are not cut off; judge "
                     "arrows against the caption, not your own idea of the process; do not recompute numbers."),
    ("d3", "Direct + sketch", "d2 plus the sketch: each dimension also gets the caption checklist and goes through it "
                              "before scoring."),
    ("d4", "Direct", "d2 with one change per dimension, from the d2 mistakes: a tight crop is not cut off (only sliced "
                     "text or an incomplete box is); an exclusion box in the main chain is fine; a label in the wrong "
                     "box is a Details problem; typos and abbreviations do not count; and the figure's display size "
                     "in the study (fitted to a ~660x360 panel) for Legibility."),
    ("d5", "Direct", "d4 keeping the Layout and exclusion-box sentences, dropping the wrong-box and typo sentences "
                     "(they excused real errors), and making small text at the display size cost less."),
    ("d6", "Direct", "d5 with a milder Legibility wording: small text that still reads at the display size is fine."),
    ("d7", "Direct", "The d6 ideas as short general rules with no sizes, plus a Presence rule: a blank, cut-off, or "
                     "hidden box does not show its component."),
    ("d8", "Direct", "Best part of each: d7 Presence, d6 Connectivity, d2 Details, a Layout rule naming a border that "
                     "lies on the edge, and Legibility told that figures are fitted to a screen, without numbers."),
    ("d9", "Direct", "d8 with Legibility judged as a person would, by the hardest text, naming weak contrast; and "
                     "Details treating a label in the wrong place as a small problem. Neither helped."),
    ("d10", "Direct", "d8 with a size sentence in words (no number) for figures shown much smaller, and a Details rule: "
                      "same thing in other words is accurate; a different thing, a contradiction, or a left-out item "
                      "is wrong."),
    ("d11", "Direct", "d8 with the first half of the d10 Details rule only (same thing in other words is accurate; a "
                      "different thing or a contradiction is wrong)."),
    ("d12", "Direct", "d11 with Legibility back to the d2 wording and Details also calling a label wrong when it drops a "
                      "count, value, or condition the caption gives."),
    ("d13", "Direct", "d12 with Connectivity checked connection by connection, plain lines and arrows to a group or panel "
                      "accepted."),
    ("d14", "Direct", "d13 plus: a missing, cut-off, or faint box breaks the connections through it."),
    ("d15", "Direct", "d12 with a Layout severity rule (one obvious layout error belongs in 3-5) and wrap-around flows "
                      "accepted."),
    ("d16", "Direct", "d12 with a full Connectivity section: a four-step check, what is not an error, and what each band "
                      "means for connections."),
    ("d17", "Direct", "d12 with Connectivity and Presence told the caption is a short summary rewritten from the figure: "
                      "flag only clear contradictions; a box beside an arrow is its label; a less detailed label still "
                      "counts as present. Tuned on HS001-HS200."),
    ("d18", "Direct", "d17 with the Connectivity rubric sentence reworded to the flow the caption describes, and Presence "
                      "told that a step the flow jumps over is missing."),
    ("d19", "Direct + sketch", "d17 + the v4 caption sketch as a checklist for Presence, Details, and Connectivity."),
]
CHOSEN = "d2"


def report() -> None:
    """report.html: the HS001-HS100 results of every version (recomputes the summaries first)."""
    import contextlib
    import html
    import io
    names = [w for w, _, _ in VERSIONS]
    for w in names:
        with contextlib.redirect_stdout(io.StringIO()):
            compare(100, w)
    sums = {w: json.load(open(HERE / f"summary{'' if w == 'v1' else '_' + w}.json")) for w in names}
    humans = [d for d in _humans() if int(d["id"][2:]) <= 100]
    count = lambda key: {v: sum(d[key] == v for d in humans) for v in sorted({d[key] for d in humans})}
    f2 = lambda v: "-" if v is None else f"{v:.2f}"

    # run-to-run noise: d1 vs its repeat d1r
    noise = {}
    for k in CORRECTNESS:
        a = [json.load(open(RESULTS / d["id"] / f"direct_d1_{k}.json"))["score"] for d in humans]
        b = [json.load(open(RESULTS / d["id"] / f"direct_d1r_{k}.json"))["score"] for d in humans]
        noise[k] = (sum(x == y for x, y in zip(a, b)), sum(abs(x - y) <= 1 for x, y in zip(a, b)))

    def best_row(values, higher=True):
        real = [v for v in values if v is not None]
        target = (max if higher else min)(real) if real else None
        return [f'<td class="num{" best" if v is not None and v == target else ""}">{f2(v)}</td>' for v in values]

    rows_overall = ""
    for label, key, higher in [("MAE (points)", "mae", False), ("Within 1 point", "within_1", True),
                               ("Spearman", "spearman", True), ("Kendall", "kendall", True), ("Pearson", "pearson", True)]:
        rows_overall += f"<tr><th>{label}</th>" + "".join(best_row([sums[w]["all"]["correctness_overall"][key] for w in names], higher)) + "</tr>"
    rows_overall += "<tr><th>Clean vs broken (AUC)</th>" + "".join(
        best_row([sums[w]["positive_vs_negative_auc"]["judge"]["correctness_overall"] for w in names])) + "</tr>"
    rows_overall += "<tr><th>False alarms on clean diagrams</th>" + "".join(
        best_row([sum(sums[w]["false_alarms"].values()) for w in names], False)).replace(".00<", "<") + "</tr>"

    rows_spearman = "".join(f"<tr><th>{k.capitalize()}</th>" + "".join(best_row([sums[w]["all"][k]["spearman"] for w in names])) + "</tr>"
                            for k in CORRECTNESS)
    rows_kendall = "".join(f"<tr><th>{k.capitalize()}</th>" + "".join(best_row([sums[w]["all"][k]["kendall"] for w in names])) + "</tr>"
                           for k in CORRECTNESS)
    rows_mae = "".join(f"<tr><th>{k.capitalize()}</th>" + "".join(best_row([sums[w]["all"][k]["mae"] for w in names], False)) + "</tr>"
                       for k in CORRECTNESS)
    rows_auc = "".join(f"<tr><th>{k.capitalize()}</th><td class='num human'>{f2(sums['v1']['positive_vs_negative_auc']['human'][k])}</td>"
                       + "".join(best_row([sums[w]["positive_vs_negative_auc"]["judge"][k] for w in names])) + "</tr>" for k in CORRECTNESS)
    rows_alarm = "".join(f"<tr><th>{k.capitalize()}</th>" + "".join(best_row([sums[w]["false_alarms"][k] for w in names], False)).replace(".00<", "<") + "</tr>"
                         for k in CORRECTNESS)
    rows_caught = ""
    for key, c in sums["v1"]["caught"].items():
        defect, dim = key.split(" (")
        rows_caught += (f"<tr><th>{defect.replace('_', ' ')}</th><td>{dim.rstrip(')')}</td><td class='num'>{c['n']}</td>"
                        f"<td class='num human'>{c['human']}</td>" + "".join(f"<td class='num'>{sums[w]['caught'][key]['judge']}</td>" for w in names) + "</tr>")
    rows_noise = "".join(f"<tr><th>{k.capitalize()}</th><td class='num'>{noise[k][0]}</td><td class='num'>{noise[k][1]}</td></tr>" for k in CORRECTNESS)
    rows_beauty = "".join(
        f"<tr><th>{k.replace('_', ' ').capitalize()}</th><td class='num'>{f2(s['human_mean'])}</td><td class='num'>{f2(s['judge_mean'])}</td>"
        f"<td class='num'>{f2(s['mae'])}</td><td class='num'>{f2(s['spearman'])}</td><td class='num'>{f2(s['kendall'])}</td>"
        f"<td class='num human'>{f2(sums['v1']['positive_vs_negative_auc']['human'][k])}</td><td class='num'>{f2(sums['v1']['positive_vs_negative_auc']['judge'][k])}</td></tr>"
        for k in BEAUTY + ["beauty_overall"] for s in [sums["v1"]["all"][k]])
    rows_versions = "".join(f"<tr><th>{w}{' ★' if w == CHOSEN else ''}</th><td>{m}</td><td>{html.escape(t)}</td></tr>" for w, m, t in VERSIONS)
    head = "".join(f"<th class='num{' chosen' if w == CHOSEN else ''}'>{w}</th>" for w in names)

    v1, ch = sums["v1"]["all"]["correctness_overall"], sums[CHOSEN]["all"]["correctness_overall"]
    auc = lambda w: sums[w]["positive_vs_negative_auc"]["judge"]["correctness_overall"]
    tiles = [("Spearman with humans", f2(v1["spearman"]), f2(ch["spearman"]), "overall correctness, higher is better"),
             ("Mean absolute error", f2(v1["mae"]), f2(ch["mae"]), "points on 0-10, lower is better"),
             ("Clean vs broken (AUC)", f2(auc("v1")), f2(auc(CHOSEN)),
              f"humans {f2(sums['v1']['positive_vs_negative_auc']['human']['correctness_overall'])}"),
             ("False alarms on clean diagrams", str(sum(sums["v1"]["false_alarms"].values())), str(sum(sums[CHOSEN]["false_alarms"].values())),
              "humans 9+, judge below 8, summed over 5 dimensions")]
    tiles_html = "".join(f"<div class='tile'><div class='tl'>{t}</div><div class='tv'><span class='old'>{a}</span> → <b>{b}</b></div>"
                         f"<div class='tn'>{n}</div></div>" for t, a, b, n in tiles)
    diff, lab = count("difficulty"), count("label")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Human Alignment Results</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6a66;--line:#e4e2dc;--card:#ffffff;--accent:#2f6f5e;--accent-bg:#e3f0eb;--human:#8a5a12;--human-bg:#f6ecdc}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#161615;--fg:#ecebe7;--muted:#a19f99;--line:#33322f;--card:#1f1f1d;--accent:#7cc4ad;--accent-bg:#1f3a32;--human:#e0b46e;--human-bg:#3a2e1b}}}}
:root[data-theme="dark"]{{--bg:#161615;--fg:#ecebe7;--muted:#a19f99;--line:#33322f;--card:#1f1f1d;--accent:#7cc4ad;--accent-bg:#1f3a32;--human:#e0b46e;--human-bg:#3a2e1b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:32px 16px 64px}}h1{{font-size:26px;margin:0 0 4px}}h2{{font-size:18px;margin:36px 0 8px}}
.sub{{color:var(--muted);margin:0 0 20px}}p,li{{max-width:780px}}.note{{color:var(--muted);font-size:13px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:20px 0}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}.tl{{color:var(--muted);font-size:13px}}
.tv{{font-size:24px;margin:4px 0}}.tv .old{{color:var(--muted)}}.tv b{{color:var(--accent)}}.tn{{color:var(--muted);font-size:12px}}
.wrap{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
thead th{{color:var(--muted);font-weight:600;font-size:12px;white-space:nowrap}}tbody tr:last-child th,tbody tr:last-child td{{border-bottom:0}}
tbody th{{font-weight:600;white-space:nowrap}}.num{{text-align:right;font-variant-numeric:tabular-nums}}
td.best{{color:var(--accent);font-weight:700;background:var(--accent-bg)}}th.chosen{{color:var(--accent)}}td.human{{color:var(--human);background:var(--human-bg)}}
</style></head><body><main>
<h1>Human alignment: correctness on HS001–HS100</h1>
<p class="sub">The prompt-tuning set of the human study · judge gpt-6-luna · every score on 0–10 · HS101–HS300 are held out and not used here</p>
<div class="tiles">{tiles_html}</div>
<p>The chosen version, <b>{CHOSEN}</b>, scores each correctness dimension directly: the model sees the image and caption and gives
1–10, with the scoring idea written into the prompt. It replaced the sketch-and-formula method (v1). Highlighted cells are the best
value in each row; brown cells are the human reference.</p>

<h2>Setup</h2>
<ul>
<li>100 diagrams, HS001–HS100: {diff.get('easy', 0)} easy, {diff.get('medium', 0)} medium, {diff.get('hard', 0)} hard; {lab.get('positive', 0)} clean and {lab.get('negative', 0)} with inserted errors.</li>
<li>Human ratings on 1–10. Judge scores on the formula's 0–1 scale are multiplied by 10.</li>
<li>Overall correctness: humans and the direct versions use the mean of the five dimensions; the sketch + formula versions use the pipeline's own overall score.</li>
<li>An error counts as caught when its dimension falls below 8 (with the formula, one detected error gives 7.5). A false alarm is a clean diagram that humans rated 9 or 10 but the judge scored below 8.</li>
</ul>

<h2>Versions tried</h2>
<div class="wrap"><table><thead><tr><th>Version</th><th>Method</th><th>What it is</th></tr></thead><tbody>{rows_versions}</tbody></table></div>

<h2>Overall correctness</h2>
<div class="wrap"><table><thead><tr><th></th>{head}</tr></thead><tbody>{rows_overall}</tbody></table></div>

<h2>Per dimension: Spearman with humans</h2>
<div class="wrap"><table><thead><tr><th></th>{head}</tr></thead><tbody>{rows_spearman}</tbody></table></div>
<h2>Per dimension: Kendall with humans</h2>
<div class="wrap"><table><thead><tr><th></th>{head}</tr></thead><tbody>{rows_kendall}</tbody></table></div>
<h2>Per dimension: mean absolute error (points)</h2>
<div class="wrap"><table><thead><tr><th></th>{head}</tr></thead><tbody>{rows_mae}</tbody></table></div>
<h2>Per dimension: clean vs broken (AUC)</h2>
<p class="note">How often a clean diagram scores above one with inserted errors (0.5 = no separation, 1 = perfect).</p>
<div class="wrap"><table><thead><tr><th></th><th class="num">humans</th>{head}</tr></thead><tbody>{rows_auc}</tbody></table></div>

<h2>False alarms on clean diagrams</h2>
<div class="wrap"><table><thead><tr><th></th>{head}</tr></thead><tbody>{rows_alarm}</tbody></table></div>

<h2>Inserted errors caught</h2>
<p class="note">Counts of diagrams whose target dimension fell below 8. Few diagrams per error type, so single counts move with noise.</p>
<div class="wrap"><table><thead><tr><th>Error</th><th>Dimension</th><th class="num">n</th><th class="num">humans</th>{head}</tr></thead><tbody>{rows_caught}</tbody></table></div>

<h2>Run-to-run noise (d1 vs its repeat d1r)</h2>
<p class="note">Of 100 diagrams, how many got the same score, and a score within 1 point, in two identical runs. Overall agreement with
humans did not change between the runs (Spearman {f2(sums['d1']['all']['correctness_overall']['spearman'])} and {f2(sums['d1r']['all']['correctness_overall']['spearman'])}),
and averaging the two runs gave only 0.75, so one call per dimension is enough.</p>
<div class="wrap"><table><thead><tr><th></th><th class="num">same score</th><th class="num">within 1</th></tr></thead><tbody>{rows_noise}</tbody></table></div>

<h2>Beauty v6 on the same 100 (not tuned)</h2>
<div class="wrap"><table><thead><tr><th></th><th class="num">human mean</th><th class="num">judge mean</th><th class="num">MAE</th><th class="num">Spearman</th><th class="num">Kendall</th><th class="num">AUC humans</th><th class="num">AUC judge</th></tr></thead><tbody>{rows_beauty}</tbody></table></div>

<h2>Conclusions</h2>
<ul>
<li><b>Direct scoring beats sketch + formula on every dimension.</b> Overall Spearman {f2(v1['spearman'])} → {f2(ch['spearman'])}, MAE {f2(v1['mae'])} → {f2(ch['mae'])}, with less than half the input tokens and a fifth of the output tokens.</li>
<li><b>The formula, not the model, was the main problem.</b> v1 detected most inserted errors, but one detected error usually cost 2.5 points (score 7.5), while humans often drop a dimension to 4–6 for a clear error; and clean real figures lost points because the caption sketch did not match how they were drawn (exclusion boxes chained into the main flow, one arrow serving several boxes).</li>
<li><b>d2's sentences helped beyond noise</b> on Details and Connectivity, the two dimensions they targeted.</li>
<li><b>Adding the sketch back (d3) did not help:</b> overall the same as d2, Connectivity worse (more false alarms), at about 1.6 times the cost.</li>
<li><b>d4–d6, from the d2 mistakes:</b> best so far is d6 (overall Spearman {f2(sums['d6']['all']['correctness_overall']['spearman'])}, MAE {f2(sums['d6']['all']['correctness_overall']['mae'])}). Layout gained most: d2 still called tightly cropped figures cut off, and saying that only sliced text or an incomplete box is cut off lifted Layout Spearman {f2(sums['d2']['all']['layout']['spearman'])} → {f2(sums['d6']['all']['layout']['spearman'])} while still catching every inserted crop.</li>
<li><b>The study's display size explains much of the Legibility gap.</b> Figures were shown fitted to a panel about 660×360 pixels, so large figures looked far smaller to raters than to the model. Telling the model the scale ranks Legibility better but overshoots the level (bias {sums['d2']['all']['legibility']['bias']:+.2f} in d2, {sums['d6']['all']['legibility']['bias']:+.2f} in d6).</li>
<li><b>d7–d8, general rules without sizes:</b> d8 is the best general prompt (overall Spearman {f2(sums['d8']['all']['correctness_overall']['spearman'])}, MAE {f2(sums['d8']['all']['correctness_overall']['mae'])}, false alarms on clean diagrams {sum(sums['d8']['false_alarms'].values())} vs {sum(sums['d2']['false_alarms'].values())} for d2). Presence gained most from taking "visibly present" literally (Spearman {f2(sums['d6']['all']['presence']['spearman'])} → {f2(sums['d8']['all']['presence']['spearman'])}). Legibility wordings without numbers score like d2 at every display size: the judge sees the full-resolution file and a prompt alone does not close that gap.</li>
<li><b>Sentences that excuse a kind of error backfire:</b> in d4, "typos do not count" made the judge pass real wrong labels and "a label in the wrong box is a Details problem" excused real arrow errors; d5 dropped both.</li>
</ul>
<h2>Caveats</h2>
<ul>
<li>The prompt was tuned on these 100 diagrams; the claim has to be confirmed once on HS101–HS300.</li>
<li>Humans often lowered every dimension of a diagram that had any error, even dimensions with no inserted error; the judge is meant to score each dimension on its own, which caps the agreement it can reach.</li>
</ul>
</main></body></html>
"""
    (HERE / "report.html").write_text(page)
    print(f"wrote {HERE / 'report.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--only", help="comma-separated diagram ids (for a quick trial)")
    for p in (s, c := sub.add_parser("compare")):
        p.add_argument("--first", type=int, help="only HS001..HSnnn")
        p.add_argument("--start", type=int, default=1, help="only HSnnn.. (with --first: a range such as 101-200)")
        p.add_argument("--wording", default="v1", choices=list(WORDINGS) + list(DIRECT))
    t = sub.add_parser("tune")
    t.add_argument("--wordings", default="v1,v1r,v2")
    sub.add_parser("report")
    k = sub.add_parser("sketches")  # stage-1 sketches of a wording, checked against the unmodified figures
    k.add_argument("--wording", default="v4", choices=list(WORDINGS))
    k.add_argument("--workers", type=int, default=12)
    k.add_argument("--first", type=int)
    k.add_argument("--start", type=int, default=1)
    args = parser.parse_args()
    if args.cmd == "sketches":
        sketches(args.wording, args.workers, args.first, args.start)
    elif args.cmd == "score":
        score(args.workers, set(args.only.split(",")) if args.only else None, args.first, args.wording, args.start)
    elif args.cmd == "compare":
        compare(args.first, args.wording, args.start)
    elif args.cmd == "tune":
        tune(args.wordings.split(","))
    else:
        report()
