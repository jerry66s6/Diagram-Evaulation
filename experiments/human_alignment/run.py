"""Human alignment: score the 300 human-study diagrams with the evaluator as it is in git
(correctness pipeline + Beauty v6, gpt-6-luna) and compare with the human ratings.

    python3 run.py score      # resumable; raw outputs in results/<id>/
    python3 run.py compare --first 100   # HS001-HS100 (prompt-tuning set); summary.json, judge_scores.json
    python3 run.py score --first 100 --wording v2     # correctness only, with the v2 prompt wording
    python3 run.py compare --first 100 --wording v2   # summary_v2.json
    python3 run.py tune --wordings v1,v1r,v2          # side by side, from the summaries

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


def _direct_prompt(dimension, caption, wording="d1"):
    change = DIRECT[wording]
    criteria = [c["description"] for c in json.load(open(ROOT / "config" / "rubric.json"))["criteria"]
                if c["dimension"] == dimension]
    name = dimension.upper()
    return (
        f"You are rating one aspect of a flowchart: {name}. The attached image is the flowchart, and the caption "
        f"below describes what it should show.\n\n{name}: {DIRECT_QUESTION[dimension]}\nIt covers: "
        + " ".join(criteria) + (" " + DIRECT_CONVENTIONS[dimension] if DIRECT_CONVENTIONS[dimension] else "")
        + change.get("conventions", {}).get(dimension, "")
        + f"\n\nEach aspect has its own rater:\n{DIRECT_OWNERS}\nJudge only {name}.\n\n"
        + DIRECT_SCORING + change.get("scoring", "") + "\n\n"
        "First list the problems you see on this dimension, each with where it is (an empty list if there are "
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


def score(workers: int, only: set[str] | None = None, first: int | None = None, wording: str = "v1") -> None:
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

    def direct(d, dimension):
        payload = beauty_backend.structured(_direct_prompt(dimension, d["caption"], wording), [ROOT / d["image"]],
                                            DIRECT_SCHEMA, f"diagram_correctness_{dimension}")
        if not isinstance(payload.get("score"), int) or not 1 <= payload["score"] <= 10:
            raise RuntimeError(f"invalid score for {dimension}: {payload.get('score')!r}")
        return payload

    jobs = []
    for d in _humans():
        if wording in DIRECT:
            if (not only or d["id"] in only) and (not first or int(d["id"][2:]) <= first):
                (RESULTS / d["id"]).mkdir(parents=True, exist_ok=True)
                jobs += [(d, f"direct_{wording}_{k}", lambda d=d, k=k: direct(d, k)) for k in CORRECTNESS]
            continue
        if (only and d["id"] not in only) or (first and int(d["id"][2:]) > first):
            continue
        (RESULTS / d["id"]).mkdir(parents=True, exist_ok=True)
        jobs.append((d, _correctness_file(wording)[:-5], lambda d=d: correctness(d)))
        if wording != "v1":  # Beauty is not part of the correctness prompt; it is scored once
            continue
        for dim in (*BEAUTY_V6_MODEL_DIMENSIONS, "canvas_fill"):
            jobs.append((d, f"beauty_{dim}", lambda d=d, dim=dim: beauty(d, dim)))
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


def compare(first: int | None = None, wording: str = "v1") -> None:
    humans = [d for d in _humans() if first is None or int(d["id"][2:]) <= first]
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
    suffix = "" if wording == "v1" else f"_{wording}"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--only", help="comma-separated diagram ids (for a quick trial)")
    for p in (s, c := sub.add_parser("compare")):
        p.add_argument("--first", type=int, help="only HS001..HSnnn")
        p.add_argument("--wording", default="v1", choices=list(WORDINGS) + list(DIRECT))
    t = sub.add_parser("tune")
    t.add_argument("--wordings", default="v1,v1r,v2")
    args = parser.parse_args()
    if args.cmd == "score":
        score(args.workers, set(args.only.split(",")) if args.only else None, args.first, args.wording)
    elif args.cmd == "compare":
        compare(args.first, args.wording)
    else:
        tune(args.wordings.split(","))
