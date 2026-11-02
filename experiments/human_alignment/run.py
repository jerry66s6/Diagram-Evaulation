"""Human alignment: score the 300 human-study diagrams with the evaluator as it is in git
(correctness pipeline + Beauty v6, gpt-6-luna) and compare with the human ratings.

    python3 run.py score      # resumable; raw outputs in results/<id>/
    python3 run.py compare    # judge_scores.json, summary.json, tables printed as markdown

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

from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = HERE / "results"
CORRECTNESS = ["layout", "connectivity", "presence", "details", "legibility"]
BEAUTY = ["centering", "space_use", "color_consistency", "contrast", "alignment", "style_uniformity", "resolution", "canvas_fill"]


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

def _pipeline(model):
    from diagram_correctness.models import PresencePolicy
    from diagram_correctness.pipeline import CorrectnessPipeline, PipelineConfig
    from diagram_correctness.rubric import load_config, load_rubric
    from diagram_correctness.vlm import ExpectationExtractor, OpenAIBackend, VLMJudge
    settings = load_config(ROOT / "config" / "evaluator.json")
    policy = PresencePolicy.from_dict(settings.get("presence_policy"))
    judge_backend, expectation_backend = OpenAIBackend(model), OpenAIBackend(model)
    pipeline = CorrectnessPipeline(PipelineConfig(
        criteria=load_rubric(ROOT / "config" / "rubric.json"),
        judges=[VLMJudge(judge_backend, policy, samples=int(settings.get("judge_samples", 1)))],
        expectation_extractor=ExpectationExtractor(expectation_backend),
        weighting=settings.get("weighting", "auto"),
        dimension_reducer=settings.get("dimension_reducer", "mean"),
        error_factor=settings.get("error_factor"),
        severity_multipliers=settings.get("severity_multipliers"),
        fallback_frequencies=settings.get("fallback_frequencies"),
    ))
    return pipeline, [judge_backend, expectation_backend]


def score(workers: int, only: set[str] | None = None) -> None:
    _load_env()
    from diagram_correctness.beauty import judge_beauty_dimension
    from diagram_correctness.beauty_v6 import BEAUTY_V6_MODEL_DIMENSIONS, BEAUTY_V6_PROMPTS
    from diagram_correctness.canvas_fill import judge_canvas_fill
    from diagram_correctness.vlm import OpenAIBackend
    model = os.environ.get("OPENAI_MODEL") or "gpt-6-luna"
    if model != "gpt-6-luna":
        raise SystemExit(f"model is {model!r}; this experiment runs on gpt-6-luna only")
    pipeline, backends = _pipeline(model)
    beauty_backend = OpenAIBackend(model)
    backends.append(beauty_backend)
    print(f"scoring with {model}", flush=True)

    def correctness(d):
        folder = RESULTS / d["id"]
        path = folder / "inventory.json"  # the expectation sketch from the caption, made once
        if not path.exists():
            inv = pipeline.config.expectation_extractor.extract(d["caption"], pipeline.config.criteria)
            path.write_text(json.dumps(inv, indent=1))
        report = pipeline.run(description=d["caption"], candidate_image=ROOT / d["image"],
                              inventory=json.loads(path.read_text()), run_deterministic=False).to_dict()
        return {"correctness": report["correctness"],
                "dimensions": {k: v["score"] for k, v in report["dimensions"].items()},
                "criteria": {k: v["criteria"] for k, v in report["dimensions"].items()}}

    def beauty(d, dimension):
        if dimension == "canvas_fill":  # computed in Python, no model call
            result = judge_canvas_fill(ROOT / d["image"])
        else:
            result = judge_beauty_dimension(beauty_backend, ROOT / d["image"], dimension, BEAUTY_V6_PROMPTS)
        return {"score": result.score, "observation": result.observation, "reasoning": result.reasoning}

    jobs = []
    for d in _humans():
        if only and d["id"] not in only:
            continue
        (RESULTS / d["id"]).mkdir(parents=True, exist_ok=True)
        jobs.append((d, "correctness", lambda d=d: correctness(d)))
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

def _judge(diagram_id):
    folder = RESULTS / diagram_id
    if not (folder / "correctness.json").exists():
        return None
    c = json.load(open(folder / "correctness.json"))  # everything on 0-10: correctness is 0-1, beauty 0-100
    scores = {k: None if c["dimensions"].get(k) is None else 10 * c["dimensions"][k] for k in CORRECTNESS}
    for dim in BEAUTY:
        path = folder / f"beauty_{dim}.json"
        scores[dim] = json.load(open(path))["score"] / 10 if path.exists() else None
    return {"correctness": 10 * c["correctness"], "scores": scores}


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
    out["pearson"] = None if constant else float(pearsonr(h, j).statistic)
    return out


def _auc(scores, labels):
    """How often a positive scores above a negative (0.5 = no separation, 1 = perfect)."""
    pos = [s for s, l in zip(scores, labels) if l == "positive" and s is not None]
    neg = [s for s, l in zip(scores, labels) if l == "negative" and s is not None]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg)) if pos and neg else None


def compare() -> None:
    humans = _humans()
    rows = []
    for d in humans:
        judge = _judge(d["id"])
        if judge is None:
            continue
        human = dict(d["scores"])  # as rated, 1-10
        rows.append({"id": d["id"], "difficulty": d["difficulty"], "label": d["label"], "rater": d["rater"],
                     "human": human, "judge": judge["scores"], "judge_correctness": judge["correctness"]})
    print(f"{len(rows)} of {len(humans)} diagrams scored by the judge\n")

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
    overall = ["correctness_overall", "beauty_overall"]

    summary = {
        "n": len(rows),
        "all": table(rows) | {k: _stats([(r["human"][k], r["judge"][k]) for r in rows]) for k in overall},
        "by_difficulty": {lv: table([r for r in rows if r["difficulty"] == lv]) for lv in ("easy", "medium", "hard")},
        "by_rater": {rt: table([r for r in rows if r["rater"] == rt]) for rt in sorted({r["rater"] for r in rows})},
        "positive_vs_negative_auc": {
            who: {k: _auc([r[who][k] for r in rows], [r["label"] for r in rows]) for k in CORRECTNESS + BEAUTY + overall}
            for who in ("human", "judge")},
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1))
    (HERE / "judge_scores.json").write_text(json.dumps(
        {"model": "gpt-6-luna", "scale": "0-10", "diagrams": {r["id"]: {"correctness": r["judge_correctness"], "scores": {
            k: r["judge"][k] for k in CORRECTNESS + BEAUTY}} for r in rows}}, indent=1))

    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    print("| Dimension | n | Human mean | Judge mean | MAE | RMSE | Spearman | Pearson | AUC human | AUC judge |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    auc = summary["positive_vs_negative_auc"]
    for k in CORRECTNESS + BEAUTY + overall:
        s = summary["all"][k]
        print(f"| {k} | {s['n']} | {fmt(s.get('human_mean'))} | {fmt(s.get('judge_mean'))} | {fmt(s.get('mae'))} | "
              f"{fmt(s.get('rmse'))} | {fmt(s.get('spearman'))} | {fmt(s.get('pearson'))} | {fmt(auc['human'][k])} | {fmt(auc['judge'][k])} |")
    for group in ("by_difficulty", "by_rater"):
        keys = list(summary[group])
        print(f"\nSpearman / MAE {group.replace('_', ' ')}\n")
        print("| Dimension | " + " | ".join(keys) + " |")
        print("|---|" + "---:|" * len(keys))
        for k in CORRECTNESS + BEAUTY:
            cells = [f"{fmt(summary[group][g][k].get('spearman'))} / {fmt(summary[group][g][k].get('mae'))}" for g in keys]
            print(f"| {k} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--only", help="comma-separated diagram ids (for a quick trial)")
    sub.add_parser("compare")
    args = parser.parse_args()
    if args.cmd == "score":
        score(args.workers, set(args.only.split(",")) if args.only else None)
    else:
        compare()
