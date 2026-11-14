"""Beauty prompt tuning against the human ratings, for the four dimensions that agree least with humans:
Centering, Space Use, Alignment, Canvas Fill.

    python3 experiments/beauty_tuning/run.py score --versions current,rater --start 1 --end 200   # gpt-6-luna; resumable
    python3 experiments/beauty_tuning/run.py compare --versions current,rater --start 101 --end 200
    python3 experiments/beauty_tuning/run.py test --choice centering=rater,space_use=current,...    # HS201-HS300, once

Splits: HS001-HS100 training (look at disagreements), HS101-HS200 validation (choose), HS201-HS300 test (scored once
with the chosen prompts). For each dimension the version with the higher validation Spearman is chosen.

Versions:
- current: the prompts in src/diagram_correctness/beauty.py (0-100, divided by 10). Its HS201-HS300 scores are the
  pipeline's (../pipeline/results), made with the same prompts.
- rater: the question, low anchor, and high anchor the human raters saw in the study (user-study/lib/metrics.ts),
  scored 1-10 like the raters did, with no isolation rules.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
PIPELINE = ROOT / "experiments" / "pipeline" / "results"
sys.path.insert(0, str(ROOT / "src"))

from diagram_correctness.beauty import judge_beauty_dimension  # noqa: E402

MODEL = "gpt-6-luna"
DIMENSIONS = ["centering", "space_use", "alignment", "canvas_fill"]

# What the raters saw for each dimension (user-study/lib/metrics.ts): name, question, low anchor, high anchor.
RATER = {
    "centering": ("Centering", "Is the diagram centered on the canvas, each label centered in its shape, and neighboring elements evenly spaced?",
                  "Off-center or unevenly spaced", "Centered, with even gaps"),
    "space_use": ("Space use", "Are elements given room, rather than packed tightly while usable canvas space is left empty?",
                  "Crammed together while space is left empty", "Space used comfortably"),
    "alignment": ("Alignment", "Is the geometry deliberate: elements lined up or in a clear structure (such as a tree or ring), nothing slightly "
                  "tilted, and arrows in one clean, consistent style?", "Misaligned, tilted, or messy arrows", "Neatly aligned, with clean, consistent arrows"),
    "canvas_fill": ("Canvas fill", "Do the diagram elements fill a reasonable part of the canvas, rather than being tiny in a mostly empty image?",
                    "Tiny elements in a mostly empty canvas", "Elements fill the canvas well"),
}
RATER_SCHEMA = {
    "type": "object",
    "properties": {"observation": {"type": "string"}, "score": {"type": "integer"}, "reasoning": {"type": "string"}},
    "required": ["observation", "score", "reasoning"],
    "additionalProperties": False,
}


def rater_prompt(dimension):
    name, question, low, high = RATER[dimension]
    return (
        f"You are a participant in a user study rating flowchart images. Each flowchart is shown on a screen, and you rate several "
        f"aspects of how it looks from 1 (very poor) to 10 (excellent). This question is about {name.upper()}.\n\n"
        f"{name}: {question}\n1 means: {low}.\n10 means: {high}.\n\n"
        "Look at the whole image as a reader would see it on a screen. First describe in one or two sentences what you see "
        "that matters for this question, then give an integer score from 1 to 10, then a one-sentence reason."
    )


def humans():
    return {d["id"]: d for d in json.load(open(ROOT / "experiments" / "human_alignment" / "human_ratings.json"))["diagrams"]}


def ids(start, end):
    return [f"HS{i:03d}" for i in range(start, end + 1)]


def path(version, i, dimension):
    if version == "current" and int(i[2:]) >= 201:
        return PIPELINE / i / f"{dimension}.json"
    return RESULTS / i / f"{version}_{dimension}.json"


def judge_score(version, i, dimension):
    """On 1-10, like the human ratings."""
    s = json.load(open(path(version, i, dimension)))["score"]
    return s / 10 if version == "current" else s


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def score(versions, start, end, workers, dimensions=DIMENSIONS):
    load_env()
    from diagram_correctness.vlm import OpenAIBackend
    if os.environ.get("OPENAI_MODEL", MODEL) != MODEL:
        raise SystemExit(f"this experiment runs on {MODEL} only")
    backend = OpenAIBackend(MODEL)
    H = humans()

    def judge(version, i, dimension):
        image = ROOT / H[i]["image"]
        if version == "current":
            result = judge_beauty_dimension(backend, image, dimension).to_dict()
        else:
            result = backend.structured(rater_prompt(dimension), [image], RATER_SCHEMA, f"rater_{dimension}")
            if not isinstance(result.get("score"), int) or not 1 <= result["score"] <= 10:
                raise RuntimeError(f"score out of range: {result.get('score')!r}")
        out = path(version, i, dimension)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1))

    jobs = [(v, i, k) for v in versions for i in ids(start, end) for k in dimensions if not path(v, i, k).exists()]
    print(f"scoring with {MODEL}: {len(jobs)} calls")
    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(judge, *job): job for job in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:  # retried on the next run
                failed += 1
                print(f"failed {futures[future]}: {error}")
            if n % 200 == 0:
                print(f"{n}/{len(jobs)} done")
    print(f"{len(jobs) - failed}/{len(jobs)} done, {failed} failed")
    print(f"api usage this run: {backend.usage}")


def agreement(version, start, end, dimension):
    H = humans()
    sel = ids(start, end)
    h = np.array([H[i]["scores"][dimension] for i in sel], float)
    j = np.array([judge_score(version, i, dimension) for i in sel], float)
    clean = np.array([H[i]["label"] == "positive" for i in sel])
    constant = lambda a, b: np.std(a) == 0 or np.std(b) == 0
    return {"spearman": float(spearmanr(h, j).statistic) if not constant(h, j) else None,
            "pearson": float(pearsonr(h, j).statistic) if not constant(h, j) else None,
            "mae": float(np.abs(j - h).mean()), "bias": float((j - h).mean()),
            "spearman_unmodified": float(spearmanr(h[clean], j[clean]).statistic) if not constant(h[clean], j[clean]) else None,
            "judge_sd": float(j.std()), "human_sd": float(h.std())}


def fmt(v, signed=False):
    return "n/a" if v is None else (f"{v:+.2f}" if signed else f"{v:.2f}")


def compare(versions, start, end, dimensions=DIMENSIONS):
    out = {}
    print(f"HS{start:03d}-HS{end:03d}\n\n| Dimension | Version | Spearman | Pearson | MAE | Bias | Spearman, unmodified only | Judge SD (human SD) |\n|---|---|---:|---:|---:|---:|---:|---:|")
    for k in dimensions:
        out[k] = {}
        for v in versions:
            a = agreement(v, start, end, k)
            out[k][v] = a
            print(f"| {k} | {v} | {fmt(a['spearman'])} | {fmt(a['pearson'])} | {a['mae']:.2f} | {fmt(a['bias'], True)} | {fmt(a['spearman_unmodified'])} | {a['judge_sd']:.2f} ({a['human_sd']:.2f}) |")
    return out


def test(choice):
    """Score HS201-HS300 with the chosen version of each dimension, then compare with current there."""
    new = sorted({v for v in choice.values() if v != "current"})
    for v in new:
        score([v], 201, 300, 12, [k for k, c in choice.items() if c == v])
    result = {"choice": choice, "validation": compare(["current", *new], 101, 200), "test": compare(["current", *new], 201, 300)}
    json.dump(result, open(HERE / "results.json", "w"), indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("score", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--versions", default="current,rater")
        p.add_argument("--start", type=int, default=1)
        p.add_argument("--end", type=int, default=200)
        p.add_argument("--dimensions", default=",".join(DIMENSIONS))
    sub.choices["score"].add_argument("--workers", type=int, default=12)
    t = sub.add_parser("test")
    t.add_argument("--choice", required=True, help="dimension=version,... for all four dimensions")
    args = parser.parse_args()
    if args.cmd == "score":
        if args.end > 200:
            raise SystemExit("HS201-HS300 is the test set: use `test`")
        score(args.versions.split(","), args.start, args.end, args.workers, args.dimensions.split(","))
    elif args.cmd == "compare":
        if args.end > 200:
            raise SystemExit("HS201-HS300 is the test set: use `test`")
        compare(args.versions.split(","), args.start, args.end, args.dimensions.split(","))
    else:
        test(dict(x.split("=") for x in args.choice.split(",")))
