"""VLM pipeline on HS201-HS300: the two judges in src score the 13 dimensions, gradient boosting calls each diagram
good or bad, and the call is compared with the true label (positive/negative, not the human scores).

    python3 experiments/pipeline/run.py score                      # gpt-6-luna; resumable; raw outputs in results/<id>/
    python3 experiments/pipeline/run.py score --only HS201,HS202   # a quick trial
    python3 experiments/pipeline/run.py classify                   # results.json, predictions.csv
    python3 experiments/pipeline/run.py classify --beauty-tuned    # the same with the four tuned Beauty prompts

Judges: `judge_correctness_dimension` (src/diagram_correctness/direct.py) for the 5 Correctness dimensions and
`judge_beauty_dimension` (src/diagram_correctness/beauty.py) for the 8 Beauty dimensions, one call per dimension.
Beauty scores (0-100) are divided by 10 to match the 1-10 scale the classifier was trained on.

Classifier: the gradient boosting model from ../classification, unchanged. It is trained on the human scores of
HS001-HS200, with its settings and good/bad cut chosen there on val. Here it gets the judges' scores instead.
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
sys.path.insert(0, str(ROOT / "src"))

from diagram_correctness.beauty import BEAUTY_DIMENSIONS, judge_beauty_dimension  # noqa: E402
from diagram_correctness.direct import CORRECTNESS_DIMENSIONS, judge_correctness_dimension  # noqa: E402

DIMENSIONS = [*CORRECTNESS_DIMENSIONS, *BEAUTY_DIMENSIONS]
MODEL = "gpt-6-luna"


def classification():
    """../classification/run.py, loaded under its own name (this file is also run.py)."""
    spec = importlib.util.spec_from_file_location("classification_run", HERE.parent / "classification" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rows():
    rows = json.load(open(ROOT / "experiments" / "human_alignment" / "human_ratings.json"))["diagrams"]
    return [r for r in rows if int(r["id"][2:]) >= 201]


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def score(workers, only):
    load_env()
    from diagram_correctness.rubric import load_rubric
    from diagram_correctness.vlm import OpenAIBackend
    model = json.load(open(ROOT / "config" / "evaluator.json"))["models"]["judge"]
    if model != MODEL or os.environ.get("OPENAI_MODEL", MODEL) != MODEL:
        raise SystemExit(f"this pipeline runs on {MODEL} only")
    backend = OpenAIBackend(MODEL)
    criteria = load_rubric(ROOT / "config" / "rubric.json")

    def judge(d, dimension):
        image = ROOT / d["image"]
        if dimension in CORRECTNESS_DIMENSIONS:
            result = judge_correctness_dimension(backend, image, d["caption"], dimension, criteria)
        else:
            result = judge_beauty_dimension(backend, image, dimension)
        path = RESULTS / d["id"] / f"{dimension}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=1))

    rows = [d for d in test_rows() if only is None or d["id"] in only]
    jobs = [(d, k) for d in rows for k in DIMENSIONS if not (RESULTS / d["id"] / f"{k}.json").exists()]
    print(f"scoring with {MODEL}: {len(jobs)} calls for {len(rows)} diagrams")
    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(judge, d, k): (d["id"], k) for d, k in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:  # a failed call is retried on the next run
                failed += 1
                print(f"failed {futures[future]}: {error}")
            if n % 100 == 0:
                print(f"{n}/{len(jobs)} done")
    print(f"{len(jobs) - failed}/{len(jobs)} done, {failed} failed")
    print(f"api usage this run: {backend.usage}")


TUNED = ("centering", "space_use", "alignment", "canvas_fill")  # rater-wording prompts from ../beauty_tuning (1-10)


def judge_scores(rows, tuned=False):
    """The judges' 13 scores per diagram, on 1-10 (Beauty divided by 10). With tuned, the four tuned Beauty dimensions
    come from ../beauty_tuning instead."""
    out = []
    for d in rows:
        scores = []
        for k in DIMENSIONS:
            if tuned and k in TUNED:
                scores.append(json.load(open(ROOT / "experiments" / "beauty_tuning" / "results" / d["id"] / f"rater_{k}.json"))["score"])
                continue
            s = json.load(open(RESULTS / d["id"] / f"{k}.json"))["score"]
            scores.append(s / 10 if k in BEAUTY_DIMENSIONS else s)
        out.append(scores)
    return np.array(out, dtype=float)


def classify(tuned=False):
    c = classification()
    rows, dims, X, y = c.load()
    develop, test, folds = c.split(rows)
    assert [rows[i]["id"] for i in test] == [d["id"] for d in test_rows()] and dims == DIMENSIONS
    missing = [(rows[i]["id"], k) for i in test for k in DIMENSIONS if not (RESULTS / rows[i]["id"] / f"{k}.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} judge scores missing, e.g. {missing[:3]}; run `score` first")

    gb = c.run_method("gradient_boosting", X, y, develop, test, folds)  # trained on the human scores of HS001-HS200
    cut = c.best_cut(gb["val_score"][develop], y[develop])
    J = judge_scores([rows[i] for i in test], tuned)
    t = y[test]
    vlm_score = c.bad_score(gb["model"], J)
    methods = {
        "pipeline: gradient boosting on judge scores": {"test_score": vlm_score, "test_called": vlm_score > cut},
        "reference: gradient boosting on human scores": {"test_score": gb["test_score"], "test_called": gb["test_called"]},
        "reference: mean of the 13 judge scores": {"test_score": -J.mean(axis=1), "test_called": None},
    }
    auc_draws, accuracy_draws = c.bootstrap(t, {n: m for n, m in methods.items() if m["test_called"] is not None})
    auc_draws["reference: mean of the 13 judge scores"] = c.bootstrap(t, {"x": {**methods["reference: mean of the 13 judge scores"],
                                                                                "test_called": np.zeros(len(t), bool)}})[0]["x"]
    summary = {}
    print(f"100 test diagrams HS201-HS300: {t.sum()} bad, {(1 - t).sum()} good. Cut on P(bad): {cut:.3f} (from val, HS001-HS200)\n")
    print("| Method | Test AUC (95% interval) | Accuracy (95% interval) | F1 | Bad caught | Clean flagged |\n|---|---:|---:|---:|---:|---:|")
    for name, m in methods.items():
        s = {"test_auc": float(roc_auc_score(t, m["test_score"])), "test_auc_interval": c.interval(auc_draws[name])}
        if m["test_called"] is not None:
            called = m["test_called"]
            s |= {"accuracy": float((called == t).mean()), "accuracy_interval": c.interval(accuracy_draws[name]),
                  "f1": float(f1_score(t, called)), "bad_caught": int(called[t == 1].sum()), "clean_flagged": int(called[t == 0].sum())}
        summary[name] = s
        acc = f"{s['accuracy']:.2f} ({s['accuracy_interval'][0]:.2f}-{s['accuracy_interval'][1]:.2f}) | {s['f1']:.3f} | {s['bad_caught']} of {t.sum()} | {s['clean_flagged']} of {(1 - t).sum()}" if "accuracy" in s else "- | - | - | -"
        print(f"| {name} | {s['test_auc']:.3f} ({s['test_auc_interval'][0]:.3f}-{s['test_auc_interval'][1]:.3f}) | {acc} |")
    a, b = "pipeline: gradient boosting on judge scores", "reference: gradient boosting on human scores"
    gap = {"auc": float(summary[a]["test_auc"] - summary[b]["test_auc"]), "auc_interval": c.interval(auc_draws[a] - auc_draws[b]),
           "accuracy": float(summary[a]["accuracy"] - summary[b]["accuracy"]), "accuracy_interval": c.interval(accuracy_draws[a] - accuracy_draws[b])}
    print(f"\nPipeline minus human-score reference: AUC {gap['auc']:+.3f} ({gap['auc_interval'][0]:+.3f} to {gap['auc_interval'][1]:+.3f}), "
          f"accuracy {gap['accuracy']:+.2f} ({gap['accuracy_interval'][0]:+.2f} to {gap['accuracy_interval'][1]:+.2f})")
    m = "reference: mean of the 13 judge scores"  # same bootstrap draws, so this difference is paired too
    gap_mean = {"auc": float(summary[a]["test_auc"] - summary[m]["test_auc"]), "auc_interval": c.interval(auc_draws[a] - auc_draws[m])}
    print(f"Pipeline minus mean of the 13 judge scores: AUC {gap_mean['auc']:+.3f} ({gap_mean['auc_interval'][0]:+.3f} to {gap_mean['auc_interval'][1]:+.3f})")

    # How the judges' scores sit next to the human scores the classifier learned from, on the same 100 diagrams.
    H = X[test]
    per_dimension = {k: {"human_mean": float(H[:, i].mean()), "judge_mean": float(J[:, i].mean()),
                         "spearman": float(spearmanr(H[:, i], J[:, i]).statistic),
                         "auc_human": float(roc_auc_score(t, -H[:, i])), "auc_judge": float(roc_auc_score(t, -J[:, i]))}
                     for i, k in enumerate(DIMENSIONS)}
    print("\n| Dimension | Human mean | Judge mean | Spearman | AUC alone, human | AUC alone, judge |\n|---|---:|---:|---:|---:|---:|")
    for k, v in per_dimension.items():
        print(f"| {k} | {v['human_mean']:.2f} | {v['judge_mean']:.2f} | {v['spearman']:.2f} | {v['auc_human']:.3f} | {v['auc_judge']:.3f} |")

    json.dump({"model": MODEL, "test": "HS201-HS300", "cut": float(cut), "methods": summary, "pipeline_minus_human": gap, "pipeline_minus_judge_mean": gap_mean,
               "per_dimension": per_dimension, "gradient_boosting_settings": gb["settings"]}, open(HERE / ("results_beauty_tuned.json" if tuned else "results.json"), "w"), indent=1)
    with open(HERE / ("predictions_beauty_tuned.csv" if tuned else "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "difficulty", "p_bad_from_judge_scores", "called_bad", "p_bad_from_human_scores", "called_bad_from_human_scores",
                    *(f"judge_{k}" for k in DIMENSIONS)])
        for n, i in enumerate(test):
            w.writerow([rows[i]["id"], rows[i]["label"], rows[i]["difficulty"], round(float(vlm_score[n]), 4), int(vlm_score[n] > cut),
                        round(float(gb["test_score"][n]), 4), int(gb["test_called"][n]), *(round(float(v), 1) for v in J[n])])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--only", help="comma-separated diagram ids (for a quick trial)")
    k = sub.add_parser("classify")
    k.add_argument("--beauty-tuned", action="store_true", help="the four tuned Beauty dimensions from ../beauty_tuning")
    args = parser.parse_args()
    if args.cmd == "score":
        score(args.workers, set(args.only.split(",")) if args.only else None)
    else:
        classify(args.beauty_tuned)
