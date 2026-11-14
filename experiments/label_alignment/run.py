"""Are the positive/negative labels of the 300 study diagrams right? Re-check the suspicious ones by eye, then compare
the two gradient boosting classifiers (trained on human scores, trained on judge scores) on the original and the
aligned labels.

    python3 experiments/label_alignment/run.py   # results.json, tables on stdout

review.csv (one row per diagram) holds the aligned labels:
  label          the original label (positive = the original figure, negative = errors inserted on purpose)
  reviewed       1 if the diagram was looked at again (suspicious by human scores, judge scores or either classifier)
  verdict        what the visual check found: for negatives bad / not_bad (was any inserted error detectable?),
                 for positives good / bad (does the original figure have a clear error of the same kinds?)
  aligned_label  negative if the diagram has a detectable error, positive otherwise; the original label if not reviewed
  reason         what was seen

Label sets: original; visual check (review.csv); both classifiers (flip a label when both out-of-fold classifiers
disagree with it); human mean < 8 (bad when the mean of the 13 human scores is below 8).

Both classifiers use the split, folds, setting grid and cut rule of ../classification (test HS201-HS300). The folds
are the same for every run, built from the original labels.
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = load_module("classification_run", EXPERIMENTS / "classification" / "run.py")
sys.argv = sys.argv[:1]
pipeline = load_module("pipeline_run", EXPERIMENTS / "pipeline" / "run.py")


def consensus_labels(X_human, X_judge, y, rows):
    """Flip a label when both classifiers disagree with it: 10-fold out-of-fold calls over all 300 (P(bad) > 0.5),
    one gradient boosting on human scores and one on judge scores, same settings for both."""
    groups = [f"{r['label']}_{r['difficulty']}" for r in rows]
    model = GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.03, random_state=0)
    called = {}
    for name, X in (("human", X_human), ("judge", X_judge)):
        p = np.zeros(len(y))
        for train, held in StratifiedKFold(10, shuffle=True, random_state=0).split(X, groups):
            p[held] = model.fit(X[train], y[train]).predict_proba(X[held])[:, 1]
        called[name] = (p > 0.5).astype(int)
    both_disagree = (called["human"] != y) & (called["judge"] != y)
    return np.where(both_disagree, 1 - y, y)


def main():
    rows, dims, X_human, y_original = c.load()
    X_judge = pipeline.judge_scores(rows, tuned=True)  # the four tuned Beauty prompts, as in the best pipeline
    review = {r["id"]: r for r in csv.DictReader(open(HERE / "review.csv"))}
    develop, test, folds = c.split(rows)
    label_sets = {
        "original": y_original,
        "visual check": np.array([review[r["id"]]["aligned_label"] == "negative" for r in rows], dtype=int),
        "both classifiers": consensus_labels(X_human, X_judge, y_original, rows),
        "human mean < 8": (X_human.mean(axis=1) < 8).astype(int),  # bad when the mean of the 13 human scores is below 8
    }

    out = {}
    print("| Labels | Changed (in test) | Classifier | Val AUC | Test AUC (95% interval) | Accuracy (95% interval) | Bad caught | Good flagged |\n"
          "|---|---:|---|---:|---:|---:|---:|---:|")
    for labels, y in label_sets.items():
        changed = np.flatnonzero(y != y_original)
        methods = {"human": c.run_method("gradient_boosting", X_human, y, develop, test, folds),
                   "judge": c.run_method("gradient_boosting", X_judge, y, develop, test, folds)}
        auc, accuracy = c.bootstrap(y[test], methods)
        n_bad, n_good = int(y[test].sum()), int((1 - y[test]).sum())
        flipped = y[test] != y_original[test]
        result = {}
        for name, m in methods.items():
            result[name] = {"settings": m["settings"], "val_auc": m["val_auc"], "test_auc": m["test_auc"], "test_auc_interval": c.interval(auc[name]),
                            "accuracy": m["accuracy"], "accuracy_interval": c.interval(accuracy[name]),
                            "bad_caught": m["bad_caught"], "bad": n_bad, "good_flagged": m["clean_flagged"], "good": n_good,
                            "accuracy_on_changed_test": float((m["test_called"][flipped] == y[test][flipped]).mean()) if flipped.any() else None,
                            "accuracy_on_unchanged_test": float((m["test_called"][~flipped] == y[test][~flipped]).mean())}
            r = result[name]
            print(f"| {labels} | {len(changed)} ({int(flipped.sum())}) | trained on {name} scores | {r['val_auc']:.3f} "
                  f"| {r['test_auc']:.3f} ({r['test_auc_interval'][0]:.3f}-{r['test_auc_interval'][1]:.3f}) "
                  f"| {r['accuracy']:.2f} ({r['accuracy_interval'][0]:.2f}-{r['accuracy_interval'][1]:.2f}) | {r['bad_caught']} of {n_bad} | {r['good_flagged']} of {n_good} |")
        gap = {"auc": result["human"]["test_auc"] - result["judge"]["test_auc"], "auc_interval": c.interval(auc["human"] - auc["judge"]),
               "accuracy": result["human"]["accuracy"] - result["judge"]["accuracy"], "accuracy_interval": c.interval(accuracy["human"] - accuracy["judge"])}
        out[labels] = {"changed": [{"id": rows[i]["id"], "from": rows[i]["label"], "to": "negative" if y[i] else "positive"} for i in changed],
                       "negative_to_positive": int((y_original[changed] == 1).sum()), "positive_to_negative": int((y_original[changed] == 0).sum()),
                       "methods": result, "human_minus_judge": gap}
    print()
    for labels, v in out.items():
        g = v["human_minus_judge"]
        print(f"{labels}: negative->positive {v['negative_to_positive']}, positive->negative {v['positive_to_negative']}; human minus judge: "
              f"AUC {g['auc']:+.3f} ({g['auc_interval'][0]:+.3f} to {g['auc_interval'][1]:+.3f}), "
              f"accuracy {g['accuracy']:+.2f} ({g['accuracy_interval'][0]:+.2f} to {g['accuracy_interval'][1]:+.2f}); "
              + "; ".join(f"{n}: accuracy on changed test {r['accuracy_on_changed_test'] if r['accuracy_on_changed_test'] is None else round(r['accuracy_on_changed_test'], 2)}, "
                          f"on the rest {r['accuracy_on_unchanged_test']:.2f}" for n, r in v["methods"].items()))
    json.dump(out, open(HERE / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
