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


# Fixed gradient boosting settings for every label set, so only the labels change: the settings 5-fold val chose on
# the original labels (../classification for human scores, ../pipeline train-on-judges for judge scores).
c.MODELS["gb_human"] = (GradientBoostingClassifier(random_state=0), {"n_estimators": [200], "max_depth": [2], "learning_rate": [0.03]})
c.MODELS["gb_judge"] = (GradientBoostingClassifier(random_state=0), {"n_estimators": [200], "max_depth": [3], "learning_rate": [0.03]})
CLASSIFIERS = {"pipeline": "trained on human scores, given judge scores",
               "judge": "trained on judge scores, given judge scores",
               "human": "trained on human scores, given human scores"}


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
    print("| Labels | Changed (in test) | Classifier | Test AUC (95% interval) | Accuracy (95% interval) | Bad caught | Good flagged |\n"
          "|---|---:|---|---:|---:|---:|---:|")
    for labels, y in label_sets.items():
        changed = np.flatnonzero(y != y_original)
        flipped = y[test] != y_original[test]
        human = c.run_method("gb_human", X_human, y, develop, test, folds)
        # The pipeline: the model trained on human scores (with its val cut), given the judges' scores of the test set.
        cut = c.best_cut(human["val_score"][develop], y[develop])
        p = c.bad_score(human["model"], X_judge[test])
        methods = {"pipeline": {"test_score": p, "test_called": p > cut},
                   "judge": c.run_method("gb_judge", X_judge, y, develop, test, folds),
                   "human": human}
        auc, accuracy = c.bootstrap(y[test], methods)
        t = y[test]
        result = {}
        for name, m in methods.items():
            called = m["test_called"]
            result[name] = {"test_auc": float(c.roc_auc_score(t, m["test_score"])), "test_auc_interval": c.interval(auc[name]),
                            "accuracy": float((called == t).mean()), "accuracy_interval": c.interval(accuracy[name]),
                            "bad_caught": int(called[t == 1].sum()), "bad": int(t.sum()), "good_flagged": int(called[t == 0].sum()), "good": int((1 - t).sum()),
                            "accuracy_on_changed_test": float((called[flipped] == t[flipped]).mean()) if flipped.any() else None}
            r = result[name]
            print(f"| {labels} | {len(changed)} ({int(flipped.sum())}) | {CLASSIFIERS[name]} "
                  f"| {r['test_auc']:.3f} ({r['test_auc_interval'][0]:.3f}-{r['test_auc_interval'][1]:.3f}) "
                  f"| {r['accuracy']:.2f} ({r['accuracy_interval'][0]:.2f}-{r['accuracy_interval'][1]:.2f}) | {r['bad_caught']} of {r['bad']} | {r['good_flagged']} of {r['good']} |")
        out[labels] = {"changed": [{"id": rows[i]["id"], "from": rows[i]["label"], "to": "negative" if y[i] else "positive"} for i in changed],
                       "negative_to_positive": int((y_original[changed] == 1).sum()), "positive_to_negative": int((y_original[changed] == 0).sum()),
                       "methods": result}
    print()
    for labels, v in out.items():
        print(f"{labels}: negative->positive {v['negative_to_positive']}, positive->negative {v['positive_to_negative']}; accuracy on the changed test diagrams: "
              + ", ".join(f"{n} {r['accuracy_on_changed_test']:.2f}" for n, r in v["methods"].items() if r["accuracy_on_changed_test"] is not None))
    json.dump(out, open(HERE / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
