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

Both classifiers use the split, folds, setting grid and cut rule of ../classification (test HS201-HS300). The folds
are the same for every run, built from the original labels.
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

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


def main():
    rows, dims, X_human, y_original = c.load()
    X_judge = pipeline.judge_scores(rows, tuned=True)  # the four tuned Beauty prompts, as in the best pipeline
    review = {r["id"]: r for r in csv.DictReader(open(HERE / "review.csv"))}
    y_aligned = np.array([review[r["id"]]["aligned_label"] == "negative" for r in rows], dtype=int)
    develop, test, folds = c.split(rows)

    changed = [r for r in rows if review[r["id"]]["aligned_label"] != r["label"]]
    print(f"Reviewed {sum(v['reviewed'] == '1' for v in review.values())} of 300; labels changed: {len(changed)} "
          f"(negative -> positive {sum(r['label'] == 'negative' for r in changed)}, positive -> negative {sum(r['label'] == 'positive' for r in changed)}); "
          f"in the test set: {sum(int(r['id'][2:]) >= 201 for r in changed)}\n")

    out = {"changed": [{"id": r["id"], "from": r["label"], "to": review[r["id"]]["aligned_label"], "reason": review[r["id"]]["reason"]} for r in changed],
           "labels": {}}
    print("| Labels | Classifier | Val AUC | Test AUC (95% interval) | Accuracy (95% interval) | Bad caught | Good flagged |\n|---|---|---:|---:|---:|---:|---:|")
    for labels, y in (("original", y_original), ("aligned", y_aligned)):
        methods = {"trained on human scores, given human scores": c.run_method("gradient_boosting", X_human, y, develop, test, folds),
                   "trained on judge scores, given judge scores": c.run_method("gradient_boosting", X_judge, y, develop, test, folds)}
        auc, accuracy = c.bootstrap(y[test], methods)
        n_bad, n_good = int(y[test].sum()), int((1 - y[test]).sum())
        result = {}
        flipped = y_aligned[test] != y_original[test]
        for name, m in methods.items():
            result[name] = {"settings": m["settings"], "val_auc": m["val_auc"], "test_auc": m["test_auc"], "test_auc_interval": c.interval(auc[name]),
                            "accuracy": m["accuracy"], "accuracy_interval": c.interval(accuracy[name]),
                            "bad_caught": m["bad_caught"], "bad": n_bad, "good_flagged": m["clean_flagged"], "good": n_good,
                            "changed_test_called_bad": [rows[test[i]]["id"] for i in np.flatnonzero(flipped & m["test_called"])],
                            "accuracy_on_unchanged_test": float((m["test_called"][~flipped] == y[test][~flipped]).mean())}
            r = result[name]
            print(f"| {labels} | {name} | {r['val_auc']:.3f} | {r['test_auc']:.3f} ({r['test_auc_interval'][0]:.3f}-{r['test_auc_interval'][1]:.3f}) "
                  f"| {r['accuracy']:.2f} ({r['accuracy_interval'][0]:.2f}-{r['accuracy_interval'][1]:.2f}) | {r['bad_caught']} of {n_bad} | {r['good_flagged']} of {n_good} |")
        a, b = methods
        gap = {"auc": result[a]["test_auc"] - result[b]["test_auc"], "auc_interval": c.interval(auc[a] - auc[b]),
               "accuracy": result[a]["accuracy"] - result[b]["accuracy"], "accuracy_interval": c.interval(accuracy[a] - accuracy[b])}
        out["labels"][labels] = {"methods": result, "human_minus_judge": gap}
    print(f"\nThe {int((y_aligned[test] != y_original[test]).sum())} test diagrams whose label changed, and the other test diagrams:")
    for labels, v in out["labels"].items():
        for name, r in v["methods"].items():
            print(f"  {labels}, {name}: calls bad {len(r['changed_test_called_bad'])} of them {r['changed_test_called_bad']}; "
                  f"accuracy on the rest {r['accuracy_on_unchanged_test']:.2f}")
    print()
    for labels, v in out["labels"].items():
        g = v["human_minus_judge"]
        print(f"{labels} labels, human minus judge: AUC {g['auc']:+.3f} ({g['auc_interval'][0]:+.3f} to {g['auc_interval'][1]:+.3f}), "
              f"accuracy {g['accuracy']:+.2f} ({g['accuracy_interval'][0]:+.2f} to {g['accuracy_interval'][1]:+.2f})")
    json.dump(out, open(HERE / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
