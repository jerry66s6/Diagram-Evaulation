"""Good vs bad diagrams from the 13 human scores. No images, no judge scores, no API calls.

    python3 experiments/classification/run.py   # results.json, predictions.csv, tables on stdout

Input: the 13 human scores in ../human_alignment/human_ratings.json.
Label: `label` in the same file. negative = deliberately made bad with inserted errors, positive = good.
Split: HS201-HS300 is the test set, scored once at the end. HS001-HS200 go through 5-fold train/val (160/40), balanced on
label and difficulty. Each method's settings and bad/good cut come from val; it is then refit on all 200 and tested.
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

HERE = Path(__file__).resolve().parent
RATINGS = HERE.parent / "human_alignment" / "human_ratings.json"
SEED = 0
TEST_START = 201  # HS201 onwards is the test set
N_FOLDS = 5
N_BOOTSTRAP = 2000

# Each grid goes from simpler to more complex; a tie on val keeps the earlier setting.
MODELS = {
    "decision_tree": (DecisionTreeClassifier(random_state=0),
                      {"max_depth": [2, 3, 4, 5], "min_samples_leaf": [1, 5, 10]}),
    "logistic_regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                            {"logisticregression__C": [0.01, 0.1, 1, 10]}),
    "random_forest": (RandomForestClassifier(n_estimators=200, random_state=0),
                      {"max_depth": [3, 5, None], "min_samples_leaf": [1, 5]}),
    "gradient_boosting": (GradientBoostingClassifier(random_state=0),
                          {"max_depth": [1, 2, 3], "n_estimators": [50, 100, 200], "learning_rate": [0.03, 0.1]}),
    "svm_rbf": (make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                {"svc__C": [0.1, 1, 10], "svc__gamma": [0.01, 0.1, "scale"]}),
}
# Nothing is trained: a lower score means more likely bad. The cut still comes from val, like the models.
BASELINES = {"mean_of_13": lambda X: -X.mean(axis=1), "lowest_of_13": lambda X: -X.min(axis=1)}
NAMES = [*BASELINES, *MODELS]


def load():
    data = json.load(open(RATINGS))
    dims = data["dimensions"]["correctness"] + data["dimensions"]["beauty"]
    rows = data["diagrams"]
    X = np.array([[r["scores"][k] for k in dims] for r in rows], dtype=float)
    y = np.array([r["label"] == "negative" for r in rows], dtype=int)  # 1 = bad
    return rows, dims, X, y


def split(rows):
    """HS201-HS300 as test, and 5 train/val folds of HS001-HS200 balanced on label and difficulty."""
    groups = np.array([f"{r['label']}_{r['difficulty']}" for r in rows])
    in_test = np.array([int(r["id"][2:]) >= TEST_START for r in rows])
    develop, test = np.flatnonzero(~in_test), np.flatnonzero(in_test)
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(develop, groups[develop])
    return develop, test, [(develop[a], develop[b]) for a, b in folds]


def bad_score(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)


def best_cut(score, y):
    """The cut with the best val accuracy; among ties, the middle one."""
    values = np.unique(score)
    cuts = np.r_[values[0] - 1, (values[:-1] + values[1:]) / 2, values[-1] + 1]
    accuracy = np.array([((score > c) == y).mean() for c in cuts])
    best = np.flatnonzero(accuracy == accuracy.max())
    return cuts[best[len(best) // 2]]


def run_method(name, X, y, develop, test, folds):
    """Choose settings and cut from val, refit on all of train/val unless it is a baseline, then score the test set."""
    if name in BASELINES:
        model, params, val = None, {}, BASELINES[name](X)
        val_auc = float(np.mean([roc_auc_score(y[v], val[v]) for _, v in folds]))
        test_score = BASELINES[name](X[test])
    else:
        estimator, grid = MODELS[name]
        val_auc, params, val = -1, None, None
        for candidate in ParameterGrid(grid):
            scores, aucs = np.zeros(len(y)), []
            for train, v in folds:
                scores[v] = bad_score(clone(estimator).set_params(**candidate).fit(X[train], y[train]), X[v])
                aucs.append(roc_auc_score(y[v], scores[v]))
            if np.mean(aucs) > val_auc:
                val_auc, params, val = float(np.mean(aucs)), candidate, scores
        model = clone(estimator).set_params(**params).fit(X[develop], y[develop])
        test_score = bad_score(model, X[test])
    cut = best_cut(val[develop], y[develop])
    called = test_score > cut
    t = y[test]
    return {"model": model, "settings": params, "val_auc": val_auc, "val_accuracy": float(((val[develop] > cut) == y[develop]).mean()),
            "val_score": val, "val_called": val > cut, "test_score": test_score, "test_called": called,
            "test_auc": float(roc_auc_score(t, test_score)), "accuracy": float((called == t).mean()), "f1": float(f1_score(t, called, zero_division=0)),
            "bad_caught": int(called[t == 1].sum()), "clean_flagged": int(called[t == 0].sum())}


def bootstrap(y, score, base):
    """95% intervals for test AUC and its gap to the baseline, resampling test diagrams within each label."""
    rng = np.random.default_rng(SEED)
    good, bad = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    aucs, gaps = [], []
    for _ in range(N_BOOTSTRAP):
        at = np.r_[rng.choice(good, len(good)), rng.choice(bad, len(bad))]
        auc = roc_auc_score(y[at], score[at])
        aucs.append(auc)
        gaps.append(auc - roc_auc_score(y[at], base[at]))
    return [float(v) for v in np.percentile(aucs, [2.5, 97.5])], [float(v) for v in np.percentile(gaps, [2.5, 97.5])]


def main():
    rows, dims, X, y = load()
    develop, test, folds = split(rows)
    print(f"{len(develop)} train/val in {N_FOLDS} folds, {len(test)} test ({y[test].sum()} bad); inputs: {dims}\n")

    methods = {name: run_method(name, X, y, develop, test, folds) for name in NAMES}
    n_bad, n_good = int(y[test].sum()), int((1 - y[test]).sum())
    print(f"| Method | Val AUC | Val accuracy | Test AUC (95% interval) | Minus mean_of_13 (95% interval) | Test accuracy | Test F1 | Bad caught of {n_bad} | Clean flagged of {n_good} |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in methods.items():
        m["test_auc_interval"], m["gap_interval"] = bootstrap(y[test], m["test_score"], methods["mean_of_13"]["test_score"])
        m["gap"] = m["test_auc"] - methods["mean_of_13"]["test_auc"]
        print(f"| {name} | {m['val_auc']:.3f} | {m['val_accuracy']:.3f} | {m['test_auc']:.3f} ({m['test_auc_interval'][0]:.3f}-{m['test_auc_interval'][1]:.3f}) "
              f"| {m['gap']:+.3f} ({m['gap_interval'][0]:+.3f} to {m['gap_interval'][1]:+.3f}) | {m['accuracy']:.3f} | {m['f1']:.3f} | {m['bad_caught']} | {m['clean_flagged']} |")
    print("\nSettings: " + json.dumps({n: methods[n]["settings"] for n in MODELS}))

    # The method to carry forward is chosen on val, not on the test table.
    chosen = max(MODELS, key=lambda n: methods[n]["val_auc"])
    print(f"\nBest val AUC: {chosen}")
    levels = {}
    for level in ("easy", "medium", "hard"):
        at = np.array([rows[i]["difficulty"] == level for i in test])
        called, t = methods[chosen]["test_called"][at], y[test][at]
        levels[level] = {"bad_caught": int(called[t == 1].sum()), "bad": int((t == 1).sum()), "clean_flagged": int(called[t == 0].sum()), "clean": int((t == 0).sum())}
    print("Test by difficulty: " + json.dumps(levels))
    caught, total = {}, {}  # out-of-fold val calls on HS001-HS200: too few of each error type in the test set
    for i in develop:
        for defect in {d["defect"] for d in rows[i]["defects"]}:
            total[defect] = total.get(defect, 0) + 1
            caught[defect] = caught.get(defect, 0) + int(methods[chosen]["val_called"][i])
    by_defect = {k: {"caught": caught[k], "of": total[k]} for k in sorted(total, key=lambda k: caught[k] / total[k])}
    print("Val by inserted error: " + json.dumps(by_defect))

    compared = ["logistic_regression", "random_forest", "svm_rbf"]
    parts = {"all 13": dims, "correctness 5": dims[:5], "beauty 8": dims[5:], "without style_uniformity": [k for k in dims if k != "style_uniformity"]}
    by_input = {}
    print("\nVal AUC / test AUC / test accuracy by input\n\n| Input | " + " | ".join(compared) + " |\n|---|" + "---:|" * len(compared))
    for part, keep in parts.items():
        at = [dims.index(k) for k in keep]
        result = {n: run_method(n, X[:, at], y, develop, test, folds) for n in compared}
        by_input[part] = {n: {k: result[n][k] for k in ("val_auc", "test_auc", "accuracy")} for n in compared}
        print(f"| {part} | " + " | ".join(f"{result[n]['val_auc']:.3f} / {result[n]['test_auc']:.3f} / {result[n]['accuracy']:.3f}" for n in compared) + " |")

    logistic = methods["logistic_regression"]["model"][-1].coef_[0]
    drops = {n: permutation_importance(methods[n]["model"], X[test], y[test], scoring="roc_auc", n_repeats=20, random_state=SEED).importances_mean for n in MODELS}
    dimensions = {k: {"auc_alone_train_val": float(roc_auc_score(y[develop], -X[develop, i])), "logistic_coefficient": float(logistic[i]),
                      "test_auc_drop": {n: float(drops[n][i]) for n in MODELS}} for i, k in enumerate(dims)}
    print("\n| Dimension | AUC alone (train/val) | Logistic coefficient (- = low score means bad) | " + " | ".join(MODELS) + " |\n|---|---:|---:|" + "---:|" * len(MODELS))
    for k in sorted(dims, key=lambda k: -dimensions[k]["auc_alone_train_val"]):
        d = dimensions[k]
        print(f"| {k} | {d['auc_alone_train_val']:.3f} | {d['logistic_coefficient']:+.2f} | " + " | ".join(f"{d['test_auc_drop'][n]:+.3f}" for n in MODELS) + " |")

    # Mean human style_uniformity by what was done to the figure, on all 300.
    style = np.array([r["scores"]["style_uniformity"] for r in rows])
    beauty_error = np.array([any(d["criterion"].startswith("beauty") for d in r["defects"]) for r in rows])
    label_size = np.array([any(d["defect"] in ("enlarge_label", "shrink_label") for d in r["defects"]) for r in rows])
    style_means = {name: {"n": int(at.sum()), "mean": float(style[at].mean())} for name, at in {
        "positive": y == 0, "negative, Beauty error": beauty_error,
        "negative, no Beauty error, label size changed": (y == 1) & ~beauty_error & label_size,
        "negative, no Beauty error, label size unchanged": (y == 1) & ~beauty_error & ~label_size}.items()}
    print("\nstyle_uniformity: " + json.dumps(style_means))

    rules = export_text(methods["decision_tree"]["model"], feature_names=dims, class_names=["good", "bad"])
    print("\nDecision tree (the one tested)\n" + rules)

    keep = ("settings", "val_auc", "val_accuracy", "test_auc", "test_auc_interval", "gap", "gap_interval", "accuracy", "f1", "bad_caught", "clean_flagged")
    json.dump({"inputs": dims, "bad_means": "label negative", "test_ids": [rows[i]["id"] for i in test],
               "methods": {n: {k: m[k] for k in keep} for n, m in methods.items()}, "best_val_auc": chosen, "test_by_difficulty": levels,
               "val_by_inserted_error": by_defect, "by_input": by_input, "dimensions": dimensions, "style_uniformity_means": style_means, "decision_tree": rules},
              open(HERE / "results.json", "w"), indent=1)
    part_of = {int(i): "test" for i in test} | {int(i): f"val_{k}" for k, (_, v) in enumerate(folds) for i in v}
    with open(HERE / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "difficulty", "part", *(c for n in NAMES for c in (f"{n}_score", f"{n}_bad"))])
        where = {int(i): k for k, i in enumerate(test)}
        for i, r in enumerate(rows):
            cells = []
            for n in NAMES:
                m = methods[n]
                score, called = (m["test_score"][where[i]], m["test_called"][where[i]]) if i in where else (m["val_score"][i], m["val_called"][i])
                cells += [round(float(score), 4), int(called)]
            w.writerow([r["id"], r["label"], r["difficulty"], part_of[i], *cells])


if __name__ == "__main__":
    main()
