"""The diagrams of the 6,000 to label again by hand in the user-study site: the low-confidence calls of the final
pipeline, plus as many of the least confident high-confidence calls (to check those too). The dataset labels are not
used: the band comes from the 100 test diagrams (HS201-HS300, corrected labels), and the review checks our call.

    python3 experiments/spot_check/run.py   # spot_check.csv, band.json, and review_manifest.csv for the site (not in git)

Low confidence, by split conformal prediction at 90% coverage: on the 100 test diagrams the score is 1 minus the
classifier's probability of the true label; qhat is its ceil(101 * 0.9) = 91st smallest value. A diagram is low
confidence when both labels stay in its prediction set, i.e. 1 - qhat <= P(bad) <= qhat.

High confidence: the rest. Their confidence is max(P(bad), 1 - P(bad)); the least confident of them, as many as there
are low-confidence diagrams (ties broken by id), are reviewed too.

Then, from user-study/ (see its README, "Review mode"):
    node scripts/import-diagrams.mjs ../experiments/spot_check/review_manifest.csv --review --sets 76 --mode research --id low-confidence-review-v2
"""
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIPELINE = ROOT / "experiments" / "pipeline"
ALPHA = 0.1


def confidence(p):
    return max(p, 1 - p)


test = list(csv.DictReader(open(PIPELINE / "predictions_final.csv")))
scores = sorted(1 - float(r["p_bad"]) if r["corrected_label"] == "negative" else float(r["p_bad"]) for r in test)
qhat = scores[math.ceil((len(scores) + 1) * (1 - ALPHA)) - 1]
low_p, high_p = 1 - qhat, qhat

data = {r["id"]: r for r in csv.DictReader(open(ROOT / "diagrams" / "data.csv"))}
pred = list(csv.DictReader(open(PIPELINE / "dataset_predictions.csv")))
dims = [k[len("judge_"):] for k in pred[0] if k.startswith("judge_")]
for r in pred:
    r["p"] = float(r["p_bad"])
    r["part"] = "low confidence" if low_p <= r["p"] <= high_p else "high confidence"
high = sorted((r for r in pred if r["part"] == "high confidence"), key=lambda r: (confidence(r["p"]), r["id"]))
n_low = sum(r["part"] == "low confidence" for r in pred)
for r in high[:n_low]:
    r["part"] = "high confidence, least confident"
limit = confidence(high[n_low - 1]["p"])  # the confidence of the last one taken
split_tie = sum(confidence(r["p"]) == limit for r in high[n_low:])  # diagrams with that same confidence left out
review = sorted((r for r in pred if r["part"] != "high confidence"), key=lambda r: r["id"])

with open(HERE / "spot_check.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "image", "difficulty", "part", "p_bad", "our_call", *(f"judge_{k}" for k in dims)])
    for r in review:
        w.writerow([r["id"], data[r["id"]]["image"], r["difficulty"], r["part"], r["p_bad"], "bad" if r["called_bad"] == "1" else "good",
                    *(r[f"judge_{k}"] for k in dims)])

with open(HERE / "review_manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "file", "difficulty", "caption", "source", "model_call", "model_scores"])
    for r in review:
        d = data[r["id"]]
        w.writerow([r["id"], f"../../diagrams/{d['image']}", r["difficulty"], d["caption"], d["source_id"],
                    "bad" if r["called_bad"] == "1" else "good", json.dumps({k: float(r[f"judge_{k}"]) for k in dims})])

# The same rule on the 100 test diagrams: how many of the classifier's errors each part holds.
cut = json.load(open(PIPELINE / "results_final.json"))["cut"]
test_parts = {}
for r in test:
    p, wrong = float(r["p_bad"]), (float(r["p_bad"]) > cut) != (r["corrected_label"] == "negative")
    part = "low confidence" if low_p <= p <= high_p else "high confidence, least confident" if confidence(p) <= limit else "high confidence"
    n, e = test_parts.get(part, (0, 0))
    test_parts[part] = (n + 1, e + wrong)

parts = {name: sum(r["part"] == name for r in pred) for name in ("low confidence", "high confidence, least confident", "high confidence")}
json.dump({"alpha": ALPHA, "test_diagrams": len(test), "qhat": qhat, "band": [low_p, high_p], "high_confidence_limit": limit, "tie_left_out": split_tie,
           "dataset": parts, "reviewed": len(review),
           "test_errors_by_part": {k: {"diagrams": n, "errors": e} for k, (n, e) in test_parts.items()}},
          open(HERE / "band.json", "w"), indent=1)
print(f"conformal {1 - ALPHA:.0%}: low confidence when {low_p:.3f} <= P(bad) <= {high_p:.3f}; "
      f"least confident high-confidence calls up to confidence {limit:.3f} ({split_tie} tied left out)")
print("dataset:", parts, "| to review:", len(review))
print("test (diagrams, errors):", test_parts)
