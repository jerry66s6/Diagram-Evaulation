"""The low-confidence calls of the final pipeline on the 6,000 dataset diagrams, to be labeled again by hand in the
user-study site. The dataset labels are not used: the band comes from the 100 test diagrams (HS201-HS300, corrected
labels), and the review checks our call.

    python3 experiments/spot_check/run.py   # spot_check.csv, and review_manifest.csv for the site (not in git)

Low confidence, by split conformal prediction at 90% coverage: on the 100 test diagrams the score is 1 minus the
classifier's probability of the true label; qhat is its ceil(101 * 0.9) = 91st smallest value. A diagram is low
confidence when both labels stay in its prediction set, i.e. 1 - qhat <= P(bad) <= qhat.

Then, from user-study/ (see its README, "Review mode"):
    node scripts/import-diagrams.mjs ../experiments/spot_check/review_manifest.csv --review --sets 38 --mode research --id low-confidence-review-v1
"""
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIPELINE = ROOT / "experiments" / "pipeline"
ALPHA = 0.1

test = list(csv.DictReader(open(PIPELINE / "predictions_final.csv")))
scores = sorted(1 - float(r["p_bad"]) if r["corrected_label"] == "negative" else float(r["p_bad"]) for r in test)
qhat = scores[math.ceil((len(scores) + 1) * (1 - ALPHA)) - 1]
low_p, high_p = 1 - qhat, qhat

data = {r["id"]: r for r in csv.DictReader(open(ROOT / "diagrams" / "data.csv"))}
pred = list(csv.DictReader(open(PIPELINE / "dataset_predictions.csv")))
dims = [k[len("judge_"):] for k in pred[0] if k.startswith("judge_")]
low = sorted((r for r in pred if low_p <= float(r["p_bad"]) <= high_p), key=lambda r: r["id"])

with open(HERE / "spot_check.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "image", "difficulty", "p_bad", "our_call", *(f"judge_{k}" for k in dims)])
    for r in low:
        w.writerow([r["id"], data[r["id"]]["image"], r["difficulty"], r["p_bad"], "bad" if r["called_bad"] == "1" else "good",
                    *(r[f"judge_{k}"] for k in dims)])

with open(HERE / "review_manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "file", "difficulty", "caption", "source", "model_call", "model_scores"])
    for r in low:
        d = data[r["id"]]
        w.writerow([r["id"], f"../../diagrams/{d['image']}", r["difficulty"], d["caption"], d["source_id"],
                    "bad" if r["called_bad"] == "1" else "good", json.dumps({k: float(r[f"judge_{k}"]) for k in dims})])

json.dump({"alpha": ALPHA, "test_diagrams": len(test), "qhat": qhat, "band": [low_p, high_p], "low_confidence": len(low),
           "of": len(pred), "called_bad": sum(r["called_bad"] == "1" for r in low)}, open(HERE / "band.json", "w"), indent=1)
print(f"conformal {1 - ALPHA:.0%}: qhat {qhat:.3f}; low confidence when {low_p:.3f} <= P(bad) <= {high_p:.3f}: "
      f"{len(low)} of {len(pred)} (called bad {sum(r['called_bad'] == '1' for r in low)})")
