"""Fresh beauty scoring and human review, separate from historical experiments."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
import mimetypes
from pathlib import Path
import random
import statistics
import sys
import threading
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from diagram_correctness.beauty import judge_beauty_dimension
from diagram_correctness.beauty_v6 import BEAUTY_V6_DIMENSIONS, BEAUTY_V6_PROMPTS

DIMENSIONS = tuple(BEAUTY_V6_DIMENSIONS)
MODEL = "dev-gpt-6-luna"
RESPONSE_INSTRUCTION = (
    "\nReturn the requested observation first, then a score from 0 to 100, then a concise "
    "one- or two-sentence justification."
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def standardized(score):
    return 1 + 9 * score / 100


def prepare(args):
    labels = read_json(args.labels)
    if [row["id"] for row in labels["diagrams"]] != [f"HS{i:03}" for i in range(1, 301)]:
        raise ValueError("Expected 300 labels in numeric HS001-HS300 order.")
    if tuple(labels["dimensions"]["beauty"]) != DIMENSIONS:
        raise ValueError("Human beauty dimensions do not match the current evaluator.")
    images = []
    for row in labels["diagrams"][:200]:
        image = args.images / Path(row["image"]).name
        if not image.is_file():
            raise FileNotFoundError(image)
        for dimension in DIMENSIONS:
            score = row["scores"][dimension]
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= score <= 10:
                raise ValueError(f"Invalid human beauty score: {row['id']} / {dimension}")
        images.append({"id": row["id"], "filename": image.name})
    args.run.mkdir(parents=True, exist_ok=False)
    (args.run / "versions").mkdir()
    mini = sorted(random.Random(args.seed).sample(range(101, 201), 20))
    write_json(args.run / "manifest.json", {
        "created_at": now(), "source_labels": str(args.labels.resolve()),
        "image_directory": str(args.images.resolve()), "model": MODEL,
        "dimensions": DIMENSIONS, "images": images,
        "training_ids": [f"HS{i:03}" for i in range(1, 101)],
        "validation_ids": [f"HS{i:03}" for i in range(101, 201)],
        "mini_validation_ids": [f"HS{i:03}" for i in mini],
        "mini_validation_seed": args.seed,
        "scaling": "Human: unchanged 1-10. Judge: 1 + 9 * score / 100.",
        "protocol": {
            "max_candidates": 5, "edits_per_candidate": 5,
            "primary_metric": "full-validation pooled RMSE",
            "stop_after_consecutive_non_improvements": 2,
            "screening": "Fixed mini-validation; promising candidates get full validation.",
            "prompt_insights": "Approved training images only; never validation errors.",
            "test_set": "HS201-HS300 excluded from scoring, selection, and review.",
        },
    })
    write_json(args.run / "human_labels_original.json", labels)
    write_json(args.run / "human_labels_corrected.json", labels)
    write_json(args.run / "versions" / "baseline.json", {
        "id": "baseline", "description": "Unmodified current beauty v6 prompts; fresh GPT-6 luna scores.",
        "created_at": now(), "prompts": BEAUTY_V6_PROMPTS,
        "response_instruction": RESPONSE_INSTRUCTION, "edits": [], "training_evidence_ids": [],
    })
    write_json(args.run / "review.json", {"images": {}, "approval": None})
    refresh(args.run)
    print(f"Prepared a new run at {args.run.resolve()}", flush=True)


def records(run, version="baseline"):
    path = run / f"{version}_scores.jsonl"
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["id"], row["dimension"])
        if key in result:
            raise ValueError(f"Duplicate saved score: {key}")
        result[key] = row
    return result


def metrics(pairs):
    human = [pair[0] for pair in pairs]
    judge = [pair[1] for pair in pairs]
    pearson = None
    if len(pairs) > 1 and len(set(human)) > 1 and len(set(judge)) > 1:
        pearson = statistics.correlation(human, judge)
    return {
        "n": len(pairs),
        "rmse": math.sqrt(statistics.fmean((h - j) ** 2 for h, j in pairs)),
        "mae": statistics.fmean(abs(h - j) for h, j in pairs),
        "pearson": pearson,
    }


def validation_metrics(ids, humans, saved):
    if any((image_id, dimension) not in saved for image_id in ids for dimension in DIMENSIONS):
        return None
    by_dimension = {
        dimension: metrics([
            (humans[image_id]["scores"][dimension], standardized(saved[image_id, dimension]["score"]))
            for image_id in ids
        ]) for dimension in DIMENSIONS
    }
    return {
        "overall": metrics([
            (humans[image_id]["scores"][dimension], standardized(saved[image_id, dimension]["score"]))
            for image_id in ids for dimension in DIMENSIONS
        ]),
        "by_dimension": by_dimension,
    }


def paired_comparison(ids, humans, baseline, candidate, draws=10000):
    if any((image_id, d) not in scores
           for scores in (baseline, candidate) for image_id in ids for d in DIMENSIONS):
        return None
    errors = [
        tuple(statistics.fmean(
            (humans[image_id]["scores"][d] - standardized(scores[image_id, d]["score"])) ** 2
            for d in DIMENSIONS
        ) for scores in (baseline, candidate))
        for image_id in ids
    ]
    baseline_mse = statistics.fmean(a for a, _ in errors)
    candidate_mse = statistics.fmean(b for _, b in errors)
    differences = [b - a for a, b in errors]
    observed = abs(sum(differences))
    rng = random.Random(20261112)
    bootstrap = []
    extreme = 0
    for _ in range(draws):
        sample = rng.choices(errors, k=len(errors))
        bootstrap.append(
            math.sqrt(statistics.fmean(b for _, b in sample))
            - math.sqrt(statistics.fmean(a for a, _ in sample))
        )
        permuted = sum(d if rng.getrandbits(1) else -d for d in differences)
        extreme += abs(permuted) >= observed - 1e-12
    quantiles = statistics.quantiles(bootstrap, n=40, method="inclusive")
    p_value = (extreme + 1) / (draws + 1)
    return {
        "n_diagrams": len(ids),
        "baseline_rmse": math.sqrt(baseline_mse),
        "candidate_rmse": math.sqrt(candidate_mse),
        "rmse_change": math.sqrt(candidate_mse) - math.sqrt(baseline_mse),
        "rmse_change_ci95": [quantiles[0], quantiles[-1]],
        "mean_squared_error_change": statistics.fmean(differences),
        "paired_permutation_p_two_sided": p_value,
        "p_bonferroni_five_candidates": min(1.0, p_value * 5),
        "resamples": draws,
        "seed": 20261112,
        "note": (
            "Candidate minus baseline; negative favors candidate. Resample whole diagrams, "
            "keeping all eight dimensions together. The two-sided permutation test swaps "
            "baseline/candidate within diagrams and tests mean squared-error change. "
            "P-values and intervals are exploratory after validation-based prompt selection; "
            "the five-candidate Bonferroni value does not remove adaptive tuning bias. "
            "These intervals do not include judge rerun variability or rater uncertainty."
        ),
    }


def snapshot(args):
    review = read_json(args.run / "review.json")
    if not review["approval"]:
        raise ValueError("Approve the image review before snapshotting edited prompts.")
    path = args.run / "versions" / f"{args.version}.json"
    if path.exists():
        raise FileExistsError(path)
    write_json(path, {
        "id": args.version, "description": args.description, "created_at": now(),
        "prompts": BEAUTY_V6_PROMPTS, "response_instruction": RESPONSE_INSTRUCTION,
        "edits": args.edits, "training_evidence_ids": review["approval"]["training_ids"],
    })
    refresh(args.run)
    print(f"Frozen current evaluator prompts at {path.resolve()}", flush=True)


def state(run):
    manifest = read_json(run / "manifest.json")
    humans = {row["id"]: row for row in read_json(run / "human_labels_corrected.json")["diagrams"][:200]}
    original = {row["id"]: row for row in read_json(run / "human_labels_original.json")["diagrams"][:200]}
    review = read_json(run / "review.json")
    baseline = records(run)
    ranked = []
    for image_id in manifest["training_ids"]:
        if not all((image_id, d) in baseline for d in DIMENSIONS):
            continue
        deltas = {d: standardized(baseline[image_id, d]["score"]) - humans[image_id]["scores"][d]
                  for d in DIMENSIONS}
        ranked.append({
            "id": image_id, "largest_delta": max(abs(delta) for delta in deltas.values()),
            "worst_dimension": max(deltas, key=lambda d: abs(deltas[d])),
            "deltas": deltas,
            "judge": {d: {**baseline[image_id, d],
                          "standardized": standardized(baseline[image_id, d]["score"])}
                      for d in DIMENSIONS},
        })
    ranked.sort(key=lambda row: (-row["largest_delta"], row["id"]))
    complete = len(baseline) == 200 * len(DIMENSIONS)
    training = ranked[:20]
    versions = []
    for path in sorted((run / "versions").glob("*.json")):
        version = read_json(path)
        saved = baseline if version["id"] == "baseline" else records(run, version["id"])
        versions.append({
            **version, "completed_calls": len(saved),
            "completed_training_calls": sum(
                (image_id, d) in saved for image_id in manifest["training_ids"] for d in DIMENSIONS
            ),
            "mini_validation": validation_metrics(manifest["mini_validation_ids"], humans, saved),
            "validation": validation_metrics(manifest["validation_ids"], humans, saved),
        })
    for row in training:
        row.update({
            "scores": {d: humans[row["id"]]["scores"][d] for d in DIMENSIONS},
            "original_scores": {d: original[row["id"]]["scores"][d] for d in DIMENSIONS},
            "review": review["images"].get(row["id"]),
        })
    mini = [{
        "id": image_id, "scores": {d: humans[image_id]["scores"][d] for d in DIMENSIONS},
        "original_scores": {d: original[image_id]["scores"][d] for d in DIMENSIONS},
        "review": review["images"].get(image_id),
    } for image_id in manifest["mini_validation_ids"]]
    return {
        "manifest": manifest, "baseline_complete": complete,
        "training": training, "mini_validation": mini, "versions": versions,
        "review": review,
        "corrections": [{
            "id": image_id, "dimension": d,
            "original": original[image_id]["scores"][d], "corrected": humans[image_id]["scores"][d],
        } for image_id in manifest["training_ids"] + manifest["validation_ids"] for d in DIMENSIONS
            if original[image_id]["scores"][d] != humans[image_id]["scores"][d]],
    }


def refresh(run):
    data = state(run)
    humans = {row["id"]: row for row in read_json(run / "human_labels_corrected.json")["diagrams"][:200]}
    baseline = records(run)
    comparisons = {}
    for version in data["versions"]:
        if version["id"] == "baseline":
            continue
        saved = records(run, version["id"])
        comparisons[version["id"]] = {
            subset: paired_comparison(data["manifest"][key], humans, baseline, saved)
            for subset, key in (
                ("mini_validation", "mini_validation_ids"), ("validation", "validation_ids")
            )
        }
    data["paired_comparisons_against_baseline"] = comparisons
    write_json(run / "dashboard_data.json", data)
    write_json(run / "metrics.json", {
        "updated_at": now(), "scale": data["manifest"]["scaling"],
        "note": "Validation metrics only. Overall Pearson pools image/dimension pairs. Undefined Pearson is null.",
        "versions": [{
            "id": v["id"], "mini_validation": v["mini_validation"], "validation": v["validation"],
        } for v in data["versions"]],
        "paired_comparisons_against_baseline": comparisons,
    })
    return data


def score(args):
    manifest = read_json(args.run / "manifest.json")
    version = read_json(args.run / "versions" / f"{args.version}.json")
    if args.version != "baseline" and not read_json(args.run / "review.json")["approval"]:
        raise ValueError("Approve the image review before scoring edited prompts.")
    subsets = {
        "all": manifest["training_ids"] + manifest["validation_ids"],
        "training": manifest["training_ids"], "mini": manifest["mini_validation_ids"],
        "validation": manifest["validation_ids"],
    }
    saved = records(args.run, args.version)
    jobs = [(image, d) for image in manifest["images"] if image["id"] in subsets[args.subset]
            for d in DIMENSIONS if (image["id"], d) not in saved]
    if not jobs:
        refresh(args.run)
        print("All requested scores are already saved in this fresh run.", flush=True)
        return
    spec = importlib.util.spec_from_file_location("fresh_beauty_backend", args.backend)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load LLMAPI backend: {args.backend}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = module.Backend(model=MODEL)
    lock = threading.Lock()
    completed = 0
    print(f"Scoring {len(jobs)} new image/dimension pairs with {MODEL}.", flush=True)

    def evaluate(job):
        nonlocal completed
        image, dimension = job
        result = judge_beauty_dimension(
            backend, Path(manifest["image_directory"]) / image["filename"], dimension, version["prompts"],
        )
        metadata = backend.last.metadata
        if metadata["requested_model"] != MODEL:
            raise RuntimeError(f"Unexpected requested model: {metadata['requested_model']}")
        record = {
            "id": image["id"], "dimension": dimension, "version": args.version,
            "scored_at": now(), **result.to_dict(), "metadata": metadata,
        }
        with lock:
            with (args.run / f"{args.version}_scores.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, allow_nan=False) + "\n")
            completed += 1
            if completed == 1 or completed % 20 == 0:
                print(f"{completed}/{len(jobs)} saved; {image['id']} {dimension} = {result.score}; "
                      f"response model: {metadata['response_model']}", flush=True)

    # Check authentication and one real response before submitting the remaining calls.
    evaluate(jobs[0])
    executor = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = {executor.submit(evaluate, job): job for job in jobs[1:]}
        for future in as_completed(futures):
            image, dimension = futures[future]
            try:
                future.result()
            except Exception as error:
                print(f"FAILED {image['id']} {dimension}: {type(error).__name__}: {error}", flush=True)
                raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        refresh(args.run)
    print("Requested scoring complete; scores and validation metrics saved.", flush=True)


class ReviewServer(ThreadingHTTPServer):
    def __init__(self, address, run):
        super().__init__(address, ReviewHandler)
        self.run = run
        self.edit_lock = threading.Lock()


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer

    def send(self, value, status=200, mime="application/json"):
        data = json.dumps(value, allow_nan=False).encode() if mime == "application/json" else value
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self.send(Path(__file__).with_name("dashboard.html").read_bytes(), mime="text/html; charset=utf-8")
        elif path == "/api/state":
            with self.server.edit_lock:
                self.send(refresh(self.server.run))
        elif path.startswith("/image/"):
            manifest = read_json(self.server.run / "manifest.json")
            image = next((r for r in manifest["images"] if r["id"] == path.removeprefix("/image/")), None)
            if image is None:
                self.send({"error": "Image is not in this run's first 200 images."}, 404)
                return
            image_path = Path(manifest["image_directory"]) / image["filename"]
            self.send(image_path.read_bytes(), mime=mimetypes.guess_type(image_path.name)[0] or "image/png")
        elif path.startswith("/download/"):
            name = path.removeprefix("/download/")
            allowed = {
                "manifest.json", "human_labels_original.json", "human_labels_corrected.json",
                "review.json", "metrics.json", "dashboard_data.json", "baseline_scores.jsonl",
            }
            target = self.server.run / name
            if name not in allowed or not target.is_file():
                self.send({"error": "Download not found."}, 404)
                return
            self.send(target.read_bytes(), mime="application/octet-stream")
        else:
            self.send({"error": "Not found."}, 404)

    def do_POST(self):
        try:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with self.server.edit_lock:
                if self.path == "/api/review":
                    self.save_review(payload)
                elif self.path == "/api/approve":
                    self.approve()
                else:
                    self.send({"error": "Not found."}, 404)
                    return
                self.send(refresh(self.server.run))
        except (ValueError, KeyError, TypeError) as error:
            self.log_error("Review input rejected: %s", error)
            self.send({"error": str(error)}, 400)
        except OSError as error:
            self.log_error("Review could not be saved: %s", error)
            self.send({"error": f"Could not save review: {error}"}, 500)

    def save_review(self, payload):
        data = state(self.server.run)
        allowed = {r["id"] for r in data["training"] + data["mini_validation"]}
        image_id = payload["id"]
        if image_id not in allowed:
            raise ValueError("Review only the displayed training or fixed mini-validation images.")
        if set(payload["scores"]) != set(DIMENSIONS):
            raise ValueError("Submit exactly the eight human beauty scores.")
        for value in payload["scores"].values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 10:
                raise ValueError("Human scores must be numbers from 1 to 10.")
        if not isinstance(payload.get("note", ""), str):
            raise ValueError("The review note must be text.")
        if payload.get("confirmed") is not True:
            raise ValueError("Confirm that you have visually checked this image.")
        labels_path = self.server.run / "human_labels_corrected.json"
        labels = read_json(labels_path)
        row = next(r for r in labels["diagrams"] if r["id"] == image_id)
        row["scores"].update(payload["scores"])
        write_json(labels_path, labels)
        review = read_json(self.server.run / "review.json")
        review["images"][image_id] = {
            "reviewed_at": now(), "scores": payload["scores"], "note": payload.get("note", ""),
        }
        review["approval"] = None
        write_json(self.server.run / "review.json", review)

    def approve(self):
        data = state(self.server.run)
        if not data["baseline_complete"]:
            raise ValueError("Finish the fresh 200-image baseline before approval.")
        expected = [r["id"] for r in data["training"] + data["mini_validation"]]
        missing = [image_id for image_id in expected if image_id not in data["review"]["images"]]
        if missing:
            raise ValueError("Visually review and save these images first: " + ", ".join(missing))
        review = data["review"]
        review["approval"] = {
            "approved_at": now(),
            "training_ids": [r["id"] for r in data["training"]],
            "mini_validation_ids": data["manifest"]["mini_validation_ids"],
            "scope": "Image labels verified. Each proposed prompt edit still requires approval in chat.",
        }
        write_json(self.server.run / "review.json", review)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("prepare")
    init.add_argument("--labels", type=Path, required=True)
    init.add_argument("--images", type=Path, required=True)
    init.add_argument("--run", type=Path, required=True)
    init.add_argument("--seed", type=int, default=20261111)
    freezing = subparsers.add_parser("snapshot")
    freezing.add_argument("--run", type=Path, required=True)
    freezing.add_argument("--version", required=True)
    freezing.add_argument("--description", required=True)
    freezing.add_argument("--edits", nargs="+", required=True)
    scoring = subparsers.add_parser("score")
    scoring.add_argument("--run", type=Path, required=True)
    scoring.add_argument("--backend", type=Path, default=ROOT / "local" / "beauty_alignment_backend.py")
    scoring.add_argument("--version", default="baseline")
    scoring.add_argument("--subset", choices=("all", "training", "mini", "validation"), default="all")
    scoring.add_argument("--workers", type=int, default=8)
    report = subparsers.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    server = subparsers.add_parser("serve")
    server.add_argument("--run", type=Path, required=True)
    server.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "snapshot":
        snapshot(args)
    elif args.command == "score":
        score(args)
    elif args.command == "report":
        refresh(args.run)
    else:
        httpd = ReviewServer(("127.0.0.1", args.port), args.run)
        print(f"Review dashboard: http://127.0.0.1:{httpd.server_port}", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
