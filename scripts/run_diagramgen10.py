#!/usr/bin/env python3
"""Prepare ten original DiagramGen pairs; optionally run the existing GPT evaluator."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from datetime import datetime, timezone
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_OUTPUT = ROOT / "examples" / "diagramgen10"
MIRROR = "https://modelscope.cn/datasets/opendatalab-raiser/DiagramGen/resolve/master/"
INDEX_SHA256 = "6060c5099b4f23d4a1269b71929212995d42a73f26a035c8bdcce1d820383783"
SEED = "20260930"
STRATA = {"model architecture diagram": 4, "flowchart": 4, "directed graph": 2}
DIMENSIONS = ["layout", "connectivity", "presence", "details", "legibility"]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def fetch(relative: str, target: Path) -> None:
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(MIRROR + relative, timeout=90) as response:
        data = response.read()
    temporary = target.with_suffix(target.suffix + ".download")
    temporary.write_bytes(data)
    temporary.replace(target)


def prepare(output: Path) -> dict:
    index_path = output / "source" / "DiagramGeneration.json"
    fetch("DiagramGeneration.json", index_path)
    fetch("README.md", output / "source" / "README.md")
    if digest(index_path.read_bytes()) != INDEX_SHA256:
        raise ValueError("Dataset index differs from the inspected version; refusing to change the sample silently.")
    records = json.loads(index_path.read_text())
    chosen = []
    for category, count in STRATA.items():
        eligible = [row for row in records if row["category"] == category]
        chosen.extend(sorted(eligible, key=lambda row: digest(f"{SEED}:{row['id']}".encode()))[:count])
    if len(chosen) != 10 or len({row["id"] for row in chosen}) != 10:
        raise ValueError("Expected exactly ten distinct examples")
    examples = []
    for number, row in enumerate(chosen, 1):
        sample = output / "samples" / row["id"]
        sample.mkdir(parents=True, exist_ok=True)
        image_relative = row["images"][0].removeprefix("./")
        if image_relative != f"images/{row['id']}.png":
            raise ValueError("Unexpected image path in dataset")
        image_path = sample / "candidate.png"
        fetch(image_relative, image_path)
        image_bytes = image_path.read_bytes()
        if image_bytes[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a PNG: {row['id']}")
        width, height = struct.unpack(">II", image_bytes[16:24])
        description = row["expanded_query"]
        if not description.strip():
            raise ValueError(f"Empty expanded_query: {row['id']}")
        (sample / "description.txt").write_text(description)
        source_format = "dot" if row["reference"].lstrip().startswith(("digraph", "graph", "strict")) else "tex"
        (sample / f"reference.{source_format}").write_text(row["reference"])
        write_json(sample / "original_record.json", row)
        examples.append({
            "number": number, "id": row["id"], "category": row["category"],
            "query": row["query"], "description": description,
            "description_field": "expanded_query", "source_format": source_format,
            "image_url": MIRROR + image_relative, "image_sha256": digest(image_bytes),
            "description_sha256": digest(description.encode()),
            "image_size": [width, height], "sample_directory": f"samples/{row['id']}",
        })
        print(f"Prepared {number:02d}/10: {row['category']} — {width} × {height}", flush=True)
    manifest = {
        "dataset": "OpenRaiser/DiagramGen", "config": "DiagramGeneration", "split": "test",
        "canonical_url": "https://huggingface.co/datasets/OpenRaiser/DiagramGen",
        "download_mirror": "https://modelscope.cn/datasets/opendatalab-raiser/DiagramGen",
        "download_branch": "master", "index_sha256": INDEX_SHA256, "population_size": len(records),
        "selection": {"seed": SEED, "strata": STRATA,
                      "rule": "Within each category, ascending SHA256(seed + ':' + id); take the stated count. No score-based filtering."},
        "caption_policy": "Use expanded_query verbatim. Expectations are extracted from text only; neither reference code nor image is used in that stage.",
        "license_note": "Dataset card: Apache-2.0; underlying sources described as CC BY 4.0 or MIT. Original dataset card is retained under source/README.md.",
        "examples": examples,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Prepared inputs differ from the saved manifest; use a new output directory.")
    write_json(manifest_path, manifest)
    return manifest


def protocol() -> dict:
    settings = json.loads((ROOT / "config/evaluator.json").read_text())
    history = json.loads((ROOT / settings["weight_history"]).read_text())["issues"] if settings.get("weight_history") else []
    return {
        "mode": "GPT-only on original dataset PNGs", "settings": settings,
        "rubric": json.loads((ROOT / "config/rubric.json").read_text()), "issue_history": history,
        "deterministic_judge": False, "vfig_used": False,
        "code_sha256": {path.name: digest(path.read_bytes()) for path in sorted((ROOT / "src/diagram_correctness").glob("*.py"))},
        "planned_calls": {"expectation": 10, "judge": 10},
    }


def publish(output: Path, manifest: dict, run_protocol: dict) -> None:
    rows = []
    for example in manifest["examples"]:
        directory = output / "results" / example["id"]
        report_file = directory / "report.json"
        state_file = directory / "state.json"
        row = dict(example)
        row["state"] = json.loads(state_file.read_text()) if state_file.exists() else {"status": "prepared", "message": "Awaiting evaluation"}
        row["report"] = json.loads(report_file.read_text()) if report_file.exists() else None
        rows.append(row)
    summary = {"updated_utc": datetime.now(timezone.utc).isoformat(), "protocol": run_protocol,
               "manifest": {key: value for key, value in manifest.items() if key != "examples"}, "examples": rows}
    write_json(output / "summary.json", summary)
    with (output / "scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["number", "id", "category", "status", "overall_0_100", *[f"{d}_0_100" for d in DIMENSIONS]])
        for row in rows:
            report = row["report"]
            scores = [100 * report["correctness"], *[100 * report["dimensions"][d]["score"] if d in report["dimensions"] else "" for d in DIMENSIONS]] if report else [""] * 6
            writer.writerow([row["number"], row["id"], row["category"], row["state"]["status"], *scores])
    # The HTML contains both the original PNG bytes and all captions/results.
    # There are no fetches, external fonts, libraries, or server dependencies.
    for row in rows:
        image_bytes = (output / row["sample_directory"] / "candidate.png").read_bytes()
        row["image_data"] = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
    template = (ROOT / "scripts/diagramgen10_report.html").read_text()
    payload = json.dumps(summary, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    (output / "index.html").write_text(template.replace("__REPORT_DATA__", payload))


def run_live(output: Path, manifest: dict, run_protocol: dict) -> int:
    pending = [row for row in manifest["examples"] if not (output / "results" / row["id"] / "report.json").exists()]
    if not pending:
        print("All ten cached reports are already complete.")
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        for row in pending:
            write_json(output / "results" / row["id"] / "state.json", {
                "status": "blocked", "message": "OPENAI_API_KEY is not configured locally. No model call was made for this example.",
            })
        publish(output, manifest, run_protocol)
        print("BLOCKED: OPENAI_API_KEY is not configured. Ten image/description pairs are ready; no scores were generated.", file=sys.stderr)
        return 2
    from diagram_correctness.models import PresencePolicy
    from diagram_correctness.pipeline import CorrectnessPipeline, PipelineConfig
    from diagram_correctness.rubric import load_rubric
    from diagram_correctness.vlm import ExpectationExtractor, OpenAIBackend, VLMJudge

    class RecordedBackend(OpenAIBackend):
        def __init__(self, model: str, directory: Path):
            super().__init__(model)
            self.directory = directory

        def structured(self, prompt, images, schema, schema_name):
            write_json(self.directory / f"{schema_name}.request.json", {
                "model": self.model, "prompt": prompt, "schema": schema,
                "images": [{"filename": Path(image).name, "sha256": digest(Path(image).read_bytes())} for image in images],
            })
            response = super().structured(prompt, images, schema, schema_name)
            write_json(self.directory / f"{schema_name}.response.json", response)
            return response

    settings = run_protocol["settings"]
    criteria = load_rubric(ROOT / "config/rubric.json")
    failures = 0
    for row in pending:
        directory = output / "results" / row["id"]
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Scoring {row['number']:02d}/10: {row['id']}", flush=True)
        state_path = directory / "state.json"
        write_json(state_path, {"status": "running", "message": "Evaluating against the source description"})
        publish(output, manifest, run_protocol)
        try:
            extractor = ExpectationExtractor(RecordedBackend(settings["models"]["expectation"], directory))
            inventory_path = directory / "inventory.json"
            if inventory_path.exists():
                inventory = json.loads(inventory_path.read_text())
            else:
                inventory = extractor.extract(row["description"], criteria)
                write_json(inventory_path, inventory)
            judge = VLMJudge(RecordedBackend(settings["models"]["judge"], directory), PresencePolicy.from_dict(settings.get("presence_policy")))
            pipeline = CorrectnessPipeline(PipelineConfig(
                criteria=criteria, judges=[judge], expectation_extractor=extractor,
                issue_history=run_protocol["issue_history"],
                severity_multipliers=settings.get("severity_multipliers"),
                fallback_frequencies=settings.get("fallback_frequencies"),
            ))
            report = pipeline.run(row["description"], candidate_image=output / row["sample_directory"] / "candidate.png", inventory=inventory, run_deterministic=False)
            write_json(directory / "report.json", report.to_dict())
            write_json(state_path, {"status": "complete", "message": "Scored by the configured GPT judge"})
            print(f"  Overall: {report.correctness * 100:.2f}/100", flush=True)
        except Exception as error:
            failures += 1
            # Do not persist SDK messages that might contain credential fragments.
            write_json(state_path, {"status": "error", "message": f"Evaluation failed ({type(error).__name__}). No final score for this example."})
            print(f"  Failed: {type(error).__name__}", file=sys.stderr)
            if type(error).__name__ in {"AuthenticationError", "PermissionDeniedError", "NotFoundError"}:
                print("Stopping because credentials or the configured model are unavailable.", file=sys.stderr)
                publish(output, manifest, run_protocol)
                return 1
        publish(output, manifest, run_protocol)
    return int(bool(failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--live", action="store_true", help="Run the configured GPT expectation extractor and image judge (normally 20 calls).")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    manifest = prepare(output)
    current_protocol = protocol()
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != current_protocol:
        existing_work = [*output.glob("results/*/report.json"), *output.glob("results/*/*.request.json")]
        if existing_work:
            raise ValueError("Evaluator configuration or code changed. Use a new --output-dir to avoid mixing runs.")
        print("Updating the prepared protocol; no previous model requests or results exist.")
    write_json(protocol_path, current_protocol)
    publish(output, manifest, current_protocol)
    code = run_live(output, manifest, current_protocol) if args.live else 0
    print(f"Self-contained report: {output / 'index.html'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
