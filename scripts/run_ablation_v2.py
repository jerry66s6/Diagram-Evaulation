#!/usr/bin/env python3
"""Versioned sensitivity experiment: original, partial, and severe SVG edits.

Default: run SVG checks and write a standalone report. --live: also score all
eleven images with the configured GPT judge and show each source separately.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import csv
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from diagram_correctness.models import (ComponentVerdict, EvaluationReport, Issue, MetricScore,
    PresencePolicy, Severity, Dimension, Verdict, aggregate_panel, component_agreement)
from diagram_correctness.deterministic import GeometryConfig
from diagram_correctness.rubric import load_config, load_rubric, renormalize_weights, severity_frequency_weights
from diagram_correctness.sensitivity import SensitivityJudge
from diagram_correctness.svg import parse_svg
from run_dimension_ablation import DIMENSIONS, EXAMPLE, INTERVENTIONS, mutate_svg

DEFAULT_OUTPUT = EXAMPLE / "ablation-v2"
SEVERE = {
    "layout": "Reverse the four internal encoder stages and reroute arrows to preserve the original directed connections. All 6 required internal order pairs are reversed.",
    "connectivity": "Remove all 10 expected arrows. Keep every component and label.",
    "presence": "Remove all 11 expected components and their labels. Keep the title and the now-dangling arrows; spillover into other dimensions is expected.",
    "details": "Replace all 9 computational-node labels with TBD. Keep the group label, repetition badge, geometry, and all 10 arrows. Roles may be recoverable from the complete graph.",
    "legibility": "Set all 12 text elements to 1 px. Keep their words, shapes, and arrows. Text remains in the SVG source but becomes visually unreadable.",
}
NODE_IDS = {"input": "input-label", "token": "token-label", "position": "position-label", "add": "add-label",
            "attention": "attention-label", "norm1": "norm1-label", "ffn": "ffn-label", "norm2": "norm2-label", "output": "output-label"}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def load_local_key():
    """Only read the explicitly offered project .env key; never execute its text."""
    if os.environ.get("OPENAI_API_KEY"):
        return True
    dotenv = ROOT / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text().splitlines():
            name, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and name.strip() == "OPENAI_API_KEY":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if value and value not in {"your_key", "your_api_key_here", "..."}:
                    os.environ["OPENAI_API_KEY"] = value
                    return True
    return False


def mutate_severe(source, dimension, destination):
    tree = ET.parse(source)
    root = tree.getroot()
    elements = {e.get("id"): e for e in root.iter() if e.get("id")}
    before = {key: ET.tostring(element, encoding="unicode") for key, element in elements.items()}
    if dimension == "layout":
        for node, new_y in [("attention", 695), ("norm1", 595), ("ffn", 501), ("norm2", 401)]:
            shape, label = elements[node + "-s"], elements[NODE_IDS[node] + "-s"]
            shift = new_y - float(shape.get("y"))
            shape.set("y", str(new_y))
            label.set("y", str(float(label.get("y")) + shift))
        pairs = [("s4", "add", "attention"), ("s5", "attention", "norm1"), ("s6", "add", "norm1"),
                 ("s7", "norm1", "ffn"), ("s8", "ffn", "norm2"), ("s9", "norm1", "norm2"), ("s10", "norm2", "output")]
        def port(name):
            element = elements[name + "-s"]
            if name == "add":
                return float(element.get("cx")) + float(element.get("r")), float(element.get("cy"))
            return float(element.get("x")) + float(element.get("width")), float(element.get("y")) + float(element.get("height")) / 2
        for index, (key, first, last) in enumerate(pairs):
            a, b = port(first), port(last)
            lane = 940 + index * 18
            element = elements[key]
            element.tag = "{http://www.w3.org/2000/svg}polyline"
            for attribute in ("x1", "y1", "x2", "y2"):
                element.attrib.pop(attribute, None)
            element.set("points", f"{a[0]},{a[1]} {lane},{a[1]} {lane},{b[1]} {b[0]},{b[1]}")
            element.set("fill", "none")
    elif dimension == "connectivity":
        for index in range(1, 11):
            root.remove(elements[f"s{index}"])
    elif dimension == "presence":
        for key in [*[node + "-s" for node in NODE_IDS], *[label + "-s" for label in NODE_IDS.values()],
                    "encoder-s", "encoder-label-s", "badge-s", "repeat-s"]:
            root.remove(elements[key])
    elif dimension == "details":
        for label in NODE_IDS.values():
            elements[label + "-s"].text = "TBD"
    elif dimension == "legibility":
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] == "text":
                element.set("font-size", "1")
    else:
        raise ValueError(dimension)
    tree.write(destination, encoding="unicode")
    after = {e.get("id"): ET.tostring(e, encoding="unicode") for e in root.iter() if e.get("id")}
    return [{"element_id": key, "before": value, "after": after.get(key)} for key, value in before.items() if value != after.get(key)]


def report_for(metrics, criteria, inventory, settings, history):
    dimensions = aggregate_panel(metrics, criteria, reducer="minimum")
    if not dimensions:
        raise ValueError("No measured dimensions; cannot create an overall score")
    weights = renormalize_weights(severity_frequency_weights(history, settings.get("severity_multipliers"), settings.get("fallback_frequencies")), set(dimensions))
    return EvaluationReport(
        correctness=sum(weights[d] * value.score for d, value in dimensions.items()),
        dimensions=dimensions, weights=weights, metrics=metrics, inventory=inventory,
        metadata={"version": "experimental-v2", "within_dimension": "minimum measurable criterion",
                  "between_judges": "median per matching criterion", "component_verdicts": component_agreement(metrics)},
    ).to_dict()


def restore_metrics(report):
    return [MetricScore(
        criterion_id=m["criterion_id"], dimension=Dimension(m["dimension"]), judge=m["judge"],
        expected_count=m["expected_count"], detected_issues=m["detected_issues"], measurable=m["measurable"], notes=m["notes"],
        issues=[Issue(i["description"], Severity(i["severity"]), i["element_ids"]) for i in m["issues"]],
        component_verdicts=[ComponentVerdict(**{**v, "verdict": Verdict(v["verdict"])}) for v in m.get("component_verdicts", [])],
    ) for m in report["metrics"]]


def publish(output, summary, publish_to=None):
    write_json(output / "summary.json", summary)
    with (output / "scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate", "severity", "target", "judge", *DIMENSIONS, "overall"])
        for row in summary["rows"]:
            for channel, report in row["panels"].items():
                writer.writerow([row["id"], row["severity"], row["target"], channel,
                    *[100 * report["dimensions"][d]["score"] if report and d in report["dimensions"] else "" for d in DIMENSIONS],
                    100 * report["correctness"] if report else ""])
    payload = {**summary, "images": {row["id"]: "data:image/svg+xml;base64," + base64.b64encode((output / "diagrams" / (row["id"] + ".svg")).read_bytes()).decode() for row in summary["rows"]}}
    template = (ROOT / "scripts/ablation_v2_report.html").read_text()
    html = template.replace("__EXPERIMENT_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"))
    (output / "index.html").write_text(html)
    if publish_to:
        publish_to.parent.mkdir(parents=True, exist_ok=True)
        backup = publish_to.with_name("index.v1.html")
        if publish_to.exists() and not backup.exists():
            backup.write_bytes(publish_to.read_bytes())
        publish_to.write_text(html)


def run(output, live=False, publish_to=None):
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "diagrams"
    assets.mkdir(exist_ok=True)
    source = EXAMPLE / "candidate_strong.svg"
    description = (EXAMPLE / "description.md").read_text()
    settings = load_config(ROOT / "config/evaluator.json")
    criteria = load_rubric(ROOT / "config/rubric_v2.json")
    history = load_config(ROOT / settings["weight_history"])["issues"]
    inventory = load_config(EXAMPLE / "results/description_expectations.json")
    spatial = load_config(EXAMPLE / "spatial_requirements_v2.json")
    if sha(description.encode()) != spatial["provenance"]["source_sha256"]:
        raise ValueError("Description changed; review the frozen spatial expectations first")
    inventory["spatial_requirements"] = spatial
    protocol = {"version": "experimental-v2", "reducer": "minimum", "settings": settings,
                "rubric": load_config(ROOT / "config/rubric_v2.json"), "inventory": inventory, "history": history,
                "description_sha256": sha(description.encode()), "source_sha256": sha(source.read_bytes()),
                "code_hashes": {p.name: sha(p.read_bytes()) for p in [Path(__file__), ROOT / "scripts/run_dimension_ablation.py", *sorted((ROOT / "src/diagram_correctness").glob("*.py"))]}}
    signature = sha(json.dumps(protocol, sort_keys=True).encode())
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and load_config(protocol_path) != protocol and list((output / "results").glob("*/gpt*.json")):
        raise ValueError("Protocol changed after GPT work. Use a fresh --output-dir; cached judgments must not be mixed.")
    write_json(protocol_path, protocol)
    write_json(output / "description_expectations.json", inventory)
    legacy = load_config(EXAMPLE / "ablation/summary.json")
    summary = {"version": "experimental-v2", "created_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": signature,
               "dimensions": DIMENSIONS, "model": settings["models"]["judge"], "description": description,
               "spatial_requirements": spatial, "legacy_svg_rows": legacy["rows"], "legacy_source_sha256": legacy["provenance"]["source_sha256"],
               "rows": [], "status": "svg_only", "planned_image_calls": 11,
               "limitations": ["One base diagram and two intervention strengths per dimension; this is an experimental sensitivity check, not a human-calibrated benchmark.",
                               "The minimum-criterion rule deliberately prevents compensation within a dimension. It is sensitive to noisy criteria and requires human validation.",
                               "Between-judge medians can still hide disagreement. Compare SVG, GPT, and Combined views; combined results require both judges.",
                               "Severe edits can affect other dimensions. Small SVG text is still readable to the source parser even when it is unreadable in the image.",
                               "The SVG parser approximates font bounds and connector geometry. Recovered graph roles require a unique complete correspondence; ambiguous cases fall back to the original matcher.",
                               "Spatial constraints are manually transcribed from the caption. Existing overall weights are unchanged and are not corpus calibrated."]}
    mutations = {}
    candidates = [("baseline", None, "original")] + [(f"{d}_{level}", d, level) for level in ["partial", "severe"] for d in DIMENSIONS]
    judge = SensitivityJudge(GeometryConfig(**settings["geometry"]), PresencePolicy.from_dict(settings.get("presence_policy")))
    for name, dimension, level in candidates:
        svg = assets / f"{name}.svg"
        if level == "original":
            svg.write_bytes(source.read_bytes())
            explanation = "Original diagram, unchanged."
        elif level == "partial":
            mutations[name] = mutate_svg(source, dimension, svg)
            explanation = INTERVENTIONS[dimension]["description"] + " " + INTERVENTIONS[dimension]["dose"] + "."
        else:
            mutations[name] = mutate_severe(source, dimension, svg)
            explanation = SEVERE[dimension]
        metrics = judge.evaluate(parse_svg(svg), criteria, inventory)
        svg_report = report_for(metrics, criteria, inventory, settings, history)
        directory = output / "results" / name
        write_json(directory / "svg.json", svg_report)
        row = {"id": name, "target": dimension, "severity": level, "description": explanation,
               "image_sha256": sha(svg.read_bytes()), "panels": {"svg": svg_report, "gpt": None, "combined": None}, "gpt_status": "not_run"}
        cached = directory / "gpt.json"
        if cached.exists():
            saved = load_config(cached)
            if saved.get("protocol_sha256") != signature or saved.get("image_sha256") != row["image_sha256"]:
                raise ValueError(f"Cached GPT judgment for {name} belongs to different inputs")
            row["panels"]["gpt"] = saved["report"]
            row["panels"]["combined"] = report_for(metrics + restore_metrics(saved["report"]), criteria, inventory, settings, history)
            row["gpt_status"] = "complete"
        summary["rows"].append(row)
    write_json(output / "mutations.json", mutations)
    publish(output, summary, publish_to)
    if not live:
        return summary, 0
    pending = [row for row in summary["rows"] if row["panels"]["gpt"] is None]
    if pending and not load_local_key():
        summary["status"] = "blocked_missing_api_key"
        for row in pending:
            row["gpt_status"] = "blocked_missing_api_key"
        publish(output, summary, publish_to)
        return summary, 2
    from diagram_correctness.vlm import OpenAIBackend, VLMJudge
    from diagram_correctness.pipeline import CorrectnessPipeline, PipelineConfig

    class RecordedBackend(OpenAIBackend):
        def __init__(self, model, directory):
            super().__init__(model)
            self.directory = directory

        def structured(self, prompt, images, schema, schema_name):
            write_json(self.directory / "gpt_request.json", {"model": self.model, "prompt": prompt, "schema": schema,
                "image_sha256": sha(Path(images[0]).read_bytes()), "created_at": datetime.now(timezone.utc).isoformat()})
            result = super().structured(prompt, images, schema, schema_name)
            write_json(self.directory / "gpt_response.json", result)
            return result

    failures = 0
    for row in pending:
        name = row["id"]
        directory = output / "results" / name
        print(f"GPT: {name}", flush=True)
        row["gpt_status"] = "running"
        summary["status"] = "running"
        publish(output, summary, publish_to)
        try:
            pipeline = CorrectnessPipeline(PipelineConfig(criteria=criteria, judges=[], expectation_extractor=None))
            png = pipeline._resolve_image(None, assets / f"{name}.svg", assets / f"{name}.png")
            gpt = VLMJudge(RecordedBackend(settings["models"]["judge"], directory), PresencePolicy.from_dict(settings.get("presence_policy")))
            # Filename, target dimension, severity, and baseline are absent from the prompt.
            metrics = gpt.evaluate(description, png, criteria, inventory)
            report = report_for(metrics, criteria, inventory, settings, history)
            write_json(directory / "gpt.json", {"protocol_sha256": signature, "image_sha256": row["image_sha256"], "report": report})
            row["panels"]["gpt"] = report
            row["panels"]["combined"] = report_for(restore_metrics(row["panels"]["svg"]) + metrics, criteria, inventory, settings, history)
            write_json(directory / "combined.json", row["panels"]["combined"])
            row["gpt_status"] = "complete"
        except Exception as error:
            failures += 1
            row["gpt_status"] = "error: " + type(error).__name__
            write_json(directory / "error.json", {"type": type(error).__name__, "status_code": getattr(error, "status_code", None)})
            print(f"  Failed: {type(error).__name__}", file=sys.stderr)
            if type(error).__name__ in {"AuthenticationError", "PermissionDeniedError", "NotFoundError", "RateLimitError", "APIConnectionError"}:
                summary["status"] = "incomplete"
                publish(output, summary, publish_to)
                return summary, 1
        publish(output, summary, publish_to)
    summary["status"] = "complete" if all(r["panels"]["gpt"] is not None for r in summary["rows"]) else "incomplete"
    publish(output, summary, publish_to)
    return summary, int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--prompt-key", action="store_true", help="Prompt locally for a hidden API key; implies --live.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--publish-current", action="store_true", help="Also update the existing localhost report's index.html, keeping index.v1.html.")
    args = parser.parse_args()
    if args.prompt_key:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key (hidden): ").strip()
    summary, code = run(args.output_dir.resolve(), live=args.live or args.prompt_key,
                        publish_to=EXAMPLE / "ablation/index.html" if args.publish_current else None)
    for row in summary["rows"]:
        report = row["panels"]["svg"]
        target = row["target"]
        value = report["dimensions"].get(target, {}).get("score") if target else report["correctness"]
        print(f"{row['id']:24s} SVG target/overall: {100 * value:.2f}  GPT: {row['gpt_status']}")
    print(f"Status: {summary['status']}\nReport: {args.output_dir.resolve() / 'index.html'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
