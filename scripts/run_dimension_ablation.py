#!/usr/bin/env python3
"""Generate real SVG interventions, evaluate them, and write a portable HTML report.

The default run needs only Python's standard library. --live additionally needs
the project's normal dependencies and OPENAI_API_KEY. Neither mode assigns scores.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagram_correctness.deterministic import DeterministicJudge, GeometryConfig
from diagram_correctness.models import (
    EvaluationReport, PresencePolicy, aggregate_panel, component_agreement,
)
from diagram_correctness.rubric import load_config, load_rubric, renormalize_weights, severity_frequency_weights
from diagram_correctness.svg import parse_svg

EXAMPLE = ROOT / "examples" / "transformer"
DIMENSIONS = ["layout", "connectivity", "presence", "details", "legibility"]
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

INTERVENTIONS = {
    "layout": {
        "title": "Swap two processing stages",
        "description": "Exchange the vertical positions of Self-Attention and the Feed-Forward Network. Reroute their incident arrows to preserve the original source and target roles.",
        "dose": "2 stages exchanged; 2 of 10 expected relations now run upward",
    },
    "connectivity": {
        "title": "Remove three arrows",
        "description": "Remove Positional Encoding → Add and both residual/skip arrows. Keep every component and label in its original position.",
        "dose": "3 of 10 arrows removed",
    },
    "presence": {
        "title": "Remove two components",
        "description": "Remove the Positional Encoding box and label, and the N× badge and label. Leave all arrows in place, including the now-dangling positional arrow.",
        "dose": "2 of 11 expected components removed",
    },
    "details": {
        "title": "Replace three meaningful labels",
        "description": "Change Token Embeddings to Embeddings, Multi-Head Self-Attention to Layer, and Position-wise Feed-Forward Network to TBD. Keep all shapes, text positions, fonts, and arrows.",
        "dose": "3 of 11 component labels made incomplete or generic",
    },
    "legibility": {
        "title": "Shrink four labels",
        "description": "Set the Input Tokens, Token Embeddings, Self-Attention, and Feed-Forward labels to 8 px, below the configured 12 px minimum. Preserve their wording and positions.",
        "dose": "4 of 12 text elements reduced to 8 px",
    },
}


def mutate_svg(source: Path, target: str, destination: Path) -> list[dict]:
    tree = ET.parse(source)
    root = tree.getroot()
    by_id = {element.get("id"): element for element in root.iter() if element.get("id")}
    before = {key: ET.tostring(value, encoding="unicode") for key, value in by_id.items()}

    if target == "layout":
        for key, displacement in (("attention-s", 194), ("attention-label-s", 194), ("ffn-s", -194), ("ffn-label-s", -194)):
            element = by_id[key]
            element.set("y", str(float(element.get("y")) + displacement))
        # Keep the ten directed component pairs intact. Side lanes avoid drawing
        # rerouted arrows through the intermediate boxes.
        routes = {
            "s4": "581,286 960,286 960,626 780,626",
            "s5": "780,642 980,642 980,529 715,529",
            "s7": "715,529 940,529 940,432 780,432",
            "s8": "780,449 1000,449 1000,723 715,723",
        }
        for key, points in routes.items():
            element = by_id[key]
            element.tag = f"{{{SVG_NS}}}polyline"
            for attr in ("x1", "y1", "x2", "y2"):
                element.attrib.pop(attr, None)
            element.set("points", points)
            element.set("fill", "none")
    elif target == "connectivity":
        for key in ("s3", "s6", "s9"):
            root.remove(by_id[key])
    elif target == "presence":
        for key in ("position-s", "position-label-s", "badge-s", "repeat-s"):
            root.remove(by_id[key])
    elif target == "details":
        for key, label in (("token-label-s", "Embeddings"), ("attention-label-s", "Layer"), ("ffn-label-s", "TBD")):
            by_id[key].text = label
    elif target == "legibility":
        for key in ("input-label-s", "token-label-s", "attention-label-s", "ffn-label-s"):
            by_id[key].set("font-size", "8")
    else:
        raise ValueError(f"Unknown intervention: {target}")

    tree.write(destination, encoding="unicode")
    after = {element.get("id"): ET.tostring(element, encoding="unicode") for element in root.iter() if element.get("id")}
    return [
        {"element_id": key, "before": value, "after": after.get(key)}
        for key, value in before.items() if value != after.get(key)
    ]


def deterministic_report(svg: Path, inventory: dict, settings: dict, criteria: list, history: list) -> dict:
    judge = DeterministicJudge(GeometryConfig(**settings["geometry"]), PresencePolicy.from_dict(settings.get("presence_policy")))
    metrics = judge.evaluate(parse_svg(svg), criteria, inventory)
    dimensions = aggregate_panel(metrics, criteria)
    weights = renormalize_weights(severity_frequency_weights(history, settings.get("severity_multipliers"), settings.get("fallback_frequencies")), set(dimensions))
    agreement = component_agreement(metrics)
    return EvaluationReport(
        correctness=sum(weights[key] * value.score for key, value in dimensions.items()),
        dimensions=dimensions, weights=weights, metrics=metrics, inventory=inventory,
        metadata={"vlm_judges": [], "deterministic_judge": True, "expectations_from": "frozen description-derived inventory", "component_verdicts": agreement},
    ).to_dict()


def summarize(reports: dict[str, dict]) -> list[dict]:
    baseline = reports["baseline"]
    rows = []
    for name, report in reports.items():
        scores = {dimension: report["dimensions"].get(dimension, {}).get("score") for dimension in DIMENSIONS}
        delta = {
            dimension: None if score is None or dimension not in baseline["dimensions"] else score - baseline["dimensions"][dimension]["score"]
            for dimension, score in scores.items()
        }
        rows.append({
            "id": name, "target": None if name == "baseline" else name,
            "label": "Original baseline" if name == "baseline" else name.capitalize() + " degraded",
            "scores": scores, "delta": delta, "overall": report["correctness"],
            "overall_delta": report["correctness"] - baseline["correctness"],
            "criteria": [{"id": metric["criterion_id"], "judge": metric["judge"], "expected": metric["expected_count"], "issues": metric["detected_issues"], "score": metric["score"], "evidence": metric["issues"]} for metric in report["metrics"]],
            "intervention": INTERVENTIONS.get(name),
        })
    return rows


def write_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(output: Path, live: bool = False) -> dict:
    if live and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --live. Omit --live to run deterministic SVG checks.")
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "diagrams"
    assets.mkdir(exist_ok=True)
    settings = load_config(ROOT / "config/evaluator.json")
    criteria = load_rubric(ROOT / "config/rubric.json")
    history = load_config(ROOT / settings["weight_history"])["issues"]
    inventory_source = EXAMPLE / "results/description_expectations.json"
    inventory = load_config(inventory_source)
    source = EXAMPLE / "candidate_strong.svg"
    (assets / "baseline.svg").write_bytes(source.read_bytes())
    mutations = {}
    for dimension in DIMENSIONS:
        mutations[dimension] = mutate_svg(source, dimension, assets / f"{dimension}.svg")
    write_json(output / "mutations.json", mutations)
    write_json(output / "description_expectations.json", inventory)

    pipeline = None
    if live:
        from run_transformer_examples import build_pipeline, render_svg
        pipeline = build_pipeline(offline=False)

    reports = {}
    for name in ["baseline", *DIMENSIONS]:
        svg = assets / f"{name}.svg"
        if pipeline is not None:
            png = assets / f"{name}.png"
            render_svg(svg, png)
            report = pipeline.run(description=(EXAMPLE / "description.md").read_text(), candidate_image=png, candidate_svg=svg, inventory=inventory).to_dict()
        else:
            report = deterministic_report(svg, inventory, settings, criteria, history)
        reports[name] = report
        write_json(output / f"{name}.json", report)

    rows = summarize(reports)
    historical = load_config(EXAMPLE / "results/summary.json")
    historical_baseline = next(row for row in historical["results"] if row["candidate"] == "candidate_strong")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": f"live {settings['models']['judge']} + deterministic SVG" if live else "deterministic SVG only",
        "live": live, "dimensions": DIMENSIONS, "rows": rows,
        "weights": reports["baseline"]["weights"],
        "historical": {"mode": historical["mode"], "baseline": historical_baseline, "used_for_deltas": False},
        "provenance": {
            "source_svg": str(source.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "inventory_source": str(inventory_source.relative_to(ROOT)),
            "inventory_sha256": hashlib.sha256(inventory_source.read_bytes()).hexdigest(),
            "settings": settings,
            "rubric": load_config(ROOT / "config/rubric.json"),
            "issue_history": history,
            "scoring_source_sha256": {name: hashlib.sha256((ROOT / "src/diagram_correctness" / name).read_bytes()).hexdigest() for name in ("deterministic.py", "models.py", "rubric.py", "svg.py", "pipeline.py", "vlm.py")},
        },
        "limitations": [
            "One base diagram and one fixed intervention per dimension; no statistical uncertainty or generalization claim.",
            "Intervention sizes differ. Drops cannot rank intrinsic dimension sensitivity.",
            "Off-diagonal changes may reflect real structural dependence or evaluator coupling; the matrix alone cannot distinguish them.",
            "Overall weights are the repository's sample issue-history weights, not corpus-calibrated weights.",
        ],
    }
    write_json(output / "summary.json", summary)
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["variant", "target", *DIMENSIONS, "overall", *[f"delta_{dimension}" for dimension in DIMENSIONS], "delta_overall"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"variant": row["id"], "target": row["target"], **row["scores"], "overall": row["overall"], **{f"delta_{key}": value for key, value in row["delta"].items()}, "delta_overall": row["overall_delta"]})
    payload = {**summary, "images": {name: "data:image/svg+xml;base64," + base64.b64encode((assets / f"{name}.svg").read_bytes()).decode() for name in reports}}
    template = (ROOT / "scripts/dimension_ablation_report.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(template.replace("__EXPERIMENT_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Re-score all six diagrams with GPT plus SVG checks (requires OPENAI_API_KEY).")
    parser.add_argument("--output-dir", type=Path, help="Default: examples/transformer/ablation, or ablation-live for --live.")
    args = parser.parse_args()
    output = (args.output_dir or EXAMPLE / ("ablation-live" if args.live else "ablation")).resolve()
    summary = run(output, live=args.live)
    print(summary["mode"])
    print("Variant                Layout  Connect  Presence  Details  Legible  Overall")
    for row in summary["rows"]:
        values = [row["scores"][key] for key in DIMENSIONS] + [row["overall"]]
        print(f"{row['label']:23}" + " ".join(f"{100 * value:7.2f}" if value is not None else "    N/A" for value in values))
    print(f"Report: {output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
