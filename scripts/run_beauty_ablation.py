#!/usr/bin/env python3
"""Generate Beauty-axis interventions and build a portable isolation report.

The default mode replays the checked-in historical scores without making API
calls. Pass --live to render the generated SVGs and score them with the public
OpenAI backend configured by OPENAI_API_KEY.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagram_correctness.beauty import BEAUTY_DIMENSIONS, judge_beauty
from diagram_correctness.rubric import load_config
from diagram_correctness.vlm import OpenAIBackend


EXAMPLE = ROOT / "examples" / "beauty_ablation"
HISTORICAL_RESULTS = EXAMPLE / "historical_results.json"
LEAK_THRESHOLD = 15
MIN_SENSITIVITY = 15
VARIANT_TARGET = {
    "baseline": None,
    "degraded_aesthetics": "aesthetics",
    "degraded_palette": "palette",
    "degraded_ink_balance": "ink_balance",
    "degraded_density": "density",
    "degraded_balance": "balance",
}
LABELS = {
    "baseline": "Baseline",
    "degraded_aesthetics": "Degraded: Aesthetics",
    "degraded_palette": "Degraded: Palette",
    "degraded_ink_balance": "Degraded: Ink Balance",
    "degraded_density": "Degraded: Density",
    "degraded_balance": "Degraded: Balance",
}
DIMENSION_LABELS = {
    "aesthetics": "Aesthetic Craft",
    "palette": "Palette",
    "ink_balance": "Ink Balance",
    "density": "Density",
    "balance": "Balance",
}
DIMENSION_DEFINITIONS = {
    "aesthetics": (
        "Execution quality of shapes, strokes, typography geometry, connectors, "
        "arrowheads, edge rendering, and consistency."
    ),
    "palette": (
        "Coordination of chromatic hues and their consistent use for stable semantic roles."
    ),
    "ink_balance": (
        "The fraction of the canvas covered by fills, strokes, text, texture, and decoration."
    ),
    "density": (
        "Local clearance between distinct neighboring elements relative to their size."
    ),
    "balance": (
        "Global foreground placement measured through opposing margins and visual-centroid displacement."
    ),
}
NODE_LABELS = ("Start", "Validate Input", "Process Data", "Generate Report", "End")
NODE_WIDTH = 100
NODE_HEIGHT = 100
BASE_POSITIONS = ((250, 300), (375, 300), (500, 300), (625, 300), (750, 300))
DENSE_POSITIONS = ((290, 300), (395, 300), (500, 300), (605, 300), (710, 300))
BALANCE_POSITIONS = ((60, 300), (185, 300), (310, 300), (435, 300), (560, 300))
BASE_COLORS = ("#4c72b0", "#55a868", "#4c72b0", "#55a868", "#4c72b0")


def _arrow(start: tuple[int, int], end: tuple[int, int], color: str = "#333333") -> str:
    x1, y1 = start
    x2, y2 = end
    if x1 == x2:
        offset = NODE_HEIGHT / 2 if y2 > y1 else -NODE_HEIGHT / 2
        return (
            f'<line class="edge" x1="{x1}" y1="{y1 + offset}" x2="{x2}" '
            f'y2="{y2 - offset}" stroke="{color}" marker-end="url(#arrow)"/>'
        )
    offset = NODE_WIDTH / 2 if x2 > x1 else -NODE_WIDTH / 2
    return (
        f'<line class="edge" x1="{x1 + offset}" y1="{y1}" x2="{x2 - offset}" '
        f'y2="{y2}" stroke="{color}" marker-end="url(#arrow)"/>'
    )


def _diagram_svg(variant: str) -> str:
    positions = DENSE_POSITIONS if variant == "degraded_density" else BASE_POSITIONS
    if variant == "degraded_balance":
        positions = BALANCE_POSITIONS
    colors = BASE_COLORS
    if variant == "degraded_palette":
        colors = ("#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231")

    definitions = """
    <defs>
      <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" fill="context-stroke"/>
      </marker>
    </defs>"""
    background = '<rect width="1000" height="600" fill="white"/>'
    if variant == "degraded_ink_balance":
        background += "".join(
            f'<rect x="0" y="{band * 50}" width="1000" height="32" fill="#666"/>'
            for band in range(12)
        )
    nodes = []
    for index, ((x, y), label, color) in enumerate(zip(positions, NODE_LABELS, colors)):
        degraded_craft = variant == "degraded_aesthetics"
        rounded = (
            ' rx="18"'
            if not degraded_craft
            else (' rx="24"' if index % 2 else "")
        )
        dash = (
            f' stroke-dasharray="{("12 7", "3 6", "16 5 3 5")[index % 3]}"'
            if degraded_craft
            else ""
        )
        stroke_width = (2, 6, 3, 8, 4)[index] if degraded_craft else 3
        font_family = "Georgia,serif" if degraded_craft and index % 2 == 0 else "Arial,sans-serif"
        font_weight = "400" if degraded_craft and index % 2 == 0 else "700"
        nodes.append(
            f'<g class="node" data-node="{index}">'
            f'<rect x="{x - NODE_WIDTH / 2}" y="{y - NODE_HEIGHT / 2}" '
            f'width="{NODE_WIDTH}" height="{NODE_HEIGHT}"{rounded}{dash} '
            f'fill="{color}" stroke="#222" stroke-width="{stroke_width}"/>'
            f'<text x="{x}" y="{y}" fill="white" font-family="{font_family}" '
            f'font-weight="{font_weight}">{escape(label)}</text></g>'
        )
    arrow_colors = (
        ("#911eb4", "#46f0f0", "#f032e6", "#e6194b")
        if variant == "degraded_palette"
        else ("#333333",) * 4
    )
    edges = [
        _arrow(positions[index], positions[index + 1], arrow_colors[index])
        for index in range(4)
    ]
    title_x = sum(x for x, _ in positions) / len(positions)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600">
{definitions}
<style>
  .edge {{ fill:none; stroke-width:4; }}
  text {{ font-size:18px;
          text-anchor:middle; dominant-baseline:middle; }}
  .title {{ fill:#222; font:700 30px Arial,sans-serif; }}
</style>
{background}
<text class="title" x="{title_x}" y="190">Order Processing Flow</text>
{''.join(edges)}
{''.join(nodes)}
</svg>
"""


def generate_variants(output: Path) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for variant in VARIANT_TARGET:
        path = output / f"{variant}.svg"
        path.write_text(_diagram_svg(variant), encoding="utf-8")
        paths[variant] = path
    return paths


def isolation_metrics(results: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Compute sensitivity, leakage, specificity, and isolation per perturbation."""
    baseline = results["baseline"]
    metrics = {}
    for variant, target in VARIANT_TARGET.items():
        if target is None:
            continue
        sensitivity = baseline[target]["score"] - results[variant][target]["score"]
        leaks = {
            dimension: baseline[dimension]["score"] - results[variant][dimension]["score"]
            for dimension in BEAUTY_DIMENSIONS
            if dimension != target
        }
        worst_dimension = max(leaks, key=leaks.get)
        positive_sensitivity = max(0, sensitivity)
        positive_leaks = sum(max(0, drop) for drop in leaks.values())
        total_drop = positive_sensitivity + positive_leaks
        specificity = 100 * positive_sensitivity / total_drop if total_drop else 0.0
        sensitivity_factor = min(1.0, positive_sensitivity / MIN_SENSITIVITY)
        metrics[variant] = {
            "target_dimension": target,
            "sensitivity": sensitivity,
            "worst_leak_dimension": worst_dimension,
            "worst_leak": leaks[worst_dimension],
            "leaks": leaks,
            "specificity": round(specificity, 1),
            "isolation_score": round(specificity * sensitivity_factor, 1),
        }
    return metrics


def check_isolation(results: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    warnings = []
    for variant, metric in isolation_metrics(results).items():
        if metric["sensitivity"] < MIN_SENSITIVITY:
            warnings.append(
                f"[WEAK] {variant}: {metric['target_dimension']} dropped only "
                f"{metric['sensitivity']} points (minimum {MIN_SENSITIVITY})"
            )
        for dimension, drop in metric["leaks"].items():
            if drop >= LEAK_THRESHOLD:
                warnings.append(
                    f"[LEAK] {variant}: unrelated dimension {dimension} dropped "
                    f"{drop} points"
                )
    return warnings


def _live_results(
    paths: dict[str, Path],
    model: str,
    output: Path,
    repeats: int,
) -> dict:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --live")
    import cairosvg

    backend = OpenAIBackend(model)
    results = {}
    for variant, svg in paths.items():
        png = output / f"{variant}.png"
        cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=1000, output_height=600)
        repeated_scores = [judge_beauty(backend, png) for _ in range(repeats)]
        results[variant] = {}
        for dimension in BEAUTY_DIMENSIONS:
            scores = [run_scores[dimension] for run_scores in repeated_scores]
            values = [score.score for score in scores]
            results[variant][dimension] = {
                "observation": scores[0].observation,
                "score": round(statistics.mean(values)),
                "std": round(statistics.pstdev(values), 1),
                "runs": values,
                "reasoning": scores[0].reasoning,
            }
    return results


def _score_color(score: int) -> str:
    if score >= 70:
        ratio = (score - 70) / 30
        return f"rgb({int(255 - ratio * 145)},200,{int(110 - ratio * 30)})"
    return f"rgb(230,{int(90 + score / 70 * 110)},80)"


def build_html(summary: dict, images: dict[str, str]) -> str:
    dimensions = summary["dimensions"]
    headers = "".join(f"<th>{escape(DIMENSION_LABELS[d])}</th>" for d in dimensions)
    definitions = "".join(
        f"<li><strong>{escape(DIMENSION_LABELS[dimension])}:</strong> "
        f"{escape(DIMENSION_DEFINITIONS[dimension])}</li>"
        for dimension in dimensions
    )
    rows = []
    cards = []
    for row in summary["rows"]:
        target = row["target"]
        cells = "".join(
            f'<td class="score{" target" if dimension == target else ""}" '
            f'style="background:{_score_color(row["scores"][dimension])}">'
            f'{row["scores"][dimension]}{" *" if dimension == target else ""}</td>'
            for dimension in dimensions
        )
        rows.append(f'<tr><th>{escape(row["label"])}</th>{cells}</tr>')
        details = "".join(
            f'<section class="detail{" target" if dimension == target else ""}">'
            f'<strong>{escape(DIMENSION_LABELS[dimension])}: '
            f'{row["scores"][dimension]}</strong>'
            + (
                f'<p><b>Observed:</b> {escape(row["observations"][dimension])}</p>'
                if row["observations"].get(dimension)
                else ""
            )
            + f'<p>{escape(row["reasoning"][dimension])}</p></section>'
            for dimension in dimensions
        )
        cards.append(
            f'<article><h3>{escape(row["label"])}</h3>'
            f'<img src="{images[row["id"]]}" alt="{escape(row["label"])}">{details}</article>'
        )
    warnings = (
        "<ul>" + "".join(f"<li>{escape(warning)}</li>" for warning in summary["warnings"]) + "</ul>"
        if summary["warnings"]
        else "<p>Every intervention lowered only its target dimension under the configured threshold.</p>"
    )
    metric_rows = "".join(
        "<tr>"
        f'<th>{escape(LABELS[variant])}</th>'
        f'<td>{metric["sensitivity"]}</td>'
        f'<td>{metric["worst_leak"]} ({escape(metric["worst_leak_dimension"])})</td>'
        f'<td>{metric["specificity"]}</td>'
        f'<td><strong>{metric["isolation_score"]}</strong></td>'
        "</tr>"
        for variant, metric in summary["metrics"].items()
    )
    metrics_table = (
        "<h2>Isolation metrics</h2><table><thead><tr><th>Variant</th>"
        "<th>Target drop</th><th>Worst leak</th><th>Specificity</th>"
        f"<th>Isolation</th></tr></thead><tbody>{metric_rows}</tbody></table>"
    )
    payload = json.dumps(summary, ensure_ascii=False).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Beauty-axis isolation experiment</title><style>
body{{font:15px system-ui,sans-serif;margin:0;padding:2rem;background:#f5f6fa;color:#1a1a2e}}
main{{max-width:1200px;margin:auto}} table{{border-collapse:collapse;background:white;width:100%}}
th,td{{padding:.65rem;border:1px solid #ddd;text-align:center}} th:first-child{{text-align:left}}
.score{{font-weight:700}} .target{{outline:2px solid #1a1a2e;outline-offset:-2px}}
.notice{{background:#fff4e5;border:1px solid #f5c16c;padding:1rem;margin:1rem 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}}
article{{background:white;border:1px solid #ddd;border-radius:8px;overflow:hidden}}
article h3,.detail{{padding:.75rem 1rem;margin:0}} article img{{width:100%;display:block}}
.detail{{border-top:1px solid #eee}} .detail p{{color:#596273;margin:.35rem 0 0}}
.detail.target{{background:#fffbea}} code{{background:#eee;padding:.1rem .25rem}}
.definitions{{background:white;border:1px solid #ddd;padding:1rem 2rem;margin:1rem 0}}
.definitions li{{margin:.4rem 0}}
</style></head><body><main>
<h1>Diagram Beauty-axis isolation experiment</h1>
<p>{escape(summary["mode"])}. Each dimension is judged independently; <code>*</code> marks the intervention target.</p>
<ul class="definitions">{definitions}</ul>
<table><thead><tr><th>Variant</th>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>
{metrics_table}
<div class="notice"><strong>Isolation check</strong>{warnings}</div>
<div class="grid">{''.join(cards)}</div>
<p>See <a href="../../docs/beauty_axis_experiment.md">the methodology and limitations</a>.</p>
<script id="experiment-data" type="application/json">{payload}</script>
</main></body></html>
"""


def run(
    output: Path,
    live: bool = False,
    model: str | None = None,
    repeats: int = 3,
) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    diagrams = generate_variants(output / "diagrams")
    if live:
        configured = load_config(ROOT / "config" / "evaluator.json")["models"]["judge"]
        selected_model = model or configured
        results = _live_results(
            diagrams,
            selected_model,
            output / "diagrams",
            repeats,
        )
        mode = f"Live public OpenAI run using {selected_model}, {repeats} repeats"
    else:
        results = json.loads(HISTORICAL_RESULTS.read_text(encoding="utf-8"))["results"]
        mode = "Historical prototype scores replayed without API calls"
    rows = []
    for variant, target in VARIANT_TARGET.items():
        rows.append(
            {
                "id": variant,
                "label": LABELS[variant],
                "target": target,
                "scores": {dimension: results[variant][dimension]["score"] for dimension in BEAUTY_DIMENSIONS},
                "reasoning": {
                    dimension: results[variant][dimension]["reasoning"]
                    for dimension in BEAUTY_DIMENSIONS
                },
                "observations": {
                    dimension: results[variant][dimension].get("observation", "")
                    for dimension in BEAUTY_DIMENSIONS
                },
            }
        )
    metrics = isolation_metrics(results)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "live": live,
        "repeats": repeats if live else None,
        "dimensions": list(BEAUTY_DIMENSIONS),
        "rows": rows,
        "metrics": metrics,
        "overall_isolation": round(
            sum(metric["isolation_score"] for metric in metrics.values()) / len(metrics),
            1,
        ),
        "warnings": check_isolation(results),
        "limitations": [
            "One synthetic flowchart and one intervention per dimension do not establish generalization.",
            "Interventions have unequal strengths, so score drops cannot rank judge sensitivity.",
            "Off-diagonal drops can reflect real visual correlations or judge coupling.",
            "Historical scores came from an internal prototype; provider and deployment metadata are intentionally omitted.",
            "The latest source fixture cleanup was not rerun because of an hourly quota, so no corrected result is inferred.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "target", *BEAUTY_DIMENSIONS])
        writer.writeheader()
        for row in rows:
            writer.writerow({"variant": row["id"], "target": row["target"], **row["scores"]})
    images = {
        variant: "data:image/svg+xml;base64,"
        + base64.b64encode(path.read_bytes()).decode("ascii")
        for variant, path in diagrams.items()
    }
    (output / "index.html").write_text(build_html(summary, images), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Score generated diagrams via the public OpenAI API.")
    parser.add_argument("--model", help="OpenAI model override for --live.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Live judge calls per image and dimension (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: examples/beauty_ablation/reproduced, or live for --live.",
    )
    args = parser.parse_args()
    output = args.output_dir or EXAMPLE / ("live" if args.live else "reproduced")
    summary = run(
        output.resolve(),
        live=args.live,
        model=args.model,
        repeats=args.repeats,
    )
    print(summary["mode"])
    print(f"Report: {output.resolve() / 'index.html'}")
    for warning in summary["warnings"]:
        print(warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
