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
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagram_correctness.beauty import BEAUTY_DIMENSIONS, judge_beauty
from diagram_correctness.rubric import load_config
from diagram_correctness.vlm import OpenAIBackend


EXAMPLE = ROOT / "examples" / "beauty_ablation"
HISTORICAL_RESULTS = EXAMPLE / "historical_results.json"
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
NODE_LABELS = ("Start", "Validate Input", "Process Data", "Generate Report", "End")
BASE_POSITIONS = ((100, 300), (300, 300), (500, 300), (700, 300), (900, 300))
DENSE_POSITIONS = ((435, 220), (565, 220), (565, 340), (435, 340), (435, 460))
BASE_COLORS = ("#4c72b0", "#55a868", "#4c72b0", "#55a868", "#4c72b0")


def _arrow(start: tuple[int, int], end: tuple[int, int], color: str = "#333333") -> str:
    x1, y1 = start
    x2, y2 = end
    if x1 == x2:
        offset = 55 if y2 > y1 else -55
        return (
            f'<line class="edge" x1="{x1}" y1="{y1 + offset}" x2="{x2}" '
            f'y2="{y2 - offset}" stroke="{color}" marker-end="url(#arrow)"/>'
        )
    offset = 80 if x2 > x1 else -80
    return (
        f'<line class="edge" x1="{x1 + offset}" y1="{y1}" x2="{x2 - offset}" '
        f'y2="{y2}" stroke="{color}" marker-end="url(#arrow)"/>'
    )


def _diagram_svg(variant: str) -> str:
    positions = DENSE_POSITIONS if variant == "degraded_density" else BASE_POSITIONS
    if variant == "degraded_balance":
        positions = tuple((x, 485) for x, _ in BASE_POSITIONS)
    colors = BASE_COLORS
    if variant == "degraded_aesthetics":
        colors = ("#ff0000", "#ffff00", "#ff0000", "#ffff00", "#ff0000")
    elif variant == "degraded_palette":
        colors = ("#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231")

    definitions = """
    <defs>
      <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" fill="context-stroke"/>
      </marker>
      <pattern id="hatch" width="12" height="12" patternUnits="userSpaceOnUse">
        <path d="M-3,3 l6,-6 M0,12 l12,-12 M9,15 l6,-6" stroke="#777" stroke-width="2"/>
      </pattern>
    </defs>"""
    background = (
        '<rect width="1000" height="600" fill="#dddddd"/>'
        '<rect width="1000" height="600" fill="url(#hatch)" opacity="0.9"/>'
        if variant == "degraded_ink_balance"
        else '<rect width="1000" height="600" fill="white"/>'
    )
    nodes = []
    for index, ((x, y), label, color) in enumerate(zip(positions, NODE_LABELS, colors)):
        rounded = "" if variant == "degraded_aesthetics" else ' rx="14"'
        stroke_width = 8 if variant == "degraded_aesthetics" else 3
        text_color = "blue" if variant == "degraded_aesthetics" else "white"
        fill = "url(#hatch)" if variant == "degraded_ink_balance" else color
        nodes.append(
            f'<g class="node" data-node="{index}">'
            f'<rect x="{x - 80}" y="{y - 55}" width="160" height="110"{rounded} '
            f'fill="{fill}" stroke="#222" stroke-width="{stroke_width}"/>'
            f'<text x="{x}" y="{y}" fill="{text_color}">{escape(label)}</text></g>'
        )
        if variant == "degraded_ink_balance":
            nodes.append(
                f'<rect x="{x - 90}" y="{y - 65}" width="180" height="130" '
                'fill="none" stroke="#555" stroke-width="3"/>'
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
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600">
{definitions}
<style>
  .edge {{ fill:none; stroke-width:4; }}
  text {{ font-family:Arial,sans-serif; font-size:22px; font-weight:700;
          text-anchor:middle; dominant-baseline:middle; }}
  .title {{ fill:#222; font-size:30px; }}
</style>
{background}
<text class="title" x="500" y="45">Order Processing Flow</text>
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


def check_isolation(results: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    warnings = []
    baseline = results["baseline"]
    for variant, target in VARIANT_TARGET.items():
        if target is None:
            continue
        for dimension in BEAUTY_DIMENSIONS:
            score = results[variant][dimension]["score"]
            base_score = baseline[dimension]["score"]
            if dimension == target and score >= base_score:
                warnings.append(
                    f"[FAIL] {variant}: expected {dimension} to drop "
                    f"({score} vs baseline {base_score})"
                )
            elif dimension != target and score < base_score - 15:
                warnings.append(
                    f"[LEAK] {variant}: unrelated {dimension} dropped by more than 15 "
                    f"points ({score} vs baseline {base_score})"
                )
    return warnings


def _live_results(paths: dict[str, Path], model: str, output: Path) -> dict:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --live")
    import cairosvg

    backend = OpenAIBackend(model)
    results = {}
    for variant, svg in paths.items():
        png = output / f"{variant}.png"
        cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=1000, output_height=600)
        results[variant] = {
            dimension: score.to_dict()
            for dimension, score in judge_beauty(backend, png).items()
        }
    return results


def _score_color(score: int) -> str:
    if score >= 70:
        ratio = (score - 70) / 30
        return f"rgb({int(255 - ratio * 145)},200,{int(110 - ratio * 30)})"
    return f"rgb(230,{int(90 + score / 70 * 110)},80)"


def build_html(summary: dict, images: dict[str, str]) -> str:
    dimensions = summary["dimensions"]
    headers = "".join(f"<th>{escape(d.replace('_', ' ').title())}</th>" for d in dimensions)
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
            f'<strong>{escape(dimension.replace("_", " ").title())}: '
            f'{row["scores"][dimension]}</strong>'
            f'<p>{escape(row["reasoning"][dimension])}</p></section>'
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
</style></head><body><main>
<h1>Diagram Beauty-axis isolation experiment</h1>
<p>{escape(summary["mode"])}. Each dimension is judged independently; <code>*</code> marks the intervention target.</p>
<table><thead><tr><th>Variant</th>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="notice"><strong>Isolation check</strong>{warnings}</div>
<div class="grid">{''.join(cards)}</div>
<p>See <a href="../../docs/beauty_axis_experiment.md">the methodology and limitations</a>.</p>
<script id="experiment-data" type="application/json">{payload}</script>
</main></body></html>
"""


def run(output: Path, live: bool = False, model: str | None = None) -> dict:
    diagrams = generate_variants(output / "diagrams")
    if live:
        configured = load_config(ROOT / "config" / "evaluator.json")["models"]["judge"]
        selected_model = model or configured
        results = _live_results(diagrams, selected_model, output / "diagrams")
        mode = f"Live public OpenAI run using {selected_model}"
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
            }
        )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "live": live,
        "dimensions": list(BEAUTY_DIMENSIONS),
        "rows": rows,
        "warnings": check_isolation(results),
        "limitations": [
            "One synthetic flowchart and one intervention per dimension do not establish generalization.",
            "Interventions have unequal strengths, so score drops cannot rank judge sensitivity.",
            "Off-diagonal drops can reflect real visual correlations or judge coupling.",
            "Historical scores came from an internal prototype; provider and deployment metadata are intentionally omitted.",
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
        "--output-dir",
        type=Path,
        help="Default: examples/beauty_ablation/reproduced, or live for --live.",
    )
    args = parser.parse_args()
    output = args.output_dir or EXAMPLE / ("live" if args.live else "reproduced")
    summary = run(output.resolve(), live=args.live, model=args.model)
    print(summary["mode"])
    print(f"Report: {output.resolve() / 'index.html'}")
    for warning in summary["warnings"]:
        print(warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
