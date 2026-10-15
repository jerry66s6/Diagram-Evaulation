#!/usr/bin/env python3
"""Beauty v6 isolation experiment: eight dimensions with sub-checks.

Centering, Alignment, and Resolution each have several single-variable fixtures so every
sub-check is tested on its own. Default mode renders fixtures and records deterministic
measurements, including Canvas Fill. Pass --live to score the seven model-judged dimensions
with the public OpenAI backend (OPENAI_API_KEY from the environment or project .env).
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from html import escape
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw, ImageFilter

from diagram_correctness.beauty import judge_beauty_dimension
from diagram_correctness.beauty_v6 import (
    BEAUTY_V6_DIMENSIONS,
    BEAUTY_V6_MODEL_DIMENSIONS,
    BEAUTY_V6_PROMPTS,
)
from diagram_correctness.canvas_fill import judge_canvas_fill
from diagram_correctness.rubric import load_config

_spec = importlib.util.spec_from_file_location("beauty_v5_shared", ROOT / "scripts" / "run_beauty_v5_ablation.py")
v5 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = v5
_spec.loader.exec_module(v5)
v4 = v5.v4

OUTPUT_ROOT = ROOT / "examples" / "beauty_ablation" / "v6"
W, H = v5.CANVAS
NODES, EDGES = v5.NODES, v5.EDGES

VARIANTS = {  # variant -> (target dimension, sub-check, description)
    "baseline": (None, None, "Centered, evenly spaced, axis-aligned six-node flowchart, two-hue theme on white."),
    "degraded_centering": ("centering", "whole diagram", "Identical diagram translated to the top-left corner."),
    "degraded_centering_labels": ("centering", "labels in shapes", "Every label jammed into its shape's bottom-left corner."),
    "degraded_centering_spacing": ("centering", "even spacing", "Row gaps alternate 80px / 220px (both comfortable); same total width."),
    "degraded_space_use": ("space_use", None, "Same nodes pulled tightly together, still centered."),
    "degraded_color_consistency": ("color_consistency", None, "Every box an unrelated hue at the same luminance."),
    "degraded_contrast": ("contrast", None, "Same hues as very pale tints, pale gray arrows."),
    "degraded_alignment": ("alignment", "grid", "Nodes shifted 20-32px vertically off their rows; horizontal gaps unchanged."),
    "degraded_alignment_tilt": ("alignment", "orientation", "Each node and its label rotated 4-9 degrees."),
    "degraded_alignment_arrows": ("alignment", "connectors", "Arrows run diagonally corner-to-corner between aligned nodes."),
    "degraded_alignment_endpoints": ("alignment", "endpoints", "Arrows 1, 3, 5 stop 45px short of both boxes; arrows 2 and 4 touch."),
    "degraded_style_uniformity": ("style_uniformity", None, "Per-node shape, size, border, font, and arrow weight."),
    "degraded_canvas_fill": ("canvas_fill", None, "Tiny boxes spread to the canvas edges."),
    "degraded_resolution_blur": ("resolution", "blur", "Whole image Gaussian-blurred (radius 2.2px)."),
    "degraded_resolution_pixel": ("resolution", "pixelation", "Rendered at 40% without anti-aliasing, then upscaled."),
}
VARIANT_TARGET = {variant: spec[0] for variant, spec in VARIANTS.items()}
DIMENSION_LABELS = v5.DIMENSION_LABELS | {"resolution": "Resolution"}
LABELS = {"baseline": "Baseline"} | {
    variant: f"Degraded: {DIMENSION_LABELS[target]}" + (f" ({sub})" if sub else "")
    for variant, (target, sub, _) in VARIANTS.items()
    if target
}
DIMENSION_DEFINITIONS = {
    "centering": "Everything centered: whole diagram on the canvas, each label in its shape, even gaps between neighbors.",
    "space_use": "Nodes must not feel condensed together while usable canvas space is left empty.",
    "color_consistency": "One coherent color theme with no random colors; each role keeps one color.",
    "contrast": "Boxes stand out from the background and touching boxes; arrows stand out from the background.",
    "alignment": "Everything axis-aligned: upright shapes and labels, nodes on shared lines, horizontal/vertical "
    "connectors with consistent endpoints.",
    "style_uniformity": "Same-role elements share one shape, size, border, font, and arrow stroke/arrowhead.",
    "resolution": "Crisp rendering: no blur, pixelation, jagged stair-steps, or compression artifacts.",
    "canvas_fill": "Script-measured: elements must not be so small that most of the canvas is empty.",
}
ALIGN_JITTER_Y = (0, -30, 24, -20, 32, -26)
TILTS = (6, -8, 4, -5, 9, -7)
UNEVEN_GAPS = (80, 220)


@dataclass(frozen=True)
class Scene(v5.Scene):
    label_anchor: str = "center"
    rotations: tuple[float, ...] = (0,) * len(NODES)
    edge_mode: str = "center"
    arrow_gaps: tuple[float, ...] = (0,) * len(EDGES)
    arrows_on_top: bool = False
    label_colors: tuple[str, ...] | None = None
    blur: float = 0.0
    render_scale: float = 1.0


def from_v5(scene) -> Scene:
    return Scene(**{field.name: getattr(scene, field.name) for field in fields(scene)})


SHARED_WITH_V5 = {
    "degraded_centering", "degraded_space_use", "degraded_color_consistency",
    "degraded_contrast", "degraded_style_uniformity", "degraded_canvas_fill",
}


def build_scene(variant: str) -> Scene:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")
    scene = from_v5(v5.build_scene(variant if variant in SHARED_WITH_V5 else "baseline"))
    if variant == "degraded_centering_labels":
        return replace(scene, label_anchor="corner")
    if variant == "degraded_centering_spacing":
        left = min(node.box[0] for node in scene.nodes)
        width = v4.NODE_W
        column_left = (left, left + width + UNEVEN_GAPS[0], left + 2 * width + sum(UNEVEN_GAPS))
        nodes = tuple(
            replace(node, box=(column_left[column], node.box[1], column_left[column] + width, node.box[3]))
            for node, (_, _, column, _) in zip(scene.nodes, NODES)
        )
        return replace(scene, nodes=nodes)
    if variant == "degraded_alignment":
        nodes = tuple(
            replace(node, box=(node.box[0], node.box[1] + dy, node.box[2], node.box[3] + dy))
            for node, dy in zip(scene.nodes, ALIGN_JITTER_Y)
        )
        return v5.recenter(replace(scene, nodes=nodes))
    if variant == "degraded_alignment_tilt":
        return replace(scene, rotations=TILTS)
    if variant == "degraded_alignment_arrows":
        return replace(scene, edge_mode="corner")
    if variant == "degraded_alignment_endpoints":
        return replace(scene, arrow_gaps=(45, 0, 45, 0, 45))
    if variant == "degraded_resolution_blur":
        return replace(scene, blur=2.2)
    if variant == "degraded_resolution_pixel":
        return replace(scene, render_scale=0.4)
    return scene


def _corner_points(source, target):
    (sx, sy), (tx, ty) = v5._center(source), v5._center(target)
    if abs(sy - ty) < 1:
        if tx > sx:
            return (source[2], source[1] + 6), (target[0], target[3] - 6)
        return (source[0], source[1] + 6), (target[2], target[3] - 6)
    return (source[0] + 10, source[3]), (target[2] - 10, target[1])


def edge_endpoints(scene: Scene, index: int) -> tuple[tuple[float, float], tuple[float, float]]:
    source, target = EDGES[index]
    a, b = scene.nodes[source].box, scene.nodes[target].box
    start, end = _corner_points(a, b) if scene.edge_mode == "corner" else v5.edge_points(a, b)
    gap = scene.arrow_gaps[index]
    if gap:
        length = math.dist(start, end)
        ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
        start = (start[0] + ux * gap, start[1] + uy * gap)
        end = (end[0] - ux * gap, end[1] - uy * gap)
    return start, end


def _draw_arrows(draw, scene: Scene, s: float) -> None:
    for index in range(len(EDGES)):
        start, end = edge_endpoints(scene, index)
        width, head = scene.arrow_widths[index], scene.arrow_heads[index]
        length = math.dist(start, end)
        ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
        tip = (end[0] - ux * 2, end[1] - uy * 2)
        base = (tip[0] - ux * head, tip[1] - uy * head)
        px, py = -uy * head * 0.55, ux * head * 0.55
        draw.line([(start[0] * s, start[1] * s), (base[0] * s, base[1] * s)], fill=scene.arrow_color,
                  width=max(1, round(width * s)))
        draw.polygon(
            [(tip[0] * s, tip[1] * s), ((base[0] + px) * s, (base[1] + py) * s), ((base[0] - px) * s, (base[1] - py) * s)],
            fill=scene.arrow_color,
        )


def render(scene: Scene, path: Path) -> Path:
    antialias = scene.render_scale >= 1
    s = v5.SUPERSAMPLE * scene.render_scale if antialias else scene.render_scale
    image = Image.new("RGBA", (round(W * s), round(H * s)), v4.BACKGROUND)
    draw = ImageDraw.Draw(image)
    if not antialias:
        draw.fontmode = "1"
    draw.text((scene.title_x * s, scene.title_top * s), v4.TITLE, fill=v4.INK,
              font=v5._font("arialbd.ttf", v4.TITLE_SIZE * s), anchor="mt")
    if not scene.arrows_on_top:
        _draw_arrows(draw, scene, s)
    label_colors = scene.label_colors or (scene.label_color,) * len(NODES)
    pad = 40
    for node, label, fill, angle, text_color in zip(scene.nodes, v5.LABEL_TEXT, scene.fills, scene.rotations, label_colors):
        x0, y0, x1, y1 = node.box
        tile = Image.new("RGBA", (round((x1 - x0 + 2 * pad) * s), round((y1 - y0 + 2 * pad) * s)), (0, 0, 0, 0))
        tile_draw = ImageDraw.Draw(tile)
        if not antialias:
            tile_draw.fontmode = "1"
        local = (pad * s, pad * s, (x1 - x0 + pad) * s, (y1 - y0 + pad) * s)
        outline = "#10233f" if node.outline_width else None
        width = round(node.outline_width * s)
        if node.kind == "ellipse":
            tile_draw.ellipse(local, fill=fill, outline=outline, width=width)
        else:
            radius = (y1 - y0) / 2 if node.kind == "pill" else node.radius
            tile_draw.rounded_rectangle(local, radius=radius * s, fill=fill, outline=outline, width=width)
        font = v5._font(node.font_face, node.font_size * s)
        if scene.label_anchor == "corner":
            inset = (y1 - y0) * 0.32 if node.kind == "pill" else 3
            tile_draw.text((local[0] + inset * s, local[3] - 3 * s), label, fill=text_color, font=font, anchor="ld")
        else:
            tile_draw.text((tile.width / 2, tile.height / 2), label, fill=text_color, font=font, anchor="mm")
        if angle:
            tile = tile.rotate(angle, resample=Image.BICUBIC)
        image.alpha_composite(tile, (round((x0 - pad) * s), round((y0 - pad) * s)))
    if scene.arrows_on_top:
        _draw_arrows(ImageDraw.Draw(image), scene, s)
    image = image.convert("RGB")
    if antialias:
        image = image.resize((W, H), Image.LANCZOS)
    else:
        image = image.resize((W, H), Image.NEAREST)
    if scene.blur:
        image = image.filter(ImageFilter.GaussianBlur(scene.blur))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def measure(scene: Scene, image_path: Path) -> dict[str, Any]:
    """Deterministic facts used to verify each fixture changes only its intended variable."""
    facts = v5.measure(scene, image_path)
    ratios = []
    for row in {spec[3] for spec in NODES}:
        members = sorted((n.box for n, spec in zip(scene.nodes, NODES) if spec[3] == row), key=lambda box: box[0])
        gaps = [b[0] - a[2] for a, b in zip(members, members[1:])]
        ratios.append(max(gaps) / min(gaps))
    diagonal = 0
    for index in range(len(EDGES)):
        (x1, y1), (x2, y2) = edge_endpoints(scene, index)
        diagonal += abs(x1 - x2) > 1 and abs(y1 - y2) > 1
    facts.update(
        {
            "label_anchor": scene.label_anchor,
            "row_gap_ratio": round(max(ratios), 2),
            "max_rotation_deg": max(abs(angle) for angle in scene.rotations),
            "diagonal_connectors": diagonal,
            "endpoint_treatments": len(set(scene.arrow_gaps)),
            "blur_px": scene.blur,
            "render_scale": scene.render_scale,
        }
    )
    return facts


def score(images: dict[str, Path], backend, repeats: int, workers: int) -> dict[str, dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, dict[str, Any]]] = {variant: {} for variant in images}
    for variant, path in images.items():
        fill = judge_canvas_fill(path)
        results[variant]["canvas_fill"] = {
            "score": fill.score, "std": 0.0, "runs": [fill.score], "observation": fill.observation,
            "reasoning": fill.reasoning, "footprint_pct": fill.footprint_pct,
        }
    if backend is None:
        return results
    jobs = [(v, d, r) for v in images for d in BEAUTY_V6_MODEL_DIMENSIONS for r in range(repeats)]

    def call(job):
        variant, dimension, _ = job
        for attempt in range(1, 4):
            try:
                return judge_beauty_dimension(backend, images[variant], dimension, BEAUTY_V6_PROMPTS)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * attempt)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(call, jobs))
    grouped: dict[tuple[str, str], list] = {}
    for (variant, dimension, _), output in zip(jobs, outputs):
        grouped.setdefault((variant, dimension), []).append(output)
    for (variant, dimension), runs in grouped.items():
        values = [run.score for run in runs]
        results[variant][dimension] = {
            "score": round(statistics.mean(values)),
            "std": round(statistics.pstdev(values), 1),
            "runs": values,
            "observation": runs[0].observation,
            "reasoning": runs[0].reasoning,
        }
    return results


def off_target_mean_abs(results: dict) -> float:
    """Mean absolute score change of non-target dimensions; 0 means perfectly uncorrelated."""
    base = results["baseline"]
    deltas = [
        abs(results[variant][d]["score"] - base[d]["score"])
        for variant, target in VARIANT_TARGET.items()
        if target
        for d in BEAUTY_V6_DIMENSIONS
        if d != target
    ]
    return round(statistics.mean(deltas), 2)


def build_html(summary: dict[str, Any], images: dict[str, str]) -> str:
    dimensions = summary["dimensions"]
    results = summary["results"]
    definitions = "".join(
        f"<li><strong>{escape(DIMENSION_LABELS[d])}:</strong> {escape(DIMENSION_DEFINITIONS[d])}</li>" for d in dimensions
    )
    header = "".join(f"<th>{escape(DIMENSION_LABELS[d])}</th>" for d in dimensions)
    rows = "".join(
        f"<tr><th>{escape(LABELS[v])}</th>"
        + "".join(
            f'<td class="score{" target" if d == target else ""}" style="background:{v4._score_color(results[v][d]["score"])}">'
            f'{results[v][d]["score"]}<small>{results[v][d]["runs"]}</small></td>'
            if d in results[v] else "<td>-</td>"
            for d in dimensions
        )
        + "</tr>"
        for v, target in VARIANT_TARGET.items()
    )
    sections = [f"<h2>Scores</h2><table><thead><tr><th>Variant</th>{header}</tr></thead><tbody>{rows}</tbody></table>"]
    if summary["metrics"]:
        metric_rows = "".join(
            f"<tr><th>{escape(LABELS[v])}</th><td>{m['sensitivity']}</td>"
            f"<td>{m['worst_leak']} ({escape(DIMENSION_LABELS[m['worst_leak_dimension']])})</td>"
            f"<td>{m['specificity']}</td><td><strong>{m['isolation_score']}</strong></td></tr>"
            for v, m in summary["metrics"].items()
        )
        sections.append(
            f"<h2>Isolation: {summary['overall_isolation']}/100 &middot; mean |off-target change| "
            f"{summary['off_target_mean_abs']} points</h2><table><thead><tr><th>Perturbation</th><th>Target drop</th>"
            f"<th>Worst leak</th><th>Specificity</th><th>Isolation</th></tr></thead><tbody>{metric_rows}</tbody></table>"
        )
    cards = "".join(
        f'<article><h3>{escape(LABELS[v])}</h3><p class="intervention">{escape(VARIANTS[v][2])}</p>'
        f'<img src="{images[v]}" alt="{escape(LABELS[v])}">'
        + "".join(
            f'<section class="detail{" target" if d == VARIANT_TARGET[v] else ""}"><strong>{escape(DIMENSION_LABELS[d])}: '
            f'{results[v][d]["score"]}</strong><p><b>Observed:</b> {escape(results[v][d]["observation"])}</p>'
            f'<p>{escape(results[v][d]["reasoning"])}</p></section>'
            for d in dimensions
            if d in results[v]
        )
        + "</article>"
        for v in VARIANTS
    )
    payload = json.dumps(summary, ensure_ascii=False).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Beauty v6 isolation experiment</title><style>
body{{font:15px system-ui,sans-serif;margin:0;padding:2rem;background:#f5f6fa;color:#1a1a2e}}
main{{max-width:1400px;margin:auto}} table{{border-collapse:collapse;background:white;width:100%;margin-bottom:1rem}}
th,td{{padding:.45rem;border:1px solid #ddd;text-align:center}} th:first-child{{text-align:left}}
.score{{font-weight:700}} .score small{{display:block;font-weight:400;opacity:.7}}
.target{{outline:2px solid #1a1a2e;outline-offset:-2px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:1rem}}
article{{background:white;border:1px solid #ddd;border-radius:8px;overflow:hidden}}
article h3,.detail,.intervention{{padding:.6rem 1rem;margin:0}} article img{{width:100%;display:block}}
.detail{{border-top:1px solid #eee}} .detail p,.intervention{{color:#596273;margin:.3rem 0 0}}
.detail.target{{background:#fffbea}} .definitions{{background:white;border:1px solid #ddd;padding:1rem 2rem}}
</style></head><body><main>
<h1>Beauty v6 isolation experiment</h1>
<p>{escape(summary["mode"])}. Outlined cells mark the intervention target.</p>
<ul class="definitions">{definitions}</ul>
{''.join(sections)}
<div class="grid">{cards}</div>
<script id="experiment-data" type="application/json">{payload}</script>
</main></body></html>
"""


def run(
    output: Path,
    live: bool = False,
    model: str | None = None,
    repeats: int = 3,
    workers: int = 6,
    backend=None,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    scenes = {variant: build_scene(variant) for variant in VARIANTS}
    paths = {variant: render(scene, output / "diagrams" / f"{variant}.png") for variant, scene in scenes.items()}
    mode = "Fixtures only; Canvas Fill measured, no judge calls"
    if live and backend is None:
        if not v4.load_local_key():
            raise RuntimeError("OPENAI_API_KEY is required for --live (environment or project .env)")
        from diagram_correctness.vlm import OpenAIBackend

        backend = OpenAIBackend(model or load_config(ROOT / "config" / "evaluator.json")["models"]["judge"])
    if backend is not None:
        mode = f"Live run using {getattr(backend, 'name', 'judge')}, {repeats} repeats per image and model-judged dimension"
    results = score(paths, backend, repeats, workers)
    full = backend is not None
    metrics = v4.isolation_metrics(results, VARIANT_TARGET, BEAUTY_V6_DIMENSIONS) if full else {}
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "repeats": repeats if full else None,
        "dimensions": list(BEAUTY_V6_DIMENSIONS),
        "variants": {v: {"target": t, "sub_check": s, "description": d} for v, (t, s, d) in VARIANTS.items()},
        "measurements": {variant: measure(scene, paths[variant]) for variant, scene in scenes.items()},
        "results": results if full else {v: {"canvas_fill": r["canvas_fill"]} for v, r in results.items()},
        "metrics": metrics,
        "overall_isolation": (
            round(sum(m["isolation_score"] for m in metrics.values()) / len(metrics), 1) if metrics else None
        ),
        "off_target_mean_abs": off_target_mean_abs(results) if full else None,
        "warnings": v4.isolation_warnings(metrics),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if full:
        with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["variant", "target", *BEAUTY_V6_DIMENSIONS])
            writer.writeheader()
            for variant, target in VARIANT_TARGET.items():
                writer.writerow({"variant": variant, "target": target, **{d: results[variant][d]["score"] for d in BEAUTY_V6_DIMENSIONS}})
    images = {
        variant: "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        for variant, path in paths.items()
    }
    (output / "index.html").write_text(build_html(summary, images), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Score fixtures via the public OpenAI API.")
    parser.add_argument("--model", help="OpenAI model override for --live.")
    parser.add_argument("--repeats", type=int, default=3, help="Judge calls per image and dimension (default: 3).")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent judge calls (default: 6).")
    parser.add_argument("--output-dir", type=Path, help="Default: examples/beauty_ablation/v6/{fixtures,live}.")
    args = parser.parse_args()
    output = (args.output_dir or OUTPUT_ROOT / ("live" if args.live else "fixtures")).resolve()
    summary = run(output, live=args.live, model=args.model, repeats=args.repeats, workers=args.workers)
    print(summary["mode"])
    print(f"Report: {output / 'index.html'}")
    if summary["overall_isolation"] is not None:
        print(f"Overall isolation: {summary['overall_isolation']}; mean |off-target change|: {summary['off_target_mean_abs']}")
    for warning in summary["warnings"]:
        print(warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
