#!/usr/bin/env python3
"""Beauty v5 isolation experiment: the four v4 dimensions plus Alignment, Style Uniformity, Canvas Fill.

Default mode renders the fixtures and records deterministic measurements, including the
script-measured Canvas Fill score. Pass --live to score the six model-judged dimensions
with the public OpenAI backend (OPENAI_API_KEY from the environment or project .env).
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass, replace
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

from PIL import Image, ImageDraw, ImageFont

from diagram_correctness.beauty import judge_beauty_dimension
from diagram_correctness.beauty_v5 import (
    BEAUTY_V5_DIMENSIONS,
    BEAUTY_V5_MODEL_DIMENSIONS,
    BEAUTY_V5_PROMPTS,
)
from diagram_correctness.canvas_fill import judge_canvas_fill
from diagram_correctness.rubric import load_config

_spec = importlib.util.spec_from_file_location("beauty_v4_shared", ROOT / "scripts" / "run_beauty_v4_ablation.py")
v4 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = v4
_spec.loader.exec_module(v4)

OUTPUT_ROOT = ROOT / "examples" / "beauty_ablation" / "v5"
CANVAS = v4.CANVAS
SUPERSAMPLE = v4.SUPERSAMPLE
W, H = CANVAS
NODES = v4.NODES
EDGES = v4.EDGES
ROLES = tuple(role for _, role, _, _ in NODES)
LABEL_TEXT = tuple(label for label, _, _, _ in NODES)

VARIANT_TARGET = {
    "baseline": None,
    "degraded_centering": "centering",
    "degraded_space_use": "space_use",
    "degraded_color_consistency": "color_consistency",
    "degraded_contrast": "contrast",
    "degraded_alignment": "alignment",
    "degraded_style_uniformity": "style_uniformity",
    "degraded_canvas_fill": "canvas_fill",
}
DIMENSION_LABELS = {
    "centering": "Centering",
    "space_use": "Space Use",
    "color_consistency": "Color Consistency",
    "contrast": "Contrast",
    "alignment": "Alignment",
    "style_uniformity": "Style Uniformity",
    "canvas_fill": "Canvas Fill",
}
LABELS = {"baseline": "Baseline"} | {
    variant: f"Degraded: {DIMENSION_LABELS[target]}" for variant, target in VARIANT_TARGET.items() if target
}
DIMENSION_DEFINITIONS = {
    "centering": "Position of the whole diagram: symmetric opposing margins and a centered visual centroid.",
    "space_use": "Nodes should not feel condensed together while usable canvas space is left empty.",
    "color_consistency": "One coherent color theme with no random colors; each role keeps one color.",
    "contrast": "Boxes stand out from the background and touching boxes; arrows stand out from the background.",
    "alignment": "Nodes in the same row/column share one center line; connectors between them run straight.",
    "style_uniformity": "Same-role elements share one shape, size, border, font, and arrow treatment.",
    "canvas_fill": "Script-measured: elements must not be so small that most of the canvas is empty.",
}
INTERVENTIONS = {
    "baseline": "Centered, well-spread six-node flowchart with a two-hue blue/teal theme on white.",
    "degraded_centering": "Identical diagram translated toward the top-left corner.",
    "degraded_space_use": "Same nodes and colors pulled tightly together, still centered.",
    "degraded_color_consistency": "Same geometry; every box gets an unrelated hue at the baseline luminance.",
    "degraded_contrast": "Same geometry and hue families as very pale tints, with pale gray arrows.",
    "degraded_alignment": "Nodes nudged 12-38px off their rows/columns (re-centered); arrows slant.",
    "degraded_style_uniformity": "Same centers and role colors; per-node shape, size, border, font, and arrow weight.",
    "degraded_canvas_fill": "Tiny boxes spread to the canvas edges, joined by long arrows; most of the canvas is empty.",
}
ARROW_COLOR = "#37474f"
JITTER = ((0, -34), (28, 26), (-30, -14), (20, 38), (-26, -30), (34, 12))
STYLE_SPECS = (  # (width, height, kind, radius, outline width, font, font size)
    (170, 90, "pill", 45, 0, "arialbd.ttf", 18),
    (150, 76, "round", 0, 0, "times.ttf", 22),
    (205, 108, "ellipse", 0, 0, "courbd.ttf", 16),
    (165, 112, "round", 34, 3, "comic.ttf", 18),
    (215, 74, "round", 0, 7, "impact.ttf", 20),
    (150, 96, "round", 6, 0, "georgiai.ttf", 20),
)


@dataclass(frozen=True)
class Node:
    box: tuple[float, float, float, float]
    kind: str = "round"
    radius: float = 12
    outline_width: float = 0
    font_face: str = "arialbd.ttf"
    font_size: float = v4.LABEL_SIZE


@dataclass(frozen=True)
class Scene:
    title_top: float
    nodes: tuple[Node, ...]
    fills: tuple[str, ...]
    label_color: str = "#ffffff"
    arrow_color: str = ARROW_COLOR
    arrow_widths: tuple[float, ...] = (4,) * len(EDGES)
    arrow_heads: tuple[float, ...] = (12,) * len(EDGES)
    title_x: float = W / 2


def _font(face: str, size: float):
    try:
        return ImageFont.truetype(face, round(size))
    except OSError:
        return v4._font(round(size))


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _title_box(scene: Scene) -> tuple[float, float, float, float]:
    width, height = v4._title_size()
    return scene.title_x - width / 2, scene.title_top, scene.title_x + width / 2, scene.title_top + height


def group_bounds(scene: Scene) -> tuple[float, float, float, float]:
    boxes = [node.box for node in scene.nodes] + [_title_box(scene)]
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def _shift(scene: Scene, dx: float, dy: float) -> Scene:
    nodes = tuple(
        replace(node, box=(node.box[0] + dx, node.box[1] + dy, node.box[2] + dx, node.box[3] + dy))
        for node in scene.nodes
    )
    return replace(scene, nodes=nodes, title_top=scene.title_top + dy, title_x=scene.title_x + dx)


def recenter(scene: Scene) -> Scene:
    """Center the node group horizontally (title above its center) and the title+nodes group vertically."""
    left = min(node.box[0] for node in scene.nodes)
    right = max(node.box[2] for node in scene.nodes)
    scene = _shift(scene, (W - (right - left)) / 2 - left, 0)
    scene = replace(scene, title_x=W / 2)
    _, top, _, bottom = group_bounds(scene)
    return _shift(scene, 0, (H - (bottom - top)) / 2 - top)


def _base_nodes(v4_variant: str = "baseline") -> tuple[tuple[Node, ...], float]:
    scene = v4.build_scene(v4_variant)
    nodes = tuple(
        Node(box=box, kind="pill" if role == "terminal" else "round", radius=v4.NODE_H / 2 if role == "terminal" else 12)
        for box, role in zip(v4.node_boxes(scene), ROLES)
    )
    return nodes, scene.origin[1]


def build_scene(variant: str) -> Scene:
    if variant not in VARIANT_TARGET:
        raise ValueError(f"Unknown variant {variant!r}")
    nodes, title_top = _base_nodes()
    scene = Scene(title_top=title_top, nodes=nodes, fills=v4._role_fills(v4.TARGET_LUMINANCE))
    if variant == "degraded_centering":
        left, top, _, _ = group_bounds(scene)
        return _shift(scene, 24 - left, 24 - top)
    if variant == "degraded_space_use":
        nodes, title_top = _base_nodes("degraded_space_use")
        return replace(scene, nodes=nodes, title_top=title_top)
    if variant == "degraded_color_consistency":
        return replace(
            scene,
            fills=tuple(v4.color_at_luminance(hue, 0.6, v4.TARGET_LUMINANCE) for hue in v4.RANDOM_HUES),
        )
    if variant == "degraded_contrast":
        return replace(scene, fills=v4._role_fills(v4.PALE_LUMINANCE), label_color=v4.INK, arrow_color="#d9dde1")
    if variant == "degraded_alignment":
        nodes = tuple(
            replace(node, box=(node.box[0] + dx, node.box[1] + dy, node.box[2] + dx, node.box[3] + dy))
            for node, (dx, dy) in zip(scene.nodes, JITTER)
        )
        return recenter(replace(scene, nodes=nodes))
    if variant == "degraded_style_uniformity":
        nodes = []
        for node, (width, height, kind, radius, outline, face, size) in zip(scene.nodes, STYLE_SPECS):
            x, y = _center(node.box)
            nodes.append(Node((x - width / 2, y - height / 2, x + width / 2, y + height / 2), kind, radius, outline, face, size))
        return replace(scene, nodes=tuple(nodes), arrow_widths=(2, 7, 3, 6, 4), arrow_heads=(9, 24, 12, 20, 10))
    if variant == "degraded_canvas_fill":
        width, height = 120, 44
        xs = (60 + width / 2, W / 2, W - 60 - width / 2)
        ys = (250, 640)
        nodes = tuple(
            Node(
                (xs[column] - width / 2, ys[row] - height / 2, xs[column] + width / 2, ys[row] + height / 2),
                "pill" if role == "terminal" else "round",
                height / 2 if role == "terminal" else 10,
                font_size=13,
            )
            for (_, role, column, row) in NODES
        )
        scene = replace(scene, nodes=nodes, arrow_widths=(3,) * len(EDGES), arrow_heads=(10,) * len(EDGES))
        return recenter(scene)
    return scene


def _clip(center, direction, box):
    half_w, half_h = (box[2] - box[0]) / 2, (box[3] - box[1]) / 2
    dx, dy = direction
    t = min(half_w / abs(dx) if dx else math.inf, half_h / abs(dy) if dy else math.inf)
    return center[0] + dx * t, center[1] + dy * t


def edge_points(source, target):
    a, b = _center(source), _center(target)
    direction = (b[0] - a[0], b[1] - a[1])
    return _clip(a, direction, source), _clip(b, (-direction[0], -direction[1]), target)


def render(scene: Scene, path: Path) -> Path:
    s = SUPERSAMPLE
    image = Image.new("RGB", (W * s, H * s), v4.BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((scene.title_x * s, scene.title_top * s), v4.TITLE, fill=v4.INK, font=_font("arialbd.ttf", v4.TITLE_SIZE * s), anchor="mt")
    for (source, target), width, head in zip(EDGES, scene.arrow_widths, scene.arrow_heads):
        start, end = edge_points(scene.nodes[source].box, scene.nodes[target].box)
        length = math.dist(start, end)
        ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
        tip = (end[0] - ux * 2, end[1] - uy * 2)
        base = (tip[0] - ux * head, tip[1] - uy * head)
        px, py = -uy * head * 0.55, ux * head * 0.55
        draw.line([(start[0] * s, start[1] * s), (base[0] * s, base[1] * s)], fill=scene.arrow_color, width=round(width * s))
        draw.polygon(
            [(tip[0] * s, tip[1] * s), ((base[0] + px) * s, (base[1] + py) * s), ((base[0] - px) * s, (base[1] - py) * s)],
            fill=scene.arrow_color,
        )
    for node, label, fill in zip(scene.nodes, LABEL_TEXT, scene.fills):
        coords = [c * s for c in node.box]
        outline = "#10233f" if node.outline_width else None
        width = round(node.outline_width * s)
        if node.kind == "ellipse":
            draw.ellipse(coords, fill=fill, outline=outline, width=width)
        else:
            radius = (node.box[3] - node.box[1]) / 2 if node.kind == "pill" else node.radius
            draw.rounded_rectangle(coords, radius=radius * s, fill=fill, outline=outline, width=width)
        x, y = _center(node.box)
        draw.text((x * s, y * s), label, fill=scene.label_color, font=_font(node.font_face, node.font_size * s), anchor="mm")
    image = image.resize(CANVAS, Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def measure(scene: Scene, image_path: Path) -> dict[str, Any]:
    """Deterministic facts used to verify each fixture changes only its intended variable."""
    left, top, right, bottom = group_bounds(scene)
    margins = (left, W - right, top, H - bottom)
    margin_ratio = max(max(margins[:2]) / min(margins[:2]), max(margins[2:]) / min(margins[2:]))
    node_left = min(n.box[0] for n in scene.nodes)
    node_right = max(n.box[2] for n in scene.nodes)
    node_top = min(n.box[1] for n in scene.nodes)
    node_bottom = max(n.box[3] for n in scene.nodes)
    gaps = []
    for source, target in EDGES:
        a, b = scene.nodes[source].box, scene.nodes[target].box
        horizontal = abs(_center(a)[0] - _center(b)[0]) >= abs(_center(a)[1] - _center(b)[1])
        if horizontal:
            gaps.append((max(a[0], b[0]) - min(a[2], b[2])) / ((a[2] - a[0] + b[2] - b[0]) / 2))
        else:
            gaps.append((max(a[1], b[1]) - min(a[3], b[3])) / ((a[3] - a[1] + b[3] - b[1]) / 2))
    offsets = []
    for spec_index, axis in ((2, 0), (3, 1)):  # columns share an x center; rows share a y center
        for key in {spec[spec_index] for spec in NODES}:
            members = [n for n, spec in zip(scene.nodes, NODES) if spec[spec_index] == key]
            centers = [_center(n.box)[axis] for n in members]
            size = statistics.mean((n.box[2] - n.box[0]) if axis == 0 else (n.box[3] - n.box[1]) for n in members)
            offsets.append((max(centers) - min(centers)) / size)
    role_styles = {}
    for node, role in zip(scene.nodes, ROLES):
        signature = (
            round(node.box[2] - node.box[0]), round(node.box[3] - node.box[1]), node.kind,
            node.radius if node.kind == "round" else None, node.outline_width, node.font_face, node.font_size,
        )
        role_styles.setdefault(role, set()).add(signature)
    roles: dict[str, set[float]] = {}
    for role, fill in zip(ROLES, scene.fills):
        roles.setdefault(role, set()).update(round(h) for h in v4._hue_families([fill]))
    fill = judge_canvas_fill(image_path)
    return {
        "worst_margin_ratio": round(margin_ratio, 2),
        "span_width_pct": round(100 * (node_right - node_left) / W, 1),
        "span_height_pct": round(100 * (node_bottom - node_top) / H, 1),
        "min_gap_to_node_pct": round(100 * min(gaps), 1),
        "hue_families": len(v4._hue_families(list(scene.fills) + [scene.arrow_color])),
        "roles_consistent": all(len(hues) <= 1 for hues in roles.values()),
        "min_box_contrast": round(min(v4.contrast_ratio(f, v4.BACKGROUND) for f in scene.fills), 2),
        "arrow_contrast": round(v4.contrast_ratio(scene.arrow_color, v4.BACKGROUND), 2),
        "max_row_col_offset_pct": round(100 * max(offsets), 1),
        "style_variants_per_role": max(len(styles) for styles in role_styles.values()),
        "arrow_treatments": len(set(zip(scene.arrow_widths, scene.arrow_heads))),
        "canvas_footprint_pct": fill.footprint_pct,
    }


def score(
    images: dict[str, Path],
    backend,
    repeats: int,
    workers: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, dict[str, Any]]] = {variant: {} for variant in images}
    for variant, path in images.items():
        fill = judge_canvas_fill(path)
        results[variant]["canvas_fill"] = {
            "score": fill.score, "std": 0.0, "runs": [fill.score], "observation": fill.observation, "reasoning": fill.reasoning,
        }
    if backend is None:
        return results
    jobs = [(v, d, r) for v in images for d in BEAUTY_V5_MODEL_DIMENSIONS for r in range(repeats)]

    def call(job):
        variant, dimension, _ = job
        for attempt in range(1, 4):
            try:
                return judge_beauty_dimension(backend, images[variant], dimension, BEAUTY_V5_PROMPTS)
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


def off_target_mean_abs(results: dict, dimensions=BEAUTY_V5_DIMENSIONS) -> float:
    """Mean absolute score change of non-target dimensions; 0 means perfectly uncorrelated."""
    base = results["baseline"]
    deltas = [
        abs(results[variant][d]["score"] - base[d]["score"])
        for variant, target in VARIANT_TARGET.items()
        if target
        for d in dimensions
        if d != target
    ]
    return round(statistics.mean(deltas), 2)


def build_html(summary: dict[str, Any], images: dict[str, str]) -> str:
    dimensions = summary["dimensions"]
    results = summary["results"]
    scored = summary["metrics"] != {}
    definitions = "".join(
        f"<li><strong>{escape(DIMENSION_LABELS[d])}:</strong> {escape(DIMENSION_DEFINITIONS[d])}</li>" for d in dimensions
    )
    measurement_rows = "".join(
        f"<tr><th>{escape(LABELS[v])}</th><td>{m['worst_margin_ratio']}x</td>"
        f"<td>{m['span_width_pct']}% x {m['span_height_pct']}%</td><td>{m['min_gap_to_node_pct']}%</td>"
        f"<td>{m['hue_families']} / {'yes' if m['roles_consistent'] else 'no'}</td>"
        f"<td>{m['min_box_contrast']} / {m['arrow_contrast']}</td><td>{m['max_row_col_offset_pct']}%</td>"
        f"<td>{m['style_variants_per_role']} / {m['arrow_treatments']}</td><td>{m['canvas_footprint_pct']}%</td></tr>"
        for v, m in summary["measurements"].items()
    )
    sections = [
        "<h2>Fixture measurements</h2><table><thead><tr><th>Variant</th><th>Margin ratio</th><th>Node span</th>"
        "<th>Min gap / node</th><th>Hue families / roles consistent</th><th>Box / arrow contrast</th>"
        "<th>Max row/col offset</th><th>Styles per role / arrow treatments</th><th>Canvas footprint</th></tr></thead>"
        f"<tbody>{measurement_rows}</tbody></table>"
    ]
    header = "".join(f"<th>{escape(DIMENSION_LABELS[d])}</th>" for d in dimensions)
    rows = "".join(
        f"<tr><th>{escape(LABELS[v])}</th>"
        + "".join(
            f'<td class="score{" target" if d == target else ""}" style="background:{v4._score_color(results[v][d]["score"])}">'
            f'{results[v][d]["score"]}{" *" if d == target else ""}<small>{results[v][d]["runs"]}</small></td>'
            if d in results[v] else "<td>-</td>"
            for d in dimensions
        )
        + "</tr>"
        for v, target in VARIANT_TARGET.items()
    )
    sections.append(f"<h2>Scores</h2><table><thead><tr><th>Variant</th>{header}</tr></thead><tbody>{rows}</tbody></table>")
    if scored:
        metric_rows = "".join(
            f"<tr><th>{escape(LABELS[v])}</th><td>{m['sensitivity']}</td>"
            f"<td>{m['worst_leak']} ({escape(DIMENSION_LABELS[m['worst_leak_dimension']])})</td>"
            f"<td>{m['specificity']}</td><td><strong>{m['isolation_score']}</strong></td></tr>"
            for v, m in summary["metrics"].items()
        )
        warnings = (
            "<ul>" + "".join(f"<li>{escape(w)}</li>" for w in summary["warnings"]) + "</ul>"
            if summary["warnings"] else "<p>Every intervention lowered only its target dimension under the configured threshold.</p>"
        )
        sections += [
            f"<h2>Isolation: {summary['overall_isolation']}/100 &middot; mean |off-target change| "
            f"{summary['off_target_mean_abs']} points</h2><table><thead><tr><th>Perturbation</th><th>Target drop</th>"
            f"<th>Worst leak</th><th>Specificity</th><th>Isolation</th></tr></thead><tbody>{metric_rows}</tbody></table>",
            f'<div class="notice"><strong>Isolation check</strong>{warnings}</div>',
        ]
    cards = []
    for v, target in VARIANT_TARGET.items():
        details = "".join(
            f'<section class="detail{" target" if d == target else ""}"><strong>{escape(DIMENSION_LABELS[d])}: '
            f'{results[v][d]["score"]}</strong><p><b>Observed:</b> {escape(results[v][d]["observation"])}</p>'
            f'<p>{escape(results[v][d]["reasoning"])}</p></section>'
            for d in dimensions
            if d in results[v]
        )
        cards.append(
            f'<article><h3>{escape(LABELS[v])}</h3><p class="intervention">{escape(INTERVENTIONS[v])}</p>'
            f'<img src="{images[v]}" alt="{escape(LABELS[v])}">{details}</article>'
        )
    payload = json.dumps(summary, ensure_ascii=False).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Beauty v5 isolation experiment</title><style>
body{{font:15px system-ui,sans-serif;margin:0;padding:2rem;background:#f5f6fa;color:#1a1a2e}}
main{{max-width:1300px;margin:auto}} table{{border-collapse:collapse;background:white;width:100%;margin-bottom:1rem}}
th,td{{padding:.5rem;border:1px solid #ddd;text-align:center}} th:first-child{{text-align:left}}
.score{{font-weight:700}} .score small{{display:block;font-weight:400;opacity:.7}}
.target{{outline:2px solid #1a1a2e;outline-offset:-2px}}
.notice{{background:#fff4e5;border:1px solid #f5c16c;padding:1rem;margin:1rem 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:1rem}}
article{{background:white;border:1px solid #ddd;border-radius:8px;overflow:hidden}}
article h3,.detail,.intervention{{padding:.6rem 1rem;margin:0}} article img{{width:100%;display:block;border-top:1px solid #eee;border-bottom:1px solid #eee}}
.detail{{border-top:1px solid #eee}} .detail p,.intervention{{color:#596273;margin:.3rem 0 0}}
.detail.target{{background:#fffbea}} .definitions{{background:white;border:1px solid #ddd;padding:1rem 2rem}}
</style></head><body><main>
<h1>Beauty v5 isolation experiment</h1>
<p>{escape(summary["mode"])}. Each dimension is judged independently; <code>*</code> marks the intervention target.</p>
<ul class="definitions">{definitions}</ul>
{''.join(sections)}
<div class="grid">{''.join(cards)}</div>
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
    scenes = {variant: build_scene(variant) for variant in VARIANT_TARGET}
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
    metrics = v4.isolation_metrics(results, VARIANT_TARGET, BEAUTY_V5_DIMENSIONS) if full else {}
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "repeats": repeats if full else None,
        "dimensions": list(BEAUTY_V5_DIMENSIONS),
        "measurements": {variant: measure(scene, paths[variant]) for variant, scene in scenes.items()},
        "results": results,
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
            writer = csv.DictWriter(handle, fieldnames=["variant", "target", *BEAUTY_V5_DIMENSIONS])
            writer.writeheader()
            for variant, target in VARIANT_TARGET.items():
                writer.writerow({"variant": variant, "target": target, **{d: results[variant][d]["score"] for d in BEAUTY_V5_DIMENSIONS}})
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
    parser.add_argument("--output-dir", type=Path, help="Default: examples/beauty_ablation/v5/{fixtures,live}.")
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
