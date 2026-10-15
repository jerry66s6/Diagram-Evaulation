#!/usr/bin/env python3
"""Four-dimension Beauty isolation experiment: Centering, Space Use, Color Consistency, Contrast.

Default mode only renders the fixtures and reports their deterministic measurements.
Pass --live to score every fixture with the public OpenAI backend (OPENAI_API_KEY from
the environment or the project's .env file).
"""
from __future__ import annotations

import argparse
import base64
import colorsys
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont

from diagram_correctness.beauty import judge_beauty_dimension
from diagram_correctness.beauty_v4 import BEAUTY_V4_DIMENSIONS, BEAUTY_V4_PROMPTS
from diagram_correctness.rubric import load_config


OUTPUT_ROOT = ROOT / "examples" / "beauty_ablation" / "v4"
LEAK_THRESHOLD = 15
MIN_SENSITIVITY = 15
CANVAS = (1200, 750)
SUPERSAMPLE = 2
NODE_W, NODE_H = 170, 90
TITLE = "Order Processing Flow"
TITLE_SIZE = 32
LABEL_SIZE = 18
BACKGROUND = "#ffffff"
INK = "#1b2733"

VARIANT_TARGET = {
    "baseline": None,
    "degraded_centering": "centering",
    "degraded_space_use": "space_use",
    "degraded_color_consistency": "color_consistency",
    "degraded_contrast": "contrast",
}
LABELS = {
    "baseline": "Baseline",
    "degraded_centering": "Degraded: Centering",
    "degraded_space_use": "Degraded: Space Use",
    "degraded_color_consistency": "Degraded: Color Consistency",
    "degraded_contrast": "Degraded: Contrast",
}
DIMENSION_LABELS = {
    "centering": "Centering",
    "space_use": "Space Use",
    "color_consistency": "Color Consistency",
    "contrast": "Contrast",
}
DIMENSION_DEFINITIONS = {
    "centering": "Position of the whole diagram: symmetric opposing margins and a centered visual centroid.",
    "space_use": "Nodes should not feel condensed together while usable canvas space is left empty.",
    "color_consistency": "One coherent color theme with no random colors; each role keeps one color.",
    "contrast": "Boxes stand out from the background and touching boxes; arrows stand out from the background.",
}
INTERVENTIONS = {
    "baseline": "Centered, well-spread six-node flowchart with a two-hue blue/teal theme on white.",
    "degraded_centering": "Identical diagram translated toward the top-left corner.",
    "degraded_space_use": "Same nodes and colors pulled tightly together, still centered.",
    "degraded_color_consistency": "Same geometry; every box gets an unrelated hue at the baseline luminance.",
    "degraded_contrast": "Same geometry and hue families as very pale tints, with pale gray arrows.",
}

# (label, role, column, row); flow snakes left-to-right on row 0, then right-to-left on row 1.
NODES = (
    ("Start", "terminal", 0, 0),
    ("Validate Input", "process", 1, 0),
    ("Process Data", "process", 2, 0),
    ("Review Results", "process", 2, 1),
    ("Generate Report", "process", 1, 1),
    ("End", "terminal", 0, 1),
)
EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))
TARGET_LUMINANCE = 0.12
PALE_LUMINANCE = 0.86
ROLE_HUES = {"terminal": 160, "process": 212}
RANDOM_HUES = (0, 42, 120, 275, 320, 190)


@dataclass(frozen=True)
class Scene:
    origin: tuple[float, float]
    gap_x: float
    gap_y: float
    title_gap: float
    fills: tuple[str, ...]
    label_color: str
    arrow_color: str


def _channel(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    red, green, blue = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * _channel(red) + 0.7152 * _channel(green) + 0.0722 * _channel(blue)


def contrast_ratio(first: str, second: str) -> float:
    lighter, darker = sorted((relative_luminance(first), relative_luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def color_at_luminance(hue: float, saturation: float, luminance: float) -> str:
    """Binary-search HSL lightness so hues can change while luminance stays fixed."""
    low, high = 0.0, 1.0
    for _ in range(40):
        lightness = (low + high) / 2
        rgb = colorsys.hls_to_rgb(hue / 360, lightness, saturation)
        color = "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb)
        if relative_luminance(color) < luminance:
            low = lightness
        else:
            high = lightness
    return color


def hue_of(hex_color: str) -> float | None:
    red, green, blue = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    hue, _, saturation = colorsys.rgb_to_hls(red, green, blue)
    return None if saturation < 0.25 else hue * 360


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _title_size() -> tuple[int, int]:
    left, top, right, bottom = _font(TITLE_SIZE).getbbox(TITLE)
    return right - left, bottom - top


def _group_size(scene: Scene) -> tuple[float, float]:
    title_w, title_h = _title_size()
    width = max(3 * NODE_W + 2 * scene.gap_x, title_w)
    height = title_h + scene.title_gap + 2 * NODE_H + scene.gap_y
    return width, height


def centered_origin(gap_x: float, gap_y: float, title_gap: float) -> tuple[float, float]:
    probe = Scene((0, 0), gap_x, gap_y, title_gap, (), "", "")
    width, height = _group_size(probe)
    return (CANVAS[0] - width) / 2, (CANVAS[1] - height) / 2


def _role_fills(luminance: float) -> tuple[str, ...]:
    return tuple(color_at_luminance(ROLE_HUES[role], 0.6, luminance) for _, role, _, _ in NODES)


def build_scene(variant: str) -> Scene:
    if variant not in VARIANT_TARGET:
        raise ValueError(f"Unknown variant {variant!r}")
    gap_x, gap_y, title_gap = 150, 140, 60
    scene = Scene(
        origin=centered_origin(gap_x, gap_y, title_gap),
        gap_x=gap_x,
        gap_y=gap_y,
        title_gap=title_gap,
        fills=_role_fills(TARGET_LUMINANCE),
        label_color="#ffffff",
        arrow_color="#37474f",
    )
    if variant == "degraded_centering":
        scene = replace(scene, origin=(24, 24))
    elif variant == "degraded_space_use":
        scene = replace(
            scene,
            origin=centered_origin(26, 26, 26),
            gap_x=26,
            gap_y=26,
            title_gap=26,
        )
    elif variant == "degraded_color_consistency":
        scene = replace(
            scene,
            fills=tuple(color_at_luminance(hue, 0.6, TARGET_LUMINANCE) for hue in RANDOM_HUES),
        )
    elif variant == "degraded_contrast":
        scene = replace(
            scene,
            fills=_role_fills(PALE_LUMINANCE),
            label_color=INK,
            arrow_color="#d9dde1",
        )
    return scene


def node_boxes(scene: Scene) -> list[tuple[float, float, float, float]]:
    _, title_h = _title_size()
    group_w, _ = _group_size(scene)
    nodes_w = 3 * NODE_W + 2 * scene.gap_x
    left = scene.origin[0] + (group_w - nodes_w) / 2
    top = scene.origin[1] + title_h + scene.title_gap
    return [
        (
            left + column * (NODE_W + scene.gap_x),
            top + row * (NODE_H + scene.gap_y),
            left + column * (NODE_W + scene.gap_x) + NODE_W,
            top + row * (NODE_H + scene.gap_y) + NODE_H,
        )
        for _, _, column, row in NODES
    ]


def _edge_points(
    source: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    if abs(source[1] - target[1]) < 1:
        y = (source[1] + source[3]) / 2
        if target[0] > source[0]:
            return (source[2], y), (target[0], y)
        return (source[0], y), (target[2], y)
    x = (source[0] + source[2]) / 2
    return (x, source[3]), (x, target[1])


def render(scene: Scene, path: Path) -> Path:
    scale = SUPERSAMPLE
    image = Image.new("RGB", (CANVAS[0] * scale, CANVAS[1] * scale), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = _font(TITLE_SIZE * scale)
    label_font = _font(LABEL_SIZE * scale)
    group_w, _ = _group_size(scene)
    title_x = scene.origin[0] + group_w / 2
    title_top = scene.origin[1]
    draw.text((title_x * scale, title_top * scale), TITLE, fill=INK, font=title_font, anchor="mt")
    boxes = node_boxes(scene)
    for source, target in EDGES:
        (x1, y1), (x2, y2) = _edge_points(boxes[source], boxes[target])
        head = 12
        if x1 == x2:
            direction = 1 if y2 > y1 else -1
            tip = (x2, y2 - 2 * direction)
            base = (x2, tip[1] - head * direction)
            wings = ((base[0] - 7, base[1]), (base[0] + 7, base[1]))
        else:
            direction = 1 if x2 > x1 else -1
            tip = (x2 - 2 * direction, y2)
            base = (tip[0] - head * direction, y2)
            wings = ((base[0], base[1] - 7), (base[0], base[1] + 7))
        draw.line(
            [(x1 * scale, y1 * scale), (base[0] * scale, base[1] * scale)],
            fill=scene.arrow_color,
            width=4 * scale,
        )
        draw.polygon(
            [(point[0] * scale, point[1] * scale) for point in (tip, *wings)],
            fill=scene.arrow_color,
        )
    for (label, role, _, _), box, fill in zip(NODES, boxes, scene.fills):
        radius = NODE_H / 2 if role == "terminal" else 12
        draw.rounded_rectangle(
            [coordinate * scale for coordinate in box],
            radius=radius * scale,
            fill=fill,
        )
        draw.text(
            ((box[0] + box[2]) / 2 * scale, (box[1] + box[3]) / 2 * scale),
            label,
            fill=scene.label_color,
            font=label_font,
            anchor="mm",
        )
    image = image.resize(CANVAS, Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _hue_families(colors: list[str], tolerance: float = 20.0) -> list[float]:
    families: list[float] = []
    for color in colors:
        hue = hue_of(color)
        if hue is None:
            continue
        if not any(min(abs(hue - known), 360 - abs(hue - known)) <= tolerance for known in families):
            families.append(hue)
    return families


def measure(scene: Scene) -> dict[str, Any]:
    """Deterministic geometry/color facts used to verify each fixture changes one variable."""
    group_w, group_h = _group_size(scene)
    left, top = scene.origin
    right = CANVAS[0] - left - group_w
    bottom = CANVAS[1] - top - group_h
    margin_ratio = max(max(left, right) / min(left, right), max(top, bottom) / min(top, bottom))
    roles: dict[str, set[float]] = {}
    for (_, role, _, _), fill in zip(NODES, scene.fills):
        families = _hue_families([fill])
        roles.setdefault(role, set()).update(round(hue) for hue in families)
    contrasts = [contrast_ratio(fill, BACKGROUND) for fill in scene.fills]
    return {
        "margins_px": [round(left), round(right), round(top), round(bottom)],
        "worst_margin_ratio": round(margin_ratio, 2),
        "span_width_pct": round(100 * group_w / CANVAS[0], 1),
        "span_height_pct": round(100 * group_h / CANVAS[1], 1),
        "min_gap_to_node_pct": round(100 * min(scene.gap_x / NODE_W, scene.gap_y / NODE_H), 1),
        "hue_families": len(_hue_families(list(scene.fills) + [scene.arrow_color])),
        "roles_consistent": all(len(hues) <= 1 for hues in roles.values()),
        "fill_luminance": [round(relative_luminance(fill), 3) for fill in scene.fills],
        "min_box_contrast": round(min(contrasts), 2),
        "arrow_contrast": round(contrast_ratio(scene.arrow_color, BACKGROUND), 2),
    }


def isolation_metrics(
    results: dict[str, dict[str, dict[str, Any]]],
    variant_target: dict[str, str | None] = VARIANT_TARGET,
    dimensions: tuple[str, ...] = BEAUTY_V4_DIMENSIONS,
) -> dict[str, dict[str, Any]]:
    baseline = results["baseline"]
    metrics = {}
    for variant, target in variant_target.items():
        if target is None:
            continue
        sensitivity = baseline[target]["score"] - results[variant][target]["score"]
        leaks = {
            dimension: baseline[dimension]["score"] - results[variant][dimension]["score"]
            for dimension in dimensions
            if dimension != target
        }
        worst = max(leaks, key=leaks.get)
        positive = max(0, sensitivity)
        total = positive + sum(max(0, drop) for drop in leaks.values())
        specificity = 100 * positive / total if total else 0.0
        metrics[variant] = {
            "target_dimension": target,
            "sensitivity": sensitivity,
            "worst_leak_dimension": worst,
            "worst_leak": leaks[worst],
            "leaks": leaks,
            "specificity": round(specificity, 1),
            "isolation_score": round(specificity * min(1.0, positive / MIN_SENSITIVITY), 1),
        }
    return metrics


def isolation_warnings(metrics: dict[str, dict[str, Any]]) -> list[str]:
    warnings = []
    for variant, metric in metrics.items():
        if metric["sensitivity"] < MIN_SENSITIVITY:
            warnings.append(
                f"[WEAK] {variant}: {metric['target_dimension']} dropped only "
                f"{metric['sensitivity']} points (minimum {MIN_SENSITIVITY})"
            )
        for dimension, drop in metric["leaks"].items():
            if drop >= LEAK_THRESHOLD:
                warnings.append(f"[LEAK] {variant}: unrelated dimension {dimension} dropped {drop} points")
    return warnings


def load_local_key() -> bool:
    """Read only OPENAI_API_KEY from the project's .env file; never execute its text."""
    if os.environ.get("OPENAI_API_KEY"):
        return True
    dotenv = ROOT / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and name.strip() == "OPENAI_API_KEY":
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if value and value not in {"your_key", "your_api_key_here", "..."}:
                    os.environ["OPENAI_API_KEY"] = value
                    return True
    return False


def _judge_with_retry(backend, image: Path, dimension: str, attempts: int = 3):
    for attempt in range(1, attempts + 1):
        try:
            return judge_beauty_dimension(backend, image, dimension, BEAUTY_V4_PROMPTS)
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)


def score_live(
    images: dict[str, Path],
    backend,
    repeats: int,
    workers: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    jobs = [
        (variant, dimension, repeat)
        for variant in images
        for dimension in BEAUTY_V4_DIMENSIONS
        for repeat in range(repeats)
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores = list(
            pool.map(lambda job: _judge_with_retry(backend, images[job[0]], job[1]), jobs)
        )
    grouped: dict[tuple[str, str], list] = {}
    for (variant, dimension, _), score in zip(jobs, scores):
        grouped.setdefault((variant, dimension), []).append(score)
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for (variant, dimension), runs in grouped.items():
        values = [run.score for run in runs]
        results.setdefault(variant, {})[dimension] = {
            "score": round(statistics.mean(values)),
            "std": round(statistics.pstdev(values), 1),
            "runs": values,
            "observation": runs[0].observation,
            "reasoning": runs[0].reasoning,
        }
    return results


def _score_color(score: int) -> str:
    if score >= 70:
        ratio = (score - 70) / 30
        return f"rgb({int(255 - ratio * 145)},200,{int(110 - ratio * 30)})"
    return f"rgb(230,{int(90 + score / 70 * 110)},80)"


def build_html(summary: dict[str, Any], images: dict[str, str]) -> str:
    dimensions = summary["dimensions"]
    scored = summary["results"] is not None
    definitions = "".join(
        f"<li><strong>{escape(DIMENSION_LABELS[d])}:</strong> {escape(DIMENSION_DEFINITIONS[d])}</li>"
        for d in dimensions
    )
    measurement_rows = "".join(
        f"<tr><th>{escape(LABELS[variant])}</th>"
        f"<td>{m['worst_margin_ratio']}x</td>"
        f"<td>{m['span_width_pct']}% x {m['span_height_pct']}%</td>"
        f"<td>{m['min_gap_to_node_pct']}%</td>"
        f"<td>{m['hue_families']} / {'yes' if m['roles_consistent'] else 'no'}</td>"
        f"<td>{m['min_box_contrast']} / {m['arrow_contrast']}</td></tr>"
        for variant, m in summary["measurements"].items()
    )
    sections = [
        "<h2>Fixture measurements</h2><table><thead><tr><th>Variant</th><th>Worst margin ratio</th>"
        "<th>Group span</th><th>Min gap / node</th><th>Hue families / roles consistent</th>"
        f"<th>Box / arrow contrast vs background</th></tr></thead><tbody>{measurement_rows}</tbody></table>"
    ]
    cards = []
    if scored:
        results = summary["results"]
        header = "".join(f"<th>{escape(DIMENSION_LABELS[d])}</th>" for d in dimensions)
        rows = "".join(
            f"<tr><th>{escape(LABELS[variant])}</th>"
            + "".join(
                f'<td class="score{" target" if d == target else ""}" '
                f'style="background:{_score_color(results[variant][d]["score"])}">'
                f'{results[variant][d]["score"]}{" *" if d == target else ""}'
                f'<small>&plusmn;{results[variant][d]["std"]}</small></td>'
                for d in dimensions
            )
            + "</tr>"
            for variant, target in VARIANT_TARGET.items()
        )
        metric_rows = "".join(
            f"<tr><th>{escape(LABELS[variant])}</th><td>{m['sensitivity']}</td>"
            f"<td>{m['worst_leak']} ({escape(DIMENSION_LABELS[m['worst_leak_dimension']])})</td>"
            f"<td>{m['specificity']}</td><td><strong>{m['isolation_score']}</strong></td></tr>"
            for variant, m in summary["metrics"].items()
        )
        warnings = (
            "<ul>" + "".join(f"<li>{escape(w)}</li>" for w in summary["warnings"]) + "</ul>"
            if summary["warnings"]
            else "<p>Every intervention lowered only its target dimension under the configured threshold.</p>"
        )
        sections += [
            f"<h2>Scores (mean of {summary['repeats']} runs)</h2><table><thead><tr><th>Variant</th>"
            f"{header}</tr></thead><tbody>{rows}</tbody></table>",
            f"<h2>Isolation: {summary['overall_isolation']}/100</h2><table><thead><tr><th>Perturbation</th>"
            "<th>Target drop</th><th>Worst leak</th><th>Specificity</th><th>Isolation</th></tr></thead>"
            f"<tbody>{metric_rows}</tbody></table>",
            f'<div class="notice"><strong>Isolation check</strong>{warnings}</div>',
        ]
    for variant, target in VARIANT_TARGET.items():
        details = ""
        if scored:
            details = "".join(
                f'<section class="detail{" target" if d == target else ""}">'
                f'<strong>{escape(DIMENSION_LABELS[d])}: {summary["results"][variant][d]["score"]} '
                f'(runs {summary["results"][variant][d]["runs"]})</strong>'
                f'<p><b>Observed:</b> {escape(summary["results"][variant][d]["observation"])}</p>'
                f'<p>{escape(summary["results"][variant][d]["reasoning"])}</p></section>'
                for d in dimensions
            )
        cards.append(
            f"<article><h3>{escape(LABELS[variant])}</h3><p class=\"intervention\">"
            f"{escape(INTERVENTIONS[variant])}</p><img src=\"{images[variant]}\" "
            f"alt=\"{escape(LABELS[variant])}\">{details}</article>"
        )
    payload = json.dumps(summary, ensure_ascii=False).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Beauty v4 isolation experiment</title><style>
body{{font:15px system-ui,sans-serif;margin:0;padding:2rem;background:#f5f6fa;color:#1a1a2e}}
main{{max-width:1200px;margin:auto}} table{{border-collapse:collapse;background:white;width:100%;margin-bottom:1rem}}
th,td{{padding:.6rem;border:1px solid #ddd;text-align:center}} th:first-child{{text-align:left}}
.score{{font-weight:700}} .score small{{display:block;font-weight:400;opacity:.7}}
.target{{outline:2px solid #1a1a2e;outline-offset:-2px}}
.notice{{background:#fff4e5;border:1px solid #f5c16c;padding:1rem;margin:1rem 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:1rem}}
article{{background:white;border:1px solid #ddd;border-radius:8px;overflow:hidden}}
article h3,.detail,.intervention{{padding:.6rem 1rem;margin:0}} article img{{width:100%;display:block;border-top:1px solid #eee;border-bottom:1px solid #eee}}
.detail{{border-top:1px solid #eee}} .detail p,.intervention{{color:#596273;margin:.3rem 0 0}}
.detail.target{{background:#fffbea}} .definitions{{background:white;border:1px solid #ddd;padding:1rem 2rem}}
</style></head><body><main>
<h1>Beauty v4 isolation experiment</h1>
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
    results = None
    mode = "Fixtures only; no judge calls"
    if live or backend is not None:
        if backend is None:
            if not load_local_key():
                raise RuntimeError("OPENAI_API_KEY is required for --live (environment or project .env)")
            from diagram_correctness.vlm import OpenAIBackend

            model = model or load_config(ROOT / "config" / "evaluator.json")["models"]["judge"]
            backend = OpenAIBackend(model)
        results = score_live(paths, backend, repeats, workers)
        mode = f"Live run using {getattr(backend, 'name', 'judge')}, {repeats} repeats per image and dimension"
    metrics = isolation_metrics(results) if results else {}
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "repeats": repeats if results else None,
        "dimensions": list(BEAUTY_V4_DIMENSIONS),
        "measurements": {variant: measure(scene) for variant, scene in scenes.items()},
        "results": results,
        "metrics": metrics,
        "overall_isolation": (
            round(sum(m["isolation_score"] for m in metrics.values()) / len(metrics), 1) if metrics else None
        ),
        "warnings": isolation_warnings(metrics),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if results:
        with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["variant", "target", *BEAUTY_V4_DIMENSIONS])
            writer.writeheader()
            for variant, target in VARIANT_TARGET.items():
                writer.writerow(
                    {"variant": variant, "target": target, **{d: results[variant][d]["score"] for d in BEAUTY_V4_DIMENSIONS}}
                )
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
    parser.add_argument("--output-dir", type=Path, help="Default: examples/beauty_ablation/v4/{fixtures,live}.")
    args = parser.parse_args()
    output = (args.output_dir or OUTPUT_ROOT / ("live" if args.live else "fixtures")).resolve()
    summary = run(output, live=args.live, model=args.model, repeats=args.repeats, workers=args.workers)
    print(summary["mode"])
    print(f"Report: {output / 'index.html'}")
    if summary["overall_isolation"] is not None:
        print(f"Overall isolation: {summary['overall_isolation']}")
    for warning in summary["warnings"]:
        print(warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
