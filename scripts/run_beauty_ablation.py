#!/usr/bin/env python3
"""Current Beauty isolation experiment: eight VLM-judged dimensions.

Alignment and Resolution have several single-variable fixtures; label and gap changes are
whole-diagram placement controls. Default mode renders fixtures and records geometry/color
measurements, without scoring any dimension. Pass --live to score all eight dimensions
with the public OpenAI backend (OPENAI_API_KEY from the environment or project .env).
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
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from diagram_correctness.beauty import BEAUTY_DIMENSIONS, judge_beauty_dimension
from diagram_correctness.rubric import load_config

OUTPUT_ROOT = ROOT / "examples" / "beauty_ablation" / "v6"
W, H = 1200, 750
SUPERSAMPLE = 2
NODE_W, NODE_H = 170, 90
TITLE = "Order Processing Flow"
TITLE_SIZE, LABEL_SIZE = 32, 18
BACKGROUND, INK = "#ffffff", "#1b2733"
ARROW_COLOR = "#37474f"
TARGET_LUMINANCE, PALE_LUMINANCE = 0.12, 0.86
ROLE_HUES = {"terminal": 160, "process": 212}
RANDOM_HUES = (0, 42, 120, 275, 320, 190)
LEAK_THRESHOLD = MIN_SENSITIVITY = 15
NODES = (
    ("Start", "terminal", 0, 0),
    ("Validate Input", "process", 1, 0),
    ("Process Data", "process", 2, 0),
    ("Review Results", "process", 2, 1),
    ("Generate Report", "process", 1, 1),
    ("End", "terminal", 0, 1),
)
EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))
ROLES = tuple(role for _, role, _, _ in NODES)
LABEL_TEXT = tuple(label for label, _, _, _ in NODES)
STYLE_SPECS = (
    (170, 90, "pill", 45, 0, "arialbd.ttf", 18),
    (150, 76, "round", 0, 0, "times.ttf", 22),
    (205, 108, "ellipse", 0, 0, "courbd.ttf", 16),
    (165, 112, "round", 34, 3, "comic.ttf", 18),
    (215, 74, "round", 0, 7, "impact.ttf", 20),
    (150, 96, "round", 6, 0, "georgiai.ttf", 20),
)

VARIANTS = {  # variant -> (target dimension, sub-check, description)
    "baseline": (None, None, "Centered, evenly spaced, axis-aligned six-node flowchart, two-hue theme on white."),
    "degraded_centering": ("centering", "whole diagram", "Identical diagram translated to the top-left corner."),
    "degraded_centering_labels": (None, "placement control", "Labels shifted inside shapes; whole-diagram margins unchanged."),
    "degraded_centering_spacing": (None, "placement control", "Unequal but comfortable row gaps; whole-diagram margins unchanged."),
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
    "control_cycle_straight": (None, "control", "Valid design: five steps in a cycle joined by straight diagonal arrows."),
    "control_cycle_curved": (None, "control", "Valid design: five steps in a cycle joined by curved arrows."),
    "control_loopback": (None, "control", "Valid design: straight left-to-right flow with one curved loop-back arrow."),
    "control_tree": (None, "control", "Valid design: a tree whose branches run diagonally from parent to child."),
    "control_hub": (None, "control", "Valid design: a hub with straight spokes out to six services."),
}
CONTROLS = {variant for variant, (_, sub, _) in VARIANTS.items() if sub == "control"}
# Keep historical fixture IDs, but no longer expect label or gap changes to lower Centering.
PLACEMENT_CONTROLS = {variant for variant, (_, sub, _) in VARIANTS.items() if sub == "placement control"}
VARIANT_TARGET = {variant: spec[0] for variant, spec in VARIANTS.items()}
DIMENSION_LABELS = {d: d.replace("_", " ").title() for d in BEAUTY_DIMENSIONS}
LABELS = {"baseline": "Baseline"} | {
    variant: f"Degraded: {DIMENSION_LABELS[target]}" + (f" ({sub})" if sub else "")
    for variant, (target, sub, _) in VARIANTS.items()
    if target
} | {
    "control_cycle_straight": "Control: cycle, straight arrows",
    "control_cycle_curved": "Control: cycle, curved arrows",
    "control_loopback": "Control: loop-back arrow",
    "control_tree": "Control: tree, angled branches",
    "control_hub": "Control: hub and spokes",
    "degraded_centering_labels": "Control: labels shifted inside shapes",
    "degraded_centering_spacing": "Control: unequal comfortable gaps",
}
DIMENSION_DEFINITIONS = {
    "centering": "Opposite-margin imbalance normalized by canvas width/height around the full diagram.",
    "space_use": "Peer-node/group clearance relative to size; intentional layered symbols count as one group.",
    "color_consistency": "Coherent color-to-role/section mapping without a three-hue limit.",
    "contrast": "Visible box boundaries (outline and/or fill) and arrows stand out from their background.",
    "alignment": "Deliberate geometry: nothing almost-aligned or slightly tilted; connectors take the cleanest path "
    "in one consistent routing style; consistent endpoints. Any layout type is valid.",
    "style_uniformity": "Semantically equivalent elements and connectors share a consistent visual treatment.",
    "resolution": "Crisp rendering: no blur, pixelation, jagged stair-steps, or compression artifacts.",
    "canvas_fill": "Model-estimated element footprint: score 0 at 3% or below, 100 at 6% or above.",
}
ALIGN_JITTER_Y = (0, -30, 24, -20, 32, -26)
TILTS = (6, -8, 4, -5, 9, -7)
UNEVEN_GAPS = (80, 220)


@dataclass(frozen=True)
class Node:
    box: tuple[float, float, float, float]
    kind: str = "round"
    radius: float = 12
    outline_width: float = 0
    font_face: str = "arialbd.ttf"
    font_size: float = LABEL_SIZE


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
    label_anchor: str = "center"
    rotations: tuple[float, ...] = (0,) * len(NODES)
    edge_mode: str = "center"
    arrow_gaps: tuple[float, ...] = (0,) * len(EDGES)
    arrows_on_top: bool = False
    label_colors: tuple[str, ...] | None = None
    blur: float = 0.0
    render_scale: float = 1.0


def _channel(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(color: str) -> float:
    red, green, blue = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * _channel(red) + 0.7152 * _channel(green) + 0.0722 * _channel(blue)


def contrast_ratio(first: str, second: str) -> float:
    lighter, darker = sorted((relative_luminance(first), relative_luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def color_at_luminance(hue: float, saturation: float, luminance: float) -> str:
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


def _hue_families(colors: list[str], tolerance: float = 20.0) -> list[float]:
    families = []
    for color in colors:
        red, green, blue = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
        hue, _, saturation = colorsys.rgb_to_hls(red, green, blue)
        if saturation < 0.25:
            continue
        hue *= 360
        if not any(min(abs(hue - known), 360 - abs(hue - known)) <= tolerance for known in families):
            families.append(hue)
    return families


def _role_fills(luminance: float) -> tuple[str, ...]:
    return tuple(color_at_luminance(ROLE_HUES[role], 0.6, luminance) for role in ROLES)


def _font(face: str, size: float):
    for name in (face, "arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, round(size))
        except OSError:
            continue
    return ImageFont.load_default(size=round(size))


def _title_size() -> tuple[int, int]:
    left, top, right, bottom = _font("arialbd.ttf", TITLE_SIZE).getbbox(TITLE)
    return right - left, bottom - top


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def group_bounds(scene: Scene) -> tuple[float, float, float, float]:
    width, height = _title_size()
    boxes = [node.box for node in scene.nodes] + [
        (scene.title_x - width / 2, scene.title_top, scene.title_x + width / 2, scene.title_top + height),
    ]
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def _shift(scene: Scene, dx: float, dy: float) -> Scene:
    nodes = tuple(replace(n, box=(n.box[0] + dx, n.box[1] + dy, n.box[2] + dx, n.box[3] + dy))
                  for n in scene.nodes)
    return replace(scene, nodes=nodes, title_top=scene.title_top + dy, title_x=scene.title_x + dx)


def recenter(scene: Scene) -> Scene:
    left, right = min(n.box[0] for n in scene.nodes), max(n.box[2] for n in scene.nodes)
    scene = _shift(scene, (W - (right - left)) / 2 - left, 0)
    scene = replace(scene, title_x=W / 2)
    _, top, _, bottom = group_bounds(scene)
    return _shift(scene, 0, (H - (bottom - top)) / 2 - top)


def _base_scene(gap_x=150, gap_y=140, title_gap=60) -> Scene:
    _, title_height = _title_size()
    nodes_width = 3 * NODE_W + 2 * gap_x
    group_height = title_height + title_gap + 2 * NODE_H + gap_y
    title_top = (H - group_height) / 2
    left, top = (W - nodes_width) / 2, title_top + title_height + title_gap
    nodes = tuple(
        Node((left + col * (NODE_W + gap_x), top + row * (NODE_H + gap_y),
              left + col * (NODE_W + gap_x) + NODE_W, top + row * (NODE_H + gap_y) + NODE_H),
             kind="pill" if role == "terminal" else "round", radius=NODE_H / 2 if role == "terminal" else 12)
        for _, role, col, row in NODES
    )
    return Scene(title_top=title_top, nodes=nodes, fills=_role_fills(TARGET_LUMINANCE))


def build_scene(variant: str) -> Scene:
    if variant not in VARIANTS or variant in CONTROLS:
        raise ValueError(f"Unknown grid variant {variant!r}")
    scene = _base_scene()
    if variant == "degraded_centering":
        left, top, _, _ = group_bounds(scene)
        return _shift(scene, 24 - left, 24 - top)
    if variant == "degraded_space_use":
        return _base_scene(26, 26, 26)
    if variant == "degraded_color_consistency":
        return replace(scene, fills=tuple(color_at_luminance(h, 0.6, TARGET_LUMINANCE) for h in RANDOM_HUES))
    if variant == "degraded_contrast":
        return replace(scene, fills=_role_fills(PALE_LUMINANCE), label_color=INK, arrow_color="#d9dde1")
    if variant == "degraded_style_uniformity":
        nodes = []
        for node, (width, height, kind, radius, outline, face, size) in zip(scene.nodes, STYLE_SPECS):
            x, y = _center(node.box)
            nodes.append(Node((x - width / 2, y - height / 2, x + width / 2, y + height / 2),
                              kind, radius, outline, face, size))
        return replace(scene, nodes=tuple(nodes), arrow_widths=(2, 7, 3, 6, 4), arrow_heads=(9, 24, 12, 20, 10))
    if variant == "degraded_canvas_fill":
        width, height = 120, 44
        xs, ys = (60 + width / 2, W / 2, W - 60 - width / 2), (250, 640)
        nodes = tuple(
            Node((xs[col] - width / 2, ys[row] - height / 2, xs[col] + width / 2, ys[row] + height / 2),
                 "pill" if role == "terminal" else "round", height / 2 if role == "terminal" else 10,
                 font_size=13)
            for _, role, col, row in NODES
        )
        return recenter(replace(scene, nodes=nodes, arrow_widths=(3,) * len(EDGES), arrow_heads=(10,) * len(EDGES)))
    if variant == "degraded_centering_labels":
        return replace(scene, label_anchor="corner")
    if variant == "degraded_centering_spacing":
        left = min(node.box[0] for node in scene.nodes)
        width = NODE_W
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
        return recenter(replace(scene, nodes=nodes))
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
    (sx, sy), (tx, ty) = _center(source), _center(target)
    if abs(sy - ty) < 1:
        if tx > sx:
            return (source[2], source[1] + 6), (target[0], target[3] - 6)
        return (source[0], source[1] + 6), (target[2], target[3] - 6)
    return (source[0] + 10, source[3]), (target[2] - 10, target[1])


def edge_endpoints(scene: Scene, index: int) -> tuple[tuple[float, float], tuple[float, float]]:
    source, target = EDGES[index]
    a, b = scene.nodes[source].box, scene.nodes[target].box
    if scene.edge_mode == "corner":
        start, end = _corner_points(a, b)
    else:
        ac, bc = _center(a), _center(b)
        direction = (bc[0] - ac[0], bc[1] - ac[1])
        start = _clip_box(ac, direction, (a[2] - a[0]) / 2, (a[3] - a[1]) / 2)
        end = _clip_box(bc, (-direction[0], -direction[1]), (b[2] - b[0]) / 2, (b[3] - b[1]) / 2)
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
    s = SUPERSAMPLE * scene.render_scale if antialias else scene.render_scale
    image = Image.new("RGBA", (round(W * s), round(H * s)), BACKGROUND)
    draw = ImageDraw.Draw(image)
    if not antialias:
        draw.fontmode = "1"
    draw.text((scene.title_x * s, scene.title_top * s), TITLE, fill=INK,
              font=_font("arialbd.ttf", TITLE_SIZE * s), anchor="mt")
    if not scene.arrows_on_top:
        _draw_arrows(draw, scene, s)
    label_colors = scene.label_colors or (scene.label_color,) * len(NODES)
    pad = 40
    for node, label, fill, angle, text_color in zip(scene.nodes, LABEL_TEXT, scene.fills, scene.rotations, label_colors):
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
        font = _font(node.font_face, node.font_size * s)
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


CYCLE_LABELS = ("Plan", "Build", "Test", "Release", "Review")
CYCLE_TITLE = "Release Cycle"
CYCLE_BOX = (150, 70)
CYCLE_RADIUS = 230


def render_cycle(variant: str, path: Path) -> Path:
    """Valid non-grid layout: five steps evenly spaced on a circle, joined by straight or curved arrows."""
    s = SUPERSAMPLE
    box_w, box_h = CYCLE_BOX
    title_font = _font("arialbd.ttf", TITLE_SIZE * s)
    _, top_px, _, bottom_px = title_font.getbbox(CYCLE_TITLE)
    title_h, title_gap = (bottom_px - top_px) / s, 60
    angles = [-90 + 72 * index for index in range(len(CYCLE_LABELS))]
    reach_up = CYCLE_RADIUS + box_h / 2
    reach_down = max(CYCLE_RADIUS * math.sin(math.radians(a)) for a in angles) + box_h / 2
    title_top = (H - (title_h + title_gap + reach_up + reach_down)) / 2
    center = (W / 2, title_top + title_h + title_gap + reach_up)
    points = [
        (center[0] + CYCLE_RADIUS * math.cos(math.radians(a)), center[1] + CYCLE_RADIUS * math.sin(math.radians(a)))
        for a in angles
    ]
    image = Image.new("RGB", (W * s, H * s), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((W / 2 * s, title_top * s), CYCLE_TITLE, fill=INK, font=title_font, anchor="mt")
    color = ARROW_COLOR

    def head(tip, direction):
        length = math.hypot(*direction)
        ux, uy = direction[0] / length, direction[1] / length
        base = (tip[0] - ux * 12, tip[1] - uy * 12)
        px, py = -uy * 6.6, ux * 6.6
        draw.polygon(
            [(tip[0] * s, tip[1] * s), ((base[0] + px) * s, (base[1] + py) * s), ((base[0] - px) * s, (base[1] - py) * s)],
            fill=color,
        )
        return base

    def clip(point, direction):
        dx, dy = direction
        t = min((box_w / 2) / abs(dx) if dx else math.inf, (box_h / 2) / abs(dy) if dy else math.inf)
        return point[0] + dx * t, point[1] + dy * t

    for index, start_angle in enumerate(angles):
        a, b = points[index], points[(index + 1) % len(points)]
        if variant == "control_cycle_curved":
            pad = 21
            arc = [
                (center[0] + CYCLE_RADIUS * math.cos(math.radians(t)), center[1] + CYCLE_RADIUS * math.sin(math.radians(t)))
                for t in (start_angle + pad + (72 - 2 * pad) * k / 40 for k in range(41))
            ]
            base = head(arc[-1], (arc[-1][0] - arc[-3][0], arc[-1][1] - arc[-3][1]))
            draw.line([(x * s, y * s) for x, y in arc[:-3]] + [(base[0] * s, base[1] * s)], fill=color, width=4 * s, joint="curve")
        else:
            direction = (b[0] - a[0], b[1] - a[1])
            start, end = clip(a, direction), clip(b, (-direction[0], -direction[1]))
            length = math.dist(start, end)
            tip = (end[0] - (end[0] - start[0]) / length * 2, end[1] - (end[1] - start[1]) / length * 2)
            base = head(tip, direction)
            draw.line([(start[0] * s, start[1] * s), (base[0] * s, base[1] * s)], fill=color, width=4 * s)
    fill = _role_fills(TARGET_LUMINANCE)[1]
    label_font = _font("arialbd.ttf", LABEL_SIZE * s)
    for point, label in zip(points, CYCLE_LABELS):
        box = (point[0] - box_w / 2, point[1] - box_h / 2, point[0] + box_w / 2, point[1] + box_h / 2)
        draw.rounded_rectangle([c * s for c in box], radius=12 * s, fill=fill)
        draw.text((point[0] * s, point[1] * s), label, fill="#ffffff", font=label_font, anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((W, H), Image.LANCZOS).save(path)
    return path


def _draw_head(draw, tip, direction, color, s):
    length = math.hypot(*direction)
    ux, uy = direction[0] / length, direction[1] / length
    base = (tip[0] - ux * 12, tip[1] - uy * 12)
    px, py = -uy * 6.6, ux * 6.6
    draw.polygon(
        [(tip[0] * s, tip[1] * s), ((base[0] + px) * s, (base[1] + py) * s), ((base[0] - px) * s, (base[1] - py) * s)],
        fill=color,
    )
    return base


def _clip_box(center, direction, half_w, half_h):
    dx, dy = direction
    t = min(half_w / abs(dx) if dx else math.inf, half_h / abs(dy) if dy else math.inf)
    return center[0] + dx * t, center[1] + dy * t


def _layout(name: str):
    """Valid non-grid layouts: (title, nodes [(label, x, y, w, h, role)], edges [(src, dst, kind)])."""
    if name == "control_loopback":
        labels = ("Start", "Draft", "Review", "Approve", "Publish")
        nodes = [(label, i * 230, 0, 160, 80, "terminal" if i in (0, 4) else "process") for i, label in enumerate(labels)]
        edges = [(i, i + 1, "straight") for i in range(4)] + [(2, 1, "loop")]
        return "Publishing Flow", nodes, edges
    if name == "control_tree":
        labels = ("Request", "Approve", "Reject", "Fulfil", "Bill", "Notify", "Archive")
        xs = (0, -220, 220, -330, -110, 110, 330)
        ys = (0, 150, 150, 300, 300, 300, 300)
        nodes = [(label, x, y, 150, 64, "terminal" if i == 0 else "process") for i, (label, x, y) in enumerate(zip(labels, xs, ys))]
        edges = [(0, 1, "straight"), (0, 2, "straight"), (1, 3, "straight"), (1, 4, "straight"), (2, 5, "straight"), (2, 6, "straight")]
        return "Request Handling", nodes, edges
    if name == "control_hub":
        labels = ("Auth", "Billing", "Search", "Email", "Storage", "Logs")
        nodes = [("Core Service", 0, 0, 190, 80, "terminal")]
        for i, label in enumerate(labels):
            angle = math.radians(60 * i)
            nodes.append((label, 360 * math.cos(angle), 210 * math.sin(angle), 150, 64, "process"))
        edges = [(0, i, "straight") for i in range(1, 7)]
        return "Service Map", nodes, edges
    raise ValueError(f"Unknown control {name!r}")


def render_control(variant: str, path: Path) -> Path:
    if variant.startswith("control_cycle"):
        return render_cycle(variant, path)
    title, nodes, edges = _layout(variant)
    s = SUPERSAMPLE
    title_font = _font("arialbd.ttf", TITLE_SIZE * s)
    _, top_px, _, bottom_px = title_font.getbbox(title)
    title_h, title_gap = (bottom_px - top_px) / s, 60
    loop_rise = 70 if any(kind == "loop" for _, _, kind in edges) else 0
    left = min(x - w / 2 for _, x, _, w, _, _ in nodes)
    right = max(x + w / 2 for _, x, _, w, _, _ in nodes)
    top = min(y - h / 2 for _, _, y, _, h, _ in nodes) - loop_rise
    bottom = max(y + h / 2 for _, _, y, _, h, _ in nodes)
    title_top = (H - (title_h + title_gap + bottom - top)) / 2
    dx, dy = W / 2 - (left + right) / 2, title_top + title_h + title_gap - top
    nodes = [(label, x + dx, y + dy, w, h, role) for label, x, y, w, h, role in nodes]
    image = Image.new("RGB", (W * s, H * s), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((W / 2 * s, title_top * s), title, fill=INK, font=title_font, anchor="mt")
    color = ARROW_COLOR
    for src, dst, kind in edges:
        _, ax, ay, aw, ah, _ = nodes[src]
        _, bx, by, bw, bh, _ = nodes[dst]
        if kind == "loop":
            start, end = (ax, ay - ah / 2), (bx, by - bh / 2)
            control = ((ax + bx) / 2, min(ay, by) - ah / 2 - 2 * loop_rise)
            curve = [
                ((1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t ** 2 * end[0],
                 (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t ** 2 * end[1])
                for t in (k / 60 for k in range(61))
            ]
            tip = curve[-2]
            base = _draw_head(draw, tip, (tip[0] - curve[-6][0], tip[1] - curve[-6][1]), color, s)
            draw.line([(x * s, y * s) for x, y in curve[:-6]] + [(base[0] * s, base[1] * s)], fill=color, width=4 * s, joint="curve")
            continue
        direction = (bx - ax, by - ay)
        start = _clip_box((ax, ay), direction, aw / 2, ah / 2)
        end = _clip_box((bx, by), (-direction[0], -direction[1]), bw / 2, bh / 2)
        length = math.dist(start, end)
        tip = (end[0] - (end[0] - start[0]) / length * 2, end[1] - (end[1] - start[1]) / length * 2)
        base = _draw_head(draw, tip, direction, color, s)
        draw.line([(start[0] * s, start[1] * s), (base[0] * s, base[1] * s)], fill=color, width=4 * s)
    fills = _role_fills(TARGET_LUMINANCE)
    role_fill = {"terminal": fills[0], "process": fills[1]}
    label_font = _font("arialbd.ttf", LABEL_SIZE * s)
    for label, x, y, w, h, role in nodes:
        box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        radius = h / 2 if role == "terminal" else 12
        draw.rounded_rectangle([c * s for c in box], radius=radius * s, fill=role_fill[role])
        draw.text((x * s, y * s), label, fill="#ffffff", font=label_font, anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((W, H), Image.LANCZOS).save(path)
    return path


def measure(scene: Scene) -> dict[str, Any]:
    """Geometry/color facts for fixture isolation, not predicted judge scores."""
    left, top, right, bottom = group_bounds(scene)
    margins = (left, W - right, top, H - bottom)
    margin_ratio = max(max(margins[:2]) / min(margins[:2]), max(margins[2:]) / min(margins[2:]))
    gaps = []
    for source, target in EDGES:
        a, b = scene.nodes[source].box, scene.nodes[target].box
        horizontal = abs(_center(a)[0] - _center(b)[0]) >= abs(_center(a)[1] - _center(b)[1])
        if horizontal:
            gaps.append((max(a[0], b[0]) - min(a[2], b[2])) / ((a[2] - a[0] + b[2] - b[0]) / 2))
        else:
            gaps.append((max(a[1], b[1]) - min(a[3], b[3])) / ((a[3] - a[1] + b[3] - b[1]) / 2))
    offsets = []
    for spec_index, axis in ((2, 0), (3, 1)):
        for key in {spec[spec_index] for spec in NODES}:
            members = [n for n, spec in zip(scene.nodes, NODES) if spec[spec_index] == key]
            centers = [_center(n.box)[axis] for n in members]
            size = statistics.mean((n.box[2] - n.box[0]) if axis == 0 else (n.box[3] - n.box[1]) for n in members)
            offsets.append((max(centers) - min(centers)) / size)
    role_styles = {}
    roles: dict[str, set[float]] = {}
    for node, role, fill in zip(scene.nodes, ROLES, scene.fills):
        signature = (
            round(node.box[2] - node.box[0]), round(node.box[3] - node.box[1]), node.kind,
            node.radius if node.kind == "round" else None, node.outline_width, node.font_face, node.font_size,
        )
        role_styles.setdefault(role, set()).add(signature)
        roles.setdefault(role, set()).update(round(h) for h in _hue_families([fill]))
    facts = {
        "worst_margin_ratio": round(margin_ratio, 2),
        "min_gap_to_node_pct": round(100 * min(gaps), 1),
        "hue_families": len(_hue_families(list(scene.fills) + [scene.arrow_color])),
        "roles_consistent": all(len(hues) <= 1 for hues in roles.values()),
        "min_box_contrast": round(min(contrast_ratio(f, BACKGROUND) for f in scene.fills), 2),
        "arrow_contrast": round(contrast_ratio(scene.arrow_color, BACKGROUND), 2),
        "max_row_col_offset_pct": round(100 * max(offsets), 1),
        "style_variants_per_role": max(len(styles) for styles in role_styles.values()),
        "arrow_treatments": len(set(zip(scene.arrow_widths, scene.arrow_heads))),
        "node_sizes": [(n.box[2] - n.box[0], n.box[3] - n.box[1]) for n in scene.nodes],
    }
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
    if backend is None:
        return results
    jobs = [(v, d, r) for v in images for d in BEAUTY_DIMENSIONS for r in range(repeats)]

    def call(job):
        variant, dimension, _ = job
        for attempt in range(1, 4):
            try:
                return judge_beauty_dimension(backend, images[variant], dimension)
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
        for d in BEAUTY_DIMENSIONS
        if d != target
    ]
    return round(statistics.mean(deltas), 2)


def isolation_metrics(results: dict) -> dict[str, dict[str, Any]]:
    baseline = results["baseline"]
    metrics = {}
    for variant, target in VARIANT_TARGET.items():
        if target is None:
            continue
        sensitivity = baseline[target]["score"] - results[variant][target]["score"]
        leaks = {d: baseline[d]["score"] - results[variant][d]["score"]
                 for d in BEAUTY_DIMENSIONS if d != target}
        worst = max(leaks, key=leaks.get)
        positive = max(0, sensitivity)
        total = positive + sum(max(0, drop) for drop in leaks.values())
        specificity = 100 * positive / total if total else 0.0
        metrics[variant] = {
            "target_dimension": target, "sensitivity": sensitivity,
            "worst_leak_dimension": worst, "worst_leak": leaks[worst], "leaks": leaks,
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
    """Read only OPENAI_API_KEY from the project's .env file."""
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


def _score_color(score: int) -> str:
    if score >= 70:
        ratio = (score - 70) / 30
        return f"rgb({int(255 - ratio * 145)},200,{int(110 - ratio * 30)})"
    return f"rgb(230,{int(90 + score / 70 * 110)},80)"


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
            f'<td class="score{" target" if d == target else ""}" style="background:{_score_color(results[v][d]["score"])}">'
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
    scenes = {variant: build_scene(variant) for variant in VARIANTS if variant not in CONTROLS}
    paths = {variant: render(scene, output / "diagrams" / f"{variant}.png") for variant, scene in scenes.items()}
    paths |= {variant: render_control(variant, output / "diagrams" / f"{variant}.png") for variant in CONTROLS}
    mode = "Fixtures only; geometry/color measurements, no judge calls or scores"
    if live and backend is None:
        if not load_local_key():
            raise RuntimeError("OPENAI_API_KEY is required for --live (environment or project .env)")
        from diagram_correctness.vlm import OpenAIBackend

        backend = OpenAIBackend(model or load_config(ROOT / "config" / "evaluator.json")["models"]["judge"])
    if backend is not None:
        mode = f"Live run using {getattr(backend, 'name', 'judge')}, {repeats} repeats per image and model-judged dimension"
    results = score(paths, backend, repeats, workers)
    full = backend is not None
    metrics = isolation_metrics(results) if full else {}
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "repeats": repeats if full else None,
        "dimensions": list(BEAUTY_DIMENSIONS),
        "variants": {v: {"target": t, "sub_check": s, "description": d} for v, (t, s, d) in VARIANTS.items()},
        "measurements": {variant: measure(scene) for variant, scene in scenes.items()},
        "results": results,
        "metrics": metrics,
        "overall_isolation": (
            round(sum(m["isolation_score"] for m in metrics.values()) / len(metrics), 1) if metrics else None
        ),
        "off_target_mean_abs": off_target_mean_abs(results) if full else None,
        "warnings": isolation_warnings(metrics),
        "controls": (
            {
                v: min(results[v][d]["score"] for d in BEAUTY_DIMENSIONS) >= 85
                for v in CONTROLS | PLACEMENT_CONTROLS
            } if full else {}
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if full:
        with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["variant", "target", *BEAUTY_DIMENSIONS])
            writer.writeheader()
            for variant, target in VARIANT_TARGET.items():
                writer.writerow({"variant": variant, "target": target, **{d: results[variant][d]["score"] for d in BEAUTY_DIMENSIONS}})
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
