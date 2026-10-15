"""Deterministic Canvas Fill check: how much of the canvas the diagram's elements occupy.

The check works on a rendered image (PNG/JPEG), so it applies to any diagram source.
1. Estimate the background as the most common border color.
2. Mark every pixel that differs from it as ink.
3. Fill enclosed holes, so outline-only boxes count by their full area.
4. Remove thin structures (arrows, text strokes, hairlines, small dots) with a
   morphological opening sized to the canvas.
5. Footprint = remaining area / canvas area. Only a too-SMALL footprint is penalized;
   heavy, full diagrams are a design choice and score 100.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

INK_THRESHOLD = 16
FULL_SCORE_AT = 8.0
ZERO_SCORE_AT = 2.0
WORKING_SIZE = 480
_BACKGROUND = 128


@dataclass(frozen=True)
class CanvasFill:
    footprint_pct: float
    score: int
    observation: str
    reasoning: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "footprint_pct": self.footprint_pct,
            "score": self.score,
            "observation": self.observation,
            "reasoning": self.reasoning,
        }


def _background(image: Image.Image) -> tuple[int, int, int]:
    width, height = image.size
    step = max(1, min(width, height) // 200)
    pixels = image.load()
    border = [pixels[x, y] for x in range(0, width, step) for y in (0, height - 1)]
    border += [pixels[x, y] for y in range(0, height, step) for x in (0, width - 1)]
    return Counter(border).most_common(1)[0][0]


def footprint_mask(image: Image.Image) -> Image.Image:
    """Binary mask (255 = element footprint) after hole filling and thin-structure removal."""
    image = image.convert("RGB")
    # Footprint is an area ratio, so a downscaled copy gives the same answer much faster.
    scale = WORKING_SIZE / max(image.size)
    if scale < 1:
        image = image.resize((max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))), Image.BOX)
    width, height = image.size
    difference = ImageChops.difference(image, Image.new("RGB", image.size, _background(image)))
    red, green, blue = difference.split()
    channel_max = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    mask = channel_max.point(lambda value: 255 if value > INK_THRESHOLD else 0)

    flooded = mask.copy()
    pixels = flooded.load()
    border = [(x, y) for x in range(width) for y in (0, height - 1)]
    border += [(x, y) for y in range(height) for x in (0, width - 1)]
    for point in border:
        if pixels[point] == 0:
            ImageDraw.floodfill(flooded, point, _BACKGROUND)
    filled = flooded.point(lambda value: 0 if value == _BACKGROUND else 255)

    kernel = max(3, round(min(width, height) / 80) | 1)
    return filled.filter(ImageFilter.MinFilter(kernel)).filter(ImageFilter.MaxFilter(kernel))


def footprint_percent(image: Image.Image) -> float:
    mask = footprint_mask(image)
    covered = mask.histogram()[255]
    return 100 * covered / (mask.size[0] * mask.size[1])


def score_footprint(percent: float) -> int:
    if percent >= FULL_SCORE_AT:
        return 100
    if percent <= ZERO_SCORE_AT:
        return 0
    return round(100 * (percent - ZERO_SCORE_AT) / (FULL_SCORE_AT - ZERO_SCORE_AT))


def judge_canvas_fill(image_path: str | Path) -> CanvasFill:
    with Image.open(image_path) as image:
        percent = footprint_percent(image)
    score = score_footprint(percent)
    observation = f"Diagram elements occupy {percent:.1f}% of the canvas (thin arrows and text strokes excluded)."
    if score == 100:
        reasoning = f"At or above the {FULL_SCORE_AT:.0f}% footprint where the canvas reads as used."
    elif score == 0:
        reasoning = f"At or below {ZERO_SCORE_AT:.0f}%: the canvas is essentially empty around tiny elements."
    else:
        reasoning = (
            f"Between {ZERO_SCORE_AT:.0f}% and {FULL_SCORE_AT:.0f}%: much of the canvas is empty relative to "
            "the size of the elements."
        )
    return CanvasFill(round(percent, 2), score, observation, reasoning)
