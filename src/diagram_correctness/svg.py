from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_TRANSFORM_RE = re.compile(r"([a-zA-Z]+)\s*\(([^)]*)\)")


@dataclass(frozen=True)
class Bounds:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.right and self.y <= y <= self.bottom

    def intersects(self, other: "Bounds") -> bool:
        return not (
            self.right <= other.x
            or other.right <= self.x
            or self.bottom <= other.y
            or other.bottom <= self.y
        )


@dataclass(frozen=True)
class SvgElement:
    id: str
    tag: str
    bounds: Bounds
    text: str = ""
    font_size: float = 16.0
    fill: str = ""
    stroke: str = ""
    marker_start: str = ""
    marker_end: str = ""
    points: tuple[tuple[float, float], ...] = ()

    @property
    def is_text(self) -> bool:
        return self.tag == "text"

    @property
    def is_connector(self) -> bool:
        if self.tag in {"line", "polyline"}:
            return True
        return self.tag == "path" and (
            bool(self.marker_start)
            or bool(self.marker_end)
            or (
                self.fill.lower() in {"none", "transparent"}
                and self.stroke.lower() not in {"", "none", "transparent"}
            )
        )

    @property
    def is_shape(self) -> bool:
        return self.tag in {"rect", "circle", "ellipse", "polygon"}


@dataclass(frozen=True)
class SvgDocument:
    width: float
    height: float
    elements: tuple[SvgElement, ...]


# Affine matrix in SVG order: a,b,c,d,e,f.
Matrix = tuple[float, float, float, float, float, float]
IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    a1, b1, c1, d1, e1, f1 = left
    a2, b2, c2, d2, e2, f2 = right
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _apply(matrix: Matrix, point: tuple[float, float]) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    x, y = point
    return a * x + c * y + e, b * x + d * y + f


def _transform_matrix(value: str) -> Matrix:
    result = IDENTITY
    for name, args_text in _TRANSFORM_RE.findall(value or ""):
        args = [float(n) for n in _NUMBER_RE.findall(args_text)]
        name = name.lower()
        current = IDENTITY
        if name == "matrix" and len(args) >= 6:
            current = tuple(args[:6])  # type: ignore[assignment]
        elif name == "translate" and args:
            current = (1, 0, 0, 1, args[0], args[1] if len(args) > 1 else 0)
        elif name == "scale" and args:
            current = (args[0], 0, 0, args[1] if len(args) > 1 else args[0], 0, 0)
        elif name == "rotate" and args:
            radians = math.radians(args[0])
            rotation = (math.cos(radians), math.sin(radians), -math.sin(radians), math.cos(radians), 0, 0)
            if len(args) >= 3:
                before = (1, 0, 0, 1, args[1], args[2])
                after = (1, 0, 0, 1, -args[1], -args[2])
                current = _multiply(_multiply(before, rotation), after)
            else:
                current = rotation
        result = _multiply(result, current)
    return result


def _number(attrs: dict[str, str], name: str, default: float = 0.0) -> float:
    match = _NUMBER_RE.search(attrs.get(name, ""))
    return float(match.group()) if match else default


def _style(attrs: dict[str, str]) -> dict[str, str]:
    style = {}
    for declaration in attrs.get("style", "").split(";"):
        if ":" in declaration:
            key, value = declaration.split(":", 1)
            style[key.strip()] = value.strip()
    for key in ("fill", "stroke", "font-size", "marker-start", "marker-end"):
        if key in attrs:
            style[key] = attrs[key]
    return style


def _bounds_from_points(points: Iterable[tuple[float, float]]) -> Bounds | None:
    values = list(points)
    if not values:
        return None
    xs = [p[0] for p in values]
    ys = [p[1] for p in values]
    return Bounds(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def _element_points(tag: str, attrs: dict[str, str], text: str, font_size: float) -> list[tuple[float, float]]:
    if tag == "rect":
        x, y = _number(attrs, "x"), _number(attrs, "y")
        w, h = _number(attrs, "width"), _number(attrs, "height")
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    if tag == "circle":
        cx, cy, r = _number(attrs, "cx"), _number(attrs, "cy"), _number(attrs, "r")
        return [(cx - r, cy - r), (cx + r, cy + r)]
    if tag == "ellipse":
        cx, cy = _number(attrs, "cx"), _number(attrs, "cy")
        rx, ry = _number(attrs, "rx"), _number(attrs, "ry")
        return [(cx - rx, cy - ry), (cx + rx, cy + ry)]
    if tag == "line":
        return [(_number(attrs, "x1"), _number(attrs, "y1")), (_number(attrs, "x2"), _number(attrs, "y2"))]
    if tag in {"polyline", "polygon", "path"}:
        source = attrs.get("points", "") if tag != "path" else attrs.get("d", "")
        numbers = [float(n) for n in _NUMBER_RE.findall(source)]
        return list(zip(numbers[0::2], numbers[1::2]))
    if tag == "text":
        x, y = _number(attrs, "x"), _number(attrs, "y")
        width = max(font_size * 0.6 * len(text), font_size * 0.4)
        anchor = attrs.get("text-anchor", "start")
        if anchor == "middle":
            x -= width / 2
        elif anchor == "end":
            x -= width
        return [(x, y - font_size), (x + width, y + font_size * 0.2)]
    return []


def parse_svg(path: str | Path) -> SvgDocument:
    root = ET.parse(path).getroot()
    attrs = dict(root.attrib)
    viewbox = [float(n) for n in _NUMBER_RE.findall(attrs.get("viewBox", ""))]
    if len(viewbox) == 4:
        width, height = viewbox[2], viewbox[3]
    else:
        width, height = _number(attrs, "width", 1000), _number(attrs, "height", 1000)

    elements: list[SvgElement] = []
    allowed = {"rect", "circle", "ellipse", "line", "polyline", "polygon", "path", "text"}

    def visit(node: ET.Element, inherited: Matrix) -> None:
        local = _multiply(inherited, _transform_matrix(node.attrib.get("transform", "")))
        tag = node.tag.rsplit("}", 1)[-1]
        if tag in {"defs", "marker", "clipPath", "mask", "pattern", "style", "metadata", "title", "desc"}:
            return
        if tag in allowed:
            node_attrs = dict(node.attrib)
            styles = _style(node_attrs)
            text = " ".join("".join(node.itertext()).split()) if tag == "text" else ""
            font_size_match = _NUMBER_RE.search(styles.get("font-size", "16"))
            font_size = float(font_size_match.group()) if font_size_match else 16.0
            raw_points = _element_points(tag, node_attrs, text, font_size)
            transformed = [_apply(local, point) for point in raw_points]
            bounds = _bounds_from_points(transformed)
            if bounds is not None:
                element_id = node_attrs.get("id", f"{tag}-{len(elements) + 1}")
                elements.append(
                    SvgElement(
                        id=element_id,
                        tag=tag,
                        bounds=bounds,
                        text=text,
                        font_size=font_size * math.sqrt(abs(local[0] * local[3] - local[1] * local[2])),
                        fill=styles.get("fill", ""),
                        stroke=styles.get("stroke", ""),
                        marker_start=styles.get("marker-start", ""),
                        marker_end=styles.get("marker-end", ""),
                        points=tuple(transformed),
                    )
                )
        for child in node:
            visit(child, local)

    visit(root, IDENTITY)
    return SvgDocument(width=width, height=height, elements=tuple(elements))
