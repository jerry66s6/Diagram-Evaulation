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
    def directed_points(self) -> tuple[tuple[float, float], ...]:
        """Points from the arrow's tail to its head.

        An arrowhead drawn only with marker-start means the arrow points back toward the
        first point, so the drawing order is reversed. Otherwise drawing order is kept,
        which is also how tools such as Graphviz draw edges with separate arrowheads.
        """
        if self.marker_start and not self.marker_end:
            return tuple(reversed(self.points))
        return self.points

    @property
    def has_arrowhead(self) -> bool:
        return bool(self.marker_start or self.marker_end)

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
    # Rendered pixels per SVG user unit, from the root width/height versus the viewBox.
    render_scale: float = 1.0


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


# Presentation properties that SVG children inherit from their parent group.
_INHERITED = ("fill", "stroke", "font-size", "marker-start", "marker-end", "text-anchor")
_LENGTH_UNITS = {"": 1.0, "px": 1.0, "pt": 4 / 3, "pc": 16.0, "mm": 96 / 25.4, "cm": 96 / 2.54, "in": 96.0}
_FONT_KEYWORDS = {"xx-small": 9.0, "x-small": 10.0, "small": 13.0, "medium": 16.0, "large": 18.0, "x-large": 24.0, "xx-large": 32.0}
_LENGTH_RE = re.compile(r"^\s*([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)\s*([a-zA-Z%]*)\s*$")


def _style(attrs: dict[str, str]) -> dict[str, str]:
    """The element's own inheritable properties; style="..." overrides attributes."""
    style = {key: attrs[key] for key in _INHERITED if key in attrs}
    for declaration in attrs.get("style", "").split(";"):
        if ":" in declaration:
            key, value = declaration.split(":", 1)
            if key.strip() in _INHERITED:
                style[key.strip()] = value.strip()
    return style


def _length_px(value: str) -> float | None:
    """An absolute CSS length in pixels, or None for percentages and unknown units."""
    match = _LENGTH_RE.match(value or "")
    if not match or match.group(2).lower() not in _LENGTH_UNITS:
        return None
    return float(match.group(1)) * _LENGTH_UNITS[match.group(2).lower()]


def _font_size(value: str, parent: float) -> float:
    """Resolve a font-size declaration against the inherited size, in user units."""
    value = (value or "").strip().lower()
    if value in _FONT_KEYWORDS:
        return _FONT_KEYWORDS[value]
    if value == "smaller":
        return parent / 1.2
    if value == "larger":
        return parent * 1.2
    match = _LENGTH_RE.match(value)
    if not match:
        return parent
    number, unit = float(match.group(1)), match.group(2)
    if unit == "em":
        return number * parent
    if unit == "rem":
        return number * 16.0
    if unit == "%":
        return number * parent / 100
    return number * _LENGTH_UNITS.get(unit, 1.0) if unit in _LENGTH_UNITS else parent


def _render_scale(attrs: dict[str, str], viewbox: list[float]) -> float:
    """Pixels per user unit when the root size differs from its viewBox (aspect ratio kept)."""
    if len(viewbox) != 4 or viewbox[2] <= 0 or viewbox[3] <= 0:
        return 1.0
    width, height = _length_px(attrs.get("width", "")), _length_px(attrs.get("height", ""))
    scales = [value for value in (width / viewbox[2] if width else None, height / viewbox[3] if height else None) if value]
    return min(scales) if scales else 1.0


def _bounds_from_points(points: Iterable[tuple[float, float]]) -> Bounds | None:
    values = list(points)
    if not values:
        return None
    xs = [p[0] for p in values]
    ys = [p[1] for p in values]
    return Bounds(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def _element_points(
    tag: str, attrs: dict[str, str], text: str, font_size: float, anchor: str = "start"
) -> list[tuple[float, float]]:
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

    def visit(node: ET.Element, inherited: Matrix, inherited_style: dict[str, str]) -> None:
        local = _multiply(inherited, _transform_matrix(node.attrib.get("transform", "")))
        tag = node.tag.rsplit("}", 1)[-1]
        if tag in {"defs", "marker", "clipPath", "mask", "pattern", "style", "metadata", "title", "desc"}:
            return
        node_attrs = dict(node.attrib)
        own = _style(node_attrs)
        styles = {**inherited_style, **own}
        parent_size = float(inherited_style.get("font-size", "16"))
        font_size = _font_size(own["font-size"], parent_size) if "font-size" in own else parent_size
        styles["font-size"] = repr(font_size)
        if tag in allowed:
            text = " ".join("".join(node.itertext()).split()) if tag == "text" else ""
            raw_points = _element_points(tag, node_attrs, text, font_size, styles.get("text-anchor", "start"))
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
            visit(child, local, styles)

    visit(root, IDENTITY, {"font-size": "16.0"})
    return SvgDocument(
        width=width, height=height, elements=tuple(elements), render_scale=_render_scale(attrs, viewbox)
    )
