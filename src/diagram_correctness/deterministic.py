from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from .models import Criterion, Dimension, Issue, MetricScore, Severity
from .svg import Bounds, SvgDocument, SvgElement


@dataclass(frozen=True)
class GeometryConfig:
    endpoint_tolerance: float = 0.035
    minimum_font_size: float = 12.0
    overlap_area_threshold: float = 0.01
    label_similarity_threshold: float = 0.45
    canvas_tolerance: float = 0.5


@dataclass(frozen=True)
class LabeledShape:
    shape: SvgElement
    text: SvgElement


def _normalize_label(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _tokens(value: str) -> set[str]:
    return set(_normalize_label(value).split())


def _label_similarity(component: dict[str, str], candidate: str) -> float:
    expected = _normalize_label(component.get("label", ""))
    identifier = _normalize_label(component.get("id", "").replace("_", " "))
    actual = _normalize_label(candidate)
    if not actual:
        return 0.0
    if actual in {expected, identifier}:
        return 1.0
    if actual in expected or actual in identifier or expected in actual:
        return 0.80
    expected_tokens = _tokens(expected)
    actual_tokens = _tokens(actual)
    union = expected_tokens | actual_tokens
    jaccard = len(expected_tokens & actual_tokens) / len(union) if union else 0.0
    acronym = "".join(token[0] for token in expected.split() if token)
    if actual == acronym or (len(actual) >= 2 and actual in acronym):
        return max(jaccard, 0.75)
    return jaccard


def _area(bounds: Bounds) -> float:
    return max(0.0, bounds.width) * max(0.0, bounds.height)


def _contains_bounds(container: Bounds, child: Bounds, tolerance: float = 0.0) -> bool:
    return (
        container.x - tolerance <= child.x
        and container.y - tolerance <= child.y
        and container.right + tolerance >= child.right
        and container.bottom + tolerance >= child.bottom
    )


def _overlap_fraction(first: Bounds, second: Bounds) -> float:
    width = max(0.0, min(first.right, second.right) - max(first.x, second.x))
    height = max(0.0, min(first.bottom, second.bottom) - max(first.y, second.y))
    intersection = width * height
    smaller_area = min(_area(first), _area(second))
    return intersection / smaller_area if smaller_area > 0 else 0.0


def _distance_to_bounds(point: tuple[float, float], bounds: Bounds) -> float:
    x, y = point
    dx = max(bounds.x - x, 0.0, x - bounds.right)
    dy = max(bounds.y - y, 0.0, y - bounds.bottom)
    return math.hypot(dx, dy)


def _labeled_shapes(document: SvgDocument) -> list[LabeledShape]:
    shapes = [element for element in document.elements if element.is_shape]
    labeled: list[LabeledShape] = []
    for text in (element for element in document.elements if element.is_text and element.text.strip()):
        containers = [shape for shape in shapes if shape.bounds.contains(*text.bounds.center)]
        if not containers:
            continue
        shape = min(containers, key=lambda item: _area(item.bounds))
        labeled.append(LabeledShape(shape=shape, text=text))
    return labeled


def _match_components(
    document: SvgDocument,
    expectations: dict[str, Any],
    threshold: float,
) -> tuple[dict[str, LabeledShape], list[dict[str, str]]]:
    components = [dict(component) for component in expectations.get("components", [])]
    candidates = _labeled_shapes(document)
    remaining = list(candidates)
    matched: dict[str, LabeledShape] = {}
    missing: list[dict[str, str]] = []
    for component in components:
        ranked = sorted(
            (
                (_label_similarity(component, candidate.text.text), candidate)
                for candidate in remaining
            ),
            key=lambda item: (-item[0], item[1].shape.bounds.y, item[1].shape.bounds.x),
        )
        if not ranked or ranked[0][0] < threshold:
            missing.append(component)
            continue
        candidate = ranked[0][1]
        matched[component["id"]] = candidate
        remaining.remove(candidate)
    return matched, missing


class DeterministicJudge:
    """Candidate-only geometry judge grounded in a description-derived spec."""

    name = "deterministic-svg"

    def __init__(self, config: GeometryConfig | None = None) -> None:
        self.config = config or GeometryConfig()

    def evaluate(
        self,
        candidate: SvgDocument,
        criteria: list[Criterion],
        expectations: dict[str, Any],
    ) -> list[MetricScore]:
        matched, missing = _match_components(
            candidate, expectations, self.config.label_similarity_threshold
        )
        handlers: dict[str, Callable[[], MetricScore]] = {
            "description_presence": lambda: self._presence(expectations, missing),
            "description_order": lambda: self._order(expectations, matched),
            "canvas_bounds": lambda: self._canvas(candidate),
            "unintended_overlap": lambda: self._overlap(candidate),
            "description_connections": lambda: self._connections(
                candidate, expectations, matched
            ),
            "connector_attachment": lambda: self._connector_attachment(candidate, matched),
            "meaningful_labels": lambda: self._labels(candidate, expectations, matched),
            "text_overflow": lambda: self._text_overflow(candidate),
            "minimum_font_size": lambda: self._minimum_font(candidate),
        }
        results: list[MetricScore] = []
        for criterion in criteria:
            if not criterion.deterministic_metric:
                continue
            handler = handlers.get(criterion.deterministic_metric)
            if not handler:
                continue
            result = handler()
            result.criterion_id = criterion.id
            result.dimension = criterion.dimension
            results.append(result)
        return results

    def _score(
        self,
        expected: int,
        issues: list[Issue],
        notes: str = "",
    ) -> MetricScore:
        return MetricScore(
            criterion_id="",
            dimension=Dimension.PRESENCE,
            judge=self.name,
            expected_count=expected,
            detected_issues=len(issues),
            issues=issues,
            notes=notes,
        )

    def _presence(
        self, expectations: dict[str, Any], missing: list[dict[str, str]]
    ) -> MetricScore:
        components = expectations.get("components", [])
        issues = [
            Issue(
                f"Expected component is not visible: {component.get('label', component.get('id', 'unknown'))}",
                Severity.HIGH,
                [component.get("id", "")],
            )
            for component in missing
        ]
        return self._score(len(components), issues)

    def _order(
        self,
        expectations: dict[str, Any],
        matched: dict[str, LabeledShape],
    ) -> MetricScore:
        connections = expectations.get("connections", [])
        applicable = [
            edge
            for edge in connections
            if edge.get("source") in matched and edge.get("target") in matched
        ]
        issues = []
        for edge in applicable:
            source = matched[edge["source"]].shape.bounds.center
            target = matched[edge["target"]].shape.bounds.center
            if target[1] + 5 < source[1]:
                issues.append(
                    Issue(
                        "Connected components violate the expected top-to-bottom reading order",
                        Severity.MEDIUM,
                        [edge["source"], edge["target"]],
                    )
                )
        return self._score(len(applicable), issues)

    def _canvas(self, candidate: SvgDocument) -> MetricScore:
        canvas = Bounds(0, 0, candidate.width, candidate.height)
        elements = [
            element
            for element in candidate.elements
            if not (
                element.is_shape
                and _contains_bounds(element.bounds, canvas, self.config.canvas_tolerance)
            )
        ]
        issues = [
            Issue("Element extends outside the SVG canvas", Severity.HIGH, [element.id])
            for element in elements
            if not _contains_bounds(canvas, element.bounds, self.config.canvas_tolerance)
        ]
        return self._score(len(elements), issues)

    def _overlap(self, candidate: SvgDocument) -> MetricScore:
        shapes = [element for element in candidate.elements if element.is_shape]
        # Containers and canvas backgrounds intentionally contain other shapes.
        containers = {
            shape.id
            for shape in shapes
            if sum(
                1
                for other in shapes
                if other.id != shape.id and _contains_bounds(shape.bounds, other.bounds, 0.1)
            )
            >= 1
        }
        leaves = [shape for shape in shapes if shape.id not in containers]
        issues: list[Issue] = []
        for index, first in enumerate(leaves):
            for second in leaves[index + 1 :]:
                if (
                    _overlap_fraction(first.bounds, second.bounds)
                    > self.config.overlap_area_threshold
                ):
                    issues.append(
                        Issue(
                            "Unintended shape overlap",
                            Severity.HIGH,
                            [first.id, second.id],
                        )
                    )
        return self._score(len(leaves), issues)

    def _endpoint_component(
        self,
        point: tuple[float, float],
        candidate: SvgDocument,
        matched: dict[str, LabeledShape],
    ) -> str | None:
        threshold = self.config.endpoint_tolerance * math.hypot(
            candidate.width, candidate.height
        )
        ranked = sorted(
            (
                (
                    _distance_to_bounds(point, labeled.shape.bounds),
                    _area(labeled.shape.bounds),
                    component_id,
                )
                for component_id, labeled in matched.items()
            ),
            key=lambda item: (item[0], item[1]),
        )
        if not ranked or ranked[0][0] > threshold:
            return None
        return ranked[0][2]

    def _actual_connections(
        self,
        candidate: SvgDocument,
        matched: dict[str, LabeledShape],
    ) -> list[tuple[str | None, str | None, SvgElement]]:
        actual = []
        for connector in (element for element in candidate.elements if element.is_connector):
            if len(connector.points) < 2:
                continue
            source = self._endpoint_component(connector.points[0], candidate, matched)
            target = self._endpoint_component(connector.points[-1], candidate, matched)
            actual.append((source, target, connector))
        return actual

    def _connections(
        self,
        candidate: SvgDocument,
        expectations: dict[str, Any],
        matched: dict[str, LabeledShape],
    ) -> MetricScore:
        expected = expectations.get("connections", [])
        actual_pairs = {
            (source, target)
            for source, target, _connector in self._actual_connections(candidate, matched)
            if source and target
        }
        issues = [
            Issue(
                "Expected directed connection is missing or joins the wrong components",
                Severity.CRITICAL,
                [edge.get("source", ""), edge.get("target", "")],
            )
            for edge in expected
            if (edge.get("source"), edge.get("target")) not in actual_pairs
        ]
        return self._score(len(expected), issues)

    def _connector_attachment(
        self,
        candidate: SvgDocument,
        matched: dict[str, LabeledShape],
    ) -> MetricScore:
        actual = self._actual_connections(candidate, matched)
        issues = []
        for source, target, connector in actual:
            if source is None or target is None:
                issues.append(
                    Issue(
                        "Connector endpoint is not attached to a description-expected component",
                        Severity.CRITICAL,
                        [connector.id],
                    )
                )
        return self._score(len(actual), issues)

    def _labels(
        self,
        candidate: SvgDocument,
        expectations: dict[str, Any],
        matched: dict[str, LabeledShape],
    ) -> MetricScore:
        components = expectations.get("components", [])
        issues: list[Issue] = []
        for component in components:
            labeled = matched.get(component.get("id"))
            if labeled is None:
                issues.append(
                    Issue(
                        "Expected meaningful label is missing",
                        Severity.HIGH,
                        [component.get("id", "")],
                    )
                )
            elif _normalize_label(labeled.text.text) != _normalize_label(
                component.get("label", "")
            ):
                issues.append(
                    Issue(
                        f"Label is incomplete or generic; expected '{component.get('label', '')}'",
                        Severity.HIGH,
                        [labeled.text.id],
                    )
                )
        forbidden = {
            _normalize_label(value)
            for value in expectations.get("forbidden_placeholders", [])
            if _normalize_label(value)
        } | {"tbd", "todo", "placeholder", "thing", "thing 1", "layer", "unknown"}
        for text in (element for element in candidate.elements if element.is_text):
            normalized = _normalize_label(text.text)
            if normalized in forbidden or "???" in text.text:
                issues.append(
                    Issue("Placeholder or meaningless label is visible", Severity.HIGH, [text.id])
                )
        return self._score(max(len(components), 1), issues)

    def _text_overflow(self, candidate: SvgDocument) -> MetricScore:
        shapes = [element for element in candidate.elements if element.is_shape]
        applicable: list[tuple[SvgElement, SvgElement]] = []
        for text in (element for element in candidate.elements if element.is_text):
            containers = [shape for shape in shapes if shape.bounds.contains(*text.bounds.center)]
            if containers:
                container = min(containers, key=lambda shape: _area(shape.bounds))
                applicable.append((text, container))
        issues = [
            Issue("Text overflows its containing shape", Severity.HIGH, [text.id, shape.id])
            for text, shape in applicable
            if not _contains_bounds(shape.bounds, text.bounds)
        ]
        return self._score(
            len(applicable),
            issues,
            "Text bounds use SVG positions and a conservative font-width estimate.",
        )

    def _minimum_font(self, candidate: SvgDocument) -> MetricScore:
        texts = [element for element in candidate.elements if element.is_text]
        issues = [
            Issue(
                f"Font size {text.font_size:g}px is below {self.config.minimum_font_size:g}px",
                Severity.HIGH,
                [text.id],
            )
            for text in texts
            if text.font_size < self.config.minimum_font_size
        ]
        return self._score(len(texts), issues)
