from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .models import Criterion, Dimension, Issue, MetricScore, Severity
from .svg import Bounds, SvgDocument, SvgElement


@dataclass(frozen=True)
class GeometryConfig:
    position_tolerance: float = 0.06
    size_tolerance: float = 0.25
    endpoint_tolerance: float = 0.08
    minimum_font_size: float = 12.0
    overlap_area_threshold: float = 0.01
    match_threshold: float = 0.50


@dataclass(frozen=True)
class ElementMatch:
    reference: SvgElement
    candidate: SvgElement


def _family(element: SvgElement) -> str:
    if element.is_text:
        return "text"
    if element.is_connector:
        return "connector"
    if element.is_shape:
        return "shape"
    return element.tag


def _normalized_center(element: SvgElement, document: SvgDocument) -> tuple[float, float]:
    x, y = element.bounds.center
    return x / max(document.width, 1), y / max(document.height, 1)


def _match_cost(
    reference: SvgElement,
    candidate: SvgElement,
    reference_doc: SvgDocument,
    candidate_doc: SvgDocument,
) -> float:
    rx, ry = _normalized_center(reference, reference_doc)
    cx, cy = _normalized_center(candidate, candidate_doc)
    position = math.hypot(rx - cx, ry - cy)
    rw = reference.bounds.width / max(reference_doc.width, 1)
    rh = reference.bounds.height / max(reference_doc.height, 1)
    cw = candidate.bounds.width / max(candidate_doc.width, 1)
    ch = candidate.bounds.height / max(candidate_doc.height, 1)
    size = abs(rw - cw) + abs(rh - ch)
    tag_penalty = 0.0 if reference.tag == candidate.tag else 0.15
    text_penalty = 0.0
    if reference.is_text and candidate.is_text and reference.text.casefold() != candidate.text.casefold():
        text_penalty = 0.20
    return position + 0.5 * size + tag_penalty + text_penalty


def match_elements(
    reference: SvgDocument,
    candidate: SvgDocument,
    threshold: float,
) -> tuple[list[ElementMatch], list[SvgElement], list[SvgElement]]:
    remaining_reference = list(reference.elements)
    remaining_candidate = list(candidate.elements)
    matches: list[ElementMatch] = []

    # Exact text matches are strong semantic anchors.
    for ref in list(remaining_reference):
        if not ref.is_text or not ref.text:
            continue
        candidates = [
            cand
            for cand in remaining_candidate
            if cand.is_text and cand.text.casefold() == ref.text.casefold()
        ]
        if candidates:
            match = min(
                candidates,
                key=lambda cand: _match_cost(ref, cand, reference, candidate),
            )
            matches.append(ElementMatch(ref, match))
            remaining_reference.remove(ref)
            remaining_candidate.remove(match)

    possibilities = []
    for ref in remaining_reference:
        for cand in remaining_candidate:
            if _family(ref) != _family(cand):
                continue
            possibilities.append((_match_cost(ref, cand, reference, candidate), ref, cand))
    for cost, ref, cand in sorted(possibilities, key=lambda item: item[0]):
        if cost > threshold or ref not in remaining_reference or cand not in remaining_candidate:
            continue
        matches.append(ElementMatch(ref, cand))
        remaining_reference.remove(ref)
        remaining_candidate.remove(cand)
    return matches, remaining_reference, remaining_candidate


class DeterministicJudge:
    name = "deterministic-svg"

    def __init__(self, config: GeometryConfig | None = None) -> None:
        self.config = config or GeometryConfig()

    def evaluate(
        self,
        reference: SvgDocument,
        candidate: SvgDocument,
        criteria: list[Criterion],
    ) -> list[MetricScore]:
        matches, missing, _extra = match_elements(reference, candidate, self.config.match_threshold)
        handlers: dict[str, Callable[[], MetricScore]] = {
            "element_presence": lambda: self._presence(reference, missing),
            "spatial_position": lambda: self._position(reference, candidate, matches),
            "size_and_aspect": lambda: self._size(reference, candidate, matches),
            "overlap_relations": lambda: self._overlap(reference, candidate, matches),
            "connector_presence": lambda: self._connector_presence(reference, missing),
            "connector_endpoints": lambda: self._connector_endpoints(reference, candidate, matches),
            "text_accuracy": lambda: self._text_accuracy(reference, matches, missing),
            "style_fidelity": lambda: self._style(matches),
            "shape_fidelity": lambda: self._shape(matches),
            "text_overflow": lambda: self._text_overflow(candidate),
            "minimum_font_size": lambda: self._minimum_font(candidate),
        }
        results = []
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

    def _presence(self, reference: SvgDocument, missing: list[SvgElement]) -> MetricScore:
        issues = [
            Issue(f"Missing {element.tag} element", Severity.HIGH, [element.id])
            for element in missing
        ]
        return self._score(len(reference.elements), issues)

    def _position(
        self, reference: SvgDocument, candidate: SvgDocument, matches: list[ElementMatch]
    ) -> MetricScore:
        applicable = [match for match in matches if not match.reference.is_connector]
        issues = []
        for match in applicable:
            rx, ry = _normalized_center(match.reference, reference)
            cx, cy = _normalized_center(match.candidate, candidate)
            if math.hypot(rx - cx, ry - cy) > self.config.position_tolerance:
                issues.append(Issue("Element position exceeds tolerance", Severity.MEDIUM, [match.reference.id, match.candidate.id]))
        return self._score(len(applicable), issues)

    def _size(
        self, reference: SvgDocument, candidate: SvgDocument, matches: list[ElementMatch]
    ) -> MetricScore:
        applicable = [match for match in matches if match.reference.is_shape]
        issues = []
        for match in applicable:
            rw = match.reference.bounds.width / max(reference.width, 1)
            rh = match.reference.bounds.height / max(reference.height, 1)
            cw = match.candidate.bounds.width / max(candidate.width, 1)
            ch = match.candidate.bounds.height / max(candidate.height, 1)
            width_error = abs(rw - cw) / max(rw, 1e-6)
            height_error = abs(rh - ch) / max(rh, 1e-6)
            if max(width_error, height_error) > self.config.size_tolerance:
                issues.append(Issue("Element size/aspect differs from reference", Severity.MEDIUM, [match.reference.id, match.candidate.id]))
        return self._score(len(applicable), issues)

    def _overlap(
        self, reference: SvgDocument, candidate: SvgDocument, matches: list[ElementMatch]
    ) -> MetricScore:
        shapes = [match for match in matches if match.reference.is_shape]
        expected = len(shapes) * (len(shapes) - 1) // 2
        issues = []
        for index, first in enumerate(shapes):
            for second in shapes[index + 1 :]:
                ref_overlap = _overlap_fraction(first.reference.bounds, second.reference.bounds) > self.config.overlap_area_threshold
                cand_overlap = _overlap_fraction(first.candidate.bounds, second.candidate.bounds) > self.config.overlap_area_threshold
                if ref_overlap != cand_overlap:
                    issues.append(Issue("Pairwise overlap relation differs", Severity.MEDIUM, [first.candidate.id, second.candidate.id]))
        return self._score(expected, issues)

    def _connector_presence(self, reference: SvgDocument, missing: list[SvgElement]) -> MetricScore:
        expected_connectors = [element for element in reference.elements if element.is_connector]
        issues = [
            Issue("Missing connector", Severity.CRITICAL, [element.id])
            for element in missing
            if element.is_connector
        ]
        return self._score(len(expected_connectors), issues)

    def _connector_endpoints(
        self, reference: SvgDocument, candidate: SvgDocument, matches: list[ElementMatch]
    ) -> MetricScore:
        connectors = [match for match in matches if match.reference.is_connector]
        issues = []
        for match in connectors:
            if len(match.reference.points) < 2 or len(match.candidate.points) < 2:
                continue
            ref_points = (match.reference.points[0], match.reference.points[-1])
            cand_points = (match.candidate.points[0], match.candidate.points[-1])
            distances = []
            for ref_point, cand_point in zip(ref_points, cand_points):
                rx, ry = ref_point[0] / max(reference.width, 1), ref_point[1] / max(reference.height, 1)
                cx, cy = cand_point[0] / max(candidate.width, 1), cand_point[1] / max(candidate.height, 1)
                distances.append(math.hypot(rx - cx, ry - cy))
            if max(distances) > self.config.endpoint_tolerance:
                issues.append(Issue("Connector endpoint or direction differs", Severity.CRITICAL, [match.reference.id, match.candidate.id]))
        return self._score(len(connectors), issues)

    def _text_accuracy(
        self,
        reference: SvgDocument,
        matches: list[ElementMatch],
        missing: list[SvgElement],
    ) -> MetricScore:
        expected_text = [element for element in reference.elements if element.is_text]
        issues = [
            Issue("Missing or incorrect text", Severity.HIGH, [element.id])
            for element in missing
            if element.is_text
        ]
        for match in matches:
            if match.reference.is_text and match.reference.text.casefold() != match.candidate.text.casefold():
                issues.append(Issue("Text differs from reference", Severity.HIGH, [match.reference.id, match.candidate.id]))
        return self._score(len(expected_text), issues)

    def _style(self, matches: list[ElementMatch]) -> MetricScore:
        applicable = [match for match in matches if not match.reference.is_text]
        issues = []
        for match in applicable:
            reference_style = (match.reference.fill.casefold(), match.reference.stroke.casefold())
            candidate_style = (match.candidate.fill.casefold(), match.candidate.stroke.casefold())
            if reference_style != candidate_style:
                issues.append(Issue("Fill or stroke differs", Severity.LOW, [match.reference.id, match.candidate.id]))
        return self._score(len(applicable), issues)

    def _shape(self, matches: list[ElementMatch]) -> MetricScore:
        applicable = [match for match in matches if match.reference.is_shape]
        issues = [
            Issue("Shape type differs", Severity.MEDIUM, [match.reference.id, match.candidate.id])
            for match in applicable
            if match.reference.tag != match.candidate.tag
        ]
        return self._score(len(applicable), issues)

    def _text_overflow(self, candidate: SvgDocument) -> MetricScore:
        shapes = [element for element in candidate.elements if element.is_shape]
        applicable: list[tuple[SvgElement, SvgElement]] = []
        for text in (element for element in candidate.elements if element.is_text):
            center = text.bounds.center
            containers = [shape for shape in shapes if shape.bounds.contains(*center)]
            if containers:
                container = min(containers, key=lambda shape: shape.bounds.width * shape.bounds.height)
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


def _contains_bounds(container: Bounds, child: Bounds) -> bool:
    return (
        container.x <= child.x
        and container.y <= child.y
        and container.right >= child.right
        and container.bottom >= child.bottom
    )


def _overlap_fraction(first: Bounds, second: Bounds) -> float:
    width = max(0.0, min(first.right, second.right) - max(first.x, second.x))
    height = max(0.0, min(first.bottom, second.bottom) - max(first.y, second.y))
    intersection = width * height
    smaller_area = min(first.width * first.height, second.width * second.height)
    return intersection / smaller_area if smaller_area > 0 else 0.0
