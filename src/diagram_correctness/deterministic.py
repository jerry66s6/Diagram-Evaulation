from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

from .models import (
    ComponentVerdict,
    Criterion,
    Dimension,
    Issue,
    MetricScore,
    PresencePolicy,
    Severity,
    Verdict,
    component_metrics,
)
from .svg import Bounds, SvgDocument, SvgElement


@dataclass(frozen=True)
class GeometryConfig:
    endpoint_tolerance: float = 0.035
    minimum_font_size: float = 12.0
    overlap_area_threshold: float = 0.01
    label_similarity_threshold: float = 0.45
    canvas_tolerance: float = 0.5
    # Number of expected connections that must be drawn before an element whose label
    # does not match is accepted as playing a component's role (e.g. a box labeled "Layer"
    # between the feed-forward network and the output).
    min_role_links: int = 2


@dataclass(frozen=True)
class LabeledShape:
    shape: SvgElement
    text: SvgElement


@dataclass(frozen=True)
class ComponentMatch:
    """A drawn element assigned to one expected component."""

    shape: SvgElement
    text: SvgElement | None
    method: str  # "label": matched by its text; "role": matched by its connections
    similarity: float = 0.0
    evidence: str = ""


# Placeholder labels rejected even when the description does not list them.
_DEFAULT_PLACEHOLDERS = {"tbd", "todo", "placeholder", "thing", "thing 1", "layer", "unknown"}


def _normalize_label(value: str) -> str:
    # Map the multiplication sign to "x" so repetition markers such as "N×" keep a
    # distinctive token ("nx") instead of collapsing to the single letter "n".
    value = value.casefold().replace("×", "x")
    return " ".join(re.findall(r"[a-z0-9]+", value))


def _tokens(value: str) -> set[str]:
    return set(_normalize_label(value).split())


def _is_abbreviation(actual: str, expected: str) -> bool:
    """True when a single-token label abbreviates the expected label ("FFN" for
    "Position-wise Feed-Forward Network"). Both arguments are normalized labels."""
    if not actual or " " in actual or len(actual) < 2:
        return False
    acronym = "".join(token[0] for token in expected.split() if token)
    return len(expected.split()) > 1 and actual in acronym


def _label_similarity(component: dict[str, str], candidate: str) -> float:
    expected = _normalize_label(component.get("label", ""))
    identifier = _normalize_label(component.get("id", "").replace("_", " "))
    actual = _normalize_label(candidate)
    if not actual:
        return 0.0
    if actual in {expected, identifier}:
        return 1.0
    expected_tokens = _tokens(expected)
    actual_tokens = _tokens(actual)
    identifier_tokens = _tokens(identifier)
    # Whole-token containment ("Self-Attention" within "Multi-Head Self-Attention").
    # Raw substring checks are avoided because a one- or two-letter label would
    # otherwise match almost every other label.
    if expected_tokens and (actual_tokens <= expected_tokens or expected_tokens <= actual_tokens):
        return 0.80
    if identifier_tokens and actual_tokens <= identifier_tokens:
        return 0.80
    union = expected_tokens | actual_tokens
    jaccard = len(expected_tokens & actual_tokens) / len(union) if union else 0.0
    if _is_abbreviation(actual, expected):
        return max(jaccard, 0.75)
    return jaccard


def _placeholders(expectations: dict[str, Any]) -> set[str]:
    listed = {
        _normalize_label(value)
        for value in expectations.get("forbidden_placeholders", [])
        if _normalize_label(value)
    }
    return listed | _DEFAULT_PLACEHOLDERS


def _is_placeholder(text: str, placeholders: set[str]) -> bool:
    return _normalize_label(text) in placeholders or "???" in text


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


def _match_by_label(
    document: SvgDocument,
    components: list[dict[str, str]],
    threshold: float,
) -> dict[str, ComponentMatch]:
    """Assign drawn labels to expected components, best pairs first.

    All (component, label) pairs are ranked together, so the result does not depend on
    inventory order: a weak partial match can no longer take a label that another
    component matches exactly. Ties go to the earlier component and the higher element,
    which keeps repeated labels (two "Add & Norm" boxes) in reading order. Each drawn
    label is used for at most one component.
    """
    labeled = _labeled_shapes(document)
    pairs = []
    for index, component in enumerate(components):
        for candidate in labeled:
            similarity = _label_similarity(component, candidate.text.text)
            if similarity >= threshold:
                pairs.append((similarity, index, candidate))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2].shape.bounds.y, item[2].shape.bounds.x))

    matched: dict[str, ComponentMatch] = {}
    used_texts: set[int] = set()
    for similarity, index, candidate in pairs:
        component_id = components[index]["id"]
        if component_id in matched or id(candidate.text) in used_texts:
            continue
        matched[component_id] = ComponentMatch(
            shape=candidate.shape,
            text=candidate.text,
            method="label",
            similarity=similarity,
            evidence=f"label '{candidate.text.text}' matches (similarity {similarity:.2f})",
        )
        used_texts.add(id(candidate.text))
    return matched


def _nearest_key(
    point: tuple[float, float],
    pool: list[tuple[Any, SvgElement]],
    threshold: float,
) -> Any:
    """Key of the closest shape to a connector endpoint (smallest shape wins ties)."""
    ranked = sorted(
        ((_distance_to_bounds(point, shape.bounds), _area(shape.bounds), index) for index, (_key, shape) in enumerate(pool)),
    )
    if not ranked or ranked[0][0] > threshold:
        return None
    return pool[ranked[0][2]][0]


def _match_by_role(
    document: SvgDocument,
    components: list[dict[str, str]],
    connections: list[dict[str, str]],
    matched: dict[str, ComponentMatch],
    config: GeometryConfig,
    policy: PresencePolicy,
    placeholders: set[str],
) -> dict[str, ComponentMatch]:
    """Find drawn elements that play an expected role despite a non-matching label.

    An unmatched component is assigned to an unused element when at least
    ``config.min_role_links`` of its expected connections to already-matched components
    are drawn to that element, and no other element has as much support. This mirrors
    the MISLABELED verdict: the role is drawn, only its label is wrong or missing.
    """
    matched = dict(matched)
    labels = {id(item.shape): item.text for item in _labeled_shapes(document)}
    used = {id(match.shape) for match in matched.values()}
    threshold = config.endpoint_tolerance * math.hypot(document.width, document.height)
    connectors = [
        element for element in document.elements if element.is_connector and len(element.points) >= 2
    ]

    def eligible(shape: SvgElement) -> bool:
        if not shape.is_shape or id(shape) in used:
            return False
        text = labels.get(id(shape))
        if text is None and shape.tag not in {"rect", "circle", "ellipse"}:
            # Unlabeled polygons are usually arrowheads, not diagram nodes.
            return False
        if not policy.placeholder_counts_as_present and (
            text is None or _is_placeholder(text.text, placeholders)
        ):
            return False
        # Groups and backgrounds that contain matched components are not candidates.
        return not any(_contains_bounds(shape.bounds, match.shape.bounds) for match in matched.values())

    for _ in range(len(components)):
        candidates = [shape for shape in document.elements if eligible(shape)]
        pool: list[tuple[Any, SvgElement]] = [
            (component_id, match.shape) for component_id, match in matched.items()
        ] + [(("candidate", index), shape) for index, shape in enumerate(candidates)]
        endpoints = [
            (
                _nearest_key(connector.points[0], pool, threshold),
                _nearest_key(connector.points[-1], pool, threshold),
            )
            for connector in connectors
        ]

        best: tuple[int, int, str, SvgElement, set[tuple[str, str]]] | None = None
        for order, component in enumerate(components):
            component_id = component["id"]
            if component_id in matched:
                continue
            incoming = {
                edge["source"]
                for edge in connections
                if edge.get("target") == component_id and edge.get("source") in matched
            }
            outgoing = {
                edge["target"]
                for edge in connections
                if edge.get("source") == component_id and edge.get("target") in matched
            }
            support: dict[Any, set[tuple[str, str]]] = {}
            for source, target in endpoints:
                if source in incoming and isinstance(target, tuple):
                    support.setdefault(target, set()).add((source, component_id))
                if target in outgoing and isinstance(source, tuple):
                    support.setdefault(source, set()).add((component_id, target))
            ranked = sorted(support.items(), key=lambda item: -len(item[1]))
            if not ranked or len(ranked[0][1]) < config.min_role_links:
                continue
            if len(ranked) > 1 and len(ranked[1][1]) == len(ranked[0][1]):
                continue  # two elements fit equally well; do not guess
            links = len(ranked[0][1])
            if best is None or (links, -order) > (best[0], -best[1]):
                best = (links, order, component_id, candidates[ranked[0][0][1]], ranked[0][1])

        if best is None:
            break
        links, _order, component_id, shape, edges = best
        text = labels.get(id(shape))
        drawn = ", ".join(f"{source}->{target}" for source, target in sorted(edges))
        matched[component_id] = ComponentMatch(
            shape=shape,
            text=text,
            method="role",
            evidence=(
                f"element labeled '{text.text if text else ''}' is connected as expected ({drawn})"
            ),
        )
        used.add(id(shape))
    return matched


def _match_components(
    document: SvgDocument,
    expectations: dict[str, Any],
    config: GeometryConfig,
    policy: PresencePolicy,
) -> dict[str, ComponentMatch]:
    components = [dict(component) for component in expectations.get("components", [])]
    matched = _match_by_label(document, components, config.label_similarity_threshold)
    return _match_by_role(
        document,
        components,
        expectations.get("connections", []),
        matched,
        config,
        policy,
        _placeholders(expectations),
    )


class DeterministicJudge:
    """Candidate-only geometry judge grounded in a description-derived spec."""

    name = "deterministic-svg"

    def __init__(
        self,
        config: GeometryConfig | None = None,
        policy: PresencePolicy | None = None,
    ) -> None:
        self.config = config or GeometryConfig()
        self.policy = policy or PresencePolicy()

    def evaluate(
        self,
        candidate: SvgDocument,
        criteria: list[Criterion],
        expectations: dict[str, Any],
    ) -> list[MetricScore]:
        matched = _match_components(candidate, expectations, self.config, self.policy)
        verdicts = self.component_verdicts(candidate, expectations, matched)
        by_metric = {criterion.deterministic_metric: criterion for criterion in criteria}
        atomic = {
            metric.criterion_id: metric
            for metric in component_metrics(
                verdicts,
                self.name,
                presence=by_metric.get("description_presence"),
                details=by_metric.get("meaningful_labels"),
            )
        }
        stray = self._stray_placeholders(candidate, expectations, matched)
        for metric in atomic.values():
            if metric.dimension == Dimension.DETAILS and stray:
                metric.notes += " Placeholder text not tied to an expected component (reported, not scored): " + ", ".join(
                    f"'{text}'" for text in stray
                )

        def atomic_metric(metric_name: str) -> MetricScore:
            return atomic[by_metric[metric_name].id]

        handlers: dict[str, Callable[[], MetricScore]] = {
            "description_presence": lambda: atomic_metric("description_presence"),
            "description_order": lambda: self._order(expectations, matched),
            "canvas_bounds": lambda: self._canvas(candidate),
            "unintended_overlap": lambda: self._overlap(candidate),
            "description_connections": lambda: self._connections(
                candidate, expectations, matched
            ),
            "connector_attachment": lambda: self._connector_attachment(candidate, matched),
            "meaningful_labels": lambda: atomic_metric("meaningful_labels"),
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

    def component_verdicts(
        self,
        candidate: SvgDocument,
        expectations: dict[str, Any],
        matched: dict[str, ComponentMatch] | None = None,
    ) -> list[ComponentVerdict]:
        """One verdict per inventory component, using the same IDs as the GPT judge."""
        if matched is None:
            matched = _match_components(candidate, expectations, self.config, self.policy)
        width = candidate.width or 1.0
        height = candidate.height or 1.0
        verdicts: list[ComponentVerdict] = []
        for component in expectations.get("components", []):
            component_id = component.get("id", "")
            match = matched.get(component_id)
            if match is None:
                verdicts.append(
                    ComponentVerdict(
                        component_id=component_id,
                        verdict=Verdict.ABSENT,
                        judge=self.name,
                        evidence="no drawn label or connected element matches this component",
                    )
                )
                continue
            visible_text = match.text.text if match.text is not None else ""
            expected = _normalize_label(component.get("label", ""))
            actual = _normalize_label(visible_text)
            correct = match.method == "label" and (
                actual == expected
                or (self.policy.accept_abbreviations and _is_abbreviation(actual, expected))
            )
            bounds = match.shape.bounds
            verdicts.append(
                ComponentVerdict(
                    component_id=component_id,
                    verdict=Verdict.PRESENT if correct else Verdict.MISLABELED,
                    judge=self.name,
                    visible_text=visible_text,
                    evidence=match.evidence,
                    bounds=(
                        round(bounds.x / width, 4),
                        round(bounds.y / height, 4),
                        round(bounds.right / width, 4),
                        round(bounds.bottom / height, 4),
                    ),
                )
            )
        return verdicts

    def _stray_placeholders(
        self,
        candidate: SvgDocument,
        expectations: dict[str, Any],
        matched: dict[str, ComponentMatch],
    ) -> list[str]:
        """Placeholder text that is not the label of any matched component."""
        placeholders = _placeholders(expectations)
        assigned = {id(match.text) for match in matched.values() if match.text is not None}
        return [
            element.text
            for element in candidate.elements
            if element.is_text and id(element) not in assigned and _is_placeholder(element.text, placeholders)
        ]

    def _order(
        self,
        expectations: dict[str, Any],
        matched: dict[str, ComponentMatch],
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
        matched: dict[str, ComponentMatch],
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
        matched: dict[str, ComponentMatch],
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
        matched: dict[str, ComponentMatch],
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
        matched: dict[str, ComponentMatch],
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
