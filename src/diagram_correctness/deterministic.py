from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from .labels import is_abbreviation, label_tokens, normalize_label
from .models import (
    METRIC_ATTACHMENT,
    METRIC_CONNECTION_LABELS,
    METRIC_CONNECTIONS,
    METRIC_LABELS,
    METRIC_MINIMUM_FONT,
    METRIC_PRESENCE,
    METRIC_UNEXPECTED,
    ComponentVerdict,
    ConnectionStatus,
    ConnectionVerdict,
    Criterion,
    Dimension,
    Issue,
    MetricScore,
    PresencePolicy,
    Severity,
    Verdict,
    component_metrics,
    connection_metrics,
    drawn_component_ids,
    feedback_edges,
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


# Shared with the GPT judge so that both judges compare labels the same way.
_normalize_label = normalize_label
_tokens = label_tokens
_is_abbreviation = is_abbreviation


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


def _distance_to_polyline(point: tuple[float, float], points: tuple[tuple[float, float], ...]) -> float:
    best = math.inf
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx, dy = x2 - x1, y2 - y1
        length = dx * dx + dy * dy
        t = 0.0 if length == 0 else max(0.0, min(1.0, ((point[0] - x1) * dx + (point[1] - y1) * dy) / length))
        best = min(best, math.hypot(point[0] - (x1 + t * dx), point[1] - (y1 + t * dy)))
    return best


def _union(first: Bounds, second: Bounds) -> Bounds:
    x, y = min(first.x, second.x), min(first.y, second.y)
    return Bounds(x, y, max(first.right, second.right) - x, max(first.bottom, second.bottom) - y)




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
                _nearest_key(connector.directed_points[0], pool, threshold),
                _nearest_key(connector.directed_points[-1], pool, threshold),
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
        matched = self.match_components(candidate, expectations)
        verdicts = self.component_verdicts(candidate, expectations, matched)
        by_metric = {criterion.deterministic_metric: criterion for criterion in criteria}
        atomic = {
            metric.criterion_id: metric
            for metric in component_metrics(
                verdicts,
                self.name,
                presence=by_metric.get(METRIC_PRESENCE),
                details=by_metric.get(METRIC_LABELS),
            )
        }
        drawn = drawn_component_ids(verdicts)
        links, extra_arrows = self.connection_verdicts(candidate, expectations, matched)
        atomic.update(
            {
                metric.criterion_id: metric
                for metric in connection_metrics(
                    links,
                    drawn,
                    extra_arrows,
                    self.name,
                    recall=by_metric.get(METRIC_CONNECTIONS),
                    precision=by_metric.get(METRIC_ATTACHMENT),
                    labels=by_metric.get(METRIC_CONNECTION_LABELS),
                    accept_abbreviations=self.policy.accept_abbreviations,
                )
            }
        )
        stray = self._stray_placeholders(candidate, expectations, matched)
        for metric in atomic.values():
            if metric.dimension == Dimension.DETAILS and stray:
                metric.notes += " Placeholder text not tied to an expected component (reported, not scored): " + ", ".join(
                    f"'{text}'" for text in stray
                )

        def atomic_metric(metric_name: str) -> MetricScore:
            return atomic[by_metric[metric_name].id]

        handlers: dict[str, Callable[[], MetricScore]] = {
            METRIC_PRESENCE: lambda: atomic_metric(METRIC_PRESENCE),
            METRIC_UNEXPECTED: lambda: self._unexpected(candidate, matched, len(drawn)),
            "description_order": lambda: self._order(expectations, matched),
            "canvas_bounds": lambda: self._canvas(candidate, matched),
            "unintended_overlap": lambda: self._overlap(candidate),
            METRIC_CONNECTIONS: lambda: atomic_metric(METRIC_CONNECTIONS),
            METRIC_ATTACHMENT: lambda: atomic_metric(METRIC_ATTACHMENT),
            METRIC_CONNECTION_LABELS: lambda: atomic_metric(METRIC_CONNECTION_LABELS),
            METRIC_LABELS: lambda: atomic_metric(METRIC_LABELS),
            "text_overflow": lambda: self._text_overflow(candidate),
            METRIC_MINIMUM_FONT: lambda: self._minimum_font(candidate),
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

    def match_components(self, candidate: SvgDocument, expectations: dict[str, Any]) -> dict[str, ComponentMatch]:
        return _match_components(candidate, expectations, self.config, self.policy)

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
            matched = self.match_components(candidate, expectations)
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
        """Connected components follow one consistent reading direction.

        The direction is inferred from the drawing itself (the dominant axis and sense of
        the expected connections), so left-to-right and bottom-to-top diagrams are not
        penalized. Connections that close a loop back to an earlier step are skipped.
        """
        connections = expectations.get("connections", [])
        feedback = feedback_edges(expectations.get("components", []), connections)
        applicable = [
            edge
            for index, edge in enumerate(connections)
            if index not in feedback
            and edge.get("source") != edge.get("target")
            and edge.get("source") in matched
            and edge.get("target") in matched
        ]
        deltas = []
        for edge in applicable:
            source = matched[edge["source"]].shape.bounds.center
            target = matched[edge["target"]].shape.bounds.center
            deltas.append((target[0] - source[0], target[1] - source[1]))
        axis = 1 if sum(abs(dy) for _dx, dy in deltas) >= sum(abs(dx) for dx, _dy in deltas) else 0
        sense = 1 if sum(delta[axis] for delta in deltas) >= 0 else -1
        direction = {(1, 1): "top-to-bottom", (1, -1): "bottom-to-top", (0, 1): "left-to-right", (0, -1): "right-to-left"}[(axis, sense)]
        issues = [
            Issue(
                f"Connection runs against the diagram's {direction} reading direction",
                Severity.MEDIUM,
                [edge["source"], edge["target"]],
            )
            for edge, delta in zip(applicable, deltas)
            if sense * delta[axis] < -5
        ]
        skipped = len(connections) - len(applicable)
        return self._score(
            len(applicable),
            issues,
            f"Reading direction inferred from the drawing: {direction}."
            + (f" {skipped} connection(s) skipped: loops back to an earlier step or a component is not drawn." if skipped else ""),
        )

    def _canvas(self, candidate: SvgDocument, matched: dict[str, ComponentMatch]) -> MetricScore:
        """Each drawn component (box plus label), arrow, or other element counts once."""
        canvas = Bounds(0, 0, candidate.width, candidate.height)
        tolerance = self.config.canvas_tolerance
        units: list[tuple[str, Bounds]] = []
        assigned: set[int] = set()
        for component_id, match in matched.items():
            bounds = match.shape.bounds if match.text is None else _union(match.shape.bounds, match.text.bounds)
            units.append((component_id, bounds))
            assigned.update({id(match.shape), id(match.text)})
        for element in candidate.elements:
            if id(element) in assigned:
                continue
            if element.is_shape and _contains_bounds(element.bounds, canvas, tolerance):
                continue  # canvas background
            units.append((element.id, element.bounds))
        issues = [
            Issue("Content extends outside the SVG canvas", Severity.HIGH, [unit_id])
            for unit_id, bounds in units
            if not _contains_bounds(canvas, bounds, tolerance)
        ]
        return self._score(len(units), issues, "A component and its label count as one unit.")

    def _overlap(self, candidate: SvgDocument) -> MetricScore:
        """Each shape that collides with another counts once, so the numerator and the
        denominator are both shapes (not pairs)."""
        shapes = [element for element in candidate.elements if element.is_shape]
        # Containers and canvas backgrounds intentionally contain other shapes.
        containers = {
            shape.id
            for shape in shapes
            if any(other.id != shape.id and _contains_bounds(shape.bounds, other.bounds, 0.1) for other in shapes)
        }
        labeled = {id(item.shape) for item in _labeled_shapes(candidate)}
        # Unlabeled polygons are arrowheads; touching their target box is not a collision.
        leaves = [
            shape
            for shape in shapes
            if shape.id not in containers and not (shape.tag == "polygon" and id(shape) not in labeled)
        ]
        partner: dict[str, str] = {}
        for index, first in enumerate(leaves):
            for second in leaves[index + 1 :]:
                if _overlap_fraction(first.bounds, second.bounds) > self.config.overlap_area_threshold:
                    partner.setdefault(first.id, second.id)
                    partner.setdefault(second.id, first.id)
        issues = [
            Issue("Shape collides with another shape", Severity.HIGH, [shape_id, other])
            for shape_id, other in sorted(partner.items())
        ]
        return self._score(len(leaves), issues, "Each colliding shape counts once.")

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
        """Drawn arrows as (source, target, element), oriented tail to head.

        Lines without an arrowhead that touch no component at either end, or that start
        and end on the same component (a divider inside a box), are decoration and are
        skipped.
        """
        actual = []
        for connector in (element for element in candidate.elements if element.is_connector):
            points = connector.directed_points
            if len(points) < 2:
                continue
            source = self._endpoint_component(points[0], candidate, matched)
            target = self._endpoint_component(points[-1], candidate, matched)
            if not connector.has_arrowhead and (source == target):
                continue
            actual.append((source, target, connector))
        return actual

    def connection_verdicts(
        self,
        candidate: SvgDocument,
        expectations: dict[str, Any],
        matched: dict[str, ComponentMatch],
    ) -> tuple[list[ConnectionVerdict], list[tuple[str, list[str]]]]:
        """One verdict per expected connection, plus every drawn arrow that is not one.

        Each drawn arrow is used once: first for an expected connection in the same
        direction, then for one in the opposite direction (REVERSED). What remains, and
        every arrow with an unattached end, is returned as extra.
        """
        expected = expectations.get("connections", [])
        actual = self._actual_connections(candidate, matched)
        attached = [(source, target, connector) for source, target, connector in actual if source and target]
        available: dict[tuple[str, str], list[SvgElement]] = {}
        for source, target, connector in attached:
            available.setdefault((source, target), []).append(connector)

        statuses: list[tuple[ConnectionStatus, SvgElement | None]] = [(ConnectionStatus.ABSENT, None)] * len(expected)
        for index, edge in enumerate(expected):  # same direction first
            pool = available.get((edge.get("source"), edge.get("target")), [])
            if pool:
                statuses[index] = (ConnectionStatus.PRESENT, pool.pop(0))
        for index, edge in enumerate(expected):  # then the opposite direction
            if statuses[index][0] != ConnectionStatus.ABSENT:
                continue
            pool = available.get((edge.get("target"), edge.get("source")), [])
            if pool:
                statuses[index] = (ConnectionStatus.REVERSED, pool.pop(0))

        labels = self._connection_labels(candidate, matched, [connector for _s, _t, connector in actual])
        verdicts = []
        for index, (edge, (status, connector)) in enumerate(zip(expected, statuses)):
            verdicts.append(
                ConnectionVerdict(
                    connection_id=f"C{index + 1}",
                    source=edge.get("source", ""),
                    target=edge.get("target", ""),
                    status=status,
                    judge=self.name,
                    expected_label=edge.get("label", "") or "",
                    visible_label=labels.get(id(connector), "") if connector is not None else "",
                    evidence=f"arrow {connector.id}" if connector is not None else "no arrow joins these components",
                )
            )
        extras = [
            (f"arrow {connector.id} joins {source} -> {target}", [connector.id])
            for (source, target), pool in sorted(available.items())
            for connector in pool
        ] + [
            (
                f"arrow {connector.id} is not attached to a described component at "
                + ("either end" if source is None and target is None else "one end"),
                [connector.id],
            )
            for source, target, connector in actual
            if not (source and target)
        ]
        return verdicts, extras

    def _connection_labels(
        self,
        candidate: SvgDocument,
        matched: dict[str, ComponentMatch],
        connectors: list[SvgElement],
    ) -> dict[int, str]:
        """Free text placed along an arrow, keyed by the arrow element.

        A text is a candidate when it is not a component's label, is not inside a
        component, and lies within the endpoint tolerance of an arrow. Each text goes to
        its nearest arrow, and each arrow keeps its nearest text.
        """
        component_texts = {id(match.text) for match in matched.values() if match.text is not None}
        component_shapes = [match.shape for match in matched.values()]
        threshold = self.config.endpoint_tolerance * math.hypot(candidate.width, candidate.height)
        best: dict[int, tuple[float, str]] = {}
        for text in (element for element in candidate.elements if element.is_text and element.text.strip()):
            if id(text) in component_texts or any(shape.bounds.contains(*text.bounds.center) for shape in component_shapes):
                continue
            ranked = sorted(
                (_distance_to_polyline(text.bounds.center, connector.points), index)
                for index, connector in enumerate(connectors)
                if len(connector.points) >= 2
            )
            if not ranked or ranked[0][0] > threshold:
                continue
            distance, index = ranked[0]
            key = id(connectors[index])
            if key not in best or distance < best[key][0]:
                best[key] = (distance, text.text)
        return {key: value[1] for key, value in best.items()}

    def _unexpected(
        self,
        candidate: SvgDocument,
        matched: dict[str, ComponentMatch],
        drawn: int,
    ) -> MetricScore:
        """Labeled boxes that do not correspond to any expected component.

        Boxes that hold a matched component's label, and containers or backgrounds that
        enclose a matched component, are not counted.
        """
        matched_shapes = {id(match.shape) for match in matched.values()}
        matched_texts = {id(match.text) for match in matched.values() if match.text is not None}
        extras: dict[int, LabeledShape] = {}
        for item in _labeled_shapes(candidate):
            if id(item.shape) in matched_shapes or id(item.text) in matched_texts:
                continue
            if any(_contains_bounds(item.shape.bounds, match.shape.bounds) for match in matched.values()):
                continue
            extras.setdefault(id(item.shape), item)
        issues = [
            Issue(
                f"Drawn component that the description does not call for: '{item.text.text}'",
                Severity.MEDIUM,
                [item.shape.id],
            )
            for item in extras.values()
        ]
        return self._score(
            drawn + len(extras),
            issues,
            "Denominator: drawn expected components plus unexpected labeled boxes.",
        )

    def _text_overflow(self, candidate: SvgDocument) -> MetricScore:
        """Text inside a box must fit the box; text outside boxes must not collide with
        other text (for example two arrow labels drawn on top of each other)."""
        shapes = [element for element in candidate.elements if element.is_shape]
        texts = [element for element in candidate.elements if element.is_text and element.text.strip()]
        contained: list[tuple[SvgElement, SvgElement]] = []
        free: list[SvgElement] = []
        for text in texts:
            containers = [shape for shape in shapes if shape.bounds.contains(*text.bounds.center)]
            if containers:
                contained.append((text, min(containers, key=lambda shape: _area(shape.bounds))))
            else:
                free.append(text)
        issues = [
            Issue("Text overflows its containing shape", Severity.HIGH, [text.id, shape.id])
            for text, shape in contained
            if not _contains_bounds(shape.bounds, text.bounds)
        ]
        for text in free:
            other = next(
                (
                    candidate_text
                    for candidate_text in texts
                    if candidate_text is not text and _overlap_fraction(text.bounds, candidate_text.bounds) > 0.2
                ),
                None,
            )
            if other is not None:
                issues.append(Issue("Text collides with other text", Severity.HIGH, [text.id, other.id]))
        return self._score(
            len(contained) + len(free),
            issues,
            "Text bounds use SVG positions and a conservative font-width estimate.",
        )

    def _minimum_font(self, candidate: SvgDocument) -> MetricScore:
        """Font sizes are compared at the size the SVG is rendered (root size vs viewBox),
        with sizes inherited from parent groups."""
        texts = [element for element in candidate.elements if element.is_text]
        minimum = self.config.minimum_font_size
        scale = candidate.render_scale
        issues = [
            Issue(
                f"Font size {text.font_size * scale:g}px is below {minimum:g}px when rendered",
                Severity.HIGH,
                [text.id],
            )
            for text in texts
            if text.font_size * scale < minimum
        ]
        return self._score(
            len(texts),
            issues,
            f"Rendered at {scale:g} pixel(s) per SVG unit.",
        )
