from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from statistics import median
from typing import Any, Iterable

from .labels import labels_match


class Dimension(StrEnum):
    PRESENCE = "presence"
    LAYOUT = "layout"
    CONNECTIVITY = "connectivity"
    DETAILS = "details"
    LEGIBILITY = "legibility"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Criterion:
    id: str
    dimension: Dimension
    description: str
    severity: Severity
    deterministic_metric: str | None = None


@dataclass
class Issue:
    description: str
    severity: Severity
    element_ids: list[str] = field(default_factory=list)


# Criteria that are scored from per-component verdicts instead of judge-chosen counts.
PRESENCE_CRITERION_ID = "presence.elements"
DETAILS_CRITERION_ID = "details.labels"
ATOMIC_CRITERIA = frozenset({PRESENCE_CRITERION_ID, DETAILS_CRITERION_ID})

# Rubric metric keys. Every judge computes the denominator of these from the frozen
# inventory and its own per-item verdicts, never from a free-form count.
METRIC_PRESENCE = "description_presence"
METRIC_UNEXPECTED = "unexpected_components"
METRIC_LABELS = "meaningful_labels"
METRIC_CONNECTIONS = "description_connections"
METRIC_ATTACHMENT = "connector_attachment"
METRIC_CONNECTION_LABELS = "connection_labels"
METRIC_MINIMUM_FONT = "minimum_font_size"


class Verdict(StrEnum):
    """Outcome of checking one expected component against the candidate."""

    PRESENT = "present"  # the role is drawn and carries the expected label
    MISLABELED = "mislabeled"  # the role is drawn, but its label is wrong, abbreviated, missing, or a placeholder
    ABSENT = "absent"  # nothing in the candidate plays this role
    UNCERTAIN = "uncertain"  # the judge cannot tell whether the role is drawn; scored as not visibly drawn


class ConnectionStatus(StrEnum):
    """Outcome of checking one expected directed connection against the candidate."""

    PRESENT = "present"  # an arrow joins the two components in the expected direction
    REVERSED = "reversed"  # an arrow joins them, but points the wrong way
    ABSENT = "absent"  # no arrow joins them
    UNCERTAIN = "uncertain"  # the judge cannot tell; scored as not drawn


@dataclass(frozen=True)
class PresencePolicy:
    """Scoring rules shared by every judge so that verdicts mean the same thing.

    accept_abbreviations: a standard abbreviation of the expected label (e.g. "FFN")
        counts as PRESENT instead of MISLABELED.
    placeholder_counts_as_present: an element in the right position whose label is a
        placeholder (e.g. "Layer", "TBD") or missing counts as MISLABELED; when False it
        counts as ABSENT.
    """

    accept_abbreviations: bool = False
    placeholder_counts_as_present: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PresencePolicy":
        data = data or {}
        return cls(
            accept_abbreviations=bool(data.get("accept_abbreviations", False)),
            placeholder_counts_as_present=bool(data.get("placeholder_counts_as_present", True)),
        )


@dataclass
class ComponentVerdict:
    """One judge's verdict for one expected component ID from the frozen inventory."""

    component_id: str
    verdict: Verdict
    judge: str
    visible_text: str = ""
    evidence: str = ""
    # Normalized (x0, y0, x1, y1) in [0, 1] relative to the candidate canvas, if located.
    bounds: tuple[float, float, float, float] | None = None
    notes: str = ""
    # False when the element is drawn but its text is too small or blurry to read. Such a
    # label cannot be checked for Details; the defect is scored once, under Legibility.
    label_legible: bool = True


@dataclass
class ConnectionVerdict:
    """One judge's verdict for one expected connection from the frozen inventory."""

    connection_id: str
    source: str
    target: str
    status: ConnectionStatus
    judge: str
    expected_label: str = ""
    visible_label: str = ""
    label_legible: bool = True
    evidence: str = ""
    notes: str = ""


@dataclass
class MetricScore:
    criterion_id: str
    dimension: Dimension
    judge: str
    expected_count: int
    detected_issues: int
    issues: list[Issue] = field(default_factory=list)
    notes: str = ""
    measurable: bool = True
    # Per-component verdicts behind an atomic criterion (empty for count-based criteria).
    component_verdicts: list[ComponentVerdict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.expected_count = max(0, int(self.expected_count))
        self.detected_issues = max(0, int(self.detected_issues))
        if self.expected_count == 0:
            self.measurable = False

    @property
    def score(self) -> float | None:
        """Return (opportunities - issues) / opportunities, clamped to [0, 1]."""
        if not self.measurable or self.expected_count == 0:
            return None
        return max(0.0, min(1.0, (self.expected_count - self.detected_issues) / self.expected_count))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["dimension"] = self.dimension.value
        result["issues"] = [
            {**asdict(issue), "severity": issue.severity.value} for issue in self.issues
        ]
        result["score"] = self.score
        return result


@dataclass
class DimensionScore:
    dimension: Dimension
    score: float
    criteria: dict[str, float]
    participating_judges: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "criteria": self.criteria,
            "participating_judges": self.participating_judges,
        }


@dataclass
class EvaluationReport:
    correctness: float
    dimensions: dict[Dimension, DimensionScore]
    weights: dict[Dimension, float]
    metrics: list[MetricScore]
    inventory: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "correctness": self.correctness,
            "formula": " + ".join(
                f"{self.weights.get(d, 0.0):.6f}*{d.value}" for d in Dimension
            ),
            "weights": {d.value: value for d, value in self.weights.items()},
            "dimensions": {
                d.value: score.to_dict() for d, score in self.dimensions.items()
            },
            "metrics": [metric.to_dict() for metric in self.metrics],
            "inventory": self.inventory,
            "metadata": self.metadata,
        }


def aggregate_panel(
    metrics: Iterable[MetricScore],
    criteria: Iterable[Criterion],
    reducer: str = "mean",
) -> dict[Dimension, DimensionScore]:
    """Aggregate judge signals criterion-first, then criteria into dimensions.

    Judges are combined per criterion with the median; see ``criterion_disagreements``
    for the cases this hides. Criteria are combined per dimension with ``reducer``:
    "mean" lets a passing check offset a failing one, "minimum" does not. Missing or
    non-measurable results do not become perfect scores.
    """
    if reducer not in {"mean", "minimum"}:
        raise ValueError(f"Unknown dimension reducer: {reducer}")
    metric_list = list(metrics)
    criterion_list = list(criteria)
    dimensions: dict[Dimension, DimensionScore] = {}

    for dimension in Dimension:
        criterion_scores: dict[str, float] = {}
        judges: set[str] = set()
        for criterion in (c for c in criterion_list if c.dimension == dimension):
            aligned = [
                metric
                for metric in metric_list
                if metric.criterion_id == criterion.id and metric.score is not None
            ]
            if not aligned:
                continue
            criterion_scores[criterion.id] = float(median(metric.score for metric in aligned if metric.score is not None))
            judges.update(metric.judge for metric in aligned)
        if criterion_scores:
            dimensions[dimension] = DimensionScore(
                dimension=dimension,
                score=(min(criterion_scores.values()) if reducer == "minimum"
                       else sum(criterion_scores.values()) / len(criterion_scores)),
                criteria=criterion_scores,
                participating_judges=sorted(judges),
            )
    return dimensions


def component_metrics(
    verdicts: list[ComponentVerdict],
    judge: str,
    presence: Criterion | None,
    details: Criterion | None,
) -> list[MetricScore]:
    """Score Presence and Details from one judge's per-component verdicts.

    Both denominators come from the frozen inventory, never from the judge:
    - Presence: every inventory component; it fails when ABSENT or UNCERTAIN. A role the
      judge cannot make out is not visibly drawn, and dropping it from the denominator
      would make a less certain judge give a higher score.
    - Details: components that are drawn (PRESENT or MISLABELED) with a readable label;
      a component fails only when it is MISLABELED. Unreadable labels are scored under
      Legibility instead.
    A missing component therefore costs Presence only, a wrong label costs Details only,
    and tiny text costs Legibility only, so no failure is counted twice.
    """
    decided = list(verdicts)
    drawn = [
        item
        for item in decided
        if item.verdict in {Verdict.PRESENT, Verdict.MISLABELED} and item.label_legible
    ]
    uncertain = [item.component_id for item in verdicts if item.verdict == Verdict.UNCERTAIN]
    unreadable = [
        item.component_id
        for item in verdicts
        if item.verdict in {Verdict.PRESENT, Verdict.MISLABELED} and not item.label_legible
    ]
    results: list[MetricScore] = []

    if presence is not None:
        issues = [
            Issue(
                f"Expected component is not visibly drawn ({item.verdict.value}): {item.component_id}",
                presence.severity,
                [item.component_id],
            )
            for item in decided
            if item.verdict in {Verdict.ABSENT, Verdict.UNCERTAIN}
        ]
        results.append(
            MetricScore(
                criterion_id=presence.id,
                dimension=presence.dimension,
                judge=judge,
                expected_count=len(decided),
                detected_issues=len(issues),
                issues=issues,
                notes=_uncertain_note(uncertain),
                component_verdicts=list(verdicts),
            )
        )

    if details is not None:
        issues = [
            Issue(
                f"Drawn component has an abbreviated, incorrect, missing, or placeholder label: "
                f"{item.component_id} (visible text: '{item.visible_text}')",
                details.severity,
                [item.component_id],
            )
            for item in drawn
            if item.verdict == Verdict.MISLABELED
        ]
        results.append(
            MetricScore(
                criterion_id=details.id,
                dimension=details.dimension,
                judge=judge,
                expected_count=len(drawn),
                detected_issues=len(issues),
                issues=issues,
                notes=(
                    "Scored only over drawn components with readable labels; missing components are scored "
                    "under Presence and unreadable labels under Legibility."
                    + (f" Unreadable, not scored here: {', '.join(unreadable)}." if unreadable else "")
                ),
                # Keep the verdict list on Presence only, unless Presence is not in the rubric.
                component_verdicts=[] if presence is not None else list(verdicts),
            )
        )
    return results


def _uncertain_note(uncertain: list[str]) -> str:
    if not uncertain:
        return ""
    return "Counted as not visibly drawn because the judge could not tell: " + ", ".join(uncertain)


def feedback_edges(components: list[dict[str, Any]], connections: list[dict[str, Any]]) -> set[int]:
    """Indices of expected connections that close a cycle (loops back to an earlier step).

    A depth-first search starting from components with no incoming connection marks the
    edges that point to a component still on the search path. Such loops necessarily run
    against the reading direction, so they are not order violations.
    """
    adjacency: dict[str, list[tuple[int, str]]] = {}
    incoming: dict[str, int] = {}
    nodes = [component.get("id", "") for component in components]
    for index, edge in enumerate(connections):
        source, target = edge.get("source", ""), edge.get("target", "")
        adjacency.setdefault(source, []).append((index, target))
        incoming[target] = incoming.get(target, 0) + 1
        nodes += [node for node in (source, target) if node not in nodes]
    roots = [node for node in nodes if incoming.get(node, 0) == 0] + nodes
    state: dict[str, int] = {}  # 1 = on the current path, 2 = finished
    feedback: set[int] = set()
    for root in roots:
        if root in state:
            continue
        state[root] = 1
        stack = [(root, iter(adjacency.get(root, [])))]
        while stack:
            node, edges = stack[-1]
            for index, nxt in edges:
                if state.get(nxt) == 1:
                    feedback.add(index)
                elif nxt not in state:
                    state[nxt] = 1
                    stack.append((nxt, iter(adjacency.get(nxt, []))))
                    break
            else:
                state[node] = 2
                stack.pop()
    return feedback


def drawn_component_ids(verdicts: Iterable[ComponentVerdict]) -> set[str]:
    return {item.component_id for item in verdicts if item.verdict in {Verdict.PRESENT, Verdict.MISLABELED}}


def connection_metrics(
    connections: list[ConnectionVerdict],
    drawn_components: set[str],
    extra_arrows: list[str] | list[tuple[str, list[str]]],
    judge: str,
    recall: Criterion | None,
    precision: Criterion | None,
    labels: Criterion | None,
    accept_abbreviations: bool = False,
) -> list[MetricScore]:
    """Score Connectivity and connection labels from one judge's per-connection verdicts.

    - Recall (every expected connection is drawn): only connections whose two components
      are both drawn are applicable. A connection to a missing component is already paid
      for under Presence, so it is not counted a second time here. It fails when no arrow
      joins the pair (ABSENT or UNCERTAIN).
    - Precision (every drawn arrow is right): the denominator is every drawn arrow, that
      is correct, reversed, and extra arrows. Reversed and extra arrows fail, so a wrong
      or invented arrow costs something even when every expected arrow is also drawn.
    - Connection labels: applicable connections that are drawn and carry an expected
      label (for example YES/NO). The visible label is compared in code, not by the judge.
      Unreadable labels are left to Legibility.
    """
    extras = [(item, []) if isinstance(item, str) else (item[0], list(item[1])) for item in extra_arrows]
    applicable = [
        item for item in connections if item.source in drawn_components and item.target in drawn_components
    ]
    skipped = [item.connection_id for item in connections if item not in applicable]
    drawn = [item for item in applicable if item.status in {ConnectionStatus.PRESENT, ConnectionStatus.REVERSED}]
    results: list[MetricScore] = []

    if recall is not None:
        issues = [
            Issue(
                f"Expected connection is not drawn ({item.status.value}): {item.source} -> {item.target}",
                recall.severity,
                [item.source, item.target],
            )
            for item in applicable
            if item.status in {ConnectionStatus.ABSENT, ConnectionStatus.UNCERTAIN}
        ]
        results.append(
            MetricScore(
                criterion_id=recall.id,
                dimension=recall.dimension,
                judge=judge,
                expected_count=len(applicable),
                detected_issues=len(issues),
                issues=issues,
                notes=(
                    "Connections to components that are not drawn are scored under Presence only"
                    + (f": {', '.join(skipped)}." if skipped else ".")
                ),
            )
        )

    if precision is not None:
        reversed_items = [item for item in applicable if item.status == ConnectionStatus.REVERSED]
        issues = [
            Issue(f"Arrow points the wrong way: {item.target} -> {item.source}", precision.severity, [item.source, item.target])
            for item in reversed_items
        ] + [Issue(f"Arrow that the description does not call for: {text}", precision.severity, ids) for text, ids in extras]
        results.append(
            MetricScore(
                criterion_id=precision.id,
                dimension=precision.dimension,
                judge=judge,
                expected_count=len(drawn) + len(extras),
                detected_issues=len(issues),
                issues=issues,
                notes="Every drawn arrow is checked: reversed, extra, dangling, and duplicate arrows fail.",
            )
        )

    if labels is not None:
        labeled = [item for item in drawn if item.expected_label.strip() and item.label_legible]
        issues = [
            Issue(
                f"Connection {item.source} -> {item.target} should be labeled '{item.expected_label}' "
                f"but shows '{item.visible_label}'",
                labels.severity,
                [item.source, item.target],
            )
            for item in labeled
            if not labels_match(item.visible_label, item.expected_label, accept_abbreviations)
        ]
        results.append(
            MetricScore(
                criterion_id=labels.id,
                dimension=labels.dimension,
                judge=judge,
                expected_count=len(labeled),
                detected_issues=len(issues),
                issues=issues,
                notes="Only drawn connections whose description gives a label or condition, with readable text.",
            )
        )
    return results


def legibility_from_verdicts(
    components: list[ComponentVerdict],
    connections: list[ConnectionVerdict],
    judge: str,
    criterion: Criterion,
) -> MetricScore:
    """Readable-text rate over drawn labeled items, from the per-item legibility flags."""
    texts = [
        (item.component_id, item.label_legible)
        for item in components
        if item.verdict in {Verdict.PRESENT, Verdict.MISLABELED} and item.visible_text.strip()
    ]
    drawn_components = drawn_component_ids(components)
    texts += [
        (item.connection_id, item.label_legible)
        for item in connections
        if item.status in {ConnectionStatus.PRESENT, ConnectionStatus.REVERSED}
        and item.source in drawn_components
        and item.target in drawn_components
        and (item.visible_label.strip() or item.expected_label.strip())
    ]
    issues = [
        Issue(f"Text is too small or blurry to read: {item_id}", criterion.severity, [item_id])
        for item_id, legible in texts
        if not legible
    ]
    return MetricScore(
        criterion_id=criterion.id,
        dimension=criterion.dimension,
        judge=judge,
        expected_count=len(texts),
        detected_issues=len(issues),
        issues=issues,
        notes="One check per drawn component label and connection label.",
    )


def criterion_disagreements(metrics: Iterable[MetricScore], threshold: float = 0.25) -> list[dict[str, Any]]:
    """Criteria where two judges' scores differ by at least ``threshold``.

    The panel combines judges with a median, which for two judges is their mean and
    would otherwise hide a disagreement such as 1.0 versus 0.5.
    """
    by_criterion: dict[str, dict[str, float]] = {}
    for metric in metrics:
        if metric.score is not None:
            by_criterion.setdefault(metric.criterion_id, {})[metric.judge] = metric.score
    rows = []
    for criterion_id, scores in sorted(by_criterion.items()):
        if len(scores) < 2:
            continue
        spread = max(scores.values()) - min(scores.values())
        if spread >= threshold:
            rows.append({"criterion_id": criterion_id, "scores": scores, "spread": round(spread, 4)})
    return rows


def component_agreement(metrics: Iterable[MetricScore]) -> list[dict[str, Any]]:
    """Line up every judge's verdict per component ID and flag disagreements.

    Disagreements are reported for review rather than changing the score, because
    dropping disputed items from the denominator would reward ambiguous drawings.
    """
    table: dict[str, dict[str, str]] = {}
    for metric in metrics:
        for item in metric.component_verdicts:
            table.setdefault(item.component_id, {})[metric.judge] = item.verdict.value
    return [
        {
            "component_id": component_id,
            "verdicts": verdicts,
            "agree": len(set(verdicts.values())) <= 1,
        }
        for component_id, verdicts in table.items()
    ]
