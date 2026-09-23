from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from statistics import median
from typing import Any, Iterable


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


class Verdict(StrEnum):
    """Outcome of checking one expected component against the candidate."""

    PRESENT = "present"  # the role is drawn and carries the expected label
    MISLABELED = "mislabeled"  # the role is drawn, but its label is wrong, abbreviated, missing, or a placeholder
    ABSENT = "absent"  # nothing in the candidate plays this role
    UNCERTAIN = "uncertain"  # the judge cannot decide from the candidate


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
) -> dict[Dimension, DimensionScore]:
    """Aggregate compatible judge signals criterion-first, using the median.

    A deterministic result is only combined with VLM results when it has the same
    criterion id. Missing/non-measurable results do not become perfect scores.
    """
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
                score=sum(criterion_scores.values()) / len(criterion_scores),
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
    - Presence: every component the judge could decide (UNCERTAIN is excluded);
      a component fails only when it is ABSENT.
    - Details: only components that are drawn (PRESENT or MISLABELED);
      a component fails only when it is MISLABELED.
    A missing component therefore costs Presence only, and a wrong label costs
    Details only, so no failure is counted twice.
    """
    decided = [item for item in verdicts if item.verdict != Verdict.UNCERTAIN]
    drawn = [item for item in decided if item.verdict in {Verdict.PRESENT, Verdict.MISLABELED}]
    uncertain = [item.component_id for item in verdicts if item.verdict == Verdict.UNCERTAIN]
    results: list[MetricScore] = []

    if presence is not None:
        issues = [
            Issue(
                f"Expected component is not drawn: {item.component_id}",
                presence.severity,
                [item.component_id],
            )
            for item in decided
            if item.verdict == Verdict.ABSENT
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
                notes="Scored only over drawn components; missing components are scored under Presence.",
                # Keep the verdict list on Presence only, unless Presence is not in the rubric.
                component_verdicts=[] if presence is not None else list(verdicts),
            )
        )
    return results


def _uncertain_note(uncertain: list[str]) -> str:
    if not uncertain:
        return ""
    return "Excluded as uncertain: " + ", ".join(uncertain)


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
