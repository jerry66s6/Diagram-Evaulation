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
