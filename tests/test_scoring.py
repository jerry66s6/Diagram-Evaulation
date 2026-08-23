from diagram_correctness.models import (
    Criterion,
    Dimension,
    MetricScore,
    Severity,
    aggregate_panel,
)
from diagram_correctness.rubric import severity_frequency_weights


def test_metric_formula_and_clamping() -> None:
    score = MetricScore("p", Dimension.PRESENCE, "judge", 10, 3)
    assert score.score == 0.7
    assert MetricScore("p", Dimension.PRESENCE, "judge", 2, 8).score == 0.0
    assert MetricScore("p", Dimension.PRESENCE, "judge", 0, 0).score is None


def test_panel_is_aligned_by_criterion_and_uses_median() -> None:
    criterion = Criterion("p", Dimension.PRESENCE, "required", Severity.HIGH, "element_presence")
    metrics = [
        MetricScore("p", Dimension.PRESENCE, "gpt-5.5", 10, 1),
        MetricScore("p", Dimension.PRESENCE, "gpt-5.6", 10, 2),
        MetricScore("p", Dimension.PRESENCE, "claude", 10, 9),
        MetricScore("p", Dimension.PRESENCE, "deterministic", 10, 2),
    ]
    result = aggregate_panel(metrics, [criterion])[Dimension.PRESENCE]
    assert result.score == 0.8


def test_frequency_and_severity_determine_weights() -> None:
    weights = severity_frequency_weights(
        [
            {"dimension": "presence", "severity": "medium", "count": 2},
            {"dimension": "connectivity", "severity": "critical", "count": 1},
        ],
        {"low": 1, "medium": 2, "high": 4, "critical": 8},
    )
    assert weights[Dimension.PRESENCE] == 1 / 3
    assert weights[Dimension.CONNECTIVITY] == 2 / 3

