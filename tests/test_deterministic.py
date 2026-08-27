from pathlib import Path

from diagram_correctness.deterministic import DeterministicJudge, GeometryConfig
from diagram_correctness.models import Criterion, Dimension, Severity
from diagram_correctness.svg import parse_svg


FIXTURES = Path(__file__).parent / "fixtures"


def _criteria() -> list[Criterion]:
    return [
        Criterion("presence.elements", Dimension.PRESENCE, "", Severity.HIGH, "description_presence"),
        Criterion("connectivity.connections", Dimension.CONNECTIVITY, "", Severity.CRITICAL, "description_connections"),
        Criterion("connectivity.endpoints", Dimension.CONNECTIVITY, "", Severity.CRITICAL, "connector_attachment"),
        Criterion("details.labels", Dimension.DETAILS, "", Severity.HIGH, "meaningful_labels"),
        Criterion("legibility.overflow", Dimension.LEGIBILITY, "", Severity.HIGH, "text_overflow"),
        Criterion("legibility.minimum_font", Dimension.LEGIBILITY, "", Severity.HIGH, "minimum_font_size"),
    ]


def test_deterministic_judge_counts_missing_text_and_small_font() -> None:
    candidate = parse_svg(FIXTURES / "candidate.svg")
    expectations = {
        "components": [
            {"id": "start", "label": "Start", "kind": "node"},
            {"id": "finish", "label": "Finish", "kind": "node"},
        ],
        "connections": [{"source": "start", "target": "finish"}],
        "forbidden_placeholders": ["TBD", "???"],
        "inventory": {dimension.value: [] for dimension in Dimension},
    }
    results = {
        result.criterion_id: result
        for result in DeterministicJudge(GeometryConfig(minimum_font_size=12)).evaluate(
            candidate, _criteria(), expectations
        )
    }
    assert results["presence.elements"].detected_issues >= 1
    assert results["connectivity.connections"].detected_issues == 1
    assert results["connectivity.endpoints"].detected_issues == 1
    assert results["details.labels"].detected_issues == 2
    assert results["legibility.minimum_font"].detected_issues == 1
    assert results["legibility.overflow"].detected_issues == 1
