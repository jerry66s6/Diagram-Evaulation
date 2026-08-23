from pathlib import Path

from diagram_correctness.deterministic import DeterministicJudge, GeometryConfig
from diagram_correctness.models import Criterion, Dimension, Severity
from diagram_correctness.svg import parse_svg


FIXTURES = Path(__file__).parent / "fixtures"


def _criteria() -> list[Criterion]:
    return [
        Criterion("presence.elements", Dimension.PRESENCE, "", Severity.HIGH, "element_presence"),
        Criterion("details.text", Dimension.DETAILS, "", Severity.HIGH, "text_accuracy"),
        Criterion("legibility.overflow", Dimension.LEGIBILITY, "", Severity.HIGH, "text_overflow"),
        Criterion("legibility.minimum_font", Dimension.LEGIBILITY, "", Severity.HIGH, "minimum_font_size"),
    ]


def test_deterministic_judge_counts_missing_text_and_small_font() -> None:
    reference = parse_svg(FIXTURES / "reference.svg")
    candidate = parse_svg(FIXTURES / "candidate.svg")
    results = {
        result.criterion_id: result
        for result in DeterministicJudge(GeometryConfig(minimum_font_size=12)).evaluate(
            reference, candidate, _criteria()
        )
    }
    assert results["presence.elements"].detected_issues >= 1
    assert results["details.text"].detected_issues == 2
    assert results["legibility.minimum_font"].detected_issues == 1
    assert results["legibility.overflow"].detected_issues == 1
