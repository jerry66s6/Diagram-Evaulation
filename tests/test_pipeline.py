from pathlib import Path

import pytest

from diagram_correctness.models import Criterion, Dimension, MetricScore, Severity
from diagram_correctness.pipeline import CorrectnessPipeline, PipelineConfig
from diagram_correctness.vlm import ExpectationExtractor


class FakeBackend:
    name = "expectation-model"


class FakeExtractor(ExpectationExtractor):
    def __init__(self) -> None:
        self.backend = FakeBackend()

    def extract(self, description: str, criteria: list[Criterion]):
        return {
            "inventory": {
                dimension.value: [f"expected {dimension.value}"]
                for dimension in Dimension
            },
            "components": [],
            "connections": [],
            "forbidden_placeholders": [],
        }


class FakeJudge:
    def __init__(self, name: str, detected: int) -> None:
        self.name = name
        self.detected = detected

    def evaluate(self, description, candidate_image, criteria, inventory):
        return [
            MetricScore(
                criterion.id,
                criterion.dimension,
                self.name,
                expected_count=10,
                detected_issues=self.detected,
            )
            for criterion in criteria
        ]


def test_pipeline_scores_all_five_dimensions(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.png"
    candidate.write_bytes(b"candidate")
    criteria = [
        Criterion(f"{dimension.value}.metric", dimension, "", Severity.MEDIUM)
        for dimension in Dimension
    ]
    pipeline = CorrectnessPipeline(
        PipelineConfig(
            criteria=criteria,
            judges=[FakeJudge("a", 1), FakeJudge("b", 2), FakeJudge("c", 3)],  # type: ignore[list-item]
            expectation_extractor=FakeExtractor(),
            fallback_frequencies={dimension.value: 1 for dimension in Dimension},
        )
    )
    report = pipeline.run("A five-part diagram", candidate)
    assert set(report.dimensions) == set(Dimension)
    assert report.correctness == pytest.approx(0.8)
    assert sum(report.weights.values()) == pytest.approx(1.0)
    assert report.metadata["deterministic_judge"] is False
    assert report.metadata["expectations_from"] == "description"


def test_pipeline_accepts_svg_as_the_only_candidate_artifact(tmp_path: Path) -> None:
    candidate_svg = tmp_path / "candidate.svg"
    candidate_svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
        '<rect x="10" y="10" width="80" height="40"/>'
        '<text x="20" y="35" font-size="14">Input</text></svg>',
        encoding="utf-8",
    )
    criterion = Criterion(
        "presence.elements", Dimension.PRESENCE, "required elements", Severity.HIGH
    )
    pipeline = CorrectnessPipeline(
        PipelineConfig(
            criteria=[criterion],
            judges=[FakeJudge("gpt", 0)],  # type: ignore[list-item]
            expectation_extractor=FakeExtractor(),
            fallback_frequencies={dimension.value: 1 for dimension in Dimension},
        )
    )

    report = pipeline.run("Show an input box.", candidate_svg=candidate_svg)

    assert report.correctness == pytest.approx(1.0)
