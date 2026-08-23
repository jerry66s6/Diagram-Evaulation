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

    def extract(self, reference_image: str | Path, criteria: list[Criterion]) -> dict[str, list[str]]:
        return {dimension.value: [f"expected {dimension.value}"] for dimension in Dimension}


class FakeJudge:
    def __init__(self, name: str, detected: int) -> None:
        self.name = name
        self.detected = detected

    def evaluate(self, reference_image, candidate_image, criteria, inventory):
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
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    reference.write_bytes(b"reference")
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
    report = pipeline.run(reference, candidate)
    assert set(report.dimensions) == set(Dimension)
    assert report.correctness == pytest.approx(0.8)
    assert sum(report.weights.values()) == pytest.approx(1.0)
    assert report.metadata["deterministic_judge"] is False
