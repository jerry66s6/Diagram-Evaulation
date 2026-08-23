from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .deterministic import DeterministicJudge, GeometryConfig
from .models import Criterion, Dimension, EvaluationReport, MetricScore, aggregate_panel
from .rubric import renormalize_weights, severity_frequency_weights
from .svg import parse_svg
from .vfig import VFigRunner
from .vlm import ExpectationExtractor, VLMJudge


@dataclass
class PipelineConfig:
    criteria: list[Criterion]
    judges: list[VLMJudge]
    expectation_extractor: ExpectationExtractor
    vfig_runner: VFigRunner | None = None
    deterministic_judge: DeterministicJudge = field(default_factory=DeterministicJudge)
    issue_history: list[dict[str, Any]] = field(default_factory=list)
    severity_multipliers: dict[str, float] | None = None
    fallback_frequencies: dict[str, float] | None = None


class CorrectnessPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(
        self,
        reference_image: str | Path,
        candidate_image: str | Path,
        reference_svg: str | Path | None = None,
        candidate_svg: str | Path | None = None,
    ) -> EvaluationReport:
        reference_image = Path(reference_image)
        candidate_image = Path(candidate_image)
        _require_file(reference_image)
        _require_file(candidate_image)

        inventory = self.config.expectation_extractor.extract(reference_image, self.config.criteria)
        metrics: list[MetricScore] = []

        # The three VLM judges are independent panel members.
        with ThreadPoolExecutor(max_workers=max(1, len(self.config.judges))) as executor:
            futures = {
                executor.submit(
                    judge.evaluate,
                    reference_image,
                    candidate_image,
                    self.config.criteria,
                    inventory,
                ): judge.name
                for judge in self.config.judges
            }
            for future in as_completed(futures):
                metrics.extend(future.result())

        with tempfile.TemporaryDirectory(prefix="diagram-correctness-") as temporary:
            temp = Path(temporary)
            resolved_reference_svg = self._resolve_svg(
                reference_image, reference_svg, temp / "reference.svg"
            )
            resolved_candidate_svg = self._resolve_svg(
                candidate_image, candidate_svg, temp / "candidate.svg"
            )
            if resolved_reference_svg and resolved_candidate_svg:
                metrics.extend(
                    self.config.deterministic_judge.evaluate(
                        parse_svg(resolved_reference_svg),
                        parse_svg(resolved_candidate_svg),
                        self.config.criteria,
                    )
                )

        dimensions = aggregate_panel(metrics, self.config.criteria)
        raw_weights = severity_frequency_weights(
            self.config.issue_history,
            self.config.severity_multipliers,
            self.config.fallback_frequencies,
        )
        weights = renormalize_weights(raw_weights, set(dimensions))
        correctness = sum(weights[dimension] * result.score for dimension, result in dimensions.items())
        return EvaluationReport(
            correctness=correctness,
            dimensions=dimensions,
            weights=weights,
            metrics=metrics,
            inventory=inventory,
            metadata={
                "vlm_judges": [judge.name for judge in self.config.judges],
                "expectation_model": self.config.expectation_extractor.backend.name,
                "deterministic_judge": bool(reference_svg and candidate_svg) or bool(self.config.vfig_runner),
            },
        )

    def _resolve_svg(
        self,
        image: Path,
        supplied_svg: str | Path | None,
        output: Path,
    ) -> Path | None:
        if supplied_svg:
            path = Path(supplied_svg)
            _require_file(path)
            return path
        if self.config.vfig_runner:
            return self.config.vfig_runner.convert(image, output)
        return None


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)

