from __future__ import annotations

import os
import sys
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
    expectation_extractor: ExpectationExtractor | None
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
        description: str,
        candidate_image: str | Path | None = None,
        candidate_svg: str | Path | None = None,
        inventory: dict[str, Any] | None = None,
        run_deterministic: bool = True,
    ) -> EvaluationReport:
        if not description.strip():
            raise ValueError("Diagram description cannot be empty")
        if candidate_image is None and candidate_svg is None:
            raise ValueError("A candidate SVG or rendered candidate image is required")
        if inventory is None:
            if self.config.expectation_extractor is None:
                raise ValueError("An expectation extractor or a frozen inventory is required")
            inventory = self.config.expectation_extractor.extract(description, self.config.criteria)
        metrics: list[MetricScore] = []

        with tempfile.TemporaryDirectory(prefix="diagram-correctness-") as temporary:
            temp = Path(temporary)
            resolved_candidate_image = self._resolve_image(
                candidate_image, candidate_svg, temp / "candidate.png"
            )

            # VLM judges consume the same description-derived frozen inventory.
            with ThreadPoolExecutor(max_workers=max(1, len(self.config.judges))) as executor:
                futures = {
                    executor.submit(
                        judge.evaluate,
                        description,
                        resolved_candidate_image,
                        self.config.criteria,
                        inventory,
                    ): judge.name
                    for judge in self.config.judges
                }
                for future in as_completed(futures):
                    metrics.extend(future.result())

            resolved_candidate_svg = (
                self._resolve_svg(resolved_candidate_image, candidate_svg, temp / "candidate.svg")
                if run_deterministic
                else None
            )
            if resolved_candidate_svg:
                metrics.extend(
                    self.config.deterministic_judge.evaluate(
                        parse_svg(resolved_candidate_svg),
                        self.config.criteria,
                        inventory,
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
                "expectation_model": (
                    self.config.expectation_extractor.backend.name
                    if self.config.expectation_extractor
                    else None
                ),
                "expectations_from": "description",
                "deterministic_judge": bool(resolved_candidate_svg),
            },
        )

    def _resolve_image(
        self,
        supplied_image: str | Path | None,
        supplied_svg: str | Path | None,
        output: Path,
    ) -> Path:
        if supplied_image:
            path = Path(supplied_image)
            _require_file(path)
            return path
        if not supplied_svg:
            raise ValueError("A candidate SVG or rendered candidate image is required")
        svg_path = Path(supplied_svg)
        _require_file(svg_path)
        if sys.platform == "darwin" and Path("/opt/homebrew/lib/libcairo.2.dylib").is_file():
            fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
            if "/opt/homebrew/lib" not in fallback.split(":"):
                os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
                    part for part in ("/opt/homebrew/lib", fallback) if part
                )
        from cairosvg import svg2png

        svg2png(url=str(svg_path), write_to=str(output))
        return output

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
