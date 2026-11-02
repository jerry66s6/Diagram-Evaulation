"""Correctness evaluation for rendered diagrams: direct per-dimension scoring (default) and the earlier
sketch-and-formula pipeline."""

from .direct import CorrectnessScore, judge_correctness, overall
from .models import Dimension, EvaluationReport, MetricScore
from .pipeline import CorrectnessPipeline, PipelineConfig

__all__ = [
    "CorrectnessScore",
    "judge_correctness",
    "overall",
    "CorrectnessPipeline",
    "Dimension",
    "EvaluationReport",
    "MetricScore",
    "PipelineConfig",
]

