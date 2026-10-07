from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .deterministic import DeterministicJudge, GeometryConfig
from .models import PresencePolicy
from .pipeline import CorrectnessPipeline, PipelineConfig
from .rubric import load_config, load_rubric
from .vlm import ExpectationExtractor, OpenAIBackend, VLMJudge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate diagram alignment with a description")
    parser.add_argument("--description", required=True, help="Path to the intended diagram description")
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument(
        "--candidate-svg",
        help="Generated candidate SVG (rendered internally for GPT and parsed for geometry)",
    )
    candidate.add_argument(
        "--candidate",
        help="Rendered candidate image (PNG or JPEG); scored by the GPT judge only, because the deterministic judge reads SVG",
    )
    parser.add_argument("--rubric", default="config/rubric.json")
    parser.add_argument("--config", default="config/evaluator.json")
    parser.add_argument("--output", default="correctness-report.json")
    parser.add_argument(
        "--skip-deterministic",
        action="store_true",
        help="Score an SVG candidate with the GPT judge only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_config(args.config)
        criteria = load_rubric(args.rubric)
        models = settings["models"]
        policy = PresencePolicy.from_dict(settings.get("presence_policy"))
        judges = [VLMJudge(OpenAIBackend(models["judge"]), policy, samples=int(settings.get("judge_samples", 1)))]
        geometry = settings.get("geometry", {})
        deterministic = DeterministicJudge(GeometryConfig(**geometry), policy)
        candidate_svg = args.candidate_svg
        if not candidate_svg and not args.skip_deterministic:
            print("note: no SVG supplied, so only the GPT judge scores this image.", file=sys.stderr)

        issue_history = []
        history_path = settings.get("weight_history")
        if history_path:
            issue_history = json.loads(Path(history_path).read_text(encoding="utf-8"))["issues"]

        pipeline = CorrectnessPipeline(
            PipelineConfig(
                criteria=criteria,
                judges=judges,
                expectation_extractor=ExpectationExtractor(
                    OpenAIBackend(models.get("expectation", models["judge"]))
                ),
                deterministic_judge=deterministic,
                issue_history=issue_history,
                severity_multipliers=settings.get("severity_multipliers"),
                fallback_frequencies=settings.get("fallback_frequencies"),
                weighting=settings.get("weighting", "auto"),
                dimension_reducer=settings.get("dimension_reducer", "mean"),
            )
        )
        description = Path(args.description).read_text(encoding="utf-8")
        report = pipeline.run(
            description=description,
            candidate_image=args.candidate,
            candidate_svg=candidate_svg,
            run_deterministic=not args.skip_deterministic and bool(candidate_svg),
        )
        output = Path(args.output)
        output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Correctness: {report.correctness:.4f}")
        for dimension, result in report.dimensions.items():
            print(f"  {dimension.value}: {result.score:.4f} (weight {report.weights[dimension]:.4f})")
        print(f"Report: {output.resolve()}")
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
