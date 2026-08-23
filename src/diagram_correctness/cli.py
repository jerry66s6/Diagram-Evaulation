from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .deterministic import DeterministicJudge, GeometryConfig
from .pipeline import CorrectnessPipeline, PipelineConfig
from .rubric import load_config, load_rubric
from .vfig import VFigRunner
from .vlm import AnthropicBackend, ExpectationExtractor, OpenAIBackend, VLMJudge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate diagram correctness with a multi-judge panel")
    parser.add_argument("--reference", required=True, help="Reference/target diagram image")
    parser.add_argument("--candidate", required=True, help="Rendered candidate diagram image")
    parser.add_argument("--reference-svg", help="Existing reference SVG; bypass VFIG for this image")
    parser.add_argument("--candidate-svg", help="Existing candidate SVG; bypass VFIG for this image")
    parser.add_argument("--rubric", default="config/rubric.json")
    parser.add_argument("--config", default="config/evaluator.json")
    parser.add_argument("--output", default="correctness-report.json")
    parser.add_argument(
        "--skip-deterministic",
        action="store_true",
        help="Run only the three VLM judges (useful before VFIG is installed)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_config(args.config)
        criteria = load_rubric(args.rubric)
        models = settings["models"]
        judges = [
            VLMJudge(OpenAIBackend(models["openai"][0])),
            VLMJudge(OpenAIBackend(models["openai"][1])),
            VLMJudge(AnthropicBackend(models["anthropic"])),
        ]
        geometry = settings.get("geometry", {})
        deterministic = DeterministicJudge(GeometryConfig(**geometry))
        vfig = None
        reference_svg = args.reference_svg
        candidate_svg = args.candidate_svg
        if not args.skip_deterministic and not (reference_svg and candidate_svg):
            vfig_command = settings.get("vfig", {}).get("command", [])
            if not vfig_command:
                raise ValueError(
                    "Deterministic evaluation requires both --reference-svg and --candidate-svg, "
                    "or a config.vfig.command. Use --skip-deterministic to omit it."
                )
            vfig = VFigRunner(
                command=vfig_command,
                timeout_seconds=int(settings.get("vfig", {}).get("timeout_seconds", 600)),
            )

        issue_history = []
        history_path = settings.get("weight_history")
        if history_path:
            issue_history = json.loads(Path(history_path).read_text(encoding="utf-8"))["issues"]

        pipeline = CorrectnessPipeline(
            PipelineConfig(
                criteria=criteria,
                judges=judges,
                expectation_extractor=ExpectationExtractor(
                    OpenAIBackend(models.get("expectation", "gpt-5.6"))
                ),
                vfig_runner=vfig,
                deterministic_judge=deterministic,
                issue_history=issue_history,
                severity_multipliers=settings.get("severity_multipliers"),
                fallback_frequencies=settings.get("fallback_frequencies"),
            )
        )
        report = pipeline.run(
            reference_image=args.reference,
            candidate_image=args.candidate,
            reference_svg=reference_svg if not args.skip_deterministic else None,
            candidate_svg=candidate_svg if not args.skip_deterministic else None,
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

