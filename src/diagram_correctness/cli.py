from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .deterministic import DeterministicJudge, GeometryConfig
from .pipeline import CorrectnessPipeline, PipelineConfig
from .rubric import load_config, load_rubric
from .vfig import VFigRunner
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
        help="Rendered candidate image (requires VFIG for deterministic geometry checks)",
    )
    parser.add_argument("--rubric", default="config/rubric.json")
    parser.add_argument("--config", default="config/evaluator.json")
    parser.add_argument("--output", default="correctness-report.json")
    parser.add_argument(
        "--skip-deterministic",
        action="store_true",
        help="Run only the configured GPT judge (useful before VFIG is installed)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_config(args.config)
        criteria = load_rubric(args.rubric)
        models = settings["models"]
        judges = [VLMJudge(OpenAIBackend(models["judge"]))]
        geometry = settings.get("geometry", {})
        deterministic = DeterministicJudge(GeometryConfig(**geometry))
        vfig = None
        candidate_svg = args.candidate_svg
        if not args.skip_deterministic and not candidate_svg:
            vfig_command = settings.get("vfig", {}).get("command", [])
            if not vfig_command:
                raise ValueError(
                    "Deterministic evaluation requires --candidate-svg "
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
                    OpenAIBackend(models.get("expectation", models["judge"]))
                ),
                vfig_runner=vfig,
                deterministic_judge=deterministic,
                issue_history=issue_history,
                severity_multipliers=settings.get("severity_multipliers"),
                fallback_frequencies=settings.get("fallback_frequencies"),
            )
        )
        description = Path(args.description).read_text(encoding="utf-8")
        report = pipeline.run(
            description=description,
            candidate_image=args.candidate,
            candidate_svg=candidate_svg,
            run_deterministic=not args.skip_deterministic,
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
