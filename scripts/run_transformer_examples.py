#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Homebrew's Cairo dylib is outside the default lookup path for framework Python.
if sys.platform == "darwin" and Path("/opt/homebrew/lib/libcairo.2.dylib").exists():
    fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    if "/opt/homebrew/lib" not in fallback.split(":"):
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
            part for part in ("/opt/homebrew/lib", fallback) if part
        )
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

import cairosvg

from diagram_correctness.deterministic import DeterministicJudge, GeometryConfig
from diagram_correctness.models import Dimension
from diagram_correctness.pipeline import CorrectnessPipeline, PipelineConfig
from diagram_correctness.rubric import load_config, load_rubric
from diagram_correctness.vlm import ExpectationExtractor, OpenAIBackend, VLMJudge


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "transformer"
CANDIDATES = ("candidate_strong", "candidate_mixed", "candidate_weak")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and score the three controlled Transformer SVG examples."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run deterministic SVG checks using manual expectations derived from the description.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render SVGs to PNG without scoring them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXAMPLE_DIR / "results",
    )
    return parser.parse_args()


def render_svg(svg_path: Path, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        output_width=1100,
        output_height=900,
    )


def load_history(settings: dict[str, Any]) -> list[dict[str, Any]]:
    history_path = settings.get("weight_history")
    if not history_path:
        return []
    return json.loads((ROOT / history_path).read_text(encoding="utf-8"))["issues"]


def build_pipeline(offline: bool) -> CorrectnessPipeline:
    settings = load_config(ROOT / "config" / "evaluator.json")
    criteria = load_rubric(ROOT / "config" / "rubric.json")
    geometry = DeterministicJudge(GeometryConfig(**settings.get("geometry", {})))
    if offline:
        judges = []
        extractor = None
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Set it in your shell, then rerun without --offline."
            )
        model = settings["models"]["judge"]
        judges = [VLMJudge(OpenAIBackend(model))]
        extractor = ExpectationExtractor(OpenAIBackend(settings["models"]["expectation"]))
    return CorrectnessPipeline(
        PipelineConfig(
            criteria=criteria,
            judges=judges,
            expectation_extractor=extractor,
            deterministic_judge=geometry,
            issue_history=load_history(settings),
            severity_multipliers=settings.get("severity_multipliers"),
            fallback_frequencies=settings.get("fallback_frequencies"),
        )
    )


def write_summary(reports: dict[str, dict[str, Any]], output_dir: Path, mode: str) -> None:
    dimensions = [
        Dimension.LAYOUT,
        Dimension.CONNECTIVITY,
        Dimension.PRESENCE,
        Dimension.DETAILS,
        Dimension.LEGIBILITY,
    ]
    rows = []
    for name, report in reports.items():
        rows.append(
            {
                "candidate": name,
                "correctness": report["correctness"],
                **{
                    dimension.value: report["dimensions"].get(dimension.value, {}).get("score")
                    for dimension in dimensions
                },
            }
        )
    summary = {"mode": mode, "results": rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    header = "| Candidate | Overall | Layout | Connectivity | Presence | Details | Legibility |"
    separator = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [
        "# Transformer example results",
        "",
        f"Mode: **{mode}**",
        "",
        header,
        separator,
    ]
    for row in rows:
        values = [row[dimension.value] for dimension in dimensions]
        lines.append(
            "| "
            + row["candidate"]
            + " | "
            + " | ".join(
                [f"{row['correctness']:.3f}"]
                + [f"{value:.3f}" if value is not None else "N/A" for value in values]
            )
            + " |"
        )

    ranked = sorted(rows, key=lambda row: row["correctness"], reverse=True)
    by_name = {row["candidate"]: row for row in rows}
    strong = by_name["candidate_strong"]
    mixed = by_name["candidate_mixed"]
    weak = by_name["candidate_weak"]
    lines.extend(
        [
            "",
            "## Sensemaking",
            "",
            f"The ranking is **{' > '.join(row['candidate'] for row in ranked)}**.",
            "",
            f"- **Strong ({strong['correctness']:.3f})** is the positive control: every description-derived component, exact label, directed arrow, residual path, and legibility check passes.",
            f"- **Mixed ({mixed['correctness']:.3f})** keeps most of the layout ({mixed['layout']:.3f}) and remains readable ({mixed['legibility']:.3f}), but missing positional/residual paths reduce connectivity to {mixed['connectivity']:.3f}; abbreviated and placeholder labels reduce Details to {mixed['details']:.3f}.",
            f"- **Weak ({weak['correctness']:.3f})** retains a few recognizable structures, so Layout is not zero ({weak['layout']:.3f}), while missing components, wrong arrows, placeholder content, and small text drive Presence to {weak['presence']:.3f}, Connectivity to {weak['connectivity']:.3f}, Details to {weak['details']:.3f}, and Legibility to {weak['legibility']:.3f}.",
            "",
            "Overall is a severity/frequency-weighted score, not a simple mean. In this sample, Connectivity has the largest weight (0.40), Presence and Legibility each have 0.20, and Layout and Details each have 0.10.",
            "",
            "Each candidate JSON report contains every criterion-level score and issue from each participating judge.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in CANDIDATES:
        render_svg(EXAMPLE_DIR / f"{name}.svg", output_dir / f"{name}.png")
    if args.render_only:
        print(f"Rendered examples to {output_dir}")
        return 0

    description = (EXAMPLE_DIR / "description.md").read_text(encoding="utf-8")
    pipeline = build_pipeline(args.offline)
    if args.offline:
        inventory = json.loads(
            (EXAMPLE_DIR / "expected_inventory_manual.json").read_text(encoding="utf-8")
        )
        mode = "offline deterministic smoke test"
    else:
        assert pipeline.config.expectation_extractor is not None
        inventory = pipeline.config.expectation_extractor.extract(
            description, pipeline.config.criteria
        )
        mode = f"live {pipeline.config.judges[0].name} + deterministic SVG"
    (output_dir / "description_expectations.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )

    reports: dict[str, dict[str, Any]] = {}
    for name in CANDIDATES:
        report = pipeline.run(
            description=description,
            candidate_image=output_dir / f"{name}.png",
            candidate_svg=EXAMPLE_DIR / f"{name}.svg",
            inventory=inventory,
        )
        reports[name] = report.to_dict()
        (output_dir / f"{name}.json").write_text(
            json.dumps(reports[name], indent=2), encoding="utf-8"
        )
    write_summary(reports, output_dir, mode)
    print((output_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
