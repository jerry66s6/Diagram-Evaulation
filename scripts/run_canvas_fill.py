#!/usr/bin/env python3
"""Score Canvas Fill only with the current Beauty v6 VLM prompt."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagram_correctness.beauty import judge_beauty_dimension
from diagram_correctness.beauty_v6 import BEAUTY_V6_PROMPTS
from diagram_correctness.rubric import load_config
from diagram_correctness.vlm import OpenAIBackend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--backend", type=Path, help="Existing local module exposing Backend().")
    parser.add_argument("--model", help="Model override; otherwise use the backend's configured model.")
    parser.add_argument("--output", type=Path, required=True, help="JSON results with the exact prompt.")
    args = parser.parse_args()
    for image in args.images:
        if not image.is_file():
            parser.error(f"Image does not exist: {image}")
    if args.backend:
        spec = importlib.util.spec_from_file_location("canvas_fill_backend", args.backend)
        if spec is None or spec.loader is None:
            parser.error(f"Cannot load backend: {args.backend}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        backend = module.Backend(model=args.model) if args.model else module.Backend()
    else:
        backend = OpenAIBackend(
            args.model or load_config(ROOT / "config" / "evaluator.json")["models"]["judge"]
        )
    results = []
    for image in args.images:
        result = judge_beauty_dimension(backend, image, "canvas_fill", BEAUTY_V6_PROMPTS)
        results.append({"image": str(image.resolve()), **result.to_dict()})
        print(f"{image.name}: {result.score}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "dimension": "canvas_fill",
        "model": getattr(backend, "model", args.model),
        "prompt": BEAUTY_V6_PROMPTS["canvas_fill"],
        "response_instruction": (
            "\nReturn the requested observation first, then a score from 0 to 100, then a concise "
            "one- or two-sentence justification."
        ),
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
