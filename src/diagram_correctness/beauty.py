"""Historical five-dimension Beauty judge; use beauty_v6 for the current eight."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .vlm import VLMBackend


BEAUTY_DIMENSIONS = (
    "aesthetics",
    "palette",
    "ink_balance",
    "density",
    "balance",
)

_DIMENSION_OWNERS = """\
  AESTHETICS  - execution craft: shape, stroke, typography, connector, and rendering consistency.
  PALETTE     - chromatic hue count, coordination, and consistent use of color roles.
  INK BALANCE - what fraction of the canvas is covered by ink (fills, strokes, texture, text).
  DENSITY     - local clearance between neighboring, unrelated elements relative to their size.
  BALANCE     - global placement of foreground diagram content: margins and visual centroid.
"""


def _preamble(dimension_label: str, not_yours: str) -> str:
    return (
        "You are ONE of five independent judges scoring a diagram image. Each judge scores exactly "
        "one visual dimension and never sees the other judges' output.\n\n"
        f"The five dimensions and who owns them:\n{_DIMENSION_OWNERS}\n"
        f"YOU OWN: {dimension_label}. You score nothing else.\n\n"
        "ISOLATION RULES (follow strictly):\n"
        "- Assume the other four dimensions are already perfect. They are someone else's job.\n"
        f"- NOT YOURS: {not_yours}\n"
        "- If the only flaws you can see belong to another judge's dimension, you MUST still score "
        "HIGH. A defect owned by another dimension is not evidence against yours.\n"
        "- Do not form an overall 'this diagram looks bad/good' impression and transfer it to your "
        "score. Judge the single observable below and nothing else.\n"
        "- Apply the stated MENTAL MASK before observing. Score the counterfactual image after sibling "
        "signals are hidden, not the unmasked overall impression.\n"
        "- First state the OBSERVATION your dimension asks for, then derive the score strictly from "
        "that observation using the bands given.\n\n"
    )


BEAUTY_PROMPTS = {
    "aesthetics": _preamble(
        "AESTHETICS",
        "hue choice, color harmony, saturation, or color contrast (Palette); how much of the canvas "
        "is covered in ink, texture, hatching, or decoration (Ink Balance); gaps between elements "
        "(Density); where the content sits on the canvas (Balance).",
    )
    + (
        "MENTAL MASK: convert every color to the same neutral gray, remove full-canvas texture/"
        "hatching, and ignore all element positions and gaps. Preserve shapes, strokes, type geometry, "
        "connectors, and rendering quality.\n\n"
        "OBSERVATION TO MAKE: in that masked image, describe execution CRAFT only -- shape treatment, "
        "stroke-weight control, typography sizing/alignment, connector and arrowhead construction, "
        "edge quality, and consistency of those treatments. Do not discuss color or whitespace.\n\n"
        "SCORE BANDS:\n"
        "- 90-100: no visible craft defects; deliberate, refined control of strokes, type, shapes, "
        "and connectors.\n"
        "- 70-89: one or two minor craft defects, or clean consistent execution that is mostly default.\n"
        "- 40-69: repeated craft defects across three or more elements, such as arbitrary stroke or "
        "font changes, crude shapes, weak alignment, or inconsistent connector treatment.\n"
        "- 0-39: pervasive inconsistent or broken rendering that looks materially unfinished.\n\n"
        "HARD CAP: if three or more equivalent nodes/connectors use visibly different stroke, type, "
        "alignment, or arrow treatments without a semantic reason, the score cannot exceed 59.\n"
        "Judging a diagram 'plain' is fine and belongs in the 60-89 band. Do NOT drop below 60 just "
        "because the diagram is simple, sparse, or uses few colors.\n"
        "CRITICAL: do not mention text-on-fill contrast, clashing colors, visual noise, clutter, "
        "centering, or spacing in the observation or reasoning. Those are explicitly masked out."
    ),
    "palette": _preamble(
        "PALETTE",
        "shape/stroke/type/connector execution (Aesthetics); how much ink covers the canvas "
        "(Ink Balance); element spacing (Density); composition placement (Balance).",
    )
    + (
        "MENTAL MASK: keep each element's color but normalize shape geometry, stroke weight, spacing, "
        "and placement. Ignore text readability; accessibility contrast is not scored here.\n\n"
        "OBSERVATION TO MAKE: count the perceptually distinct CHROMATIC hue families and identify what "
        "roles they serve (for example process, decision, connector, emphasis). Then state whether the "
        "hues coordinate and whether repeated roles use color consistently. Rules for counting:\n"
        "- Ignore neutrals entirely: white, black, and grays do not count.\n"
        "- Shades/tints of the SAME hue family count as ONE (navy + light blue = 1).\n"
        "- State the count, list the hue families, and describe their role mapping.\n\n"
        "SCORE BANDS:\n"
        "- 90-100: usually 2-4 coordinated hues with clear, consistent roles and restrained emphasis.\n"
        "- 70-89: coordinated but limited/monochromatic, or 5 hues with mostly clear roles.\n"
        "- 40-69: excessive or weakly coordinated hues, or repeated semantic roles colored inconsistently.\n"
        "- 0-39: arbitrary rainbow-like assignment with no stable role system or evident coordination.\n\n"
        "The hue count is a guardrail, not the score by itself. A deliberate neutral or monochrome "
        "diagram can still score 70-89; a two-color clash with arbitrary role assignment cannot score "
        "high merely because its count is small. Saturation alone is not a defect."
    ),
    "ink_balance": _preamble(
        "INK BALANCE",
        "styling taste (Aesthetics); color count (Palette); the gaps between elements (Density); "
        "whether the content is centered on the canvas (Balance).",
    )
    + (
        "MENTAL MASK: ignore what the marks depict, their colors, their local gaps, and where their "
        "foreground group is positioned. Treat texture and decoration as ink.\n\n"
        "OBSERVATION TO MAKE: estimate roughly what PERCENTAGE of the total canvas area is covered by "
        "non-background ink -- fills, strokes, hatching/texture, and text. State the estimate.\n\n"
        "SCORE BANDS:\n"
        "- 90-100: roughly 8-40% coverage -- substantive content that reads clearly at a glance.\n"
        "- 60-89: roughly 5-8% or 40-55% -- a little thin or a little heavy, still fine.\n"
        "- 30-59: roughly 2-5% or 55-75% -- noticeably bare, or busy with heavy fills/texture.\n"
        "- 0-29: under 2% (essentially blank) or over 75% (canvas smothered in ink/texture).\n\n"
        "CRITICAL: empty whitespace is NOT a defect here as long as coverage is above ~8%. A compact "
        "diagram surrounded by generous white space is HEALTHY and belongs in the 90-100 band. Do not "
        "deduct for 'large unused areas', for content sitting in one band of the canvas, or for where "
        "the content is placed -- that is the Balance judge's job, not yours."
    ),
    "density": _preamble(
        "DENSITY",
        "how much of the canvas is used or how much whitespace surrounds the content (Ink Balance); "
        "where the group sits on the canvas (Balance); color (Palette); styling (Aesthetics).",
    )
    + (
        "MENTAL MASK: crop conceptually to the foreground diagram group, recolor it uniformly, and "
        "ignore styling. Exclude only the exact attachment point where a connector touches its source "
        "or target; do not exempt the box-to-box gap between connected nodes.\n\n"
        "OBSERVATION TO MAKE: inspect local clearance between neighboring DISTINCT elements: "
        "box-to-box, label-to-border, label-to-connector, and connectors that should remain separate. "
        "Estimate typical and minimum clearances relative to the nearby element size.\n\n"
        "SCORE BANDS:\n"
        "- 90-100: typical gaps are ~30-100% of element size, with comfortable minimum clearances.\n"
        "- 70-89: typical gaps are ~20-30% or ~100-150%; readable but slightly tight or loose.\n"
        "- 40-69: typical gaps are ~5-20% or over ~150%; cramped or fragmented.\n"
        "- 0-39: unrelated elements touch/overlap, or labels/connectors materially collide.\n\n"
        "HARD CAP: if the visible box-to-box gap is under 5% of box width, the score cannot exceed 39, "
        "even when those boxes are joined by a connector.\n"
        "CRITICAL: this dimension is LOCAL only. A small tight cluster of elements sitting inside a "
        "huge empty canvas should score based ONLY on the gaps inside that cluster. Do NOT deduct "
        "because the content occupies a single band or row, because there is large unused space above/"
        "below/around, or because the canvas is 'unevenly used' -- none of that is yours. Generous "
        "whitespace AROUND the group is irrelevant; only gaps BETWEEN elements matter."
    ),
    "balance": _preamble(
        "BALANCE",
        "how much ink is on the canvas (Ink Balance); the spacing between elements (Density); color "
        "(Palette); styling refinement (Aesthetics).",
    )
    + (
        "MENTAL MASK: remove any uniform/full-canvas background fill, texture, border, or decoration. "
        "Keep only foreground diagram elements and titles; normalize their colors and styling.\n\n"
        "OBSERVATION TO MAKE: draw the bounding box around that foreground content and estimate its "
        "visual centroid. Compare left vs right and top vs bottom margins, and state whether the "
        "centroid is displaced from canvas center.\n\n"
        "SCORE BANDS (use the WORST of the two axes):\n"
        "- 90-100: both margin pairs are near-symmetric (larger margin < ~1.5x the smaller).\n"
        "- 70-89: mildly off-center on one axis (larger margin ~1.5-2.5x the smaller).\n"
        "- 40-69: clearly shifted to one side/edge (larger margin ~2.5-5x the smaller).\n"
        "- 0-39: severely lopsided -- content jammed against one edge or corner (larger margin > ~5x)."
        "\n\nA title sitting above the diagram is foreground content, but a full-canvas pattern is "
        "background and must not force a perfect score. Judge only global foreground placement: total "
        "whitespace, local packing, color, and styling belong to other judges."
    ),
}

BEAUTY_SCORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "observation": {"type": "string"},
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "reasoning": {"type": "string"},
    },
    "required": ["observation", "score", "reasoning"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class BeautyScore:
    observation: str
    score: int
    reasoning: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "observation": self.observation,
            "score": self.score,
            "reasoning": self.reasoning,
        }


def judge_beauty_dimension(
    backend: VLMBackend,
    image: str | Path,
    dimension: str,
    prompts: dict[str, str] | None = None,
) -> BeautyScore:
    prompts = BEAUTY_PROMPTS if prompts is None else prompts
    if dimension not in prompts:
        raise ValueError(
            f"Unknown beauty dimension {dimension!r}; expected one of {tuple(prompts)}"
        )
    prompt = (
        prompts[dimension]
        + "\nReturn the requested observation first, then a score from 0 to 100, then a concise "
        "one- or two-sentence justification."
    )
    payload = backend.structured(
        prompt,
        [image],
        BEAUTY_SCORE_SCHEMA,
        f"diagram_beauty_{dimension}",
    )
    observation = payload.get("observation")
    score = payload.get("score")
    reasoning = payload.get("reasoning")
    if not isinstance(observation, str) or not observation.strip():
        raise RuntimeError(f"Beauty judge returned no observation for {dimension}")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
        raise RuntimeError(f"Beauty judge returned an invalid score for {dimension}: {score!r}")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise RuntimeError(f"Beauty judge returned no reasoning for {dimension}")
    return BeautyScore(
        observation=observation.strip(),
        score=score,
        reasoning=reasoning.strip(),
    )


def judge_beauty(
    backend: VLMBackend,
    image: str | Path,
) -> dict[str, BeautyScore]:
    return {
        dimension: judge_beauty_dimension(backend, image, dimension)
        for dimension in BEAUTY_DIMENSIONS
    }
