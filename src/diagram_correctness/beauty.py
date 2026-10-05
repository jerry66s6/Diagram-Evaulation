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

BEAUTY_PROMPTS = {
    "aesthetics": (
        "Judge only the AESTHETICS of this diagram: its overall visual "
        "sophistication, coherence, and richness. Do not judge correctness, color "
        "count, whitespace balance, or legibility. Score 90-100 for polished and "
        "professional design, 60-89 for attractive but plain or slightly inconsistent "
        "styling, 30-59 for crude or amateurish styling, and 0-29 for jarring design "
        "with no evident visual care."
    ),
    "palette": (
        "Judge only the PALETTE of this diagram: whether it uses a healthy middle "
        "range of perceptually distinct, coordinated colors. Count similar shades as "
        "one perceptual color. Score 90-100 for roughly 2-5 coordinated purposeful "
        "colors, 60-89 for slightly too few or too many, 30-59 for clearly too few or "
        "too many uncoordinated colors, and 0-29 for monochrome or a chaotic rainbow."
    ),
    "ink_balance": (
        "Judge only the INK BALANCE of this diagram: whether the canvas contains a "
        "healthy amount of visual content rather than being nearly empty or overloaded "
        "with fills, hatching, borders, or decoration. Score 90-100 for healthy and "
        "uncluttered ink, 60-89 for slightly sparse or heavy, 30-59 for noticeably "
        "empty or busy, and 0-29 for almost blank or extremely overloaded."
    ),
    "density": (
        "Judge only the DENSITY of this diagram: whether elements are comfortably "
        "spaced rather than forming an overly dense or cramped cluster. Score 90-100 "
        "for comfortable even spacing, 60-89 for somewhat uneven but uncramped "
        "spacing, 30-59 for a noticeable cluster, and 0-29 for a tight or overlapping "
        "blob. Do not judge color, aesthetics, balance, or overall ink coverage."
    ),
    "balance": (
        "Judge only the BALANCE of this diagram: whether the composition is centered "
        "rather than visually lopsided toward one side or corner. Score 90-100 for "
        "balanced whitespace, 60-89 for slightly off-center composition, 30-59 for a "
        "noticeable shift with a large empty area, and 0-29 for a severe corner bias. "
        "Do not judge color, aesthetics, ink amount, or element spacing."
    ),
}

BEAUTY_SCORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class BeautyScore:
    score: int
    reasoning: str

    def to_dict(self) -> dict[str, int | str]:
        return {"score": self.score, "reasoning": self.reasoning}


def judge_beauty_dimension(
    backend: VLMBackend,
    image: str | Path,
    dimension: str,
) -> BeautyScore:
    if dimension not in BEAUTY_PROMPTS:
        raise ValueError(
            f"Unknown beauty dimension {dimension!r}; expected one of {BEAUTY_DIMENSIONS}"
        )
    prompt = (
        BEAUTY_PROMPTS[dimension]
        + "\nReturn a score from 0 to 100 and a concise one- or two-sentence justification."
    )
    payload = backend.structured(
        prompt,
        [image],
        BEAUTY_SCORE_SCHEMA,
        f"diagram_beauty_{dimension}",
    )
    score = payload.get("score")
    reasoning = payload.get("reasoning")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
        raise RuntimeError(f"Beauty judge returned an invalid score for {dimension}: {score!r}")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise RuntimeError(f"Beauty judge returned no reasoning for {dimension}")
    return BeautyScore(score=score, reasoning=reasoning.strip())


def judge_beauty(
    backend: VLMBackend,
    image: str | Path,
) -> dict[str, BeautyScore]:
    return {
        dimension: judge_beauty_dimension(backend, image, dimension)
        for dimension in BEAUTY_DIMENSIONS
    }
