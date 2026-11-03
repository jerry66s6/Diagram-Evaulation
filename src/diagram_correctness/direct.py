"""Correctness by direct scoring: five independent model judgments, one per dimension.

Each dimension (Layout, Connectivity, Presence, Details, Legibility) is asked on its own with
the image and the caption, and the model returns the problems it sees and an integer score
from 1 to 10, the scale the human raters used. There is no expectation sketch and no scoring
formula: the prompt carries the scoring idea in words (what the dimension covers, what belongs
to the other dimensions, and how much a problem costs). Overall correctness is the mean of the
five scores.

This is version d17 of experiments/human_alignment, tuned on human-study diagrams HS001-HS200
(HS201-HS300 were not used for tuning). The earlier sketch-and-formula method (CorrectnessPipeline) is kept for the experiments that
compare against it; branch correctness-v-formula has main as it was before this change.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .models import Criterion
from .vlm import VLMBackend

CORRECTNESS_DIMENSIONS = ("layout", "connectivity", "presence", "details", "legibility")

# The question the human raters saw for each dimension (user-study/lib/metrics.ts).
QUESTIONS = {
    "layout": "Are elements placed in a sensible order, without unintended overlap, content outside the canvas, or "
              "arrows running through boxes or text?",
    "connectivity": "Are the required connections present, with arrows joining the right elements in the right "
                    "direction?",
    "presence": "Are all the components mentioned in the caption visibly present?",
    "details": "Are labels complete, accurate, and meaningful, rather than missing or placeholders?",
    "legibility": "Is the text comfortably readable, without tiny type, clipping, overflow, or low contrast against "
                  "its background?",
}
OWNERS = """\
  LAYOUT       - reading order, nothing cut off at the canvas edge, no unintended overlap, clean arrow routing.
  CONNECTIVITY - the connections the caption calls for are drawn between the right components, in the right
                 direction; no wrong or dangling arrows.
  PRESENCE     - every component the caption calls for is drawn, and nothing out of place is added.
  DETAILS      - the labels on components and arrows are complete, accurate, and meaningful.
  LEGIBILITY   - text is comfortably readable: large and sharp enough, not clipped or overflowing, enough contrast."""
# How real figures draw things, added after the rubric text of each dimension. The caption is a short summary
# rewritten from the figure, not a specification, so Connectivity and Presence flag only clear contradictions.
CONVENTIONS = {
    "layout": "Steps should appear along the reading direction in the order the caption gives. A box or label that "
              "touches the image edge is not cut off, even where its border lies on the edge and looks thin or "
              "missing. Content is cut off only when part of it is visibly missing, such as text sliced through or a "
              "box missing part of its contents.",
    "connectivity": "The caption is a short summary rewritten from the figure: it may merge, split, or reorder steps, "
                    "and it does not describe every line. Use it to understand the flow, and count a connection as "
                    "wrong only when the figure clearly contradicts that flow: two steps that must follow each other "
                    "have nothing joining them, an arrow points the wrong way, or an arrow leads to the wrong "
                    "component. One line or arrow can serve several components (a branching trunk line, a line ending "
                    "at a row of boxes, a large background arrow with the steps placed along it, an arrow ending at "
                    "the border of a group or panel) and connects each of them. A small box beside an arrow that "
                    "states its condition or a note (a threshold, a criterion, a count) labels that arrow and needs no "
                    "connection of its own. Two boxes that restate the same step need no arrow between them. Excluded "
                    "or removed items are often a side branch that leaves the main flow and ends there, but a figure "
                    "may also put the exclusion box in the main chain, meaning the next step is what remains; both "
                    "are correct. Judge the arrows against what the caption states, not against your own idea of how "
                    "the process should work.",
    "presence": "The caption is a short summary rewritten from the figure and may give more detail than the figure's "
                "labels (where the data come from, what each arm receives). Go through the components the caption "
                "names and find each in the figure: a component is present when the figure shows that step or group, "
                "even if its label says less than the caption (that is a DETAILS matter). Count a component as "
                "present only when a reader can see what it is: a box with no text, cut off, or hidden under another "
                "does not show its component; a box whose text is merely small or faint is present (that is a "
                "LEGIBILITY matter). Boxes that repeat a name in different arms or branches, or restate a stage with "
                "added detail, are not duplicates.",
    "details": "A label that is cut off, or that names the wrong step for its place in the flow, is not accurate. A "
               "label that names the same thing as the caption is accurate even if it is worded differently, "
               "abbreviated, or slightly misspelled; a label that names a different thing, contradicts the caption, "
               "or drops a count, value, or condition the caption gives for that component is wrong. Judge the "
               "labels themselves: do not recompute numbers across boxes or second-guess wording that fits the "
               "caption.",
    "legibility": "Judge text at the size the image is shown, without zooming in: text you would need to zoom in on "
                  "or guess at is not comfortably readable.",
}
SCORING = (
    "How to score, as an integer from 1 to 10. Look over the whole diagram for problems on this dimension only, "
    "then let each problem lower the score by how much it hurts a reader:\n"
    "- 9-10: no real problem on this dimension.\n"
    "- 7-8: small problems that do not change what the diagram says.\n"
    "- 4-6: at least one clear error that a reader would notice, or that changes part of the meaning.\n"
    "- 1-3: several such errors, or the diagram is misleading or unusable on this dimension.\n"
    "Be sensitive: a single clear error is enough to leave the top band. Errors that change the meaning cost more "
    "than cosmetic ones, and several errors cost more than one. A figure that is organized or worded differently "
    "from what you would draw, but still shows what the caption describes, has no problem. Problems that belong to "
    "another dimension do not lower this score. A problem is small only when a reader would barely notice it; one "
    "that makes part of the diagram harder to read, follow, or trust is a clear error."
)
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "problems": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "integer", "minimum": 1, "maximum": 10},
        "reasoning": {"type": "string"},
    },
    "required": ["problems", "score", "reasoning"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class CorrectnessScore:
    problems: list[str]
    score: int
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def correctness_prompt(dimension: str, description: str, criteria: list[Criterion]) -> str:
    covers = [c.description for c in criteria if c.dimension.value == dimension]
    name = dimension.upper()
    return (
        f"You are rating one aspect of a flowchart: {name}. The attached image is the flowchart, and the caption "
        f"below describes what it should show.\n\n{name}: {QUESTIONS[dimension]}\nIt covers: "
        + " ".join(covers) + (" " + CONVENTIONS[dimension] if CONVENTIONS[dimension] else "")
        + f"\n\nEach aspect has its own rater:\n{OWNERS}\nJudge only {name}.\n\n"
        + SCORING + "\n\n"
        + "First list the problems you see on this dimension, each with where it is (an empty list if there are "
        "none), then give the score and a one- or two-sentence reason.\n\nCaption:\n" + description
    )


def judge_correctness_dimension(
    backend: VLMBackend,
    image: str | Path,
    description: str,
    dimension: str,
    criteria: list[Criterion],
) -> CorrectnessScore:
    payload = backend.structured(
        correctness_prompt(dimension, description, criteria), [image], SCHEMA, f"diagram_correctness_{dimension}"
    )
    score = payload.get("score")
    if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 10:
        raise RuntimeError(f"Correctness judge returned an invalid score for {dimension}: {score!r}")
    return CorrectnessScore([str(p) for p in payload.get("problems", [])], score, str(payload.get("reasoning", "")))


def judge_correctness(
    backend: VLMBackend,
    image: str | Path,
    description: str,
    criteria: list[Criterion],
) -> dict[str, CorrectnessScore]:
    return {
        dimension: judge_correctness_dimension(backend, image, description, dimension, criteria)
        for dimension in CORRECTNESS_DIMENSIONS
    }


def overall(scores: dict[str, CorrectnessScore]) -> float:
    """Overall correctness: the mean of the five dimension scores (1-10), as for the human ratings."""
    return mean(s.score for s in scores.values())
