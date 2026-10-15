"""Four-dimension Beauty judges: Centering, Space Use, Color Consistency, Contrast."""
from __future__ import annotations

from pathlib import Path

from .beauty import BeautyScore, judge_beauty_dimension
from .vlm import VLMBackend


BEAUTY_V4_DIMENSIONS = (
    "centering",
    "space_use",
    "color_consistency",
    "contrast",
)

_DIMENSION_OWNERS = """\
  CENTERING         - where the whole diagram sits on the canvas: opposing margins and visual centroid.
  SPACE USE         - whether nodes feel condensed together while usable canvas space is left empty.
  COLOR CONSISTENCY - whether colors form one coherent theme with stable color-to-role mapping.
  CONTRAST          - whether boxes and arrows visibly separate from the background and from touching boxes.
"""


def _preamble(dimension_label: str, not_yours: str) -> str:
    return (
        "You are ONE of four independent judges scoring a flowchart image. Each judge scores exactly "
        "one visual dimension and never sees the other judges' output. Content and structure are "
        "judged elsewhere; this is purely visual.\n\n"
        f"The four dimensions and who owns them:\n{_DIMENSION_OWNERS}\n"
        f"YOU OWN: {dimension_label}. You score nothing else.\n\n"
        "ISOLATION RULES (follow strictly):\n"
        "- Assume the other three dimensions are already perfect. They are someone else's job.\n"
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


BEAUTY_V4_PROMPTS = {
    "centering": _preamble(
        "CENTERING",
        "how large the diagram is or how tightly its nodes are packed (Space Use); which colors are "
        "used (Color Consistency); how visible boxes and arrows are (Contrast).",
    )
    + (
        "MENTAL MASK: replace every foreground element (boxes, arrows, labels, and any title) with a "
        "solid uniform dark silhouette, then treat any uniform background fill as empty canvas. Ignore "
        "the size of the silhouette and the gaps inside it.\n\n"
        "OBSERVATION TO MAKE: draw one bounding box around all foreground content. Estimate the left, "
        "right, top, and bottom margins as percentages of the canvas width/height, and state whether "
        "the visual centroid is displaced from the canvas center.\n\n"
        "SCORE BANDS (use the WORST of the two axes):\n"
        "- 90-100: both margin pairs near-symmetric (larger margin < ~1.5x the smaller).\n"
        "- 70-89: mildly off-center on one axis (larger margin ~1.5-2.5x the smaller).\n"
        "- 40-69: clearly shifted toward a side or edge (larger margin ~2.5-5x the smaller).\n"
        "- 0-39: severely lopsided; content pushed into a corner or against an edge (> ~5x).\n\n"
        "CRITICAL: a small or compact diagram surrounded by large but EQUAL margins is perfectly "
        "centered and belongs in 90-100. Do not deduct for unused canvas, cramped nodes, colors, or "
        "faint elements. Only the position of the whole group matters."
    ),
    "space_use": _preamble(
        "SPACE USE",
        "where the whole group sits on the canvas or whether its margins are symmetric (Centering); "
        "which colors are used (Color Consistency); how visible boxes and arrows are (Contrast).",
    )
    + (
        "MENTAL MASK: mentally slide the whole diagram so it is centered on the canvas, and recolor "
        "every element as a uniform mid-gray outline. Keep the node sizes and the distances between "
        "nodes exactly as drawn.\n\n"
        "OBSERVATION TO MAKE: estimate (a) what fraction of the canvas width and height the node group "
        "spans after re-centering, (b) the typical gap between neighboring nodes relative to node "
        "size, and (c) whether the nodes feel condensed together while usable canvas space remains "
        "empty around them. Identify the layout's MAIN direction (the axis with more nodes).\n\n"
        "SCORE BANDS (use the WORSE of the span and gap readings):\n"
        "- 90-100: nodes spread along the main direction (roughly 55%+ of the canvas on that axis) with "
        "comfortable gaps (about 40%+ of node size); nothing feels squeezed. A layout with only one or "
        "two rows/columns does NOT need to fill the cross axis.\n"
        "- 70-89: slightly compact; main-direction span ~45-55%, or gaps ~25-40% of node size.\n"
        "- 40-69: noticeably condensed; main-direction span ~30-45%, or gaps ~10-25% of node size, "
        "while plenty of empty canvas remains.\n"
        "- 0-39: crammed; main-direction span under ~30% or nodes nearly touching, while most of the "
        "canvas is empty.\n\n"
        "CRITICAL: an off-center diagram is NOT condensed. Do not deduct because empty space sits on "
        "one side; the re-centering mask removes position. Do not deduct for colors or faint elements. "
        "If the canvas is genuinely full, tight gaps are not a Space Use defect."
    ),
    "color_consistency": _preamble(
        "COLOR CONSISTENCY",
        "how light, pale, washed-out, or hard to see a color is (Contrast); where the diagram sits "
        "(Centering); how spread out or packed it is (Space Use).",
    )
    + (
        "MENTAL MASK: re-tint every color to the same medium lightness and saturation, keeping only "
        "its hue identity. Ignore element positions and gaps.\n\n"
        "OBSERVATION TO MAKE: list the distinct CHROMATIC hue families in the diagram and map each to "
        "the element roles that use it (for example start/end terminal, process step, decision, "
        "arrow). Rules:\n"
        "- White, black, and grays are neutrals and do not count as theme colors.\n"
        "- Tints and shades of the same hue are ONE family (pale blue + navy = blue).\n"
        "- State the count, list the families, and say whether elements with the same role share one "
        "family.\n\n"
        "SCORE BANDS:\n"
        "- 90-100: 1-3 related hue families, each role colored consistently; a coherent theme.\n"
        "- 70-89: coherent, but one extra hue or one minor role inconsistency.\n"
        "- 40-69: several unrelated hues, or the same role colored differently in two or more "
        "elements.\n"
        "- 0-39: arbitrary rainbow-like coloring with no stable color-to-role mapping.\n\n"
        "CRITICAL: pale, washed-out, low-saturation, or hard-to-see colors are NOT inconsistency. A "
        "theme of very light tints is perfectly consistent if the hues and roles are stable. Do not "
        "mention visibility, contrast, centering, or spacing."
    ),
    "contrast": _preamble(
        "CONTRAST",
        "whether the hues form a coherent theme or how many different hues appear (Color Consistency); "
        "where the diagram sits (Centering); how spread out or packed it is (Space Use). Text "
        "legibility inside boxes is also not scored here.",
    )
    + (
        "MENTAL MASK: convert the image to grayscale so only lightness remains. Ignore hue identity, "
        "element positions, and the gaps between elements.\n\n"
        "OBSERVATION TO MAKE: for (a) each box versus the background, (b) boxes that touch or overlap "
        "other boxes, and (c) arrows/connectors versus the background, estimate how strongly they "
        "separate in lightness. Name the WEAKEST pair and how visible it is.\n\n"
        "SCORE BANDS (driven by the weakest pair):\n"
        "- 90-100: every box and arrow stands out immediately from what it sits against.\n"
        "- 70-89: everything is clearly visible, but one class of element is somewhat soft.\n"
        "- 40-69: one class of element is FAINT: pastel/near-white box fills or pale gray connectors "
        "on a light background (or the dark-background equivalent).\n"
        "- 0-39: boxes AND arrows are both faint, or any element nearly blends into the background.\n\n"
        "CALIBRATION: a pastel or near-white tint on a white background (for example very light blue "
        "on white) is FAINT even though its outline can still be made out. Pale gray arrows on white "
        "are FAINT. Dark text inside a box does not make the box itself stand out.\n\n"
        "A dark background is acceptable when elements separate clearly from it; a light background "
        "with near-white boxes is not. CRITICAL: many unrelated hues are NOT a contrast defect if each "
        "one separates clearly from the background. Do not mention theme coherence, centering, or "
        "spacing."
    ),
}


def judge_beauty_v4(
    backend: VLMBackend,
    image: str | Path,
) -> dict[str, BeautyScore]:
    return {
        dimension: judge_beauty_dimension(backend, image, dimension, BEAUTY_V4_PROMPTS)
        for dimension in BEAUTY_V4_DIMENSIONS
    }
