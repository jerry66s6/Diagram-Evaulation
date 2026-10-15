"""Beauty v5: six model-judged dimensions plus the deterministic Canvas Fill check.

v5 keeps the four v4 dimensions and adds Alignment, Style Uniformity, and Canvas Fill.
Canvas Fill is measured in Python (see canvas_fill.py) but appears in the ownership map so
the model judges know not to score emptiness themselves.
"""
from __future__ import annotations

from pathlib import Path

from .beauty import BeautyScore, judge_beauty_dimension
from .canvas_fill import judge_canvas_fill
from .vlm import VLMBackend


BEAUTY_V5_MODEL_DIMENSIONS = (
    "centering",
    "space_use",
    "color_consistency",
    "contrast",
    "alignment",
    "style_uniformity",
)
BEAUTY_V5_DIMENSIONS = (*BEAUTY_V5_MODEL_DIMENSIONS, "canvas_fill")

_DIMENSION_OWNERS = """\
  CENTERING         - where the whole diagram sits on the canvas: opposing margins and visual centroid.
  SPACE USE         - whether nodes feel condensed together while usable canvas space is left empty.
  COLOR CONSISTENCY - whether colors form one coherent theme with stable color-to-role mapping.
  CONTRAST          - whether boxes and arrows visibly separate from the background and from touching boxes.
  ALIGNMENT         - whether nodes that form a row or column share one center line, with straight connectors.
  STYLE UNIFORMITY  - whether elements of the same role share one shape, size, border, font, and arrow treatment.
  CANVAS FILL       - (measured by a script, not a judge) whether elements are too small for the canvas,
                      leaving most of it empty.
"""

_SIBLINGS = {
    "centering": "where the whole group sits on the canvas or whether its margins are symmetric (Centering)",
    "space_use": "whether nodes are packed too tightly together (Space Use)",
    "color_consistency": "which hues are used or whether they form a coherent theme (Color Consistency)",
    "contrast": "how light, pale, or hard to see boxes and arrows are (Contrast)",
    "alignment": "whether nodes line up in rows/columns or connectors are slanted (Alignment)",
    "style_uniformity": "whether same-role elements differ in shape, size, border, font, or arrow weight (Style Uniformity)",
    "canvas_fill": "how much of the canvas is empty or how small the elements are relative to it (Canvas Fill)",
}


def _not_yours(dimension: str, extra: str = "") -> str:
    parts = [text for key, text in _SIBLINGS.items() if key != dimension]
    return "; ".join(parts) + ("; " + extra if extra else "") + "."


def _preamble(dimension: str, label: str, extra: str = "") -> str:
    return (
        "You are ONE of six independent judges scoring a flowchart image; a seventh dimension is measured "
        "by a script. Each judge scores exactly one visual dimension and never sees the other judges' "
        "output. Content and structure are judged elsewhere; this is purely visual.\n\n"
        f"The seven dimensions and who owns them:\n{_DIMENSION_OWNERS}\n"
        f"YOU OWN: {label}. You score nothing else.\n\n"
        "ISOLATION RULES (follow strictly):\n"
        "- Assume every other dimension is already perfect. They are someone else's job.\n"
        f"- NOT YOURS: {_not_yours(dimension, extra)}\n"
        "- If the only flaws you can see belong to another dimension, you MUST still score HIGH. A defect "
        "owned by another dimension is not evidence against yours.\n"
        "- Do not form an overall 'this diagram looks bad/good' impression and transfer it to your "
        "score. Judge the single observable below and nothing else.\n"
        "- Apply the stated MENTAL MASK before observing. Score the counterfactual image after sibling "
        "signals are hidden, not the unmasked overall impression.\n"
        "- First state the OBSERVATION your dimension asks for, then derive the score strictly from "
        "that observation using the bands given.\n\n"
    )


BEAUTY_V5_PROMPTS = {
    "centering": _preamble("centering", "CENTERING")
    + (
        "MENTAL MASK: replace every foreground element (boxes, arrows, labels, and any title) with a "
        "solid uniform silhouette, then treat any uniform background fill, light OR dark, as empty "
        "canvas. Ignore the size of the silhouette and the gaps and alignment inside it.\n\n"
        "OBSERVATION TO MAKE: draw one bounding box around all foreground content. Estimate the left, "
        "right, top, and bottom margins as percentages of the canvas width/height, and state whether "
        "the visual centroid is displaced from the canvas center.\n\n"
        "SCORE BANDS (use the WORST of the two axes):\n"
        "- 90-100: both margin pairs near-symmetric (larger margin < ~1.5x the smaller).\n"
        "- 70-89: mildly off-center on one axis (larger margin ~1.5-2.5x the smaller).\n"
        "- 40-69: clearly shifted toward a side or edge (larger margin ~2.5-5x the smaller).\n"
        "- 0-39: severely lopsided; content pushed into a corner or against an edge (> ~5x).\n\n"
        "CRITICAL: a small or compact diagram surrounded by large but EQUAL margins is perfectly "
        "centered and belongs in 90-100. Do not deduct for unused canvas, cramped or misaligned nodes, "
        "styling, colors, or faint elements. Only the position of the whole group matters."
    ),
    "space_use": _preamble("space_use", "SPACE USE")
    + (
        "MENTAL MASK: mentally slide the whole diagram so it is centered on the canvas, recolor every "
        "element as a uniform mid-gray outline, and snap every node onto clean rows/columns. Keep the "
        "node sizes and the distances between nodes as drawn.\n\n"
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
        "CRITICAL: only TOO-TIGHT packing is yours. Generous or even very large gaps, tiny elements, and "
        "a mostly empty canvas are NOT Space Use defects. An off-center or misaligned diagram is not "
        "condensed. If the canvas is genuinely full, tight gaps are not a defect."
    ),
    "color_consistency": _preamble(
        "color_consistency", "COLOR CONSISTENCY", "shape, size, font, or border differences between elements"
    )
    + (
        "MENTAL MASK: re-tint every color to the same medium lightness and saturation, keeping only "
        "its hue identity. Ignore element positions, gaps, shapes, and fonts.\n\n"
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
        "mention visibility, contrast, centering, spacing, alignment, or non-color styling."
    ),
    "contrast": _preamble("contrast", "CONTRAST", "text legibility inside boxes")
    + (
        "MENTAL MASK: convert the image to grayscale so only lightness remains. Ignore hue identity, "
        "element positions, the gaps between elements, and shape/font/border styling.\n\n"
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
        "are FAINT. Dark text inside a box does not make the box itself stand out. A thinner but dark "
        "arrow is still high contrast.\n\n"
        "A dark background is acceptable when elements separate clearly from it; a light background "
        "with near-white boxes is not. CRITICAL: many unrelated hues are NOT a contrast defect if each "
        "one separates clearly from the background. Do not mention theme coherence, centering, spacing, "
        "alignment, or styling."
    ),
    "alignment": _preamble("alignment", "ALIGNMENT")
    + (
        "MENTAL MASK: shrink every node to a single dot at its CENTER. Ignore node sizes and shapes, "
        "fonts, borders, colors, the size of the gaps, and where the diagram sits on the canvas.\n\n"
        "OBSERVATION TO MAKE: identify the rows and columns the layout intends (nodes placed roughly "
        "side by side or stacked, and nodes joined by a connector that should run horizontally or "
        "vertically). For each row/column, estimate how far the node centers stray from one shared "
        "center line, as a fraction of node height (rows) or width (columns). Note connectors that are "
        "slanted only because their endpoints are slightly offset.\n\n"
        "SCORE BANDS:\n"
        "- 90-100: every intended row/column shares one center line (offsets under ~5% of node size); "
        "connectors between them run straight.\n"
        "- 70-89: one node is slightly off its line (~5-15% of node size).\n"
        "- 40-69: several nodes are visibly off their lines (~15-40% of node size), or several "
        "connectors are slanted by small offsets.\n"
        "- 0-39: no consistent grid; most nodes sit at arbitrary offsets.\n\n"
        "CRITICAL: boxes of different sizes or shapes are perfectly aligned when their CENTERS line up. "
        "Do not deduct for box sizes, shapes, fonts, borders, colors, how large or small the gaps are, "
        "or the diagram's position on the canvas."
    ),
    "style_uniformity": _preamble("style_uniformity", "STYLE UNIFORMITY")
    + (
        "MENTAL MASK: render everything in grayscale (ignore hue entirely) and ignore every element's "
        "position, alignment, spacing, connector direction, and how much canvas is used.\n\n"
        "OBSERVATION TO MAKE: group elements by ROLE (start/end terminals, process steps, decisions, "
        "connectors, title). Within each role, list the visual treatments: box shape and corner radius, "
        "box size, border presence and weight, font family/size/weight/style, label placement, and for "
        "connectors the stroke width and arrowhead shape/size. Name every within-role difference.\n\n"
        "SCORE BANDS:\n"
        "- 90-100: each role has one consistent treatment.\n"
        "- 70-89: one minor within-role variation (for example one box somewhat wider for a longer "
        "label).\n"
        "- 40-69: several same-role elements differ in shape, size, border, or font, or connectors vary "
        "in weight or arrowhead.\n"
        "- 0-39: nearly every element is styled differently; no recognizable visual system.\n\n"
        "CRITICAL: DIFFERENT roles may and should look different (pill terminals vs rectangular steps vs "
        "diamond decisions); that is correct, not inconsistency. Color differences are not yours. Size "
        "differences of up to ~15% driven by label length are fine. Do not consider positions, "
        "alignment, spacing, or canvas usage."
    ),
}


def judge_beauty_v5(
    backend: VLMBackend,
    image: str | Path,
) -> dict[str, BeautyScore]:
    scores = {
        dimension: judge_beauty_dimension(backend, image, dimension, BEAUTY_V5_PROMPTS)
        for dimension in BEAUTY_V5_MODEL_DIMENSIONS
    }
    fill = judge_canvas_fill(image)
    scores["canvas_fill"] = BeautyScore(fill.observation, fill.score, fill.reasoning)
    return scores
