"""Beauty v6: seven model-judged dimensions plus the deterministic Canvas Fill check.

Changes from v5:
- Centering covers three levels: the whole diagram on the canvas, each label inside its
  shape, and even gaps between neighboring nodes.
- Alignment covers everything being axis-aligned: upright shapes and labels, nodes on
  shared row/column lines, and connectors that run horizontally/vertically and attach
  consistently.
- Resolution is new: crisp rendering with no blur, pixelation, or compression damage.
"""
from __future__ import annotations

from pathlib import Path

from .beauty import BeautyScore, judge_beauty_dimension
from .canvas_fill import judge_canvas_fill
from .vlm import VLMBackend


BEAUTY_V6_MODEL_DIMENSIONS = (
    "centering",
    "space_use",
    "color_consistency",
    "contrast",
    "alignment",
    "style_uniformity",
    "resolution",
)
BEAUTY_V6_DIMENSIONS = (*BEAUTY_V6_MODEL_DIMENSIONS, "canvas_fill")

_DIMENSION_OWNERS = """\
  CENTERING         - everything is centered: the whole diagram on the canvas, each label inside its
                      shape, and each node evenly spaced between its row/column neighbors (equal gaps).
  SPACE USE         - whether nodes are packed too tightly overall while usable canvas space is left empty.
  COLOR CONSISTENCY - whether colors form one coherent theme with stable color-to-role mapping.
  CONTRAST          - whether boxes and arrows visibly separate from the background and from touching boxes.
  ALIGNMENT         - everything is axis-aligned: shapes and labels upright, nodes on shared row/column
                      lines, connectors running horizontally/vertically and attaching consistently.
  STYLE UNIFORMITY  - whether elements of the same role share one shape, size, border, font, and arrow
                      stroke/arrowhead treatment.
  RESOLUTION        - whether the image renders crisply: no blur, pixelation, jagged stair-steps, or
                      compression artifacts.
  CANVAS FILL       - (measured by a script, not a judge) whether elements are too small for the canvas,
                      leaving most of it empty.
"""

_SIBLINGS = {
    "centering": "whether the whole diagram, the labels inside their shapes, or the gaps between neighbors are "
    "centered and even (Centering)",
    "space_use": "whether nodes are packed too tightly overall while canvas space is empty (Space Use)",
    "color_consistency": "which hues are used or whether they form a coherent theme (Color Consistency)",
    "contrast": "how light, pale, or hard to see boxes and arrows are against the background (Contrast)",
    "alignment": "whether shapes, labels, or connectors are tilted or diagonal, nodes sit off their rows/columns, "
    "or connectors attach inconsistently (Alignment)",
    "style_uniformity": "whether same-role elements differ in shape, size, border, font, or arrow stroke/arrowhead "
    "(Style Uniformity)",
    "resolution": "blur, pixelation, jagged edges, or compression artifacts (Resolution)",
    "canvas_fill": "how much of the canvas is empty or how small the elements are relative to it (Canvas Fill)",
}


def _not_yours(dimension: str, extra: str = "") -> str:
    parts = [text for key, text in _SIBLINGS.items() if key != dimension]
    return "; ".join(parts) + ("; " + extra if extra else "") + "."


def _preamble(dimension: str, label: str, extra: str = "") -> str:
    return (
        "You are ONE of seven independent judges scoring a flowchart image; an eighth dimension is measured "
        "by a script. Each judge scores exactly one visual dimension and never sees the other judges' "
        "output. Content and structure are judged elsewhere; this is purely visual.\n\n"
        f"The eight dimensions and who owns them:\n{_DIMENSION_OWNERS}\n"
        f"YOU OWN: {label}. You score nothing else.\n\n"
        "ISOLATION RULES (follow strictly):\n"
        "- Assume every other dimension is already perfect. They are someone else's job.\n"
        f"- NOT YOURS: {_not_yours(dimension, extra)}\n"
        "- If the only flaws you can see belong to another dimension, you MUST still score HIGH. A defect "
        "owned by another dimension is not evidence against yours.\n"
        "- Do not form an overall 'this diagram looks bad/good' impression and transfer it to your "
        "score. Judge the observable below and nothing else.\n"
        "- Apply the stated MENTAL MASK before observing. Score the counterfactual image after sibling "
        "signals are hidden, not the unmasked overall impression.\n"
        "- First state the OBSERVATION your dimension asks for, then derive the score strictly from "
        "that observation using the bands given.\n\n"
    )


BEAUTY_V6_PROMPTS = {
    "centering": _preamble("centering", "CENTERING")
    + (
        "MENTAL MASK: recolor everything in uniform gray, render it crisp, upright, and uniformly styled, "
        "and treat any uniform background fill, light OR dark, as empty canvas. Keep exactly three things: "
        "where the whole group sits on the canvas, where each label sits inside its shape, and how large "
        "each gap between neighboring nodes is.\n\n"
        "OBSERVATION TO MAKE (three checks):\n"
        "(A) WHOLE DIAGRAM: draw one bounding box around all foreground content (title included). Estimate "
        "the left/right and top/bottom margins and the larger-to-smaller ratio on each axis.\n"
        "(B) LABELS: for each node, is its label centered horizontally and vertically inside its shape? "
        "Estimate the largest label offset as a fraction of the shape's width/height.\n"
        "(C) EVEN SPACING: for each row and column of connected neighboring nodes, list the gaps between "
        "consecutive neighbors and give the largest-to-smallest gap ratio within that row/column.\n\n"
        "SCORE BANDS (score = the WORST of A, B, C):\n"
        "- 90-100: A margins near-symmetric (ratio < ~1.5); B every label centered (offset < ~5%); C gaps "
        "even (ratio < ~1.25).\n"
        "- 70-89: one mild issue: A ratio ~1.5-2.5, or B one label offset ~5-15%, or C ratio ~1.25-1.6.\n"
        "- 40-69: A clearly shifted (ratio ~2.5-5), or B labels visibly off-center (~15-35%), or C clearly "
        "uneven gaps (ratio ~1.6-3).\n"
        "- 0-39: A pushed into a corner or against an edge (> ~5x), or B labels jammed against a side or "
        "corner of their shapes, or C wildly uneven gaps (> ~3x).\n\n"
        "CRITICAL: a compact diagram with large but EQUAL margins is centered. Uniformly tight or uniformly "
        "loose gaps are EVEN and score high here (tightness is Space Use's). Do not deduct for empty canvas, "
        "tilted or misaligned nodes, diagonal arrows, styling, colors, faint elements, or blur."
    ),
    "space_use": _preamble("space_use", "SPACE USE")
    + (
        "MENTAL MASK: mentally slide the whole diagram so it is centered on the canvas, recolor every "
        "element as a crisp uniform mid-gray outline, and snap every node upright onto clean rows/columns. "
        "Keep the node sizes and the typical distance between nodes as drawn.\n\n"
        "OBSERVATION TO MAKE: estimate (a) what fraction of the canvas width and height the node group "
        "spans after re-centering, (b) the TYPICAL gap between neighboring nodes relative to node size, "
        "and (c) whether the nodes feel condensed together while usable canvas space remains empty around "
        "them. Identify the layout's MAIN direction (the axis with more nodes).\n\n"
        "SCORE BANDS (use the WORSE of the span and gap readings):\n"
        "- 90-100: nodes spread along the main direction (roughly 55%+ of the canvas on that axis) with "
        "comfortable typical gaps (about 40%+ of node size); nothing feels squeezed. A layout with only one "
        "or two rows/columns does NOT need to fill the cross axis.\n"
        "- 70-89: slightly compact; main-direction span ~45-55%, or typical gaps ~25-40% of node size.\n"
        "- 40-69: noticeably condensed; main-direction span ~30-45%, or typical gaps ~10-25% of node size, "
        "while plenty of empty canvas remains.\n"
        "- 0-39: crammed; main-direction span under ~30% or nodes nearly touching, while most of the "
        "canvas is empty.\n\n"
        "CRITICAL: only TOO-TIGHT overall packing is yours. Whether gaps are EQUAL to each other is "
        "Centering's. Generous or very large gaps, tiny elements, and a mostly empty canvas are NOT Space "
        "Use defects. An off-center or misaligned diagram is not condensed. If the canvas is genuinely "
        "full, tight gaps are not a defect."
    ),
    "color_consistency": _preamble(
        "color_consistency", "COLOR CONSISTENCY", "shape, size, font, or border differences between elements"
    )
    + (
        "MENTAL MASK: re-tint every color to the same medium lightness and saturation, keeping only "
        "its hue identity. Ignore element positions, gaps, shapes, fonts, orientation, and sharpness.\n\n"
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
        "mention visibility, contrast, centering, spacing, alignment, sharpness, or non-color styling."
    ),
    "contrast": _preamble("contrast", "CONTRAST", "text legibility inside boxes")
    + (
        "MENTAL MASK: convert the image to crisp grayscale so only lightness remains. Ignore hue identity, "
        "element positions and orientation, the gaps between elements, and shape/font/border styling.\n\n"
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
        "alignment, styling, or sharpness."
    ),
    "alignment": _preamble("alignment", "ALIGNMENT")
    + (
        "MENTAL MASK: recolor everything in crisp uniform gray and give every shape the same size. Ignore "
        "fonts, borders, colors, how large the gaps are or whether they are equal, where each label sits "
        "inside its shape, and where the diagram sits on the canvas.\n\n"
        "OBSERVATION TO MAKE (four checks):\n"
        "(A) ORIENTATION: is every shape upright and every label horizontal? Upright means: rectangles, "
        "rounded rectangles, and pills have level top/bottom edges; ellipses have horizontal/vertical "
        "axes; polygons rest on a horizontal edge or have a vertical axis of symmetry; a diamond (decision) "
        "has vertical and horizontal diagonals. Circles are exempt. A tilted whole diagram counts. Name "
        "each tilted shape or label and its approximate angle.\n"
        "(B) GRID: identify intended rows and columns. For each, estimate how far node centers stray from "
        "one shared center line, as a fraction of node height (rows) or width (columns).\n"
        "(C) CONNECTORS: where two connected nodes share a row or column, the connector should run "
        "straight horizontally or vertically between facing edges; connectors between nodes in different "
        "rows and columns should use horizontal/vertical (elbow) segments. Note diagonal or slanted "
        "segments.\n"
        "(D) ENDPOINTS: compare the connectors WITH EACH OTHER. For each connector, say whether there is "
        "clearly visible white space (roughly 8px or more) between its line and the shape it leaves, and "
        "between its arrowhead and the shape it enters. A gap of only a few pixels counts as touching. "
        "Compare like with like (tails with tails, arrowheads with arrowheads): a tail that touches while "
        "its arrowhead tip stops a few pixels short is normal and consistent. Inconsistency means some "
        "connectors clearly float with visible white space at their ends while others run right up to "
        "their shapes.\n\n"
        "SCORE BANDS (score = the WORST of A, B, C, D):\n"
        "- 90-100: every shape and label upright; centers share their lines (offsets < ~5% of node size); "
        "every connector horizontal/vertical; all connectors treat their ends the same way.\n"
        "- 70-89: one minor deviation: one element tilted ~1-3 degrees, one node ~5-15% off its line, one "
        "connector slightly slanted, or one connector visibly detached while the rest touch (or vice versa).\n"
        "- 40-69: several deviations: multiple shapes or labels tilted ~3-10 degrees, several nodes "
        "~15-40% off their lines, several diagonal connectors between nodes that could be joined "
        "straight, or several connectors visibly detached while others touch.\n"
        "- 0-39: no consistent orthogonal structure: most shapes tilted, nodes at arbitrary offsets, or "
        "most connectors diagonal or haphazard.\n\n"
        "CRITICAL: connectors that ALL stop the same small distance short of their shapes are a valid "
        "style, not a defect. Different shapes and sizes are aligned when their centers line up. Do not "
        "deduct for unequal or tight gaps, label position inside a shape, canvas position, styling, "
        "colors, faintness, or blur."
    ),
    "style_uniformity": _preamble("style_uniformity", "STYLE UNIFORMITY")
    + (
        "MENTAL MASK: render everything crisp, upright, and in grayscale (ignore hue entirely), and ignore "
        "every element's position, orientation, alignment, spacing, connector direction and attachment, "
        "where labels sit inside their shapes, and how much canvas is used.\n\n"
        "OBSERVATION TO MAKE: group elements by ROLE (start/end terminals, process steps, decisions, "
        "connectors, title). Within each role, list the visual treatments: box shape and corner radius, "
        "box size, border presence and weight, font family/size/weight/style, and for connectors the "
        "stroke width, dash pattern, and arrowhead shape/size. Name every within-role difference.\n\n"
        "SCORE BANDS:\n"
        "- 90-100: each role has one consistent treatment.\n"
        "- 70-89: one minor within-role variation (for example one box somewhat wider for a longer "
        "label).\n"
        "- 40-69: several same-role elements differ in shape, size, border, or font, or connectors vary "
        "in weight, dash, or arrowhead.\n"
        "- 0-39: nearly every element is styled differently; no recognizable visual system.\n\n"
        "CRITICAL: DIFFERENT roles may and should look different (pill terminals vs rectangular steps vs "
        "diamond decisions); that is correct, not inconsistency. Color differences are not yours. Size "
        "differences of up to ~15% driven by label length are fine. Rotation, alignment, spacing, label "
        "position, canvas usage, and sharpness are not yours."
    ),
    "resolution": _preamble("resolution", "RESOLUTION")
    + (
        "MENTAL MASK: ignore what is drawn, where it is, its colors, sizes, orientation, and styling. "
        "Inspect only rendering quality at the displayed size, as if zooming into box edges, curves, "
        "arrowheads, and letters.\n\n"
        "OBSERVATION TO MAKE: describe how edges and text are rendered. Crisp means color transitions "
        "happen within about one pixel with smooth anti-aliasing, and letters have clean strokes. Note any "
        "BLUR (soft halos several pixels wide, fuzzy text), PIXELATION or ALIASING (stair-stepped curves "
        "and diagonals, blocky letters, visible square pixels), or COMPRESSION ARTIFACTS (ringing around "
        "edges, 8x8 blocks, color smearing).\n\n"
        "SCORE BANDS:\n"
        "- 90-100: crisp everywhere; text and edges are clean.\n"
        "- 70-89: slightly soft, or mild stair-stepping on a few curves, while text stays crisp.\n"
        "- 40-69: clearly blurry, or clearly pixelated/blocky, or visible compression artifacts; text is "
        "readable but visibly degraded.\n"
        "- 0-39: severely degraded; text is hard to read because of blur, blockiness, or artifacts.\n\n"
        "CRITICAL: small elements or small text that render cleanly are NOT low resolution. Thin lines, "
        "pale colors, tilted or misaligned shapes, and empty canvas are not yours."
    ),
}


def judge_beauty_v6(
    backend: VLMBackend,
    image: str | Path,
) -> dict[str, BeautyScore]:
    scores = {
        dimension: judge_beauty_dimension(backend, image, dimension, BEAUTY_V6_PROMPTS)
        for dimension in BEAUTY_V6_MODEL_DIMENSIONS
    }
    fill = judge_canvas_fill(image)
    scores["canvas_fill"] = BeautyScore(fill.observation, fill.score, fill.reasoning)
    return scores
