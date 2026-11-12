"""Beauty v6: eight independently model-judged dimensions.

Changes from v5:
- Centering compares opposite canvas margins around the whole diagram only.
- Alignment is about deliberate geometry: no near-misses (almost aligned, almost level,
  almost straight), connectors taking the cleanest path in one consistent routing style,
  and consistent endpoints. No layout type (grid, ring, hub, tree) is preferred.
- Resolution is new: crisp rendering with no blur, pixelation, or compression damage.
"""
from __future__ import annotations

from pathlib import Path

from .beauty import BeautyScore, judge_beauty_dimension
from .vlm import VLMBackend


BEAUTY_V6_MODEL_DIMENSIONS = (
    "centering",
    "space_use",
    "color_consistency",
    "contrast",
    "alignment",
    "style_uniformity",
    "resolution",
    "canvas_fill",
)
BEAUTY_V6_DIMENSIONS = BEAUTY_V6_MODEL_DIMENSIONS

_DIMENSION_OWNERS = """\
  CENTERING         - whole-diagram placement: equal left/right and top/bottom canvas margins.
  SPACE USE         - crowding and edge-to-edge clearance between neighboring nodes relative to node size.
  COLOR CONSISTENCY - whether colors form one coherent theme with stable color-to-role mapping.
  CONTRAST          - whether boxes and arrows visibly separate from the background and from touching boxes.
  ALIGNMENT         - geometry is deliberate: nothing almost-aligned or slightly tilted; connectors take
                      the cleanest path in one consistent routing style and attach consistently.
  STYLE UNIFORMITY  - whether elements of the same role share one shape, size, border, font, and arrow
                      stroke/arrowhead treatment.
  RESOLUTION        - whether the image renders crisply: no blur, pixelation, jagged stair-steps, or
                      compression artifacts.
  CANVAS FILL       - combined node silhouette area divided by the entire canvas area,
                      excluding gaps and thin connectors.
"""

_SIBLINGS = {
    "centering": "whether opposite canvas margins around the whole diagram are equal (Centering)",
    "space_use": "crowding and edge-to-edge clearance between neighboring nodes relative to node size (Space Use)",
    "color_consistency": "which hues are used or whether they form a coherent theme (Color Consistency)",
    "contrast": "how light, pale, or hard to see boxes and arrows are against the background (Contrast)",
    "alignment": "whether nodes, shapes, labels, or connectors are almost-but-not-quite aligned or upright, "
    "connectors take unclean paths, or routing and endpoint styles are inconsistent (Alignment)",
    "style_uniformity": "whether same-role elements differ in shape, size, border, font, or arrow stroke/arrowhead "
    "(Style Uniformity)",
    "resolution": "blur, pixelation, jagged edges, or compression artifacts (Resolution)",
    "canvas_fill": "combined node silhouette area relative to the entire canvas area (Canvas Fill)",
}


def _not_yours(dimension: str, extra: str = "") -> str:
    parts = [text for key, text in _SIBLINGS.items() if key != dimension]
    return "; ".join(parts) + ("; " + extra if extra else "") + "."


def _preamble(dimension: str, label: str, extra: str = "") -> str:
    return (
        "You are ONE of eight independent judges scoring a flowchart image. "
        "Each judge scores exactly one visual dimension and never sees the other judges' "
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
        "MENTAL MASK: treat any uniform background, light OR dark, as empty canvas and make foreground "
        "content crisp and equally visible without moving or resizing it. Judge the full diagram as "
        "one group, not the positions of individual labels or nodes within that group.\n\n"
        "OBSERVATION TO MAKE: draw one bounding box around the full diagram, including nodes, arrows, "
        "labels, and title. Estimate the empty margins L (left), R (right), T (top), and B (bottom), "
        "using the same units for each opposing pair. These are the distances the entire group could "
        "move in each direction before reaching the canvas edge. Report all four margins.\n\n"
        "SCORING RULE (whole-diagram placement only):\n"
        "- Horizontal score H = 100 if L + R = 0; otherwise H = 100 * (1 - abs(L - R) / (L + R)).\n"
        "- Vertical score V = 100 if T + B = 0; otherwise V = 100 * (1 - abs(T - B) / (T + B)).\n"
        "- Final score = round(min(H, V)). Report both axis scores before the final score.\n"
        "- Equal opposite margins score 100 regardless of how large those margins are.\n"
        "- No room to move horizontally means H = 100; no room to move vertically means V = 100. "
        "Still evaluate the other axis independently.\n"
        "- Examples: L=R=200 and T=B=100 -> 100; L=0,R=0,T=50,B=50 -> 100; "
        "L=20,R=80,T=50,B=50 -> 40; L=0,R=0,T=0,B=100 -> 0.\n\n"
        "CRITICAL: do not score label centering inside shapes, equal gaps between nodes, crowding, "
        "node area, or the amount of empty canvas. If internal labels or gaps change without changing "
        "the full diagram's bounding box, this score must not change. Colors, styling, blur, and "
        "internal alignment are not yours."
    ),
    "space_use": _preamble("space_use", "SPACE USE")
    + (
        "MENTAL MASK: recolor nodes as crisp uniform mid-gray silhouettes. Hide labels, connectors, "
        "colors, styling, and the canvas boundary. Keep every node's size, shape, and position unchanged "
        "so the actual clearances are preserved.\n\n"
        "OBSERVATION TO MAKE: estimate the TYPICAL edge-to-edge gap between neighboring nodes relative "
        "to their size. For each neighboring pair, compare the gap with the average size of the two "
        "nodes measured in the direction separating them (width for horizontal neighbors, height for "
        "vertical neighbors). Note whether narrow gaps recur or nodes nearly touch. Judge local "
        "clearance in any layout, not only rows or columns.\n\n"
        "SCORE BANDS (based ONLY on inter-node clearance):\n"
        "- 90-100: comfortable typical gaps, about 40%+ of node size; nothing feels squeezed.\n"
        "- 70-89: slightly tight typical gaps, about 25-40% of node size.\n"
        "- 40-69: noticeably crowded, with typical gaps about 10-25% of node size.\n"
        "- 0-39: crammed, with typical gaps below about 10% of node size or repeated nearly touching "
        "nodes.\n\n"
        "CRITICAL: only crowding and inter-node clearance are yours. Do not score canvas coverage, "
        "group span, outer margins, or absolute node size. Generous gaps are not a defect. Unequal "
        "gaps are not a defect here unless they create crowding. Adding empty padding or uniformly "
        "scaling the nodes AND their gaps must not change this score. Tight gaps remain a defect even "
        "when the canvas is full. With fewer than two nodes there is no inter-node crowding; score high."
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
        "PRINCIPLE: the geometry must be DELIBERATE. Whatever structure the diagram uses, everything that is "
        "meant to line up, stand upright, or follow that structure does so exactly, and connectors take the "
        "cleanest path for that structure in one consistent style. Near-misses (almost aligned, almost "
        "level, almost straight) are defects. Grids, rings, hubs, and trees are all valid.\n\n"
        "OBSERVATION TO MAKE (four checks):\n"
        "(A) ORIENTATION: is every shape upright and every label horizontal? Upright means: rectangles, "
        "rounded rectangles, and pills have level top/bottom edges; ellipses have horizontal/vertical "
        "axes; polygons rest on a horizontal edge or have a vertical axis of symmetry; a diamond (decision) "
        "has vertical and horizontal diagonals. Circles are exempt. A tilted whole diagram counts. Name "
        "each tilted shape or label and its approximate angle.\n"
        "(B) STRUCTURE: name the layout's structure. The DEFAULT is a GRID (nodes in rows and columns). "
        "Other valid structures are a RING (nodes evenly around a circle), a HUB (a central node with "
        "spokes), a TREE (parents above children in level rows), or an EVEN STAIRCASE (identical steps). "
        "Choose a non-grid structure only when the nodes clearly and regularly form it. Nodes that sit "
        "within about one node height of a shared row (or one node width of a shared column) by uneven "
        "amounts are a MISALIGNED GRID, not a free-form design; when unsure, treat the layout as a grid. "
        "Then measure: for a grid, how far node centers in each row/column stray from one shared line, as "
        "a fraction of node size; for a ring, hub, or tree, whether nodes sit evenly on the circle, around "
        "the hub, or level within each tree row.\n"
        "(C) CONNECTORS: judge routing against the structure. GRID: connectors between nodes that share a "
        "row or column run straight and exactly horizontal/vertical between facing edges; connectors "
        "between nodes in different rows and columns use horizontal/vertical (elbow) segments. RING, HUB, "
        "TREE, or STAIRCASE: straight lines toward node centers, or smooth arcs that follow a ring, are "
        "correct; do not require horizontal/vertical routing. In ANY structure, deduct for: a diagonal "
        "connector between two nodes that share (or nearly share) a row or column; a connector aimed at a "
        "shape's corner instead of its center or an edge midpoint; and routing styles mixed without reason. "
        "A curved loop-back or feedback arrow returning to an earlier step is a valid reason and is not a "
        "defect.\n"
        "(D) ENDPOINTS: compare the connectors WITH EACH OTHER. For each connector, say whether there is "
        "clearly visible white space (roughly 8px or more) between its line and the shape it leaves, and "
        "between its arrowhead and the shape it enters. A gap of only a few pixels counts as touching. "
        "Compare like with like (tails with tails, arrowheads with arrowheads): a tail that touches while "
        "its arrowhead tip stops a few pixels short is normal and consistent. Inconsistency means some "
        "connectors clearly float with visible white space at their ends while others run right up to "
        "their shapes.\n\n"
        "SCORE BANDS (score = the WORST of A, B, C, D):\n"
        "- 90-100: every shape and label upright; nodes exactly on their structure (offsets < ~5% of node "
        "size); every connector routed as its structure requires; all connectors treat their ends the "
        "same way.\n"
        "- 70-89: one minor deviation: one element tilted ~1-3 degrees, one node ~5-15% off its line, one "
        "connector slightly slanted or misdirected, or one connector visibly detached while the rest "
        "touch (or vice versa).\n"
        "- 40-69: several deviations: multiple shapes or labels tilted ~3-15 degrees, several nodes "
        "~15-100% of a node size off their lines in uneven amounts, several connectors routed against the "
        "structure (diagonal between nodes that share a row, or corner-to-corner), or several connectors "
        "visibly detached while others touch.\n"
        "- 0-39: no deliberate structure: most shapes tilted, nodes at arbitrary offsets, or most "
        "connectors haphazard.\n\n"
        "CRITICAL: never penalize a ring, hub, tree, or loop-back for not being a grid; their diagonal or "
        "curved connectors are correct. A near-grid with unevenly offset nodes is a misaligned grid. "
        "Connectors that ALL stop the same small distance short of their shapes are a valid style. "
        "Different shapes and sizes are aligned when their centers line up. Do not deduct for unequal or "
        "tight gaps, label position inside a shape, canvas position, styling, colors, faintness, or blur."
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
    "canvas_fill": _preamble("canvas_fill", "CANVAS FILL")
    + (
        "MENTAL MASK: treat any uniform background, light OR dark, as empty canvas. Mentally recolor "
        "each node as a solid neutral silhouette, including the full interior of outline-only shapes. "
        "Keep the original canvas dimensions and node sizes. Ignore color, contrast, blur, styling, "
        "placement, alignment, label positioning, and gaps between nodes.\n\n"
        "OBSERVATION TO MAKE: estimate the combined area occupied by diagram node silhouettes as a "
        "percentage of the ENTIRE image canvas. Sum the individual shape areas, counting overlapping "
        "areas only once. Exclude thin connectors, arrows, text strokes, decoration, and background "
        "texture. Empty space between nodes or enclosed by a loop of arrows does NOT count. Do not "
        "use the bounding box of the whole diagram: tiny nodes spread to the edges still have a "
        "small footprint. State your estimated footprint percentage explicitly.\n\n"
        "SCORING RULE (keep these exact thresholds):\n"
        "- Footprint at or below 3%: score 0.\n"
        "- Footprint at or above 6%: score 100.\n"
        "- Between 3% and 6%: score = round(100 * (footprint_percent - 3) / 3).\n"
        "- Examples: 3% -> 0; 3.75% -> 25; 4.5% -> 50; 5.25% -> 75; 6% -> 100.\n\n"
        "CRITICAL: penalize ONLY a too-small element footprint. Dense or oversized diagrams are a "
        "design choice and still score 100 here if their footprint is at least 6%. A pale or "
        "outline-only node counts by its full silhouette, not by its visible ink. Do not deduct "
        "for off-centering, tight gaps, unreadable labels, connector defects, or poor rendering."
    ),
}


def judge_beauty_v6(
    backend: VLMBackend,
    image: str | Path,
) -> dict[str, BeautyScore]:
    return {
        dimension: judge_beauty_dimension(backend, image, dimension, BEAUTY_V6_PROMPTS)
        for dimension in BEAUTY_V6_MODEL_DIMENSIONS
    }
