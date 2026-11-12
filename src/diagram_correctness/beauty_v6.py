"""Beauty v6: eight independently model-judged dimensions.

Changes from v5:
- Centering compares opposite canvas margins around the whole diagram only.
- Alignment is about deliberate geometry: no near-misses (almost aligned, almost level,
  almost straight), connectors taking the cleanest path in one consistent routing style,
  and consistent endpoints. No layout type (grid, ring, hub, tree) is preferred.
- Resolution is new: crisp rendering with no blur, pixelation, or compression damage.

The current prompts include the five human-alignment v2 edits: canvas-relative
Centering, grouped-symbol Space Use, outline-or-fill Contrast, semantic-role Style
Uniformity, and coherent multi-hue Color Consistency.
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
        "one group, not individual labels or nodes within that group.\n\n"
        "OBSERVATION TO MAKE: identify the actual rectangular canvas, without assuming it is square. "
        "Draw one bounding box around all diagram content, including nodes, arrows, labels, and title. "
        "Report canvas width W_canvas, canvas height H_canvas, and empty margins L, R, T, B in the "
        "same units. Fractions of width/height are acceptable if each axis uses consistent units.\n\n"
        "SCORING RULE (whole-diagram placement only):\n"
        "- Horizontal score = 100 * (1 - abs(L - R) / W_canvas).\n"
        "- Vertical score = 100 * (1 - abs(T - B) / H_canvas).\n"
        "- Final score = round(min(horizontal score, vertical score)). Report both axis scores.\n"
        "- Normalize by the ENTIRE canvas dimension, NOT by L+R or T+B. A tiny difference between "
        "two near-zero margins is not a severe displacement.\n"
        "- Equal opposite margins score 100 regardless of their size; two zero margins also score 100 "
        "on that axis. Evaluate the other axis independently.\n"
        "- Examples: W_canvas=1000, L=0, R=2 gives horizontal score 99.8; "
        "W_canvas=1000, L=0, R=400 gives horizontal score 60. "
        "H_canvas=1000, T=60, B=1 gives vertical score 94.1.\n\n"
        "CRITICAL: do not score label centering inside shapes, equal inter-node gaps, crowding, "
        "node area, or empty canvas amount. Internal changes that preserve the full diagram's "
        "bounding box must not change this score. Colors, styling, blur, and internal alignment "
        "are not yours."
    ),
    "space_use": _preamble("space_use", "SPACE USE")
    + (
        "MENTAL MASK: recolor nodes as crisp uniform mid-gray silhouettes. Hide colors, styling, "
        "connectors, and the canvas boundary. Keep sizes, shapes, positions, and actual clearances.\n\n"
        "OBSERVATION TO MAKE: first distinguish peer nodes from intentionally grouped symbols. "
        "Layered sheets, overlapping collection symbols, and containing regions can deliberately "
        "represent one group; use a legend, labels, or a clear repeated visual convention as evidence. "
        "Do not call every overlap intentional. Treat such a group as one footprint when comparing "
        "its clearance with neighboring peer nodes/groups; otherwise compare individual nodes. "
        "Estimate TYPICAL edge-to-edge gaps relative to the average size of the two neighboring "
        "nodes/groups in the direction separating them (width for horizontal neighbors, height for "
        "vertical neighbors). Name recurring narrow gaps, near-touching nodes, or accidental overlaps. "
        "Report whether any apparent overlap is a deliberate group or an actual crowding defect.\n\n"
        "SCORE BANDS (based ONLY on inter-node/group clearance):\n"
        "- 90-100: comfortable typical gaps, about 40%+ of node/group size; nothing feels squeezed.\n"
        "- 70-89: slightly tight typical gaps, about 25-40% of node/group size.\n"
        "- 40-69: noticeably crowded, with typical gaps about 10-25% of node/group size.\n"
        "- 0-39: crammed, with typical gaps below about 10% or repeated accidental touching/overlap.\n\n"
        "CRITICAL: intentional layering is not itself crowding, but real cramped gaps and accidental "
        "overlaps remain defects. Do not score canvas coverage, group span, outer margins, absolute "
        "node size, or gap equality. Generous gaps are not a defect. Padding the canvas or uniformly "
        "scaling nodes AND gaps must not change this score. Tight gaps remain a defect on a full "
        "canvas. With fewer than two peer nodes/groups there is no inter-node crowding; score high."
    ),
    "color_consistency": _preamble(
        "color_consistency", "COLOR CONSISTENCY", "shape, size, font, or border differences between elements"
    )
    + (
        "MENTAL MASK: re-tint every color to the same medium lightness and saturation, keeping hue "
        "identity. Ignore positions, gaps, shapes, fonts, orientation, and sharpness.\n\n"
        "OBSERVATION TO MAKE: list chromatic hue families and map them to semantic roles or sections. "
        "White, black, and gray are neutrals; shades of one hue are one family. Use labels, function, "
        "or a legend to establish categories. Check whether equivalent elements share a stable "
        "color treatment and whether different colors communicate distinct categories coherently.\n\n"
        "SCORE BANDS:\n"
        "- 90-100: a coherent theme with stable color-to-role/section mapping, including monochrome "
        "and palettes with more than three hue families.\n"
        "- 70-89: coherent overall, with one minor unexplained role-color inconsistency.\n"
        "- 40-69: several unexplained color changes among equivalent elements or competing palettes "
        "without a coherent role/section mapping.\n"
        "- 0-39: arbitrary rainbow coloring with no stable mapping or recognizable theme.\n\n"
        "CRITICAL: hue count alone is not inconsistency. A multi-section scientific illustration "
        "can coherently use many colors for distinct entities and relationship types. Do not invent "
        "categories merely to excuse random colors. Pale, washed-out, or hard-to-see colors are "
        "not inconsistent when their role mapping is stable. Visibility, contrast, centering, "
        "spacing, alignment, sharpness, and non-color styling are not yours."
    ),
    "contrast": _preamble("contrast", "CONTRAST", "text legibility inside boxes")
    + (
        "MENTAL MASK: convert the image to crisp grayscale so only lightness remains. Ignore hue "
        "identity, placement, orientation, gaps, and styling differences except the visible lightness "
        "separation needed to distinguish a boundary.\n\n"
        "OBSERVATION TO MAKE: assess (a) each box's visible boundary against its background, "
        "(b) boundaries between touching boxes, and (c) connectors against their background. "
        "A boundary can be made visible by its OUTLINE, its FILL, or both. Name the weakest genuinely "
        "visible boundary or connector and describe how readily it can be distinguished.\n\n"
        "SCORE BANDS (driven by the weakest visible boundary/connector):\n"
        "- 90-100: every box boundary and arrow stands out immediately from what it sits against.\n"
        "- 70-89: everything is clearly visible, but one class of boundary/connector is somewhat soft.\n"
        "- 40-69: one class of boundary/connector is faint and takes effort to distinguish.\n"
        "- 0-39: boxes AND arrows are faint, or an element's boundary nearly disappears.\n\n"
        "CALIBRATION: a white or pastel box with a clear dark outline can have excellent contrast "
        "against white. Do not penalize a pale interior when its boundary is already distinct. "
        "A box with both a near-white fill AND a faint outline is weak contrast. Pale gray arrows "
        "on white are faint; thin dark arrows can be high contrast. Dark text alone does not rescue "
        "an otherwise invisible box boundary.\n\n"
        "A dark background is acceptable when boundaries separate clearly from it. Unrelated hues "
        "are not a contrast defect if their boundaries remain distinct. Do not score theme coherence, "
        "centering, spacing, alignment, style uniformity, text legibility, or sharpness."
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
        "MENTAL MASK: render everything crisp, upright, and in grayscale. Ignore positions, "
        "orientation, alignment, spacing, connector direction/attachment, label placement, and canvas use.\n\n"
        "OBSERVATION TO MAKE: group SEMANTICALLY EQUIVALENT elements by role before comparing style. "
        "Identify roles from function, labels, or a legend, not merely from differing styles. "
        "Process boxes, molecular illustrations, containing regions, and section headings are not "
        "one role. Similarly, ordinary flow arrows, alternative pathways, and cross-layer explanatory "
        "links can have distinct roles and deliberately use solid, dashed, or dotted lines. "
        "Do not invent a separate role for every inconsistent element.\n"
        "Within each comparable role, list shape/corner radius, size, border presence/weight, "
        "font family/size/weight, and connector stroke/dash/arrowhead treatment. Name unexplained "
        "within-role differences. Different biological entities can legitimately have different "
        "illustrative silhouettes; that does not make the process-box style inconsistent.\n\n"
        "SCORE BANDS:\n"
        "- 90-100: each semantically equivalent role has one consistent visual treatment.\n"
        "- 70-89: one minor unexplained within-role variation.\n"
        "- 40-69: several equivalent elements have inconsistent shapes, sizes, borders, fonts, or "
        "connector treatments.\n"
        "- 0-39: nearly every equivalent element is styled differently; no recognizable system.\n\n"
        "CRITICAL: deliberate style differences BETWEEN roles are not defects, but unexplained "
        "differences WITHIN a role still count. Size differences up to about 15% driven by label "
        "length are fine. Color, rotation, alignment, spacing, label placement, canvas use, and "
        "rendering sharpness are not yours."
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
