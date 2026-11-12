"""Freeze the five approved experimental v2 edits without changing the evaluator."""
from __future__ import annotations

import argparse
from pathlib import Path

from run import now, read_json, refresh, write_json


BODIES = {
    "centering": (
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
    "space_use": (
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
    "contrast": (
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
    "style_uniformity": (
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
    "color_consistency": (
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
}

EDITS = [
    "Centering: normalize opposite-margin imbalance by canvas width/height, not available margin.",
    "Space Use: distinguish intentional layered/grouped symbols from accidental peer-node crowding.",
    "Contrast: judge visible boundaries using outline and/or fill, not fill alone.",
    "Style Uniformity: compare semantically equivalent elements and connector roles.",
    "Color Consistency: allow more than three hues when role/section mapping is coherent.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    parent = read_json(args.run / "versions" / "current_001.json")
    review = read_json(args.run / "review.json")
    if not review["approval"]:
        raise ValueError("Approve the image review before creating v2.")
    target = args.run / "versions" / "v2.json"
    if target.exists():
        raise FileExistsError(target)
    prompts = dict(parent["prompts"])
    for dimension, body in BODIES.items():
        preamble, separator, _ = prompts[dimension].partition("MENTAL MASK:")
        if not separator:
            raise ValueError(f"No dimension-instruction boundary in {dimension}")
        prompts[dimension] = preamble + body
    write_json(target, {
        "id": "v2", "parent_version": "current_001", "experimental": True,
        "description": "Approved five-edit experimental v2; source evaluator prompts unchanged.",
        "created_at": now(), "prompts": prompts,
        "response_instruction": parent["response_instruction"],
        "edits": EDITS,
        "training_evidence_ids": [
            "HS001", "HS005", "HS008", "HS023", "HS056", "HS067", "HS076", "HS085",
        ],
        "approval": "User approved experimental v2 and dashboard on 2026-11-12.",
    })
    refresh(args.run)
    print(f"Frozen experimental v2 at {target.resolve()}", flush=True)


if __name__ == "__main__":
    main()
