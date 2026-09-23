from __future__ import annotations

import base64
import json
import mimetypes
import struct
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .models import (
    ATOMIC_CRITERIA,
    DETAILS_CRITERION_ID,
    PRESENCE_CRITERION_ID,
    ComponentVerdict,
    Criterion,
    Dimension,
    Issue,
    MetricScore,
    PresencePolicy,
    Severity,
    Verdict,
    component_metrics,
)

# Two component verdicts whose boxes overlap this much are treated as the same drawn element.
SAME_ELEMENT_IOU = 0.7


def _data_url(path: str | Path) -> str:
    file_path = Path(path)
    mime = mimetypes.guess_type(file_path.name)[0] or "image/png"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _criterion_payload(criteria: list[Criterion]) -> list[dict[str, str]]:
    return [
        {
            "id": criterion.id,
            "dimension": criterion.dimension.value,
            "description": criterion.description,
            "severity": criterion.severity.value,
        }
        for criterion in criteria
    ]


INVENTORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "inventory": {
            "type": "object",
            "properties": {
                dimension.value: {"type": "array", "items": {"type": "string"}}
                for dimension in Dimension
            },
            "required": [dimension.value for dimension in Dimension],
            "additionalProperties": False,
        },
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "kind": {"type": "string"},
                },
                "required": ["id", "label", "kind"],
                "additionalProperties": False,
            },
        },
        "connections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["source", "target"],
                "additionalProperties": False,
            },
        },
        "forbidden_placeholders": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["inventory", "components", "connections", "forbidden_placeholders"],
    "additionalProperties": False,
}


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion_id": {"type": "string"},
                    "expected_count": {"type": "integer", "minimum": 0},
                    "detected_issues": {"type": "integer", "minimum": 0},
                    "issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "severity": {
                                    "type": "string",
                                    "enum": [severity.value for severity in Severity],
                                },
                                "element_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["description", "severity", "element_ids"],
                            "additionalProperties": False,
                        },
                    },
                    "notes": {"type": "string"},
                },
                "required": [
                    "criterion_id",
                    "expected_count",
                    "detected_issues",
                    "issues",
                    "notes",
                ],
                "additionalProperties": False,
            },
        },
        # One verdict per inventory component (atomic Presence/Details checks).
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "component_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": [verdict.value for verdict in Verdict],
                    },
                    "visible_text": {"type": "string"},
                    "location": {
                        "type": "object",
                        "properties": {
                            "x0": {"type": "number"},
                            "y0": {"type": "number"},
                            "x1": {"type": "number"},
                            "y1": {"type": "number"},
                        },
                        "required": ["x0", "y0", "x1", "y1"],
                        "additionalProperties": False,
                    },
                    "evidence": {"type": "string"},
                },
                "required": ["component_id", "verdict", "visible_text", "location", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["metrics", "components"],
    "additionalProperties": False,
}


def _png_size(path: str | Path) -> tuple[int, int] | None:
    """Read width and height from a PNG header without extra dependencies."""
    try:
        with open(path, "rb") as handle:
            header = handle.read(24)
    except OSError:
        return None
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return (width, height) if width and height else None


def _component_checklist(inventory: dict[str, Any]) -> list[dict[str, str]]:
    """Checklist items for the atomic component questions.

    Each item carries its connection context so that repeated labels (two "Add & Norm"
    components) can be told apart by what they receive from and feed into.
    """
    components = inventory.get("components", [])
    labels = {component["id"]: component.get("label", "") for component in components}
    label_counts: dict[str, int] = {}
    for component in components:
        key = component.get("label", "").strip().casefold()
        label_counts[key] = label_counts.get(key, 0) + 1
    connections = inventory.get("connections", [])

    def name(component_id: str) -> str:
        # Repeated labels are shown with their id so the context stays unambiguous.
        label = labels.get(component_id, component_id)
        return f"{label} [{component_id}]" if label_counts.get(label.strip().casefold(), 0) > 1 else label

    items = []
    for component in components:
        component_id = component["id"]
        sources = [name(edge["source"]) for edge in connections if edge.get("target") == component_id]
        targets = [name(edge["target"]) for edge in connections if edge.get("source") == component_id]
        context = []
        if sources:
            context.append("receives from " + ", ".join(sources))
        if targets:
            context.append("feeds " + ", ".join(targets))
        repeats = label_counts.get(component.get("label", "").strip().casefold(), 0)
        if repeats > 1:
            context.append(
                f"one of {repeats} components with this label; judge it separately and use a different drawn element"
            )
        items.append(
            {
                "component_id": component_id,
                "expected_label": component.get("label", ""),
                "kind": component.get("kind", ""),
                "context": "; ".join(context),
            }
        )
    return items


def _policy_rules(policy: PresencePolicy) -> str:
    abbreviation = (
        "A standard abbreviation of the expected label (for example 'FFN' for 'Feed-Forward Network') counts as present."
        if policy.accept_abbreviations
        else "An abbreviation of the expected label (for example 'FFN' for 'Feed-Forward Network') counts as mislabeled."
    )
    placeholder = (
        "An element in the component's position whose label is a placeholder (such as 'Layer', 'TBD', '???') or is missing counts as mislabeled."
        if policy.placeholder_counts_as_present
        else "An element whose label is a placeholder (such as 'Layer', 'TBD', '???') or is missing counts as absent, even in the right position."
    )
    return abbreviation + " " + placeholder


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _normalized_box(
    location: dict[str, float],
    image_size: tuple[int, int] | None,
) -> tuple[float, float, float, float] | None:
    """Return a valid (x0, y0, x1, y1) box in [0, 1], or None if the location is unusable.

    Pixel coordinates are converted when the image size is known, so a judge that
    ignores the 0-1 instruction does not silently lose all of its evidence.
    """
    x0, y0, x1, y1 = (float(location[key]) for key in ("x0", "y0", "x1", "y1"))
    if max(x0, y0, x1, y1) > 1.5:
        if image_size is None:
            return None
        width, height = image_size
        x0, x1 = x0 / width, x1 / width
        y0, y1 = y0 / height, y1 / height
    x0, y0, x1, y1 = (min(1.0, max(0.0, value)) for value in (x0, y0, x1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))


def parse_component_verdicts(
    items: list[dict[str, Any]],
    component_ids: list[str],
    judge: str,
    image_size: tuple[int, int] | None = None,
) -> list[ComponentVerdict]:
    """Validate a judge's per-component answers and return them in inventory order.

    Rejected outright: unknown IDs, duplicate IDs, and missing IDs, because the scoring
    denominator must be exactly the frozen inventory.
    Downgraded to ABSENT (with a note), to guard against "yes" answers without evidence:
    - PRESENT without quoted visible text or without a usable location;
    - MISLABELED without a usable location;
    - a component that claims the same drawn element as an earlier component.
    """
    expected = set(component_ids)
    seen: dict[str, dict[str, Any]] = {}
    for item in items:
        component_id = item["component_id"]
        if component_id not in expected:
            raise RuntimeError(f"{judge} returned an unknown component id: {component_id}")
        if component_id in seen:
            raise RuntimeError(f"{judge} returned component id twice: {component_id}")
        seen[component_id] = item
    missing = [component_id for component_id in component_ids if component_id not in seen]
    if missing:
        raise RuntimeError(f"{judge} omitted component ids: {missing}")

    verdicts: list[ComponentVerdict] = []
    for component_id in component_ids:
        item = seen[component_id]
        verdict = Verdict(item["verdict"])
        visible_text = item.get("visible_text", "").strip()
        bounds = _normalized_box(item["location"], image_size)
        notes = ""
        if verdict == Verdict.PRESENT and (not visible_text or bounds is None):
            verdict, notes = Verdict.ABSENT, "downgraded from present: no quoted text or usable location"
        elif verdict == Verdict.MISLABELED and bounds is None:
            verdict, notes = Verdict.ABSENT, "downgraded from mislabeled: no usable location"
        if verdict in {Verdict.PRESENT, Verdict.MISLABELED} and bounds is not None:
            for earlier in verdicts:
                if (
                    earlier.verdict in {Verdict.PRESENT, Verdict.MISLABELED}
                    and earlier.bounds is not None
                    and _iou(earlier.bounds, bounds) >= SAME_ELEMENT_IOU
                ):
                    verdict = Verdict.ABSENT
                    notes = f"downgraded: claims the same drawn element as {earlier.component_id}"
                    break
        verdicts.append(
            ComponentVerdict(
                component_id=component_id,
                verdict=verdict,
                judge=judge,
                visible_text=visible_text,
                evidence=item.get("evidence", ""),
                bounds=bounds,
                notes=notes,
            )
        )
    return verdicts


class VLMBackend(ABC):
    name: str

    @abstractmethod
    def structured(self, prompt: str, images: list[str | Path], schema: dict[str, Any], schema_name: str) -> dict[str, Any]:
        raise NotImplementedError


class OpenAIBackend(VLMBackend):
    def __init__(self, model: str) -> None:
        from openai import OpenAI

        self.model = model
        self.name = model
        self.client = OpenAI()

    def structured(self, prompt: str, images: list[str | Path], schema: dict[str, Any], schema_name: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content.extend(
            {"type": "input_image", "image_url": _data_url(image), "detail": "high"}
            for image in images
        )
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
                "verbosity": "low",
            },
        )
        return json.loads(response.output_text)


class ExpectationExtractor:
    def __init__(self, backend: VLMBackend) -> None:
        self.backend = backend

    def extract(self, description: str, criteria: list[Criterion]) -> dict[str, Any]:
        prompt = (
            "You are stage 1 of a description-to-diagram alignment evaluator. Read only the written diagram "
            "description below. There is no candidate image in this stage. Infer a frozen inventory of concrete, "
            "countable expectations for all five dimensions. Presence must enumerate every expected component, "
            "group, input, output, and repeated block, and nothing else: connections belong under Connectivity "
            "only. Details must enumerate exact meaningful labels and content "
            "and must explicitly reject missing or placeholder labels. Connectivity must list each intended "
            "directed source-to-target connection. Layout must describe sensible ordering, containment, overlap, "
            "and canvas expectations. Legibility must describe readable text expectations. Do not score. Do not "
            "add decorative requirements not implied by the description. Also return an ordered components list. "
            "Give each component a unique snake_case id, its exact expected visible label, and a short kind. Preserve "
            "repeated components as separate entries with separate ids. Return connections using only those component "
            "ids, directed from source to target. Return explicit forbidden placeholder strings. Use short, atomic "
            "strings.\n\nShared rubric:\n"
            + json.dumps(_criterion_payload(criteria), ensure_ascii=False)
            + "\n\nDiagram description:\n"
            + description
        )
        data = self.backend.structured(prompt, [], INVENTORY_SCHEMA, "diagram_expectation_inventory")
        data["inventory"] = {
            dimension.value: list(data["inventory"][dimension.value]) for dimension in Dimension
        }
        return data


class VLMJudge:
    def __init__(self, backend: VLMBackend, policy: PresencePolicy | None = None) -> None:
        self.backend = backend
        self.name = backend.name
        self.policy = policy or PresencePolicy()

    def evaluate(
        self,
        description: str,
        candidate_image: str | Path,
        criteria: list[Criterion],
        inventory: dict[str, Any],
    ) -> list[MetricScore]:
        # Presence and Details come from one verdict per inventory component; every other
        # criterion keeps the count-based answer.
        counted = [criterion for criterion in criteria if criterion.id not in ATOMIC_CRITERIA]
        checklist = _component_checklist(inventory)
        prompt = (
            "You are stage 2 of a description-to-diagram alignment evaluator. The attached image is the only "
            "CANDIDATE. Evaluate it against the written description and the frozen stage-1 inventory. Do not "
            "invent a different target. Evidence must refer to visible candidate elements.\n\n"
            "PART A - component checklist (field 'components'). Return exactly one entry for every component_id "
            "below, in any order, and no other ids. Answer each item on its own:\n"
            "- present: a visible element plays this component's role and its label matches the expected label "
            "(ignore case, punctuation, and line breaks).\n"
            "- mislabeled: a visible element plays this role (recognizable from its label, position, grouping, or "
            "arrows), but its label is abbreviated, incomplete, different, missing, or a placeholder.\n"
            "- absent: no visible element plays this role.\n"
            "- uncertain: the image does not allow a decision.\n"
            + _policy_rules(self.policy)
            + "\nRules: quote the element's visible text exactly as drawn in visible_text ('' if it has none). Give "
            "location as the element's bounding box in normalized image coordinates from 0 to 1 (x0, y0 top-left; "
            "x1, y1 bottom-right); use all zeros when absent. Each drawn element may be used for at most one "
            "component_id. Judge only whether the component itself is drawn and labeled; arrows and connections "
            "are judged in Part B, not here.\n\n"
            "PART B - counted criteria (field 'metrics'). For every criterion listed here, expected_count is the "
            "number of relevant countable opportunities and detected_issues is the number that fail. If no "
            "opportunity exists, return both counts as zero. Count each failing opportunity once. Layout must "
            "penalize overlap and off-canvas content. Connectivity must verify visible source and target "
            "endpoints, not merely arrow count.\n\n"
            "Component checklist (Part A):\n"
            + json.dumps(checklist, ensure_ascii=False)
            + "\n\nCounted criteria (Part B):\n"
            + json.dumps(_criterion_payload(counted), ensure_ascii=False)
            + "\n\nDiagram description:\n"
            + description
            + "\n\nFrozen description-derived inventory:\n"
            + json.dumps(inventory, ensure_ascii=False)
        )
        data = self.backend.structured(
            prompt,
            [candidate_image],
            JUDGE_SCHEMA,
            "diagram_correctness_judgment",
        )
        criteria_by_id = {criterion.id: criterion for criterion in counted}
        returned_ids: set[str] = set()
        results: list[MetricScore] = []
        for item in data["metrics"]:
            criterion_id = item["criterion_id"]
            if criterion_id not in criteria_by_id or criterion_id in returned_ids:
                continue
            returned_ids.add(criterion_id)
            criterion = criteria_by_id[criterion_id]
            results.append(
                MetricScore(
                    criterion_id=criterion_id,
                    dimension=criterion.dimension,
                    judge=self.name,
                    expected_count=item["expected_count"],
                    detected_issues=item["detected_issues"],
                    issues=[
                        Issue(
                            description=issue["description"],
                            severity=Severity(issue["severity"]),
                            element_ids=list(issue["element_ids"]),
                        )
                        for issue in item["issues"]
                    ],
                    notes=item["notes"],
                )
            )
        missing_ids = set(criteria_by_id) - returned_ids
        if missing_ids:
            raise RuntimeError(f"{self.name} omitted rubric criteria: {sorted(missing_ids)}")

        verdicts = parse_component_verdicts(
            data["components"],
            [item["component_id"] for item in checklist],
            self.name,
            _png_size(candidate_image),
        )
        all_criteria = {criterion.id: criterion for criterion in criteria}
        results.extend(
            component_metrics(
                verdicts,
                self.name,
                presence=all_criteria.get(PRESENCE_CRITERION_ID),
                details=all_criteria.get(DETAILS_CRITERION_ID),
            )
        )
        return results
