from __future__ import annotations

import base64
import json
import mimetypes
import struct
import threading
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median_low
from typing import Any

from .models import (
    DETAILS_CRITERION_ID,
    METRIC_ATTACHMENT,
    METRIC_CONNECTION_LABELS,
    METRIC_CONNECTIONS,
    METRIC_LABELS,
    METRIC_MINIMUM_FONT,
    METRIC_PRESENCE,
    METRIC_UNEXPECTED,
    PRESENCE_CRITERION_ID,
    ComponentVerdict,
    ConnectionStatus,
    ConnectionVerdict,
    Criterion,
    Dimension,
    Issue,
    MetricScore,
    PresencePolicy,
    Severity,
    Verdict,
    component_metrics,
    connection_metrics,
    drawn_component_ids,
    feedback_edges,
    legibility_from_verdicts,
    typical_items,
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
                    # "required": the description calls for it; "typical": such a diagram usually shows it.
                    "tier": {"type": "string", "enum": ["required", "typical"]},
                },
                "required": ["id", "label", "kind", "tier"],
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
                    # Text the description says the arrow carries, e.g. "YES"; "" if none.
                    "label": {"type": "string"},
                    "tier": {"type": "string", "enum": ["required", "typical"]},
                },
                "required": ["source", "target", "label", "tier"],
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


_LOCATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "x0": {"type": "number"},
        "y0": {"type": "number"},
        "x1": {"type": "number"},
        "y1": {"type": "number"},
    },
    "required": ["x0", "y0", "x1", "y1"],
    "additionalProperties": False,
}

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # Part A: one verdict per inventory component (Presence, Details, Legibility).
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
                    "label_legible": {"type": "boolean"},
                    "location": _LOCATION_SCHEMA,
                    "evidence": {"type": "string"},
                },
                "required": ["component_id", "verdict", "visible_text", "label_legible", "location", "evidence"],
                "additionalProperties": False,
            },
        },
        # Part B: one verdict per inventory connection (Connectivity, connection labels).
        "connections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "connection_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": [status.value for status in ConnectionStatus],
                    },
                    "visible_label": {"type": "string"},
                    "label_legible": {"type": "boolean"},
                    "evidence": {"type": "string"},
                },
                "required": ["connection_id", "verdict", "visible_label", "label_legible", "evidence"],
                "additionalProperties": False,
            },
        },
        # Part C: drawn things the inventory does not call for.
        "extra_components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "visible_text": {"type": "string"},
                    "location": _LOCATION_SCHEMA,
                    "evidence": {"type": "string"},
                },
                "required": ["visible_text", "location", "evidence"],
                "additionalProperties": False,
            },
        },
        "extra_connections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from_text": {"type": "string"},
                    "to_text": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["from_text", "to_text", "evidence"],
                "additionalProperties": False,
            },
        },
        # Part C (continued): other drawn content that fits the diagram; it only enters denominators.
        "other_components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "visible_text": {"type": "string"},
                    "location": _LOCATION_SCHEMA,
                },
                "required": ["visible_text", "location"],
                "additionalProperties": False,
            },
        },
        "other_connections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from_text": {"type": "string"},
                    "to_text": {"type": "string"},
                },
                "required": ["from_text", "to_text"],
                "additionalProperties": False,
            },
        },
        # Part D: failure counts only; the denominators are computed from Parts A-C.
        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion_id": {"type": "string"},
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
                "required": ["criterion_id", "detected_issues", "issues", "notes"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["components", "connections", "extra_components", "extra_connections", "other_components", "other_connections", "metrics"],
    "additionalProperties": False,
}

# Metrics scored in code from per-item verdicts. Every other criterion is a counted
# criterion: the judge reports failures, and the code supplies the opportunities.
ATOMIC_METRICS = frozenset(
    {
        METRIC_PRESENCE,
        METRIC_LABELS,
        METRIC_UNEXPECTED,
        METRIC_CONNECTIONS,
        METRIC_ATTACHMENT,
        METRIC_CONNECTION_LABELS,
        METRIC_MINIMUM_FONT,
    }
)

# What one opportunity is for each counted criterion; the judge counts failures only.
COUNTING_RULES = {
    "description_order": (
        "Opportunities: required Part B connections whose two components are both drawn, except connections that "
        "loop back to an earlier step. Count one issue per such connection that runs against the diagram's main "
        "reading direction. Typical connections are guesses and are not checked here."
    ),
    "canvas_bounds": (
        "Opportunities: every drawn component and every drawn arrow. Count one issue per component or arrow that "
        "is cut off by, or lies outside, the image edge."
    ),
    "unintended_overlap": (
        "Opportunities: every drawn component. Count one issue per component that unintentionally overlaps or "
        "collides with another component (count components, not pairs). A container drawn around its members is "
        "not an overlap."
    ),
    "text_overflow": (
        "Opportunities: every drawn component label and every drawn connection label. Count one issue per text "
        "that spills out of its box, is clipped, or collides with other text."
    ),
    "connector_routing": (
        "Opportunities: every drawn arrow. Count one issue per arrow that passes through a component or across "
        "text, crosses other arrows where a cleaner route was possible, or takes a confusing detour."
    ),
    "text_contrast": (
        "Opportunities: every drawn component label and every drawn connection label. Count one issue per text "
        "whose color is too close to its background to read easily, such as light text on a light fill."
    ),
}
_DEFAULT_COUNTING_RULE = (
    "Opportunities: every drawn component. Count one issue per drawn component that fails this criterion."
)


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
                "tier": component.get("tier", "required"),
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
    labels: dict[str, str] | None = None,
) -> list[ComponentVerdict]:
    """Validate a judge's per-component answers and return them in inventory order.

    Rejected outright: unknown IDs, duplicate IDs, and missing IDs, because the scoring
    denominator must be exactly the frozen inventory. A missing ``label_legible`` flag
    (older recorded answers) means the label was readable.
    Downgraded to ABSENT (with a note), to guard against "yes" answers without evidence:
    - PRESENT without quoted visible text or without a usable location;
    - MISLABELED without a usable location;
    - a component that claims the same drawn element as an earlier component. With
      ``labels``, only components with the same expected label (repeated blocks) conflict;
      one drawn element may otherwise cover several roles, such as a box combining two steps.
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
                    and (
                        labels is None
                        or labels.get(earlier.component_id, "").strip().casefold()
                        == labels.get(component_id, "").strip().casefold()
                    )
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
                label_legible=bool(item.get("label_legible", True)),
            )
        )
    return verdicts


def _connection_checklist(inventory: dict[str, Any]) -> list[dict[str, str]]:
    """Checklist items for the per-connection questions, with stable ids C1, C2, ..."""
    components = inventory.get("components", [])
    labels = {component["id"]: component.get("label", "") for component in components}
    counts = Counter(label.strip().casefold() for label in labels.values())

    def name(component_id: str) -> str:
        label = labels.get(component_id, component_id)
        return f"{label} [{component_id}]" if counts[label.strip().casefold()] > 1 else label

    return [
        {
            "connection_id": f"C{index + 1}",
            "source_id": edge.get("source", ""),
            "target_id": edge.get("target", ""),
            "from": name(edge.get("source", "")),
            "to": name(edge.get("target", "")),
            "expected_label": edge.get("label", "") or "",
            "tier": edge.get("tier", "required"),
        }
        for index, edge in enumerate(inventory.get("connections", []))
    ]


def parse_connection_verdicts(
    items: list[dict[str, Any]],
    checklist: list[dict[str, str]],
    judge: str,
) -> list[ConnectionVerdict]:
    """Validate a judge's per-connection answers and return them in inventory order.

    Unknown, duplicate, and missing connection ids are rejected, because the Connectivity
    denominator must be exactly the frozen inventory.
    """
    expected = {item["connection_id"]: item for item in checklist}
    seen: dict[str, dict[str, Any]] = {}
    for item in items:
        connection_id = item["connection_id"]
        if connection_id not in expected:
            raise RuntimeError(f"{judge} returned an unknown connection id: {connection_id}")
        if connection_id in seen:
            raise RuntimeError(f"{judge} returned connection id twice: {connection_id}")
        seen[connection_id] = item
    missing = [connection_id for connection_id in expected if connection_id not in seen]
    if missing:
        raise RuntimeError(f"{judge} omitted connection ids: {missing}")
    return [
        ConnectionVerdict(
            connection_id=connection_id,
            source=spec["source_id"],
            target=spec["target_id"],
            status=ConnectionStatus(seen[connection_id]["verdict"]),
            judge=judge,
            expected_label=spec["expected_label"],
            visible_label=seen[connection_id].get("visible_label", "").strip(),
            label_legible=bool(seen[connection_id].get("label_legible", True)),
            evidence=seen[connection_id].get("evidence", ""),
        )
        for connection_id, spec in expected.items()
    ]


@dataclass
class JudgeSample:
    """One parsed judge response."""

    components: list[ComponentVerdict]
    connections: list[ConnectionVerdict]
    extra_components: list[str] = field(default_factory=list)
    extra_connections: list[str] = field(default_factory=list)
    # Drawn content beyond the checklist that fits the diagram; counted in denominators only.
    other_components: list[str] = field(default_factory=list)
    other_connections: list[str] = field(default_factory=list)
    # criterion id -> (detected issues, issues, notes)
    counted: dict[str, tuple[int, list[Issue], str]] = field(default_factory=dict)


_COMPONENT_ORDER = [Verdict.PRESENT, Verdict.MISLABELED, Verdict.UNCERTAIN, Verdict.ABSENT]
_CONNECTION_ORDER = [ConnectionStatus.PRESENT, ConnectionStatus.REVERSED, ConnectionStatus.UNCERTAIN, ConnectionStatus.ABSENT]


def _vote(values: list[Any], order: list[Any]) -> Any:
    """The most common value; on a tie, the median of all values on the given order."""
    counts = Counter(values)
    top = max(counts.values())
    winners = [value for value in counts if counts[value] == top]
    if len(winners) == 1:
        return winners[0]
    ranked = sorted(values, key=order.index)
    return ranked[(len(ranked) - 1) // 2]


def merge_samples(samples: list[JudgeSample]) -> JudgeSample:
    """Combine repeated judge responses: majority verdict per component and connection,
    median failure count per counted criterion, and the extras of the median sample."""
    if len(samples) == 1:
        return samples[0]
    components = []
    for items in zip(*(sample.components for sample in samples)):
        verdict = _vote([item.verdict for item in items], _COMPONENT_ORDER)
        chosen = next(item for item in items if item.verdict == verdict)
        legible = Counter(item.label_legible for item in items)
        chosen.label_legible = legible[True] >= legible[False] if legible[True] != legible[False] else chosen.label_legible
        chosen.notes = (chosen.notes + " " if chosen.notes else "") + "votes: " + "/".join(item.verdict.value for item in items)
        components.append(chosen)
    connections = []
    for items in zip(*(sample.connections for sample in samples)):
        status = _vote([item.status for item in items], _CONNECTION_ORDER)
        chosen = next(item for item in items if item.status == status)
        legible = Counter(item.label_legible for item in items)
        chosen.label_legible = legible[True] >= legible[False] if legible[True] != legible[False] else chosen.label_legible
        chosen.notes = "votes: " + "/".join(item.status.value for item in items)
        connections.append(chosen)

    def median_sample(key) -> JudgeSample:
        ranked = sorted(samples, key=key)
        return ranked[(len(ranked) - 1) // 2]

    counted: dict[str, tuple[int, list[Issue], str]] = {}
    for criterion_id in samples[0].counted:
        values = [sample.counted[criterion_id][0] for sample in samples if criterion_id in sample.counted]
        chosen_value = median_low(values)
        source = next(sample for sample in samples if sample.counted.get(criterion_id, (None,))[0] == chosen_value)
        _value, issues, notes = source.counted[criterion_id]
        counted[criterion_id] = (chosen_value, issues, notes)
    return JudgeSample(
        components=components,
        connections=connections,
        extra_components=median_sample(lambda sample: len(sample.extra_components)).extra_components,
        extra_connections=median_sample(lambda sample: len(sample.extra_connections)).extra_connections,
        other_components=median_sample(lambda sample: len(sample.other_components)).other_components,
        other_connections=median_sample(lambda sample: len(sample.other_connections)).other_connections,
        counted=counted,
    )


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
        # Token counts over every call this backend makes, so a run's cost can be worked out.
        self.usage = {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        self._usage_lock = threading.Lock()

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
        usage = response.usage
        with self._usage_lock:
            self.usage["calls"] += 1
            if usage is not None:
                self.usage["input_tokens"] += usage.input_tokens or 0
                self.usage["output_tokens"] += usage.output_tokens or 0
                self.usage["cached_input_tokens"] += getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
                self.usage["reasoning_tokens"] += getattr(usage.output_tokens_details, "reasoning_tokens", 0) or 0
        return json.loads(response.output_text)


class ExpectationExtractor:
    def __init__(self, backend: VLMBackend) -> None:
        self.backend = backend

    def extract(self, description: str, criteria: list[Criterion]) -> dict[str, Any]:
        prompt = (
            "You are stage 1 of a description-to-diagram alignment evaluator. Read only the written diagram "
            "description below; it may be a detailed specification or a short figure caption. There is no "
            "candidate image in this stage. Imagine you were asked to draw this diagram, and write down what it "
            "should contain: the components (steps, stages, groups, inputs, outputs, decisions, repeated blocks) "
            "and the directed connections of the intended flow. Describe each component by its role; when the "
            "description gives a component an exact name, use that name as its label, otherwise give a short label "
            "naming the role (the drawing may word it differently). Mark each component and connection with a "
            "tier: 'required' when the description states it or it follows directly from the description, and "
            "'typical' when such a diagram usually shows it but the description does not call for it. For a "
            "detailed specification most items are required; for a short caption keep required items to what the "
            "caption actually says and put the rest under typical. Do not invent specific names, numbers, or "
            "details the description does not give. Components are the boxes or nodes a reader would see: the "
            "diagram's overall subject or title (such as 'Experimental procedure' or 'Flowchart of the trial') is "
            "not a component, and notes, footnotes, abbreviation definitions, and remarks about the analysis are "
            "not components unless they name a step that would be drawn as its own box. Presence must enumerate the components and nothing else: "
            "connections belong under Connectivity only. Details must describe meaningful labels and reject missing "
            "or placeholder labels. Layout must describe sensible ordering, containment, overlap, routing, and "
            "canvas expectations. Legibility must describe readable text expectations. Do not score. Do not add "
            "decorative requirements. Also return an ordered components list. Give each component a unique "
            "snake_case id, its label, a short kind, and its tier. Preserve repeated components as separate entries "
            "with separate ids. Return connections using only those component ids, directed from source to target, "
            "each with a tier; a connection is required only when the description implies it and both its "
            "components are required. Give a connection a label only when the description says that text is written "
            "on the arrow itself, such as YES/NO on the branches of a decision; categories, conditions, or outcomes "
            "the description names as steps or groups are components, not arrow labels. Otherwise use an empty "
            "string. Return explicit forbidden placeholder strings. Use short, atomic strings.\n\nShared rubric:\n"
            + json.dumps(_criterion_payload(criteria), ensure_ascii=False)
            + "\n\nDiagram description:\n"
            + description
        )
        data = self.backend.structured(prompt, [], INVENTORY_SCHEMA, "diagram_expectation_inventory")
        data["inventory"] = {
            dimension.value: list(data["inventory"][dimension.value]) for dimension in Dimension
        }
        for edge in data.get("connections", []):
            edge.setdefault("label", "")
            edge.setdefault("tier", "required")
        for component in data.get("components", []):
            component.setdefault("tier", "required")
        return data


class VLMJudge:
    """Image judge: per-item verdicts for components and connections, failure counts for
    the remaining criteria. Every denominator comes from the frozen inventory and the
    item verdicts, never from a number the model chooses.

    ``samples`` > 1 asks the same question several times and keeps the majority verdict
    per item and the median failure count per criterion; the per-sample scores are
    recorded in each metric's notes so run-to-run variation is visible.
    """

    def __init__(self, backend: VLMBackend, policy: PresencePolicy | None = None, samples: int = 1) -> None:
        if samples < 1:
            raise ValueError("samples must be at least 1")
        self.backend = backend
        self.name = backend.name
        self.policy = policy or PresencePolicy()
        self.samples = samples

    def prompt(
        self,
        description: str,
        criteria: list[Criterion],
        inventory: dict[str, Any],
    ) -> str:
        counted = [criterion for criterion in criteria if _metric_key(criterion) not in ATOMIC_METRICS]
        checklist = _component_checklist(inventory)
        connections = [
            {key: item[key] for key in ("connection_id", "from", "to", "expected_label", "tier")}
            for item in _connection_checklist(inventory)
        ]
        return (
            "You are stage 2 of a description-to-diagram alignment evaluator. The attached image is the only "
            "CANDIDATE. Evaluate it against the written description and the frozen stage-1 inventory. Do not "
            "invent a different target. Evidence must refer to visible candidate elements. Judge every part on its "
            "own: a placement problem is not a missing component, a crossing arrow is not a missing connection, and "
            "tiny text is not a wrong label. Each checklist item has a tier: 'required' items are what the "
            "description calls for; 'typical' items are only a guess at what such a diagram often shows. Answer "
            "typical items the same way, but never treat the drawing as wrong because it is organized differently "
            "from them (different groups, steps, branches, or directions).\n\n"
            "PART A - component checklist (field 'components'). Return exactly one entry for every component_id "
            "below, in any order, and no other ids. Answer each item on its own:\n"
            "- present: a visible element plays this component's role and its label names that role. Match by "
            "meaning, not wording: different words with the same meaning count as present, and the expected label "
            "only names the role. Ignore case, punctuation, and line breaks.\n"
            "- mislabeled: a visible element plays this role (recognizable from its label, position, grouping, or "
            "arrows), but its label is missing, a placeholder, or names something else.\n"
            "- absent: no visible element plays this role.\n"
            "- uncertain: the image is too small, blurry, or low-resolution to tell whether any element plays this "
            "role. This is not counted as missing; it is scored as unreadable under Legibility. Use it only when image "
            "quality prevents a decision; if you can see clearly that no element plays this role, answer absent.\n"
            + _policy_rules(self.policy)
            + "\nSet label_legible to false when the element is drawn but its text is too small, blurry, or clipped "
            "to read. In that case still decide present or mislabeled from what you can see, quote what you can "
            "read, and never call a label wrong only because it is hard to read. "
            "Rules: quote the element's visible text exactly as drawn in visible_text ('' if it has none). Give "
            "location as the element's bounding box in normalized image coordinates from 0 to 1 (x0, y0 top-left; "
            "x1, y1 bottom-right); use all zeros when absent. One drawn element may cover several checklist items when "
            "it clearly plays all those roles (for example one box that combines two steps), and several drawn "
            "elements may together cover one item; give the location of the main one. Components with the same "
            "expected label (repeated blocks) must each match a different drawn element. Judge only whether the "
            "component itself is drawn and labeled; placement and arrows are judged elsewhere.\n\n"
            "PART B - connection checklist (field 'connections'). Return exactly one entry for every connection_id "
            "below, and no other ids:\n"
            "- present: arrows lead from 'from' to 'to', directly or through drawn intermediate steps that are not "
            "in the checklist (extra detail between the two is fine).\n"
            "- reversed: they lead from 'to' to 'from' instead.\n"
            "- absent: no arrows lead from one to the other, including when either component is not drawn.\n"
            "- uncertain: you cannot tell; this is scored as absent.\n"
            "Trace each arrow from end to end. Long routes, bends, crossings, and poor placement do not make a "
            "connection absent if you can follow it. Quote the text written on or next to the arrow in "
            "visible_label ('' if there is none), whether or not a label is expected, and set label_legible to "
            "false if that text is too small or blurry to read.\n\n"
            "PART C - extras. The checklist is a sketch of what the diagram should contain, not a complete list, and "
            "its typical items are guesses; judge the drawing against the description, not against the sketch. "
            "additional detail that fits the described diagram (extra steps, sub-steps, counts, exclusion or "
            "side boxes) is fine and is NOT extra. In 'extra_components', list only drawn boxes or nodes that do "
            "not belong in this diagram: unrelated to what the description is about, duplicated by mistake, or "
            "contradicting the description. Do not list titles, captions, legends, arrowheads, or a container frame "
            "that groups components. In 'extra_connections', list only drawn arrows that are wrong: they contradict "
            "the described flow, connect parts that should not be connected, dangle with an end attached to "
            "nothing, or duplicate another arrow; give the text of the elements at its two ends. Arrows to or from "
            "additional detail that fits the diagram are not extra. When unsure whether something belongs, it "
            "belongs: list it under the other lists below, not as extra. Then list everything else that is drawn: in "
            "'other_components', every other box or node that is neither a checklist component nor extra (the "
            "additional detail that fits), with its visible text and location; in 'other_connections', every other "
            "drawn arrow that is neither a Part B connection nor extra, with the text at its two ends. These lists "
            "are not penalized; they make sure every drawn element is counted.\n\n"
            "PART D - counted criteria (field 'metrics'). Return one entry for every criterion listed here, with "
            "detected_issues as the number of failures under its counting rule. The number of opportunities is "
            "computed from your Part A-C answers, so report failures only, and count each failing item once.\n\n"
            "Component checklist (Part A):\n"
            + json.dumps(checklist, ensure_ascii=False)
            + "\n\nConnection checklist (Part B):\n"
            + json.dumps(connections, ensure_ascii=False)
            + "\n\nCounted criteria (Part D):\n"
            + json.dumps(
                [
                    {**payload, "counting_rule": COUNTING_RULES.get(_metric_key(criterion) or "", _DEFAULT_COUNTING_RULE)}
                    for payload, criterion in zip(_criterion_payload(counted), counted)
                ],
                ensure_ascii=False,
            )
            + "\n\nDiagram description:\n"
            + description
            + "\n\nFrozen description-derived inventory (a sketch of the expected diagram):\n"
            + json.dumps(inventory, ensure_ascii=False)
        )

    def evaluate(
        self,
        description: str,
        candidate_image: str | Path,
        criteria: list[Criterion],
        inventory: dict[str, Any],
    ) -> list[MetricScore]:
        prompt = self.prompt(description, criteria, inventory)
        if self.samples == 1:
            responses = [self.backend.structured(prompt, [candidate_image], JUDGE_SCHEMA, "diagram_correctness_judgment")]
        else:
            with ThreadPoolExecutor(max_workers=self.samples) as executor:
                responses = list(
                    executor.map(
                        lambda _index: self.backend.structured(
                            prompt, [candidate_image], JUDGE_SCHEMA, "diagram_correctness_judgment"
                        ),
                        range(self.samples),
                    )
                )
        samples = [self._parse(response, criteria, inventory, candidate_image) for response in responses]
        results = self._score(merge_samples(samples), criteria, inventory)
        if len(samples) > 1:
            per_sample = [
                {metric.criterion_id: metric.score for metric in self._score(sample, criteria, inventory)}
                for sample in self._reparse(responses, criteria, inventory, candidate_image)
            ]
            for metric in results:
                values = [scores.get(metric.criterion_id) for scores in per_sample]
                shown = ", ".join("n/a" if value is None else f"{value:.2f}" for value in values)
                metric.notes = (metric.notes + " " if metric.notes else "") + f"Scores across {len(values)} samples: {shown}."
        return results

    def _reparse(self, responses, criteria, inventory, candidate_image) -> list[JudgeSample]:
        # merge_samples updates the chosen verdict objects, so score each sample from a fresh parse.
        return [self._parse(response, criteria, inventory, candidate_image) for response in responses]

    def _parse(
        self,
        data: dict[str, Any],
        criteria: list[Criterion],
        inventory: dict[str, Any],
        candidate_image: str | Path,
    ) -> JudgeSample:
        checklist = _component_checklist(inventory)
        connection_checklist = _connection_checklist(inventory)
        components = parse_component_verdicts(
            data["components"],
            [item["component_id"] for item in checklist],
            self.name,
            _png_size(candidate_image),
            {item["component_id"]: item["expected_label"] for item in checklist},
        )
        if "connections" not in data and connection_checklist:
            raise RuntimeError(f"{self.name} returned no connection verdicts")
        connections = parse_connection_verdicts(data.get("connections", []), connection_checklist, self.name)

        drawn_boxes = [item.bounds for item in components if item.bounds is not None and item.verdict in {Verdict.PRESENT, Verdict.MISLABELED}]
        extra_components = []
        for item in data.get("extra_components", []):
            box = _normalized_box(item["location"], _png_size(candidate_image)) if "location" in item else None
            if box is not None and any(_iou(box, other) >= SAME_ELEMENT_IOU for other in drawn_boxes):
                continue  # the same element as a listed component, not an extra one
            extra_components.append(f"'{item.get('visible_text', '').strip()}'")
        extra_connections = [
            f"'{item.get('from_text', '').strip()}' -> '{item.get('to_text', '').strip()}'"
            for item in data.get("extra_connections", [])
        ]
        other_components = []
        for item in data.get("other_components", []):
            box = _normalized_box(item["location"], _png_size(candidate_image)) if "location" in item else None
            if box is not None and any(_iou(box, other) >= SAME_ELEMENT_IOU for other in drawn_boxes):
                continue  # the same element as a listed component
            other_components.append(item.get("visible_text", "").strip())
        other_connections = [
            f"'{item.get('from_text', '').strip()}' -> '{item.get('to_text', '').strip()}'"
            for item in data.get("other_connections", [])
        ]

        counted_ids = {criterion.id for criterion in criteria if _metric_key(criterion) not in ATOMIC_METRICS}
        counted: dict[str, tuple[int, list[Issue], str]] = {}
        for item in data.get("metrics", []):
            criterion_id = item["criterion_id"]
            if criterion_id not in counted_ids or criterion_id in counted:
                continue
            counted[criterion_id] = (
                int(item["detected_issues"]),
                [
                    Issue(
                        description=issue["description"],
                        severity=Severity(issue["severity"]),
                        element_ids=list(issue["element_ids"]),
                    )
                    for issue in item.get("issues", [])
                ],
                item.get("notes", ""),
            )
        missing = counted_ids - set(counted)
        if missing:
            raise RuntimeError(f"{self.name} omitted rubric criteria: {sorted(missing)}")
        return JudgeSample(
            components, connections, extra_components, extra_connections, other_components, other_connections, counted
        )

    def _score(
        self,
        sample: JudgeSample,
        criteria: list[Criterion],
        inventory: dict[str, Any],
    ) -> list[MetricScore]:
        by_metric = {_metric_key(criterion): criterion for criterion in criteria}
        by_id = {criterion.id: criterion for criterion in criteria}
        presence = by_metric.get(METRIC_PRESENCE) or by_id.get(PRESENCE_CRITERION_ID)
        details = by_metric.get(METRIC_LABELS) or by_id.get(DETAILS_CRITERION_ID)
        drawn = drawn_component_ids(sample.components)
        typical_components, typical_connections = typical_items(inventory)
        results = component_metrics(
            sample.components, self.name, presence=presence, details=details, typical=typical_components
        )
        results += connection_metrics(
            sample.connections,
            drawn,
            sample.extra_connections,
            self.name,
            recall=by_metric.get(METRIC_CONNECTIONS),
            precision=by_metric.get(METRIC_ATTACHMENT),
            labels=by_metric.get(METRIC_CONNECTION_LABELS),
            accept_abbreviations=self.policy.accept_abbreviations,
            typical=typical_connections,
            other_arrows=len(sample.other_connections),
        )
        unexpected = by_metric.get(METRIC_UNEXPECTED)
        if unexpected is not None:
            results.append(
                MetricScore(
                    criterion_id=unexpected.id,
                    dimension=unexpected.dimension,
                    judge=self.name,
                    expected_count=len(drawn) + len(sample.other_components) + len(sample.extra_components),
                    detected_issues=len(sample.extra_components),
                    issues=[
                        Issue(f"Drawn component that does not belong in this diagram: {text}", unexpected.severity)
                        for text in sample.extra_components
                    ],
                    notes="Denominator: every drawn component (checklist, other fitting detail, and extra).",
                )
            )
        font = by_metric.get(METRIC_MINIMUM_FONT)
        if font is not None:
            results.append(legibility_from_verdicts(sample.components, sample.connections, self.name, font))

        opportunities = self._opportunities(sample, inventory, drawn)
        for criterion in criteria:
            if criterion.id not in sample.counted:
                continue
            expected = opportunities.get(_metric_key(criterion) or "", len(drawn))
            detected, issues, notes = sample.counted[criterion.id]
            results.append(
                MetricScore(
                    criterion_id=criterion.id,
                    dimension=criterion.dimension,
                    judge=self.name,
                    expected_count=expected,
                    detected_issues=min(detected, expected),
                    issues=issues,
                    notes=notes,
                )
            )
        return results

    @staticmethod
    def _opportunities(sample: JudgeSample, inventory: dict[str, Any], drawn: set[str]) -> dict[str, int]:
        connections = inventory.get("connections", [])
        feedback = feedback_edges(inventory.get("components", []), connections)
        drawn_links = [
            item
            for item in sample.connections
            if item.status in {ConnectionStatus.PRESENT, ConnectionStatus.REVERSED}
            and item.source in drawn
            and item.target in drawn
        ]
        component_texts = [
            item for item in sample.components
            if item.verdict in {Verdict.PRESENT, Verdict.MISLABELED} and item.visible_text.strip()
        ]
        # Every drawn element counts: checklist items, other fitting detail, and extras.
        boxes = [*drawn, *sample.other_components, *sample.extra_components]
        arrows = [*drawn_links, *sample.other_connections, *sample.extra_connections]
        texts = [
            *component_texts,
            *(text for text in sample.other_components if text),
            *(item for item in drawn_links if item.visible_label.strip()),
        ]
        return {
            "description_order": sum(
                1
                for index, edge in enumerate(connections)
                if index not in feedback
                and edge.get("tier") != "typical"
                and edge.get("source") != edge.get("target")
                and edge.get("source") in drawn
                and edge.get("target") in drawn
            ),
            "canvas_bounds": len(boxes) + len(arrows),
            "unintended_overlap": len(boxes),
            "text_overflow": len(texts),
            "connector_routing": len(arrows),
            "text_contrast": len(texts),
        }


def _metric_key(criterion: Criterion) -> str | None:
    """The rubric metric a criterion measures; older criteria are recognized by id."""
    if criterion.deterministic_metric:
        return criterion.deterministic_metric
    return {PRESENCE_CRITERION_ID: METRIC_PRESENCE, DETAILS_CRITERION_ID: METRIC_LABELS}.get(criterion.id)
