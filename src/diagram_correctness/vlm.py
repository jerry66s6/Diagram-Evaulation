from __future__ import annotations

import base64
import json
import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .models import Criterion, Dimension, Issue, MetricScore, Severity


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
        }
    },
    "required": ["metrics"],
    "additionalProperties": False,
}


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
            "group, input, output, and repeated block. Details must enumerate exact meaningful labels and content "
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
    def __init__(self, backend: VLMBackend) -> None:
        self.backend = backend
        self.name = backend.name

    def evaluate(
        self,
        description: str,
        candidate_image: str | Path,
        criteria: list[Criterion],
        inventory: dict[str, Any],
    ) -> list[MetricScore]:
        prompt = (
            "You are stage 2 of a description-to-diagram alignment evaluator. The attached image is the only "
            "CANDIDATE. Evaluate it against the written description and the frozen stage-1 inventory. Do not "
            "invent a different target. Presence and Details must use the frozen expectations rather than "
            "deciding what should exist while scoring. For every criterion, expected_count is the number of "
            "relevant countable opportunities and detected_issues is the number that fail. If no opportunity "
            "exists, return both counts as zero. Count each failing opportunity once. Layout must penalize overlap "
            "and off-canvas content. Connectivity must verify visible source and target endpoints, not merely arrow "
            "count. Details must penalize missing, meaningless, and placeholder labels. Evidence must refer to "
            "visible candidate elements.\n\nShared rubric:\n"
            + json.dumps(_criterion_payload(criteria), ensure_ascii=False)
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
        criteria_by_id = {criterion.id: criterion for criterion in criteria}
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
        return results
