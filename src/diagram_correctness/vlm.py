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


def _raw_image(path: str | Path) -> tuple[str, str]:
    file_path = Path(path)
    mime = mimetypes.guess_type(file_path.name)[0] or "image/png"
    return mime, base64.b64encode(file_path.read_bytes()).decode("ascii")


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
        }
    },
    "required": ["inventory"],
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


class AnthropicBackend(VLMBackend):
    def __init__(self, model: str) -> None:
        from anthropic import Anthropic

        self.model = model
        self.name = model
        self.client = Anthropic()

    def structured(self, prompt: str, images: list[str | Path], schema: dict[str, Any], schema_name: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for image in images:
            mime, data = _raw_image(image)
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": data},
                }
            )
        content.append({"type": "text", "text": prompt})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            messages=[{"role": "user", "content": content}],
            tools=[
                {
                    "name": schema_name,
                    "description": "Return the requested evaluation as structured data.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": schema_name},
        )
        tool_block = next((block for block in response.content if block.type == "tool_use"), None)
        if tool_block is None:
            raise RuntimeError(f"{self.model} did not return the required structured result")
        return dict(tool_block.input)


class ExpectationExtractor:
    def __init__(self, backend: VLMBackend) -> None:
        self.backend = backend

    def extract(self, reference_image: str | Path, criteria: list[Criterion]) -> dict[str, list[str]]:
        prompt = (
            "You are the expectation-discovery stage of a diagram evaluator. The attached image is the "
            "REFERENCE diagram, never the candidate. Inventory concrete, countable expectations visible in "
            "the image for all five dimensions. Presence and details must be especially explicit (objects, "
            "labels, colors, styles, and small marks). Connectivity entries must state source, destination, "
            "and direction. Layout entries must state relative positions. Legibility entries must identify "
            "text regions. Do not score anything. Use short, atomic strings.\n\nShared rubric:\n"
            + json.dumps(_criterion_payload(criteria), ensure_ascii=False)
        )
        data = self.backend.structured(prompt, [reference_image], INVENTORY_SCHEMA, "diagram_expectation_inventory")
        return {dimension.value: list(data["inventory"][dimension.value]) for dimension in Dimension}


class VLMJudge:
    def __init__(self, backend: VLMBackend) -> None:
        self.backend = backend
        self.name = backend.name

    def evaluate(
        self,
        reference_image: str | Path,
        candidate_image: str | Path,
        criteria: list[Criterion],
        inventory: dict[str, list[str]],
    ) -> list[MetricScore]:
        prompt = (
            "You are one judge in a correctness panel. Image 1 is the REFERENCE and image 2 is the CANDIDATE. "
            "Evaluate every rubric criterion independently. Use the frozen inventory below; do not invent a "
            "different task. For each metric, expected_count is the number of relevant countable opportunities "
            "and detected_issues is the number that fail. If no opportunity exists, return both counts as zero. "
            "Count each failing opportunity once. Evidence must refer to visible elements.\n\nShared rubric:\n"
            + json.dumps(_criterion_payload(criteria), ensure_ascii=False)
            + "\n\nFrozen reference inventory:\n"
            + json.dumps(inventory, ensure_ascii=False)
        )
        data = self.backend.structured(
            prompt,
            [reference_image, candidate_image],
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

