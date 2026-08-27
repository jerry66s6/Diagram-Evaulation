from pathlib import Path

from diagram_correctness.models import Criterion, Dimension, Severity
from diagram_correctness.vlm import ExpectationExtractor, VLMJudge


class RecordingBackend:
    name = "recording-gpt"

    def __init__(self) -> None:
        self.calls = []

    def structured(self, prompt, images, schema, schema_name):
        self.calls.append(
            {"prompt": prompt, "images": list(images), "schema_name": schema_name}
        )
        if schema_name == "diagram_expectation_inventory":
            return {
                "inventory": {
                    dimension.value: [f"expected {dimension.value}"]
                    for dimension in Dimension
                },
                "components": [
                    {"id": "input_tokens", "label": "Input Tokens", "kind": "input"}
                ],
                "connections": [],
                "forbidden_placeholders": ["TBD"],
            }
        return {
            "metrics": [
                {
                    "criterion_id": "presence.elements",
                    "expected_count": 1,
                    "detected_issues": 0,
                    "issues": [],
                    "notes": "visible",
                }
            ]
        }


def test_presence_expectations_are_frozen_from_description_before_image_judging(
    tmp_path: Path,
) -> None:
    backend = RecordingBackend()
    criterion = Criterion(
        "presence.elements", Dimension.PRESENCE, "required elements", Severity.HIGH
    )
    description = "Show Input Tokens and Token Embeddings."
    inventory = ExpectationExtractor(backend).extract(description, [criterion])
    assert backend.calls[0]["images"] == []
    assert description in backend.calls[0]["prompt"]
    assert inventory["components"][0]["id"] == "input_tokens"

    candidate = tmp_path / "candidate.png"
    candidate.write_bytes(b"fake image")
    results = VLMJudge(backend).evaluate(description, candidate, [criterion], inventory)
    assert backend.calls[1]["images"] == [candidate]
    assert "Frozen description-derived inventory" in backend.calls[1]["prompt"]
    assert results[0].score == 1.0
