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
            # A count-based Presence answer is ignored: Presence is scored per component.
            "metrics": [
                {
                    "criterion_id": "presence.elements",
                    "expected_count": 5,
                    "detected_issues": 5,
                    "issues": [],
                    "notes": "ignored",
                }
            ],
            "components": [
                {
                    "component_id": "input_tokens",
                    "verdict": "present",
                    "visible_text": "Input Tokens",
                    "location": {"x0": 0.1, "y0": 0.1, "x1": 0.4, "y1": 0.2},
                    "evidence": "box at the top",
                }
            ],
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
    assert '"component_id": "input_tokens"' in backend.calls[1]["prompt"]
    presence = next(result for result in results if result.criterion_id == "presence.elements")
    assert presence.score == 1.0
    assert presence.expected_count == 1
    assert [item.verdict.value for item in presence.component_verdicts] == ["present"]
