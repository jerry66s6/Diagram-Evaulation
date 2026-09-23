"""Tests for per-component (atomic) Presence and Details scoring."""

import json
from pathlib import Path

import pytest

from diagram_correctness.deterministic import DeterministicJudge
from diagram_correctness.models import (
    ComponentVerdict,
    Criterion,
    Dimension,
    MetricScore,
    PresencePolicy,
    Severity,
    Verdict,
    component_agreement,
    component_metrics,
)
from diagram_correctness.rubric import load_rubric
from diagram_correctness.svg import parse_svg
from diagram_correctness.vlm import VLMJudge, parse_component_verdicts


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "transformer"
PRESENCE = Criterion("presence.elements", Dimension.PRESENCE, "", Severity.HIGH, "description_presence")
DETAILS = Criterion("details.labels", Dimension.DETAILS, "", Severity.HIGH, "meaningful_labels")


def _verdicts(pairs: list[tuple[str, Verdict]]) -> list[ComponentVerdict]:
    return [ComponentVerdict(component_id, verdict, "judge") for component_id, verdict in pairs]


def _box(x0: float, y0: float, x1: float, y1: float) -> dict[str, float]:
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _write_svg(tmp_path: Path, body: str, name: str = "candidate.svg") -> Path:
    path = tmp_path / name
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">' + body + "</svg>",
        encoding="utf-8",
    )
    return path


def _by_criterion(metrics: list[MetricScore]) -> dict[str, MetricScore]:
    return {metric.criterion_id: metric for metric in metrics}


def test_each_failure_is_counted_in_exactly_one_dimension() -> None:
    # The mixed Transformer candidate: 1 missing component and 4 wrong labels.
    verdicts = _verdicts(
        [("c%d" % index, Verdict.PRESENT) for index in range(6)]
        + [("m%d" % index, Verdict.MISLABELED) for index in range(4)]
        + [("absent", Verdict.ABSENT)]
    )
    results = _by_criterion(component_metrics(verdicts, "judge", PRESENCE, DETAILS))
    assert (results["presence.elements"].detected_issues, results["presence.elements"].expected_count) == (1, 11)
    assert (results["details.labels"].detected_issues, results["details.labels"].expected_count) == (4, 10)
    assert results["details.labels"].score == pytest.approx(0.6)


def test_uncertain_items_leave_the_presence_denominator() -> None:
    verdicts = _verdicts([("a", Verdict.PRESENT), ("b", Verdict.UNCERTAIN), ("c", Verdict.ABSENT)])
    presence = _by_criterion(component_metrics(verdicts, "judge", PRESENCE, DETAILS))["presence.elements"]
    assert (presence.detected_issues, presence.expected_count) == (1, 2)
    assert "b" in presence.notes


def test_details_is_not_applicable_when_nothing_is_drawn() -> None:
    verdicts = _verdicts([("a", Verdict.ABSENT)])
    details = _by_criterion(component_metrics(verdicts, "judge", PRESENCE, DETAILS))["details.labels"]
    assert details.score is None


def test_parser_rejects_ids_outside_the_frozen_inventory() -> None:
    item = {"verdict": "absent", "visible_text": "", "location": _box(0, 0, 0, 0), "evidence": ""}
    with pytest.raises(RuntimeError, match="unknown"):
        parse_component_verdicts([{**item, "component_id": "x"}], ["a"], "gpt")
    with pytest.raises(RuntimeError, match="twice"):
        parse_component_verdicts(
            [{**item, "component_id": "a"}, {**item, "component_id": "a"}], ["a"], "gpt"
        )
    with pytest.raises(RuntimeError, match="omitted"):
        parse_component_verdicts([{**item, "component_id": "a"}], ["a", "b"], "gpt")


def test_parser_downgrades_present_without_evidence() -> None:
    items = [
        {"component_id": "a", "verdict": "present", "visible_text": "", "location": _box(0.1, 0.1, 0.3, 0.2), "evidence": ""},
        {"component_id": "b", "verdict": "present", "visible_text": "B", "location": _box(0, 0, 0, 0), "evidence": ""},
        {"component_id": "c", "verdict": "mislabeled", "visible_text": "", "location": _box(0.5, 0.5, 0.7, 0.6), "evidence": ""},
    ]
    verdicts = parse_component_verdicts(items, ["a", "b", "c"], "gpt")
    assert [item.verdict for item in verdicts] == [Verdict.ABSENT, Verdict.ABSENT, Verdict.MISLABELED]
    assert "downgraded" in verdicts[0].notes


def test_parser_allows_one_component_per_drawn_element() -> None:
    # Both "Add & Norm" components point at the same box, so only the first keeps it.
    items = [
        {"component_id": "norm_1", "verdict": "present", "visible_text": "Add & Norm", "location": _box(0.3, 0.5, 0.6, 0.55), "evidence": ""},
        {"component_id": "norm_2", "verdict": "present", "visible_text": "Add & Norm", "location": _box(0.31, 0.5, 0.6, 0.56), "evidence": ""},
    ]
    verdicts = parse_component_verdicts(items, ["norm_1", "norm_2"], "gpt")
    assert [item.verdict for item in verdicts] == [Verdict.PRESENT, Verdict.ABSENT]
    assert "norm_1" in verdicts[1].notes


def test_parser_converts_pixel_locations_when_image_size_is_known() -> None:
    items = [
        {"component_id": "a", "verdict": "present", "visible_text": "A", "location": _box(110, 90, 330, 180), "evidence": ""}
    ]
    verdict = parse_component_verdicts(items, ["a"], "gpt", image_size=(1100, 900))[0]
    assert verdict.verdict == Verdict.PRESENT
    assert verdict.bounds == pytest.approx((0.1, 0.1, 0.3, 0.2))
    assert parse_component_verdicts(items, ["a"], "gpt")[0].verdict == Verdict.ABSENT


class AtomicBackend:
    name = "fake-gpt"

    def structured(self, prompt, images, schema, schema_name):
        self.prompt = prompt
        return {
            "metrics": [
                {"criterion_id": "presence.elements", "expected_count": 13, "detected_issues": 5, "issues": [], "notes": ""},
                {"criterion_id": "layout.order", "expected_count": 4, "detected_issues": 1, "issues": [], "notes": ""},
            ],
            "components": [
                {"component_id": "a", "verdict": "present", "visible_text": "Alpha", "location": _box(0.1, 0.1, 0.3, 0.2), "evidence": ""},
                {"component_id": "b", "verdict": "mislabeled", "visible_text": "Layer", "location": _box(0.1, 0.4, 0.3, 0.5), "evidence": ""},
                {"component_id": "c", "verdict": "absent", "visible_text": "", "location": _box(0, 0, 0, 0), "evidence": ""},
            ],
        }


def test_vlm_judge_scores_presence_from_verdicts_not_counts(tmp_path: Path) -> None:
    order = Criterion("layout.order", Dimension.LAYOUT, "", Severity.MEDIUM, "description_order")
    inventory = {
        "components": [
            {"id": "a", "label": "Alpha", "kind": "node"},
            {"id": "b", "label": "Beta", "kind": "node"},
            {"id": "c", "label": "Gamma", "kind": "node"},
        ],
        "connections": [{"source": "a", "target": "b"}],
    }
    image = tmp_path / "candidate.png"
    image.write_bytes(b"not a real png")
    backend = AtomicBackend()
    results = _by_criterion(VLMJudge(backend).evaluate("desc", image, [PRESENCE, order, DETAILS], inventory))
    assert (results["presence.elements"].detected_issues, results["presence.elements"].expected_count) == (1, 3)
    assert (results["details.labels"].detected_issues, results["details.labels"].expected_count) == (1, 2)
    assert results["layout.order"].score == pytest.approx(0.75)
    # Presence is asked per component and is not in the counted criteria list.
    counted_part = backend.prompt.split("Counted criteria (Part B):")[1].split("Diagram description:")[0]
    assert "presence.elements" not in counted_part
    assert '"context": "feeds Beta"' in backend.prompt
    assert "counts as mislabeled" in backend.prompt


def test_repetition_marker_no_longer_takes_other_labels(tmp_path: Path) -> None:
    svg = _write_svg(
        tmp_path,
        '<rect x="100" y="20" width="200" height="50"/><text x="200" y="50" font-size="14" text-anchor="middle">Input Tokens</text>'
        '<rect x="100" y="300" width="200" height="50"/><text x="200" y="330" font-size="14" text-anchor="middle">Contextual Embeddings</text>',
    )
    inventory = {
        "components": [
            {"id": "input_tokens", "label": "Input Tokens", "kind": "input"},
            {"id": "repeat_marker", "label": "N×", "kind": "annotation"},
            {"id": "contextual_embeddings", "label": "Contextual Embeddings", "kind": "output"},
        ],
        "connections": [],
    }
    verdicts = {item.component_id: item.verdict for item in DeterministicJudge().component_verdicts(parse_svg(svg), inventory)}
    assert verdicts == {
        "input_tokens": Verdict.PRESENT,
        "repeat_marker": Verdict.ABSENT,
        "contextual_embeddings": Verdict.PRESENT,
    }


CHAIN_INVENTORY = {
    "components": [
        {"id": "alpha", "label": "Alpha", "kind": "node"},
        {"id": "beta", "label": "Beta", "kind": "node"},
        {"id": "gamma", "label": "Gamma", "kind": "node"},
    ],
    "connections": [{"source": "alpha", "target": "beta"}, {"source": "beta", "target": "gamma"}],
    "forbidden_placeholders": ["Layer"],
}

CHAIN_SVG = (
    '<rect x="150" y="20" width="100" height="40"/><text x="200" y="45" font-size="14" text-anchor="middle">Alpha</text>'
    '<rect x="150" y="160" width="100" height="40"/><text x="200" y="185" font-size="14" text-anchor="middle">Layer</text>'
    '<rect x="150" y="300" width="100" height="40"/><text x="200" y="325" font-size="14" text-anchor="middle">Gamma</text>'
    '<line x1="200" y1="60" x2="200" y2="160" stroke="#000"/>'
    '<line x1="200" y1="200" x2="200" y2="300" stroke="#000"/>'
)


def test_placeholder_box_in_the_right_position_is_mislabeled(tmp_path: Path) -> None:
    criteria = [criterion for criterion in load_rubric(ROOT / "config" / "rubric.json")]
    results = _by_criterion(DeterministicJudge().evaluate(parse_svg(_write_svg(tmp_path, CHAIN_SVG)), criteria, CHAIN_INVENTORY))
    verdicts = {item.component_id: item.verdict for item in results["presence.elements"].component_verdicts}
    assert verdicts["beta"] == Verdict.MISLABELED
    assert results["connectivity.connections"].detected_issues == 0


def test_placeholder_policy_can_count_it_as_absent(tmp_path: Path) -> None:
    criteria = [criterion for criterion in load_rubric(ROOT / "config" / "rubric.json")]
    judge = DeterministicJudge(policy=PresencePolicy(placeholder_counts_as_present=False))
    results = _by_criterion(judge.evaluate(parse_svg(_write_svg(tmp_path, CHAIN_SVG)), criteria, CHAIN_INVENTORY))
    verdicts = {item.component_id: item.verdict for item in results["presence.elements"].component_verdicts}
    assert verdicts["beta"] == Verdict.ABSENT
    assert results["connectivity.connections"].detected_issues == 2


def test_abbreviation_policy(tmp_path: Path) -> None:
    svg = parse_svg(
        _write_svg(tmp_path, '<rect x="100" y="100" width="200" height="50"/><text x="200" y="130" font-size="14" text-anchor="middle">FFN</text>')
    )
    inventory = {"components": [{"id": "feed_forward", "label": "Position-wise Feed-Forward Network", "kind": "op"}], "connections": []}
    strict = DeterministicJudge().component_verdicts(svg, inventory)[0]
    lenient = DeterministicJudge(policy=PresencePolicy(accept_abbreviations=True)).component_verdicts(svg, inventory)[0]
    assert (strict.verdict, lenient.verdict) == (Verdict.MISLABELED, Verdict.PRESENT)


def test_transformer_mixed_candidate_verdicts() -> None:
    inventory = json.loads((EXAMPLES / "expected_inventory_manual.json").read_text(encoding="utf-8"))
    criteria = load_rubric(ROOT / "config" / "rubric.json")
    results = _by_criterion(DeterministicJudge().evaluate(parse_svg(EXAMPLES / "candidate_mixed.svg"), criteria, inventory))
    verdicts = {item.component_id: item.verdict.value for item in results["presence.elements"].component_verdicts}
    assert verdicts == {
        "input_tokens": "present",
        "token_embeddings": "present",
        "positional_encoding": "absent",
        "add": "present",
        "encoder_block": "mislabeled",
        "repeat_marker": "present",
        "attention": "mislabeled",
        "norm_1": "present",
        "ffn": "mislabeled",
        "norm_2": "mislabeled",
        "contextual_embeddings": "present",
    }
    assert results["presence.elements"].score == pytest.approx(10 / 11)
    assert results["details.labels"].score == pytest.approx(6 / 10)
    # Only the three connections that are really missing: positional encoding and both residuals.
    assert results["connectivity.connections"].detected_issues == 3


def test_agreement_table_flags_disagreements() -> None:
    first = MetricScore("presence.elements", Dimension.PRESENCE, "gpt", 2, 0, component_verdicts=[
        ComponentVerdict("a", Verdict.PRESENT, "gpt"), ComponentVerdict("b", Verdict.MISLABELED, "gpt")])
    second = MetricScore("presence.elements", Dimension.PRESENCE, "svg", 2, 1, component_verdicts=[
        ComponentVerdict("a", Verdict.PRESENT, "svg"), ComponentVerdict("b", Verdict.ABSENT, "svg")])
    rows = {row["component_id"]: row for row in component_agreement([first, second])}
    assert rows["a"]["agree"] is True
    assert rows["b"]["agree"] is False
    assert rows["b"]["verdicts"] == {"gpt": "mislabeled", "svg": "absent"}
