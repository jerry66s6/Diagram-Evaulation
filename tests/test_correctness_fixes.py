"""Tests for the correctness-metric fixes: precision, connection labels, decoupled
dimensions, judge-independent denominators, repeated sampling, and SVG parsing."""

import itertools
import threading
from pathlib import Path

import pytest

from diagram_correctness.deterministic import DeterministicJudge
from diagram_correctness.labels import labels_match
from diagram_correctness.models import Criterion, Dimension, MetricScore, Severity
from diagram_correctness.pipeline import CorrectnessPipeline, PipelineConfig
from diagram_correctness.rubric import load_rubric, severity_weights
from diagram_correctness.svg import parse_svg
from diagram_correctness.vlm import VLMJudge

ROOT = Path(__file__).resolve().parents[1]
RUBRIC = load_rubric(ROOT / "config" / "rubric.json")
ARROW = 'marker-end="url(#a)"'


def _svg(tmp_path: Path, body: str, root: str = 'width="400" height="400"') -> Path:
    path = tmp_path / "candidate.svg"
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" {root}>{body}</svg>', encoding="utf-8")
    return path


def _box(name: str, x: float, y: float, label: str | None = None) -> str:
    text = label if label is not None else name
    return (
        f'<rect id="{name}" x="{x}" y="{y}" width="80" height="40" fill="#fff" stroke="#000"/>'
        f'<text x="{x + 40}" y="{y + 25}" font-size="14" text-anchor="middle">{text}</text>'
    )


def _inventory(names: list[str], edges: list[tuple[str, str, str]]) -> dict:
    return {
        "components": [{"id": name.lower(), "label": name, "kind": "node"} for name in names],
        "connections": [{"source": s.lower(), "target": t.lower(), "label": label} for s, t, label in edges],
        "forbidden_placeholders": [],
    }


def _deterministic(path: Path, inventory: dict) -> dict[str, MetricScore]:
    return {metric.criterion_id: metric for metric in DeterministicJudge().evaluate(parse_svg(path), RUBRIC, inventory)}


# --- Connectivity precision and direction -------------------------------------------------

THREE_IN_A_ROW = _box("Alpha", 20, 20) + _box("Beta", 160, 20) + _box("Gamma", 300, 20)


def test_reversed_and_extra_arrows_fail_precision_but_not_recall(tmp_path: Path) -> None:
    body = THREE_IN_A_ROW + (
        f'<line id="ab" x1="100" y1="40" x2="160" y2="40" stroke="#000" {ARROW}/>'
        f'<line id="cb" x1="300" y1="40" x2="240" y2="40" stroke="#000" {ARROW}/>'  # Gamma -> Beta: reversed
        f'<polyline id="ac" points="60,60 60,120 340,120 340,60" fill="none" stroke="#000" {ARROW}/>'  # extra
    )
    inventory = _inventory(["Alpha", "Beta", "Gamma"], [("Alpha", "Beta", ""), ("Beta", "Gamma", "")])
    results = _deterministic(_svg(tmp_path, body), inventory)
    assert (results["connectivity.connections"].detected_issues, results["connectivity.connections"].expected_count) == (0, 2)
    endpoints = results["connectivity.endpoints"]
    assert (endpoints.detected_issues, endpoints.expected_count) == (2, 3)
    assert {tuple(issue.element_ids) for issue in endpoints.issues} >= {("ac",)}


def test_marker_start_points_the_arrow_back_to_the_first_point(tmp_path: Path) -> None:
    # Drawn from Beta to Alpha, but the arrowhead is at the start, so it points at Beta.
    body = _box("Alpha", 20, 20) + _box("Beta", 160, 20) + (
        '<line x1="160" y1="40" x2="100" y2="40" stroke="#000" marker-start="url(#a)"/>'
    )
    results = _deterministic(_svg(tmp_path, body), _inventory(["Alpha", "Beta"], [("Alpha", "Beta", "")]))
    assert results["connectivity.endpoints"].detected_issues == 0
    assert results["connectivity.connections"].detected_issues == 0


def test_decorative_lines_are_not_arrows(tmp_path: Path) -> None:
    body = THREE_IN_A_ROW + (
        f'<line x1="100" y1="40" x2="160" y2="40" stroke="#000" {ARROW}/>'
        '<line x1="25" y1="45" x2="95" y2="45" stroke="#999"/>'  # divider inside Alpha
        '<line x1="20" y1="380" x2="380" y2="380" stroke="#999"/>'  # footer rule touching nothing
    )
    results = _deterministic(_svg(tmp_path, body), _inventory(["Alpha", "Beta", "Gamma"], [("Alpha", "Beta", "")]))
    assert (results["connectivity.endpoints"].detected_issues, results["connectivity.endpoints"].expected_count) == (0, 1)


# --- Connection labels -----------------------------------------------------------------

def _decision(yes_label: str, no_label: str) -> str:
    return (
        _box("Check", 160, 20) + _box("Accept", 40, 300) + _box("Reject", 280, 300)
        + f'<line x1="200" y1="60" x2="80" y2="300" stroke="#000" {ARROW}/>'
        + f'<line x1="200" y1="60" x2="320" y2="300" stroke="#000" {ARROW}/>'
        + f'<text x="140" y="185" font-size="12" text-anchor="middle">{yes_label}</text>'
        + f'<text x="260" y="185" font-size="12" text-anchor="middle">{no_label}</text>'
    )


DECISION = _inventory(["Check", "Accept", "Reject"], [("Check", "Accept", "YES"), ("Check", "Reject", "NO")])


def test_swapped_branch_labels_fail_connection_labels(tmp_path: Path) -> None:
    right = _deterministic(_svg(tmp_path, _decision("YES", "NO")), DECISION)["details.connection_labels"]
    swapped = _deterministic(_svg(tmp_path, _decision("NO", "YES")), DECISION)["details.connection_labels"]
    assert (right.detected_issues, right.expected_count) == (0, 2)
    assert (swapped.detected_issues, swapped.expected_count) == (2, 2)
    # Connectivity alone cannot see the swap.
    assert _deterministic(_svg(tmp_path, _decision("NO", "YES")), DECISION)["connectivity.connections"].score == 1


def test_label_matching_is_tolerant_of_notation_but_not_of_meaning() -> None:
    assert labels_match("Error 1 ≤ ε₁", "Error 1 <= epsilon_1")
    assert labels_match("yes", "YES")
    assert not labels_match("YES", "NO")
    assert not labels_match("Error 2", "Error 1")


# --- Presence precision ----------------------------------------------------------------

def test_unexpected_boxes_count_against_presence_precision(tmp_path: Path) -> None:
    body = _box("Alpha", 20, 20) + _box("Beta", 160, 20) + _box("Logging", 160, 200)
    results = _deterministic(_svg(tmp_path, body), _inventory(["Alpha", "Beta"], []))
    unexpected = results["presence.unexpected"]
    assert (unexpected.detected_issues, unexpected.expected_count) == (1, 3)
    assert "Logging" in unexpected.issues[0].description
    assert results["presence.elements"].score == 1


# --- Layout --------------------------------------------------------------------------------

def test_order_infers_the_reading_direction_and_skips_loops(tmp_path: Path) -> None:
    loop = THREE_IN_A_ROW + (
        f'<line x1="100" y1="40" x2="160" y2="40" stroke="#000" {ARROW}/>'
        f'<line x1="240" y1="40" x2="300" y2="40" stroke="#000" {ARROW}/>'
        f'<polyline points="340,60 340,120 60,120 60,60" fill="none" stroke="#000" {ARROW}/>'
    )
    inventory = _inventory(["Alpha", "Beta", "Gamma"], [("Alpha", "Beta", ""), ("Beta", "Gamma", ""), ("Gamma", "Alpha", "")])
    order = _deterministic(_svg(tmp_path, loop), inventory)["layout.order"]
    assert (order.detected_issues, order.expected_count) == (0, 2)
    assert "left-to-right" in order.notes

    backwards = _box("Alpha", 20, 20) + _box("Beta", 160, 20) + _box("Gamma", 300, 20) + _box("Delta", 160, 200)
    inventory = _inventory(
        ["Alpha", "Beta", "Gamma", "Delta"], [("Alpha", "Beta", ""), ("Beta", "Gamma", ""), ("Gamma", "Delta", "")]
    )
    # Gamma -> Delta runs left, against the left-to-right direction of the other two.
    order = _deterministic(_svg(tmp_path, backwards), inventory)["layout.order"]
    assert (order.detected_issues, order.expected_count) == (1, 3)


def test_overlap_counts_each_colliding_shape_once_and_ignores_arrowheads(tmp_path: Path) -> None:
    body = (
        '<rect x="100" y="100" width="100" height="100"/>'
        '<rect x="180" y="100" width="60" height="40"/>'
        '<rect x="180" y="160" width="60" height="40"/>'
        '<rect x="60" y="180" width="60" height="40"/>'
        '<rect x="300" y="300" width="40" height="40"/>'
        '<rect x="20" y="20" width="40" height="40"/>'
        '<polygon points="300,320 290,315 290,325"/>'  # arrowhead touching a box
    )
    overlap = _deterministic(_svg(tmp_path, body), _inventory([], []))["layout.overlap"]
    # Three pairs collide, involving four distinct shapes out of six.
    assert (overlap.detected_issues, overlap.expected_count) == (4, 6)


# --- Legibility ------------------------------------------------------------------------

def test_font_size_is_inherited_and_measured_at_rendered_size(tmp_path: Path) -> None:
    inherited = '<g font-size="8"><text x="10" y="20">tiny</text></g><text x="10" y="60" font-size="14">fine</text>'
    font = _deterministic(_svg(tmp_path, inherited), _inventory([], []))["legibility.minimum_font"]
    assert (font.detected_issues, font.expected_count) == (1, 2)

    # 14 user units drawn at half size render at 7 px.
    shrunk = '<text x="10" y="20" font-size="14">shrunk</text>'
    root = 'width="400" height="200" viewBox="0 0 800 400"'
    assert _deterministic(_svg(tmp_path, shrunk, root), _inventory([], []))["legibility.minimum_font"].detected_issues == 1
    # 8 user units drawn at double size render at 16 px.
    root = 'width="1600" height="800" viewBox="0 0 800 400"'
    small = '<text x="10" y="20" font-size="8">enlarged</text>'
    assert _deterministic(_svg(tmp_path, small, root), _inventory([], []))["legibility.minimum_font"].detected_issues == 0


def test_text_outside_boxes_must_not_collide(tmp_path: Path) -> None:
    body = '<text x="100" y="100" font-size="14">first label</text><text x="105" y="102" font-size="14">second label</text>'
    overflow = _deterministic(_svg(tmp_path, body), _inventory([], []))["legibility.overflow"]
    assert (overflow.detected_issues, overflow.expected_count) == (2, 2)


# --- GPT judge -------------------------------------------------------------------------

def _loc(x0: float, y0: float, x1: float, y1: float) -> dict:
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


GPT_INVENTORY = {
    "components": [{"id": key, "label": label, "kind": "node"} for key, label in
                   [("a", "Alpha"), ("b", "Beta"), ("c", "Gamma"), ("d", "Delta")]],
    "connections": [
        {"source": "a", "target": "b", "label": "YES"},
        {"source": "b", "target": "c", "label": ""},
        {"source": "c", "target": "d", "label": ""},
        {"source": "a", "target": "d", "label": ""},
    ],
}


def _gpt_answer(a_verdict: str = "present", overlap: int = 9) -> dict:
    return {
        "components": [
            {"component_id": "a", "verdict": a_verdict, "visible_text": "Alpha", "label_legible": True, "location": _loc(0.1, 0.1, 0.3, 0.2), "evidence": ""},
            {"component_id": "b", "verdict": "present", "visible_text": "Beta", "label_legible": True, "location": _loc(0.4, 0.1, 0.6, 0.2), "evidence": ""},
            # Drawn, but the text is unreadable: not a Details failure, a Legibility one.
            {"component_id": "c", "verdict": "mislabeled", "visible_text": "Gam", "label_legible": False, "location": _loc(0.7, 0.1, 0.9, 0.2), "evidence": ""},
            {"component_id": "d", "verdict": "absent", "visible_text": "", "label_legible": True, "location": _loc(0, 0, 0, 0), "evidence": ""},
        ],
        "connections": [
            {"connection_id": "C1", "verdict": "present", "visible_label": "NO", "label_legible": True, "evidence": ""},
            {"connection_id": "C2", "verdict": "reversed", "visible_label": "", "label_legible": True, "evidence": ""},
            {"connection_id": "C3", "verdict": "absent", "visible_label": "", "label_legible": True, "evidence": ""},
            {"connection_id": "C4", "verdict": "absent", "visible_label": "", "label_legible": True, "evidence": ""},
        ],
        "extra_components": [
            {"visible_text": "Logger", "location": _loc(0.1, 0.7, 0.3, 0.8), "evidence": ""},
            # Same box as Alpha: a duplicate listing, not an extra component.
            {"visible_text": "Alpha", "location": _loc(0.1, 0.1, 0.3, 0.2), "evidence": ""},
        ],
        "extra_connections": [{"from_text": "Alpha", "to_text": "Gamma", "evidence": ""}],
        "metrics": [
            {"criterion_id": "layout.order", "detected_issues": 0, "issues": [], "notes": ""},
            {"criterion_id": "layout.canvas", "detected_issues": 0, "issues": [], "notes": ""},
            {"criterion_id": "layout.overlap", "detected_issues": overlap, "issues": [], "notes": ""},
            {"criterion_id": "legibility.overflow", "detected_issues": 0, "issues": [], "notes": ""},
        ],
    }


class ScriptedBackend:
    name = "fake-gpt"

    def __init__(self, answers: list[dict]) -> None:
        self.answers = itertools.cycle(answers)
        self.lock = threading.Lock()
        self.calls = 0
        self.prompt = ""

    def structured(self, prompt, images, schema, schema_name):
        with self.lock:
            self.calls += 1
            self.prompt = prompt
            return next(self.answers)


def _gpt(tmp_path: Path, answers: list[dict], samples: int = 1) -> tuple[dict[str, MetricScore], ScriptedBackend]:
    image = tmp_path / "candidate.png"
    image.write_bytes(b"not a real png")
    backend = ScriptedBackend(answers)
    metrics = VLMJudge(backend, samples=samples).evaluate("desc", image, RUBRIC, GPT_INVENTORY)
    return {metric.criterion_id: metric for metric in metrics}, backend


def _counts(metric: MetricScore) -> tuple[int, int]:
    return metric.detected_issues, metric.expected_count


def test_gpt_scores_every_criterion_from_item_verdicts(tmp_path: Path) -> None:
    results, backend = _gpt(tmp_path, [_gpt_answer()])
    assert _counts(results["presence.elements"]) == (1, 4)  # Delta is absent
    assert _counts(results["presence.unexpected"]) == (1, 4)  # Logger; the Alpha duplicate is dropped
    assert _counts(results["details.labels"]) == (0, 2)  # Gamma is unreadable, so not judged here
    assert _counts(results["legibility.minimum_font"]) == (1, 4)  # ...but here
    # C3 and C4 end at the missing Delta and are paid for under Presence only.
    assert _counts(results["connectivity.connections"]) == (0, 2)
    assert _counts(results["connectivity.endpoints"]) == (2, 3)  # reversed C2 and the extra arrow
    assert _counts(results["details.connection_labels"]) == (1, 1)  # YES expected, NO drawn
    # The model said 9 overlaps, but only three components are drawn.
    assert _counts(results["layout.overlap"]) == (3, 3)
    assert _counts(results["layout.order"]) == (0, 2)
    assert "Connection checklist (Part B)" in backend.prompt
    assert "a crossing arrow is not a missing connection" in backend.prompt


def test_gpt_samples_use_majority_verdicts_and_median_counts(tmp_path: Path) -> None:
    answers = [_gpt_answer("present", 0), _gpt_answer("absent", 2), _gpt_answer("present", 1)]
    results, backend = _gpt(tmp_path, answers, samples=3)
    assert backend.calls == 3
    assert _counts(results["presence.elements"]) == (1, 4)  # Alpha: present in 2 of 3
    assert results["layout.overlap"].detected_issues == 1  # median of 0, 2, 1
    assert "Scores across 3 samples" in results["presence.elements"].notes


# --- Aggregation ---------------------------------------------------------------------------

def test_weights_follow_rubric_severity() -> None:
    weights = severity_weights(RUBRIC)
    assert weights[Dimension.CONNECTIVITY] == pytest.approx(1 / 3)
    assert weights[Dimension.PRESENCE] == pytest.approx(1 / 6)  # its medium criterion does not dilute it


class FixedJudge:
    def __init__(self, name: str, detected: int) -> None:
        self.name, self.detected = name, detected

    def evaluate(self, description, candidate_image, criteria, inventory):
        return [MetricScore(c.id, c.dimension, self.name, 10, self.detected) for c in criteria]


class EmptyExtractor:
    class backend:
        name = "none"

    def extract(self, description, criteria):
        return {"components": [], "connections": []}


def test_pipeline_reports_judge_disagreements_and_uses_severity_weights(tmp_path: Path) -> None:
    image = tmp_path / "candidate.png"
    image.write_bytes(b"x")
    criteria = [
        Criterion("p", Dimension.PRESENCE, "", Severity.HIGH),
        Criterion("c", Dimension.CONNECTIVITY, "", Severity.CRITICAL),
    ]
    report = CorrectnessPipeline(
        PipelineConfig(criteria=criteria, judges=[FixedJudge("a", 0), FixedJudge("b", 5)], expectation_extractor=EmptyExtractor())  # type: ignore[list-item]
    ).run("desc", candidate_image=image, run_deterministic=False)
    assert {row["criterion_id"] for row in report.metadata["criterion_disagreements"]} == {"p", "c"}
    assert report.weights[Dimension.CONNECTIVITY] == pytest.approx(2 / 3)
    assert report.correctness == pytest.approx(0.75)
