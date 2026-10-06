import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

from diagram_correctness.models import Criterion, Dimension, MetricScore, Severity, aggregate_panel
from diagram_correctness.sensitivity import SensitivityJudge
from diagram_correctness.svg import parse_svg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("ablation_v2", ROOT / "scripts/run_ablation_v2.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    output = tmp_path_factory.mktemp("sensitivity")
    summary, code = experiment.run(output)
    assert code == 0
    return output, summary, {row["id"]: row for row in summary["rows"]}


def test_original_unchanged_and_target_scores_follow_edit_strength(run):
    output, summary, rows = run
    assert (output / "diagrams/baseline.svg").read_bytes() == (experiment.EXAMPLE / "candidate_strong.svg").read_bytes()
    assert len(rows) == 11
    for dimension in experiment.DIMENSIONS:
        values = [rows[name]["panels"]["svg"]["dimensions"][dimension]["score"]
                  for name in ["baseline", dimension + "_partial", dimension + "_severe"]]
        assert values[0] == 1
        assert values[0] > values[1] > values[2]
    # These are fully failed criteria with real edits, not manually assigned cells.
    for dimension in ["layout", "connectivity", "presence", "legibility"]:
        assert rows[dimension + "_severe"]["panels"]["svg"]["dimensions"][dimension]["score"] == 0
    assert rows["details_severe"]["panels"]["svg"]["dimensions"]["details"]["score"] == pytest.approx(2 / 11)


def test_placement_and_label_edits_preserve_component_roles_and_connections(run):
    _, _, rows = run
    for name in ["layout_partial", "layout_severe", "details_partial", "details_severe"]:
        dimensions = rows[name]["panels"]["svg"]["dimensions"]
        assert dimensions["presence"]["score"] == 1
        assert dimensions["connectivity"]["score"] == 1


def test_pending_gpt_is_not_fabricated_or_substituted_with_svg(run):
    output, summary, rows = run
    for row in rows.values():
        assert row["panels"]["gpt"] is None
        assert row["panels"]["combined"] is None
    html = (output / "index.html").read_text()
    data = json.loads(html.split('<script id="experiment-data" type="application/json">')[1].split('</script>')[0])
    assert data["rows"] == json.loads(json.dumps(summary["rows"]))
    assert len(data["images"]) == 11
    assert all(image.startswith("data:image/svg+xml;base64,") for image in data["images"].values())


def test_complete_role_recovery_does_not_use_svg_element_ids(run, tmp_path):
    output, _, _ = run
    tree = ET.parse(output / "diagrams/details_severe.svg")
    for number, element in enumerate(tree.getroot().iter()):
        if element.get("id") and element.get("id") != "arrow":
            element.set("id", f"arbitrary_{number}")
    renamed = tmp_path / "renamed.svg"
    tree.write(renamed, encoding="unicode")
    inventory = json.loads((output / "description_expectations.json").read_text())
    matches = SensitivityJudge().match_components(parse_svg(renamed), inventory)
    assert len(matches) == 11
    assert sum(m.method == "role" for m in matches.values()) == 9


def test_minimum_rule_exposes_failure_and_preserves_between_judge_disagreement():
    criteria = [Criterion("legibility.font", Dimension.LEGIBILITY, "", Severity.HIGH),
                Criterion("legibility.overflow", Dimension.LEGIBILITY, "", Severity.HIGH)]
    svg = [MetricScore(c.id, c.dimension, "svg", 10, 10 if c.id.endswith("font") else 0) for c in criteria]
    gpt = [MetricScore(c.id, c.dimension, "gpt", 10, 0) for c in criteria]
    assert aggregate_panel(svg, criteria)[Dimension.LEGIBILITY].score == .5
    assert aggregate_panel(svg, criteria, "minimum")[Dimension.LEGIBILITY].score == 0
    assert aggregate_panel(svg + gpt, criteria, "minimum")[Dimension.LEGIBILITY].score == .5
    with pytest.raises(ValueError):
        aggregate_panel(svg, criteria, "arbitrary")


def small_svg(tmp_path, shapes):
    path = tmp_path / "candidate.svg"
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200">' + shapes + '</svg>')
    return parse_svg(path)


def metric(document, inventory, name, function):
    criterion = Criterion(name, Dimension.LAYOUT, "", Severity.HIGH, function)
    return SensitivityJudge().evaluate(document, [criterion], inventory)[0]


def test_explicit_horizontal_sequence_and_unspecified_direction(tmp_path):
    document = small_svg(tmp_path, '<rect id="a" x="10" y="60" width="60" height="40"/><text x="20" y="85">A</text>'
                         '<rect id="b" x="200" y="10" width="60" height="40"/><text x="210" y="35">B</text>')
    inventory = {"components": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}], "connections": [{"source": "a", "target": "b"}]}
    assert metric(document, inventory, "layout.order", "description_order").score is None
    inventory["spatial_requirements"] = {"sequences": [{"axis": "left_to_right", "components": ["a", "b"]}]}
    assert metric(document, inventory, "layout.order", "description_order").score == 1
    inventory["spatial_requirements"]["sequences"][0]["components"].reverse()
    assert metric(document, inventory, "layout.order", "description_order").score == 0


def test_group_containment_is_checked_from_explicit_requirement(tmp_path):
    document = small_svg(tmp_path, '<rect id="group" x="10" y="10" width="120" height="120"/><text x="20" y="30">Group</text>'
                         '<rect id="a" x="190" y="40" width="80" height="40"/><text x="205" y="65">A</text>')
    inventory = {"components": [{"id": "g", "label": "Group", "kind": "group"}, {"id": "a", "label": "A"}],
                 "spatial_requirements": {"groups": [{"container": "g", "members": ["a"]}]}}
    result = metric(document, inventory, "layout.containment", "group_containment")
    assert result.expected_count == result.detected_issues == 1


def test_accidental_enclosure_does_not_become_intentional_group(tmp_path):
    document = small_svg(tmp_path, '<rect id="a" x="10" y="10" width="150" height="140"/><text x="20" y="35">A</text>'
                         '<rect id="b" x="50" y="70" width="60" height="40"/><text x="60" y="95">B</text>')
    inventory = {"components": [{"id": "a", "label": "A", "kind": "node"}, {"id": "b", "label": "B", "kind": "node"}]}
    result = metric(document, inventory, "layout.overlap", "unintended_overlap")
    assert result.expected_count == result.detected_issues == 2


def test_symmetric_unlabeled_graph_does_not_invent_unique_roles(tmp_path):
    document = small_svg(tmp_path, '<rect id="x" x="10" y="10" width="60" height="40"/><text x="20" y="35">TBD</text>'
                         '<rect id="y" x="200" y="10" width="60" height="40"/><text x="210" y="35">TBD</text>'
                         '<line x1="70" y1="20" x2="200" y2="20"/><line x1="200" y1="40" x2="70" y2="40"/>')
    inventory = {"components": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
                 "connections": [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}]}
    assert not SensitivityJudge().match_components(document, inventory)
