"""Invariants for the portable Beauty-axis intervention experiment."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from diagram_correctness.beauty import BEAUTY_DIMENSIONS, judge_beauty


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "beauty_ablation", ROOT / "scripts/run_beauty_ablation.py"
)
beauty_ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(beauty_ablation)


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.calls = []

    def structured(self, prompt, images, schema, schema_name):
        self.calls.append((prompt, images, schema, schema_name))
        return {
            "observation": "Measured only the owned observable.",
            "score": 75,
            "reasoning": "Focused dimension assessment.",
        }


class BeautyAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        cls.summary = beauty_ablation.run(cls.output)
        cls.rows = {row["id"]: row for row in cls.summary["rows"]}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_each_intervention_lowers_its_target(self):
        baseline = self.rows["baseline"]["scores"]
        for variant, target in beauty_ablation.VARIANT_TARGET.items():
            if target is not None:
                with self.subTest(variant=variant):
                    self.assertLess(self.rows[variant]["scores"][target], baseline[target])

    def test_judge_uses_one_independent_structured_call_per_dimension(self):
        backend = FakeBackend()
        scores = judge_beauty(backend, "candidate.png")
        self.assertEqual(tuple(scores), BEAUTY_DIMENSIONS)
        self.assertEqual(len(backend.calls), len(BEAUTY_DIMENSIONS))
        for dimension, (prompt, images, schema, schema_name) in zip(
            BEAUTY_DIMENSIONS, backend.calls
        ):
            self.assertEqual(images, ["candidate.png"])
            self.assertEqual(schema["properties"]["score"]["maximum"], 100)
            self.assertEqual(schema_name, f"diagram_beauty_{dimension}")
            self.assertIn("MENTAL MASK", prompt)
            self.assertIn("OBSERVATION TO MAKE", prompt)
            self.assertIn(dimension.replace("_", " ").upper(), prompt)

    def test_diagrams_preserve_labels_and_connection_count(self):
        expected = set(beauty_ablation.NODE_LABELS) | {"Order Processing Flow"}
        for variant in beauty_ablation.VARIANT_TARGET:
            root = ET.parse(self.output / "diagrams" / f"{variant}.svg").getroot()
            labels = {
                " ".join(element.itertext())
                for element in root.iter()
                if element.tag.endswith("text")
            }
            edges = [element for element in root.iter() if element.get("class") == "edge"]
            with self.subTest(variant=variant):
                self.assertEqual(labels, expected)
                self.assertEqual(len(edges), 4)

    def test_historical_leaks_are_explicit(self):
        self.assertEqual(len(self.summary["warnings"]), 4)
        self.assertEqual(
            sum(warning.startswith("[LEAK]") for warning in self.summary["warnings"]),
            3,
        )
        self.assertEqual(
            sum(warning.startswith("[WEAK]") for warning in self.summary["warnings"]),
            1,
        )

    def test_dashboard_embeds_result_data_and_all_six_images(self):
        html = (self.output / "index.html").read_text(encoding="utf-8")
        payload = html.split(
            '<script id="experiment-data" type="application/json">', 1
        )[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(payload)["rows"], self.summary["rows"])
        self.assertEqual(html.count("data:image/svg+xml;base64,"), 6)

    def test_single_variable_fixture_geometry(self):
        baseline = ET.parse(self.output / "diagrams" / "baseline.svg").getroot()
        ink = ET.parse(
            self.output / "diagrams" / "degraded_ink_balance.svg"
        ).getroot()
        balance = ET.parse(
            self.output / "diagrams" / "degraded_balance.svg"
        ).getroot()

        def node_rects(root):
            return [
                next(child for child in group if child.tag.endswith("rect"))
                for group in root.iter()
                if group.get("class") == "node"
            ]

        baseline_rects = node_rects(baseline)
        ink_rects = node_rects(ink)
        balance_rects = node_rects(balance)
        self.assertEqual(
            [(rect.get("width"), rect.get("height")) for rect in ink_rects],
            [(rect.get("width"), rect.get("height")) for rect in baseline_rects],
        )
        self.assertEqual(
            [
                float(right.get("x")) - float(left.get("x"))
                for left, right in zip(balance_rects, balance_rects[1:])
            ],
            [
                float(right.get("x")) - float(left.get("x"))
                for left, right in zip(baseline_rects, baseline_rects[1:])
            ],
        )


def _metric_results(target_drop=30, unrelated_drop=0):
    baseline = {dimension: {"score": 90} for dimension in BEAUTY_DIMENSIONS}
    results = {"baseline": baseline}
    for target in BEAUTY_DIMENSIONS:
        scores = {dimension: {"score": 90} for dimension in BEAUTY_DIMENSIONS}
        scores[target]["score"] -= target_drop
        for dimension in BEAUTY_DIMENSIONS:
            if dimension != target:
                scores[dimension]["score"] -= unrelated_drop
                break
        results[f"degraded_{target}"] = scores
    return results


class IsolationMetricsTests(unittest.TestCase):
    def test_perfectly_specific_perturbations_score_100(self):
        metrics = beauty_ablation.isolation_metrics(_metric_results())
        self.assertTrue(
            all(metric["specificity"] == 100.0 for metric in metrics.values())
        )
        self.assertTrue(
            all(metric["isolation_score"] == 100.0 for metric in metrics.values())
        )
        self.assertEqual(beauty_ablation.check_isolation(_metric_results()), [])

    def test_leak_reduces_specificity(self):
        results = _metric_results(target_drop=30, unrelated_drop=20)
        metrics = beauty_ablation.isolation_metrics(results)
        self.assertTrue(
            all(metric["specificity"] == 60.0 for metric in metrics.values())
        )
        self.assertTrue(
            all(metric["isolation_score"] == 60.0 for metric in metrics.values())
        )
        self.assertEqual(len(beauty_ablation.check_isolation(results)), 5)

    def test_low_target_sensitivity_cannot_look_isolated(self):
        results = _metric_results(target_drop=5)
        metrics = beauty_ablation.isolation_metrics(results)
        self.assertTrue(
            all(metric["specificity"] == 100.0 for metric in metrics.values())
        )
        self.assertTrue(
            all(
                33.0 <= metric["isolation_score"] <= 34.0
                for metric in metrics.values()
            )
        )
        self.assertEqual(len(beauty_ablation.check_isolation(results)), 5)


if __name__ == "__main__":
    unittest.main()
