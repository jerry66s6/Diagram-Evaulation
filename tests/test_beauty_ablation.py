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
        return {"score": 75, "reasoning": "Focused dimension assessment."}


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
        self.assertEqual(len(self.summary["warnings"]), 3)
        self.assertTrue(
            all(warning.startswith("[LEAK]") for warning in self.summary["warnings"])
        )

    def test_dashboard_embeds_result_data_and_all_six_images(self):
        html = (self.output / "index.html").read_text(encoding="utf-8")
        payload = html.split(
            '<script id="experiment-data" type="application/json">', 1
        )[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(payload)["rows"], self.summary["rows"])
        self.assertEqual(html.count("data:image/svg+xml;base64,"), 6)


if __name__ == "__main__":
    unittest.main()
