"""Scientific invariants for the intervention experiment (stdlib test runner)."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ablation", ROOT / "scripts/run_dimension_ablation.py")
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


class AblationExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        cls.source = ablation.EXAMPLE / "candidate_strong.svg"
        cls.original_bytes = cls.source.read_bytes()
        cls.summary = ablation.run(cls.output)
        cls.rows = {row["id"]: row for row in cls.summary["rows"]}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_original_is_byte_identical_and_history_is_not_used_as_control(self):
        self.assertEqual(self.source.read_bytes(), self.original_bytes)
        self.assertEqual((self.output / "diagrams/baseline.svg").read_bytes(), self.original_bytes)
        self.assertFalse(self.summary["historical"]["used_for_deltas"])
        self.assertEqual(len(self.rows), 6)
        self.assertEqual(self.rows["baseline"]["overall"], 1.0)

    def test_all_interventions_lower_the_target_and_weighted_deltas_reconcile(self):
        base = self.rows["baseline"]
        for target in ablation.DIMENSIONS:
            with self.subTest(target=target):
                row = self.rows[target]
                self.assertLess(row["scores"][target], base["scores"][target])
                for dimension in ablation.DIMENSIONS:
                    self.assertAlmostEqual(row["delta"][dimension], row["scores"][dimension] - base["scores"][dimension])
                self.assertAlmostEqual(row["overall_delta"], sum(self.summary["weights"][d] * row["delta"][d] for d in ablation.DIMENSIONS))

    def test_layout_preserves_connections_and_presence_exposes_dangling_arrow(self):
        layout = {m["id"]: m for m in self.rows["layout"]["criteria"]}
        self.assertEqual(layout["layout.order"]["issues"], 2)
        self.assertEqual(layout["connectivity.connections"]["issues"], 0)
        self.assertEqual(layout["connectivity.endpoints"]["issues"], 0)
        presence = {m["id"]: m for m in self.rows["presence"]["criteria"]}
        self.assertEqual(presence["presence.elements"]["issues"], 2)
        self.assertEqual(presence["details.labels"]["expected"], 9)
        self.assertEqual(presence["connectivity.connections"]["issues"], 1)
        self.assertEqual(presence["connectivity.endpoints"]["issues"], 1)
        self.assertEqual(presence["connectivity.endpoints"]["evidence"][0]["element_ids"], ["s3"])

    def test_font_and_label_interventions_do_not_change_geometry_or_arrows(self):
        def elements(path):
            return {element.get("id"): element for element in ET.parse(path).getroot().iter() if element.get("id")}
        original = elements(self.source)
        for target in ("details", "legibility"):
            edited = elements(self.output / f"diagrams/{target}.svg")
            self.assertEqual(set(original), set(edited))
            for key, element in original.items():
                changed = edited[key]
                self.assertEqual(element.tag, changed.tag)
                if target == "details":
                    self.assertEqual(element.attrib, changed.attrib)
                    if not element.tag.endswith("}text"):
                        self.assertEqual(element.text, changed.text)
                else:
                    self.assertEqual(element.text, changed.text)
                    self.assertEqual({k: v for k, v in element.attrib.items() if k != "font-size"}, {k: v for k, v in changed.attrib.items() if k != "font-size"})

    def test_csv_and_portable_html_use_the_actual_result_payload(self):
        with (self.output / "scores.csv").open(newline="") as handle:
            exported = list(csv.DictReader(handle))
        for csv_row in exported:
            row = self.rows[csv_row["variant"]]
            self.assertAlmostEqual(float(csv_row["overall"]), row["overall"])
        html = (self.output / "index.html").read_text()
        payload = html.split('<script id="experiment-data" type="application/json">', 1)[1].split('</script>', 1)[0]
        embedded = json.loads(payload)
        self.assertEqual(embedded["rows"], self.summary["rows"])
        self.assertEqual(len(embedded["images"]), 6)
        self.assertNotIn("__EXPERIMENT_DATA__", html)
        self.assertTrue(all(value.startswith("data:image/svg+xml;base64,") for value in embedded["images"].values()))


if __name__ == "__main__":
    unittest.main()
