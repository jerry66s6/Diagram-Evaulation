"""Invariants for the Beauty v5 dimensions and the deterministic Canvas Fill check."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

from diagram_correctness.beauty_v5 import (
    BEAUTY_V5_DIMENSIONS,
    BEAUTY_V5_MODEL_DIMENSIONS,
    BEAUTY_V5_PROMPTS,
    judge_beauty_v5,
)
from diagram_correctness.canvas_fill import footprint_percent, judge_canvas_fill, score_footprint


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("beauty_v5_ablation", ROOT / "scripts/run_beauty_v5_ablation.py")
v5 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v5
spec.loader.exec_module(v5)


class ScriptedBackend:
    """High scores everywhere except the model-judged dimension each fixture targets."""

    name = "scripted"

    def structured(self, prompt, images, schema, schema_name):
        dimension = schema_name.removeprefix("diagram_beauty_")
        target = v5.VARIANT_TARGET.get(Path(images[0]).stem)
        return {"observation": "Scripted.", "score": 40 if target == dimension else 92, "reasoning": "Scripted."}


class CanvasFillTests(unittest.TestCase):
    def _image(self, boxes, outline_only=False, background="white"):
        image = Image.new("RGB", (1200, 750), background)
        draw = ImageDraw.Draw(image)
        for box in boxes:
            if outline_only:
                draw.rectangle(box, outline="black", width=3)
            else:
                draw.rectangle(box, fill="#1f5fa8")
        draw.line([(0, 375), (1200, 375)], fill="black", width=4)  # thin arrow-like line is ignored
        return image

    def test_outline_boxes_count_by_area_and_thin_lines_are_ignored(self):
        filled = footprint_percent(self._image([(100, 100, 380, 260)]))
        outlined = footprint_percent(self._image([(100, 100, 380, 260)], outline_only=True))
        self.assertAlmostEqual(filled, 100 * 280 * 160 / (1200 * 750), delta=0.5)
        self.assertAlmostEqual(outlined, filled, delta=0.5)

    def test_large_enclosed_regions_are_not_counted(self):
        # A ring of thin lines enclosing ~25% of the canvas (like a cycle of arrows) is not an element.
        image = Image.new("RGB", (1200, 750), "white")
        ImageDraw.Draw(image).rectangle((300, 150, 900, 600), outline="black", width=4)
        self.assertLess(footprint_percent(image), 1.0)

    def test_dark_background_is_detected(self):
        image = Image.new("RGB", (1200, 750), "black")
        ImageDraw.Draw(image).rectangle((100, 100, 400, 400), fill="#ff00ff")
        self.assertAlmostEqual(footprint_percent(image), 10.0, delta=0.5)

    def test_score_is_one_sided(self):
        self.assertEqual(score_footprint(1.0), 0)
        self.assertEqual(score_footprint(4.5), 50)
        self.assertEqual(score_footprint(10.0), 100)
        self.assertEqual(score_footprint(60.0), 100)


class BeautyV5FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        output = Path(cls.temp.name)
        cls.scenes = {v: v5.build_scene(v) for v in v5.VARIANT_TARGET}
        cls.paths = {v: v5.render(scene, output / f"{v}.png") for v, scene in cls.scenes.items()}
        cls.measurements = {v: v5.measure(cls.scenes[v], cls.paths[v]) for v in v5.VARIANT_TARGET}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_baseline_is_good_on_every_measurement(self):
        base = self.measurements["baseline"]
        self.assertLessEqual(base["worst_margin_ratio"], 1.05)
        self.assertGreaterEqual(base["min_gap_to_node_pct"], 40)
        self.assertTrue(base["roles_consistent"])
        self.assertGreaterEqual(base["min_box_contrast"], 4.5)
        self.assertEqual(base["max_row_col_offset_pct"], 0)
        self.assertEqual(base["style_variants_per_role"], 1)
        self.assertEqual(base["arrow_treatments"], 1)
        self.assertEqual(judge_canvas_fill(self.paths["baseline"]).score, 100)

    def test_each_new_fixture_moves_its_own_measurement(self):
        self.assertGreater(self.measurements["degraded_alignment"]["max_row_col_offset_pct"], 15)
        self.assertGreater(self.measurements["degraded_style_uniformity"]["style_variants_per_role"], 2)
        self.assertGreater(self.measurements["degraded_style_uniformity"]["arrow_treatments"], 2)
        self.assertLess(judge_canvas_fill(self.paths["degraded_canvas_fill"]).score, 50)

    def test_canvas_fill_ignores_every_other_fixture(self):
        for variant in v5.VARIANT_TARGET:
            if variant == "degraded_canvas_fill":
                continue
            with self.subTest(variant=variant):
                self.assertEqual(judge_canvas_fill(self.paths[variant]).score, 100)

    def test_new_fixtures_keep_v4_measurements_intact(self):
        base = self.measurements["baseline"]
        for variant in ("degraded_alignment", "degraded_style_uniformity", "degraded_canvas_fill"):
            measured = self.measurements[variant]
            with self.subTest(variant=variant):
                self.assertLessEqual(measured["worst_margin_ratio"], 1.15)
                self.assertGreaterEqual(measured["min_gap_to_node_pct"], 40)
                self.assertEqual(measured["hue_families"], base["hue_families"])
                self.assertTrue(measured["roles_consistent"])
                self.assertAlmostEqual(measured["min_box_contrast"], base["min_box_contrast"], delta=0.25)
                self.assertEqual(measured["arrow_contrast"], base["arrow_contrast"])

    def test_v4_fixtures_keep_new_measurements_intact(self):
        for variant in ("degraded_centering", "degraded_space_use", "degraded_color_consistency", "degraded_contrast"):
            measured = self.measurements[variant]
            with self.subTest(variant=variant):
                self.assertEqual(measured["max_row_col_offset_pct"], 0)
                self.assertEqual(measured["style_variants_per_role"], 1)
                self.assertEqual(measured["arrow_treatments"], 1)


class BeautyV5PromptTests(unittest.TestCase):
    def test_prompts_name_every_sibling_as_not_owned(self):
        for dimension, prompt in BEAUTY_V5_PROMPTS.items():
            not_yours = prompt.split("NOT YOURS:", 1)[1].split("\n", 1)[0]
            with self.subTest(dimension=dimension):
                self.assertIn("MENTAL MASK", prompt)
                for sibling, label in v5.DIMENSION_LABELS.items():
                    if sibling != dimension:
                        self.assertIn(f"({label})", not_yours)

    def test_judge_returns_all_seven_dimensions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = v5.render(v5.build_scene("baseline"), Path(temp) / "baseline.png")
            scores = judge_beauty_v5(ScriptedBackend(), path)
        self.assertEqual(tuple(scores), BEAUTY_V5_DIMENSIONS)
        self.assertEqual(scores["canvas_fill"].score, 100)
        self.assertTrue(all(scores[d].score == 92 for d in BEAUTY_V5_MODEL_DIMENSIONS))

    def test_scored_run_reports_perfect_isolation_for_ideal_judge(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = v5.run(Path(temp), repeats=1, workers=4, backend=ScriptedBackend())
        self.assertEqual(summary["warnings"], [])
        self.assertEqual(summary["off_target_mean_abs"], 0)
        self.assertEqual(summary["overall_isolation"], 100.0)


if __name__ == "__main__":
    unittest.main()
