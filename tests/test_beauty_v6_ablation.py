"""Invariants for the Beauty v6 dimensions and their single-variable fixtures."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

from diagram_correctness.beauty_v6 import (
    BEAUTY_V6_DIMENSIONS,
    BEAUTY_V6_MODEL_DIMENSIONS,
    BEAUTY_V6_PROMPTS,
    judge_beauty_v6,
)
from diagram_correctness.canvas_fill import judge_canvas_fill


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("beauty_v6_ablation", ROOT / "scripts/run_beauty_v6_ablation.py")
v6 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v6
spec.loader.exec_module(v6)

# Measurement groups owned by each dimension; every other group must match the baseline.
GROUPS = {
    "centering": ("worst_margin_ratio", "label_anchor", "row_gap_ratio"),
    "space_use": ("span_width_pct", "span_height_pct", "min_gap_to_node_pct"),
    "color_consistency": ("hue_families", "roles_consistent"),
    "contrast": ("min_box_contrast", "arrow_contrast"),
    "alignment": ("max_row_col_offset_pct", "max_rotation_deg", "diagonal_connectors", "endpoint_treatments"),
    "style_uniformity": ("style_variants_per_role", "arrow_treatments"),
    "resolution": ("blur_px", "render_scale"),
    "canvas_fill": ("canvas_footprint_pct",),
}
# Known, unavoidable side effects that stay inside the sibling's healthy band.
TOLERATED = {
    ("degraded_centering_spacing", "min_gap_to_node_pct"),
    ("degraded_style_uniformity", "min_gap_to_node_pct"),
    ("degraded_style_uniformity", "span_width_pct"),
    ("degraded_style_uniformity", "span_height_pct"),
    ("degraded_style_uniformity", "worst_margin_ratio"),
    ("degraded_style_uniformity", "row_gap_ratio"),
    ("degraded_alignment", "span_height_pct"),
    ("degraded_canvas_fill", "span_width_pct"),
    ("degraded_canvas_fill", "span_height_pct"),
    ("degraded_canvas_fill", "min_gap_to_node_pct"),
}


class ScriptedBackend:
    name = "scripted"

    def structured(self, prompt, images, schema, schema_name):
        dimension = schema_name.removeprefix("diagram_beauty_")
        target = v6.VARIANT_TARGET.get(Path(images[0]).stem)
        return {"observation": "Scripted.", "score": 40 if target == dimension else 92, "reasoning": "Scripted."}


class BeautyV6FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        output = Path(cls.temp.name)
        cls.scenes = {v: v6.build_scene(v) for v in v6.VARIANTS}
        cls.paths = {v: v6.render(scene, output / f"{v}.png") for v, scene in cls.scenes.items()}
        cls.measurements = {v: v6.measure(cls.scenes[v], cls.paths[v]) for v in v6.VARIANTS}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_each_fixture_moves_its_own_group_only(self):
        base = self.measurements["baseline"]
        for variant, target in v6.VARIANT_TARGET.items():
            if target is None:
                continue
            measured = self.measurements[variant]
            with self.subTest(variant=variant):
                self.assertTrue(any(measured[k] != base[k] for k in GROUPS[target]), "target unchanged")
                for dimension, keys in GROUPS.items():
                    if dimension == target:
                        continue
                    for key in keys:
                        if (variant, key) in TOLERATED:
                            continue
                        if key in {"min_box_contrast", "canvas_footprint_pct"}:
                            self.assertAlmostEqual(measured[key], base[key], delta=1.5, msg=key)
                        else:
                            self.assertEqual(measured[key], base[key], key)

    def test_tolerated_side_effects_stay_in_healthy_bands(self):
        self.assertGreaterEqual(self.measurements["degraded_centering_spacing"]["min_gap_to_node_pct"], 40)
        self.assertGreaterEqual(self.measurements["degraded_alignment"]["min_gap_to_node_pct"], 40)
        self.assertLessEqual(self.measurements["degraded_style_uniformity"]["row_gap_ratio"], 1.25)

    def test_subcheck_fixtures_cross_their_thresholds(self):
        m = self.measurements
        self.assertGreater(m["degraded_centering_spacing"]["row_gap_ratio"], 1.6)
        self.assertEqual(m["degraded_centering_labels"]["label_anchor"], "corner")
        self.assertGreater(m["degraded_alignment"]["max_row_col_offset_pct"], 15)
        self.assertGreaterEqual(m["degraded_alignment_tilt"]["max_rotation_deg"], 4)
        self.assertGreaterEqual(m["degraded_alignment_arrows"]["diagonal_connectors"], 3)
        self.assertEqual(m["degraded_alignment_endpoints"]["endpoint_treatments"], 2)
        self.assertGreater(m["degraded_resolution_blur"]["blur_px"], 1.5)
        self.assertLess(m["degraded_resolution_pixel"]["render_scale"], 0.5)

    def test_canvas_fill_only_drops_on_its_fixture(self):
        for variant, path in self.paths.items():
            with self.subTest(variant=variant):
                score = judge_canvas_fill(path).score
                if variant == "degraded_canvas_fill":
                    self.assertLess(score, 50)
                else:
                    self.assertEqual(score, 100)


class BeautyV6PromptTests(unittest.TestCase):
    def test_prompts_name_every_sibling_as_not_owned(self):
        for dimension, prompt in BEAUTY_V6_PROMPTS.items():
            not_yours = prompt.split("NOT YOURS:", 1)[1].split("\n", 1)[0]
            with self.subTest(dimension=dimension):
                self.assertIn("MENTAL MASK", prompt)
                for sibling, label in v6.DIMENSION_LABELS.items():
                    if sibling != dimension:
                        self.assertIn(f"({label})", not_yours)

    def test_judge_returns_all_eight_dimensions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = v6.render(v6.build_scene("baseline"), Path(temp) / "baseline.png")
            scores = judge_beauty_v6(ScriptedBackend(), path)
        self.assertEqual(tuple(scores), BEAUTY_V6_DIMENSIONS)
        self.assertEqual(scores["canvas_fill"].score, 100)
        self.assertTrue(all(scores[d].score == 92 for d in BEAUTY_V6_MODEL_DIMENSIONS))

    def test_scored_run_reports_perfect_isolation_for_ideal_judge(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = v6.run(Path(temp), repeats=1, workers=4, backend=ScriptedBackend())
        self.assertEqual(summary["warnings"], [])
        self.assertEqual(summary["off_target_mean_abs"], 0)
        self.assertEqual(summary["overall_isolation"], 100.0)


if __name__ == "__main__":
    unittest.main()
