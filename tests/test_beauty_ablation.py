"""Invariants for the current Beauty dimensions and their single-variable fixtures."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

from diagram_correctness.beauty import (
    BEAUTY_DIMENSIONS,
    BEAUTY_PROMPTS,
    judge_beauty,
)


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("beauty_ablation", ROOT / "scripts" / "run_beauty_ablation.py")
v6 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v6
spec.loader.exec_module(v6)

# Measurement groups owned by each dimension; every other group must match the baseline.
GROUPS = {
    "centering": ("worst_margin_ratio",),
    "space_use": ("min_gap_to_node_pct",),
    "color_consistency": ("hue_families", "roles_consistent"),
    "contrast": ("min_box_contrast", "arrow_contrast"),
    "alignment": ("max_row_col_offset_pct", "max_rotation_deg", "diagonal_connectors", "endpoint_treatments"),
    "style_uniformity": ("style_variants_per_role", "arrow_treatments"),
    "resolution": ("blur_px", "render_scale"),
    "canvas_fill": ("node_sizes",),
}
# Known, unavoidable side effects that stay inside the sibling's healthy band.
TOLERATED = {
    ("degraded_centering_spacing", "min_gap_to_node_pct"),
    ("degraded_style_uniformity", "min_gap_to_node_pct"),
    ("degraded_style_uniformity", "worst_margin_ratio"),
    ("degraded_style_uniformity", "node_sizes"),
    ("degraded_canvas_fill", "min_gap_to_node_pct"),
}


class ScriptedBackend:
    name = "scripted"

    def __init__(self):
        self.calls = []

    def structured(self, prompt, images, schema, schema_name):
        dimension = schema_name.removeprefix("diagram_beauty_")
        self.calls.append((dimension, prompt, images, schema))
        target = v6.VARIANT_TARGET.get(Path(images[0]).stem)
        return {"observation": "Scripted.", "score": 40 if target == dimension else 92, "reasoning": "Scripted."}


class BeautyFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        output = Path(cls.temp.name)
        cls.scenes = {v: v6.build_scene(v) for v in v6.VARIANTS if v not in v6.CONTROLS}
        cls.paths = {v: v6.render(scene, output / f"{v}.png") for v, scene in cls.scenes.items()}
        cls.measurements = {v: v6.measure(cls.scenes[v]) for v in cls.scenes}
        cls.controls = {v: v6.render_control(v, output / f"{v}.png") for v in v6.CONTROLS}

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
                        if key == "min_box_contrast":
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

class BeautyPromptTests(unittest.TestCase):
    def test_prompts_name_every_sibling_as_not_owned(self):
        for dimension, prompt in BEAUTY_PROMPTS.items():
            not_yours = prompt.split("NOT YOURS:", 1)[1].split("\n", 1)[0]
            with self.subTest(dimension=dimension):
                self.assertIn("MENTAL MASK", prompt)
                for sibling, label in v6.DIMENSION_LABELS.items():
                    if sibling != dimension:
                        self.assertIn(f"({label})", not_yours)

    def test_judge_returns_all_eight_dimensions(self):
        backend = ScriptedBackend()
        with tempfile.TemporaryDirectory() as temp:
            path = v6.render(v6.build_scene("baseline"), Path(temp) / "baseline.png")
            scores = judge_beauty(backend, path)
        self.assertEqual(tuple(scores), BEAUTY_DIMENSIONS)
        self.assertEqual(tuple(call[0] for call in backend.calls), BEAUTY_DIMENSIONS)
        self.assertTrue(all(scores[d].score == 92 for d in BEAUTY_DIMENSIONS))

    def test_canvas_fill_prompt_preserves_scoring_thresholds(self):
        prompt = BEAUTY_PROMPTS["canvas_fill"]
        self.assertIn("at or below 3%: score 0", prompt)
        self.assertIn("at or above 6%: score 100", prompt)
        self.assertIn("round(100 * (footprint_percent - 3) / 3)", prompt)
        self.assertIn("outline-only", prompt)
        self.assertIn("loop of arrows does NOT count", prompt)

    def test_space_use_scores_clearance_not_canvas_coverage(self):
        prompt = BEAUTY_PROMPTS["space_use"]
        self.assertIn("based ONLY on inter-node/group clearance", prompt)
        self.assertIn("edge-to-edge gap", prompt)
        self.assertIn("Do not score canvas coverage", prompt)
        self.assertIn("Tight gaps remain a defect on a full canvas", prompt)
        self.assertIn("Do not call every overlap intentional", prompt)
        self.assertIn("intentional layering is not itself crowding", prompt)
        self.assertNotIn("main-direction span", prompt)
        self.assertNotIn("WORSE of the span and gap", prompt)
        self.assertIn("gaps between nodes", BEAUTY_PROMPTS["canvas_fill"])

    def test_centering_scores_whole_diagram_margins_only(self):
        prompt = BEAUTY_PROMPTS["centering"]
        self.assertIn("100 * (1 - abs(L - R) / W_canvas)", prompt)
        self.assertIn("100 * (1 - abs(T - B) / H_canvas)", prompt)
        self.assertIn("round(min(horizontal score, vertical score))", prompt)
        self.assertIn("NOT by L+R or T+B", prompt)
        self.assertIn("two zero margins also score 100", prompt)
        self.assertIn("do not score label centering inside shapes, equal inter-node gaps", prompt)
        for variant in v6.PLACEMENT_CONTROLS:
            self.assertIsNone(v6.VARIANT_TARGET[variant])

    def test_scored_run_reports_perfect_isolation_for_ideal_judge(self):
        backend = ScriptedBackend()
        with tempfile.TemporaryDirectory() as temp:
            summary = v6.run(Path(temp), repeats=2, workers=4, backend=backend)
        self.assertEqual(len(backend.calls), len(v6.VARIANTS) * 8 * 2)
        self.assertEqual(summary["results"]["baseline"]["canvas_fill"]["runs"], [92, 92])
        self.assertEqual(summary["results"]["degraded_canvas_fill"]["canvas_fill"]["runs"], [40, 40])
        self.assertEqual(summary["warnings"], [])
        self.assertEqual(summary["off_target_mean_abs"], 0)
        self.assertEqual(summary["overall_isolation"], 100.0)
        self.assertEqual(summary["controls"], {v: True for v in v6.CONTROLS | v6.PLACEMENT_CONTROLS})


if __name__ == "__main__":
    unittest.main()
