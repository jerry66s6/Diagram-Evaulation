"""Invariants for the four-dimension Beauty v4 isolation experiment."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from diagram_correctness.beauty_v4 import BEAUTY_V4_DIMENSIONS, BEAUTY_V4_PROMPTS, judge_beauty_v4


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("beauty_v4_ablation", ROOT / "scripts/run_beauty_v4_ablation.py")
v4 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v4
spec.loader.exec_module(v4)


class ScriptedBackend:
    """Returns baseline-high scores, dropping only the dimension each fixture targets."""

    name = "scripted"

    def __init__(self, images_to_variant):
        self.images_to_variant = images_to_variant

    def structured(self, prompt, images, schema, schema_name):
        dimension = schema_name.removeprefix("diagram_beauty_")
        variant = self.images_to_variant.get(Path(images[0]).stem, "baseline")
        score = 40 if v4.VARIANT_TARGET.get(variant) == dimension else 92
        return {"observation": f"{dimension} observed", "score": score, "reasoning": "Scripted."}


class BeautyV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenes = {variant: v4.build_scene(variant) for variant in v4.VARIANT_TARGET}
        cls.measurements = {variant: v4.measure(scene) for variant, scene in cls.scenes.items()}

    def test_baseline_is_good_on_every_dimension(self):
        base = self.measurements["baseline"]
        self.assertLessEqual(base["worst_margin_ratio"], 1.05)
        self.assertGreaterEqual(min(base["span_width_pct"], base["span_height_pct"]), 50)
        self.assertGreaterEqual(base["min_gap_to_node_pct"], 40)
        self.assertLessEqual(base["hue_families"], 3)
        self.assertTrue(base["roles_consistent"])
        self.assertGreaterEqual(base["min_box_contrast"], 4.5)
        self.assertGreaterEqual(base["arrow_contrast"], 4.5)

    def test_each_fixture_changes_only_its_target_measurement(self):
        base = self.measurements["baseline"]
        groups = {
            "centering": ("worst_margin_ratio",),
            "space_use": ("span_width_pct", "span_height_pct", "min_gap_to_node_pct"),
            "color_consistency": ("hue_families", "roles_consistent"),
            "contrast": ("min_box_contrast", "arrow_contrast"),
        }
        for variant, target in v4.VARIANT_TARGET.items():
            if target is None:
                continue
            measured = self.measurements[variant]
            with self.subTest(variant=variant):
                self.assertTrue(any(measured[key] != base[key] for key in groups[target]))
                for dimension, keys in groups.items():
                    if dimension == target:
                        continue
                    for key in keys:
                        if key in {"min_box_contrast"}:
                            self.assertAlmostEqual(measured[key], base[key], delta=0.25)
                        else:
                            self.assertEqual(measured[key], base[key], key)

    def test_target_measurements_cross_failure_thresholds(self):
        self.assertGreater(self.measurements["degraded_centering"]["worst_margin_ratio"], 5)
        space = self.measurements["degraded_space_use"]
        self.assertLess(space["min_gap_to_node_pct"], 25)
        self.assertLess(space["span_height_pct"], 45)
        color = self.measurements["degraded_color_consistency"]
        self.assertGreaterEqual(color["hue_families"], 5)
        self.assertFalse(color["roles_consistent"])
        contrast = self.measurements["degraded_contrast"]
        self.assertLess(contrast["min_box_contrast"], 1.5)
        self.assertLess(contrast["arrow_contrast"], 1.6)

    def test_color_fixture_keeps_luminance_and_contrast_fixture_keeps_hues(self):
        base = self.scenes["baseline"]
        color = self.scenes["degraded_color_consistency"]
        for fill in color.fills:
            self.assertAlmostEqual(v4.relative_luminance(fill), v4.TARGET_LUMINANCE, delta=0.01)
        pale = self.scenes["degraded_contrast"]
        for dark, light in zip(base.fills, pale.fills):
            difference = abs(v4.hue_of(dark) - v4.hue_of(light))
            self.assertLess(min(difference, 360 - difference), 12)

    def test_prompts_name_every_sibling_as_not_owned(self):
        labels = {"centering": "Centering", "space_use": "Space Use",
                  "color_consistency": "Color Consistency", "contrast": "Contrast"}
        for dimension, prompt in BEAUTY_V4_PROMPTS.items():
            not_yours = prompt.split("NOT YOURS:", 1)[1].split("\n", 1)[0]
            with self.subTest(dimension=dimension):
                self.assertIn(f"YOU OWN: {labels[dimension].upper()}", prompt)
                self.assertIn("MENTAL MASK", prompt)
                for sibling, label in labels.items():
                    if sibling != dimension:
                        self.assertIn(f"({label})", not_yours)

    def test_judge_returns_all_four_dimensions(self):
        backend = ScriptedBackend({})
        scores = judge_beauty_v4(backend, "baseline.png")
        self.assertEqual(tuple(scores), BEAUTY_V4_DIMENSIONS)
        self.assertTrue(all(score.score == 92 for score in scores.values()))

    def test_scored_run_reports_perfect_isolation_for_ideal_judge(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            backend = ScriptedBackend({variant: variant for variant in v4.VARIANT_TARGET})
            summary = v4.run(output, repeats=2, workers=2, backend=backend)
            self.assertEqual(summary["overall_isolation"], 100.0)
            self.assertEqual(summary["warnings"], [])
            for metric in summary["metrics"].values():
                self.assertEqual(metric["sensitivity"], 52)
                self.assertEqual(metric["worst_leak"], 0)
            saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["results"]["baseline"]["centering"]["runs"], [92, 92])
            self.assertTrue((output / "scores.csv").is_file())
            self.assertIn("Isolation: 100.0/100", (output / "index.html").read_text(encoding="utf-8"))

    def test_isolation_metrics_penalize_leaks_and_weak_targets(self):
        def row(**scores):
            return {d: {"score": scores.get(d, 90)} for d in BEAUTY_V4_DIMENSIONS}

        results = {
            "baseline": row(),
            "degraded_centering": row(centering=50, space_use=70),
            "degraded_space_use": row(space_use=85),
            "degraded_color_consistency": row(color_consistency=40),
            "degraded_contrast": row(contrast=40, color_consistency=90),
        }
        metrics = v4.isolation_metrics(results)
        self.assertAlmostEqual(metrics["degraded_centering"]["specificity"], 66.7)
        self.assertEqual(metrics["degraded_centering"]["worst_leak_dimension"], "space_use")
        self.assertAlmostEqual(metrics["degraded_space_use"]["isolation_score"], 33.3)
        warnings = v4.isolation_warnings(metrics)
        self.assertTrue(any("[LEAK] degraded_centering" in w for w in warnings))
        self.assertTrue(any("[WEAK] degraded_space_use" in w for w in warnings))

    def test_fixtures_only_run_makes_no_judge_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = v4.run(Path(temp))
            self.assertIsNone(summary["results"])
            self.assertIsNone(summary["overall_isolation"])
            self.assertEqual(len(list((Path(temp) / "diagrams").glob("*.png"))), 5)


if __name__ == "__main__":
    unittest.main()
