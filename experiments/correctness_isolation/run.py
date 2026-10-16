#!/usr/bin/env python3
"""Correctness isolation experiment: does each kind of error lower only its own dimension?

Same idea and metrics as the Beauty isolation experiment (docs/beauty_axis_experiment.md),
but on real PMC flowcharts: each diagram is scored as is and with exactly one error added,
using the 16 error types that made the dataset's negatives (diagrams/pipeline/negatives.py).
Two harmless controls per diagram (white margin added, converted to grayscale) should not
lower any dimension; they show false alarms.

    python3 run.py make     # pick diagrams; write baseline + single-error images (no API calls)
    python3 run.py score    # score every image with the evaluator (gpt-6-luna); resumable
    python3 run.py report   # metrics, results.csv and report.html (no API calls)

Nothing outside this folder is changed. Images and results stay here.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "diagrams"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(DATA / "pipeline"))

IMAGES, RESULTS = HERE / "images", HERE / "results"
MIN_CHANGED_PIXELS = 60  # same as negatives.visibly_changed; an error image that changed fewer pixels than this has no visible error; it is left out

# Candidate wording fix, tried here in memory only (config/ and src/ are not changed).
# It targets the two leaks found in the first run: enlarged text counted as Layout overlap,
# and blur at the image edge counted as cut off by the canvas.
WORDING_V2 = {
    "layout.overlap": " Text that spills out of its own box is judged under Legibility (text overflow), not here; "
                      "count only shapes that collide.",
    "layout.canvas": " A blurry or low-resolution image is not cut off: count only a component or arrow whose drawn "
                     "border or line visibly ends at the image edge.",
}
# v3 = v2 plus one sentence each for the two Connectivity criteria (erased and detached arrows were mostly missed).
WORDING_V3 = {
    **WORDING_V2,
    "connectivity.connections": " Look for a visible line for each expected connection: if no line joins the two "
                                "components, the connection is absent even when they are aligned or next to each other.",
    "connectivity.endpoints": " Trace every arrow end to end: an arrow that stops short of its component, leaving a "
                              "visible gap, is dangling.",
}
WORDING_V2_RULES = {"unintended_overlap": WORDING_V2["layout.overlap"], "canvas_bounds": WORDING_V2["layout.canvas"]}
WORDING_V2_KINDS = ("enlarge_label", "low_resolution", "crop_edge", "overlap_box") + ("control_margin", "control_gray")
DIMENSIONS = ["layout", "connectivity", "presence", "details", "legibility"]
PER_LEVEL, BASELINE_REPEATS, SEED = 10, 3, 2026
# Errors need at least two boxes the box finder can see. Hard diagrams with three or more were
# almost all used for the dataset's negatives or the human study, so two are accepted there.
MIN_BOXES = {"easy": 3, "medium": 3, "hard": 2}
CONTROLS = ("control_margin", "control_gray")
MIN_SENSITIVITY = 15  # points on a 0-100 scale, as in the Beauty experiment


# ---------- make ----------

def _nice():
    os.nice(15)  # keep the machine cool


def _flow_box_count(path):
    from boxes import find_boxes
    from negatives import flow_boxes
    try:
        return len(flow_boxes(find_boxes(path)[3]))
    except Exception:
        return 0


def _make_variant(task):
    from PIL import Image
    from negatives import CRITERION, apply_defect
    diagram_id, image, kind = task
    rng = random.Random(f"{SEED}:{diagram_id}:{kind}")
    out, target = apply_defect(kind, Image.open(image).convert("RGB"), rng)
    if out is None:
        return diagram_id, kind, None
    path = IMAGES / diagram_id / f"{kind}.png"
    out.save(path)
    return diagram_id, kind, {"id": diagram_id, "kind": kind, "criterion": CRITERION[kind],
                              "dimension": CRITERION[kind].split(".")[0], "target": target,
                              "image": str(path.relative_to(HERE))}


def _make_control(diagram, kind):
    from PIL import Image, ImageOps
    img = Image.open(HERE / diagram["baseline"]).convert("RGB")
    if kind == "control_margin":  # 8% white margin on every side; nothing drawn changes
        img = ImageOps.expand(img, border=int(0.08 * max(img.size)), fill="white")
    else:
        img = ImageOps.grayscale(img).convert("RGB")
    path = IMAGES / diagram["id"] / f"{kind}.png"
    img.save(path)
    return {"id": diagram["id"], "kind": kind, "criterion": None, "dimension": None, "target": None,
            "image": str(path.relative_to(HERE))}


def make(workers: int) -> None:
    from PIL import Image
    from negatives import CRITERION
    Image.MAX_IMAGE_PIXELS = None
    rows = list(csv.DictReader(open(DATA / "data.csv", encoding="utf-8")))
    study_articles = {r["source_id"].split("/")[0] for r in rows if r["human_study_id"]}
    rng = random.Random(SEED)
    chosen, articles = [], set()
    with Pool(workers, initializer=_nice) as pool:
        for level in ("easy", "medium", "hard"):
            pool_rows = [r for r in rows if r["label"] == "positive" and not r["human_study_id"]
                         and r["difficulty"] == level and r["source_id"].split("/")[0] not in study_articles]
            rng.shuffle(pool_rows)
            picked = []
            for start in range(0, len(pool_rows), 8):  # count boxes 8 at a time until 10 qualify
                batch = [r for r in pool_rows[start:start + 8] if r["source_id"].split("/")[0] not in articles
                         and max(Image.open(DATA / r["image"]).size) >= 600]
                for r, n in zip(batch, pool.map(_flow_box_count, [DATA / r["image"] for r in batch])):
                    if n >= MIN_BOXES[level] and len(picked) < PER_LEVEL and r["source_id"].split("/")[0] not in articles:
                        picked.append(r); articles.add(r["source_id"].split("/")[0])
                if len(picked) == PER_LEVEL:
                    break
            chosen += picked
            print(f"{level}: {len(picked)} diagrams", flush=True)
        diagrams = []
        for r in chosen:
            (IMAGES / r["id"]).mkdir(parents=True, exist_ok=True)
            baseline = IMAGES / r["id"] / "baseline.png"  # re-encoded like the variants, so format is not a difference
            Image.open(DATA / r["image"]).convert("RGB").save(baseline)
            diagrams.append({"id": r["id"], "difficulty": r["difficulty"], "caption": r["caption"],
                             "source_id": r["source_id"], "baseline": str(baseline.relative_to(HERE))})
        tasks = [(d["id"], HERE / d["baseline"], kind) for d in diagrams for kind in CRITERION]
        variants = []
        for n, (_, _, v) in enumerate(pool.imap_unordered(_make_variant, tasks, chunksize=2), 1):
            if v:
                variants.append(v)
            if n % 60 == 0 or n == len(tasks):
                print(f"variants: {n}/{len(tasks)} tried, {len(variants)} made", flush=True)
    variants += [_make_control(d, kind) for d in diagrams for kind in CONTROLS]
    variants.sort(key=lambda v: (v["id"], v["kind"]))
    json.dump({"diagrams": diagrams, "variants": variants}, open(HERE / "manifest.json", "w"), indent=1)
    print(f"{len(diagrams)} diagrams, {len(variants)} single-error variants -> manifest.json")


def remake(kinds: list[str], workers: int) -> None:
    """Regenerate some error types (after a generator fix) and drop their old scores."""
    manifest = json.load(open(HERE / "manifest.json"))
    old = [v for v in manifest["variants"] if v["kind"] in kinds]
    for v in old:
        (HERE / v["image"]).unlink(missing_ok=True)
        for results in (RESULTS, _results("v2")):
            (results / v["id"] / f"{v['kind']}.json").unlink(missing_ok=True)
    tasks = [(d["id"], HERE / d["baseline"], kind) for d in manifest["diagrams"] for kind in kinds]
    with Pool(workers, initializer=_nice) as pool:
        new = [v for _, _, v in pool.imap_unordered(_make_variant, tasks) if v]
    manifest["variants"] = sorted([v for v in manifest["variants"] if v["kind"] not in kinds] + new,
                                  key=lambda v: (v["id"], v["kind"]))
    json.dump(manifest, open(HERE / "manifest.json", "w"), indent=1)
    print(f"{', '.join(kinds)}: {len(old)} old images replaced by {len(new)} new ones")


# ---------- score ----------

def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _pipeline(wording="v1"):
    from dataclasses import replace
    from diagram_correctness import vlm
    from diagram_correctness.models import PresencePolicy
    from diagram_correctness.pipeline import CorrectnessPipeline, PipelineConfig
    from diagram_correctness.rubric import load_config, load_rubric
    from diagram_correctness.vlm import ExpectationExtractor, OpenAIBackend, VLMJudge
    settings = load_config(ROOT / "config" / "evaluator.json")
    model = os.environ.get("OPENAI_MODEL") or settings["models"]["judge"]
    if model != "gpt-6-luna":
        raise SystemExit(f"model is {model!r}; this experiment runs on gpt-6-luna only")
    policy = PresencePolicy.from_dict(settings.get("presence_policy"))
    judge_backend, expectation_backend = OpenAIBackend(model), OpenAIBackend(model)
    criteria = load_rubric(ROOT / "config" / "rubric.json")
    if wording in ("v2", "v3"):
        extra = WORDING_V3 if wording == "v3" else WORDING_V2
        criteria = [replace(c, description=c.description + extra.get(c.id, "")) for c in criteria]
        for key, extra in WORDING_V2_RULES.items():
            vlm.COUNTING_RULES[key] = vlm.COUNTING_RULES[key] + extra
    pipeline = CorrectnessPipeline(PipelineConfig(
        criteria=criteria,
        judges=[VLMJudge(judge_backend, policy, samples=int(settings.get("judge_samples", 1)))],
        expectation_extractor=ExpectationExtractor(expectation_backend),
        weighting=settings.get("weighting", "auto"),
        dimension_reducer=settings.get("dimension_reducer", "mean"),
        error_factor=settings.get("error_factor"),
        severity_multipliers=settings.get("severity_multipliers"),
        fallback_frequencies=settings.get("fallback_frequencies"),
    ))
    return pipeline, (judge_backend, expectation_backend), model


def score(workers: int, per_level: int | None = None, wording: str = "v1", only_kinds: list[str] | None = None) -> None:
    _load_env()
    manifest = json.load(open(HERE / "manifest.json"))
    results = _results(wording)
    if only_kinds:
        manifest["variants"] = [v for v in manifest["variants"] if v["kind"] in only_kinds]
        with_variants = {v["id"] for v in manifest["variants"]}  # baselines only where they are needed
        manifest["diagrams"] = [d for d in manifest["diagrams"] if d["id"] in with_variants]
    if wording == "v2":  # only the error types the wording fix is about, plus controls and baselines
        manifest["variants"] = [v for v in manifest["variants"] if v["kind"] in WORDING_V2_KINDS]
    if per_level:  # score only the first N diagrams (by id) of each difficulty
        keep = {d["id"] for level in ("easy", "medium", "hard")
                for d in sorted((d for d in manifest["diagrams"] if d["difficulty"] == level), key=lambda d: d["id"])[:per_level]}
        manifest = {"diagrams": [d for d in manifest["diagrams"] if d["id"] in keep],
                    "variants": [v for v in manifest["variants"] if v["id"] in keep]}
    pipeline, backends, model = _pipeline(wording)
    print(f"scoring with {model}", flush=True)
    captions = {d["id"]: d["caption"] for d in manifest["diagrams"]}

    def inventory(diagram_id):
        path = RESULTS / diagram_id / "inventory.json"  # the same frozen sketch for both wordings
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            inv = pipeline.config.expectation_extractor.extract(captions[diagram_id], pipeline.config.criteria)
            path.write_text(json.dumps(inv, indent=1))
        return json.loads(path.read_text())

    with ThreadPoolExecutor(workers) as pool:  # one frozen expectation sketch per diagram, shared by its variants
        inventories = dict(zip(captions, pool.map(inventory, captions)))

    jobs = [(d["id"], f"baseline_r{k}", d["baseline"]) for d in manifest["diagrams"] for k in range(BASELINE_REPEATS)]
    jobs += [(v["id"], v["kind"], v["image"]) for v in manifest["variants"]]
    jobs = [j for j in jobs if not (results / j[0] / f"{j[1]}.json").exists()]
    print(f"{len(jobs)} images to score", flush=True)

    def run(job):
        diagram_id, name, image = job
        report = pipeline.run(description=captions[diagram_id], candidate_image=HERE / image,
                              inventory=inventories[diagram_id], run_deterministic=False).to_dict()
        keep = {"correctness": report["correctness"],
                "dimensions": {d: v["score"] for d, v in report["dimensions"].items()},
                "criteria": {d: v["criteria"] for d, v in report["dimensions"].items()},
                "metrics": report.get("metrics")}
        (results / diagram_id).mkdir(parents=True, exist_ok=True)
        (results / diagram_id / f"{name}.json").write_text(json.dumps(keep, indent=1))
        return job

    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for future in as_completed([pool.submit(run, j) for j in jobs]):
            try:
                future.result()
            except Exception as error:  # a failed call is retried on the next `score` run
                print(f"failed: {error}", flush=True)
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"scored {done}/{len(jobs)}", flush=True)
    usage = {k: sum(b.usage[k] for b in backends) for k in backends[0].usage}
    print("api usage this run:", usage)


# ---------- report ----------

def _results(wording):
    return RESULTS if wording == "v1" else HERE / f"results_{wording}"


def _changed_pixels(baseline, image):
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(HERE / baseline).convert("L"), dtype=int)
    b = np.asarray(Image.open(HERE / image).convert("L"), dtype=int)
    return int((abs(a - b) > 60).sum()) if a.shape == b.shape else -1  # -1: canvas changed (crop, margin)


def _metrics(manifest, results=RESULTS):
    rows, noise = [], {d: [] for d in DIMENSIONS}
    baselines = {}
    for diagram in manifest["diagrams"]:
        runs = [json.load(open(p))["dimensions"] for p in sorted((results / diagram["id"]).glob("baseline_r*.json"))]
        if not runs:
            continue
        baselines[diagram["id"]] = {d: 100 * statistics.mean(r[d] for r in runs if d in r) for d in DIMENSIONS}
        for d in DIMENSIONS:
            if len(runs) > 1:
                noise[d].append(100 * statistics.pstdev([r[d] for r in runs if d in r]))
    difficulty = {d["id"]: d["difficulty"] for d in manifest["diagrams"]}
    baseline_image = {d["id"]: d["baseline"] for d in manifest["diagrams"]}
    for v in manifest["variants"]:
        path = results / v["id"] / f"{v['kind']}.json"
        if not path.exists() or v["id"] not in baselines:
            continue
        scores = json.load(open(path))["dimensions"]
        drop = {d: round(baselines[v["id"]][d] - 100 * scores[d], 1) for d in DIMENSIONS if d in scores}
        target = v["dimension"]
        changed = _changed_pixels(baseline_image[v["id"]], v["image"])
        if target is not None and 0 <= changed < MIN_CHANGED_PIXELS:  # the error did not visibly change the image
            rows.append({"id": v["id"], "difficulty": difficulty[v["id"]], "kind": v["kind"], "target": "invalid",
                         **{f"drop_{d}": drop.get(d, "") for d in DIMENSIONS}, "sensitivity": "", "worst_leak": "",
                         "specificity": "", "isolation": "", "changed_px": changed, "image": v["image"]})
            continue
        if target is None:  # a control: every dimension should stay put
            rows.append({"id": v["id"], "difficulty": difficulty[v["id"]], "kind": v["kind"], "target": "",
                         **{f"drop_{d}": drop.get(d, "") for d in DIMENSIONS}, "sensitivity": "",
                         "worst_leak": round(max([x for x in drop.values()], default=0), 1),
                         "specificity": "", "isolation": "", "changed_px": changed, "image": v["image"]})
            continue
        sensitivity = max(0.0, drop.get(target, 0.0))
        leaks = {d: max(0.0, x) for d, x in drop.items() if d != target}
        total = sensitivity + sum(leaks.values())
        specificity = 100 * sensitivity / total if total else 0.0
        rows.append({"id": v["id"], "difficulty": difficulty[v["id"]], "kind": v["kind"], "target": target,
                     **{f"drop_{d}": drop.get(d, "") for d in DIMENSIONS},
                     "sensitivity": round(sensitivity, 1), "worst_leak": round(max(leaks.values(), default=0), 1),
                     "specificity": round(specificity, 1),
                     "isolation": round(specificity * min(1.0, sensitivity / MIN_SENSITIVITY), 1),
                     "changed_px": changed, "image": v["image"]})
    return rows, baselines, {d: round(statistics.mean(x), 1) if x else None for d, x in noise.items()}


def _mean(values):
    values = [v for v in values if v != ""]
    return round(statistics.mean(values), 1) if values else None


def _cell(value, diagonal=False):
    if value is None:
        return "<td>-</td>"
    shade = min(abs(value), 60) / 60
    color = f"rgba(200,60,40,{shade:.2f})" if value > 0 else f"rgba(40,120,200,{shade:.2f})"
    style = f"background:{color};" + ("outline:2px solid #222;" if diagonal else "")
    return f'<td style="{style}">{value:+.1f}</td>'


def report(wording: str = "v1") -> None:
    from negatives import CRITERION
    manifest = json.load(open(HERE / "manifest.json"))
    rows, baselines, noise = _metrics(manifest, _results(wording))
    suffix = "" if wording == "v1" else f"_{wording}"
    if not rows:
        raise SystemExit("no scored variants yet; run `score` first")
    controls = [r for r in rows if not r["target"]]
    invalid = [r for r in rows if r["target"] == "invalid"]
    errors = [r for r in rows if r["target"] and r["target"] != "invalid"]
    with open(HERE / f"results{suffix}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    by_kind = {k: [r for r in errors if r["kind"] == k] for k in CRITERION}
    by_dim = {d: [r for r in errors if r["target"] == d] for d in DIMENSIONS}
    kind_rows = []
    for kind, rs in by_kind.items():
        if not rs:
            continue
        caught = 100 * sum(r["sensitivity"] >= MIN_SENSITIVITY for r in rs) / len(rs)
        kind_rows.append((kind, CRITERION[kind], len(rs), {d: _mean([r[f"drop_{d}"] for r in rs]) for d in DIMENSIONS},
                          round(caught), _mean([r["isolation"] for r in rs])))
    overall = _mean([r["isolation"] for r in errors])
    off_target = _mean([abs(r[f"drop_{d}"]) for r in errors for d in DIMENSIONS if d != r["target"] and r[f"drop_{d}"] != ""])
    missed = [r for r in errors if max(r[f"drop_{d}"] for d in DIMENSIONS if r[f"drop_{d}"] != "") < MIN_SENSITIVITY]
    control_drop = {k: {d: _mean([r[f"drop_{d}"] for r in controls if r["kind"] == k]) for d in DIMENSIONS} for k in CONTROLS}
    summary = {"diagrams": len(baselines), "variants": len(errors), "overall_isolation": overall,
               "off_target_mean_abs": off_target, "missed_by_every_dimension": len(missed),
               "invalid_fixtures": {k: sum(r["kind"] == k for r in invalid) for k in CRITERION if any(r["kind"] == k for r in invalid)},
               "controls_mean_drop": control_drop, "baseline_noise_sd": noise,
               "by_dimension": {d: {"n": len(rs), "isolation": _mean([r["isolation"] for r in rs]),
                                    "mean_drop": {m: _mean([r[f"drop_{m}"] for r in rs]) for m in DIMENSIONS}}
                                for d, rs in by_dim.items()},
               "by_kind": {k: {"criterion": c, "n": n, "mean_drop": md, "caught_pct": cp, "isolation": iso}
                           for k, c, n, md, cp, iso in kind_rows}}
    json.dump(summary, open(HERE / f"summary{suffix}.json", "w"), indent=1)

    head = "".join(f"<th>{d}</th>" for d in DIMENSIONS)
    dim_table = "".join(
        f"<tr><th>{d}</th><td>{len(by_dim[d])}</td>"
        + "".join(_cell(summary["by_dimension"][d]["mean_drop"][m], m == d) for m in DIMENSIONS)
        + f"<td>{summary['by_dimension'][d]['isolation']}</td></tr>" for d in DIMENSIONS)
    kind_table = "".join(
        f"<tr><th>{k}</th><td>{c}</td><td>{n}</td>"
        + "".join(_cell(md[m], m == c.split('.')[0]) for m in DIMENSIONS)
        + f"<td>{cp}%</td><td>{iso}</td></tr>" for k, c, n, md, cp, iso in kind_rows)
    gallery = []
    for kind, rs in by_kind.items():
        for r in sorted(rs, key=lambda r: -r["sensitivity"])[:1] + sorted(rs, key=lambda r: r["isolation"])[:1]:
            drops = ", ".join(f"{d} {r[f'drop_{d}']:+}" for d in DIMENSIONS if r[f"drop_{d}"] != "")
            gallery.append(f'<figure><div class="pair"><img loading="lazy" src="images/{r["id"]}/baseline.png">'
                           f'<img loading="lazy" src="{html.escape(r["image"])}"></div>'
                           f'<figcaption><b>{kind}</b> on {r["id"]} ({r["difficulty"]}): isolation {r["isolation"]}; '
                           f'drop {html.escape(drops)}</figcaption></figure>')
    noise_text = ", ".join(f"{d} {v}" for d, v in noise.items())
    control_table = "".join(f"<tr><th>{k}</th>" + "".join(_cell(control_drop[k][d]) for d in DIMENSIONS) + "</tr>"
                            for k in CONTROLS)
    (HERE / f"report{suffix}.html").write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Correctness isolation</title>
<style>body{{margin:0;background:#f6f7f9;color:#18202b;font:14px/1.5 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:24px 16px}}
table{{border-collapse:collapse;background:#fff;margin:8px 0 24px}}td,th{{border:1px solid #d9dee5;padding:4px 8px;text-align:right}}th{{text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}}figure{{margin:0;background:#fff;border:1px solid #d9dee5;padding:8px}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}.pair img{{width:100%;height:200px;object-fit:contain;background:#fff;border:1px solid #eee}}figcaption{{font-size:12px;margin-top:6px}}</style>
<main><h1>Correctness isolation experiment{' (wording ' + wording + ')' if suffix else ''}</h1>
<p>{summary['diagrams']} PMC flowcharts (10 easy, 10 medium, 10 hard; none in the human study), each scored as is
({BASELINE_REPEATS} repeats, averaged) and with exactly one added error ({summary['variants']} single-error images), by gpt-6-luna with the
frozen expectation sketch of its caption. Cells are the mean <b>score drop</b> from baseline in points (0-100); red is a drop, blue a rise;
the outlined cell is the error's own dimension. Isolation follows the Beauty experiment: specificity (own drop / all drops) x min(own drop / {MIN_SENSITIVITY}, 1).</p>
<p><b>Overall isolation: {overall}</b>; mean |off-target change| {off_target} points; errors no dimension noticed
(no drop &ge; {MIN_SENSITIVITY}): {len(missed)} of {len(errors)}. Baseline run-to-run noise (SD, points): {noise_text}.</p>
<p>Left out as broken test images (changed fewer than {MIN_CHANGED_PIXELS} pixels, so no visible error):
{html.escape(json.dumps(summary["invalid_fixtures"]))}.</p>
<h2>Controls (should not drop)</h2><table><tr><th>control</th>{head}</tr>{control_table}</table>
<h2>By the dimension the error targets</h2><table><tr><th>error targets</th><th>n</th>{head}<th>isolation</th></tr>{dim_table}</table>
<h2>By error type</h2><table><tr><th>error</th><th>criterion</th><th>n</th>{head}<th>caught (own drop &ge; {MIN_SENSITIVITY})</th><th>isolation</th></tr>{kind_table}</table>
<h2>Examples</h2><p>Per error type: the case with the largest own drop, and the least isolated case. Left: baseline; right: with the error.</p>
<div class="grid">{''.join(gallery)}</div></main></html>
""", encoding="utf-8")
    print(f"overall isolation {overall}; off-target {off_target}; missed {len(missed)}/{len(errors)}; "
          f"controls {control_drop}; report.html, summary.json and results.csv written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["make", "remake", "score", "report"])
    parser.add_argument("--kinds", default="erase_arrow,detach_arrow,reverse_arrow", help="remake: error types to regenerate")
    parser.add_argument("--workers", type=int, default=None, help="make: CPU processes (4); score: parallel API calls (8)")
    parser.add_argument("--per-level", type=int, default=None, help="score: only the first N diagrams of each difficulty")
    parser.add_argument("--only-kinds", default=None, help="score: only these error types (comma-separated)")
    parser.add_argument("--wording", choices=["v1", "v2", "v3"], default="v1",
                        help="v2: try the Layout wording fix (in memory) on the error types it is about")
    args = parser.parse_args()
    if args.step == "make":
        make(args.workers or 4)
    elif args.step == "remake":
        remake(args.kinds.split(","), args.workers or 4)
    elif args.step == "score":
        score(args.workers or 8, args.per_level, args.wording, args.only_kinds.split(",") if args.only_kinds else None)
    else:
        report(args.wording)
