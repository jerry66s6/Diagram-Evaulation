#!/usr/bin/env python3
"""Extract the curated 33/33/33 DiagramGen subset; no generation or scoring."""
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MIRROR = "https://modelscope.cn/datasets/opendatalab-raiser/DiagramGen/resolve/master/"
TIERS = ("easy", "medium", "hard")
RUBRIC = {
    "easy": "Few main components; short chains, shallow branches, or a simple cycle; little routing to trace.",
    "medium": "More stages or groups, multiple branches and merges, feedback, parallel transitions, or routed connections.",
    "hard": "Many components, dense connectivity, multiple branching levels, or substantial connections within and across groups.",
}


def write_gallery(output: Path, rows: list[dict]) -> None:
    sections = []
    escape = html.escape
    for tier in TIERS:
        cards = []
        for row in rows:
            if row["difficulty"] != tier:
                continue
            caption = (output / row["caption_file"]).read_text()
            cards.append(f"""<article>
<a href="{row['file']}" target="_blank"><img src="{row['file']}" loading="lazy" alt="{escape(row['title'])}"></a>
<h3>{row['id']} · {escape(row['title'])}</h3>
<p>{escape(row['complexity_reason'])}</p>
<p class="meta">Source category: {escape(row['source_category'])}<br>{row['source_id']}</p>
<p><a href="{row['file']}">Original PNG</a> · <a href="{row['caption_file']}">Description</a> · <a href="{row['reference_file']}">Source code</a></p>
<details><summary>Original expanded query</summary><p class="caption">{escape(caption)}</p></details>
</article>""")
        sections.append(f'<section id="{tier}"><h2>{tier.title()} · {len(cards)}</h2><p>{RUBRIC[tier]}</p><div class="grid">' + "\n".join(cards) + "</div></section>")
    (output / "index.html").write_text("""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DiagramGen · 99 balanced diagrams</title>
<style>
body{margin:0;background:#f4f6f8;color:#17212e;font:15px/1.55 system-ui,sans-serif}
main{max-width:1440px;margin:auto;padding:32px}h1{font-size:32px;margin:0}h2{margin:0}
nav{display:flex;gap:24px;flex-wrap:wrap;margin:20px 0}a{color:#1758a8}
section{margin-top:40px;scroll-margin-top:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
article{background:white;border:1px solid #d9e0e7;border-radius:8px;padding:16px;min-width:0}
img{display:block;width:100%;height:250px;object-fit:contain;background:white;border-bottom:1px solid #eef0f3}
h3{font-size:16px}.meta{font-size:12px;color:#617083;overflow-wrap:anywhere}.caption{white-space:pre-wrap;font-size:13px}
summary{cursor:pointer}header>p{max-width:900px}
</style><main><header><h1>DiagramGen · 99 balanced diagrams</h1>
<p>33 easy, 33 medium, 33 hard. Original flowcharts and related node-and-arrow diagrams, including state machines, pipelines, dependencies, and architectures. Difficulty describes structural complexity; these are curated research labels, not evaluation scores.</p>
<p>Click an image for its original resolution. Descriptions are the dataset's unmodified expanded queries. Keep this HTML beside its easy, medium, and hard folders.</p>
<nav><a href="#easy">Easy · 33</a><a href="#medium">Medium · 33</a><a href="#hard">Hard · 33</a><a href="data.csv">CSV index</a><a href="README.md">Dataset notes</a></nav>
</header>""" + "\n".join(sections) + "</main></html>\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "diagrams/diagramgen_balanced_99")
    parser.add_argument("--image-cache", type=Path, help="Optional directory containing original <source_id>.png files.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (ROOT / "scripts/diagramgen99_selection.csv").open() as source:
        selection = list(csv.DictReader(source))
    records = json.loads((ROOT / "examples/diagramgen10/source/DiagramGeneration.json").read_text())
    by_id = {row["id"]: row for row in records}
    counts = Counter(row["difficulty"] for row in selection)
    if counts != {tier: 33 for tier in TIERS} or len({row["source_id"] for row in selection}) != 99:
        raise ValueError("The curated selection must contain 99 distinct IDs, 33 per tier.")

    rows, original_records = [], []
    numbers = Counter()
    for chosen in selection:
        record = by_id[chosen["source_id"]]
        tier = chosen["difficulty"]
        numbers[tier] += 1
        sample_id = f"{tier}_{numbers[tier]:02d}"
        stem = Path(tier) / sample_id
        image = stem.with_suffix(".png")
        caption = stem.with_suffix(".txt")
        code = record["reference"]
        reference = stem.with_suffix(".dot" if code.lstrip().startswith(("digraph", "graph", "strict")) else ".tex")
        (output / tier).mkdir(exist_ok=True)
        target = output / image
        image_url = MIRROR + record["images"][0].removeprefix("./")
        if not target.exists():
            cached = [ROOT / "examples/diagramgen10/samples" / record["id"] / "candidate.png"]
            if args.image_cache:
                cached.insert(0, args.image_cache / f"{record['id']}.png")
            local = next((path for path in cached if path.is_file()), None)
            if local:
                shutil.copyfile(local, target)
            else:
                with urlopen(image_url, timeout=60) as response:
                    target.write_bytes(response.read())
        (output / caption).write_text(record["expanded_query"])
        (output / reference).write_text(code)
        rows.append({
            "id": sample_id, "file": str(image), "difficulty": tier,
            "caption_file": str(caption), "title": chosen["title"],
            "source": "OpenRaiser/DiagramGen:DiagramGeneration:test",
            "source_id": record["id"], "source_category": record["category"],
            "reference_file": str(reference), "complexity_reason": chosen["complexity_reason"],
            "image_url": image_url,
        })
        original_records.append(record)

    with (output / "data.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "original_records.json").write_text(json.dumps(original_records, indent=2, ensure_ascii=False) + "\n")
    shutil.copyfile(ROOT / "examples/diagramgen10/source/README.md", output / "SOURCE.md")
    readme = ROOT / "diagrams/diagramgen_balanced_99/README.md"
    if output / "README.md" != readme:
        shutil.copyfile(readme, output / "README.md")
    write_gallery(output, rows)
    print(f"Prepared {len(rows)} diagrams: {dict(numbers)}")
    print(f"Folder: {output}")
    print(f"Source categories: {dict(Counter(row['source_category'] for row in rows))}")


if __name__ == "__main__":
    main()
