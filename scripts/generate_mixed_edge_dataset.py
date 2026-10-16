#!/usr/bin/env python3
"""Generate ten original flowcharts for human review; does not alter a study catalog."""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]

# Nodes: (label, column, row, role). Edges: (source, target, branch label).
SCENARIOS = [
    ("Order fulfillment", "easy", "horizontal", [
        ("Order received", 0, 1, "terminal"), ("Validate order", 1, 1, "process"),
        ("Stock available?", 2, 1, "decision"), ("Pack order", 3, 0, "process"),
        ("Notify backorder", 3, 2, "process"), ("Dispatch parcel", 4, 0, "process"),
        ("Await restock", 4, 2, "terminal")],
     [(0, 1, ""), (1, 2, ""), (2, 3, "Yes"), (2, 4, "No"), (3, 5, ""), (4, 6, "")]),
    ("Expense reimbursement", "medium", "vertical", [
        ("Submit claim", 1, 0, "terminal"), ("Check receipts", 1, 1, "process"),
        ("Complete?", 1, 2, "decision"), ("Request correction", 0, 3, "process"),
        ("Manager review", 2, 3, "process"), ("Approved?", 2, 4, "decision"),
        ("Reject claim", 1, 5, "terminal"), ("Schedule payment", 3, 5, "process"),
        ("Reimbursed", 3, 6, "terminal")],
     [(0, 1, ""), (1, 2, ""), (2, 3, "No"), (2, 4, "Yes"), (4, 5, ""),
      (5, 6, "No"), (5, 7, "Yes"), (7, 8, "")]),
    ("Production incident response", "hard", "horizontal", [
        ("Alert fired", 0, 1, "terminal"), ("Triage impact", 1, 1, "process"),
        ("Customer impact?", 2, 1, "decision"), ("Monitor trend", 3, 0, "process"),
        ("Declare incident", 3, 2, "process"), ("Assign commander", 4, 2, "process"),
        ("Apply mitigation", 5, 2, "process"), ("Service healthy?", 6, 2, "decision"),
        ("Escalate response", 7, 3, "process"), ("Resolve incident", 7, 1, "process"),
        ("Write postmortem", 8, 1, "terminal")],
     [(0, 1, ""), (1, 2, ""), (2, 3, "No"), (2, 4, "Yes"), (4, 5, ""),
      (5, 6, ""), (6, 7, ""), (7, 8, "No"), (7, 9, "Yes"), (9, 10, "")]),
    ("Clinic appointment intake", "easy", "vertical", [
        ("Booking request", 1, 0, "terminal"), ("Collect symptoms", 1, 1, "process"),
        ("Urgent symptoms?", 1, 2, "decision"), ("Direct to urgent care", 0, 3, "terminal"),
        ("Offer appointment", 2, 3, "process"), ("Confirm booking", 2, 4, "terminal")],
     [(0, 1, ""), (1, 2, ""), (2, 3, "Yes"), (2, 4, "No"), (4, 5, "")]),
    ("Software release approval", "hard", "horizontal", [
        ("Release candidate", 0, 1, "terminal"), ("Run test suite", 1, 1, "process"),
        ("Tests pass?", 2, 1, "decision"), ("Return to developer", 3, 0, "terminal"),
        ("Security review", 3, 2, "process"), ("Approved?", 4, 2, "decision"),
        ("Remediate findings", 5, 3, "terminal"), ("Deploy canary", 5, 1, "process"),
        ("Metrics healthy?", 6, 1, "decision"), ("Roll back", 7, 2, "terminal"),
        ("Expand rollout", 7, 0, "process"), ("Release complete", 8, 0, "terminal")],
     [(0, 1, ""), (1, 2, ""), (2, 3, "No"), (2, 4, "Yes"), (4, 5, ""),
      (5, 6, "No"), (5, 7, "Yes"), (7, 8, ""), (8, 9, "No"), (8, 10, "Yes"), (10, 11, "")]),
    ("Supplier invoice processing", "medium", "vertical", [
        ("Invoice received", 1, 0, "terminal"), ("Match purchase order", 1, 1, "process"),
        ("Amounts match?", 1, 2, "decision"), ("Resolve discrepancy", 0, 3, "terminal"),
        ("Verify bank details", 2, 3, "process"), ("Details verified?", 2, 4, "decision"),
        ("Hold payment", 1, 5, "terminal"), ("Authorize transfer", 3, 5, "process"),
        ("Archive invoice", 3, 6, "terminal")],
     [(0, 1, ""), (1, 2, ""), (2, 3, "No"), (2, 4, "Yes"), (4, 5, ""),
      (5, 6, "No"), (5, 7, "Yes"), (7, 8, "")]),
    ("Data ingestion quality gate", "medium", "horizontal", [
        ("File uploaded", 0, 1, "terminal"), ("Validate schema", 1, 1, "process"),
        ("Schema valid?", 2, 1, "decision"), ("Quarantine file", 3, 0, "terminal"),
        ("Check data quality", 3, 2, "process"), ("Quality accepted?", 4, 2, "decision"),
        ("Send to review", 5, 3, "terminal"), ("Load warehouse", 5, 1, "process"),
        ("Publish dataset", 6, 1, "terminal")],
     [(0, 1, ""), (1, 2, ""), (2, 3, "No"), (2, 4, "Yes"), (4, 5, ""),
      (5, 6, "No"), (5, 7, "Yes"), (7, 8, "")]),
    ("Employee access request", "hard", "vertical", [
        ("Access requested", 1, 0, "terminal"), ("Verify employment", 1, 1, "process"),
        ("Active employee?", 1, 2, "decision"), ("Deny request", 0, 3, "terminal"),
        ("Manager approval", 2, 3, "process"), ("Approved?", 2, 4, "decision"),
        ("Close request", 1, 5, "terminal"), ("Check least privilege", 3, 5, "process"),
        ("Provision access", 3, 6, "process"), ("Record audit entry", 3, 7, "terminal")],
     [(0, 1, ""), (1, 2, ""), (2, 3, "No"), (2, 4, "Yes"), (4, 5, ""),
      (5, 6, "No"), (5, 7, "Yes"), (7, 8, ""), (8, 9, "")]),
    ("Library book reservation", "easy", "horizontal", [
        ("Reserve title", 0, 1, "terminal"), ("Check availability", 1, 1, "process"),
        ("Copy available?", 2, 1, "decision"), ("Add to waitlist", 3, 2, "terminal"),
        ("Hold copy", 3, 0, "process"), ("Notify reader", 4, 0, "terminal")],
     [(0, 1, ""), (1, 2, ""), (2, 3, "No"), (2, 4, "Yes"), (4, 5, "")]),
    ("Product return inspection", "medium", "vertical", [
        ("Return received", 1, 0, "terminal"), ("Inspect item", 1, 1, "process"),
        ("Resalable?", 1, 2, "decision"), ("Restock item", 0, 3, "process"),
        ("Recycle item", 2, 3, "process"), ("Issue refund", 1, 4, "process"),
        ("Notify customer", 1, 5, "terminal")],
     [(0, 1, ""), (1, 2, ""), (2, 3, "Yes"), (2, 4, "No"),
      (3, 5, ""), (4, 5, ""), (5, 6, "")]),
]

THEMES = [
    ("#f8fafc", "#dceafa", "#d9eee6", "#fff0cd", "#203247"),
    ("#fffaf3", "#efe4d8", "#e4ecdd", "#f5dfc8", "#44362c"),
    ("#152132", "#284461", "#26544e", "#624b36", "#f4f7fc"),
    ("#f8f6fc", "#e7e0f6", "#dbeee9", "#fae8d4", "#352d4b"),
    ("#fcfcf8", "#dce8df", "#e5e6d3", "#f3e1cf", "#293b32"),
]
INKS = ("#244e7b", "#9c3046", "#397056", "#75549a", "#9a5b21", "#286979")
PATTERNS = ("solid", "dashed", "dotted", "dashdot")


def font(size, serif=False):
    return ImageFont.truetype("C:\\Windows\\Fonts\\" + ("georgia.ttf" if serif else "arial.ttf"), size)


class Drawing:
    def __init__(self, width, height, background):
        self.image = Image.new("RGB", (width, height), background)
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    f'<rect width="100%" height="100%" fill="{background}"/>']

    def text(self, xy, text, color, size, serif=False):
        self.draw.text(xy, text, fill=color, font=font(size, serif), anchor="mm")
        self.svg.append(f'<text x="{xy[0]}" y="{xy[1]}" fill="{color}" font-family="{("Georgia" if serif else "Arial")}" font-size="{size}" text-anchor="middle" dominant-baseline="central">{escape(text)}</text>')

    def line(self, points, color, width, pattern):
        dash = {"solid": (), "dashed": (14, 8), "dotted": (3, 7), "dashdot": (14, 5, 3, 5)}[pattern]
        phase, remaining = 0, dash[0] if dash else 0
        for a, b in zip(points, points[1:]):
            distance = math.dist(a, b)
            if not dash:
                self.draw.line((a, b), fill=color, width=width)
                continue
            t = 0
            while t < distance:
                end = min(distance, t + remaining)
                if phase % 2 == 0:
                    self.draw.line([(a[0] + (b[0]-a[0])*u/distance, a[1] + (b[1]-a[1])*u/distance) for u in (t, end)], fill=color, width=width)
                remaining -= end-t
                t = end
                if remaining < 0.001:
                    phase = (phase+1) % len(dash)
                    remaining = dash[phase]
        coords = " ".join(f"{x},{y}" for x, y in points)
        dash_attr = f' stroke-dasharray="{",".join(map(str, dash))}"' if dash else ""
        self.svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attr}/>')

    def polygon(self, points, fill):
        self.draw.polygon(points, fill=fill)
        self.svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in points)}" fill="{fill}"/>')


def generate(index, scenario, output):
    title, difficulty, orientation, nodes, edges = scenario
    horizontal = orientation == "horizontal"
    nw, nh = (210, 96) if index % 3 else (230, 104)
    gx, gy = nw + (95 if horizontal else 115), nh + (100 if horizontal else 90)
    margin = 85 + (index % 3)*20
    width = int(2*margin + max(n[1] for n in nodes)*gx + nw)
    height = int(2*margin + 70 + max(n[2] for n in nodes)*gy + nh)
    bg, process, terminal, decision, ink = THEMES[index % len(THEMES)]
    canvas = Drawing(width, height, bg)
    canvas.text((width/2, 48), title, ink, 30, index % 4 == 1)
    boxes = [(margin+c*gx, margin+70+r*gy, margin+c*gx+nw, margin+70+r*gy+nh) for _, c, r, _ in nodes]
    for ei, (source, target, label) in enumerate(edges):
        a, b = boxes[source], boxes[target]
        ac, bc = ((a[0]+a[2])/2, (a[1]+a[3])/2), ((b[0]+b[2])/2, (b[1]+b[3])/2)
        if horizontal:
            start, end = (a[2], ac[1]), (b[0], bc[1])
            if ac[0] == bc[0]:
                start, end = (ac[0], a[3]), (bc[0], b[1])
            mid = (start[0]+end[0])/2
            points = [start, (mid, start[1]), (mid, end[1]), end] if start[1] != end[1] and start[0] != end[0] else [start, end]
        else:
            start, end = (ac[0], a[3]), (bc[0], b[1])
            mid = (start[1]+end[1])/2
            points = [start, (start[0], mid), (end[0], mid), end] if start[0] != end[0] else [start, end]
        color = INKS[(ei+index) % len(INKS)] if bg != "#152132" else ("#94c9f5", "#edb58f", "#b8dfb1", "#d2b3ed")[ei % 4]
        canvas.line(points, color, 3 + (ei % 2), PATTERNS[(ei+index) % 4])
        prev = next(p for p in reversed(points[:-1]) if p != end)
        angle = math.atan2(end[1]-prev[1], end[0]-prev[0])
        head = [end, (end[0]-13*math.cos(angle)+6*math.sin(angle), end[1]-13*math.sin(angle)-6*math.cos(angle)),
                (end[0]-13*math.cos(angle)-6*math.sin(angle), end[1]-13*math.sin(angle)+6*math.cos(angle))]
        canvas.polygon(head, color)
        if label:
            if horizontal:
                pos = (start[0]+25, start[1]-18 if end[1] <= start[1] else start[1]+20)
            else:
                pos = (start[0]+(-30 if end[0]<start[0] else 30), start[1]+24)
            canvas.text(pos, label, ink, 17)
    for ni, ((label, _, _, role), box) in enumerate(zip(nodes, boxes)):
        x0,y0,x1,y1 = box
        cx,cy = (x0+x1)/2,(y0+y1)/2
        fill = {"process":process,"terminal":terminal,"decision":decision}[role]
        border = INKS[(ni*2+index) % len(INKS)] if bg != "#152132" else ("#9bc9f1","#e3b598","#b7e0c4")[ni % 3]
        if role == "decision":
            points = [(cx,y0),(x1,cy),(cx,y1),(x0,cy),(cx,y0)]
        elif role == "terminal":
            radius = 22
            points = []
            for ox,oy,start in [(x1-radius,y0+radius,-90),(x1-radius,y1-radius,0),(x0+radius,y1-radius,90),(x0+radius,y0+radius,180)]:
                points.extend((ox+radius*math.cos(math.radians(start+t)),oy+radius*math.sin(math.radians(start+t))) for t in range(0,91,10))
            points.append(points[0])
        else:
            points = [(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)]
        canvas.polygon(points, fill)
        canvas.line(points, border, 3+(ni % 2), PATTERNS[(ni+index) % 4])
        words = label.split()
        lines = [label]
        limit = 17 if role == "decision" else 21
        if len(label)>limit:
            split = min(range(1,len(words)),key=lambda i:abs(len(" ".join(words[:i]))-len(" ".join(words[i:]))))
            lines = [" ".join(words[:split])," ".join(words[split:])]
        for li,line in enumerate(lines):
            canvas.text((cx,cy+(li-(len(lines)-1)/2)*24),line,ink,19 if index % 3 else 20,index % 4 == 1)
    item_id = f"MIXEDGE{index+1:02d}"
    canvas.image.save(output / f"{item_id}.png")
    (output / f"{item_id}.svg").write_text("\n".join(canvas.svg+["</svg>"]),encoding="utf-8")
    relations = "; ".join(f"{nodes[a][0]} -> {nodes[b][0]}" + (f" ({label})" if label else "") for a,b,label in edges)
    caption = f"{title}. Workflow connections: {relations}."
    return {"id":item_id,"file":f"{item_id}.png","difficulty":difficulty,
            "source":"Original synthetic flowchart; mixed-edge human-review batch","caption":caption,
            "label":"negative","defects":"Inconsistent box-border patterns/colors and connector patterns/colors",
            "orientation":orientation,"nodes":len(nodes),"connections":len(edges)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"examples"/"mixed_edges_review")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows = [generate(i,scenario,args.output) for i,scenario in enumerate(SCENARIOS)]
    sheet = Image.new("RGB",(1400,1800),"#e9eef5")
    sheet_draw = ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        with Image.open(args.output/row["file"]) as image:
            thumb = ImageOps.contain(image,(680,320))
            sheet.paste(thumb,((i%2)*700+(700-thumb.width)//2,(i//2)*360+30))
        sheet_draw.text(((i%2)*700+15,(i//2)*360+8),row["id"],fill="#172b43")
    sheet.save(args.output/"contact_sheet.png")
    fields = ["id","file","difficulty","source","caption","label","defects"]
    with (args.output/"manifest.csv").open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.output/"metadata.json").write_text(json.dumps({
        "purpose":"Candidate negative examples for human review; not model-scored or human-validated.",
        "difficulty_note":"Provisional structural complexity: easy 6-7 nodes; medium 7-9; hard 10-12. Not balanced for direct study import.",
        "boundary":"No active study catalog modified. Captions describe workflow content without revealing injected style defects.",
        "items":rows},indent=2),encoding="utf-8")
    cards = "".join(f'<article><h2>{r["id"]}: {escape(SCENARIOS[i][0])}</h2><p>{r["difficulty"]} · {r["nodes"]} nodes · {r["orientation"]}</p><a href="{r["file"]}"><img loading="lazy" src="{r["file"]}" alt="{escape(SCENARIOS[i][0])}"/></a><p>{escape(r["caption"])}</p><a href="{r["id"]}.svg">Editable SVG</a> · <a href="{r["file"]}">Full-size PNG</a></article>' for i,r in enumerate(rows))
    (args.output/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mixed-edge flowcharts · Researcher review</title><style>body{font:16px system-ui;background:#f2f5fa;color:#21334b;max-width:1400px;margin:auto;padding:28px}article{background:white;border:1px solid #d3deea;border-radius:12px;padding:22px;margin:24px 0}img{max-width:100%;max-height:720px;object-fit:contain}p{line-height:1.6}a{color:#205aac}</style><h1>Mixed-edge flowcharts: 10 candidate negatives</h1><p>Researcher-only gallery. Both box borders and arrows mix patterns and colors. Original realistic workflows; diverse layouts, palettes, complexity, typography and proportions. Difficulty and negative labels are provisional, not human-validated. No study catalog has been changed.</p><p><a href="manifest.csv">Import manifest</a> · <a href="metadata.json">Research metadata</a></p>'+cards+'</html>',encoding="utf-8")
    print(f"Generated {len(rows)} PNGs and SVGs, manifest, metadata, and gallery at {args.output}")


if __name__=="__main__":
    main()
