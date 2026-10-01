"""Synthetic SOSpine-shaped cases for the live guidance tests.

A timeline maps release indices to label rows ``(label, x1, y1, x2, y2)`` in
pixels. Equal corners are written as points to the tool-tip table and other
rows as boxes to the box table, which is how the importer distinguishes them.
"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image


def make_case(root, case_id, timeline, *, frame_count=None, size=(64, 36),
              missing=(), unlabeled=(), append=False):
    """Write frames and both label tables; return the dataset root."""
    root = Path(root)
    frame_dir = root / "frames" / case_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    count = frame_count or max(timeline, default=1)
    for index in range(1, count + 1):
        if index in missing:
            continue
        shade = (index * 37) % 200 + 30
        Image.new("RGB", size, (shade, 90, 120)).save(frame_dir / f"{case_id}_frame_{index:08d}.jpeg", "JPEG")
    points, boxes = [], []
    for index in range(1, count + 1):
        if index in unlabeled or index in missing:
            continue
        name = f"{case_id}_frame_{index:08d}.jpeg"
        rows = timeline.get(index, [])
        if not rows:
            points.append({"trial_frame": name, "x1": "", "y1": "", "x2": "", "y2": "", "label": ""})
        for label, x1, y1, x2, y2 in rows:
            row = {"trial_frame": name, "x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2), "label": label}
            (points if (x1, y1) == (x2, y2) else boxes).append(row)
    _write(root / "sospine_tool_tips.csv", ["", "trial_frame", "x1", "y1", "x2", "y2", "label"],
           [{"": str(i), **row} for i, row in enumerate(points)], append)
    _write(root / "sospine_bbox.csv", ["trial_frame", "x1", "y1", "x2", "y2", "label"], boxes, append)
    _write(root / "sospine_outcomes.csv", ["Trial ID", "Time for repair", "Leak At 40mmHg"],
           [{"Trial ID": case_id, "Time for repair": str(count), "Leak At 40mmHg": "N"}], append)
    return root


def _write(path, fields, rows, append):
    exists = append and path.exists()
    with path.open("a" if exists else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def tip(label, x, y):
    return (label, x, y, x, y)


def box(label, x1, y1, x2, y2):
    return (label, x1, y1, x2, y2)


def scripted_timeline():
    """A 12-frame repair: hook explores, driver sutures near the tear, needle briefly lost."""
    durotomy = box("durotomy", 20, 10, 40, 26)
    timeline = {}
    for index in range(1, 13):
        rows = [durotomy]
        if 2 <= index <= 4:
            rows.append(tip("nerve hook", 10, 30))
        if 5 <= index <= 11:
            rows.append(tip("needle driver tip", 30 if index >= 7 else 55, 18 if index >= 7 else 5))
            rows.append(tip("needle driver base", 60, 2))
        if 5 <= index <= 8 or index == 11:
            rows.append(tip("needle tip", 33, 20))
        timeline[index] = rows
    timeline[12] = []
    return timeline
