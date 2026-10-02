"""SOSpine tool-tip and box tables as per-frame normalized geometry.

Rows with equal corners are points; other rows are boxes. A frame that appears
in either table is annotated, even when its only row has an empty label, which
the release uses for "nothing labeled here". Frames absent from both tables are
unlabeled and carry no ground truth.
"""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import math
from pathlib import Path

from . import LiveError
from .frames import FRAME_NAME

POINTS = "sospine_tool_tips.csv"
BOXES = "sospine_bbox.csv"
INSTRUMENTS = ("grasper", "needle driver", "nerve hook", "needle")
ANATOMY = ("durotomy",)


def normalize_label(raw):
    """Return ``(name, part, kind)`` for a known label, else ``None``."""
    text = " ".join(str(raw or "").split()).lower()
    if not text:
        return None
    part = "body"
    for suffix in ("tip", "base"):
        if text.endswith(" " + suffix):
            text, part = text[: -len(suffix) - 1], suffix
    if text in INSTRUMENTS:
        return text, part, "instrument"
    if text in ANATOMY:
        return text, part, "anatomy"
    return None


@dataclass(frozen=True)
class LabelGeometry:
    label: str
    kind: str
    part: str
    point: tuple | None
    box: tuple | None
    source_table: str = ""


class CaseLabels:
    """Normalized label geometry for one case, keyed by release index."""

    def __init__(self, case_id, items, annotated, unknown_labels, invalid_rows):
        self.case_id = case_id
        self._items = items
        self._annotated = annotated
        self.unknown_labels = dict(unknown_labels)
        self.invalid_rows = invalid_rows

    @property
    def indices(self):
        return sorted(self._annotated)

    def annotated(self, index):
        return index in self._annotated

    def items(self, index):
        return list(self._items.get(index, []))

    @classmethod
    def load(cls, dataset_root, case_id, frame_size):
        root = Path(dataset_root)
        width, height = frame_size
        items, annotated, unknown, invalid = {}, set(), Counter(), 0
        found_table = False
        for name in (POINTS, BOXES):
            path = root / name
            if not path.is_file():
                continue
            found_table = True
            with path.open(newline="", encoding="utf-8-sig") as stream:
                for row in csv.DictReader(stream):
                    match = FRAME_NAME.match((row.get("trial_frame") or "").strip())
                    if not match or match["case"] != case_id:
                        continue
                    index = int(match["index"])
                    annotated.add(index)
                    raw_label = row.get("label") or ""
                    normalized = normalize_label(raw_label)
                    if normalized is None:
                        if raw_label.strip():
                            unknown[" ".join(raw_label.split()).lower()] += 1
                        continue
                    try:
                        x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
                        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                            raise ValueError
                    except (KeyError, TypeError, ValueError):
                        invalid += 1
                        continue
                    nx1, nx2 = (_clamp(value / width) for value in (x1, x2))
                    ny1, ny2 = (_clamp(value / height) for value in (y1, y2))
                    if (x1, y1) == (x2, y2):
                        point, box = (nx1, ny1), None
                    else:
                        point, box = None, (min(nx1, nx2), min(ny1, ny2), max(nx1, nx2), max(ny1, ny2))
                    label, part, kind = normalized
                    geometry = LabelGeometry(label, kind, part, point, box, name)
                    bucket = items.setdefault(index, [])
                    if geometry not in bucket:
                        bucket.append(geometry)
        if not found_table:
            raise LiveError(f"No SOSpine label tables ({POINTS}, {BOXES}) under {root}")
        return cls(case_id, items, annotated, unknown, invalid)


def _clamp(value):
    return min(max(value, 0.0), 1.0)
