"""Perception sources: what the fast loop believes is in each frame.

Every source returns an ``Observation`` with one ``Detection`` per label, in
normalized coordinates. ``LabelPerception`` replays the dataset's own labels and
is an upper bound, not a model. ``DegradedPerception`` wraps any source with
seeded misses, false positives, jitter and confidence noise, so the agent can be
measured against a dial of perception quality.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from .labels import INSTRUMENTS


@dataclass(frozen=True)
class Detection:
    label: str
    kind: str
    confidence: float
    tip: tuple | None = None
    box: tuple | None = None

    def to_json(self):
        return {"label": self.label, "kind": self.kind, "confidence": round(self.confidence, 4),
                "tip": list(self.tip) if self.tip else None, "box": list(self.box) if self.box else None}

    @classmethod
    def from_json(cls, value):
        return cls(value["label"], value["kind"], value["confidence"],
                   tuple(value["tip"]) if value.get("tip") else None,
                   tuple(value["box"]) if value.get("box") else None)


@dataclass(frozen=True)
class Observation:
    frame_index: int
    t_ms: int
    detections: tuple = ()
    annotated: bool = True
    producer: str = ""

    @property
    def instruments(self):
        return tuple(d for d in self.detections if d.kind == "instrument")

    @property
    def anatomy(self):
        return tuple(d for d in self.detections if d.kind == "anatomy")

    def find(self, label):
        return next((d for d in self.detections if d.label == label), None)

    def to_json(self):
        return {"frame_index": self.frame_index, "t_ms": self.t_ms, "annotated": self.annotated,
                "producer": self.producer, "detections": [d.to_json() for d in self.detections]}


def crop_observation(observation, crop):
    """Keep detections inside a normalized crop and express them in crop space."""
    x1, y1, x2, y2 = crop
    width, height = x2 - x1, y2 - y1

    def inside(point):
        return x1 <= point[0] < x2 and y1 <= point[1] < y2

    def remap(point):
        return ((point[0] - x1) / width, (point[1] - y1) / height)

    kept = []
    for detection in observation.detections:
        box = None
        if detection.box:
            bx1, by1, bx2, by2 = max(detection.box[0], x1), max(detection.box[1], y1), min(detection.box[2], x2), min(detection.box[3], y2)
            if bx2 > bx1 and by2 > by1:
                box = remap((bx1, by1)) + remap((bx2, by2))
        if detection.tip is not None:
            if not inside(detection.tip):
                continue
            kept.append(Detection(detection.label, detection.kind, detection.confidence, remap(detection.tip), box))
        elif box is not None:
            kept.append(Detection(detection.label, detection.kind, detection.confidence, None, box))
    return Observation(observation.frame_index, observation.t_ms, tuple(kept), observation.annotated,
                       observation.producer + "+crop")


class LabelPerception:
    """Dataset labels replayed as perception: the ceiling every model is compared with."""

    producer = "labels/v1"

    def __init__(self, labels):
        self.labels = labels

    def identity(self):
        return {"kind": "labels", "producer": self.producer, "case_id": self.labels.case_id,
                "note": "Ground-truth labels replayed as perception; an upper bound, not a model."}

    def observe(self, frame):
        grouped = {}
        for item in self.labels.items(frame.index):
            grouped.setdefault(item.label, []).append(item)
        detections = []
        for label in sorted(grouped):
            items = grouped[label]
            points = [item for item in items if item.point is not None]
            tip = next((item.point for item in points if item.part == "tip"), None)
            if tip is None:
                tip = next((item.point for item in points if item.part == "body"), None)
            box = next((item.box for item in items if item.box is not None), None)
            if box is None and len(points) >= 2:
                xs, ys = [item.point[0] for item in points], [item.point[1] for item in points]
                box = (min(xs), min(ys), max(xs), max(ys))
            detections.append(Detection(label, items[0].kind, 1.0, tip, box))
        return Observation(frame.index, frame.t_ms, tuple(detections), self.labels.annotated(frame.index),
                           self.producer)

    def observe_crop(self, frame, crop):
        return crop_observation(self.observe(frame), crop)


class DegradedPerception:
    """Seeded corruption of another source, deterministic per frame and crop."""

    def __init__(self, inner, miss_rate=0.0, false_positive_rate=0.0, jitter=0.0, confidence_noise=0.0, seed=0):
        for name, value in (("miss_rate", miss_rate), ("false_positive_rate", false_positive_rate)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name, value in (("jitter", jitter), ("confidence_noise", confidence_noise)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        self.inner = inner
        self.miss_rate = miss_rate
        self.false_positive_rate = false_positive_rate
        self.jitter = jitter
        self.confidence_noise = confidence_noise
        self.seed = seed
        self.producer = f"degraded({inner.producer})/v1"

    def identity(self):
        return {"kind": "degraded", "producer": self.producer, "inner": self.inner.identity(),
                "miss_rate": self.miss_rate, "false_positive_rate": self.false_positive_rate,
                "jitter": self.jitter, "confidence_noise": self.confidence_noise, "seed": self.seed}

    def observe(self, frame):
        return self._degrade(self.inner.observe(frame), f"{self.seed}:{frame.index}")

    def observe_crop(self, frame, crop):
        return self._degrade(self.inner.observe_crop(frame, crop), f"{self.seed}:{frame.index}:{crop}")

    def _degrade(self, observation, key):
        rng = random.Random(key)
        kept = []
        for detection in observation.detections:
            if rng.random() < self.miss_rate:
                continue
            confidence = detection.confidence
            if self.confidence_noise:
                confidence = min(max(confidence - abs(rng.gauss(0, self.confidence_noise)), 0.05), 1.0)
            kept.append(Detection(detection.label, detection.kind, confidence,
                                  self._shift(detection.tip, rng), self._shift(detection.box, rng)))
        if rng.random() < self.false_positive_rate:
            present = {d.label for d in kept}
            choices = [label for label in INSTRUMENTS if label not in present]
            if choices:
                kept.append(Detection(rng.choice(choices), "instrument", rng.uniform(0.3, 0.7),
                                      (rng.random(), rng.random()), None))
        return Observation(observation.frame_index, observation.t_ms, tuple(kept), observation.annotated,
                           self.producer)

    def _shift(self, values, rng):
        if values is None or not self.jitter:
            return values
        shifted = [min(max(value + rng.gauss(0, self.jitter), 0.0), 1.0) for value in values]
        if len(shifted) == 4:
            shifted = [min(shifted[0], shifted[2]), min(shifted[1], shifted[3]),
                       max(shifted[0], shifted[2]), max(shifted[1], shifted[3])]
        return tuple(shifted)
