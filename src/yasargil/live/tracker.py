"""From observations to events, and from events to procedure steps.

``EventBuilder`` applies hysteresis: an object enters after ``enter_frames``
consecutive confident detections and leaves after ``exit_frames`` consecutive
misses. Frames that were never observed (missing releases, unlabeled frames
under label perception) are not evidence of absence and do not advance either
counter. Distances use units of image height, with x scaled by the aspect ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from . import LiveError


@dataclass(frozen=True)
class TrackerConfig:
    enter_frames: int = 1
    exit_frames: int = 2
    min_confidence: float = 0.5
    near_distance: float = 0.08
    clear_factor: float = 1.5
    near_frames: int = 1
    empty_frames: int = 3
    aspect: float = 16 / 9
    structures: tuple = ("durotomy",)

    def __post_init__(self):
        for name in ("enter_frames", "exit_frames", "near_frames", "empty_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise LiveError(f"{name} must be a positive integer")
        if not (0 <= self.min_confidence <= 1 and self.near_distance > 0 and self.clear_factor >= 1 and self.aspect > 0):
            raise LiveError("Invalid tracker thresholds")

    def to_json(self):
        return {key: (list(value) if isinstance(value, tuple) else value) for key, value in self.__dict__.items()}


def tip_distance(tip, detection, aspect):
    """Distance from an instrument tip to a structure's box (zero inside) or point."""
    x, y = tip
    if detection.box:
        x1, y1, x2, y2 = detection.box
        dx, dy = max(x1 - x, 0.0, x - x2), max(y1 - y, 0.0, y - y2)
    elif detection.tip:
        dx, dy = abs(x - detection.tip[0]), abs(y - detection.tip[1])
    else:
        return None
    return math.hypot(dx * aspect, dy)


class _Track:
    def __init__(self, kind):
        self.kind = kind
        self.visible = False
        self.streak = []
        self.absent = []
        self.near = {}
        self.near_streak = {}


class EventBuilder:
    producer = "tracker/v1"

    def __init__(self, log, config=None):
        self.log = log
        self.config = config or TrackerConfig()
        self.tracks = {}
        self.empty = []
        self.empty_flag = False
        self.latest = None

    def update(self, observation):
        if not observation.annotated:
            return []
        config = self.config
        self.latest = observation
        present = {d.label: d for d in observation.detections if d.confidence >= config.min_confidence}
        moment = (observation.frame_index, observation.t_ms)
        emitted = []

        def emit(type, subject, onset_ms, frames, **fields):
            emitted.append(self.log.append(observation.t_ms, observation.frame_index, type, subject,
                                           onset_ms=onset_ms, evidence_frames=frames, producer=self.producer, **fields))

        for label in sorted(set(present) | set(self.tracks)):
            detection = present.get(label)
            track = self.tracks.setdefault(label, _Track(detection.kind if detection else "instrument"))
            if detection is not None:
                track.absent = []
                track.streak.append((moment, detection.confidence))
                if not track.visible and len(track.streak) >= config.enter_frames:
                    track.visible = True
                    window = track.streak[-config.enter_frames:]
                    emit("instrument_entered" if track.kind == "instrument" else "anatomy_visible", label,
                         window[0][0][1], tuple(m[0] for m, _ in window),
                         confidence=sum(c for _, c in window) / len(window))
            else:
                track.streak = []
                if track.visible:
                    track.absent.append(moment)
                    if len(track.absent) >= config.exit_frames:
                        track.visible = False
                        track.near, track.near_streak = {}, {}
                        emit("instrument_left" if track.kind == "instrument" else "anatomy_hidden", label,
                             track.absent[0][1], tuple(m[0] for m in track.absent))
                        track.absent = []

        for label in sorted(present):
            detection, track = present[label], self.tracks[label]
            if track.kind != "instrument" or not track.visible or detection.tip is None:
                continue
            for structure in config.structures:
                target = present.get(structure)
                anatomy = self.tracks.get(structure)
                if target is None or anatomy is None or not anatomy.visible:
                    track.near_streak.pop(structure, None)
                    continue
                distance = tip_distance(detection.tip, target, config.aspect)
                if distance is None:
                    continue
                if structure in track.near:
                    if distance > config.near_distance * config.clear_factor:
                        del track.near[structure]
                        emit("tip_cleared_structure", label, observation.t_ms, (observation.frame_index,),
                             object=structure, data={"distance": round(distance, 4)})
                elif distance <= config.near_distance:
                    streak = track.near_streak.get(structure, []) + [moment]
                    track.near_streak[structure] = streak
                    if len(streak) >= config.near_frames:
                        track.near[structure] = True
                        track.near_streak.pop(structure)
                        emit("tip_near_structure", label, streak[0][1], tuple(m[0] for m in streak),
                             object=structure, confidence=detection.confidence * target.confidence,
                             data={"distance": round(distance, 4)})
                else:
                    track.near_streak.pop(structure, None)

        if present:
            if self.empty_flag:
                self.empty_flag = False
                emit("view_restored", "view", observation.t_ms, (observation.frame_index,))
            self.empty = []
        else:
            self.empty.append(moment)
            if not self.empty_flag and len(self.empty) >= config.empty_frames:
                self.empty_flag = True
                emit("view_empty", "view", self.empty[0][1], tuple(m[0] for m in self.empty))
        return emitted


class ProcedureTracker:
    """Coarse steps from visible instruments, with dwell and transition checks."""

    producer = "procedure/v1"

    def __init__(self, log, spec):
        self.log = log
        self.spec = spec
        self.current = spec.initial
        self.visited = set()
        self.candidate = None
        self.streak = []

    def update(self, now_ms, frame_index, state):
        visible = set(state.visible_instruments)
        step = self.spec.classify(visible)
        if step is None or step == self.current:
            self.candidate, self.streak = None, []
            return []
        if step != self.candidate:
            self.candidate, self.streak = step, []
        self.streak.append((frame_index, now_ms))
        if len(self.streak) < self.spec.dwell_frames:
            return []
        previous = self.current
        cites = tuple(sorted(state.visible_instruments.values()))
        changed = self.log.append(now_ms, frame_index, "step_changed", step, onset_ms=self.streak[0][1],
                                  evidence_frames=tuple(i for i, _ in self.streak), cites=cites,
                                  producer=self.producer, data={"from": previous,
                                                                "description": self.spec.description(step)})
        events = [changed]
        reason = None
        if step not in self.spec.transitions.get(previous, ()):
            reason = f"{previous} -> {step} is not an allowed transition"
        elif step in self.spec.expected_order:
            earlier = self.spec.expected_order[: self.spec.expected_order.index(step)]
            skipped = [name for name in earlier if name not in self.visited]
            if skipped and step not in self.visited:
                reason = f"{step} began before {', '.join(skipped)} was observed"
        if reason:
            events.append(self.log.append(now_ms, frame_index, "step_unexpected", step, cites=(changed.event_id,),
                                          producer=self.producer, data={"from": previous, "reason": reason}))
        self.visited.add(step)
        self.current, self.candidate, self.streak = step, None, []
        return events
