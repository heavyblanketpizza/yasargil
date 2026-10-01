"""The append-only event log every loop reads.

Events are state changes, not frames. Each records when it was emitted
(``t_ms``), when the change began (``onset_ms``), the frames that support it and
any events it cites. Queries always take ``now_ms`` and never return the
future, so an agent answering at time t sees exactly what was known at t.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading

from . import LiveError

EVENT_TYPES = (
    "instrument_entered", "instrument_left", "anatomy_visible", "anatomy_hidden",
    "tip_near_structure", "tip_cleared_structure", "view_empty", "view_restored",
    "step_changed", "step_unexpected", "alert",
)


@dataclass(frozen=True)
class Event:
    event_id: str
    t_ms: int
    onset_ms: int
    frame_index: int
    type: str
    subject: str
    object: str | None = None
    confidence: float = 1.0
    evidence_frames: tuple = ()
    cites: tuple = ()
    producer: str = ""
    data: dict = field(default_factory=dict)

    def to_json(self):
        return {"event_id": self.event_id, "t_ms": self.t_ms, "onset_ms": self.onset_ms,
                "frame_index": self.frame_index, "type": self.type, "subject": self.subject,
                "object": self.object, "confidence": round(self.confidence, 4),
                "evidence_frames": list(self.evidence_frames), "cites": list(self.cites),
                "producer": self.producer, "data": self.data}

    @classmethod
    def from_json(cls, value):
        return cls(value["event_id"], value["t_ms"], value["onset_ms"], value["frame_index"], value["type"],
                   value["subject"], value.get("object"), value.get("confidence", 1.0),
                   tuple(value.get("evidence_frames", ())), tuple(value.get("cites", ())),
                   value.get("producer", ""), dict(value.get("data") or {}))

    def line(self):
        """One compact line for the agent's context and tool results."""
        who = f"{self.subject} -> {self.object}" if self.object else self.subject
        parts = [self.event_id, f"t={self.t_ms / 1000:.1f}s", self.type, who]
        details = []
        if self.onset_ms != self.t_ms:
            details.append(f"onset={self.onset_ms / 1000:.1f}s")
        if self.confidence < 1.0:
            details.append(f"conf={self.confidence:.2f}")
        if self.evidence_frames:
            frames = self.evidence_frames
            details.append(f"frames={frames[0]}" if len(frames) == 1 else f"frames={frames[0]}-{frames[-1]}")
        for key, value in sorted(self.data.items()):
            details.append(f"{key}={value:.3g}" if isinstance(value, float) else f"{key}={value}")
        if self.cites:
            details.append("cites=" + ",".join(self.cites))
        return " ".join(parts) + (f" ({', '.join(details)})" if details else "")


@dataclass
class State:
    """What the event log implies at one moment."""

    now_ms: int
    visible_instruments: dict = field(default_factory=dict)
    visible_anatomy: dict = field(default_factory=dict)
    near: dict = field(default_factory=dict)
    step: tuple | None = None
    view_empty: str | None = None

    def to_json(self):
        return {"now_ms": self.now_ms, "visible_instruments": self.visible_instruments,
                "visible_anatomy": self.visible_anatomy,
                "near": {key: list(value) for key, value in self.near.items()},
                "step": list(self.step) if self.step else None, "view_empty": self.view_empty}


class EventLog:
    """Thread-safe append-only log; optionally mirrored to JSONL as it grows."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._events = []
        self._by_id = {}
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path):
        log = cls()
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    event = Event.from_json(json.loads(line))
                    log._events.append(event)
                    log._by_id[event.event_id] = event
        return log

    def append(self, t_ms, frame_index, type, subject, *, object=None, onset_ms=None, confidence=1.0,
               evidence_frames=(), cites=(), producer="", data=None):
        if type not in EVENT_TYPES:
            raise LiveError(f"Unknown event type: {type}")
        with self._lock:
            if self._events and t_ms < self._events[-1].t_ms:
                raise LiveError("Event time went backwards; the log is append-only in time.")
            missing = [event_id for event_id in cites if event_id not in self._by_id]
            if missing:
                raise LiveError(f"Event cites unknown events: {', '.join(missing)}")
            event = Event(f"E{len(self._events) + 1:06d}", int(t_ms), int(t_ms if onset_ms is None else onset_ms),
                          int(frame_index), type, subject, object, float(confidence), tuple(evidence_frames),
                          tuple(cites), producer, dict(data or {}))
            self._events.append(event)
            self._by_id[event.event_id] = event
            if self.path:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event.to_json(), ensure_ascii=False) + "\n")
            return event

    def get(self, event_id):
        return self._by_id.get(event_id)

    def all(self):
        with self._lock:
            return list(self._events)

    def query(self, now_ms, types=None, subject=None, since_ms=None, until_ms=None, limit=50):
        """Events known at ``now_ms``, oldest first; the most recent ``limit`` when truncated."""
        end = now_ms if until_ms is None else min(now_ms, until_ms)
        wanted = set(types) if types else None
        result = [event for event in self.all()
                  if event.t_ms <= end and (since_ms is None or event.t_ms >= since_ms)
                  and (wanted is None or event.type in wanted)
                  and (subject is None or subject in (event.subject, event.object))]
        return result[-limit:] if limit else result

    def state_at(self, now_ms):
        state = State(now_ms)
        for event in self.query(now_ms, limit=0):
            if event.type == "instrument_entered":
                state.visible_instruments[event.subject] = event.event_id
            elif event.type == "instrument_left":
                state.visible_instruments.pop(event.subject, None)
                state.near.pop(event.subject, None)
            elif event.type == "anatomy_visible":
                state.visible_anatomy[event.subject] = event.event_id
            elif event.type == "anatomy_hidden":
                state.visible_anatomy.pop(event.subject, None)
            elif event.type == "tip_near_structure":
                state.near[event.subject] = (event.object, event.event_id)
            elif event.type == "tip_cleared_structure":
                state.near.pop(event.subject, None)
            elif event.type == "step_changed":
                state.step = (event.subject, event.event_id)
            elif event.type == "view_empty":
                state.view_empty = event.event_id
            elif event.type == "view_restored":
                state.view_empty = None
        return state
