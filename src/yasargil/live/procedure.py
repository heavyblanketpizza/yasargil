"""Procedure specs and reflex rule defaults, as data.

The default procedure describes coarse tool-usage segments of a simulated
durotomy repair. SOSpine has no phase labels, so these steps are illustrative
and unvalidated; replace them with ``--procedure spec.json`` for real work.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from . import LiveError

DEFAULT_PROCEDURE = {
    "id": "durotomy-repair-illustrative-v0",
    "note": "Illustrative tool-usage segments for simulated durotomy repair; not validated against expert phase labels.",
    "initial": "idle",
    "dwell_frames": 3,
    "steps": [
        {"id": "suturing", "description": "Needle driver in view: suture passes or knot tying",
         "requires_any": ["needle driver"]},
        {"id": "exposure", "description": "Nerve hook or grasper without the needle driver: inspecting or exposing the tear",
         "requires_any": ["nerve hook", "grasper"]},
        {"id": "idle", "description": "No instruments in view", "requires_none": True},
    ],
    "transitions": {"idle": ["exposure", "suturing"], "exposure": ["suturing", "idle"],
                    "suturing": ["exposure", "idle"]},
    "expected_order": ["exposure", "suturing"],
}

DEFAULT_RULES = [
    {"id": "needle_out_of_view", "when": {"type": "instrument_left", "subject": "needle"},
     "require": {"visible_instruments": ["needle driver"]}, "priority": "critical", "cooldown_ms": 10000,
     "message": "Needle left the view while the needle driver is still in."},
    {"id": "tip_near_durotomy", "when": {"type": "tip_near_structure", "object": "durotomy"},
     "priority": "high", "cooldown_ms": 15000, "message": "{subject} tip is close to the durotomy."},
    {"id": "view_lost", "when": {"type": "view_empty"}, "priority": "medium", "cooldown_ms": 30000,
     "message": "No instruments or durotomy in view."},
    {"id": "unexpected_step", "when": {"type": "step_unexpected"}, "priority": "low", "cooldown_ms": 60000,
     "message": "Unexpected step: {reason}."},
]


@dataclass(frozen=True)
class ProcedureSpec:
    id: str
    initial: str
    dwell_frames: int
    steps: tuple
    transitions: dict
    expected_order: tuple
    note: str = ""

    @classmethod
    def load(cls, path):
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise LiveError(f"Cannot read procedure spec {path}: {exc}") from exc

    @classmethod
    def from_dict(cls, value):
        def require(condition, message):
            if not condition:
                raise LiveError(f"Invalid procedure spec: {message}")

        require(isinstance(value, dict) and isinstance(value.get("id"), str) and value["id"], "missing id")
        steps = value.get("steps")
        require(isinstance(steps, list) and steps, "steps must be a nonempty list")
        ids = []
        for step in steps:
            require(isinstance(step, dict) and isinstance(step.get("id"), str) and step["id"], "every step needs an id")
            conditions = [key for key in ("requires_any", "requires_all", "requires_none") if key in step]
            require(len(conditions) == 1, f"step {step['id']} needs exactly one condition")
            if "requires_none" in step:
                require(step["requires_none"] is True, f"step {step['id']} requires_none must be true")
            else:
                listed = step[conditions[0]]
                require(isinstance(listed, list) and listed and all(isinstance(i, str) for i in listed),
                        f"step {step['id']} condition must list instruments")
            ids.append(step["id"])
        require(len(set(ids)) == len(ids), "step ids must be unique")
        initial = value.get("initial", ids[-1])
        require(initial in ids, "initial must name a step")
        dwell = value.get("dwell_frames", 3)
        require(type(dwell) is int and dwell >= 1, "dwell_frames must be a positive integer")
        transitions = value.get("transitions", {step: [other for other in ids if other != step] for step in ids})
        require(isinstance(transitions, dict) and all(key in ids and isinstance(targets, list)
                and all(target in ids for target in targets) for key, targets in transitions.items()),
                "transitions must map step ids to lists of step ids")
        order = value.get("expected_order", [])
        require(isinstance(order, list) and all(step in ids for step in order), "expected_order must list step ids")
        return cls(value["id"], initial, dwell, tuple(steps), {k: tuple(v) for k, v in transitions.items()},
                   tuple(order), value.get("note", ""))

    def classify(self, visible):
        """The first step whose condition matches the set of visible instruments."""
        for step in self.steps:
            if "requires_none" in step and not visible:
                return step["id"]
            if "requires_any" in step and visible.intersection(step["requires_any"]):
                return step["id"]
            if "requires_all" in step and visible.issuperset(step["requires_all"]):
                return step["id"]
        return None

    def description(self, step_id):
        return next((step.get("description", "") for step in self.steps if step["id"] == step_id), "")

    def to_json(self):
        return {"id": self.id, "note": self.note, "initial": self.initial, "dwell_frames": self.dwell_frames,
                "steps": list(self.steps), "transitions": {k: list(v) for k, v in self.transitions.items()},
                "expected_order": list(self.expected_order)}
