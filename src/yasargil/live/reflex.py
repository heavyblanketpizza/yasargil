"""Reflex rules: declarative, event-driven, and never an LLM.

A rule matches an event's type (and optionally subject/object), may require
current state, and produces a templated message. Each firing is also written to
the event log as an ``alert`` that cites its trigger, so the agent can see what
was already said.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import LiveError
from .events import EVENT_TYPES
from .gate import PRIORITIES, Candidate


class _Fields(dict):
    def __missing__(self, key):
        return "unknown"


def _validate(rules):
    if not isinstance(rules, list):
        raise LiveError("Reflex rules must be a list")
    seen = set()
    for rule in rules:
        if not isinstance(rule, dict) or not isinstance(rule.get("id"), str) or not rule["id"] or rule["id"] in seen:
            raise LiveError("Every reflex rule needs a unique id")
        seen.add(rule["id"])
        when = rule.get("when")
        if not isinstance(when, dict) or when.get("type") not in EVENT_TYPES:
            raise LiveError(f"Rule {rule['id']}: 'when.type' must be an event type")
        if rule.get("priority") not in PRIORITIES:
            raise LiveError(f"Rule {rule['id']}: priority must be one of {', '.join(PRIORITIES)}")
        if not isinstance(rule.get("message"), str) or not rule["message"]:
            raise LiveError(f"Rule {rule['id']}: message is required")
        cooldown = rule.get("cooldown_ms")
        if cooldown is not None and (type(cooldown) is not int or cooldown < 0):
            raise LiveError(f"Rule {rule['id']}: cooldown_ms must be a nonnegative integer")
        require = rule.get("require", {})
        if not isinstance(require, dict) or any(key not in ("visible_instruments", "absent_instruments", "step")
                                                for key in require):
            raise LiveError(f"Rule {rule['id']}: unsupported 'require' keys")
    return rules


def load_rules(path):
    try:
        return _validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise LiveError(f"Cannot read reflex rules {path}: {exc}") from exc


class ReflexEngine:
    producer = "reflex/v1"

    def __init__(self, rules, log=None):
        self.rules = _validate(rules)
        self.log = log

    def on_events(self, events, state, now_ms):
        candidates = []
        for event in events:
            for rule in self.rules:
                when = rule["when"]
                if (event.type != when["type"] or when.get("subject", event.subject) != event.subject
                        or when.get("object", event.object) != event.object):
                    continue
                support = self._satisfied(rule.get("require", {}), state)
                if support is None:
                    continue
                fields = _Fields({key: value for key, value in event.data.items()})
                fields.update(subject=event.subject, object=event.object or "", step=state.step[0] if state.step else "")
                text = rule["message"].format_map(fields)
                text = text[:1].upper() + text[1:]
                cites = (event.event_id,) + support
                if self.log is not None:
                    alert = self.log.append(now_ms, event.frame_index, "alert", rule["id"], object=event.subject,
                                            confidence=event.confidence, cites=cites, producer=self.producer,
                                            data={"priority": rule["priority"], "message": text})
                    cites = (alert.event_id,) + cites
                candidates.append(Candidate("reflex", rule["priority"], f"rule:{rule['id']}:{event.subject}", text,
                                            cites, event.confidence, event.t_ms, None, rule.get("cooldown_ms")))
        return candidates

    @staticmethod
    def _satisfied(require, state):
        """Event IDs supporting the requirement, or None when it does not hold."""
        support = []
        for label in require.get("visible_instruments", []):
            if label not in state.visible_instruments:
                return None
            support.append(state.visible_instruments[label])
        if any(label in state.visible_instruments for label in require.get("absent_instruments", [])):
            return None
        if "step" in require:
            if not state.step or state.step[0] != require["step"]:
                return None
            support.append(state.step[1])
        return tuple(support)
