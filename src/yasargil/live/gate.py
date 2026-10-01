"""The speak gate: the only path from any loop to the surgeon.

Unsolicited messages must be fresh, confident, outside their key's cooldown and
within a rolling per-minute budget; critical messages skip only the budget.
Answers to a direct question are always spoken. Every decision, spoken or not,
is recorded with its reason so silence is auditable too.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

PRIORITIES = ("low", "medium", "high", "critical")


@dataclass(frozen=True)
class Candidate:
    source: str
    priority: str
    key: str
    text: str
    cites: tuple = ()
    confidence: float = 1.0
    t_ms: int = 0
    question_id: str | None = None
    cooldown_ms: int | None = None

    def to_json(self):
        return {"source": self.source, "priority": self.priority, "key": self.key, "text": self.text,
                "cites": list(self.cites), "confidence": round(self.confidence, 4), "t_ms": self.t_ms,
                "question_id": self.question_id}


@dataclass(frozen=True)
class GateConfig:
    min_confidence: float = 0.5
    default_cooldown_ms: int = 20000
    max_per_minute: int = 4
    max_age_ms: int = 5000

    def to_json(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class Decision:
    spoken: bool
    reason: str
    record: dict


class SpeakGate:
    def __init__(self, config=None, path=None):
        self.config = config or GateConfig()
        self.path = Path(path) if path else None
        self.records = []
        self._last_spoken = {}
        self._unsolicited = []

    def offer(self, candidate, now_ms):
        config = self.config
        if candidate.question_id is not None:
            reason = "answer"
        elif now_ms - candidate.t_ms > config.max_age_ms:
            reason = "stale"
        elif candidate.confidence < config.min_confidence:
            reason = "low_confidence"
        elif (candidate.key in self._last_spoken and now_ms - self._last_spoken[candidate.key]
              < (config.default_cooldown_ms if candidate.cooldown_ms is None else candidate.cooldown_ms)):
            reason = "cooldown"
        elif (candidate.priority != "critical"
              and sum(1 for t in self._unsolicited if now_ms - t < 60000) >= config.max_per_minute):
            reason = "rate_limited"
        else:
            reason = "spoken"
        spoken = reason in ("spoken", "answer")
        if spoken:
            self._last_spoken[candidate.key] = now_ms
            if candidate.question_id is None:
                self._unsolicited.append(now_ms)
        record = {"at_ms": now_ms, "spoken": spoken, "reason": reason, **candidate.to_json()}
        self.records.append(record)
        if self.path:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return Decision(spoken, reason, record)
