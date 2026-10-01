"""One replay: the fast loop per frame, the agent beside it, everything on disk.

Per frame, at the replay clock: perception, event building, procedure tracking,
reflex rules and the speak gate, each timed. Questions due at a frame go to the
agent with a causal context. In blocking mode the replay waits for the answer
(deterministic, for evaluation); otherwise the agent runs on a worker thread and
its answers are spoken whenever they arrive, so their staleness is measured. An
agent failure marks it unavailable; the frame loop never stops for it.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

from . import LiveError
from .clock import SimulatedClock, make_clock, wall_ms
from .events import EventLog
from .gate import Candidate, GateConfig, SpeakGate
from .procedure import DEFAULT_PROCEDURE, DEFAULT_RULES, ProcedureSpec
from .reflex import ReflexEngine
from .tools import ToolContext
from .tracker import EventBuilder, ProcedureTracker, TrackerConfig


@dataclass(frozen=True)
class ScheduledQuestion:
    question_id: str
    at_ms: int
    text: str
    value_hint: str | None = None


@dataclass(frozen=True)
class SessionConfig:
    speed: float = 0.0
    blocking_questions: bool = True
    start_index: int | None = None
    end_index: int | None = None
    agent_wait_s: float = 600.0

    def to_json(self):
        return dict(self.__dict__)


def load_questions(path):
    """Questions from JSON Lines or a JSON array: ``at_s`` (or ``at_ms``), ``text``, optional ``question_id``/``value_hint``."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        items = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines()
                                                                         if line.strip()]
    except (OSError, ValueError) as exc:
        raise LiveError(f"Cannot read questions from {path}: {exc}") from exc
    questions = []
    for number, item in enumerate(items, 1):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            raise LiveError(f"Question {number} in {path} needs a 'text'.")
        at_ms = item.get("at_ms", round(float(item.get("at_s", 0)) * 1000))
        questions.append(ScheduledQuestion(str(item.get("question_id") or f"Q{number:03d}"), int(at_ms), item["text"],
                                           item.get("value_hint")))
    return questions


def percentiles(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"count": 0}

    def at(q):
        return round(values[min(len(values) - 1, int(round(q * (len(values) - 1))))], 2)
    return {"count": len(values), "p50": at(0.5), "p95": at(0.95), "max": round(values[-1], 2),
            "mean": round(sum(values) / len(values), 2)}


class _SerializedPerception:
    """One perception call at a time: the agent thread shares the model (and GPU) with the frame loop."""

    def __init__(self, inner):
        self.inner = inner
        self.producer = inner.producer
        self._lock = threading.Lock()

    def identity(self):
        return self.inner.identity()

    def observe(self, frame):
        with self._lock:
            return self.inner.observe(frame)

    def observe_crop(self, frame, crop):
        with self._lock:
            return self.inner.observe_crop(frame, crop)


class Session:
    def __init__(self, frames, perception, output_dir, *, agent=None, questions=(), tracker_config=None,
                 procedure=None, rules=None, gate_config=None, config=None, specialist=None, run_info=None):
        self.frames = frames
        self.perception = _SerializedPerception(perception)
        self.output_dir = Path(output_dir)
        self.agent = agent
        self.questions = sorted(questions, key=lambda question: question.at_ms)
        self.tracker_config = tracker_config
        self.procedure = procedure or ProcedureSpec.from_dict(DEFAULT_PROCEDURE)
        self.rules = rules if rules is not None else DEFAULT_RULES
        self.gate_config = gate_config or GateConfig()
        self.config = config or SessionConfig()
        self.specialist = specialist
        self.run_info = run_info or {}
        self.agent_unavailable = False

    def run(self):
        if self.output_dir.exists() and any(entry.name != "runtime" for entry in self.output_dir.iterdir()):
            raise LiveError(f"Refusing to reuse an existing run directory: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir
        selected = [frame for frame in self.frames.frames
                    if (self.config.start_index is None or frame.index >= self.config.start_index)
                    and (self.config.end_index is None or frame.index <= self.config.end_index)]
        if not selected:
            raise LiveError("No frames in the requested window.")
        width, height = self.frames.size(selected[0])
        tracker_config = self.tracker_config or TrackerConfig(aspect=width / height)
        if self.agent is not None:
            self.agent.evidence_dir = out / "agent"
        self._write_json(out / "run.json", {
            "started_at": datetime.now(timezone.utc).isoformat(), "case_id": self.frames.case_id,
            "frames": {"directory": str(self.frames.directory), "first": selected[0].index, "last": selected[-1].index,
                       "count": len(selected), "size": [width, height], "fps": self.frames.fps},
            "perception": self.perception.identity(), "tracker": tracker_config.to_json(),
            "procedure": self.procedure.to_json(), "rules": self.rules, "gate": self.gate_config.to_json(),
            "session": self.config.to_json(),
            "agent": self.agent.config.to_json() if hasattr(self.agent, "config") else None,
            "specialist": self.specialist is not None, "questions": len(self.questions), **self.run_info})
        log = EventLog(out / "events.jsonl")
        builder = EventBuilder(log, tracker_config)
        tracker = ProcedureTracker(log, self.procedure)
        reflex = ReflexEngine(self.rules, log)
        gate = SpeakGate(self.gate_config, out / "utterances.jsonl")
        clock = make_clock(self.config.speed)
        start_offset = selected[0].t_ms
        executor = ThreadPoolExecutor(max_workers=1) if self.agent is not None and not self.config.blocking_questions else None
        pending = list(self.questions)
        futures, answers, timings = [], [], []
        frame_period = 1000 / self.frames.fps
        started = time.perf_counter()
        interrupted = False
        last_t = selected[0].t_ms

        def session_now(frame_t):
            return frame_t if isinstance(clock, SimulatedClock) else clock.now_ms() + start_offset

        try:
            for frame in selected:
                clock.sleep_until(frame.t_ms - start_offset)
                last_t = frame.t_ms
                lag = session_now(frame.t_ms) - frame.t_ms
                w0 = wall_ms()
                observation = self.perception.observe(frame)
                w1 = wall_ms()
                events = builder.update(observation)
                events += tracker.update(frame.t_ms, frame.index, log.state_at(frame.t_ms))
                w2 = wall_ms()
                candidates = reflex.on_events(events, log.state_at(frame.t_ms), frame.t_ms)
                w3 = wall_ms()
                for candidate in candidates:
                    gate.offer(candidate, session_now(frame.t_ms))
                w4 = wall_ms()
                timing = {"frame_index": frame.index, "t_ms": frame.t_ms, "lag_ms": round(lag, 2),
                          "perception_ms": round(w1 - w0, 3), "events_ms": round(w2 - w1, 3),
                          "reflex_ms": round(w3 - w2, 3), "gate_ms": round(w4 - w3, 3), "total_ms": round(w4 - w0, 3),
                          "events": len(events), "alerts": len(candidates)}
                timing["missed_deadline"] = timing["total_ms"] > frame_period
                timings.append(timing)
                self._append(out / "timing.jsonl", timing)
                self._append(out / "observations.jsonl", observation.to_json())
                while pending and pending[0].at_ms <= frame.t_ms:
                    question = pending.pop(0)
                    ctx = ToolContext(log, self.frames, self.perception, frame.t_ms, frame.index, latest=observation,
                                      specialist=self.specialist)
                    self._append(out / "questions.jsonl", {"question_id": question.question_id, "text": question.text,
                                                           "scheduled_ms": question.at_ms, "asked_at_ms": frame.t_ms,
                                                           "frame_index": frame.index, "value_hint": question.value_hint})
                    if executor is not None and not self.agent_unavailable:
                        futures.append((question, ctx, executor.submit(self._ask, question, ctx)))
                    else:
                        answers.append(self._deliver(question, ctx, self._ask(question, ctx), gate,
                                                     session_now(frame.t_ms)))
                for item in [item for item in futures if item[2].done()]:
                    futures.remove(item)
                    answers.append(self._deliver(item[0], item[1], item[2].result(), gate, session_now(frame.t_ms)))
            deadline = time.monotonic() + self.config.agent_wait_s
            for question, ctx, future in futures:
                outcome = future.result(timeout=max(0.0, deadline - time.monotonic()))
                answers.append(self._deliver(question, ctx, outcome, gate, max(session_now(last_t), last_t)))
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=not interrupted, cancel_futures=True)
            summary = self._summary(log, gate, answers, timings, started, interrupted,
                                    [question.question_id for question in pending])
            self._write_json(out / "summary.json", summary)
        return summary

    def _ask(self, question, ctx):
        """Returns an AgentResult, or a (status, error) pair when no answer could be produced."""
        if self.agent is None or self.agent_unavailable:
            return ("agent_unavailable", None)
        try:
            return self.agent.answer(question.text, ctx, question.question_id, value_hint=question.value_hint)
        except Exception as exc:  # the frame loop must survive any agent failure
            self.agent_unavailable = True
            return ("agent_error", f"{type(exc).__name__}: {exc}")

    def _deliver(self, question, ctx, outcome, gate, now_ms):
        record = {"question_id": question.question_id, "text": question.text, "asked_at_ms": ctx.now_ms,
                  "frame_index": ctx.frame_index, "delivered_at_ms": now_ms, "staleness_ms": now_ms - ctx.now_ms}
        if isinstance(outcome, tuple):
            status, error = outcome
            record.update(status=status, error=error, spoken_text=None, supported=False, value=None)
        else:
            cites = tuple(dict.fromkeys(event_id for claim in outcome.verdict.verified
                                        for event_id in claim.get("event_ids", [])))
            decision = gate.offer(Candidate("agent", "medium", f"answer:{question.question_id}", outcome.spoken_text,
                                            cites, 1.0, ctx.now_ms, question.question_id), now_ms)
            record.update(status=outcome.status, error=outcome.error, spoken_text=outcome.spoken_text,
                          spoken=decision.spoken, supported=outcome.verdict.supported,
                          value=(outcome.final or {}).get("value"), elapsed_ms=round(outcome.elapsed_ms, 1),
                          tool_calls=outcome.tool_calls, images=outcome.images, steps=len(outcome.steps),
                          rejected_claims=len(outcome.verdict.rejected))
        self._append(self.output_dir / "answers.jsonl", record)
        return record

    def _summary(self, log, gate, answers, timings, started, interrupted, not_asked):
        events = {}
        for event in log.all():
            events[event.type] = events.get(event.type, 0) + 1
        reasons = {}
        for record in gate.records:
            reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
        statuses = {}
        for answer in answers:
            statuses[answer["status"]] = statuses.get(answer["status"], 0) + 1
        return {
            "case_id": self.frames.case_id, "frames": len(timings), "interrupted": interrupted,
            "wall_s": round(time.perf_counter() - started, 3), "events": events,
            "utterances": {"spoken": sum(1 for r in gate.records if r["spoken"]),
                           "suppressed": sum(1 for r in gate.records if not r["spoken"]), "by_reason": reasons},
            "questions": {"asked": len(answers), "answered": statuses.get("answered", 0), "by_status": statuses,
                          "not_asked": not_asked,
                          "supported": sum(1 for a in answers if a.get("supported")),
                          "answer_ms": percentiles([a.get("elapsed_ms") for a in answers]),
                          "staleness_ms": percentiles([a["staleness_ms"] for a in answers])},
            "timing_ms": {key: percentiles([t[key] for t in timings])
                          for key in ("perception_ms", "events_ms", "reflex_ms", "gate_ms", "total_ms")},
            "missed_deadlines": sum(1 for t in timings if t["missed_deadline"]),
            "agent_unavailable": self.agent_unavailable,
        }

    @staticmethod
    def _append(path, value):
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_json(path, value):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
