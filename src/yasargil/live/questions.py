"""Auto-graded questions with causal ground truth.

Ground truth is what perfect perception would make of the case under the same
event definitions the tracker uses: the dataset labels replayed through
``EventBuilder``. The same ``answer_from_log`` function answers from any event
log, which gives the symbolic baseline. Comparing the three separates error
sources: symbolic answers from label perception must equal the truth; symbolic
answers from a degraded or learned perception measure perception/event loss;
agent answers measure what the language model loses on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from .events import EventLog
from .labels import INSTRUMENTS
from .perception import LabelPerception
from .tracker import EventBuilder

TIME_TOLERANCE_S = 2.0
TEMPLATES = {
    "visible_now": ("Which instruments are in view right now?",
                    "a list of instrument names from: grasper, needle driver, nerve hook, needle"),
    "first_seen": ("When did the {instrument} first come into view?",
                   "seconds since the recording started when it first came into view, or null if it has not appeared"),
    "entry_count": ("How many times has the {instrument} come into view so far?", "an integer count"),
    "near_now": ("Is any instrument tip close to the durotomy right now?", "true or false"),
    "time_in_view": ("How long has the {instrument} been in view without interruption?",
                     "seconds it has been continuously in view so far, or 0 if it is not in view"),
    "last_left": ("When did the {instrument} last leave the view?",
                  "seconds since the recording started when it last left the view, or null if it has not left"),
}
NEEDS_INSTRUMENT = {"first_seen", "entry_count", "time_in_view", "last_left"}


@dataclass(frozen=True)
class Question:
    question_id: str
    at_ms: int
    text: str
    template: str
    params: dict = field(default_factory=dict)
    value_hint: str = ""
    truth: object = None

    def to_json(self):
        return {"question_id": self.question_id, "at_ms": self.at_ms, "text": self.text, "template": self.template,
                "params": self.params, "value_hint": self.value_hint, "truth": self.truth}

    @classmethod
    def from_json(cls, value):
        return cls(value["question_id"], value["at_ms"], value["text"], value["template"], dict(value.get("params") or {}),
                   value.get("value_hint", ""), value.get("truth"))


def _seconds(ms):
    return round(ms / 1000, 1)


def answer_from_log(template, params, log, now_ms):
    """The template's answer from an event log, using only events known at ``now_ms``."""
    instrument = params.get("instrument")
    if template == "visible_now":
        return sorted(log.state_at(now_ms).visible_instruments)
    if template == "near_now":
        return any(target == "durotomy" for target, _ in log.state_at(now_ms).near.values())
    entered = log.query(now_ms, types=["instrument_entered"], subject=instrument, limit=0)
    if template == "first_seen":
        return _seconds(entered[0].onset_ms) if entered else None
    if template == "entry_count":
        return len(entered)
    if template == "time_in_view":
        event_id = log.state_at(now_ms).visible_instruments.get(instrument)
        return _seconds(now_ms - log.get(event_id).onset_ms) if event_id else 0.0
    if template == "last_left":
        left = log.query(now_ms, types=["instrument_left"], subject=instrument, limit=0)
        return _seconds(left[-1].onset_ms) if left else None
    raise ValueError(f"Unknown question template: {template}")


class GroundTruth:
    def __init__(self, frames, labels, tracker_config):
        self.frames = frames
        self.log = EventLog()
        builder = EventBuilder(self.log, tracker_config)
        perception = LabelPerception(labels)
        for frame in frames.frames:
            builder.update(perception.observe(frame))

    def value(self, template, params, at_ms):
        return answer_from_log(template, params, self.log, at_ms)

    def instruments_seen(self):
        return sorted({event.subject for event in self.log.all() if event.type == "instrument_entered"})


def make_questions(truth, count, seed=0):
    """Questions at frame times stratified across the case, cycling through every template."""
    rng = random.Random(seed)
    frames = truth.frames.frames
    instruments = truth.instruments_seen() or list(INSTRUMENTS)
    order = []
    while len(order) < count:
        batch = list(TEMPLATES)
        rng.shuffle(batch)
        order += batch
    questions = []
    for k in range(count):
        window = frames[k * len(frames) // count: (k + 1) * len(frames) // count] or frames
        at_ms = rng.choice(window).t_ms
        template = order[k]
        params = {"instrument": rng.choice(instruments)} if template in NEEDS_INSTRUMENT else {}
        text, hint = TEMPLATES[template]
        questions.append(Question(f"Q{k + 1:03d}", at_ms, text.format(**params), template, params, hint,
                                  truth.value(template, params, at_ms)))
    return questions


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        text = value.strip().lower().removesuffix("seconds").removesuffix("s").strip()
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("yes", "true", "no", "false"):
        return value.strip().lower() in ("yes", "true")
    return None


def grade(question, value):
    truth, template = question.truth, question.template
    correct, error = False, None
    if template == "visible_now":
        if isinstance(value, str):
            value = [part for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            predicted = {" ".join(str(item).split()).lower() for item in value}
            correct = predicted == set(truth)
    elif template in ("first_seen", "last_left", "time_in_view"):
        number = _number(value)
        if truth is None:
            correct = value is None or (isinstance(value, str) and value.strip().lower() in ("null", "none", "never"))
        elif number is not None:
            error = round(number - truth, 3)
            correct = abs(error) <= TIME_TOLERANCE_S
    elif template == "entry_count":
        number = _number(value)
        correct = number is not None and round(number) == truth
    elif template == "near_now":
        correct = _boolean(value) == truth
    return {"correct": correct, "truth": truth, "value": value, "error": error}
