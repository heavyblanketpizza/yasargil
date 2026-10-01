"""Evaluation runs: replay a case, ask generated questions, grade every answerer.

Each case directory holds the replay (``run/``), the questions with their truth,
one grade per answerer per question, and a summary. The symbolic baseline is
always graded; the agent is graded when one is supplied. A non-answer from the
agent is wrong, even when the truth is "not yet".
"""
from __future__ import annotations

import json
from pathlib import Path

from . import LiveError
from .events import EventLog
from .frames import case_frames
from .labels import CaseLabels
from .questions import GroundTruth, answer_from_log, grade, make_questions
from .session import ScheduledQuestion, Session, SessionConfig, percentiles
from .tracker import TrackerConfig


def evaluate_case(dataset_root, case_id, output_dir, perception_factory, *, questions=20, seed=0, agent=None,
                  specialist=None, session_config=None, procedure=None, rules=None, run_info=None):
    output = Path(output_dir)
    if output.exists():
        raise LiveError(f"Refusing to reuse an existing evaluation directory: {output}")
    frames = case_frames(dataset_root, case_id)
    width, height = frames.size(frames.frames[0])
    labels = CaseLabels.load(dataset_root, case_id, (width, height))
    tracker_config = TrackerConfig(aspect=width / height)
    truth = GroundTruth(frames, labels, tracker_config)
    asked = make_questions(truth, questions, seed)
    perception = perception_factory(labels)
    output.mkdir(parents=True)
    with (output / "questions.jsonl").open("w", encoding="utf-8") as stream:
        for question in asked:
            stream.write(json.dumps(question.to_json(), ensure_ascii=False) + "\n")
    scheduled = [ScheduledQuestion(q.question_id, q.at_ms, q.text, q.value_hint) for q in asked] if agent else []
    run_summary = Session(frames, perception, output / "run", agent=agent, questions=scheduled,
                          tracker_config=tracker_config, procedure=procedure, rules=rules,
                          config=session_config or SessionConfig(blocking_questions=True), specialist=specialist,
                          run_info={"evaluation": {"questions": questions, "seed": seed}, **(run_info or {})}).run()
    log = EventLog.load(output / "run" / "events.jsonl")
    grades = []
    for question in asked:
        grades.append({"answerer": "symbolic", "case_id": case_id, "question_id": question.question_id,
                       "template": question.template, "at_ms": question.at_ms,
                       **grade(question, answer_from_log(question.template, question.params, log, question.at_ms))})
    if agent is not None:
        answers = {}
        answers_path = output / "run" / "answers.jsonl"
        if answers_path.exists():
            for line in answers_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                answers[record["question_id"]] = record
        for question in asked:
            record = answers.get(question.question_id, {"status": "missing"})
            result = grade(question, record.get("value"))
            if record.get("status") != "answered":
                result["correct"] = False
            grades.append({"answerer": "agent", "case_id": case_id, "question_id": question.question_id,
                           "template": question.template, "at_ms": question.at_ms, **result,
                           "status": record.get("status"), "supported": record.get("supported", False),
                           "elapsed_ms": record.get("elapsed_ms"), "tool_calls": record.get("tool_calls"),
                           "images": record.get("images"), "spoken_text": record.get("spoken_text")})
    with (output / "grades.jsonl").open("w", encoding="utf-8") as stream:
        for record in grades:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    summary = {"case_id": case_id, "perception": perception.identity(), "questions": len(asked),
               "run": {key: run_summary[key] for key in ("frames", "events", "utterances", "timing_ms",
                                                         "missed_deadlines", "agent_unavailable")},
               **summarize(grades)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
                                         encoding="utf-8")
    return summary


def summarize(grades):
    result = {}
    for answerer in sorted({g["answerer"] for g in grades}):
        rows = [g for g in grades if g["answerer"] == answerer]
        by_template = {}
        for template in sorted({g["template"] for g in rows}):
            subset = [g for g in rows if g["template"] == template]
            by_template[template] = {"count": len(subset),
                                     "accuracy": round(sum(g["correct"] for g in subset) / len(subset), 4)}
        entry = {"graded": len(rows), "correct": sum(g["correct"] for g in rows),
                 "accuracy": round(sum(g["correct"] for g in rows) / len(rows), 4), "by_template": by_template}
        if answerer == "agent":
            entry.update(
                answered_rate=round(sum(g.get("status") == "answered" for g in rows) / len(rows), 4),
                supported_rate=round(sum(bool(g.get("supported")) for g in rows) / len(rows), 4),
                answer_ms=percentiles([g.get("elapsed_ms") for g in rows]),
                tool_calls_mean=_mean([g.get("tool_calls") for g in rows]),
                images_mean=_mean([g.get("images") for g in rows]))
        result[answerer] = entry
    return result


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None
