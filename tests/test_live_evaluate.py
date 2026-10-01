"""Ground truth, generated questions, the symbolic baseline, grading and evaluation runs."""
import json
from pathlib import Path
import re
import tempfile
import unittest

from live_fixtures import make_case, scripted_timeline
from yasargil.live.agent import AgentConfig, GuidanceAgent
from yasargil.live.evaluate import evaluate_case, summarize
from yasargil.live.events import EventLog
from yasargil.live.frames import case_frames
from yasargil.live.labels import CaseLabels
from yasargil.live.perception import DegradedPerception, LabelPerception
from yasargil.live.questions import TEMPLATES, GroundTruth, Question, answer_from_log, grade, make_questions
from yasargil.live.tracker import TrackerConfig
from yasargil.live.transport import ScriptedTransport


class QuestionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        make_case(self.root, "S1A1", scripted_timeline())
        self.frames = case_frames(self.root, "S1A1")
        self.labels = CaseLabels.load(self.root, "S1A1", (64, 36))
        self.truth = GroundTruth(self.frames, self.labels, TrackerConfig(aspect=64 / 36))

    def test_ground_truth_per_template(self):
        cases = [
            ("visible_now", {}, 6000, ["needle", "needle driver"]),
            ("first_seen", {"instrument": "needle driver"}, 6000, 4.0),
            ("first_seen", {"instrument": "grasper"}, 6000, None),
            ("entry_count", {"instrument": "needle"}, 11000, 2),
            ("near_now", {}, 6000, True),
            ("near_now", {}, 1000, False),
            ("time_in_view", {"instrument": "needle driver"}, 6000, 2.0),
            ("time_in_view", {"instrument": "nerve hook"}, 6000, 0.0),
            ("last_left", {"instrument": "nerve hook"}, 6000, 4.0),
            ("last_left", {"instrument": "needle driver"}, 6000, None),
        ]
        for template, params, at_ms, expected in cases:
            with self.subTest(template=template, params=params, at_ms=at_ms):
                self.assertEqual(self.truth.value(template, params, at_ms), expected)

    def test_question_generation_is_seeded_and_covers_templates(self):
        first = make_questions(self.truth, 12, seed=3)
        self.assertEqual(first, make_questions(self.truth, 12, seed=3))
        self.assertNotEqual(first, make_questions(self.truth, 12, seed=4))
        self.assertEqual([q.question_id for q in first][:2], ["Q001", "Q002"])
        self.assertEqual({q.template for q in first}, set(TEMPLATES))
        frame_times = {f.t_ms for f in self.frames.frames}
        for question in first:
            self.assertIn(question.at_ms, frame_times)
            self.assertEqual(question.truth, self.truth.value(question.template, question.params, question.at_ms))
            self.assertTrue(question.value_hint)
            self.assertEqual(Question.from_json(question.to_json()), question)

    def test_grading_tolerances_and_lenient_parsing(self):
        def q(template, truth):
            return Question("Q1", 0, "?", template, {"instrument": "needle"}, "", truth)
        self.assertTrue(grade(q("visible_now", ["needle", "needle driver"]), ["Needle Driver", "needle"])["correct"])
        self.assertFalse(grade(q("visible_now", ["needle"]), ["needle", "grasper"])["correct"])
        self.assertTrue(grade(q("first_seen", 4.0), 5.9)["correct"])
        self.assertFalse(grade(q("first_seen", 4.0), 6.5)["correct"])
        self.assertTrue(grade(q("first_seen", 4.0), "4.5")["correct"])
        self.assertTrue(grade(q("first_seen", None), None)["correct"])
        self.assertFalse(grade(q("first_seen", None), 3.0)["correct"])
        self.assertTrue(grade(q("entry_count", 2), 2.0)["correct"])
        self.assertTrue(grade(q("near_now", True), "yes")["correct"])
        self.assertFalse(grade(q("near_now", True), None)["correct"])

    def test_symbolic_answers_match_truth_under_label_perception(self):
        from yasargil.live.session import Session
        Session(self.frames, LabelPerception(self.labels), self.root / "run").run()
        log = EventLog.load(self.root / "run" / "events.jsonl")
        for question in make_questions(self.truth, 24, seed=1):
            with self.subTest(question=question.text, at=question.at_ms):
                self.assertEqual(answer_from_log(question.template, question.params, log, question.at_ms), question.truth)


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        make_case(self.root, "S1A1", scripted_timeline())

    def test_symbolic_evaluation_is_perfect_with_labels_and_degrades(self):
        perfect = evaluate_case(self.root, "S1A1", self.root / "eval-labels", lambda labels: LabelPerception(labels),
                                questions=10, seed=2)
        self.assertEqual(perfect["symbolic"]["accuracy"], 1.0)
        self.assertNotIn("agent", perfect)
        blind = evaluate_case(self.root, "S1A1", self.root / "eval-blind",
                              lambda labels: DegradedPerception(LabelPerception(labels), miss_rate=1.0),
                              questions=10, seed=2)
        self.assertLess(blind["symbolic"]["accuracy"], 1.0)
        grades = [json.loads(line) for line in (self.root / "eval-blind" / "grades.jsonl").read_text().splitlines()]
        self.assertEqual({g["answerer"] for g in grades}, {"symbolic"})
        self.assertIn("by_template", blind["symbolic"])

    def test_agent_answers_are_graded_alongside_the_symbolic_baseline(self):
        def wrong_count(request):
            opening = request["messages"][1]["content"]
            event_id = re.findall(r"E\d{6}", opening)[0]
            arguments = {"answer": "Seven.", "value": 7, "claims": [{"text": "Seven.", "event_ids": [event_id]}]}
            return {"model": "qwen-agent", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {
                    "name": "final_answer", "arguments": json.dumps(arguments)}}]}}], "usage": {}}
        agent = GuidanceAgent(ScriptedTransport([wrong_count] * 4), AgentConfig())
        result = evaluate_case(self.root, "S1A1", self.root / "eval-agent", lambda labels: LabelPerception(labels),
                               questions=4, seed=5, agent=agent)
        self.assertEqual(result["agent"]["graded"], 4)
        self.assertLess(result["agent"]["accuracy"], 1.0)
        self.assertEqual(result["symbolic"]["accuracy"], 1.0)
        self.assertIn("answer_ms", result["agent"])
        self.assertEqual(result["agent"]["supported_rate"], 1.0)

    def test_summary_aggregates_cases(self):
        grades = [{"answerer": "symbolic", "template": "near_now", "correct": True, "case_id": "A"},
                  {"answerer": "symbolic", "template": "near_now", "correct": False, "case_id": "B"}]
        summary = summarize(grades)
        self.assertEqual(summary["symbolic"]["accuracy"], 0.5)
        self.assertEqual(summary["symbolic"]["by_template"]["near_now"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
