"""The frame loop: perception -> events -> reflex/tracker -> gate, with the agent on the side."""
import json
from pathlib import Path
import tempfile
import unittest

from live_fixtures import make_case, scripted_timeline
from yasargil.live import LiveError
from yasargil.live.agent import AgentConfig, GuidanceAgent
from yasargil.live.frames import case_frames
from yasargil.live.labels import CaseLabels
from yasargil.live.perception import LabelPerception
from yasargil.live.session import ScheduledQuestion, Session, SessionConfig
from yasargil.live.transport import ScriptedTransport


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def final_call(event_id):
    arguments = {"answer": "The needle driver is in view.", "value": ["needle driver"],
                 "claims": [{"text": "The needle driver is in view.", "event_ids": [event_id]}]}
    return {"model": "qwen-agent", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "final_answer", "arguments": json.dumps(arguments)}}]}}], "usage": {}}


def answer_with_driver(request):
    """Cite the needle driver's entry event, read from the opening context like a model would."""
    import re
    opening = request["messages"][1]["content"]
    event_id = re.search(r"needle driver \((E\d{6})\)", opening).group(1)
    return final_call(event_id)


class FailingAgent:
    evidence_dir = None

    def answer(self, *args, **kwargs):
        raise RuntimeError("model crashed")


class SessionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        make_case(self.root, "S1A1", scripted_timeline())
        self.frames = case_frames(self.root, "S1A1")
        self.perception = LabelPerception(CaseLabels.load(self.root, "S1A1", (64, 36)))

    def session(self, name="run", **options):
        return Session(self.frames, self.perception, self.root / name, **options)

    def test_replay_without_agent_produces_events_alerts_and_audit_files(self):
        summary = self.session().run()
        run = self.root / "run"
        events = read_jsonl(run / "events.jsonl")
        types = {(e["type"], e["subject"]) for e in events}
        self.assertIn(("instrument_entered", "needle driver"), types)
        self.assertIn(("tip_near_structure", "needle driver"), types)
        self.assertIn(("alert", "needle_out_of_view"), types)
        spoken = [u for u in read_jsonl(run / "utterances.jsonl") if u["spoken"]]
        self.assertTrue(any(u["priority"] == "critical" and "Needle left" in u["text"] for u in spoken))
        self.assertEqual(len(read_jsonl(run / "observations.jsonl")), 12)
        timing = read_jsonl(run / "timing.jsonl")
        self.assertEqual(len(timing), 12)
        for key in ("perception_ms", "events_ms", "reflex_ms", "gate_ms", "total_ms", "missed_deadline"):
            self.assertIn(key, timing[0])
        self.assertEqual(summary["frames"], 12)
        self.assertGreater(summary["utterances"]["spoken"], 0)
        manifest = json.loads((run / "run.json").read_text())
        self.assertEqual(manifest["perception"]["kind"], "labels")
        self.assertEqual(manifest["procedure"]["id"], "durotomy-repair-illustrative-v0")
        self.assertEqual(json.loads((run / "summary.json").read_text())["frames"], 12)

    def test_blocking_questions_are_answered_through_the_gate(self):
        agent = GuidanceAgent(ScriptedTransport([answer_with_driver]), AgentConfig())
        questions = [ScheduledQuestion("Q1", 6000, "Which instruments are in view?", "a list of instrument names")]
        summary = self.session(agent=agent, questions=questions).run()
        run = self.root / "run"
        answers = read_jsonl(run / "answers.jsonl")
        self.assertEqual(answers[0]["question_id"], "Q1")
        self.assertEqual(answers[0]["status"], "answered")
        self.assertEqual(answers[0]["frame_index"], 7)
        self.assertEqual(answers[0]["value"], ["needle driver"])
        self.assertTrue(answers[0]["supported"])
        utterance = next(u for u in read_jsonl(run / "utterances.jsonl") if u["question_id"] == "Q1")
        self.assertEqual(utterance["text"], "The needle driver is in view.")
        self.assertTrue((run / "agent" / "Q1" / "result.json").is_file())
        self.assertEqual(summary["questions"]["answered"], 1)

    def test_background_questions_are_delivered_by_the_end(self):
        agent = GuidanceAgent(ScriptedTransport([answer_with_driver, answer_with_driver]), AgentConfig())
        questions = [ScheduledQuestion("Q1", 6000, "What is in view?"), ScheduledQuestion("Q2", 8000, "And now?")]
        summary = self.session(agent=agent, questions=questions, config=SessionConfig(blocking_questions=False)).run()
        answers = read_jsonl(self.root / "run" / "answers.jsonl")
        self.assertEqual(sorted(a["question_id"] for a in answers), ["Q1", "Q2"])
        self.assertTrue(all(a["delivered_at_ms"] >= a["asked_at_ms"] for a in answers))
        self.assertEqual(summary["questions"]["answered"], 2)

    def test_one_agent_across_runs_keeps_each_runs_evidence_separate(self):
        agent = GuidanceAgent(ScriptedTransport([answer_with_driver, answer_with_driver]), AgentConfig())
        for name in ("first", "second"):
            self.session(name, agent=agent, questions=[ScheduledQuestion("Q1", 6000, "What is in view?")]).run()
            self.assertTrue((self.root / name / "agent" / "Q1" / "result.json").is_file())

    def test_agent_failure_does_not_stop_the_frame_loop(self):
        questions = [ScheduledQuestion("Q1", 2000, "a?"), ScheduledQuestion("Q2", 9000, "b?")]
        summary = self.session(agent=FailingAgent(), questions=questions).run()
        self.assertEqual(summary["frames"], 12)
        self.assertTrue(summary["agent_unavailable"])
        answers = read_jsonl(self.root / "run" / "answers.jsonl")
        self.assertEqual([a["status"] for a in answers], ["agent_error", "agent_unavailable"])
        self.assertIn("model crashed", answers[0]["error"])

    def test_questions_without_an_agent_are_recorded_unanswered(self):
        summary = self.session(questions=[ScheduledQuestion("Q1", 0, "a?")]).run()
        self.assertEqual(read_jsonl(self.root / "run" / "answers.jsonl")[0]["status"], "agent_unavailable")
        self.assertEqual(summary["questions"]["asked"], 1)

    def test_questions_after_the_last_frame_are_reported_not_dropped(self):
        summary = self.session(questions=[ScheduledQuestion("Q1", 2000, "a?"), ScheduledQuestion("Q9", 99000, "late?")]).run()
        self.assertEqual(summary["questions"]["not_asked"], ["Q9"])
        self.assertEqual(summary["questions"]["asked"], 1)

    def test_frame_window_and_existing_directory(self):
        summary = self.session(config=SessionConfig(start_index=3, end_index=6)).run()
        self.assertEqual(summary["frames"], 4)
        with self.assertRaises(LiveError):
            self.session().run()

    def test_a_prepared_runtime_folder_is_allowed_but_nothing_else(self):
        (self.root / "prepared" / "runtime").mkdir(parents=True)
        self.assertEqual(self.session("prepared").run()["frames"], 12)
        (self.root / "dirty").mkdir()
        (self.root / "dirty" / "notes.txt").write_text("x")
        with self.assertRaises(LiveError):
            self.session("dirty").run()


if __name__ == "__main__":
    unittest.main()
