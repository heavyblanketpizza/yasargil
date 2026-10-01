"""Agent tools over the causal event log, and claim verification."""
import base64
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from live_fixtures import make_case, scripted_timeline
from yasargil.live.citations import check_claims, mentioned_labels
from yasargil.live.events import EventLog
from yasargil.live.frames import case_frames
from yasargil.live.labels import CaseLabels
from yasargil.live.perception import LabelPerception
from yasargil.live.tools import ToolContext, ToolError, run_tool, tool_schemas, validate_args
from yasargil.live.tracker import EventBuilder, TrackerConfig


class ToolTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        make_case(root, "S1A1", scripted_timeline(), size=(64, 36))
        self.frames = case_frames(root, "S1A1")
        self.perception = LabelPerception(CaseLabels.load(root, "S1A1", (64, 36)))
        self.log = EventLog()
        builder = EventBuilder(self.log, TrackerConfig(aspect=64 / 36))
        self.observations = {}
        for frame in self.frames.frames:
            observation = self.perception.observe(frame)
            self.observations[frame.index] = observation
            builder.update(observation)

    def context(self, index, **options):
        frame = self.frames.get(index)
        return ToolContext(self.log, self.frames, self.perception, frame.t_ms, index,
                           latest=self.observations[index], **options)

    def test_schemas_are_openai_function_tools(self):
        names = [tool["function"]["name"] for tool in tool_schemas()]
        self.assertEqual(names, ["query_events", "current_state", "view_frame", "perceive_frame", "final_answer"])
        self.assertIn("ask_specialist", [t["function"]["name"] for t in tool_schemas(specialist=True)])
        for tool in tool_schemas(specialist=True):
            self.assertEqual(tool["type"], "function")
            self.assertEqual(tool["function"]["parameters"]["type"], "object")

    def test_query_events_is_causal(self):
        early = run_tool("query_events", {}, self.context(3)).text
        self.assertIn("nerve hook", early)
        self.assertNotIn("needle driver", early)
        later = run_tool("query_events", {"types": ["instrument_entered"], "subject": "needle driver"},
                         self.context(12)).text
        self.assertIn("instrument_entered needle driver", later)
        self.assertNotIn("nerve hook", later)

    def test_current_state_lists_visible_objects_with_event_ids(self):
        text = run_tool("current_state", {}, self.context(7)).text
        self.assertIn("t=6.0s", text)
        self.assertIn("needle driver", text)
        self.assertIn("durotomy", text)
        self.assertRegex(text, r"E\d{6}")

    def test_view_frame_returns_image_and_records_viewing(self):
        context = self.context(6)
        result = run_tool("view_frame", {"frame_index": 4, "crop": [0, 0, 0.5, 0.5]}, context)
        with Image.open(io.BytesIO(base64.b64decode(result.image_b64))) as image:
            self.assertEqual(image.size, (32, 18))
        self.assertEqual(result.viewed_frame, 4)
        self.assertIn(4, context.viewed_frames)
        self.assertEqual(context.images_used, 1)
        current = run_tool("view_frame", {}, context)
        self.assertEqual(current.viewed_frame, 6)

    def test_view_frame_refuses_the_future_and_budget(self):
        with self.assertRaisesRegex(ToolError, "future"):
            run_tool("view_frame", {"frame_index": 9}, self.context(6))
        with self.assertRaisesRegex(ToolError, "future"):
            run_tool("view_frame", {"t_s": 8.5}, self.context(6))
        context = self.context(6, max_images=1)
        run_tool("view_frame", {"frame_index": 1}, context)
        with self.assertRaisesRegex(ToolError, "budget"):
            run_tool("view_frame", {"frame_index": 2}, context)

    def test_perceive_frame_reports_detections(self):
        context = self.context(8)
        text = run_tool("perceive_frame", {"frame_index": 7}, context).text
        self.assertIn("needle driver", text)
        self.assertIn(7, context.viewed_frames)
        cropped = run_tool("perceive_frame", {"frame_index": 7, "crop": [0, 0, 0.2, 0.2]}, context).text
        self.assertIn("no detections", cropped)

    def test_specialist_tool_calls_the_backend(self):
        seen = []
        context = self.context(5, specialist=lambda image, question: seen.append((image, question)) or "a needle")
        text = run_tool("ask_specialist", {"frame_index": 5, "question": "what is held?"}, context).text
        self.assertIn("a needle", text)
        self.assertEqual(seen[0][1], "what is held?")
        with self.assertRaisesRegex(ToolError, "not available"):
            run_tool("ask_specialist", {"frame_index": 5, "question": "x"}, self.context(5))

    def test_query_and_state_results_get_citable_ids(self):
        context = self.context(6)
        first = run_tool("query_events", {"subject": "needle driver", "types": ["instrument_entered"]}, context)
        self.assertTrue(first.text.startswith("Result R1"))
        state = run_tool("current_state", {}, context)
        self.assertTrue(state.text.startswith("Result R2"))
        self.assertEqual(context.results["R1"], {"tool": "query_events", "subject": "needle driver",
                                                 "types": ["instrument_entered"], "count": 1})
        self.assertEqual(context.results["R2"]["tool"], "current_state")

    def test_schemas_use_patterns_llama_cpp_grammars_support(self):
        import json
        from yasargil.live.tools import SCHEMAS
        self.assertNotIn("\\d", json.dumps(SCHEMAS))
        validate_args("final_answer", {"answer": "x", "claims": [{"text": "y", "result_ids": ["R0"]}]})
        with self.assertRaises(ToolError):
            validate_args("final_answer", {"answer": "x", "claims": [{"text": "y", "result_ids": ["E000001"]}]})

    def test_invalid_arguments_and_unknown_tools(self):
        with self.assertRaises(ToolError):
            validate_args("query_events", {"limit": 500})
        with self.assertRaises(ToolError):
            validate_args("view_frame", {"frame_index": 1, "surprise": True})
        with self.assertRaises(ToolError):
            run_tool("teleport", {}, self.context(3))
        validate_args("final_answer", {"answer": "x", "value": ["grasper"], "claims": []})
        with self.assertRaises(ToolError):
            validate_args("final_answer", {"answer": "x", "claims": [{"text": "y", "event_ids": ["bad"]}]})


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.log = EventLog()
        self.hook = self.log.append(1000, 2, "instrument_entered", "nerve hook", evidence_frames=(2,))
        self.driver = self.log.append(4000, 5, "instrument_entered", "needle driver", evidence_frames=(5,))
        self.future = self.log.append(9000, 10, "instrument_left", "needle driver")

    def check(self, claims, viewed=()):
        return check_claims({"answer": "a", "claims": claims}, self.log, now_ms=6000, current_index=7,
                            viewed_frames=set(viewed))

    def test_supported_claims_are_spoken(self):
        verdict = self.check([{"text": "The needle driver entered at 4 s.", "event_ids": [self.driver.event_id]}])
        self.assertTrue(verdict.supported)
        self.assertEqual(verdict.spoken_text, "The needle driver entered at 4 s.")

    def test_unknown_future_and_uncited_claims_are_withheld(self):
        verdict = self.check([
            {"text": "Nerve hook came first.", "event_ids": [self.hook.event_id]},
            {"text": "The driver left.", "event_ids": [self.future.event_id]},
            {"text": "Something else.", "event_ids": ["E000777"]},
            {"text": "No citation at all."},
        ])
        self.assertFalse(verdict.supported)
        self.assertEqual([c["text"] for c in verdict.verified], ["Nerve hook came first."])
        reasons = [r for claim in verdict.rejected for r in claim["reasons"]]
        self.assertIn(f"future event {self.future.event_id}", reasons)
        self.assertIn("unknown event E000777", reasons)
        self.assertIn("no citation", reasons)
        self.assertTrue(verdict.spoken_text.startswith("Nerve hook came first."))
        self.assertIn("could not be verified", verdict.spoken_text)

    def test_frames_must_be_past_and_inspected(self):
        verdict = self.check([{"text": "Seen in frame 3.", "frame_indices": [3]},
                              {"text": "Seen in frame 9.", "frame_indices": [9]},
                              {"text": "Grasper in frame 4.", "frame_indices": [4]}], viewed={3, 9})
        self.assertEqual([c["text"] for c in verdict.verified], ["Seen in frame 3."])
        reasons = [r for claim in verdict.rejected for r in claim["reasons"]]
        self.assertIn("future frame 9", reasons)
        self.assertIn("frame 4 was not inspected", reasons)

    def test_event_evidence_frames_count_as_inspected(self):
        verdict = self.check([{"text": "Frame 5 shows it.", "event_ids": [self.driver.event_id], "frame_indices": [5]}])
        self.assertTrue(verdict.supported)

    def test_mentioned_instruments_must_match_cited_events(self):
        verdict = self.check([{"text": "The grasper entered at 1 s.", "event_ids": [self.hook.event_id]}])
        self.assertFalse(verdict.supported)
        self.assertIn("cites no event about grasper", verdict.rejected[0]["reasons"])

    def test_mention_detection_handles_needle_versus_needle_driver(self):
        self.assertEqual(mentioned_labels("The needle driver holds it."), {"needle driver"})
        self.assertEqual(mentioned_labels("The needle and the Needle Driver."), {"needle", "needle driver"})
        self.assertEqual(mentioned_labels("Needles everywhere"), set())

    def test_absence_claims_can_cite_query_and_state_results(self):
        results = {"R0": {"tool": "current_state"},
                   "R1": {"tool": "query_events", "subject": "grasper", "types": ["instrument_entered"], "count": 0},
                   "R2": {"tool": "query_events", "subject": None, "types": None, "count": 3}}
        verdict = check_claims({"answer": "a", "claims": [
            {"text": "The grasper has not entered.", "result_ids": ["R1"]},
            {"text": "The needle driver is not in view.", "result_ids": ["R0"]},
            {"text": "No needle has appeared.", "result_ids": ["R2"]},
            {"text": "The needle driver never entered.", "result_ids": ["R1"]},
            {"text": "Made up.", "result_ids": ["R9"]},
        ]}, self.log, 6000, 7, set(), results=results)
        self.assertEqual([c["text"] for c in verdict.verified],
                         ["The grasper has not entered.", "The needle driver is not in view.", "No needle has appeared."])
        reasons = [r for claim in verdict.rejected for r in claim["reasons"]]
        self.assertIn("cites no event about needle driver", reasons)
        self.assertIn("unknown result R9", reasons)

    def test_no_claims_gives_an_honest_non_answer(self):
        verdict = check_claims({"answer": "unsure", "claims": [], "unresolved": "Too dark."}, self.log, 6000, 7, set())
        self.assertFalse(verdict.supported)
        self.assertIn("can't confirm", verdict.spoken_text)
        self.assertIn("Too dark.", verdict.spoken_text)


if __name__ == "__main__":
    unittest.main()
