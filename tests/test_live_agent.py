"""The tool-calling agent loop, driven by scripted model responses."""
import json
from pathlib import Path
import tempfile
import unittest

from live_fixtures import make_case, scripted_timeline
from yasargil.live.agent import AgentConfig, GuidanceAgent
from yasargil.live.citations import NO_ANSWER
from yasargil.live.events import EventLog
from yasargil.live.frames import case_frames
from yasargil.live.labels import CaseLabels
from yasargil.live.perception import LabelPerception
from yasargil.live.tools import ToolContext
from yasargil.live.tracker import EventBuilder, TrackerConfig
from yasargil.live.transport import ScriptedTransport, TransportError


def call(name, args, call_id=None, raw=None):
    return {"id": call_id or f"call_{name}", "type": "function",
            "function": {"name": name, "arguments": raw if raw is not None else json.dumps(args)}}


def tool_reply(*calls, finish="tool_calls"):
    return {"model": "qwen-agent", "choices": [{"index": 0, "finish_reason": finish,
            "message": {"role": "assistant", "content": "", "tool_calls": list(calls)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}, "timings": {"cache_n": 40, "prompt_n": 60}}


def text_reply(content, finish="stop"):
    return {"model": "qwen-agent", "choices": [{"index": 0, "finish_reason": finish,
            "message": {"role": "assistant", "content": content}}], "usage": {"prompt_tokens": 50, "completion_tokens": 5}}


class AgentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        make_case(self.root, "S1A1", scripted_timeline())
        self.frames = case_frames(self.root, "S1A1")
        self.perception = LabelPerception(CaseLabels.load(self.root, "S1A1", (64, 36)))
        self.log = EventLog()
        builder = EventBuilder(self.log, TrackerConfig(aspect=64 / 36))
        for frame in self.frames.frames:
            builder.update(self.perception.observe(frame))
        self.driver = next(e for e in self.log.all() if e.subject == "needle driver" and e.type == "instrument_entered")

    def context(self, index=8):
        frame = self.frames.get(index)
        return ToolContext(self.log, self.frames, self.perception, frame.t_ms, index, latest=self.perception.observe(frame))

    def final(self, claims=None, value=None, answer="The needle driver is in view."):
        return {"answer": answer, "value": value, "claims": claims if claims is not None else
                [{"text": "The needle driver is in view.", "event_ids": [self.driver.event_id]}]}

    def test_native_loop_queries_then_answers_with_verified_citations(self):
        transport = ScriptedTransport([tool_reply(call("query_events", {"types": ["instrument_entered"]})),
                                       tool_reply(call("final_answer", self.final(value=["needle driver"])))])
        result = GuidanceAgent(transport, AgentConfig()).answer("Which instruments are in view?", self.context(), "Q1")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.final["value"], ["needle driver"])
        self.assertTrue(result.verdict.supported)
        self.assertEqual(result.spoken_text, "The needle driver is in view.")
        first, second = transport.requests
        self.assertEqual(first["tool_choice"], "required")
        self.assertEqual([t["function"]["name"] for t in first["tools"]][-1], "final_answer")
        self.assertEqual(first["messages"][0]["role"], "system")
        self.assertIn("Which instruments are in view?", first["messages"][1]["content"])
        self.assertIn(self.driver.event_id, first["messages"][1]["content"])
        tool_message = second["messages"][-1]
        self.assertEqual((tool_message["role"], tool_message["tool_call_id"]), ("tool", "call_query_events"))
        self.assertIn("instrument_entered needle driver", tool_message["content"])
        self.assertEqual(second["messages"][:3], first["messages"] + [second["messages"][2]])
        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(result.steps[0]["cached_tokens"], 40)

    def test_opening_state_is_citable_as_r0(self):
        final = {"answer": "No grasper is in view.", "value": [],
                 "claims": [{"text": "No grasper is in view.", "result_ids": ["R0"]}]}
        transport = ScriptedTransport([tool_reply(call("final_answer", final))])
        result = GuidanceAgent(transport, AgentConfig()).answer("Is the grasper in view?", self.context(), "Q13")
        self.assertIn("Current state (R0)", transport.requests[0]["messages"][1]["content"])
        self.assertTrue(result.verdict.supported)
        self.assertEqual(result.spoken_text, "No grasper is in view.")

    def test_schema_mode_uses_constrained_json_actions(self):
        transport = ScriptedTransport([text_reply(json.dumps({"tool": "current_state", "args": {}})),
                                       text_reply(json.dumps({"tool": "final_answer", "args": self.final()}))])
        result = GuidanceAgent(transport, AgentConfig(mode="schema")).answer("What is in view?", self.context(), "Q2")
        self.assertEqual(result.status, "answered")
        first, second = transport.requests
        self.assertNotIn("tools", first)
        self.assertEqual(first["response_format"]["type"], "json_schema")
        self.assertEqual(second["messages"][-1]["role"], "user")
        self.assertIn("Result of current_state", second["messages"][-1]["content"])

    def test_model_mistakes_are_fed_back_and_the_loop_continues(self):
        transport = ScriptedTransport([
            tool_reply(call("teleport", {})),
            tool_reply(call("query_events", None, raw="{not json")),
            tool_reply(call("query_events", {"limit": 999})),
            text_reply("I think it is the driver."),
            tool_reply(call("final_answer", self.final())),
        ])
        result = GuidanceAgent(transport, AgentConfig(max_steps=6)).answer("q", self.context(), "Q3")
        self.assertEqual(result.status, "answered")
        feedback = [m["content"] for m in transport.requests[-1]["messages"] if m["role"] in ("tool", "user")][1:]
        self.assertTrue(any("Unknown tool" in text for text in feedback))
        self.assertTrue(any("not valid JSON" in text for text in feedback))
        self.assertTrue(any("Invalid arguments" in text for text in feedback))
        self.assertTrue(any("call a tool" in text for text in feedback))
        self.assertEqual(len(result.steps), 5)

    def test_tool_runtime_errors_are_fed_back_not_raised(self):
        transport = ScriptedTransport([tool_reply(call("view_frame", {"frame_index": 2, "crop": [0.6, 0.6, 0.2, 0.2]})),
                                       tool_reply(call("final_answer", self.final()))])
        result = GuidanceAgent(transport, AgentConfig()).answer("q", self.context(), "Q12")
        self.assertEqual(result.status, "answered")
        self.assertIn("Crop box", transport.requests[1]["messages"][-1]["content"])

    def test_invalid_final_answer_is_rejected_and_retried(self):
        transport = ScriptedTransport([tool_reply(call("final_answer", {"answer": "x"})),
                                       tool_reply(call("final_answer", self.final()))])
        result = GuidanceAgent(transport, AgentConfig()).answer("q", self.context(), "Q4")
        self.assertEqual(result.status, "answered")
        self.assertIn("Invalid arguments for final_answer", transport.requests[1]["messages"][-1]["content"])

    def test_budget_exhaustion_is_an_honest_non_answer(self):
        transport = ScriptedTransport([tool_reply(call("current_state", {}, f"c{i}")) for i in range(3)])
        result = GuidanceAgent(transport, AgentConfig(max_steps=3)).answer("q", self.context(), "Q5")
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(result.spoken_text, NO_ANSWER)
        self.assertIn("final_answer now", transport.requests[-1]["messages"][-1]["content"])

    def test_old_images_are_evicted_beyond_the_keep_limit(self):
        transport = ScriptedTransport([tool_reply(call("view_frame", {"frame_index": 2}, "v1")),
                                       tool_reply(call("view_frame", {"frame_index": 3}, "v2")),
                                       tool_reply(call("final_answer", self.final()))])
        result = GuidanceAgent(transport, AgentConfig(max_images=3, keep_images=1)).answer("q", self.context(), "Q6")
        self.assertEqual(result.status, "answered")
        messages = transport.requests[-1]["messages"]
        images = [part for m in messages if isinstance(m["content"], list) for part in m["content"]
                  if part["type"] == "image_url"]
        self.assertEqual(len(images), 1)
        stubs = [part["text"] for m in messages if isinstance(m["content"], list) for part in m["content"]
                 if part["type"] == "text" and "removed" in part["text"]]
        self.assertEqual(len(stubs), 1)
        self.assertEqual(result.images, 2)
        self.assertEqual(result.evictions, 1)

    def test_transport_failure_returns_a_result_instead_of_raising(self):
        transport = ScriptedTransport([TransportError("server died")])
        result = GuidanceAgent(transport, AgentConfig()).answer("q", self.context(), "Q7")
        self.assertEqual(result.status, "transport_error")
        self.assertIn("server died", result.error)
        self.assertEqual(result.spoken_text, NO_ANSWER)

    def test_deadline_stops_the_loop(self):
        ticks = iter([0.0, 0.0, 10.0, 10.0, 10.0])
        transport = ScriptedTransport([tool_reply(call("current_state", {})), tool_reply(call("current_state", {}, "c2"))])
        agent = GuidanceAgent(transport, AgentConfig(deadline_s=5), clock=lambda: next(ticks))
        result = agent.answer("q", self.context(), "Q8")
        self.assertEqual(result.status, "deadline")
        self.assertEqual(len(transport.requests), 1)

    def test_truncated_output_is_fed_back(self):
        transport = ScriptedTransport([text_reply('{"partial', finish="length"),
                                       tool_reply(call("final_answer", self.final()))])
        result = GuidanceAgent(transport, AgentConfig()).answer("q", self.context(), "Q9")
        self.assertEqual(result.status, "answered")
        self.assertIn("truncated", transport.requests[1]["messages"][-1]["content"])

    def test_evidence_is_saved_with_exact_bytes(self):
        transport = ScriptedTransport([tool_reply(call("final_answer", self.final()))])
        evidence = self.root / "agent"
        result = GuidanceAgent(transport, AgentConfig(), evidence_dir=evidence).answer("q", self.context(), "Q10")
        step = evidence / "Q10" / "step-01"
        self.assertEqual(json.loads((step / "request.json").read_bytes()), transport.requests[0])
        self.assertEqual(json.loads((step / "response.json").read_bytes())["model"], "qwen-agent")
        self.assertEqual(json.loads((step / "meta.json").read_text())["actions"][0]["tool"], "final_answer")
        saved = json.loads((evidence / "Q10" / "result.json").read_text())
        self.assertEqual(saved["status"], "answered")
        self.assertEqual(saved["verdict"]["spoken_text"], result.spoken_text)
        with self.assertRaises(Exception):
            GuidanceAgent(ScriptedTransport([]), AgentConfig(), evidence_dir=evidence).answer("q", self.context(), "Q10")

    def test_value_hint_and_specialist_tool_are_offered(self):
        context = self.context()
        context.specialist = lambda image, question: "ok"
        transport = ScriptedTransport([tool_reply(call("final_answer", self.final()))])
        GuidanceAgent(transport, AgentConfig()).answer("q", context, "Q11", value_hint="a list of instrument names")
        request = transport.requests[0]
        self.assertIn("ask_specialist", [t["function"]["name"] for t in request["tools"]])
        self.assertIn("a list of instrument names", request["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
