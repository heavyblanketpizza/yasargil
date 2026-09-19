"""Context/provenance integration checks with real tiny media and scripted Qwen."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import ValidationError
from PIL import Image

from yasargil.contract import ContractError, sha256_file
from yasargil.smart_selection import (
    REVIEW_POLICY_SHA256, TIMING_POLICY_SHA256, SelectionConfig, _config_sha256,
    _recover_verified_result, review_loop, run_selection, selection_config_from_saved, validate_review,
)
from yasargil.video_source import prepare_video_source


def answer(ids, *, searches=(), ready=True, context_check="consistent", drop=()):
    return {"scene_summary": "Visible instruments change position over the supplied timeline.",
            "context_check": context_check,
            "decisions": {frame_id: {"decision": "drop" if frame_id in drop else "keep",
                                      "reason": "Scripted visible-evidence judgment for this test."} for frame_id in ids},
            "searches": list(searches), "ready": ready}


def search(start=900, end=4200, replace_frame_id=None):
    return {"start_ms": start, "end_ms": end, "question": "Which actual intermediate frames clarify instrument motion?",
            "replace_frame_id": replace_frame_id}


class SavedResultInterruption(RuntimeError):
    pass


class ScriptedRuntime:
    """Publish the same recoverable result envelope as the real transport."""
    def __init__(self, scripts, *, interrupt_after_save=False):
        self.scripts = list(scripts)
        self.requests = []
        self.interrupt_after_save = interrupt_after_save

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def chat(self, messages, *, schema, max_tokens, round_dir):
        index = len(self.requests)
        self.requests.append(copy.deepcopy(messages))
        ids = schema["properties"]["decisions"]["required"]
        output = self.scripts[index](ids)
        request = {"messages": copy.deepcopy(messages), "max_tokens": max_tokens,
                   "response_format": {"type": "json_schema", "json_schema": {"schema": schema}}}
        response = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(output)}}]}
        overview = json.loads(messages[1]["content"][0]["text"])
        video_url = next(part["input_video"]["url"] for part in messages[1]["content"] if part["type"] == "input_video")
        verification = {"accepted": True, "full_source_video_verified": True,
                        "decoded_frame_ids": list(range(6)), "video_fps_setting": 0,
                        "decoded_frames": 6, "expected_video_frames": 6, "context_truncation_observed": False,
                        "video_sha256": overview["video_sha256"], "video_relative_path": video_url.removeprefix("file://"),
                        "finish_reason": "stop"}
        result = {"output": output, "response": response, "verification": verification}
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "request.json").write_text(json.dumps(request))
        verification["request_sha256"] = sha256_file(round_dir / "request.json")
        for name, value in (("request.json", request), ("response.json", response),
                            ("verification.json", verification), ("result.json", result)):
            (round_dir / name).write_text(json.dumps(value))
        if self.interrupt_after_save:
            raise SavedResultInterruption("Simulated interruption after an accepted result was published")
        return result


class ReviewValidationTests(unittest.TestCase):
    def test_new_defaults_disallow_retrieval_and_invalid_request_timeouts(self):
        config = SelectionConfig()
        self.assertEqual(config.review_mode, "keep_drop")
        self.assertEqual(config.max_retrieval_rounds, 0)
        config.validate()
        with self.assertRaisesRegex(ContractError, "does not allow retrieval"):
            replace(config, max_retrieval_rounds=1).validate()
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ContractError, "Request timeout"):
                replace(config, request_timeout_seconds=timeout).validate()

    def test_unknown_ids_missing_decisions_and_bad_intervals_are_rejected(self):
        ids = ["first", "last"]
        cases = []
        unknown = answer(ids)
        unknown["decisions"]["invented-frame"] = {"decision": "keep", "reason": "unsupported ID"}
        cases.append(unknown)
        missing = answer(ids)
        del missing["decisions"]["last"]
        cases.append(missing)
        for start, end in ((-1, 500), (100, 100), (900, 800), (500, 2001), (0, float("nan"))):
            cases.append(answer(ids, searches=[search(start, end)], ready=False))
        cases.append(answer(ids, searches=[search(100, 500)], ready=True))
        cases.append(answer(ids, searches=[search(100, 500, "invented-frame")], ready=False))
        for value in cases:
            with self.subTest(value=value), self.assertRaises((ContractError, ValidationError)):
                validate_review(value, ids, 2000)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required for real source provenance fixtures")
class SmartSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.release = self.root / "release"
        self.release.mkdir()
        for index, color in enumerate(("red", "green", "blue", "yellow", "orange", "purple"), start=1):
            Image.new("RGB", (48, 32), color).save(self.release / f"frame_{index:08d}.jpeg")
        self.output = self.root / "run"
        self.output.mkdir()
        self.source = prepare_video_source(self.release, self.output / "source", released_fps=1)
        self.ids = [frame["frame_id"] for frame in self.source["frames"]]
        self.initial = {"selected_ids": [self.ids[0], self.ids[2], self.ids[5]],
                        "protected_ids": [self.ids[0], self.ids[5]]}
        self.config = SelectionConfig(candidate_budget=3, retrieval_frames=2, max_candidates=6,
                                      procedure_context="A synthetic six-frame test sequence.")
        self.legacy_config = replace(self.config, max_retrieval_rounds=1, review_mode="retrieval")

    def run_loop(self, runtime, *, config=None, state=None):
        return review_loop(self.source, self.initial, self.output, config or self.config,
                           runtime, state=state, progress=lambda _: None)

    def write_plan(self):
        plan = {"config": asdict(self.config), "input_path": str(self.release),
                "timing_policy_sha256": TIMING_POLICY_SHA256,
                "review_policy_sha256": REVIEW_POLICY_SHA256,
                "config_sha256": _config_sha256(asdict(self.config)),
                "source_manifest_sha256": sha256_file(self.output / "source/source.json")}
        (self.output / "run.json").write_text(json.dumps(plan))
        (self.output / "initial-selection.json").write_text(json.dumps(self.initial))

    def test_keep_drop_reviews_fixed_candidates_once_with_full_video_and_no_retrieval(self):
        runtime = ScriptedRuntime([lambda ids: answer(ids, drop=(self.ids[0], self.ids[2]))])
        with patch("yasargil.smart_selection.retrieve_requests", side_effect=AssertionError("No additions are allowed")):
            result = self.run_loop(runtime)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(runtime.requests), 1)
        self.assertEqual([frame["frame_id"] for frame in result["frames"]], self.initial["selected_ids"])
        self.assertEqual(result["selected_frame_ids"], self.initial["protected_ids"])
        self.assertTrue(all(frame["introduced_round"] == 0 for frame in result["frames"]))
        self.assertEqual(result["unresolved_searches"], [])
        request = json.loads((self.output / "rounds/round-00/request.json").read_text())
        self.assertEqual(request["response_format"]["json_schema"]["schema"]["properties"]["searches"]["maxItems"], 0)
        messages = request["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("only keep or drop", messages[0]["content"])
        self.assertNotIn("Request a bounded time", messages[0]["content"])
        overview = json.loads(messages[1]["content"][0]["text"])
        self.assertTrue(overview["candidate_set_frozen"])
        self.assertEqual(overview["complete_video_frame_count"], 6)
        self.assertEqual(overview["candidate_ids"], self.initial["selected_ids"])
        blocks = messages[1]["content"]
        self.assertEqual(sum(part["type"] == "input_video" for part in blocks), 1)
        self.assertEqual(sum(part["type"] == "image_url" for part in blocks), 3)
        self.assertFalse((self.output / "rounds/round-00/retrieval.json").exists())

    def test_keep_drop_rejects_model_addition_request_before_retrieval_or_publication(self):
        runtime = ScriptedRuntime([lambda ids: answer(ids, searches=[search()], ready=False)])
        with patch("yasargil.smart_selection.retrieve_requests", side_effect=AssertionError("No additions are allowed")):
            with self.assertRaises(ValidationError):
                self.run_loop(runtime)
        self.assertEqual(len(runtime.requests), 1)
        self.assertFalse((self.output / "selection.json").exists())
        self.assertFalse((self.output / "rounds/round-01").exists())

    def test_keep_drop_rejects_changed_candidate_checkpoint_before_inference(self):
        interrupted = ScriptedRuntime([lambda ids: answer(ids)], interrupt_after_save=True)
        with self.assertRaises(SavedResultInterruption):
            self.run_loop(interrupted)
        state = json.loads((self.output / "state.json").read_text())
        state["candidate_ids"].append(self.ids[1])
        state["introduced_rounds"][self.ids[1]] = 1
        runtime = ScriptedRuntime([])
        with self.assertRaisesRegex(ContractError, "original sampled set"):
            self.run_loop(runtime, state=state)
        self.assertEqual(runtime.requests, [])

    def test_historical_completed_retrieval_schema_still_verifies_without_rerunning(self):
        completed = self.run_loop(ScriptedRuntime([lambda ids: answer(ids)]), config=self.legacy_config)
        old_config = asdict(self.legacy_config)
        old_config.pop("review_mode")
        old_config.pop("request_timeout_seconds")
        reconstructed = selection_config_from_saved(old_config)
        self.assertEqual(reconstructed.review_mode, "retrieval")
        state = json.loads((self.output / "state.json").read_text())
        verified = _recover_verified_result(self.output / "rounds/round-00", self.source,
            state["messages"][:-1], state["candidate_ids"], reconstructed, "video.mp4")
        self.assertEqual(verified["output"], state["last_output"])
        self.assertEqual(completed["status"], "completed")

    def test_old_active_retrieval_run_cannot_resume_as_new_selection(self):
        self.write_plan()
        path = self.output / "run.json"
        plan = json.loads(path.read_text())
        plan["config"] = asdict(self.legacy_config)
        plan["config"].pop("review_mode")
        path.write_text(json.dumps(plan))
        with patch("yasargil.llama_video.LocalVideoRuntime") as runtime:
            with self.assertRaisesRegex(ContractError, "review policy changed"):
                run_selection(None, self.output, resume=True, progress=lambda _: None)
            runtime.assert_not_called()

    def test_saved_request_timeout_cannot_change_on_resume(self):
        self.write_plan()
        path = self.output / "run.json"
        plan = json.loads(path.read_text())
        plan["config"]["request_timeout_seconds"] = 21600
        path.write_text(json.dumps(plan))
        with patch("yasargil.llama_video.LocalVideoRuntime") as runtime:
            with self.assertRaisesRegex(ContractError, "configuration changed"):
                run_selection(None, self.output, resume=True, progress=lambda _: None)
            runtime.assert_not_called()

    def test_two_round_review_preserves_whole_video_history_and_retrieves_real_provenance(self):
        before = {path: path.read_bytes() for path in self.release.iterdir()}
        runtime = ScriptedRuntime([
            lambda ids: answer(ids, searches=[search(replace_frame_id=self.ids[2])], ready=False),
            lambda ids: answer(ids, drop=(self.ids[0], self.ids[2])),
        ])
        result = self.run_loop(runtime, config=self.legacy_config)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(runtime.requests), 2)
        first, second = runtime.requests
        self.assertEqual(second[:len(first)], first)
        self.assertEqual(second[2]["role"], "assistant")
        for messages in runtime.requests:
            videos = [part for message in messages if isinstance(message["content"], list)
                      for part in message["content"] if part["type"] == "input_video"]
            self.assertEqual(videos, [{"type": "input_video", "input_video": {"url": "file://video.mp4"}}])
        first_images = [part for part in first[1]["content"] if part["type"] == "image_url"]
        self.assertEqual(len(first_images), 3)
        self.assertEqual([part for part in second[1]["content"] if part["type"] == "image_url"], first_images)
        new_evidence = [json.loads(part["text"].removeprefix("Candidate evidence: ")) for part in second[3]["content"]
                        if part["type"] == "text" and part["text"].startswith("Candidate evidence: ")]
        self.assertEqual([frame["frame_id"] for frame in new_evidence], [self.ids[1], self.ids[4]])
        self.assertEqual([frame["timestamp_ms"] for frame in new_evidence], [1000, 4000])
        for frame in new_evidence:
            self.assertEqual(frame["timestamp_basis"], "reconstructed_nominal")
            self.assertIsNone(frame["source_pts"])
            self.assertEqual(sha256_file(Path(frame["source_path"])), frame["source_sha256"])
            self.assertEqual(sha256_file(Path(frame["image_path"])), frame["image_sha256"])
        decisions = {frame["frame_id"]: frame for frame in result["frames"]}
        self.assertEqual(decisions[self.ids[0]]["model_decision"], "drop")
        self.assertEqual(decisions[self.ids[0]]["effective_decision"], "keep")
        self.assertTrue(decisions[self.ids[0]]["coverage_override"])
        self.assertNotIn(self.ids[2], result["selected_frame_ids"])
        self.assertTrue(set(self.initial["protected_ids"]) <= set(result["selected_frame_ids"]))
        self.assertEqual(decisions[self.ids[1]]["introduced_round"], 1)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertFalse(result["training_eligible"])
        self.assertEqual(json.loads((self.output / "selection.json").read_text()), result)

    def test_unreviewed_retrieval_checkpoint_can_resume_without_losing_context(self):
        runtime = ScriptedRuntime([lambda ids: answer(ids, searches=[search()], ready=False)])
        def pause_after_checkpoint(message):
            if message.endswith("awaiting_review."):
                raise SavedResultInterruption("Pause between review rounds")
        with self.assertRaises(SavedResultInterruption):
            review_loop(self.source, self.initial, self.output, self.legacy_config, runtime, progress=pause_after_checkpoint)
        state = json.loads((self.output / "state.json").read_text())
        partial = json.loads((self.output / "selection.json").read_text())
        added = [frame for frame in partial["frames"] if frame["introduced_round"] == 1]
        self.assertEqual(len(added), 2)
        self.assertTrue(all(frame["model_decision"] == "unreviewed" for frame in added))
        resumed = ScriptedRuntime([lambda ids: answer(ids)])
        result = self.run_loop(resumed, state=state, config=self.legacy_config)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(resumed.requests[0][:2], runtime.requests[0])
        self.assertEqual(len(result["rounds"]), 2)

    def test_no_new_evidence_stops_with_unresolved_request_without_second_inference(self):
        runtime = ScriptedRuntime([lambda ids: answer(ids, searches=[search(10, 900)], ready=False)])
        result = self.run_loop(runtime, config=self.legacy_config)
        self.assertEqual(result["status"], "available_evidence_exhausted")
        self.assertEqual(len(runtime.requests), 1)
        self.assertEqual(result["unresolved_searches"], [search(10, 900)])
        receipt = json.loads((self.output / "rounds/round-00/retrieval.json").read_text())
        self.assertEqual(receipt[0]["returned_frame_ids"], [])

    def test_round_budget_stops_before_unbudgeted_retrieval(self):
        runtime = ScriptedRuntime([lambda ids: answer(ids, searches=[search()], ready=False)])
        result = self.run_loop(runtime, config=replace(self.legacy_config, max_retrieval_rounds=0))
        self.assertEqual(result["status"], "retrieval_budget_exhausted")
        self.assertEqual(len(runtime.requests), 1)
        self.assertEqual(len(result["frames"]), 3)
        self.assertFalse((self.output / "rounds/round-00/retrieval.json").exists())

    def test_candidate_budget_and_context_conflict_stop_without_retrieval(self):
        for config, context, expected in ((replace(self.legacy_config, max_candidates=3), "consistent", "candidate_budget_exhausted"),
                                           (self.legacy_config, "conflict", "context_conflict")):
            with self.subTest(expected=expected):
                # Separate round stores for each independent run.
                self.output = self.root / expected
                self.output.mkdir()
                runtime = ScriptedRuntime([lambda ids: answer(ids, searches=[search()], ready=False, context_check=context)])
                result = self.run_loop(runtime, config=config)
                self.assertEqual(result["status"], expected)
                self.assertEqual(len(runtime.requests), 1)
                self.assertTrue(result["unresolved_searches"])

    def test_recover_published_result_without_repeating_model_inference(self):
        interrupted = ScriptedRuntime([lambda ids: answer(ids)], interrupt_after_save=True)
        with self.assertRaises(SavedResultInterruption):
            self.run_loop(interrupted)
        state = json.loads((self.output / "state.json").read_text())
        request_bytes = (self.output / "rounds/round-00/request.json").read_bytes()
        result_bytes = (self.output / "rounds/round-00/result.json").read_bytes()
        resumed = ScriptedRuntime([])
        result = self.run_loop(resumed, state=state)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(resumed.requests, [])
        self.assertEqual((self.output / "rounds/round-00/request.json").read_bytes(), request_bytes)
        self.assertEqual((self.output / "rounds/round-00/result.json").read_bytes(), result_bytes)

    def test_completed_resume_recreates_missing_final_artifact_without_runtime(self):
        self.write_plan()
        result = self.run_loop(ScriptedRuntime([lambda ids: answer(ids)]))
        (self.output / "selection.json").unlink()
        with patch("yasargil.llama_video.LocalVideoRuntime", side_effect=AssertionError("Completed resume must not start Qwen")):
            resumed = run_selection(None, self.output, resume=True, progress=lambda _: None)
        self.assertEqual(resumed, result)
        self.assertEqual(json.loads((self.output / "selection.json").read_text()), result)

    def test_saved_recovery_requires_current_video_coverage_and_unchanged_request_response(self):
        interrupted = ScriptedRuntime([lambda ids: answer(ids)], interrupt_after_save=True)
        with self.assertRaises(SavedResultInterruption):
            self.run_loop(interrupted)
        state = json.loads((self.output / "state.json").read_text())
        round_dir = self.output / "rounds/round-00"
        original = {name: (round_dir / name).read_bytes() for name in ("result.json", "request.json", "response.json")}

        def change_result(key, value):
            result = json.loads(original["result.json"])
            result["verification"][key] = value
            (round_dir / "result.json").write_text(json.dumps(result))

        changes = [lambda: change_result("accepted", False),
                   lambda: change_result("full_source_video_verified", False),
                   lambda: change_result("decoded_frame_ids", [0, 1, 2, 3, 4, 4]),
                   lambda: change_result("video_sha256", "f" * 64),
                   lambda: change_result("expected_video_frames", 5),
                   lambda: change_result("context_truncation_observed", True),
                   lambda: (round_dir / "request.json").write_bytes(original["request.json"] + b"\n"),
                   lambda: (round_dir / "response.json").write_text('{"choices":[]}')]
        for index, modify in enumerate(changes):
            with self.subTest(index=index):
                for name, contents in original.items():
                    (round_dir / name).write_bytes(contents)
                modify()
                resumed = ScriptedRuntime([])
                with self.assertRaises(ContractError):
                    self.run_loop(resumed, state=state)
                self.assertEqual(resumed.requests, [])
        for name, contents in original.items():
            (round_dir / name).write_bytes(contents)
        self.assertEqual(self.run_loop(ScriptedRuntime([]), state=state)["status"], "completed")

    def test_interrupted_runtime_resume_uses_fresh_log_and_decode_receipt_directory(self):
        self.write_plan()
        attempts = []
        runtimes = []

        class ArtifactGuardRuntime(ScriptedRuntime):
            def __init__(inner, config, *, interrupt):
                super().__init__([lambda ids: answer(ids)], interrupt_after_save=interrupt)
                inner.log_dir = config.log_dir

            def __enter__(inner):
                # Model the real runtime's refusal to overwrite a prior
                # server.log or native decoder verification directory.
                if (inner.log_dir / "server.log").exists():
                    raise RuntimeError("Refusing to replace an existing server log")
                inner.log_dir.mkdir(parents=True, exist_ok=True)
                (inner.log_dir / "server.log").write_text("immutable server evidence")
                (inner.log_dir / "native-decode").mkdir(exist_ok=False)
                (inner.log_dir / "native-decode/verification.json").write_text('{"accepted":true}')
                return inner

        def factory(config, **kwargs):
            runtime = ArtifactGuardRuntime(config, interrupt=not attempts)
            attempts.append(config.log_dir)
            runtimes.append(runtime)
            return runtime

        with self.assertRaises(SavedResultInterruption):
            run_selection(None, self.output, resume=True, runtime_factory=factory, progress=lambda _: None)
        old_log = (attempts[0] / "server.log").read_bytes()
        old_receipt = (attempts[0] / "native-decode/verification.json").read_bytes()
        result = run_selection(None, self.output, resume=True, runtime_factory=factory, progress=lambda _: None)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(attempts), 2)
        self.assertNotEqual(attempts[0], attempts[1])
        self.assertTrue(all(path.parent == self.output / "runtime" for path in attempts))
        self.assertEqual((attempts[0] / "server.log").read_bytes(), old_log)
        self.assertEqual((attempts[0] / "native-decode/verification.json").read_bytes(), old_receipt)
        self.assertEqual(runtimes[1].requests, [])

    def test_changed_source_is_rejected_before_any_model_request(self):
        Path(self.source["frames"][1]["source_path"]).write_bytes(b"changed-source")
        runtime = ScriptedRuntime([lambda ids: answer(ids)])
        with self.assertRaisesRegex(ContractError, "Evidence changed"):
            self.run_loop(runtime)
        self.assertEqual(runtime.requests, [])

    def test_legacy_prepared_selection_requires_new_run_before_loading_model(self):
        self.write_plan()
        plan_path = self.output / "run.json"
        plan = json.loads(plan_path.read_text())
        del plan["timing_policy_sha256"]
        plan_path.write_text(json.dumps(plan))
        source_bytes = (self.output / "source/source.json").read_bytes()
        with patch("yasargil.llama_video.LocalVideoRuntime") as runtime:
            with self.assertRaisesRegex(ContractError, "timing policy changed"):
                run_selection(None, self.output, resume=True, progress=lambda _: None)
            runtime.assert_not_called()
        self.assertEqual((self.output / "source/source.json").read_bytes(), source_bytes)

    def test_saved_prompt_change_is_rejected_before_loading_model_or_replacing_receipts(self):
        self.write_plan()
        interrupted = ScriptedRuntime([lambda ids: answer(ids)], interrupt_after_save=True)
        with self.assertRaises(SavedResultInterruption):
            self.run_loop(interrupted)
        state_path = self.output / "state.json"
        state = json.loads(state_path.read_text())
        state["messages"][0]["content"] = "Use the recorded repair time to locate frames."
        state_path.write_text(json.dumps(state))
        request_path = self.output / "rounds/round-00/request.json"
        request_bytes = request_path.read_bytes()
        with patch("yasargil.llama_video.LocalVideoRuntime") as runtime:
            with self.assertRaisesRegex(ContractError, "Selection prompt changed"):
                run_selection(None, self.output, resume=True, progress=lambda _: None)
            runtime.assert_not_called()
        self.assertEqual(request_path.read_bytes(), request_bytes)

    def test_completed_selection_without_new_policy_remains_historical_and_needs_no_model(self):
        self.write_plan()
        completed = self.run_loop(ScriptedRuntime([lambda ids: answer(ids)]))
        plan_path = self.output / "run.json"
        plan = json.loads(plan_path.read_text())
        plan.pop("timing_policy_sha256")
        plan_path.write_text(json.dumps(plan))
        with patch("yasargil.llama_video.LocalVideoRuntime") as runtime:
            recovered = run_selection(None, self.output, resume=True, progress=lambda _: None)
            runtime.assert_not_called()
        self.assertEqual(recovered, completed)

    def test_prepare_only_then_resume_uses_saved_source_and_starts_local_runtime_once(self):
        output = self.root / "prepared-workflow"
        runtime = ScriptedRuntime([lambda ids: answer(ids)])
        factories = []
        def factory(config, *, expected_video_frames, video_relative_path):
            factories.append({"config": config, "frames": expected_video_frames, "video": video_relative_path})
            return runtime
        def propose(frames, budget, **kwargs):
            self.assertEqual(len(frames), 6)
            self.assertEqual(budget, 3)
            return {"selected_ids": [frames[i]["frame_id"] for i in (0, 2, 5)],
                    "protected_ids": [frames[i]["frame_id"] for i in (0, 5)]}
        with patch("yasargil.frame_selection.select_candidates", side_effect=propose) as selector:
            prepared = run_selection(self.release, output, self.config, released_fps=1,
                                     prepare_only=True, runtime_factory=factory, progress=lambda _: None)
            self.assertEqual(prepared["status"], "prepared")
            self.assertEqual(prepared["source_frames"], 6)
            self.assertEqual(prepared["native_video_fps"], 0)
            self.assertEqual(factories, [])
            completed = run_selection(None, output, resume=True, runtime_factory=factory, progress=lambda _: None)
            self.assertEqual(selector.call_count, 1)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(len(factories), 1)
        self.assertEqual(factories[0]["frames"], 6)
        self.assertEqual(factories[0]["video"], "video.mp4")
        self.assertEqual(factories[0]["config"].request_timeout, self.config.request_timeout_seconds)
        self.assertEqual(len(runtime.requests), 1)


if __name__ == "__main__":
    unittest.main()
