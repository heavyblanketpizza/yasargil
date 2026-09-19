"""Fresh-session gap audits using real tiny media and recoverable model receipts."""
import copy
import fcntl
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image

from yasargil.contract import ContractError, sha256_file
from yasargil.gap_experiment import (
    GapConfig, experiment_status, request_experiment_pause, run_experiment,
)
from yasargil.video_source import prepare_video_source


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value))


def review(ids, *, searches=(), ready=True):
    return {"scene_summary": "Distinct synthetic colors occur in chronological order.",
            "context_check": "consistent", "ready": ready, "searches": list(searches),
            "decisions": {key: {"decision": "keep", "reason": "A distinct visible moment."} for key in ids}}


def query(start, end):
    return {"start_ms": start, "end_ms": end,
            "question": "Retrieve the visible change in this interval.", "replace_frame_id": None}


class AcceptedResultInterruption(RuntimeError):
    pass


class RecordedRuntime:
    def __init__(self, owner, config, *, expected_video_frames, video_relative_path):
        self.owner, self.config = owner, config
        self.frame_count, self.video_name = expected_video_frames, video_relative_path
        self.stage = config.log_dir.parent.parent.name
        self.requests, self.closed = [], False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True
        return False

    def chat(self, messages, *, schema, max_tokens, round_dir):
        request = {"messages": copy.deepcopy(messages), "max_tokens": max_tokens,
                   "response_format": {"type": "json_schema", "json_schema": {"schema": schema}}}
        self.requests.append(request)
        self.owner.calls.append((self.stage, request))
        if "ranking" in schema["properties"]:
            ids = schema["properties"]["ranking"]["items"]["properties"]["frame_id"]["enum"]
            chronological = sorted(ids)
            presented = json.loads(messages[1]["content"][0]["text"])["candidate_ids"]
            # Keep the omitted 1000-ms frame in the bottom half for retrieval tests.
            order = [4, 2, 6, 0, 5, 1, 7, 3] if len(ids) == 8 else [2, 0, 4, 1, 3]
            ranked = [chronological[index] for index in order]
            if ranked == presented:
                ranked[:2] = reversed(ranked[:2])
            if self.owner.ranking_mode in ("chronological", "chronological_unsorted_scores"):
                ranked = chronological
            elif self.owner.ranking_mode == "reverse":
                ranked = chronological[::-1]
            elif self.owner.ranking_mode == "presentation":
                ranked = presented
            output = {"scene_summary": "Eight distinct synthetic visible moments.", "context_check": "consistent",
                      "ranking": [{"frame_id": key, "moment_id": index + 1,
                                   "importance_score": 50 if self.owner.ranking_mode == "flat" else 100 - index * 10,
                                   "reason": f"Distinct visible reference moment {index + 1}."}
                                  for index, key in enumerate(ranked)]}
            if self.owner.ranking_mode in ("unsorted_scores", "unsorted_scores_ties"):
                if self.owner.ranking_mode == "unsorted_scores_ties":
                    output["ranking"][2]["importance_score"] = output["ranking"][1]["importance_score"]
                # Preserve each row's model-assigned evidence while scrambling only
                # list order. Equal-score rows retain their original relative order.
                output["ranking"].insert(0, output["ranking"].pop(4))
                if [row["frame_id"] for row in output["ranking"]] == presented:
                    output["ranking"][-2:] = reversed(output["ranking"][-2:])
            elif self.owner.ranking_mode == "chronological_unsorted_scores":
                # Chronological copying remains suspect even if sorting its scores
                # would produce a different order.
                output["ranking"][0]["importance_score"] = 85
        else:
            ids = schema["properties"]["decisions"]["required"]
            script = self.owner.scripts.get(self.stage)
            output = script(ids, len(self.requests) - 1) if script else review(ids)
        response = {"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": json.dumps(output)}}], "usage": {"prompt_tokens": 20}}
        overview = json.loads(messages[1]["content"][0]["text"])
        round_dir.mkdir(parents=True, exist_ok=True)
        write(round_dir / "request.json", request)
        verification = {"accepted": True, "full_source_video_verified": True,
                        "context_truncation_observed": False, "finish_reason": "stop",
                        "video_sha256": overview["video_sha256"], "video_relative_path": self.video_name,
                        "video_fps_setting": 0, "expected_video_frames": self.frame_count,
                        "decoded_frame_ids": list(range(self.frame_count)), "decoded_frames": self.frame_count,
                        "request_sha256": sha256_file(round_dir / "request.json")}
        result = {"output": output, "response": response, "verification": verification}
        for name, value in (("response.json", response), ("verification.json", verification),
                            ("output.json", output), ("result.json", result)):
            write(round_dir / name, value)
        if self.owner.interrupt_stage == self.stage:
            raise AcceptedResultInterruption("An accepted result was saved before the process stopped")
        return result


class RuntimeFactory:
    def __init__(self, scripts=None, *, interrupt_stage=None, ranking_mode="importance"):
        self.scripts, self.interrupt_stage = scripts or {}, interrupt_stage
        self.ranking_mode = ranking_mode
        self.sessions, self.calls = [], []

    def __call__(self, config, **kwargs):
        runtime = RecordedRuntime(self, config, **kwargs)
        self.sessions.append(runtime)
        return runtime


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Real provenance fixtures need FFmpeg")
class GapExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.release = self.root / "released"
        self.release.mkdir()
        for index, color in enumerate(("red", "green", "blue", "yellow", "orange", "purple", "black", "white"), 1):
            Image.new("RGB", (32, 24), color).save(self.release / f"frame_{index:08d}.jpeg")
        self.selection = self.root / "selection"
        self.selection.mkdir()
        self.source = prepare_video_source(self.release, self.selection / "source", released_fps=1)
        self.ids = [frame["frame_id"] for frame in self.source["frames"]]
        write(self.selection / "run.json", {
            "source_manifest_sha256": sha256_file(self.selection / "source/source.json"),
            "config": {"procedure_context": "A synthetic eight-frame sequence."}})
        write(self.selection / "initial-selection.json", {"selected_ids": self.ids, "protected_ids": [self.ids[0]]})
        # The prior model's reduced selection must never become the reference pool.
        write(self.selection / "selection.json", {"selected_frame_ids": [self.ids[4], self.ids[7]]})
        self.output = self.root / "experiment"
        self.config = GapConfig(retrieval_frames=4, max_retrieval_rounds=2,
                                max_request_span_ms=1200, max_moment_span_ms=2000, tolerance_ms=0)

    def run_exp(self, factory, **kwargs):
        resume = kwargs.get("resume", False)
        return run_experiment(None if resume else self.selection, self.output, self.config,
                              runtime_factory=factory, progress=lambda _: None, **kwargs)

    def test_five_blinded_sessions_share_only_source_and_use_original_candidate_pool(self):
        factory = RuntimeFactory()
        summary = self.run_exp(factory)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual([session.stage for session in factory.sessions],
                         ["reference", "drop_50", "drop_70", "drop_90", "control_all"])
        self.assertTrue(all(session.closed for session in factory.sessions))
        reference = read(self.output / "reference/reference.json")
        ranked_ids = [row["frame_id"] for row in reference["ranking"]]
        self.assertEqual(set(ranked_ids), set(self.ids))
        self.assertNotEqual(ranked_ids, self.ids)
        self.assertNotEqual(ranked_ids, list(reversed(self.ids)))
        self.assertEqual(set(ranked_ids[4:]), {self.ids[index] for index in (5, 1, 7, 3)})
        ranking_overview = json.loads(factory.sessions[0].requests[0]["messages"][1]["content"][0]["text"])
        presented_ids = ranking_overview["candidate_ids"]
        self.assertEqual(set(presented_ids), set(self.ids))
        self.assertNotEqual(presented_ids, self.ids)
        self.assertNotEqual(ranked_ids, presented_ids)
        self.assertTrue(read(self.output / "reference/ranking-quality.json")["accepted"])
        scores = [row["importance_score"] for row in reference["ranking"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(len(set(scores)), 1)
        conditions = read(self.output / "conditions.json")
        self.assertEqual([row["actual_withheld_count"] for row in conditions], [4, 6, 7, 0])
        self.assertEqual([row["actual_supplied_count"] for row in conditions], [4, 2, 1, 8])
        self.assertLess(set(conditions[0]["withheld_frame_ids"]), set(conditions[1]["withheld_frame_ids"]))
        self.assertLess(set(conditions[1]["withheld_frame_ids"]), set(conditions[2]["withheld_frame_ids"]))
        private_ids = [read(self.output / "reference/session.json")["session_id"]]
        systems = []
        for condition, session in zip(conditions, factory.sessions[1:]):
            messages = session.requests[0]["messages"]
            self.assertEqual([message["role"] for message in messages], ["system", "user"])
            self.assertEqual(sum(block["type"] == "input_video" for block in messages[1]["content"]), 1)
            overview = json.loads(messages[1]["content"][0]["text"])
            self.assertEqual(overview["candidate_ids"], condition["supplied_frame_ids"])
            self.assertEqual(overview["complete_video_frame_count"], 8)
            text = json.dumps(messages)
            for secret in ("drop_50", "drop_70", "drop_90", "control_all", "withheld_frame_ids", "reference_sha256", "moment_id", "importance_score"):
                self.assertNotIn(secret, text)
            self.assertNotIn("Distinct visible reference moment", text)
            systems.append(messages[0])
            private_ids.append(read(self.output / "conditions" / condition["id"] / "state.json")["session_id"])
            self.assertEqual(set(condition["supplied_frame_ids"]) | set(condition["withheld_frame_ids"]), set(self.ids))
            self.assertFalse(set(condition["supplied_frame_ids"]) & set(condition["withheld_frame_ids"]))
            supplied_count = condition["actual_supplied_count"]
            self.assertEqual(set(condition["supplied_frame_ids"]), set(ranked_ids[:supplied_count]))
            self.assertEqual(set(condition["withheld_frame_ids"]), set(ranked_ids[supplied_count:]))
            self.assertEqual(condition["supplied_frame_ids"], sorted(condition["supplied_frame_ids"]))
        self.assertEqual(len(set(private_ids)), 5)
        self.assertTrue(all(system == systems[0] for system in systems))
        self.assertFalse(summary["training_eligible"])

    def test_half_up_rounding_is_frozen_from_single_ranking(self):
        pool = [self.ids[index] for index in (0, 2, 3, 5, 7)]
        write(self.selection / "initial-selection.json", {"selected_ids": pool, "protected_ids": []})
        self.run_exp(RuntimeFactory())
        conditions = read(self.output / "conditions.json")
        self.assertEqual([row["actual_withheld_count"] for row in conditions], [3, 4, 4, 0])
        self.assertEqual([row["actual_supplied_count"] for row in conditions], [2, 1, 1, 5])

    def test_suspect_rankings_stop_before_any_blinded_auditor(self):
        for mode in ("chronological", "chronological_unsorted_scores", "reverse", "presentation", "flat"):
            with self.subTest(mode=mode):
                self.output = self.root / f"experiment-{mode}"
                factory = RuntimeFactory(ranking_mode=mode)
                with self.assertRaises(ContractError):
                    self.run_exp(factory)
                self.assertEqual([session.stage for session in factory.sessions], ["reference"])
                self.assertTrue(factory.sessions[0].closed)
                quality = read(self.output / "reference/ranking-quality.json")
                self.assertFalse(quality["accepted"])
                self.assertTrue(quality["flags"])
                self.assertFalse((self.output / "reference/reference.json").exists())
                self.assertFalse((self.output / "conditions.json").exists())
                self.assertFalse((self.output / "conditions").exists())
                # Replaying the accepted raw response cannot bypass the quality gate.
                resumed = RuntimeFactory()
                with self.assertRaises(ContractError):
                    self.run_exp(resumed, resume=True)
                self.assertEqual(resumed.sessions, [])

    def test_unsorted_scores_are_normalized_without_changing_model_evidence(self):
        for mode in ("unsorted_scores", "unsorted_scores_ties"):
            with self.subTest(mode=mode):
                self.output = self.root / f"experiment-{mode}"
                factory = RuntimeFactory(ranking_mode=mode)
                summary = self.run_exp(factory)
                self.assertEqual(summary["status"], "completed")
                self.assertEqual([stage for stage, _ in factory.calls],
                                 ["reference", "drop_50", "drop_70", "drop_90", "control_all"])
                directory = self.output / "reference"
                raw = read(directory / "round-00/result.json")["output"]
                original = raw["ranking"]
                expected = sorted(original, key=lambda row: -row["importance_score"])
                original_ids = [row["frame_id"] for row in original]
                expected_ids = [row["frame_id"] for row in expected]
                self.assertNotEqual(original_ids, expected_ids)
                self.assertNotEqual(original_ids, self.ids)
                self.assertEqual(read(directory / "round-00/output.json"), raw)
                response = read(directory / "round-00/response.json")
                self.assertEqual(json.loads(response["choices"][0]["message"]["content"]), raw)
                reference = read(directory / "reference.json")
                self.assertEqual([row["frame_id"] for row in reference["ranking"]], expected_ids)
                evidence_keys = ("frame_id", "importance_score", "moment_id", "reason")
                self.assertEqual([{key: row[key] for key in evidence_keys} for row in reference["ranking"]],
                                 [{key: row[key] for key in evidence_keys} for row in expected])
                receipt_path = directory / "ranking-normalization.json"
                receipt = read(receipt_path)
                self.assertEqual(receipt["method"], "stable_descending_qwen_importance_score")
                self.assertTrue(receipt["changed"])
                self.assertEqual(receipt["original_frame_ids"], original_ids)
                self.assertEqual(receipt["normalized_frame_ids"], expected_ids)
                self.assertTrue(receipt["tie_policy"])
                if mode == "unsorted_scores_ties":
                    tied_raw = [row["frame_id"] for row in original if row["importance_score"] == 90]
                    tied_frozen = [row["frame_id"] for row in reference["ranking"] if row["importance_score"] == 90]
                    self.assertEqual(len(tied_raw), 2)
                    self.assertEqual(tied_frozen, tied_raw)
                raw_names = ("request.json", "response.json", "result.json", "output.json", "verification.json")
                saved_raw = {name: (directory / "round-00" / name).read_bytes() for name in raw_names}
                saved_receipt = receipt_path.read_bytes()
                resumed = RuntimeFactory()
                self.assertEqual(self.run_exp(resumed, resume=True)["status"], "completed")
                self.assertEqual(resumed.calls, [])
                self.assertEqual(resumed.sessions, [])
                self.assertEqual(receipt_path.read_bytes(), saved_receipt)
                self.assertEqual({name: (directory / "round-00" / name).read_bytes() for name in raw_names}, saved_raw)

    def test_retrieval_stays_in_condition_and_preserves_full_history_and_provenance(self):
        factory = RuntimeFactory({"drop_50": lambda ids, index: review(
            ids, searches=[query(900, 1100)] if index == 0 else [], ready=index != 0)})
        before = {path: sha256_file(path) for path in self.release.iterdir()}
        self.run_exp(factory)
        session = next(session for session in factory.sessions if session.stage == "drop_50")
        self.assertEqual(len(session.requests), 2)
        first, followup = [request["messages"] for request in session.requests]
        self.assertEqual(followup[:len(first)], first)
        self.assertEqual([message["role"] for message in followup], ["system", "user", "assistant", "user"])
        for messages in (first, followup):
            self.assertEqual(sum(block["type"] == "input_video" for message in messages
                                 if isinstance(message["content"], list) for block in message["content"]), 1)
        directory = self.output / "conditions/drop_50"
        retrieved = read(directory / "rounds/round-00/retrieved-manifest.json")
        self.assertEqual([frame["frame_id"] for frame in retrieved], [self.ids[1]])
        frame = retrieved[0]
        self.assertEqual(frame, self.source["frames"][1])
        self.assertEqual(frame["timestamp_ms"], 1000)
        self.assertEqual(frame["timestamp_basis"], "reconstructed_nominal")
        self.assertEqual(sha256_file(Path(frame["source_path"])), frame["source_sha256"])
        self.assertEqual(sha256_file(Path(frame["image_path"])), frame["image_sha256"])
        metrics = read(directory / "metrics.json")
        for key in ("initial_requested", "cumulative_retrieved_exact", "final_retained_exact"):
            self.assertEqual(metrics[key]["count"], 1)
            self.assertEqual(metrics[key]["denominator"], 4)
        self.assertIsNone(metrics["false_completion_against_provisional_reference"])
        self.assertTrue(metrics["declared_complete_with_unretained_reference_moments"])
        self.assertEqual(before, {path: sha256_file(path) for path in self.release.iterdir()})
        html = (self.output / "report.html").read_text()
        self.assertIn(Path(frame["source_path"]).as_uri(), html)
        self.assertIn(frame["source_sha256"], html)
        self.assertIn("reconstructed_nominal", html)

    def test_overbroad_query_receives_no_frames_or_detection_credit(self):
        factory = RuntimeFactory({"drop_50": lambda ids, index: review(
            ids, searches=[query(0, 7000)] if index == 0 else [], ready=index != 0)})
        self.run_exp(factory)
        directory = self.output / "conditions/drop_50"
        receipt = read(directory / "rounds/round-00/retrieval.json")[0]
        self.assertEqual(receipt["status"], "rejected_too_broad")
        self.assertEqual(receipt["returned_frame_ids"], [])
        self.assertEqual(read(directory / "rounds/round-00/retrieved-manifest.json"), [])
        metrics = read(directory / "metrics.json")
        self.assertEqual(metrics["cumulative_requested"]["count"], 0)
        self.assertEqual(metrics["request_cost"]["broad_request_count"], 1)
        self.assertEqual(metrics["request_cost"]["returned_frame_count"], 0)

    def test_six_targeted_requests_are_valid_and_share_the_frame_budget(self):
        searches = [query(max(0, index * 1000 - 100), index * 1000 + 100) for index in (0, 1, 3, 5, 6, 7)]
        factory = RuntimeFactory({"drop_50": lambda ids, index: review(
            ids, searches=searches if index == 0 else [], ready=index != 0)})
        self.run_exp(factory)
        directory = self.output / "conditions/drop_50"
        receipts = read(directory / "rounds/round-00/retrieval.json")
        self.assertEqual(len(receipts), 6)
        returned = [frame_id for receipt in receipts for frame_id in receipt["returned_frame_ids"]]
        self.assertEqual(set(returned), {self.ids[index] for index in (1, 3, 5, 7)})
        self.assertEqual(len(returned), self.config.retrieval_frames)
        metrics = read(directory / "metrics.json")
        self.assertEqual(metrics["final_retained_exact"]["recall"], 1)
        self.assertEqual(metrics["request_cost"]["request_count"], 6)

    def test_short_video_whole_duration_request_is_rejected_under_default_span_limit(self):
        self.config = GapConfig(retrieval_frames=4, max_retrieval_rounds=2, tolerance_ms=0)
        self.assertEqual(self.source["duration_ms"], 8000)
        self.assertEqual(self.config.max_request_span_ms, 10000)
        factory = RuntimeFactory({"drop_50": lambda ids, index: review(
            ids, searches=[query(0, 8000)] if index == 0 else [], ready=index != 0)})
        self.run_exp(factory)
        session = next(item for item in factory.sessions if item.stage == "drop_50")
        overview = json.loads(session.requests[0]["messages"][1]["content"][0]["text"])
        self.assertEqual(overview["maximum_request_span_ms"], 4000)
        directory = self.output / "conditions/drop_50"
        receipt = read(directory / "rounds/round-00/retrieval.json")[0]
        self.assertEqual(receipt["status"], "rejected_too_broad")
        self.assertEqual(receipt["maximum_span_ms"], 4000)
        self.assertEqual(receipt["returned_frame_ids"], [])
        self.assertEqual(read(directory / "rounds/round-00/retrieved-manifest.json"), [])
        metrics = read(directory / "metrics.json")
        self.assertEqual(metrics["cumulative_requested"]["count"], 0)
        self.assertEqual(metrics["request_cost"]["returned_frame_count"], 0)
        self.assertEqual(metrics["request_cost"]["broad_request_count"], 1)

    def test_pause_after_ranking_and_completed_resume_do_not_repeat_model_calls(self):
        factory = RuntimeFactory()
        summary = self.run_exp(factory, should_stop=lambda: len(factory.calls) >= 1)
        self.assertEqual(summary["status"], "paused")
        self.assertEqual([stage for stage, _ in factory.calls], ["reference"])
        saved_reference = (self.output / "reference/round-00/result.json").read_bytes()
        resumed = RuntimeFactory()
        self.assertEqual(self.run_exp(resumed, resume=True)["status"], "completed")
        self.assertEqual([stage for stage, _ in resumed.calls], ["drop_50", "drop_70", "drop_90", "control_all"])
        self.assertEqual((self.output / "reference/round-00/result.json").read_bytes(), saved_reference)
        finished = RuntimeFactory()
        self.assertEqual(self.run_exp(finished, resume=True)["status"], "completed")
        self.assertEqual(finished.sessions, [])

    def test_accepted_condition_result_is_recovered_before_state_checkpoint(self):
        interrupted = RuntimeFactory(interrupt_stage="drop_50")
        with self.assertRaises(AcceptedResultInterruption):
            self.run_exp(interrupted)
        self.assertEqual(read(self.output / "summary.json")["status"], "failed")
        round_dir = self.output / "conditions/drop_50/rounds/round-00"
        saved = {name: (round_dir / name).read_bytes() for name in ("request.json", "response.json", "result.json")}
        resumed = RuntimeFactory()
        self.assertEqual(self.run_exp(resumed, resume=True)["status"], "completed")
        self.assertEqual([stage for stage, _ in resumed.calls], ["drop_70", "drop_90", "control_all"])
        self.assertEqual(saved, {name: (round_dir / name).read_bytes() for name in saved})
        self.assertFalse((self.output / "last-error.json").exists())

    def test_corrupted_recovery_receipt_fails_before_any_model_starts(self):
        self.run_exp(RuntimeFactory())
        path = self.output / "reference/round-00/result.json"
        result = read(path)
        result["verification"]["decoded_frame_ids"][-1] = 6
        write(path, result)
        factory = RuntimeFactory()
        with self.assertRaisesRegex(ContractError, "every frame"):
            self.run_exp(factory, resume=True)
        self.assertEqual(factory.sessions, [])

    def test_resume_rejects_altered_followup_history_and_completed_selection_state(self):
        factory = RuntimeFactory({"drop_50": lambda ids, index: review(
            ids, searches=[query(900, 1100)] if index == 0 else [], ready=index != 0)})
        self.assertEqual(self.run_exp(factory, should_stop=lambda: len(factory.calls) >= 2)["status"], "paused")
        path = self.output / "conditions/drop_50/state.json"
        original = path.read_bytes()
        state = read(path)
        self.assertEqual(state["status"], "awaiting_review")
        state["messages"][2]["content"] = '{"invented":"assistant answer"}'
        write(path, state)
        resumed = RuntimeFactory()
        with self.assertRaisesRegex(ContractError, "history or candidate state changed"):
            self.run_exp(resumed, resume=True)
        self.assertEqual(resumed.sessions, [])
        path.write_bytes(original)
        self.assertEqual(self.run_exp(resumed, resume=True)["status"], "completed")
        state = read(path)
        state["selected_frame_ids"] = [self.ids[0]]
        write(path, state)
        completed = RuntimeFactory()
        with self.assertRaisesRegex(ContractError, "history or candidate state changed"):
            self.run_exp(completed, resume=True)
        self.assertEqual(completed.sessions, [])

    def test_lock_pause_status_and_frozen_evidence_guard(self):
        factory = RuntimeFactory()
        self.assertEqual(self.run_exp(factory, prepare_only=True)["status"], "prepared")
        self.assertEqual(factory.sessions, [])
        request_experiment_pause(self.output)
        self.assertTrue(experiment_status(self.output)["pause_requested"])
        with (self.output / ".experiment.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertTrue(experiment_status(self.output)["writer_active"])
            with self.assertRaisesRegex(ContractError, "already active"):
                self.run_exp(factory, resume=True)
        path = self.output / "candidate-manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ContractError, "Saved evidence changed"):
            self.run_exp(factory, resume=True)
        self.assertEqual(factory.sessions, [])


if __name__ == "__main__":
    unittest.main()
