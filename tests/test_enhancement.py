"""End-to-end orchestration tests with fabricated images and scripted model responses."""
import copy
import hashlib
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import test_contract
from yasargil.contract import ContractError, validate_record
from yasargil.enhancement import EnhancementConfig, enhance_sospine, plan_enhancement, select_search_indices
from yasargil.ollama import encode_request
from yasargil.checkpoint import directory_lock
import yasargil.enhancement as enhancement


class ScriptedClient:
    def __init__(self, *, search=True, fail_stage=None, uncertain=False):
        self.search = search
        self.fail_stage = fail_stage
        self.uncertain = uncertain
        self.requests = []
        self.last_response_bytes = None

    def model_info(self, model):
        digest = "sha256:" + hashlib.sha256(model.encode()).hexdigest()
        details = {"quantization_level": "Q4_K_M"}
        return {"name": model, "digest": digest, "quantization": "Q4_K_M", "runtime_version": "test-only",
                "capabilities": ["vision"], "model_name": model, "model_digest": digest,
                "tags_response": {"models": [{"name": model, "model": model, "digest": digest, "details": details}]},
                "show_response": {"details": details, "capabilities": ["vision"]},
                "version_response": {"version": "test-only"}}

    def chat_raw(self, request):
        self.requests.append(copy.deepcopy(request))
        payload = json.loads(request["messages"][1]["content"])
        stage = payload["stage"]
        frame = payload["frames"][-1]["frame_id"]
        annotation_ids = [a["annotation_id"] for a in payload["original_annotations"]
                          if frame in a["frame_ids"] and a["original_kind"] == "keypoint"][:1]
        output = {"events": [{"event_id": "event-1", "description": f"A grasper is visible in {frame}.",
                   "type": "visible_observation", "evidence_frame_ids": [frame], "annotation_ids": annotation_ids,
                   "assessment": "supported", "uncertainty": ""}],
                  "questions": [{"question_id": "question-1", "question": "Which tool is visible in the last supplied frame?",
                   "answer": f"A grasper is visible in {frame}.", "evidence_frame_ids": [frame],
                   "annotation_ids": annotation_ids, "answerability": "visible"}],
                  "searches": [], "disagreements": []}
        if self.uncertain:
            output["events"][0].update(assessment="uncertain", uncertainty="The available images do not establish closure adequacy.")
        if stage == "review" and self.search:
            output["searches"] = [{"query": "What is visible in the middle frame?", "start_frame_index": 2,
                                   "end_frame_index": 2, "reason": "The middle released frame has not been inspected."}]
        envelope = {"model": request["model"], "done": True, "done_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(output)},
                    "prompt_eval_count": 100, "eval_count": 50}
        if stage == self.fail_stage:
            envelope["message"]["content"] = "not valid JSON"
        self.last_response_bytes = encode_request(envelope)
        return self.last_response_bytes


class EnhancementTests(unittest.TestCase):
    def setUp(self):
        fixture = test_contract.ContractTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.base = fixture.root, fixture.base
        self.config = EnhancementConfig("S1A2", 1, 3, initial_frames=2, search_frames=1, max_frames=3)

    def test_full_loop_preserves_exposure_and_produces_pending_review(self):
        client = ScriptedClient()
        dest = self.base / "enhanced"
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob("*") if p.is_file()}
        result = enhance_sospine(self.root, dest, self.config, client=client)
        self.assertEqual(result["model_calls"], 5)
        self.assertEqual(result["observed_frame_indices"], [1, 2, 3])
        self.assertFalse(result["training_eligible"])
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
        payloads = [json.loads(r["messages"][1]["content"]) for r in client.requests]
        self.assertEqual([p["stage"] for p in payloads], ["propose", "independent_observe", "review", "search", "final_revise"])
        self.assertEqual(payloads[1]["parent_outputs"], [])
        self.assertEqual([f["frame_id"] for f in payloads[3]["frames"]], ["f000002"])
        for payload in payloads:
            self.assertEqual(payload["window"]["cutoff_frame_index"], 3)
            self.assertTrue(all(a["original_kind"] in {"keypoint", "bbox"} for a in payload["original_annotations"]))
        archive = json.loads((dest / "archive.json").read_text())
        validate_record(archive, dataset_root=self.root, artifact_root=dest)
        self.assertEqual(archive["reviews"], [])
        self.assertTrue(all(c["disposition"] == "pending" for c in archive["claims"]))
        self.assertEqual(len(archive["training_view"]["turn_links"]), 2)
        self.assertIn("released frame index 2", json.dumps(archive["training_view"]["messages"][0]))
        self.assertTrue((dest / "audit.html").is_file())
        self.assertTrue((dest / "review.review-template.json").is_file())
        self.assertTrue((dest / "training.preview.jsonl").is_file())
        for request, run in zip(client.requests, archive["generation_runs"]):
            self.assertEqual((dest / "calls" / run["run_id"] / "request.json").read_bytes(), encode_request(request))
        with self.assertRaises(ContractError):
            validate_record(archive, dataset_root=self.root, artifact_root=dest, training=True)

    def test_no_requested_search_stops_at_three_calls(self):
        result = enhance_sospine(self.root, self.base / "nos", self.config, client=ScriptedClient(search=False))
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(result["stop_reason"], "no_search_requested")

    def test_uncertainty_is_expressed_and_response_bytes_distinguish_revisions(self):
        records = []
        for uncertain in (False, True):
            dest = self.base / f"uncertain-{uncertain}"
            enhance_sospine(self.root, dest, self.config, client=ScriptedClient(search=False, uncertain=uncertain))
            records.append(json.loads((dest / "archive.json").read_text()))
        self.assertNotEqual(records[0]["revision"]["revision_id"], records[1]["revision"]["revision_id"])
        blocks = records[1]["training_view"]["messages"][1]["content"]
        self.assertIn("The available images do not establish closure adequacy.", [b["text"] for b in blocks])
        expressed = [c for c in records[1]["claims"] if c["output_locations"] and c["type"] == "uncertainty_statement"]
        self.assertTrue(expressed)
        self.assertTrue(all(c["disposition"] == "pending" for c in expressed))

    def test_distinct_tags_cannot_alias_same_model_weights(self):
        client = ScriptedClient()
        original_info = client.model_info
        def alias_info(name):
            info = original_info(name)
            info["digest"] = "same-weights"
            return info
        client.model_info = alias_info
        with self.assertRaisesRegex(ContractError, "same model weights"):
            enhance_sospine(self.root, self.base / "aliased", self.config, client=client)
        self.assertFalse(client.requests)

    def test_round_and_frame_budgets_preserve_unresolved_search(self):
        for config, expected in ((replace(self.config, max_rounds=0), "round_budget_exhausted"),
                                 (replace(self.config, max_frames=2), "frame_budget_exhausted")):
            result = enhance_sospine(self.root, self.base / expected, config, client=ScriptedClient())
            self.assertEqual(result["stop_reason"], expected)
            self.assertEqual(result["model_calls"], 3)
            self.assertTrue(result["unresolved_searches"])

    def test_failed_generation_retains_exact_response_and_failure(self):
        dest = self.base / "failure"
        client = ScriptedClient(fail_stage="review")
        with self.assertRaises(ContractError):
            enhance_sospine(self.root, dest, self.config, client=client)
        self.assertEqual((dest / "calls/run-003-review/response.json").read_bytes(), client.last_response_bytes)
        self.assertEqual(json.loads((dest / "failure.json").read_text())["successful_model_calls"], 2)
        self.assertFalse((dest / "archive.json").exists())

    def test_dry_plan_and_output_guards(self):
        plan = plan_enhancement(self.root, self.config)
        self.assertEqual(plan["initial_frame_indices"], [1, 3])
        self.assertFalse(plan["native_video_processor"])
        for dest in (self.root / "generated", self.base):
            with self.assertRaises(ContractError):
                enhance_sospine(self.root, dest, self.config, client=ScriptedClient())
        for config in (replace(self.config, cutoff_index=0), replace(self.config, max_frames=33),
                       replace(self.config, initial_frames=4), replace(self.config, max_rounds=4)):
            with self.assertRaises(ContractError):
                plan_enhancement(self.root, config)

    def test_search_balances_queries_and_never_reuses_seen_frames(self):
        searches = [{"start_frame_index": 1, "end_frame_index": 5}, {"start_frame_index": 7, "end_frame_index": 10}]
        chosen = select_search_indices(searches, list(range(1, 11)), {1, 10}, 2)
        self.assertEqual(len(chosen), 2)
        self.assertTrue(any(i < 6 for i in chosen))
        self.assertTrue(any(i > 6 for i in chosen))
        self.assertFalse(set(chosen) & {1, 10})

    def test_pause_and_resume_reuses_calls_and_replays_targeted_search(self):
        first = ScriptedClient()
        dest = self.base / "resumed-search"
        paused = enhance_sospine(self.root, dest, self.config, client=first,
                                 pause_requested=lambda: len(first.requests) >= 3)
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["successful_model_calls"], 3)
        self.assertEqual(paused["observed_frame_indices"], [1, 3])
        self.assertEqual(paused["selected_frame_indices"], [1, 2, 3])
        originals = {p: p.read_bytes() for p in (dest / "calls").glob("*/response.json")}
        second = ScriptedClient()
        completed = enhance_sospine(self.root, dest, self.config, client=second, resume=True)
        self.assertEqual(completed["model_calls"], 5)
        self.assertEqual([json.loads(r["messages"][1]["content"])["stage"] for r in second.requests],
                         ["search", "final_revise"])
        self.assertEqual(originals, {p: p.read_bytes() for p in originals})
        archive = json.loads((dest / "archive.json").read_text())
        validate_record(archive, dataset_root=self.root, artifact_root=dest)

    def test_completed_resume_verifies_without_inference(self):
        dest = self.base / "complete-replay"
        first = enhance_sospine(self.root, dest, self.config, client=ScriptedClient(search=False))
        original = (dest / "completion.json").read_bytes()
        client = ScriptedClient(search=False)
        replayed = enhance_sospine(self.root, dest, self.config, client=client, resume=True)
        self.assertFalse(client.requests)
        self.assertEqual(first, replayed)
        self.assertEqual(original, (dest / "completion.json").read_bytes())

    def test_failed_response_is_retained_and_only_failed_call_retried(self):
        dest = self.base / "retry-failure"
        with self.assertRaises(ContractError):
            enhance_sospine(self.root, dest, self.config, client=ScriptedClient(search=False, fail_stage="review"))
        failed_path = dest / "calls/run-003-review/response.json"
        failed_bytes = failed_path.read_bytes()
        client = ScriptedClient(search=False)
        result = enhance_sospine(self.root, dest, self.config, client=client, resume=True)
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(failed_path.read_bytes(), failed_bytes)
        self.assertTrue((dest / "calls/run-003-review/attempts/0002/success.json").is_file())
        archive = json.loads((dest / "archive.json").read_text())
        self.assertTrue(any("attempts/0002/response.json" in a["location"] for a in archive["assets"]))

    def test_complete_response_survives_interruption_before_success_checkpoint(self):
        dest = self.base / "salvage-response"
        real = enhancement.atomic_json
        def interrupt_parsed(path, value, **kwargs):
            if path.name == "parsed.json":
                raise KeyboardInterrupt()
            return real(path, value, **kwargs)
        with patch.object(enhancement, "atomic_json", side_effect=interrupt_parsed):
            with self.assertRaises(KeyboardInterrupt):
                enhance_sospine(self.root, dest, self.config, client=ScriptedClient(search=False))
        self.assertTrue((dest / "calls/run-001-propose/response.json").is_file())
        self.assertFalse((dest / "calls/run-001-propose/success.json").exists())
        client = ScriptedClient(search=False)
        enhance_sospine(self.root, dest, self.config, client=client, resume=True)
        self.assertEqual(len(client.requests), 2)
        self.assertTrue((dest / "calls/run-001-propose/success.json").is_file())

    def test_interrupted_finalization_recovers_without_inference(self):
        dest = self.base / "finalization"
        real = enhancement.atomic_bytes
        def interrupt_publish(path, value, **kwargs):
            if path == dest.resolve() / "review.html":
                raise KeyboardInterrupt()
            return real(path, value, **kwargs)
        with patch.object(enhancement, "atomic_bytes", side_effect=interrupt_publish):
            with self.assertRaises(KeyboardInterrupt):
                enhance_sospine(self.root, dest, self.config, client=ScriptedClient(search=False))
        self.assertTrue((dest / "archive.json").is_file())
        self.assertTrue((dest / ".finalize/ready.json").is_file())
        self.assertFalse((dest / "completion.json").exists())
        client = ScriptedClient(search=False)
        result = enhance_sospine(self.root, dest, self.config, client=client, resume=True)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(client.requests)

    def test_resume_rejects_changed_config_and_unseen_source_bytes(self):
        dest = self.base / "pinned-sources"
        enhance_sospine(self.root, dest, self.config, client=ScriptedClient(), pause_requested=lambda: True)
        client = ScriptedClient()
        with self.assertRaisesRegex(ContractError, "source bytes, config"):
            enhance_sospine(self.root, dest, replace(self.config, seed=43), client=client, resume=True)
        unseen = self.root / "frames/S1A2/S1A2_frame_00000002.jpeg"
        unseen.write_bytes(unseen.read_bytes() + b"changed")
        with self.assertRaisesRegex(ContractError, "source bytes, config"):
            enhance_sospine(self.root, dest, self.config, client=client, resume=True)
        self.assertFalse(client.requests)

    def test_resume_rejects_changed_model_runtime_and_protocol(self):
        dest = self.base / "pinned-protocol"
        enhance_sospine(self.root, dest, self.config, client=ScriptedClient(), pause_requested=lambda: True)
        client = ScriptedClient()
        info = client.model_info
        client.model_info = lambda name: info(name) | {"runtime_version": "changed-runtime"}
        with self.assertRaisesRegex(ContractError, "runtime version changed"):
            enhance_sospine(self.root, dest, self.config, client=client, resume=True)
        with patch.object(enhancement, "enhancement_protocol_fingerprint", return_value={"changed": True}):
            with self.assertRaisesRegex(ContractError, "prompt identity changed"):
                enhance_sospine(self.root, dest, self.config, client=ScriptedClient(), resume=True)
        self.assertFalse(client.requests)

    def test_model_identity_is_rechecked_between_inferences(self):
        client = ScriptedClient()
        original = client.model_info
        def changed(name):
            value = original(name)
            if client.requests:
                value["model_digest"] = value["digest"] = "changed-mid-window"
            return value
        client.model_info = changed
        with self.assertRaisesRegex(ContractError, "Pinned model identity changed"):
            enhance_sospine(self.root, self.base / "model-changed", self.config, client=client)
        self.assertEqual(len(client.requests), 1)

    def test_tampered_successful_response_is_not_rerun(self):
        client = ScriptedClient()
        dest = self.base / "tampered-call"
        enhance_sospine(self.root, dest, self.config, client=client,
                       pause_requested=lambda: len(client.requests) >= 1)
        response = dest / "calls/run-001-propose/response.json"
        response.write_bytes(response.read_bytes() + b" ")
        resumed = ScriptedClient()
        with self.assertRaisesRegex(ContractError, "Successful call artifact changed"):
            enhance_sospine(self.root, dest, self.config, client=resumed, resume=True)
        self.assertFalse(resumed.requests)

    def test_window_pause_file_and_writer_lock_are_respected(self):
        dest = self.base / "pause-file"
        client = ScriptedClient(search=False)
        original_chat = client.chat_raw
        def pause_after_first(request):
            raw = original_chat(request)
            (dest / "PAUSE").write_text("pause")
            return raw
        client.chat_raw = pause_after_first
        result = enhance_sospine(self.root, dest, self.config, client=client)
        self.assertEqual(result["status"], "paused")
        with directory_lock(dest):
            with self.assertRaises(ContractError):
                enhance_sospine(self.root, dest, self.config, client=ScriptedClient(), resume=True)
            self.assertTrue((dest / "PAUSE").exists())
        resumed = ScriptedClient(search=False)
        enhance_sospine(self.root, dest, self.config, client=resumed, resume=True)
        self.assertFalse((dest / "PAUSE").exists())
        self.assertEqual(len(resumed.requests), 2)

    def test_existing_user_output_is_preserved_during_finalization(self):
        dest = self.base / "user-output"
        enhance_sospine(self.root, dest, self.config, client=ScriptedClient(), pause_requested=lambda: True)
        user_path = dest / "review.html"
        user_path.write_text("User-authored review notes")
        with self.assertRaisesRegex(ContractError, "Existing output differs"):
            enhance_sospine(self.root, dest, self.config, client=ScriptedClient(search=False), resume=True)
        self.assertEqual(user_path.read_text(), "User-authored review notes")


if __name__ == "__main__":
    unittest.main()
