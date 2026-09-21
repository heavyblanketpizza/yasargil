"""End-to-end review orchestration with real media and scripted model replies."""
import base64
import copy
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import test_frame_annotation as qwen_fixtures
from yasargil.contract import ContractError, sha256_file
from yasargil.frame_annotation import AnnotationConfig, run_annotation
from yasargil.medgemma_review import (
    ReviewConfig, request_review_pause, review_status, run_review,
)
from yasargil.llama_cpp import LlamaCppError
from yasargil.smart_selection import SelectionConfig, review_loop
from yasargil.video_source import prepare_video_source


read, write = qwen_fixtures.read, qwen_fixtures.write


def review_answer(request, *, needs_more=False):
    context = json.loads(request["messages"][-1]["content"])
    target = context["target_frame_id"]
    return {
        "target_frame_id": target,
        "status": "needs_more_evidence" if needs_more else "review_complete",
        "assessment": "uncertain" if needs_more else "revised",
        "revised_annotation": {
            "visible_observation": "A flat colored image is visible.", "visibility": "clear",
            "contextual_claims": [],
            "uncertainties": ["The synthetic image cannot establish any surgical action."],
        },
        "corrections": [{"original_text": context["qwen_annotation"]["visible_observation"],
                         "revised_text": "A flat colored image is visible.",
                         "reason": "The target supports a color description only.",
                         "evidence_frame_ids": [target]}],
        "evidence_requests": [{"question": "Can a clearer original target image establish the action?",
                               "reason": "Neither neighboring image resolves the target detail.",
                               "target": "target_detail", "start_ms": None, "end_ms": None}] if needs_more else [],
    }


class FakeLlamaCpp:
    def __init__(self, *, deferred_ids=(), transform=None, transport_error=False):
        self.deferred_ids, self.transform = set(deferred_ids), transform
        self.transport_error = transport_error
        self.info_calls, self.requests, self.responses = [], [], []
        self.last_response_bytes = None

    def model_info(self, model):
        self.info_calls.append(model)
        return {"name": model, "digest": "sha256:" + "a" * 64, "quantization": "Q8_0",
                "runtime_version": "0.test", "capabilities": ["vision"], "runtime": "llama.cpp",
                "model_file": {"path": "/models/medgemma.gguf", "sha256": "a" * 64},
                "projector_file": {"path": "/models/mmproj.gguf", "sha256": "b" * 64},
                "runtime_binary": {"path": "/bin/llama-server", "sha256": "c" * 64, "libraries": []},
                "binary_version": "0.test"}

    def chat_raw(self, request):
        self.requests.append(copy.deepcopy(request))
        target = json.loads(request["messages"][-1]["content"])["target_frame_id"]
        envelope = {
            "model": request["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "diagnostic_detail": "MODEL_DIAGNOSTIC_PRESERVE_VERBATIM",
                        "content": json.dumps(review_answer(request, needs_more=target in self.deferred_ids))},
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 60, "total_tokens": 160},
            "extra_runtime_metadata": {"preserve": [1, "raw", True]},
        }
        transformed = self.transform(envelope) if self.transform else envelope
        raw = transformed if isinstance(transformed, bytes) else json.dumps(transformed, indent=3).encode() + b"\n"
        self.responses.append(raw)
        self.last_response_bytes = raw
        if self.transport_error:
            raise LlamaCppError("Scripted transport validation failure")
        return raw


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Native provenance fixtures need FFmpeg")
class MedGemmaReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.dataset = self.root / "dataset"
        self.release = self.dataset / "frames" / "S1A1"
        self.release.mkdir(parents=True)
        for index, color in enumerate(("red", "green", "blue", "yellow", "orange", "purple", "black", "white"), 1):
            Image.new("RGB", (48, 32), color).save(self.release / f"S1A1_frame_{index:08d}.jpeg")
        for name in ("sospine_tool_tips.csv", "sospine_bbox.csv"):
            with (self.dataset / name).open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("trial_frame", "x1", "y1", "x2", "y2", "label"))
                for index in range(1, 9):
                    writer.writerow((f"S1A1_frame_{index:08d}.jpeg", index, 2, 3, 4, f" SOURCE_LABEL_{index} "))
        self.selection = self.root / "selection"
        self.selection.mkdir()
        self.source = prepare_video_source(self.release, self.selection / "source", released_fps=1)
        self.ids = [frame["frame_id"] for frame in self.source["frames"]]
        self.selected_ids = [self.ids[index] for index in (0, 2, 5, 7)]
        selection_config = SelectionConfig(candidate_budget=8, max_candidates=12, max_retrieval_rounds=0,
                                           procedure_context="Documented synthetic sequence context.")
        initial = {"selected_ids": self.ids, "protected_ids": [self.ids[0]]}
        write(self.selection / "run.json", {
            "schema_version": "smart-frame-selection-run-v1", "input_path": str(self.release),
            "config": asdict(selection_config),
            "source_manifest_sha256": sha256_file(self.selection / "source/source.json")})
        write(self.selection / "initial-selection.json", initial)
        review_loop(self.source, initial, self.selection, selection_config,
                    qwen_fixtures.SelectionRuntime(self.selected_ids[1:], len(self.ids)), progress=lambda _: None)
        self.annotation = self.root / "annotation"
        self.qwen_factory = qwen_fixtures.RuntimeFactory()
        run_annotation(self.selection, self.annotation,
                       AnnotationConfig(context_size=16384, image_max_tokens=256,
                                        max_tokens=2048, max_evidence_span_ms=2000),
                       runtime_factory=self.qwen_factory, progress=lambda _: None)
        self.output = self.root / "review"
        self.config = ReviewConfig(before_frames=1, after_frames=1, max_context_frames=5,
                                   num_ctx=16384, num_predict=2048)

    def run_pass(self, client, **kwargs):
        return run_review(None if kwargs.get("resume") else self.annotation,
                          self.output, self.config, client=client,
                          progress=kwargs.pop("progress", lambda _: None), **kwargs)

    def test_one_fresh_review_per_target_contains_exact_ordered_pixels_and_source_labels(self):
        frozen_parent = {path: path.read_bytes() for path in self.annotation.rglob("*.json")}
        client = FakeLlamaCpp()
        summary = self.run_pass(client)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["reviewed_frame_count"], len(self.selected_ids))
        self.assertEqual(len(client.requests), len(self.selected_ids))
        for index, request in enumerate(client.requests):
            packet = read(self.output / f"evidence/frame-{index:04d}.json")
            target_id = self.selected_ids[index]
            self.assertEqual(packet["target_frame_id"], target_id)
            messages = request["messages"]
            self.assertEqual([m["role"] for m in messages], ["system"] + ["user"] * (len(packet["frames"]) + 1))
            self.assertEqual(request["response_format"]["json_schema"]["schema"]["properties"]["target_frame_id"]["enum"], [target_id])
            self.assertEqual(request["n_ctx"], self.config.num_ctx)
            self.assertFalse(request["stream"])
            frame_indices = []
            for message, frame in zip(messages[1:-1], packet["frames"]):
                locator = json.loads(message["content"][0]["text"].removeprefix("Evidence image: "))
                self.assertEqual(locator["frame_id"], frame["frame_id"])
                self.assertEqual(locator["evidence_roles"], frame["evidence_roles"])
                self.assertEqual(sum(block["type"] == "image_url" for block in message["content"]), 1)
                self.assertEqual(base64.b64decode(message["content"][1]["image_url"]["url"].split(",", 1)[1]), Path(frame["image_path"]).read_bytes())
                frame_indices.append(locator["frame_index"])
            self.assertEqual(frame_indices, sorted(frame_indices))
            target_index = self.ids.index(target_id)
            for role, neighbor_index in (("before", target_index - 1), ("after", target_index + 1)):
                if 0 <= neighbor_index < len(self.ids):
                    self.assertTrue(any(f["frame_id"] == self.ids[neighbor_index] and role in f["evidence_roles"]
                                        for f in packet["frames"]))
            context = json.loads(messages[-1]["content"])
            self.assertEqual(context["target_frame_id"], target_id)
            self.assertEqual(context["qwen_annotation"]["visible_observation"], "A solid color fills the visible image.")
            self.assertEqual(context["dataset_context"]["status"], "available")
            self.assertEqual(context["dataset_context"]["documented_procedure_context"],
                             "Documented synthetic sequence context.")
            original = context["dataset_context"]["original_annotations"]
            self.assertTrue(any(row["frame_id"] == target_id
                                and row["raw_value"]["label"] == f" SOURCE_LABEL_{target_index + 1} " for row in original))
            self.assertEqual(context["media_timeline"]["timestamp_basis"], "reconstructed_nominal")
            self.assertFalse(context["media_timeline"]["original_procedure_elapsed_time_verified"])
            self.assertNotIn("PRIVATE_SELECTOR", json.dumps(request))
        self.assertEqual(frozen_parent, {path: path.read_bytes() for path in frozen_parent})

    def test_raw_response_envelope_thinking_and_runtime_fields_are_preserved_exactly(self):
        client = FakeLlamaCpp()
        self.run_pass(client)
        reviews = read(self.output / "reviews.json")["reviews"]
        for row, raw in zip(reviews, client.responses):
            attempt = self.output / row["call_directory"]
            self.assertEqual((attempt / "response.json").read_bytes(), raw)
            envelope = read(attempt / "response.json")
            self.assertEqual(envelope["choices"][0]["message"]["diagnostic_detail"], "MODEL_DIAGNOSTIC_PRESERVE_VERBATIM")
            self.assertEqual(envelope["extra_runtime_metadata"], {"preserve": [1, "raw", True]})
            self.assertTrue(row["human_review_required"])
            self.assertFalse(row["training_eligible"])
            self.assertFalse(row["automated_followup"])
            receipt = read(attempt / "receipt.json")
            self.assertEqual(receipt["files"]["response.json"], sha256_file(attempt / "response.json"))

    def test_needs_more_evidence_is_deferred_while_other_key_frames_continue(self):
        client = FakeLlamaCpp(deferred_ids=self.selected_ids[:1])
        with patch("yasargil.frame_annotation.LocalVideoRuntime", side_effect=AssertionError("Qwen must not run")):
            summary = self.run_pass(client)
        self.assertEqual(summary["status"], "completed_with_deferred_evidence")
        self.assertEqual(summary["reviewed_frame_count"], 4)
        self.assertEqual(summary["deferred_frame_count"], 1)
        self.assertFalse(summary["automated_followup"])
        self.assertEqual(len(client.requests), 4)
        self.assertEqual(len(self.qwen_factory.calls), 1)
        deferred = read(self.output / "deferred-evidence.json")
        self.assertFalse(deferred["automated_followup"])
        self.assertEqual(len(deferred["requests"]), 1)
        row = deferred["requests"][0]
        self.assertEqual(row["status"], "deferred_not_dispatched")
        self.assertEqual(row["target_frame_id"], self.selected_ids[0])
        raw_response = read(self.output / row["response_path"])
        self.assertEqual(row["evidence_requests"], json.loads(raw_response["choices"][0]["message"]["content"])["evidence_requests"])
        self.assertEqual((self.output / row["response_path"]).read_bytes(), client.responses[0])

    def test_prepare_only_freezes_evidence_without_model_info_or_inference(self):
        client = FakeLlamaCpp()
        summary = self.run_pass(client, prepare_only=True)
        self.assertEqual(summary["status"], "prepared")
        self.assertEqual(client.info_calls, [])
        self.assertEqual(client.requests, [])
        self.assertFalse((self.output / "calls").exists())
        self.assertEqual(len(list((self.output / "evidence").glob("*.json"))), 4)
        self.assertEqual(review_status(self.output)["status"], "prepared")
        self.assertEqual(self.run_pass(client, resume=True)["status"], "completed")

    def test_completed_resume_validates_without_loading_or_calling_medgemma(self):
        self.run_pass(FakeLlamaCpp())
        originals = {path: path.read_bytes() for path in (self.output / "calls").rglob("*.json")}
        resumed = FakeLlamaCpp()
        summary = self.run_pass(resumed, resume=True)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(resumed.info_calls, [])
        self.assertEqual(resumed.requests, [])
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})

    def test_interruption_after_raw_save_recovers_without_repeating_that_frame(self):
        interrupted = FakeLlamaCpp()
        with patch("yasargil.medgemma_review._parse", side_effect=KeyboardInterrupt("Stopped after raw save")):
            with self.assertRaises(KeyboardInterrupt):
                self.run_pass(interrupted)
        attempt = self.output / "calls/frame-0000/attempt-0001"
        self.assertEqual((attempt / "response.json").read_bytes(), interrupted.responses[0])
        self.assertFalse((attempt / "review.json").exists())
        self.assertEqual(review_status(self.output)["status"], "interrupted")
        resumed = FakeLlamaCpp()
        self.assertEqual(self.run_pass(resumed, resume=True)["status"], "completed")
        self.assertEqual(len(resumed.requests), 3)
        self.assertEqual(json.loads(resumed.requests[0]["messages"][-1]["content"])["target_frame_id"], self.selected_ids[1])
        self.assertTrue((attempt / "receipt.json").exists())
        self.assertEqual(len(list(attempt.parent.glob("attempt-*"))), 1)
        self.assertTrue((attempt / "failure.json").is_file())

    def test_malformed_unfinished_truncated_and_tool_responses_are_saved_without_retry(self):
        def unfinished(value):
            value["choices"][0]["finish_reason"] = None
            return value
        def truncated(value):
            value["choices"][0]["finish_reason"] = "length"
            return value
        def tool_call(value):
            value["choices"][0]["message"]["tool_calls"] = [{"function": {"name": "qwen_search", "arguments": {}}}]
            return value
        def invalid_annotation(value):
            value["choices"][0]["message"]["content"] = '{"target_frame_id": "invented"}'
            return value
        for index, transform in enumerate((lambda _: b"{malformed runtime response", unfinished,
                                            truncated, tool_call, invalid_annotation)):
            with self.subTest(case=index):
                self.output = self.root / f"bad-review-{index}"
                client = FakeLlamaCpp(transform=transform)
                with self.assertRaises((ContractError, LlamaCppError)):
                    self.run_pass(client)
                attempt = self.output / "calls/frame-0000/attempt-0001"
                self.assertEqual((attempt / "response.json").read_bytes(), client.responses[0])
                self.assertTrue((attempt / "failure.json").is_file())
                self.assertFalse((attempt / "review.json").exists())
                self.assertEqual(len(client.requests), 1)
                self.assertEqual(read(self.output / "reviews.json")["reviews"], [])
                self.assertEqual(review_status(self.output)["status"], "failed")

    def test_transport_rejection_retains_its_body(self):
        client = FakeLlamaCpp(transform=lambda _: b"partial HTTP body", transport_error=True)
        with self.assertRaises(LlamaCppError):
            self.run_pass(client)
        attempt = self.output / "calls/frame-0000/attempt-0001"
        self.assertEqual((attempt / "response.json").read_bytes(), b"partial HTTP body")
        self.assertEqual(len(client.requests), 1)

    def test_explicit_resume_creates_new_attempt_and_preserves_prior_failure(self):
        failed = FakeLlamaCpp(transform=lambda _: b"{bad json")
        with self.assertRaises(LlamaCppError):
            self.run_pass(failed)
        prior = self.output / "calls/frame-0000/attempt-0001"
        originals = {path: path.read_bytes() for path in prior.iterdir()}
        resumed = FakeLlamaCpp()
        self.assertEqual(self.run_pass(resumed, resume=True)["status"], "completed")
        self.assertEqual(len(resumed.requests), 4)
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})
        accepted = self.output / "calls/frame-0000/attempt-0002"
        self.assertTrue((accepted / "receipt.json").is_file())
        self.assertEqual(read(self.output / "reviews.json")["reviews"][0]["call_directory"],
                         "calls/frame-0000/attempt-0002")

    def test_changed_plan_evidence_or_frozen_qwen_inputs_fail_before_inference(self):
        self.run_pass(FakeLlamaCpp(), prepare_only=True)
        for relative in ("run.json", "evidence/frame-0000.json", "qwen/source/source.json",
                         "qwen/annotations.json", "qwen/round-00/response.json"):
            path = self.output / relative
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            client = FakeLlamaCpp()
            with self.subTest(path=relative), self.assertRaises(ContractError):
                self.run_pass(client, resume=True)
            self.assertEqual(client.info_calls, [])
            self.assertEqual(client.requests, [])
            path.write_bytes(original)

    def test_pre_migration_review_cannot_resume_with_llama_cpp(self):
        self.run_pass(FakeLlamaCpp(), prepare_only=True)
        plan = read(self.output / "run.json")
        plan.pop("runtime")
        write(self.output / "run.json", plan)
        write(self.output / "session.json", {"run_sha256": sha256_file(self.output / "run.json")})
        client = FakeLlamaCpp()
        with self.assertRaisesRegex(ContractError, "predates the llama.cpp migration"):
            self.run_pass(client, resume=True)
        self.assertEqual(client.info_calls, [])
        self.assertEqual(client.requests, [])

    def test_resume_pins_projector_and_runtime_binary_as_well_as_model(self):
        failed = FakeLlamaCpp(transform=lambda _: b"{partial reply")
        with self.assertRaises(LlamaCppError):
            self.run_pass(failed)
        for key in ("model_file", "projector_file", "runtime_binary"):
            with self.subTest(changed=key):
                client = FakeLlamaCpp()
                metadata = client.model_info(self.config.medgemma_model)
                metadata[key]["sha256"] = "d" * 64
                with patch.object(client, "model_info", return_value=metadata):
                    with self.assertRaisesRegex(ContractError, "Pinned MedGemma model or runtime changed"):
                        self.run_pass(client, resume=True)
                self.assertEqual(client.requests, [])

    def test_invalid_completion_usage_cannot_be_recovered_as_an_accepted_response(self):
        failed = FakeLlamaCpp(transform=lambda envelope: {**envelope, "usage": {
            **envelope["usage"], "completion_tokens": self.config.num_predict + 1}})
        with self.assertRaisesRegex(ContractError, "completion usage"):
            self.run_pass(failed)
        attempt = self.output / "calls/frame-0000/attempt-0001"
        self.assertEqual((attempt / "response.json").read_bytes(), failed.responses[0])
        self.assertFalse((attempt / "receipt.json").exists())
        resumed = FakeLlamaCpp()
        self.assertEqual(self.run_pass(resumed, resume=True)["status"], "completed")
        self.assertEqual(len(resumed.requests), len(self.selected_ids))
        self.assertFalse((attempt / "receipt.json").exists())

    def test_changed_source_or_decoded_image_fails_before_inference(self):
        self.run_pass(FakeLlamaCpp(), prepare_only=True)
        for key in ("source_path", "image_path"):
            path = Path(self.source["frames"][0][key])
            original = path.read_bytes()
            Image.new("RGB", (48, 32), "pink").save(path)
            client = FakeLlamaCpp()
            with self.subTest(path=key), self.assertRaises(ContractError):
                self.run_pass(client, resume=True)
            self.assertEqual(client.info_calls, [])
            self.assertEqual(client.requests, [])
            path.write_bytes(original)

    def test_completed_resume_rejects_changes_to_accepted_response_request_and_review(self):
        self.run_pass(FakeLlamaCpp())
        attempt = self.output / "calls/frame-0000/attempt-0001"
        for name in ("response.json", "request.json", "review.json", "model.json"):
            path = attempt / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            client = FakeLlamaCpp()
            with self.subTest(name=name), self.assertRaises(ContractError):
                self.run_pass(client, resume=True)
            self.assertEqual(client.info_calls, [])
            self.assertEqual(client.requests, [])
            path.write_bytes(original)

    def test_pause_request_saves_active_response_then_resume_continues(self):
        active_states = []
        def pause_first(envelope):
            if not active_states:
                self.assertTrue(request_review_pause(self.output)["pause_requested"])
                active_states.append(review_status(self.output))
            return envelope
        client = FakeLlamaCpp(transform=pause_first)
        summary = self.run_pass(client)
        self.assertEqual(summary["status"], "paused")
        self.assertEqual(summary["reviewed_frame_count"], 1)
        self.assertEqual(len(client.requests), 1)
        self.assertTrue(active_states[0]["writer_active"])
        self.assertTrue(active_states[0]["pause_requested"])
        self.assertFalse(review_status(self.output)["writer_active"])
        self.assertTrue((self.output / "calls/frame-0000/attempt-0001/response.json").exists())
        paused_again = self.run_pass(FakeLlamaCpp(), resume=True, should_stop=lambda: True)
        self.assertEqual(paused_again["status"], "paused")
        self.assertEqual(paused_again["reviewed_frame_count"], 1)
        resumed = FakeLlamaCpp()
        self.assertEqual(self.run_pass(resumed, resume=True)["status"], "completed")
        self.assertEqual(len(resumed.requests), 3)
        self.assertFalse(review_status(self.output)["pause_requested"])

    def test_stop_callback_can_pause_without_any_model_call(self):
        client = FakeLlamaCpp()
        self.assertEqual(self.run_pass(client, should_stop=lambda: True)["status"], "paused")
        self.assertEqual(client.info_calls, [])
        self.assertEqual(client.requests, [])
        self.assertEqual(read(self.output / "reviews.json")["reviews"], [])

    def test_unfinished_or_mutated_parent_qwen_run_is_rejected_before_preparation(self):
        summary_path, annotation_path = self.annotation / "summary.json", self.annotation / "annotations.json"
        originals = {path: path.read_bytes() for path in (summary_path, annotation_path)}
        for mutation in ("unfinished", "annotation"):
            for path, raw in originals.items():
                path.write_bytes(raw)
            if mutation == "unfinished":
                summary = read(summary_path)
                summary["status"] = "running"
                write(summary_path, summary)
            else:
                draft = read(annotation_path)
                draft["annotations"][0]["visible_observation"] = "Invented later surgical finding."
                write(annotation_path, draft)
            client = FakeLlamaCpp()
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                self.run_pass(client)
            self.assertEqual(client.info_calls, [])
            self.assertEqual(client.requests, [])
            self.assertFalse(self.output.exists())

    def test_changed_frame_id_inventory_is_rejected_before_medgemma_runs(self):
        path = self.annotation / "round-00/request.json"
        request = read(path)
        overview = json.loads(request["messages"][1]["content"][0]["text"])
        overview["evidence_frame_inventory"]["rows"][-1][1] += 1000
        request["messages"][1]["content"][0]["text"] = json.dumps(overview)
        write(path, request)
        client = FakeLlamaCpp()
        with self.assertRaisesRegex(ContractError, "canonical evidence inventory"):
            self.run_pass(client)
        self.assertEqual(client.requests, [])
        self.assertFalse(self.output.exists())

    def test_legacy_timestamp_annotations_still_pass_review_intake(self):
        qwen_fixtures.make_legacy_annotation(self.annotation)
        original = read(self.annotation / "annotations.json")
        client = FakeLlamaCpp()
        self.assertEqual(self.run_pass(client)["status"], "completed")
        self.assertEqual(read(self.output / "qwen/annotations.json"), original)
        self.assertEqual(len(client.requests), len(self.selected_ids))

    def test_parent_context_conflict_remains_explicit_in_every_review_request(self):
        def conflicting(ids):
            output = qwen_fixtures.annotation_answer(ids)
            output["context_check"] = "conflict"
            return output
        self.annotation = self.root / "conflicting-annotation"
        summary = run_annotation(self.selection, self.annotation,
                                 AnnotationConfig(context_size=16384, image_max_tokens=256,
                                                  max_tokens=2048, max_evidence_span_ms=2000),
                                 runtime_factory=qwen_fixtures.RuntimeFactory(conflicting), progress=lambda _: None)
        self.assertEqual(summary["status"], "context_conflict")
        client = FakeLlamaCpp()
        self.run_pass(client)
        for request in client.requests:
            self.assertEqual(json.loads(request["messages"][-1]["content"])["qwen_context_check"], "conflict")
        self.assertTrue(all(row["evidence"]["qwen_context_check"] == "conflict"
                            for row in read(self.output / "reviews.json")["reviews"]))


if __name__ == "__main__":
    unittest.main()
