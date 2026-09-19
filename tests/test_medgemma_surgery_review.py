"""One joint, recoverable MedGemma call reviews all selected frames of a surgery."""
import base64
import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import test_medgemma_review as fixtures
from yasargil.contract import ContractError, sha256_file
from yasargil.dataset_inspector import InspectorStore
from yasargil.medgemma_surgery_review import SurgeryReviewConfig, run_review
from yasargil.llama_cpp import MEDGEMMA_MODEL, LlamaCppError


read, write = fixtures.read, fixtures.write


class JointLlamaCpp:
    def __init__(self, *, deferred_ids=(), transform=None, transport_error=False):
        self.deferred_ids = set(deferred_ids)
        self.transform, self.transport_error = transform, transport_error
        self.info_calls, self.requests, self.responses = [], [], []
        self.last_response_bytes = None

    def model_info(self, model):
        self.info_calls.append(model)
        return {"name": model, "digest": "sha256:" + "b" * 64, "quantization": "Q8_0",
                "runtime_version": "0.test", "capabilities": ["vision"], "runtime": "llama.cpp",
                "model_file": {"path": "/models/medgemma.gguf", "sha256": "a" * 64},
                "projector_file": {"path": "/models/mmproj.gguf", "sha256": "b" * 64},
                "runtime_binary": {"path": "/bin/llama-server", "sha256": "c" * 64, "libraries": []},
                "binary_version": "0.test"}

    def chat_raw(self, request):
        self.requests.append(copy.deepcopy(request))
        context = json.loads(request["messages"][-1]["content"])
        drafts = {row["frame_id"]: row for row in context["qwen_annotations"]}
        targets = request["response_format"]["json_schema"]["schema"]["properties"]["reviews"]["required"]
        judgments = {}
        for target in targets:
            # Reuse the existing valid per-target contract inside the joint envelope.
            individual = {"messages": [{"content": json.dumps({
                "target_frame_id": target, "qwen_annotation": drafts[target]})}]}
            judgments[target] = fixtures.review_answer(individual, needs_more=target in self.deferred_ids)
        envelope = {
            "model": request["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "diagnostic_detail": "JOINT_DIAGNOSTIC_RETAIN_EXACTLY",
                        "content": json.dumps({"reviews": judgments})},
            }],
            "usage": {"prompt_tokens": 200, "completion_tokens": 160, "total_tokens": 360},
            "extra_runtime_metadata": {"preserve": ["whole-surgery", 123, True]},
        }
        transformed = self.transform(envelope) if self.transform else envelope
        raw = transformed if isinstance(transformed, bytes) else json.dumps(transformed, indent=3).encode() + b"\n\n"
        self.responses.append(raw)
        self.last_response_bytes = raw
        if self.transport_error:
            raise LlamaCppError("Scripted joint transport failure")
        return raw


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Native provenance fixtures need FFmpeg")
class MedGemmaSurgeryReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MedGemmaReviewTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.annotation = self.fixture.root, self.fixture.annotation
        self.source, self.selected_ids = self.fixture.source, self.fixture.selected_ids
        self.output = self.root / "joint-review"
        self.config = SurgeryReviewConfig(num_ctx=65536, num_predict=16384)

    def run_pass(self, client, **kwargs):
        return run_review(None if kwargs.get("resume") else self.annotation,
                          self.output, self.config, dataset_root=self.fixture.dataset,
                          client=client, progress=lambda _: None, **kwargs)

    def attempt(self, number=1):
        return self.output / "calls/surgery" / f"attempt-{number:04d}"

    def test_one_joint_call_contains_each_selected_image_and_all_drafts_exactly_once(self):
        parent_bytes = {path: path.read_bytes() for path in self.annotation.rglob("*.json")}
        client = JointLlamaCpp()
        summary = self.run_pass(client)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["review_unit"], "surgery")
        self.assertEqual(summary["reviewed_frame_count"], 4)
        self.assertEqual(client.info_calls, [MEDGEMMA_MODEL])
        self.assertEqual(len(client.requests), 1)
        request = client.requests[0]
        self.assertEqual(request["model"], self.config.medgemma_model)
        self.assertFalse(request["stream"])
        self.assertEqual(request["n_ctx"], 65536)
        self.assertEqual(request["max_tokens"], 16384)
        self.assertEqual(request["seed"], 42)
        messages = request["messages"]
        self.assertEqual([m["role"] for m in messages], ["system"] + ["user"] * 5)
        image_messages = [message for message in messages if isinstance(message.get("content"), list)]
        self.assertEqual(len(image_messages), 4)
        by_id = {frame["frame_id"]: frame for frame in self.source["frames"]}
        for message, target in zip(image_messages, self.selected_ids):
            locator = json.loads(message["content"][0]["text"].split(": ", 1)[1])
            self.assertEqual(locator["frame_id"], target)
            self.assertEqual(locator["timestamp_ms"], by_id[target]["timestamp_ms"])
            self.assertEqual(locator["frame_index"], by_id[target]["frame_index"])
            self.assertEqual(sum(block["type"] == "image_url" for block in message["content"]), 1)
            self.assertEqual(base64.b64decode(message["content"][1]["image_url"]["url"].split(",", 1)[1]), Path(by_id[target]["image_path"]).read_bytes())
        schema = request["response_format"]["json_schema"]["schema"]["properties"]["reviews"]
        self.assertEqual(schema["required"], self.selected_ids)
        self.assertEqual(set(schema["properties"]), set(self.selected_ids))
        self.assertFalse(schema["additionalProperties"])
        context = json.loads(messages[-1]["content"])
        self.assertEqual(context["target_frame_ids"], self.selected_ids)
        self.assertEqual([draft["frame_id"] for draft in context["qwen_annotations"]], self.selected_ids)
        original = {frame["frame_id"]: frame for frame in read(self.annotation / "annotations.json")["annotations"]}
        for draft in context["qwen_annotations"]:
            original_draft = original[draft["frame_id"]]
            for key in ("visible_observation", "visibility", "uncertainties"):
                self.assertEqual(draft[key], original_draft[key])
            self.assertEqual(len(draft["contextual_claims"]), len(original_draft["contextual_claims"]))
            for claim, full_claim in zip(draft["contextual_claims"], original_draft["contextual_claims"]):
                self.assertEqual(claim["claim"], full_claim["claim"])
                self.assertEqual(len(claim["evidence_intervals"]), len(full_claim["evidence_intervals"]))
                for interval, full_interval in zip(claim["evidence_intervals"], full_claim["evidence_intervals"]):
                    self.assertEqual((interval["start_ms"], interval["end_ms"]),
                                     (full_interval["start_ms"], full_interval["end_ms"]))
                    supplied = [row["frame_id"] for row in full_interval["supporting_frames"]
                                if row["frame_id"] in self.selected_ids]
                    self.assertEqual(interval["supplied_supporting_frame_ids"], supplied)
                    self.assertEqual(interval["omitted_supporting_frame_count"],
                                     len(full_interval["supporting_frames"]) - len(supplied))
            self.assertNotIn("source_path", draft)
            self.assertNotIn("supporting_frames", json.dumps(draft))
        self.assertEqual(context["media_timeline"]["timestamp_basis"], "reconstructed_nominal")
        self.assertEqual(context["dataset_context"]["status"], "available")
        self.assertEqual(context["dataset_context"]["documented_procedure_context"], "Documented synthetic sequence context.")
        serialized = json.dumps(context["dataset_context"])
        for frame_id in self.selected_ids:
            self.assertIn(f" SOURCE_LABEL_{by_id[frame_id]['frame_index'] + 1} ", serialized)
        self.assertNotIn("PRIVATE_SELECTOR", json.dumps(request))
        self.assertEqual(parent_bytes, {path: path.read_bytes() for path in parent_bytes})

    def test_raw_joint_envelope_is_saved_once_and_every_review_points_to_that_call(self):
        client = JointLlamaCpp()
        self.run_pass(client)
        attempt = self.attempt()
        self.assertEqual((attempt / "response.json").read_bytes(), client.responses[0])
        self.assertEqual(len(list((self.output / "calls").rglob("response.json"))), 1)
        response = read(attempt / "response.json")
        self.assertEqual(response["choices"][0]["message"]["diagnostic_detail"], "JOINT_DIAGNOSTIC_RETAIN_EXACTLY")
        self.assertEqual(response["extra_runtime_metadata"], {"preserve": ["whole-surgery", 123, True]})
        combined = read(attempt / "review.json")
        self.assertEqual(combined["schema_version"], "medgemma-surgery-review-v1")
        self.assertEqual([row["target_frame_id"] for row in combined["reviews"]], self.selected_ids)
        rows = read(self.output / "reviews.json")["reviews"]
        self.assertEqual([row["target_frame_id"] for row in rows], self.selected_ids)
        self.assertEqual({row["call_directory"] for row in rows}, {"calls/surgery/attempt-0001"})
        self.assertTrue(all(row["human_review_required"] and not row["training_eligible"] for row in rows))
        self.assertEqual(read(attempt / "receipt.json")["files"]["response.json"], sha256_file(attempt / "response.json"))

    def test_prepare_only_and_pause_make_no_model_calls(self):
        client = JointLlamaCpp()
        self.assertEqual(self.run_pass(client, prepare_only=True)["status"], "prepared")
        self.assertEqual(client.info_calls, [])
        self.assertEqual(client.requests, [])
        self.assertFalse((self.output / "calls").exists())
        plan = read(self.output / "run.json")
        batch = read(self.output / plan["surgery_evidence_file"])
        self.assertEqual(batch["target_frame_ids"], self.selected_ids)
        self.assertEqual([frame["frame_id"] for frame in batch["frames"]], self.selected_ids)
        self.assertEqual([row["frame_id"] for row in batch["qwen_annotations"]], self.selected_ids)
        self.assertEqual(len(plan["evidence_files"]), len(self.selected_ids))
        for target, name in zip(self.selected_ids, plan["evidence_files"]):
            projection = read(self.output / name)
            self.assertEqual(projection["target_frame_id"], target)
            self.assertEqual(projection["qwen_annotation"]["frame_id"], target)
            self.assertEqual([frame["frame_id"] for frame in projection["frames"]], self.selected_ids)
            self.assertEqual(plan["input_sha256"][name], sha256_file(self.output / name))
        self.assertEqual(self.run_pass(client, resume=True, should_stop=lambda: True)["status"], "paused")
        self.assertEqual(client.requests, [])
        self.assertEqual(self.run_pass(client, resume=True)["status"], "completed")
        self.assertEqual(len(client.requests), 1)

    def test_inspector_shows_one_case_with_all_joint_reviews_and_shared_raw_artifacts(self):
        client = JointLlamaCpp()
        self.run_pass(client)
        store = InspectorStore(self.root, self.fixture.dataset)
        records = store.records()["records"]
        self.assertEqual(len(records), 1, "Frozen Qwen/selection copies must not appear as extra cases")
        self.assertEqual(records[0]["medgemma_review_count"], len(self.selected_ids))
        reviews = {row["target_frame_id"]: row for row in read(self.output / "reviews.json")["reviews"]}
        for target in self.selected_ids:
            frame = store.frame(records[0]["id"], target)
            self.assertEqual(frame["raw"]["medgemma"], reviews[target])
            self.assertEqual(frame["medgemma"], reviews[target]["medgemma_review"])
            self.assertEqual([row["frame_id"] for row in frame["evidence"]], self.selected_ids)
            artifacts = [store.media(row["url"].rsplit("/", 1)[-1]) for row in frame["artifacts"]]
            self.assertIn(self.attempt() / "request.json", artifacts)
            self.assertIn(self.attempt() / "response.json", artifacts)
            response = next(path for path in artifacts if path.name == "response.json")
            self.assertEqual(response.read_bytes(), client.responses[0])

    def test_completed_resume_neither_loads_nor_calls_medgemma(self):
        self.run_pass(JointLlamaCpp())
        before = {path: path.read_bytes() for path in self.attempt().iterdir()}
        client = JointLlamaCpp()
        self.assertEqual(self.run_pass(client, resume=True)["status"], "completed")
        self.assertEqual(client.info_calls, [])
        self.assertEqual(client.requests, [])
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_pre_migration_surgery_review_cannot_resume_with_llama_cpp(self):
        self.run_pass(JointLlamaCpp(), prepare_only=True)
        plan = read(self.output / "run.json")
        plan.pop("runtime")
        write(self.output / "run.json", plan)
        write(self.output / "session.json", {"run_sha256": sha256_file(self.output / "run.json")})
        client = JointLlamaCpp()
        with self.assertRaisesRegex(ContractError, "predates the llama.cpp migration"):
            self.run_pass(client, resume=True)
        self.assertEqual(client.info_calls, [])
        self.assertEqual(client.requests, [])

    def test_invalid_completion_usage_cannot_be_recovered_as_an_accepted_response(self):
        failed = JointLlamaCpp(transform=lambda envelope: {**envelope, "usage": {
            **envelope["usage"], "completion_tokens": self.config.num_predict + 1}})
        with self.assertRaisesRegex(ContractError, "completion usage"):
            self.run_pass(failed)
        self.assertEqual((self.attempt() / "response.json").read_bytes(), failed.responses[0])
        self.assertFalse((self.attempt() / "receipt.json").exists())
        resumed = JointLlamaCpp()
        self.assertEqual(self.run_pass(resumed, resume=True)["status"], "completed")
        self.assertEqual(len(resumed.requests), 1)
        self.assertFalse((self.attempt() / "receipt.json").exists())
        self.assertTrue((self.attempt(2) / "receipt.json").exists())

    def test_saved_raw_joint_response_recovers_after_crash_without_any_repeat_call(self):
        client = JointLlamaCpp()
        with patch("yasargil.medgemma_surgery_review._parse", side_effect=KeyboardInterrupt("After raw response save")):
            with self.assertRaises(KeyboardInterrupt):
                self.run_pass(client)
        self.assertEqual((self.attempt() / "response.json").read_bytes(), client.responses[0])
        self.assertFalse((self.attempt() / "review.json").exists())
        resumed = JointLlamaCpp()
        self.assertEqual(self.run_pass(resumed, resume=True)["status"], "completed")
        self.assertEqual(resumed.info_calls, [])
        self.assertEqual(resumed.requests, [])
        self.assertEqual(len(list(self.attempt().parent.glob("attempt-*"))), 1)

    def test_missing_extra_or_mismatched_target_ids_never_publish_partial_reviews(self):
        def change_output(envelope, change):
            value = json.loads(envelope["choices"][0]["message"]["content"])
            change(value["reviews"])
            envelope["choices"][0]["message"]["content"] = json.dumps(value)
            return envelope
        def missing(rows):
            rows.pop(self.selected_ids[-1])
        def extra(rows):
            rows["invented-frame"] = copy.deepcopy(rows[self.selected_ids[0]])
        def mismatch(rows):
            rows[self.selected_ids[0]]["target_frame_id"] = self.selected_ids[1]
        for index, change in enumerate((missing, extra, mismatch)):
            with self.subTest(change=change.__name__):
                self.output = self.root / f"invalid-targets-{index}"
                client = JointLlamaCpp(transform=lambda envelope: change_output(envelope, change))
                with self.assertRaises((ContractError, LlamaCppError)):
                    self.run_pass(client)
                self.assertEqual(len(client.requests), 1)
                self.assertEqual((self.attempt() / "response.json").read_bytes(), client.responses[0])
                self.assertFalse((self.attempt() / "review.json").exists())
                self.assertEqual(read(self.output / "reviews.json")["reviews"], [])

    def test_unknown_evidence_citation_rejects_the_entire_joint_response(self):
        def bad_citation(envelope):
            value = json.loads(envelope["choices"][0]["message"]["content"])
            value["reviews"][self.selected_ids[-1]]["corrections"][0]["evidence_frame_ids"] = ["not-supplied"]
            envelope["choices"][0]["message"]["content"] = json.dumps(value)
            return envelope
        client = JointLlamaCpp(transform=bad_citation)
        with self.assertRaises((ContractError, LlamaCppError)):
            self.run_pass(client)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(read(self.output / "reviews.json")["reviews"], [])
        self.assertEqual((self.attempt() / "response.json").read_bytes(), client.responses[0])

    def test_positive_prompt_usage_and_reserved_output_capacity_are_required(self):
        for index, count in enumerate((None, 0, -1, True, self.config.num_ctx - self.config.num_predict + 1)):
            with self.subTest(count=count):
                self.output = self.root / f"bad-budget-{index}"
                client = JointLlamaCpp(transform=lambda envelope: {**envelope, "usage": {**envelope["usage"], "prompt_tokens": count}})
                with self.assertRaises((ContractError, LlamaCppError)):
                    self.run_pass(client)
                self.assertEqual(len(client.requests), 1)
                self.assertEqual((self.attempt() / "response.json").read_bytes(), client.responses[0])
                self.assertFalse((self.attempt() / "review.json").exists())

    def test_malformed_unfinished_and_truncated_responses_have_no_per_frame_fallback(self):
        transforms = (lambda _: b"{partial joint response", lambda value: {**value, "choices": [{**value["choices"][0], "finish_reason": None}]},
                      lambda value: {**value, "choices": [{**value["choices"][0], "finish_reason": "length"}]})
        for index, transform in enumerate(transforms):
            with self.subTest(index=index):
                self.output = self.root / f"bad-response-{index}"
                client = JointLlamaCpp(transform=transform)
                with self.assertRaises((ContractError, LlamaCppError)):
                    self.run_pass(client)
                self.assertEqual(len(client.requests), 1)
                self.assertEqual((self.attempt() / "response.json").read_bytes(), client.responses[0])
                self.assertTrue((self.attempt() / "failure.json").exists())
                self.assertEqual(read(self.output / "reviews.json")["reviews"], [])

    def test_partial_transport_body_is_preserved_and_explicit_retry_is_one_new_joint_call(self):
        failed = JointLlamaCpp(transform=lambda _: b"partial HTTP body", transport_error=True)
        with self.assertRaises(LlamaCppError):
            self.run_pass(failed)
        before = {path: path.read_bytes() for path in self.attempt().iterdir()}
        self.assertEqual((self.attempt() / "response.json").read_bytes(), b"partial HTTP body")
        resumed = JointLlamaCpp()
        self.assertEqual(self.run_pass(resumed, resume=True)["status"], "completed")
        self.assertEqual(len(resumed.requests), 1)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual({row["call_directory"] for row in read(self.output / "reviews.json")["reviews"]},
                         {"calls/surgery/attempt-0002"})

    def test_accepted_receipt_and_all_raw_artifacts_are_immutable_on_resume(self):
        self.run_pass(JointLlamaCpp())
        for name in ("request.json", "response.json", "model.json", "review.json", "receipt.json"):
            with self.subTest(name=name):
                path = self.attempt() / name
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                resumed = JointLlamaCpp()
                with self.assertRaises(ContractError):
                    self.run_pass(resumed, resume=True)
                self.assertEqual(resumed.info_calls, [])
                self.assertEqual(resumed.requests, [])
                path.write_bytes(original)

    def test_changed_completed_reviews_are_rejected_without_overwriting_them_or_calling_model(self):
        self.run_pass(JointLlamaCpp())
        path = self.output / "reviews.json"
        changed = read(path)
        changed["reviews"][0]["medgemma_review"]["revised_annotation"]["visible_observation"] = (
            "TAMPERED: this text was never returned by MedGemma.")
        write(path, changed)
        before = path.read_bytes()
        raw_before = {item: item.read_bytes() for item in self.attempt().iterdir()}
        resumed = JointLlamaCpp()
        with self.assertRaises(ContractError):
            self.run_pass(resumed, resume=True)
        self.assertEqual(path.read_bytes(), before, "A rejected publication must remain available for inspection")
        self.assertEqual(raw_before, {item: item.read_bytes() for item in raw_before})
        self.assertEqual(resumed.info_calls, [])
        self.assertEqual(resumed.requests, [])

    def test_invalid_accepted_receipt_does_not_erase_published_reviews(self):
        self.run_pass(JointLlamaCpp())
        published = (self.output / "reviews.json").read_bytes()
        receipt = self.attempt() / "receipt.json"
        receipt.write_bytes(receipt.read_bytes() + b"\n")
        resumed = JointLlamaCpp()
        with self.assertRaises(ContractError):
            self.run_pass(resumed, resume=True)
        self.assertEqual((self.output / "reviews.json").read_bytes(), published)
        self.assertEqual(resumed.info_calls, [])
        self.assertEqual(resumed.requests, [])

    def test_evidence_request_is_deferred_without_new_model_or_image_calls(self):
        client = JointLlamaCpp(deferred_ids=self.selected_ids[:1])
        with patch("yasargil.frame_annotation.LocalVideoRuntime", side_effect=AssertionError("No Qwen followup")):
            summary = self.run_pass(client)
        self.assertEqual(summary["status"], "completed_with_deferred_evidence")
        self.assertEqual(summary["reviewed_frame_count"], 4)
        self.assertEqual(summary["deferred_frame_count"], 1)
        self.assertEqual(len(client.requests), 1)
        self.assertFalse(summary["automated_followup"])
        deferred = read(self.output / "deferred-evidence.json")["requests"]
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0]["target_frame_id"], self.selected_ids[0])
        self.assertEqual((self.output / deferred[0]["response_path"]).read_bytes(), client.responses[0])


if __name__ == "__main__":
    unittest.main()
