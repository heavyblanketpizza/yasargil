"""Teacher provenance tests use fabricated images, model receipts and reviews."""
from __future__ import annotations

import base64
import copy
import json
import unittest

import test_contract as source_fixture

from yasargil import enhancement_prompts as prompts
from yasargil.contract import ContractError, sha256_file, validate_record
from yasargil.teacher import (ADAPTER, ENVELOPE_BUILDER_VERSION, EVENT_QUESTION,
                              build_request, canonical_bytes, frame_caption, parse_response,
                              validate_teacher_runs)


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.source = source_fixture.ContractTests(methodName="runTest")
        self.source.setUp()
        self.addCleanup(self.source.doCleanups)
        self.record = self.source.record
        self.root, self.artifacts = self.source.root, self.source.artifacts
        self.metadata = {"model_name": "fixture-vlm:1", "model_digest": "sha256:" + "1" * 64,
                         "quantization": "Q4_K_M", "runtime_version": "0.fixture",
                         "tags_response": {"models": [{"name": "fixture-vlm:1", "model": "fixture-vlm:1",
                             "digest": "sha256:" + "1" * 64, "details": {"quantization_level": "Q4_K_M"}}]},
                         "show_response": {"details": {"quantization_level": "Q4_K_M"}, "capabilities": ["vision"]},
                         "version_response": {"version": "0.fixture"}}
        self._put("model-info", canonical_bytes(self.metadata), "reference_document")

    def _put(self, aid, raw, role):
        path = self.artifacts / f"{aid}.json"
        path.write_bytes(raw)
        asset = {"asset_id": aid, "role": role, "origin": "model_generated" if role == "teacher_response"
                 else "deterministically_derived", "location": f"artifact:{aid}.json", "sha256": sha256_file(path),
                 "media_type": "application/json", "derived_from_asset_ids": [], "transformation_id": None}
        self.record["assets"][:] = [a for a in self.record["assets"] if a["asset_id"] != aid]
        self.record["assets"].append(asset)

    def _output(self):
        return {"events": [{"event_id": "e1", "description": "A grasper is visible.",
                            "type": "visible_observation", "evidence_frame_ids": ["f000001"],
                            "annotation_ids": [], "assessment": "supported", "uncertainty": ""}],
                "questions": [{"question_id": "q1", "question": "Which instrument is visible?",
                               "answer": "A grasper is visible.", "evidence_frame_ids": ["f000001"],
                               "annotation_ids": [], "answerability": "visible"}],
                "searches": [], "disagreements": []}

    def _response(self, run, output):
        return {"model": run["model_name"], "created_at": "2026-09-11T00:00:00Z", "done": True,
                "done_reason": "stop", "message": {"role": "assistant", "content": json.dumps(output)},
                "total_duration": 123, "eval_count": 32}

    def _run(self, rid="run1", stage="propose", parents=(), frame_ids=None, output=None, cutoff=3):
        frame_ids = frame_ids or ["f000001", "f000002", "f000003"]
        run = {"run_id": rid, "model_name": self.metadata["model_name"],
               "model_digest": self.metadata["model_digest"], "quantization": self.metadata["quantization"],
               "runtime": "ollama", "runtime_version": self.metadata["runtime_version"],
               "request_asset_id": f"{rid}-request", "response_asset_id": f"{rid}-response",
               "prompt_version": prompts.PROMPT_VERSION, "input_mode": "causal_prefix",
               "input_frame_ids": frame_ids,
               "input_annotation_ids": [a["annotation_id"] for a in self.record["original_annotations"]
                                        if a["frame_ids"] and set(a["frame_ids"]) <= set(frame_ids)],
               "input_reference_asset_ids": [], "previous_message_indices": [], "outcome_ids_seen": [],
               "maximum_frame_index_seen": cutoff,
               "generation_parameters": {"adapter": ADAPTER, "envelope_builder_version": ENVELOPE_BUILDER_VERSION,
                   "stage": stage, "parent_run_ids": list(parents), "start_frame_index": 1,
                   "cutoff_frame_index": cutoff, "stage_question": prompts.stage_question(stage),
                   "options": {"temperature": 0, "num_predict": 512}, "model_metadata_asset_id": "model-info",
                   "think": False, "keep_alive": 0}}
        self.record["generation_runs"].append(run)
        raw = build_request(self.record, run, dataset_root=self.root, artifact_root=self.artifacts)
        self._put(run["request_asset_id"], raw, "teacher_request")
        self._put(run["response_asset_id"], canonical_bytes(self._response(run, output or self._output())), "teacher_response")
        return run

    def _rewrite_request(self, run, mutate):
        request = json.loads((self.artifacts / f"{run['request_asset_id']}.json").read_bytes())
        mutate(request)
        self._put(run["request_asset_id"], canonical_bytes(request), "teacher_request")

    def _rewrite_response(self, run, mutate):
        response = json.loads((self.artifacts / f"{run['response_asset_id']}.json").read_bytes())
        mutate(response)
        self._put(run["response_asset_id"], canonical_bytes(response), "teacher_response")

    def _verify(self):
        return validate_teacher_runs(self.record, dataset_root=self.root, artifact_root=self.artifacts)

    def _bind(self, run, question=False):
        output = self._output()
        item = output["questions"][0] if question else output["events"][0]
        text = item["answer"] if question else item["description"]
        cid = f"{run['run_id']}.q1.answer" if question else f"{run['run_id']}.e1"
        self.record["claims"] = [{"claim_id": cid, "text": text, "type": "visible_observation",
            "origin": "model_generated", "contribution": "adds_proposed_observation", "generation_run_id": run["run_id"],
            "transformation_id": None, "supersedes_claim_id": None,
            "evidence": {"frame_ids": item["evidence_frame_ids"], "original_annotation_ids": item["annotation_ids"],
                         "supporting_claim_ids": [], "reference_asset_ids": [], "regions": []},
            "output_locations": [{"message_index": 1, "content_block_index": 0,
                                  "start_character": 0, "end_character": len(text)}],
            "review_ids": [], "disposition": "pending"}]
        self.record["transformations"] = []
        assets = {a["asset_id"]: a for a in self.record["assets"]}
        images = [block for frame in self.record["frame_selection"]["frames"] for block in
                  ({"type": "text", "text": frame_caption(frame)},
                   {"type": "image", "image": assets[frame["asset_id"]]["location"]})]
        self.record["training_view"]["messages"] = [
            {"role": "user", "content": [{"type": "text", "text": item["question"] if question else EVENT_QUESTION}] + images},
            {"role": "assistant", "content": [{"type": "text", "text": text}]}]
        self.record["training_view"]["turn_links"] = [{"turn_id": "turn1", "user_message_index": 0,
            "assistant_message_index": 1, "task": "evidence_based_assessment", "input_mode": "causal_prefix",
            "cutoff_frame_index": 3, "student_frame_ids": ["f000001", "f000002", "f000003"],
            "student_annotation_ids": [], "student_reference_asset_ids": [], "expressed_claim_ids": [cid],
            "question_origin": "model_generated" if question else "deterministic_template",
            "generation_run_ids": [r["run_id"] for r in self.record["generation_runs"]], "review_ids": []}]

    def test_canonical_original_pixels_and_rows_are_retained(self):
        run = self._run()
        raw = (self.artifacts / f"{run['request_asset_id']}.json").read_bytes()
        request = json.loads(raw)
        self.assertEqual(raw, canonical_bytes(request))
        self.assertEqual(request["format"], prompts.response_schema())
        payload = json.loads(request["messages"][1]["content"])
        self.assertNotIn("case_outcomes", payload)
        self.assertEqual(len(payload["frames"]), 3)
        self.assertEqual(payload["original_annotations"][0]["raw_value"], self.source.raw_points[0])
        expected = (self.root / "frames/S1A2/S1A2_frame_00000001.jpeg").read_bytes()
        self.assertEqual(base64.b64decode(request["messages"][1]["images"][0], validate=True), expected)
        self.assertEqual(self._verify(), {"run1"})

    def test_reviewed_fabricated_teacher_record_can_pass_training_gate(self):
        run = self._run()
        self._bind(run, question=True)
        self.source.record = self.record
        reviewed = self.source._reviewed_fixture()
        for review in reviewed["reviews"]:
            review.update(reviewer_role="surgeon", clinical_appropriateness="appropriate")
        validate_record(reviewed, dataset_root=self.root, artifact_root=self.artifacts, training=True)
        self.assertEqual(self.record["status"], "draft")
        self.assertFalse(self.record["reviews"])

    def test_model_visible_observation_still_requires_clinical_domain_review(self):
        run = self._run()
        self._bind(run)
        self.source.record = self.record
        reviewed = self.source._reviewed_fixture()
        self.assertEqual(reviewed["claims"][0]["type"], "visible_observation")
        with self.assertRaisesRegex(ContractError, "Claim lacks passing applicable review"):
            validate_record(reviewed, dataset_root=self.root, artifact_root=self.artifacts, training=True)

    def test_generated_turn_requires_domain_review_even_when_claim_review_passes(self):
        run = self._run()
        self._bind(run, question=True)
        self.source.record = self.record
        reviewed = self.source._reviewed_fixture()
        claim_review = reviewed["reviews"][0]
        turn_review = copy.deepcopy(claim_review)
        turn_review.update(review_id="fabricated-turn-only-review", target_claim_ids=[])
        claim_review.update(reviewer_role="clinical_domain_expert", clinical_appropriateness="appropriate",
                            target_message_indices=[])
        reviewed["reviews"].append(turn_review)
        reviewed["training_view"]["turn_links"][0]["review_ids"] = [turn_review["review_id"]]
        with self.assertRaisesRegex(ContractError, "Turn needs reviewed input/answer"):
            validate_record(reviewed, dataset_root=self.root, artifact_root=self.artifacts, training=True)
        turn_review.update(reviewer_role="clinical_domain_expert", clinical_appropriateness="appropriate")
        validate_record(reviewed, dataset_root=self.root, artifact_root=self.artifacts, training=True)

    def test_clinician_review_must_explicitly_assess_clinical_appropriateness(self):
        run = self._run()
        self._bind(run, question=True)
        self.source.record = self.record
        reviewed = self.source._reviewed_fixture()
        for review in reviewed["reviews"]:
            review.update(reviewer_role="surgeon", clinical_appropriateness="not_applicable")
        with self.assertRaisesRegex(ContractError, "Claim lacks passing applicable review"):
            validate_record(reviewed, dataset_root=self.root, artifact_root=self.artifacts, training=True)

    def test_pending_teacher_record_stays_excluded_from_training(self):
        run = self._run()
        self._bind(run)
        validate_record(self.record, dataset_root=self.root, artifact_root=self.artifacts)
        with self.assertRaises(ContractError):
            validate_record(self.record, dataset_root=self.root, artifact_root=self.artifacts, training=True)

    def test_swapped_image_bytes_fail_even_when_request_hash_updated(self):
        run = self._run()
        self._rewrite_request(run, lambda r: r["messages"][1]["images"].reverse())
        with self.assertRaisesRegex(ContractError, "request bytes"):
            self._verify()

    def test_extra_outcome_or_hidden_message_fails(self):
        for key in ("outcome", "message", "options"):
            with self.subTest(key=key):
                run = self._run(rid=f"run-{key}")
                if key == "outcome":
                    def mutate(request):
                        payload = json.loads(request["messages"][1]["content"])
                        payload["outcome"] = True
                        request["messages"][1]["content"] = canonical_bytes(payload).decode()
                elif key == "message":
                    def mutate(request):
                        request["messages"].insert(1, {"role": "assistant", "content": "future answer"})
                else:
                    def mutate(request):
                        request["options"]["temperature"] = 1
                self._rewrite_request(run, mutate)
                with self.assertRaisesRegex(ContractError, "request bytes"):
                    self._verify()
                self.record["generation_runs"].remove(run)

    def test_freeform_stage_question_cannot_smuggle_case_metadata(self):
        run = self._run()
        run["generation_parameters"]["stage_question"] += " The repair leaked."
        with self.assertRaisesRegex(ContractError, "stage question"):
            self._verify()

    def test_outcome_annotations_are_rejected(self):
        run = self._run()
        run["input_annotation_ids"].append(self.record["original_annotations"][-1]["annotation_id"])
        with self.assertRaisesRegex(ContractError, "outcome"):
            self._verify()

    def test_forged_annotation_cannot_be_sent_by_request_builder(self):
        run = self._run()
        self.record["original_annotations"][0]["raw_value"]["label"] = "leak-free repair"
        with self.assertRaisesRegex(ContractError, "original source row"):
            build_request(self.record, run, dataset_root=self.root, artifact_root=self.artifacts)

    def test_annotation_cannot_be_joined_to_another_supplied_frame(self):
        run = self._run()
        self.record["original_annotations"][0]["frame_ids"] = ["f000002"]
        with self.assertRaisesRegex(ContractError, "annotation/frame"):
            build_request(self.record, run, dataset_root=self.root, artifact_root=self.artifacts)

    def test_parent_outputs_rebuilt_from_exact_response(self):
        parent = self._run()
        child = self._run("run2", "review", ["run1"])
        self.assertEqual(self._verify(), {"run1", "run2"})
        request = json.loads((self.artifacts / f"{child['request_asset_id']}.json").read_bytes())
        self.assertEqual(json.loads(request["messages"][1]["content"])["parent_outputs"][0]["output"], self._output())
        self._rewrite_response(parent, lambda r: r["message"].update(content=json.dumps(self._output() | {"disagreements": ["Changed parent output"]})))
        with self.assertRaisesRegex(ContractError, "request bytes"):
            self._verify()

    def test_transitive_future_exposure_is_rejected(self):
        self._run()
        self._run("run2", "review", ["run1"])
        child = self._run("run3", "final_revise", ["run2"])
        child["generation_parameters"]["cutoff_frame_index"] = 2
        child["maximum_frame_index_seen"] = 2
        child["input_frame_ids"] = ["f000001", "f000002"]
        with self.assertRaisesRegex(ContractError, "ancestor.*future"):
            self._verify()

    def test_unknown_and_cyclic_ancestors_are_rejected(self):
        parent = self._run()
        child = self._run("run2", "review", ["run1"])
        child["generation_parameters"]["parent_run_ids"] = ["missing"]
        with self.assertRaisesRegex(ContractError, "Missing teacher ancestor"):
            self._verify()
        child["generation_parameters"]["parent_run_ids"] = ["run1"]
        parent["generation_parameters"]["parent_run_ids"] = ["run2"]
        with self.assertRaisesRegex(ContractError, "Cyclic"):
            self._verify()

    def test_independent_observer_cannot_receive_proposals(self):
        self._run()
        with self.assertRaisesRegex(ContractError, "Independent observation"):
            self._run("run2", "independent_observe", ["run1"])

    def test_transitive_maximum_cannot_omit_ancestor_pixels(self):
        self._run()
        self._run("run2", "review", ["run1"])
        run = self._run("run3", "search", ["run2"], frame_ids=["f000001"])
        run["maximum_frame_index_seen"] = 1
        with self.assertRaisesRegex(ContractError, "transitive inputs"):
            self._verify()

    def test_metadata_digest_quantization_and_runtime_must_agree(self):
        self._run()
        for key in ("digest", "quantization", "version"):
            with self.subTest(key=key):
                value = copy.deepcopy(self.metadata)
                if key == "digest":
                    value["tags_response"]["models"][0]["digest"] = "sha256:" + "2" * 64
                elif key == "quantization":
                    value["show_response"]["details"]["quantization_level"] = "F16"
                else:
                    value["version_response"]["version"] = "different"
                self._put("model-info", canonical_bytes(value), "reference_document")
                with self.assertRaisesRegex(ContractError, "digest|quantization|runtime"):
                    self._verify()

    def test_text_only_model_receipt_is_rejected(self):
        self._run()
        self.metadata["show_response"]["capabilities"] = ["completion"]
        self._put("model-info", canonical_bytes(self.metadata), "reference_document")
        with self.assertRaisesRegex(ContractError, "vision capability"):
            self._verify()

    def test_failed_truncated_wrong_model_and_hidden_reasoning_rejected(self):
        run = self._run()
        original = self._response(run, self._output())
        variants = [original | {"done": False}, original | {"done_reason": "length"},
                    original | {"model": "other"}, original | {"error": "failure"},
                    original | {"message": original["message"] | {"thinking": "secret reasoning"}}]
        for response in variants:
            with self.subTest(response=response):
                with self.assertRaises(ContractError):
                    parse_response(canonical_bytes(response), run)

    def test_duplicate_json_keys_are_rejected(self):
        run = self._run()
        with self.assertRaisesRegex(ContractError, "Duplicate"):
            parse_response(b'{"done":true,"done":false}', run)

    def test_response_cannot_cite_unexposed_frames(self):
        run = self._run()
        output = self._output()
        output["events"][0]["evidence_frame_ids"] = ["future-frame"]
        self._rewrite_response(run, lambda r: r["message"].update(content=json.dumps(output)))
        with self.assertRaisesRegex(ContractError, "structured output"):
            self._verify()

    def test_swapped_generated_claim_or_question_rejected(self):
        run = self._run()
        self._bind(run, question=True)
        self.record["claims"][0]["text"] = "The repair is watertight."
        with self.assertRaisesRegex(ContractError, "claim text"):
            self._verify()
        self._bind(run, question=True)
        self.record["training_view"]["messages"][0]["content"][0]["text"] = "Did the repair leak?"
        with self.assertRaisesRegex(ContractError, "question/answer"):
            self._verify()

    def test_claim_evidence_cannot_be_swapped(self):
        run = self._run()
        self._bind(run)
        self.record["claims"][0]["evidence"]["frame_ids"] = ["f000002"]
        with self.assertRaisesRegex(ContractError, "claim evidence"):
            self._verify()

    def test_event_uncertainty_is_bound_to_exact_returned_text(self):
        output = self._output()
        output["events"][0]["uncertainty"] = "The instrument tip is partially obscured."
        run = self._run(output=output)
        self._bind(run)
        claim = copy.deepcopy(self.record["claims"][0])
        claim.update(claim_id="run1.e1.uncertainty", text=output["events"][0]["uncertainty"],
                     type="uncertainty_statement", output_locations=[])
        self.record["claims"].append(claim)
        self.assertEqual(self._verify(), {"run1"})
        claim["text"] = "The instrument tip is fully visible."
        with self.assertRaisesRegex(ContractError, "claim text"):
            self._verify()

    def test_student_frame_alias_cannot_smuggle_extra_facts(self):
        run = self._run()
        self._bind(run)
        self.record["training_view"]["messages"][0]["content"][1]["text"] += " The repair leaked."
        with self.assertRaisesRegex(ContractError, "frame alias"):
            self._verify()

    def test_teacher_ancestor_context_must_be_visible_to_student(self):
        self._run()
        run = self._run("run2", "final_revise", ["run1"])
        self._bind(run)
        self.record["training_view"]["turn_links"][0]["student_frame_ids"] = ["f000001"]
        with self.assertRaisesRegex(ContractError, "ancestor frame context"):
            self._verify()

    def test_unsupported_adapter_and_no_byte_roots_grant_no_training_gate(self):
        run = self._run()
        self.assertEqual(validate_teacher_runs(self.record), set())
        run["generation_parameters"]["adapter"] = "unknown"
        self.assertEqual(self._verify(), set())
        self._bind(run)
        self.source.record = self.record
        reviewed = self.source._reviewed_fixture()
        with self.assertRaisesRegex(ContractError, "verified teacher-request adapter"):
            validate_record(reviewed, dataset_root=self.root, artifact_root=self.artifacts, training=True)


if __name__ == "__main__":
    unittest.main()
