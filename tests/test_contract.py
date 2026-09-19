"""Contract boundary tests use fabricated temporary source data and reviewers only."""
from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from yasargil.contract import (ContractError, canonical_hash, export_records, load_export,
                              resolve_asset, sha256_file, validate_corpus, validate_record)
from yasargil.sospine import import_case


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "source"
        self.artifacts = self.base / "artifacts"
        self.artifacts.mkdir()
        (self.root / "metadata").mkdir(parents=True)
        (self.root / "documentation").mkdir()
        (self.root / "metadata/source_manifest.json").write_text('{"fixture": "fabricated unit-test source"}')
        (self.root / "documentation/readme.txt").write_text("Fabricated unit-test fixture; no actual dataset claims.")
        image_dir = self.root / "frames/S1A2"
        image_dir.mkdir(parents=True)
        self.raw_points = []
        self.raw_boxes = []
        for i in (1, 2, 3):
            name = f"S1A2_frame_{i:08d}.jpeg"
            Image.new("RGB", (16, 8), (i * 50, 0, 0)).save(image_dir / name)
            for label in ("grasper ", "needle driver tip"):
                self.raw_points.append({"": str(len(self.raw_points)), "trial_frame": name, "x1": "2.5", "y1": "3", "x2": "2.5", "y2": "3", "label": label})
                self.raw_boxes.append({"trial_frame": name, "x1": "0", "y1": "0", "x2": "9", "y2": "7", "label": label.strip()})
        self._csv("sospine_tool_tips.csv", self.raw_points)
        self._csv("sospine_bbox.csv", self.raw_boxes)
        self._csv("sospine_outcomes.csv", [{"Trial ID": "S1A2", "Length": "0:00:03", "Postgraduate year": "2",
                  "Prior experience with MISS": "N", "Number prior cases": "0", "Time for repair": "50",
                  "Leak At 40mmHg": "Y", "Total Frames at 1 FPS": "3"}])
        self.record = import_case(self.root, "S1A2", [1, 2, 3])

    def _csv(self, name, rows):
        with (self.root / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _reviewed_fixture(self):
        """Fabricated gate fixture only; never used for a real dataset or review packet."""
        record = copy.deepcopy(self.record)
        record["status"] = "reviewed"
        record["training_view"]["eligibility"] = "eligible"
        manifest = self.artifacts / "split.json"
        manifest.write_text(json.dumps({"grouping_strategy": "surgeon", "assignments": {"S1": "train"}}))
        record["assets"].append({"asset_id": "split", "role": "split_manifest", "origin": "human_authored",
            "location": "artifact:split.json", "sha256": sha256_file(manifest), "media_type": "application/json",
            "derived_from_asset_ids": [], "transformation_id": None})
        record["partition"].update(name="train", split_manifest_asset_id="split")
        for turn in record["training_view"]["turn_links"]:
            rid = "fabricated-review-" + turn["turn_id"]
            record["reviews"].append({"review_id": rid, "reviewer_id": "FABRICATED_UNIT_TEST_REVIEWER",
                "reviewer_role": "trained_annotator", "reviewed_at": "2026-09-10T12:00:00Z",
                "rubric_id": "unit-test-only", "rubric_version": "1", "target_claim_ids": turn["expressed_claim_ids"],
                "target_message_indices": [turn["user_message_index"], turn["assistant_message_index"]],
                "outcome_hidden_during_review": True, "generator_identity_hidden": True, "verdict": "supported",
                "evidence_adequacy": "adequate", "temporal_correctness": "correct", "clinical_appropriateness": "not_applicable",
                "question_answerability": "from_supplied_input", "adds_useful_supervision": "yes", "error_severity": "none",
                "correction_time_seconds": None, "replacement_claim_ids": [], "notes": "Synthetic test fixture, never an actual human approval."})
            turn["review_ids"] = [rid]
            for claim in record["claims"]:
                if claim["claim_id"] in turn["expressed_claim_ids"]:
                    claim.update(disposition="retained", review_ids=[rid])
        return record

    def _validate(self, record=None, training=False):
        return validate_record(record or self.record, dataset_root=self.root, artifact_root=self.artifacts, training=training)

    def _export(self, record=None, preview=True):
        path = self.base / ("rows.preview.jsonl" if preview else "rows.jsonl")
        export_records([record or self.record], path, dataset_root=self.root, artifact_root=self.artifacts,
                       preview=preview, partition=None if preview else "train")
        return path

    def test_exact_raw_import_and_no_automatic_review(self):
        self._validate()
        point = self.record["original_annotations"][0]
        self.assertEqual(point["raw_value"], self.raw_points[0])
        self.assertEqual(point["source_locator"]["locator"], "2")
        self.assertEqual(point["original_origin"], "manual")
        self.assertFalse(self.record["reviews"])
        self.assertFalse(self.record["generation_runs"])
        self.assertTrue(self.record["case_outcomes"][0]["value"])
        self.assertEqual(self.record["patient_risk_adjustment"]["status"], "unavailable")

    def test_pending_draft_cannot_train(self):
        with self.assertRaises(ContractError):
            self._validate(training=True)

    def test_reviewed_fixture_can_export_and_hydrate_in_order(self):
        path = self._export(self._reviewed_fixture(), preview=False)
        rows = load_export(path, self.root, artifact_root=self.artifacts)
        images = [b["image"] for m in rows[0]["messages"] for b in m["content"] if b["type"] == "image"]
        self.assertEqual([im.mode for im in images], ["RGB"] * 3)
        self.assertLess(images[0].getpixel((0, 0))[0], images[1].getpixel((0, 0))[0])
        self.assertEqual(set(rows[0]), {"messages"})

    def test_preview_requires_explicit_preview_and_partition(self):
        path = self._export()
        with self.assertRaisesRegex(ContractError, "Draft preview"):
            load_export(path, self.root)
        rows = load_export(path, self.root, allow_preview=True, expected_partition="unassigned")
        self.assertEqual(len(rows), 1)

    def test_no_output_overwrite(self):
        self._export()
        with self.assertRaisesRegex(ContractError, "already exists"):
            self._export()

    def test_export_api_protects_source_directory_and_jsonl_contract(self):
        for output in (self.root / "output.preview.jsonl", self.base / "output.json"):
            with self.subTest(output=output), self.assertRaises(ContractError):
                export_records([self.record], output, dataset_root=self.root, preview=True)

    def test_wrong_hash_and_raw_row_rejected(self):
        for change in ("hash", "raw"):
            record = copy.deepcopy(self.record)
            if change == "hash":
                record["assets"][0]["sha256"] = "0" * 64
            else:
                record["original_annotations"][0]["raw_value"]["label"] = "invented"
            with self.subTest(change=change), self.assertRaises(ContractError):
                self._validate(record)

    def test_source_locator_cannot_skip_csv_verification(self):
        for target in ("original_annotations", "case_outcomes"):
            record = copy.deepcopy(self.record)
            record[target][0]["source_locator"]["locator_type"] = "whole_asset"
            with self.subTest(target=target), self.assertRaisesRegex(ContractError, "CSV"):
                self._validate(record)

    def test_wrong_outcome_value_rejected(self):
        self.record["case_outcomes"][0]["value"] = False
        with self.assertRaisesRegex(ContractError, "measured leak"):
            self._validate()

    def test_future_student_evidence_rejected(self):
        self.record["training_view"]["turn_links"][0]["cutoff_frame_index"] = 0
        with self.assertRaisesRegex(ContractError, "Future student"):
            self._validate()

    def test_claim_span_must_match_actual_text(self):
        self.record["claims"][0]["text"] = "A made-up statement."
        with self.assertRaisesRegex(ContractError, "exact conversation span"):
            self._validate()

    def test_unreviewed_text_cannot_hide_between_claims(self):
        record = self._reviewed_fixture()
        record["training_view"]["messages"][1]["content"][0]["text"] += " This step guarantees recovery."
        with self.assertRaisesRegex(ContractError, "unreviewed span"):
            self._validate(record, training=True)

    def test_earlier_context_requires_review_with_final_turn_loss(self):
        record = self._reviewed_fixture()
        record["claims"][0]["disposition"] = "pending"
        with self.assertRaisesRegex(ContractError, "unretained"):
            self._validate(record, training=True)

    def test_negative_claim_review_cannot_be_omitted(self):
        record = self._reviewed_fixture()
        adverse = copy.deepcopy(record["reviews"][0])
        adverse.update(review_id="adverse", verdict="unsupported", error_severity="major")
        record["reviews"].append(adverse)
        with self.assertRaisesRegex(ContractError, "all applicable reviews"):
            self._validate(record, training=True)

    def test_positive_turn_review_does_not_override_negative(self):
        record = self._reviewed_fixture()
        adverse = copy.deepcopy(record["reviews"][0])
        adverse.update(review_id="adverse", verdict="unsupported", error_severity="major", target_claim_ids=[])
        record["reviews"].append(adverse)
        record["training_view"]["turn_links"][0]["review_ids"].append("adverse")
        with self.assertRaisesRegex(ContractError, "adverse"):
            self._validate(record, training=True)

    def test_unresolved_relevant_conflict_rejected(self):
        record = self._reviewed_fixture()
        record["source_conflicts"].append({"conflict_id": "conflict", "annotation_ids": [],
            "claim_ids": [record["claims"][0]["claim_id"]], "description": "Fabricated test dispute",
            "status": "unresolved", "resolution_review_id": None})
        with self.assertRaisesRegex(ContractError, "conflict remains"):
            self._validate(record, training=True)

    def test_future_derived_image_rejected(self):
        record = copy.deepcopy(self.record)
        image = next(a for a in record["assets"] if a["asset_id"] == "image-f000001")
        image.update(role="derived_image", origin="deterministically_derived", derived_from_asset_ids=["image-f000003"])
        with self.assertRaisesRegex(ContractError, "future or unverified"):
            self._validate(record)

    def test_transform_cycle_rejected(self):
        record = copy.deepcopy(self.record)
        image = next(a for a in record["assets"] if a["asset_id"] == "image-f000001")
        image.update(transformation_id="cycle", origin="deterministically_derived", role="derived_image")
        record["transformations"].append({"transformation_id": "cycle", "operation": "crop", "implementation": "test",
            "implementation_version": "1", "input_asset_ids": [image["asset_id"]], "input_annotation_ids": [],
            "output_asset_ids": [image["asset_id"]], "output_claim_ids": [], "parameters": {}, "description": "Fabricated cycle"})
        with self.assertRaisesRegex(ContractError, "Cyclic"):
            self._validate(record)

    def test_bounding_box_positive_area_required(self):
        self.record["claims"][0]["evidence"]["regions"].append({"frame_id": "f000001", "type": "bbox_xyxy",
            "coordinate_system": "normalized_original_image_edges_0_1", "coordinates": [.5, .2, .5, .8], "derivation": "converted_source_geometry"})
        with self.assertRaisesRegex(ContractError, "Degenerate"):
            self._validate()

    def test_corpus_split_leakage_detected_before_export_filter(self):
        train = self._reviewed_fixture()
        other = copy.deepcopy(self.record)
        other["record_id"] = "another-window"
        with self.assertRaisesRegex(ContractError, "Cross-partition"):
            export_records([train, other], self.base / "train.jsonl", dataset_root=self.root,
                           artifact_root=self.artifacts, partition="train")

    def test_receipt_tampering(self):
        path = self._export()
        receipt_path = path.with_suffix(".receipt.json")
        original = json.loads(receipt_path.read_text())
        for field in ("row_count", "schema_sha256", "row_binding", "duplicate_media", "loss_scope"):
            receipt = copy.deepcopy(original)
            if field == "row_count":
                receipt[field] = 2
            elif field == "schema_sha256":
                receipt[field] = "0" * 64
            elif field == "row_binding":
                receipt["records"][0]["row_sha256"] = "0" * 64
            elif field == "duplicate_media":
                receipt["media"].append(receipt["media"][0])
            else:
                receipt[field] = "all_assistant_turns"
            receipt_path.write_text(json.dumps(receipt))
            with self.subTest(field=field), self.assertRaises(ContractError):
                load_export(path, self.root, allow_preview=True, expected_partition="unassigned")

    def test_text_only_followup_can_reuse_earlier_images(self):
        record = copy.deepcopy(self.record)
        messages = record["training_view"]["messages"]
        messages[2]["content"] = [{"type": "text", "text": "Repeat the first assessment."}]
        first, second = record["training_view"]["turn_links"][:2]
        second["student_frame_ids"] = first["student_frame_ids"].copy()
        messages[3]["content"] = copy.deepcopy(messages[1]["content"])
        for original, repeated in zip(record["claims"][:2], record["claims"][2:4]):
            repeated["text"] = original["text"]
            repeated["evidence"] = copy.deepcopy(original["evidence"])
            repeated["transformation_id"] = None
            repeated["output_locations"][0]["end_character"] = len(repeated["text"])
        record["training_view"]["turn_links"][2]["student_frame_ids"] = ["f000001", "f000003"]
        self._validate(record)

    def test_path_traversal_and_symlink_escape(self):
        outside = self.base / "outside.txt"
        outside.write_text("outside")
        (self.root / "escape.txt").symlink_to(outside)
        for location in ("../outside.txt", str(outside), "escape.txt", "https://example.org/image.jpeg"):
            with self.subTest(location=location), self.assertRaises(ContractError):
                resolve_asset(location, self.root)

    def test_unlabeled_frames_preserved_without_negative_targets(self):
        for row in self.raw_points:
            row["label"] = ""
        self._csv("sospine_tool_tips.csv", self.raw_points)
        record = import_case(self.root, "S1A2", [1, 2, 3])
        self.assertEqual(len(record["frame_selection"]["frames"]), 3)
        self.assertFalse(record["claims"])
        self.assertFalse(record["training_view"]["messages"])
        with self.assertRaises(ContractError):
            self._export(record)

    def test_model_proposals_stay_archivable_but_request_adapter_gates_training(self):
        record = self._reviewed_fixture()
        for role in ("teacher_request", "teacher_response"):
            path = self.artifacts / (role + ".json")
            path.write_text('{"fixture": "not an executed model request"}')
            record["assets"].append({"asset_id": role, "role": role, "origin": "model_generated",
                "location": "artifact:" + path.name, "sha256": sha256_file(path), "media_type": "application/json",
                "derived_from_asset_ids": [], "transformation_id": None})
        record["generation_runs"].append({"run_id": "fabricated-generator", "model_name": "test-only",
            "model_digest": "0" * 64, "quantization": "test-only", "runtime": "test-only", "runtime_version": "1",
            "request_asset_id": "teacher_request", "response_asset_id": "teacher_response", "prompt_version": "1",
            "input_mode": "causal_prefix", "input_frame_ids": ["f000001"], "input_annotation_ids": [],
            "input_reference_asset_ids": [], "previous_message_indices": [], "outcome_ids_seen": [],
            "maximum_frame_index_seen": 1, "generation_parameters": {}})
        record["claims"][0].update(origin="model_generated", generation_run_id="fabricated-generator")
        record["training_view"]["turn_links"][0]["generation_run_ids"] = ["fabricated-generator"]
        self._validate(record)
        with self.assertRaisesRegex(ContractError, "teacher-request adapter"):
            self._validate(record, training=True)
        record["generation_runs"][0]["maximum_frame_index_seen"] = 3
        with self.assertRaisesRegex(ContractError, "Generator saw future"):
            self._validate(record)

    def test_review_packet_is_blinded_escaped_and_never_approval(self):
        from yasargil.review import write_review_packet
        record = copy.deepcopy(self.record)
        record["training_view"]["messages"][0]["content"][0]["text"] += " <script>attack()</script>"
        before = canonical_hash(record)
        output = self.base / "review.html"
        result = write_review_packet(record, output, dataset_root=self.root)
        rendered = Path(result["html"]).read_text()
        worksheet = json.loads(Path(result["review_template"]).read_text())
        self.assertIn("data:image/jpeg;base64,", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script>attack()", rendered)
        self.assertNotIn("sospine_outcomes.csv", rendered)
        self.assertNotIn("source_reexpression", rendered)
        self.assertIsNone(worksheet["reviewer_id"])
        self.assertEqual(worksheet["archive_sha256"], before)
        self.assertTrue(all(item["verdict"] is None for item in worksheet["items"]))
        self.assertEqual(canonical_hash(record), before)
        with self.assertRaises(ContractError):
            write_review_packet(record, output, dataset_root=self.root)

    def test_review_packet_cannot_write_into_source(self):
        from yasargil.review import write_review_packet
        with self.assertRaises(ContractError):
            write_review_packet(self.record, self.root / "review.html", dataset_root=self.root)


if __name__ == "__main__":
    unittest.main()
