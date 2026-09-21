"""One surgery packet keeps exactly the selected images and all review targets."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator
from PIL import Image

from yasargil.contract import ContractError, sha256_file
from yasargil.medgemma_evidence import _dataset_context
from yasargil.medgemma_surgery_contract import (
    build_surgery_evidence, build_surgery_reviews, surgery_schema, target_evidence,
)
from yasargil.video_source import media_timeline


def review(frame_id, supplied_ids, *, needs_more=False):
    return {"target_frame_id": frame_id,
            "status": "needs_more_evidence" if needs_more else "review_complete",
            "assessment": "uncertain" if needs_more else "retained",
            "revised_annotation": {"visible_observation": "A colored image is visible.", "visibility": "clear",
                "contextual_claims": [{"claim": "Other supplied images show different colors.",
                                       "evidence_frame_ids": list(supplied_ids)}],
                "uncertainties": ["The omitted observations cannot be inspected."] if needs_more else []},
            "corrections": [],
            "evidence_requests": [{"question": "What appears between the selected observations?",
                "reason": "The cited observations were not supplied.", "target": "temporal_context",
                "start_ms": 1000, "end_ms": 4000}] if needs_more else []}


class MedGemmaSurgeryContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.dataset = self.root / "SOSpine"
        release = self.dataset / "frames/S1A2"
        release.mkdir(parents=True)
        frames = []
        for index in range(6):
            path = release / f"S1A2_frame_{index + 1:08d}.jpeg"
            Image.new("RGB", (16, 16), (index * 35, 0, 0)).save(path)
            frames.append({"frame_id": f"f{index:06d}", "frame_index": index,
                "release_frame_index": index + 1, "source_path": str(path), "source_sha256": sha256_file(path),
                "image_path": str(path), "image_sha256": sha256_file(path), "timestamp_ms": float(index * 1000),
                "timestamp_basis": "reconstructed_nominal", "source_pts": None, "time_base": None,
                "source_timestamp_ms": None, "source_acquisition_time": None, "video_pts": index * 1000,
                "video_time_base": "1/1000", "width": 16, "height": 16})
        self.source = {"source_kind": "released_image_sequence", "source_path": str(release),
            "source_sha256": "a" * 64, "frames": frames, "expected_video_frames": 6,
            "released_fps": 1., "duration_ms": 6000., "timestamp_basis": "reconstructed_nominal",
            "timestamp_origin_source_pts": None,
            "duration_basis": "released_frame_count_divided_by_explicit_nominal_fps", "video_stream_index": 0,
            "ffprobe": {"streams": [{"index": 0, "time_base": "1/1000"}],
                        "frames": [{"pts": index * 1000, "duration": 1000} for index in range(6)]}}
        self.source["media_timeline"] = media_timeline(self.source)
        self.ids = [frames[index]["frame_id"] for index in (0, 3, 5)]
        drafts = [{**copy.deepcopy(frames[index]), "visible_observation": "A colored image is visible.",
                   "visibility": "clear", "contextual_claims": [], "uncertainties": [],
                   "review_required": True, "training_eligible": False} for index in (0, 3, 5)]
        drafts[0]["contextual_claims"] = [{"claim": "The video shows intermediate colors.", "evidence_intervals": [{
            "start_ms": 1000, "end_ms": 4000, "supporting_frames": copy.deepcopy(frames[1:5])}]}]
        self.document = {"schema_version": "contextual-frame-annotations-v1", "context_check": "consistent",
                         "annotations": drafts, "training_eligible": False}

    def batch(self):
        return build_surgery_evidence(self.source, self.document, procedure_context="Documented synthetic context.")

    def response(self):
        return {"reviews": {frame_id: review(frame_id, self.ids) for frame_id in self.ids}}

    def write_tables(self):
        fields = ["trial_frame", "x1", "y1", "x2", "y2", "label"]
        for filename in ("sospine_tool_tips.csv", "sospine_bbox.csv"):
            with (self.dataset / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for frame in self.source["frames"]:
                    writer.writerow({"trial_frame": Path(frame["source_path"]).name,
                                     "x1": "1.20", "y1": "2", "x2": "3", "y2": "4",
                                     "label": f" ORIGINAL_LABEL_{frame['frame_index']} "})
        (self.dataset / "sospine_outcomes.csv").write_text(
            "Trial ID,Leak At 40mmHg,Time for repair\nS1A2,Y,SECRET_OUTCOME_DURATION\n")

    def test_selected_images_are_supplied_once_without_neighbors_or_cited_additions(self):
        batch = self.batch()
        self.assertEqual(batch["target_frame_ids"], self.ids)
        self.assertEqual([frame["frame_id"] for frame in batch["frames"]], self.ids)
        self.assertTrue(all(frame["evidence_roles"] == ["selected_key_frame"] for frame in batch["frames"]))
        self.assertEqual(batch["qwen_annotations"], self.document["annotations"])
        self.assertEqual(batch["omitted_qwen_supporting_frame_ids"], ["f000001", "f000002", "f000004"])
        coverage = batch["qwen_evidence_coverage"][self.ids[0]][0]
        self.assertEqual(coverage["supplied_frame_ids"], [self.ids[1]])
        self.assertEqual(coverage["omitted_frame_ids"], batch["omitted_qwen_supporting_frame_ids"])
        self.assertEqual(coverage["omission_reason"], "not_in_final_selected_set")
        self.assertFalse(batch["automatic_neighbor_expansion"])
        self.assertFalse(batch["automated_followup"])
        self.assertEqual(batch["media_timeline"]["duration_ms"], 6000)
        self.assertEqual(batch["media_timeline"]["timestamp_basis"], "reconstructed_nominal")

    def test_inputs_and_full_original_drafts_are_preserved_without_aliasing(self):
        self.document["annotations"].reverse()
        before = copy.deepcopy((self.source, self.document))
        batch = self.batch()
        self.assertEqual(batch["target_frame_ids"], self.ids)
        packet = target_evidence(batch, self.ids[1])
        packet["frames"][0]["source_path"] = "/edited"
        packet["qwen_annotation"]["uncertainties"].append("edited")
        packet["dataset_context"]["documented_procedure_context"] = "edited"
        self.assertNotEqual(packet["frames"][0]["source_path"], batch["frames"][0]["source_path"])
        batch["qwen_annotations"][0]["contextual_claims"][0]["claim"] = "edited"
        batch["frames"][0]["timestamp_ms"] = 9000
        self.assertEqual((self.source, self.document), before)

    def test_v2_point_evidence_uses_only_the_canonical_selected_endpoint(self):
        self.document["schema_version"] = "contextual-frame-annotations-v2"
        last = self.source["frames"][-1]
        interval = {"start_frame_id": last["frame_id"], "end_frame_id": last["frame_id"],
                    "start_ms": last["timestamp_ms"], "end_ms": last["timestamp_ms"],
                    "supporting_frames": [copy.deepcopy(last)]}
        self.document["annotations"][0]["contextual_claims"][0]["evidence_intervals"] = [interval]
        batch = self.batch()
        coverage = batch["qwen_evidence_coverage"][self.ids[0]][0]
        self.assertEqual(coverage["available_frame_ids"], [last["frame_id"]])
        self.assertEqual(coverage["supplied_frame_ids"], [last["frame_id"]])
        self.assertTrue(coverage["complete"])
        self.assertEqual(batch["qwen_annotations"], self.document["annotations"])
        self.assertEqual(len(batch["frames"]), len(self.ids))

    def test_target_adapter_uses_all_selected_frames_and_per_target_roles(self):
        batch = self.batch()
        packet = target_evidence(batch, self.ids[1])
        self.assertEqual(packet["target_frame_id"], self.ids[1])
        self.assertEqual([frame["frame_id"] for frame in packet["frames"]], self.ids)
        self.assertEqual([frame["evidence_roles"] for frame in packet["frames"]],
                         [["earlier_selected"], ["target"], ["later_selected"]])
        self.assertEqual(packet["qwen_annotation"], self.document["annotations"][1])
        self.assertEqual(packet["dataset_context"], batch["dataset_context"])
        self.assertEqual(packet["limitations"], batch["limitations"])
        with self.assertRaisesRegex(ContractError, "Unknown surgery"):
            target_evidence(batch, "f000001")

    def test_original_selected_label_context_is_built_once_and_reused_without_outcomes(self):
        self.write_tables()
        with patch("yasargil.medgemma_surgery_contract._dataset_context", wraps=_dataset_context) as context:
            batch = self.batch()
            packets = [target_evidence(batch, frame_id) for frame_id in self.ids]
        self.assertEqual(context.call_count, 1)
        self.assertEqual(batch["dataset_context"]["status"], "available")
        labels = batch["dataset_context"]["original_annotations"]
        self.assertEqual(len(labels), 6)
        self.assertEqual({row["frame_id"] for row in labels}, set(self.ids))
        self.assertEqual(labels[0]["raw_value"]["x1"], "1.20")
        self.assertEqual(labels[0]["raw_value"]["label"], " ORIGINAL_LABEL_0 ")
        self.assertTrue(all(packet["dataset_context"] == batch["dataset_context"] for packet in packets))
        self.assertNotIn("SECRET_OUTCOME_DURATION", json.dumps(batch))
        self.assertEqual(batch["dataset_context"]["excluded_context"],
                         ["case_outcomes", "surgeon_experience", "repair_duration"])

    def test_missing_duplicate_unknown_or_changed_source_targets_are_rejected(self):
        cases = []
        missing = copy.deepcopy(self.document)
        missing["annotations"][0].pop("frame_id")
        cases.append(missing)
        duplicate = copy.deepcopy(self.document)
        duplicate["annotations"].append(copy.deepcopy(duplicate["annotations"][0]))
        cases.append(duplicate)
        unknown = copy.deepcopy(self.document)
        unknown["annotations"][0]["frame_id"] = "invented"
        cases.append(unknown)
        changed = copy.deepcopy(self.document)
        changed["annotations"][0]["timestamp_ms"] = 0
        cases.append(changed)  # Int and float locators must not be silently substituted.
        no_drafts = copy.deepcopy(self.document)
        no_drafts["annotations"] = []
        cases.append(no_drafts)
        for document in cases:
            with self.subTest(document=document), self.assertRaises(ContractError):
                build_surgery_evidence(self.source, document)
        source = copy.deepcopy(self.source)
        source["frames"][1]["frame_id"] = source["frames"][0]["frame_id"]
        with self.assertRaises(ContractError):
            build_surgery_evidence(source, self.document)

    def test_changed_qwen_supporting_frame_provenance_is_rejected(self):
        changed = copy.deepcopy(self.document)
        interval = changed["annotations"][0]["contextual_claims"][0]["evidence_intervals"][0]
        interval["supporting_frames"][0]["source_path"] = "/invented"
        with self.assertRaisesRegex(ContractError, "supporting frames differ"):
            build_surgery_evidence(self.source, changed)

    def test_response_requires_exactly_one_review_for_every_target(self):
        batch = self.batch()
        schema = surgery_schema(self.ids, 6000)
        Draft202012Validator.check_schema(schema)
        cases = []
        missing = self.response()
        missing["reviews"].pop(self.ids[0])
        cases.append(missing)
        extra = self.response()
        extra["reviews"]["invented"] = review("invented", self.ids)
        cases.append(extra)
        wrong_target = self.response()
        wrong_target["reviews"][self.ids[0]]["target_frame_id"] = self.ids[1]
        cases.append(wrong_target)
        cases.append({"reviews": list(self.response()["reviews"].values())})
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                build_surgery_reviews(raw, batch)
        for ids in ([], self.ids + [self.ids[0]], "f000000"):
            with self.subTest(ids=ids), self.assertRaises(ContractError):
                surgery_schema(ids, 6000)

    def test_only_supplied_images_can_support_cross_frame_claims_or_corrections(self):
        batch = self.batch()
        raw = self.response()
        rows = build_surgery_reviews(raw, batch)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["medgemma_review"]["revised_annotation"]["contextual_claims"][0]["evidence_frame_ids"], self.ids)
        for unknown in ("f000001", "invented"):
            for field in ("claim", "correction"):
                changed = self.response()
                judgment = changed["reviews"][self.ids[0]]
                if field == "claim":
                    judgment["revised_annotation"]["contextual_claims"][0]["evidence_frame_ids"] = [unknown]
                else:
                    judgment["corrections"] = [{"original_text": "Original", "revised_text": "Revised",
                                               "reason": "Evidence", "evidence_frame_ids": [unknown]}]
                with self.subTest(unknown=unknown, field=field), self.assertRaises(ContractError):
                    build_surgery_reviews(changed, batch)

    def test_deferred_requests_and_original_drafts_are_retained_in_legacy_shaped_reviews(self):
        batch = self.batch()
        raw = self.response()
        raw["reviews"][self.ids[0]] = review(self.ids[0], self.ids, needs_more=True)
        original = copy.deepcopy((raw, batch))
        rows = build_surgery_reviews(raw, batch)
        self.assertEqual([row["target_frame_id"] for row in rows], self.ids)
        self.assertEqual(rows[0]["deferred_evidence_requests"], raw["reviews"][self.ids[0]]["evidence_requests"])
        for index, row in enumerate(rows):
            self.assertEqual(row["qwen_annotation"], self.document["annotations"][index])
            self.assertEqual([frame["frame_id"] for frame in row["evidence"]["frames"]], self.ids)
            self.assertTrue(row["human_review_required"])
            self.assertFalse(row["training_eligible"])
            self.assertFalse(row["automated_followup"])
        rows[0]["qwen_annotation"]["visible_observation"] = "edited"
        self.assertEqual((raw, batch), original)

    def test_target_adapter_rejects_duplicate_or_missing_batch_images(self):
        for key in ("frames", "qwen_annotations"):
            for duplicate in (True, False):
                batch = self.batch()
                if duplicate:
                    batch[key][1] = copy.deepcopy(batch[key][0])
                else:
                    batch[key].pop()
                with self.subTest(key=key, duplicate=duplicate), self.assertRaises(ContractError):
                    target_evidence(batch, self.ids[0])
        batch = self.batch()
        batch["frames"][0]["image_path"] = "/replacement.jpeg"
        with self.assertRaisesRegex(ContractError, "provenance differs"):
            target_evidence(batch, self.ids[0])


if __name__ == "__main__":
    unittest.main()
