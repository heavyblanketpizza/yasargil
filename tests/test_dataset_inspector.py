"""Read-only inspector joins canonical evidence and serves registered files only."""
import copy
import csv
import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from yasargil.dataset_inspector import CurationError, InspectorStore, PreviewError, make_server


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class InspectorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.outputs = self.root / "outputs"
        self.selection = self.outputs / "selection" / "S1A1"
        self.annotation = self.outputs / "annotations" / "S1A1"
        self.review = self.outputs / "reviews" / "S1A1"
        self.dataset = self.root / "dataset"
        source_path = self.dataset / "frames" / "S1A1"
        source_path.mkdir(parents=True)
        self.frames = []
        for index in range(5):
            path = source_path / f"S1A1_frame_{index + 1:08d}.jpeg"
            path.write_bytes(b"fixture jpeg " + str(index).encode())
            self.frames.append({"frame_id": f"f{index}", "frame_index": index, "release_frame_index": index + 1,
                "timestamp_ms": index * 1000, "timestamp_basis": "reconstructed_nominal", "source_path": str(path),
                "source_sha256": digest(path), "image_path": str(path), "image_sha256": digest(path),
                "width": 16, "height": 16, "source_pts": None, "time_base": None})
        video = self.selection / "source" / "video.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"0123456789")
        self.source = {"source_kind": "released_image_sequence", "source_path": str(source_path),
            "source_sha256": "a" * 64, "video_path": str(video), "video_sha256": digest(video),
            "frames": self.frames, "duration_ms": 5000, "expected_video_frames": 5,
            "timestamp_basis": "reconstructed_nominal", "timeline_note": "Nominal playback; original timing unavailable.",
            "ffprobe": {"huge": "not in record metadata"}, "commands": ["not in record metadata"]}
        write(self.selection / "source/source.json", self.source)
        write(self.selection / "run.json", {"schema_version": "smart-frame-selection-run-v1",
              "created_at": "2026-01-01", "config": {"procedure_context": "Simulated surgical repair"}})
        self.selection_data = {"schema_version": "smart-frame-selection-v1", "status": "completed",
            "video_sha256": self.source["video_sha256"], "selected_frame_ids": ["f0", "f2"], "frames": [
                {**self.frames[0], "model_decision": "keep", "effective_decision": "keep", "model_reason": "Overview"},
                {**self.frames[1], "model_decision": "drop", "effective_decision": "drop", "model_reason": "Redundant"},
                {**self.frames[2], "model_decision": "drop", "effective_decision": "keep", "coverage_override": True},
                {**self.frames[3], "model_decision": "unreviewed", "effective_decision": "unreviewed"}]}
        write(self.selection / "selection.json", self.selection_data)
        self.annotation_data = {"annotations": [{**self.frames[0], "visible_observation": "Qwen draft",
            "visibility": "partial", "contextual_claims": [], "uncertainties": ["Instrument uncertain"]}],
            "selection_run": str(self.selection), "verification": {"video_sha256": self.source["video_sha256"]}}
        self.annotation_plan = {"schema_version": "full-video-frame-annotation-v1", "created_at": "2026-01-02",
            "selection_run": str(self.selection), "input_sha256": {"selection.json": digest(self.selection / "selection.json"),
                "source/source.json": digest(self.selection / "source/source.json")}}
        write(self.annotation / "run.json", self.annotation_plan)
        write(self.annotation / "source/source.json", self.source)
        write(self.annotation / "annotations.json", self.annotation_data)
        write(self.annotation / "summary.json", {"status": "completed"})
        self.packet = {"target_frame_id": "f0", "qwen_annotation": self.annotation_data["annotations"][0],
            "frames": [{**self.frames[0], "evidence_roles": ["target"]},
                       {**self.frames[1], "evidence_roles": ["after"]}],
            "dataset_context": {"excluded_context": ["case_outcomes", "surgeon_experience"]}}
        self.judgment = {"target_frame_id": "f0", "status": "needs_more_evidence", "assessment": "uncertain",
            "revised_annotation": {"visible_observation": "Revised draft", "contextual_claims": [],
                                   "visibility": "partial", "uncertainties": ["Target detail needed"]},
            "corrections": [], "evidence_requests": [{"question": "Need closer view", "target": "target_detail"}]}
        self.review_row = {"target_frame_id": "f0", "qwen_annotation": self.annotation_data["annotations"][0],
            "medgemma_review": self.judgment, "evidence": self.packet, "call_directory": "calls/frame-0000/attempt-0000",
            "deferred_evidence_requests": self.judgment["evidence_requests"], "automated_followup": False}
        self.review_plan = {"schema_version": "medgemma-frame-review-v1", "created_at": "2026-01-03",
            "annotation_run": str(self.annotation), "evidence_files": ["evidence/frame-0000.json"],
            "input_sha256": {"qwen/annotations.json": digest(self.annotation / "annotations.json")}}
        write(self.review / "run.json", self.review_plan)
        write(self.review / "qwen/run.json", self.annotation_plan)
        write(self.review / "qwen/source/source.json", self.source)
        write(self.review / "evidence/frame-0000.json", self.packet)
        write(self.review / "reviews.json", {"reviews": [self.review_row]})
        write(self.review / "summary.json", {"status": "completed"})
        write(self.review / "calls/frame-0000/attempt-0000/response.json", {"full": "raw model response"})
        for filename in ("sospine_tool_tips.csv", "sospine_bbox.csv"):
            with (self.dataset / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, ["trial_frame", "x1", "y1", "x2", "y2", "label"])
                writer.writeheader()
                writer.writerow({"trial_frame": "S1A1_frame_00000001.jpeg", "label": "needle driver tip ",
                                 "x1": "1.2300", "y1": "-1", "x2": "1.2300", "y2": "3000"})
                writer.writerow({"trial_frame": "S1A1_frame_00000002.jpeg", "label": ""})
                writer.writerow({"trial_frame": "S1A2_frame_00000001.jpeg", "label": "OTHER CASE"})
        (self.dataset / "sospine_outcomes.csv").write_text(
            "Trial ID,Length,Time for repair,Leak At 40mmHg,Postgraduate year,Prior experience with MISS,Extra\n"
            "S1A1,99:99:99,99999,N,5,Y,all source columns\n"
            "S1A2,01:00:00,11111,Y,2,N,other case\n")

    def store(self):
        store = InspectorStore(self.outputs)
        record_id = store.records()["records"][0]["id"]
        return store, record_id

    def test_v2_frame_bound_annotations_remain_visible_in_the_inspector(self):
        self.annotation_plan["schema_version"] = "full-video-frame-annotation-v2"
        write(self.annotation / "run.json", self.annotation_plan)
        self.annotation_data["schema_version"] = "contextual-frame-annotations-v2"
        self.annotation_data["annotations"][0]["contextual_claims"] = [{"claim": "Source evidence.",
            "evidence_intervals": [{"start_frame_id": "f4", "end_frame_id": "f4",
                "start_ms": 4000, "end_ms": 4000, "supporting_frames": [self.frames[-1]]}]}]
        write(self.annotation / "annotations.json", self.annotation_data)
        store, record_id = self.store()
        detail = store.frame(record_id, "f0")
        self.assertEqual(detail["qwen"], self.annotation_data["annotations"][0])

    def test_full_canonical_timeline_and_protected_drop_semantics(self):
        store, record_id = self.store()
        summary = store.records()["records"]
        self.assertEqual(len(summary), 1, "frozen upstream run snapshots must not appear as records")
        self.assertEqual((summary[0]["frame_count"], summary[0]["duration_ms"]), (5, 5000))
        self.assertEqual((summary[0]["selected_count"], summary[0]["dropped_count"]), (2, 1))
        record = store.record(record_id)
        self.assertEqual([frame["status"] for frame in record["frames"]],
                         ["selected", "dropped", "selected", "candidate", "source"])
        self.assertTrue(record["frames"][2]["coverage_override"])
        self.assertEqual(record["frames"][2]["model_decision"], "drop")
        self.assertNotIn("ffprobe", record["metadata"])
        self.assertNotIn("commands", record["metadata"])

    def test_qwen_revision_evidence_and_complete_response_are_preserved(self):
        store, record_id = self.store()
        summary = store.record(record_id)
        self.assertEqual((summary["qwen_annotation_count"], summary["medgemma_review_count"]), (1, 1))
        frame = store.frame(record_id, "f0")
        self.assertEqual(frame["qwen"]["visible_observation"], "Qwen draft")
        self.assertEqual(frame["medgemma"], self.judgment)
        self.assertEqual(frame["raw"]["medgemma"], self.review_row)
        self.assertEqual([row["roles"] for row in frame["evidence"]], [["target"], ["after"]])
        self.assertFalse(frame["raw"]["medgemma"]["automated_followup"])
        self.assertEqual(frame["artifacts"][0]["label"], "Complete MedGemma response")
        response_path = store.media(frame["artifacts"][0]["url"].rsplit("/", 1)[-1])
        self.assertEqual(json.loads(response_path.read_text()), {"full": "raw model response"})
        self.assertIsNone(store.frame(record_id, "f4")["qwen"])
        self.assertIsNone(store.frame(record_id, "f2")["medgemma"])
        frame["raw"]["medgemma"]["target_frame_id"] = "edited"
        self.assertEqual(store.frame(record_id, "f0")["raw"]["medgemma"]["target_frame_id"], "f0")

    def test_outcomes_are_exact_complete_human_only_and_never_define_timing(self):
        store, record_id = self.store()
        frame = store.frame(record_id, "f0")
        outcomes = frame["outcomes"]
        self.assertEqual(outcomes["status"], "available")
        self.assertEqual(outcomes["raw"]["Postgraduate year"], "5")
        self.assertEqual(outcomes["raw"]["Extra"], "all source columns")
        self.assertEqual(outcomes["source_locator"]["locator"], 2)
        self.assertFalse(outcomes["used_for_model_inference"])
        self.assertEqual(store.record(record_id)["duration_ms"], 5000)
        self.assertNotIn("99999", json.dumps(frame["raw"]))
        self.assertNotIn("99999", json.dumps(frame["dataset_context"]))
        self.assertEqual(len(frame["original_annotations"]), 2)
        self.assertEqual(frame["original_annotations"][0]["raw_value"]["x1"], "1.2300")
        self.assertEqual(frame["original_annotations"][0]["raw_value"]["label"], "needle driver tip ")
        self.assertNotIn("OTHER CASE", json.dumps(frame))
        for frame_id, reason in (("f1", "blank_placeholder_rows"), ("f4", "no_matching_source_rows")):
            availability = store.frame(record_id, frame_id)["label_availability"]
            self.assertTrue(all(row["status"] == "unavailable" and not row["negative_label"] for row in availability))
            self.assertTrue(all(row["reason"] == reason for row in availability))

    def test_outcome_mismatches_and_duplicates_never_join(self):
        store = InspectorStore(self.outputs, self.root / "wrong-dataset")
        row = store.records()["records"][0]
        self.assertEqual(store.record(row["id"])["outcomes"]["status"], "unavailable")
        path = self.dataset / "sospine_outcomes.csv"
        path.write_text(path.read_text() + "S1A1,1,2,Y,5,Y,duplicate\n")
        store, record_id = self.store()
        self.assertIsNone(store.record(record_id)["outcomes"]["raw"])

    def test_other_runs_with_same_video_cannot_supply_annotations(self):
        self.annotation_plan["selection_run"] = str(self.outputs / "unrelated-selection")
        write(self.annotation / "run.json", self.annotation_plan)
        store, record_id = self.store()
        record = store.record(record_id)
        self.assertEqual(record["qwen_annotation_count"], 0)
        self.assertEqual(record["medgemma_review_count"], 0)
        self.assertIsNone(record["runs"]["qwen"])

    def test_changed_video_or_selection_hash_cannot_supply_annotations(self):
        other = copy.deepcopy(self.source)
        other["video_sha256"] = "b" * 64
        write(self.annotation / "source/source.json", other)
        store, record_id = self.store()
        self.assertEqual(store.record(record_id)["qwen_annotation_count"], 0)
        self.assertTrue(store.records()["warnings"])
        write(self.annotation / "source/source.json", self.source)
        self.annotation_plan["input_sha256"]["selection.json"] = "b" * 64
        write(self.annotation / "run.json", self.annotation_plan)
        store, record_id = self.store()
        self.assertEqual(store.record(record_id)["qwen_annotation_count"], 0)

    def test_medgemma_requires_exact_qwen_parent_and_evidence(self):
        self.review_row["evidence"]["frames"][1]["timestamp_ms"] = 4444
        write(self.review / "reviews.json", {"reviews": [self.review_row]})
        store, record_id = self.store()
        self.assertEqual(store.record(record_id)["medgemma_review_count"], 0)
        self.assertIsNone(store.frame(record_id, "f0")["medgemma"])
        self.review_plan["annotation_run"] = str(self.outputs / "different-annotation")
        write(self.review / "run.json", self.review_plan)
        store, record_id = self.store()
        self.assertIsNone(store.record(record_id)["runs"]["medgemma"])

    def test_prepared_and_partial_runs_are_visible_without_invented_outputs(self):
        (self.selection / "selection.json").unlink()
        (self.annotation / "annotations.json").unlink()
        store, record_id = self.store()
        record = store.record(record_id)
        self.assertEqual(record["frame_count"], 5)
        self.assertEqual(record["selected_count"], 0)
        self.assertTrue(all(frame["status"] == "source" for frame in record["frames"]))
        self.assertIsNone(store.frame(record_id, "f0")["medgemma"])
        (self.selection / "source/source.json").write_text("{")
        self.assertEqual(store.records()["records"], [])
        self.assertTrue(store.records()["warnings"])

    def test_refresh_discovers_new_results_and_missing_media_is_explicit(self):
        (self.review / "reviews.json").unlink()
        store, record_id = self.store()
        self.assertIsNone(store.frame(record_id, "f0")["medgemma"])
        self.assertEqual(len(store.frame(record_id, "f0")["evidence"]), 2)
        write(self.review / "reviews.json", {"reviews": [self.review_row]})
        self.assertEqual(store.records()["records"][0]["medgemma_review_count"], 1)
        Path(self.frames[4]["image_path"]).unlink()
        store.records()
        self.assertIsNone(store.frame(record_id, "f4")["frame"]["image_url"])

    def test_review_identity_is_stable_and_changes_with_displayed_revisions(self):
        store, record_id = self.store()
        original = store.record(record_id)["review_identity"]
        store.records()
        self.assertEqual(store.record(record_id)["review_identity"], original)
        self.annotation_data["annotations"][0]["visible_observation"] = "Updated Qwen draft"
        write(self.annotation / "annotations.json", self.annotation_data)
        store.records()
        updated = store.record(record_id)["review_identity"]
        self.assertNotEqual(original, updated)
        self.assertEqual(store.frame(record_id, "f0")["qwen"]["visible_observation"], "Updated Qwen draft")
        store.records()
        self.assertEqual(store.record(record_id)["review_identity"], updated)
        # Identical canonical source media in another selection run still has
        # separate human-review identity, even without model output attached.
        other = self.outputs / "selection" / "separate-run"
        write(other / "run.json", {"schema_version": "smart-frame-selection-run-v1"})
        write(other / "source/source.json", self.source)
        write(other / "selection.json", self.selection_data)
        records = store.records()["records"]
        identities = {store.record(record["id"])["review_identity"] for record in records}
        self.assertEqual(len(identities), 2)

    def test_pending_candidates_are_neutral_and_original_video_links_are_registered(self):
        (self.selection / "selection.json").unlink()
        write(self.selection / "state.json", {"status": "awaiting_review", "candidate_ids": ["f0", "f2"]})
        linked_video = self.selection / "source" / "original-video.mp4"
        linked_video.symlink_to(self.source["video_path"])
        self.source["video_path"] = str(linked_video)
        write(self.selection / "source/source.json", self.source)
        store, record_id = self.store()
        record = store.record(record_id)
        self.assertEqual(record["status"], "awaiting_review")
        self.assertEqual([frame["status"] for frame in record["frames"]],
                         ["candidate", "source", "candidate", "source", "source"])
        self.assertEqual(record["selected_count"], 0)
        self.assertIsNotNone(record["video_url"])

    def test_curation_edit_delete_restore_is_durable_and_keeps_original_evidence(self):
        store, record_id = self.store()
        identity = store.record(record_id)["review_identity"]
        original = {path: digest(path) for path in self.root.rglob("*") if path.is_file()}
        self.assertIsNone(store.frame(record_id, "f0")["curation"])
        self.assertTrue(all(frame["curation"] is None for frame in store.record(record_id)["frames"]))

        result = store.curate(record_id, "f0", {"review_identity": identity, "action": "edit",
                                               "annotations": {"qwen": "Human-corrected draft", "medgemma": ""}})
        self.assertEqual(result["annotations"], {"qwen": "Human-corrected draft", "medgemma": ""})
        self.assertFalse(result["deleted"])
        self.assertEqual(result["worksheet_status"], "draft")
        self.assertFalse(result["training_eligible"])
        self.assertEqual(result["provenance"]["source_frame"], self.frames[0])
        self.assertEqual(result["provenance"]["original_annotations"]["qwen"], self.annotation_data["annotations"][0])
        self.assertEqual(result["provenance"]["original_annotations"]["medgemma"], self.judgment)
        result["annotations"]["qwen"] = "Caller cannot modify stored value"
        restarted = InspectorStore(self.outputs)
        frame = restarted.frame(record_id, "f0")
        self.assertEqual(frame["curation"]["annotations"]["qwen"], "Human-corrected draft")
        self.assertEqual(frame["frame"]["curation"], frame["curation"])
        self.assertEqual(frame["qwen"]["visible_observation"], "Qwen draft")
        self.assertEqual(frame["medgemma"], self.judgment)

        deleted = restarted.curate(record_id, "f0", {"review_identity": identity, "action": "delete"})
        self.assertTrue(deleted["deleted"])
        self.assertEqual(restarted.record(record_id)["frames"][0]["status"], "selected")
        self.assertEqual(restarted.record(record_id)["frames"][0]["curation"], deleted)
        restored = store.curate(record_id, "f0", {"review_identity": identity, "action": "restore"})
        self.assertFalse(restored["deleted"])
        self.assertEqual(restored["annotations"], deleted["annotations"])
        self.assertEqual([row["action"] for row in restored["history"]], ["edit", "delete", "restore"])
        self.assertEqual(store.record(record_id)["review_identity"], identity)
        self.assertIsNone(store.frame(record_id, "f1")["curation"])
        self.assertEqual({path: digest(path) for path in original}, original)

    def test_curation_rejects_stale_lineage_and_does_not_migrate_to_new_revisions(self):
        store, record_id = self.store()
        identity = store.record(record_id)["review_identity"]
        request = {"review_identity": identity, "action": "edit", "annotations": {"qwen": "Human draft"}}
        saved = store.curate(record_id, "f0", request)
        outcome_path = self.dataset / "sospine_outcomes.csv"
        original_outcomes = outcome_path.read_text()
        outcome_path.write_text(original_outcomes.replace("all source columns", "changed outcome"))
        # No explicit refresh: accepting a POST must itself detect changed data.
        with self.assertRaises(CurationError) as rejected:
            store.curate(record_id, "f0", request)
        self.assertEqual(rejected.exception.status, 409)
        updated_identity = store.record(record_id)["review_identity"]
        self.assertNotEqual(identity, updated_identity)
        self.assertIsNone(store.frame(record_id, "f0")["curation"])
        newer = store.curate(record_id, "f0", {**request, "review_identity": updated_identity,
                                               "annotations": {"qwen": "New revision draft"}})
        self.assertNotEqual(saved["review_identity"], newer["review_identity"])
        outcome_path.write_text(original_outcomes)
        store.records()
        self.assertEqual(store.frame(record_id, "f0")["curation"], saved)

        other = self.outputs / "selection" / "separate-run"
        write(other / "run.json", {"schema_version": "smart-frame-selection-run-v1"})
        write(other / "source/source.json", self.source)
        write(other / "selection.json", self.selection_data)
        store.records()
        other_id = next(row["id"] for row in store.records()["records"] if row["id"] != record_id)
        self.assertIsNone(store.frame(other_id, "f0")["curation"])
        self.assertEqual(store.frame(record_id, "f0")["curation"], saved)

    def test_curation_validates_actions_sources_and_saved_files_before_writing(self):
        store, record_id = self.store()
        identity = store.record(record_id)["review_identity"]
        base = {"review_identity": identity, "action": "edit", "annotations": {"qwen": "Human draft"}}
        for invalid in (None, [], {}, {**base, "unexpected": True}, {**base, "review_identity": 1},
                        {**base, "action": []}, {**base, "action": "erase"}, {**base, "annotations": {}},
                        {**base, "annotations": {"other": "text"}}, {**base, "annotations": {"qwen": 1}},
                        {**base, "annotations": {"qwen": "a" * 100001}}, {**base, "action": "delete"}):
            with self.subTest(invalid=str(invalid)[:100]):
                with self.assertRaises(CurationError) as rejected:
                    store.curate(record_id, "f0", invalid)
                self.assertEqual(rejected.exception.status, 400)
        self.assertFalse(store.curation_root.exists())
        for frame_id in ("f1", "f2", "f4"):
            with self.assertRaises(CurationError) as rejected:
                store.curate(record_id, frame_id, base)
            self.assertEqual(rejected.exception.status, 409)
        with self.assertRaises(CurationError) as rejected:
            store.curate(record_id, "missing", base)
        self.assertEqual(rejected.exception.status, 404)
        (self.review / "reviews.json").unlink()
        store.records()
        identity = store.record(record_id)["review_identity"]
        with self.assertRaises(CurationError) as rejected:
            store.curate(record_id, "f0", {**base, "review_identity": identity, "annotations": {"medgemma": "not present"}})
        self.assertEqual(rejected.exception.status, 409)
        deleted = store.curate(record_id, "f0", {"review_identity": identity, "action": "delete"})
        with self.assertRaisesRegex(CurationError, "Restore"):
            store.curate(record_id, "f0", {**base, "review_identity": identity})
        self.assertEqual(store.frame(record_id, "f0")["curation"], deleted)
        path = next(store.curation_root.rglob("*.json"))
        path.write_text("invalid saved data")
        with self.assertRaisesRegex(CurationError, "not overwritten"):
            store.curate(record_id, "f0", {"review_identity": identity, "action": "restore"})
        self.assertEqual(path.read_text(), "invalid saved data")

    def test_curation_failed_commit_preserves_previous_value_and_removes_temporary_file(self):
        store, record_id = self.store()
        identity = store.record(record_id)["review_identity"]
        request = {"review_identity": identity, "action": "edit", "annotations": {"qwen": "Saved draft"}}
        saved = store.curate(record_id, "f0", request)
        path = next(store.curation_root.rglob("*.json"))
        before = path.read_bytes()
        with patch("yasargil.dataset_inspector.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                store.curate(record_id, "f0", {**request, "annotations": {"qwen": "Failed draft"}})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(store.frame(record_id, "f0")["curation"], saved)
        self.assertEqual(list(path.parent.glob(".curation-*")), [])

    def test_curation_conditional_updates_reject_stale_tabs_without_mutating_history(self):
        store, record_id = self.store()
        identity = store.record(record_id)["review_identity"]
        request = {"review_identity": identity, "action": "edit", "annotations": {"qwen": "First draft"},
                   "expected_updated_at": None}
        for invalid in (False, 1, [], {}):
            with self.assertRaises(CurationError) as rejected:
                store.curate(record_id, "f0", {**request, "expected_updated_at": invalid})
            self.assertEqual(rejected.exception.status, 400)
        self.assertFalse(store.curation_root.exists())
        first = store.curate(record_id, "f0", request)
        other = InspectorStore(self.outputs)
        for action in ("edit", "delete", "restore"):
            stale = {"review_identity": identity, "action": action, "expected_updated_at": None}
            if action == "edit":
                stale["annotations"] = {"qwen": "Stale draft"}
            with self.assertRaisesRegex(CurationError, "another window") as rejected:
                other.curate(record_id, "f0", stale)
            self.assertEqual(rejected.exception.status, 409)
            self.assertEqual(store.frame(record_id, "f0")["curation"], first)
        updated = other.curate(record_id, "f0", {**request, "expected_updated_at": first["updated_at"],
                                                 "annotations": {"qwen": "Current draft"}})
        with self.assertRaises(CurationError):
            store.curate(record_id, "f0", {**request, "expected_updated_at": first["updated_at"]})
        self.assertEqual(store.frame(record_id, "f0")["curation"], updated)
        self.assertEqual(len(updated["history"]), 2)
        deleted = store.curate(record_id, "f0", {"review_identity": identity, "action": "delete",
                                                 "expected_updated_at": updated["updated_at"]})
        restored = other.curate(record_id, "f0", {"review_identity": identity, "action": "restore",
                                                  "expected_updated_at": deleted["updated_at"]})
        self.assertFalse(restored["deleted"])
        self.assertEqual(restored["annotations"], updated["annotations"])
        self.assertEqual(store.record(record_id)["review_identity"], identity)

    def test_concurrent_inspectors_merge_annotation_fields_under_file_lock(self):
        store, record_id = self.store()
        other = InspectorStore(self.outputs)
        identity = store.record(record_id)["review_identity"]
        barrier = threading.Barrier(2)
        errors = []

        def update(instance, name):
            try:
                barrier.wait(timeout=3)
                instance.curate(record_id, "f0", {"review_identity": identity, "action": "edit",
                                                   "annotations": {name: name + " human draft"}})
            except Exception as exc:
                errors.append(exc)

        workers = [threading.Thread(target=update, args=pair) for pair in ((store, "qwen"), (other, "medgemma"))]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        result = store.frame(record_id, "f0")["curation"]
        self.assertEqual(result["annotations"], {"qwen": "qwen human draft", "medgemma": "medgemma human draft"})
        self.assertEqual(len(result["history"]), 2)

    def test_curation_storage_symlinks_are_never_followed(self):
        store, record_id = self.store()
        identity = store.record(record_id)["review_identity"]
        outside = self.root / "outside"
        outside.mkdir()
        store.curation_root.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(CurationError):
            store.curate(record_id, "f0", {"review_identity": identity, "action": "delete"})
        self.assertEqual(list(outside.iterdir()), [])
        self.assertIsNone(store.frame(record_id, "f0")["curation"])

    def test_http_curation_endpoint_enforces_same_origin_and_strict_request_contract(self):
        store, record_id = self.store()
        server = make_server(store, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        identity = store.record(record_id)["review_identity"]
        request = {"review_identity": identity, "action": "edit", "annotations": {"qwen": "Human edit"}}
        route = f"/api/records/{record_id}/frames/f0/curation"

        def submit(value=request, *, raw=None, headers=None, path=route):
            payload = raw if raw is not None else json.dumps(value).encode()
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            connection.request("POST", path, body=payload, headers={"Content-Type": "application/json", **(headers or {})})
            response = connection.getresponse()
            result = response.status, json.loads(response.read())
            connection.close()
            return result

        status, saved = submit()
        self.assertEqual(status, 200)
        self.assertEqual(saved["annotations"], {"qwen": "Human edit"})
        self.assertEqual(InspectorStore(self.outputs).frame(record_id, "f0")["curation"], saved)
        for raw in (b"NaN", b"[]", b'{"action":"edit","action":"delete"}', b"\xff",
                    b'{"annotations":{"qwen":"\\ud800"}}'):
            self.assertEqual(submit(raw=raw)[0], 400)
        self.assertEqual(submit({**request, "review_identity": "b" * 64})[0], 409)
        self.assertEqual(submit(path=f"/api/records/{record_id}/frames/missing/curation")[0], 404)
        self.assertEqual(submit(headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(submit(headers={"Content-Length": str(2 * 1024 * 1024 + 1)})[0], 413)
        self.assertEqual(submit(headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(submit(headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(submit(headers={"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(store.frame(record_id, "f0")["curation"], saved)

    def test_original_video_does_not_join_sospine_metadata(self):
        self.source["source_kind"] = "original_video"
        write(self.selection / "source/source.json", self.source)
        store, record_id = self.store()
        record = store.record(record_id)
        self.assertEqual(record["outcomes"]["status"], "unavailable")
        self.assertEqual(store.frame(record_id, "f0")["original_annotations"], [])

    def test_http_registered_media_ranges_and_path_protection(self):
        store, record_id = self.store()
        server = make_server(store, 0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        def request(path, method="GET", headers=None):
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            connection.request(method, path, headers=headers or {})
            result = connection.getresponse()
            payload = result.read()
            output = (result.status, dict(result.getheaders()), payload)
            connection.close()
            return output

        self.assertEqual(request("/api/records")[0], 200)
        status, headers, body = request("/style.css")
        self.assertEqual(status, 200)
        self.assertIn("text/css", headers["Content-Type"])
        self.assertEqual(request(f"/api/records/{record_id}/frames/f0")[0], 200)
        self.assertEqual(request(f"/api/records/{record_id}/frames/not-real")[0], 404)
        video = store.record(record_id)["source_video_url"]
        status, headers, body = request(video, headers={"Range": "bytes=2-5"})
        self.assertEqual((status, body, headers["Content-Range"]), (206, b"2345", "bytes 2-5/10"))
        self.assertEqual(request(video, headers={"Range": "bytes=-3"})[2], b"789")
        self.assertEqual(request(video, headers={"Range": "bytes=8-"})[2], b"89")
        self.assertEqual(request(video, "HEAD")[2], b"")
        for invalid in ("bytes=11-", "bytes=5-2", "bytes=-0", "bytes=0-1,4-5", "invalid"):
            self.assertEqual(request(video, headers={"Range": invalid})[0], 416)
        for invalid in ("/media/../../etc/passwd", "/media/%2e%2e/%2e%2e/etc/passwd", "/media/" + "f" * 32,
                        "/source/source.json", "/api/records/../../etc/passwd", "/index.html/../run.json",
                        "/fonts.css", "/fonts/MesloLGS-NF-Regular.ttf", "/fonts/MesloLGS-NF-Bold.ttf",
                        "/fonts/not-registered.ttf", "/fonts/%2e%2e/app.js"):
            self.assertEqual(request(invalid)[0], 404)
        self.assertEqual(request("/api/records", method="POST")[0], 405)
        self.assertEqual(request("/api/records", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(request("/api/records", headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
        preview = store.record(record_id)["video_url"]
        store.preview_cache = self.root / "preview-cache"
        self.assertFalse(store.preview_cache.exists())
        with patch("yasargil.dataset_inspector.shutil.which", return_value=None):
            status, _, body = request(preview)
        self.assertEqual(status, 503)
        self.assertIn("ffmpeg", json.loads(body)["error"])
        self.assertFalse(store.preview_cache.exists())
        token = video.rsplit("/", 1)[-1]
        target = store.media(token)
        target.unlink()
        target.symlink_to(self.dataset / "sospine_outcomes.csv")
        self.assertEqual(request(video)[0], 404)

    def test_http_home_page_allows_external_navigation_without_exposing_dataset_routes(self):
        store, record_id = self.store()
        server = make_server(store, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        navigation = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate",
                      "Sec-Fetch-Dest": "document", "Sec-Fetch-User": "?1"}

        def request(path="/", method="GET", headers=None):
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            connection.request(method, path, headers={**navigation, **(headers or {})})
            response = connection.getresponse()
            result = response.status, dict(response.getheaders()), response.read()
            connection.close()
            return result

        # Opening the GUI from a link in another app/site is a valid navigation.
        for origin in (None, "https://elsewhere.example", "null"):
            for method in ("GET", "HEAD"):
                with self.subTest(origin=origin, method=method):
                    status, headers, body = request(method=method, headers={"Origin": origin} if origin else {})
                    self.assertEqual(status, 200)
                    self.assertIn("text/html", headers["Content-Type"])
                    self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
                    self.assertNotIn("Access-Control-Allow-Origin", headers)
                    self.assertIn("Sec-Fetch-Mode", headers["Vary"])
                    if method == "GET":
                        self.assertIn(b"Yasargil", body)
                        self.assertNotIn(b"Qwen draft", body)
                    else:
                        self.assertEqual(body, b"")

        # The exception is only for the home document, never data or actions.
        protected = ["/api/records", f"/api/records/{record_id}",
                     f"/api/records/{record_id}/frames/f0",
                     f"/api/records/{record_id}/frames/f0/context-video",
                     store.record(record_id)["source_video_url"],
                     store.record(record_id)["video_url"]]
        for path in protected:
            for method in ("GET", "HEAD"):
                with self.subTest(path=path, method=method):
                    self.assertEqual(request(path, method)[0], 403)
        for path in ("/", "/api/open-dataset-folder", "/api/review-export",
                     f"/api/records/{record_id}/frames/f0/curation"):
            with self.subTest(post=path):
                self.assertEqual(request(path, "POST")[0], 403)

        for headers in ({"Host": "elsewhere.example"}, {"Sec-Fetch-Mode": "cors"},
                        {"Sec-Fetch-Mode": "no-cors"}, {"Sec-Fetch-Dest": "iframe"},
                        {"Sec-Fetch-Dest": "object"}, {"Sec-Fetch-Dest": "embed"},
                        {"Sec-Fetch-Mode": "", "Sec-Fetch-Dest": ""}):
            with self.subTest(blocked=headers):
                self.assertEqual(request(headers=headers)[0], 403)

        # Once loaded, the document can fetch its assets and dataset as usual.
        own_origin = {"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors",
                      "Sec-Fetch-Dest": "empty", "Origin": f"http://127.0.0.1:{server.server_port}"}
        for path in ("/app.js", "/style.css", "/api/records"):
            with self.subTest(same_origin=path):
                self.assertEqual(request(path, headers=own_origin)[0], 200)

    def test_preview_source_hash_is_verified_before_any_encoding(self):
        store, record_id = self.store()
        store.preview_cache = self.root / "preview-cache"
        token = store.record(record_id)["video_url"].rsplit("/", 1)[-1]
        Path(self.source["video_path"]).write_bytes(b"changed source")
        with patch("yasargil.dataset_inspector.subprocess.run") as process:
            with self.assertRaisesRegex(PreviewError, "hash"):
                store.preview(token)
            process.assert_not_called()
        self.assertFalse(store.preview_cache.exists())
        self.assertIsNone(store.preview("not-registered"))

    def test_review_export_is_a_validated_stateless_attachment(self):
        store, _ = self.store()
        server = make_server(store, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        before = {path: digest(path) for path in self.root.rglob("*") if path.is_file()}

        def submit(value, *, raw=None, headers=None, path="/api/review-export"):
            payload = raw if raw is not None else urlencode({"review": json.dumps(value)}).encode()
            request_headers = {"Content-Type": "application/x-www-form-urlencoded", **(headers or {})}
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            connection.request("POST", path, body=payload, headers=request_headers)
            response = connection.getresponse()
            result = response.status, dict(response.getheaders()), response.read()
            connection.close()
            return result

        worksheet = {"schema_version": "yasargil-inspector-review-v1", "case_id": "../../S1A1\r\nunsafe",
                     "notes": [{"frame_id": "f0", "note": "Review this frame", "status": "flagged"}],
                     "worksheet_status": "approved", "training_eligible": True}
        status, headers, body = submit(worksheet)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Content-Disposition"], 'attachment; filename="S1A1-unsafe-human-review.json"')
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        result = json.loads(body)
        self.assertEqual(result["notes"], worksheet["notes"])
        self.assertEqual(result["worksheet_status"], "draft")
        self.assertFalse(result["training_eligible"])
        for invalid in ({}, [], {"schema_version": "wrong", "notes": []},
                        {"schema_version": "yasargil-inspector-review-v1", "notes": "wrong"}):
            self.assertEqual(submit(invalid)[0], 400)
        self.assertEqual(submit(None, raw=b"review=NaN")[0], 400)
        self.assertEqual(submit(None, raw=urlencode({"review": '{"notes":[],"notes":[]}'}).encode())[0], 400)
        self.assertEqual(submit(None, raw=b"review=%FF")[0], 400)
        self.assertEqual(submit(worksheet, headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(submit(worksheet, headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(submit(worksheet, headers={"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(submit(worksheet, headers={"Content-Length": str(2 * 1024 * 1024 + 1)})[0], 413)
        self.assertEqual(submit(worksheet, headers={"Content-Type": "application/json"})[0], 415)
        self.assertEqual(submit(worksheet, path="/api/records")[0], 405)
        self.assertEqual({path: digest(path) for path in self.root.rglob("*") if path.is_file()}, before)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Preview encoding requires ffmpeg and ffprobe")
    def test_lazy_browser_preview_preserves_frame_timing_and_uses_yuv420(self):
        source_path = Path(self.source["video_path"])
        subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=16x16:rate=1", "-c:v", "libx264rgb",
            "-crf", "0", "-threads", "2", str(source_path)], check=True, capture_output=True, timeout=30)
        self.source["video_sha256"] = digest(source_path)
        self.selection_data["video_sha256"] = self.source["video_sha256"]
        write(self.selection / "source/source.json", self.source)
        write(self.selection / "selection.json", self.selection_data)
        original = source_path.read_bytes()
        store, record_id = self.store()
        store.preview_cache = self.root / "preview-cache"
        record = store.record(record_id)
        self.assertTrue(record["video_url"].startswith("/preview/"))
        self.assertTrue(record["source_video_url"].startswith("/media/"))
        self.assertTrue(record["metadata"]["browser_preview"]["display_only"])
        self.assertFalse(store.preview_cache.exists(), "Discovery and record viewing must not transcode")
        token = record["video_url"].rsplit("/", 1)[-1]
        preview = store.preview(token)
        self.assertEqual(preview.name, self.source["video_sha256"] + ".mp4")
        self.assertEqual(source_path.read_bytes(), original)
        receipt = json.loads(preview.with_suffix(".json").read_text())
        self.assertEqual(receipt["verification"], {
            "frame_count": 5, "duration_ms": 5000, "codec": "h264", "pixel_format": "yuv420p"})
        self.assertEqual(receipt["input"]["timestamps_ms"], [0, 1000, 2000, 3000, 4000])
        self.assertEqual(receipt["preview_sha256"], digest(preview))
        self.assertEqual(receipt["input"]["source_video_sha256"], digest(source_path))
        with patch("yasargil.dataset_inspector.subprocess.run") as process:
            self.assertEqual(store.preview(token), preview)
            process.assert_not_called()
        self.assertFalse(list(store.preview_cache.glob(".*.mp4")))
        server = make_server(store, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request("GET", record["video_url"], headers={"Range": "bytes=0-15"})
        response = connection.getresponse()
        self.assertEqual(response.status, 206)
        self.assertEqual(response.getheader("Content-Type"), "video/mp4")
        self.assertEqual(response.read(), preview.read_bytes()[:16])
        connection.close()

    def test_dataset_folder_action_opens_only_configured_outputs_and_requires_same_origin(self):
        store, _ = self.store()
        server = make_server(store, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        def request(method="POST", path="/api/open-dataset-folder", body="", headers=None):
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            result = response.status, json.loads(response.read())
            connection.close()
            return result
        with patch("yasargil.dataset_inspector.sys.platform", "darwin"), patch("yasargil.dataset_inspector.subprocess.run") as opener:
            status, result = request()
            self.assertEqual(status, 200)
            self.assertEqual(result, {"opened": True, "path": str(self.outputs)})
            self.assertEqual(opener.call_args.args[0], ["open", str(self.outputs)])
            opener.reset_mock()
            self.assertEqual(request(headers={"Origin": "https://elsewhere.example"})[0], 403)
            self.assertEqual(request(path="/api/open-dataset-folder?path=/tmp")[0], 400)
            self.assertEqual(request(body='{"path":"/tmp"}')[0], 400)
            self.assertEqual(request(method="GET")[0], 404)
            opener.assert_not_called()
            opener.side_effect = OSError("Desktop unavailable")
            self.assertEqual(request()[0], 503)

    def _prepare_context_images(self, count=13, fps=1):
        images = self.root / "canonical-images"
        images.mkdir()
        subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=32x24:rate={fps}", "-frames:v", str(count),
            "-threads", "2", "-start_number", "0", str(images / "frame-%08d.png")],
            check=True, capture_output=True, timeout=30)
        self.frames = []
        for i in range(count):
            path = images / f"frame-{i:08d}.png"
            self.frames.append({"frame_id": f"f{i}", "frame_index": i, "release_frame_index": i + 1,
                "timestamp_ms": i * 1000 / fps, "timestamp_basis": "reconstructed_nominal",
                "source_path": str(path), "source_sha256": digest(path),
                "image_path": str(path), "image_sha256": digest(path), "width": 32, "height": 24,
                "source_pts": None, "time_base": None})
        self.source.update(frames=self.frames, expected_video_frames=count, duration_ms=count * 1000 / fps)
        self.selection_data["frames"] = [{**self.frames[0], "model_decision": "keep", "effective_decision": "keep"}]
        write(self.selection / "source/source.json", self.source)
        write(self.selection / "selection.json", self.selection_data)
        store, record_id = self.store()
        store.preview_cache = self.root / "context-cache"
        return store, record_id

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Context video requires ffmpeg and ffprobe")
    def test_context_video_uses_exact_nine_images_without_reading_full_video(self):
        store, record_id = self._prepare_context_images()
        original_images = [Path(frame["image_path"]).read_bytes() for frame in self.frames]
        with patch.object(store, "_verified_hash", wraps=store._verified_hash) as hashes:
            video = store.context_preview(record_id, "f6")
            self.assertNotIn(Path(self.source["video_path"]), [call.args[0] for call in hashes.call_args_list])
        receipt = json.loads(video.with_suffix(".json").read_text())
        self.assertEqual(receipt["input"]["image_sha256"], [frame["image_sha256"] for frame in self.frames[2:11]])
        self.assertEqual(receipt["input"]["timestamps_ms"], [i * 1000 for i in range(9)])
        self.assertEqual(receipt["verification"], {"frame_count": 9, "duration_ms": 9000, "codec": "h264", "pixel_format": "yuv420p"})
        with patch("yasargil.dataset_inspector.subprocess.run") as encoder:
            self.assertEqual(store.context_preview(record_id, "f6"), video)
            encoder.assert_not_called()
        self.assertEqual(original_images, [Path(frame["image_path"]).read_bytes() for frame in self.frames])
        self.assertFalse(list(store.preview_cache.glob(".context-*")))
        for target in ["f0", "f12"]:
            boundary = store.context_preview(record_id, target)
            self.assertEqual(json.loads(boundary.with_suffix(".json").read_text())["verification"]["frame_count"], 5)
        self.assertIsNone(store.context_preview(record_id, "missing"))
        with self.assertRaisesRegex(PreviewError, "dataset changed"):
            store.context_preview(record_id, "f6", "stale-revision")
        Path(self.frames[6]["image_path"]).write_bytes(b"changed source image")
        with self.assertRaisesRegex(PreviewError, "saved provenance"):
            store.context_preview(record_id, "f6")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Context video requires ffmpeg and ffprobe")
    def test_context_video_preserves_native_cadence_and_is_served_as_video(self):
        store, record_id = self._prepare_context_images(fps=30)
        server = make_server(store, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        record = store.record(record_id)
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        url = f"/api/records/{record_id}/frames/f6/context-video?revision={record['review_identity']}"
        connection.request("GET", url, headers={"Range": "bytes=0-15"})
        response = connection.getresponse()
        self.assertEqual(response.status, 206)
        self.assertEqual(response.getheader("Content-Type"), "video/mp4")
        self.assertEqual(len(response.read()), 16)
        connection.close()
        video = store.context_preview(record_id, "f6")
        receipt = json.loads(video.with_suffix(".json").read_text())
        self.assertEqual(receipt["verification"]["frame_count"], 9)
        self.assertAlmostEqual(receipt["verification"]["duration_ms"], 300, delta=1.1)
        with patch("yasargil.dataset_inspector.subprocess.run") as encoder:
            store._records[record_id]["public"]["frames"][6]["timestamp_ms"] += 7
            with self.assertRaisesRegex(PreviewError, "irregular timing"):
                store.context_preview(record_id, "f6")
            encoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
