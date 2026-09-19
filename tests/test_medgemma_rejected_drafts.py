"""Explicit review of failed Qwen citations preserves the rejected source evidence."""
import base64
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import test_medgemma_review as fixtures
from test_medgemma_surgery_review import JointOllama
from test_qwen_draft_intake import make_rejected_fixture
from yasargil.contract import ContractError, sha256_file
from yasargil.dataset_inspector import InspectorStore
from yasargil.medgemma_surgery_review import SurgeryReviewConfig, run_review


read, write = fixtures.read, fixtures.write


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Native provenance fixtures need FFmpeg")
class MedGemmaRejectedDraftTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MedGemmaReviewTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.parent = self.root / "failed-annotation"
        self.output = self.root / "flagged-review"
        self.selected_ids = self.fixture.selected_ids
        self.config = SurgeryReviewConfig(allow_rejected_temporal_citations=True)

    def rejected_annotation(self, *, beyond_duration=False):
        self.parent = self.fixture.annotation
        make_rejected_fixture(self.parent, schema_rejected=beyond_duration)
        config = read(self.parent / "run.json")["config"]
        raw = read(self.parent / "round-00/response.json")
        output = json.loads(raw["choices"][0]["message"]["content"])
        duration = self.fixture.source["duration_ms"]
        violations = [(frame_id, interval)
            for frame_id, row in output["annotations"].items()
            for claim in row["contextual_claims"]
            for interval in claim["evidence_intervals"]
            if interval["end_ms"] > duration
            or interval["end_ms"] - interval["start_ms"] > config["max_evidence_span_ms"]]
        self.assertTrue(violations)
        self.bad_target, self.bounds = violations[0]
        self.assertEqual(read(self.parent / "summary.json")["status"], "failed")
        self.assertFalse((self.parent / "annotations.json").exists())
        self.parent_bytes = {path.relative_to(self.parent).as_posix(): path.read_bytes()
                             for path in self.parent.rglob("*") if path.is_file()}

    def review(self, client, *, resume=False, config=None, **kwargs):
        with patch("yasargil.frame_annotation.LocalVideoRuntime", side_effect=AssertionError("No new Qwen inference")):
            return run_review(None if resume else self.parent, self.output,
                self.config if config is None else config, dataset_root=self.fixture.dataset,
                resume=resume, client=client, progress=lambda _: None, **kwargs)

    def assert_original_unchanged(self):
        self.assertEqual(self.parent_bytes, {
            path.relative_to(self.parent).as_posix(): path.read_bytes()
            for path in self.parent.rglob("*") if path.is_file()})
        self.assertEqual(read(self.parent / "summary.json")["status"], "failed")
        self.assertFalse((self.parent / "annotations.json").exists())
        self.assertFalse((self.parent / "integrity.json").exists())

    def test_failed_qwen_input_still_requires_explicit_opt_in(self):
        self.rejected_annotation()
        client = JointOllama()
        with self.assertRaises(ContractError):
            self.review(client, config=SurgeryReviewConfig())
        self.assertEqual(client.info_calls, [])
        self.assertEqual(client.requests, [])
        self.assertFalse(self.output.exists())
        self.assert_original_unchanged()

    def test_explicit_review_is_one_call_with_all_selected_images_and_visible_validation_warning(self):
        self.rejected_annotation()
        client = JointOllama()
        summary = self.review(client)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["selected_frame_count"], len(self.selected_ids))
        self.assertEqual(summary["reviewed_frame_count"], len(self.selected_ids))
        self.assertEqual(len(client.requests), 1)
        request = client.requests[0]
        context = json.loads(request["messages"][-1]["content"])
        self.assertEqual(context["qwen_validation"]["status"], "rejected_temporal_citations")
        self.assertTrue(context["qwen_validation"]["issues"])
        self.assertIn("failed", request["messages"][0]["content"])
        image_messages = [row for row in request["messages"] if row.get("images")]
        self.assertEqual(len(image_messages), len(self.selected_ids))
        source = {f["frame_id"]: f for f in self.fixture.source["frames"]}
        for frame_id, message in zip(self.selected_ids, image_messages):
            locator = json.loads(message["content"].split(": ", 1)[1])
            self.assertEqual(locator["frame_id"], frame_id)
            self.assertEqual(len(message["images"]), 1)
            self.assertEqual(base64.b64decode(message["images"][0]), Path(source[frame_id]["image_path"]).read_bytes())
        self.assertEqual([row["frame_id"] for row in context["qwen_annotations"]], self.selected_ids)
        drafts = {row["frame_id"]: row for row in context["qwen_annotations"]}
        interval = drafts[self.bad_target]["contextual_claims"][0]["evidence_intervals"][0]
        self.assertEqual({key: interval[key] for key in self.bounds}, self.bounds)
        self.assert_original_unchanged()

    def test_beyond_duration_raw_citation_remains_unclipped_with_no_invented_frames(self):
        self.rejected_annotation(beyond_duration=True)
        self.assertFalse((self.parent / "round-00/result.json").exists())
        self.assertFalse((self.parent / "round-00/output.json").exists())
        client = JointOllama()
        self.review(client)
        context = json.loads(client.requests[0]["messages"][-1]["content"])
        self.assertEqual(context["media_timeline"]["duration_ms"], 8000)
        self.assertEqual(sum(len(m.get("images", [])) for m in client.requests[0]["messages"]), 4)
        batch = read(self.output / "surgery-evidence.json")
        self.assertEqual(batch["target_frame_ids"], self.selected_ids)
        draft = next(row for row in batch["qwen_annotations"] if row["frame_id"] == self.bad_target)
        interval = draft["contextual_claims"][0]["evidence_intervals"][0]
        self.assertEqual(interval["end_ms"], self.bounds["end_ms"], "Invalid endpoint must remain the original Qwen text")
        self.assertEqual(interval["start_ms"], self.bounds["start_ms"])
        self.assertGreater(interval["end_ms"], batch["media_timeline"]["duration_ms"])
        actual = [frame for frame in self.fixture.source["frames"]
                  if self.bounds["start_ms"] <= frame["timestamp_ms"] <= self.bounds["end_ms"]]
        self.assertEqual(interval["supporting_frames"], actual)
        self.assertTrue(batch["qwen_evidence_coverage"][self.bad_target][0]["extends_beyond_media"])
        record = next(row for row in read(self.output / "reviews.json")["reviews"] if row["target_frame_id"] == self.bad_target)
        self.assertEqual(record["qwen_annotation"], draft)
        self.assertEqual(record["evidence"]["qwen_validation"], context["qwen_validation"])
        self.assertTrue(record["human_review_required"])
        self.assertFalse(record["training_eligible"])
        self.assertEqual((self.output / "qwen/round-00/response.json").read_bytes(), self.parent_bytes["round-00/response.json"])
        self.assertFalse((self.output / "qwen/round-00/result.json").exists())
        self.assert_original_unchanged()

    def test_intake_freezes_every_raw_artifact_and_shared_medgemma_response_with_hashes(self):
        self.rejected_annotation()
        client = JointOllama()
        self.review(client)
        plan = read(self.output / "run.json")
        audit = read(self.output / "draft-intake.json")
        self.assertEqual(plan["qwen_input_status"], "rejected_temporal_citations")
        self.assertEqual(audit["validation_status"], "rejected_temporal_citations")
        for name in ("draft-intake.json", "qwen-drafts.json"):
            self.assertEqual(plan["input_sha256"][name], sha256_file(self.output / name))
        for name, digest in audit["artifact_sha256"].items():
            copied = self.output / "qwen" / name
            self.assertEqual(copied.read_bytes(), self.parent_bytes[name])
            self.assertEqual(plan["input_sha256"]["qwen/" + name], digest)
            self.assertEqual(sha256_file(copied), digest)
        for required in ("run.json", "summary.json", "last-error.json", "round-00/request.json", "round-00/response.json", "round-00/verification.json"):
            self.assertIn(required, audit["artifact_sha256"])
        reviews = read(self.output / "reviews.json")["reviews"]
        self.assertEqual({row["call_directory"] for row in reviews}, {"calls/surgery/attempt-0001"})
        raw = self.output / "calls/surgery/attempt-0001/response.json"
        self.assertEqual(raw.read_bytes(), client.responses[0])
        receipt = read(raw.parent / "receipt.json")
        self.assertEqual(receipt["files"]["response.json"], sha256_file(raw))
        self.assert_original_unchanged()

    def test_inspector_displays_flagged_qwen_drafts_with_joint_reviews_and_original_citations(self):
        self.rejected_annotation(beyond_duration=True)
        self.review(JointOllama())
        store = InspectorStore(self.root, self.fixture.dataset)
        records = store.records()["records"]
        self.assertEqual(len(records), 1, "Frozen copies and flagged drafts must not create duplicate case cards")
        record = records[0]
        self.assertEqual(record["medgemma_review_count"], len(self.selected_ids))
        self.assertEqual(record["qwen_annotation_count"], len(self.selected_ids))
        frame = store.frame(record["id"], self.bad_target)
        interval = frame["qwen"]["contextual_claims"][0]["evidence_intervals"][0]
        self.assertEqual({key: interval[key] for key in self.bounds}, self.bounds)
        self.assertIsNotNone(frame["medgemma"])
        self.assertEqual(frame["raw"]["medgemma"]["qwen_annotation"]["contextual_claims"][0]["evidence_intervals"][0], interval)
        artifacts = [store.media(row["url"].rsplit("/", 1)[-1]) for row in frame["artifacts"]]
        self.assertIn(self.output / "calls/surgery/attempt-0001/request.json", artifacts)
        self.assertIn(self.output / "calls/surgery/attempt-0001/response.json", artifacts)
        self.assert_original_unchanged()

    def test_completed_flagged_review_resumes_without_model_calls_or_source_rewrite(self):
        self.rejected_annotation(beyond_duration=True)
        self.review(JointOllama())
        originals = {path: path.read_bytes() for path in (self.output / "calls").rglob("*.json")}
        client = JointOllama()
        self.assertEqual(self.review(client, resume=True)["status"], "completed")
        self.assertEqual(client.info_calls, [])
        self.assertEqual(client.requests, [])
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})
        self.assert_original_unchanged()

    def test_tampered_frozen_qwen_raw_response_rejects_resume_and_preserves_published_reviews(self):
        self.rejected_annotation()
        self.review(JointOllama())
        published = (self.output / "reviews.json").read_bytes()
        raw = self.output / "qwen/round-00/response.json"
        raw.write_bytes(raw.read_bytes() + b"\n")
        client = JointOllama()
        with self.assertRaises(ContractError):
            self.review(client, resume=True)
        self.assertEqual(client.info_calls, [])
        self.assertEqual(client.requests, [])
        self.assertEqual((self.output / "reviews.json").read_bytes(), published)
        self.assert_original_unchanged()

    def test_flagged_intake_does_not_bypass_incomplete_video_receipts(self):
        self.rejected_annotation(beyond_duration=True)
        path = self.parent / "round-00/verification.json"
        verification = read(path)
        verification["decoded_frame_ids"] = verification["decoded_frame_ids"][:-1]
        verification["decoded_frames"] -= 1
        write(path, verification)
        client = JointOllama()
        with self.assertRaises(ContractError):
            self.review(client)
        self.assertEqual(client.info_calls, [])
        self.assertEqual(client.requests, [])
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
