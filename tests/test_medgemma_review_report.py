"""MedGemma reports preserve evidence provenance and deferred review state."""
import json
from pathlib import Path
import tempfile
import unittest

from yasargil.medgemma_review_report import write_review_report


class MedGemmaReviewReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def write_json(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def packet(self):
        return {"target_frame_id": "frame-0001", "frames": [
            {"frame_id": f"frame-{index:04d}", "timestamp_ms": 1000 * index,
             "timestamp_basis": "reconstructed_nominal", "evidence_roles": [role],
             "image_path": str(self.root / f'image {index} #.png'),
             "source_path": str(self.root / f'source {index}.jpg')}
            for index, role in enumerate(("before", "target", "after"))],
            "qwen_annotation": {"visible_observation": "Qwen <instrument>.", "visibility": "partial",
                "contextual_claims": [{"claim": "Qwen contextual action.",
                    "evidence_intervals": [{"start_ms": 0, "end_ms": 2000}]}],
                "uncertainties": ["Qwen uncertainty."]},
            "dataset_context": {"original_annotations": [{"text": "Original dataset <label>."}]},
            "limitations": ["Original acquisition timestamps unavailable."]}

    def record(self, status="review_complete"):
        return {"target_frame_id": "frame-0001", "call_directory": "calls/frame-0001/attempt-0001",
            "medgemma_review": {"status": status, "assessment": "revised",
                "revised_annotation": {"visible_observation": 'MedGemma <script>alert("x")</script>.',
                    "visibility": "partly_obscured", "contextual_claims": [{"claim": "MedGemma contextual action.",
                        "evidence_frame_ids": ["frame-0000", "frame-0002"]}],
                    "uncertainties": ['Cannot identify "structure".']},
                "corrections": [{"original_text": "Original phrase", "revised_text": "Corrected phrase",
                    "reason": "Instrument tip obscured", "evidence_frame_ids": ["frame-0001"]}],
                "evidence_requests": []}, "human_review_required": True, "training_eligible": False}

    def test_complete_review_compares_annotations_and_shows_canonical_evidence(self):
        packet, record = self.packet(), self.record()
        self.write_json("summary.json", {"status": "completed", "selected_frame_count": 1,
            "reviewed_frame_count": 1, "deferred_frame_count": 0})
        self.write_json("evidence/frame-0001.json", packet)
        self.write_json("reviews.json", {"reviews": [record]})
        self.write_json("deferred-evidence.json", {"automated_followup": False, "requests": []})
        page = write_review_report(self.root).read_text()
        self.assertIn("1 / 1 reviews saved", page)
        self.assertIn("Review complete · Model draft", page)
        self.assertIn("Qwen original annotation", page)
        self.assertIn("MedGemma revised annotation", page)
        self.assertIn("Qwen &lt;instrument&gt;.", page)
        self.assertIn("MedGemma &lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;.", page)
        self.assertNotIn("<script>", page)
        self.assertIn("Visible in the target frame", page)
        self.assertIn("Video interval: 00:00.000–00:02.000", page)
        self.assertIn("Evidence frame: frame-0002", page)
        self.assertIn("Original phrase", page)
        self.assertIn("Corrected phrase", page)
        self.assertIn("Instrument tip obscured", page)
        self.assertIn("Cannot identify &quot;structure&quot;", page)
        for frame in packet["frames"]:
            self.assertIn(Path(frame["image_path"]).as_uri(), page)
            self.assertIn(Path(frame["source_path"]).as_uri(), page)
            self.assertIn(frame["evidence_roles"][0], page)
        self.assertIn("image%201%20%23.png", page)
        self.assertIn("reconstructed_nominal", page)
        self.assertIn("Original dataset &lt;label&gt;", page)
        self.assertIn("Original acquisition timestamps unavailable", page)
        self.assertIn("Full evidence packet and coverage", page)
        self.assertIn((self.root / "deferred-evidence.json").as_uri(), page)
        self.assertIn((self.root / record["call_directory"] / "response.json").as_uri(), page)
        self.assertIn((self.root / record["call_directory"] / "request.json").as_uri(), page)
        self.assertIn("Not eligible for training", page)
        self.assertNotIn("https://", page)
        self.assertEqual(list(self.root.glob(".medgemma-review-report-*")), [])

    def test_deferred_review_retains_requests_without_claiming_search_was_dispatched(self):
        record = self.record("needs_more_evidence")
        record["medgemma_review"]["evidence_requests"] = [{"question": "Is the target <tip> visible?",
            "reason": "Before and after frames do not resolve occlusion", "target": "key_frame",
            "start_ms": 750, "end_ms": 1250}]
        self.write_json("evidence/frame-0001.json", self.packet())
        self.write_json("reviews.json", {"reviews": [record]})
        self.write_json("summary.json", {"status": "completed", "selected_frame_count": 1})
        page = write_review_report(self.root).read_text()
        self.assertIn("Needs more evidence · Saved for later", page)
        self.assertIn("1 frame(s) need more evidence", page)
        self.assertIn("Is the target &lt;tip&gt; visible?", page)
        self.assertIn("Before and after frames do not resolve occlusion", page)
        self.assertIn("00:00.750–00:01.250", page)
        self.assertIn("No Qwen or TimeLens2 search has been dispatched", page)
        self.assertIn("Full raw MedGemma response", page)

    def test_pending_and_incomplete_reports_preserve_prepared_evidence(self):
        self.assertIn("summary unavailable", write_review_report(self.root).read_text())
        self.write_json("evidence/frame-0001.json", self.packet())
        for status in ("prepared", "paused", "failed", "partial"):
            with self.subTest(status=status):
                self.write_json("summary.json", {"status": status, "selected_frame_count": 1,
                    "error": "Interrupted <attempt>" if status == "failed" else None})
                page = write_review_report(self.root).read_text()
                self.assertIn(f'<span class="status">{status}</span>', page)
                self.assertIn("0 / 1 reviews saved", page)
                self.assertIn("MedGemma review pending", page)
                self.assertIn("Qwen &lt;instrument&gt;", page)
                self.assertIn("Target and supporting frames · 3 image(s)", page)
                if status == "failed":
                    self.assertIn("Interrupted &lt;attempt&gt;", page)

    def test_model_metadata_cannot_replace_saved_evidence_provenance(self):
        record = self.record()
        record["frames"] = [{"image_path": "model-invented.png", "timestamp_ms": 999000}]
        record["qwen_annotation"] = {"visible_observation": "Model changed the original"}
        self.write_json("evidence/frame-0001.json", self.packet())
        self.write_json("reviews.json", {"reviews": [record]})
        page = write_review_report(self.root).read_text()
        self.assertNotIn('src="' + (self.root / "model-invented.png").as_uri(), page)
        self.assertIn("Qwen &lt;instrument&gt;", page)
        self.assertNotIn("<p>Model changed the original</p>", page)

    def test_malformed_artifacts_do_not_hide_saved_records_or_invent_evidence(self):
        self.write_json("reviews.json", {"reviews": [self.record()]})
        self.write_json("evidence/frame-0001.json", self.packet())
        (self.root / "evidence/frame-0001.json").write_text("{")
        page = write_review_report(self.root).read_text()
        self.assertIn("Artifact warnings", page)
        self.assertIn("Could not read frame-0001.json", page)
        self.assertIn("No saved evidence packet is available", page)
        self.assertIn("MedGemma revised annotation", page)
        (self.root / "reviews.json").write_text("{")
        page = write_review_report(self.root).read_text()
        self.assertIn("Could not read reviews.json", page)
        self.assertIn("No saved evidence packets or accepted reviews", page)


if __name__ == "__main__":
    unittest.main()
