"""Annotation review preserves provenance and the visible/context distinction."""
import json
from pathlib import Path
import tempfile
import unittest

from yasargil.annotation_report import write_annotation_report


class AnnotationReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def write_json(self, filename, value):
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def frame(self, index):
        return {"frame_id": f"frame-{index}", "frame_index": index,
                "timestamp_ms": index * 1000, "timestamp_basis": "reconstructed_nominal",
                "source_path": str(self.root / f"original image {index}.jpg"),
                "source_sha256": f"source-hash-{index}",
                "image_path": str(self.root / f"extracted image {index}.png"),
                "image_sha256": f"image-hash-{index}"}

    def test_pending_failed_and_conflict_reports_do_not_invent_annotations(self):
        page = write_annotation_report(self.root).read_text()
        self.assertIn("summary unavailable", page)
        self.assertIn("0 / 0 annotations saved", page)
        self.assertIn("manifest is not available yet", page)
        self.write_json("selected-frames.json", [self.frame(12), self.frame(20)])
        self.write_json("source/source.json", {"duration_ms": 288000,
            "expected_video_frames": 288, "timestamp_basis": "reconstructed_nominal"})
        for status in ("prepared", "running", "failed", "context_conflict", "paused"):
            with self.subTest(status=status):
                self.write_json("summary.json", {"status": status, "selected_frame_count": 2,
                    "error": "Interrupted <attempt>" if status == "failed" else None})
                page = write_annotation_report(self.root).read_text()
                self.assertIn(f'<span class="status">{status}</span>', page)
                self.assertIn("0 / 2 annotations saved", page)
                self.assertIn("Annotation pending", page)
                self.assertIn("04:48.000", page)
                self.assertIn("288 available video frames", page)
                self.assertIn("Not eligible for training", page)
                if status == "failed":
                    self.assertIn("Interrupted &lt;attempt&gt;", page)
        self.assertEqual(list(self.root.glob(".annotation-report-*")), [])

    def test_separates_visible_and_context_claims_and_links_exact_source_evidence(self):
        selected, support = self.frame(12), self.frame(20)
        self.write_json("selected-frames.json", [selected])
        self.write_json("source/source.json", {"frames": [selected, support],
            "source_path": str(self.root / "source sequence"), "duration_ms": 288000,
            "expected_video_frames": 288, "timestamp_basis": "reconstructed_nominal",
            "timeline_note": "Original acquisition times are unavailable."})
        self.write_json("summary.json", {"status": "completed", "selected_frame_count": 1,
            "session_id": "annotation-session", "source": {}, "annotations_path": "annotations.json"})
        self.write_json("annotations.json", {"schema_version": "contextual-frame-annotations-v1",
            "context_check": "consistent", "clinical_validation": "not_performed",
            "temporal_exposure": "retrospective_full_video", "evidence_validation": "locator_only_not_semantic",
            "annotations": [{**selected, "visible_observation": 'Visible <instrument> near tissue.',
                "visibility": "partly_obscured", "uncertainties": ['Cannot identify "tip".'],
                "contextual_claims": [{"claim": 'Later <script>alert("claim")</script> action.',
                    "evidence_intervals": [{"start_ms": 19500, "end_ms": 20500,
                        "supporting_frames": [support]}]}], "review_required": True, "training_eligible": False}]})
        for filename in ("run.json", "session.json", "round-00/request.json", "round-00/response.json",
                         "round-00/result.json"):
            self.write_json(filename, {})
        self.write_json("round-00/verification.json", {"full_source_video_verified": True, "decoded_frames": 288})
        page = write_annotation_report(self.root).read_text()
        self.assertIn("1 / 1 annotations saved", page)
        self.assertIn("Visible in this frame · model draft", page)
        self.assertIn("Added by the surrounding video · model draft", page)
        self.assertLess(page.index("Visible &lt;instrument&gt;"), page.index("Added by the surrounding video"))
        self.assertIn("Evidence locator · 00:19.500–00:20.500 · 1 source observation(s)", page)
        self.assertIn("00:12.000", page)
        self.assertIn("00:20.000", page)
        self.assertIn(Path(selected["image_path"]).as_uri(), page)
        self.assertIn(Path(support["source_path"]).as_uri(), page)
        self.assertIn("source-hash-12", page)
        self.assertIn("image-hash-20", page)
        self.assertIn("reconstructed_nominal", page)
        self.assertIn("Original acquisition times are unavailable", page)
        self.assertIn("Full-video verification", page)
        self.assertIn("Complete video verified (288 decoded frames)", page)
        self.assertIn("annotation-session", page)
        self.assertIn("Exact model request", page)
        self.assertIn("Raw model response", page)
        self.assertIn("Accepted call result", page)
        self.assertIn("&lt;script&gt;alert(&quot;claim&quot;)&lt;/script&gt;", page)
        self.assertNotIn("<script>", page)
        self.assertNotIn("https://", page)
        self.assertIn("Cannot identify &quot;tip&quot;", page)
        self.assertIn("Evidence validation checks source locators only", page)
        self.assertIn("Clinical validation has not been performed", page)

    def test_frozen_selected_provenance_cannot_be_replaced_by_annotation_metadata(self):
        frame = self.frame(12)
        self.write_json("selected-frames.json", [frame])
        self.write_json("annotations.json", {"annotations": [{**frame,
            "timestamp_ms": 999000, "image_path": "model-invented.png", "source_path": "model-invented.jpg",
            "visible_observation": "Draft", "contextual_claims": [], "uncertainties": []}]})
        page = write_annotation_report(self.root).read_text()
        self.assertIn('<h2>00:12.000</h2>', page)
        self.assertIn(f'src="{Path(frame["image_path"]).as_uri()}"', page)
        self.assertNotIn('src="' + (self.root / "model-invented.png").as_uri(), page)
        self.assertNotIn('<h2>16:39.000</h2>', page)

    def test_malformed_optional_annotation_artifact_is_reported_and_selected_frames_survive(self):
        self.write_json("summary.json", {"status": "failed"})
        self.write_json("selected-frames.json", [self.frame(12)])
        (self.root / "annotations.json").write_text('{"annotations":')
        page = write_annotation_report(self.root).read_text()
        self.assertIn("Artifact warnings", page)
        self.assertIn("Could not read annotations.json", page)
        self.assertIn("0 / 1 annotations saved", page)
        self.assertIn("Annotation pending", page)


if __name__ == "__main__":
    unittest.main()
