"""Independent reports keep claim evidence inspectable and model text escaped."""
import json
from pathlib import Path
import tempfile
import unittest

from yasargil.medgemma_annotation_report import write_annotation_report


class MedGemmaAnnotationReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def packet(self):
        return {"schema_version": "medgemma-annotation-evidence-v1", "target_frame_id": "frame-1",
            "views": [{"view_id": view_id, "frame_id": frame_id, "role": role,
                       "image_path": str(self.root / f"{view_id} image #.png"),
                       "width": 100, "height": 100, "bounds": [0, 0, 100, 100]}
                      for view_id, frame_id, role in (("target", "frame-1", "target"),
                        ("detail", "frame-1", "target_detail"), ("after", "frame-2", "context_after"))],
            "procedure_context": "Simulated <repair>", "limitations": ["Nominal reconstructed timestamps."],
            "neighbor_coverage": {"before": {"unavailable_count": 1}}}

    def record(self):
        return {"target_frame_id": "frame-1", "call_directory": "calls/frame-0000/attempt-0000",
            "annotation": {"status": "needs_more_evidence", "visibility": "partial", "claims": [
                {"category": "instrument", "statement": '<script>alert("model")</script>',
                 "support": "target_visible", "evidence_view_ids": ["detail"], "uncertainty": "Subtype uncertain"},
                {"category": "action", "statement": "Instrument moves toward tissue.",
                 "support": "context_supported", "evidence_view_ids": ["target", "after"], "uncertainty": ""}],
                "unresolved_questions": [{"question": "Which <tissue>?", "reason": "Boundary obscured",
                                           "kind": "target_detail"}]}}

    def test_claim_groups_link_to_target_crop_and_context_without_qwen_artifacts(self):
        packet, row = self.packet(), self.record()
        self.write("evidence/frame-0000.json", packet)
        self.write("annotations.json", {"annotations": [row]})
        self.write("summary.json", {"status": "partial", "selected_frame_count": 2, "annotated_frame_count": 1})
        page = write_annotation_report(self.root).read_text()
        self.assertIn("1 / 2 annotations saved", page)
        self.assertIn("Visible in the target frame", page)
        self.assertIn("Supported by temporal or procedure context", page)
        self.assertIn("target_detail", page)
        self.assertIn("context_after", page)
        self.assertIn("Bounds in source image: [0, 0, 100, 100]", page)
        self.assertIn("&lt;script&gt;alert(&quot;model&quot;)&lt;/script&gt;", page)
        self.assertNotIn("<script>", page)
        self.assertIn("Which &lt;tissue&gt;?", page)
        self.assertIn("Subtype uncertain", page)
        self.assertIn("Simulated &lt;repair&gt;", page)
        self.assertNotIn("Qwen", page)
        for view in packet["views"]:
            self.assertIn(Path(view["image_path"]).as_uri(), page)
        self.assertIn('href="#view-', page)
        self.assertIn((self.root / row["call_directory"] / "request.json").as_uri(), page)
        self.assertIn("Human review required", page)
        self.assertIn("Not eligible for training", page)
        self.assertFalse(list(self.root.glob(".medgemma-annotation-report-*")))

    def test_prepared_failed_and_missing_artifacts_remain_inspectable(self):
        self.assertIn("summary unavailable", write_annotation_report(self.root).read_text())
        self.write("evidence/frame-0000.json", self.packet())
        self.write("summary.json", {"status": "failed", "error": "Failed <call>", "selected_frame_count": 1})
        page = write_annotation_report(self.root).read_text()
        self.assertIn("0 / 1 annotations saved", page)
        self.assertIn("Independent MedGemma annotation pending", page)
        self.assertIn("Failed &lt;call&gt;", page)
        row = self.record()
        row["evidence"] = {"views": [{"image_path": "/model-invented.png"}]}
        self.write("annotations.json", {"annotations": [row]})
        (self.root / "evidence/frame-0000.json").write_text("{")
        page = write_annotation_report(self.root).read_text()
        self.assertIn("Artifact warnings", page)
        self.assertIn("No saved evidence packet is available", page)
        self.assertIn("saved view unavailable", page)
        self.assertNotIn('src="file:///model-invented.png"', page)
        self.assertIn("Instrument moves toward tissue.", page)


if __name__ == "__main__":
    unittest.main()
