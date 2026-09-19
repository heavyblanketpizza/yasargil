"""The experiment report remains reviewable before and after model responses."""
import json
from pathlib import Path
import tempfile
import unittest

from yasargil.gap_report import write_experiment_report


class GapReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def write_json(self, path, value):
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value), encoding="utf-8")

    def test_missing_and_incomplete_artifacts_render_without_inventing_results(self):
        path = write_experiment_report(self.root)
        self.assertEqual(path, self.root / "report.html")
        page = path.read_text()
        self.assertIn("summary unavailable", page)
        self.assertIn("Scores will appear", page)
        self.assertIn("drop the least-important 50%, 70%, or 90%", page)
        self.write_json("summary.json", {
            "status": "running", "source": {}, "reference": {"status": "running"},
            "conditions": [{"id": "bottom_50", "status": "pending", "metrics": None}]})
        page = write_experiment_report(self.root).read_text()
        self.assertIn("No frozen reference ranking", page)
        self.assertIn("No review rounds saved yet", page)
        self.assertIn("Bottom 50% supplied", page)
        self.assertNotIn("0.0%", page)
        self.assertEqual(list(self.root.glob(".report-*")), [])

    def test_full_report_links_provenance_and_round_evidence_and_escapes_model_text(self):
        source = self.root / "source with space.jpg"
        image = self.root / "frame image.png"
        frame = {"rank": 1, "frame_id": "frame-1", "moment_id": "moment-1",
                 "reason": '<script>alert("rank")</script>', "source_path": str(source),
                 "source_sha256": "source-hash", "image_path": str(image), "image_sha256": "image-hash",
                 "timestamp_ms": 12500, "timestamp_basis": "reconstructed_nominal", "frame_index": 12}
        added = {**frame, "frame_id": "retrieved-2", "timestamp_ms": 13000, "frame_index": 13}
        added.pop("rank")
        self.write_json("source.json", {"frames": [frame, added]})
        self.write_json("reference/reference.json", {"ranking": [frame], "moments": [
            {"moment_id": "moment-1", "start_ms": 12000, "end_ms": 13000,
             "frame_ids": ["frame-1"], "description": "visible <tool>"}]})
        self.write_json("summary.json", {
            "status": "completed", "source": {"source_path": str(source),
                "expected_video_frames": 288, "duration_ms": 288000, "timestamp_basis": "reconstructed_nominal"},
            "reference": {"status": "completed", "path": "reference/reference.json"},
            "conditions": [{"id": "bottom_50", "status": "completed", "supplied_frame_ids": [],
                "withheld_frame_ids": ["frame-1"], "actual_supplied_count": 12, "actual_withheld_count": 12,
                "actual_supplied_percent": 50, "actual_withheld_percent": 50,
                "missing_moment_ids": ["moment-1"], "rounds_completed": 1,
                "selected_frame_ids": ["retrieved-2"],
                "artifact_dir": "conditions/bottom_50", "metrics": {"detection_recall": 0.5,
                    "novel_evidence": {"human_review_required": True}, "empty_recall": None}}]})
        self.write_json("conditions/bottom_50/metrics.json", {"detection_recall": 0.5})
        self.write_json("reference/round-00/output.json", {"ranking": [frame]})
        self.write_json("conditions/bottom_50/rounds/round-00/output.json", {
            "searches": [{"start_ms": 12000, "end_ms": 13000, "question": "Find <missing> evidence"}]})
        self.write_json("conditions/bottom_50/rounds/round-00/verification.json", {
            "full_source_video_verified": True, "decoded_frames": 288})
        self.write_json("conditions/bottom_50/rounds/round-00/retrieval.json", [{"returned_frame_ids": ["frame-1"]}])
        self.write_json("conditions/bottom_50/rounds/round-00/candidate-manifest.json", [frame])
        page = write_experiment_report(self.root).read_text()
        self.assertIn("00:12.500", page)
        self.assertIn("reconstructed_nominal", page)
        self.assertIn("source-hash", page)
        self.assertIn(source.as_uri(), page)
        self.assertIn(image.as_uri(), page)
        self.assertIn("50.0%", page)
        self.assertIn("Latest retained candidates · 1", page)
        self.assertIn("Outside reference pool", page)
        self.assertIn("00:13.000", page)
        self.assertIn("Reference ranking: 1 candidates", page)
        self.assertIn("Detection recall", page)
        self.assertIn("Not applicable", page)
        self.assertIn("Complete metrics JSON", page)
        self.assertIn("1 evidence request(s)", page)
        self.assertIn("Complete video verified (288 decoded frames)", page)
        self.assertIn("Candidate provenance", page)
        self.assertIn("Returned evidence", page)
        self.assertIn("&lt;script&gt;alert(&quot;rank&quot;)&lt;/script&gt;", page)
        self.assertNotIn('<script>alert("rank")</script>', page)
        self.assertIn("Find &lt;missing&gt; evidence", page)
        self.assertNotIn("https://", page)

    def test_v2_report_shows_surgical_scores_drop_conditions_and_quality_receipt(self):
        frame = {"rank": 1, "frame_id": "frame-1", "importance_score": 95,
                 "reason": "Clear view of needle–tissue interaction", "moment_id": "moment-1",
                 "timestamp_ms": 4000, "source_path": str(self.root / "original.jpg")}
        added = {"frame_id": "retrieved-2", "timestamp_ms": 5000,
                 "source_path": str(self.root / "retrieved.jpg"), "frame_index": 5}
        self.write_json("source/source.json", {"frames": [frame, added]})
        self.write_json("reference/reference.json", {
            "schema_version": "gap-reference-v2", "ranking": [frame]})
        self.write_json("reference/ranking-quality.json", {
            "accepted": True, "flags": [], "scores_descending": True,
            "chronological_order": False, "reverse_chronological_order": False,
            "presentation_order_copied": False, "distinct_score_count": 24,
            "reference_is_ground_truth": False})
        self.write_json("summary.json", {
            "schema_version": "native-video-gap-experiment-v2", "status": "running",
            "reference": {"status": "frozen"}, "conditions": [{
                "condition_id": "drop_70", "supplied_frame_ids": ["frame-1"],
                "withheld_frame_ids": [], "actual_supplied_count": 7,
                "actual_withheld_count": 17, "actual_supplied_percent": 100 * 7 / 24,
                "actual_withheld_percent": 100 * 17 / 24, "selected_frame_ids": ["retrieved-2"],
                "rounds_completed": 1, "metrics": {
                    "declared_complete_with_unretained_reference_moments": True,
                    "false_completion_against_provisional_reference": None}}]})
        page = write_experiment_report(self.root).read_text()
        self.assertIn("Drop least-important 70%", page)
        self.assertNotIn("Bottom 70% supplied", page)
        self.assertIn("Surgical importance:</b> 95 / 100", page)
        self.assertIn("needle–tissue interaction", page)
        self.assertIn("Reference quality check: Passed", page)
        self.assertIn("Ranking quality receipt", page)
        self.assertIn("29.2%", page)
        self.assertIn("Declared complete with unretained reference moments", page)
        self.assertIn("recorded as an observation, not a failure", page)
        self.assertIn("00:05.000", page)
        self.assertIn((self.root / "source/source.json").as_uri(), page)

    def test_rejected_ranking_quality_remains_visible_without_frozen_reference(self):
        self.write_json("summary.json", {
            "schema_version": "native-video-gap-experiment-v2", "status": "failed",
            "reference": {"status": "rejected"}, "conditions": []})
        self.write_json("reference/ranking-quality.json", {
            "accepted": False, "flags": ["chronological_order", "unsafe <text>"],
            "chronological_order": True, "checked_at": "2026-09-13T00:00:00Z"})
        self.write_json("reference/round-00/output.json", {"ranking": [{"frame_id": "frame-1"}]})
        page = write_experiment_report(self.root).read_text()
        self.assertIn("Reference quality check: Failed — condition audits blocked", page)
        self.assertIn("chronological_order", page)
        self.assertIn("unsafe &lt;text&gt;", page)
        self.assertIn("Reference ranking: 1 candidates", page)
        self.assertIn("No frozen reference ranking is available yet", page)
        self.assertIn("Passing them does not establish surgical importance", page)

    def test_unreadable_optional_response_is_visible_without_losing_report(self):
        self.write_json("summary.json", {
            "status": "failed", "reference": {"status": "completed"},
            "conditions": [{"id": "bottom_90", "status": "failed", "artifact_dir": "conditions/bottom_90"}]})
        directory = self.root / "conditions/bottom_90/rounds/round-00"
        directory.mkdir(parents=True)
        (directory / "output.json").write_text('{"searches":')
        page = write_experiment_report(self.root).read_text()
        self.assertIn("Artifact warnings", page)
        self.assertIn("Could not read output.json", page)
        self.assertIn("Response pending or interrupted", page)


if __name__ == "__main__":
    unittest.main()
