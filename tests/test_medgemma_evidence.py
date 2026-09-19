"""Bounded context selection and exact original-dataset annotation provenance."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from yasargil.contract import ContractError, sha256_file
from yasargil.medgemma_evidence import build_evidence
from yasargil.video_source import VideoSourceError, media_timeline


class MedGemmaEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def source(self, count=20, *, case=None, original_pts=None):
        release = self.root / "dataset" / "frames" / case if case else self.root / "released"
        release.mkdir(parents=True, exist_ok=True)
        original = original_pts is not None
        if original:
            count = len(original_pts)
            video = self.root / "original.mp4"
            video.write_bytes(b"fixture provenance; decoding is outside this unit")
        frames, decoded = [], []
        for index in range(count):
            filename = f"{case}_frame_{index + 1:08d}.jpeg" if case else f"frame_{index + 1:08d}.jpeg"
            path = release / filename
            Image.new("RGB", (16, 16), (index * 7 % 255, 0, 0)).save(path)
            pts = original_pts[index] if original else index * 1000
            duration = (original_pts[-1] - original_pts[-2]) if original else 1000
            decoded.append({"pts": pts, "duration": duration})
            basis = "source_pts" if original else "reconstructed_nominal"
            frames.append({"frame_id": f"f{index:06d}", "frame_index": index,
                           "release_frame_index": None if original else index + 1,
                           "source_path": str(video if original else path),
                           "source_sha256": sha256_file(video if original else path),
                           "image_path": str(path), "image_sha256": sha256_file(path),
                           "timestamp_ms": float(pts - original_pts[0] if original else pts),
                           "timestamp_basis": basis, "source_pts": pts if original else None,
                           "time_base": "1/1000" if original else None,
                           "source_timestamp_ms": float(pts) if original else None,
                           "source_acquisition_time": None, "video_pts": pts,
                           "video_time_base": "1/1000", "width": 16, "height": 16})
        duration_ms = float(original_pts[-1] - original_pts[0] + duration) if original else float(count * 1000)
        source = {"source_kind": "original_video" if original else "released_image_sequence",
                  "source_path": str(video if original else release), "source_sha256": "a" * 64,
                  "frames": frames, "expected_video_frames": count, "released_fps": None if original else 1.,
                  "duration_ms": duration_ms, "timestamp_basis": basis,
                  "timestamp_origin_source_pts": original_pts[0] if original else None,
                  "duration_basis": "last_source_pts_plus_decoded_frame_duration" if original else
                      "released_frame_count_divided_by_explicit_nominal_fps",
                  "video_stream_index": 0,
                  "ffprobe": {"streams": [{"index": 0, "time_base": "1/1000"}], "frames": decoded}}
        source["media_timeline"] = media_timeline(source)
        return source

    def annotation(self, source, index, intervals=()):
        claims = [{"claim": f"Context {i}.", "evidence_intervals": [{
            "start_ms": start, "end_ms": end,
            "supporting_frames": [copy.deepcopy(frame) for frame in source["frames"]
                                  if start <= frame["timestamp_ms"] <= end],
        }]} for i, (start, end) in enumerate(intervals)]
        return {**copy.deepcopy(source["frames"][index]), "visible_observation": "An instrument is visible.",
                "visibility": "partial", "contextual_claims": claims,
                "uncertainties": ["Instrument contact is uncertain."],
                "review_required": True, "training_eligible": False}

    def tables(self, source, *, extra_field=None):
        root = Path(source["source_path"]).parent.parent
        case = Path(source["source_path"]).name
        fields = ["trial_frame", "x1", "y1", "x2", "y2", "label"]
        if extra_field:
            fields.append(extra_field)
        for table in ("sospine_tool_tips.csv", "sospine_bbox.csv"):
            with (root / table).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[""] + fields if "tool_tips" in table else fields)
                writer.writeheader()
                for index in (1, 2, 3, 6):
                    row = {"trial_frame": f"{case}_frame_{index:08d}.jpeg", "x1": "1.2300", "y1": "-4",
                           "x2": "1.2300", "y2": "9999", "label": "needle driver tip "}
                    if index == 2:
                        row.update({key: "" for key in fields if key != "trial_frame"})
                    if extra_field:
                        row[extra_field] = "OUTCOME_SECRET"
                    if "tool_tips" in table:
                        row[""] = str(index + 100)
                    writer.writerow(row)
                writer.writerow({"trial_frame": "S1A2_frame_00000003.jpeg", "label": "OTHER_CASE_SECRET"})
        (root / "sospine_outcomes.csv").write_text("Trial ID,Leak At 40mmHg,Time for repair\n" + case + ",Y,OUTCOME_SECRET\n")
        return root

    def test_nearest_neighbors_come_from_complete_inventory_and_inputs_are_unchanged(self):
        source = self.source()
        annotation = self.annotation(source, 8)
        before = copy.deepcopy((source, annotation))
        evidence = build_evidence(source, annotation)
        self.assertEqual([f["frame_index"] for f in evidence["frames"]], [6, 7, 8, 9, 10])
        self.assertEqual([f["evidence_roles"] for f in evidence["frames"]],
                         [["before"], ["before"], ["target"], ["after"], ["after"]])
        self.assertEqual(evidence["qwen_annotation"], annotation)
        self.assertEqual(evidence, build_evidence(source, annotation))
        self.assertEqual(evidence["dataset_context"]["status"], "unavailable")
        self.assertIsNone(evidence["dataset_context"]["dataset_name"])
        evidence["frames"][0]["source_path"] = "modified copy"
        evidence["qwen_annotation"]["uncertainties"].append("modified copy")
        self.assertEqual((source, annotation), before)

    def test_boundary_missing_frames_are_explicit_and_not_fabricated(self):
        source = self.source(count=1)
        evidence = build_evidence(source, self.annotation(source, 0))
        self.assertEqual(len(evidence["frames"]), 1)
        for role in ("before", "after"):
            self.assertEqual(evidence["neighbor_coverage"][role], {
                "requested": 2, "included_frame_ids": [], "unavailable_count": 2, "availability": "unavailable"})
            self.assertTrue(any(f"2 requested {role}" in value for value in evidence["limitations"]))
        source = self.source(count=4)
        evidence = build_evidence(source, self.annotation(source, 1))
        self.assertEqual(evidence["neighbor_coverage"]["before"]["availability"], "partial")
        self.assertEqual(evidence["neighbor_coverage"]["after"]["availability"], "complete")

    def test_qwen_interval_sampling_spreads_budget_and_records_every_omission(self):
        source = self.source()
        annotation = self.annotation(source, 8, [(0, 2000), (15000, 18000)])
        evidence = build_evidence(source, annotation, max_context_frames=7)
        indices = [frame["frame_index"] for frame in evidence["frames"]]
        self.assertEqual(indices, [1, 6, 7, 8, 9, 10, 16])
        for interval in evidence["qwen_evidence_coverage"]:
            self.assertEqual(len(interval["supplied_frame_ids"]), 1)
            self.assertEqual(set(interval["available_frame_ids"]),
                             set(interval["supplied_frame_ids"] + interval["omitted_frame_ids"]))
            self.assertFalse(set(interval["supplied_frame_ids"]) & set(interval["omitted_frame_ids"]))
            self.assertEqual(interval["omission_reason"], "image_budget")
        self.assertTrue(any("image budget" in value for value in evidence["limitations"]))
        large = build_evidence(source, annotation, max_context_frames=12)
        self.assertTrue(all(interval["complete"] for interval in large["qwen_evidence_coverage"]))

    def test_uncovered_intervals_take_priority_and_roles_can_overlap(self):
        source = self.source()
        annotation = self.annotation(source, 8, [(6000, 11000), (15000, 19000)])
        evidence = build_evidence(source, annotation, max_context_frames=6)
        self.assertIn(17, [frame["frame_index"] for frame in evidence["frames"]])
        self.assertEqual(next(frame for frame in evidence["frames"] if frame["frame_index"] == 8)["evidence_roles"],
                         ["target", "qwen_context"])

    def test_nominal_media_timing_is_preserved_and_repair_metadata_is_not_used(self):
        source = self.source()
        source.update(recorded_repair_time_ms=999999999, surgeon_experience="SECRET_EXPERIENCE")
        evidence = build_evidence(source, self.annotation(source, 5))
        self.assertEqual(evidence["media_timeline"]["duration_ms"], 20000)
        self.assertFalse(evidence["media_timeline"]["repair_time_used"])
        self.assertNotIn("SECRET_EXPERIENCE", json.dumps(evidence))
        self.assertIsNone(evidence["frames"][0]["source_pts"])
        self.assertTrue(any("nominal playback" in value for value in evidence["limitations"]))
        source["duration_ms"] = 999999999
        with self.assertRaises(VideoSourceError):
            build_evidence(source, self.annotation(source, 5))

    def test_original_video_keeps_variable_source_pts_without_dataset_fabrication(self):
        source = self.source(original_pts=[5000, 5120, 6000, 6200, 7100])
        evidence = build_evidence(source, self.annotation(source, 2), procedure_context="Documented color sequence")
        self.assertEqual([frame["source_pts"] for frame in evidence["frames"]], [5000, 5120, 6000, 6200, 7100])
        self.assertEqual([frame["timestamp_ms"] for frame in evidence["frames"]], [0., 120., 1000., 1200., 2100.])
        self.assertEqual(evidence["dataset_context"]["documented_procedure_context"], "Documented color sequence")
        self.assertEqual(evidence["dataset_context"]["original_annotations"], [])
        with self.assertRaises(ContractError):
            build_evidence(source, self.annotation(source, 2), dataset_root=self.root)

    def test_budget_and_neighbor_bounds_reject_booleans_and_invalid_limits(self):
        source = self.source()
        annotation = self.annotation(source, 8)
        for arguments in ({"before_frames": 0}, {"before_frames": True}, {"after_frames": 9},
                          {"after_frames": 1.5}, {"max_context_frames": 4}, {"max_context_frames": 33},
                          {"before_frames": 8, "after_frames": 8, "max_context_frames": 16}):
            with self.subTest(arguments=arguments), self.assertRaises(ContractError):
                build_evidence(source, annotation, **arguments)
        self.assertEqual(len(build_evidence(source, annotation, before_frames=8, after_frames=8,
                                            max_context_frames=17)["frames"]), 17)

    def test_canonical_target_and_cited_supporting_rows_cannot_be_rewritten(self):
        source = self.source()
        annotation = self.annotation(source, 8, [(0, 2000)])
        annotation["source_sha256"] = "b" * 64
        with self.assertRaisesRegex(ContractError, "target provenance"):
            build_evidence(source, annotation)
        annotation = self.annotation(source, 8, [(0, 2000)])
        annotation["contextual_claims"][0]["evidence_intervals"][0]["supporting_frames"][0]["timestamp_ms"] = 123
        with self.assertRaisesRegex(ContractError, "supporting frames"):
            build_evidence(source, annotation)
        for start, end in ((0, 0), (500, 750), (0, 999999), (float("nan"), 1000)):
            with self.subTest(start=start, end=end), self.assertRaises(ContractError):
                build_evidence(source, self.annotation(source, 8, [(start, end)]))

    def test_sospine_exact_rows_raw_coordinates_and_table_receipts_are_preserved(self):
        source = self.source(count=6, case="S6A3")
        root = self.tables(source)
        original_bytes = {path: path.read_bytes() for path in root.glob("*.csv")}
        annotation = self.annotation(source, 2)
        evidence = build_evidence(source, annotation, procedure_context="Simulated spinal durotomy repair")
        context = evidence["dataset_context"]
        self.assertEqual(context["status"], "available")
        self.assertEqual(context["case_id"], "S6A3")
        self.assertEqual(context["procedure_context_provenance"], "parent_annotation_run")
        self.assertFalse(context["procedure_context_is_visual_evidence"])
        self.assertEqual(evidence, build_evidence(source, annotation, dataset_root=root,
                                                procedure_context="Simulated spinal durotomy repair"))
        self.assertEqual(len(context["original_annotations"]), 6)
        points = context["original_annotations"][0]
        self.assertEqual(points["original_origin"], "manual")
        self.assertEqual(points["raw_value"], {"": "101", "trial_frame": "S6A3_frame_00000001.jpeg",
                         "x1": "1.2300", "y1": "-4", "x2": "1.2300", "y2": "9999", "label": "needle driver tip "})
        self.assertEqual(points["source_locator"]["locator"], 2)
        self.assertEqual(points["source_locator"]["physical_end_line_1based"], 2)
        self.assertEqual(points["source_frame_sha256"], source["frames"][0]["source_sha256"])
        self.assertEqual(context["original_annotations"][3]["original_origin"], "computed")
        for table in context["tables"]:
            self.assertEqual(table["sha256"], hashlib.sha256(original_bytes[root / table["filename"]]).hexdigest())
        availability = {(row["frame_id"], row["filename"]): row for row in context["label_availability"]}
        blank = availability[("f000001", "sospine_tool_tips.csv")]
        self.assertEqual((blank["status"], blank["reason"]), ("unavailable", "blank_placeholder_rows"))
        missing = availability[("f000004", "sospine_tool_tips.csv")]
        self.assertEqual((missing["status"], missing["reason"]), ("unavailable", "no_matching_source_rows"))
        self.assertTrue(all(row["negative_label"] is False for row in availability.values()))
        serialized = json.dumps(evidence)
        self.assertNotIn("OTHER_CASE_SECRET", serialized)
        self.assertNotIn("OUTCOME_SECRET", serialized)
        self.assertNotIn("S6A3_frame_00000006.jpeg", serialized)
        self.assertEqual(original_bytes, {path: path.read_bytes() for path in original_bytes})

    def test_source_mapping_rejects_same_basename_wrong_case_and_wrong_root(self):
        source = self.source(count=6, case="S6A3")
        root = self.tables(source)
        wrong_root = self.root / "wrong"
        wrong_root.mkdir()
        with self.assertRaises(ContractError):
            build_evidence(source, self.annotation(source, 2), dataset_root=wrong_root)
        frame = source["frames"][2]
        frame["source_path"] = str(root / "frames" / "S1A2" / Path(frame["source_path"]).name)
        with self.assertRaisesRegex(ContractError, "exact SOSpine"):
            build_evidence(source, self.annotation(source, 2), dataset_root=root)

    def test_source_hash_mismatch_and_symlink_table_escape_are_rejected(self):
        source = self.source(count=6, case="S6A3")
        root = self.tables(source)
        image = Path(source["frames"][2]["source_path"])
        previous = image.read_bytes()
        image.write_bytes(b"changed source image")
        with self.assertRaisesRegex(ContractError, "image hash mismatch"):
            build_evidence(source, self.annotation(source, 2))
        image.write_bytes(previous)
        table = root / "sospine_tool_tips.csv"
        escaped = self.root / "outside.csv"
        escaped.write_bytes(table.read_bytes())
        table.unlink()
        table.symlink_to(escaped)
        with self.assertRaisesRegex(ContractError, "symlink escape"):
            build_evidence(source, self.annotation(source, 2))

    def test_missing_tables_do_not_create_dataset_context_and_extra_columns_fail_closed(self):
        source = self.source(count=6, case="S6A3")
        self.assertEqual(build_evidence(source, self.annotation(source, 2))["dataset_context"]["status"], "unavailable")
        root = self.tables(source, extra_field="Time for repair")
        with self.assertRaisesRegex(ContractError, "Unexpected SOSpine annotation columns"):
            build_evidence(source, self.annotation(source, 2), dataset_root=root)
        (root / "sospine_bbox.csv").unlink()
        with self.assertRaises(ContractError):
            build_evidence(source, self.annotation(source, 2), dataset_root=root)


if __name__ == "__main__":
    unittest.main()
