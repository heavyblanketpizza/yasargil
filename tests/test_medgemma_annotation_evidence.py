"""Source-only packets, faithful crops, explicit boundaries and timing."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from yasargil.contract import ContractError, sha256_file
from yasargil.medgemma_annotation_evidence import build_evidence, canonical_frame
from yasargil.video_source import VideoSourceError, media_timeline


def source_fixture(root, count=7, *, original_pts=None):
    directory = root / "source"
    directory.mkdir(exist_ok=True)
    original = original_pts is not None
    if original:
        count = len(original_pts)
        video = root / "source.mp4"
        video.write_bytes(b"video-provenance-fixture")
    frames, decoded = [], []
    basis = "source_pts" if original else "reconstructed_nominal"
    for index in range(count):
        path = directory / f"frame-{index:06d}.png"
        image = Image.new("RGB", (20, 12))
        image.putdata([(x * 11, y * 19, index * 20) for y in range(12) for x in range(20)])
        image.save(path)
        pts = original_pts[index] if original else index * 1000
        duration = original_pts[-1] - original_pts[-2] if original else 1000
        decoded.append({"pts": pts, "duration": duration})
        frames.append({"frame_id": f"f{index}", "frame_index": index,
                       "release_frame_index": None if original else index + 1,
                       "source_path": str(video if original else path),
                       "source_sha256": sha256_file(video if original else path),
                       "image_path": str(path), "image_sha256": sha256_file(path),
                       "timestamp_ms": float(pts - original_pts[0] if original else pts),
                       "timestamp_basis": basis, "source_pts": pts if original else None,
                       "time_base": "1/1000" if original else None,
                       "source_timestamp_ms": float(pts) if original else None,
                       "source_acquisition_time": None, "video_pts": pts, "video_time_base": "1/1000",
                       "width": 20, "height": 12})
    source = {"source_kind": "original_video" if original else "released_image_sequence",
              "source_path": str(video if original else directory), "source_sha256": "a" * 64,
              "frames": frames, "expected_video_frames": count, "released_fps": None if original else 1.,
              "duration_ms": float(original_pts[-1] - original_pts[0] + duration) if original else float(count * 1000),
              "timestamp_basis": basis, "timestamp_origin_source_pts": original_pts[0] if original else None,
              "duration_basis": "last_source_pts_plus_decoded_frame_duration" if original else
                  "released_frame_count_divided_by_explicit_nominal_fps", "video_stream_index": 0,
              "ffprobe": {"streams": [{"index": 0, "time_base": "1/1000"}], "frames": decoded}}
    source["media_timeline"] = media_timeline(source)
    return source


class AnnotationEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = source_fixture(self.root)
        self.output = self.root / "crops"

    def packet(self, target="f3", **kwargs):
        return build_evidence(self.source, target, self.output, **kwargs)

    def test_nearest_source_neighbors_and_native_crops_are_reproducible(self):
        before = copy.deepcopy(self.source)
        packet = self.packet()
        self.assertEqual([frame["frame_id"] for frame in packet["frames"]], ["f1", "f2", "f3", "f4", "f5"])
        self.assertEqual([view["role"] for view in packet["views"]],
                         ["target"] + ["target_detail"] * 4 + ["context_before"] * 2 + ["context_after"] * 2)
        with Image.open(self.source["frames"][3]["image_path"]) as target:
            for view in packet["views"][1:5]:
                with Image.open(view["image_path"]) as crop:
                    self.assertEqual(crop.size, (12, 8))
                    self.assertEqual(crop.tobytes(), target.crop(view["bounds"]).tobytes())
                    self.assertEqual(view["image_sha256"], sha256_file(view["image_path"]))
        self.assertEqual(self.packet(), packet)
        self.assertEqual(self.source, before)
        packet["target"]["width"] = 1
        self.assertEqual(self.source, before)

    def test_target_only_baseline_and_source_boundaries(self):
        packet = self.packet(before_frames=0, after_frames=0, detail_crops=False)
        self.assertEqual([frame["frame_id"] for frame in packet["frames"]], ["f3"])
        self.assertEqual(len(packet["views"]), 1)
        self.assertFalse(self.output.exists())
        packet = self.packet("f0", detail_crops=False)
        self.assertEqual(packet["neighbor_coverage"]["before"], {
            "requested": 2, "included_frame_ids": [], "unavailable_count": 2, "availability": "unavailable"})
        self.assertTrue(any("source boundary" in text for text in packet["limitations"]))
        packet = self.packet("f6", detail_crops=False)
        self.assertEqual(packet["neighbor_coverage"]["after"]["unavailable_count"], 2)

    def test_annotations_and_outcome_metadata_cannot_leak_through_source_rows(self):
        self.source.update(qwen_annotation="SECRET_QWEN", dataset_context="SECRET_CSV",
                           surgeon_experience="SECRET_EXPERIENCE", recorded_repair_time_ms=999999)
        for frame in self.source["frames"]:
            frame.update(visible_observation="SECRET_QWEN", annotations={"label": "SECRET_CSV"},
                         contextual_claims=["SECRET_CLINICAL_OUTCOME"])
        # Even an adjacent answer-label file is outside the evidence data path.
        (Path(self.source["source_path"]) / "sospine_tool_tips.csv").write_text("label\nSECRET_CSV\n")
        packet = self.packet(procedure_context="Documented simulation context")
        encoded = json.dumps(packet)
        self.assertNotIn("SECRET", encoded)
        self.assertEqual(packet["target"], canonical_frame(self.source["frames"][3]))
        self.assertEqual(packet["procedure_context"], "Documented simulation context")
        self.assertEqual(packet["media_timeline"]["duration_ms"], 7000.)

    def test_source_and_image_hashes_and_dimensions_are_verified(self):
        for field, value in (("source_sha256", "b" * 64), ("image_sha256", "b" * 64), ("width", 21)):
            source = copy.deepcopy(self.source)
            source["frames"][2][field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                build_evidence(source, "f3", self.output)
        path = Path(self.source["frames"][3]["image_path"])
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ContractError, "hash mismatch"):
            self.packet()

    def test_invalid_source_manifest_is_rejected(self):
        cases = [None, {}, {**self.source, "frames": []}, {**self.source, "expected_video_frames": 6},
                 {**self.source, "duration_ms": float("nan")}, {**self.source, "source_sha256": "invalid"}]
        for field, value in (("frame_id", "f0"), ("frame_index", 0), ("timestamp_ms", 0),
                             ("timestamp_ms", float("inf")), ("timestamp_ms", 7001),
                             ("source_sha256", "bad"), ("image_path", "relative.png"),
                             ("source_pts", 16000), ("timestamp_basis", "guessed"), ("timestamp_basis", [])):
            source = copy.deepcopy(self.source)
            source["frames"][1][field] = value
            cases.append(source)
        for source in cases:
            with self.subTest(source=str(source)[:100]), self.assertRaises(ContractError):
                build_evidence(source, "f3", self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_targets_options_and_timeline_fail(self):
        for options in ({"before_frames": -1}, {"before_frames": True}, {"after_frames": 9},
                        {"after_frames": 1.5}, {"detail_crops": 1}, {"procedure_context": {}},
                        {"procedure_context": "x" * 6001}):
            with self.subTest(options=options), self.assertRaises(ContractError):
                self.packet(**options)
        with self.assertRaises(ContractError):
            self.packet("unknown")
        self.source["duration_ms"] = 999999
        with self.assertRaises(VideoSourceError):
            self.packet()

    def test_original_video_variable_pts_and_nominal_release_caveats(self):
        self.assertTrue(any("nominal playback" in text for text in self.packet()["limitations"]))
        source = source_fixture(self.root, original_pts=[5000, 5100, 5900, 7050])
        packet = build_evidence(source, "f2", self.output)
        self.assertEqual([frame["timestamp_ms"] for frame in packet["frames"]], [0., 100., 900., 2050.])
        self.assertEqual([frame["source_pts"] for frame in packet["frames"]], [5000, 5100, 5900, 7050])
        self.assertFalse(any("nominal playback" in text for text in packet["limitations"]))

    def test_crop_output_does_not_follow_symlinks(self):
        self.output.mkdir()
        path = self.output / "target-00000003-detail-1.png"
        original = Path(self.source["frames"][3]["image_path"])
        path.symlink_to(original)
        before = original.read_bytes()
        with self.assertRaisesRegex(ContractError, "symlink"):
            self.packet()
        self.assertEqual(original.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
