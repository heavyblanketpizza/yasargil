"""Real tiny media checks for complete decoding and honest source timestamps."""

from fractions import Fraction
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from PIL import Image

from yasargil.video_source import VideoSourceError, file_sha256, media_timeline, prepare_video_source, retrieve_interval, validate_native_timeline


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required for provenance integration checks")
class VideoSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.release = self.root / "release"
        self.release.mkdir()
        for index, color in enumerate(("red", "green", "blue", "yellow"), start=1):
            Image.new("RGB", (64, 48), color).save(self.release / f"S6A3_frame_{index:08d}.jpeg")

    def test_complete_release_has_nominal_times_original_hashes_and_lossless_pixels(self):
        result = prepare_video_source(self.release, self.root / "prepared", released_fps=1)
        self.assertEqual(result["expected_video_frames"], 4)
        self.assertEqual(result["duration_ms"], 4000)
        self.assertEqual(result["source_kind"], "released_image_sequence")
        self.assertEqual(result["media_timeline"], media_timeline(result))
        self.assertFalse(result["video_is_original_bytes"])
        self.assertEqual([frame["timestamp_ms"] for frame in result["frames"]], [0, 1000, 2000, 3000])
        self.assertEqual([frame["release_frame_index"] for frame in result["frames"]], [1, 2, 3, 4])
        for frame in result["frames"]:
            self.assertEqual(frame["timestamp_basis"], "reconstructed_nominal")
            self.assertIsNone(frame["source_pts"])
            self.assertIsNone(frame["source_acquisition_time"])
            self.assertEqual(frame["source_sha256"], file_sha256(Path(frame["source_path"])))
            with Image.open(frame["source_path"]) as original, Image.open(frame["image_path"]) as staged:
                self.assertEqual(original.convert("RGB").tobytes(), staged.tobytes())
        self.assertEqual(json.loads((self.root / "prepared/source.json").read_text()), result)
        decoded = subprocess.check_output([
            "ffmpeg", "-v", "error", "-i", result["video_path"], "-map", "0:v:0",
            "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ])
        expected = b""
        for frame in result["frames"]:
            with Image.open(frame["image_path"]) as image:
                expected += image.tobytes()
        self.assertEqual(decoded, expected)

    def test_original_video_keeps_bytes_and_every_variable_rate_source_pts(self):
        video = self.root / "original.mkv"
        subprocess.run([
            "ffmpeg", "-v", "error", "-framerate", "10", "-i", str(self.release / "S6A3_frame_%08d.jpeg"),
            "-vf", "setpts='3/TB+if(lt(N,2),N,2*N-1)/(10*TB)'", "-fps_mode", "passthrough",
            "-c:v", "ffv1", str(video),
        ], check=True)
        original_hash = file_sha256(video)
        result = prepare_video_source(video, self.root / "native")
        self.assertEqual(result["expected_video_frames"], 4)
        self.assertTrue(result["video_is_original_bytes"])
        self.assertTrue(Path(result["video_path"]).is_symlink())
        self.assertEqual(result["video_sha256"], original_hash)
        self.assertEqual(file_sha256(video), original_hash)
        self.assertEqual([f["timestamp_ms"] for f in result["frames"]], [0, 100, 300, 500])
        timing = media_timeline(result)
        self.assertEqual(timing["timestamp_basis"], "source_pts")
        self.assertEqual(timing["duration_ms"], 600)
        self.assertIsNone(timing["playback_fps"])
        self.assertEqual(timing["zero_based_frame_timestamp_formula"],
                         "(source_pts - first_source_pts) * source_time_base * 1000")
        self.assertGreater(result["timestamp_origin_source_pts"], 0)
        for frame in result["frames"]:
            self.assertEqual(frame["source_path"], str(video))
            self.assertEqual(frame["timestamp_basis"], "source_pts")
            self.assertEqual(frame["source_timestamp_ms"], float(frame["source_pts"] * Fraction(frame["time_base"]) * 1000))
        self.assertEqual(len({f["image_sha256"] for f in result["frames"]}), 4)
        with self.assertRaisesRegex(VideoSourceError, "Irregular/VFR timeline"):
            validate_native_timeline(result)

    def test_native_timeline_accepts_complete_nominal_release_and_checks_every_timestamp(self):
        result = prepare_video_source(self.release, self.root / "prepared", released_fps=1)
        receipt = validate_native_timeline(result)
        self.assertTrue(receipt["constant_frame_rate_verified"])
        self.assertEqual(receipt["fps"], 1)
        self.assertEqual(receipt["expected_count"], 4)
        self.assertEqual(receipt["max_deviation_ms"], 0)
        changed = copy.deepcopy(result)
        changed["frames"][2]["timestamp_ms"] += 0.01
        with self.assertRaisesRegex(VideoSourceError, "nominal timing"):
            validate_native_timeline(changed)

    def test_native_timeline_accepts_regular_nonzero_source_pts_origin(self):
        video = self.root / "regular.mkv"
        subprocess.run([
            "ffmpeg", "-v", "error", "-framerate", "10", "-i", str(self.release / "S6A3_frame_%08d.jpeg"),
            "-vf", "setpts=PTS+3/TB", "-fps_mode", "passthrough", "-c:v", "ffv1", str(video),
        ], check=True)
        result = prepare_video_source(video, self.root / "native")
        receipt = validate_native_timeline(result)
        self.assertEqual(receipt["fps"], 10)
        self.assertGreater(receipt["timestamp_origin_source_pts"], 0)
        self.assertEqual(receipt["max_deviation_ms"], 0)

    def test_native_timeline_rejects_missing_pts_rate_count_and_stream_mapping(self):
        result = prepare_video_source(self.release, self.root / "prepared", released_fps=1)
        mutations = [
            lambda changed: changed["ffprobe"]["streams"][0].pop("r_frame_rate"),
            lambda changed: changed["ffprobe"]["streams"][0].update(r_frame_rate="0/0"),
            lambda changed: changed.update(expected_video_frames=3),
            lambda changed: changed.update(video_stream_index=10),
            lambda changed: changed["frames"][2].update(frame_index=3),
            lambda changed: changed["frames"][2].update(video_pts=1),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(result)
            mutate(changed)
            with self.assertRaisesRegex(VideoSourceError, "Native timeline is not safe"):
                validate_native_timeline(changed)

    def test_native_timeline_allows_fractional_cfr_quantized_to_container_time_base(self):
        video = self.root / "fractional-rate.mkv"
        subprocess.run([
            "ffmpeg", "-v", "error", "-framerate", "30000/1001", "-i", str(self.release / "S6A3_frame_%08d.jpeg"),
            "-fps_mode", "passthrough", "-c:v", "ffv1", str(video),
        ], check=True)
        result = prepare_video_source(video, self.root / "native")
        receipt = validate_native_timeline(result)
        self.assertEqual(receipt["fps_rational"], "30000/1001")
        self.assertGreater(receipt["max_deviation_ms"], 0)
        self.assertLess(receipt["max_deviation_ms"], receipt["tolerance_ms"])

    def test_coarse_time_base_does_not_hide_a_whole_missing_frame(self):
        result = prepare_video_source(self.release, self.root / "prepared", released_fps=1)
        # A legal coarse time base should not grant a full frame of tolerance.
        result["ffprobe"]["streams"][0]["time_base"] = "1/1"
        for index, (frame, decoded) in enumerate(zip(result["frames"], result["ffprobe"]["frames"])):
            frame["video_time_base"] = "1/1"
            frame["video_pts"] = decoded["pts"] = index if index < 2 else index + 1
        with self.assertRaisesRegex(VideoSourceError, "Irregular/VFR timeline"):
            validate_native_timeline(result)

    def test_frame_ids_are_stable_across_preparation_directories(self):
        first = prepare_video_source(self.release, self.root / "first", released_fps=2)
        second = prepare_video_source(self.release, self.root / "second", released_fps=2)
        self.assertEqual([f["frame_id"] for f in first["frames"]], [f["frame_id"] for f in second["frames"]])
        self.assertEqual(first["duration_ms"], 2000)

    def test_reconstruction_uses_number_of_media_frames_divided_by_explicit_fps(self):
        result = prepare_video_source(self.release, self.root / "prepared", released_fps=2)
        timeline = media_timeline(result)
        self.assertEqual(timeline["duration_ms"], 2000)
        self.assertEqual(timeline["playback_fps"], 2)
        self.assertEqual(timeline["frame_count"], 4)
        self.assertEqual(timeline["zero_based_frame_timestamp_formula"], "frame_index / playback_fps * 1000")
        self.assertEqual([f["timestamp_ms"] for f in result["frames"]], [0, 500, 1000, 1500])
        self.assertEqual([f["frame_index"] for f in retrieve_interval(result, 1250, 2000, 4)], [3])
        with self.assertRaisesRegex(VideoSourceError, "exceeds.*playback duration"):
            retrieve_interval(result, 1250, 2000.01, 4)

    def test_repair_duration_declared_length_and_counts_never_change_media_timing(self):
        result = prepare_video_source(self.release, self.root / "prepared", released_fps=1)
        before = media_timeline(result)
        result.update({"Time for repair": 551, "Length": "0:04:46", "declared_frame_count": 286,
                       "repair_duration_seconds": 551,
                       "case_outcomes": [{"endpoint": "repair_duration_seconds", "value": 551}]})
        self.assertEqual(media_timeline(result), before)
        self.assertEqual(before["duration_ms"], 4000)
        self.assertEqual(before["authority"], "supplied_video_playback")
        for field in ("repair_time_used", "original_procedure_elapsed_time_verified", "full_procedure_coverage_verified"):
            self.assertIs(before[field], False)
        self.assertEqual([f["frame_index"] for f in retrieve_interval(result, 1000, 2000, 4)], [1, 2])
        self.assertEqual(validate_native_timeline(result)["expected_count"], 4)

    def test_corrupt_media_duration_basis_count_and_policy_fail_before_retrieval(self):
        result = prepare_video_source(self.release, self.root / "prepared", released_fps=1)
        mutations = [
            lambda changed: changed.update(duration_ms=551000),
            lambda changed: changed.update(duration_ms=float("nan")),
            lambda changed: changed.update(duration_ms=float("inf")),
            lambda changed: changed.update(duration_ms=True),
            lambda changed: changed.update(duration_basis="recorded_repair_duration"),
            lambda changed: changed.update(expected_video_frames=286),
            lambda changed: changed.update(timestamp_basis="source_pts"),
            lambda changed: changed["media_timeline"].update(repair_time_used=True),
            lambda changed: changed["media_timeline"].update(full_procedure_coverage_verified=0),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = copy.deepcopy(result)
                mutate(changed)
                with self.assertRaises(VideoSourceError):
                    media_timeline(changed)
                with self.assertRaisesRegex(VideoSourceError, "Native timeline is not safe"):
                    validate_native_timeline(changed)
                with self.assertRaises(VideoSourceError):
                    retrieve_interval(changed, 0, 1000, 2)

    def test_release_encoded_extent_is_checked_with_bounded_pts_quantization_allowance(self):
        result = prepare_video_source(self.release, self.root / "prepared", released_fps=2)
        result["ffprobe"]["streams"][0]["time_base"] = "1/1001"
        for index, (frame, decoded) in enumerate(zip(result["frames"], result["ffprobe"]["frames"])):
            frame["video_time_base"] = "1/1001"
            frame["video_pts"] = decoded["pts"] = round(index * 1001 / 2)
            decoded["duration"] = 501
        self.assertEqual(media_timeline(result)["duration_ms"], 2000)
        result["ffprobe"]["frames"][-1]["duration"] += 1001
        with self.assertRaisesRegex(VideoSourceError, "Encoded release extent"):
            media_timeline(result)

    def test_historical_manifest_without_policy_uses_same_validated_media_timeline(self):
        result = prepare_video_source(self.release, self.root / "prepared", released_fps=1)
        expected = result.pop("media_timeline")
        self.assertEqual(media_timeline(result), expected)
        self.assertNotIn("media_timeline", result)
        self.assertEqual(validate_native_timeline(result)["expected_count"], 4)
        self.assertEqual([f["frame_index"] for f in retrieve_interval(result, 2000, 3000, 2)], [2, 3])

    def test_interval_fetch_uses_existing_timestamps_and_respects_exclusions(self):
        result = prepare_video_source(self.release, self.root / "prepared", released_fps=1)
        found = retrieve_interval(result, 0, 3000, 2)
        self.assertEqual([f["timestamp_ms"] for f in found], [0, 3000])
        found = retrieve_interval(result, 500, 2200, 1)
        self.assertEqual([f["timestamp_ms"] for f in found], [1000])
        self.assertEqual(retrieve_interval(result, 100, 900, 4), [])
        found = retrieve_interval(result, 0, 3000, 5, exclude_ids=(result["frames"][1]["frame_id"],))
        self.assertEqual([f["frame_index"] for f in found], [0, 2, 3])
        for start, end, budget in ((-1, 2, 1), (4, 3, 1), (0, float("nan"), 1), (0, 3, 0), (0, 3, True)):
            with self.assertRaises(VideoSourceError):
                retrieve_interval(result, start, end, budget)

    def test_missing_duplicate_unreadable_and_unspecified_release_timing_fail(self):
        with self.assertRaisesRegex(VideoSourceError, "Missing video source"):
            prepare_video_source(self.root / "missing.mp4", self.root / "missing")
        with self.assertRaisesRegex(VideoSourceError, "explicit released_fps"):
            prepare_video_source(self.release, self.root / "no-rate")
        second = self.release / "S6A3_frame_00000002.jpeg"
        second.rename(self.release / "S6A3_frame_00000006.jpeg")
        with self.assertRaisesRegex(VideoSourceError, "Missing released frames"):
            prepare_video_source(self.release, self.root / "gap", released_fps=1)
        (self.release / "S6A3_frame_00000006.jpeg").rename(second)
        duplicate = self.release / "S6A3_frame_2.png"
        Image.new("RGB", (64, 48)).save(duplicate)
        with self.assertRaisesRegex(VideoSourceError, "Duplicate released frame"):
            prepare_video_source(self.release, self.root / "duplicate", released_fps=1)
        duplicate.unlink()
        second.write_bytes(b"not an image")
        with self.assertRaisesRegex(VideoSourceError, "Unreadable released frame"):
            prepare_video_source(self.release, self.root / "corrupt", released_fps=1)
        self.assertFalse((self.root / "corrupt/source.json").exists())


if __name__ == "__main__":
    unittest.main()
