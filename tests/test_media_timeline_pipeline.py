"""Playback time stays authoritative when repair metadata disagrees with media."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from jsonschema import Draft202012Validator, ValidationError
from PIL import Image

from yasargil.gap_experiment import GapConfig, _audit_schema, _initial_messages as gap_messages
from yasargil.smart_selection import (
    SelectionConfig, initial_messages, retrieve_requests, review_schema, stage_media,
)
from yasargil.video_source import VideoSourceError, media_timeline, prepare_video_source, validate_native_timeline


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required for media-clock checks")
class MediaTimelinePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name).resolve()
        release = cls.root / "dataset" / "frames" / "S6A3"
        release.mkdir(parents=True)
        for index, color in enumerate(("red", "green", "blue", "yellow"), start=1):
            Image.new("RGB", (64, 48), color).save(release / f"S6A3_frame_{index:08d}.png")
        # This is deliberately incompatible with four observations at 2 fps.
        cls.metadata = cls.root / "dataset" / "sospine_outcomes.csv"
        cls.metadata.write_text("Sequence,Repair Time,Video Duration\nS6A3,03:00:00,02:00:00\n")
        cls.prepared = prepare_video_source(release, cls.root / "source", released_fps=2)
        staging = cls.root / "staging"
        staging.mkdir()
        _, cls.video_name, cls.names = stage_media(cls.prepared, staging)

    def setUp(self):
        self.source = copy.deepcopy(self.prepared)
        self.source.update(recorded_repair_time_ms=10_800_000,
                           dataset_video_duration_ms=7_200_000,
                           original_procedure_duration_ms=10_800_000)
        self.frames = self.source["frames"]
        self.ids = [frame["frame_id"] for frame in self.frames]
        self.context = "A color demonstration; dataset metadata records a three-hour repair."

    def prompts(self):
        yield "selection", initial_messages(
            self.source, self.frames, [self.ids[0], self.ids[-1]], self.names, self.video_name,
            SelectionConfig(procedure_context=self.context))
        for ranking in (True, False):
            yield "ranking" if ranking else "gap_audit", gap_messages(
                self.source, self.frames, self.names, self.video_name,
                GapConfig(procedure_context=self.context), ranking=ranking)

    def test_preparation_uses_frames_divided_by_fps_despite_neighboring_repair_metadata(self):
        self.assertEqual(self.prepared["duration_ms"], 2000)
        self.assertEqual([frame["timestamp_ms"] for frame in self.frames], [0, 500, 1000, 1500])
        self.assertEqual(self.prepared["released_fps"], 2)
        self.assertEqual(self.prepared["expected_video_frames"], 4)
        self.assertTrue(all(frame["timestamp_basis"] == "reconstructed_nominal" for frame in self.frames))
        self.assertTrue(all(frame["source_acquisition_time"] is None for frame in self.frames))
        self.assertEqual(self.metadata.read_text(),
                         "Sequence,Repair Time,Video Duration\nS6A3,03:00:00,02:00:00\n")
        before = copy.deepcopy(self.source)
        timeline = media_timeline(self.source)
        self.assertEqual(timeline["authority"], "supplied_video_playback")
        self.assertEqual(timeline["duration_ms"], 2000)
        self.assertIs(timeline["repair_time_used"], False)
        self.assertIs(timeline["original_procedure_elapsed_time_verified"], False)
        self.assertIs(timeline["full_procedure_coverage_verified"], False)
        self.assertEqual(self.source, before)

    def test_every_model_stage_gets_media_clock_and_all_unchanged_candidate_locators(self):
        before = copy.deepcopy(self.source)
        for stage, messages in self.prompts():
            with self.subTest(stage=stage):
                self.assertEqual([message["role"] for message in messages], ["system", "user"])
                system = " ".join(messages[0]["content"].lower().split())
                self.assertIn("repair", system)
                self.assertRegex(system, r"ignore.{0,100}repair|repair.{0,100}ignore")
                self.assertRegex(system, r"do not.{0,100}(rescale|stretch|speed)|never.{0,100}rescale")
                blocks = messages[1]["content"]
                overview = json.loads(blocks[0]["text"])
                self.assertEqual(overview["verified_source_context"], self.context)
                self.assertEqual(overview["duration_ms"], 2000)
                self.assertEqual(overview["media_timeline"]["duration_ms"], 2000)
                self.assertEqual(overview["media_timeline"]["authority"], "supplied_video_playback")
                self.assertIs(overview["media_timeline"]["repair_time_used"], False)
                self.assertEqual(overview["complete_video_frame_count"], 4)
                self.assertEqual(overview["candidate_ids"], self.ids)
                videos = [block for block in blocks if block["type"] == "input_video"]
                self.assertEqual(videos, [{"type": "input_video", "input_video": {"url": "file://" + self.video_name}}])
                images = [block["image_url"]["url"] for block in blocks if block["type"] == "image_url"]
                self.assertEqual(images, ["file://" + self.names[frame_id] for frame_id in self.ids])
                locators = [json.loads(block["text"].removeprefix("Candidate evidence: "))
                            for block in blocks if block["type"] == "text"
                            and block["text"].startswith("Candidate evidence: ")]
                self.assertEqual(len(locators), 4)
                for locator, canonical in zip(locators, self.frames):
                    self.assertEqual(locator, {key: canonical.get(key) for key in locator})
                    self.assertEqual(locator["timestamp_ms"], canonical["frame_index"] * 500)
                    self.assertEqual(locator["timestamp_basis"], "reconstructed_nominal")
                if stage == "gap_audit":
                    self.assertEqual(overview["maximum_request_span_ms"], 1000)
        self.assertEqual(self.source, before)

    def review(self, end_ms):
        return {"scene_summary": "Colored fields change.", "context_check": "uncertain",
                "decisions": {frame_id: {"decision": "keep", "reason": "A distinct color."} for frame_id in self.ids},
                "searches": [{"start_ms": 1500, "end_ms": end_ms,
                              "question": "Inspect the last visible color.", "replace_frame_id": None}],
                "ready": False}

    def test_selection_and_gap_retrieval_use_playback_endpoint_and_source_frames(self):
        schemas = [review_schema(self.ids, media_timeline(self.source)["duration_ms"]),
                   _audit_schema(self.ids, self.source, GapConfig())]
        for schema in schemas:
            Draft202012Validator(schema).validate(self.review(2000))
            for outside in (2000.001, self.source["recorded_repair_time_ms"]):
                with self.subTest(outside=outside), self.assertRaises(ValidationError):
                    Draft202012Validator(schema).validate(self.review(outside))
        result, receipts = retrieve_requests(self.source, self.review(2000)["searches"], self.ids[:3], 3)
        self.assertEqual(result, [self.frames[-1]])
        self.assertEqual(receipts[0]["returned_frame_ids"], [self.ids[-1]])
        self.assertEqual(result[0]["timestamp_ms"], 1500)
        for outside in (2000.001, self.source["recorded_repair_time_ms"]):
            with self.subTest(direct_retrieval_outside=outside), self.assertRaises(VideoSourceError):
                retrieve_requests(self.source, self.review(outside)["searches"], self.ids[:3], 3)

    def test_duration_relabeling_is_rejected_even_when_every_frame_timestamp_is_unchanged(self):
        self.source["duration_ms"] = self.source["recorded_repair_time_ms"]
        self.assertEqual([frame["timestamp_ms"] for frame in self.frames], [0, 500, 1000, 1500])
        with self.assertRaises(VideoSourceError):
            validate_native_timeline(self.source)
        with self.assertRaises(VideoSourceError):
            media_timeline(self.source)
        for stage in ("selection", "ranking", "gap_audit"):
            with self.subTest(stage=stage), self.assertRaises(VideoSourceError):
                # Construct just this stage: generators fail at the requested prompt.
                if stage == "selection":
                    initial_messages(self.source, self.frames, [], self.names, self.video_name, SelectionConfig())
                else:
                    gap_messages(self.source, self.frames, self.names, self.video_name, GapConfig(), ranking=stage == "ranking")


if __name__ == "__main__":
    unittest.main()
