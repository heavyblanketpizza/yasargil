"""Frame citations cannot invent a playback time or relabel source observations."""
import copy
import json
import unittest

from jsonschema import Draft202012Validator, ValidationError

from test_annotation_contract import response, source_fixture
from yasargil.annotation_contract import FRAME_EVIDENCE, annotation_schema, build_annotations
from yasargil.contract import ContractError


class FrameEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.source = source_fixture()
        self.selected = [self.source["frames"][1]]

    def answer(self, start="f0", end="f2"):
        raw = response([self.selected[0]["frame_id"]])
        row = next(iter(raw["annotations"].values()))
        row["contextual_claims"][0]["evidence_intervals"] = [
            {"start_frame_id": start, "end_frame_id": end}]
        return raw

    def schema(self):
        return annotation_schema([frame["frame_id"] for frame in self.selected], self.source["duration_ms"],
                                 source_frames=self.source["frames"], evidence_mode=FRAME_EVIDENCE)

    def build(self, raw=None, **kwargs):
        return build_annotations(self.answer() if raw is None else raw, self.selected, self.source,
                                 evidence_mode=FRAME_EVIDENCE, **kwargs)

    def interval(self, document):
        return document["annotations"][0]["contextual_claims"][0]["evidence_intervals"][0]

    def test_schema_only_allows_existing_frame_ids_and_never_model_written_times(self):
        schema = self.schema()
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        validator.validate(self.answer())
        for interval in ({"start_ms": 0, "end_ms": 2000},
                         {"start_frame_id": "f0", "end_frame_id": "f4"},
                         {"start_frame_id": True, "end_frame_id": "f2"},
                         {"start_frame_id": "f0", "end_frame_id": "f2", "end_ms": 1260000}):
            raw = self.answer()
            raw["annotations"]["f1"]["contextual_claims"][0]["evidence_intervals"] = [interval]
            with self.subTest(interval=interval):
                with self.assertRaises(ValidationError):
                    validator.validate(raw)
                with self.assertRaises(ContractError):
                    self.build(raw)
        # A single shared enum keeps the schema size independent of target
        # count times source count, and produces one reusable native grammar.
        self.assertEqual(json.dumps(schema).count('"f3"'), 1)

    def test_unselected_source_frames_are_valid_evidence_and_inputs_remain_unchanged(self):
        raw = self.answer("f2", "f3")
        before = copy.deepcopy((raw, self.source, self.selected))
        document = self.build(raw)
        interval = self.interval(document)
        self.assertEqual(interval["supporting_frames"], self.source["frames"][2:])
        self.assertEqual((interval["start_ms"], interval["end_ms"]), (2000, 3000))
        self.assertEqual((raw, self.source, self.selected), before)
        self.assertEqual(document["schema_version"], "contextual-frame-annotations-v2")
        self.assertEqual(document["evidence_validation"], "locator_only_not_semantic")
        self.assertFalse(document["training_eligible"])
        interval["supporting_frames"][0]["timestamp_ms"] = 999999
        self.assertEqual(self.source, before[1])

    def test_original_pts_and_sparse_frame_indices_are_looked_up_without_rounding(self):
        for frame, index, timestamp in zip(self.source["frames"], [0, 3, 9, 17], [0, 33.375, 91.125, 122.5]):
            frame.update(frame_index=index, timestamp_ms=timestamp, timestamp_basis="source_pts",
                         source_pts=int(timestamp * 8), time_base="1/8000", source_timestamp_ms=timestamp,
                         release_frame_index=None)
        self.source["duration_ms"] = 150
        interval = self.interval(self.build(self.answer("f1", "f3")))
        self.assertEqual((interval["start_ms"], interval["end_ms"]), (33.375, 122.5))
        self.assertEqual(interval["supporting_frames"], self.source["frames"][1:])

    def test_single_frame_citations_cover_first_and_last_observation_without_duration(self):
        for frame in (self.source["frames"][0], self.source["frames"][-1]):
            with self.subTest(frame=frame["frame_id"]):
                interval = self.interval(self.build(self.answer(frame["frame_id"], frame["frame_id"])))
                self.assertEqual((interval["start_ms"], interval["end_ms"]), (frame["timestamp_ms"],) * 2)
                self.assertEqual(interval["supporting_frames"], [frame])

    def test_reversed_and_overlong_spans_are_rejected_instead_of_repaired(self):
        for start, end in (("f2", "f1"), ("f0", "f3")):
            with self.subTest(start=start, end=end), self.assertRaises(ContractError):
                self.build(self.answer(start, end), max_evidence_span_ms=2000)

    def test_2054_clip_cannot_produce_2100_as_an_evidence_endpoint(self):
        template = self.source["frames"][0]
        self.source["frames"] = [{**copy.deepcopy(template), "frame_id": f"f{i}", "frame_index": i,
                                  "release_frame_index": i + 1, "timestamp_ms": i * 1000}
                                 for i in range(1254)]
        self.source.update(expected_video_frames=1254, duration_ms=1254000)
        self.selected = [self.source["frames"][-1]]
        interval = self.interval(self.build(self.answer("f1250", "f1253")))
        self.assertEqual((interval["start_ms"], interval["end_ms"]), (1250000, 1253000))
        with self.assertRaises(ContractError):
            self.build(self.answer("f1250", "f1260"))

    def test_unsupported_claim_can_be_omitted(self):
        raw = self.answer()
        raw["annotations"]["f1"]["contextual_claims"] = []
        self.assertEqual(self.build(raw)["annotations"][0]["contextual_claims"], [])


if __name__ == "__main__":
    unittest.main()
