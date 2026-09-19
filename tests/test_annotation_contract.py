"""Annotation locators must bind to source observations without becoming truth."""
import copy
import unittest

from jsonschema import Draft202012Validator, ValidationError

from yasargil.annotation_contract import annotation_schema, build_annotations
from yasargil.contract import ContractError


def source_fixture():
    frames = [{"frame_id": f"f{index}", "frame_index": index, "release_frame_index": index + 1,
               "source_path": f"/release/frame-{index}.jpeg", "source_sha256": f"{index:064x}",
               "image_path": f"/decoded/frame-{index}.png", "image_sha256": f"{index + 10:064x}",
               "timestamp_ms": index * 1000.0, "timestamp_basis": "reconstructed_nominal",
               "source_pts": None, "time_base": None, "source_timestamp_ms": None,
               "source_acquisition_time": None, "video_pts": index * 16000,
               "video_time_base": "1/16000", "width": 320, "height": 240} for index in range(4)]
    return {"source_path": "/release", "source_sha256": "f" * 64, "duration_ms": 4000,
            "expected_video_frames": 4, "frames": frames}


def response(ids=("f1",)):
    return {"context_check": "consistent", "annotations": {frame_id: {
        "visible_observation": "A colored field is visible.", "visibility": "clear",
        "contextual_claims": [{"claim": "The field changes color in the surrounding video.",
                               "evidence_intervals": [{"start_ms": 0.5, "end_ms": 2000}]}],
        "uncertainties": [],
    } for frame_id in ids}}


class AnnotationContractTests(unittest.TestCase):
    def setUp(self):
        self.source = source_fixture()
        self.selected = [copy.deepcopy(self.source["frames"][1])]

    def build(self, raw=None, **kwargs):
        return build_annotations(response() if raw is None else raw, self.selected, self.source, **kwargs)

    def test_schema_requires_exact_ids_and_rejects_extra_fields_at_each_level(self):
        schema = annotation_schema(["f1"], 4000)
        Draft202012Validator.check_schema(schema)
        raw = response()
        Draft202012Validator(schema).validate(raw)
        cases = []
        for target in (lambda value: value, lambda value: value["annotations"],
                       lambda value: value["annotations"]["f1"],
                       lambda value: value["annotations"]["f1"]["contextual_claims"][0],
                       lambda value: value["annotations"]["f1"]["contextual_claims"][0]["evidence_intervals"][0]):
            changed = copy.deepcopy(raw)
            target(changed)["invented"] = "not permitted"
            cases.append(changed)
        missing = response()
        del missing["annotations"]["f1"]
        cases.extend([missing, response(["unknown"]), response(["f1", "f2"])])
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(ValidationError):
                Draft202012Validator(schema).validate(changed)
            with self.assertRaises(ContractError):
                self.build(changed)

    def test_source_evidence_is_exact_and_preserves_unknown_capture_times(self):
        raw = response()
        before = copy.deepcopy((raw, self.selected, self.source))
        result = self.build(raw)
        annotation = result["annotations"][0]
        self.assertEqual({key: annotation[key] for key in self.selected[0]}, self.selected[0])
        interval = annotation["contextual_claims"][0]["evidence_intervals"][0]
        self.assertEqual((interval["start_ms"], interval["end_ms"]), (0.5, 2000))
        self.assertEqual(interval["supporting_frames"], self.source["frames"][1:3])
        self.assertNotIn(0.5, [frame["timestamp_ms"] for frame in interval["supporting_frames"]])
        self.assertTrue(all(frame["source_acquisition_time"] is None for frame in interval["supporting_frames"]))
        self.assertEqual((raw, self.selected, self.source), before)
        self.assertEqual(result, self.build(raw))
        self.source["frames"][1]["image_path"] = "/changed.png"
        self.assertEqual(interval["supporting_frames"][0]["image_path"], before[2]["frames"][1]["image_path"])
        self.assertEqual(annotation["image_path"], before[1][0]["image_path"])

    def test_unsupported_intervals_nonfinite_numbers_and_overbroad_evidence_fail(self):
        intervals = [(100, 900), (1000, 1000), (2000, 1000), (-1, 2000), (3000, 4001),
                     (0, 3001), (float("nan"), 1000), (0, float("nan")),
                     (0, float("inf")), (True, 1000), (0, False)]
        for start, end in intervals:
            raw = response()
            raw["annotations"]["f1"]["contextual_claims"][0]["evidence_intervals"] = [{"start_ms": start, "end_ms": end}]
            with self.subTest(start=start, end=end), self.assertRaises(ContractError):
                self.build(raw, max_evidence_span_ms=3000)

    def test_poor_visibility_requires_uncertainty_and_abstention_is_preserved(self):
        for visibility in ("poor", "uninterpretable"):
            raw = response()
            raw["annotations"]["f1"].update(visibility=visibility, contextual_claims=[])
            with self.subTest(visibility=visibility), self.assertRaises(ContractError):
                self.build(raw)
            raw["annotations"]["f1"]["uncertainties"] = ["The image is obscured; the action cannot be determined."]
            annotation = self.build(raw)["annotations"][0]
            self.assertEqual(annotation["contextual_claims"], [])
            self.assertEqual(annotation["uncertainties"], raw["annotations"]["f1"]["uncertainties"])

    def test_context_conflicts_and_drafts_never_become_training_eligible(self):
        for context in ("consistent", "uncertain", "conflict", "not_supplied"):
            raw = response()
            raw["context_check"] = context
            result = self.build(raw)
            self.assertEqual(result["context_check"], context)
            self.assertEqual(result["clinical_validation"], "not_performed")
            self.assertEqual(result["temporal_exposure"], "retrospective_full_video")
            self.assertEqual(result["evidence_validation"], "locator_only_not_semantic")
            self.assertFalse(result["training_eligible"])
            self.assertTrue(result["annotations"][0]["review_required"])
            self.assertFalse(result["annotations"][0]["training_eligible"])

    def test_selected_provenance_cannot_be_edited_or_invented(self):
        for key, replacement in {"frame_id": "unknown", "source_path": "/other.jpeg", "source_sha256": "d" * 64,
                                 "image_path": "/other.png", "image_sha256": "e" * 64, "frame_index": True,
                                 "timestamp_ms": 1001, "timestamp_basis": "source_pts", "release_frame_index": 7,
                                 "source_acquisition_time": "2026-01-01T00:00:00Z", "video_pts": 0}.items():
            selected = copy.deepcopy(self.selected)
            selected[0][key] = replacement
            with self.subTest(field=key), self.assertRaises(ContractError):
                build_annotations(response(), selected, self.source)
        for selected in ([], [self.selected[0], self.selected[0]], [{}]):
            with self.subTest(selected=selected), self.assertRaises(ContractError):
                build_annotations(response(), selected, self.source)
        selected = copy.deepcopy(self.selected)
        del selected[0]["source_pts"]
        with self.assertRaises(ContractError):
            build_annotations(response(), selected, self.source)
        selected = copy.deepcopy(self.selected)
        selected[0]["selection_reason"] = "Earlier model judgment must not be copied."
        self.assertNotIn("selection_reason", build_annotations(response(), selected, self.source)["annotations"][0])

    def test_invalid_source_manifest_is_rejected(self):
        cases = [None, {}, {**self.source, "frames": []}, {**self.source, "expected_video_frames": 3},
                 {**self.source, "duration_ms": float("nan")}, {**self.source, "source_sha256": "invalid"}]
        for field, value in (("frame_id", "f0"), ("frame_index", 0), ("timestamp_ms", 0),
                             ("timestamp_ms", float("inf")), ("timestamp_ms", 4001),
                             ("source_sha256", "bad"), ("image_path", "relative.png"),
                             ("source_pts", 16000), ("timestamp_basis", "guessed"), ("timestamp_basis", [])):
            source = copy.deepcopy(self.source)
            source["frames"][1][field] = value
            cases.append(source)
        for source in cases:
            with self.subTest(source=source), self.assertRaises(ContractError):
                build_annotations(response(), self.selected, source)

    def test_source_observations_need_not_have_dense_frame_indices(self):
        source = copy.deepcopy(self.source)
        for frame in source["frames"]:
            frame["frame_index"] *= 3
        result = build_annotations(response(), [source["frames"][1]], source)
        self.assertEqual(result["annotations"][0]["frame_index"], 3)

    def test_blank_claims_and_invalid_configuration_are_rejected(self):
        for field in ("visible_observation", "uncertainties", "contextual_claims"):
            raw = response()
            row = raw["annotations"]["f1"]
            if field == "contextual_claims":
                row[field][0]["claim"] = "   "
            else:
                row[field] = ["  "] if field == "uncertainties" else "  "
            with self.subTest(field=field), self.assertRaises(ContractError):
                self.build(raw)
        for value in (0, -1, True, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ContractError):
                annotation_schema(["f1"], 4000, max_evidence_span_ms=value)
        for ids in ([], ["same", "same"], [""], None, "f1"):
            with self.subTest(ids=ids), self.assertRaises(ContractError):
                annotation_schema(ids, 4000)


if __name__ == "__main__":
    unittest.main()
