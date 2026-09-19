"""Structural guards for the v2 enhancement archive.

Fixtures below are synthetic, in-memory unit-test data. Their review fields are
not actual human reviews, and no fixture is exported or written as an accepted
annotation. Reference resolution, source-byte checks, factual support, timing
exposure and substantive training eligibility belong to the semantic validator.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator, FormatChecker


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "enhancement-record.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())


def fixture_asset(asset_id: str, role: str) -> dict:
    return {
        "asset_id": asset_id,
        "role": role,
        "origin": "original_dataset",
        "location": f"unit-test-only/{asset_id}",
        "sha256": "0" * 64,
        "media_type": "application/json",
        "derived_from_asset_ids": [],
        "transformation_id": None,
    }


def fixture_claim() -> dict:
    return {
        "claim_id": "claim.unit-test",
        "text": "Synthetic unit-test claim; no clinical assessment occurred.",
        "type": "visible_observation",
        "origin": "model_generated",
        "contribution": "adds_proposed_observation",
        "generation_run_id": "generation.unit-test",
        "transformation_id": None,
        "supersedes_claim_id": None,
        "evidence": {
            "frame_ids": [],
            "original_annotation_ids": [],
            "supporting_claim_ids": [],
            "reference_asset_ids": [],
            "regions": [],
        },
        "output_locations": [],
        "review_ids": [],
        "disposition": "pending",
    }


def fixture_draft() -> dict:
    """An incomplete but structurally meaningful archive, awaiting enrichment."""
    frame_asset = fixture_asset("asset.frame", "released_frame")
    frame_asset.update(media_type="image/jpeg", width=1920, height=1080)
    return {
        "schema_version": "2.0.0",
        "record_id": "record.unit-test-only",
        "revision": {
            "revision_id": "revision.unit-test",
            "parent_revision_id": None,
            "created_at": "2026-09-10T00:00:00Z",
        },
        "status": "draft",
        "intended_use": "causal_intraoperative_assistance",
        "source": {
            "dataset_name": "SOSpine",
            "dataset_version": "unit-test-only",
            "dataset_uri": "https://example.org/unit-test-only",
            "setting": "cadaveric_simulation",
            "procedure": "unit-test-only",
            "case_id": "S1A1",
            "surgeon_id": "S1",
            "patient_id": None,
            "institution_id": None,
            "source_manifest_asset_id": "asset.manifest",
            "license_evidence": {
                "status": "unresolved",
                "source_asset_ids": [],
                "notes": "Unit-test fixture; no licensing determination.",
            },
        },
        "assets": [fixture_asset("asset.manifest", "source_manifest"), frame_asset],
        "original_annotations": [],
        "frame_selection": {
            "sequence_id": "sequence.unit-test",
            "video_asset_id": None,
            "released_fps": 1,
            "method": "all_available",
            "selector_version": "unit-test-only",
            "selector_exposure": "causal_prefix_only",
            "parameters": {},
            "frames": [{
                "frame_id": "frame.unit-test",
                "asset_id": "asset.frame",
                "frame_index": 1,
                "timestamp_ms": None,
                "timestamp_basis": "unavailable",
                "role": "keyframe",
                "selection_reason": "Synthetic structural test.",
                "context_frame_ids": [],
            }],
            "known_sampling_limitations": ["No actual image is loaded by this fixture."],
        },
        "transformations": [],
        "generation_runs": [],
        "claims": [],
        "source_conflicts": [],
        "reviews": [],
        "enrichment_quality": {
            "status": "not_assessed",
            "protocol_id": None,
            "review_ids": [],
            "measurements": [],
        },
        "case_outcomes": [],
        "patient_risk_adjustment": {
            "status": "unavailable",
            "reason": "cadaveric_simulation_without_patient_baseline_or_recovery",
        },
        "training_view": {
            "dialogue_origin": "synthetic_training_dialogue",
            "eligibility": "pending_review",
            "exclusion_reasons": [],
            "assistant_format": "plain_text",
            "objective": "supervised_fine_tuning",
            "loss_scope": "final_assistant_turn",
            "messages": [],
            "turn_links": [],
        },
        "partition": {
            "name": "unassigned",
            "grouping_strategy": "surgeon",
            "group_ids": ["S1"],
            "split_manifest_asset_id": None,
        },
        "training_value": {"status": "not_evaluated", "study_links": []},
    }


def fixture_structurally_eligible() -> dict:
    """Populate shape prerequisites; this is not clinically eligible training data."""
    record = fixture_draft()
    record["status"] = "reviewed"
    record["partition"].update(name="train", split_manifest_asset_id="asset.split")
    record["assets"].append(fixture_asset("asset.split", "split_manifest"))
    claim = fixture_claim()
    claim.update(origin="expert_authored", generation_run_id=None, disposition="retained")
    claim["evidence"]["frame_ids"] = ["frame.unit-test"]
    claim["review_ids"] = ["review.unit-test"]
    record["claims"] = [claim]
    record["reviews"] = [{
        "review_id": "review.unit-test",
        "reviewer_id": "unit-test-only-not-a-real-reviewer",
        "reviewer_role": "trained_annotator",
        "reviewed_at": "2026-09-10T00:00:00Z",
        "rubric_id": "rubric.unit-test",
        "rubric_version": "unit-test-only",
        "target_claim_ids": ["claim.unit-test"],
        "target_message_indices": [1],
        "outcome_hidden_during_review": True,
        "generator_identity_hidden": True,
        "verdict": "supported",
        "evidence_adequacy": "adequate",
        "temporal_correctness": "correct",
        "clinical_appropriateness": "not_applicable",
        "question_answerability": "from_supplied_input",
        "adds_useful_supervision": "yes",
        "error_severity": "none",
        "correction_time_seconds": None,
        "replacement_claim_ids": [],
        "notes": "Ephemeral unit-test fixture. No human review took place.",
    }]
    record["training_view"].update(
        eligibility="eligible",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Unit-test input."},
                {"type": "image", "image": "unit-test-only/asset.frame"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": claim["text"]}]},
        ],
        turn_links=[{
            "turn_id": "turn.unit-test",
            "user_message_index": 0,
            "assistant_message_index": 1,
            "task": "instrument_identification",
            "input_mode": "causal_prefix",
            "cutoff_frame_index": 1,
            "student_frame_ids": ["frame.unit-test"],
            "student_annotation_ids": [],
            "student_reference_asset_ids": [],
            "expressed_claim_ids": ["claim.unit-test"],
            "question_origin": "deterministic_template",
            "generation_run_ids": [],
            "review_ids": ["review.unit-test"],
        }],
    )
    return record


class EnhancementSchemaTests(unittest.TestCase):
    def assert_valid(self, value: dict, definition: str | None = None) -> None:
        validator = VALIDATOR.evolve(schema={"$ref": f"#/$defs/{definition}"}) if definition else VALIDATOR
        errors = list(validator.iter_errors(value))
        self.assertFalse(errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors))

    def assert_invalid(self, value: dict, definition: str | None = None) -> None:
        validator = VALIDATOR.evolve(schema={"$ref": f"#/$defs/{definition}"}) if definition else VALIDATOR
        self.assertTrue(list(validator.iter_errors(value)), "Unexpectedly accepted invalid structure")

    def test_schema_is_valid_draft_2020_12(self):
        self.assertEqual(SCHEMA["$schema"], "https://json-schema.org/draft/2020-12/schema")
        Draft202012Validator.check_schema(SCHEMA)

    def test_incomplete_draft_does_not_require_invented_reviews_or_dialogue(self):
        self.assert_valid(fixture_draft())

    def test_pending_proposal_can_have_empty_evidence(self):
        self.assert_valid(fixture_claim(), "claim")

    def test_model_claim_requires_a_typed_generation_run_link(self):
        for invalid_id in (None, 7, {}, [], "invalid id"):
            with self.subTest(generation_run_id=invalid_id):
                claim = fixture_claim()
                claim["generation_run_id"] = invalid_id
                self.assert_invalid(claim, "claim")

    def test_deterministic_claim_requires_its_transformation_link(self):
        claim = fixture_claim()
        claim.update(origin="deterministic_derivation", generation_run_id=None)
        self.assert_invalid(claim, "claim")
        claim["transformation_id"] = "transformation.unit-test"
        self.assert_valid(claim, "claim")

    def test_timestamps_agree_with_their_declared_basis(self):
        for basis in ("unavailable", "estimated_from_sampling", "verified_original_pts"):
            for timestamp in (None, 0, -1):
                with self.subTest(basis=basis, timestamp=timestamp):
                    frame = fixture_draft()["frame_selection"]["frames"][0]
                    frame.update(timestamp_basis=basis, timestamp_ms=timestamp)
                    valid = (basis == "unavailable" and timestamp is None) or (
                        basis != "unavailable" and timestamp == 0
                    )
                    (self.assert_valid if valid else self.assert_invalid)(frame, "frame")

    def test_geometry_requires_correct_coordinate_counts_and_bounds(self):
        region = {
            "frame_id": "frame.unit-test",
            "type": "point_xy",
            "coordinate_system": "normalized_original_image_edges_0_1",
            "coordinates": [0.2, 0.3],
            "derivation": "model_proposed_geometry",
        }
        for kind, good in (("point_xy", [0.2, 0.3]), ("bbox_xyxy", [0.1, 0.2, 0.7, 0.8])):
            candidate = {**region, "type": kind, "coordinates": good}
            self.assert_valid(candidate, "region")
            for coordinates in (good[:-1], good + [0.9], [-0.1] + good[1:], [1.1] + good[1:]):
                with self.subTest(kind=kind, coordinates=coordinates):
                    self.assert_invalid({**candidate, "coordinates": coordinates}, "region")

    def test_sospine_cannot_be_relabeled_as_clinical_patient_data(self):
        for field, value in (("setting", "clinical_recording"), ("patient_id", "patient.invented")):
            with self.subTest(field=field):
                record = fixture_draft()
                record["source"][field] = value
                self.assert_invalid(record)

    def test_sospine_outcomes_remain_simulated_technical_endpoints(self):
        record = fixture_draft()
        outcome = {
            "outcome_id": "outcome.unit-test",
            "scope": "simulated_technical_outcome",
            "endpoint": "unit-test-endpoint",
            "status": "measured",
            "value": False,
            "unit": "boolean",
            "measurement_timepoint": "unit-test-only",
            "followup_days": None,
            "source_locator": {"asset_id": "asset.manifest", "locator_type": "whole_asset", "locator": ""},
            "missing_reason": None,
        }
        record["case_outcomes"] = [outcome]
        self.assert_valid(record)
        outcome["scope"] = "clinical_patient_outcome"
        self.assert_invalid(record)

    def test_sospine_cannot_claim_a_patient_risk_estimate(self):
        risk = {
            "status": "research_estimate",
            "baseline_annotation_ids": ["annotation.unit-test-baseline"],
            "baseline_model": {
                "name": "unit-test-only",
                "version": "unit-test-only",
                "applicable_population": "Synthetic fixture only.",
                "evaluation_reference": "unit-test-only",
                "calibration_reference": "unit-test-only",
            },
            "preoperative_information_only": True,
            "predictions_out_of_sample": True,
            "comparisons": [{
                "outcome_id": "outcome.unit-test",
                "prediction_type": "event_probability",
                "expected_value": 0.25,
                "observed_value_numeric": 1,
                "observed_minus_expected": 0.75,
                "unit": "probability",
                "higher_value_means": "worse_outcome",
                "computed_at": "2026-09-10T00:00:00Z",
                "method_reference": "unit-test-only",
            }],
            "interpretation": "descriptive_case_comparison_not_proof_of_action_quality_or_text_correctness",
        }
        self.assert_valid(risk, "patient_risk_adjustment")
        record = fixture_draft()
        record["patient_risk_adjustment"] = risk
        self.assert_invalid(record)

    def test_sospine_risk_unavailability_has_the_specific_reason(self):
        record = fixture_draft()
        record["patient_risk_adjustment"]["reason"] = "not_yet_assessed"
        self.assert_invalid(record)

    def test_reviewed_eligibility_shape_is_representable_without_automatic_acceptance(self):
        self.assert_valid(fixture_structurally_eligible())

    def test_nonreviewed_status_cannot_be_training_eligible(self):
        for status in ("draft", "in_review", "rejected"):
            with self.subTest(status=status):
                record = fixture_structurally_eligible()
                record["status"] = status
                self.assert_invalid(record)

    def test_eligible_records_need_partition_claims_reviews_and_dialogue(self):
        complete = fixture_structurally_eligible()
        mutations = {
            "unassigned_partition": lambda r: r["partition"].update(name="unassigned", split_manifest_asset_id=None),
            "no_claims": lambda r: r.update(claims=[]),
            "no_reviews": lambda r: r.update(reviews=[]),
            "no_messages": lambda r: r["training_view"].update(messages=[]),
            "no_turn_links": lambda r: r["training_view"].update(turn_links=[]),
            "exclusion_still_present": lambda r: r["training_view"].update(exclusion_reasons=["Unresolved exclusion."]),
        }
        for label, mutate in mutations.items():
            with self.subTest(missing=label):
                record = deepcopy(complete)
                mutate(record)
                self.assert_invalid(record)


if __name__ == "__main__":
    unittest.main()
