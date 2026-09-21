"""Independent claim grounding and target/context separation are enforceable."""
import copy
import unittest

from jsonschema import Draft202012Validator

from yasargil.contract import ContractError
from yasargil.medgemma_annotation_contract import (
    ANNOTATION_SYSTEM, annotation_schema, build_annotation, validate_annotation,
)


def packet():
    return {"target_frame_id": "f1", "target": {"frame_id": "f1", "timestamp_ms": 1000},
            "procedure_context": "", "frames": [{"frame_id": f"f{i}"} for i in range(3)],
            "views": [{"view_id": "f1:full", "frame_id": "f1", "role": "target"},
                      {"view_id": "f1:detail:1", "frame_id": "f1", "role": "target_detail"},
                      {"view_id": "f1:detail:2", "frame_id": "f1", "role": "target_detail"},
                      {"view_id": "f0:full", "frame_id": "f0", "role": "context_before"},
                      {"view_id": "f2:full", "frame_id": "f2", "role": "context_after"}]}


def response():
    return {"target_frame_id": "f1", "visibility": "partial", "claims": [
        {"claim_id": "c1", "category": "instrument", "statement": "A metal instrument is visible.",
         "support": "target_visible", "evidence_view_ids": ["f1:full", "f1:detail:1"], "uncertainty": ""},
        {"claim_id": "c2", "category": "action", "statement": "The instrument approaches the tissue edge.",
         "support": "context_supported", "evidence_view_ids": ["f0:full", "f1:full"],
         "uncertainty": "The intervening trajectory is unobserved."}], "unresolved_questions": []}


class AnnotationContractTests(unittest.TestCase):
    def test_strict_schema_and_independent_prompt(self):
        Draft202012Validator.check_schema(annotation_schema(packet()))
        raw = response()
        self.assertEqual(validate_annotation(raw, packet()), raw)
        self.assertIn("independent", ANNOTATION_SYSTEM)
        self.assertIn("ONE selected", ANNOTATION_SYSTEM)
        self.assertNotIn("Retain correct", ANNOTATION_SYSTEM)

    def test_captions_derive_only_from_their_support_partition(self):
        evidence, raw = packet(), response()
        before = copy.deepcopy((evidence, raw))
        result = build_annotation(raw, evidence)
        self.assertEqual(result["visible_observation"], "A metal instrument is visible.")
        self.assertEqual(result["contextual_observation"],
                         "The instrument approaches the tissue edge. (Uncertainty: The intervening trajectory is unobserved.)")
        self.assertEqual(result["claims"][0]["evidence_frame_ids"], ["f1"])
        self.assertEqual(result["claims"][1]["evidence_frame_ids"], ["f0", "f1"])
        self.assertEqual(result["target_claim_ids"], ["c1"])
        self.assertEqual(result["context_claim_ids"], ["c2"])
        self.assertTrue(result["review_required"])
        self.assertFalse(result["training_eligible"])
        self.assertEqual((evidence, raw), before)

    def test_unknown_citations_duplicate_claim_ids_and_provenance_fabrication_fail(self):
        cases = []
        for value in (["f99:full"], [], ["f1:full", "f1:full"]):
            raw = response()
            raw["claims"][0]["evidence_view_ids"] = value
            cases.append(raw)
        raw = response()
        raw["claims"][1]["claim_id"] = "c1"
        cases.append(raw)
        raw = response()
        raw["target_frame_id"] = "f2"
        cases.append(raw)
        for accessor in (lambda raw: raw, lambda raw: raw["claims"][0]):
            raw = response()
            accessor(raw)["training_eligible"] = True
            cases.append(raw)
        raw = response()
        raw["visible_observation"] = "A fabricated caption."
        cases.append(raw)
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                validate_annotation(raw, packet())

    def test_all_response_fields_are_required(self):
        for accessor in (lambda raw: raw, lambda raw: raw["claims"][0]):
            for key in accessor(response()):
                raw = response()
                del accessor(raw)[key]
                with self.subTest(key=key), self.assertRaises(ContractError):
                    validate_annotation(raw, packet())

    def test_context_cannot_launder_into_target_and_every_claim_refers_to_target(self):
        for citations in (["f0:full", "f1:full"], ["f0:full"]):
            raw = response()
            raw["claims"][0]["evidence_view_ids"] = citations
            with self.subTest(citations=citations), self.assertRaises(ContractError):
                validate_annotation(raw, packet())
        raw = response()
        raw["claims"][1]["evidence_view_ids"] = ["f0:full", "f2:full"]
        with self.assertRaisesRegex(ContractError, "target view"):
            validate_annotation(raw, packet())

    def test_crops_do_not_supply_temporal_evidence(self):
        raw = response()
        raw["claims"][1]["evidence_view_ids"] = ["f1:full", "f1:detail:1", "f1:detail:2"]
        with self.assertRaisesRegex(ContractError, "distinct source frames"):
            validate_annotation(raw, packet())
        raw["claims"][1]["support"] = "target_visible"
        with self.assertRaisesRegex(ContractError, "context-supported"):
            validate_annotation(raw, packet())

    def test_context_only_target_requires_documented_context_and_step_uncertainty(self):
        raw = response()
        raw["claims"][1].update(category="procedure_step", evidence_view_ids=["f1:full"])
        with self.assertRaisesRegex(ContractError, "documented procedure"):
            validate_annotation(raw, packet())
        evidence = packet()
        evidence["procedure_context"] = "Documented cadaveric repair simulation"
        validate_annotation(raw, evidence)
        raw["claims"][1]["uncertainty"] = ""
        with self.assertRaisesRegex(ContractError, "specific uncertainty"):
            validate_annotation(raw, evidence)

    def test_unusable_evidence_can_abstain_but_must_identify_missing_evidence(self):
        raw = {"target_frame_id": "f1", "visibility": "uninterpretable", "claims": [],
               "unresolved_questions": [{"question": "Can the target field be resolved?",
                                          "reason": "The target is obscured.", "kind": "target_detail"}]}
        built = build_annotation(raw, packet())
        self.assertEqual(built["visible_observation"], "")
        self.assertEqual(built["status"], "needs_more_evidence")
        for key in ("question", "reason", "kind"):
            invalid = copy.deepcopy(raw)
            del invalid["unresolved_questions"][0][key]
            with self.subTest(key=key), self.assertRaises(ContractError):
                validate_annotation(invalid, packet())
        raw["unresolved_questions"] = []
        with self.assertRaises(ContractError):
            validate_annotation(raw, packet())
        raw = response()
        raw["visibility"] = "poor"
        with self.assertRaisesRegex(ContractError, "unresolved question"):
            validate_annotation(raw, packet())

    def test_blank_claims_questions_and_uncertainty_are_rejected(self):
        for field in ("statement", "uncertainty"):
            raw = response()
            raw["claims"][0][field] = " "
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_annotation(raw, packet())
        raw = response()
        raw["unresolved_questions"] = [{"question": " ", "reason": "Missing detail", "kind": "target_detail"}]
        with self.assertRaises(ContractError):
            validate_annotation(raw, packet())

    def test_packet_view_roles_cannot_relabel_a_context_frame_as_target(self):
        for role in ("target", "target_detail"):
            evidence = packet()
            evidence["views"][-1]["role"] = role
            with self.subTest(role=role), self.assertRaisesRegex(ContractError, "view role"):
                annotation_schema(evidence)


if __name__ == "__main__":
    unittest.main()
