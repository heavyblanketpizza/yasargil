"""Independent claim grounding and target/context separation are enforceable."""
import copy
import unittest

from jsonschema import Draft202012Validator

from yasargil.contract import ContractError
from yasargil.medgemma_annotation_contract import (
    ANNOTATION_SYSTEM, PROTOCOL_V1, annotation_from_answer, annotation_schema, answer_schema,
    build_annotation, validate_annotation, view_labels,
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


def answer():
    """The model-facing v2 form of response(): short labels, no IDs."""
    return {"visibility": "partial", "claims": [
        {"statement": "A metal instrument is visible.", "category": "instrument",
         "support": "target_visible", "evidence_view_ids": ["T", "T1"], "uncertainty": ""},
        {"statement": "The instrument approaches the tissue edge.", "category": "action",
         "support": "context_supported", "evidence_view_ids": ["B1", "T"],
         "uncertainty": "The intervening trajectory is unobserved."}], "unresolved_questions": []}


def with_statements(statements):
    raw = response()
    raw["claims"] = [{"claim_id": f"c{number}", "category": "instrument", "statement": text,
                      "support": "target_visible", "evidence_view_ids": ["f1:full"], "uncertainty": ""}
                     for number, text in enumerate(statements, 1)]
    return raw


DISTINCT = ["A grasper is visible.", "A suture strand crosses the field.", "Blood pools inferiorly.",
            "The dura is pale.", "A ruler lies across the port.", "Glare obscures the upper rim.",
            "The tube wall is metallic.", "A needle tip is visible.", "Tissue edges are irregular.",
            "The opening is circular.", "Fluid reflects the light.", "Two instruments overlap."]


class AnnotationContractTests(unittest.TestCase):
    def test_views_get_short_labels_numbered_outward_from_the_target(self):
        self.assertEqual(view_labels(packet()), {"f1:full": "T", "f1:detail:1": "T1", "f1:detail:2": "T2",
                                                 "f0:full": "B1", "f2:full": "A1"})
        evidence = packet()
        evidence["frames"].insert(0, {"frame_id": "fa"})
        evidence["views"].insert(3, {"view_id": "fa:full", "frame_id": "fa", "role": "context_before"})
        self.assertEqual(view_labels(evidence)["fa:full"], "B2")
        self.assertEqual(view_labels(evidence)["f0:full"], "B1")

    def test_context_view_direction_must_match_its_source_position(self):
        evidence = packet()
        evidence["views"][-1]["role"] = "context_before"
        with self.assertRaisesRegex(ContractError, "before"):
            view_labels(evidence)

    def test_answer_schema_puts_assessment_and_statements_before_classification(self):
        schema = answer_schema(packet())
        Draft202012Validator.check_schema(schema)
        self.assertEqual(list(schema["properties"]), ["visibility", "claims", "unresolved_questions"])
        claim = schema["properties"]["claims"]["items"]
        self.assertEqual(list(claim["properties"]),
                         ["statement", "category", "support", "evidence_view_ids", "uncertainty"])
        self.assertEqual(claim["properties"]["evidence_view_ids"]["items"]["enum"], ["T", "T1", "T2", "B1", "A1"])

    def test_answer_maps_labels_back_and_software_assigns_identity(self):
        evidence, model_answer = packet(), answer()
        before = copy.deepcopy((evidence, model_answer))
        raw, flags = annotation_from_answer(model_answer, evidence)
        self.assertEqual(raw, response())
        self.assertEqual(flags, [])
        self.assertEqual((evidence, model_answer), before)

    def test_repeated_citations_are_dropped_and_flagged(self):
        model_answer = answer()
        model_answer["claims"][0]["evidence_view_ids"] = ["T", "T1", "T"]
        raw, flags = annotation_from_answer(model_answer, packet())
        self.assertEqual(raw["claims"][0]["evidence_view_ids"], ["f1:full", "f1:detail:1"])
        self.assertEqual(flags, ["duplicate_citations_removed"])

    def test_filling_the_claim_cap_is_flagged(self):
        model_answer = answer()
        model_answer["claims"] = [dict(answer()["claims"][0], statement=text) for text in DISTINCT]
        _, flags = annotation_from_answer(model_answer, packet())
        self.assertEqual(flags, ["claim_cap_reached"])

    def test_answers_outside_the_v2_contract_are_rejected(self):
        def unknown_label(value):
            value["claims"][0]["evidence_view_ids"] = ["X"]
        def model_written_id(value):
            value["claims"][0]["claim_id"] = "c9"
        def model_written_target(value):
            value["target_frame_id"] = "f1"
        def too_many_claims(value):
            value["claims"] = [dict(answer()["claims"][0], statement=text) for text in DISTINCT + ["One more."]]
        def overlong_statement(value):
            value["claims"][0]["statement"] = "x" * 201
        for change in (unknown_label, model_written_id, model_written_target, too_many_claims, overlong_statement):
            value = answer()
            change(value)
            with self.subTest(change=change.__name__), self.assertRaises(ContractError):
                annotation_from_answer(value, packet())

    def test_repetition_loops_are_rejected(self):
        loops = {"exact": ["The needle is positioned within a hole.", "the needle is positioned within a hole"],
                 "templated": [f"The instrument is contacting a structure marked with '{n}'." for n in (1, 2, 3)]}
        for name, statements in loops.items():
            with self.subTest(loop=name), self.assertRaisesRegex(ContractError, "Degenerate"):
                validate_annotation(with_statements(statements), packet())

    def test_similar_but_distinct_findings_are_accepted(self):
        # A real 12-claim answer whose closest pairs score 0.906 and 0.780.
        raw = with_statements([
            "A surgical instrument is visible.", "The instrument has a working end.", "The instrument has a handle.",
            "The instrument has a scale.", "The instrument is a retractor.",
            "The retractor is positioned in a surgical site.", "Tissue is visible within the surgical site.",
            "The tissue appears to be dura.", "The dura appears torn or damaged.",
            "The retractor is positioned to expose the damaged dura.", "The retractor is holding the tissue.",
            "The procedure is likely a surgical repair."])
        self.assertEqual(validate_annotation(raw, packet()), raw)

    def test_v1_records_keep_their_original_rules_and_version(self):
        raw = with_statements(DISTINCT + ["A thirteenth distinct finding.", "x" * 700])
        with self.assertRaises(ContractError):
            build_annotation(raw, packet())
        self.assertEqual(build_annotation(raw, packet(), protocol=PROTOCOL_V1)["schema_version"],
                         "medgemma-frame-annotation-v1")
        loop = with_statements([f"Marked with '{n}'." for n in range(3)])
        self.assertEqual(build_annotation(loop, packet(), protocol=PROTOCOL_V1)["status"], "annotated")

    def test_new_annotations_are_stamped_v2(self):
        self.assertEqual(build_annotation(response(), packet())["schema_version"], "medgemma-frame-annotation-v2")

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
