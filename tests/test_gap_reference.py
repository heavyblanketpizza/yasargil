"""Independent references, exact omission cohorts and conservative recovery scores."""
import copy
import unittest

from yasargil.contract import ContractError, canonical_hash
from yasargil.gap_reference import build_reference, make_conditions, ranking_schema, score_condition


def frame(index, timestamp=None):
    return {"frame_id": f"f{index}", "frame_index": index, "timestamp_ms": index * 1000 if timestamp is None else timestamp,
            "timestamp_basis": "reconstructed_nominal", "source_path": f"/release/frame_{index:08d}.jpeg",
            "source_sha256": f"{index:064x}", "image_path": f"/decoded/frame_{index:08d}.png",
            "image_sha256": f"{index + 100:064x}", "source_pts": None, "time_base": None,
            "source_timestamp_ms": None, "source_acquisition_time": None}


def reference(frames, *, groups=None, tolerance_ms=400, duration_ms=24000, version="gap-reference-v1"):
    raw = {"scene_summary": "Synthetic changing moments.", "context_check": "consistent",
           "ranking": [{"frame_id": item["frame_id"], "moment_id": (groups or list(range(1, len(frames) + 1)))[i],
                        "reason": f"Visible synthetic moment {i}."} for i, item in enumerate(frames)]}
    result = build_reference(raw, frames, duration_ms=duration_ms, tolerance_ms=tolerance_ms)
    result["schema_version"] = version
    result["reference_sha256"] = canonical_hash({key: value for key, value in result.items() if key != "reference_sha256"})
    return result


def request(start, end):
    return {"start_ms": start, "end_ms": end, "question": "Find useful evidence for this interval.", "replace_frame_id": None}


def review(ids, *, searches=(), drop=(), ready=True, retrieved=(), number=0):
    return {"round": number,
            "output": {"scene_summary": "Synthetic evidence review.", "context_check": "consistent",
                       "decisions": {fid: {"decision": "drop" if fid in drop else "keep", "reason": "Synthetic judgment."} for fid in ids},
                       "searches": list(searches), "ready": ready},
            "retrieval": [{"request": query, "returned_frame_ids": frame_ids, "status": "retrieved" if frame_ids else "no_new_evidence_within_budget"}
                          for query, frame_ids in retrieved]}


class ReferenceTests(unittest.TestCase):
    def test_one_frozen_permutation_produces_nested_chronological_conditions(self):
        indices = [23, 1, 18, 4, 20, 6, 17, 8, 9, 10, 11, 12, 13, 14, 15, 16, 7, 2, 19, 5, 21, 22, 0, 3]
        ref = reference([frame(index) for index in indices])
        conditions = make_conditions(ref)
        self.assertEqual([row["condition_id"] for row in conditions], ["bottom_50", "bottom_70", "bottom_90", "control_all"])
        self.assertEqual([row["actual_withheld_count"] for row in conditions], [12, 7, 2, 0])
        self.assertEqual([row["actual_supplied_count"] for row in conditions], [12, 17, 22, 24])
        self.assertLess(set(conditions[2]["withheld_frame_ids"]), set(conditions[1]["withheld_frame_ids"]))
        self.assertLess(set(conditions[1]["withheld_frame_ids"]), set(conditions[0]["withheld_frame_ids"]))
        for condition in conditions:
            ids = condition["supplied_frame_ids"]
            self.assertEqual(ids, sorted(ids, key=lambda fid: int(fid[1:])))
            self.assertFalse(set(ids) & set(condition["withheld_frame_ids"]))
            self.assertEqual(len(set(ids) | set(condition["withheld_frame_ids"])), 24)
            self.assertFalse(condition["protected_anchor_overrides"])

    def test_half_up_rounding_and_small_cohort_clamping(self):
        conditions = make_conditions(reference([frame(i) for i in range(5)]))
        self.assertEqual([row["actual_withheld_count"] for row in conditions], [3, 2, 1, 0])
        conditions = make_conditions(reference([frame(0), frame(1)]))
        self.assertEqual([row["actual_withheld_count"] for row in conditions], [1, 1, 1, 0])

    def test_known_equivalent_supplied_frame_removes_artificial_gap(self):
        ref = reference([frame(0), frame(8), frame(12), frame(1)], groups=[1, 2, 3, 1])
        condition = make_conditions(ref)[2]
        self.assertEqual(condition["withheld_frame_ids"], ["f0"])
        self.assertEqual(condition["missing_moment_ids"], [])
        self.assertEqual(condition["already_represented_withheld_moment_ids"], [1])

    def test_provenance_is_copied_and_intervals_clamped_without_capture_time_invention(self):
        frames = [frame(0), frame(1, timestamp=2000)]
        before = copy.deepcopy(frames)
        ref = reference(frames, tolerance_ms=1500, duration_ms=2400)
        self.assertEqual(frames, before)
        self.assertEqual((ref["moments"][0]["start_ms"], ref["moments"][1]["end_ms"]), (0, 2400))
        self.assertIsNone(ref["ranking"][0]["source_acquisition_time"])
        self.assertEqual(ref["ranking"][0]["source_path"], before[0]["source_path"])
        frames[0]["source_path"] = "/changed"
        self.assertEqual(ref["ranking"][0]["source_path"], before[0]["source_path"])
        self.assertFalse(ref["clinical_ground_truth"])
        self.assertFalse(ref["training_eligible"])
        self.assertEqual(len(ref["reference_sha256"]), 64)

    def test_duplicate_unknown_missing_candidates_and_distant_groups_rejected(self):
        frames = [frame(0), frame(12)]
        valid = {"scene_summary": "Synthetic.", "context_check": "not_supplied", "ranking": [
            {"frame_id": "f0", "moment_id": 1, "reason": "First."},
            {"frame_id": "f12", "moment_id": 2, "reason": "Second."}]}
        cases = []
        for changed in ("f0", "invented"):
            raw = copy.deepcopy(valid)
            raw["ranking"][1]["frame_id"] = changed
            cases.append(raw)
        raw = copy.deepcopy(valid)
        raw["ranking"].pop()
        cases.append(raw)
        raw = copy.deepcopy(valid)
        raw["ranking"][1]["moment_id"] = 1
        cases.append(raw)
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                build_reference(raw, frames, duration_ms=14000)
        with self.assertRaises(ContractError):
            ranking_schema(["same", "same"])
        with self.assertRaises(ContractError):
            build_reference(valid, frames, duration_ms=float("nan"))


class LeastImportantOmissionTests(unittest.TestCase):
    def test_v2_drops_nested_ranking_tails_and_supplies_chronological_prefixes(self):
        indices = [23, 1, 18, 4, 20, 6, 17, 8, 9, 10, 11, 12, 13, 14, 15, 16, 7, 2, 19, 5, 21, 22, 0, 3]
        ranked_frames = [frame(index) for index in indices]
        ref = reference(ranked_frames, version="gap-reference-v2")
        before = copy.deepcopy(ref)
        conditions = make_conditions(ref)
        self.assertEqual(ref, before)
        self.assertEqual([row["condition_id"] for row in conditions], ["drop_50", "drop_70", "drop_90", "control_all"])
        self.assertEqual([row["requested_withheld_percent"] for row in conditions], [50, 70, 90, 0])
        self.assertEqual([row["requested_supplied_percent"] for row in conditions], [50, 30, 10, 100])
        self.assertEqual([row["actual_withheld_count"] for row in conditions], [12, 17, 22, 0])
        self.assertEqual([row["actual_supplied_count"] for row in conditions], [12, 7, 2, 24])
        self.assertLess(set(conditions[0]["withheld_frame_ids"]), set(conditions[1]["withheld_frame_ids"]))
        self.assertLess(set(conditions[1]["withheld_frame_ids"]), set(conditions[2]["withheld_frame_ids"]))
        for condition in conditions:
            split = condition["actual_supplied_count"]
            self.assertEqual(condition["withheld_frame_ids"], [row["frame_id"] for row in ranked_frames[split:]])
            self.assertEqual(condition["supplied_frame_ids"], sorted(
                [row["frame_id"] for row in ranked_frames[:split]], key=lambda fid: int(fid[1:])))
            self.assertEqual(condition["omission_policy"], "drop_least_surgically_important_ranking_tail")
            self.assertFalse(condition["protected_anchor_overrides"])

    def test_v2_rounds_number_dropped_half_up_and_clamps_small_groups(self):
        conditions = make_conditions(reference([frame(i) for i in range(5)], version="gap-reference-v2"))
        self.assertEqual([row["actual_withheld_count"] for row in conditions], [3, 4, 4, 0])
        self.assertEqual([row["actual_supplied_count"] for row in conditions], [2, 1, 1, 5])
        conditions = make_conditions(reference([frame(0), frame(1)], version="gap-reference-v2"))
        self.assertEqual([row["withheld_frame_ids"] for row in conditions], [["f1"], ["f1"], ["f1"], []])

    def test_v2_equivalent_high_ranked_frame_excludes_redundant_omission_from_gaps(self):
        source = [frame(i) for i in range(13)]
        ref = reference([source[i] for i in (1, 8, 12, 0)], groups=[1, 2, 3, 1],
                        duration_ms=13000, version="gap-reference-v2")
        condition = make_conditions(ref)[0]
        self.assertEqual(condition["withheld_frame_ids"], ["f12", "f0"])
        self.assertEqual(condition["supplied_frame_ids"], ["f1", "f8"])
        self.assertEqual(condition["missing_moment_ids"], [3])
        self.assertEqual(condition["already_represented_withheld_moment_ids"], [1])
        query = request(11800, 12200)
        supplied = condition["supplied_frame_ids"]
        rounds = [review(supplied, searches=[query], ready=False, retrieved=[(query, ["f12"])]),
                  review([*supplied, "f12"], number=1)]
        score = score_condition(ref, condition, source, rounds)
        self.assertEqual(score["initial_requested"]["moment_ids"], [3])
        self.assertEqual(score["initial_requested"]["denominator"], 1)
        self.assertEqual(score["cumulative_retrieved_exact"]["recall"], 1)
        self.assertEqual(score["final_retained"]["recall"], 1)
        self.assertIsNone(score["false_completion_against_provisional_reference"])
        self.assertFalse(score["declared_complete_with_unretained_reference_moments"])

    def test_v2_recovery_scoring_targets_tail_and_distinguishes_requests_from_retention(self):
        source = [frame(i) for i in range(13)]
        ref = reference([source[i] for i in (1, 5, 9, 11)], duration_ms=13000, version="gap-reference-v2")
        condition = make_conditions(ref)[0]
        self.assertEqual(condition["withheld_frame_ids"], ["f9", "f11"])
        self.assertEqual(condition["supplied_frame_ids"], ["f1", "f5"])
        first, second = request(8800, 9200), request(10800, 11200)
        supplied = condition["supplied_frame_ids"]
        rounds = [review(supplied, searches=[first, second], ready=False,
                         retrieved=[(first, ["f9"]), (second, ["f11"])]),
                  review([*supplied, "f9", "f11"], drop=["f11"], number=1)]
        score = score_condition(ref, condition, source, rounds)
        self.assertEqual(score["initial_requested"]["moment_ids"], [3, 4])
        self.assertEqual(score["cumulative_retrieved_exact"]["recall"], 1)
        self.assertEqual(score["final_retained"]["recall"], 0.5)
        self.assertEqual(score["missed_moment_ids"], [4])
        self.assertEqual(score["schema_version"], "gap-recovery-score-v2")
        self.assertIsNone(score["false_completion_against_provisional_reference"])
        self.assertTrue(score["declared_complete_with_unretained_reference_moments"])
        self.assertIn("Zero retrieval requests", score["completion_interpretation"])
        control = make_conditions(ref)[3]
        score = score_condition(ref, control, source, [review(control["supplied_frame_ids"])])
        self.assertIsNone(score["initial_requested"]["recall"])
        self.assertIsNone(score["false_completion_against_provisional_reference"])
        self.assertFalse(score["declared_complete_with_unretained_reference_moments"])

    def test_v2_zero_requests_with_omissions_is_observation_not_automatic_failure(self):
        source = [frame(i) for i in range(4)]
        ref = reference(source, version="gap-reference-v2")
        condition = make_conditions(ref)[0]
        score = score_condition(ref, condition, source, [review(condition["supplied_frame_ids"])])
        self.assertEqual(score["request_cost"]["request_count"], 0)
        self.assertEqual(score["final_retained"]["recall"], 0)
        self.assertTrue(score["declared_complete_with_unretained_reference_moments"])
        self.assertIsNone(score["false_completion_against_provisional_reference"])
        self.assertIn("alone do not establish a failure", score["completion_interpretation"])

    def test_unknown_reference_version_does_not_silently_change_cohorts(self):
        ref = reference([frame(0), frame(1)], version="gap-reference-future")
        with self.assertRaisesRegex(ContractError, "Unknown gap reference version"):
            make_conditions(ref)


class SurgicalImportanceReferenceTests(unittest.TestCase):
    def setUp(self):
        self.frames = [frame(i) for i in (3, 1, 2, 0)]
        self.raw = {"scene_summary": "Visible activity varies in information and clarity.",
                    "context_check": "consistent", "ranking": [
                        {"frame_id": item["frame_id"], "moment_id": ordinal + 1,
                         "importance_score": score, "reason": "Distinct visible evidence for this priority."}
                        for ordinal, (item, score) in enumerate(zip(self.frames, (100, 75, 75, 0)))]}

    def build(self, raw):
        return build_reference(raw, self.frames, duration_ms=4000, surgical_importance=True)

    def test_surgical_reference_emits_scored_objective_and_frozen_hash(self):
        before = copy.deepcopy(self.raw)
        result = self.build(self.raw)
        self.assertEqual(self.raw, before)
        self.assertEqual(result["schema_version"], "gap-reference-v2")
        self.assertEqual(result["ranking_objective"], "surgical_information_value_in_complete_procedure")
        self.assertEqual(result["ranking_score_range"], [0, 100])
        self.assertEqual(result["score_order"], "descending")
        self.assertEqual([row["importance_score"] for row in result["ranking"]], [100, 75, 75, 0])
        self.assertEqual([row["frame_id"] for row in result["ranking"]], ["f3", "f1", "f2", "f0"])
        self.assertEqual(result["ranking"][0]["source_sha256"], self.frames[0]["source_sha256"])
        self.assertEqual(result["reference_sha256"], canonical_hash(
            {key: value for key, value in result.items() if key != "reference_sha256"}))
        changed = copy.deepcopy(result)
        changed["ranking"][0]["importance_score"] = 99
        with self.assertRaisesRegex(ContractError, "Frozen reference has changed"):
            score_condition(changed, make_conditions(result)[0], self.frames, [])

    def test_missing_out_of_range_and_noninteger_scores_are_rejected(self):
        cases = {}
        for value in (-1, 101, 99.5, 100.0, True, "100", None):
            changed = copy.deepcopy(self.raw)
            changed["ranking"][0]["importance_score"] = value
            cases[repr(value)] = changed
        missing = copy.deepcopy(self.raw)
        del missing["ranking"][0]["importance_score"]
        cases["missing"] = missing
        for label, raw in cases.items():
            with self.subTest(label=label), self.assertRaises(ContractError):
                self.build(raw)

    def test_flat_and_ascending_scores_are_rejected_but_partial_ties_are_valid(self):
        for scores in ((75, 75, 75, 75), (0, 30, 60, 100), (100, 20, 75, 0)):
            changed = copy.deepcopy(self.raw)
            for row, score in zip(changed["ranking"], scores):
                row["importance_score"] = score
            with self.subTest(scores=scores), self.assertRaisesRegex(ContractError, "meaningful differentiation"):
                self.build(changed)
        self.assertEqual([row["importance_score"] for row in self.build(self.raw)["ranking"]], [100, 75, 75, 0])

    def test_default_legacy_schema_and_reference_continue_accepting_unscored_rankings(self):
        raw = copy.deepcopy(self.raw)
        for row in raw["ranking"]:
            del row["importance_score"]
        result = build_reference(raw, self.frames, duration_ms=4000)
        self.assertEqual(result["schema_version"], "gap-reference-v1")
        self.assertNotIn("ranking_objective", result)
        self.assertNotIn("importance_score", result["ranking"][0])
        legacy_schema = ranking_schema([item["frame_id"] for item in self.frames])
        surgical_schema = ranking_schema([item["frame_id"] for item in self.frames], surgical_importance=True)
        self.assertNotIn("importance_score", legacy_schema["properties"]["ranking"]["items"]["properties"])
        score_property = surgical_schema["properties"]["ranking"]["items"]["properties"]["importance_score"]
        self.assertEqual(score_property, {"type": "integer", "minimum": 0, "maximum": 100})
        self.assertIn("importance_score", surgical_schema["properties"]["ranking"]["items"]["required"])
        with self.assertRaises(ContractError):
            self.build(raw)


class RecoveryScoreTests(unittest.TestCase):
    def setUp(self):
        self.source = [frame(i) for i in range(13)]
        self.ref = reference([self.source[i] for i in (1, 5, 9, 11)], duration_ms=13000)
        self.condition = make_conditions(self.ref)[0]
        self.supplied = self.condition["supplied_frame_ids"]

    def score(self, rounds, *, ref=None, condition=None):
        return score_condition(ref or self.ref, condition or self.condition, self.source, rounds)

    def test_request_retrieval_and_final_retention_are_separate(self):
        first, second = request(900, 1500), request(4800, 5200)
        rounds = [review(self.supplied, searches=[first, second], ready=False,
                         retrieved=[(first, ["f1"]), (second, ["f5"])]),
                  review([*self.supplied, "f1", "f5"], drop=["f5"], number=1)]
        score = self.score(rounds)
        self.assertEqual(score["initial_requested"]["recall"], 1)
        self.assertEqual(score["cumulative_retrieved_exact"]["recall"], 1)
        self.assertEqual(score["final_retained"]["recall"], 0.5)
        self.assertEqual(score["missed_moment_ids"], [2])
        self.assertTrue(score["false_completion_against_provisional_reference"])
        self.assertEqual(score["request_cost"]["unique_returned_frame_count"], 2)
        self.assertEqual(score["retrieved_frames"][0]["source_path"], self.source[1]["source_path"])

    def test_initial_vs_cumulative_recall_and_budget_denominator(self):
        first, second = request(900, 1500), request(4800, 5200)
        rounds = [review(self.supplied, searches=[first], ready=False, retrieved=[(first, ["f1"])]),
                  review([*self.supplied, "f1"], searches=[second], ready=False,
                         retrieved=[(second, ["f5"])], number=1),
                  review([*self.supplied, "f1", "f5"], number=2)]
        score = self.score(rounds)
        self.assertEqual(score["initial_requested"], {"moment_ids": [1], "count": 1, "denominator": 2, "recall": 0.5, "missed_moment_ids": [2]})
        self.assertEqual(score["cumulative_requested"]["recall"], 1)
        self.assertEqual(score["final_retained"]["recall"], 1)
        budget_limited = self.score(rounds[:1])
        self.assertEqual(budget_limited["cumulative_retrieved"]["denominator"], 2)
        self.assertEqual(budget_limited["cumulative_retrieved"]["recall"], 0.5)
        self.assertEqual(budget_limited["unreviewed_retrieved_frame_ids"], ["f1"])
        self.assertEqual(budget_limited["final_retained"]["recall"], 0)

    def test_broad_requests_get_no_detection_credit_even_if_retrieval_succeeds(self):
        broad = request(0, 13000)
        narrow = request(800, 1600)
        score = self.score([review(self.supplied, searches=[broad, narrow], ready=False,
                                   retrieved=[(broad, ["f5"])])])
        self.assertEqual(score["cumulative_requested"]["moment_ids"], [1])
        self.assertEqual(score["cumulative_retrieved"]["moment_ids"], [2])
        self.assertEqual(score["request_cost"]["broad_request_count"], 1)
        self.assertEqual(score["request_cost"]["summed_duration_ms"], 13800)
        self.assertEqual(score["request_cost"]["union_duration_ms"], 13000)
        self.assertFalse(score["false_completion_against_provisional_reference"])

    def test_new_nearby_frame_earns_explicit_provisional_temporal_credit(self):
        ref = reference([self.source[i] for i in (1, 5, 9, 11)], tolerance_ms=1100, duration_ms=13000)
        condition = make_conditions(ref)[0]
        query = request(1500, 2500)
        rounds = [review(condition["supplied_frame_ids"], searches=[query], ready=False, retrieved=[(query, ["f2"])]),
                  review([*condition["supplied_frame_ids"], "f2"], number=1)]
        score = self.score(rounds, ref=ref, condition=condition)
        self.assertEqual(score["cumulative_retrieved_exact"]["count"], 0)
        self.assertEqual(score["cumulative_retrieved_temporal_proxy"]["moment_ids"], [1])
        self.assertEqual(score["final_retained"]["recall"], 0.5)
        self.assertEqual(score["retrieved_frames"][0]["match_kind"], "temporal_proxy")
        self.assertTrue(score["retrieved_frames"][0]["review_required"])

    def test_ambiguous_temporal_match_cannot_cover_multiple_gaps_automatically(self):
        ref = reference([self.source[i] for i in (2, 4, 9, 11)], tolerance_ms=1200, duration_ms=13000)
        condition = make_conditions(ref)[0]
        query = request(2800, 3200)
        score = self.score([review(condition["supplied_frame_ids"], searches=[query], ready=False,
                                   retrieved=[(query, ["f3"])])], ref=ref, condition=condition)
        self.assertEqual(score["cumulative_requested"]["recall"], 1)
        self.assertEqual(score["cumulative_retrieved"]["recall"], 0)
        self.assertEqual(score["ambiguous_retrieved_frame_ids_needing_review"], ["f3"])

    def test_group_envelope_does_not_turn_unseen_middle_into_matching_evidence(self):
        ref = reference([self.source[i] for i in (1, 5, 9, 11)], groups=[1, 1, 2, 3], duration_ms=13000)
        condition = make_conditions(ref)[0]
        query = request(2900, 3100)
        score = self.score([review(condition["supplied_frame_ids"], searches=[query], ready=False,
                                   retrieved=[(query, ["f3"])])], ref=ref, condition=condition)
        self.assertEqual(score["missing_moment_count"], 1)
        self.assertEqual(score["cumulative_requested"]["recall"], 0)
        self.assertEqual(score["cumulative_retrieved"]["recall"], 0)
        self.assertEqual(score["unmatched_retrieved_frame_ids_needing_review"], ["f3"])
        self.assertIn("never automatically a false positive", score["unmatched_evidence_policy"])

    def test_control_and_redundant_omissions_have_null_recall(self):
        condition = make_conditions(self.ref)[3]
        query = request(2900, 3100)
        score = self.score([review(condition["supplied_frame_ids"], searches=[query], ready=False,
                                   retrieved=[(query, ["f3"])])], condition=condition)
        self.assertEqual(score["missing_moment_count"], 0)
        for name in ("initial_requested", "cumulative_requested", "cumulative_retrieved", "final_retained"):
            self.assertIsNone(score[name]["recall"])
        self.assertFalse(score["false_completion_against_provisional_reference"])
        self.assertEqual(score["unmatched_retrieved_frame_ids_needing_review"], ["f3"])

    def test_fabricated_ids_provenance_interval_and_unsupplied_decisions_rejected(self):
        query = request(900, 1500)
        cases = [review(self.supplied, searches=[query], ready=False, retrieved=[(query, ["invented"])]),
                 review(self.supplied, searches=[query], ready=False, retrieved=[(query, ["f5"])]),
                 review([*self.supplied, "f1"])]
        for row in cases:
            with self.subTest(row=row), self.assertRaises(ContractError):
                self.score([row])
        altered = copy.deepcopy(self.source)
        altered[1]["source_sha256"] = "0" * 64
        with self.assertRaises(ContractError):
            score_condition(self.ref, self.condition, altered, [])

    def test_frozen_ranking_and_gap_denominator_cannot_change_during_scoring(self):
        changed_reference = copy.deepcopy(self.ref)
        changed_reference["ranking"][0]["reason"] = "Changed after exposure."
        with self.assertRaises(ContractError):
            self.score([], ref=changed_reference)
        changed_condition = copy.deepcopy(self.condition)
        changed_condition["missing_moment_ids"].pop()
        with self.assertRaises(ContractError):
            self.score([], condition=changed_condition)


if __name__ == "__main__":
    unittest.main()
