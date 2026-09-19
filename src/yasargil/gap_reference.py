"""Frozen Qwen references and provisional, auditable gap-recovery measurements.

The reference is a model's judgment, not surgical ground truth. Exact candidate
membership and temporal proximity are reported separately; new discoveries and
ambiguous temporal matches require visual review.
"""
from __future__ import annotations

import copy
import math

from jsonschema import Draft202012Validator, ValidationError

from .contract import ContractError, canonical_hash, require


def ranking_schema(candidate_ids, *, surgical_importance=False):
    ids = list(candidate_ids)
    require(len(ids) >= 2 and len(ids) == len(set(ids)), "Ranking needs at least two unique candidate IDs")
    schema = {"type": "object", "additionalProperties": False, "properties": {
        "scene_summary": {"type": "string", "minLength": 1, "maxLength": 1600},
        "context_check": {"type": "string", "enum": ["consistent", "uncertain", "conflict", "not_supplied"]},
        "ranking": {"type": "array", "minItems": len(ids), "maxItems": len(ids), "items": {
            "type": "object", "additionalProperties": False, "properties": {
                "frame_id": {"type": "string", "enum": ids},
                "moment_id": {"type": "integer", "minimum": 1, "maximum": len(ids)},
                "reason": {"type": "string", "minLength": 1, "maxLength": 600},
            }, "required": ["frame_id", "moment_id", "reason"]}},
    }, "required": ["scene_summary", "context_check", "ranking"]}
    if surgical_importance:
        row = schema["properties"]["ranking"]["items"]
        row["properties"]["importance_score"] = {"type": "integer", "minimum": 0, "maximum": 100}
        row["required"].append("importance_score")
    return schema


def _number(value, label, *, minimum=0):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= minimum, f"Invalid {label}")
    return value


def _frame_index(frames, *, duration_ms):
    result = {}
    for frame in frames:
        require(isinstance(frame, dict), "A source frame must be an object")
        frame_id = frame.get("frame_id")
        require(isinstance(frame_id, str) and frame_id and frame_id not in result, "Invalid or duplicate source frame ID")
        timestamp = _number(frame.get("timestamp_ms"), "frame timestamp")
        require(timestamp <= duration_ms, "Frame timestamp exceeds source duration")
        for field in ("source_path", "source_sha256", "timestamp_basis"):
            require(isinstance(frame.get(field), str) and frame[field], f"Frame lacks required provenance: {field}")
        result[frame_id] = frame
    return result


def build_reference(raw, candidate_frames, *, duration_ms, tolerance_ms=1000, max_moment_span_ms=10000,
                    surgical_importance=False):
    """Freeze a complete ranking, deriving all locators from canonical source data."""
    _number(duration_ms, "source duration", minimum=1)
    _number(tolerance_ms, "temporal tolerance")
    _number(max_moment_span_ms, "maximum moment span")
    candidates = _frame_index(candidate_frames, duration_ms=duration_ms)
    try:
        Draft202012Validator(ranking_schema(candidates, surgical_importance=surgical_importance)).validate(raw)
    except ValidationError as exc:
        raise ContractError(f"Invalid reference ranking: {exc.message}") from exc
    ranked_ids = [row["frame_id"] for row in raw["ranking"]]
    require(len(set(ranked_ids)) == len(candidates) and set(ranked_ids) == set(candidates),
            "Ranking must contain every candidate exactly once")
    require(all(isinstance(row["moment_id"], int) and not isinstance(row["moment_id"], bool)
                for row in raw["ranking"]), "Moment IDs must be integers")
    if surgical_importance:
        scores = [row["importance_score"] for row in raw["ranking"]]
        require(all(type(score) is int for score in scores), "Importance scores must be integers")
        require(scores == sorted(scores, reverse=True) and len(set(scores)) >= 2,
                "Surgical importance scores must be decreasing with meaningful differentiation")
    ranking, groups = [], {}
    for ordinal, judgment in enumerate(raw["ranking"], start=1):
        row = {**copy.deepcopy(candidates[judgment["frame_id"]]), **copy.deepcopy(judgment), "rank": ordinal}
        ranking.append(row)
        groups.setdefault(row["moment_id"], []).append(row)
    moments = []
    for moment_id, members in groups.items():
        timestamps = sorted(set(row["timestamp_ms"] for row in members))
        require(timestamps[-1] - timestamps[0] <= max_moment_span_ms,
                f"Moment {moment_id} groups frames more than {max_moment_span_ms} ms apart")
        moments.append({"moment_id": moment_id, "frame_ids": [row["frame_id"] for row in members],
                        "best_rank": members[0]["rank"], "anchor_timestamps_ms": timestamps,
                        "start_ms": max(0, timestamps[0] - tolerance_ms),
                        "end_ms": min(duration_ms, timestamps[-1] + tolerance_ms),
                        "member_span_ms": timestamps[-1] - timestamps[0],
                        "grouping_basis": "qwen_provisional_visual_equivalence",
                        "temporal_match_basis": "within_tolerance_of_a_member_anchor_not_entire_group_span"})
    reference = {"schema_version": "gap-reference-v1", "reference_kind": "provisional_qwen_ranking_and_moment_groups",
                 "clinical_ground_truth": False, "clinical_validation": "not_performed", "training_eligible": False,
                 "scene_summary": raw["scene_summary"], "context_check": raw["context_check"],
                 "duration_ms": duration_ms, "tolerance_ms": tolerance_ms,
                 "max_moment_span_ms": max_moment_span_ms, "candidate_count": len(ranking),
                 "ranking": ranking, "moments": moments}
    if surgical_importance:
        reference.update(schema_version="gap-reference-v2", ranking_objective="surgical_information_value_in_complete_procedure",
                         ranking_score_range=[0, 100], score_order="descending")
    reference["reference_sha256"] = canonical_hash(reference)
    return reference


def make_conditions(reference):
    """Derive nested omissions; v2 drops least-important frames from the tail.

    Frozen v1 experiments retain their original most-important omission protocol
    so old receipts can still be independently scored without reinterpretation.
    """
    ranking = reference["ranking"]
    total = len(ranking)
    require(total >= 2 and len({row["frame_id"] for row in ranking}) == total,
            "Conditions require at least two uniquely ranked frames")
    version = reference.get("schema_version")
    require(version in {"gap-reference-v1", "gap-reference-v2"}, "Unknown gap reference version")
    drop_least_important = version == "gap-reference-v2"
    cohort_spec = (("drop_50", 50), ("drop_70", 70), ("drop_90", 90), ("control_all", 0)) if drop_least_important else (
        ("bottom_50", 50), ("bottom_70", 30), ("bottom_90", 10), ("control_all", 0))
    result = []
    for condition_id, withheld_percent in cohort_spec:
        count = min(total - 1, max(1, (total * withheld_percent + 50) // 100)) if withheld_percent else 0
        if drop_least_important:
            split = total - count
            withheld_rows, supplied_rows = ranking[split:], ranking[:split]
        else:
            withheld_rows, supplied_rows = ranking[:count], ranking[count:]
        withheld_ids = [row["frame_id"] for row in withheld_rows]
        supplied = sorted(supplied_rows, key=lambda row: (row["timestamp_ms"], row.get("frame_index", 0), row["frame_id"]))
        supplied_ids = [row["frame_id"] for row in supplied]
        withheld, supplied_set = set(withheld_ids), set(supplied_ids)
        missing, represented = [], []
        for moment in reference["moments"]:
            members = set(moment["frame_ids"])
            if members & withheld:
                (represented if members & supplied_set else missing).append(moment["moment_id"])
        result.append({"id": condition_id, "condition_id": condition_id, "reference_sha256": reference.get("reference_sha256"),
                       "requested_withheld_percent": withheld_percent,
                       "requested_supplied_percent": 100 - withheld_percent,
                       "actual_withheld_count": count, "actual_supplied_count": total - count,
                       "actual_withheld_percent": count / total * 100,
                       "actual_supplied_percent": (total - count) / total * 100,
                       "rounding": "nearest_integer_half_up_clamped_to_one_through_n_minus_one_except_control",
                       "withheld_frame_ids": withheld_ids, "supplied_frame_ids": supplied_ids,
                       "missing_moment_ids": missing,
                       "already_represented_withheld_moment_ids": represented,
                       "protected_anchor_overrides": False})
        if drop_least_important:
            result[-1]["omission_policy"] = "drop_least_surgically_important_ranking_tail"
            result[-1]["supply_policy"] = "retain_most_surgically_important_ranking_prefix_presented_chronologically"
    return result


def _rate(ids, gaps):
    matched = [moment_id for moment_id in gaps if moment_id in ids]
    return {"moment_ids": matched, "count": len(matched), "denominator": len(gaps),
            "recall": len(matched) / len(gaps) if gaps else None,
            "missed_moment_ids": [moment_id for moment_id in gaps if moment_id not in ids]}


def _union_duration(intervals):
    end, total = None, 0
    for start, stop in sorted(intervals):
        total += stop - (start if end is None or start > end else min(stop, end))
        end = stop if end is None else max(end, stop)
    return total


def score_condition(reference, condition, source_frames, rounds, *, max_request_span_ms=10000):
    """Measure gap recovery without turning temporal/model proxies into truth.

    Detection excludes overly broad queries. Retrieval and final retention remain
    separate even if a broad query happened to return relevant evidence. A new
    frame matching several moment groups earns no automatic coverage credit.
    """
    _number(max_request_span_ms, "maximum credited request span", minimum=1)
    frozen_value = {key: value for key, value in reference.items() if key != "reference_sha256"}
    require(reference.get("reference_sha256") == canonical_hash(frozen_value), "Frozen reference has changed")
    expected_condition = next((row for row in make_conditions(reference)
                               if row["condition_id"] == condition.get("condition_id")), None)
    require(expected_condition is not None, "Unknown omission condition")
    require(all(condition.get(key) == expected_condition[key] for key in (
        "reference_sha256", "supplied_frame_ids", "withheld_frame_ids", "missing_moment_ids",
        "already_represented_withheld_moment_ids", "actual_withheld_count", "actual_supplied_count")),
        "Condition cohorts or denominator differ from the frozen ranking")
    duration, tolerance = reference["duration_ms"], reference["tolerance_ms"]
    source = _frame_index(source_frames, duration_ms=duration)
    moments = {row["moment_id"]: row for row in reference["moments"]}
    membership = {frame_id: moment["moment_id"] for moment in reference["moments"] for frame_id in moment["frame_ids"]}
    gaps = list(condition["missing_moment_ids"])
    require(len(gaps) == len(set(gaps)) and set(gaps) <= moments.keys(), "Invalid condition gap IDs")
    require(set(membership) <= source.keys(), "Reference candidates missing from canonical source")
    for row in reference["ranking"]:
        require(all(source[row["frame_id"]].get(key) == row.get(key)
                    for key in ("timestamp_ms", "timestamp_basis", "source_path", "source_sha256",
                                "image_path", "image_sha256", "frame_index", "release_frame_index", "source_pts",
                                "time_base", "source_timestamp_ms", "source_acquisition_time", "video_pts", "video_time_base")),
                "Reference provenance differs from source manifest")
    require(set(condition["supplied_frame_ids"]) <= source.keys(), "Unknown supplied source frame")

    def match_frame(frame_id):
        require(frame_id in source, f"Unknown retrieved or judged source frame: {frame_id}")
        frame = source[frame_id]
        if frame_id in membership:
            return {"frame_id": frame_id, "timestamp_ms": frame["timestamp_ms"],
                    "match_kind": "exact_reference_member", "moment_ids": [membership[frame_id]], "review_required": False}
        possible = [moment_id for moment_id, moment in moments.items()
                    if any(abs(frame["timestamp_ms"] - anchor) <= tolerance for anchor in moment["anchor_timestamps_ms"])]
        kind = "temporal_proxy" if len(possible) == 1 else "ambiguous_temporal_proxy" if possible else "outside_reference_needs_review"
        return {"frame_id": frame_id, "timestamp_ms": frame["timestamp_ms"], "match_kind": kind,
                "moment_ids": possible, "review_required": True}

    requested, first_requested, retrieved, exact_retrieved, proxy_retrieved = set(), set(), set(), set(), set()
    request_records, retrieval_records, progress, intervals = [], [], [], []
    returned_ids, introduced = [], set(condition["supplied_frame_ids"])
    previous_round = -1
    for ordinal, row in enumerate(rounds):
        round_number = row.get("round", ordinal)
        require(isinstance(round_number, int) and not isinstance(round_number, bool) and round_number > previous_round,
                "Scoring rounds must have unique increasing integer ordinals")
        previous_round = round_number
        output = row["output"]
        for frame_id, decision in output["decisions"].items():
            require(frame_id in introduced, "A decision judges a frame not supplied to this session")
            require(decision.get("decision") in {"keep", "drop"}, "Invalid selection decision")
        require(set(output["decisions"]) == introduced, "Every supplied candidate needs a decision before scoring")
        round_requested = set()
        searches = output.get("searches", [])
        for search_index, search in enumerate(searches):
            start = _number(search["start_ms"], "requested interval start")
            end = _number(search["end_ms"], "requested interval end")
            require(start < end <= duration, "Invalid requested interval")
            span = end - start
            eligible = span <= max_request_span_ms
            hits = [moment_id for moment_id in gaps if eligible and any(
                start <= min(duration, anchor + tolerance) and end >= max(0, anchor - tolerance)
                for anchor in moments[moment_id]["anchor_timestamps_ms"])]
            round_requested.update(hits)
            intervals.append((start, end))
            request_records.append({"round": round_number, "request_index": search_index,
                                    "request": copy.deepcopy(search), "span_ms": span,
                                    "credited_for_detection": eligible,
                                    "exclusion_reason": None if eligible else "request_exceeds_maximum_credited_span",
                                    "matched_missing_moment_ids": hits,
                                    "match_basis": "request_overlaps_reference_member_anchor_plus_or_minus_tolerance"})
        if ordinal == 0:
            first_requested.update(round_requested)
        requested.update(round_requested)
        for receipt in row.get("retrieval", []):
            require(receipt["request"] in searches, "Retrieval receipt has no corresponding model request")
            ids = receipt.get("returned_frame_ids", [])
            require(len(ids) == len(set(ids)), "Duplicate frame within retrieval receipt")
            for frame_id in ids:
                require(frame_id not in introduced, "Retrieval receipt repeats an already supplied frame")
                match = match_frame(frame_id)
                require(receipt["request"]["start_ms"] <= match["timestamp_ms"] <= receipt["request"]["end_ms"],
                        "Retrieved frame lies outside its requested interval")
                match.update({"round": round_number, "request": copy.deepcopy(receipt["request"]),
                              "source_path": source[frame_id]["source_path"],
                              "source_sha256": source[frame_id]["source_sha256"],
                              "timestamp_basis": source[frame_id]["timestamp_basis"]})
                retrieval_records.append(match)
                returned_ids.append(frame_id)
                introduced.add(frame_id)
                if match["match_kind"] in {"exact_reference_member", "temporal_proxy"}:
                    target = exact_retrieved if match["match_kind"] == "exact_reference_member" else proxy_retrieved
                    target.update(match["moment_ids"])
                    retrieved.update(match["moment_ids"])
        progress.append({"round": round_number, "requested": _rate(requested, gaps),
                         "retrieved": _rate(retrieved, gaps)})

    final_output = rounds[-1]["output"] if rounds else None
    retained, exact_retained, proxy_retained, retained_records = set(), set(), set(), []
    if final_output:
        for frame_id, decision in final_output["decisions"].items():
            if decision["decision"] != "keep":
                continue
            match = match_frame(frame_id)
            retained_records.append(match)
            if match["match_kind"] in {"exact_reference_member", "temporal_proxy"}:
                target = exact_retained if match["match_kind"] == "exact_reference_member" else proxy_retained
                target.update(match["moment_ids"])
                retained.update(match["moment_ids"])
    missed = [moment_id for moment_id in gaps if moment_id not in retained]
    final_decision_ids = set(final_output["decisions"]) if final_output else set()
    dropped_lower_importance = reference["schema_version"] == "gap-reference-v2"
    declared_complete_with_omissions = bool(final_output and final_output.get("ready") and missed)
    return {"schema_version": "gap-recovery-score-v2" if dropped_lower_importance else "gap-recovery-score-v1", "condition_id": condition["condition_id"],
            "reference_sha256": reference.get("reference_sha256"),
            "scoring_basis": "provisional_qwen_reference_with_exact_membership_and_temporal_proxies",
            "clinical_ground_truth": False, "clinical_validation": "not_performed", "training_eligible": False,
            "temporal_proxy_review_required": True, "max_request_span_ms": max_request_span_ms,
            "missing_moment_ids": gaps, "missing_moment_count": len(gaps), "completed_review_rounds": len(rounds),
            "initial_requested": _rate(first_requested, gaps), "cumulative_requested": _rate(requested, gaps),
            "cumulative_retrieved": _rate(retrieved, gaps),
            "cumulative_retrieved_exact": _rate(exact_retrieved, gaps),
            "cumulative_retrieved_temporal_proxy": _rate(proxy_retrieved, gaps),
            "final_retained": _rate(retained, gaps), "final_retained_exact": _rate(exact_retained, gaps),
            "final_retained_temporal_proxy": _rate(proxy_retained, gaps),
            "missed_moment_ids": missed, "model_declared_complete": final_output.get("ready", False) if final_output else None,
            "false_completion_against_provisional_reference": None if dropped_lower_importance else declared_complete_with_omissions,
            **({"declared_complete_with_unretained_reference_moments": declared_complete_with_omissions,
                "completion_interpretation": "Lower-ranked omitted moments may be unnecessary for the surgical summary. "
                "Zero retrieval requests and declaring completion can be appropriate; unretained reference moments "
                "alone do not establish a failure. Surgical relevance requires review."} if dropped_lower_importance else {}),
            "request_cost": {"request_count": len(request_records),
                             "broad_request_count": sum(not item["credited_for_detection"] for item in request_records),
                             "summed_duration_ms": sum(end - start for start, end in intervals),
                             "union_duration_ms": _union_duration(intervals),
                             "returned_frame_count": len(returned_ids), "unique_returned_frame_count": len(set(returned_ids))},
            "requests": request_records, "retrieved_frames": retrieval_records, "retained_frames": retained_records,
            "round_progress": progress,
            "unreviewed_retrieved_frame_ids": [frame_id for frame_id in returned_ids if frame_id not in final_decision_ids],
            "unmatched_retrieved_frame_ids_needing_review": [item["frame_id"] for item in retrieval_records
                                                            if item["match_kind"] == "outside_reference_needs_review"],
            "ambiguous_retrieved_frame_ids_needing_review": [item["frame_id"] for item in retrieval_records
                                                            if item["match_kind"] == "ambiguous_temporal_proxy"],
            "unmatched_evidence_policy": "May be a useful discovery outside the candidate pool; never automatically a false positive."}
