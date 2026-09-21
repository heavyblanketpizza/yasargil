"""Bind retrospective Qwen annotation drafts to canonical source observations.

These checks validate structure and evidence locators. They cannot establish
whether an observation or a contextual claim is visually or clinically true.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import re

from jsonschema import Draft202012Validator, ValidationError

from .contract import ContractError, require


TIMESTAMP_EVIDENCE = "timestamps"
FRAME_EVIDENCE = "source_frame_ids"


def _number(value, label, *, minimum=None):
    try:
        finite = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite and (minimum is None or value >= minimum), f"Invalid {label}")
    return value


def _positive(value, label):
    _number(value, label, minimum=0)
    require(value > 0, f"Invalid {label}")
    return value


def annotation_schema(frame_ids, duration_ms, *, max_evidence_span_ms=10000,
                      source_frames=None, evidence_mode=TIMESTAMP_EVIDENCE):
    """Return the strict response schema for exactly the frozen selected IDs.

    Interval order, finite numbers, maximum span, and actual observations are
    cross-field checks performed by :func:`build_annotations`.
    """
    _positive(duration_ms, "source duration")
    _positive(max_evidence_span_ms, "maximum evidence span")
    require(evidence_mode in (TIMESTAMP_EVIDENCE, FRAME_EVIDENCE), "Unknown annotation evidence mode")
    require(not isinstance(frame_ids, (str, bytes, dict)), "Selected frame IDs must be a sequence")
    try:
        ids = list(frame_ids)
    except TypeError as exc:
        raise ContractError("Selected frame IDs must be a sequence") from exc
    require(ids and all(isinstance(frame_id, str) and frame_id.strip() for frame_id in ids),
            "Annotation requires nonempty selected frame IDs")
    require(len(ids) == len(set(ids)), "Duplicate selected frame ID")
    interval = {"type": "object", "additionalProperties": False, "properties": {
        "start_ms": {"type": "number", "minimum": 0, "maximum": duration_ms},
        "end_ms": {"type": "number", "minimum": 0, "maximum": duration_ms},
    }, "required": ["start_ms", "end_ms"]}
    if evidence_mode == FRAME_EVIDENCE:
        require(isinstance(source_frames, (list, tuple)) and source_frames,
                "Frame citations require the complete source inventory")
        source_ids = [frame.get("frame_id") if isinstance(frame, dict) else None for frame in source_frames]
        require(all(isinstance(frame_id, str) and frame_id.strip() for frame_id in source_ids)
                and len(source_ids) == len(set(source_ids)) and set(ids) <= set(source_ids),
                "Frame citations require unique source IDs containing every selected frame")
        # One shared enum avoids repeating thousands of IDs for every target.
        # These are direct local references; the definition has no nested refs.
        interval = {"type": "object", "additionalProperties": False, "properties": {
            "start_frame_id": {"$ref": "#/$defs/source_frame_id"},
            "end_frame_id": {"$ref": "#/$defs/source_frame_id"},
        }, "required": ["start_frame_id", "end_frame_id"]}
    claim = {"type": "object", "additionalProperties": False, "properties": {
        "claim": {"type": "string", "minLength": 1, "maxLength": 600},
        "evidence_intervals": {"type": "array", "minItems": 1, "maxItems": 3, "items": interval},
    }, "required": ["claim", "evidence_intervals"]}
    annotation = {"type": "object", "additionalProperties": False, "properties": {
        "visible_observation": {"type": "string", "minLength": 1, "maxLength": 800},
        "visibility": {"type": "string", "enum": ["clear", "partial", "poor", "uninterpretable"]},
        "contextual_claims": {"type": "array", "minItems": 0, "maxItems": 3, "items": claim},
        "uncertainties": {"type": "array", "minItems": 0, "maxItems": 6,
                          "items": {"type": "string", "minLength": 1, "maxLength": 400}},
    }, "required": ["visible_observation", "visibility", "contextual_claims", "uncertainties"]}
    schema = {"type": "object", "additionalProperties": False, "properties": {
        "context_check": {"type": "string", "enum": ["consistent", "uncertain", "conflict", "not_supplied"]},
        "annotations": {"type": "object", "additionalProperties": False,
                        "properties": {frame_id: copy.deepcopy(annotation) for frame_id in ids}, "required": ids},
    }, "required": ["context_check", "annotations"]}
    if evidence_mode == FRAME_EVIDENCE:
        schema["$defs"] = {"source_frame_id": {"type": "string", "enum": source_ids}}
    return schema


def _path(value, label):
    require(isinstance(value, str) and value.strip() and Path(value).is_absolute(), f"Invalid {label}")


def _hash(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None,
            f"Invalid {label}")


def _source_index(source):
    require(isinstance(source, dict), "Source manifest must be an object")
    duration = _positive(source.get("duration_ms"), "source duration")
    _path(source.get("source_path"), "source path")
    _hash(source.get("source_sha256"), "source hash")
    frames = source.get("frames")
    require(isinstance(frames, list) and frames, "Source manifest needs observed frames")
    count = source.get("expected_video_frames")
    require(type(count) is int and count == len(frames), "Source frame count differs from expected video frames")
    result, previous_index, previous_time = {}, -1, -1
    for frame in frames:
        require(isinstance(frame, dict), "Canonical source frame must be an object")
        frame_id = frame.get("frame_id")
        require(isinstance(frame_id, str) and frame_id.strip() and frame_id not in result,
                "Invalid or duplicate canonical source frame ID")
        index = frame.get("frame_index")
        require(type(index) is int and index > previous_index, "Source frame indices must be unique and increasing")
        timestamp = _number(frame.get("timestamp_ms"), "source frame timestamp", minimum=0)
        require(previous_time < timestamp <= duration, "Source frame timestamps must increase within source duration")
        require(frame.get("timestamp_basis") in ("source_pts", "reconstructed_nominal"),
                "Invalid source frame timestamp basis")
        for field in ("source_path", "image_path"):
            _path(frame.get(field), f"canonical frame {field}")
        for field in ("source_sha256", "image_sha256"):
            _hash(frame.get(field), f"canonical frame {field}")
        release_index = frame.get("release_frame_index")
        require(release_index is None or (type(release_index) is int and release_index > 0),
                "Invalid released source frame index")
        if frame["timestamp_basis"] == "reconstructed_nominal":
            require(all(frame.get(field) is None for field in ("source_pts", "time_base", "source_timestamp_ms")),
                    "Reconstructed frames cannot claim original capture PTS")
        result[frame_id] = frame
        previous_index, previous_time = index, timestamp
    return result, duration


def build_annotations(raw, selected_frames, source, *, max_evidence_span_ms=10000,
                      evidence_mode=TIMESTAMP_EVIDENCE):
    """Validate drafts and attach unchanged source rows to evidence intervals.

    Intervals are closed: observations exactly at either endpoint are included.
    Frame citations map their endpoints to exact source timestamps. A single
    frame is a point citation; its display duration is never invented.
    Selection and source objects are never modified. The returned records remain
    review-required even when every locator is valid and Qwen reports agreement.
    """
    canonical, duration = _source_index(source)
    _positive(max_evidence_span_ms, "maximum evidence span")
    require(isinstance(selected_frames, (list, tuple)) and selected_frames,
            "Annotation requires frozen selected frames")
    selected_ids = []
    for frame in selected_frames:
        require(isinstance(frame, dict), "Selected frame must be an object")
        frame_id = frame.get("frame_id")
        require(isinstance(frame_id, str) and frame_id in canonical, "Unknown selected source frame")
        require(frame_id not in selected_ids, "Duplicate selected frame ID")
        require(all(key in frame and type(frame[key]) is type(value) and frame[key] == value
                    for key, value in canonical[frame_id].items()),
                f"Selected frame provenance differs from canonical source: {frame_id}")
        selected_ids.append(frame_id)
    try:
        Draft202012Validator(annotation_schema(selected_ids, duration,
                                               max_evidence_span_ms=max_evidence_span_ms,
                                               source_frames=source["frames"],
                                               evidence_mode=evidence_mode)).validate(raw)
    except ValidationError as exc:
        raise ContractError(f"Invalid frame annotations: {exc.message}") from exc
    annotations = []
    for frame_id in selected_ids:
        judgment = raw["annotations"][frame_id]
        require(judgment["visible_observation"].strip(), "Visible observation cannot be blank")
        require(all(value.strip() for value in judgment["uncertainties"]), "Uncertainty cannot be blank")
        require(judgment["visibility"] not in {"poor", "uninterpretable"} or judgment["uncertainties"],
                "Poor or uninterpretable visibility requires an uncertainty statement")
        claims = []
        for claim in judgment["contextual_claims"]:
            require(claim["claim"].strip(), "Contextual claim cannot be blank")
            intervals = []
            for interval in claim["evidence_intervals"]:
                locators = {}
                if evidence_mode == FRAME_EVIDENCE:
                    first, last = canonical[interval["start_frame_id"]], canonical[interval["end_frame_id"]]
                    require(first["frame_index"] <= last["frame_index"],
                            "Evidence frame endpoints must be in source order")
                    start, end = first["timestamp_ms"], last["timestamp_ms"]
                    locators = {"start_frame_id": first["frame_id"], "end_frame_id": last["frame_id"]}
                else:
                    start = _number(interval["start_ms"], "evidence interval start", minimum=0)
                    end = _number(interval["end_ms"], "evidence interval end", minimum=0)
                    require(start < end <= duration, "Evidence interval must have increasing endpoints within source duration")
                require(end - start <= max_evidence_span_ms, "Evidence interval exceeds maximum evidence span")
                observations = [copy.deepcopy(frame) for frame in canonical.values() if start <= frame["timestamp_ms"] <= end]
                require(observations, "Evidence interval contains no actual source observations")
                intervals.append({**locators, "start_ms": start, "end_ms": end, "supporting_frames": observations})
            claims.append({"claim": claim["claim"], "evidence_intervals": intervals})
        annotations.append({**copy.deepcopy(canonical[frame_id]),
                            "visible_observation": judgment["visible_observation"],
                            "visibility": judgment["visibility"], "contextual_claims": claims,
                            "uncertainties": copy.deepcopy(judgment["uncertainties"]),
                            "review_required": True, "training_eligible": False})
    return {"schema_version": ("contextual-frame-annotations-v2" if evidence_mode == FRAME_EVIDENCE
                               else "contextual-frame-annotations-v1"), "context_check": raw["context_check"],
            "annotations": annotations, "clinical_validation": "not_performed", "training_eligible": False,
            "temporal_exposure": "retrospective_full_video", "evidence_validation": "locator_only_not_semantic"}
