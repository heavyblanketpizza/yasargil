"""Deterministic, bounded evidence for reviewing one retrospective Qwen draft.

The packet contains source observations and original annotation rows, never
outcome-derived context. This module does not invoke models or edit its inputs.
"""
from __future__ import annotations

import copy
import csv
import hashlib
from io import StringIO
from pathlib import Path
import re

from .annotation_contract import _number, _source_index
from .contract import ContractError, require, sha256_file
from .video_source import media_timeline


PROTOCOL_VERSION = "medgemma-evidence-v1"
_TABLES = (("sospine_tool_tips.csv", "keypoint", "manual"),
           ("sospine_bbox.csv", "bbox", "computed"))
_FIELDS = {"trial_frame", "x1", "y1", "x2", "y2", "label"}
_CASES = re.compile(r"(?:S[1-8]A[1-3]|Clip[01])")
_SOURCE_CAVEATS = [
    "Original labels are evidence to inspect, not verified truth or exhaustive object inventories.",
    "Missing rows and blank placeholders are unavailable annotations, not negative labels.",
    "Tool-tip annotations are manual; bounding boxes are computed, not manually traced object outlines.",
    "Raw coordinates and label whitespace are preserved. Coordinate conventions are not converted; "
    "tool-tip rows may have nonzero extent and boxes may be degenerate or out of bounds.",
]


def _integer(value, low, high, label):
    require(type(value) is int and low <= value <= high, f"{label} must be an integer from {low} to {high}")


def _dataset_root(source, dataset_root):
    """Infer only an exact SOSpine source directory; explicit mismatches fail."""
    source_path = Path(source["source_path"])
    supported_layout = (source["source_kind"] == "released_image_sequence"
                        and source_path.parent.name == "frames"
                        and _CASES.fullmatch(source_path.name) is not None)
    if dataset_root is None:
        if not supported_layout:
            return None
        candidate = source_path.parent.parent.resolve()
        if not all((candidate / table).is_file() for table, _, _ in _TABLES):
            return None
    else:
        candidate = Path(dataset_root).expanduser().resolve()
        require(supported_layout, "Dataset context requires an exact SOSpine released-frame source layout")
        require(candidate.is_dir(), "SOSpine dataset root is not a directory")
    case = source_path.name
    expected_directory = candidate / "frames" / case
    require(source_path == expected_directory and source_path.resolve() == expected_directory,
            "Source directory does not match the SOSpine dataset root/case; symlink escapes are not allowed")
    for frame in source["frames"]:
        release_index = frame.get("release_frame_index")
        require(type(release_index) is int and release_index == frame["frame_index"] + 1,
                "SOSpine source rows must preserve contiguous one-based release indices")
        expected = expected_directory / f"{case}_frame_{release_index:08d}.jpeg"
        actual = Path(frame["source_path"])
        require(actual == expected and actual.resolve() == expected,
                "Source frame does not match its exact SOSpine case/release-index path")
    return candidate


def _dataset_context(source, frames, dataset_root, procedure_context):
    context = {
        "status": "unavailable", "dataset_name": None, "case_id": None,
        "documented_procedure_context": procedure_context,
        "procedure_context_provenance": "parent_annotation_run" if procedure_context else "not_supplied",
        "procedure_context_is_visual_evidence": False,
        "source_annotation_caveats": [], "tables": [], "original_annotations": [],
        "label_availability": [],
        "excluded_context": ["case_outcomes", "surgeon_experience", "repair_duration"],
    }
    root = _dataset_root(source, dataset_root)
    if root is None:
        context["unavailable_reason"] = "No verified SOSpine source mapping and annotation tables supplied or inferred."
        return context
    context.update(status="available", dataset_name="SOSpine", case_id=Path(source["source_path"]).name,
                   source_annotation_caveats=copy.deepcopy(_SOURCE_CAVEATS))
    by_filename = {}
    for frame in frames:
        path = Path(frame["source_path"])
        require(path.is_file() and sha256_file(path) == frame["source_sha256"],
                f"SOSpine source image hash mismatch: {frame['frame_id']}")
        by_filename[path.name] = frame
    for filename, kind, origin in _TABLES:
        path = root / filename
        require(path.is_file() and path.resolve() == path,
                f"Missing SOSpine annotation table or symlink escape: {filename}")
        try:
            content = path.read_bytes()
            reader = csv.DictReader(StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
            fields = reader.fieldnames
            require(fields is not None and len(fields) == len(set(fields)) and _FIELDS <= set(fields)
                    and set(fields) <= _FIELDS | ({""} if kind == "keypoint" else set()),
                    f"Unexpected SOSpine annotation columns: {filename}")
            digest = hashlib.sha256(content).hexdigest()
            context["tables"].append({"filename": filename, "source_path": str(path), "sha256": digest,
                                      "original_kind": kind, "original_origin": origin,
                                      "header_fields": fields})
            matches = {frame["frame_id"]: [] for frame in frames}
            for row_index, raw in enumerate(reader, start=2):
                require(None not in raw and all(value is not None for value in raw.values()),
                        f"Malformed SOSpine annotation row in {filename}")
                if raw["trial_frame"] not in by_filename:
                    continue
                frame = by_filename[raw["trial_frame"]]
                annotation_id = f"{filename}:row:{row_index}"
                context["original_annotations"].append({
                    "annotation_id": annotation_id, "frame_id": frame["frame_id"],
                    "source_frame_filename": raw["trial_frame"], "source_frame_sha256": frame["source_sha256"],
                    "original_kind": kind, "original_origin": origin,
                    "source_locator": {"filename": filename, "source_path": str(path), "sha256": digest,
                                       "locator_type": "csv_record_1based_including_header", "locator": row_index,
                                       "physical_end_line_1based": reader.line_num},
                    "raw_value": raw, "coordinate_conversion": "none",
                })
                matches[frame["frame_id"]].append((annotation_id, bool(raw["label"].strip())))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ContractError(f"Cannot read SOSpine annotation table {filename}: {exc}") from exc
        for frame in frames:
            rows = matches[frame["frame_id"]]
            context["label_availability"].append({
                "frame_id": frame["frame_id"], "filename": filename,
                "status": "provided" if any(labeled for _, labeled in rows) else "unavailable",
                "reason": None if any(labeled for _, labeled in rows) else
                    ("blank_placeholder_rows" if rows else "no_matching_source_rows"),
                "annotation_ids": [annotation_id for annotation_id, _ in rows],
                "negative_label": False,
            })
    return context


def _intervals(annotation, canonical, duration):
    intervals = []
    claims = annotation.get("contextual_claims")
    require(isinstance(claims, list), "Qwen annotation requires contextual_claims")
    for claim_index, claim in enumerate(claims):
        require(isinstance(claim, dict) and isinstance(claim.get("evidence_intervals"), list)
                and claim["evidence_intervals"],
                "Invalid Qwen contextual claim")
        for interval_index, interval in enumerate(claim["evidence_intervals"]):
            require(isinstance(interval, dict), "Invalid Qwen evidence interval")
            start = _number(interval.get("start_ms"), "Qwen evidence interval start", minimum=0)
            end = _number(interval.get("end_ms"), "Qwen evidence interval end", minimum=0)
            endpoints = {}
            if "start_frame_id" in interval or "end_frame_id" in interval:
                first_id, last_id = interval.get("start_frame_id"), interval.get("end_frame_id")
                require(isinstance(first_id, str) and isinstance(last_id, str)
                        and first_id in canonical and last_id in canonical,
                        "Qwen evidence endpoints must identify canonical source frames")
                first, last = canonical[first_id], canonical[last_id]
                require(first["frame_index"] <= last["frame_index"]
                        and start == first["timestamp_ms"] and end == last["timestamp_ms"]
                        and start <= end <= duration,
                        "Qwen evidence timestamps differ from their canonical frame endpoints")
                endpoints = {"start_frame_id": first_id, "end_frame_id": last_id}
            else:
                require(start < end <= duration,
                        "Qwen evidence interval must use increasing supplied-media timestamps")
            actual = [frame for frame in canonical.values() if start <= frame["timestamp_ms"] <= end]
            require(actual, "Qwen evidence interval contains no source observations")
            if "supporting_frames" in interval:
                require(interval["supporting_frames"] == actual,
                        "Qwen interval supporting frames differ from canonical source rows")
            intervals.append({"claim_index": claim_index, "interval_index": interval_index,
                              **endpoints, "start_ms": start, "end_ms": end,
                              "available_frame_ids": [frame["frame_id"] for frame in actual]})
    return intervals


def build_evidence(source, annotation, *, before_frames=2, after_frames=2,
                   max_context_frames=12, dataset_root=None, procedure_context=""):
    """Build a stable review packet, counting the target against the image cap.

    Neighbors come from the complete source inventory. Extra Qwen-cited frames
    are sampled across intervals, prioritizing intervals with no supplied image
    and then the greatest temporal distance from already supplied observations.
    Closed interval endpoints match the parent annotation contract.
    """
    _integer(before_frames, 1, 8, "Before-frame count")
    _integer(after_frames, 1, 8, "After-frame count")
    _integer(max_context_frames, 1 + before_frames + after_frames, 32, "Total evidence-frame budget")
    require(isinstance(procedure_context, str), "Procedure context must be text")
    canonical, duration = _source_index(source)
    timeline = media_timeline(source)
    require(isinstance(annotation, dict) and annotation.get("frame_id") in canonical,
            "Qwen annotation target is not a canonical source frame")
    target_id = annotation["frame_id"]
    target = canonical[target_id]
    require(all(key in annotation and type(annotation[key]) is type(value) and annotation[key] == value
                for key, value in target.items()), "Qwen target provenance differs from canonical source")
    source_frames = source["frames"]
    position = target["frame_index"]
    before = source_frames[max(0, position - before_frames):position]
    after = source_frames[position + 1:position + after_frames + 1]
    selected = {target_id} | {frame["frame_id"] for frame in before + after}
    intervals = _intervals(annotation, canonical, duration)
    while len(selected) < max_context_frames:
        added = False
        ordered = sorted(intervals, key=lambda interval: bool(selected.intersection(interval["available_frame_ids"])))
        for interval in ordered:
            ids = interval["available_frame_ids"]
            remaining = [frame_id for frame_id in ids if frame_id not in selected]
            if not remaining:
                continue
            covered_times = [canonical[frame_id]["timestamp_ms"] for frame_id in ids if frame_id in selected]
            midpoint = (interval["start_ms"] + interval["end_ms"]) / 2
            def priority(frame_id):
                timestamp = canonical[frame_id]["timestamp_ms"]
                distance = (min(abs(timestamp - other) for other in covered_times) if covered_times
                            else -abs(timestamp - midpoint))
                return distance, -canonical[frame_id]["frame_index"]
            selected.add(max(remaining, key=priority))
            added = True
            if len(selected) >= max_context_frames:
                break
        if not added:
            break
    cited = {frame_id for interval in intervals for frame_id in interval["available_frame_ids"]}
    before_ids, after_ids = {f["frame_id"] for f in before}, {f["frame_id"] for f in after}
    frames = []
    for frame in source_frames:
        frame_id = frame["frame_id"]
        if frame_id in selected:
            roles = [role for role, applies in (("target", frame_id == target_id),
                     ("before", frame_id in before_ids), ("after", frame_id in after_ids),
                     ("qwen_context", frame_id in cited)) if applies]
            frames.append({**copy.deepcopy(frame), "evidence_roles": roles})
    limitations = ["Only supplied still images are reviewed; they cannot establish unseen motion or fill temporal gaps."]
    if timeline["timestamp_basis"] == "reconstructed_nominal":
        limitations.append("Timestamps are reconstructed nominal playback offsets; original acquisition PTS and "
                           "images between released frames are unavailable.")
    neighbors = {}
    for role, actual, requested in (("before", before, before_frames), ("after", after, after_frames)):
        unavailable = requested - len(actual)
        neighbors[role] = {"requested": requested, "included_frame_ids": [frame["frame_id"] for frame in actual],
                           "unavailable_count": unavailable,
                           "availability": "complete" if not unavailable else ("partial" if actual else "unavailable")}
        if unavailable:
            limitations.append(f"{unavailable} requested {role} frame(s) unavailable at the source sequence boundary.")
    for interval in intervals:
        interval["supplied_frame_ids"] = [frame_id for frame_id in interval["available_frame_ids"] if frame_id in selected]
        interval["omitted_frame_ids"] = [frame_id for frame_id in interval["available_frame_ids"] if frame_id not in selected]
        interval["complete"] = not interval["omitted_frame_ids"]
        interval["omission_reason"] = "image_budget" if interval["omitted_frame_ids"] else None
    if any(interval["omitted_frame_ids"] for interval in intervals):
        limitations.append("Some Qwen-cited source observations were not supplied because of the image budget; "
                           "see qwen_evidence_coverage for the exact omitted frame IDs.")
    context = _dataset_context(source, frames, dataset_root, procedure_context)
    if context["status"] == "unavailable":
        limitations.append(context["unavailable_reason"])
    return {"schema_version": PROTOCOL_VERSION, "target_frame_id": target_id, "frames": frames,
            "qwen_annotation": copy.deepcopy(annotation), "dataset_context": context,
            "media_timeline": timeline, "neighbor_coverage": neighbors,
            "qwen_evidence_coverage": intervals, "limitations": limitations}
