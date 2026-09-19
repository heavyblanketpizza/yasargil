"""Bounded, deterministic SOSpine import. Never writes the source dataset.

Released frames are selected by explicit index, without visual or outcome-based
ranking. Imported labels propose training targets; human review remains pending.
"""
from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .contract import VERSION, canonical_hash, require, resolve_asset, sha256_file, validate_record

POINTS = "sospine_tool_tips.csv"
BOXES = "sospine_bbox.csv"
OUTCOMES = "sospine_outcomes.csv"
CLASSES = {"grasper": "grasper", "needle driver base": "needle driver", "needle driver tip": "needle driver"}


def _rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return [(reader.line_num, row) for row in reader]


def _asset(root, location, role, asset_id, media_type):
    path = resolve_asset(location, root)
    result = {"asset_id": asset_id, "role": role, "origin": "original_dataset",
              "location": location, "sha256": sha256_file(path), "media_type": media_type,
              "derived_from_asset_ids": [], "transformation_id": None}
    if media_type.startswith("image/"):
        with Image.open(path) as im:
            require(im.format == "JPEG", "SOSpine released image is not JPEG")
            result.update(width=im.width, height=im.height)
            im.verify()
    return result


def _locator(asset_id, line):
    return {"asset_id": asset_id, "locator_type": "csv_row_1based_including_header", "locator": str(line)}


def _import_outcomes(root, case_id):
    """Outcome import has no role in choosing frames or building answers."""
    matches = [(line, raw) for line, raw in _rows(root / OUTCOMES) if raw["Trial ID"] == case_id]
    require(len(matches) == 1, "Expected one exact outcome/metadata row for the case")
    line, raw = matches[0]
    require(raw["Leak At 40mmHg"] in {"Y", "N", ""}, "Unknown leak code")
    leak = {"Y": True, "N": False, "": None}[raw["Leak At 40mmHg"]]
    duration = float(raw["Time for repair"]) if raw["Time for repair"] else None
    outcomes = []
    for endpoint, value, unit, timepoint in (
        ("csf_leak_at_40_mmhg", leak, "boolean_leak_present", "post_repair_pressurization_40_mmhg"),
        ("repair_duration_seconds", duration, "seconds", "completion_of_simulated_repair"),
    ):
        outcomes.append({"outcome_id": f"{case_id}.{endpoint}", "scope": "simulated_technical_outcome",
                         "endpoint": endpoint, "status": "missing" if value is None else "measured",
                         "value": value, "unit": unit, "measurement_timepoint": timepoint,
                         "followup_days": None, "source_locator": _locator("outcomes-csv", line),
                         "missing_reason": "not_reported" if value is None else None})
    annotation = {"annotation_id": f"outcomes-row-{line}", "source_locator": _locator("outcomes-csv", line),
                  "original_kind": "outcome", "original_origin": "measured_metadata", "raw_value": raw,
                  "frame_ids": [], "usage": {"considered": True, "used_for_generation": False,
                  "used_as_target_evidence": False, "exclusion_reason": "Retrospective case outcomes and surgeon experience; excluded from causal answer generation."}}
    return outcomes, annotation


def import_case(dataset_root, case_id, frame_indices):
    root = Path(dataset_root)
    require(re.fullmatch(r"S[1-8]A[1-3]|Clip[01]", case_id), "Unknown SOSpine case ID syntax")
    require(1 <= len(frame_indices) <= 32, "Select between 1 and 32 released frames per bounded record")
    require(all(type(i) is int and i > 0 for i in frame_indices), "Frame indices must be positive integers")
    require(frame_indices == sorted(set(frame_indices)), "Frame indices must be unique and increasing")
    assets = [
        _asset(root, "metadata/source_manifest.json", "source_manifest", "source-manifest", "application/json"),
        _asset(root, "documentation/readme.txt", "license_notice", "author-readme", "text/plain"),
        _asset(root, POINTS, "annotation_table", "points-csv", "text/csv"),
        _asset(root, BOXES, "annotation_table", "boxes-csv", "text/csv"),
        _asset(root, OUTCOMES, "case_metadata", "outcomes-csv", "text/csv"),
    ]
    frames, filenames = [], {}
    for index in frame_indices:
        fid = f"f{index:06d}"
        filename = f"{case_id}_frame_{index:08d}.jpeg"
        asset_id = f"image-{fid}"
        assets.append(_asset(root, f"frames/{case_id}/{filename}", "released_frame", asset_id, "image/jpeg"))
        filenames[filename] = fid
        frames.append({"frame_id": fid, "asset_id": asset_id, "frame_index": index,
                       "timestamp_ms": None, "timestamp_basis": "unavailable", "role": "keyframe",
                       "selection_reason": "Explicit release index; no image-content or outcome ranking.",
                       "context_frame_ids": [f["frame_id"] for f in frames]})
    annotations = []
    for filename, asset_id, kind, origin in (
        (POINTS, "points-csv", "keypoint", "manual"),
        (BOXES, "boxes-csv", "bbox", "computed"),
    ):
        for line, raw in _rows(root / filename):
            if raw.get("trial_frame") not in filenames:
                continue
            fid = filenames[raw["trial_frame"]]
            usable = kind == "keypoint" and raw.get("label", "").strip() in CLASSES
            annotations.append({"annotation_id": f"{asset_id}-row-{line}", "source_locator": _locator(asset_id, line),
                                "original_kind": kind, "original_origin": origin, "raw_value": raw,
                                "frame_ids": [fid], "usage": {"considered": True, "used_for_generation": usable,
                                "used_as_target_evidence": usable, "exclusion_reason": None if usable else
                                "Retained original row; baseline uses only positive grasper/needle-driver point labels, without geometry conversion."}})

    # Build targets from selected frame labels only. Outcome import happens later.
    messages, turns, claims, transforms, visible = [], [], [], [], []
    image_assets = {a["asset_id"]: a for a in assets}
    for frame in frames:
        fid = frame["frame_id"]
        relevant = [a for a in annotations if a["original_kind"] == "keypoint" and fid in a["frame_ids"]
                    and a["raw_value"].get("label", "").strip() in CLASSES]
        labels = sorted({CLASSES[a["raw_value"]["label"].strip()] for a in relevant})
        if not labels:
            continue  # Missing labels cannot support an absence or confident-uncertainty target.
        ui = len(messages)
        ai = ui + 1
        question = ("Research task: identify the grasper and needle driver when visible in the supplied microscope image. "
                    "Use only the images available at this turn; avoid inferring absence from incomplete visibility. "
                    f"Assess frame {fid}, release index {frame['frame_index']}; acquisition time is unavailable.")
        messages.append({"role": "user", "content": [{"type": "text", "text": question},
                         {"type": "image", "image": image_assets[frame["asset_id"]]["location"]}]})
        content, claim_ids = [], []
        for label in labels:
            cid = f"{fid}.{label.replace(' ', '_')}"
            tid = f"text-{cid}"
            text = f"A {label} is visible in frame {fid}."
            ann_ids = [a["annotation_id"] for a in relevant if CLASSES[a["raw_value"]["label"].strip()] == label]
            claims.append({"claim_id": cid, "text": text, "type": "visible_observation", "origin": "source_reexpression",
                           "contribution": "restates_existing_information", "generation_run_id": None,
                           "transformation_id": tid, "supersedes_claim_id": None,
                           "evidence": {"frame_ids": [fid], "original_annotation_ids": ann_ids, "supporting_claim_ids": [],
                                        "reference_asset_ids": [], "regions": []},
                           "output_locations": [{"message_index": ai, "content_block_index": len(content),
                                                 "start_character": 0, "end_character": len(text)}],
                           "review_ids": [], "disposition": "pending"})
            transforms.append({"transformation_id": tid, "operation": "deterministic_text",
                               "implementation": "yasargil.sospine.import_case", "implementation_version": VERSION,
                               "input_asset_ids": [frame["asset_id"]], "input_annotation_ids": ann_ids,
                               "output_asset_ids": [], "output_claim_ids": [cid],
                               "parameters": {"trim_label_whitespace": True, "instrument_mapping": CLASSES},
                               "description": "Positive instrument-class re-expression from manual point labels; coordinates are not converted. No outcome inputs."})
            content.append({"type": "text", "text": text})
            claim_ids.append(cid)
        messages.append({"role": "assistant", "content": content})
        visible.append(fid)
        turns.append({"turn_id": f"turn-{len(turns) + 1}", "user_message_index": ui, "assistant_message_index": ai,
                      "task": "instrument_identification", "input_mode": "causal_prefix", "cutoff_frame_index": frame["frame_index"],
                      "student_frame_ids": visible.copy(), "student_annotation_ids": [], "student_reference_asset_ids": [],
                      "expressed_claim_ids": claim_ids, "question_origin": "deterministic_template", "generation_run_ids": [], "review_ids": []})
    outcomes, case_annotation = _import_outcomes(root, case_id)
    annotations.append(case_annotation)
    selection_hash = canonical_hash(frame_indices)[:12]
    surgeon = case_id[:2] if case_id.startswith("S") else None
    record = {
        "schema_version": VERSION, "record_id": f"sospine.{case_id}.baseline.{selection_hash}",
        "revision": {"revision_id": f"sospine.{case_id}.baseline.{selection_hash}.r1", "parent_revision_id": None,
                     "created_at": datetime.now(timezone.utc).isoformat()},
        "status": "draft", "intended_use": "causal_intraoperative_assistance",
        "source": {"dataset_name": "SOSpine", "dataset_version": "downloaded_release_snapshot",
                   "dataset_uri": "https://figshare.com/projects/Simulated_Outcomes_for_Durotomy_Repair_in_Minimally_Invasive_Spine_Surgery_SOSpine_/142508",
                   "setting": "cadaveric_simulation", "procedure": "simulated_spinal_durotomy_repair",
                   "case_id": case_id, "surgeon_id": surgeon, "patient_id": None, "institution_id": None,
                   "source_manifest_asset_id": "source-manifest",
                   "license_evidence": {"status": "conflicting_notices", "source_asset_ids": ["source-manifest", "author-readme"],
                                        "notes": "Audited release: Figshare metadata says CC BY 4.0; author readme says CC BY-NC 4.0. Unresolved; preserve both notices."}},
        "assets": assets, "original_annotations": annotations,
        "frame_selection": {"sequence_id": case_id, "video_asset_id": None, "released_fps": 1,
                            "method": "explicit_indices",
                            "selector_version": "explicit-release-indices-v1", "selector_exposure": "causal_prefix_only",
                            "parameters": {"requested_frame_indices": frame_indices, "content_based_selection": False},
                            "frames": frames, "known_sampling_limitations": ["Released 1-fps stills; original video/PTS unavailable in inspected release.",
                            "No recovery of motion between samples. Explicit indices must be chosen without future or outcome knowledge for causal experiments."]},
        "transformations": transforms, "generation_runs": [], "claims": claims, "source_conflicts": [], "reviews": [],
        "enrichment_quality": {"status": "not_assessed", "protocol_id": None, "review_ids": [], "measurements": []},
        "case_outcomes": outcomes, "patient_risk_adjustment": {"status": "unavailable", "reason": "cadaveric_simulation_without_patient_baseline_or_recovery"},
        "training_view": {"dialogue_origin": "synthetic_training_dialogue", "eligibility": "pending_review", "exclusion_reasons": [],
                          "assistant_format": "plain_text", "objective": "supervised_fine_tuning", "loss_scope": "final_assistant_turn",
                          "messages": messages, "turn_links": turns},
        "partition": {"name": "unassigned", "grouping_strategy": "surgeon", "group_ids": [surgeon] if surgeon else [], "split_manifest_asset_id": None},
        "training_value": {"status": "not_evaluated", "study_links": []},
    }
    validate_record(record, dataset_root=root)
    return record
