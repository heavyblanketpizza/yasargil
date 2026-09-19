"""Validate enhancement archives and export only reviewed multimodal messages.

These checks establish structural, provenance, and declared review consistency.
They cannot establish clinical truth or authenticate a claimed human review.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import re
import sysconfig
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

VERSION = "2.0.0"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas/enhancement-record.schema.json"
if not SCHEMA_PATH.is_file():
    SCHEMA_PATH = Path(sysconfig.get_path("data")) / "share/yasargil/schemas/enhancement-record.schema.json"


class ContractError(ValueError):
    """An archive or export violates the active contract."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    return value if isinstance(value, list) else [value]


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def resolve_asset(location, dataset_root, artifact_root=None):
    """Resolve local references only; traversal, remote URLs and symlink escapes fail."""
    root = artifact_root if location.startswith("artifact:") else dataset_root
    relative = location.removeprefix("artifact:") if location.startswith("artifact:") else location
    require(root is not None, f"A root is required to resolve {location}")
    require(relative and not Path(relative).is_absolute() and ":" not in relative,
            f"Asset location must be a relative local path: {location}")
    base = Path(root).resolve()
    candidate = (base / relative).resolve()
    require(candidate.is_relative_to(base), f"Asset escapes its root: {location}")
    require(candidate.is_file(), f"Missing asset: {location}")
    return candidate


def unique_index(items, key):
    result = {item[key]: item for item in items}
    require(len(result) == len(items), f"Duplicate {key}")
    return result


def references(ids, index, label):
    require(set(ids) <= index.keys(), f"Unknown {label}: {sorted(set(ids) - index.keys())}")


def _check_acyclic(index, parents, label):
    active, done = set(), set()

    def visit(key):
        require(key not in active, f"Cyclic {label}: {key}")
        if key in done:
            return
        active.add(key)
        for parent in parents(index[key]):
            visit(parent)
        active.remove(key)
        done.add(key)

    for key in index:
        visit(key)


def _source_row(locator, assets, paths, cache):
    asset_id = locator["asset_id"]
    require(asset_id in assets, "Unknown source locator asset")
    if locator["locator_type"] != "csv_row_1based_including_header" or not paths:
        return None
    path = paths[asset_id]
    if path not in cache:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            # Physical row number is preserved; quoted multiline rows use their ending line.
            cache[path] = {reader.line_num: row for row in reader}
    try:
        line = int(locator["locator"])
    except ValueError as exc:
        raise ContractError("CSV locator must be an integer row number") from exc
    require(line in cache[path], "Source CSV row is missing")
    return cache[path][line]


def _portable_messages(row):
    require(set(row) == {"messages"}, "Portable rows must contain only messages")
    messages = row["messages"]
    require(isinstance(messages, list) and len(messages) >= 2 and len(messages) % 2 == 0,
            "Conversation must contain complete user/assistant pairs")
    image_seen = False
    for i, message in enumerate(messages):
        require(set(message) == {"role", "content"}, "Unexpected message fields")
        require(message["role"] == ("user" if i % 2 == 0 else "assistant"), "Roles must alternate")
        require(isinstance(message["content"], list) and message["content"], "Empty message content")
        for block in message["content"]:
            kind = block.get("type")
            require(kind in {"text", "image"}, "Unsupported content block")
            require(set(block) == {"type", kind}, "Unexpected content block fields")
            require(isinstance(block[kind], str) and block[kind].strip(), "Empty or non-string content")
            if kind == "image":
                require(message["role"] == "user", "Images belong in user messages")
                image_seen = True
        if message["role"] == "assistant":
            require(image_seen, "Vision answer has no preceding image context")


def validate_record(record, *, dataset_root=None, artifact_root=None, training=False):
    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    error = next(validator.iter_errors(record), None)
    if error:
        raise ContractError(f"Schema {list(error.path)}: {error.message}")
    # Reject NaN/Infinity even when a permissive JSON parser admitted them.
    try:
        canonical_hash(record)
    except ValueError as exc:
        raise ContractError("Non-finite JSON number") from exc

    assets = unique_index(record["assets"], "asset_id")
    annotations = unique_index(record["original_annotations"], "annotation_id")
    frames = unique_index(record["frame_selection"]["frames"], "frame_id")
    transformations = unique_index(record["transformations"], "transformation_id")
    generations = unique_index(record["generation_runs"], "run_id")
    claims = unique_index(record["claims"], "claim_id")
    reviews = unique_index(record["reviews"], "review_id")
    outcomes = unique_index(record["case_outcomes"], "outcome_id")
    unique_index(record["source_conflicts"], "conflict_id")
    require(len({a["location"] for a in assets.values()}) == len(assets), "Duplicate asset locations")
    paths = {}
    if dataset_root is not None:
        from PIL import Image
        for aid, asset in assets.items():
            path = resolve_asset(asset["location"], dataset_root, artifact_root)
            require(sha256_file(path) == asset["sha256"], f"Asset hash mismatch: {aid}")
            paths[aid] = path
            if asset["media_type"].startswith("image/"):
                with Image.open(path) as im:
                    require(im.size == (asset.get("width"), asset.get("height")), f"Image dimensions mismatch: {aid}")
                    im.verify()
    if training:
        require(dataset_root is not None, "Training export requires source byte verification")

    source = record["source"]
    if training:
        require(source["dataset_name"] == "SOSpine", "Only the SOSpine source verifier is implemented for training export")
    references([source["source_manifest_asset_id"]], assets, "source manifest")
    require(assets[source["source_manifest_asset_id"]]["role"] == "source_manifest", "Wrong manifest role")
    references(source["license_evidence"]["source_asset_ids"], assets, "license evidence")
    for asset in assets.values():
        references(asset["derived_from_asset_ids"], assets, "asset ancestors")
        if asset["transformation_id"]:
            references([asset["transformation_id"]], transformations, "asset transformation")
            require(asset["asset_id"] in transformations[asset["transformation_id"]]["output_asset_ids"], "Asset transformation output mismatch")
    _check_acyclic(assets, lambda a: a["derived_from_asset_ids"], "asset ancestry")
    for frame in frames.values():
        references([frame["asset_id"]], assets, "frame asset")
        require(assets[frame["asset_id"]]["media_type"].startswith("image/"), "Frame asset must be an image")
        references(frame["context_frame_ids"], frames, "frame context")
        require(frame["frame_id"] not in frame["context_frame_ids"], "Frame cannot be its own context")
    require(len({f["frame_index"] for f in frames.values()}) == len(frames), "Duplicate sequence frame indices")
    require(len({f["asset_id"] for f in frames.values()}) == len(frames), "Image reused under multiple frame IDs")
    if record["frame_selection"]["video_asset_id"]:
        references([record["frame_selection"]["video_asset_id"]], assets, "video asset")

    cache = {}
    for aid, annotation in annotations.items():
        references(annotation["frame_ids"], frames, "annotation frames")
        if source["dataset_name"] == "SOSpine":
            locator = annotation["source_locator"]
            require(locator["locator_type"] == "csv_row_1based_including_header", "SOSpine annotations need exact CSV-row locators")
            references([locator["asset_id"]], assets, "annotation source")
            expected_table = {"keypoint": "sospine_tool_tips.csv", "bbox": "sospine_bbox.csv", "outcome": "sospine_outcomes.csv"}.get(annotation["original_kind"])
            require(expected_table and assets[locator["asset_id"]]["location"] == expected_table,
                    "SOSpine annotation kind/table mismatch; unsupported source adapter")
            require(annotation["original_origin"] == {"keypoint": "manual", "bbox": "computed", "outcome": "measured_metadata"}[annotation["original_kind"]], "SOSpine original annotation origin mismatch")
        row = _source_row(annotation["source_locator"], assets, paths, cache)
        if row is not None:
            require(row == annotation["raw_value"], f"Raw source annotation mismatch: {aid}")
        if source["dataset_name"] == "SOSpine" and annotation["frame_ids"]:
            raw = annotation["raw_value"]
            for fid in annotation["frame_ids"]:
                expected = Path(assets[frames[fid]["asset_id"]]["location"]).name
                require(isinstance(raw, dict) and raw.get("trial_frame") == expected, "Annotation/frame source join mismatch")

    for transform in transformations.values():
        references(transform["input_asset_ids"] + transform["output_asset_ids"], assets, "transformation assets")
        references(transform["input_annotation_ids"], annotations, "transformation annotations")
        references(transform["output_claim_ids"], claims, "transformation claims")
    _check_acyclic(assets, lambda a: a["derived_from_asset_ids"] + (
        transformations[a["transformation_id"]]["input_asset_ids"] if a["transformation_id"] else []),
        "asset transformation inputs")
    for run in generations.values():
        references(run["input_frame_ids"], frames, "generator frames")
        references(run["input_annotation_ids"], annotations, "generator annotations")
        references(run["input_reference_asset_ids"], assets, "generator references")
        references(run["outcome_ids_seen"], outcomes, "generator outcomes")
        references([run["request_asset_id"], run["response_asset_id"]], assets, "teacher artifacts")
        require(assets[run["request_asset_id"]]["role"] == "teacher_request" and assets[run["response_asset_id"]]["role"] == "teacher_response", "Wrong teacher artifact roles")
        seen_frames = set(run["input_frame_ids"])
        for aid in run["input_annotation_ids"]:
            seen_frames.update(annotations[aid]["frame_ids"])
            if run["input_mode"] == "causal_prefix":
                require(annotations[aid]["original_kind"] != "outcome", "Causal generator saw outcome annotations")
        require(all(frames[f]["frame_index"] <= run["maximum_frame_index_seen"] for f in seen_frames), "Generator exposure understates supplied frames")
        require(all(assets[a]["role"] == "reference_document" for a in run["input_reference_asset_ids"]), "Teacher general references require reference-document assets")

    messages = record["training_view"]["messages"]
    if messages:
        _portable_messages({"messages": messages})
    else:
        require(not training, "No training conversation")
    for claim in claims.values():
        evidence = claim["evidence"]
        references(evidence["frame_ids"], frames, "claim frames")
        references(evidence["original_annotation_ids"], annotations, "claim annotations")
        references(evidence["supporting_claim_ids"], claims, "supporting claims")
        references(evidence["reference_asset_ids"], assets, "claim reference assets")
        references(claim["review_ids"], reviews, "claim reviews")
        if claim["supersedes_claim_id"]:
            references([claim["supersedes_claim_id"]], claims, "superseded claim")
        if claim["generation_run_id"]:
            references([claim["generation_run_id"]], generations, "claim generator")
        if claim["transformation_id"]:
            references([claim["transformation_id"]], transformations, "claim transformation")
            require(claim["claim_id"] in transformations[claim["transformation_id"]]["output_claim_ids"], "Claim missing from transformation outputs")
        for region in evidence["regions"]:
            require(region["frame_id"] in evidence["frame_ids"], "Geometry is missing its evidence frame")
            if region["type"] == "bbox_xyxy":
                x1, y1, x2, y2 = region["coordinates"]
                require(x1 < x2 and y1 < y2, "Degenerate or inverted target geometry")
        for location in claim["output_locations"]:
            mi, bi = location["message_index"], location["content_block_index"]
            require(mi < len(messages) and messages[mi]["role"] == "assistant", "Claim location must target an assistant message")
            require(bi < len(messages[mi]["content"]), "Invalid claim content block")
            text = messages[mi]["content"][bi]["text"]
            start, end = location["start_character"], location["end_character"]
            require(0 <= start < end <= len(text) and text[start:end] == claim["text"], "Claim text differs from exact conversation span")
    _check_acyclic(claims, lambda c: c["evidence"]["supporting_claim_ids"] + ([c["supersedes_claim_id"]] if c["supersedes_claim_id"] else []), "claim evidence/revisions")

    for review in reviews.values():
        references(review["target_claim_ids"] + review["replacement_claim_ids"], claims, "review claims")
        require(all(i < len(messages) for i in review["target_message_indices"]), "Review targets a missing message")
    for claim in claims.values():
        require(all(claim["claim_id"] in reviews[r]["target_claim_ids"] for r in claim["review_ids"]), "Claim/review target mismatch")
        actual_reviews = {rid for rid, review in reviews.items() if claim["claim_id"] in review["target_claim_ids"]}
        require(set(claim["review_ids"]) == actual_reviews, "Claim must retain all applicable reviews, including negative judgments")
    for conflict in record["source_conflicts"]:
        references(conflict["annotation_ids"], annotations, "conflict annotations")
        references(conflict["claim_ids"], claims, "conflict claims")
        if conflict["resolution_review_id"]:
            references([conflict["resolution_review_id"]], reviews, "conflict resolution")
            require(set(conflict["claim_ids"]) <= set(reviews[conflict["resolution_review_id"]]["target_claim_ids"]), "Resolution review omits conflicting claims")

    for outcome in outcomes.values():
        if source["dataset_name"] == "SOSpine":
            locator = outcome["source_locator"]
            references([locator["asset_id"]], assets, "outcome source")
            require(locator["locator_type"] == "csv_row_1based_including_header" and assets[locator["asset_id"]]["location"] == "sospine_outcomes.csv", "SOSpine outcomes require exact outcome CSV rows")
        row = _source_row(outcome["source_locator"], assets, paths, cache)
        if source["dataset_name"] == "SOSpine" and row is not None:
            require(row.get("Trial ID") == source["case_id"], "Outcome belongs to another case")
            endpoint = outcome["endpoint"]
            require(endpoint in {"csf_leak_at_40_mmhg", "repair_duration_seconds"}, "Unsupported SOSpine outcome endpoint")
            if endpoint == "csf_leak_at_40_mmhg":
                require(row["Leak At 40mmHg"] in {"Y", "N", ""}, "Unknown raw leak code")
                expected = {"Y": True, "N": False, "": None}[row["Leak At 40mmHg"]]
                require(outcome["value"] is expected, "Outcome value does not match measured leak label")
            else:
                expected = float(row["Time for repair"]) if row["Time for repair"] else None
                require(outcome["value"] == expected, "Outcome duration differs from raw source")
    risk = record["patient_risk_adjustment"]
    if risk["status"] == "research_estimate":
        references(risk["baseline_annotation_ids"], annotations, "patient baseline")
        require(all(annotations[a]["original_kind"] == "baseline_patient_variable" for a in risk["baseline_annotation_ids"]), "Risk model needs baseline variables")
        for comparison in risk["comparisons"]:
            references([comparison["outcome_id"]], outcomes, "risk outcome")
            outcome = outcomes[comparison["outcome_id"]]
            require(outcome["scope"] == "clinical_patient_outcome" and outcome["status"] == "measured", "Risk comparison needs measured clinical outcome")
            require(outcome["value"] == comparison["observed_value_numeric"], "Risk observed value mismatch")
            require(math.isclose(comparison["observed_minus_expected"], comparison["observed_value_numeric"] - comparison["expected_value"], abs_tol=1e-9), "Risk comparison arithmetic mismatch")

    if source["dataset_name"] == "SOSpine":
        case = source["case_id"]
        match = re.fullmatch(r"(S[1-8])A[1-3]", case)
        require(source["surgeon_id"] == (match[1] if match else None), "SOSpine surgeon/case mismatch")
        require(record["frame_selection"]["sequence_id"] == case, "Sequence/case mismatch")
        for f in frames.values():
            asset = assets[f["asset_id"]]
            if asset["role"] == "released_frame":
                require(asset["location"] == f"frames/{case}/{case}_frame_{f['frame_index']:08d}.jpeg", "SOSpine frame path/index/case mismatch")
        if record["partition"]["name"] != "unassigned":
            require(match and record["partition"]["grouping_strategy"] == "surgeon", "SOSpine training splits require known surgeons")
            require(record["partition"]["group_ids"] == [source["surgeon_id"]], "SOSpine partition must group its surgeon")
    partition = record["partition"]
    if partition["name"] != "unassigned":
        require(partition["split_manifest_asset_id"] is not None, "Assigned split needs a manifest")
        references([partition["split_manifest_asset_id"]], assets, "split manifest")
        require(assets[partition["split_manifest_asset_id"]]["role"] == "split_manifest", "Wrong split manifest role")
        if paths:
            manifest = json.loads(paths[partition["split_manifest_asset_id"]].read_text())
            require(manifest.get("grouping_strategy") == partition["grouping_strategy"], "Split manifest grouping mismatch")
            require(all(manifest.get("assignments", {}).get(g) == partition["name"] for g in partition["group_ids"]), "Split assignment differs from manifest")

    quality = record["enrichment_quality"]
    references(quality["review_ids"], reviews, "quality reviews")
    for measurement in quality["measurements"]:
        references(measurement["supporting_review_ids"], reviews, "measurement reviews")
        if measurement["unit"] == "fraction":
            require(measurement["denominator"] is not None and 0 <= measurement["value"] <= 1, "Fraction needs a denominator and bounded value")
        if measurement["unit"] == "count":
            require(float(measurement["value"]).is_integer(), "Count must be an integer")
    for study in record["training_value"]["study_links"]:
        if study["evaluation_report_asset_id"]:
            references([study["evaluation_report_asset_id"]], assets, "evaluation report")
            require(assets[study["evaluation_report_asset_id"]]["role"] == "evaluation_report", "Wrong evaluation report role")

    from .teacher import validate_teacher_runs
    verified_teacher_runs = validate_teacher_runs(record, dataset_root=dataset_root, artifact_root=artifact_root)
    _validate_turns(record, frames, assets, annotations, claims, generations, reviews, training,
                    verified_teacher_runs)
    if training:
        require(record["status"] == "reviewed" and record["training_view"]["eligibility"] == "eligible", "Training requires a reviewed eligible archive")
        require(not record["training_view"]["exclusion_reasons"], "Eligible record has exclusion reasons")
        require(partition["name"] in {"train", "validation", "test"}, "Assign a partition before training export")
    return record


def _validate_turns(record, frames, assets, annotations, claims, generations, reviews, training,
                    verified_teacher_runs=None):
    view = record["training_view"]
    verified_runs = verified_teacher_runs or set()
    messages, turns = view["messages"], view["turn_links"]
    unique_index(turns, "turn_id")
    require(len(turns) * 2 == len(messages), "Every message pair needs one turn link")
    visible, cutoff_previous = set(), -1
    locations_to_frames = {assets[f["asset_id"]]["location"]: fid for fid, f in frames.items()}
    causal = record["intended_use"] == "causal_intraoperative_assistance"
    if causal:
        require(record["frame_selection"]["selector_exposure"] == "causal_prefix_only", "Causal examples cannot use future-informed selection")

    def closure(cid):
        result = {cid}
        for parent in claims[cid]["evidence"]["supporting_claim_ids"]:
            result.update(closure(parent))
        return result

    def positive_review(rid, clinical=False):
        review = reviews[rid]
        good = (review["verdict"] == "supported" and review["evidence_adequacy"] == "adequate"
                and review["temporal_correctness"] == "correct" and review["error_severity"] == "none")
        if causal:
            good = good and review["outcome_hidden_during_review"]
        if clinical:
            good = good and review["reviewer_role"] in {"surgeon", "clinical_domain_expert"} and review["clinical_appropriateness"] == "appropriate"
        return good

    def causal_asset(aid, cutoff):
        asset = assets[aid]
        require(asset["role"] not in {"case_metadata", "evaluation_report", "teacher_response", "original_video"}, "Causal input has unverified retrospective/video exposure")
        if asset["media_type"].startswith("image/"):
            matching = [f for f in frames.values() if f["asset_id"] == aid]
            require(matching and all(f["frame_index"] <= cutoff for f in matching), "Causal asset has future or unverified frame timing")
        for parent in asset["derived_from_asset_ids"]:
            causal_asset(parent, cutoff)
        if asset["transformation_id"]:
            transform = next(t for t in record["transformations"] if t["transformation_id"] == asset["transformation_id"])
            for parent in transform["input_asset_ids"]:
                require(parent != aid, "Asset transformation is self-referential")
                # The full transformation graph is checked for cycles before this traversal.
                causal_asset(parent, cutoff)
            for annotation_id in transform["input_annotation_ids"]:
                annotation = annotations[annotation_id]
                require(annotation["original_kind"] != "outcome", "Causal derivative used outcomes")
                require(all(frames[f]["frame_index"] <= cutoff for f in annotation["frame_ids"]), "Causal derivative used future annotation")

    for i, turn in enumerate(turns):
        ui, ai = turn["user_message_index"], turn["assistant_message_index"]
        require((ui, ai) == (i * 2, i * 2 + 1), "Turn indices must match ordered message pairs")
        cutoff = turn["cutoff_frame_index"]
        require(cutoff >= cutoff_previous, "Turn cutoff moves backwards")
        cutoff_previous = cutoff
        references(turn["student_frame_ids"], frames, "student frames")
        references(turn["student_annotation_ids"], annotations, "student annotations")
        references(turn["student_reference_asset_ids"], assets, "student references")
        references(turn["expressed_claim_ids"], claims, "turn claims")
        references(turn["generation_run_ids"], generations, "turn generators")
        references(turn["review_ids"], reviews, "turn reviews")
        actual_reviews = {rid for rid, review in reviews.items() if set(review["target_message_indices"]) & {ui, ai}}
        require(set(turn["review_ids"]) == actual_reviews, "Turn must retain every applicable message review")
        if turn["question_origin"] == "model_generated":
            require(turn["generation_run_ids"], "Generated question lacks a generator receipt")
        images = [b["image"] for b in messages[ui]["content"] if b["type"] == "image"]
        require(all(p in locations_to_frames for p in images), "Message image is not a selected frame")
        new_frames = [locations_to_frames[p] for p in images]
        require([frames[f]["frame_index"] for f in new_frames] == sorted(frames[f]["frame_index"] for f in new_frames), "Turn images are not chronological")
        visible.update(new_frames)
        require(set(turn["student_frame_ids"]) == visible, "Student frame manifest must match cumulative conversation images")
        actual_claims = {cid for cid, c in claims.items() if any(p["message_index"] == ai for p in c["output_locations"])}
        require(set(turn["expressed_claim_ids"]) == actual_claims, "Turn claims differ from actual assistant spans")
        if causal:
            require(turn["input_mode"] == "causal_prefix", "Causal record has a retrospective turn")
            require(all(frames[f]["frame_index"] <= cutoff for f in visible), "Future student image")
            require(all(annotations[a]["original_kind"] != "outcome" for a in turn["student_annotation_ids"]), "Outcome annotation supplied to causal student")
            for fid in visible:
                causal_asset(frames[fid]["asset_id"], cutoff)
        for aid in turn["student_annotation_ids"]:
            require(set(annotations[aid]["frame_ids"]) <= visible, "Student annotation concerns an unseen frame")
        require(all(assets[a]["role"] == "reference_document" for a in turn["student_reference_asset_ids"]), "Student references must be general documents")
        supporting = set().union(*(closure(cid) for cid in actual_claims)) if actual_claims else set()
        used_runs = set(turn["generation_run_ids"])
        for cid in supporting:
            claim = claims[cid]
            evidence = claim["evidence"]
            require(set(evidence["frame_ids"]) <= visible, "Claim cites a frame the student has not seen")
            for aid in evidence["original_annotation_ids"]:
                require(set(annotations[aid]["frame_ids"]) <= visible, "Claim annotation concerns an unseen frame")
                if causal:
                    require(annotations[aid]["original_kind"] != "outcome", "Causal claim uses a measured outcome as explanatory evidence")
            if claim["generation_run_id"]:
                require(claim["generation_run_id"] in used_runs, "Turn omits a claim's generation run")
            if claim["transformation_id"] and causal:
                transform = next(t for t in record["transformations"] if t["transformation_id"] == claim["transformation_id"])
                for aid in transform["input_annotation_ids"]:
                    require(annotations[aid]["original_kind"] != "outcome", "Causal text transformation used outcomes")
                    require(all(frames[f]["frame_index"] <= cutoff for f in annotations[aid]["frame_ids"]), "Text transformation used future annotations")
                for aid in transform["input_asset_ids"]:
                    causal_asset(aid, cutoff)
            if training:
                require(claim["disposition"] == "retained", "Training conversation/support contains an unretained claim")
                # A model's self-assigned type cannot waive clinical review of its text.
                clinical = (claim["type"] in {"clinical_interpretation", "suggested_check", "outcome_prediction"}
                            or claim["generation_run_id"] in verified_runs)
                require(any(positive_review(rid, clinical) for rid in claim["review_ids"]), "Claim lacks passing applicable review")
                require(not any(reviews[rid]["verdict"] != "supported" for rid in claim["review_ids"]), "Correct or supersede disputed claims before export")
                require(any(evidence[k] for k in ("frame_ids", "original_annotation_ids", "supporting_claim_ids", "reference_asset_ids")), "Retained claim lacks evidence")
        for rid in used_runs:
            run = generations[rid]
            require(all(mi < ui for mi in run["previous_message_indices"]), "Generator saw current/future conversation messages")
            if causal:
                require(run["input_mode"] == "causal_prefix" and not run["outcome_ids_seen"], "Causal turn used an outcome-exposed generator")
                require(run["maximum_frame_index_seen"] <= cutoff, "Generator saw future frames")
                for fid in run["input_frame_ids"]:
                    causal_asset(frames[fid]["asset_id"], cutoff)
        if training:
            require(used_runs <= verified_runs,
                    "Model-generated dialogue needs a verified teacher-request adapter")
            require(turn["task"] != "outcome_prediction", "Measured outcome prediction export needs a dedicated target adapter; not implemented in v2")
            # Original manual labels are teacher/target evidence unless a deployment input adapter exists.
            require(not turn["student_annotation_ids"], "Student annotation inputs need a deployment-availability adapter; not implemented")
            require(not turn["student_reference_asset_ids"], "Student reference injection needs an exact-text adapter; not implemented")
            clinical_turn = turn["task"] == "decision_support" or bool(used_runs & verified_runs)
            require(any(positive_review(rid, clinical_turn)
                        and {ui, ai} <= set(reviews[rid]["target_message_indices"])
                        and reviews[rid]["question_answerability"] == "from_supplied_input"
                        for rid in turn["review_ids"]), "Turn needs reviewed input/answer and answerability")
            require(all(positive_review(rid, clinical_turn) for rid in turn["review_ids"]), "Turn has unresolved adverse or incomplete review")
            for bi, block in enumerate(messages[ai]["content"]):
                covered = set()
                for cid in actual_claims:
                    for span in claims[cid]["output_locations"]:
                        if (span["message_index"], span["content_block_index"]) == (ai, bi):
                            covered.update(range(span["start_character"], span["end_character"]))
                require(all(j in covered or char.isspace() for j, char in enumerate(block["text"])), "Assistant text contains an unreviewed span")
            for conflict in record["source_conflicts"]:
                relevant_annotations = {aid for cid in supporting for aid in claims[cid]["evidence"]["original_annotation_ids"]}
                if set(conflict["claim_ids"]) & supporting or set(conflict["annotation_ids"]) & relevant_annotations:
                    require(conflict["status"] == "resolved", "Relevant source conflict remains unresolved")


def validate_corpus(records, **kwargs):
    require(bool(records), "Empty corpus")
    unique_index(records, "record_id")
    partitions = {}
    for record in records:
        validate_record(record, **kwargs)
        part = record["partition"]
        source = record["source"]
        keys = [("case", source["dataset_name"], source["case_id"])]
        keys += [("group", source["dataset_name"], part["grouping_strategy"], g) for g in part["group_ids"]]
        keys += [(key, source["dataset_name"], source[key]) for key in ("surgeon_id", "patient_id") if source[key]]
        keys += [("image", a["sha256"]) for a in record["assets"] if a["media_type"].startswith("image/")]
        for key in keys:
            if key in partitions:
                require(partitions[key] == part["name"], f"Cross-partition source leakage: {key}")
            partitions[key] = part["name"]
    return records


def export_records(records, output, *, dataset_root, artifact_root=None, preview=False,
                   partition=None, allow_retrospective=False):
    # Validate all groups before selecting a partition.
    validate_corpus(records, dataset_root=dataset_root, artifact_root=artifact_root)
    selected = [r for r in records if partition is None or r["partition"]["name"] == partition]
    require(bool(selected), "No selected records")
    output = Path(output)
    receipt_path = output.with_suffix(".receipt.json")
    require(output.suffix == ".jsonl", "Portable export must use a .jsonl filename")
    require(not output.resolve().is_relative_to(Path(dataset_root).resolve()), "Export output must be outside the source dataset")
    require(not output.exists() and not receipt_path.exists(), "Export or receipt already exists")
    if preview:
        require(output.name.endswith(".preview.jsonl"), "Preview name must end .preview.jsonl")
    else:
        require(partition in {"train", "validation", "test"}, "Select an explicit training/evaluation partition")
        for record in selected:
            validate_record(record, dataset_root=dataset_root, artifact_root=artifact_root, training=True)
    for getter, label in ((lambda r: r["training_view"]["loss_scope"], "loss policies"),
                          (lambda r: r["intended_use"], "causal and retrospective uses"),
                          (lambda r: r["partition"]["name"], "partitions")):
        require(len({getter(r) for r in selected}) == 1, f"Separate exports for different {label}")
    require(allow_retrospective or selected[0]["intended_use"] == "causal_intraoperative_assistance", "Retrospective export requires --allow-retrospective")
    rows = [{"messages": copy.deepcopy(r["training_view"]["messages"])} for r in selected]
    for row in rows:
        _portable_messages(row)
    media = {}
    for record in selected:
        used = {b["image"] for m in record["training_view"]["messages"] for b in m["content"] if b["type"] == "image"}
        for asset in record["assets"]:
            if asset["location"] in used:
                entry = {k: asset[k] for k in ("location", "sha256", "width", "height")}
                require(asset["location"] not in media or media[asset["location"]] == entry, "Conflicting exported image bytes")
                media[asset["location"]] = entry
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    receipt = {
        "schema_version": VERSION, "purpose": "format_preview_only" if preview else "reviewed_sft_export",
        "row_count": len(rows), "partition": selected[0]["partition"]["name"],
        "loss_scope": selected[0]["training_view"]["loss_scope"], "intended_use": selected[0]["intended_use"],
        "schema_sha256": sha256_file(SCHEMA_PATH), "output_sha256": sha256_file(output),
        "records": [{"record_id": r["record_id"], "revision_id": r["revision"]["revision_id"],
                     "archive_sha256": canonical_hash(r), "row_sha256": canonical_hash(row)}
                    for r, row in zip(selected, rows)], "media": list(media.values()),
    }
    write_new_json(receipt_path, receipt)
    return receipt


def hydrate_messages(row, dataset_root, artifact_root=None):
    from PIL import Image
    _portable_messages(row)
    result = copy.deepcopy(row)
    for message in result["messages"]:
        for block in message["content"]:
            if block["type"] == "image":
                with Image.open(resolve_asset(block["image"], dataset_root, artifact_root)) as im:
                    block["image"] = im.convert("RGB")
    return result


def load_export(path, dataset_root, *, artifact_root=None, allow_preview=False,
                expected_partition="train", expected_loss_scope="final_assistant_turn",
                expected_intended_use="causal_intraoperative_assistance"):
    """Verify receipts, then eagerly load a bounded experiment's images into memory."""
    path = Path(path)
    receipt = json.loads(path.with_suffix(".receipt.json").read_text())
    required = {"schema_version", "purpose", "row_count", "partition", "loss_scope", "intended_use",
                "schema_sha256", "output_sha256", "records", "media"}
    require(set(receipt) == required, "Malformed export receipt fields")
    require(receipt["schema_version"] == VERSION, "Wrong receipt version")
    require(receipt["purpose"] in {"format_preview_only", "reviewed_sft_export"}, "Unknown export purpose")
    require(receipt["purpose"] == "reviewed_sft_export" or allow_preview, "Draft preview is not reviewed training data")
    require(receipt["partition"] == expected_partition, "Requested partition differs from receipt")
    require(receipt["loss_scope"] == expected_loss_scope, "Requested loss policy differs from receipt")
    require(receipt["intended_use"] == expected_intended_use, "Requested temporal use differs from receipt")
    require(receipt["schema_sha256"] == sha256_file(SCHEMA_PATH), "Schema digest mismatch")
    require(receipt["output_sha256"] == sha256_file(path), "Export digest mismatch")
    rows = read_records(path)
    require(len(rows) == receipt["row_count"] == len(receipt["records"]) and rows, "Receipt row count mismatch")
    unique_index(receipt["records"], "record_id")
    media = unique_index(receipt["media"], "location")
    used = set()
    for record, row in zip(receipt["records"], rows):
        require(set(record) == {"record_id", "revision_id", "archive_sha256", "row_sha256"}, "Malformed row receipt")
        require(canonical_hash(row) == record["row_sha256"], "Receipt row binding mismatch")
        _portable_messages(row)
        used.update(b["image"] for m in row["messages"] for b in m["content"] if b["type"] == "image")
    require(used == media.keys(), "Receipt media does not match conversation images")
    from PIL import Image
    for entry in media.values():
        require(set(entry) == {"location", "sha256", "width", "height"}, "Malformed media receipt")
        asset_path = resolve_asset(entry["location"], dataset_root, artifact_root)
        require(sha256_file(asset_path) == entry["sha256"], "Export image hash mismatch")
        with Image.open(asset_path) as im:
            require(im.size == (entry["width"], entry["height"]), "Export image dimensions mismatch")
    return [hydrate_messages(row, dataset_root, artifact_root) for row in rows]
