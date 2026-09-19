"""Bounded Qwen discovery and MedGemma inspection of released SOSpine frames.

The llama.cpp transport carries ordered original images, not native video tensors.
All model output is a draft. Exact requests, responses and revisions are kept.
"""
from __future__ import annotations

import html
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .contract import (SCHEMA_PATH, ContractError, canonical_hash, require, resolve_asset,
                       sha256_file, validate_record, write_new_json)
from .checkpoint import atomic_bytes, atomic_json, directory_lock, durable_mkdir
from .sospine import import_case
from .llama_cpp import LlamaCppClient, MEDGEMMA_MODEL, QWEN_MODEL
from .teacher import EVENT_QUESTION, frame_caption


@dataclass(frozen=True)
class EnhancementConfig:
    case_id: str
    start_index: int
    cutoff_index: int
    initial_frames: int = 8
    search_frames: int = 4
    max_frames: int = 24
    max_rounds: int = 1
    qwen_model: str = QWEN_MODEL
    medgemma_model: str = MEDGEMMA_MODEL
    num_ctx: int = 65536
    num_predict: int = 4096
    seed: int = 42

    def check(self):
        require(bool(re.fullmatch(r"S[1-8]A[1-3]|Clip[01]", self.case_id)), "Invalid SOSpine case ID")
        require(1 <= self.start_index <= self.cutoff_index, "Invalid inclusive frame window")
        require(self.cutoff_index - self.start_index < 3600, "A window is limited to 3600 released indices")
        require(1 <= self.initial_frames <= self.max_frames <= 32, "Require 1 <= initial_frames <= max_frames <= 32")
        require(1 <= self.search_frames <= self.max_frames, "Search frame budget is invalid")
        require(0 <= self.max_rounds <= 3, "Use 0–3 targeted search rounds")
        require(8192 <= self.num_ctx <= 131072, "Context must be between 8192 and 131072 tokens")
        require(512 <= self.num_predict <= 8192, "Output budget must be between 512 and 8192 tokens")
        require(self.qwen_model != self.medgemma_model, "Discovery and medical review must use distinct models")


def uniform_indices(indices, count):
    """Deterministic evenly spaced indices, including both endpoints when possible."""
    indices = sorted(set(indices))
    if not indices or count <= 0:
        return []
    if len(indices) <= count:
        return indices
    if count == 1:
        return [indices[len(indices) // 2]]
    return [indices[round(i * (len(indices) - 1) / (count - 1))] for i in range(count)]


def plan_enhancement(dataset_root, config):
    """Inspect filenames within the requested window, without model calls or writes."""
    config.check()
    root = Path(dataset_root).resolve()
    directory = root / "frames" / config.case_id
    require(directory.is_dir() and directory.resolve().is_relative_to(root), "Missing or escaped source sequence")
    pattern = re.compile(re.escape(config.case_id) + r"_frame_(\d{8})\.jpeg")
    pool = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match and config.start_index <= int(match[1]) <= config.cutoff_index:
            require(path.resolve().is_relative_to(root) and path.is_file(), "Source frame escapes dataset root")
            pool.append(int(match[1]))
    pool.sort()
    require(pool, "No released frames in the requested window")
    return {"config": asdict(config), "available_frame_indices": pool,
            "initial_frame_indices": uniform_indices(pool, config.initial_frames),
            "transport": "llama_cpp_ordered_images", "native_video_processor": False,
            "grounder": "qwen_targeted_reinspection", "outcomes_supplied_to_models": False,
            "maximum_model_calls": 3 + 2 * config.max_rounds,
            "timestamp_basis": "release_indices_only; original capture timestamps unavailable"}


def select_search_indices(searches, pool, seen, count):
    """Round-robin across requested intervals so the first query cannot take all slots."""
    candidates = [uniform_indices([i for i in pool if s["start_frame_index"] <= i <= s["end_frame_index"]
                                  and i not in seen], count) for s in searches]
    picked = []
    for offset in range(count):
        for group in candidates:
            if offset < len(group) and group[offset] not in picked:
                picked.append(group[offset])
                if len(picked) == count:
                    return sorted(picked)
    return sorted(picked)


def _bytes_new(path, data):
    durable_mkdir(path.parent)
    atomic_bytes(path, data)


def _artifact(path, output_dir, asset_id, role, *, origin="model_generated"):
    return {"asset_id": asset_id, "role": role, "origin": origin,
            "location": "artifact:" + path.relative_to(output_dir).as_posix(),
            "sha256": sha256_file(path), "media_type": "application/json",
            "derived_from_asset_ids": [], "transformation_id": None}


def _source_record(root, config, indices):
    record = import_case(root, config.case_id, sorted(indices))
    # The deterministic baseline is a separate comparison, not part of these model proposals.
    record["claims"] = []
    record["transformations"] = []
    record["training_view"]["messages"] = []
    record["training_view"]["turn_links"] = []
    for annotation in record["original_annotations"]:
        annotation["usage"].update(used_for_generation=False, used_as_target_evidence=False)
        if annotation["original_kind"] != "outcome":
            annotation["usage"]["exclusion_reason"] = "Not yet supplied to a model in this run."
    return record


def _merge_source(record, refreshed):
    """Refresh original assets only, retaining every successful call and its artifacts."""
    original_ids = {a["asset_id"] for a in refreshed["assets"]}
    record["assets"] = refreshed["assets"] + [a for a in record["assets"] if a["asset_id"] not in original_ids]
    record["original_annotations"] = refreshed["original_annotations"]
    record["frame_selection"]["frames"] = refreshed["frame_selection"]["frames"]


def _claims_for_output(run_id, output):
    result = []
    for item in output["events"] + output["questions"]:
        event = "event_id" in item
        result.append({
            "claim_id": f"{run_id}.{item['event_id']}" if event else f"{run_id}.{item['question_id']}.answer",
            "text": item["description"] if event else item["answer"],
            "type": item["type"] if event else (
                "uncertainty_statement" if item["answerability"] == "uncertain" else "visible_observation"),
            "origin": "model_generated", "contribution": "adds_proposed_observation",
            "generation_run_id": run_id, "transformation_id": None, "supersedes_claim_id": None,
            "evidence": {"frame_ids": item["evidence_frame_ids"], "original_annotation_ids": item["annotation_ids"],
                         "supporting_claim_ids": [], "reference_asset_ids": [], "regions": []},
            "output_locations": [], "review_ids": [], "disposition": "pending",
        })
        if event and item["uncertainty"].strip():
            uncertainty = dict(result[-1])
            uncertainty.update(claim_id=f"{run_id}.{item['event_id']}.uncertainty",
                               text=item["uncertainty"], type="uncertainty_statement")
            result.append(uncertainty)
    return result


def _make_conversation(record, outputs, final_run_id, cutoff):
    record["claims"] = [claim for run_id, output in outputs.items() for claim in _claims_for_output(run_id, output)]
    claims = {c["claim_id"]: c for c in record["claims"]}
    final = outputs[final_run_id]
    frames = record["frame_selection"]["frames"]
    assets = {a["asset_id"]: a for a in record["assets"]}
    messages, turns = [], []

    def add_turn(question, ids, origin, task):
        if not ids:
            return
        ui = len(messages)
        blocks = [{"type": "text", "text": question}]
        if not messages:
            for frame in frames:
                blocks.extend([{"type": "text", "text": frame_caption(frame)},
                               {"type": "image", "image": assets[frame["asset_id"]]["location"]}])
        answers = []
        for cid in ids:
            text = claims[cid]["text"]
            claims[cid]["output_locations"] = [{"message_index": ui + 1, "content_block_index": len(answers),
                                                 "start_character": 0, "end_character": len(text)}]
            answers.append({"type": "text", "text": text})
        messages.extend([{"role": "user", "content": blocks}, {"role": "assistant", "content": answers}])
        turns.append({"turn_id": f"turn-{len(turns) + 1}", "user_message_index": ui, "assistant_message_index": ui + 1,
                      "task": task, "input_mode": "causal_prefix", "cutoff_frame_index": cutoff,
                      "student_frame_ids": [f["frame_id"] for f in frames], "student_annotation_ids": [],
                      "student_reference_asset_ids": [], "expressed_claim_ids": ids,
                      "question_origin": origin, "generation_run_ids": list(outputs), "review_ids": []})

    event_claims = []
    for event in final["events"]:
        if event["assessment"] != "contradicted":
            event_claims.append(f"{final_run_id}.{event['event_id']}")
            if event["uncertainty"].strip():
                event_claims.append(f"{final_run_id}.{event['event_id']}.uncertainty")
    add_turn(EVENT_QUESTION, event_claims, "deterministic_template", "evidence_based_assessment")
    for question in final["questions"]:
        add_turn(question["question"], [f"{final_run_id}.{question['question_id']}.answer"], "model_generated",
                 "uncertainty_assessment" if question["answerability"] == "uncertain" else "evidence_based_assessment")
    record["training_view"].update(messages=messages, turn_links=turns, loss_scope="all_assistant_turns")
    used_annotations = {aid for run in record["generation_runs"] for aid in run["input_annotation_ids"]}
    target_annotations = {aid for c in record["claims"] if c["output_locations"] for aid in c["evidence"]["original_annotation_ids"]}
    for annotation in record["original_annotations"]:
        aid = annotation["annotation_id"]
        annotation["usage"].update(used_for_generation=aid in used_annotations,
                                   used_as_target_evidence=aid in target_annotations)
        if aid in used_annotations:
            annotation["usage"]["exclusion_reason"] = None


def _write_audit(output_dir, record, calls, completion):
    """Readable, explicitly unblinded audit of model changes. Images use the review packet."""
    def block(value):
        return "<pre>" + html.escape(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; base-uri \'none\'">',
             '<title>Yasargil model inspection audit</title><style>body{max-width:1050px;margin:2rem auto;padding:0 1rem;font:16px/1.5 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:1rem}section{border-top:1px solid #ccc;margin-top:2rem}a{color:#175b8e}</style>',
             '<h1>Model inspection audit</h1><p>Model identities are visible here. All descriptions and answers are pending human review. Model agreement is not a human judgment.</p>',
             '<p><a href="review.html">Inspect images, original labels, and the proposed conversation</a></p>', block(completion)]
    previous = {}
    for call in calls:
        output = call["output"]
        revisions = []
        for event in output["events"]:
            eid = event["event_id"]
            if eid in previous and previous[eid] != event:
                revisions.append({"event_id": eid, "previous": previous[eid], "revised": event})
        # Only review stages are revisions; independent observations are separate assessments.
        if call["stage"] in {"propose", "review", "final_revise"}:
            previous = {e["event_id"]: e for e in output["events"]}
        parts.extend(["<section><h2>" + html.escape(call["run_id"]) + "</h2>",
                      block({k: call[k] for k in ("model", "stage", "input_frame_ids", "parent_run_ids", "elapsed_seconds")}),
                      "<h3>Output</h3>", block(output)])
        if revisions and call["stage"] in {"review", "final_revise"}:
            parts.extend(["<h3>Changes to matching event IDs</h3>", block(revisions)])
        parts.append("</section>")
    parts.append("</html>")
    with (output_dir / "audit.html").open("x", encoding="utf-8") as stream:
        stream.write("".join(parts))


class _PauseRequested(Exception):
    pass


def _source_snapshot(root, plan, record):
    locations = {a["location"] for a in record["assets"] if a["origin"] == "original_dataset"}
    case = plan["config"]["case_id"]
    locations.update(f"frames/{case}/{case}_frame_{index:08d}.jpeg"
                     for index in plan["available_frame_indices"])
    return {location: sha256_file(resolve_asset(location, root)) for location in sorted(locations)}


def enhancement_protocol_fingerprint():
    from .enhancement_prompts import PROMPT_VERSION, STAGES, response_schema, stage_question, system_prompt
    from .teacher import ADAPTER, ENVELOPE_BUILDER_VERSION
    return {"checkpoint_version": "1", "prompt_version": PROMPT_VERSION, "adapter": ADAPTER,
            "envelope_builder_version": ENVELOPE_BUILDER_VERSION,
            "archive_schema_sha256": sha256_file(SCHEMA_PATH),
            "prompts_sha256": canonical_hash({"format": response_schema(),
                "stages": {stage: {"system": system_prompt(stage), "question": stage_question(stage)}
                           for stage in STAGES}})}


def _model_identity(info):
    return {key: info[key] for key in ("model_name", "model_digest", "quantization", "runtime_version",
                                      "runtime", "model_file", "projector_file", "runtime_binary")}


def _read_json(path):
    require(path.is_file() and not path.is_symlink(), f"Missing or symlinked checkpoint artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ContractError(f"Invalid checkpoint JSON: {path}") from exc


def _new_or_equal_json(path, value):
    if path.exists() or path.is_symlink():
        require(_read_json(path) == value, f"Existing checkpoint artifact differs: {path}")
    else:
        atomic_json(path, value)


def _attempt_directories(call_dir):
    if not call_dir.exists():
        return []
    require(call_dir.is_dir() and not call_dir.is_symlink(), "Invalid retained call directory")
    # Upgrade durability of receipts created before parent-directory syncing
    # was introduced, before treating a completed call as reusable.
    durable_mkdir(call_dir.parent)
    durable_mkdir(call_dir)
    result = [call_dir]
    container = call_dir / "attempts"
    if container.exists():
        require(container.is_dir() and not container.is_symlink(), "Invalid call attempts directory")
        durable_mkdir(container)
        for path in sorted(container.iterdir()):
            require(path.is_dir() and not path.is_symlink() and re.fullmatch(r"\d{4}", path.name),
                    "Unrecognized retained attempt directory")
            durable_mkdir(path)
            result.append(path)
    return result


def _publish_finalization(destination, record, calls, completion):
    """Stage a complete bundle, then publish without overwriting any user file."""
    from .review import write_review_packet
    from .contract import export_records
    staging_root = destination / ".finalize"
    require(not staging_root.is_symlink(), "Finalization directory must not be a symlink")
    durable_mkdir(staging_root)
    ready_path = staging_root / "ready.json"
    if ready_path.exists():
        ready = _read_json(ready_path)
        require(ready.get("archive_sha256") == canonical_hash(record),
                "Rebuilt archive differs from retained finalization")
        stage_name = ready.get("stage", "")
        require(isinstance(stage_name, str) and re.fullmatch(r"attempt-\d{4}", stage_name),
                "Invalid finalization stage")
        stage_dir = staging_root / stage_name
        require(stage_dir.is_dir() and not stage_dir.is_symlink(), "Missing finalization stage")
        durable_mkdir(stage_dir)
    else:
        attempts = [p for p in staging_root.iterdir() if re.fullmatch(r"attempt-\d{4}", p.name)]
        next_index = max([int(p.name.removeprefix("attempt-")) for p in attempts] or [0]) + 1
        stage_dir = staging_root / f"attempt-{next_index:04d}"
        durable_mkdir(stage_dir)
        atomic_json(stage_dir / "archive.json", record)
        write_review_packet(record, stage_dir / "review.html", dataset_root=completion["dataset_root"],
                            artifact_root=destination)
        if record["training_view"]["messages"]:
            export_records([record], stage_dir / "training.preview.jsonl",
                           dataset_root=completion["dataset_root"], artifact_root=destination, preview=True)
        _write_audit(stage_dir, record, calls, completion)
        atomic_json(stage_dir / "completion.json", completion)
        # The existing render/export helpers write into this private staging
        # directory. Flush those files before committing the ready manifest.
        for path in stage_dir.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        descriptor = os.open(stage_dir, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        files = {p.name: sha256_file(p) for p in sorted(stage_dir.iterdir())}
        ready = {"archive_sha256": canonical_hash(record), "stage": stage_dir.name, "files": files}
        atomic_json(ready_path, ready)
    require(isinstance(ready.get("files"), dict) and {"archive.json", "completion.json", "review.html",
            "review.review-template.json", "audit.html"} <= ready["files"].keys(),
            "Incomplete finalization manifest")
    allowed = {"archive.json", "completion.json", "review.html", "review.review-template.json",
               "audit.html", "training.preview.jsonl", "training.preview.receipt.json"}
    require(ready["files"].keys() <= allowed, "Unknown finalization artifact")
    # completion.json is the commit marker and is always published last.
    names = sorted(set(ready["files"]) - {"completion.json"}) + ["completion.json"]
    for name in names:
        source = stage_dir / name
        require(source.is_file() and not source.is_symlink()
                and sha256_file(source) == ready["files"][name], f"Finalization artifact changed: {name}")
        target = destination / name
        if target.exists() or target.is_symlink():
            require(target.is_file() and not target.is_symlink()
                    and sha256_file(target) == ready["files"][name],
                    f"Existing output differs; preserve it and resolve manually: {target}")
        else:
            atomic_bytes(target, source.read_bytes())
    return _read_json(destination / "completion.json")


def enhance_sospine(dataset_root, output_dir, config, *, client=None, progress=None,
                    resume=False, pause_requested=None):
    """Run or resume one pinned window; never repeat a valid retained model call.

    A pause is observed before each new inference. Request/response files are the
    durable replay log; checkpoint.json is a status view, never a substitute for
    verifying that log. Failed attempts and partial finalization are retained.
    """
    from .enhancement_prompts import PROMPT_VERSION, stage_question, validate_output
    from .teacher import ADAPTER, ENVELOPE_BUILDER_VERSION, build_request, parse_response

    plan = plan_enhancement(dataset_root, config)
    root, destination = Path(dataset_root).resolve(), Path(output_dir)
    require(not destination.is_symlink(), "Output directory must not be a symlink")
    require(not destination.resolve().is_relative_to(root), "Output must be outside the source dataset")
    if destination.exists():
        require(resume and destination.is_dir(), "Output directory already exists; use resume or a new run directory")
        require((destination / "session.json").is_file(),
                "Cannot resume a legacy or unrecognized directory without session.json")
    else:
        durable_mkdir(destination)
    destination = destination.resolve()
    client = client or LlamaCppClient()

    with directory_lock(destination):
        durable_mkdir(destination)
        require(resume or not (destination / "session.json").exists(),
                "Output directory was initialized by another writer; use resume")
        pause_path = destination / "PAUSE"
        seen = set(plan["initial_frame_indices"])
        record = _source_record(root, config, seen)
        snapshot = _source_snapshot(root, plan, record)
        session_path = destination / "session.json"
        expected_session = {"checkpoint_version": "1", "dataset_root": str(root),
                            "plan_sha256": canonical_hash(plan), "source_files": snapshot,
                            "prompt_identity": enhancement_protocol_fingerprint()}
        if session_path.exists():
            session = _read_json(session_path)
            require(session.get("prompt_identity", {}).get("adapter") == ADAPTER,
                    "Resume rejected: legacy Ollama jobs are read-only; start a new llama.cpp run")
            require(all(session.get(k) == v for k, v in expected_session.items()),
                    "Resume rejected: source bytes, config, dataset root, or prompt identity changed")
        else:
            session = expected_session | {"created_at": datetime.now(timezone.utc).isoformat()}
            atomic_json(session_path, session)
        _new_or_equal_json(destination / "plan.json", plan)
        if resume and pause_path.exists():
            require(pause_path.is_file() and not pause_path.is_symlink(), "Invalid PAUSE control file")
            pause_path.unlink()
        calls, outputs, selections = [], {}, []
        started = time.monotonic()
        checkpoint_path = destination / "checkpoint.json"
        previous = _read_json(checkpoint_path) if checkpoint_path.exists() else {}
        previous_elapsed = previous.get("elapsed_seconds", 0)
        require(isinstance(previous_elapsed, (int, float)) and previous_elapsed >= 0,
                "Invalid elapsed time in checkpoint")

        def report(text):
            if progress:
                progress(text)

        def checkpoint(status, **details):
            observed_ids = {fid for call in calls for fid in call["input_frame_ids"]}
            observed = sorted(f["frame_index"] for f in record["frame_selection"]["frames"]
                              if f["frame_id"] in observed_ids)
            state = {"status": status, "dataset_root": str(root), "output_dir": str(destination),
                     "case_id": config.case_id, "start_index": config.start_index,
                     "cutoff_index": config.cutoff_index, "successful_model_calls": len(calls),
                     "successful_run_ids": [call["run_id"] for call in calls],
                     "observed_frame_indices": observed, "selected_frame_indices": sorted(seen),
                     "human_review_status": "pending",
                     "training_eligible": False, "elapsed_seconds": previous_elapsed + time.monotonic() - started,
                     "updated_at": datetime.now(timezone.utc).isoformat(), **details}
            atomic_json(checkpoint_path, state, overwrite=True)
            return state

        def record_failure(exc):
            failure = checkpoint("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                 error_type=type(exc).__name__, error=str(exc) or "Interrupted by user")
            directory = destination / "failures"
            durable_mkdir(directory)
            index = max([int(p.stem) for p in directory.glob("*.json") if p.stem.isdigit()] or [0]) + 1
            atomic_json(directory / f"{index:04d}.json", failure)
            # Compatibility summary: retain the first failure rather than overwrite history.
            if not (destination / "failure.json").exists():
                atomic_json(destination / "failure.json", failure)

        try:
            checkpoint("initializing")
            metadata = {}
            model_dir = destination / "models"
            durable_mkdir(model_dir)
            for role, name in (("qwen", config.qwen_model), ("medgemma", config.medgemma_model)):
                fresh = client.model_info(name)
                fresh.setdefault("model_name", fresh["name"])
                fresh.setdefault("model_digest", fresh["digest"])
                path = model_dir / f"{role}.json"
                if path.exists():
                    info = _read_json(path)
                    require(_model_identity(info) == _model_identity(fresh),
                            "Resume rejected: model digest, quantization, name, or runtime version changed")
                else:
                    info = fresh
                    atomic_json(path, info)
                aid = f"model-{role}"
                record["assets"].append(_artifact(path, destination, aid, "reference_document",
                                                  origin="deterministically_derived"))
                metadata[name] = (info, aid)
            require(metadata[config.qwen_model][0]["digest"] != metadata[config.medgemma_model][0]["digest"],
                    "Discovery and medical review models resolve to the same model weights")
            model_manifest = {name: {"identity": _model_identity(info), "asset_id": aid,
                                    "receipt_sha256": next(a["sha256"] for a in record["assets"] if a["asset_id"] == aid)}
                              for name, (info, aid) in metadata.items()}
            _new_or_equal_json(model_dir / "manifest.json", model_manifest)

            def invoke(stage, model, indices, parents):
                run_id = f"run-{len(calls) + 1:03d}-{stage}"
                info, metadata_id = metadata[model]
                fids = [f"f{i:06d}" for i in sorted(indices)]
                aids = [a["annotation_id"] for a in record["original_annotations"]
                        if a["original_kind"] in {"keypoint", "bbox"} and set(a["frame_ids"]) <= set(fids)]
                parameters = {"adapter": ADAPTER, "envelope_builder_version": ENVELOPE_BUILDER_VERSION, "stage": stage,
                              "parent_run_ids": parents, "cutoff_frame_index": config.cutoff_index,
                              "start_frame_index": config.start_index, "stage_question": stage_question(stage),
                              "options": {"temperature": 0, "seed": config.seed, "num_ctx": config.num_ctx,
                                          "num_predict": config.num_predict},
                              "model_metadata_asset_id": metadata_id}
                run = {"run_id": run_id, "model_name": model, "model_digest": info["digest"],
                       "quantization": info["quantization"], "runtime": "llama.cpp", "runtime_version": info["runtime_version"],
                       "request_asset_id": f"{run_id}-request", "response_asset_id": f"{run_id}-response",
                       "prompt_version": PROMPT_VERSION, "input_mode": "causal_prefix", "input_frame_ids": fids,
                       "input_annotation_ids": aids, "input_reference_asset_ids": [], "previous_message_indices": [],
                       "outcome_ids_seen": [], "maximum_frame_index_seen": max(seen), "generation_parameters": parameters}
                for asset in record["assets"]:
                    if asset["origin"] == "original_dataset":
                        require(snapshot.get(asset["location"]) == asset["sha256"]
                                and sha256_file(resolve_asset(asset["location"], root)) == asset["sha256"],
                                "Pinned source bytes changed during enhancement")
                request = build_request(record, run, dataset_root=root, artifact_root=destination)
                ancestor_ids, ancestor_annotations, parent_ids = set(fids), set(aids), set(parents)
                for prior in reversed(record["generation_runs"]):
                    if prior["run_id"] in parent_ids:
                        ancestor_ids.update(prior["input_frame_ids"])
                        ancestor_annotations.update(prior["input_annotation_ids"])
                        parent_ids.update(prior["generation_parameters"]["parent_run_ids"])

                def parsed(raw):
                    return validate_output(parse_response(raw, run), ancestor_ids, ancestor_annotations,
                                           config.start_index, config.cutoff_index)

                call_dir = destination / "calls" / run_id
                attempts = _attempt_directories(call_dir)
                successful = []
                for attempt in attempts:
                    request_path, run_path = attempt / "request.json", attempt / "run.json"
                    response_path, success_path = attempt / "response.json", attempt / "success.json"
                    if request_path.exists():
                        require(not request_path.is_symlink() and request_path.read_bytes() == request,
                                f"Retained request differs from rebuilt evidence: {run_id}")
                    if run_path.exists():
                        require(_read_json(run_path) == run, f"Retained generation receipt changed: {run_id}")
                    committed = _read_json(success_path) if success_path.exists() else None
                    if committed:
                        for name, digest in committed["files"].items():
                            require(name in {"request.json", "run.json", "response.json", "parsed.json"},
                                    "Unexpected successful-call artifact")
                            require(sha256_file(attempt / name) == digest, "Successful call artifact changed")
                    if not response_path.exists():
                        require(not committed, "Completed call response is missing")
                        continue
                    require(request_path.exists() and run_path.exists() and not response_path.is_symlink(),
                            "Response has an incomplete retained request receipt")
                    raw = response_path.read_bytes()
                    try:
                        output = parsed(raw)
                    except (ContractError, ValueError) as exc:
                        require(not committed, f"Completed call is no longer valid: {exc}")
                        if not (attempt / "failure.json").exists():
                            atomic_json(attempt / "failure.json", {"error_type": type(exc).__name__,
                                "error": str(exc), "recovered_after_interruption": True})
                        continue
                    successful.append((attempt, raw, output))
                require(len(successful) <= 1, "Multiple successful attempts for one logical call")
                if successful:
                    attempt, raw, output = successful[0]
                    report(f"{run_id}: replayed verified response; no inference")
                else:
                    if pause_path.exists() or (pause_requested is not None and pause_requested()):
                        raise _PauseRequested()
                    fresh = client.model_info(model)
                    fresh.setdefault("model_name", fresh["name"])
                    fresh.setdefault("model_digest", fresh["digest"])
                    require(_model_identity(fresh) == _model_identity(info),
                            "Pinned model identity changed before inference")
                    if pause_path.exists() or (pause_requested is not None and pause_requested()):
                        raise _PauseRequested()
                    if not attempts:
                        attempt = call_dir
                        durable_mkdir(attempt)
                    else:
                        attempt_root = call_dir / "attempts"
                        durable_mkdir(attempt_root)
                        number = max([int(p.name) for p in attempts if p != call_dir] or [1]) + 1
                        attempt = attempt_root / f"{number:04d}"
                        durable_mkdir(attempt)
                    request_path, response_path = attempt / "request.json", attempt / "response.json"
                    atomic_bytes(request_path, request)
                    atomic_json(attempt / "run.json", run)
                    atomic_json(attempt / "attempt.json", {"status": "started",
                        "started_at": datetime.now(timezone.utc).isoformat()})
                    checkpoint("running", current_run_id=run_id, current_stage=stage)
                    report(f"{run_id}: {model}, {len(indices)} original frames")
                    before = time.monotonic()
                    try:
                        raw = client.chat_raw(json.loads(request))
                        atomic_bytes(response_path, raw)
                        output = parsed(raw)
                    except (Exception, KeyboardInterrupt) as exc:
                        raw_failure = getattr(client, "last_response_bytes", None)
                        if raw_failure and not response_path.exists():
                            atomic_bytes(response_path, raw_failure)
                        atomic_json(attempt / "failure.json", {"error_type": type(exc).__name__,
                            "error": str(exc) or "Interrupted by user",
                            "elapsed_seconds": time.monotonic() - before})
                        raise
                    elapsed = round(time.monotonic() - before, 3)
                request_path, response_path = attempt / "request.json", attempt / "response.json"
                parsed_path = attempt / "parsed.json"
                if parsed_path.exists():
                    call = _read_json(parsed_path)
                    require(call.get("output") == output and call.get("run_id") == run_id
                            and call.get("stage") == stage and call.get("model") == model
                            and call.get("input_frame_ids") == fids and call.get("parent_run_ids") == parents,
                            "Retained parsed call differs from raw response or lineage")
                else:
                    envelope = json.loads(raw)
                    call = {"run_id": run_id, "stage": stage, "model": model, "input_frame_ids": fids,
                            "parent_run_ids": parents, "elapsed_seconds": None if successful else elapsed,
                            "output": output, "prompt_tokens": envelope.get("usage", {}).get("prompt_tokens"),
                            "completion_tokens": envelope.get("usage", {}).get("completion_tokens")}
                    atomic_json(parsed_path, call)
                success = {"files": {name: sha256_file(attempt / name)
                                     for name in ("request.json", "run.json", "response.json", "parsed.json")},
                           "output_sha256": canonical_hash(output)}
                _new_or_equal_json(attempt / "success.json", success)
                record["assets"].extend([
                    _artifact(request_path, destination, run["request_asset_id"], "teacher_request",
                              origin="deterministically_derived"),
                    _artifact(response_path, destination, run["response_asset_id"], "teacher_response")])
                record["generation_runs"].append(run)
                outputs[run_id] = output
                calls.append(call)
                checkpoint("running", last_completed_run_id=run_id)
                report(f"{run_id}: {len(output['events'])} events, {len(output['questions'])} questions, "
                       f"{len(output['searches'])} search requests")
                return run_id, output

            proposal_id, _ = invoke("propose", config.qwen_model, seen, [])
            observation_id, _ = invoke("independent_observe", config.medgemma_model, seen, [])
            final_id, final = invoke("review", config.medgemma_model, seen, [proposal_id, observation_id])
            stop_reason = "no_search_requested"
            for round_index in range(config.max_rounds):
                if not final["searches"]:
                    break
                budget = min(config.search_frames, config.max_frames - len(seen))
                extra = select_search_indices(final["searches"], plan["available_frame_indices"], seen, budget)
                selections.append({"round": round_index + 1, "requested_by": final_id, "searches": final["searches"],
                                   "new_frame_indices": extra,
                                   "remaining_frame_budget": config.max_frames - len(seen) - len(extra)})
                if not extra:
                    stop_reason = "frame_budget_exhausted" if budget == 0 else "no_additional_released_evidence"
                    break
                seen.update(extra)
                _merge_source(record, _source_record(root, config, seen))
                search_id, _ = invoke("search", config.qwen_model, extra, [final_id])
                final_id, final = invoke("final_revise", config.medgemma_model, seen, [final_id, search_id])
            if final["searches"] and stop_reason == "no_search_requested":
                stop_reason = "round_budget_exhausted"
            expected_dirs = {call["run_id"] for call in calls}
            require({p.name for p in (destination / "calls").iterdir()} == expected_dirs,
                    "Retained calls do not match the replayed deterministic plan")
            require(_source_snapshot(root, plan, record) == snapshot, "Pinned source bytes changed before finalization")
            receipt_hashes = {a["asset_id"]: a["sha256"] for a in record["assets"]
                              if a["role"] in {"teacher_request", "teacher_response"}}
            identity = {"config": asdict(config), "receipts": receipt_hashes,
                        "models": [r["model_digest"] for r in record["generation_runs"]]}
            identifier = f"sospine.{config.case_id}.enhanced.{canonical_hash(identity)[:20]}"
            record["record_id"] = identifier
            record["revision"].update(revision_id=identifier + ".r1", created_at=session["created_at"])
            record["frame_selection"].update(method="hybrid", selector_version="qwen-medgemma-inspection-v1",
                parameters={"initial_sampling": "uniform_in_declared_window", "start_frame_index": config.start_index,
                            "cutoff_frame_index": config.cutoff_index, "max_frames": config.max_frames,
                            "selection_rounds": selections, "native_video_processor": False,
                            "grounder": "qwen_targeted_reinspection", "stop_reason": stop_reason})
            initial = set(plan["initial_frame_indices"])
            for frame in record["frame_selection"]["frames"]:
                frame["selection_reason"] = ("Uniform initial sample within the declared cutoff."
                    if frame["frame_index"] in initial else
                    "Additional released evidence requested by medical review; exact search retained in selection_rounds.")
            _make_conversation(record, outputs, final_id, config.cutoff_index)
            validate_record(record, dataset_root=root, artifact_root=destination)
            completion = {"status": "completed", "dataset_root": str(root), "output_dir": str(destination),
                          "human_review_status": "pending", "record_id": identifier,
                          "final_run_id": final_id, "stop_reason": stop_reason, "model_calls": len(calls),
                          "successful_model_calls": len(calls), "final_events": len(final["events"]),
                          "final_questions": len(final["questions"]),
                          "conversation_turns": len(record["training_view"]["turn_links"]),
                          "observed_frame_indices": sorted(seen), "unresolved_searches": final["searches"],
                          "elapsed_seconds": previous_elapsed + time.monotonic() - started,
                          "transport": "llama_cpp_ordered_images", "native_video_processor": False,
                          "grounder": "qwen_targeted_reinspection", "training_eligible": False}
            checkpoint("finalizing")
            completion = _publish_finalization(destination, record, calls, completion)
            checkpoint("completed", record_id=identifier, completion_path=str(destination / "completion.json"))
            report(f"Completed: {destination / 'archive.json'}; human review pending")
            return completion
        except _PauseRequested:
            result = checkpoint("paused")
            report(f"Paused after {len(calls)} successful model calls: {destination}")
            return result
        except (Exception, KeyboardInterrupt) as exc:
            record_failure(exc)
            raise
