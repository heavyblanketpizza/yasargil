"""Read rejected temporal citations as drafts without accepting or rewriting them.

Only overlong evidence spans and ends beyond the supplied video are eligible.
All other schema, source, request, native-input, and response checks still apply.
Runtime paths are rebound lexically to a frozen copy; original runtime files are
never consulted when a copy is supplied.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import fcntl
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import re
import struct

from jsonschema import Draft202012Validator

from .annotation_contract import _number, annotation_schema
from .annotation_integrity import _FROZEN, _hashes, _transport_binding
from .contract import ContractError, require
from .frame_annotation import (
    AnnotationConfig, LEGACY_PROTOCOL_VERSION, _messages, _read, _validate_selection,
    normalized_annotation_config,
)
from .llama_video import _strict_json
from .smart_selection import verify_assets
from .video_source import validate_native_timeline


SCHEMA_VERSION = "qwen-rejected-annotation-intake-v1"
VALIDATION_STATUS = "rejected_temporal_citations"
REVIEWER_WARNINGS = [
    "These original Qwen drafts failed temporal-citation validation and are supplied for review by explicit request.",
    "Original text and interval bounds are unchanged. Overlong or out-of-video citations are not validated evidence.",
    "Qwen processed every supplied video frame, but its non-specialist observations remain unverified.",
    "Retain supported content; inspect cited times critically and record uncertainty or missing-evidence requests.",
    "This intake does not mark the Qwen annotation completed, clinically validated, or eligible for training.",
]


def _paths(root):
    required = _FROZEN | {
        ".annotation.lock", "run.json", "summary.json", "last-error.json",
        "native-timeline-verification.json", "round-00/annotation-context.json",
        "round-00/request.json", "round-00/response.json", "round-00/verification.json",
        "round-00/server-segment.log",
    }
    require(not (root / "annotations.json").exists() and not (root / "integrity.json").exists(),
            "Rejected-draft intake cannot replace published or sealed annotations")
    for name in required:
        require((root / name).is_file(), f"Missing rejected Qwen evidence: {name}")
    paths = set(required)
    for name in ("round-00/result.json", "round-00/output.json"):
        if (root / name).exists():
            paths.add(name)
    runtime = root / "runtime"
    require(runtime.is_dir() and not runtime.is_symlink(), "Missing local Qwen runtime evidence")
    for path in runtime.rglob("*"):
        if path.is_file():
            paths.add(path.relative_to(root).as_posix())
    return sorted(paths)


def _absolute(value, label):
    require(isinstance(value, str) and Path(value).is_absolute()
            and ".." not in Path(value).parts and str(Path(value)) == value,
            f"Invalid original {label}")
    return Path(value)


def _runtime(root, verification, source, config):
    """Validate raw receipts locally, retaining their original path identities."""
    original_runtime = _absolute(verification.get("runtime_directory"), "runtime directory")
    require(original_runtime.parent.name == "runtime", "Unexpected original annotation runtime layout")
    original_root = original_runtime.parent.parent
    runtime = root / original_runtime.relative_to(original_root)
    metadata = _read(runtime / "runtime.json")
    native = _read(runtime / "native-decode/verification.json")
    count = source["expected_video_frames"]
    require(metadata.get("context_size") == config.context_size
            and metadata.get("image_max_tokens") == config.image_max_tokens
            and metadata.get("expected_video_frames") == count
            and metadata.get("video_sha256") == source["video_sha256"]
            and metadata.get("video_fps_setting") == 0
            and metadata.get("native_decode_verification") == native,
            "Qwen runtime differs from its frozen settings or complete source")
    command = metadata.get("command")
    require(isinstance(command, list) and "--no-context-shift" in command,
            "Qwen runtime does not disable context shifting")
    for flag, expected in {
        "-c": str(config.context_size), "--image-max-tokens": str(config.image_max_tokens),
        "--video-fps": "0", "--timeout": str(math.ceil(config.request_timeout_seconds)),
        "--media-path": str(original_root / "media"),
    }.items():
        require(command.count(flag) == 1 and command.index(flag) + 1 < len(command)
                and command[command.index(flag) + 1] == expected, f"Qwen runtime setting changed: {flag}")
    require(native.get("accepted") is True and native.get("ordered_rgb_frames_identical") is True
            and native.get("video_sha256") == source["video_sha256"]
            and all(native.get(key) == count for key in
                    ("expected_video_frames", "source_decoded_frames", "native_decoded_frames"))
            and native.get("video_path") == str(original_root / "media" / verification["video_relative_path"]),
            "Native decode does not verify the complete source video")
    probe = _read(runtime / "native-decode/ffprobe.json")
    streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"]
    require(len(streams) == 1, "Native preflight does not identify exactly one video stream")
    rate = Fraction(streams[0].get("r_frame_rate", "0/1"))
    require(rate > 0 and native.get("source_r_frame_rate") == str(rate)
            and native.get("video_stream_index") == streams[0].get("index"), "Native preflight rate changed")
    fps = struct.unpack("f", struct.pack("f", float(rate)))[0]
    require(native.get("native_filter") == f"fps={fps:.6f}", "Native preflight filter changed")
    records = []
    for name in ("source", "native"):
        rows = []
        for line in (runtime / f"native-decode/{name}.framehash").read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = [field.strip() for field in line.split(",")]
            require(len(fields) == 6 and re.fullmatch(r"[0-9a-f]{64}", fields[-1]) is not None,
                    "Malformed native RGB frame hashes")
            size = int(fields[-2])
            require(size > 0, "Native RGB frame has an invalid byte count")
            rows.append((size, fields[-1]))
        records.append(rows)
        require((runtime / f"native-decode/{name}.log").is_file(), "Missing native preflight log")
    require(len(records[0]) == count and records[0] == records[1],
            "Ordered native RGB frame hashes do not match the complete source")
    log = (runtime / "server.log").read_bytes()
    segment = (root / "round-00/server-segment.log").read_bytes()
    start, end = verification.get("log_start_byte"), verification.get("log_end_byte")
    require(type(start) is int and type(end) is int and 0 <= start < end <= len(log)
            and log[start:end] == segment, "Qwen inference segment differs from the runtime log")
    text = segment.decode(errors="replace")
    decoded = [int(value) for value in re.findall(r"read_next_frame: frame (\d+) read OK", text)]
    require(decoded == list(range(count)), "Qwen log does not show every video frame exactly once")
    require(not any(int(value) for value in re.findall(r"\btruncated\s*=\s*(\d+)", text))
            and re.search(r"\bcontext shift:|\btruncating (?:the )?prompt", text, re.IGNORECASE) is None,
            "Qwen log shows context truncation or shifting")
    # Verify recorded absolute locations before translating only these two
    # lookup fields for the existing transport checker. Raw artifacts stay intact.
    transport_root = original_runtime / "transport-receipts"
    require(metadata.get("ffmpeg_transport_receipts_directory") == str(transport_root),
            "Recorded byte-transport directory changed")
    receipt = _absolute(verification.get("byte_transport_receipt_path"), "byte-transport receipt")
    require(receipt.is_relative_to(transport_root), "Byte-transport receipt escapes its recorded runtime")
    mapped_metadata = {**metadata, "ffmpeg_transport_receipts_directory": str(runtime / "transport-receipts")}
    mapped_verification = {**verification,
        "byte_transport_receipt_path": str(runtime / receipt.relative_to(original_runtime))}
    _transport_binding(runtime, mapped_metadata, native, mapped_verification, source)
    return original_root


def _drafts(raw, selected, source, config, schema):
    """Permit only the two named temporal errors; preserve every original bound."""
    errors = sorted(Draft202012Validator(schema).iter_errors(raw), key=lambda error: str(list(error.path)))
    for error in errors:
        path = list(error.path)
        require(error.validator == "maximum" and len(path) == 7
                and path[0] == "annotations" and path[2] == "contextual_claims"
                and path[4] == "evidence_intervals" and path[6] == "end_ms",
                f"Qwen draft has an ineligible schema error: {error.message}")
    duration = source["duration_ms"]
    issues, drafts = [], []
    for frame in selected:
        frame_id = frame["frame_id"]
        row = raw["annotations"][frame_id]
        require(row["visible_observation"].strip() and all(value.strip() for value in row["uncertainties"]),
                "Qwen draft contains blank observation or uncertainty text")
        require(row["visibility"] not in {"poor", "uninterpretable"} or row["uncertainties"],
                "Poor Qwen visibility requires uncertainty")
        claims = []
        for claim_index, claim in enumerate(row["contextual_claims"]):
            require(claim["claim"].strip(), "Qwen contextual claim is blank")
            intervals = []
            for interval_index, interval in enumerate(claim["evidence_intervals"]):
                start = _number(interval["start_ms"], "Qwen evidence start", minimum=0)
                end = _number(interval["end_ms"], "Qwen evidence end", minimum=0)
                require(start < end and start <= duration, "Qwen interval must increase from the video timeline")
                supporting = [copy.deepcopy(item) for item in source["frames"] if start <= item["timestamp_ms"] <= end]
                require(supporting, "Qwen evidence interval contains no actual source observation")
                locator = {"frame_id": frame_id, "claim_index": claim_index, "interval_index": interval_index,
                           "start_ms": start, "end_ms": end, "span_ms": end - start,
                           "maximum_evidence_span_ms": config.max_evidence_span_ms, "video_duration_ms": duration}
                if end - start > config.max_evidence_span_ms:
                    issues.append({"code": "evidence_span_exceeds_limit", **locator})
                if end > duration:
                    issues.append({"code": "evidence_end_exceeds_video", **locator, "overflow_ms": end - duration})
                intervals.append({**copy.deepcopy(interval), "supporting_frames": supporting})
            claims.append({**copy.deepcopy(claim), "evidence_intervals": intervals})
        drafts.append({**copy.deepcopy(frame), **copy.deepcopy(row), "contextual_claims": claims,
                       "review_required": True, "training_eligible": False})
    require(issues, "Qwen annotation has no eligible temporal-citation rejection")
    return {"schema_version": "contextual-frame-annotations-v1", "context_check": raw["context_check"],
            "annotations": drafts, "validation_status": VALIDATION_STATUS, "validation_issues": issues,
            "clinical_validation": "not_performed", "training_eligible": False,
            "temporal_exposure": "retrospective_full_video",
            "evidence_validation": "rejected_temporal_citations_not_semantic_validation"}, errors


def _load(root):
    names = _paths(root)
    before = _hashes(root, names)
    plan, session = _read(root / "run.json"), _read(root / "session.json")
    require(plan.get("schema_version") == LEGACY_PROTOCOL_VERSION,
            "Rejected timestamp intake supports only the legacy Qwen annotation protocol")
    config = AnnotationConfig(**plan["config"])
    config.validate()
    require(normalized_annotation_config(session.get("config")) == asdict(config),
            "Qwen settings differ from the frozen session")
    require(_FROZEN <= set(plan["input_sha256"]), "Qwen run does not pin its complete frozen inputs")
    for name, digest in plan["input_sha256"].items():
        require(name in before and before[name]["sha256"] == digest, f"Frozen Qwen input changed: {name}")
    _, source, selected, _ = _validate_selection(root, plan["selection_run"], snapshot=True)
    ids = [frame["frame_id"] for frame in selected]
    require(_read(root / "selected-frames.json") == selected and plan["frozen_frame_ids"] == ids,
            "Qwen selected frames differ from the frozen final selection")
    require(_read(root / "native-timeline-verification.json") == validate_native_timeline(source),
            "Qwen native timeline verification changed")
    require(_read(root / "round-00/annotation-context.json") == session, "Qwen annotation context changed")
    request = _read(root / "round-00/request.json")
    schema = annotation_schema(ids, source["duration_ms"], max_evidence_span_ms=config.max_evidence_span_ms)
    video_name = "video" + (Path(source["video_path"]).suffix.lower() or ".mp4")
    names_by_id = {frame["frame_id"]: f"frame-{index:08d}{Path(frame['image_path']).suffix.lower()}"
                   for index, frame in enumerate(source["frames"])}
    messages = _messages(source, selected, names_by_id, video_name, config,
                         protocol_version=LEGACY_PROTOCOL_VERSION)
    require(request == {
        "model": "qwen-video", "messages": messages, "max_tokens": config.max_tokens,
        "temperature": 0.1, "seed": 42, "stream": False, "cache_prompt": True, "id_slot": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "frame_selection", "strict": True, "schema": schema}}},
        "Saved Qwen request differs from its complete original annotation configuration")
    require(plan.get("protocol_sha256") == hashlib.sha256((LEGACY_PROTOCOL_VERSION + messages[0]["content"]).encode()).hexdigest(),
            "Qwen annotation prompt policy changed")
    response = _read(root / "round-00/response.json")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict)
            and choices[0].get("finish_reason") == "stop" and not response.get("error"), "Qwen response is incomplete")
    message = choices[0].get("message")
    require(isinstance(message, dict) and message.get("role") == "assistant" and not message.get("tool_calls")
            and isinstance(message.get("content"), str) and message["content"].strip(),
            "Qwen response is not a complete direct assistant draft")
    raw = _strict_json(message["content"])
    usage = response.get("usage")
    require(isinstance(usage, dict) and type(usage.get("prompt_tokens")) is int
            and 0 < usage["prompt_tokens"] <= config.context_size - config.max_tokens,
            "Qwen response exceeds the frozen context budget")
    verification = _read(root / "round-00/verification.json")
    require(type(verification.get("accepted")) is bool
            and verification.get("full_source_video_verified") is True
            and verification.get("context_truncation_observed") is False
            and verification.get("request_sha256") == before["round-00/request.json"]["sha256"]
            and verification.get("video_sha256") == source["video_sha256"]
            and verification.get("video_relative_path") == video_name
            and verification.get("video_fps_setting") == 0
            and verification.get("expected_video_frames") == source["expected_video_frames"]
            and verification.get("decoded_frames") == source["expected_video_frames"]
            and verification.get("decoded_frame_ids") == list(range(source["expected_video_frames"]))
            and verification.get("message_count") == 2 and verification.get("prior_history_preserved") is False
            and verification.get("finish_reason") == "stop"
            and verification.get("usage") == usage and verification.get("timings") == response.get("timings")
            and verification.get("system_fingerprint") == response.get("system_fingerprint"),
            "Qwen response lacks unchanged complete-video verification")
    document, errors = _drafts(raw, selected, source, config, schema)
    error = _read(root / "last-error.json")
    if verification["accepted"]:
        require(not errors and not verification.get("error"), "Accepted transport result has a schema failure")
        result = _read(root / "round-00/result.json")
        require(result == {"output": raw, "response": response, "verification": verification}
                and _read(root / "round-00/output.json") == raw, "Qwen result differs from the original raw response")
        require(error.get("type") == "ContractError"
                and error.get("message") == "Evidence interval exceeds maximum evidence span",
                "Qwen annotation failed for an ineligible reason")
    else:
        require(errors and not (root / "round-00/result.json").exists()
                and not (root / "round-00/output.json").exists(), "Rejected Qwen transport artifacts are inconsistent")
        expected = "Qwen's answer does not match the required schema: " + errors[0].message
        require(verification.get("error") == expected and error.get("type") == "VideoRuntimeError"
                and error.get("message") == expected, "Qwen transport failed for an ineligible reason")
    summary = _read(root / "summary.json")
    require(summary.get("status") == "failed" and summary.get("error") == error
            and summary.get("config") == plan["config"] and summary.get("session_id") == session["session_id"]
            and summary.get("frame_ids") == ids and summary.get("selected_frame_count") == len(ids),
            "Rejected Qwen annotation summary changed")
    original_root = _runtime(root, verification, source, config)
    verify_assets(source)
    require(names == _paths(root) and before == _hashes(root, names), "Qwen artifacts changed during draft intake")
    audit = {"schema_version": SCHEMA_VERSION, "status": VALIDATION_STATUS, "validation_status": VALIDATION_STATUS,
             "issues": copy.deepcopy(document["validation_issues"]), "annotation_count": len(ids), "frame_ids": ids,
             "source_frame_count": source["expected_video_frames"], "source_video_sha256": source["video_sha256"],
             "original_annotation_root": str(original_root),
             "raw_response_sha256": before["round-00/response.json"]["sha256"],
             "artifact_sha256": {name: details["sha256"] for name, details in before.items()},
             "reviewer_warnings": list(REVIEWER_WARNINGS),
             "original_transport_accepted": verification["accepted"],
             "full_source_video_verified": True, "clinical_validation": "not_performed", "training_eligible": False}
    return plan, source, document, audit


def load_rejected_annotation(root):
    """Read an original failed pass or its exact frozen artifact copy, without writes."""
    root = Path(root).expanduser().resolve()
    try:
        with (root / ".annotation.lock").open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ContractError("Qwen annotation is still active") from exc
            return _load(root)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"Invalid rejected Qwen annotation evidence: {exc}") from exc
