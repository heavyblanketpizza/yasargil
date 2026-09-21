"""Annotate a frozen final selection in a fresh complete-video conversation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
import uuid

from jsonschema import Draft202012Validator, ValidationError

from .annotation_contract import FRAME_EVIDENCE, TIMESTAMP_EVIDENCE, annotation_schema, build_annotations
from .contract import ContractError, require, sha256_file
from .llama_video import LocalVideoRuntime, RuntimeConfig, VideoRuntimeError, _strict_json
from .smart_selection import (
    PROJECT_ROOT, TIMELINE_SYSTEM, SelectionConfig, _image_blocks, _recover_verified_result,
    _selection_record, _write, selection_config_from_saved, stage_media, validate_review, verify_assets,
)
from .video_source import media_timeline, validate_native_timeline


LEGACY_PROTOCOL_VERSION = "full-video-frame-annotation-v1"
PROTOCOL_VERSION = "full-video-frame-annotation-v2"
LEGACY_ANNOTATION_SYSTEM = """You annotate a FROZEN set of selected still frames using the COMPLETE native
video for context. Watch the whole video and examine every supplied still before answering.
This is a fresh annotation task: the frame list is final. Do not select, rank, drop, replace,
or request additional frames. Return one annotation for EVERY specified frame ID.

Keep two kinds of information explicitly separate:
1. visible_observation: describe only what can be seen in THIS still. Mention concrete visual
   details, relevant instrument/tissue positions, and visibility limits. A still alone does not
   establish motion, the ordinal number of a stitch, successful repair, or a clinical outcome.
2. contextual_claims: describe an interpretation supported by the surrounding VIDEO. Each
   claim must cite narrow evidence_intervals and must be useful for understanding this frame.
   Cite the actual location on the supplied video timeline. Do not compress the procedure
   into the beginning of the video, invent evenly spaced stages, or treat timestamps as proof
   of procedural stages. Follow the stated maximum evidence interval length.

Context may explain visible evidence, but an event elsewhere in the video must never become
something supposedly visible in this still. Do not infer anatomy, maneuvers, stitch counts,
completion, safety, or treatment success from the documented procedure name. If uncertain,
use cautious language or omit the contextual claim. An empty contextual_claims list is valid.
Report uncertainties explicitly; poor or uninterpretable views must include a limitation.
Do not fill gaps with plausible surgical storytelling. If the video is not surgical, annotate
the actual visible activity without imposing a surgical interpretation.

Source context is documented background, not independent visual evidence. Return
context_check consistent, uncertain, or conflict when background is provided, or not_supplied
otherwise. All times are milliseconds on the documented source timeline. Copy frame IDs
exactly. Source filenames and authoritative timestamps are attached by software, not authored
by you. These annotations are drafts awaiting review. Return only the required JSON.""" + "\n\n" + TIMELINE_SYSTEM


ANNOTATION_SYSTEM = """You annotate a FROZEN set of selected still frames using the COMPLETE native
video for context. Watch the whole video and examine every supplied still before answering.
This is a fresh annotation task: the frame list is final. Do not select, rank, drop, replace,
or request additional target frames. Return one annotation for EVERY specified target frame ID.

Keep two kinds of information explicitly separate:
1. visible_observation: describe only what can be seen in THIS still. Mention concrete visual
   details, relevant instrument/tissue positions, and visibility limits. A still alone does not
   establish motion, the ordinal number of a stitch, successful repair, or a clinical outcome.
2. contextual_claims: describe an interpretation supported by the surrounding VIDEO. Each
   claim must cite narrow evidence_intervals using start_frame_id and end_frame_id copied
   exactly from evidence_frame_inventory. This inventory covers ALL source frames, including
   frames outside the selected targets. Its rows map an existing frame ID to its exact playback
   timestamp in milliseconds; the timestamps help you locate frames but are not proof of events.

Choose endpoints by matching visible evidence in the video. Include both endpoint frames and
all source frames between them. The end must be at or after the start in source order. The
difference between their inventory timestamps must not exceed maximum_evidence_interval_ms.
For one observed frame, use the SAME frame ID for both endpoints: this is a point citation,
not evidence of an action's duration. This is also valid for the final frame of the clip.
Do not invent or calculate timestamps, frame IDs, unseen intermediate observations, or a
boundary beyond the last frame. Do not put timecodes in descriptive text. Software attaches
timestamps from the cited source rows. Never fill neat time windows merely to produce a citation.

Context may explain visible evidence, but an event elsewhere in the video must never become
something supposedly visible in this still. Do not infer anatomy, maneuvers, stitch counts,
completion, safety, or treatment success from the documented procedure name. If you cannot
locate supporting frames, omit the contextual claim. An empty contextual_claims list is valid.
Report uncertainties explicitly; poor or uninterpretable views must include a limitation.
Do not fill gaps with plausible surgical storytelling. If the video is not surgical, annotate
the actual visible activity without imposing a surgical interpretation.

Source context is documented background, not independent visual evidence. Return
context_check consistent, uncertain, or conflict when background is provided, or not_supplied
otherwise. Copy frame IDs exactly. Source filenames and authoritative timestamps are attached
by software. These annotations are drafts awaiting review. Return only the required JSON.""" + "\n\n" + TIMELINE_SYSTEM


@dataclass(frozen=True)
class AnnotationConfig:
    context_size: int = 131072
    image_max_tokens: int = 256
    max_tokens: int = 12288
    max_evidence_span_ms: int = 10000
    procedure_context: str = ""
    request_timeout_seconds: float = 3600

    def validate(self):
        require(0 < self.max_tokens < self.context_size <= 262144, "Invalid annotation context/output budget")
        require(self.image_max_tokens >= 64, "Image budget must be at least 64 tokens")
        require(self.max_evidence_span_ms > 0, "Evidence interval limit must be positive")
        require(type(self.request_timeout_seconds) in {int, float}
                and math.isfinite(self.request_timeout_seconds) and self.request_timeout_seconds > 0,
                "Annotation request timeout must be finite and positive")


def normalized_annotation_config(value):
    """Compare old saved settings using their original implicit timeout."""
    require(isinstance(value, dict), "Annotation session lacks its frozen configuration")
    return {**value, "request_timeout_seconds": value.get("request_timeout_seconds", 3600)}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    try:
        return _strict_json(Path(path).read_bytes())
    except (ValueError, OSError) as exc:
        raise ContractError(f"Cannot read annotation evidence {path}: {exc}") from exc


def _system(protocol_version):
    require(protocol_version in (LEGACY_PROTOCOL_VERSION, PROTOCOL_VERSION), "Unsupported annotation protocol")
    return LEGACY_ANNOTATION_SYSTEM if protocol_version == LEGACY_PROTOCOL_VERSION else ANNOTATION_SYSTEM


def _protocol_hash(protocol_version=PROTOCOL_VERSION):
    return hashlib.sha256((protocol_version + _system(protocol_version)).encode()).hexdigest()


def _evidence_mode(plan):
    version = plan.get("schema_version")
    _system(version)
    return FRAME_EVIDENCE if version == PROTOCOL_VERSION else TIMESTAMP_EVIDENCE


def _annotation_schema(plan, source):
    return annotation_schema(plan["frozen_frame_ids"], source["duration_ms"],
                             max_evidence_span_ms=plan["config"]["max_evidence_span_ms"],
                             source_frames=source["frames"], evidence_mode=_evidence_mode(plan))


def _build_annotations(raw, selected, source, plan):
    return build_annotations(raw, selected, source,
                             max_evidence_span_ms=plan["config"]["max_evidence_span_ms"],
                             evidence_mode=_evidence_mode(plan))


def _validate_selection(directory, selection_run, *, snapshot=False):
    """Tie effective keeps to canonical source rows and the completed raw review."""
    directory, selection_run = Path(directory), Path(selection_run)
    parent = _read(directory / ("selection-run.json" if snapshot else "run.json"))
    source = _read(directory / "source/source.json")
    record = _read(directory / "selection.json")
    state = _read(directory / ("selection-state.json" if snapshot else "state.json"))
    initial = _read(directory / "initial-selection.json")
    require(parent.get("schema_version") == "smart-frame-selection-run-v1"
            and record.get("schema_version") == "smart-frame-selection-v1", "Expected a normal smart-selection run")
    require(parent["source_manifest_sha256"] == sha256_file(directory / "source/source.json"),
            "Selection source manifest changed")
    require(record.get("status") == state.get("status") == "completed"
            and record.get("completed_rounds_full_video_verified") is True
            and not record.get("unresolved_searches"), "Annotate only a completed, fully reviewed selection")
    frames = source["frames"]
    by_id = {f["frame_id"]: f for f in frames}
    require(len(by_id) == len(frames) == source["expected_video_frames"], "Invalid canonical source frame inventory")
    ids = state["candidate_ids"]
    require(len(ids) == len(set(ids)) and set(ids) <= by_id.keys(), "Selection contains unknown/duplicate candidates")
    require(set(initial["protected_ids"]) <= set(ids), "Invalid protected selection anchors")
    require(state["rounds"] and state["next_round"] == len(state["rounds"]), "Selection lacks completed review receipts")
    require(record == _selection_record(source, state, initial["protected_ids"], selection_run),
            "Final selection differs from the saved source and review state")
    for row in record["rounds"]:
        receipt = row["verification"]
        require(receipt.get("accepted") is True and receipt.get("full_source_video_verified") is True
                and receipt.get("context_truncation_observed") is False
                and receipt.get("video_sha256") == source["video_sha256"]
                and receipt.get("decoded_frame_ids") == list(range(len(frames))),
                "Selection review did not verify the complete video")
    last = state["rounds"][-1]
    original_round = selection_run / "rounds" / f"round-{last['round']:02d}"
    require(Path(last["directory"]).resolve() == original_round.resolve(), "Selection review location changed")
    round_dir = directory / "selection-review" if snapshot else original_round
    video_name = "video" + (Path(source["video_path"]).suffix.lower() or ".mp4")
    require(len(state["messages"]) >= 3 and state["messages"][-1].get("role") == "assistant",
            "Selection lacks its final assistant response")
    parent_config = selection_config_from_saved(parent["config"])
    result = _recover_verified_result(round_dir, source, state["messages"][:-1], ids,
                                     parent_config, video_name)
    raw_message = result["response"]["choices"][0]["message"]
    require(state["messages"][-1] == {"role": "assistant", "content": raw_message["content"]}
            and state["last_output"] == result["output"], "Selection state differs from its verified raw response")
    validate_review(result["output"], ids, source["duration_ms"], review_mode=parent_config.review_mode)
    require(result["output"]["ready"] is True and result["output"]["context_check"] != "conflict",
            "Selection is not ready for annotation")
    selected_ids = record["selected_frame_ids"]
    require(1 <= len(selected_ids) <= 96 and len(set(selected_ids)) == len(selected_ids),
            "Annotation requires 1–96 uniquely selected frames")
    selected = sorted([by_id[f] for f in selected_ids], key=lambda f: (f["timestamp_ms"], f["frame_index"]))
    return parent, source, selected, original_round


def _prepare(selection_run, output, config):
    require(selection_run is not None, "--selection-run is required for a new annotation pass")
    selection_run = Path(selection_run).expanduser().resolve()
    require(selection_run.is_dir(), "Selection run does not exist")
    with (selection_run / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("Selection is still active; its final frame set cannot be frozen yet") from exc
        parent, source, selected, last_round = _validate_selection(selection_run, selection_run)
        source_root = Path(source["source_path"])
        if source["source_kind"] == "original_video":
            source_root = source_root.parent
        require(not output.is_relative_to(source_root.resolve()), "Annotation output must be outside source media")
        verify_assets(source)
        config = config or AnnotationConfig()
        if not config.procedure_context:
            config = AnnotationConfig(**{**asdict(config), "procedure_context": parent["config"].get("procedure_context", "")})
        config.validate()
        output.mkdir(parents=True, exist_ok=False)
        inputs = {"selection-run.json": selection_run / "run.json",
                  "selection.json": selection_run / "selection.json",
                  "selection-state.json": selection_run / "state.json",
                  "initial-selection.json": selection_run / "initial-selection.json",
                  "source/source.json": selection_run / "source/source.json"}
        for name in ("request.json", "response.json", "result.json", "verification.json", "output.json"):
            if (last_round / name).is_file():
                inputs["selection-review/" + name] = last_round / name
        hashes = {}
        for name, original in inputs.items():
            digest = sha256_file(original)
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
            require(sha256_file(destination) == digest == sha256_file(original), "Selection changed while freezing input")
            hashes[name] = digest
        _write(output / "selected-frames.json", selected)
        _write(output / "session.json", {"session_id": str(uuid.uuid4()), "created_at": _now(),
                                          "purpose": "fresh_full_video_annotation", "prior_selection_history_included": False,
                                          "config": asdict(config)})
        hashes.update({name: sha256_file(output / name) for name in ("selected-frames.json", "session.json")})
        plan = {"schema_version": PROTOCOL_VERSION, "created_at": _now(), "selection_run": str(selection_run),
                "config": asdict(config), "protocol_sha256": _protocol_hash(), "input_sha256": hashes,
                "frozen_frame_ids": [f["frame_id"] for f in selected],
                "temporal_exposure": "retrospective_full_video", "training_eligible": False}
        _write(output / "run.json", plan)
        return plan


def _messages(source, selected, names, video_name, config, *, protocol_version=PROTOCOL_VERSION):
    overview = {"task": "Annotate every final selected frame using direct visual evidence and complete-video context",
                "verified_source_context": config.procedure_context.strip() or None,
                "source_kind": source["source_kind"], "video_sha256": source["video_sha256"],
                "duration_ms": source["duration_ms"], "complete_video_frame_count": source["expected_video_frames"],
                "media_timeline": media_timeline(source),
                "timestamp_basis": source["timestamp_basis"], "timestamp_units": "milliseconds",
                "frame_ids": [f["frame_id"] for f in selected],
                "maximum_evidence_interval_ms": config.max_evidence_span_ms}
    if protocol_version == PROTOCOL_VERSION:
        overview["evidence_frame_inventory"] = {
            "columns": ["frame_id", "timestamp_ms"],
            "rows": [[frame["frame_id"], frame["timestamp_ms"]] for frame in source["frames"]],
        }
    return [{"role": "system", "content": _system(protocol_version)}, {"role": "user", "content": [
        {"type": "text", "text": json.dumps(overview, ensure_ascii=False)},
        {"type": "input_video", "input_video": {"url": "file://" + video_name}},
        *_image_blocks(selected, names),
        {"type": "text", "text": "Use the entire video and all supplied stills together. Return one grounded draft for every frame ID."},
    ]}]


def _verified_result(directory, source, messages, schema, config, video_name):
    request = _read(directory / "request.json")
    result = _read(directory / "result.json")
    require(request.get("messages") == messages and request.get("max_tokens") == config.max_tokens
            and request.get("response_format", {}).get("json_schema", {}).get("schema") == schema,
            "Saved annotation request differs from the fresh conversation")
    verification = result.get("verification", {})
    require(verification.get("accepted") is True and verification.get("full_source_video_verified") is True
            and verification.get("context_truncation_observed") is False
            and verification.get("request_sha256") == sha256_file(directory / "request.json")
            and verification.get("video_sha256") == source["video_sha256"]
            and verification.get("video_relative_path") == video_name
            and verification.get("video_fps_setting") == 0
            and verification.get("expected_video_frames") == source["expected_video_frames"]
            and verification.get("decoded_frames") == source["expected_video_frames"]
            and verification.get("decoded_frame_ids") == list(range(source["expected_video_frames"])),
            "Annotation response lacks verified complete native video evidence")
    response = _read(directory / "response.json")
    require(response == result.get("response"), "Annotation raw response differs from result")
    choices = response.get("choices", [])
    require(len(choices) == 1 and choices[0].get("finish_reason") == verification.get("finish_reason") == "stop",
            "Annotation response is unfinished")
    message = choices[0].get("message", {})
    require(message.get("role") == "assistant" and not message.get("tool_calls")
            and _strict_json(message.get("content", "")) == result.get("output"),
            "Annotation draft differs from the actual assistant response")
    prompt_tokens = response.get("usage", {}).get("prompt_tokens")
    require(type(prompt_tokens) is int and 0 < prompt_tokens <= config.context_size - config.max_tokens,
            "Annotation exceeded the configured context capacity")
    for name, expected in (("output.json", result["output"]), ("verification.json", verification)):
        if (directory / name).exists():
            require(_read(directory / name) == expected, f"Saved annotation artifact changed: {name}")
    try:
        Draft202012Validator(schema).validate(result["output"])
    except ValidationError as exc:
        raise ContractError(f"Invalid annotation response: {exc.message}") from exc
    return result


def _summary(output, plan, source, status, *, error=None):
    from .annotation_report import write_annotation_report
    try:
        session = _read(output / "session.json")
    except ContractError:
        session = {}
    summary = {"schema_version": plan["schema_version"], "status": status, "created_at": plan["created_at"],
               "updated_at": _now(), "config": plan["config"], "session_id": session.get("session_id"),
               "source": {key: source[key] for key in ("source_path", "video_path", "video_sha256", "duration_ms",
                                                       "expected_video_frames", "timestamp_basis")},
               "selected_frame_count": len(plan["frozen_frame_ids"]), "frame_ids": plan["frozen_frame_ids"],
               "annotations_path": str(output / "annotations.json"), "error": error,
               "review_required": True, "clinical_validation": "not_performed", "training_eligible": False}
    _write(output / "summary.json", summary)
    write_annotation_report(output)
    return summary


def run_annotation(selection_run, output_dir, config=None, *, resume=False, prepare_only=False,
                   runtime_factory=None, should_stop=lambda: False, progress=print):
    output = Path(output_dir).expanduser().resolve()
    if not resume:
        require(not output.exists(), "Annotation output already exists; use --resume for that exact pass")
        _prepare(selection_run, output, config)
    require((output / "run.json").is_file(), "No prepared annotation pass found")
    with (output / ".annotation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("This annotation pass is already active") from exc
        if not prepare_only:
            (output / ".pause-requested").unlink(missing_ok=True)
        plan = _read(output / "run.json")
        source = _read(output / "source/source.json")
        try:
            version = plan.get("schema_version")
            require(plan.get("protocol_sha256") == _protocol_hash(version),
                    "Annotation protocol changed; begin a separate pass")
            if resume and selection_run is not None:
                require(str(Path(selection_run).expanduser().resolve()) == plan["selection_run"], "Resume selection differs")
            config = AnnotationConfig(**plan["config"])
            config.validate()
            for name, digest in plan["input_sha256"].items():
                require(sha256_file(output / name) == digest, f"Frozen annotation input changed: {name}")
            # Historical passes used RuntimeConfig's 3,600-second default and
            # therefore did not serialize an annotation-specific timeout. Keep
            # their original session bytes and hashes intact while comparing.
            session_config = normalized_annotation_config(_read(output / "session.json").get("config"))
            require(session_config == asdict(config),
                    "Annotation configuration differs from the frozen session")
            _, source, selected, _ = _validate_selection(output, plan["selection_run"], snapshot=True)
            require(_read(output / "selected-frames.json") == selected
                    and plan["frozen_frame_ids"] == [f["frame_id"] for f in selected], "Frozen final frame set changed")
            verify_assets(source)
            _write(output / "native-timeline-verification.json", validate_native_timeline(source))
            if prepare_only:
                return _summary(output, plan, source, "prepared")
            pause = lambda: should_stop() or (output / ".pause-requested").exists()
            if pause():
                return _summary(output, plan, source, "paused")
            require(version == PROTOCOL_VERSION or (output / "round-00/result.json").is_file(),
                    "Legacy timestamp generation cannot resume; begin a new annotation pass with frame citations")
            media, video_name, names = stage_media(source, output)
            messages = _messages(source, selected, names, video_name, config, protocol_version=version)
            schema = _annotation_schema(plan, source)
            directory = output / "round-00"
            _summary(output, plan, source, "running")
            if not (directory / "result.json").exists():
                require(not (output / "annotations.json").exists(), "Annotation drafts exist without a verified model result")
                if directory.exists() and any(directory.iterdir()):
                    directory.rename(directory.with_name("round-00-interrupted-" + str(time.time_ns())))
                directory.mkdir(exist_ok=True)
                _write(directory / "annotation-context.json", _read(output / "session.json"))
                runtime_config = RuntimeConfig(PROJECT_ROOT, media, output / "runtime" / f"attempt-{time.time_ns()}",
                                               context_size=config.context_size, image_max_tokens=config.image_max_tokens,
                                               request_timeout=config.request_timeout_seconds)
                progress(f"Annotating {len(selected)} final frames together in a fresh session with the entire {source['expected_video_frames']}-frame video.")
                factory = runtime_factory or LocalVideoRuntime
                with factory(runtime_config, expected_video_frames=source["expected_video_frames"],
                             video_relative_path=video_name) as runtime:
                    runtime.chat(messages, schema=schema, max_tokens=config.max_tokens, round_dir=directory)
            result = _verified_result(directory, source, messages, schema, config, video_name)
            verify_assets(source)
            annotations = _build_annotations(result["output"], selected, source, plan)
            annotations.update(session_id=_read(output / "session.json")["session_id"],
                               selection_run=plan["selection_run"],
                               selection_sha256=plan["input_sha256"]["selection.json"],
                               selected_frames_sha256=plan["input_sha256"]["selected-frames.json"],
                               model_result_sha256=sha256_file(directory / "result.json"),
                               verification=result["verification"])
            destination = output / "annotations.json"
            if destination.exists():
                require(_read(destination) == annotations, "Saved annotations differ from the verified raw response")
            else:
                _write(destination, annotations)
            (output / "last-error.json").unlink(missing_ok=True)
            status = "context_conflict" if result["output"]["context_check"] == "conflict" else "completed"
            progress(f"Annotation saved: {len(selected)} drafts; {status}; human review required.")
            return _summary(output, plan, source, status)
        except BaseException as error:
            detail = {"type": type(error).__name__, "message": str(error), "at": _now()}
            _write(output / "last-error.json", detail)
            _summary(output, plan, source, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=detail)
            raise


def annotation_status(output_dir):
    output = Path(output_dir).expanduser().resolve()
    require((output / "run.json").is_file(), "No saved annotation pass found")
    summary = _read(output / "summary.json") if (output / "summary.json").exists() else {"status": "initializing"}
    with (output / ".annotation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            summary["writer_active"] = False
        except BlockingIOError:
            summary["writer_active"] = True
    summary["pause_requested"] = (output / ".pause-requested").exists()
    logs = sorted(output.glob("runtime/attempt-*/server.log"))
    if summary["writer_active"] and logs:
        with logs[-1].open("rb") as stream:
            stream.seek(max(0, logs[-1].stat().st_size - 32768))
            tail = stream.read().decode(errors="replace")
        prefill = re.findall(r"prompt processing, n_tokens =\s*(\d+), progress = ([0-9.]+)", tail)
        generated = re.findall(r"n_gen = (\d+), n_remaining = (\d+)", tail)
        summary["live_progress"] = {"log_path": str(logs[-1]),
            "prompt_fraction": float(prefill[-1][1]) if prefill else None,
            "generated_tokens": int(generated[-1][0]) if generated else None}
    return summary


def request_annotation_pause(output_dir):
    output = Path(output_dir).expanduser().resolve()
    require((output / "run.json").is_file(), "No saved annotation pass found")
    (output / ".pause-requested").touch()
    return {"pause_requested": True, "message": "An in-flight annotation call will finish and save its drafts; no new call will start."}


def add_annotation_parser(subparsers):
    parser = subparsers.add_parser("annotate-video-frames", help="Annotate final selected frames in a fresh complete-video session")
    parser.add_argument("--selection-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--procedure-context", default="")
    for name, default in asdict(AnnotationConfig()).items():
        if name != "procedure_context":
            parser.add_argument("--" + name.replace("_", "-"),
                                type=float if name == "request_timeout_seconds" else int, default=default)
    for name in ("annotation-status", "pause-annotation"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)


def annotation_cli(args, *, should_stop=lambda: False):
    config = None if args.resume else AnnotationConfig(**{key: getattr(args, key) for key in asdict(AnnotationConfig())})
    try:
        result = run_annotation(args.selection_run, args.output_dir, config, resume=args.resume,
                                prepare_only=args.prepare_only, should_stop=should_stop,
                                progress=lambda message: print(message, flush=True))
    except (VideoRuntimeError, ValidationError, ValueError) as error:
        raise ContractError(str(error)) from error
    print(json.dumps({"status": result["status"], "report": str(args.output_dir.resolve() / "report.html")}, indent=2))
