"""Seal completed annotation evidence without rewriting model output or provenance.

The seal detects later byte changes; it is a local integrity record, not a
signature or a claim that Qwen's clinical interpretations are correct.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .contract import ContractError, require, sha256_file
from .frame_annotation import (
    AnnotationConfig, _annotation_schema, _build_annotations, _messages, _protocol_hash,
    _read, _validate_selection, _verified_result,
    normalized_annotation_config,
)
from .smart_selection import verify_assets
from .video_source import validate_native_timeline


SCHEMA_VERSION = "annotation-output-integrity-v1"
MANIFEST_NAME = "integrity.json"
_FROZEN = {
    "session.json", "source/source.json", "selected-frames.json", "selection-run.json",
    "selection.json", "selection-state.json", "initial-selection.json",
    *(f"selection-review/{name}.json" for name in ("request", "response", "result", "output", "verification")),
}
_REQUIRED = _FROZEN | {
    "run.json", "native-timeline-verification.json", "annotations.json",
    "round-00/annotation-context.json",
    *(f"round-00/{name}.json" for name in ("request", "response", "result", "output", "verification")),
}


def _artifact_paths(output):
    paths = set(_REQUIRED)
    for name in _REQUIRED:
        require((output / name).is_file(), f"Missing annotation integrity evidence: {name}")
    segment = output / "round-00/server-segment.log"
    if segment.exists():
        paths.add(segment.relative_to(output).as_posix())
    runtime = output / "runtime"
    if runtime.exists():
        for path in runtime.rglob("*"):
            if path.is_file() and (path.name in {"runtime.json", "server.log"}
                                   or "native-decode" in path.relative_to(runtime).parts
                                   or (path.suffix == ".json"
                                       and "transport-receipts" in path.relative_to(runtime).parts)):
                paths.add(path.relative_to(output).as_posix())
    return sorted(paths)


def _hashes(output, names):
    result = {}
    for name in names:
        path = output / name
        require(isinstance(name, str) and Path(name).as_posix() == name
                and not Path(name).is_absolute() and ".." not in Path(name).parts
                and path.resolve().is_relative_to(output) and path.is_file() and not path.is_symlink(),
                f"Invalid annotation evidence path: {name}")
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns, before.st_ino)
                == (after.st_size, after.st_mtime_ns, after.st_ino), f"Evidence changed while reading: {name}")
        result[name] = {"sha256": digest, "size_bytes": after.st_size}
    return result


def _runtime_binding(output, result, source, config):
    """Validate advertised native receipts; scripted test runtimes may omit them."""
    verification = result["verification"]
    runtime_name = verification.get("runtime_directory")
    if runtime_name is None:
        require(verification.get("byte_transport_receipt") is None
                and verification.get("byte_transport_receipt_path") is None,
                "Byte transport receipt lacks its annotation runtime")
        return
    runtime = Path(runtime_name).resolve()
    require(runtime.is_relative_to(output / "runtime"), "Annotation runtime evidence is outside this pass")
    metadata = _read(runtime / "runtime.json")
    native = _read(runtime / "native-decode/verification.json")
    require(metadata.get("context_size") == config.context_size
            and metadata.get("image_max_tokens") == config.image_max_tokens
            and metadata.get("expected_video_frames") == source["expected_video_frames"]
            and metadata.get("video_sha256") == source["video_sha256"]
            and metadata.get("video_fps_setting") == 0
            and metadata.get("native_decode_verification") == native,
            "Annotation runtime metadata differs from its source or frozen settings")
    command = metadata.get("command", [])
    require(isinstance(command, list) and command.count("--timeout") == 1
            and command.index("--timeout") + 1 < len(command)
            and command[command.index("--timeout") + 1] == str(math.ceil(config.request_timeout_seconds)),
            "Annotation runtime timeout differs from its frozen settings")
    require(native.get("accepted") is True and native.get("ordered_rgb_frames_identical") is True
            and native.get("video_sha256") == source["video_sha256"]
            and all(native.get(key) == source["expected_video_frames"] for key in
                    ("expected_video_frames", "source_decoded_frames", "native_decoded_frames")),
            "Native decode receipt does not verify the complete source")
    for name in ("server.log", "native-decode/source.framehash", "native-decode/native.framehash",
                 "native-decode/ffprobe.json", "native-decode/source.log", "native-decode/native.log"):
        require((runtime / name).is_file(), f"Missing advertised runtime evidence: {name}")
    require((output / "round-00/server-segment.log").is_file(), "Missing annotation inference log segment")
    _transport_binding(runtime, metadata, native, verification, source)


def _transport_binding(runtime, metadata, native, verification, source):
    """Bind saved spool receipts to their exact input and frozen adapter identity.

    Compare with the adapter identity recorded at inference, rather than current
    installed code: later adapter updates must not rewrite or invalidate old runs.
    """
    transport = metadata.get("ffmpeg_transport")
    advertised = (transport, metadata.get("ffmpeg_transport_receipts_directory"),
                  native.get("ffmpeg_transport"), native.get("byte_transport_receipts"),
                  verification.get("byte_transport_receipt"), verification.get("byte_transport_receipt_path"))
    if all(value is None for value in advertised):
        return
    from .ffmpeg_transport import SCHEMA_VERSION as TRANSPORT_VERSION
    require(isinstance(transport, dict) and transport.get("schema_version") == TRANSPORT_VERSION,
            "Missing or invalid pinned FFmpeg transport metadata")
    require(isinstance(transport.get("module"), dict), "Missing pinned FFmpeg transport module")
    module_hash = transport["module"].get("sha256")
    manifest_hash = transport.get("manifest_sha256")
    require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                for value in (module_hash, manifest_hash)), "Invalid pinned FFmpeg transport digests")
    require(native.get("ffmpeg_transport") == transport,
            "Native preflight used a different pinned FFmpeg transport")
    receipt_root = runtime / "transport-receipts"
    require(metadata.get("ffmpeg_transport_receipts_directory") == str(receipt_root)
            and receipt_root.is_dir() and receipt_root.resolve() == receipt_root,
            "FFmpeg transport receipt directory differs from its annotation runtime")
    location = verification.get("byte_transport_receipt_path")
    require(isinstance(location, str) and Path(location).is_absolute(),
            "Missing absolute native byte transport receipt path")
    receipt_path = Path(location)
    require(receipt_path.suffix == ".json" and receipt_path.is_file() and not receipt_path.is_symlink()
            and receipt_path.resolve() == receipt_path and receipt_path.is_relative_to(receipt_root),
            "Native byte transport receipt is outside its runtime transport directory")
    receipt = _read(receipt_path)
    require(receipt == verification.get("byte_transport_receipt"),
            "Saved byte transport receipt differs from the accepted model result")
    preflight = native.get("byte_transport_receipts")
    require(isinstance(preflight, dict) and set(preflight) == {"source", "native"},
            "Native preflight lacks both byte transport receipts")
    video_size = Path(source["video_path"]).stat().st_size
    for item in (receipt, preflight["source"], preflight["native"]):
        require(isinstance(item, dict) and item.get("schema_version") == TRANSPORT_VERSION
                and item.get("input_sha256") == source["video_sha256"]
                and type(item.get("input_bytes")) is int and item["input_bytes"] == video_size
                and item.get("module_sha256") == module_hash
                and item.get("transport_manifest_sha256") == manifest_hash,
                "Native byte transport receipt differs from original video bytes or pinned adapter")


def _validated_evidence(output):
    names = _artifact_paths(output)
    before = _hashes(output, names)
    plan, session = _read(output / "run.json"), _read(output / "session.json")
    protocol_version = plan.get("schema_version")
    require(plan.get("protocol_sha256") == _protocol_hash(protocol_version),
            "Annotation protocol differs from its frozen request")
    config = AnnotationConfig(**plan["config"])
    config.validate()
    require(normalized_annotation_config(session["config"]) == asdict(config),
            "Annotation settings differ from the frozen session")
    require(_FROZEN <= set(plan["input_sha256"]), "Annotation plan does not pin all frozen evidence")
    for name, digest in plan["input_sha256"].items():
        require(name in before and before[name]["sha256"] == digest, f"Frozen annotation input changed: {name}")
    _, source, selected, _ = _validate_selection(output, plan["selection_run"], snapshot=True)
    frame_ids = [frame["frame_id"] for frame in selected]
    require(_read(output / "selected-frames.json") == selected and plan["frozen_frame_ids"] == frame_ids,
            "Frozen annotation frames differ from the final selection")
    require(_read(output / "native-timeline-verification.json") == validate_native_timeline(source),
            "Saved native timeline receipt differs from the source timeline")
    require(_read(output / "round-00/annotation-context.json") == session,
            "Annotation inference session differs from its frozen context")
    # Derive the existing aliases without creating links or touching source media.
    video_name = "video" + (Path(source["video_path"]).suffix.lower() or ".mp4")
    aliases = {frame["frame_id"]: f"frame-{index:08d}{Path(frame['image_path']).suffix.lower()}"
               for index, frame in enumerate(source["frames"])}
    messages = _messages(source, selected, aliases, video_name, config, protocol_version=protocol_version)
    schema = _annotation_schema(plan, source)
    result = _verified_result(output / "round-00", source, messages, schema, config, video_name)
    _runtime_binding(output, result, source, config)
    verify_assets(source)
    expected = _build_annotations(result["output"], selected, source, plan)
    expected.update(session_id=session["session_id"], selection_run=plan["selection_run"],
                    selection_sha256=plan["input_sha256"]["selection.json"],
                    selected_frames_sha256=plan["input_sha256"]["selected-frames.json"],
                    model_result_sha256=before["round-00/result.json"]["sha256"],
                    verification=result["verification"])
    require(_read(output / "annotations.json") == expected,
            "Published annotations differ from the raw Qwen response and canonical provenance")
    status = "context_conflict" if result["output"]["context_check"] == "conflict" else "completed"
    summary = _read(output / "summary.json")
    require(summary.get("status") == status and summary.get("session_id") == session["session_id"]
            and summary.get("frame_ids") == frame_ids and summary.get("selected_frame_count") == len(frame_ids),
            "Only a completed annotation pass can be sealed")
    require(names == _artifact_paths(output) and before == _hashes(output, names),
            "Annotation evidence changed during validation")
    return {"status": status, "session_id": session["session_id"], "frame_ids": frame_ids,
            "annotation_count": len(frame_ids), "source_video_sha256": source["video_sha256"],
            "source_frame_count": source["expected_video_frames"], "artifacts": before}


def _verify(output):
    manifest = _read(output / MANIFEST_NAME)
    require(manifest.get("schema_version") == SCHEMA_VERSION and manifest.get("output_directory") == str(output)
            and manifest.get("clinical_validation") == "not_performed" and manifest.get("training_eligible") is False,
            "Invalid annotation completion manifest")
    require(manifest.get("artifacts") == _hashes(output, _artifact_paths(output)),
            "Sealed annotation evidence bytes have changed")
    evidence = _validated_evidence(output)
    require(all(manifest.get(key) == value for key, value in evidence.items()),
            "Sealed annotation evidence has changed")
    return manifest


def _locked(output, action):
    output = Path(output).expanduser().resolve()
    require((output / ".annotation.lock").is_file(), "No annotation pass lock exists")
    with (output / ".annotation.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("Annotation is still running; cannot seal or verify output") from exc
        try:
            return action(output)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"Invalid annotation integrity evidence: {exc}") from exc


def verify_annotation_output(output):
    """Revalidate the saved seal and exact evidence bytes without rewriting files."""
    return _locked(output, _verify)


def seal_annotation_output(output):
    """Publish a new completion seal, or verify the existing seal without replacing it."""
    def seal(directory):
        destination = directory / MANIFEST_NAME
        if destination.exists():
            return _verify(directory)
        evidence = _validated_evidence(directory)
        manifest = {"schema_version": SCHEMA_VERSION, "output_directory": str(directory),
                    "created_at": datetime.now(timezone.utc).isoformat(), **evidence,
                    "clinical_validation": "not_performed", "training_eligible": False}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                             prefix=".annotation-integrity-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                return _verify(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return manifest
    return _locked(output, seal)
