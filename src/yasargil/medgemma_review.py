"""Review frozen Qwen annotations with local MedGemma; defer evidence searches.

One fresh ordered-image request per key frame. Exact responses, including failed
or unfinished responses, are retained. This stage never calls a search model.
"""
from __future__ import annotations

import base64
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path

from .checkpoint import atomic_bytes, atomic_json, directory_is_locked, directory_lock, durable_mkdir
from .contract import ContractError, canonical_hash, require, sha256_file
from .frame_annotation import (
    AnnotationConfig, LEGACY_PROTOCOL_VERSION, PROTOCOL_VERSION as ANNOTATION_PROTOCOL_VERSION,
    _annotation_schema, _build_annotations, _messages as _annotation_messages,
    _protocol_hash as _annotation_protocol_hash, _verified_result,
)
from .llama_video import _strict_json
from .medgemma_evidence import build_evidence
from .medgemma_review_contract import REVIEW_SYSTEM, build_review, review_schema
from .llama_cpp import MEDGEMMA_MODEL, LlamaCppClient, LlamaCppError, _object, build_chat_request, encode_request
from .smart_selection import _image_blocks, verify_assets
from .video_source import media_timeline


PROTOCOL_VERSION = "medgemma-frame-review-v1"


@dataclass(frozen=True)
class ReviewConfig:
    medgemma_model: str = MEDGEMMA_MODEL
    before_frames: int = 2
    after_frames: int = 2
    max_context_frames: int = 12
    num_ctx: int = 65536
    num_predict: int = 4096
    seed: int = 42

    def validate(self):
        require(isinstance(self.medgemma_model, str) and self.medgemma_model.strip()
                and self.medgemma_model == self.medgemma_model.strip(), "An explicit MedGemma model is required")
        for key in ("before_frames", "after_frames"):
            require(type(getattr(self, key)) is int and 1 <= getattr(self, key) <= 8,
                    "Supply 1–8 context frames on each side")
        require(type(self.max_context_frames) is int
                and 1 + self.before_frames + self.after_frames <= self.max_context_frames <= 32,
                "Total image budget must fit the target and configured neighbors, and be at most 32")
        require(type(self.num_ctx) is int and 8192 <= self.num_ctx <= 131072, "Invalid MedGemma context budget")
        require(type(self.num_predict) is int and 256 <= self.num_predict < self.num_ctx, "Invalid generation budget")
        require(type(self.seed) is int and self.seed >= 0, "Invalid seed")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing or symlinked review artifact: {path}")
    try:
        return _strict_json(path.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise ContractError(f"Invalid review JSON: {path}: {exc}") from exc


def _relative(root, name):
    path = Path(name)
    require(not path.is_absolute() and path.parts and ".." not in path.parts,
            "Invalid artifact-relative path")
    destination = root / path
    require(destination.resolve().is_relative_to(root.resolve()), "Artifact path escapes its run")
    return destination


def _protocol_hash():
    return canonical_hash({"version": PROTOCOL_VERSION, "runtime": "llama.cpp", "transport": "openai-chat-v1", "system": REVIEW_SYSTEM,
                           "schema": review_schema("target", ["target", "before", "after"], 1000)})


def _load_annotation(root):
    """Validate saved Qwen output against its original request, including older prompts.

    Prompt changes in new annotation code must not invalidate a completed earlier
    run. Its original request and complete-video receipts are the authority.
    """
    plan, summary = _read(root / "run.json"), _read(root / "summary.json")
    require(plan.get("schema_version") in {LEGACY_PROTOCOL_VERSION, ANNOTATION_PROTOCOL_VERSION}
            and summary.get("status") in {"completed", "context_conflict"},
            "MedGemma needs a finished Qwen annotation pass")
    for name, digest in plan["input_sha256"].items():
        require(sha256_file(_relative(root, name)) == digest, f"Qwen input changed: {name}")
    source, selected = _read(root / "source/source.json"), _read(root / "selected-frames.json")
    media_timeline(source)
    ids = [frame["frame_id"] for frame in selected]
    require(ids == plan["frozen_frame_ids"] and 1 <= len(ids) <= 96, "Qwen selected frame set changed")
    config = AnnotationConfig(**plan["config"])
    config.validate()
    session = _read(root / "session.json")
    require(session.get("config") == plan["config"], "Qwen session settings changed")
    request = _read(root / "round-00/request.json")
    messages = request.get("messages", [])
    require(len(messages) == 2 and [m.get("role") for m in messages] == ["system", "user"],
            "Expected a fresh Qwen annotation conversation")
    blocks = messages[1]["content"]
    require(isinstance(blocks, list) and len(blocks) == 3 + 2 * len(selected), "Qwen annotation inputs changed")
    overview = _strict_json(blocks[0]["text"])
    require(overview.get("frame_ids") == ids and overview.get("video_sha256") == source["video_sha256"]
            and overview.get("duration_ms") == source["duration_ms"]
            and overview.get("complete_video_frame_count") == source["expected_video_frames"],
            "Qwen request source or final frames changed")
    video_name = "video" + (Path(source["video_path"]).suffix.lower() or ".mp4")
    names = {f["frame_id"]: f"frame-{index:08d}{Path(f['image_path']).suffix.lower()}"
             for index, f in enumerate(source["frames"])}
    require(blocks[1] == {"type": "input_video", "input_video": {"url": "file://" + video_name}}
            and blocks[2:-1] == _image_blocks(selected, names), "Qwen request omitted or changed visual evidence")
    if plan["schema_version"] == ANNOTATION_PROTOCOL_VERSION:
        require(messages == _annotation_messages(source, selected, names, video_name, config,
                                                 protocol_version=plan["schema_version"])
                and plan.get("protocol_sha256") == _annotation_protocol_hash(plan["schema_version"]),
                "Qwen frame-ID request differs from its canonical evidence inventory or protocol")
    result = _verified_result(root / "round-00", source, messages,
                              _annotation_schema(plan, source), config, video_name)
    annotations = _read(root / "annotations.json")
    expected = _build_annotations(result["output"], selected, source, plan)
    expected.update(session_id=session["session_id"], selection_run=plan["selection_run"],
                    selection_sha256=plan["input_sha256"]["selection.json"],
                    selected_frames_sha256=plan["input_sha256"]["selected-frames.json"],
                    model_result_sha256=sha256_file(root / "round-00/result.json"), verification=result["verification"])
    require(annotations == expected, "Qwen annotations differ from their verified raw response")
    return plan, source, annotations


def _prepare(annotation_run, output, config, dataset_root):
    require(annotation_run is not None, "--annotation-run is required for a new MedGemma review")
    parent = Path(annotation_run).expanduser().resolve()
    require(parent.is_dir(), "Qwen annotation run does not exist")
    require(not output.exists() and not output.is_relative_to(parent),
            "Use a new review output directory outside the Qwen annotation run")
    config = config or ReviewConfig()
    config.validate()
    dataset_root = Path(dataset_root).expanduser().resolve() if dataset_root is not None else None
    with (parent / ".annotation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("Qwen annotation is still active") from exc
        qwen_plan, source, annotations = _load_annotation(parent)
        source_root = Path(source["source_path"]).resolve()
        if source["source_kind"] == "original_video":
            source_root = source_root.parent
        require(not output.is_relative_to(source_root)
                and (dataset_root is None or not output.is_relative_to(dataset_root)),
                "Review output must be outside the source dataset")
        verify_assets(source)
        packets = [build_evidence(source, annotation, before_frames=config.before_frames,
                                  after_frames=config.after_frames, max_context_frames=config.max_context_frames,
                                  dataset_root=dataset_root,
                                  procedure_context=qwen_plan["config"].get("procedure_context", ""))
                   for annotation in annotations["annotations"]]
        for packet in packets:
            packet["qwen_context_check"] = annotations["context_check"]
            for table in packet["dataset_context"]["tables"]:
                require(not output.is_relative_to(Path(table["source_path"]).parent),
                        "Review output must be outside the inferred source dataset")
        durable_mkdir(output)
        hashes = {}
        names = set(qwen_plan["input_sha256"]) | {
            "run.json", "summary.json", "annotations.json", "round-00/request.json",
            "round-00/response.json", "round-00/result.json", "round-00/verification.json"}
        for name in sorted(names):
            original = _relative(parent, name)
            content = original.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            destination = _relative(output / "qwen", name)
            durable_mkdir(destination.parent)
            atomic_bytes(destination, content)
            require(sha256_file(original) == digest, "Qwen input changed while freezing the review")
            hashes["qwen/" + name] = digest
        durable_mkdir(output / "evidence")
        evidence_files = []
        for index, packet in enumerate(packets):
            name = f"evidence/frame-{index:04d}.json"
            atomic_json(output / name, packet)
            hashes[name] = sha256_file(output / name)
            evidence_files.append(name)
        plan = {"schema_version": PROTOCOL_VERSION, "runtime": "llama.cpp", "created_at": _now(),
                "annotation_run": str(parent), "dataset_root": str(dataset_root) if dataset_root else None,
                "config": asdict(config), "protocol_sha256": _protocol_hash(), "input_sha256": hashes,
                "frame_ids": [p["target_frame_id"] for p in packets], "evidence_files": evidence_files,
                "automated_followup": False, "human_review_required": True, "training_eligible": False}
        atomic_json(output / "run.json", plan)
        # Pin settings independently so edits to run.json cannot silently alter resume.
        atomic_json(output / "session.json", {"run_sha256": sha256_file(output / "run.json")})
    return plan


def build_request(evidence, config):
    """Hydrate ordered images with a locator in each image-bearing message."""
    messages = [{"role": "system", "content": REVIEW_SYSTEM}]
    for frame in evidence["frames"]:
        path = Path(frame["image_path"])
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == frame["image_sha256"], "Review image changed")
        require(sha256_file(frame["source_path"]) == frame["source_sha256"], "Original source evidence changed")
        locator = {key: frame.get(key) for key in (
            "frame_id", "frame_index", "release_frame_index", "timestamp_ms", "timestamp_basis",
            "source_pts", "time_base", "width", "height", "evidence_roles", "source_sha256", "image_sha256")}
        messages.append({"role": "user", "content": "Evidence image: " + json.dumps(locator, ensure_ascii=False),
                         "images": [base64.b64encode(raw).decode("ascii")]})
    draft = evidence["qwen_annotation"]
    qwen = {key: copy.deepcopy(draft[key]) for key in
            ("visible_observation", "visibility", "contextual_claims", "uncertainties")}
    for claim in qwen["contextual_claims"]:
        for interval in claim["evidence_intervals"]:
            interval["supporting_frame_ids"] = [f["frame_id"] for f in interval.pop("supporting_frames", [])]
    context = {key: value for key, value in evidence.items()
               if key not in {"frames", "qwen_annotation", "schema_version"}}
    # Full row hashes and source locators stay in the frozen packet. Repeating
    # paths and hashes for every label wastes the model's visual context budget.
    context["dataset_context"] = copy.deepcopy(evidence["dataset_context"])
    context["dataset_context"]["original_annotations"] = [
        {key: row[key] for key in ("annotation_id", "frame_id", "original_kind", "original_origin", "raw_value")}
        for row in evidence["dataset_context"]["original_annotations"]]
    context["qwen_annotation"] = qwen
    context["task"] = "Inspect the supplied evidence, then retain, correct or revise this key frame's Qwen annotation. Save unresolved evidence requests for later."
    messages.append({"role": "user", "content": json.dumps(context, ensure_ascii=False)})
    return build_chat_request(config.medgemma_model, messages,
        review_schema(evidence["target_frame_id"], [f["frame_id"] for f in evidence["frames"]],
                      evidence["media_timeline"]["duration_ms"]),
        num_ctx=config.num_ctx, num_predict=config.num_predict, seed=config.seed)


def _identity(info):
    require(info.get("runtime") == "llama.cpp", "Review model identity is not from llama.cpp")
    return {key: info[key] for key in ("name", "digest", "quantization", "runtime_version", "runtime",
                                     "model_file", "projector_file", "runtime_binary", "binary_version")}


def _parse(raw, evidence, config):
    envelope = _object(raw, "MedGemma review")
    LlamaCppClient._validate_chat(envelope, config.medgemma_model)
    message = envelope["choices"][0]["message"]
    require(not message.get("tool_calls"), "MedGemma review cannot dispatch tools")
    usage = envelope.get("usage")
    require(isinstance(usage, dict), "MedGemma token usage is unavailable")
    count = usage.get("prompt_tokens")
    require(type(count) is int and 0 < count <= config.num_ctx - config.num_predict,
            "MedGemma prompt usage is unavailable or leaves insufficient context capacity")
    completion = usage.get("completion_tokens")
    require(type(completion) is int and 0 < completion <= config.num_predict,
            "MedGemma completion usage is unavailable or exceeds the answer budget")
    try:
        content = _strict_json(message["content"])
    except (ValueError, UnicodeError) as exc:
        raise ContractError(f"Invalid MedGemma response JSON: {exc}") from exc
    return build_review(content, evidence)


def _save_equal(path, value):
    if path.exists():
        require(_read(path) == value, f"Saved review artifact changed: {path.name}")
    else:
        atomic_json(path, value)


def _call(output, index, evidence, config, client_factory, *, allow_inference=lambda: True):
    request = build_request(evidence, config)
    request_bytes = encode_request(request)
    call_dir = output / "calls" / f"frame-{index:04d}"
    attempts = sorted(call_dir.glob("attempt-*"))
    accepted = []
    for attempt in attempts:
        require(attempt.is_dir() and not attempt.is_symlink(), "Invalid review attempt directory")
        receipt = _read(attempt / "receipt.json") if (attempt / "receipt.json").exists() else None
        if receipt:
            require(set(receipt["files"]) == {"request.json", "response.json", "model.json", "review.json"},
                    "Review receipt has an incomplete artifact inventory")
            for name, digest in receipt["files"].items():
                require(sha256_file(attempt / name) == digest, f"Accepted review artifact changed: {name}")
        if (attempt / "request.json").exists():
            require((attempt / "request.json").read_bytes() == request_bytes, "Saved MedGemma request changed")
        if not (attempt / "response.json").exists():
            require(not receipt, "Accepted review response is missing")
            continue
        require((attempt / "request.json").exists() and (attempt / "model.json").exists(),
                "Saved MedGemma response lacks its request or model identity")
        require(_identity(_read(attempt / "model.json")) == _identity(_read(output / "model-info.json")),
                "Saved MedGemma model differs from the pinned model")
        try:
            review = _parse((attempt / "response.json").read_bytes(), evidence, config)
        except (ContractError, LlamaCppError, ValueError):
            require(not receipt, "Accepted MedGemma response is no longer valid")
            continue
        accepted.append((attempt, review))
    require(len(accepted) <= 1, "Multiple accepted MedGemma responses for one frame")
    if accepted:
        attempt, review = accepted[0]
    else:
        if not allow_inference():
            return None
        client = client_factory()
        metadata = client.model_info(config.medgemma_model)
        require(metadata.get("name") == config.medgemma_model and "vision" in metadata.get("capabilities", []),
                "Review requires the configured vision model")
        identity = _identity(metadata)
        pin = output / "model-info.json"
        if pin.exists():
            require(_identity(_read(pin)) == identity, "Pinned MedGemma model or runtime changed")
        else:
            atomic_json(pin, metadata)
        if not allow_inference():
            return None
        attempt = call_dir / f"attempt-{len(attempts) + 1:04d}"
        durable_mkdir(attempt)
        atomic_bytes(attempt / "request.json", request_bytes)
        atomic_json(attempt / "model.json", metadata)
        # Clear stale transport state: a failed connection must not archive an
        # earlier model-info or chat body as this frame's response.
        client.last_response_bytes = None
        try:
            raw = client.chat_raw(request)
            atomic_bytes(attempt / "response.json", raw)
            review = _parse(raw, evidence, config)
        except BaseException as exc:
            raw = getattr(client, "last_response_bytes", None)
            if raw is not None and not (attempt / "response.json").exists():
                atomic_bytes(attempt / "response.json", raw)
            atomic_json(attempt / "failure.json", {"type": type(exc).__name__, "message": str(exc), "at": _now()})
            raise
    _save_equal(attempt / "review.json", review)
    _save_equal(attempt / "receipt.json", {"files": {name: sha256_file(attempt / name)
                for name in ("request.json", "response.json", "model.json", "review.json")}})
    return {**review, "call_directory": str(attempt.relative_to(output))}


def _publish(output, plan, reviews, status, error=None):
    from .medgemma_review_report import write_review_report
    atomic_json(output / "reviews.json", {"schema_version": PROTOCOL_VERSION, "reviews": reviews,
                "human_review_required": True, "training_eligible": False}, overwrite=True)
    deferred = [{"target_frame_id": review["target_frame_id"], "status": "deferred_not_dispatched",
                 "evidence_requests": review["deferred_evidence_requests"],
                 "response_path": review["call_directory"] + "/response.json",
                 "review_path": review["call_directory"] + "/review.json"}
                for review in reviews if review["deferred_evidence_requests"]]
    atomic_json(output / "deferred-evidence.json", {"automated_followup": False, "requests": deferred}, overwrite=True)
    summary = {"schema_version": PROTOCOL_VERSION, "created_at": plan["created_at"], "updated_at": _now(),
               "status": status, "selected_frame_count": len(plan["frame_ids"]), "reviewed_frame_count": len(reviews),
               "deferred_frame_count": len(deferred), "config": plan["config"], "error": error,
               "automated_followup": False, "human_review_required": True, "training_eligible": False}
    atomic_json(output / "summary.json", summary, overwrite=True)
    write_review_report(output)
    return summary


def run_review(annotation_run, output_dir, config=None, *, dataset_root=None, resume=False,
               prepare_only=False, client=None, should_stop=lambda: False, progress=print):
    output = Path(output_dir).expanduser()
    require(not output.is_symlink(), "Review output cannot be a symlink")
    output = output.resolve()
    if not resume:
        _prepare(annotation_run, output, config, dataset_root)
    require((output / "run.json").is_file(), "No prepared MedGemma review found")
    with directory_lock(output):
        plan = _read(output / "run.json")
        reviews = []
        validated = False
        try:
            require(plan.get("runtime") == "llama.cpp",
                    "This review predates the llama.cpp migration; start a new review directory")
            require(_read(output / "session.json")["run_sha256"] == sha256_file(output / "run.json"),
                    "Frozen review plan changed")
            require(plan["schema_version"] == PROTOCOL_VERSION and plan["protocol_sha256"] == _protocol_hash(),
                    "Review protocol changed; start a new review")
            if annotation_run is not None:
                require(str(Path(annotation_run).expanduser().resolve()) == plan["annotation_run"], "Resume annotation differs")
            if dataset_root is not None:
                require(str(Path(dataset_root).expanduser().resolve()) == plan["dataset_root"], "Resume dataset differs")
            if config is not None:
                require(asdict(config) == plan["config"], "Resume settings differ")
            config = ReviewConfig(**plan["config"])
            config.validate()
            for name, digest in plan["input_sha256"].items():
                require(sha256_file(_relative(output, name)) == digest, f"Frozen MedGemma evidence changed: {name}")
            _load_annotation(output / "qwen")
            packets = [_read(output / name) for name in plan["evidence_files"]]
            require([packet["target_frame_id"] for packet in packets] == plan["frame_ids"], "Review target set changed")
            if prepare_only:
                require(not (output / "calls").exists(), "Prepare-only cannot reset an already started review")
                return _publish(output, plan, reviews, "prepared")
            validated = True
            (output / ".pause-requested").unlink(missing_ok=True)
            def get_client():
                nonlocal client
                if client is None:
                    client = LlamaCppClient()
                return client
            for index, evidence in enumerate(packets):
                _publish(output, plan, reviews, "running")
                progress(f"MedGemma review {index + 1}/{len(packets)}: {evidence['target_frame_id']}, {len(evidence['frames'])} images")
                review = _call(output, index, evidence, config, get_client,
                               allow_inference=lambda: not should_stop() and not (output / ".pause-requested").exists())
                if review is None:
                    return _publish(output, plan, reviews, "paused")
                reviews.append(review)
                _publish(output, plan, reviews, "running")
            status = "completed_with_deferred_evidence" if any(r["deferred_evidence_requests"] for r in reviews) else "completed"
            (output / "last-error.json").unlink(missing_ok=True)
            return _publish(output, plan, reviews, status)
        except BaseException as exc:
            detail = {"type": type(exc).__name__, "message": str(exc), "at": _now()}
            atomic_json(output / "last-error.json", detail, overwrite=True)
            if validated:
                _publish(output, plan, reviews, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", detail)
            raise


def review_status(output_dir):
    output = Path(output_dir).expanduser().resolve()
    result = _read(output / "summary.json")
    return {**result, "writer_active": directory_is_locked(output),
            "pause_requested": (output / ".pause-requested").exists()}


def request_review_pause(output_dir):
    output = Path(output_dir).expanduser().resolve()
    require((output / "run.json").is_file(), "No saved MedGemma review found")
    atomic_json(output / ".pause-requested", {"at": _now()}, overwrite=True)
    return {"pause_requested": True, "message": "The active MedGemma response will be saved before pausing."}


def add_review_parser(subparsers):
    parser = subparsers.add_parser("review-frame-annotations", help="MedGemma reviews Qwen drafts with before/after evidence; saves further evidence requests")
    parser.add_argument("--annotation-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    for key, default in asdict(ReviewConfig()).items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(default), default=None,
                            help=f"Default: {default}; resume uses the frozen value")
    for name in ("frame-review-status", "pause-frame-review"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)


def review_cli(args, *, should_stop=lambda: False):
    settings = {key: getattr(args, key) for key in asdict(ReviewConfig()) if getattr(args, key) is not None}
    if args.resume:
        saved = _read(args.output_dir.expanduser().resolve() / "run.json")["config"]
        require(all(saved[key] == value for key, value in settings.items()), "Resume settings differ from frozen settings")
        config = None
    else:
        config = ReviewConfig(**settings)
    result = run_review(args.annotation_run, args.output_dir, config, dataset_root=args.dataset_root,
                        resume=args.resume, prepare_only=args.prepare_only,
                        client=LlamaCppClient(args.project_root, timeout=args.timeout), should_stop=should_stop,
                        progress=lambda message: print(message, flush=True))
    print(json.dumps({"status": result["status"], "reviewed": result["reviewed_frame_count"],
                      "deferred": result["deferred_frame_count"],
                      "report": str(args.output_dir.resolve() / "report.html")}, indent=2))
