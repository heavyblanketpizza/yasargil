"""Independent, evidence-grounded MedGemma annotation of selected surgical frames."""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path

from .checkpoint import atomic_bytes, atomic_json, directory_is_locked, directory_lock, durable_mkdir
from .contract import ContractError, canonical_hash, require, sha256_file
from .llama_cpp import MEDGEMMA_MODEL, LlamaCppClient, LlamaCppError, _object, build_chat_request, encode_request
from .llama_video import _strict_json
from .medgemma_annotation_contract import ANNOTATION_SYSTEM, annotation_schema, build_annotation
from .medgemma_annotation_evidence import build_evidence
from .smart_selection import validate_completed_selection, verify_assets


PROTOCOL_VERSION = "medgemma-frame-annotation-v1"


@dataclass(frozen=True)
class AnnotationConfig:
    medgemma_model: str = MEDGEMMA_MODEL
    before_frames: int = 2
    after_frames: int = 2
    detail_crops: bool = True
    num_ctx: int = 32768
    num_predict: int = 4096
    seed: int = 42
    procedure_context: str = ""

    def validate(self):
        require(self.medgemma_model == MEDGEMMA_MODEL, "Use the configured local MedGemma vision model")
        for name in ("before_frames", "after_frames"):
            require(type(getattr(self, name)) is int and 0 <= getattr(self, name) <= 8,
                    "Supply 0–8 source context frames on each side")
        require(type(self.detail_crops) is bool, "detail_crops must be boolean")
        require(type(self.num_ctx) is int and 8192 <= self.num_ctx <= 131072, "Invalid context budget")
        require(type(self.num_predict) is int and 256 <= self.num_predict <= 8192
                and self.num_predict < self.num_ctx, "Invalid answer budget")
        require(type(self.seed) is int and self.seed >= 0, "Invalid seed")
        require(isinstance(self.procedure_context, str) and len(self.procedure_context) <= 4000,
                "Procedure context must be at most 4000 characters")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing or symlinked annotation artifact: {path}")
    return _strict_json(path.read_bytes())


def _relative(root, name):
    path = Path(name)
    require(not path.is_absolute() and path.parts and ".." not in path.parts, "Invalid artifact path")
    result = root / path
    require(result.resolve().is_relative_to(root.resolve()) and not result.is_symlink(), "Artifact escapes run")
    return result


def _save_equal(path, value):
    if path.exists():
        require(_read(path) == value, f"Saved annotation artifact changed: {path.name}")
    else:
        atomic_json(path, value)


def _protocol_hash():
    # Pin both the prompt and its evidence/schema implementation for safe resume.
    from . import medgemma_annotation_contract, medgemma_annotation_evidence
    return canonical_hash({"version": PROTOCOL_VERSION, "system": ANNOTATION_SYSTEM,
        "contract": sha256_file(Path(medgemma_annotation_contract.__file__)),
        "evidence": sha256_file(Path(medgemma_annotation_evidence.__file__)),
        "runner": sha256_file(Path(__file__)),
        "request_layout": "target_views_then_ordered_context_then_allowlisted_metadata_v1"})


def _prepare(selection_run, output, config):
    require(selection_run is not None, "--selection-run is required for a new MedGemma annotation")
    parent = Path(selection_run).expanduser().resolve()
    require(parent.is_dir() and not output.exists() and not output.is_relative_to(parent),
            "Use a new annotation directory outside the selection run")
    config = config or AnnotationConfig()
    config.validate()
    with (parent / ".run.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("Selection is still active") from exc
        selection_plan, source, selected, last_round = validate_completed_selection(parent, parent)
        source_root = Path(source["source_path"]).resolve()
        if source["source_kind"] == "original_video":
            source_root = source_root.parent
        require(not output.is_relative_to(source_root), "Annotation output must be outside source media")
        if source["source_kind"] == "released_image_sequence" and source_root.parent.name == "frames":
            require(not output.is_relative_to(source_root.parent.parent), "Annotation output must be outside the dataset")
        if not config.procedure_context:
            config = replace(config, procedure_context=selection_plan["config"].get("procedure_context", ""))
        config.validate()
        verify_assets(source)
        output.mkdir(parents=True, exist_ok=False)
        durable_mkdir(output)
        inputs = {"selection/selection-run.json": parent / "run.json",
                  "selection/selection.json": parent / "selection.json",
                  "selection/selection-state.json": parent / "state.json",
                  "selection/initial-selection.json": parent / "initial-selection.json",
                  "selection/source/source.json": parent / "source/source.json",
                  "source.json": parent / "source/source.json"}
        for name in ("request.json", "response.json", "result.json", "verification.json", "output.json"):
            if (last_round / name).is_file():
                inputs["selection/selection-review/" + name] = last_round / name
        hashes = {}
        for name, original in inputs.items():
            content = original.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            destination = _relative(output, name)
            durable_mkdir(destination.parent)
            atomic_bytes(destination, content)
            require(sha256_file(original) == digest, "Selection changed while freezing evidence")
            hashes[name] = digest
        atomic_json(output / "selected-frames.json", selected)
        hashes["selected-frames.json"] = sha256_file(output / "selected-frames.json")
        durable_mkdir(output / "evidence")
        names = []
        for index, target in enumerate(selected):
            packet = build_evidence(source, target["frame_id"], output / "assets" / f"frame-{index:04d}",
                before_frames=config.before_frames, after_frames=config.after_frames,
                detail_crops=config.detail_crops, procedure_context=config.procedure_context)
            name = f"evidence/frame-{index:04d}.json"
            atomic_json(output / name, packet)
            hashes[name] = sha256_file(output / name)
            names.append(name)
        plan = {"schema_version": PROTOCOL_VERSION, "runtime": "llama.cpp", "created_at": _now(),
                "selection_run": str(parent), "source_file": "source.json", "selected_file": "selected-frames.json",
                "config": asdict(config), "protocol_sha256": _protocol_hash(), "input_sha256": hashes,
                "frame_ids": [row["frame_id"] for row in selected], "evidence_files": names,
                "source_frame_count": len(source["frames"]),
                "qwen_annotations_included": False, "source_labels_included": False,
                "human_review_required": True, "training_eligible": False}
        plan_bytes = (json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        atomic_json(output / "session.json", {"run_sha256": hashlib.sha256(plan_bytes).hexdigest()})
        # The run is resumable only after every preparation artifact is durable.
        atomic_bytes(output / "run.json", plan_bytes)


def build_request(packet, config):
    """Only explicit visual evidence and documented context enter the model prompt."""
    config.validate()
    messages = [{"role": "system", "content": ANNOTATION_SYSTEM}]
    frames = {row["frame_id"]: row for row in packet["frames"]}
    verified = set()
    for frame in frames.values():
        for path_key, hash_key in (("source_path", "source_sha256"), ("image_path", "image_sha256")):
            identity = (frame[path_key], frame[hash_key])
            if identity not in verified:
                require(sha256_file(identity[0]) == identity[1], "Source image changed")
                verified.add(identity)
    for view in packet["views"]:
        raw = Path(view["image_path"]).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == view["image_sha256"], "Annotation view changed")
        locator = {key: view[key] for key in ("view_id", "frame_id", "role", "bounds", "width", "height")}
        locator["timestamp_ms"] = frames[view["frame_id"]]["timestamp_ms"]
        messages.append({"role": "user", "content": json.dumps(locator),
                         "images": [base64.b64encode(raw).decode("ascii")]})
    context = {"task": "Author an independent surgical annotation for this target frame.",
               "target_frame_id": packet["target_frame_id"], "procedure_context": packet["procedure_context"],
               "media_timeline": packet["media_timeline"], "limitations": packet["limitations"],
               "neighbor_coverage": packet["neighbor_coverage"]}
    messages.append({"role": "user", "content": json.dumps(context, ensure_ascii=False)})
    return build_chat_request(config.medgemma_model, messages, annotation_schema(packet),
        num_ctx=config.num_ctx, num_predict=config.num_predict, seed=config.seed)


def _identity(info):
    require(info.get("runtime") == "llama.cpp", "Expected llama.cpp model identity")
    return {key: info[key] for key in ("name", "digest", "quantization", "runtime_version", "runtime",
                                     "model_file", "projector_file", "runtime_binary", "binary_version")}


def _parse(raw, packet, config):
    envelope = _object(raw, "MedGemma annotation")
    LlamaCppClient._validate_chat(envelope, config.medgemma_model)
    usage = envelope.get("usage", {})
    count, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    require(type(count) is int and 0 < count <= config.num_ctx - config.num_predict,
            "Prompt usage is unavailable or leaves insufficient answer capacity")
    require(type(completion) is int and 0 < completion <= config.num_predict,
            "Completion usage is unavailable or exceeds the answer budget")
    return build_annotation(_strict_json(envelope["choices"][0]["message"]["content"]), packet)


def _call(output, index, packet, config, client_factory, allow_inference):
    request = build_request(packet, config)
    request_bytes = encode_request(request)
    root = output / "calls" / f"frame-{index:04d}"
    marker = root / "accepted-response.json"
    if marker.exists():
        acceptance = _read(marker)
        receipt_path = _relative(output, acceptance["receipt_path"])
        require(receipt_path.parent.parent == root and receipt_path.name == "receipt.json"
                and sha256_file(receipt_path) == acceptance["receipt_sha256"], "Accepted response receipt changed")
    attempts = sorted(root.glob("attempt-*"))
    accepted = []
    for attempt in attempts:
        require(attempt.is_dir() and not attempt.is_symlink(), "Invalid annotation attempt")
        receipt = _read(attempt / "receipt.json") if (attempt / "receipt.json").exists() else None
        if receipt:
            require(set(receipt["files"]) == {"request.json", "response.json", "model.json", "annotation.json"},
                    "Incomplete annotation receipt")
            for name, digest in receipt["files"].items():
                require(sha256_file(attempt / name) == digest, "Accepted annotation bytes changed")
        if (attempt / "request.json").exists():
            require((attempt / "request.json").read_bytes() == request_bytes, "Frozen annotation request changed")
        if not (attempt / "response.json").exists():
            require(not receipt, "Accepted annotation response missing")
            continue
        require((attempt / "request.json").exists(), "Response lacks its request")
        require(_identity(_read(attempt / "model.json")) == _identity(_read(output / "model-info.json")),
                "Annotation model identity changed")
        try:
            result = _parse((attempt / "response.json").read_bytes(), packet, config)
        except (ContractError, LlamaCppError, ValueError):
            require(not receipt, "Accepted annotation no longer validates")
            continue
        accepted.append((attempt, result))
    require(len(accepted) <= 1, "Multiple accepted annotations for one target")
    if accepted:
        attempt, result = accepted[0]
    else:
        if not allow_inference():
            return None
        client = client_factory()
        info = client.model_info(config.medgemma_model)
        require(info.get("name") == config.medgemma_model and "vision" in info.get("capabilities", []),
                "Independent annotation requires the configured vision model")
        identity = _identity(info)
        if (output / "model-info.json").exists():
            require(_identity(_read(output / "model-info.json")) == identity, "Pinned model or runtime changed")
        else:
            atomic_json(output / "model-info.json", info)
        if not allow_inference():
            return None
        attempt = root / f"attempt-{len(attempts) + 1:04d}"
        durable_mkdir(attempt)
        atomic_bytes(attempt / "request.json", request_bytes)
        atomic_json(attempt / "model.json", info)
        client.last_response_bytes = None
        try:
            raw = client.chat_raw(request)
            atomic_bytes(attempt / "response.json", raw)
            result = _parse(raw, packet, config)
        except BaseException as exc:
            raw = getattr(client, "last_response_bytes", None)
            if raw is not None and not (attempt / "response.json").exists():
                atomic_bytes(attempt / "response.json", raw)
            atomic_json(attempt / "failure.json", {"type": type(exc).__name__, "message": str(exc), "at": _now()})
            raise
    _save_equal(attempt / "annotation.json", result)
    _save_equal(attempt / "receipt.json", {"files": {name: sha256_file(attempt / name)
        for name in ("request.json", "response.json", "model.json", "annotation.json")}})
    _save_equal(root / "accepted-response.json", {"receipt_path": str((attempt / "receipt.json").relative_to(output)),
                                               "receipt_sha256": sha256_file(attempt / "receipt.json")})
    return {"target_frame_id": packet["target_frame_id"], "target": packet["target"], "evidence": packet,
            "annotation": result, "call_directory": str(attempt.relative_to(output)),
            "human_review_required": True, "training_eligible": False}


def _document(rows):
    return {"schema_version": PROTOCOL_VERSION, "annotations": rows,
            "human_review_required": True, "training_eligible": False}


def _publish(output, plan, rows, status, error=None):
    from .medgemma_annotation_report import write_annotation_report
    atomic_json(output / "annotations.json", _document(rows), overwrite=True)
    unresolved = sum(bool(row["annotation"]["unresolved_questions"]) for row in rows)
    summary = {"schema_version": PROTOCOL_VERSION, "created_at": plan["created_at"], "updated_at": _now(),
               "status": status, "selected_frame_count": len(plan["frame_ids"]), "annotated_frame_count": len(rows),
               "source_frame_count": plan["source_frame_count"],
               "unresolved_frame_count": unresolved, "config": plan["config"], "error": error,
               "qwen_annotations_included": False, "source_labels_included": False,
               "human_review_required": True, "training_eligible": False}
    atomic_json(output / "summary.json", summary, overwrite=True)
    write_annotation_report(output)
    return summary


def run_annotation(selection_run, output_dir, config=None, *, resume=False, prepare_only=False,
                   client=None, should_stop=lambda: False, progress=print):
    output = Path(output_dir).expanduser()
    require(not output.is_symlink(), "Annotation output cannot be symlinked")
    output = output.resolve()
    if not resume:
        _prepare(selection_run, output, config)
    require((output / "run.json").is_file(), "No prepared independent MedGemma annotation found")
    with directory_lock(output):
        plan = _read(output / "run.json")
        require(plan.get("schema_version") == PROTOCOL_VERSION and plan.get("runtime") == "llama.cpp",
                "Expected an independent MedGemma annotation run")
        require(_read(output / "session.json")["run_sha256"] == sha256_file(output / "run.json"), "Frozen plan changed")
        require(plan["protocol_sha256"] == _protocol_hash(), "Annotation protocol changed; start a new run")
        if selection_run is not None:
            require(str(Path(selection_run).expanduser().resolve()) == plan["selection_run"], "Resume selection differs")
        saved_config = AnnotationConfig(**plan["config"])
        saved_config.validate()
        if config is not None:
            effective = replace(config, procedure_context=config.procedure_context or saved_config.procedure_context)
            require(effective == saved_config, "Resume settings differ")
        config = saved_config
        for name, digest in plan["input_sha256"].items():
            require(sha256_file(_relative(output, name)) == digest, f"Frozen annotation input changed: {name}")
        _, source, selected, _ = validate_completed_selection(output / "selection", Path(plan["selection_run"]), snapshot=True)
        require(source == _read(output / "source.json") and selected == _read(output / "selected-frames.json")
                and len(source["frames"]) == plan["source_frame_count"]
                and [row["frame_id"] for row in selected] == plan["frame_ids"], "Frozen target set changed")
        require(len(plan["evidence_files"]) == len(selected), "Evidence inventory changed")
        published = _read(output / "annotations.json") if (output / "annotations.json").exists() else _document([])
        previous = published.get("annotations", [])
        require(published == _document(previous), "Published annotation metadata changed")
        previous_by_id = {row["target_frame_id"]: row for row in previous}
        require([row["target_frame_id"] for row in previous] == plan["frame_ids"][:len(previous)], "Published target order changed")
        if prepare_only:
            require(not (output / "calls").exists(), "Prepare-only cannot reset a started annotation")
            return _publish(output, plan, [], "prepared")
        (output / ".pause-requested").unlink(missing_ok=True)
        rows = []
        if not previous:
            _publish(output, plan, [], "running")
        def get_client():
            nonlocal client
            if client is None:
                client = LlamaCppClient(timeout=1800)
            return client
        try:
            for index, (target, name) in enumerate(zip(selected, plan["evidence_files"])):
                packet = _read(_relative(output, name))
                require(packet["target_frame_id"] == target["frame_id"], "Target evidence changed")
                progress(f"MedGemma annotation {index + 1}/{len(selected)}: {target['frame_id']}")
                result = _call(output, index, packet, config, get_client,
                    lambda: target["frame_id"] not in previous_by_id and not should_stop()
                    and not (output / ".pause-requested").exists())
                if result is None:
                    require(target["frame_id"] not in previous_by_id, "Published annotation lacks an accepted response")
                    return _publish(output, plan, rows, "paused")
                if target["frame_id"] in previous_by_id:
                    require(result == previous_by_id[target["frame_id"]], "Published annotation differs from accepted response")
                rows.append(result)
                if len(rows) >= len(previous):
                    _publish(output, plan, rows, "running")
            status = "completed_with_unresolved_questions" if any(r["annotation"]["unresolved_questions"] for r in rows) else "completed"
            (output / "last-error.json").unlink(missing_ok=True)
            return _publish(output, plan, rows, status)
        except BaseException as exc:
            detail = {"type": type(exc).__name__, "message": str(exc), "at": _now()}
            atomic_json(output / "last-error.json", detail, overwrite=True)
            if len(rows) >= len(previous):
                _publish(output, plan, rows, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", detail)
            raise


def annotation_status(output_dir):
    output = Path(output_dir).expanduser().resolve()
    require(_read(output / "run.json").get("schema_version") == PROTOCOL_VERSION, "Expected independent MedGemma annotation")
    return {**_read(output / "summary.json"), "writer_active": directory_is_locked(output)}


def request_annotation_pause(output_dir):
    output = Path(output_dir).expanduser().resolve()
    annotation_status(output)
    atomic_bytes(output / ".pause-requested", b"pause\n", overwrite=True)
    return {"pause_requested": True, "message": "The current frame will finish and save before pausing."}


def add_annotation_parser(subparsers):
    parser = subparsers.add_parser("annotate-selected-frames", help="Independent MedGemma surgical annotation from selected source frames")
    parser.add_argument("--selection-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    for name, default in asdict(AnnotationConfig()).items():
        if name == "detail_crops":
            parser.add_argument("--no-detail-crops", dest=name, action="store_false", default=None)
        else:
            parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=None)
    for name in ("medgemma-annotation-status", "pause-medgemma-annotation"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)


def annotation_cli(args, *, should_stop=lambda: False):
    overrides = {key: getattr(args, key) for key in asdict(AnnotationConfig()) if getattr(args, key) is not None}
    if args.resume:
        saved = _read(args.output_dir.expanduser().resolve() / "run.json")["config"]
        require(all(saved[key] == value for key, value in overrides.items()), "Resume settings differ")
        config = None
    else:
        config = AnnotationConfig(**overrides)
    result = run_annotation(args.selection_run, args.output_dir, config, resume=args.resume,
        prepare_only=args.prepare_only, client=LlamaCppClient(args.project_root, timeout=args.timeout),
        should_stop=should_stop, progress=lambda message: print(message, flush=True))
    print(json.dumps(result, indent=2))
