"""Review every selected frame from one surgery in one fresh MedGemma request.

The raw joint response is saved once. Validated per-frame records share that
receipt, while retaining their original Qwen drafts and canonical provenance.
"""
from __future__ import annotations

import argparse
import base64
import copy
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
from pathlib import Path

from .checkpoint import atomic_bytes, atomic_json, directory_lock, durable_mkdir
from .contract import ContractError, canonical_hash, require, sha256_file
from .llama_video import _strict_json
from .medgemma_review import _identity, _load_annotation, _now, _read, _relative, _save_equal
from .medgemma_surgery_contract import (
    SURGERY_SYSTEM, build_surgery_evidence, build_surgery_reviews,
    surgery_schema, target_evidence,
)
from .llama_cpp import MEDGEMMA_MODEL, LlamaCppClient, LlamaCppError, _object, build_chat_request, encode_request
from .smart_selection import verify_assets


PROTOCOL_VERSION = "medgemma-surgery-review-v1"


@dataclass(frozen=True)
class SurgeryReviewConfig:
    medgemma_model: str = MEDGEMMA_MODEL
    num_ctx: int = 65536
    num_predict: int = 16384
    seed: int = 42
    allow_rejected_temporal_citations: bool = False

    def validate(self):
        require(isinstance(self.medgemma_model, str) and self.medgemma_model.strip()
                and self.medgemma_model == self.medgemma_model.strip(), "An explicit MedGemma model is required")
        require(type(self.num_ctx) is int and 8192 <= self.num_ctx <= 131072, "Invalid MedGemma context budget")
        require(type(self.num_predict) is int and 256 <= self.num_predict < self.num_ctx,
                "Invalid joint review answer budget")
        require(type(self.seed) is int and self.seed >= 0, "Invalid seed")
        require(type(self.allow_rejected_temporal_citations) is bool, "Invalid rejected-draft review setting")


def _protocol_hash():
    return canonical_hash({"version": PROTOCOL_VERSION, "runtime": "llama.cpp", "transport": "openai-chat-v1", "system": SURGERY_SYSTEM,
                           "schema": surgery_schema(["first", "second"], 1000),
                           "evidence_policy": "all_final_selected_images_once_no_neighbor_expansion",
                           "prompt_layout": "shared_metadata_and_drafts_with_explicit_temporal_audit_v2"})


def _qwen_input(parent, config):
    if config.allow_rejected_temporal_citations and _read(parent / "summary.json").get("status") == "failed":
        from .qwen_draft_intake import load_rejected_annotation
        return load_rejected_annotation(parent)
    plan, source, annotations = _load_annotation(parent)
    return plan, source, annotations, None


def _prepare(annotation_run, output, config, dataset_root):
    require(annotation_run is not None, "--annotation-run is required for a new surgery review")
    parent = Path(annotation_run).expanduser().resolve()
    require(parent.is_dir() and not output.exists() and not output.is_relative_to(parent),
            "Use a new review directory outside the finished Qwen annotation run")
    config = config or SurgeryReviewConfig()
    config.validate()
    dataset_root = Path(dataset_root).expanduser().resolve() if dataset_root is not None else None
    with (parent / ".annotation.lock").open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("Qwen annotation is still active") from exc
        qwen_plan, source, annotations, audit = _qwen_input(parent, config)
        source_root = Path(source["source_path"]).resolve()
        if source["source_kind"] == "original_video":
            source_root = source_root.parent
        require(not output.is_relative_to(source_root)
                and (dataset_root is None or not output.is_relative_to(dataset_root)),
                "Review output must be outside the original dataset")
        verify_assets(source)
        batch = build_surgery_evidence(source, annotations, dataset_root=dataset_root,
            procedure_context=qwen_plan["config"].get("procedure_context", ""), rejected_draft_audit=audit)
        for table in batch["dataset_context"]["tables"]:
            require(not output.is_relative_to(Path(table["source_path"]).parent),
                    "Review output must be outside the inferred source dataset")
        durable_mkdir(output)
        names = set(audit["artifact_sha256"]) if audit is not None else set(qwen_plan["input_sha256"]) | {
            "run.json", "summary.json", "session.json", "selected-frames.json", "annotations.json",
            "round-00/request.json", "round-00/response.json", "round-00/result.json", "round-00/verification.json"}
        if audit is None and (parent / "integrity.json").is_file():
            from .annotation_integrity import verify_annotation_output
            seal = verify_annotation_output(parent)
            names.update(seal["artifacts"])
            names.add("integrity.json")
        hashes = {}
        for name in sorted(names):
            original = _relative(parent, name)
            content = original.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if audit is not None:
                require(digest == audit["artifact_sha256"][name], "Rejected Qwen evidence changed before freezing")
            destination = _relative(output / "qwen", name)
            durable_mkdir(destination.parent)
            atomic_bytes(destination, content)
            require(sha256_file(original) == digest, "Qwen evidence changed while freezing the review")
            hashes["qwen/" + name] = digest
        if audit is not None:
            for name, document in (("draft-intake.json", audit), ("qwen-drafts.json", annotations)):
                atomic_json(output / name, document)
                hashes[name] = sha256_file(output / name)
        atomic_json(output / "surgery-evidence.json", batch)
        hashes["surgery-evidence.json"] = sha256_file(output / "surgery-evidence.json")
        durable_mkdir(output / "evidence")
        evidence_files = []
        for index, frame_id in enumerate(batch["target_frame_ids"]):
            name = f"evidence/frame-{index:04d}.json"
            atomic_json(output / name, target_evidence(batch, frame_id))
            hashes[name] = sha256_file(output / name)
            evidence_files.append(name)
        plan = {"schema_version": PROTOCOL_VERSION, "runtime": "llama.cpp", "created_at": _now(), "review_unit": "surgery",
                "annotation_run": str(parent), "dataset_root": str(dataset_root) if dataset_root else None,
                "config": asdict(config), "protocol_sha256": _protocol_hash(), "input_sha256": hashes,
                "frame_ids": batch["target_frame_ids"], "evidence_files": evidence_files,
                "surgery_evidence_file": "surgery-evidence.json", "automated_followup": False,
                "qwen_input_status": "rejected_temporal_citations" if audit is not None else "completed",
                "human_review_required": True, "training_eligible": False}
        atomic_json(output / "run.json", plan)
        atomic_json(output / "session.json", {"run_sha256": sha256_file(output / "run.json")})
    return plan


def _prompt_context(batch):
    """Share metadata once; full original rows and locators remain in frozen evidence."""
    supplied = set(batch["target_frame_ids"])
    drafts = []
    for original in batch["qwen_annotations"]:
        draft = {key: copy.deepcopy(original[key]) for key in
                 ("frame_id", "visible_observation", "visibility", "contextual_claims", "uncertainties")}
        for claim in draft["contextual_claims"]:
            for interval in claim["evidence_intervals"]:
                supporting = interval.pop("supporting_frames", [])
                interval["supplied_supporting_frame_ids"] = [f["frame_id"] for f in supporting
                                                              if f["frame_id"] in supplied]
                interval["omitted_supporting_frame_count"] = sum(f["frame_id"] not in supplied for f in supporting)
        drafts.append(draft)
    dataset = copy.deepcopy(batch["dataset_context"])
    # The frame/table associations and every raw coordinate/label value stay
    # intact; lengthy filesystem locators are retained in surgery-evidence.json.
    dataset["original_annotations"] = [
        {key: row[key] for key in ("annotation_id", "frame_id", "original_kind", "original_origin", "raw_value")}
        for row in dataset["original_annotations"]]
    context = {"task": "Review every target frame and its Qwen draft together; return one review per target ID in reviews.",
            "review_unit": "surgery", "target_frame_ids": list(batch["target_frame_ids"]),
            "media_timeline": copy.deepcopy(batch["media_timeline"]),
            "qwen_context_check": batch["qwen_context_check"], "dataset_context": dataset,
            "qwen_annotations": drafts, "limitations": copy.deepcopy(batch.get("limitations", [])),
            "evidence_note": "All supplied images are final selected key frames. Other Qwen-cited source images are not supplied; their absence is not proof a claim is false."}
    if "qwen_validation" in batch:
        context["qwen_validation"] = copy.deepcopy(batch["qwen_validation"])
    return context


def build_request(batch, config):
    config.validate()
    require([frame["frame_id"] for frame in batch["frames"]] == batch["target_frame_ids"],
            "Joint review image order or target coverage changed")
    messages = [{"role": "system", "content": SURGERY_SYSTEM}]
    for frame in batch["frames"]:
        raw = Path(frame["image_path"]).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == frame["image_sha256"]
                and sha256_file(frame["source_path"]) == frame["source_sha256"],
                "A selected MedGemma source image changed")
        locator = {key: frame.get(key) for key in ("frame_id", "frame_index", "release_frame_index",
            "timestamp_ms", "timestamp_basis", "width", "height", "source_sha256", "image_sha256")}
        messages.append({"role": "user", "content": "Selected key-frame image: " + json.dumps(locator),
                         "images": [base64.b64encode(raw).decode("ascii")]})
    messages.append({"role": "user", "content": json.dumps(_prompt_context(batch), ensure_ascii=False)})
    return build_chat_request(config.medgemma_model, messages,
        surgery_schema(batch["target_frame_ids"], batch["media_timeline"]["duration_ms"]),
        num_ctx=config.num_ctx, num_predict=config.num_predict, seed=config.seed)


def _parse(raw, batch, config):
    envelope = _object(raw, "MedGemma surgery review")
    LlamaCppClient._validate_chat(envelope, config.medgemma_model)
    message = envelope["choices"][0]["message"]
    require(not message.get("tool_calls"), "MedGemma cannot dispatch tools")
    count = envelope.get("usage", {}).get("prompt_tokens")
    require(type(count) is int and 0 < count <= config.num_ctx - config.num_predict,
            "MedGemma prompt usage is unavailable or leaves insufficient answer capacity")
    try:
        content = _strict_json(message["content"])
    except (ValueError, UnicodeError) as exc:
        raise ContractError(f"Invalid joint MedGemma response JSON: {exc}") from exc
    return {"schema_version": PROTOCOL_VERSION, "reviews": build_surgery_reviews(content, batch)}


def _call(output, batch, config, client_factory, *, allow_inference):
    acceptance_path = output / "accepted-response.json"
    if acceptance_path.exists():
        acceptance = _read(acceptance_path)
        recorded = _relative(output, acceptance["receipt_path"])
        require(recorded.parent.parent == output / "calls/surgery" and recorded.name == "receipt.json"
                and sha256_file(recorded) == acceptance["receipt_sha256"],
                "Accepted surgery receipt bytes changed")
    request = build_request(batch, config)
    request_bytes = encode_request(request)
    call_dir = output / "calls/surgery"
    attempts = sorted(call_dir.glob("attempt-*"))
    accepted = []
    for attempt in attempts:
        require(attempt.is_dir() and not attempt.is_symlink(), "Invalid surgery review attempt")
        receipt = _read(attempt / "receipt.json") if (attempt / "receipt.json").exists() else None
        if receipt:
            require(set(receipt["files"]) == {"request.json", "response.json", "model.json", "review.json"},
                    "Surgery review receipt has incomplete evidence")
            for name, digest in receipt["files"].items():
                require(sha256_file(attempt / name) == digest, f"Accepted surgery review artifact changed: {name}")
        if (attempt / "request.json").exists():
            require((attempt / "request.json").read_bytes() == request_bytes, "Saved surgery review request changed")
        if not (attempt / "response.json").exists():
            require(not receipt, "Accepted surgery response is missing")
            continue
        require((attempt / "request.json").exists() and (attempt / "model.json").exists(),
                "Saved surgery response lacks its request or model identity")
        require(_identity(_read(attempt / "model.json")) == _identity(_read(output / "model-info.json")),
                "Saved MedGemma identity differs from the pinned model")
        try:
            result = _parse((attempt / "response.json").read_bytes(), batch, config)
        except (ContractError, LlamaCppError, ValueError):
            require(not receipt, "Accepted joint MedGemma response is no longer valid")
            continue
        accepted.append((attempt, result))
    require(len(accepted) <= 1, "Multiple accepted MedGemma responses for one surgery")
    if accepted:
        attempt, result = accepted[0]
    else:
        if not allow_inference():
            return None
        client = client_factory()
        metadata = client.model_info(config.medgemma_model)
        require(metadata.get("name") == config.medgemma_model and "vision" in metadata.get("capabilities", []),
                "Surgery review requires the configured vision model")
        pin = output / "model-info.json"
        if pin.exists():
            require(_identity(_read(pin)) == _identity(metadata), "Pinned MedGemma model or runtime changed")
        else:
            atomic_json(pin, metadata)
        if not allow_inference():
            return None
        attempt = call_dir / f"attempt-{len(attempts) + 1:04d}"
        durable_mkdir(attempt)
        atomic_bytes(attempt / "request.json", request_bytes)
        atomic_json(attempt / "model.json", metadata)
        client.last_response_bytes = None
        try:
            raw = client.chat_raw(request)
            atomic_bytes(attempt / "response.json", raw)
            result = _parse(raw, batch, config)
        except BaseException as exc:
            raw = getattr(client, "last_response_bytes", None)
            if raw is not None and not (attempt / "response.json").exists():
                atomic_bytes(attempt / "response.json", raw)
            atomic_json(attempt / "failure.json", {"type": type(exc).__name__, "message": str(exc), "at": _now()})
            raise
    _save_equal(attempt / "review.json", result)
    _save_equal(attempt / "receipt.json", {"files": {name: sha256_file(attempt / name)
                for name in ("request.json", "response.json", "model.json", "review.json")}})
    _save_equal(acceptance_path, {"receipt_path": str((attempt / "receipt.json").relative_to(output)),
                               "receipt_sha256": sha256_file(attempt / "receipt.json")})
    return [{**review, "call_directory": str(attempt.relative_to(output)), "review_unit": "surgery"}
            for review in result["reviews"]]


def _review_document(reviews):
    return {"schema_version": PROTOCOL_VERSION, "review_unit": "surgery", "reviews": reviews,
            "human_review_required": True, "training_eligible": False}


def _publish(output, plan, reviews, status, error=None):
    from .medgemma_review_report import write_review_report
    atomic_json(output / "reviews.json", _review_document(reviews), overwrite=True)
    deferred = [{"target_frame_id": row["target_frame_id"], "status": "deferred_not_dispatched",
                 "evidence_requests": row["deferred_evidence_requests"],
                 "response_path": row["call_directory"] + "/response.json",
                 "review_path": row["call_directory"] + "/review.json"}
                for row in reviews if row["deferred_evidence_requests"]]
    atomic_json(output / "deferred-evidence.json", {"automated_followup": False, "requests": deferred}, overwrite=True)
    summary = {"schema_version": PROTOCOL_VERSION, "created_at": plan["created_at"], "updated_at": _now(),
               "review_unit": "surgery", "status": status, "selected_frame_count": len(plan["frame_ids"]),
               "reviewed_frame_count": len(reviews), "deferred_frame_count": len(deferred),
               "accepted_surgery_response_count": int(bool(reviews)), "config": plan["config"], "error": error,
               "qwen_input_status": plan.get("qwen_input_status", "completed"),
               "automated_followup": False, "human_review_required": True, "training_eligible": False}
    atomic_json(output / "summary.json", summary, overwrite=True)
    write_review_report(output)
    return summary


def run_review(annotation_run, output_dir, config=None, *, dataset_root=None, resume=False,
               prepare_only=False, client=None, should_stop=lambda: False, progress=print):
    output = Path(output_dir).expanduser()
    require(not output.is_symlink(), "Surgery review output cannot be a symlink")
    output = output.resolve()
    if not resume:
        _prepare(annotation_run, output, config, dataset_root)
    require((output / "run.json").is_file(), "No prepared MedGemma surgery review found")
    with directory_lock(output):
        plan = _read(output / "run.json")
        reviews, validated = [], False
        published = _read(output / "reviews.json") if (output / "reviews.json").exists() else None
        previous_summary = _read(output / "summary.json") if (output / "summary.json").exists() else {}
        preserve_publication = bool(published and published.get("reviews")) or previous_summary.get("status") in {
            "completed", "completed_with_deferred_evidence"}
        try:
            require(plan.get("runtime") == "llama.cpp",
                    "This review predates the llama.cpp migration; start a new review directory")
            require(_read(output / "session.json")["run_sha256"] == sha256_file(output / "run.json"),
                    "Frozen surgery review plan changed")
            require(plan["schema_version"] == PROTOCOL_VERSION and plan["protocol_sha256"] == _protocol_hash(),
                    "Surgery review protocol changed; start a new run")
            if annotation_run is not None:
                require(str(Path(annotation_run).expanduser().resolve()) == plan["annotation_run"], "Resume annotation differs")
            if dataset_root is not None:
                require(str(Path(dataset_root).expanduser().resolve()) == plan["dataset_root"], "Resume dataset differs")
            if config is not None:
                require(asdict(config) == plan["config"], "Resume surgery settings differ")
            config = SurgeryReviewConfig(**plan["config"])
            config.validate()
            for name, digest in plan["input_sha256"].items():
                require(sha256_file(_relative(output, name)) == digest, f"Frozen surgery evidence changed: {name}")
            _, source, annotations, audit = _qwen_input(output / "qwen", config)
            batch = _read(output / plan["surgery_evidence_file"])
            if audit is not None:
                require(plan.get("qwen_input_status") == "rejected_temporal_citations"
                        and _read(output / "draft-intake.json") == audit
                        and _read(output / "qwen-drafts.json") == annotations,
                        "Rejected Qwen draft intake differs from its frozen raw evidence")
                require(batch.get("qwen_validation", {}).get("issues") == audit["issues"]
                        and batch["qwen_annotations"] == annotations["annotations"],
                        "Rejected Qwen prompt drafts or validation warnings changed")
            else:
                require(plan.get("qwen_input_status", "completed") == "completed",
                        "Rejected Qwen draft provenance cannot be replaced with a completion claim")
            require(batch["target_frame_ids"] == plan["frame_ids"], "Surgery review target set changed")
            require(len(plan["evidence_files"]) == len(plan["frame_ids"]), "Per-frame evidence inventory changed")
            for frame_id, name in zip(plan["frame_ids"], plan["evidence_files"]):
                require(_read(output / name) == target_evidence(batch, frame_id), "Per-frame evidence projection changed")
            if prepare_only:
                require(not (output / "calls").exists(), "Prepare-only cannot reset a started surgery review")
                return _publish(output, plan, [], "prepared")
            validated = True
            (output / ".pause-requested").unlink(missing_ok=True)
            def get_client():
                nonlocal client
                if client is None:
                    client = LlamaCppClient(timeout=21600)
                return client
            if not preserve_publication:
                _publish(output, plan, [], "running")
            progress(f"MedGemma surgery review: all {len(batch['target_frame_ids'])} selected images and Qwen drafts in one request.")
            result = _call(output, batch, config, get_client,
                allow_inference=lambda: not preserve_publication and not should_stop()
                and not (output / ".pause-requested").exists())
            if result is None:
                require(not preserve_publication, "Published surgery reviews lack their accepted response")
                return _publish(output, plan, [], "paused")
            reviews = result
            require([row["target_frame_id"] for row in reviews] == plan["frame_ids"], "Surgery review coverage changed")
            if preserve_publication:
                expected = (json.dumps(_review_document(reviews), ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n").encode("utf-8")
                require(published == _review_document(reviews)
                        and (output / "reviews.json").read_bytes() == expected,
                        "Published surgery reviews differ from accepted response")
            status = "completed_with_deferred_evidence" if any(r["deferred_evidence_requests"] for r in reviews) else "completed"
            (output / "last-error.json").unlink(missing_ok=True)
            return _publish(output, plan, reviews, status)
        except BaseException as exc:
            detail = {"type": type(exc).__name__, "message": str(exc), "at": _now()}
            atomic_json(output / "last-error.json", detail, overwrite=True)
            if validated:
                status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                if preserve_publication:
                    # Retain published evidence byte-for-byte when validation
                    # fails; record the failure without replacing it with [].
                    atomic_json(output / "summary.json", {**previous_summary, "status": status,
                                "updated_at": _now(), "error": detail}, overwrite=True)
                else:
                    _publish(output, plan, reviews, status, detail)
            raise


def main():
    from .__main__ import cooperative_stop
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--timeout", type=float, default=21600)
    for name, default in asdict(SurgeryReviewConfig()).items():
        if isinstance(default, bool):
            parser.add_argument("--" + name.replace("_", "-"), action="store_true", default=None)
        else:
            parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=None)
    args = parser.parse_args()
    overrides = {name: getattr(args, name) for name in asdict(SurgeryReviewConfig()) if getattr(args, name) is not None}
    if args.resume:
        saved = _read(args.output_dir.expanduser().resolve() / "run.json")["config"]
        require(all(saved[name] == value for name, value in overrides.items()), "Resume settings differ")
        config = None
    else:
        config = SurgeryReviewConfig(**overrides)
    with cooperative_stop() as should_stop:
        summary = run_review(args.annotation_run, args.output_dir, config, dataset_root=args.dataset_root,
            resume=args.resume, prepare_only=args.prepare_only, should_stop=should_stop,
            client=LlamaCppClient(args.project_root, timeout=args.timeout), progress=lambda message: print(message, flush=True))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
