"""Review each authorized surgery's selected frames together after Qwen finishes."""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import fcntl
import json
from pathlib import Path
import time

from .annotation_integrity import verify_annotation_output
from .checkpoint import atomic_json
from .contract import require, sha256_file
from .medgemma_surgery_review import SurgeryReviewConfig, _protocol_hash, run_review
from .ollama import OllamaClient
from .smart_selection import _write


CASES = ("S2A2", "S1A2")
PROTOCOL = "sospine-first-two-medgemma-surgery-v1"


def _read(path):
    from .llama_video import _strict_json
    return _strict_json(Path(path).read_bytes())


def _active(path):
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def _save(output, state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write(output / "state.json", state)
    return state


def _case_evidence(directory, status, allow_rejected, verifier):
    if status in {"completed", "context_conflict"}:
        manifest = verifier(directory)
        return {"kind": "sealed_annotation", "annotation_count": manifest["annotation_count"],
                "manifest": manifest}
    require(status == "failed" and allow_rejected,
            "Qwen annotation needs attention; MedGemma has not started")
    from .qwen_draft_intake import load_rejected_annotation
    _, _, annotations, audit = load_rejected_annotation(directory)
    require(audit["status"] == audit["validation_status"] == "rejected_temporal_citations"
            and audit["annotation_count"] == len(annotations["annotations"]) > 0
            and audit["artifact_sha256"] and audit["issues"],
            "Failed Qwen annotation lacks a temporal-citation rejection audit")
    return {"kind": "rejected_temporal_citations", "annotation_count": audit["annotation_count"],
            "audit": audit}


def _pin_evidence(output, state, evidence):
    path = output / "qwen-evidence.json"
    pinned = state.get("qwen_evidence_sha256")
    if pinned is not None:
        require(path.is_file() and sha256_file(path) == pinned, "Frozen Qwen intake evidence bytes changed")
    if path.exists():
        require(_read(path) == evidence, "Qwen intake evidence changed since the pair was started")
    else:
        atomic_json(path, evidence)
    state["qwen_evidence_sha256"] = sha256_file(path)
    _save(output, state)


def run_pair(annotation_pair, output_dir, *, resume=False, prepare_only=False,
             allow_rejected_temporal_citations=None,
             should_stop=lambda: False, progress=print, poll_seconds=15,
             review_runner=None, integrity_verifier=None):
    output = Path(output_dir).expanduser().resolve()
    if not resume:
        parent = Path(annotation_pair).expanduser().resolve()
        upstream = _read(parent / "run.json")
        require(tuple(job["case_id"] for job in upstream["cases"]) == CASES,
                "MedGemma may review only the authorized S2A2 and S1A2 cases")
        require(not output.exists() and not output.is_relative_to(parent), "Use a new separate MedGemma output directory")
        require(allow_rejected_temporal_citations is None or type(allow_rejected_temporal_citations) is bool,
                "The rejected temporal-citation override must be boolean")
        config = SurgeryReviewConfig(allow_rejected_temporal_citations=bool(allow_rejected_temporal_citations))
        config.validate()
        output.mkdir(parents=True)
        plan = {"schema_version": PROTOCOL, "annotation_pair": str(parent),
                "annotation_pair_sha256": sha256_file(parent / "run.json"), "config": asdict(config),
                "review_protocol_sha256": _protocol_hash(), "request_timeout_seconds": 21600,
                "cases": list(CASES)}
        _write(output / "run.json", plan)
        _save(output, {"status": "prepared", "active_case": None, "plan_sha256": sha256_file(output / "run.json"),
                       "jobs": [{"case_id": case, "status": "pending"} for case in CASES]})
    with (output / ".pair.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, state = _read(output / "run.json"), _read(output / "state.json")
        require(plan["schema_version"] == PROTOCOL and state["plan_sha256"] == sha256_file(output / "run.json"),
                "Frozen MedGemma pair configuration changed")
        require(plan["review_protocol_sha256"] == _protocol_hash(), "MedGemma review prompt changed")
        require(tuple(plan["cases"]) == CASES and tuple(job["case_id"] for job in state["jobs"]) == CASES,
                "MedGemma pair scope changed")
        parent = Path(plan["annotation_pair"])
        require(sha256_file(parent / "run.json") == plan["annotation_pair_sha256"], "Parent annotation plan changed")
        config = SurgeryReviewConfig(**plan["config"])
        config.validate()
        if allow_rejected_temporal_citations is not None:
            require(type(allow_rejected_temporal_citations) is bool
                    and allow_rejected_temporal_citations == config.allow_rejected_temporal_citations,
                    "Rejected temporal-citation override differs from the frozen pair configuration")
        if prepare_only:
            return state
        pause = lambda: should_stop() or (output / ".pause-requested").exists()
        state.update(status="waiting_for_annotation", active_case=None, error=None)
        _save(output, state)
        try:
            deadline = time.monotonic() + 48 * 3600
            while True:
                if pause():
                    state["status"] = "paused"
                    return _save(output, state)
                upstream = _read(parent / "state.json")
                require(tuple(job["case_id"] for job in upstream["jobs"]) == CASES, "Parent annotation scope changed")
                terminal = {"completed", "context_conflict"}
                if config.allow_rejected_temporal_citations:
                    terminal.add("failed")
                else:
                    require(upstream["status"] != "failed" and not any(job["status"] == "failed" for job in upstream["jobs"]),
                            "Qwen annotation needs attention; MedGemma has not started")
                if all(job["status"] in terminal for job in upstream["jobs"]) \
                        and not _active(parent / ".pair.lock"):
                    break
                require(time.monotonic() < deadline, "Timed out waiting for Qwen annotations")
                time.sleep(poll_seconds)
            verify, runner = integrity_verifier or verify_annotation_output, review_runner or run_review
            # Both sources must pass their respective intake contracts before any model request.
            statuses = {job["case_id"]: job["status"] for job in upstream["jobs"]}
            evidence_for = lambda case: _case_evidence(parent / case, statuses[case],
                config.allow_rejected_temporal_citations, verify)
            evidence = {case: evidence_for(case) for case in CASES}
            _pin_evidence(output, state, evidence)
            for job in state["jobs"]:
                if pause():
                    state.update(status="paused", active_case=None)
                    return _save(output, state)
                case = job["case_id"]
                directory = output / case
                if job["status"] in {"completed", "completed_with_deferred_evidence"}:
                    require(sha256_file(directory / "reviews.json") == job["reviews_sha256"],
                            "Saved MedGemma review bundle changed")
                require(not _active(parent / ".pair.lock"), "Qwen annotation worker became active before MedGemma review")
                require(evidence_for(case) == evidence[case], "Qwen annotation changed before MedGemma review")
                state.update(status="running", active_case=case)
                job.update(status="running", error=None)
                _save(output, state)
                progress(f"MedGemma surgery review {case}: all {evidence[case]['annotation_count']} selected frames in one request.")
                try:
                    can_resume = (directory / "run.json").exists()
                    if directory.exists() and not can_resume:
                        archive = output / "interrupted-preparations"
                        archive.mkdir(exist_ok=True)
                        directory.rename(archive / f"{case}-{time.time_ns()}")
                    summary = runner(parent / case, directory, config, resume=can_resume,
                        client=OllamaClient(timeout=plan["request_timeout_seconds"]), should_stop=pause, progress=progress)
                    if summary["status"] == "paused":
                        job["status"] = "pending"
                        state.update(status="paused", active_case=None)
                        return _save(output, state)
                    require(summary["status"] in {"completed", "completed_with_deferred_evidence"},
                            "MedGemma review did not complete")
                    require(summary["reviewed_frame_count"] == summary["selected_frame_count"] == evidence[case]["annotation_count"],
                            "MedGemma did not review every annotated key frame")
                    require(evidence_for(case) == evidence[case], "Qwen annotation changed during MedGemma review")
                    job.update(status=summary["status"], reviewed_frames=summary["reviewed_frame_count"],
                               deferred_frames=summary["deferred_frame_count"], report_path=str(directory / "report.html"),
                               reviews_sha256=sha256_file(directory / "reviews.json"))
                except Exception as exc:
                    job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    progress(f"MedGemma {case} failed: {job['error']}")
                _save(output, state)
            state.update(status="completed" if all(job["status"] == "completed" for job in state["jobs"])
                         else "completed_with_issues", active_case=None)
            return _save(output, state)
        except BaseException as exc:
            state.update(status="paused" if isinstance(exc, KeyboardInterrupt) else "failed",
                         active_case=None, error=f"{type(exc).__name__}: {exc}")
            _save(output, state)
            raise


def main():
    from .__main__ import cooperative_stop
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-pair", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-rejected-temporal-citations", action="store_true", default=None,
                        help="Accept only audited Qwen temporal-citation rejections; freeze this override in a new plan")
    args = parser.parse_args()
    with cooperative_stop() as should_stop:
        state = run_pair(args.annotation_pair, args.output_dir, resume=args.resume, prepare_only=args.prepare_only,
                        allow_rejected_temporal_citations=args.allow_rejected_temporal_citations,
                        should_stop=should_stop, progress=lambda message: print(message, flush=True))
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
