"""Independently annotate the two authorized SOSpine selections with MedGemma."""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

from .checkpoint import atomic_bytes, atomic_json
from .contract import require, sha256_file
from .llama_cpp import LlamaCppClient
from .medgemma_annotation import AnnotationConfig, _protocol_hash, _read, run_annotation
from .selection_batch import selection_batch_status

CASES = ("S2A2", "S1A2")
PROTOCOL = "sospine-first-two-medgemma-annotations-v1"


def _save(output, state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output / "state.json", state, overwrite=True)
    return state


def run_pair(selection_batch, output_dir, *, resume=False, prepare_only=False,
             project_root=None, should_stop=lambda: False, progress=print,
             poll_seconds=15, annotation_runner=None):
    output = Path(output_dir).expanduser()
    require(not output.is_symlink(), "Pair output cannot be symlinked")
    output = output.resolve()
    if not resume:
        require(selection_batch is not None, "--selection-batch is required for a new pair")
        batch = Path(selection_batch).expanduser().resolve()
        queue = _read(batch / "queue.json")
        require(tuple(job["case_id"] for job in queue["jobs"][:2]) == CASES,
                "Expected the authorized S2A2 and S1A2 selections")
        require(not output.exists() and not output.is_relative_to(batch)
                and not output.is_relative_to(Path(queue["dataset_root"]).resolve()),
                "Use a new pair directory outside selection and source data")
        config = AnnotationConfig()
        config.validate()
        output.mkdir(parents=True)
        plan = {"schema_version": PROTOCOL, "runtime": "llama.cpp", "selection_batch": str(batch),
                "queue_sha256": sha256_file(batch / "queue.json"), "config": asdict(config),
                "annotation_protocol_sha256": _protocol_hash(), "request_timeout_seconds": 1800,
                "cases": [{"case_id": job["case_id"], "expected_video_frames": job["frame_count"],
                    "selection_run": str(batch / "runs" / f"{job['position']:02d}-{job['case_id']}")}
                    for job in queue["jobs"][:2]]}
        plan_bytes = (json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        _save(output, {"status": "prepared", "active_case": None, "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                       "jobs": [{"case_id": case, "status": "pending"} for case in CASES]})
        atomic_bytes(output / "run.json", plan_bytes)
    with (output / ".pair.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, state = _read(output / "run.json"), _read(output / "state.json")
        require(plan.get("schema_version") == PROTOCOL and plan.get("runtime") == "llama.cpp",
                "Historical Qwen-review pairs cannot resume; prepare an independent annotation pair")
        require(state["plan_sha256"] == sha256_file(output / "run.json")
                and plan["annotation_protocol_sha256"] == _protocol_hash(), "Frozen pair configuration changed")
        require(tuple(row["case_id"] for row in plan["cases"]) == CASES
                and tuple(row["case_id"] for row in state["jobs"]) == CASES, "Pair scope changed")
        batch = Path(plan["selection_batch"])
        if selection_batch is not None:
            require(Path(selection_batch).expanduser().resolve() == batch, "Resume selection batch differs")
        require(sha256_file(batch / "queue.json") == plan["queue_sha256"], "Selection queue changed")
        config = AnnotationConfig(**plan["config"])
        config.validate()
        if prepare_only:
            return state
        (output / ".pause-requested").unlink(missing_ok=True)
        paused = lambda: should_stop() or (output / ".pause-requested").exists()
        try:
            state.update(status="waiting_for_selection", active_case=None, error=None)
            _save(output, state)
            deadline = time.monotonic() + 24 * 3600
            while True:
                if paused():
                    state["status"] = "paused"
                    return _save(output, state)
                upstream = selection_batch_status(batch)
                require(tuple(job["case_id"] for job in upstream["jobs"][:2]) == CASES, "Selection case identities changed")
                require(all(job["status"] == "pending" for job in upstream["jobs"][2:]),
                        "A sequence outside the authorized pair has been attempted")
                controller = batch / "first-two-controller.json"
                if controller.exists():
                    require(_read(controller)["status"] not in {"failed", "needs_attention"}, "Selection controller needs attention")
                require(not any(job["status"] in {"failed", "needs_review"} for job in upstream["jobs"][:2]),
                        "MedGemma requires completed, ready selections")
                if all(job["status"] == "completed" for job in upstream["jobs"][:2]) and not upstream["writer_active"]:
                    break
                require(time.monotonic() < deadline, "Timed out waiting for selection")
                time.sleep(poll_seconds)
            runner = annotation_runner or run_annotation
            for source, job in zip(plan["cases"], state["jobs"]):
                if paused():
                    state.update(status="paused", active_case=None)
                    return _save(output, state)
                directory = output / job["case_id"]
                state.update(status="running", active_case=job["case_id"])
                job.update(status="running", error=None)
                _save(output, state)
                try:
                    can_resume = (directory / "run.json").is_file()
                    if directory.exists() and not can_resume:
                        archive = output / "interrupted-preparations"
                        archive.mkdir(exist_ok=True)
                        directory.rename(archive / f"{job['case_id']}-{time.time_ns()}")
                    result = runner(source["selection_run"], directory, config, resume=can_resume,
                        client=LlamaCppClient(project_root, timeout=plan["request_timeout_seconds"]),
                        should_stop=paused, progress=progress)
                    if result["status"] == "paused":
                        job["status"] = "pending"
                        state.update(status="paused", active_case=None)
                        return _save(output, state)
                    require(result["status"] in {"completed", "completed_with_unresolved_questions"}
                            and result["annotated_frame_count"] == result["selected_frame_count"] > 0,
                            "MedGemma did not annotate every selected frame")
                    require(result["source_frame_count"] == source["expected_video_frames"], "Selection source coverage changed")
                    job.update(status=result["status"], annotation_count=result["annotated_frame_count"],
                               unresolved_frame_count=result["unresolved_frame_count"],
                               report_path=str(directory / "report.html"),
                               annotations_sha256=sha256_file(directory / "annotations.json"))
                except Exception as exc:
                    job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    progress(f"MedGemma {job['case_id']} failed: {job['error']}")
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
    parser.add_argument("--selection-batch", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    with cooperative_stop() as should_stop:
        state = run_pair(args.selection_batch, args.output_dir, resume=args.resume, prepare_only=args.prepare_only,
                         project_root=args.project_root, should_stop=should_stop,
                         progress=lambda message: print(message, flush=True))
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
