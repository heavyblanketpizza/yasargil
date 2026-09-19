"""Wait for the two authorized SOSpine selections, then annotate and seal them."""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import fcntl
import json
from pathlib import Path
import time

from .contract import require, sha256_file
from .frame_annotation import AnnotationConfig, _protocol_hash, run_annotation
from .selection_batch import PROCEDURE_CONTEXT, selection_batch_status
from .smart_selection import _write


CASES = ("S2A2", "S1A2")
PROTOCOL = "sospine-first-two-annotations-v1"


def _read(path):
    from .llama_video import _strict_json
    return _strict_json(Path(path).read_bytes())


def _save(output, state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write(output / "state.json", state)
    return state


def run_pair(selection_batch, output_dir, *, resume=False, prepare_only=False,
             should_stop=lambda: False, progress=print, poll_seconds=15,
             annotation_runner=None, integrity_sealer=None):
    from .annotation_integrity import MANIFEST_NAME, seal_annotation_output
    output = Path(output_dir).expanduser().resolve()
    if not resume:
        batch = Path(selection_batch).expanduser().resolve()
        queue = _read(batch / "queue.json")
        require(tuple(job["case_id"] for job in queue["jobs"][:2]) == CASES,
                "Expected the authorized S2A2 and S1A2 selections")
        require(not output.is_relative_to(Path(queue["dataset_root"]).resolve()),
                "Annotation output must be outside source media")
        require(not output.exists(), "Annotation pair exists; use --resume")
        config = AnnotationConfig(context_size=262144, image_max_tokens=256,
            max_tokens=12288, request_timeout_seconds=21600, procedure_context=PROCEDURE_CONTEXT)
        config.validate()
        output.mkdir(parents=True)
        plan = {"schema_version": PROTOCOL, "selection_batch": str(batch),
                "queue_sha256": sha256_file(batch / "queue.json"), "config": asdict(config),
                "annotation_protocol_sha256": _protocol_hash(),
                "cases": [{"case_id": job["case_id"], "expected_video_frames": job["frame_count"],
                    "selection_run": str(batch / "runs" / f"{job['position']:02d}-{job['case_id']}")}
                    for job in queue["jobs"][:2]]}
        _write(output / "run.json", plan)
        _save(output, {"status": "prepared", "active_case": None,
            "plan_sha256": sha256_file(output / "run.json"),
            "jobs": [{"case_id": case, "status": "pending"} for case in CASES]})
    with (output / ".pair.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, state = _read(output / "run.json"), _read(output / "state.json")
        require(plan["schema_version"] == PROTOCOL and state["plan_sha256"] == sha256_file(output / "run.json"),
                "Saved annotation pair settings changed")
        require(plan["annotation_protocol_sha256"] == _protocol_hash(), "Annotation prompt changed")
        require(tuple(job["case_id"] for job in plan["cases"]) == CASES
                and tuple(job["case_id"] for job in state["jobs"]) == CASES, "Annotation scope changed")
        batch = Path(plan["selection_batch"])
        require(sha256_file(batch / "queue.json") == plan["queue_sha256"], "Selection queue changed")
        config = AnnotationConfig(**plan["config"])
        config.validate()
        if prepare_only:
            return state
        pause = lambda: should_stop() or (output / ".pause-requested").exists()
        state.update(status="waiting_for_selection", active_case=None, error=None)
        _save(output, state)
        try:
            deadline = time.monotonic() + 24 * 3600
            while True:
                if pause():
                    state.update(status="paused")
                    return _save(output, state)
                status = selection_batch_status(batch)
                require(tuple(job["case_id"] for job in status["jobs"][:2]) == CASES,
                        "Selection case identities changed")
                require(all(job["status"] == "pending" for job in status["jobs"][2:]),
                        "A sequence outside the authorized pair has been attempted")
                controller = batch / "first-two-controller.json"
                if controller.exists():
                    require(_read(controller)["status"] not in {"failed", "needs_attention"},
                            "Selection controller needs attention; annotation has not started")
                first = status["jobs"][:2]
                require(not any(job["status"] in {"failed", "needs_review"} for job in first),
                        "Annotation requires both selections to finish with verified, ready frame sets")
                if all(job["status"] == "completed" for job in first) and not status["writer_active"]:
                    break
                require(time.monotonic() < deadline, "Timed out waiting for the two selections")
                time.sleep(poll_seconds)
            runner, seal = annotation_runner or run_annotation, integrity_sealer or seal_annotation_output
            for source, job in zip(plan["cases"], state["jobs"]):
                directory = output / job["case_id"]
                if job["status"] in {"completed", "context_conflict"}:
                    seal(directory)
                    continue
                if pause():
                    state.update(status="paused", active_case=None)
                    return _save(output, state)
                state.update(status="running", active_case=job["case_id"])
                job.update(status="running", error=None)
                _save(output, state)
                progress(f"Annotating {job['case_id']}: all {source['expected_video_frames']} video frames and every final selected still.")
                try:
                    saved_run = directory / "run.json"
                    can_resume = saved_run.exists()
                    if can_resume:
                        saved = _read(saved_run)
                        require(saved["config"] == asdict(config)
                                and saved["selection_run"] == source["selection_run"], "Saved annotation settings changed")
                    elif directory.exists():
                        archive = output / "interrupted-preparations"
                        archive.mkdir(exist_ok=True)
                        directory.rename(archive / f"{job['case_id']}-{time.time_ns()}")
                    summary = runner(source["selection_run"], directory, config, resume=can_resume,
                                     should_stop=pause, progress=progress)
                    if summary["status"] == "paused":
                        job["status"] = "pending"
                        state.update(status="paused", active_case=None)
                        return _save(output, state)
                    require(summary["status"] in {"completed", "context_conflict"}, "Annotation did not finish")
                    require(summary["source"]["expected_video_frames"] == source["expected_video_frames"],
                            "Annotation source frame coverage changed")
                    seal(directory)
                    job.update(status=summary["status"], annotation_count=summary["selected_frame_count"],
                               report_path=str(directory / "report.html"), annotations_path=str(directory / "annotations.json"),
                               integrity_path=str(directory / MANIFEST_NAME))
                except Exception as exc:
                    job.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    progress(f"Annotation {job['case_id']} failed: {job['error']}")
                _save(output, state)
            state.update(status="completed" if all(j["status"] == "completed" for j in state["jobs"])
                         else "completed_with_issues", active_case=None)
            return _save(output, state)
        except BaseException as exc:
            state.update(status="paused" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error=f"{type(exc).__name__}: {exc}", active_case=None)
            _save(output, state)
            raise


def main():
    from .__main__ import cooperative_stop
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-batch", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    with cooperative_stop() as should_stop:
        state = run_pair(args.selection_batch, args.output_dir, resume=args.resume,
            prepare_only=args.prepare_only, should_stop=should_stop,
            progress=lambda message: print(message, flush=True))
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
