"""Run complete SOSpine sequences serially through frozen keep/drop review."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import html
import json
from pathlib import Path
import time

from .contract import ContractError, require, sha256_file
from .selection_queue import build_selection_queue, verify_selection_queue
from .smart_selection import REVIEW_POLICY_SHA256, TIMING_POLICY_SHA256, SelectionConfig, _write, run_selection
from .video_source import _release_files


PROTOCOL = "sospine-keep-drop-batch-v1"
PROCEDURE_CONTEXT = (
    "SOSpine: simulated durotomy repair in minimally invasive spine surgery on a "
    "perfusion-based cadaveric model. This context comes from dataset documentation; "
    "no outcome or per-frame annotations are supplied."
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    from .llama_video import _strict_json
    return _strict_json(Path(path).read_bytes())


def _default_config():
    return SelectionConfig(context_size=262144, image_max_tokens=256,
                           request_timeout_seconds=21600, procedure_context=PROCEDURE_CONTEXT)


def _active(output):
    path = output / ".batch.lock"
    if not path.exists():
        return False
    with path.open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def _save(output, plan, state):
    state["updated_at"] = _now()
    _write(output / "state.json", state)
    counts = {name: sum(job["status"] == name for job in state["jobs"])
              for name in ("pending", "running", "completed", "needs_review", "failed")}
    summary = {"schema_version": PROTOCOL, "status": state["status"],
               "created_at": plan["created_at"], "updated_at": state["updated_at"],
               "active_case": state.get("active_case"), "total_cases": len(state["jobs"]),
               "counts": counts, "jobs": state["jobs"],
               "ordering": _read(output / "queue.json")["ordering"],
               "report_path": str(output / "report.html"), "error": state.get("error")}
    _write(output / "summary.json", summary)
    _report(output, summary)
    return summary


def _report(output, summary):
    esc = lambda value: html.escape(str(value), quote=True)
    queue = _read(output / "queue.json")
    rows = []
    for source, job in zip(queue["jobs"], summary["jobs"]):
        duration = source["duration_ms"] / 1000
        link = f"runs/{job['directory_name']}/selection.html"
        exists = (output / link).is_file()
        label = f'<a href="{esc(link)}">{esc(job["case_id"])}</a>' if exists else esc(job["case_id"])
        outcome = {"N": "No leak", "Y": "Leak"}.get(source["leak_result"], "Unavailable")
        rows.append(f'<tr><td>{source["position"]}</td><td>{label}</td><td>{outcome}</td>'
                    f'<td>{source["frame_count"]}</td><td>{int(duration // 60)}:{int(duration % 60):02d}</td>'
                    f'<td>{esc(job["status"])}</td><td>{esc(job.get("kept_frames", "—"))}</td>'
                    f'<td>{esc(job.get("error") or "")}</td></tr>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>SOSpine keep/drop batch</title><style>
body{{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#17232b}}
table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #dbe3e9}}
th{{background:#edf4f7}}a{{color:#12617c}}.note{{padding:16px;background:#f3f6f8;border-radius:8px}}
</style><h1>SOSpine keep/drop batch</h1>
<p>Status: <b>{esc(summary['status'])}</b> · Active: {esc(summary.get('active_case') or 'None')}
 · Updated: {esc(summary['updated_at'])}</p>
<p class="note">Each case supplies every available released frame in a native 1-fps video,
plus the sampled stills. Qwen only keeps or drops those stills; it cannot add candidates.
Protected sampling anchors retain coverage. Timestamps locate frames in this reconstruction.
Leak-test outcomes order this queue only and are not supplied to Qwen. These are simulated
repair outcomes, not patient recovery. Outcome ties use {esc(queue['ordering']['tie_break'])}.</p>
<p>Completed: {summary['counts']['completed']} / {summary['total_cases']} ·
Needs review: {summary['counts']['needs_review']} · Failed: {summary['counts']['failed']}</p>
<p><a href="queue.json">Frozen queue and provenance</a> · <a href="summary.json">Status JSON</a>
 · <a href="run.json">Model settings</a> · <a href="driver.log">Worker log</a></p>
<table><thead><tr><th>Order</th><th>Sequence</th><th>Recorded outcome</th><th>Frames</th>
<th>Playback</th><th>Status</th><th>Kept</th><th>Error</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></html>'''
    (output / "report.html").write_text(page, encoding="utf-8")


def run_selection_batch(dataset_root, output_dir, config=None, *, tie_break="trial_id",
                        resume=False, prepare_only=False, should_stop=lambda: False,
                        progress=print, selection_runner=None):
    """Freeze the outcome-ordered queue, then checkpoint every complete-video case.

    A failure is recorded before continuing to the next case. Explicit resume
    retries failed cases and recovers an interrupted accepted selection. Finished
    cases are never rerun. Outcomes stay in the scheduler's queue, outside Qwen.
    """
    output = Path(output_dir).expanduser().resolve()
    if not resume:
        require(dataset_root is not None, "--dataset-root is required for a new batch")
        root = Path(dataset_root).expanduser().resolve()
        require(not output.is_relative_to(root), "Batch output must be outside source media")
        require(not output.exists(), "Batch output exists; use --resume for the saved queue")
        config = config or _default_config()
        config.validate()
        require(config.review_mode == "keep_drop" and config.max_retrieval_rounds == 0,
                "Batch review must only keep/drop the sampled candidates")
        # This trusted procedure description deliberately excludes per-case outcomes.
        require(config.procedure_context == PROCEDURE_CONTEXT,
                "Batch source context must exclude per-case outcome metadata")
        queue = build_selection_queue(root, tie_break=tie_break)
        output.mkdir(parents=True)
        _write(output / "queue.json", queue)
        plan = {"schema_version": PROTOCOL, "created_at": _now(), "dataset_root": str(root),
                "config": asdict(config), "queue_sha256": sha256_file(output / "queue.json"),
                "review_policy_sha256": REVIEW_POLICY_SHA256, "timing_policy_sha256": TIMING_POLICY_SHA256}
        _write(output / "run.json", plan)
        state = {"status": "prepared", "active_case": None,
                 "plan_sha256": sha256_file(output / "run.json"), "jobs": [
                     {"case_id": job["case_id"], "position": job["position"], "status": "pending",
                      "directory_name": f"{job['position']:02d}-{job['case_id']}"}
                     for job in queue["jobs"]]}
        _save(output, plan, state)
    require((output / "run.json").is_file(), "No prepared selection batch found")
    with (output / ".batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("This selection batch is already running") from exc
        plan, state = _read(output / "run.json"), _read(output / "state.json")
        require(plan["schema_version"] == PROTOCOL
                and state["plan_sha256"] == sha256_file(output / "run.json"), "Batch settings changed")
        require(plan["queue_sha256"] == sha256_file(output / "queue.json"), "Frozen selection queue changed")
        require(plan.get("review_policy_sha256") == REVIEW_POLICY_SHA256
                and plan.get("timing_policy_sha256") == TIMING_POLICY_SHA256,
                "Batch prompt policy changed; begin a separate batch")
        if dataset_root is not None:
            require(str(Path(dataset_root).expanduser().resolve()) == plan["dataset_root"], "Batch source changed")
        queue = _read(output / "queue.json")
        verify_selection_queue(queue)
        config = SelectionConfig(**plan["config"])
        config.validate()
        require(config.review_mode == "keep_drop" and config.max_retrieval_rounds == 0
                and config.procedure_context == PROCEDURE_CONTEXT, "Batch must preserve outcome-blind keep/drop review")
        require(len(state["jobs"]) == len(queue["jobs"]) and all(
            saved["case_id"] == frozen["case_id"] and saved["position"] == frozen["position"]
            and saved["directory_name"] == f"{frozen['position']:02d}-{frozen['case_id']}"
            for saved, frozen in zip(state["jobs"], queue["jobs"])), "Batch job order changed")
        require(all(job["status"] in {"pending", "running", "completed", "needs_review", "failed"}
                    for job in state["jobs"]), "Unknown batch job status")
        if prepare_only:
            return _save(output, plan, state)
        (output / ".pause-requested").unlink(missing_ok=True)
        runner = selection_runner or run_selection
        state.update(status="running", error=None)
        (output / "runs").mkdir(exist_ok=True)
        for source, job in zip(queue["jobs"], state["jobs"]):
            directory = output / "runs" / job["directory_name"]
            if job["status"] in {"completed", "needs_review"}:
                require((directory / "selection.json").is_file()
                        and sha256_file(directory / "selection.json") == job["selection_sha256"],
                        f"Saved selection changed for {job['case_id']}")
                continue
            if should_stop() or (output / ".pause-requested").exists():
                state.update(status="paused", active_case=None)
                return _save(output, plan, state)
            state["active_case"] = job["case_id"]
            job.update(status="running", started_at=_now(), error=None)
            _save(output, plan, state)
            progress(f"Case {job['position']}/{len(queue['jobs'])}: {job['case_id']}; "
                     f"all {source['frame_count']} released frames at 1 fps, "
                     f"{config.candidate_budget} candidate stills; keep/drop only.")
            try:
                inventory = [{"frame_index": index, "filename": path.name, "size_bytes": path.stat().st_size}
                             for index, path in _release_files(Path(source["input_path"]))]
                require(inventory == source["frame_inventory"], "Source inventory changed before case preparation")
                can_resume = (directory / "run.json").is_file()
                if can_resume:
                    existing = _read(directory / "run.json")
                    require(existing["config"] == asdict(config), "Saved case settings differ from frozen batch")
                elif directory.exists():
                    archive = output / "interrupted-preparations"
                    archive.mkdir(exist_ok=True)
                    directory.rename(archive / f"{job['directory_name']}-{time.time_ns()}")
                result = runner(source["input_path"], directory, config,
                                released_fps=1, resume=can_resume, progress=progress)
                require(result == _read(directory / "selection.json"),
                        "Returned selection differs from its saved artifact")
                require(result.get("expected_video_frames") == source["frame_count"], "Selection source coverage changed")
                require(result.get("completed_rounds_full_video_verified") is True,
                        "Selection has not verified the entire released video")
                require(not result.get("unresolved_searches"), "Keep/drop review returned additional-frame requests")
                require(len(result["frames"]) == min(config.candidate_budget, source["frame_count"]),
                        "Review changed the sampled candidate count")
                job.update(status="completed" if result["status"] == "completed" else "needs_review",
                           selection_status=result["status"], finished_at=_now(),
                           kept_frames=len(result["selected_frame_ids"]), candidate_frames=len(result["frames"]),
                           selection_sha256=sha256_file(directory / "selection.json"))
            except KeyboardInterrupt:
                job.update(status="pending", error="Interrupted; resume recovers accepted results or retries this case.")
                state.update(status="paused", active_case=None)
                _save(output, plan, state)
                raise
            except Exception as exc:
                job.update(status="failed", finished_at=_now(), error=f"{type(exc).__name__}: {exc}")
                progress(f"Case {job['case_id']} failed: {job['error']}")
            _save(output, plan, state)
        state.update(status="completed" if all(job["status"] == "completed" for job in state["jobs"])
                     else "completed_with_issues", active_case=None)
        return _save(output, plan, state)


def selection_batch_status(output_dir):
    output = Path(output_dir).expanduser().resolve()
    result = _read(output / "summary.json")
    result.update(writer_active=_active(output), pause_requested=(output / ".pause-requested").exists())
    if result.get("active_case"):
        job = next(job for job in result["jobs"] if job["case_id"] == result["active_case"])
        result["active_output_dir"] = str(output / "runs" / job["directory_name"])
    return result


def pause_selection_batch(output_dir):
    output = Path(output_dir).expanduser().resolve()
    require((output / "run.json").is_file(), "No selection batch found")
    (output / ".pause-requested").touch()
    return selection_batch_status(output)


def add_selection_batch_parser(subparsers):
    parser = subparsers.add_parser("select-sospine-batch", help="Review every full released sequence in outcome-alternating order")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tie-break", choices=["trial_id", "repair_time"], default="trial_id")
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--context-size", type=int, default=262144)
    parser.add_argument("--image-max-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout-seconds", type=float, default=21600)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    for name in ("selection-batch-status", "pause-selection-batch"):
        item = subparsers.add_parser(name)
        item.add_argument("--output-dir", type=Path, required=True)


def selection_batch_cli(args, *, should_stop=lambda: False):
    config = None if args.resume else SelectionConfig(
        candidate_budget=args.candidates, max_candidates=max(48, args.candidates),
        context_size=args.context_size, image_max_tokens=args.image_max_tokens,
        max_tokens=args.max_tokens, request_timeout_seconds=args.request_timeout_seconds,
        procedure_context=PROCEDURE_CONTEXT)
    result = run_selection_batch(args.dataset_root, args.output_dir, config,
                                 tie_break=args.tie_break, resume=args.resume, prepare_only=args.prepare_only,
                                 should_stop=should_stop, progress=lambda message: print(message, flush=True))
    print(json.dumps({key: result[key] for key in ("status", "counts", "active_case", "report_path")}, indent=2))
