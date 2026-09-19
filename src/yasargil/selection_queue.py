"""Read-only SOSpine scheduling by released technical outcomes.

The leak endpoint describes a cadaveric repair experiment, not patient recovery.
Outcome metadata orders jobs only; it must never become a model prompt or a
frame-selection input. Playback time always comes from available images at the
explicit reconstruction rate, independently of the recorded repair time.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import re

from .contract import canonical_hash
from .video_source import _release_files, file_sha256


SCHEMA_VERSION = "sospine-selection-queue-v1"
OUTCOMES = "sospine_outcomes.csv"
_CASE_ID = re.compile(r"(?:S[0-9]+A[0-9]+|Clip[0-9]+)")


class SelectionQueueError(ValueError):
    """The available release cannot support the requested frozen job order."""


def _natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part
                 for part in re.split(r"([0-9]+)", value))


def _outcomes(path: Path) -> tuple[dict, dict]:
    if not path.is_file():
        raise SelectionQueueError(f"Missing SOSpine outcomes table: {path}")
    before = file_sha256(path)
    rows = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"Trial ID", "Leak At 40mmHg", "Time for repair"}
        if not required.issubset(reader.fieldnames or []):
            raise SelectionQueueError("SOSpine outcome table lacks required columns")
        for raw in reader:
            case_id = (raw.get("Trial ID") or "").strip()
            if not _CASE_ID.fullmatch(case_id):
                raise SelectionQueueError(f"Invalid SOSpine trial ID at CSV line {reader.line_num}")
            if case_id in rows:
                raise SelectionQueueError(f"Duplicate outcome row for {case_id}")
            leak = (raw.get("Leak At 40mmHg") or "").strip()
            if leak not in {"N", "Y", ""}:
                raise SelectionQueueError(f"Unknown leak outcome for {case_id}: {leak!r}")
            duration_text = (raw.get("Time for repair") or "").strip()
            try:
                repair_seconds = float(duration_text) if duration_text else None
            except ValueError as error:
                raise SelectionQueueError(f"Invalid recorded repair time for {case_id}") from error
            if repair_seconds is not None and (not math.isfinite(repair_seconds) or repair_seconds < 0):
                raise SelectionQueueError(f"Invalid recorded repair time for {case_id}")
            rows[case_id] = {
                "leak_result": leak or None,
                "outcome_rank_group": {"N": "better", "Y": "worse", "": "unknown"}[leak],
                "repair_time_seconds": repair_seconds,
                "metadata_locator": {
                    "path": str(path), "sha256": before,
                    "locator_type": "csv_row_1based_including_header",
                    "line": reader.line_num, "trial_id": case_id,
                },
            }
    if file_sha256(path) != before:
        raise SelectionQueueError("Outcome table changed while building the queue")
    return rows, {"path": str(path), "sha256": before}


def build_selection_queue(dataset_root, *, tie_break: str = "trial_id") -> dict:
    """Plan every available sequence once, alternating better and worse trials.

    ``trial_id`` uses natural trial-ID order within the two binary leak groups.
    ``repair_time`` uses shorter repairs first among no-leak trials and longer
    repairs first among leak trials, with trial ID resolving remaining ties.
    Missing repair times come after measured times in either group. This second
    option is an explicit scheduling preference, not a measured recovery rank.

    Inventories pin ordered filenames and byte sizes without reading every image
    twice. The source-preparation stage remains responsible for hashing image
    contents before inference. This function performs no filesystem writes.
    """
    if tie_break not in {"trial_id", "repair_time"}:
        raise SelectionQueueError("tie_break must be 'trial_id' or 'repair_time'")
    root = Path(dataset_root).expanduser().resolve()
    frame_root = root / "frames"
    if not frame_root.is_dir():
        raise SelectionQueueError(f"Missing released frame directory: {frame_root}")
    outcomes, metadata = _outcomes(root / OUTCOMES)
    available = {}
    for directory in sorted(frame_root.iterdir(), key=lambda item: _natural_key(item.name)):
        if directory.name.startswith(".") or not directory.is_dir():
            continue
        case_id = directory.name
        if not _CASE_ID.fullmatch(case_id):
            raise SelectionQueueError(f"Unrecognized released sequence directory: {directory}")
        numbered = _release_files(directory)
        inventory = [{"frame_index": index, "filename": path.name,
                      "size_bytes": path.stat().st_size} for index, path in numbered]
        if any(item["size_bytes"] <= 0 for item in inventory):
            raise SelectionQueueError(f"Empty released frame in {case_id}")
        details = outcomes.get(case_id, {
            "leak_result": None, "outcome_rank_group": "unknown",
            "repair_time_seconds": None, "metadata_locator": None,
        })
        available[case_id] = {
            "case_id": case_id,
            "input_path": str(directory.resolve()),
            "frame_count": len(inventory),
            "duration_ms": len(inventory) * 1000,
            **details,
            "frame_inventory": inventory,
            "inventory_sha256": canonical_hash(inventory),
        }
    if not available:
        raise SelectionQueueError("No available SOSpine image sequences")

    def order_key(job: dict) -> tuple:
        natural = _natural_key(job["case_id"])
        if tie_break == "trial_id" or job["outcome_rank_group"] == "unknown":
            return natural
        duration = job["repair_time_seconds"]
        signed_duration = (duration if job["outcome_rank_group"] == "better" else -duration) \
            if duration is not None else 0
        return (duration is None, signed_duration, natural)

    groups = {group: sorted((job for job in available.values() if job["outcome_rank_group"] == group),
                            key=order_key) for group in ("better", "worse", "unknown")}
    jobs = []
    for index in range(max(len(groups["better"]), len(groups["worse"]))):
        for group in ("better", "worse"):
            if index < len(groups[group]):
                jobs.append(groups[group][index])
    jobs.extend(groups["unknown"])
    jobs = [{"position": index, **job} for index, job in enumerate(jobs, start=1)]
    excluded = [{"case_id": case_id, "reason": "released_sequence_not_available",
                 "expected_input_path": str(frame_root / case_id), **outcomes[case_id]}
                for case_id in sorted(set(outcomes) - set(available), key=_natural_key)]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(root),
        "reconstructed_fps": 1,
        "timing_basis": "reconstructed_nominal",
        "duration_rule": "actual_frame_count / reconstructed_fps; recorded repair time is not a timeline input",
        "metadata": metadata,
        "ordering": {
            "endpoint": "Leak At 40mmHg", "scope": "simulated_technical_outcome_not_patient_recovery",
            "better": "N", "worse": "Y", "unknown": None, "tie_break": tie_break,
            "description": "Alternate better and worse groups, then remaining known outcomes, then unknown outcomes.",
            "outcome_metadata_used_for_model_input": False,
        },
        "inventory_validation": "ordered_filenames_and_byte_sizes; source preparation verifies image content hashes",
        "dataset_inventory_sha256": canonical_hash([
            {"case_id": job["case_id"], "input_path": job["input_path"],
             "inventory_sha256": job["inventory_sha256"]}
            for job in sorted(jobs, key=lambda job: _natural_key(job["case_id"]))]),
        "job_count": len(jobs),
        "outcome_group_counts": {group: len(items) for group, items in groups.items()},
        "jobs": jobs,
        "excluded": excluded,
    }


def verify_selection_queue(queue: dict) -> None:
    """Reject changed metadata, inventories, or order before using a saved plan.

    Same-size content changes require the source manifest's stronger image hashes;
    this check deliberately does not claim to verify image bytes.
    """
    try:
        if queue["schema_version"] != SCHEMA_VERSION:
            raise SelectionQueueError("Unsupported selection queue schema")
        current = build_selection_queue(queue["dataset_root"], tie_break=queue["ordering"]["tie_break"])
    except (KeyError, TypeError) as error:
        raise SelectionQueueError("Malformed frozen selection queue") from error
    if canonical_hash(current) != canonical_hash(queue):
        raise SelectionQueueError("Frozen selection queue differs from current metadata, inventory, or job order")
