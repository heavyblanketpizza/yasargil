"""Durable, sequential enhancement of complete released SOSpine sequences.

Windows are independent evidence records. A batch coordinates them without
injecting earlier windows, outcomes, or a global narrative into model requests.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .checkpoint import atomic_json, directory_lock, durable_mkdir
from .contract import require, sha256_file
from .enhancement import EnhancementConfig, enhance_sospine, enhancement_protocol_fingerprint
from .llama_cpp import LlamaCppClient, MEDGEMMA_MODEL, QWEN_MODEL


@dataclass(frozen=True)
class BatchConfig:
    case_ids: tuple[str, ...]
    window_size: int = 12
    initial_frames: int = 4
    search_frames: int = 4
    max_frames: int = 8
    max_rounds: int = 1
    qwen_model: str = QWEN_MODEL
    medgemma_model: str = MEDGEMMA_MODEL
    num_ctx: int = 65536
    num_predict: int = 4096
    seed: int = 42

    def window_config(self, case_id, start, cutoff):
        settings = asdict(self)
        del settings["case_ids"], settings["window_size"]
        return EnhancementConfig(case_id, start, cutoff, **settings)

    def check(self):
        require(bool(self.case_ids) and len(set(self.case_ids)) == len(self.case_ids),
                "Choose at least one case, without duplicates")
        require(1 <= self.window_size <= 3600, "Window size must be 1–3600 released indices")
        for case in self.case_ids:
            self.window_config(case, 1, self.window_size).check()


def _json(path):
    require(path.is_file() and not path.is_symlink(), f"Missing or unsafe job file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Job file must contain a JSON object: {path}")
    return value


def plan_batch(dataset_root, config):
    """Inventory actual filenames; no writes, source decoding, or model requests."""
    config.check()
    root = Path(dataset_root).resolve()
    windows, counts = [], {}
    for case in config.case_ids:
        folder = root / "frames" / case
        require(folder.is_dir() and folder.resolve().is_relative_to(root), "Missing or escaped source sequence")
        pattern = re.compile(re.escape(case) + r"_frame_(\d{8})\.jpeg")
        indices = []
        for path in folder.iterdir():
            match = pattern.fullmatch(path.name)
            if match:
                require(path.is_file() and path.resolve().is_relative_to(root), "Source frame escapes dataset root")
                indices.append(int(match[1]))
        indices.sort()
        require(indices and indices == list(range(1, indices[-1] + 1)),
                f"{case}: full-sequence batches require contiguous released indices starting at 1")
        counts[case] = len(indices)
        for start in range(1, indices[-1] + 1, config.window_size):
            cutoff = min(start + config.window_size - 1, indices[-1])
            windows.append({"window_id": f"{case}-{start:08d}-{cutoff:08d}",
                            "case_id": case, "start_index": start, "cutoff_index": cutoff})
    return {"batch_version": 1, "dataset_root": str(root),
            "config": json.loads(json.dumps(asdict(config))), "frame_counts": counts,
            "total_frames": sum(counts.values()), "total_windows": len(windows), "windows": windows,
            "transport": "llama_cpp_ordered_images", "native_video_processor": False,
            "cross_window_context": False, "overlap_indices": 0,
            "minimum_model_calls": len(windows) * 3,
            "maximum_model_calls": len(windows) * (3 + 2 * config.max_rounds),
            "training_eligible": False}


def _source_hashes(root, plan):
    paths = ["metadata/source_manifest.json", "documentation/readme.txt",
             "sospine_tool_tips.csv", "sospine_bbox.csv", "sospine_outcomes.csv"]
    for case, count in plan["frame_counts"].items():
        paths.extend(f"frames/{case}/{case}_frame_{i:08d}.jpeg" for i in range(1, count + 1))
    result = {}
    for relative in paths:
        path = root / relative
        require(path.is_file() and path.resolve().is_relative_to(root), f"Missing or escaped source file: {relative}")
        result[relative] = sha256_file(path)
    return result


def _identity(info):
    return {key: info[key] for key in ("name", "digest", "quantization", "runtime_version",
                                      "runtime", "model_file", "projector_file", "runtime_binary")}


class _PinnedClient:
    def __init__(self, client, identities):
        self.client, self.identities = client, identities

    def model_info(self, name):
        info = self.client.model_info(name)
        require(_identity(info) == self.identities[name],
                f"Model or llama.cpp runtime changed during this batch: {name}; restore the pinned version")
        return info

    def chat_raw(self, request):
        return self.client.chat_raw(request)

    @property
    def last_response_bytes(self):
        return getattr(self.client, "last_response_bytes", None)


def enhance_batch(dataset_root, output_dir, config, *, client=None, progress=None,
                  resume=False, pause_requested=None):
    """Checkpoint after each model call/window; explicit resume keeps prior records."""
    plan = plan_batch(dataset_root, config)
    root = Path(dataset_root).resolve()
    destination = Path(output_dir)
    require(not destination.is_symlink() and not destination.resolve().is_relative_to(root),
            "Output must be outside the source dataset and cannot be a symlink")
    if resume:
        require(destination.is_dir(), "Cannot resume a missing batch directory")
    else:
        require(not destination.exists(), "Output directory already exists; explicitly resume it")
        # Complete preflight hashing before leaving a new directory behind.
        source_hashes = _source_hashes(root, plan)
        durable_mkdir(destination)
    destination = destination.resolve()
    with directory_lock(destination):
        durable_mkdir(destination)
        manifest_path = destination / "batch.json"
        if resume:
            saved = _json(manifest_path)
            require(saved.get("plan", {}).get("transport") == "llama_cpp_ordered_images",
                    "Resume rejected: legacy Ollama jobs are read-only; start a new llama.cpp batch")
            require(saved["plan"] == plan, "Batch settings or source sequence inventory changed; restore the original inputs")
            require(saved["source_hashes"] == _source_hashes(root, plan),
                    "Batch source bytes changed; restore the original source snapshot")
            require(saved["protocol"] == enhancement_protocol_fingerprint(),
                    "Batch prompt/schema protocol changed; restore the original version")
        else:
            saved = {"plan": plan, "source_hashes": source_hashes,
                     "protocol": enhancement_protocol_fingerprint(),
                     "created_at": datetime.now(timezone.utc).isoformat()}
            atomic_json(manifest_path, saved)
        pause_path = destination / "PAUSE"
        if resume and pause_path.exists():
            require(not pause_path.is_symlink(), "Unsafe pause marker")
            pause_path.unlink()
        should_pause = lambda: pause_path.exists() or bool(pause_requested and pause_requested())
        completed = []
        state = {"status": "running", "total_windows": len(plan["windows"]),
                 "completed_windows": 0, "current_window": None, "last_progress": "Checking saved work",
                 "pid": os.getpid(), "training_eligible": False}

        def report(message=None, **updates):
            state.update(updates)
            if message is not None:
                state["last_progress"] = message
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(destination / "progress.json", state, overwrite=True)
            if message and progress:
                progress(message)

        def retain_results():
            atomic_json(destination / "results.json", {"status": state["status"], "windows": completed,
                        "training_eligible": False}, overwrite=True)

        try:
            report("Verifying source snapshot and model identities")
            client = client or LlamaCppClient()
            models_path = destination / "models.json"
            metadata = {name: client.model_info(name) for name in (config.qwen_model, config.medgemma_model)}
            identities = {name: _identity(info) for name, info in metadata.items()}
            require(identities[config.qwen_model]["digest"] != identities[config.medgemma_model]["digest"],
                    "Discovery and review resolve to the same model weights")
            if models_path.exists():
                require(_json(models_path)["identities"] == identities,
                        "Batch model weights or runtime changed; restore the pinned versions")
            else:
                atomic_json(models_path, {"identities": identities, "metadata": metadata})
            pinned = _PinnedClient(client, identities)
            for window in plan["windows"]:
                window_id = window["window_id"]
                dest = destination / "windows" / window_id
                # Completed windows are replay-validated too, never trusted from progress.json alone.
                if should_pause():
                    report("Paused; saved calls and completed windows are retained", status="paused", current_window=window_id)
                    retain_results()
                    return state
                # Pin future windows to the batch snapshot, including files changed
                # while an earlier window was running. Window sessions then guard
                # the same bytes at every individual call and before finalization.
                shared = {"metadata/source_manifest.json", "documentation/readme.txt",
                          "sospine_tool_tips.csv", "sospine_bbox.csv", "sospine_outcomes.csv"}
                shared.update(f"frames/{window['case_id']}/{window['case_id']}_frame_{i:08d}.jpeg"
                              for i in range(window["start_index"], window["cutoff_index"] + 1))
                for relative in shared:
                    path = root / relative
                    require(path.is_file() and path.resolve().is_relative_to(root)
                            and sha256_file(path) == saved["source_hashes"][relative],
                            f"Batch source bytes changed before window {window_id}: {relative}")
                report(f"Window {len(completed) + 1}/{len(plan['windows'])}: {window_id}", current_window=window_id)
                durable_mkdir(destination / "windows")
                result = enhance_sospine(root, dest,
                    config.window_config(window["case_id"], window["start_index"], window["cutoff_index"]),
                    client=pinned, progress=report, resume=dest.exists(), pause_requested=should_pause)
                if result["status"] == "paused":
                    report("Paused at a model-call checkpoint", status="paused")
                    retain_results()
                    return state
                require(result["status"] == "completed", "Unexpected window completion status")
                completed.append({**window, "archive": f"windows/{window_id}/archive.json", "completion": result})
                report(f"Saved {len(completed)}/{len(plan['windows'])} windows", completed_windows=len(completed))
                retain_results()
            report("All windows completed; every generated record still needs human review",
                   status="completed", current_window=None)
            retain_results()
            return state
        except (Exception, KeyboardInterrupt) as exc:
            report(str(exc) or "Forced interruption; resume will reuse completed calls",
                   status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                   error_type=type(exc).__name__)
            retain_results()
            raise


def request_pause(output_dir):
    """Request a cooperative stop without taking the long-running writer lock."""
    destination = Path(output_dir)
    require(destination.is_dir() and not destination.is_symlink(), "Missing or unsafe enhancement directory")
    require((destination / "batch.json").is_file() or (destination / "session.json").is_file(),
            "Directory is not a resumable enhancement job")
    atomic_json(destination / "PAUSE", {"requested_at": datetime.now(timezone.utc).isoformat()}, overwrite=True)
    return {"pause_requested": True, "output_dir": str(destination.resolve()),
            "behavior": "Finish the current model call, save it, then stop before the next call"}


def enhancement_status(output_dir):
    """Read saved status even while the source drive or llama.cpp is unavailable."""
    from .checkpoint import directory_is_locked
    destination = Path(output_dir)
    require(destination.is_dir() and not destination.is_symlink(), "Missing or unsafe enhancement directory")
    if (destination / "batch.json").is_file():
        plan = _json(destination / "batch.json")["plan"]
        state = (_json(destination / "progress.json") if (destination / "progress.json").exists()
                 else {"status": "initialized", "total_windows": plan["total_windows"]})
        # Count durable completion receipts rather than an occasionally lagging status counter.
        state["completed_windows"] = sum(
            (destination / "windows" / w["window_id"] / "completion.json").is_file() for w in plan["windows"])
        current = state.get("current_window")
        if current in {w["window_id"] for w in plan["windows"]}:
            checkpoint = destination / "windows" / current / "checkpoint.json"
            if checkpoint.exists():
                state["current_window_checkpoint"] = _json(checkpoint)
    else:
        require((destination / "session.json").is_file(), "Directory is not a resumable enhancement job")
        path = destination / "checkpoint.json"
        state = _json(path) if path.exists() else {"status": "initialized"}
        if (destination / "completion.json").exists():
            state = _json(destination / "completion.json")
    state["writer_active"] = directory_is_locked(destination)
    state["pause_requested"] = (destination / "PAUSE").exists()
    if state.get("status") in {"running", "initializing", "finalizing"} and not state["writer_active"]:
        state["status"] = "interrupted"
        state["note"] = "No active writer; saved progress can be resumed"
    return state
