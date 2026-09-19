"""Frozen-reference gap experiments with separate native-video conversations."""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
import uuid

from .contract import ContractError, canonical_hash, require, sha256_file
from .llama_video import LocalVideoRuntime, RuntimeConfig, _strict_json
from .smart_selection import (
    PROJECT_ROOT, TIMELINE_SYSTEM, _image_blocks, _read, _write, retrieve_requests,
    review_schema, stage_media, verify_assets,
)
from .video_source import media_timeline, validate_native_timeline


PROTOCOL_VERSION = "native-video-gap-experiment-v2"
RANK_SYSTEM = """Watch the COMPLETE native video first to understand the surgical activity shown and
how visible actions change the operative field. Then compare ALL candidate stills together.
Your task is to rank surgical importance, with the most valuable evidence first.
Importance means how much a still helps someone understand what was visibly done in this
procedure. A frame is valuable because of its surgical information, not because it is early,
late, visually different, or next in the input list. Timestamps and frame IDs are locators.
Candidate stills are deliberately presented out of time order; the video remains in order.

Use this rubric in the context of the WHOLE supplied video:
- Highest priority: a clear, informative view of a defining procedural action or major
  visible change, relevant tissue-instrument interaction, or a distinct state needed to
  understand that action. Useful baseline and later inspection views can provide context.
- Medium priority: a clear supporting view that adds different procedural information.
- Lowest priority: repeated views already represented more clearly, routine instrument
  motion without a meaningful visible change, blur, obstruction, or empty/uninformative views.
  Visual novelty alone does not make an obscured frame important.
For repeated views of the same nearby moment, prioritize the clearest representative and
give redundant alternatives lower scores. Distant repetitions are separate moments.

Assign EVERY candidate an importance_score from 0 to 100: 90-100 defining evidence,
60-89 useful distinct procedural evidence, 30-59 supporting information, 0-29 redundant
or weak evidence. These are relative judgments, not confidence or clinical outcome scores.
Return the ranking array sorted by DECREASING importance_score. Break ties by additional
surgical information and clarity, never by timestamp, frame ID, or presentation position.
Do not output a timeline or a frame-by-frame chronological narration as the ranking.
Use each candidate ID exactly once. Give a short reason stating what is visibly informative
and why it deserves that priority, including redundancy or visibility limitations.
Assign moment_id to group nearby alternative views of the SAME visible moment; use a
singleton for a distinct moment. Respect the supplied maximum moment-group timestamp span.

Documented procedure context is background, not proof that anatomy, a maneuver, a successful
repair, or an outcome is visible. Do not infer those from the procedure name or timestamp.
If the media is not surgical, apply the same information/clarity/redundancy criteria to its
documented visible activity. Use consistent/uncertain/conflict for context_check, or
not_supplied without background. Return only the required JSON; invent no source IDs.""" + "\n\n" + TIMELINE_SYSTEM
AUDIT_SYSTEM = """Watch the complete native video and jointly review the supplied candidate stills.
The goal is a concise set of stills that explains the important VISIBLE surgical actions,
relevant tissue-instrument interactions, procedural changes, and useful baseline/inspection
views in the context of the supplied video. Clarity and distinct surgical information matter.
Keep useful candidates and drop redundant, obscured, or uninformative ones. Routine motion
and visual novelty alone do not establish surgical importance. If an important visible moment is not
adequately represented by the stills, request a specific time interval and evidence question.
Nearby equivalent views can already represent a moment. Gaps may or may not exist;
request additions only when useful. Do not request frames merely to fill regular time gaps.
Give a keep/drop decision with a short concrete reason for EVERY current candidate ID.
All times are milliseconds on the documented timeline. Respect the stated interval and
retrieval budgets. A replacement ID is a candidate to reconsider, not an instruction to delete
source evidence. Use the full video and the accumulated evidence in every follow-up review.
Mark ready only when no further useful evidence is needed; uncertainty is a valid outcome.
Source context is documented background, not proof of anatomy, successful repair or outcome.
If it conflicts with what is visible, use conflict or uncertain; without background use
not_supplied. Do not invent source files, frame IDs, or events. Return the required JSON.""" + "\n\n" + TIMELINE_SYSTEM


@dataclass(frozen=True)
class GapConfig:
    context_size: int = 131072
    image_max_tokens: int = 256
    ranking_max_tokens: int = 6144
    review_max_tokens: int = 4096
    retrieval_frames: int = 6
    requests_per_round: int = 6
    max_retrieval_rounds: int = 4
    max_request_span_ms: int = 10000
    max_moment_span_ms: int = 10000
    tolerance_ms: int = 1000
    procedure_context: str = ""

    def validate(self, candidate_count=None):
        require(0 < self.context_size <= 262144, "Invalid context capacity")
        require(0 < min(self.ranking_max_tokens, self.review_max_tokens)
                and max(self.ranking_max_tokens, self.review_max_tokens) < self.context_size,
                "Output budgets must fit context")
        require(self.image_max_tokens >= 64, "Image budget must be at least 64 tokens")
        require(1 <= self.retrieval_frames <= 24 and 1 <= self.max_retrieval_rounds <= 4,
                "Use 1–24 retrieved frames and 1–4 retrieval rounds")
        require(1 <= self.requests_per_round <= 24, "Use 1–24 requests per review")
        require(self.max_request_span_ms > 0 and self.max_moment_span_ms > 0
                and 0 <= self.tolerance_ms <= self.max_moment_span_ms, "Invalid time bounds")
        if candidate_count is not None:
            require(4 <= candidate_count <= 96, "The experiment requires 4–96 original candidates")
            largest = min(candidate_count - 1, max(1, (candidate_count * 90 + 50) // 100))
            require(self.retrieval_frames * self.max_retrieval_rounds >= largest,
                    "Retrieval allowance must be large enough for the largest withheld set")
            require(self.requests_per_round * self.max_retrieval_rounds >= largest,
                    "Request allowance must be large enough for the largest withheld set")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _protocol_hash():
    return hashlib.sha256((PROTOCOL_VERSION + RANK_SYSTEM + AUDIT_SYSTEM).encode()).hexdigest()


def _request_span(source, config):
    # Even a short fixture must not receive full detection credit for asking
    # for nearly its entire video. The CLI span is an upper bound.
    return min(config.max_request_span_ms, source["duration_ms"] / 2)


def _ranking_candidates(source, candidates):
    """Stable shuffled still presentation; native video and provenance stay intact."""
    chronological = sorted(candidates, key=lambda f: (f["timestamp_ms"], f["frame_index"], f["frame_id"]))
    chronological_ids = [f["frame_id"] for f in chronological]
    salt = 0
    while True:
        presented = sorted(candidates, key=lambda f: hashlib.sha256(
            f"surgical-ranking-v2:{source['video_sha256']}:{salt}:{f['frame_id']}".encode()).digest())
        ids = [f["frame_id"] for f in presented]
        if ids != chronological_ids and ids != chronological_ids[::-1]:
            return presented
        salt += 1


def _ranking_quality(raw, candidates, presented):
    """Block order-copying patterns for inspection; this is not an accuracy score."""
    rows = raw["ranking"]
    ids = [row["frame_id"] for row in rows]
    chronological = [f["frame_id"] for f in sorted(candidates,
        key=lambda f: (f["timestamp_ms"], f["frame_index"], f["frame_id"]))]
    scores = [row["importance_score"] for row in rows]
    checks = {"chronological_order": ids == chronological,
              "reverse_chronological_order": ids == chronological[::-1],
              "presentation_order_copied": ids == [f["frame_id"] for f in presented],
              "scores_descending": scores == sorted(scores, reverse=True),
              "distinct_score_count": len(set(scores)),
              "complete_unique_ranking": len(ids) == len(set(ids)) == len(chronological)
                                         and set(ids) == set(chronological)}
    flags = [key for key in ("chronological_order", "reverse_chronological_order", "presentation_order_copied")
             if checks[key]]
    if not checks["scores_descending"]:
        flags.append("scores_not_descending")
    if checks["distinct_score_count"] < 2:
        flags.append("flat_importance_scores")
    if not checks["complete_unique_ranking"]:
        flags.append("incomplete_or_duplicate_ranking")
    return {"accepted": not flags, "flags": flags, **checks, "checked_at": _now(),
            "reference_is_ground_truth": False,
            "interpretation": "Heuristic gate for inspection, not proof of surgical accuracy. "
                              "A flagged order may be defensible; it must not silently define this comparison."}


def _initial_messages(source, frames, media_names, video_name, config, *, ranking=False):
    overview = {
        "task": "Rank the candidate stills" if ranking else "Review the candidate stills and request useful missing evidence",
        "verified_source_context": config.procedure_context.strip() or None,
        "source_kind": source["source_kind"], "video_sha256": source["video_sha256"],
        "duration_ms": source["duration_ms"], "complete_video_frame_count": source["expected_video_frames"],
        "media_timeline": media_timeline(source),
        "timestamp_basis": source["frames"][0]["timestamp_basis"],
        "candidate_ids": [f["frame_id"] for f in frames],
    }
    if ranking:
        overview["maximum_moment_group_span_ms"] = config.max_moment_span_ms
        overview["ranking_objective"] = "Surgical information value in the supplied video; highest importance_score first"
        overview["candidate_presentation"] = "Deterministically shuffled stills with original timestamps; video remains chronological"
    else:
        overview.update(maximum_request_span_ms=_request_span(source, config),
                        frames_per_retrieval_round=config.retrieval_frames,
                        requests_per_review=config.requests_per_round,
                        maximum_retrieval_rounds=config.max_retrieval_rounds)
    return [{"role": "system", "content": RANK_SYSTEM if ranking else AUDIT_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": json.dumps(overview, ensure_ascii=False)},
                {"type": "input_video", "input_video": {"url": "file://" + video_name}},
                *_image_blocks(frames, media_names),
                {"type": "text", "text": "Consider the complete video and all candidate stills together before answering."},
            ]}]


def _audit_schema(ids, source, config):
    schema = review_schema(ids, source["duration_ms"])
    schema["properties"]["searches"]["maxItems"] = config.requests_per_round
    return schema


def _validate_audit(review, ids, source, config):
    from jsonschema import Draft202012Validator
    Draft202012Validator(_audit_schema(ids, source, config)).validate(review)
    for request in review["searches"]:
        require(math.isfinite(request["start_ms"]) and math.isfinite(request["end_ms"])
                and request["start_ms"] < request["end_ms"], "Invalid retrieval interval")
    require(not (review["ready"] and review["searches"]), "Ready output cannot request more evidence")


class _Session(AbstractContextManager):
    """Lazy, private server per experiment stage; recovery never needs a GPU."""
    def __init__(self, directory, source, config, factory):
        self.directory, self.source, self.config = Path(directory), source, config
        self.factory = factory or LocalVideoRuntime
        self.media, self.video_name, self.names = stage_media(source, self.directory)
        self.runtime = None

    def __enter__(self):
        return self

    def chat(self, *args, **kwargs):
        if self.runtime is None:
            attempt = self.directory / "runtime" / f"attempt-{time.time_ns()}"
            runtime_config = RuntimeConfig(PROJECT_ROOT, self.media, attempt,
                                           context_size=self.config.context_size,
                                           image_max_tokens=self.config.image_max_tokens)
            self.runtime = self.factory(runtime_config, expected_video_frames=self.source["expected_video_frames"],
                                        video_relative_path=self.video_name)
            try:
                self.runtime.__enter__()
            except BaseException:
                self.runtime = None
                raise
        return self.runtime.chat(*args, **kwargs)

    def __exit__(self, *args):
        if self.runtime is not None:
            return self.runtime.__exit__(*args)
        return False


def _checked_result(directory, source, messages, schema, max_tokens, context_size, video_name):
    """Verify recoverable results against exact requests and source receipts."""
    from jsonschema import Draft202012Validator
    request_path = directory / "request.json"
    request = _strict_json(request_path.read_bytes())
    result = _strict_json((directory / "result.json").read_bytes())
    require(request.get("messages") == messages and request.get("max_tokens") == max_tokens
            and request.get("response_format", {}).get("json_schema", {}).get("schema") == schema,
            "Saved request does not match this isolated conversation")
    receipt = result.get("verification", {})
    require(receipt.get("accepted") is True and receipt.get("full_source_video_verified") is True
            and receipt.get("context_truncation_observed") is False,
            "Saved result lacks accepted full-video verification")
    require(receipt.get("request_sha256") == sha256_file(request_path)
            and receipt.get("video_sha256") == source["video_sha256"]
            and receipt.get("video_relative_path") == video_name
            and receipt.get("video_fps_setting") == 0
            and receipt.get("expected_video_frames") == source["expected_video_frames"]
            and receipt.get("decoded_frame_ids") == list(range(source["expected_video_frames"])),
            "Saved result is not tied to every frame of the current video")
    response = _strict_json((directory / "response.json").read_bytes())
    require(response == result.get("response"), "Saved response and result disagree")
    choices = response.get("choices", [])
    require(len(choices) == 1 and choices[0].get("finish_reason") == "stop"
            and receipt.get("finish_reason") == "stop", "Saved output is unfinished")
    message = choices[0].get("message", {})
    require(message.get("role") == "assistant" and not message.get("tool_calls")
            and _strict_json(message.get("content", "")) == result.get("output"),
            "Saved parsed result differs from the actual assistant answer")
    prompt_tokens = response.get("usage", {}).get("prompt_tokens")
    require(type(prompt_tokens) is int and 0 < prompt_tokens <= context_size - max_tokens,
            "Saved output exceeded the configured context")
    Draft202012Validator(schema).validate(result["output"])
    return result


def _call(session, directory, source, messages, schema, max_tokens, config, *, metadata):
    directory = Path(directory)
    verify_assets(source)
    if (directory / "result.json").exists():
        result = _checked_result(directory, source, messages, schema, max_tokens,
                                 config.context_size, session.video_name)
    else:
        if directory.exists() and any(directory.iterdir()):
            directory.rename(directory.with_name(directory.name + f"-interrupted-{time.time_ns()}"))
        directory.mkdir(parents=True, exist_ok=True)
        _write(directory / "experiment-context.json", metadata)
        session.chat(messages, schema=schema, max_tokens=max_tokens, round_dir=directory)
        result = _checked_result(directory, source, messages, schema, max_tokens,
                                 config.context_size, session.video_name)
    verify_assets(source)
    return result


def _summary(output, plan, source, status, *, active_stage=None, error=None):
    from .gap_report import write_experiment_report
    conditions = _read(output / "conditions.json") if (output / "conditions.json").exists() else []
    rows = []
    for condition in conditions:
        directory = output / "conditions" / condition["id"]
        state = _read(directory / "state.json") if (directory / "state.json").exists() else {}
        rows.append({**condition, "status": state.get("status", "pending"),
                     "rounds_completed": len(state.get("rounds", [])), "artifact_dir": str(directory),
                     "selected_frame_ids": state.get("selected_frame_ids", []),
                     "metrics": _read(directory / "metrics.json") if (directory / "metrics.json").exists() else None})
    summary = {"schema_version": PROTOCOL_VERSION, "status": status, "updated_at": _now(),
               "created_at": plan["created_at"], "config": plan["config"], "active_stage": active_stage,
               "source": {k: source[k] for k in ("source_path", "expected_video_frames", "duration_ms", "timestamp_basis")},
               "reference": {"status": "frozen" if (output / "reference/reference.json").exists() else "pending",
                             "path": str(output / "reference/reference.json")},
               "conditions": rows, "error": error,
               "session_isolation": "new server and conversation for ranking and each condition; full history within a condition",
               "reference_is_clinical_ground_truth": False, "training_eligible": False}
    _write(output / "summary.json", summary)
    write_experiment_report(output)
    return summary


def _run_reference(output, source, candidates, config, factory, progress):
    from .gap_reference import build_reference, ranking_schema
    directory = output / "reference"
    directory.mkdir(exist_ok=True)
    with _Session(directory, source, config, factory) as session:
        presented = _ranking_candidates(source, candidates)
        messages = _initial_messages(source, presented, session.names, session.video_name, config, ranking=True)
        schema = ranking_schema([f["frame_id"] for f in presented], surgical_importance=True)
        state_path = directory / "session.json"
        presented_ids = [f["frame_id"] for f in presented]
        state = _read(state_path) if state_path.exists() else {"session_id": str(uuid.uuid4()), "created_at": _now(),
                                                             "presentation_frame_ids": presented_ids}
        require(state.get("presentation_frame_ids") == presented_ids, "Ranking presentation changed")
        _write(state_path, state)
        progress("Ranking: complete native video and all original candidates in a separate session.")
        result = _call(session, directory / "round-00", source, messages, schema, config.ranking_max_tokens,
                       config, metadata={**state, "stage": "ranking", "source_sha256": source["video_sha256"]})
        raw = result["output"]
        raw_quality = _ranking_quality(raw, candidates, presented)
        raw_quality_path = directory / "ranking-quality-raw.json"
        if not raw_quality_path.exists():
            # Retain the original rejection when recovering an out-of-order answer.
            previous_quality = directory / "ranking-quality.json"
            _write(raw_quality_path, _read(previous_quality) if previous_quality.exists() else raw_quality)
        blocking = [flag for flag in raw_quality["flags"] if flag != "scores_not_descending"]
        if blocking:
            _write(directory / "ranking-quality.json", raw_quality)
            raise ContractError("Surgical importance ranking needs inspection before audits: " + ", ".join(blocking))
        normalized = copy.deepcopy(raw)
        normalized["ranking"] = sorted(normalized["ranking"], key=lambda row: -row["importance_score"])
        original_ids = [row["frame_id"] for row in raw["ranking"]]
        normalized_ids = [row["frame_id"] for row in normalized["ranking"]]
        normalization = {"schema_version": "ranking-normalization-v1",
                         "method": "stable_descending_qwen_importance_score",
                         "tie_policy": "preserve_original_qwen_array_order_within_equal_scores",
                         "changed": original_ids != normalized_ids,
                         "original_frame_ids": original_ids, "normalized_frame_ids": normalized_ids,
                         "raw_output_canonical_sha256": canonical_hash(raw),
                         "normalized_output_canonical_sha256": canonical_hash(normalized),
                         "raw_result_sha256": sha256_file(directory / "round-00/result.json"),
                         "interpretation": "Only array order is normalized from Qwen's own scores. "
                                           "Scores, reasons, moment IDs, and source identities are unchanged. "
                                           "This does not validate surgical descriptions or distinguish tied scores."}
        for name, artifact in (("ranking-normalization.json", normalization), ("normalized-output.json", normalized)):
            path = directory / name
            if path.exists():
                require(_read(path) == artifact, f"Saved ranking derivation changed: {name}")
            else:
                _write(path, artifact)
        quality = _ranking_quality(normalized, candidates, presented)
        quality.update(evaluated_order="normalized_descending_qwen_scores",
                       normalization_changed_order=normalization["changed"], raw_flags=raw_quality["flags"])
        _write(directory / "ranking-quality.json", quality)
        require(quality["accepted"], "Surgical importance ranking needs inspection before audits: " + ", ".join(quality["flags"]))
        reference = build_reference(normalized, candidates, duration_ms=source["duration_ms"],
                                    tolerance_ms=config.tolerance_ms, max_moment_span_ms=config.max_moment_span_ms,
                                    surgical_importance=True)
        require(result["output"]["context_check"] != "conflict", "Ranking conflicts with the verified source context")
        destination = directory / "reference.json"
        if destination.exists():
            require(_read(destination) == reference, "The frozen reference changed")
        else:
            _write(destination, reference)
        return reference


def _condition_metrics(directory, reference, condition, source, state, config):
    from .gap_reference import score_condition
    result = score_condition(reference, condition, source["frames"], state["rounds"],
                             max_request_span_ms=_request_span(source, config))
    _write(directory / "metrics.json", result)
    return result


def _advance_review(previous, source, result, round_dir, names, config, *, write_artifacts):
    """Deterministically apply a result; also reconstructs checkpoints on resume."""
    state = copy.deepcopy(previous)
    by_id = {f["frame_id"]: f for f in source["frames"]}
    index, ids = state["next_round"], list(state["candidate_ids"])
    review = result["output"]
    _validate_audit(review, ids, source, config)
    artifacts = {"candidate-manifest.json": [by_id[f] for f in ids]}
    state["messages"].append({"role": "assistant", "content": result["response"]["choices"][0]["message"]["content"]})
    row = {"round": index, "directory": str(round_dir), "candidate_ids": ids,
           "output": review, "verification": result["verification"], "retrieval": []}
    state["rounds"].append(row)
    state["next_round"] = index + 1
    state["selected_frame_ids"] = [f for f in ids if review["decisions"][f]["decision"] == "keep"]
    if review["context_check"] == "conflict":
        state["status"] = "context_conflict"
    elif not review["searches"]:
        state["status"] = "completed" if review["ready"] else "model_uncertain"
    elif index >= config.max_retrieval_rounds:
        state["status"] = "retrieval_budget_exhausted"
    else:
        span_limit = _request_span(source, config)
        narrow = [q for q in review["searches"] if q["end_ms"] - q["start_ms"] <= span_limit]
        added, receipts = retrieve_requests(source, narrow, ids, config.retrieval_frames)
        iterator = iter(receipts)
        receipts = [next(iterator) if q in narrow else {"request": q, "returned_frame_ids": [],
                    "status": "rejected_too_broad", "maximum_span_ms": span_limit}
                    for q in review["searches"]]
        row["retrieval"] = receipts
        artifacts.update({"retrieval.json": receipts, "retrieved-manifest.json": added})
        if not added and len(narrow) == len(review["searches"]):
            state["status"] = "available_evidence_exhausted"
        else:
            state["candidate_ids"] = sorted([*ids, *[f["frame_id"] for f in added]], key=lambda f: by_id[f]["frame_index"])
            state["messages"].append({"role": "user", "content": [
                {"type": "text", "text": "Evidence retrieval receipts: " + json.dumps(receipts)},
                *_image_blocks(added, names),
                {"type": "text", "text": "Keep the original whole video and history in context. Reconsider ALL current candidates: "
                 + json.dumps(state["candidate_ids"]) + f". Remaining retrieval rounds: {config.max_retrieval_rounds-index-1}. "
                 + f"Requests longer than {span_limit} ms return no frames; narrow any such request."},
            ]})
            state["status"] = "awaiting_review"
    for name, value in artifacts.items():
        if write_artifacts:
            _write(round_dir / name, value)
        else:
            require((round_dir / name).is_file() and _read(round_dir / name) == value,
                    f"Saved round evidence differs from reconstructed source retrieval: {name}")
    return state


def _run_condition(output, source, reference, condition, config, factory, should_pause, progress, changed):
    directory = output / "conditions" / condition["id"]
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / "condition.json", condition)  # Private evaluator data, never included in model messages.
    by_id = {f["frame_id"]: f for f in source["frames"]}
    with _Session(directory, source, config, factory) as session:
        initial = _initial_messages(source, [by_id[f] for f in condition["supplied_frame_ids"]],
                                    session.names, session.video_name, config)
        state_path = directory / "state.json"
        if state_path.exists():
            state = _read(state_path)
            require(state["messages"][:2] == initial, "Condition's original blinded input changed")
            reconstructed = {"session_id": state["session_id"], "status": "prepared", "next_round": 0,
                             "candidate_ids": condition["supplied_frame_ids"], "messages": initial,
                             "rounds": [], "selected_frame_ids": []}
            for row in state["rounds"]:
                require(reconstructed["status"] in {"prepared", "awaiting_review"}, "Saved condition continues after completion")
                round_dir = directory / "rounds" / f"round-{reconstructed['next_round']:02d}"
                result = _checked_result(round_dir, source, reconstructed["messages"],
                                         _audit_schema(reconstructed["candidate_ids"], source, config),
                                         config.review_max_tokens, config.context_size, session.video_name)
                reconstructed = _advance_review(reconstructed, source, result, round_dir, session.names,
                                                config, write_artifacts=False)
                require(row == reconstructed["rounds"][-1], "Stored round differs from its accepted result")
            require(state == reconstructed, "Saved condition history or candidate state changed")
        else:
            state = {"session_id": str(uuid.uuid4()), "status": "prepared", "next_round": 0,
                     "candidate_ids": condition["supplied_frame_ids"], "messages": initial,
                     "rounds": [], "selected_frame_ids": []}
            _write(state_path, state)
        if state["status"] not in {"prepared", "awaiting_review"}:
            _condition_metrics(directory, reference, condition, source, state, config)
            return True
        for index in range(state["next_round"], config.max_retrieval_rounds + 1):
            if should_pause():
                return False
            round_dir = directory / "rounds" / f"round-{index:02d}"
            ids = list(state["candidate_ids"])
            progress(f"{condition['id']}, review {index + 1}: entire video + {len(ids)} stills; private session {state['session_id']}.")
            result = _call(session, round_dir, source, state["messages"], _audit_schema(ids, source, config),
                           config.review_max_tokens, config,
                           metadata={"session_id": state["session_id"], "stage": condition["id"], "round": index,
                                     "candidate_ids": ids, "reference_sha256": sha256_file(output / "reference/reference.json")})
            state = _advance_review(state, source, result, round_dir, session.names, config, write_artifacts=True)
            _write(state_path, state)
            _condition_metrics(directory, reference, condition, source, state, config)
            changed()
            progress(f"{condition['id']}, review {index + 1} saved: {state['status']}.")
            if state["status"] != "awaiting_review":
                return True
        return True


def run_experiment(selection_run, output_dir, config=None, *, resume=False, prepare_only=False,
                   runtime_factory=None, should_stop=lambda: False, progress=print):
    from .gap_reference import make_conditions
    output = Path(output_dir).expanduser().resolve()
    if resume:
        require((output / "experiment.json").is_file(), "No saved gap experiment found")
    else:
        require(selection_run is not None, "Provide an existing full-video selection run")
        selection = Path(selection_run).expanduser().resolve()
        require(not output.is_relative_to(selection), "Experiment output must be outside the original selection run")
        output.mkdir(parents=True, exist_ok=False)
    with (output / ".experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ContractError("This experiment is already active") from error
        if not prepare_only:
            # Clear an old pause at explicit startup, before any slow evidence
            # validation; a new pause arriving during that work must survive.
            (output / ".pause-requested").unlink(missing_ok=True)
        if resume:
            plan = _read(output / "experiment.json")
            config = GapConfig(**plan["config"])
            if selection_run is not None:
                require(str(Path(selection_run).expanduser().resolve()) == plan["selection_run"], "Resume source differs")
        else:
            config = config or GapConfig()
            parent = _read(selection / "run.json")
            source_path = selection / "source/source.json"
            require(sha256_file(source_path) == parent["source_manifest_sha256"], "Original source manifest changed")
            source = _read(source_path)
            require(not output.is_relative_to(Path(source["source_path"]).parent if source["source_kind"] == "original_video"
                                               else Path(source["source_path"])), "Output must be outside original source media")
            initial = _read(selection / "initial-selection.json")
            ids = initial["selected_ids"]
            require(len(ids) == len(set(ids)), "Duplicate initial candidates")
            by_id = {f["frame_id"]: f for f in source["frames"]}
            require(set(ids) <= set(by_id), "Unknown initial candidate")
            candidates = sorted([by_id[f] for f in ids], key=lambda f: f["frame_index"])
            config.validate(len(candidates))
            if not config.procedure_context:
                config = GapConfig(**{**asdict(config), "procedure_context": parent["config"].get("procedure_context", "")})
            (output / "source").mkdir()
            shutil.copyfile(source_path, output / "source/source.json")
            shutil.copyfile(selection / "initial-selection.json", output / "initial-selection.json")
            _write(output / "candidate-manifest.json", candidates)
            plan = {"schema_version": PROTOCOL_VERSION, "created_at": _now(), "selection_run": str(selection),
                    "config": asdict(config), "protocol_sha256": _protocol_hash(),
                    "source_manifest_sha256": sha256_file(output / "source/source.json"),
                    "candidate_manifest_sha256": sha256_file(output / "candidate-manifest.json"),
                    "original_selection_sha256": sha256_file(output / "initial-selection.json"),
                    "candidate_pool": "original_embedding_candidates_before_Qwen_keep_drop",
                    "ranking_objective": "surgical_importance",
                    "omission_policy": "drop_least_important_50_70_90_percent",
                    "condition_order": ["drop_50", "drop_70", "drop_90", "control_all"]}
            _write(output / "experiment.json", plan)
        source = _read(output / "source/source.json")
        candidates = _read(output / "candidate-manifest.json")
        active = None
        try:
            config.validate(len(candidates))
            require(plan["schema_version"] == PROTOCOL_VERSION and plan["protocol_sha256"] == _protocol_hash(), "Experiment protocol changed")
            for name, key in (("source/source.json", "source_manifest_sha256"), ("candidate-manifest.json", "candidate_manifest_sha256"),
                              ("initial-selection.json", "original_selection_sha256")):
                require(sha256_file(output / name) == plan[key], f"Saved evidence changed: {name}")
            verify_assets(source)
            _write(output / "native-timeline-verification.json", validate_native_timeline(source))
            if prepare_only:
                return _summary(output, plan, source, "prepared")
            should_pause = lambda: should_stop() or (output / ".pause-requested").exists()
            if should_pause():
                return _summary(output, plan, source, "paused")
            active = "ranking"
            _summary(output, plan, source, "running", active_stage=active)
            reference = _run_reference(output, source, candidates, config, runtime_factory, progress)
            conditions = make_conditions(reference)
            if (output / "conditions.json").exists():
                require(_read(output / "conditions.json") == conditions, "Frozen test conditions changed")
            else:
                _write(output / "conditions.json", conditions)
            for condition in conditions:
                active = condition["id"]
                _summary(output, plan, source, "running", active_stage=active)
                if should_pause():
                    return _summary(output, plan, source, "paused", active_stage=active)
                complete = _run_condition(output, source, reference, condition, config, runtime_factory,
                                          should_pause, progress,
                                          lambda: _summary(output, plan, source, "running", active_stage=active))
                if not complete:
                    return _summary(output, plan, source, "paused", active_stage=active)
            (output / "last-error.json").unlink(missing_ok=True)
            return _summary(output, plan, source, "completed")
        except BaseException as error:
            detail = {"type": type(error).__name__, "message": str(error), "at": _now(), "stage": active}
            _write(output / "last-error.json", detail)
            _summary(output, plan, source, "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     active_stage=active, error=detail)
            raise


def experiment_status(output_dir):
    output = Path(output_dir).expanduser().resolve()
    require((output / "experiment.json").exists(), "No saved gap experiment found")
    status = _read(output / "summary.json") if (output / "summary.json").exists() else {"status": "initializing"}
    with (output / ".experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            active = False
        except BlockingIOError:
            active = True
    status["writer_active"] = active
    status["pause_requested"] = (output / ".pause-requested").exists()
    if active and status.get("active_stage"):
        stage = status["active_stage"]
        directory = output / "reference" if stage == "ranking" else output / "conditions" / stage
        logs = sorted(directory.glob("runtime/attempt-*/server.log"))
        if logs:
            with logs[-1].open("rb") as handle:
                handle.seek(max(0, logs[-1].stat().st_size - 32768))
                tail = handle.read().decode(errors="replace")
            prefill = re.findall(r"prompt processing, n_tokens =\s*(\d+), progress = ([0-9.]+)", tail)
            generated = re.findall(r"n_gen = (\d+), n_remaining = (\d+)", tail)
            status["live_progress"] = {"log_path": str(logs[-1]),
                "prompt_tokens_processed": int(prefill[-1][0]) if prefill else None,
                "prompt_fraction": float(prefill[-1][1]) if prefill else None,
                "generated_tokens": int(generated[-1][0]) if generated else None}
    return status


def request_experiment_pause(output_dir):
    output = Path(output_dir).expanduser().resolve()
    require((output / "experiment.json").exists(), "No saved gap experiment found")
    (output / ".pause-requested").touch()
    return {"pause_requested": True, "message": "The current model call will finish and checkpoint before pausing."}


def add_gap_parser(subparsers):
    parser = subparsers.add_parser("experiment-frame-gaps", help="Rank once, then audit complementary still sets in fresh full-video sessions")
    parser.add_argument("--selection-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--procedure-context", default="")
    for name, default in asdict(GapConfig()).items():
        if name != "procedure_context":
            parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name in ("gap-experiment-status", "pause-gap-experiment"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-dir", type=Path, required=True)


def experiment_cli(args, *, should_stop=lambda: False):
    from .llama_video import VideoRuntimeError
    from jsonschema import ValidationError
    config = None if args.resume else GapConfig(**{key: getattr(args, key) for key in asdict(GapConfig())})
    try:
        result = run_experiment(args.selection_run, args.output_dir, config, resume=args.resume,
                                prepare_only=args.prepare_only, should_stop=should_stop,
                                progress=lambda message: print(message, flush=True))
    except (VideoRuntimeError, ValidationError) as error:
        raise ContractError(str(error)) from error
    print(json.dumps({"status": result["status"], "report": str(args.output_dir.resolve() / "report.html")}, indent=2))
