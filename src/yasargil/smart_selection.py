"""Review frame candidates jointly with an unchanged, complete native video."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import math
from pathlib import Path
import time

from jsonschema import Draft202012Validator

from .contract import ContractError, require, sha256_file
from .video_source import media_timeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TIMELINE_SYSTEM = """Use only the supplied video's playback timeline for every timestamp, interval,
coverage decision, and evidence citation. The media_timeline object is authoritative.
For a reconstructed image sequence, duration is frame_count / playback_fps seconds;
the zero-based frame i starts at i / playback_fps seconds. At 1 fps, 288 frames last
288 seconds and the final frame starts at 287 seconds. For an original video, use
its supplied presentation timestamps, not an assumed 1 fps.
Ignore recorded repair time, trial duration, outcome metadata, and dataset CSV video
length or frame counts when locating evidence. Do not infer a speed multiplier,
stretch or rescale timestamps, or request unseen footage to reconcile those values.
The complete supplied video means all available source frames, not verified coverage
of an entire operation or repair. Reconstruction times locate these images accurately
within this video; original procedure elapsed time and coverage remain unverified."""
TIMING_POLICY_SHA256 = hashlib.sha256(TIMELINE_SYSTEM.encode()).hexdigest()

KEEP_DROP_SYSTEM = """You review sampled candidate frames for a chronological visual record. Process the
entire supplied native video before judging the candidate set. Review all candidates
jointly, using earlier and later video context. Judge visible action/state changes,
coverage and redundancy. Image appearance alone does not establish anatomy, procedure,
successful repair or outcome. Do not invent events. Source context, when supplied, is
documented background; still ground observations in visible evidence. If appearance
conflicts with that context, say conflict or uncertain and explain in scene_summary.
Use not_supplied when no source context was provided.

Return the required JSON. Give a keep/drop decision and short visible-evidence reason
for EVERY listed candidate ID. The sampled candidate set is fixed: only keep or drop
these IDs. Additional frames, replacements, and evidence searches are unavailable;
return searches as an empty array. If an observation is unclear, explain the uncertainty
in the decision reason or scene_summary. Set ready when you can complete this keep/drop
review; otherwise set it false and explain why. Do not invent source filenames, frame IDs
or exact observations between source frames. All source files are retained; drop only
affects the selected set. Protected temporal anchors remain in the final set for coverage
even if you recommend dropping them. These are provisional selection suggestions, not
clinical validation or training labels."""
REVIEW_POLICY_SHA256 = hashlib.sha256(KEEP_DROP_SYSTEM.encode()).hexdigest()


@dataclass(frozen=True)
class SelectionConfig:
    candidate_budget: int = 24
    retrieval_frames: int = 6
    max_candidates: int = 48
    max_retrieval_rounds: int = 0
    context_size: int = 131072
    image_max_tokens: int = 256
    max_tokens: int = 4096
    embedding_backend: str = "dinov2"
    embedding_model_path: str | None = None
    procedure_context: str = ""
    review_mode: str = "keep_drop"
    request_timeout_seconds: float = 3600

    def validate(self):
        require(self.review_mode in {"keep_drop", "retrieval"}, "Unknown selection review mode")
        require(2 <= self.candidate_budget <= self.max_candidates <= 96,
                "Use 2 <= candidate budget <= max candidates <= 96")
        require(1 <= self.retrieval_frames <= 24, "Retrieval budget must be 1–24 frames per round")
        require(0 <= self.max_retrieval_rounds <= 4, "Use 0–4 retrieval rounds")
        require(self.review_mode != "keep_drop" or self.max_retrieval_rounds == 0,
                "Keep/drop selection does not allow retrieval rounds")
        require(0 < self.max_tokens < self.context_size <= 262144, "Invalid token/context budget")
        require(self.image_max_tokens >= 64, "Image budget must be at least 64 tokens")
        require(self.embedding_backend in {"dinov2", "dinov3"}, "Unknown embedding backend")
        require(math.isfinite(self.request_timeout_seconds) and self.request_timeout_seconds > 0,
                "Request timeout must be finite and positive")


def selection_config_from_saved(value):
    """Preserve old retrieval schemas when verifying historical completed runs."""
    return SelectionConfig(**{**value, "review_mode": value.get("review_mode", "retrieval")})


def _config_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write(path, value):
    """Replace a checkpoint atomically; source media are never written here."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def review_schema(candidate_ids, duration_ms, *, review_mode="retrieval"):
    require(review_mode in {"keep_drop", "retrieval"}, "Unknown selection review mode")
    decision = {"type": "object", "additionalProperties": False,
                "properties": {"decision": {"type": "string", "enum": ["keep", "drop"]},
                               "reason": {"type": "string", "minLength": 1, "maxLength": 400}},
                "required": ["decision", "reason"]}
    return {"type": "object", "additionalProperties": False, "properties": {
        "scene_summary": {"type": "string", "minLength": 1, "maxLength": 1600},
        "context_check": {"type": "string", "enum": ["consistent", "uncertain", "conflict", "not_supplied"]},
        "decisions": {"type": "object", "additionalProperties": False,
                      "properties": {frame_id: decision for frame_id in candidate_ids},
                      "required": list(candidate_ids)},
        "searches": {"type": "array", "maxItems": 0 if review_mode == "keep_drop" else 3, "items": {
            "type": "object", "additionalProperties": False, "properties": {
                "start_ms": {"type": "number", "minimum": 0, "maximum": duration_ms},
                "end_ms": {"type": "number", "minimum": 0, "maximum": duration_ms},
                "question": {"type": "string", "minLength": 1, "maxLength": 400},
                "replace_frame_id": {"enum": [None, *candidate_ids]},
            }, "required": ["start_ms", "end_ms", "question", "replace_frame_id"]}},
        "ready": {"type": "boolean"},
    }, "required": ["scene_summary", "context_check", "decisions", "searches", "ready"]}


def validate_review(value, candidate_ids, duration_ms, *, review_mode="retrieval"):
    Draft202012Validator(review_schema(candidate_ids, duration_ms, review_mode=review_mode)).validate(value)
    for search in value["searches"]:
        require(math.isfinite(search["start_ms"]) and math.isfinite(search["end_ms"])
                and search["start_ms"] < search["end_ms"], "Invalid retrieval interval")
    require(not (value["ready"] and value["searches"]), "A ready selection cannot still request evidence")


def _frame_public(frame):
    """The exact evidence locator shown to Qwen and retained with its decisions."""
    keys = ("frame_id", "frame_index", "timestamp_ms", "timestamp_basis", "source_pts",
            "time_base", "source_timestamp_ms", "source_path", "source_sha256",
            "image_path", "image_sha256")
    return {key: frame.get(key) for key in keys}


def verify_assets(source):
    expected = {str(Path(source["video_path"]).resolve()): source["video_sha256"]}
    for frame in source["frames"]:
        for path_key, hash_key in (("source_path", "source_sha256"), ("image_path", "image_sha256")):
            path = str(Path(frame[path_key]).resolve())
            require(path not in expected or expected[path] == frame[hash_key], "Conflicting source hashes")
            expected[path] = frame[hash_key]
    for path, digest in expected.items():
        require(Path(path).is_file() and sha256_file(path) == digest, f"Evidence changed or is missing: {path}")


def stage_media(source, output):
    media = Path(output) / "media"
    media.mkdir(exist_ok=True)
    video_name = "video" + (Path(source["video_path"]).suffix.lower() or ".mp4")
    links = {video_name: Path(source["video_path"]).resolve()}
    frame_names = {}
    for ordinal, frame in enumerate(source["frames"]):
        suffix = Path(frame["image_path"]).suffix.lower()
        require(suffix in {".png", ".jpg", ".jpeg"}, "Unsupported extracted image format")
        name = f"frame-{ordinal:08d}{suffix}"
        frame_names[frame["frame_id"]] = name
        links[name] = Path(frame["image_path"]).resolve()
    for name, target in links.items():
        link = media / name
        if link.is_symlink() or link.exists():
            require(link.is_symlink() and link.resolve() == target, f"Media alias changed: {link}")
        else:
            link.symlink_to(target)
    return media, video_name, frame_names


def _image_blocks(frames, names):
    blocks = []
    for frame in frames:
        blocks.extend([
            {"type": "text", "text": "Candidate evidence: " + json.dumps(_frame_public(frame), ensure_ascii=False)},
            {"type": "image_url", "image_url": {"url": "file://" + names[frame["frame_id"]]}},
        ])
    return blocks


def initial_messages(source, candidates, protected_ids, names, video_name, config):
    context = config.procedure_context.strip()
    system = """You review candidate frames for a chronological visual record. Process the entire supplied
native video before judging the candidate set. Review candidates jointly, using earlier and later
video context. Judge visible action/state changes, coverage and redundancy. Image appearance
alone does not establish anatomy, procedure, successful repair or outcome. Do not invent events.
Source context, when supplied, is documented background; still ground observations in visible
evidence. If appearance conflicts with that context, say conflict or uncertain and explain in
scene_summary. Otherwise use not_supplied when no source context was provided.

Return the required JSON. Give a keep/drop decision and short visible-evidence reason for EVERY
listed candidate ID. Request a bounded time INTERVAL and a question when evidence is unclear or
a better frame is needed. A replacement request identifies the candidate to reconsider; it does
not establish that the requested event occurred. Not found/unclear are valid outcomes. Times are
milliseconds on the supplied video's timeline, with the stated timestamp basis. Do not invent
source filenames, frame IDs or exact observations between source frames. Mark ready only if no
further evidence is requested. All source files are retained; drop only affects the selected set.
Protected temporal anchors remain in the final set for coverage even if you recommend dropping
them. These are provisional selection suggestions, not clinical validation or training labels."""
    if config.review_mode == "keep_drop":
        system = KEEP_DROP_SYSTEM
    system += "\n\n" + TIMELINE_SYSTEM
    overview = {"task": "Select useful frames after reviewing the entire video and the candidates together",
                "verified_source_context": context or None,
                "source_kind": source["source_kind"], "video_sha256": source["video_sha256"],
                "duration_ms": source["duration_ms"], "complete_video_frame_count": source["expected_video_frames"],
                "media_timeline": media_timeline(source),
                "video_input_mode": "native_video_all_decoded_frames",
                "timestamp_basis": source["frames"][0]["timestamp_basis"],
                "candidate_ids": [frame["frame_id"] for frame in candidates],
                "protected_temporal_anchor_ids": protected_ids,
                "maximum_retrieval_rounds": config.max_retrieval_rounds}
    if config.review_mode == "keep_drop":
        overview.pop("maximum_retrieval_rounds")
        overview.update({"task": "Keep or drop only the sampled frames after reviewing the entire video and all candidates together",
                         "review_mode": "keep_drop", "candidate_set_frozen": True})
    return [{"role": "system", "content": system}, {"role": "user", "content": [
        {"type": "text", "text": json.dumps(overview, ensure_ascii=False)},
        {"type": "input_video", "input_video": {"url": "file://" + video_name}},
        *_image_blocks(candidates, names),
        {"type": "text", "text": "Review the whole video and all listed candidates before returning your JSON decisions."},
    ]}]


def retrieve_requests(source, searches, known_ids, budget):
    from .video_source import retrieve_interval
    pools = [retrieve_interval(source, search["start_ms"], search["end_ms"], budget,
                               exclude_ids=known_ids) for search in searches]
    added, seen = [], set(known_ids)
    receipts = [{"request": search, "returned_frame_ids": []} for search in searches]
    # Distribute the shared budget across requests instead of exhausting it on the first.
    while len(added) < budget and any(pools):
        for index, pool in enumerate(pools):
            while pool and pool[0]["frame_id"] in seen:
                pool.pop(0)
            if pool and len(added) < budget:
                frame = pool.pop(0)
                seen.add(frame["frame_id"])
                added.append(frame)
                receipts[index]["returned_frame_ids"].append(frame["frame_id"])
    for receipt in receipts:
        receipt["status"] = "retrieved" if receipt["returned_frame_ids"] else "no_new_evidence_within_budget"
    added.sort(key=lambda frame: frame["frame_index"])
    return added, receipts


def _selection_record(source, state, protected_ids, output):
    latest = state.get("last_output")
    by_id = {frame["frame_id"]: frame for frame in source["frames"]}
    decisions = []
    for frame_id in state["candidate_ids"]:
        proposed = (latest["decisions"].get(frame_id, {"decision": "unreviewed", "reason": "Retrieved for the next joint review"})
                    if latest else {"decision": "unreviewed", "reason": "Prepared candidate"})
        protected = frame_id in protected_ids
        effective = "keep" if protected else proposed["decision"]
        decisions.append({**by_id[frame_id], "model_decision": proposed["decision"], "model_reason": proposed["reason"],
                          "effective_decision": effective, "protected_temporal_anchor": protected,
                          "coverage_override": protected and proposed["decision"] == "drop",
                          "introduced_round": state["introduced_rounds"][frame_id]})
    return {"schema_version": "smart-frame-selection-v1", "status": state["status"],
            "source_manifest": str(Path(output) / "source" / "source.json"),
            "source_kind": source["source_kind"], "video_path": source["video_path"],
            "video_sha256": source["video_sha256"], "expected_video_frames": source["expected_video_frames"],
            "timestamp_basis": source["frames"][0]["timestamp_basis"],
            "selector_exposure": "retrospective_full_video", "native_video_required_every_round": True,
            "completed_rounds_full_video_verified": bool(state["rounds"]) and all(
                row["verification"].get("full_source_video_verified", False) for row in state["rounds"]),
            "clinical_validation": "not_performed", "training_eligible": False,
            "scene_summary": latest["scene_summary"] if latest else None,
            "context_check": latest["context_check"] if latest else None,
            "frames": decisions, "selected_frame_ids": [f["frame_id"] for f in decisions if f["effective_decision"] == "keep"],
            "rounds": state["rounds"], "unresolved_searches": state.get("unresolved_searches", [])}


def write_selection_report(record, output):
    """A local visual review of every candidate, including excluded evidence."""
    escape = lambda value: html.escape(str(value), quote=True)
    cards = []
    for frame in record["frames"]:
        milliseconds = frame["timestamp_ms"]
        seconds = milliseconds / 1000
        timestamp = f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"
        source = Path(frame["source_path"])
        original = Path(frame["image_path"])
        image_uri = original.as_uri()
        cards.append(f'''<article class="frame" data-decision="{escape(frame['effective_decision'])}">
<a href="{escape(image_uri)}"><img loading="lazy" src="{escape(image_uri)}" alt="Frame {frame['frame_index']} at {timestamp}"></a>
<div class="body"><div class="meta">{escape(frame['effective_decision'].upper())} · {timestamp}
{' · COVERAGE ANCHOR' if frame['protected_temporal_anchor'] else ''}</div>
<p>{escape(frame['model_reason'])}</p>
<p class="muted">Qwen: {escape(frame['model_decision'])}{' · retained by coverage rule' if frame['coverage_override'] else ''}</p>
<p><b>Source:</b> <a href="{escape(source.as_uri())}">{escape(source.name)}</a><br>
<b>Time basis:</b> {escape(frame['timestamp_basis'])}<br><b>Source PTS:</b> {escape(frame.get('source_pts'))}
 · <b>Time base:</b> {escape(frame.get('time_base'))}</p>
<details><summary>Complete provenance</summary><pre>{escape(json.dumps(frame, indent=2))}</pre></details>
</div></article>''')
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Contextual frame selection</title><style>
body{{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f3f5f7;color:#18222d}}main{{max-width:1300px;margin:auto;padding:28px}}
h1{{margin:0 0 8px}}.notice{{background:#fff3d5;padding:14px;border-radius:8px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}}
.frame{{background:white;border:1px solid #dce3ea;border-radius:10px;overflow:hidden}}.frame img{{width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}}
.body{{padding:14px}}.meta{{font-weight:700;color:#135f67}}.muted{{color:#5c6571}}pre{{font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere}}
a{{color:#165ea8}}select{{font:inherit;padding:6px;margin:16px 0}}.stats{{margin:12px 0}}details{{border-top:1px solid #ddd;padding-top:8px}}
</style><main><h1>Contextual frame selection</h1><div class="stats">Status: <b>{escape(record['status'])}</b> ·
{len(record['selected_frame_ids'])} retained / {len(record['frames'])} candidates · {record['expected_video_frames']} video frames</div>
<p class="notice">These are Qwen's provisional selection suggestions. Clinical correctness has not been validated.
Source files are retained. Timestamp basis: <b>{escape(record['timestamp_basis'])}</b>.</p>
<p>{escape(record['scene_summary'] or 'Candidate preparation; Qwen has not reviewed these frames yet.')}</p>
<p><a href="selection.json">Selection JSON</a> · <a href="source/source.json">Complete source manifest</a>
 · <a href="initial-selection.json">Embedding selection</a></p>
<label>Show <select id="filter"><option value="all">all candidates</option><option value="keep">retained</option><option value="drop">dropped</option><option value="unreviewed">awaiting review</option></select></label>
<div class="grid">{''.join(cards)}</div>
<details><summary>Unresolved evidence requests</summary><pre>{escape(json.dumps(record['unresolved_searches'], indent=2))}</pre></details>
<script>document.getElementById('filter').addEventListener('change',function(){{document.querySelectorAll('.frame').forEach(card=>{{card.hidden=this.value!=='all'&&card.dataset.decision!==this.value;}});}});</script>
</main></html>'''
    Path(output, "selection.html").write_text(report, encoding="utf-8")


def _recover_verified_result(round_dir, source, messages, candidate_ids, config, video_name):
    """Recover a published transport result only with matching saved evidence."""
    request_path = round_dir / "request.json"
    saved_request = _read(request_path)
    require(saved_request.get("messages") == messages, "Saved round does not match its full-video conversation")
    require(saved_request.get("max_tokens") == config.max_tokens
            and saved_request.get("response_format", {}).get("json_schema", {}).get("schema")
            == review_schema(candidate_ids, source["duration_ms"], review_mode=config.review_mode),
            "Saved round request configuration changed")
    result = _read(round_dir / "result.json")
    verification = result.get("verification")
    require(isinstance(verification, dict), "Saved round is missing its verification receipt")
    require(verification.get("accepted") is True
            and verification.get("full_source_video_verified") is True
            and verification.get("context_truncation_observed") is False,
            "Saved round was not accepted with complete untruncated video context")
    require(verification.get("request_sha256") == sha256_file(request_path), "Saved request bytes changed")
    expected_frames = source["expected_video_frames"]
    require(verification.get("video_sha256") == source["video_sha256"]
            and verification.get("video_relative_path") == video_name
            and verification.get("video_fps_setting") == 0
            and verification.get("expected_video_frames") == expected_frames
            and verification.get("decoded_frames") == expected_frames
            and verification.get("decoded_frame_ids") == list(range(expected_frames)),
            "Saved round does not prove every frame of the current source video")
    response = result.get("response")
    require(isinstance(response, dict) and response == _read(round_dir / "response.json"),
            "Saved result does not match the raw response artifact")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict),
            "Saved response must contain exactly one answer")
    choice = choices[0]
    message = choice.get("message")
    require(choice.get("finish_reason") == "stop" and verification.get("finish_reason") == "stop"
            and isinstance(message, dict) and message.get("role") == "assistant"
            and isinstance(message.get("content"), str) and not message.get("tool_calls"),
            "Saved response is incomplete or not an assistant selection answer")
    require(json.loads(message["content"]) == result.get("output"),
            "Saved parsed selection differs from the model's raw answer")
    return result


def validate_completed_selection(directory, selection_run, *, snapshot=False):
    """Tie effective keeps to canonical source rows and the completed raw review."""
    from .llama_video import _strict_json

    def read(path):
        try:
            return _strict_json(Path(path).read_bytes())
        except (ValueError, OSError) as exc:
            raise ContractError(f"Cannot read selection evidence {path}: {exc}") from exc

    directory, selection_run = Path(directory), Path(selection_run)
    parent = read(directory / ("selection-run.json" if snapshot else "run.json"))
    source = read(directory / "source/source.json")
    record = read(directory / "selection.json")
    state = read(directory / ("selection-state.json" if snapshot else "state.json"))
    initial = read(directory / "initial-selection.json")
    require(parent.get("schema_version") == "smart-frame-selection-run-v1"
            and record.get("schema_version") == "smart-frame-selection-v1", "Expected a normal smart-selection run")
    require(parent["source_manifest_sha256"] == sha256_file(directory / "source/source.json"),
            "Selection source manifest changed")
    require(record.get("status") == state.get("status") == "completed"
            and record.get("completed_rounds_full_video_verified") is True
            and not record.get("unresolved_searches"), "Annotate only a completed, fully reviewed selection")
    frames = source["frames"]
    by_id = {f["frame_id"]: f for f in frames}
    require(len(by_id) == len(frames) == source["expected_video_frames"], "Invalid canonical source frame inventory")
    ids = state["candidate_ids"]
    require(len(ids) == len(set(ids)) and set(ids) <= by_id.keys(), "Selection contains unknown/duplicate candidates")
    require(set(initial["protected_ids"]) <= set(ids), "Invalid protected selection anchors")
    require(state["rounds"] and state["next_round"] == len(state["rounds"]), "Selection lacks completed review receipts")
    require(record == _selection_record(source, state, initial["protected_ids"], selection_run),
            "Final selection differs from the saved source and review state")
    for row in record["rounds"]:
        receipt = row["verification"]
        require(receipt.get("accepted") is True and receipt.get("full_source_video_verified") is True
                and receipt.get("context_truncation_observed") is False
                and receipt.get("video_sha256") == source["video_sha256"]
                and receipt.get("decoded_frame_ids") == list(range(len(frames))),
                "Selection review did not verify the complete video")
    last = state["rounds"][-1]
    original_round = selection_run / "rounds" / f"round-{last['round']:02d}"
    require(Path(last["directory"]).resolve() == original_round.resolve(), "Selection review location changed")
    round_dir = directory / "selection-review" if snapshot else original_round
    video_name = "video" + (Path(source["video_path"]).suffix.lower() or ".mp4")
    require(len(state["messages"]) >= 3 and state["messages"][-1].get("role") == "assistant",
            "Selection lacks its final assistant response")
    parent_config = selection_config_from_saved(parent["config"])
    result = _recover_verified_result(round_dir, source, state["messages"][:-1], ids,
                                     parent_config, video_name)
    raw_message = result["response"]["choices"][0]["message"]
    require(state["messages"][-1] == {"role": "assistant", "content": raw_message["content"]}
            and state["last_output"] == result["output"], "Selection state differs from its verified raw response")
    validate_review(result["output"], ids, source["duration_ms"], review_mode=parent_config.review_mode)
    require(result["output"]["ready"] is True and result["output"]["context_check"] != "conflict",
            "Selection is not ready for annotation")
    selected_ids = record["selected_frame_ids"]
    require(1 <= len(selected_ids) <= 96 and len(set(selected_ids)) == len(selected_ids),
            "Annotation requires 1–96 uniquely selected frames")
    selected = sorted([by_id[f] for f in selected_ids], key=lambda f: (f["timestamp_ms"], f["frame_index"]))
    return parent, source, selected, original_round


def review_loop(source, initial, output, config, runtime, *, state=None, progress=print):
    """Review a frozen set once; retain legacy retrieval replay for historical callers."""
    config.validate()
    output = Path(output)
    by_id = {frame["frame_id"]: frame for frame in source["frames"]}
    media, video_name, names = stage_media(source, output)
    protected = initial["protected_ids"]
    require(len(initial["selected_ids"]) == len(set(initial["selected_ids"]))
            and set(initial["selected_ids"]) <= set(by_id)
            and set(protected) <= set(initial["selected_ids"]), "Invalid initial candidate IDs")
    candidates = [by_id[frame_id] for frame_id in initial["selected_ids"]]
    expected_messages = initial_messages(source, candidates, protected, names, video_name, config)
    if state is None:
        state = {"status": "prepared", "next_round": 0, "candidate_ids": initial["selected_ids"],
                 "introduced_rounds": {frame["frame_id"]: 0 for frame in candidates}, "rounds": [],
                 "messages": expected_messages}
    else:
        state = copy.deepcopy(state)
        require(state["messages"][:2] == expected_messages,
                "Selection prompt changed; begin a separate run with a new output directory")
    if config.review_mode == "keep_drop":
        require(state["candidate_ids"] == initial["selected_ids"]
                and state["introduced_rounds"] == {frame_id: 0 for frame_id in initial["selected_ids"]},
                "Keep/drop candidates must remain the original sampled set")
        require(state["next_round"] == 0 and not state["rounds"] and state["messages"] == expected_messages,
                "Keep/drop selection permits exactly one review of the sampled set")
    _write(output / "state.json", state)
    for round_index in range(state["next_round"], config.max_retrieval_rounds + 1):
        verify_assets(source)
        round_dir = output / "rounds" / f"round-{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        _write(round_dir / "candidate-manifest.json", [by_id[f] for f in state["candidate_ids"]])
        progress(f"Round {round_index + 1}: native video ({source['expected_video_frames']} frames) + {len(state['candidate_ids'])} candidates; processing full context.")
        if (round_dir / "result.json").exists():
            result = _recover_verified_result(round_dir, source, state["messages"], state["candidate_ids"], config, video_name)
        else:
            # Keep any interrupted attempt rather than overwriting its evidence.
            if (round_dir / "request.json").exists():
                attempt = round_dir.with_name(round_dir.name + f"-interrupted-{time.time_ns()}")
                round_dir.rename(attempt)
                round_dir.mkdir()
                _write(round_dir / "candidate-manifest.json", [by_id[f] for f in state["candidate_ids"]])
            result = runtime.chat(state["messages"], schema=review_schema(state["candidate_ids"], source["duration_ms"],
                                                                         review_mode=config.review_mode),
                                  max_tokens=config.max_tokens, round_dir=round_dir)
        review = result["output"]
        validate_review(review, state["candidate_ids"], source["duration_ms"], review_mode=config.review_mode)
        verify_assets(source)
        raw_content = result["response"]["choices"][0]["message"]["content"]
        state["messages"].append({"role": "assistant", "content": raw_content})
        state["last_output"] = review
        state["next_round"] = round_index + 1
        state["rounds"].append({"round": round_index, "directory": str(round_dir),
                                "verification": result["verification"]})
        state["unresolved_searches"] = review["searches"]
        if review["context_check"] == "conflict":
            state["status"] = "context_conflict"
        elif not review["searches"]:
            state["status"] = "completed" if review["ready"] else "model_uncertain"
        elif round_index >= config.max_retrieval_rounds:
            state["status"] = "retrieval_budget_exhausted"
        else:
            capacity = min(config.retrieval_frames, config.max_candidates - len(state["candidate_ids"]))
            if capacity <= 0:
                state["status"] = "candidate_budget_exhausted"
            else:
                added, receipts = retrieve_requests(source, review["searches"], state["candidate_ids"], capacity)
                _write(round_dir / "retrieval.json", receipts)
                if not added:
                    state["status"] = "available_evidence_exhausted"
                else:
                    state["candidate_ids"] = sorted([*state["candidate_ids"], *[f["frame_id"] for f in added]],
                                                    key=lambda frame_id: by_id[frame_id]["frame_index"])
                    state["introduced_rounds"].update({frame["frame_id"]: round_index + 1 for frame in added})
                    state["messages"].append({"role": "user", "content": [
                        {"type": "text", "text": "Retrieved actual evidence for your requests: " + json.dumps(receipts, ensure_ascii=False)},
                        *_image_blocks(added, names),
                        {"type": "text", "text": "Retain the original video context and revise decisions for ALL current candidates: "
                         + json.dumps(state["candidate_ids"]) + ". If evidence is absent or ambiguous, say so. "
                         + f"Remaining retrieval rounds: {config.max_retrieval_rounds - round_index - 1}."},
                    ]})
                    state["status"] = "awaiting_review"
        _write(output / "state.json", state)
        record = _selection_record(source, state, protected, output)
        _write(output / "selection.json", record)
        write_selection_report(record, output)
        progress(f"Round {round_index + 1} saved: {state['status']}.")
        if state["status"] != "awaiting_review":
            return record
    return _selection_record(source, state, protected, output)


def run_selection(input_path, output_dir, config=None, *, released_fps=None, resume=False,
                  prepare_only=False, progress=print, runtime_factory=None):
    from .frame_selection import select_candidates
    from .llama_video import LocalVideoRuntime, RuntimeConfig
    from .video_source import prepare_video_source, validate_native_timeline

    output = Path(output_dir).expanduser().resolve()
    if not resume:
        require(input_path is not None, "An input video or released-frame directory is required")
        source_path = Path(input_path).expanduser().resolve()
        require(source_path.exists(), "Input does not exist")
        require(not output.is_relative_to(source_path if source_path.is_dir() else source_path.parent),
                "Output must be outside the source directory")
        output.mkdir(parents=True, exist_ok=False)
    else:
        require(output.is_dir() and (output / "run.json").is_file(), "No saved selection run found")
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("This selection run is already active") from exc
        try:
            if resume:
                plan = _read(output / "run.json")
                config = selection_config_from_saved(plan["config"])
                source = _read(output / "source" / "source.json")
                initial = _read(output / "initial-selection.json")
                state = _read(output / "state.json") if (output / "state.json").exists() else None
                if input_path is not None:
                    require(str(Path(input_path).expanduser().resolve()) == plan["input_path"], "Resume input differs from saved source")
            else:
                config = config or SelectionConfig()
                config.validate()
                require(config.review_mode == "keep_drop", "New normal selection runs support keep/drop review only")
                progress("Reading every source frame and recording timestamps and file hashes…")
                source = prepare_video_source(source_path, output / "source", released_fps=released_fps)
                progress(f"Prepared complete source: {source['expected_video_frames']} frames. Computing {config.embedding_backend} candidates…")
                initial = select_candidates(source["frames"], min(config.candidate_budget, len(source["frames"])),
                                            embedding_cache=output / "embeddings", backend=config.embedding_backend,
                                            model_path=config.embedding_model_path)
                _write(output / "initial-selection.json", initial)
                _write(output / "run.json", {"schema_version": "smart-frame-selection-run-v1",
                    "created_at": datetime.now(timezone.utc).isoformat(), "input_path": str(source_path),
                    "config": asdict(config), "config_sha256": _config_sha256(asdict(config)),
                    "released_fps": released_fps,
                    "source_manifest_sha256": sha256_file(output / "source" / "source.json"),
                    "timing_policy_sha256": TIMING_POLICY_SHA256,
                    "review_policy_sha256": REVIEW_POLICY_SHA256,
                    "native_video_fps": 0, "temporal_exposure": "retrospective_full_video"})
                state = None
            config.validate()
            verify_assets(source)
            plan = _read(output / "run.json")
            if config.review_mode == "keep_drop":
                require(plan.get("config_sha256") == _config_sha256(plan["config"]), "Selection configuration changed")
            require(sha256_file(output / "source" / "source.json") == plan["source_manifest_sha256"], "Source manifest changed")
            _write(output / "native-timeline-verification.json", validate_native_timeline(source))
            media, video_name, names = stage_media(source, output)
            if state and state["status"] not in {"prepared", "awaiting_review"}:
                record = _selection_record(source, state, initial["protected_ids"], output)
                _write(output / "selection.json", record)
                write_selection_report(record, output)
                return record
            require(plan.get("timing_policy_sha256") == TIMING_POLICY_SHA256,
                    "Selection timing policy changed; begin a separate run with a new output directory")
            require(config.review_mode == "keep_drop" and plan.get("review_policy_sha256") == REVIEW_POLICY_SHA256,
                    "Selection review policy changed; begin a separate keep/drop run with a new output directory")
            if state:
                by_id = {frame["frame_id"]: frame for frame in source["frames"]}
                expected_messages = initial_messages(source,
                    [by_id[frame_id] for frame_id in initial["selected_ids"]],
                    initial["protected_ids"], names, video_name, config)
                require(state["messages"][:2] == expected_messages,
                        "Selection prompt changed; begin a separate run with a new output directory")
            if prepare_only:
                return {"status": "prepared", "output_dir": str(output), "source_frames": source["expected_video_frames"],
                        "candidate_ids": initial["selected_ids"], "native_video_fps": 0}
            # Server logs and native decode receipts are immutable evidence.
            # A resumed run starts a new process and must not reuse the previous
            # process's exclusive log/preflight paths.
            runtime_attempt = output / "runtime" / f"attempt-{time.time_ns()}"
            runtime_config = RuntimeConfig(project_root=PROJECT_ROOT, media_path=media, log_dir=runtime_attempt,
                                           context_size=config.context_size, image_max_tokens=config.image_max_tokens,
                                           request_timeout=config.request_timeout_seconds)
            factory = runtime_factory or LocalVideoRuntime
            with factory(runtime_config, expected_video_frames=source["expected_video_frames"], video_relative_path=video_name) as runtime:
                return review_loop(source, initial, output, config, runtime, state=state, progress=progress)
        except BaseException as exc:
            _write(output / "last-error.json", {"type": type(exc).__name__, "message": str(exc),
                                               "at": datetime.now(timezone.utc).isoformat()})
            raise


def add_selection_parser(subparsers):
    parser = subparsers.add_parser("select-video-frames", help="Joint frame selection with complete native video context and source/timestamp provenance")
    parser.add_argument("--input", type=Path, help="Original video or complete released-frame directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--released-fps", type=float, help="Explicit nominal cadence for a released-frame directory")
    parser.add_argument("--procedure-context", default="", help="Verified source context; do not supply desired outcomes")
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--context-size", type=int, default=131072)
    parser.add_argument("--image-max-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout-seconds", type=float, default=3600,
                        help="Maximum time for the full-video model request")
    parser.add_argument("--embedding-backend", choices=["dinov2", "dinov3"], default="dinov2")
    parser.add_argument("--embedding-model-path")
    parser.add_argument("--prepare-only", action="store_true", help="Save source provenance and embedding candidates without calling Qwen")
    parser.add_argument("--resume", action="store_true", help="Resume using the saved source and configuration")


def selection_cli(args):
    config = None if args.resume else SelectionConfig(
        candidate_budget=args.candidates, max_candidates=args.candidates,
        context_size=args.context_size, image_max_tokens=args.image_max_tokens, max_tokens=args.max_tokens,
        request_timeout_seconds=args.request_timeout_seconds,
        embedding_backend=args.embedding_backend, embedding_model_path=args.embedding_model_path,
        procedure_context=args.procedure_context)
    from .llama_video import VideoRuntimeError
    try:
        result = run_selection(args.input, args.output_dir, config, released_fps=args.released_fps,
                               resume=args.resume, prepare_only=args.prepare_only, progress=lambda text: print(text, flush=True))
    except VideoRuntimeError as error:
        raise ContractError(str(error)) from error
    print(json.dumps({key: result[key] for key in ("status", "selected_frame_ids", "source_frames", "candidate_ids") if key in result}, indent=2))
    print(f"Selection artifacts: {args.output_dir.resolve()}")
