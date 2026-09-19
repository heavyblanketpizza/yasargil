"""Byte-bound provenance for bounded llama.cpp evidence and historical archives.

This verifies retained requests, responses and declared local model metadata. It
does not attest a remote server's execution or establish clinical correctness.
The first adapter supports original still images and outcome-blind causal runs.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import re
from pathlib import Path

from .contract import ContractError, require, resolve_asset

ADAPTER = "llama-cpp-evidence-v1"
LEGACY_ADAPTER = "ollama-evidence-v1"
ENVELOPE_BUILDER_VERSION = "1"
EVENT_QUESTION = "Describe the visible events and uncertainties in the supplied ordered frames."
_PARAMETERS = {
    "adapter", "envelope_builder_version", "stage", "parent_run_ids",
    "start_frame_index", "cutoff_frame_index", "stage_question", "options",
    "model_metadata_asset_id",
}

_LEGACY_PARAMETERS = _PARAMETERS | {"think", "keep_alive"}


def frame_caption(frame):
    """An exact student-visible alias for an original image and its timing."""
    if frame["timestamp_basis"] == "unavailable":
        timing = "acquisition timestamp unavailable"
    else:
        timing = f"timestamp {frame['timestamp_ms']} ms ({frame['timestamp_basis']})"
    return f"{frame['frame_id']}; released frame index {frame['frame_index']}; {timing}."


def canonical_bytes(value):
    """The exact UTF-8 encoding sent to llama.cpp and retained as teacher_request."""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError("Teacher envelope must contain finite JSON values") from exc


def _json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, f"Duplicate teacher JSON key: {key}")
            result[key] = value
        return result

    def bad_constant(value):
        raise ContractError(f"Non-finite teacher JSON value: {value}")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ContractError("Malformed teacher JSON") from exc


def parse_response(raw_bytes, run):
    """Parse complete llama.cpp responses or retained legacy Ollama receipts."""
    response = _json(raw_bytes)
    require(isinstance(response, dict) and "error" not in response,
            "Teacher response is incomplete or failed")
    require(response.get("model") == run["model_name"], "Teacher response model mismatch")
    require(not response.get("truncated"), "Teacher response stopped before completion")
    if run["generation_parameters"].get("adapter") == LEGACY_ADAPTER:
        require(response.get("done") is True, "Teacher response is incomplete or failed")
        require(response.get("done_reason") == "stop", "Teacher response stopped before completion")
        message = response.get("message")
    else:
        choices = response.get("choices")
        require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict),
                "Teacher response must contain one completed choice")
        require(choices[0].get("finish_reason") == "stop", "Teacher response stopped before completion")
        message = choices[0].get("message")
        options = run["generation_parameters"]["options"]
        usage = response.get("usage")
        require(isinstance(usage, dict), "Teacher response is missing token usage")
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        require(type(prompt) is int and 0 < prompt <= options["num_ctx"] - options["num_predict"]
                and type(completion) is int and 0 < completion <= options["num_predict"],
                "Teacher response token usage exceeds the context/output budget")
    require(isinstance(message, dict) and message.get("role") == "assistant",
            "Teacher response must contain an assistant message")
    require(not message.get("tool_calls") and not message.get("thinking")
            and not message.get("reasoning_content") and not message.get("reasoning"),
            "Teacher response contains unrecorded tool use or hidden reasoning")
    require(isinstance(message.get("content"), str), "Teacher content must be JSON text")
    output = _json(message["content"])
    require(isinstance(output, dict), "Teacher content must be a JSON object")
    return output


class _Verifier:
    def __init__(self, record, dataset_root, artifact_root):
        self.record = record
        self.dataset_root, self.artifact_root = dataset_root, artifact_root
        self.assets = {a["asset_id"]: a for a in record["assets"]}
        self.frames = {f["frame_id"]: f for f in record["frame_selection"]["frames"]}
        self.annotations = {a["annotation_id"]: a for a in record["original_annotations"]}
        self.runs = {r["run_id"]: r for r in record["generation_runs"]}
        self.outputs, self.exposures, self.active = {}, {}, set()
        self.source_rows = {}

    def asset_bytes(self, aid, role=None):
        require(aid in self.assets, f"Unknown teacher asset: {aid}")
        asset = self.assets[aid]
        if role:
            require(asset["role"] == role, f"Wrong teacher asset role: {aid}")
        path = resolve_asset(asset["location"], self.dataset_root, self.artifact_root)
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == asset["sha256"],
                f"Teacher asset hash mismatch: {aid}")
        return data

    def parameters(self, run):
        from . import enhancement_prompts as prompts

        params = run["generation_parameters"]
        legacy = params.get("adapter") == LEGACY_ADAPTER
        require(set(params) == (_LEGACY_PARAMETERS if legacy else _PARAMETERS), "Unsupported or missing teacher adapter parameters")
        require(params["adapter"] in {ADAPTER, LEGACY_ADAPTER}
                and params["envelope_builder_version"] == ENVELOPE_BUILDER_VERSION,
                "Unsupported teacher request adapter/version")
        require(run["runtime"] == ("ollama" if legacy else "llama.cpp"),
                "Teacher adapter/runtime mismatch")
        require(run["prompt_version"] == prompts.PROMPT_VERSION, "Unknown teacher prompt version")
        try:
            question = prompts.stage_question(params["stage"])
            prompts.system_prompt(params["stage"])
        except (KeyError, ValueError) as exc:
            raise ContractError("Unknown teacher stage") from exc
        require(params["stage_question"] == question, "Teacher stage question differs from versioned prompt")
        if legacy:
            require(params["think"] is False, "Legacy teacher adapter requires think=false")
            require(isinstance(params["keep_alive"], (str, int))
                    and not isinstance(params["keep_alive"], bool), "Invalid teacher keep_alive")
        require(isinstance(params["options"], dict), "Teacher options must be an object")
        canonical_bytes(params["options"])
        if not legacy:
            options = params["options"]
            require(set(options) == {"temperature", "seed", "num_ctx", "num_predict"},
                    "Unsupported or missing llama.cpp generation options")
            require(type(options["seed"]) is int and type(options["num_ctx"]) is int
                    and options["num_ctx"] > 0 and type(options["num_predict"]) is int
                    and options["num_predict"] > 0 and type(options["temperature"]) in {int, float}
                    and options["temperature"] >= 0, "Invalid llama.cpp generation options")
        start, cutoff = params["start_frame_index"], params["cutoff_frame_index"]
        require(type(start) is int and type(cutoff) is int and 0 <= start <= cutoff,
                "Invalid teacher temporal window")
        require(run["input_mode"] == "causal_prefix" and not run["outcome_ids_seen"],
                "Evidence adapter accepts only outcome-blind causal inputs")
        require(not run["input_reference_asset_ids"] and not run["previous_message_indices"],
                "Teacher references and hidden conversation history are unsupported")
        parents = params["parent_run_ids"]
        require(isinstance(parents, list) and all(isinstance(p, str) for p in parents)
                and len(parents) == len(set(parents)), "Invalid teacher parent runs")
        require(run["run_id"] not in parents, "Teacher run cannot be its own ancestor")
        require(params["stage"] != "independent_observe" or not parents,
                "Independent observation cannot see parent outputs")
        require(all(p in self.runs for p in parents), "Missing teacher ancestor run")
        return params

    def model_metadata(self, run, params):
        metadata = _json(self.asset_bytes(params["model_metadata_asset_id"], "reference_document"))
        require(isinstance(metadata, dict), "Invalid teacher model metadata")
        for key in ("model_name", "model_digest", "quantization", "runtime_version"):
            require(metadata.get(key) == run[key], f"Teacher model metadata mismatch: {key}")
        if params["adapter"] == ADAPTER:
            require(metadata.get("runtime") == "llama.cpp", "Teacher metadata runtime mismatch")
            def fingerprint(value, kind):
                require(isinstance(value, dict) and isinstance(value.get("path"), str)
                        and bool(value["path"]) and isinstance(value.get("sha256"), str)
                        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
                        f"Teacher metadata requires a {kind} fingerprint")
            for kind in ("model_file", "projector_file", "runtime_binary"):
                fingerprint(metadata.get(kind), kind)
            libraries = metadata["runtime_binary"].get("libraries")
            require(isinstance(libraries, list), "Teacher metadata requires runtime library fingerprints")
            for library in libraries:
                fingerprint(library, "runtime library")
            require(len({library["path"] for library in libraries}) == len(libraries),
                    "Duplicate teacher runtime library fingerprint")
            require(metadata["model_digest"] == "sha256:" + metadata["model_file"]["sha256"],
                    "Teacher model digest differs from GGUF receipt")
            require(metadata.get("binary_version") == run["runtime_version"],
                    "Teacher runtime differs from llama.cpp binary receipt")
            require(isinstance(metadata.get("capabilities"), list) and "vision" in metadata["capabilities"],
                    "Teacher model receipt does not establish vision capability")
            return
        tags, show, version = (metadata.get(k) for k in
                               ("tags_response", "show_response", "version_response"))
        require(isinstance(tags, dict) and isinstance(tags.get("models"), list)
                and isinstance(show, dict) and isinstance(version, dict),
                "Teacher metadata requires raw tags/show/version receipts")
        matching = [m for m in tags["models"] if isinstance(m, dict)
                    and (m.get("name") == run["model_name"] or m.get("model") == run["model_name"])]
        require(len(matching) == 1 and matching[0].get("digest") == run["model_digest"],
                "Teacher model digest differs from Ollama tags receipt")
        details = matching[0].get("details", {})
        require(isinstance(details, dict) and details.get("quantization_level", run["quantization"])
                == run["quantization"], "Teacher quantization differs from Ollama tags receipt")
        require(isinstance(show.get("details"), dict)
                and show["details"].get("quantization_level") == run["quantization"],
                "Teacher quantization differs from Ollama show receipt")
        require(isinstance(show.get("capabilities"), list) and "vision" in show["capabilities"],
                "Teacher model receipt does not establish vision capability")
        require(version.get("version") == run["runtime_version"],
                "Teacher runtime differs from Ollama version receipt")

    def annotation_source(self, annotation):
        locator = annotation["source_locator"]
        aid = locator["asset_id"]
        require(locator["locator_type"] == "csv_row_1based_including_header",
                "Teacher annotation locator needs a supported exact CSV row")
        raw = self.asset_bytes(aid, "annotation_table")
        if aid not in self.source_rows:
            try:
                reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
                self.source_rows[aid] = {str(reader.line_num): row for row in reader}
            except (UnicodeDecodeError, csv.Error) as exc:
                raise ContractError("Malformed teacher annotation source") from exc
        require(self.source_rows[aid].get(locator["locator"]) == annotation["raw_value"],
                "Teacher annotation differs from its original source row")
        if self.record["source"]["dataset_name"] == "SOSpine":
            table = {"keypoint": "sospine_tool_tips.csv", "bbox": "sospine_bbox.csv"}.get(annotation["original_kind"])
            require(table and self.assets[aid]["location"] == table,
                    "Teacher SOSpine annotation/table mismatch")
            require(all(annotation["raw_value"].get("trial_frame") ==
                        Path(self.assets[self.frames[f]["asset_id"]]["location"]).name
                        for f in annotation["frame_ids"]), "Teacher annotation/frame source join mismatch")

    def request(self, run):
        from . import enhancement_prompts as prompts

        params = self.parameters(run)
        self.model_metadata(run, params)
        parent_outputs, parent_frames, parent_annotations = [], set(), set()
        for rid in params["parent_run_ids"]:
            output = self.verify(rid)
            parent = self.runs[rid]
            require(parent["generation_parameters"]["cutoff_frame_index"] <= params["cutoff_frame_index"],
                    "Teacher ancestor used a future cutoff")
            parent_frames.update(self.exposures[rid][0])
            parent_annotations.update(self.exposures[rid][1])
            parent_outputs.append({"run_id": rid, "response_asset_id": parent["response_asset_id"],
                                   "output": output})
        frame_ids, annotation_ids = run["input_frame_ids"], run["input_annotation_ids"]
        require(frame_ids and len(frame_ids) == len(set(frame_ids))
                and set(frame_ids) <= self.frames.keys(), "Invalid teacher input frames")
        require(len(annotation_ids) == len(set(annotation_ids))
                and set(annotation_ids) <= self.annotations.keys(), "Invalid teacher input annotations")
        require([self.frames[f]["frame_index"] for f in frame_ids]
                == sorted(self.frames[f]["frame_index"] for f in frame_ids),
                "Teacher input frames must be chronological")
        frames, images, annotations = [], [], []
        for fid in frame_ids:
            frame = self.frames[fid]
            require(params["start_frame_index"] <= frame["frame_index"] <= params["cutoff_frame_index"],
                    "Teacher input frame is outside its window")
            asset = self.assets[frame["asset_id"]]
            require(asset["origin"] == "original_dataset" and asset["role"] == "released_frame"
                    and not asset["derived_from_asset_ids"] and asset["transformation_id"] is None,
                    "Teacher adapter requires original untransformed released frames")
            require(asset["media_type"].startswith("image/"), "Teacher frame is not an image")
            images.append(base64.b64encode(self.asset_bytes(frame["asset_id"])).decode("ascii"))
            frames.append({key: frame[key] for key in
                           ("frame_id", "asset_id", "frame_index", "timestamp_ms", "timestamp_basis")}
                          | {"sha256": asset["sha256"]})
        for aid in annotation_ids:
            annotation = self.annotations[aid]
            require(annotation["original_kind"] in {"keypoint", "bbox", "segmentation", "instrument",
                    "anatomy", "action", "workflow"}, "Teacher received outcome or nonvisual case metadata")
            require(annotation["frame_ids"] and set(annotation["frame_ids"]) <= set(frame_ids),
                    "Teacher annotation concerns an unsupplied frame")
            self.annotation_source(annotation)
            annotations.append({key: annotation[key] for key in
                                ("annotation_id", "source_locator", "original_kind", "original_origin",
                                 "raw_value", "frame_ids")})
        exposed_frames = set(frame_ids) | parent_frames
        exposed_annotations = set(annotation_ids) | parent_annotations
        maximum = max(self.frames[f]["frame_index"] for f in exposed_frames)
        require(maximum <= params["cutoff_frame_index"]
                and run["maximum_frame_index_seen"] == maximum,
                "Teacher exposure maximum differs from actual transitive inputs")
        self.exposures[run["run_id"]] = (exposed_frames, exposed_annotations)
        payload = {"adapter": params["adapter"], "envelope_builder_version": ENVELOPE_BUILDER_VERSION,
                   "stage": params["stage"], "question": params["stage_question"],
                   "window": {"start_frame_index": params["start_frame_index"],
                              "cutoff_frame_index": params["cutoff_frame_index"]},
                   "frames": frames, "original_annotations": annotations, "parent_outputs": parent_outputs}
        messages = [{"role": "system", "content": prompts.system_prompt(params["stage"])},
                    {"role": "user", "content": canonical_bytes(payload).decode("utf-8"), "images": images}]
        if params["adapter"] == LEGACY_ADAPTER:
            return canonical_bytes({"model": run["model_name"], "stream": False,
                                    "format": prompts.response_schema(), "options": params["options"],
                                    "think": params["think"], "keep_alive": params["keep_alive"],
                                    "messages": messages})
        from .llama_cpp import build_chat_request
        return canonical_bytes(build_chat_request(model=run["model_name"], messages=messages,
                                                   schema=prompts.response_schema(), **params["options"]))


    def verify(self, rid):
        require(rid in self.runs, "Missing teacher ancestor run")
        require(rid not in self.active, "Cyclic teacher parent runs")
        if rid in self.outputs:
            return self.outputs[rid]
        self.active.add(rid)
        try:
            run = self.runs[rid]
            expected = self.request(run)
            require(self.asset_bytes(run["request_asset_id"], "teacher_request") == expected,
                    f"Teacher request bytes differ from declared evidence envelope: {rid}")
            output = parse_response(self.asset_bytes(run["response_asset_id"], "teacher_response"), run)
            from .enhancement_prompts import validate_output
            try:
                validate_output(output, self.exposures[rid][0], self.exposures[rid][1],
                                run["generation_parameters"]["start_frame_index"],
                                run["generation_parameters"]["cutoff_frame_index"])
            except ValueError as exc:
                raise ContractError(f"Invalid teacher structured output: {exc}") from exc
            self.outputs[rid] = output
            return output
        finally:
            self.active.remove(rid)


def build_request(record, run, *, dataset_root, artifact_root=None):
    """Build a new request, verifying retained ancestors before including them.

    The current run may be a not-yet-appended draft receipt. Its request/response
    artifacts do not need to exist yet; every ancestor artifact must exist.
    """
    require(run["generation_parameters"].get("adapter") == ADAPTER,
            "New teacher requests require llama.cpp; legacy Ollama archives are read-only")
    verifier = _Verifier(record, dataset_root, artifact_root)
    verifier.active.add(run["run_id"])
    return verifier.request(run)


def _bind_outputs(record, verifier):
    for claim in record["claims"]:
        rid = claim["generation_run_id"]
        if rid not in verifier.outputs:
            continue
        output = verifier.outputs[rid]
        candidates = []
        for event in output["events"]:
            candidates.append((f"{rid}.{event['event_id']}", event["description"], event["type"],
                               event["evidence_frame_ids"], event["annotation_ids"]))
            if event["uncertainty"].strip():
                candidates.append((f"{rid}.{event['event_id']}.uncertainty", event["uncertainty"],
                                   "uncertainty_statement", event["evidence_frame_ids"], event["annotation_ids"]))
        for question in output["questions"]:
            kind = "uncertainty_statement" if question["answerability"] == "uncertain" else "visible_observation"
            candidates.append((f"{rid}.{question['question_id']}.answer", question["answer"], kind,
                               question["evidence_frame_ids"], question["annotation_ids"]))
        matching = [entry for entry in candidates if entry[0] == claim["claim_id"]]
        require(len(matching) == 1, "Generated claim ID is absent from teacher response")
        _, text, kind, frames, annotations = matching[0]
        evidence = claim["evidence"]
        require(claim["origin"] == "model_generated" and claim["text"] == text and claim["type"] == kind,
                "Generated claim text/type differs from teacher response")
        require(set(evidence["frame_ids"]) == set(frames)
                and set(evidence["original_annotation_ids"]) == set(annotations)
                and not evidence["supporting_claim_ids"] and not evidence["reference_asset_ids"]
                and not evidence["regions"] and claim["transformation_id"] is None,
                "Generated claim evidence differs from teacher response")
    messages = record["training_view"]["messages"]
    for turn in record["training_view"]["turn_links"]:
        run_ids = set(turn["generation_run_ids"]) & verifier.outputs.keys()
        if not run_ids:
            continue
        blocks = messages[turn["user_message_index"]]["content"]
        require(blocks and blocks[0].get("type") == "text", "Teacher-bound turn needs a question first")
        question_text = blocks[0]["text"]
        by_location = {verifier.assets[f["asset_id"]]["location"]: f for f in verifier.frames.values()}
        require(len(blocks[1:]) % 2 == 0, "Teacher-bound frame aliases must precede each image")
        for index in range(1, len(blocks), 2):
            caption, image = blocks[index:index + 2]
            require(caption.get("type") == "text" and image.get("type") == "image"
                    and image.get("image") in by_location, "Teacher-bound frame alias/image pair is invalid")
            require(caption["text"] == frame_caption(by_location[image["image"]]),
                    "Teacher-bound frame alias differs from source metadata")
        if turn["question_origin"] == "model_generated":
            matches = [(rid, q) for rid in run_ids for q in verifier.outputs[rid]["questions"]
                       if q["question"] == question_text
                       and f"{rid}.{q['question_id']}.answer" in turn["expressed_claim_ids"]]
            require(matches, "Generated question/answer differs from teacher response")
        else:
            require(turn["question_origin"] == "deterministic_template" and question_text == EVENT_QUESTION,
                    "Unsupported teacher-bound deterministic question")
        for rid in run_ids:
            require(verifier.exposures[rid][0] <= set(turn["student_frame_ids"]),
                    "Student lacks teacher ancestor frame context")
            require(verifier.runs[rid]["generation_parameters"]["cutoff_frame_index"] <= turn["cutoff_frame_index"],
                    "Teacher bound to an earlier student cutoff")


def validate_teacher_runs(record, *, dataset_root=None, artifact_root=None):
    """Return exactly those adapter run IDs whose bytes and bindings verified.

    Unsupported adapters remain archiveable, but are never returned as verified.
    Without source roots this performs no byte-verification and grants no gate.
    """
    if dataset_root is None:
        return set()
    verifier = _Verifier(record, dataset_root, artifact_root)
    for rid, run in verifier.runs.items():
        if run["generation_parameters"].get("adapter") in {ADAPTER, LEGACY_ADAPTER}:
            verifier.verify(rid)
    _bind_outputs(record, verifier)
    return set(verifier.outputs)
