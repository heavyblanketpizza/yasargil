"""Annotation isolation, source provenance, and recovery with real tiny video."""
import copy
import argparse
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image

from yasargil.contract import ContractError, sha256_file
from yasargil.frame_annotation import (
    AnnotationConfig, LEGACY_PROTOCOL_VERSION, PROTOCOL_VERSION,
    _annotation_schema, _build_annotations, _messages, _protocol_hash,
    add_annotation_parser, annotation_status, request_annotation_pause, run_annotation,
)
from yasargil.smart_selection import SelectionConfig, review_loop
from yasargil.video_source import prepare_video_source


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value))


class AcceptedResultInterruption(RuntimeError):
    pass


def publish_result(messages, schema, max_tokens, round_dir, output, frame_count):
    """Save the immutable transport receipts used by real native inference."""
    request = {"messages": copy.deepcopy(messages), "max_tokens": max_tokens,
               "response_format": {"type": "json_schema", "json_schema": {"schema": schema}}}
    response = {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": json.dumps(output)}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}
    overview = json.loads(messages[1]["content"][0]["text"])
    video_url = next(block["input_video"]["url"] for block in messages[1]["content"]
                     if block["type"] == "input_video")
    round_dir.mkdir(parents=True, exist_ok=True)
    write(round_dir / "request.json", request)
    verification = {
        "accepted": True, "full_source_video_verified": True,
        "context_truncation_observed": False, "finish_reason": "stop",
        "video_sha256": overview["video_sha256"],
        "video_relative_path": video_url.removeprefix("file://"),
        "video_fps_setting": 0, "expected_video_frames": frame_count,
        "decoded_frame_ids": list(range(frame_count)), "decoded_frames": frame_count,
        "request_sha256": sha256_file(round_dir / "request.json"),
    }
    result = {"output": output, "response": response, "verification": verification}
    for name, value in (("response.json", response), ("verification.json", verification),
                        ("output.json", output), ("result.json", result)):
        write(round_dir / name, value)
    return result


class SelectionRuntime:
    def __init__(self, kept_ids, frame_count):
        self.kept_ids, self.frame_count = set(kept_ids), frame_count

    def chat(self, messages, *, schema, max_tokens, round_dir):
        ids = schema["properties"]["decisions"]["required"]
        output = {
            "scene_summary": "PRIVATE_SELECTOR_SCENE_DESCRIPTION",
            "context_check": "consistent", "ready": True, "searches": [],
            "decisions": {key: {"decision": "keep" if key in self.kept_ids else "drop",
                                "reason": "PRIVATE_SELECTOR_KEEP_DROP_REASON"} for key in ids},
        }
        return publish_result(messages, schema, max_tokens, round_dir, output, self.frame_count)


def annotation_answer(ids):
    return {"context_check": "consistent", "annotations": {
        frame_id: {
            "visible_observation": "A solid color fills the visible image.",
            "visibility": "clear",
            "contextual_claims": [{
                "claim": "The surrounding sequence contains a change of background color.",
                "evidence_intervals": [{"start_frame_id": ids[0], "end_frame_id": ids[min(1, len(ids) - 1)]}],
            }],
            "uncertainties": ["This synthetic sequence contains no anatomy or surgical action."],
        } for frame_id in ids}}


def make_legacy_annotation(output):
    """Construct a completed historical fixture without a legacy inference path."""
    plan, source = read(output / "run.json"), read(output / "source/source.json")
    plan.update(schema_version=LEGACY_PROTOCOL_VERSION, protocol_sha256=_protocol_hash(LEGACY_PROTOCOL_VERSION))
    write(output / "run.json", plan)
    config = AnnotationConfig(**plan["config"])
    selected = read(output / "selected-frames.json")
    aliases = {frame["frame_id"]: f"frame-{index:08d}{Path(frame['image_path']).suffix.lower()}"
               for index, frame in enumerate(source["frames"])}
    video_name = "video" + Path(source["video_path"]).suffix.lower()
    messages = _messages(source, selected, aliases, video_name, config, protocol_version=LEGACY_PROTOCOL_VERSION)
    canonical = {frame["frame_id"]: frame for frame in source["frames"]}
    raw = read(output / "round-00/output.json")
    for row in raw["annotations"].values():
        for claim in row["contextual_claims"]:
            for interval in claim["evidence_intervals"]:
                first, last = interval.pop("start_frame_id"), interval.pop("end_frame_id")
                interval.update(start_ms=canonical[first]["timestamp_ms"], end_ms=canonical[last]["timestamp_ms"])
    result = publish_result(messages, _annotation_schema(plan, source), config.max_tokens,
                            output / "round-00", raw, source["expected_video_frames"])
    document = read(output / "annotations.json")
    document.update(_build_annotations(raw, selected, source, plan), verification=result["verification"],
                    model_result_sha256=sha256_file(output / "round-00/result.json"))
    write(output / "annotations.json", document)
    summary = read(output / "summary.json")
    summary["schema_version"] = LEGACY_PROTOCOL_VERSION
    write(output / "summary.json", summary)


class AnnotationRuntime:
    def __init__(self, owner, config, *, expected_video_frames, video_relative_path):
        self.owner, self.config = owner, config
        self.frame_count, self.video_relative_path = expected_video_frames, video_relative_path
        self.requests, self.closed = [], False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True
        return False

    def chat(self, messages, *, schema, max_tokens, round_dir):
        self.requests.append(copy.deepcopy(messages))
        self.owner.calls.append((copy.deepcopy(messages), copy.deepcopy(schema), max_tokens))
        ids = schema["properties"]["annotations"]["required"]
        output = self.owner.answer(ids)
        result = publish_result(messages, schema, max_tokens, round_dir, output, self.frame_count)
        if self.owner.interrupt_after_save:
            raise AcceptedResultInterruption("Accepted annotation response saved before interruption")
        return result


class RuntimeFactory:
    def __init__(self, answer=annotation_answer, *, interrupt_after_save=False):
        self.answer, self.interrupt_after_save = answer, interrupt_after_save
        self.sessions, self.calls = [], []

    def __call__(self, config, **kwargs):
        runtime = AnnotationRuntime(self, config, **kwargs)
        self.sessions.append(runtime)
        return runtime


class AnnotationConfigTests(unittest.TestCase):
    def test_timeout_accepts_positive_numbers_and_rejects_bool_nonfinite_or_nonpositive(self):
        self.assertEqual(AnnotationConfig().request_timeout_seconds, 3600)
        for timeout in (21600, 1.5):
            AnnotationConfig(request_timeout_seconds=timeout).validate()
        for timeout in (True, False, 0, -1, float("nan"), float("inf"), "21600", None):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ContractError, "request timeout"):
                AnnotationConfig(request_timeout_seconds=timeout).validate()

    def test_cli_parses_numeric_annotation_timeout(self):
        parser = argparse.ArgumentParser()
        add_annotation_parser(parser.add_subparsers(dest="command", required=True))
        args = parser.parse_args(["annotate-video-frames", "--output-dir", "annotation",
                                  "--request-timeout-seconds", "21600.5"])
        self.assertEqual(args.request_timeout_seconds, 21600.5)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Native provenance fixtures need FFmpeg")
class FrameAnnotationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.release = self.root / "released"
        self.release.mkdir()
        for index, color in enumerate(("red", "green", "blue", "yellow", "orange", "purple", "black", "white"), 1):
            Image.new("RGB", (48, 32), color).save(self.release / f"frame_{index:08d}.jpeg")
        self.make_selection("selection", review_mode="retrieval")
        self.output = self.root / "annotation"
        self.config = AnnotationConfig(context_size=16384, image_max_tokens=256,
                                       max_tokens=2048, max_evidence_span_ms=2000)

    def make_selection(self, directory, *, review_mode):
        self.selection = self.root / directory
        self.selection.mkdir()
        self.source = prepare_video_source(self.release, self.selection / "source", released_fps=1)
        self.ids = [frame["frame_id"] for frame in self.source["frames"]]
        self.selected_ids = [self.ids[index] for index in (0, 2, 5, 7)]
        self.selection_config = SelectionConfig(
            candidate_budget=8, max_candidates=12, review_mode=review_mode,
            max_retrieval_rounds=1 if review_mode == "retrieval" else 0,
            procedure_context="A synthetic eight-frame color sequence, not surgery.")
        initial = {"selected_ids": self.ids, "protected_ids": [self.ids[0]]}
        write(self.selection / "run.json", {
            "schema_version": "smart-frame-selection-run-v1",
            "input_path": str(self.release), "config": asdict(self.selection_config),
            "source_manifest_sha256": sha256_file(self.selection / "source/source.json")})
        write(self.selection / "initial-selection.json", initial)
        review_loop(self.source, initial, self.selection, self.selection_config,
                    SelectionRuntime(self.selected_ids[1:], len(self.ids)), progress=lambda _: None)

    def run_pass(self, factory, **kwargs):
        return run_annotation(None if kwargs.get("resume") else self.selection,
                              self.output, self.config, runtime_factory=factory,
                              progress=lambda _: None, **kwargs)

    def test_annotation_has_fresh_full_video_context_and_exact_effective_selected_set(self):
        original = {path: path.read_bytes() for path in self.selection.rglob("*.json")}
        factory = RuntimeFactory()
        summary = self.run_pass(factory)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(len(factory.sessions), 1)
        self.assertEqual(len(factory.calls), 1)
        self.assertTrue(factory.sessions[0].closed)
        messages, schema, max_tokens = factory.calls[0]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(schema["properties"]["annotations"]["required"], self.selected_ids)
        self.assertEqual(max_tokens, self.config.max_tokens)
        runtime = factory.sessions[0]
        self.assertEqual(runtime.frame_count, 8)
        self.assertEqual(runtime.config.context_size, self.config.context_size)
        self.assertEqual(runtime.config.image_max_tokens, self.config.image_max_tokens)
        self.assertEqual(runtime.config.request_timeout, self.config.request_timeout_seconds)
        blocks = messages[1]["content"]
        self.assertEqual(sum(block["type"] == "input_video" for block in blocks), 1)
        self.assertEqual(sum(block["type"] == "image_url" for block in blocks), 4)
        overview = json.loads(blocks[0]["text"])
        self.assertEqual(overview["complete_video_frame_count"], 8)
        self.assertEqual(overview["video_sha256"], self.source["video_sha256"])
        self.assertEqual(overview["evidence_frame_inventory"], {
            "columns": ["frame_id", "timestamp_ms"],
            "rows": [[frame["frame_id"], frame["timestamp_ms"]] for frame in self.source["frames"]]})
        self.assertEqual(schema["$defs"]["source_frame_id"]["enum"], self.ids)
        serialized = json.dumps(messages)
        self.assertIn(self.selection_config.procedure_context, serialized)
        for secret in ("PRIVATE_SELECTOR_SCENE_DESCRIPTION", "PRIVATE_SELECTOR_KEEP_DROP_REASON",
                       '"model_reason"', '"effective_decision"', '"protected_temporal_anchor"'):
            self.assertNotIn(secret, serialized)
        for frame in self.source["frames"]:
            if frame["frame_id"] in self.selected_ids:
                for key in ("frame_id", "source_path", "source_sha256", "image_path", "image_sha256"):
                    self.assertIn(frame[key], serialized)
        self.assertEqual(original, {path: path.read_bytes() for path in original})
        for name in ("run.json", "session.json", "source/source.json", "selection.json",
                     "selected-frames.json", "annotations.json", "summary.json", "report.html"):
            self.assertTrue((self.output / name).is_file(), name)
        draft = read(self.output / "annotations.json")
        self.assertEqual(draft["schema_version"], "contextual-frame-annotations-v2")
        self.assertEqual(read(self.output / "run.json")["schema_version"], PROTOCOL_VERSION)
        self.assertFalse(draft["training_eligible"])
        self.assertEqual([frame["frame_id"] for frame in draft["annotations"]], self.selected_ids)
        canonical = {frame["frame_id"]: frame for frame in self.source["frames"]}
        for frame in draft["annotations"]:
            self.assertTrue(frame["review_required"])
            for key, value in canonical[frame["frame_id"]].items():
                self.assertEqual(frame[key], value, key)
            interval = frame["contextual_claims"][0]["evidence_intervals"][0]
            self.assertEqual(interval["supporting_frames"], self.source["frames"][:3])

    def test_prepare_pins_inputs_without_loading_a_model(self):
        factory = RuntimeFactory()
        summary = self.run_pass(factory, prepare_only=True)
        self.assertEqual(summary["status"], "prepared")
        self.assertEqual(factory.sessions, [])
        self.assertFalse((self.output / "annotations.json").exists())
        self.assertEqual(read(self.output / "selection.json"), read(self.selection / "selection.json"))
        self.assertEqual(annotation_status(self.output)["status"], "prepared")
        self.assertEqual(self.run_pass(factory, resume=True)["status"], "completed")
        self.assertEqual(len(factory.calls), 1)

    def test_long_video_runtime_timeout_is_pinned_and_propagated_after_prepare(self):
        self.config = replace(self.config, context_size=262144, image_max_tokens=256,
                              max_tokens=12288, request_timeout_seconds=21600)
        factory = RuntimeFactory()
        self.run_pass(factory, prepare_only=True)
        self.assertEqual(factory.sessions, [])
        for name in ("run.json", "session.json"):
            self.assertEqual(read(self.output / name)["config"]["request_timeout_seconds"], 21600)
        plan = read(self.output / "run.json")
        self.assertEqual(plan["input_sha256"]["session.json"], sha256_file(self.output / "session.json"))
        self.assertEqual(self.run_pass(factory, resume=True)["status"], "completed")
        runtime = factory.sessions[0].config
        self.assertEqual(runtime.request_timeout, 21600)
        self.assertEqual(runtime.context_size, 262144)
        self.assertEqual(runtime.image_max_tokens, 256)
        self.assertEqual(factory.calls[0][2], 12288)
        self.assertEqual(read(self.output / "round-00/annotation-context.json")["config"]["request_timeout_seconds"], 21600)

    def test_historical_annotation_without_explicit_timeout_resumes_without_rewriting_evidence(self):
        self.run_pass(RuntimeFactory())
        plan_path, session_path = self.output / "run.json", self.output / "session.json"
        plan, session = read(plan_path), read(session_path)
        plan["config"].pop("request_timeout_seconds")
        session["config"].pop("request_timeout_seconds")
        write(session_path, session)
        plan["input_sha256"]["session.json"] = sha256_file(session_path)
        write(plan_path, plan)
        # Match the artifact layout written before timeout was an annotation
        # setting; neither the original raw result nor derived drafts change.
        write(self.output / "round-00/annotation-context.json", session)
        preserved = {name: (self.output / name).read_bytes() for name in
                     ("run.json", "session.json", "round-00/annotation-context.json", "round-00/request.json",
                      "round-00/response.json", "round-00/result.json", "annotations.json")}
        resumed = RuntimeFactory()
        self.assertEqual(self.run_pass(resumed, resume=True)["status"], "completed")
        self.assertEqual(resumed.sessions, [])
        self.assertEqual(preserved, {name: (self.output / name).read_bytes() for name in preserved})

    def test_historical_prepared_annotation_uses_original_default_timeout(self):
        self.run_pass(RuntimeFactory(), prepare_only=True)
        plan_path, session_path = self.output / "run.json", self.output / "session.json"
        plan, session = read(plan_path), read(session_path)
        plan["config"].pop("request_timeout_seconds")
        session["config"].pop("request_timeout_seconds")
        write(session_path, session)
        plan["input_sha256"]["session.json"] = sha256_file(session_path)
        write(plan_path, plan)
        original_session = session_path.read_bytes()
        resumed = RuntimeFactory()
        self.assertEqual(self.run_pass(resumed, resume=True)["status"], "completed")
        self.assertEqual(resumed.sessions[0].config.request_timeout, 3600)
        self.assertEqual(session_path.read_bytes(), original_session)

    def test_annotation_accepts_frozen_keep_drop_selection_and_its_exact_saved_schema(self):
        self.make_selection("keep-drop-selection", review_mode="keep_drop")
        request = read(self.selection / "rounds/round-00/request.json")
        self.assertEqual(request["response_format"]["json_schema"]["schema"]["properties"]["searches"]["maxItems"], 0)
        selected = read(self.selection / "selection.json")
        self.assertEqual(selected["selected_frame_ids"], self.selected_ids)
        self.assertEqual(selected["frames"][0]["model_decision"], "drop")
        self.assertEqual(selected["frames"][0]["effective_decision"], "keep")
        factory = RuntimeFactory()
        self.assertEqual(self.run_pass(factory)["status"], "completed")
        messages, schema, _ = factory.calls[0]
        self.assertEqual(schema["properties"]["annotations"]["required"], self.selected_ids)
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(sum(block["type"] == "image_url" for block in messages[1]["content"]), 4)
        self.assertEqual(read(self.output / "selection-review/request.json"), request)
        self.assertEqual(read(self.output / "selection-run.json")["config"]["review_mode"], "keep_drop")
        self.assertEqual([frame["frame_id"] for frame in read(self.output / "annotations.json")["annotations"]], self.selected_ids)
        resumed = RuntimeFactory()
        self.assertEqual(self.run_pass(resumed, resume=True)["status"], "completed")
        self.assertEqual(resumed.sessions, [])

    def test_annotation_accepts_historical_selection_config_without_review_mode(self):
        path = self.selection / "run.json"
        historical = read(path)
        historical["config"].pop("review_mode")
        historical["config"].pop("request_timeout_seconds")
        write(path, historical)
        request = read(self.selection / "rounds/round-00/request.json")
        self.assertEqual(request["response_format"]["json_schema"]["schema"]["properties"]["searches"]["maxItems"], 3)
        factory = RuntimeFactory()
        self.assertEqual(self.run_pass(factory)["status"], "completed")
        self.assertEqual(read(self.output / "selection-run.json"), historical)
        self.assertEqual(read(self.output / "selection-review/request.json"), request)
        self.assertEqual(factory.calls[0][1]["properties"]["annotations"]["required"], self.selected_ids)
        resumed = RuntimeFactory()
        self.assertEqual(self.run_pass(resumed, resume=True)["status"], "completed")
        self.assertEqual(resumed.sessions, [])

    def test_completed_resume_verifies_receipts_without_new_inference(self):
        self.run_pass(RuntimeFactory())
        before = {name: (self.output / name).read_bytes() for name in
                  ("session.json", "round-00/request.json", "round-00/response.json", "round-00/result.json", "annotations.json")}
        resumed = RuntimeFactory()
        self.assertEqual(self.run_pass(resumed, resume=True)["status"], "completed")
        self.assertEqual(resumed.sessions, [])
        self.assertEqual(before, {name: (self.output / name).read_bytes() for name in before})

    def test_completed_legacy_annotation_remains_verifiable_without_rewriting_raw_evidence(self):
        self.run_pass(RuntimeFactory())
        make_legacy_annotation(self.output)
        before = {name: (self.output / name).read_bytes() for name in
                  ("run.json", "round-00/request.json", "round-00/response.json", "annotations.json")}
        resumed = RuntimeFactory()
        self.assertEqual(self.run_pass(resumed, resume=True)["status"], "completed")
        self.assertEqual(resumed.sessions, [])
        self.assertEqual(before, {name: (self.output / name).read_bytes() for name in before})

    def test_legacy_prepared_pass_cannot_start_new_timestamp_generation(self):
        self.run_pass(RuntimeFactory(), prepare_only=True)
        plan = read(self.output / "run.json")
        plan.update(schema_version=LEGACY_PROTOCOL_VERSION, protocol_sha256=_protocol_hash(LEGACY_PROTOCOL_VERSION))
        write(self.output / "run.json", plan)
        resumed = RuntimeFactory()
        with self.assertRaisesRegex(ContractError, "Legacy timestamp generation cannot resume"):
            self.run_pass(resumed, resume=True)
        self.assertEqual(resumed.sessions, [])

    def test_single_last_frame_evidence_has_its_exact_time_and_survives_resume(self):
        def last_frame(ids):
            raw = annotation_answer(ids)
            for row in raw["annotations"].values():
                row["contextual_claims"][0]["evidence_intervals"] = [
                    {"start_frame_id": self.ids[-1], "end_frame_id": self.ids[-1]}]
            return raw
        self.run_pass(RuntimeFactory(last_frame))
        interval = read(self.output / "annotations.json")["annotations"][0]["contextual_claims"][0]["evidence_intervals"][0]
        self.assertEqual((interval["start_ms"], interval["end_ms"]), (7000, 7000))
        self.assertEqual(interval["supporting_frames"], self.source["frames"][-1:])
        self.assertEqual(self.run_pass(RuntimeFactory(), resume=True)["status"], "completed")

    def test_accepted_response_recovers_after_crash_without_repeating_inference(self):
        interrupted = RuntimeFactory(interrupt_after_save=True)
        with self.assertRaises(AcceptedResultInterruption):
            self.run_pass(interrupted)
        self.assertTrue(interrupted.sessions[0].closed)
        self.assertTrue((self.output / "round-00/result.json").is_file())
        self.assertFalse((self.output / "annotations.json").exists())
        resumed = RuntimeFactory()
        self.assertEqual(self.run_pass(resumed, resume=True)["status"], "completed")
        self.assertEqual(resumed.sessions, [])

    def test_recovery_rejects_internally_inconsistent_native_frame_count(self):
        with self.assertRaises(AcceptedResultInterruption):
            self.run_pass(RuntimeFactory(interrupt_after_save=True))
        directory = self.output / "round-00"
        result = read(directory / "result.json")
        result["verification"]["decoded_frames"] = 7
        write(directory / "result.json", result)
        write(directory / "verification.json", result["verification"])
        resumed = RuntimeFactory()
        with self.assertRaises(ContractError):
            self.run_pass(resumed, resume=True)
        self.assertEqual(resumed.sessions, [])
        self.assertFalse((self.output / "annotations.json").exists())

    def test_resume_rejects_changed_frozen_selection_and_source_manifest(self):
        self.run_pass(RuntimeFactory(), prepare_only=True)
        paths = [self.output / name for name in
                 ("selection.json", "source/source.json", "selected-frames.json", "session.json")]
        for path in paths:
            with self.subTest(path=path.name):
                before = path.read_bytes()
                path.write_bytes(before + b"\n")
                resumed = RuntimeFactory()
                with self.assertRaises(ContractError):
                    self.run_pass(resumed, resume=True)
                self.assertEqual(resumed.sessions, [])
                path.write_bytes(before)

    def test_frozen_snapshot_survives_later_changes_to_original_selection_prose(self):
        self.run_pass(RuntimeFactory(), prepare_only=True)
        parent = read(self.selection / "selection.json")
        parent["scene_summary"] = "A later edited parent description must not replace frozen inputs."
        write(self.selection / "selection.json", parent)
        factory = RuntimeFactory()
        self.assertEqual(self.run_pass(factory, resume=True)["status"], "completed")
        self.assertNotIn(parent["scene_summary"], json.dumps(factory.calls[0][0]))
        self.assertEqual(read(self.output / "selection.json")["scene_summary"], "PRIVATE_SELECTOR_SCENE_DESCRIPTION")

    def test_source_bytes_changed_after_prepare_stop_before_model_load(self):
        self.run_pass(RuntimeFactory(), prepare_only=True)
        frame = self.source["frames"][2]
        Image.new("RGB", (48, 32), "pink").save(frame["source_path"])
        factory = RuntimeFactory()
        with self.assertRaises(ContractError):
            self.run_pass(factory, resume=True)
        self.assertEqual(factory.sessions, [])

    def test_completed_resume_rejects_altered_request_result_and_derived_annotations(self):
        self.run_pass(RuntimeFactory())
        round_dir = self.output / "round-00"
        paths = [round_dir / "request.json", round_dir / "result.json", self.output / "annotations.json"]
        original = {path: path.read_bytes() for path in paths}
        def change_request():
            value = read(paths[0])
            value["messages"][0]["content"] += " Altered instructions."
            write(paths[0], value)
        def change_result():
            value = read(paths[1])
            value["verification"]["decoded_frame_ids"][-1] = 6
            write(paths[1], value)
        def change_derived():
            text = paths[2].read_text()
            self.assertIn("A solid color fills the visible image.", text)
            paths[2].write_text(text.replace("A solid color fills the visible image.", "An invented surgical event."))
        for change in (change_request, change_result, change_derived):
            with self.subTest(change=change.__name__):
                for path, data in original.items():
                    path.write_bytes(data)
                change()
                resumed = RuntimeFactory()
                with self.assertRaises(ContractError):
                    self.run_pass(resumed, resume=True)
                self.assertEqual(resumed.sessions, [])
        for path, data in original.items():
            path.write_bytes(data)

    def test_resume_cannot_relabel_saved_inference_with_different_runtime_config(self):
        self.run_pass(RuntimeFactory())
        path = self.output / "run.json"
        original = read(path)
        for key, value in (("context_size", 32768), ("image_max_tokens", 512), ("request_timeout_seconds", 21600)):
            with self.subTest(key=key):
                modified = copy.deepcopy(original)
                modified["config"][key] = value
                write(path, modified)
                resumed = RuntimeFactory()
                with self.assertRaises(ContractError):
                    self.run_pass(resumed, resume=True)
                self.assertEqual(resumed.sessions, [])
        write(path, original)

    def test_invalid_evidence_interval_never_publishes_final_annotations(self):
        cases = [(1000, 1000), (2000, 1000), (0, 7000), (7000, 9000)]
        for index, (start, end) in enumerate(cases):
            with self.subTest(start=start, end=end):
                self.output = self.root / f"invalid-interval-{index}"
                def invalid(ids):
                    answer = annotation_answer(ids)
                    answer["annotations"][ids[0]]["contextual_claims"][0]["evidence_intervals"] = [
                        {"start_ms": start, "end_ms": end}]
                    return answer
                factory = RuntimeFactory(invalid)
                with self.assertRaises(ContractError):
                    self.run_pass(factory)
                self.assertEqual(len(factory.calls), 1)
                self.assertFalse((self.output / "annotations.json").exists())

    def test_context_conflict_is_preserved_as_review_required_draft(self):
        def conflicting(ids):
            output = annotation_answer(ids)
            output["context_check"] = "conflict"
            return output
        result = self.run_pass(RuntimeFactory(conflicting))
        self.assertEqual(result["status"], "context_conflict")
        draft = read(self.output / "annotations.json")
        self.assertEqual(draft["context_check"], "conflict")
        self.assertFalse(draft["training_eligible"])
        self.assertTrue(all(frame["review_required"] for frame in draft["annotations"]))
        resumed = RuntimeFactory()
        self.assertEqual(self.run_pass(resumed, resume=True)["status"], "context_conflict")
        self.assertEqual(resumed.sessions, [])

    def test_selection_must_be_finished_and_match_last_actual_model_review(self):
        selection_path = self.selection / "selection.json"
        original = read(selection_path)
        def unfinished(value):
            value["status"] = "awaiting_review"
        def changed_selected_ids(value):
            value["selected_frame_ids"] = self.ids
        def invented_reason(value):
            value["frames"][0]["model_reason"] = "Changed after model review."
        def changed_source_locator(value):
            value["frames"][0]["timestamp_ms"] = 999
        def unsupported_full_video(value):
            value["completed_rounds_full_video_verified"] = False
        for index, change in enumerate((unfinished, changed_selected_ids, invented_reason,
                                        changed_source_locator, unsupported_full_video)):
            with self.subTest(change=change.__name__):
                self.output = self.root / f"invalid-selection-{index}"
                value = copy.deepcopy(original)
                change(value)
                write(selection_path, value)
                factory = RuntimeFactory()
                with self.assertRaises(ContractError):
                    self.run_pass(factory)
                self.assertEqual(factory.sessions, [])
                self.assertFalse((self.output / "annotations.json").exists())
        write(selection_path, original)

    def test_cooperative_pause_before_inference_can_resume(self):
        factory = RuntimeFactory()
        paused = self.run_pass(factory, should_stop=lambda: True)
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(factory.sessions, [])
        self.assertEqual(self.run_pass(factory, resume=True)["status"], "completed")
        self.assertEqual(len(factory.calls), 1)

    def test_active_writer_cannot_be_resumed_and_pause_request_is_persisted(self):
        self.run_pass(RuntimeFactory(), prepare_only=True)
        with (self.output / ".annotation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertTrue(annotation_status(self.output)["writer_active"])
            resumed = RuntimeFactory()
            with self.assertRaises(ContractError):
                self.run_pass(resumed, resume=True)
            self.assertEqual(resumed.sessions, [])
            request_annotation_pause(self.output)
            self.assertTrue((self.output / ".pause-requested").exists())
        self.assertFalse(annotation_status(self.output)["writer_active"])


if __name__ == "__main__":
    unittest.main()
