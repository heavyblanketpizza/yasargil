"""Rejected temporal drafts retain raw content and full native-video evidence."""
import copy
from dataclasses import asdict
import fcntl
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import unittest

from jsonschema import Draft202012Validator
from PIL import Image

from tests import test_frame_annotation as fixtures
from yasargil.contract import ContractError, sha256_file
from yasargil.frame_annotation import AnnotationConfig
from yasargil.qwen_draft_intake import load_rejected_annotation


read, write = fixtures.read, fixtures.write


def make_rejected_fixture(root, *, schema_rejected=False):
    """Give a completed scripted annotation realistic rejected native artifacts.

    This modifies test fixture files only. It returns the offending target and
    unchanged model interval so integration tests can assert preservation.
    """
    root = Path(root).resolve()
    plan, source = read(root / "run.json"), read(root / "source/source.json")
    config = AnnotationConfig(**plan["config"])
    directory = root / "round-00"
    result = read(directory / "result.json")
    raw = result["output"]
    frame_id = plan["frozen_frame_ids"][-1]
    start = source["duration_ms"] - 1000 if schema_rejected else 0
    end = source["duration_ms"] + 1000 if schema_rejected else config.max_evidence_span_ms + 1000
    interval = {"start_ms": start, "end_ms": end}
    raw["annotations"][frame_id]["contextual_claims"][0]["evidence_intervals"] = [interval]
    response = result["response"]
    response["choices"][0]["message"]["content"] = json.dumps(raw, ensure_ascii=False)
    request = read(directory / "request.json")
    request.update(model="qwen-video", temperature=0.1, seed=42, stream=False, cache_prompt=True,
                   id_slot=0, chat_template_kwargs={"enable_thinking": False})
    request["response_format"]["json_schema"].update(name="frame_selection", strict=True)
    write(directory / "request.json", request)
    write(directory / "response.json", response)
    runtime = root / "runtime/rejected-fixture"
    native_root = runtime / "native-decode"
    native_root.mkdir(parents=True)
    receipt_root = runtime / "transport-receipts"
    receipt_root.mkdir()
    receipt_path = receipt_root / "input.json"
    adapter = {"schema_version": "ffmpeg-byte-spool-transport-v1",
               "module": {"path": "/archived/transport.py", "sha256": "a" * 64},
               "manifest_sha256": "b" * 64}
    receipt = {"schema_version": adapter["schema_version"],
               "input_sha256": source["video_sha256"], "input_bytes": Path(source["video_path"]).stat().st_size,
               "module_sha256": adapter["module"]["sha256"], "transport_manifest_sha256": adapter["manifest_sha256"]}
    write(receipt_path, receipt)
    count = source["expected_video_frames"]
    rate = Fraction(source["ffprobe"]["streams"][0]["r_frame_rate"])
    fps = struct.unpack("f", struct.pack("f", float(rate)))[0]
    native = {"accepted": True, "ordered_rgb_frames_identical": True, "video_sha256": source["video_sha256"],
              "video_path": str(root / "media/video.mp4"), "expected_video_frames": count,
              "source_decoded_frames": count, "native_decoded_frames": count,
              "source_r_frame_rate": str(rate), "native_filter": f"fps={fps:.6f}", "video_stream_index": 0,
              "ffmpeg_transport": adapter, "byte_transport_receipts": {"source": receipt, "native": receipt}}
    metadata = {"context_size": config.context_size, "image_max_tokens": config.image_max_tokens,
                "expected_video_frames": count, "video_sha256": source["video_sha256"], "video_fps_setting": 0,
                "native_decode_verification": native, "ffmpeg_transport": adapter,
                "ffmpeg_transport_receipts_directory": str(receipt_root),
                "command": ["llama-server", "--no-context-shift", "-c", str(config.context_size),
                            "--image-max-tokens", str(config.image_max_tokens), "--video-fps", "0",
                            "--timeout", str(math.ceil(config.request_timeout_seconds)),
                            "--media-path", str(root / "media")]}
    write(native_root / "verification.json", native)
    write(runtime / "runtime.json", metadata)
    write(native_root / "ffprobe.json", {"streams": [{"index": 0, "codec_type": "video", "r_frame_rate": str(rate)}]})
    records = []
    for index, frame in enumerate(source["frames"]):
        with Image.open(frame["image_path"]) as image:
            pixels = image.convert("RGB").tobytes()
        records.append(f"0, {index}, {index}, 1, {len(pixels)}, {hashlib.sha256(pixels).hexdigest()}\n")
    for name in ("source", "native"):
        (native_root / f"{name}.framehash").write_text("".join(records))
        (native_root / f"{name}.log").write_text("Saved synthetic byte-transport evidence.\n")
    segment = "".join(f"read_next_frame: frame {index} read OK\n" for index in range(count)) + "truncated = 0\n"
    prefix = "server startup\n"
    (runtime / "server.log").write_text(prefix + segment + "server stopped\n")
    (directory / "server-segment.log").write_text(segment)
    verification = result["verification"]
    verification.update(accepted=not schema_rejected, runtime_directory=str(runtime), message_count=2,
                        prior_history_preserved=False, usage=response["usage"], timings=response.get("timings"),
                        system_fingerprint=response.get("system_fingerprint"),
                        log_start_byte=len(prefix.encode()), log_end_byte=len((prefix + segment).encode()),
                        request_sha256=sha256_file(directory / "request.json"),
                        byte_transport_receipt=receipt, byte_transport_receipt_path=str(receipt_path))
    if schema_rejected:
        errors = sorted(Draft202012Validator(request["response_format"]["json_schema"]["schema"]).iter_errors(raw),
                        key=lambda error: str(list(error.path)))
        message = "Qwen's answer does not match the required schema: " + errors[0].message
        verification["error"] = message
        error = {"type": "VideoRuntimeError", "message": message, "at": "fixture"}
        (directory / "result.json").unlink()
        (directory / "output.json").unlink()
    else:
        error = {"type": "ContractError", "message": "Evidence interval exceeds maximum evidence span", "at": "fixture"}
        write(directory / "result.json", {"output": raw, "response": response, "verification": verification})
        write(directory / "output.json", raw)
    write(directory / "verification.json", verification)
    write(root / "last-error.json", error)
    summary = read(root / "summary.json")
    summary.update(status="failed", error=error)
    write(root / "summary.json", summary)
    (root / "annotations.json").unlink()
    return {"frame_id": frame_id, "interval": interval, "schema_rejected": schema_rejected}


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Native provenance fixtures need FFmpeg")
class RejectedQwenIntakeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.FrameAnnotationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.output
        self.fixture.run_pass(fixtures.RuntimeFactory())
        self.details = make_rejected_fixture(self.root)

    def files(self):
        return {str(path.relative_to(self.root)): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file() and not path.is_symlink()}

    def test_overlong_drafts_preserve_exact_text_bounds_and_canonical_support(self):
        before = self.files()
        _, source, drafts, audit = load_rejected_annotation(self.root)
        self.assertEqual(audit["status"], "rejected_temporal_citations")
        self.assertEqual(audit["annotation_count"], 4)
        self.assertEqual(audit["issues"], drafts["validation_issues"])
        target = next(row for row in drafts["annotations"] if row["frame_id"] == self.details["frame_id"])
        interval = target["contextual_claims"][0]["evidence_intervals"][0]
        self.assertEqual({key: interval[key] for key in ("start_ms", "end_ms")}, self.details["interval"])
        self.assertEqual(interval["supporting_frames"], source["frames"][:4])
        raw = json.loads(read(self.root / "round-00/response.json")["choices"][0]["message"]["content"])
        for row in drafts["annotations"]:
            for key in ("visible_observation", "visibility", "uncertainties"):
                self.assertEqual(row[key], raw["annotations"][row["frame_id"]][key])
        self.assertEqual(before, self.files())
        self.assertFalse((self.root / "annotations.json").exists())
        self.assertFalse((self.root / "integrity.json").exists())
        self.assertFalse(drafts["training_eligible"])
        self.assertEqual(audit["raw_response_sha256"], sha256_file(self.root / "round-00/response.json"))
        self.assertTrue(set(before) - {"report.html"} <= set(audit["artifact_sha256"]))

    def test_frozen_copy_revalidates_identically_without_original_runtime_access(self):
        original = load_rejected_annotation(self.root)
        frozen = self.fixture.root / "frozen"
        for name in original[3]["artifact_sha256"]:
            destination = frozen / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.root / name, destination)
        runtime = self.root / "runtime"
        runtime.rename(self.root / "runtime-unavailable")
        self.assertEqual(load_rejected_annotation(frozen), original)

    def test_out_of_video_end_is_preserved_without_fabricating_an_accepted_result(self):
        other = self.fixture.root / "end-outside"
        self.fixture.output = other
        self.fixture.run_pass(fixtures.RuntimeFactory())
        details = make_rejected_fixture(other, schema_rejected=True)
        _, source, drafts, audit = load_rejected_annotation(other)
        self.assertFalse(audit["original_transport_accepted"])
        row = next(item for item in drafts["annotations"] if item["frame_id"] == details["frame_id"])
        interval = row["contextual_claims"][0]["evidence_intervals"][0]
        self.assertEqual(interval["end_ms"], source["duration_ms"] + 1000)
        self.assertEqual(interval["supporting_frames"], source["frames"][-1:])
        self.assertEqual(audit["issues"][0]["code"], "evidence_end_exceeds_video")
        self.assertFalse((other / "round-00/result.json").exists())
        self.assertFalse((other / "round-00/output.json").exists())

    def test_non_temporal_schema_or_semantic_failures_are_rejected(self):
        paths = [self.root / f"round-00/{name}.json" for name in ("response", "verification", "result", "output")]
        originals = {path: path.read_bytes() for path in paths}
        changes = [
            lambda raw: raw["annotations"].pop(self.details["frame_id"]),
            lambda raw: raw["annotations"][self.details["frame_id"]].update(visible_observation=" "),
            lambda raw: raw["annotations"][self.details["frame_id"]]["contextual_claims"][0]["evidence_intervals"][0].update(start_ms=-1),
            lambda raw: raw["annotations"][self.details["frame_id"]]["contextual_claims"][0]["evidence_intervals"][0].update(start_ms=4000, end_ms=3000),
            lambda raw: raw["annotations"][self.details["frame_id"]]["contextual_claims"][0]["evidence_intervals"][0].update(start_ms=8000, end_ms=9000),
        ]
        for change in changes:
            for path, data in originals.items():
                path.write_bytes(data)
            result = read(paths[2])
            change(result["output"])
            result["response"]["choices"][0]["message"]["content"] = json.dumps(result["output"])
            for path, value in zip(paths, (result["response"], result["verification"], result, result["output"])):
                write(path, value)
            with self.subTest(change=change), self.assertRaises(ContractError):
                load_rejected_annotation(self.root)

    def test_missing_transport_changed_log_or_decoder_receipts_are_rejected(self):
        paths = [
            self.root / "runtime/rejected-fixture/transport-receipts/input.json",
            self.root / "round-00/server-segment.log",
            self.root / "runtime/rejected-fixture/native-decode/native.framehash",
        ]
        for path in paths:
            original = path.read_bytes()
            path.write_text("{}")
            with self.subTest(path=path), self.assertRaises(ContractError):
                load_rejected_annotation(self.root)
            path.write_bytes(original)

    def test_frozen_input_or_request_changes_are_rejected(self):
        for name in ("selected-frames.json", "session.json", "round-00/request.json"):
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(name=name), self.assertRaises(ContractError):
                load_rejected_annotation(self.root)
            path.write_bytes(original)

    def test_active_writer_and_false_completion_claim_are_rejected(self):
        with (self.root / ".annotation.lock").open("r") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ContractError, "still active"):
                load_rejected_annotation(self.root)
        write(self.root / "integrity.json", {"status": "completed"})
        with self.assertRaisesRegex(ContractError, "published or sealed"):
            load_rejected_annotation(self.root)


if __name__ == "__main__":
    unittest.main()
