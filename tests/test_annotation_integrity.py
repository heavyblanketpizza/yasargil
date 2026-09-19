"""Completed Qwen drafts remain tied to exact raw bytes and canonical evidence."""
import copy
import fcntl
import json
import math
from pathlib import Path
import shutil
import unittest

import test_frame_annotation as fixtures
from yasargil.annotation_contract import build_annotations
from yasargil.annotation_integrity import MANIFEST_NAME, seal_annotation_output, verify_annotation_output
from yasargil.contract import ContractError, sha256_file


read, write = fixtures.read, fixtures.write


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Integrity fixtures need FFmpeg")
class AnnotationIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.FrameAnnotationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.make_selection("integrity-keep-drop", review_mode="keep_drop")
        self.output = self.fixture.output
        self.factory = fixtures.RuntimeFactory()
        self.fixture.run_pass(self.factory)

    def files(self):
        return {path.relative_to(self.output).as_posix(): path.read_bytes()
                for path in self.output.rglob("*") if path.is_file() and not path.is_symlink()}

    def replace_result(self, result):
        """Consistently alter all derived copies, leaving the prior seal intact."""
        directory = self.output / "round-00"
        for name in ("output", "response", "verification"):
            write(directory / f"{name}.json", result[name])
        write(directory / "result.json", result)
        current = read(self.output / "annotations.json")
        built = build_annotations(result["output"], read(self.output / "selected-frames.json"),
                                  read(self.output / "source/source.json"),
                                  max_evidence_span_ms=self.fixture.config.max_evidence_span_ms)
        current.update(built, verification=result["verification"],
                       model_result_sha256=sha256_file(directory / "result.json"))
        write(self.output / "annotations.json", current)

    def add_runtime(self, *, transport=True):
        """Supply realistic archived metadata without running a decoder or model."""
        runtime = self.output / "runtime" / "transport-attempt"
        native_root = runtime / "native-decode"
        native_root.mkdir(parents=True)
        source = read(self.output / "source/source.json")
        native = {"accepted": True, "ordered_rgb_frames_identical": True,
                  "video_sha256": source["video_sha256"],
                  "expected_video_frames": 8, "source_decoded_frames": 8, "native_decoded_frames": 8}
        metadata = {"context_size": self.fixture.config.context_size,
                    "image_max_tokens": self.fixture.config.image_max_tokens,
                    "expected_video_frames": 8, "video_sha256": source["video_sha256"],
                    "video_fps_setting": 0,
                    "command": ["llama-server", "--timeout", str(math.ceil(self.fixture.config.request_timeout_seconds))]}
        result = read(self.output / "round-00/result.json")
        result["verification"]["runtime_directory"] = str(runtime)
        receipt_path = None
        if transport:
            adapter = {"schema_version": "ffmpeg-byte-spool-transport-v1",
                       "module": {"path": "/archived/ffmpeg_transport.py", "sha256": "a" * 64},
                       "manifest_sha256": "b" * 64}
            receipt_root = runtime / "transport-receipts"
            receipt_root.mkdir()
            receipt_path = receipt_root / "native-input.json"
            receipt = {"schema_version": "ffmpeg-byte-spool-transport-v1",
                       "input_sha256": source["video_sha256"],
                       "input_bytes": Path(source["video_path"]).stat().st_size,
                       "module_sha256": adapter["module"]["sha256"],
                       "transport_manifest_sha256": adapter["manifest_sha256"],
                       "original_input": "cache:pipe:0", "temporary_input": "/removed/complete-input.bin",
                       "arguments": ["-i", "cache:pipe:0"],
                       "decoder_arguments": ["-i", "/removed/complete-input.bin"]}
            write(receipt_path, receipt)
            native.update(ffmpeg_transport=adapter,
                          byte_transport_receipts={"source": copy.deepcopy(receipt), "native": copy.deepcopy(receipt)})
            metadata.update(ffmpeg_transport=adapter, ffmpeg_transport_receipts_directory=str(receipt_root))
            result["verification"].update(byte_transport_receipt=receipt,
                                          byte_transport_receipt_path=str(receipt_path))
        metadata["native_decode_verification"] = native
        write(native_root / "verification.json", native)
        write(runtime / "runtime.json", metadata)
        for name in ("server.log", "native-decode/source.framehash", "native-decode/native.framehash",
                     "native-decode/ffprobe.json", "native-decode/source.log", "native-decode/native.log"):
            (runtime / name).write_text("Archived synthetic decoder evidence\n")
        (self.output / "round-00/server-segment.log").write_text("Archived synthetic inference evidence\n")
        self.replace_result(result)
        return runtime, receipt_path

    def test_seal_preserves_exact_evidence_and_maps_raw_annotations_to_source(self):
        before = self.files()
        sealed = seal_annotation_output(self.output)
        self.assertEqual(sealed["status"], "completed")
        self.assertEqual(sealed["frame_ids"], self.fixture.selected_ids)
        self.assertEqual(sealed["source_frame_count"], 8)
        self.assertEqual(sealed["annotation_count"], 4)
        raw = read(self.output / "round-00/output.json")["annotations"]
        draft = read(self.output / "annotations.json")
        for frame in draft["annotations"]:
            self.assertEqual(frame["visible_observation"], raw[frame["frame_id"]]["visible_observation"])
            self.assertEqual(frame["source_sha256"], self.fixture.source["frames"][frame["frame_index"]]["source_sha256"])
        for name in ("run.json", "session.json", "source/source.json", "selected-frames.json",
                     "selection-review/response.json", "selection-state.json", "native-timeline-verification.json",
                     "round-00/request.json", "round-00/response.json", "round-00/result.json",
                     "round-00/output.json", "round-00/verification.json", "annotations.json"):
            self.assertEqual(sealed["artifacts"][name], {
                "sha256": sha256_file(self.output / name), "size_bytes": len(before[name])})
        after = self.files()
        self.assertEqual(set(after) - set(before), {MANIFEST_NAME})
        self.assertEqual(before, {name: after[name] for name in before})
        self.assertEqual(verify_annotation_output(self.output), sealed)
        manifest_bytes = (self.output / MANIFEST_NAME).read_bytes()
        self.assertEqual(seal_annotation_output(self.output), sealed)
        self.assertEqual((self.output / MANIFEST_NAME).read_bytes(), manifest_bytes)

    def test_raw_whitespace_change_is_rejected_without_replacing_the_seal(self):
        seal_annotation_output(self.output)
        original_manifest = (self.output / MANIFEST_NAME).read_bytes()
        response = self.output / "round-00/response.json"
        response.write_bytes(response.read_bytes() + b"\n")
        for action in (verify_annotation_output, seal_annotation_output):
            with self.subTest(action=action.__name__), self.assertRaises(ContractError):
                action(self.output)
        self.assertEqual((self.output / MANIFEST_NAME).read_bytes(), original_manifest)

    def test_coordinated_changes_to_raw_and_derived_annotations_cannot_replace_a_seal(self):
        seal_annotation_output(self.output)
        manifest = (self.output / MANIFEST_NAME).read_bytes()
        result = read(self.output / "round-00/result.json")
        result["output"]["annotations"][self.fixture.selected_ids[0]]["visible_observation"] = "An invented observation."
        result["response"]["choices"][0]["message"]["content"] = json.dumps(result["output"])
        self.replace_result(result)
        with self.assertRaises(ContractError):
            seal_annotation_output(self.output)
        self.assertEqual((self.output / MANIFEST_NAME).read_bytes(), manifest)

    def test_unsealed_raw_response_and_annotation_disagreement_is_rejected(self):
        path = self.output / "annotations.json"
        value = read(path)
        value["annotations"][0]["visible_observation"] = "An invented annotation not in Qwen's response."
        write(path, value)
        with self.assertRaises(ContractError):
            seal_annotation_output(self.output)
        self.assertFalse((self.output / MANIFEST_NAME).exists())

    def test_missing_required_receipt_cannot_be_sealed(self):
        for name in ("round-00/verification.json", "selection-review/response.json", "native-timeline-verification.json"):
            with self.subTest(name=name):
                path = self.output / name
                data = path.read_bytes()
                path.unlink()
                with self.assertRaises(ContractError):
                    seal_annotation_output(self.output)
                self.assertFalse((self.output / MANIFEST_NAME).exists())
                path.write_bytes(data)

    def test_optional_runtime_and_log_evidence_are_pinned_when_present(self):
        runtime = self.output / "runtime" / "scripted-attempt"
        native = runtime / "native-decode"
        native.mkdir(parents=True)
        write(runtime / "runtime.json", {"test_runtime": True})
        (runtime / "server.log").write_text("scripted runtime log\n")
        (native / "source.framehash").write_text("scripted decoded frame hashes\n")
        (self.output / "round-00/server-segment.log").write_text("scripted inference log\n")
        sealed = seal_annotation_output(self.output)
        self.assertIn("runtime/scripted-attempt/native-decode/source.framehash", sealed["artifacts"])
        self.assertIn("round-00/server-segment.log", sealed["artifacts"])
        (runtime / "server.log").write_text("changed log\n")
        with self.assertRaises(ContractError):
            verify_annotation_output(self.output)

    def test_advertised_runtime_receipts_cannot_be_missing(self):
        result = read(self.output / "round-00/result.json")
        result["verification"]["runtime_directory"] = str(self.output / "runtime" / "missing-attempt")
        self.replace_result(result)
        with self.assertRaises(ContractError):
            seal_annotation_output(self.output)
        self.assertFalse((self.output / MANIFEST_NAME).exists())

    def test_spool_receipt_bytes_are_sealed_and_whitespace_changes_are_rejected(self):
        _, receipt = self.add_runtime()
        before = self.files()
        sealed = seal_annotation_output(self.output)
        name = receipt.relative_to(self.output).as_posix()
        self.assertEqual(sealed["artifacts"][name],
                         {"sha256": sha256_file(receipt), "size_bytes": len(before[name])})
        self.assertEqual(before, {name: self.files()[name] for name in before})
        self.assertEqual(verify_annotation_output(self.output), sealed)
        original_manifest = (self.output / MANIFEST_NAME).read_bytes()
        receipt.write_bytes(receipt.read_bytes() + b"\n")
        with self.assertRaisesRegex(ContractError, "bytes have changed"):
            verify_annotation_output(self.output)
        self.assertEqual((self.output / MANIFEST_NAME).read_bytes(), original_manifest)

    def test_spool_receipt_must_match_the_copy_in_the_accepted_result(self):
        _, path = self.add_runtime()
        receipt = read(path)
        receipt["input_bytes"] += 1
        write(path, receipt)
        with self.assertRaisesRegex(ContractError, "differs from the accepted model result"):
            seal_annotation_output(self.output)
        self.assertFalse((self.output / MANIFEST_NAME).exists())

    def test_spool_receipt_cannot_relabel_input_bytes_or_the_pinned_adapter(self):
        _, path = self.add_runtime()
        original = read(self.output / "round-00/result.json")
        receipt = read(path)
        for key, changed in (("input_bytes", receipt["input_bytes"] - 1),
                             ("input_sha256", "f" * 64), ("module_sha256", "c" * 64),
                             ("transport_manifest_sha256", "d" * 64)):
            with self.subTest(field=key):
                result = copy.deepcopy(original)
                result["verification"]["byte_transport_receipt"][key] = changed
                write(path, result["verification"]["byte_transport_receipt"])
                self.replace_result(result)
                with self.assertRaisesRegex(ContractError, "original video bytes or pinned adapter"):
                    seal_annotation_output(self.output)
                self.assertFalse((self.output / MANIFEST_NAME).exists())

    def test_spool_receipt_location_cannot_escape_its_runtime_directory(self):
        runtime, receipt = self.add_runtime()
        outside = runtime / "outside-transport.json"
        outside.write_bytes(receipt.read_bytes())
        original = read(self.output / "round-00/result.json")
        for location in (str(outside), "transport-receipts/native-input.json"):
            with self.subTest(location=location):
                result = copy.deepcopy(original)
                result["verification"]["byte_transport_receipt_path"] = location
                self.replace_result(result)
                with self.assertRaisesRegex(ContractError, "transport receipt"):
                    seal_annotation_output(self.output)
                self.assertFalse((self.output / MANIFEST_NAME).exists())

    def test_advertised_transport_requires_receipt_file_runtime_and_matching_preflight(self):
        runtime, receipt_path = self.add_runtime()
        original_result = read(self.output / "round-00/result.json")
        original_metadata = read(runtime / "runtime.json")
        receipt_bytes = receipt_path.read_bytes()
        receipt_path.unlink()
        with self.assertRaisesRegex(ContractError, "transport receipt"):
            seal_annotation_output(self.output)
        receipt_path.write_bytes(receipt_bytes)
        result = copy.deepcopy(original_result)
        result["verification"].pop("runtime_directory")
        self.replace_result(result)
        with self.assertRaisesRegex(ContractError, "lacks its annotation runtime"):
            seal_annotation_output(self.output)
        self.replace_result(original_result)
        native = copy.deepcopy(original_metadata["native_decode_verification"])
        native["byte_transport_receipts"]["source"]["input_bytes"] -= 1
        changed_metadata = copy.deepcopy(original_metadata)
        changed_metadata["native_decode_verification"] = native
        write(runtime / "runtime.json", changed_metadata)
        write(runtime / "native-decode/verification.json", native)
        with self.assertRaisesRegex(ContractError, "original video bytes or pinned adapter"):
            seal_annotation_output(self.output)
        self.assertFalse((self.output / MANIFEST_NAME).exists())

    def test_legacy_native_runtime_without_adapter_remains_verifiable(self):
        self.add_runtime(transport=False)
        before = self.files()
        sealed = seal_annotation_output(self.output)
        self.assertEqual(verify_annotation_output(self.output), sealed)
        self.assertEqual(before, {name: self.files()[name] for name in before})

    def test_active_annotation_writer_prevents_sealing(self):
        with (self.output / ".annotation.lock").open("r") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(ContractError):
                seal_annotation_output(self.output)
        self.assertFalse((self.output / MANIFEST_NAME).exists())

    def test_historical_session_without_timeout_keeps_its_original_bytes(self):
        plan = read(self.output / "run.json")
        session = read(self.output / "session.json")
        plan["config"].pop("request_timeout_seconds")
        session["config"].pop("request_timeout_seconds")
        write(self.output / "session.json", session)
        write(self.output / "round-00/annotation-context.json", session)
        plan["input_sha256"]["session.json"] = sha256_file(self.output / "session.json")
        write(self.output / "run.json", plan)
        before = self.files()
        sealed = seal_annotation_output(self.output)
        self.assertEqual(verify_annotation_output(self.output), sealed)
        self.assertEqual(before, {name: self.files()[name] for name in before})

    def test_context_conflict_can_be_sealed_as_a_review_required_draft(self):
        def conflict(ids):
            value = fixtures.annotation_answer(ids)
            value["context_check"] = "conflict"
            return value
        self.fixture.output = self.fixture.root / "conflict-annotation"
        self.fixture.run_pass(fixtures.RuntimeFactory(conflict))
        sealed = seal_annotation_output(self.fixture.output)
        self.assertEqual(sealed["status"], "context_conflict")
        self.assertFalse(sealed["training_eligible"])
        self.assertEqual(verify_annotation_output(self.fixture.output), sealed)


if __name__ == "__main__":
    unittest.main()
