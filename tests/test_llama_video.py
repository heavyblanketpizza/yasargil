"""Network- and GPU-free tests of native video coverage and failure handling."""
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from yasargil.llama_video import (
    LocalVideoRuntime, RuntimeConfig, VideoRuntimeError, _NoRedirect, _sha256,
    verify_native_video_decode,
)


SCHEMA = {"type": "object", "properties": {"keep": {"type": "array", "items": {"type": "string"}}},
          "required": ["keep"], "additionalProperties": False}


class NativeVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.media = self.root / "media"
        self.media.mkdir()
        (self.media / "video.mkv").write_bytes(b"complete synthetic test video")
        (self.media / "F000001.png").write_bytes(b"synthetic still")
        self.log_dir = self.root / "runtime"
        self.log_dir.mkdir()
        self.log = self.log_dir / "server.log"
        self.log.write_text("startup log\n")
        self.config = RuntimeConfig(self.root, self.media, self.log_dir, context_size=65536)
        self.runtime = LocalVideoRuntime(self.config, expected_video_frames=3, video_relative_path="video.mkv")
        self.runtime._server = Mock()
        self.runtime._server.poll.return_value = None
        self.runtime._video_sha256 = _sha256(self.media / "video.mkv")
        self.runtime._http = Mock()
        self.runtime.base_url = "http://127.0.0.1:12345"
        self.messages = [{"role": "system", "content": "Use verified procedure context."},
                         {"role": "user", "content": [
                             {"type": "input_video", "input_video": {"url": "file://video.mkv"}},
                             {"type": "text", "text": "Review the entire video and these candidates together."},
                             {"type": "image_url", "image_url": {"url": "file://F000001.png"}},
                         ]}]
        self.result = {"choices": [{"finish_reason": "stop", "index": 0,
                                   "message": {"role": "assistant", "content": '{ "keep": ["F000001"] }'}}],
                       "usage": {"prompt_tokens": 1000, "completion_tokens": 20,
                                 "prompt_tokens_details": {"cached_tokens": 0}},
                       "timings": {"cache_n": 0}, "system_fingerprint": "b10809-test"}

    def response(self, result=None, frame_ids=(0, 1, 2), extra_log=""):
        result = self.result if result is None else result
        def receive(request, timeout):
            with self.log.open("a") as handle:
                for frame in frame_ids:
                    handle.write(f"0.00.000 D read_next_frame: frame {frame} read OK\n")
                handle.write("0.01.000 I slot release: stop processing: n_tokens = 1020, truncated = 0\n")
                handle.write(extra_log)
            return io.BytesIO(result if isinstance(result, bytes) else json.dumps(result).encode())
        self.runtime._http.open.side_effect = receive

    def call(self, name="round0", messages=None):
        return self.runtime.chat(self.messages if messages is None else messages,
                                 schema=SCHEMA, max_tokens=100, round_dir=self.root / name)

    def test_full_native_video_request_and_exact_evidence(self):
        original = deepcopy(self.messages)
        self.response()
        result = self.call()
        self.assertEqual(result["output"], {"keep": ["F000001"]})
        request = self.runtime._http.open.call_args.args[0]
        saved = (self.root / "round0/request.json").read_bytes()
        self.assertEqual(saved, request.data)
        payload = json.loads(saved)
        self.assertEqual(payload["messages"], original)
        self.assertEqual(self.messages, original)
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], SCHEMA)
        self.assertTrue(payload["cache_prompt"])
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        self.assertTrue(result["verification"]["full_source_video_verified"])
        self.assertEqual(result["verification"]["decoded_frame_ids"], [0, 1, 2])
        self.assertEqual(json.loads((self.root / "round0/result.json").read_text()), result)
        self.assertEqual(result["response"]["choices"][0]["message"]["content"], self.result["choices"][0]["message"]["content"])
        self.assertNotIn("startup log", (self.root / "round0/server-segment.log").read_text())

    def test_later_round_keeps_video_history_and_reports_actual_cache(self):
        self.response()
        first = self.call()
        messages = self.messages + [first["response"]["choices"][0]["message"],
                                    {"role": "user", "content": "No extra frame found; finalize."}]
        cached = deepcopy(self.result)
        cached["usage"]["prompt_tokens_details"]["cached_tokens"] = 980
        cached["timings"]["cache_n"] = 980
        self.response(cached)
        result = self.call("round1", messages)
        self.assertTrue(result["verification"]["prior_history_preserved"])
        self.assertEqual(result["verification"]["decoded_frames"], 3)
        self.assertTrue(result["verification"]["cache"]["prefix_cache_hit_reported"])
        self.assertFalse(result["verification"]["cache"]["vision_encoding_reuse_verified"])
        self.assertEqual(result["verification"]["cache"]["cached_tokens"], 980)

    def test_cached_prefix_does_not_excuse_absent_video_decode_evidence(self):
        cached = deepcopy(self.result)
        cached["usage"]["prompt_tokens_details"]["cached_tokens"] = 980
        self.response(cached, frame_ids=())
        with self.assertRaisesRegex(VideoRuntimeError, "Complete video decoding"):
            self.call()
        self.assertFalse((self.root / "round0/result.json").exists())

    def test_partial_duplicate_out_of_order_or_excess_frames_rejected(self):
        for index, ids in enumerate(((0, 1), (0, 1, 1), (1, 0, 2), (0, 1, 2, 3))):
            with self.subTest(ids=ids):
                self.response(frame_ids=ids)
                with self.assertRaisesRegex(VideoRuntimeError, "Complete video decoding"):
                    self.call(str(index))
                self.assertFalse((self.root / str(index) / "result.json").exists())

    def test_truncated_context_and_output_are_rejected(self):
        self.response(extra_log="D slot release: truncated = 1\n")
        with self.assertRaisesRegex(VideoRuntimeError, "truncated or shifted"):
            self.call("context")
        unfinished = deepcopy(self.result)
        unfinished["choices"][0]["finish_reason"] = "length"
        self.response(unfinished)
        with self.assertRaisesRegex(VideoRuntimeError, "Unfinished"):
            self.call("length")
        self.assertFalse((self.root / "length/result.json").exists())

    def test_malformed_empty_or_schema_invalid_outputs_leave_failure_receipt(self):
        cases = ["", " ", '{"keep":[]', '{"keep":[],"keep":[]}', '{"keep":NaN}',
                 '{"keep":7}', '{"keep":[],"extra":1}', "```json\n{}\n```"]
        for index, content in enumerate(cases):
            with self.subTest(content=content):
                result = deepcopy(self.result)
                result["choices"][0]["message"]["content"] = content
                self.response(result)
                with self.assertRaises(VideoRuntimeError):
                    self.call(str(index))
                receipt = json.loads((self.root / str(index) / "verification.json").read_text())
                self.assertFalse(receipt["accepted"])
                self.assertIn("error", receipt)
                self.assertFalse((self.root / str(index) / "result.json").exists())

    def test_prompt_plus_answer_must_fit_without_silent_truncation(self):
        result = deepcopy(self.result)
        result["usage"]["prompt_tokens"] = 65500
        self.response(result)
        with self.assertRaisesRegex(VideoRuntimeError, "exceeds context"):
            self.call()

    def test_complete_prefix_and_previous_answer_cannot_be_removed(self):
        self.response()
        self.call()
        for messages in (self.messages, self.messages + [{"role": "user", "content": "Continue"}],
                         [{"role": "system", "content": "Changed"}] + self.messages[1:] + [
                             {"role": "assistant", "content": '{"keep":["F000001"]}'}]):
            with self.subTest(messages=messages), self.assertRaisesRegex(VideoRuntimeError, "prefix|previous validated"):
                self.call("bad", messages)

    def test_missing_duplicate_or_remote_video_rejected_before_http(self):
        no_video = deepcopy(self.messages)
        no_video[1]["content"].pop(0)
        duplicate = deepcopy(self.messages)
        duplicate[1]["content"].append(duplicate[1]["content"][0])
        remote = deepcopy(self.messages)
        remote[1]["content"][0]["input_video"]["url"] = "https://example.com/video.mp4"
        for messages in (no_video, duplicate, remote):
            with self.subTest(messages=messages), self.assertRaises(VideoRuntimeError):
                self.call(messages=messages)
        self.runtime._http.open.assert_not_called()

    def test_source_change_is_rejected_and_media_paths_cannot_escape_lexically(self):
        (self.media / "video.mkv").write_bytes(b"modified")
        with self.assertRaisesRegex(VideoRuntimeError, "changed"):
            self.call()
        for path in ("../outside.mp4", "/tmp/outside.mp4", "nested/../video.mkv", "video.mkv?x=1", "a%2Fb.mp4"):
            with self.subTest(path=path), self.assertRaises(VideoRuntimeError):
                LocalVideoRuntime(self.config, expected_video_frames=3, video_relative_path=path)

    def test_staged_external_source_symlink_is_supported(self):
        external = self.root / "original.mkv"
        external.write_bytes(b"external source")
        (self.media / "source.mkv").symlink_to(external)
        runtime = LocalVideoRuntime(self.config, expected_video_frames=1, video_relative_path="source.mkv")
        self.assertEqual(runtime.video_path, self.media / "source.mkv")

    def test_http_failure_keeps_body_log_and_stops_context_manager_server(self):
        self.runtime._http.open.side_effect = HTTPError("http://127.0.0.1", 400, "context too long", {}, io.BytesIO(b"prompt exceeds context"))
        server = self.runtime._server
        with self.assertRaisesRegex(VideoRuntimeError, "HTTP 400"):
            try:
                self.call()
            finally:
                self.runtime.__exit__(None, None, None)
        server.terminate.assert_called_once()
        server.wait.assert_called_once_with(timeout=15)
        self.assertIsNone(self.runtime._server)
        self.assertEqual((self.root / "round0/http-error-body.txt").read_bytes(), b"prompt exceeds context")
        self.assertFalse((self.root / "round0/result.json").exists())

    def test_close_kills_unresponsive_server(self):
        server = self.runtime._server
        server.wait.side_effect = [subprocess.TimeoutExpired("server", 15), None]
        self.runtime.close()
        server.kill.assert_called_once()
        self.assertIsNone(self.runtime._server)

    def test_invalid_configuration_and_redirects(self):
        for kwargs in ({"context_size": 262145}, {"port": -1}, {"image_max_tokens": 1},
                       {"request_timeout": float("nan")}, {"startup_timeout": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LocalVideoRuntime(RuntimeConfig(self.root, self.media, self.log_dir, **kwargs),
                                  expected_video_frames=3, video_relative_path="video.mkv")
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "moved", {}, "https://example.com"))

    def test_existing_successful_round_evidence_cannot_be_overwritten(self):
        self.response()
        self.call()
        # Reset only the prefix tracker to reach the independent artifact guard.
        self.runtime._previous_messages = None
        with self.assertRaisesRegex(VideoRuntimeError, "overwrite"):
            self.call()

    def test_startup_failure_cleans_process_and_pins_native_options(self):
        runtime_dir = self.root / ".runtime/llama.cpp/b10809"
        runtime_dir.mkdir(parents=True)
        (runtime_dir / "llama-server").write_text("test binary")
        models = self.root / ".runtime/models"
        models.mkdir()
        (models / "qwen3.8-27b-q4_k_m.gguf").write_bytes(b"weights")
        (models / "qwen3.8-27b-mmproj-bf16.gguf").write_bytes(b"projector")
        config = RuntimeConfig(self.root, self.media, self.root / "startup")
        runtime = LocalVideoRuntime(config, expected_video_frames=3, video_relative_path="video.mkv")
        server = Mock()
        server.poll.return_value = 1
        with patch("yasargil.llama_video.shutil.which", side_effect=lambda name: "/fake/bin/" + name), \
             patch("yasargil.llama_video.subprocess.check_output", return_value="version: 10809 (5266f24da)"), \
             patch("yasargil.llama_video.verify_native_video_decode", return_value={"video_sha256": "f" * 64}), \
             patch("yasargil.llama_video.subprocess.Popen", return_value=server), \
             patch("yasargil.llama_video.socket.socket") as socket_factory:
            socket_factory.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 12345)
            with self.assertRaisesRegex(VideoRuntimeError, "stopped during startup"):
                runtime.__enter__()
        self.assertIsNone(runtime._server)
        self.assertIsNone(runtime._log_handle)
        self.assertEqual(runtime.command[runtime.command.index("--video-fps") + 1], "0")
        self.assertIn("--no-context-shift", runtime.command)
        self.assertEqual(runtime.command[runtime.command.index("--fit") + 1], "off")
        self.assertEqual(runtime.command[runtime.command.index("--host") + 1], "127.0.0.1")

    def test_installed_transport_is_used_by_preflight_and_actual_native_server(self):
        from yasargil.ffmpeg_transport import RECEIPT_DIRECTORY_ENV
        runtime_dir = self.root / ".runtime/llama.cpp/b10809"
        runtime_dir.mkdir(parents=True)
        (runtime_dir / "llama-server").write_text("test binary")
        models = self.root / ".runtime/models"
        models.mkdir()
        for name in ("qwen3.8-27b-q4_k_m.gguf", "qwen3.8-27b-mmproj-bf16.gguf"):
            (models / name).write_bytes(b"test model")
        adapter = self.root / ".runtime/ffmpeg-safe"
        adapter.mkdir()
        metadata = {"directory": str(adapter), "module": {"sha256": "a" * 64}, "manifest_sha256": "b" * 64}
        runtime = LocalVideoRuntime(RuntimeConfig(self.root, self.media, self.root / "startup"),
                                    expected_video_frames=3, video_relative_path="video.mkv")
        server = Mock()
        server.poll.return_value = 1
        with patch("yasargil.ffmpeg_transport.transport_metadata", return_value=metadata), \
             patch("yasargil.llama_video.subprocess.check_output", return_value="version: 10809"), \
             patch("yasargil.llama_video.verify_native_video_decode", return_value={"video_sha256": "f" * 64}) as preflight, \
             patch("yasargil.llama_video.subprocess.Popen", return_value=server) as popen, \
             patch("yasargil.llama_video.socket.socket") as socket_factory:
            socket_factory.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 12345)
            with self.assertRaisesRegex(VideoRuntimeError, "stopped during startup"):
                runtime.__enter__()
        self.assertEqual(preflight.call_args.kwargs["ffmpeg"], str(adapter / "ffmpeg"))
        self.assertEqual(preflight.call_args.kwargs["ffprobe"], str(adapter / "ffprobe"))
        self.assertEqual(runtime.command[runtime.command.index("--video-ffmpeg-dir") + 1], str(adapter))
        self.assertEqual(popen.call_args.kwargs["env"][RECEIPT_DIRECTORY_ENV], str(self.root / "startup/transport-receipts"))
        self.assertEqual(json.loads((self.root / "startup/runtime.json").read_text())["ffmpeg_transport"], metadata)

    def test_native_transport_receipt_must_bind_the_complete_original_bytes(self):
        directory = self.log_dir / "transport-receipts"
        directory.mkdir()
        metadata = {"directory": str(self.root / "adapter"), "module": {"sha256": "a" * 64}, "manifest_sha256": "b" * 64}
        self.runtime._ffmpeg_transport = metadata
        self.runtime._transport_receipt_dir = directory
        self.response()
        original_receive = self.runtime._http.open.side_effect
        receipt = {"input_bytes": self.runtime.video_path.stat().st_size,
                   "input_sha256": self.runtime._video_sha256,
                   "module_sha256": "a" * 64, "transport_manifest_sha256": "b" * 64}
        def receive(request, timeout):
            (directory / "input-1.json").write_text(json.dumps(receipt))
            return original_receive(request, timeout)
        self.runtime._http.open.side_effect = receive
        with patch("yasargil.ffmpeg_transport.transport_metadata", return_value=metadata):
            result = self.call()
        self.assertEqual(result["verification"]["byte_transport_receipt"], receipt)
        self.runtime._previous_messages = None
        def changed_receive(request, timeout):
            changed = {**receipt, "input_sha256": "0" * 64}
            (directory / "input-2.json").write_text(json.dumps(changed))
            return original_receive(request, timeout)
        self.runtime._http.open.side_effect = changed_receive
        with patch("yasargil.ffmpeg_transport.transport_metadata", return_value=metadata), \
             self.assertRaisesRegex(VideoRuntimeError, "original video bytes"):
            self.call("changed-input")
        self.assertFalse((self.root / "changed-input/result.json").exists())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required for native decode integration checks")
class NativeDecodeIntegrationTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.ffmpeg, self.ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        for index, color in enumerate(("red", "green", "blue", "white")):
            Image.new("RGB", (32, 32), color).save(self.root / f"f{index}.png")

    def make_video(self, name="video.mkv", extra=()):
        path = self.root / name
        subprocess.run([self.ffmpeg, "-v", "error", "-framerate", "4", "-i", str(self.root / "f%d.png"),
                        *extra, "-c:v", "ffv1", "-pix_fmt", "bgr0", str(path)], check=True)
        return path

    def verify(self, video, frames=4):
        return verify_native_video_decode(video, ffmpeg=self.ffmpeg, ffprobe=self.ffprobe,
                                          expected_video_frames=frames, output_dir=self.root / "verify")

    def test_real_cfr_decoder_preserves_every_ordered_rgb_frame(self):
        result = self.verify(self.make_video())
        self.assertTrue(result["accepted"])
        self.assertTrue(result["ordered_rgb_frames_identical"])
        self.assertEqual(result["native_decoded_frames"], 4)
        self.assertEqual(result["native_filter"], "fps=4.000000")
        self.assertEqual(len([line for line in (self.root / "verify/native.framehash").read_text().splitlines()
                              if line and not line.startswith("#")]), 4)

    def test_real_irregular_timeline_cannot_silently_duplicate_source_frames(self):
        video = self.make_video(extra=("-vf", "setpts=if(eq(N\\,3)\\,5\\,N)/(4*TB)", "-fps_mode", "vfr"))
        with self.assertRaisesRegex(VideoRuntimeError, "changes the frame sequence"):
            self.verify(video)
        result = json.loads((self.root / "verify/verification.json").read_text())
        self.assertFalse(result["accepted"])
        self.assertFalse(result["ordered_rgb_frames_identical"])

    def test_multiple_real_video_streams_rejected_before_decode(self):
        source = self.make_video("one.mkv")
        video = self.root / "two.mkv"
        subprocess.run([self.ffmpeg, "-v", "error", "-i", str(source), "-map", "0:v:0", "-map", "0:v:0",
                        "-c", "copy", str(video)], check=True)
        with self.assertRaisesRegex(VideoRuntimeError, "exactly one video stream"):
            self.verify(video)
        self.assertFalse((self.root / "verify/source.framehash").exists())

    def test_same_count_different_hash_order_is_rejected(self):
        # Fabricate only the FFmpeg outputs to exercise the equal-count case;
        # the adjacent integration tests execute the real decoder paths.
        video = self.make_video()
        hashes = ["a" * 64, "b" * 64, "c" * 64, "d" * 64]
        def hash_output(command, *, stdin, stdout, stderr, check):
            order = [0, 1, 2, 3] if "-vf" not in command else [0, 1, 1, 3]
            stdout.write("".join(f"0, {i}, {i}, 1, 3072, {hashes[frame]}\n" for i, frame in enumerate(order)).encode())
            return subprocess.CompletedProcess(command, 0)
        with patch("yasargil.llama_video.subprocess.check_output", return_value=b'{"streams":[{"codec_type":"video","index":0,"r_frame_rate":"4/1"}]}'), \
                patch("yasargil.llama_video.subprocess.run", side_effect=hash_output), \
                self.assertRaisesRegex(VideoRuntimeError, "changes the frame sequence"):
            self.verify(video)
        receipt = json.loads((self.root / "verify/verification.json").read_text())
        self.assertEqual(receipt["source_decoded_frames"], receipt["native_decoded_frames"])
        self.assertEqual(receipt["first_mismatched_frame_index"], 2)


if __name__ == "__main__":
    unittest.main()
