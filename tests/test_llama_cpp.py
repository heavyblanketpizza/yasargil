"""Network-free coverage of the owned native llama.cpp image transport."""
import base64
from contextlib import ExitStack
import copy
import hashlib
import http.client
import io
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib import error

from yasargil.llama_cpp import (
    LlamaCppClient, LlamaCppError, MEDGEMMA_MODEL, MODEL_FILES, QWEN_MODEL,
    _gguf_metadata, _object, build_chat_request, encode_request,
)


def gguf(metadata):
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw
    content = b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata))
    for key, value in metadata.items():
        content += string(key)
        if isinstance(value, str):
            content += struct.pack("<I", 8) + string(value)
        else:
            content += struct.pack("<II", 4, value)
    return content


def reply(model=MEDGEMMA_MODEL):
    return {"model": model, "choices": [{"index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": '{"result":"visible"}'}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        "native_metadata": {"preserve": [1, "all bytes", True]}}


def response(raw=None, failure=None):
    result = MagicMock()
    result.__enter__.return_value = result
    if failure is not None:
        result.read.side_effect = failure
    else:
        result.read.return_value = raw
    return result


class LlamaCppTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.binary = self.root / ".runtime/llama.cpp/b10809/llama-server"
        self.binary.parent.mkdir(parents=True)
        self.binary.write_bytes(b"synthetic pinned runtime")
        model_root = self.root / ".runtime/models"
        model_root.mkdir()
        for model_file, projector_file, architecture in MODEL_FILES.values():
            (model_root / model_file).write_bytes(gguf({
                "general.architecture": architecture, "general.file_type": 15}))
            (model_root / projector_file).write_bytes(gguf({"general.architecture": "clip"}))
        self.client = LlamaCppClient(self.root, timeout=10)
        self.request = build_chat_request(MEDGEMMA_MODEL,
            [{"role": "user", "content": "Inspect the supplied image."}],
            {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"]},
            num_ctx=8192, num_predict=1024, seed=42)
        self.version = patch("yasargil.llama_cpp.subprocess.check_output", return_value="llama.cpp version: 10809\n")
        self.version_mock = self.version.start()
        self.addCleanup(self.version.stop)
        # No test may reach a network endpoint, even if a path under test changes.
        self.opener = MagicMock()
        self.client._opener = self.opener
        self.process = MagicMock()
        self.process.poll.return_value = None
        self.socket = MagicMock()
        self.socket.__enter__.return_value = self.socket
        self.socket.getsockname.return_value = ("127.0.0.1", 43210)

    def runtime_mocks(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.popen = stack.enter_context(patch("yasargil.llama_cpp.subprocess.Popen", return_value=self.process))
        stack.enter_context(patch("yasargil.llama_cpp.socket.socket", return_value=self.socket))
        stack.enter_context(patch("yasargil.llama_cpp.time.sleep"))
        return stack

    def serve(self, raw, *, read_failure=None):
        self.opener.open.side_effect = [response(b'{"status":"ok"}'), response(raw, read_failure)]

    def test_request_preserves_image_order_and_schema_without_mutating_inputs(self):
        first = base64.b64encode(b"\x89PNG\r\n\x1a\nfirst image").decode()
        second = base64.b64encode(b"\xff\xd8\xffsecond image").decode()
        messages = [{"role": "system", "content": "Review precisely."},
                    {"role": "user", "content": "Frame first, then second.", "images": [first, second]},
                    {"role": "user", "content": "Return the review."}]
        original = copy.deepcopy(messages)
        schema = self.request["response_format"]["json_schema"]["schema"]
        request = build_chat_request(MEDGEMMA_MODEL, messages, schema, 8192, 1024, 7)
        self.assertEqual(messages, original)
        self.assertEqual(request["messages"][0], messages[0])
        self.assertEqual(request["messages"][-1], messages[-1])
        self.assertEqual(request["messages"][1], {"role": "user", "content": [
            {"type": "text", "text": "Frame first, then second."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + first}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + second}}]})
        self.assertEqual(request["response_format"], {"type": "json_schema", "json_schema": {
            "name": "response", "strict": True, "schema": schema}})
        self.assertEqual((request["n_ctx"], request["max_tokens"], request["seed"]), (8192, 1024, 7))
        self.assertFalse({"format", "options", "images"} & set(request))

    def test_invalid_images_are_rejected_before_any_runtime_call(self):
        for image in ("", "not base64!", None):
            with self.subTest(image=image), self.assertRaises(LlamaCppError):
                build_chat_request(MEDGEMMA_MODEL, [{"role": "user", "content": "x", "images": [image]}],
                                   {}, 8192, 1024, 42)
        self.opener.open.assert_not_called()

    def test_exact_native_wire_bytes_and_raw_response_survive_success(self):
        raw = json.dumps(reply(), indent=3).encode() + b"\n\n"
        self.serve(raw)
        with self.runtime_mocks(), patch.dict("os.environ", {"LLAMA_ARG_MODEL": "untrusted-model", "KEEP_THIS": "yes"}):
            self.assertEqual(self.client.chat_raw(self.request), raw)
        self.assertEqual(self.client.last_response_bytes, raw)
        health, chat = [call.args[0] for call in self.opener.open.call_args_list]
        self.assertEqual(health.full_url, "http://127.0.0.1:43210/health")
        self.assertEqual(chat.full_url, "http://127.0.0.1:43210/v1/chat/completions")
        self.assertEqual(chat.data, encode_request(self.request))
        command = self.popen.call_args.args[0]
        for option, value in (("-c", "8192"), ("--parallel", "1"), ("--alias", MEDGEMMA_MODEL),
                              ("--host", "127.0.0.1"), ("--port", "43210")):
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertIn("--offline", command)
        self.assertIn("--no-context-shift", command)
        self.assertIn("--no-jinja", command)
        self.assertEqual(command[command.index("--chat-template") + 1], "gemma")
        self.assertNotIn("LLAMA_ARG_MODEL", self.popen.call_args.kwargs["env"])
        self.assertEqual(self.popen.call_args.kwargs["env"]["KEEP_THIS"], "yes")
        self.process.terminate.assert_called_once()
        self.process.wait.assert_called_once_with(timeout=15)
        receipt = json.loads((self.client.last_runtime_dir / "response-receipt.json").read_bytes())
        self.assertEqual(receipt["response_sha256"], hashlib.sha256(raw).hexdigest())
        runtime = json.loads((self.client.last_runtime_dir / "runtime.json").read_bytes())
        self.assertEqual(runtime["request_sha256"], hashlib.sha256(chat.data).hexdigest())

    def test_qwen_preserves_default_jinja_template_without_gemma_override(self):
        raw = encode_request(reply(QWEN_MODEL))
        self.serve(raw)
        with self.runtime_mocks():
            self.assertEqual(self.client.chat_raw({**self.request, "model": QWEN_MODEL}), raw)
        command = self.popen.call_args.args[0]
        self.assertEqual(command[command.index("--alias") + 1], QWEN_MODEL)
        self.assertNotIn("--no-jinja", command)
        self.assertNotIn("--chat-template", command)
        self.process.terminate.assert_called_once()

    def test_model_info_checks_both_ggufs_and_fingerprints_exact_local_bytes(self):
        for model in (MEDGEMMA_MODEL, QWEN_MODEL):
            with self.subTest(model=model):
                info = self.client.model_info(model)
                self.assertEqual((info["name"], info["runtime"], info["quantization"]), (model, "llama.cpp", "Q4_K_M"))
                for key in ("model_file", "projector_file", "runtime_binary"):
                    path = Path(info[key]["path"])
                    self.assertEqual(info[key]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(info["digest"], "sha256:" + info["model_file"]["sha256"])
                self.assertEqual(info["runtime_version"], "llama.cpp version: 10809")
        self.opener.open.assert_not_called()

    def test_changed_projector_bytes_refresh_its_fingerprint(self):
        before = self.client.model_info(MEDGEMMA_MODEL)
        path = Path(before["projector_file"]["path"])
        path.write_bytes(path.read_bytes() + b"changed tensors")
        after = self.client.model_info(MEDGEMMA_MODEL)
        self.assertNotEqual(before["projector_file"]["sha256"], after["projector_file"]["sha256"])
        self.assertEqual(before["model_file"], after["model_file"])

    def test_runtime_libraries_are_fingerprinted_deduplicated_and_refreshed(self):
        library = self.binary.parent / "libllama.dylib"
        library.write_bytes(b"native implementation one")
        (self.binary.parent / "libllama.1.dylib").symlink_to(library.name)
        before = self.client.model_info(MEDGEMMA_MODEL)
        self.assertEqual(before["runtime_binary"]["libraries"], [{
            "path": str(library), "sha256": hashlib.sha256(library.read_bytes()).hexdigest()}])
        library.write_bytes(b"native implementation two")
        after = self.client.model_info(MEDGEMMA_MODEL)
        self.assertNotEqual(before["runtime_binary"], after["runtime_binary"])
        self.assertEqual(before["runtime_binary"]["sha256"], after["runtime_binary"]["sha256"])

    def test_http_transport_disables_proxy_configuration_and_redirects(self):
        with patch("yasargil.llama_cpp.urllib_request.build_opener") as build:
            LlamaCppClient(self.root)
        proxy, redirects = build.call_args.args
        self.assertEqual(proxy.proxies, {})
        self.assertIsNone(redirects.redirect_request(None, None, 302, "redirect", {}, "https://remote.invalid"))

    def test_wrong_architecture_quantization_projector_or_runtime_version_fail(self):
        model_file, projector_file, _ = MODEL_FILES[MEDGEMMA_MODEL]
        model = self.root / ".runtime/models" / model_file
        projector = self.root / ".runtime/models" / projector_file
        originals = {path: path.read_bytes() for path in (model, projector)}
        variants = [(model, gguf({"general.architecture": "qwen35", "general.file_type": 15})),
                    (model, gguf({"general.architecture": "gemma3", "general.file_type": 7})),
                    (projector, gguf({"general.architecture": "gemma3"}))]
        for path, raw in variants:
            with self.subTest(path=path.name, raw=raw):
                path.write_bytes(raw)
                with self.assertRaises(LlamaCppError):
                    self.client.model_info(MEDGEMMA_MODEL)
                path.write_bytes(originals[path])
        self.version_mock.return_value = "llama.cpp version: 99999"
        with self.assertRaisesRegex(LlamaCppError, "pinned llama.cpp"):
            self.client.model_info(MEDGEMMA_MODEL)
        self.opener.open.assert_not_called()

    def test_malformed_gguf_headers_are_rejected(self):
        path = self.root / "invalid.gguf"
        for raw in (b"not a model", b"GGUF", b"GGUF" + struct.pack("<IQQ", 999, 0, 0),
                    b"GGUF" + struct.pack("<IQQ", 3, 0, 100001)):
            with self.subTest(raw=raw):
                path.write_bytes(raw)
                with self.assertRaises(LlamaCppError):
                    _gguf_metadata(path)

    def test_missing_files_legacy_aliases_and_urls_never_launch_or_download(self):
        with self.runtime_mocks():
            for alias in ("medgemma:27b", "qwen3.5:27b", "http://127.0.0.1:11434", "https://models.invalid/model.gguf"):
                with self.subTest(alias=alias), self.assertRaisesRegex(LlamaCppError, "Unknown llama.cpp model alias"):
                    self.client.chat_raw({**self.request, "model": alias})
            projector = self.root / ".runtime/models" / MODEL_FILES[MEDGEMMA_MODEL][1]
            projector.unlink()
            with self.assertRaisesRegex(LlamaCppError, "No models are downloaded automatically"):
                self.client.chat_raw(self.request)
            self.popen.assert_not_called()
        self.opener.open.assert_not_called()
        self.assertFalse((self.root / ".runtime/logs").exists())

    def test_invalid_request_budgets_fail_before_launch(self):
        variants = [{"n_ctx": True}, {"n_ctx": 511}, {"n_ctx": 131073}, {"max_tokens": 0},
                    {"max_tokens": 8192}, {"max_tokens": True}, {"stream": True}, {"messages": []}]
        with self.runtime_mocks():
            for changed in variants:
                with self.subTest(changed=changed), self.assertRaises(LlamaCppError):
                    self.client.chat_raw({**self.request, **changed})
            self.popen.assert_not_called()
        self.opener.open.assert_not_called()

    def test_invalid_timeout_is_rejected(self):
        for timeout in (True, 0, -1, float("inf"), float("nan"), "ten"):
            with self.subTest(timeout=timeout), self.assertRaises(LlamaCppError):
                LlamaCppClient(self.root, timeout=timeout)

    def test_unfinished_malformed_and_wrong_model_replies_retain_raw_bytes_and_stop_server(self):
        variants = [b'{"choices":', b'{"model":"x","model":"y"}', b'{"value":NaN}']
        for reason in (None, "length", "tool_calls"):
            value = reply()
            value["choices"][0]["finish_reason"] = reason
            variants.append(encode_request(value))
        for changed in ({"model": "another-model"}, {"truncated": True}, {"choices": []}):
            variants.append(encode_request({**reply(), **changed}))
        for key in ("thinking", "reasoning_content", "reasoning", "tool_calls"):
            value = reply()
            value["choices"][0]["message"][key] = "unexpected hidden output"
            variants.append(encode_request(value))
        for index, raw in enumerate(variants):
            with self.subTest(index=index), self.runtime_mocks():
                self.process.reset_mock()
                self.serve(raw)
                with self.assertRaises(LlamaCppError):
                    self.client.chat_raw(self.request)
                self.assertEqual(self.client.last_response_bytes, raw)
                self.process.terminate.assert_called_once()
                self.assertFalse((self.client.last_runtime_dir / "response-receipt.json").exists())

    def test_missing_or_over_budget_native_usage_keeps_rejected_raw_response(self):
        variants = [None, [], {}, {"prompt_tokens": 0, "completion_tokens": 1},
                    {"prompt_tokens": 7169, "completion_tokens": 1},
                    {"prompt_tokens": True, "completion_tokens": 1},
                    {"prompt_tokens": 1, "completion_tokens": 0},
                    {"prompt_tokens": 1, "completion_tokens": 1025},
                    {"prompt_tokens": 1, "completion_tokens": True}]
        for usage in variants:
            raw = encode_request({**reply(), "usage": usage})
            with self.subTest(usage=usage), self.runtime_mocks():
                self.process.reset_mock()
                self.serve(raw)
                with self.assertRaises(LlamaCppError):
                    self.client.chat_raw(self.request)
                self.assertEqual(self.client.last_response_bytes, raw)
                self.process.terminate.assert_called_once()

    def test_http_error_and_truncated_body_keep_exact_partial_chat_bytes(self):
        errors = [error.HTTPError("http://127.0.0.1/chat", 500, "failed", {}, io.BytesIO(b"server failure body")),
                  http.client.IncompleteRead(b"partial response bytes", 90)]
        for exc in errors:
            with self.subTest(kind=type(exc).__name__), self.runtime_mocks():
                self.process.reset_mock()
                if isinstance(exc, error.HTTPError):
                    self.opener.open.side_effect = [response(b'{"status":"ok"}'), exc]
                    expected = b"server failure body"
                else:
                    self.serve(None, read_failure=exc)
                    expected = exc.partial
                with self.assertRaises(LlamaCppError):
                    self.client.chat_raw(self.request)
                self.assertEqual(self.client.last_response_bytes, expected)
                self.process.terminate.assert_called_once()

    def test_connection_failure_after_health_does_not_save_health_as_chat_response(self):
        self.client.last_response_bytes = b"stale earlier chat"
        self.opener.open.side_effect = [response(b'{"status":"ok"}'), error.URLError("connection closed")]
        with self.runtime_mocks(), self.assertRaises(LlamaCppError):
            self.client.chat_raw(self.request)
        self.assertIsNone(self.client.last_response_bytes)
        self.process.terminate.assert_called_once()

    def test_http_failure_reports_server_detail_and_retains_its_exact_body(self):
        raw = b'{"error":{"message":"Failed to initialize grammar sampler","type":"server_error"}}\n'
        failure = error.HTTPError("http://127.0.0.1/chat", 400, "Bad Request", {}, io.BytesIO(raw))
        self.opener.open.side_effect = [response(b'{"status":"ok"}'), failure]
        with self.runtime_mocks(), self.assertRaisesRegex(LlamaCppError, "HTTP 400.*Failed to initialize grammar sampler") as caught:
            self.client.chat_raw(self.request)
        self.assertNotIn("redirect", str(caught.exception))
        self.assertEqual(self.client.last_response_bytes, raw)
        self.process.terminate.assert_called_once()

    def test_startup_timeout_preserves_health_log_but_no_chat_response(self):
        self.opener.open.return_value = response(b'{"status":"loading"}')
        with self.runtime_mocks(), patch("yasargil.llama_cpp.time.monotonic", side_effect=[0, 181]):
            with self.assertRaisesRegex(LlamaCppError, "startup timed out"):
                self.client.chat_raw(self.request)
        self.assertEqual(self.opener.open.call_count, 1)
        self.assertIsNone(self.client.last_response_bytes)
        self.process.terminate.assert_called_once()
        self.assertTrue((self.client.last_runtime_dir / "server.log").exists())

    def test_process_exit_during_startup_does_not_attempt_chat_or_terminate_exited_process(self):
        self.process.poll.return_value = 1
        with self.runtime_mocks(), self.assertRaisesRegex(LlamaCppError, "stopped during startup"):
            self.client.chat_raw(self.request)
        self.opener.open.assert_not_called()
        self.process.terminate.assert_not_called()
        self.assertIsNone(self.client.last_response_bytes)

    def test_cancellation_stops_owned_server_and_escalates_if_terminate_times_out(self):
        self.opener.open.side_effect = [response(b'{"status":"ok"}'), KeyboardInterrupt()]
        self.process.wait.side_effect = [subprocess.TimeoutExpired("llama-server", 15), 0]
        with self.runtime_mocks(), self.assertRaises(KeyboardInterrupt):
            self.client.chat_raw(self.request)
        self.process.terminate.assert_called_once()
        self.process.kill.assert_called_once()
        self.assertEqual(self.process.wait.call_count, 2)
        self.assertIsNone(self.client.last_response_bytes)

    def test_json_encoder_and_parser_reject_ambiguous_nonfinite_data(self):
        for value in ({"n": float("nan")}, {1: "not a string key"}, {"nested": {None: "x"}}, []):
            with self.subTest(value=value), self.assertRaises(LlamaCppError):
                encode_request(value)
        for raw in (b'[]', b'{"a":1,"a":2}', b'{"n":Infinity}', b'{"error":"failed"}', b'\xff'):
            with self.subTest(raw=raw), self.assertRaises(LlamaCppError):
                _object(raw, "chat")


if __name__ == "__main__":
    unittest.main()
