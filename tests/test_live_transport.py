"""Replay clocks and the persistent loopback llama-server transport."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib import error

from yasargil.live.clock import RealClock, SimulatedClock, make_clock
from yasargil.live.transport import LlamaServer, ScriptedTransport, TransportError


class ClockTests(unittest.TestCase):
    def test_simulated_clock_only_moves_forward_on_request(self):
        clock = SimulatedClock()
        clock.sleep_until(3000)
        clock.sleep_until(1000)
        self.assertEqual(clock.now_ms(), 3000)
        clock.advance(250)
        self.assertEqual(clock.now_ms(), 3250)

    def test_real_clock_scales_by_speed(self):
        with patch("yasargil.live.clock.time.monotonic", side_effect=[10.0, 10.5]):
            clock = RealClock(speed=4.0)
            self.assertEqual(clock.now_ms(), 2000)

    def test_make_clock(self):
        self.assertIsInstance(make_clock(0), SimulatedClock)
        self.assertIsInstance(make_clock(1), RealClock)
        with self.assertRaises(ValueError):
            make_clock(-1)


def reply(content="ok"):
    return {"model": "qwen-agent", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": content}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}


class ScriptedTransportTests(unittest.TestCase):
    def test_records_requests_and_replays_responses(self):
        transport = ScriptedTransport([reply("a"), lambda request: reply(request["messages"][0]["content"]),
                                       TransportError("boom")])
        self.assertEqual(transport.complete({"messages": [{"role": "user", "content": "q"}]}).envelope, reply("a"))
        self.assertEqual(transport.complete({"messages": [{"role": "user", "content": "echo"}]}).envelope, reply("echo"))
        with self.assertRaises(TransportError):
            transport.complete({"messages": []})
        with self.assertRaises(TransportError):
            transport.complete({"messages": []})
        self.assertEqual(len(transport.requests), 4)


class SpecialistTests(unittest.TestCase):
    def test_specialist_sends_one_image_question_and_returns_text(self):
        from yasargil.live.transport import specialist_from_transport
        transport = ScriptedTransport([reply("A curved needle held by the driver.")])
        ask = specialist_from_transport(transport, "medgemma-specialist")
        self.assertEqual(ask("QUJD", "What is held?"), "A curved needle held by the driver.")
        request = transport.requests[0]
        self.assertEqual(request["model"], "medgemma-specialist")
        parts = request["messages"][0]["content"]
        self.assertIn("What is held?", parts[0]["text"])
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/jpeg;base64,QUJD")

    def test_specialist_errors_surface_as_transport_errors(self):
        from yasargil.live.transport import specialist_from_transport
        ask = specialist_from_transport(ScriptedTransport([{"choices": []}]), "medgemma-specialist")
        with self.assertRaises(TransportError):
            ask("QUJD", "?")


def http_response(payload):
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(payload).encode()
    return response


class LlamaServerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.info = {"runtime_binary": {"path": "/rt/llama-server"}, "model_file": {"path": "/m/qwen.gguf", "sha256": "a"},
                     "projector_file": {"path": "/m/proj.gguf", "sha256": "b"}, "runtime_version": "b10809"}

    def start(self, opener, **options):
        process = MagicMock()
        process.poll.return_value = None
        patches = [patch("yasargil.live.transport.LlamaCppClient.model_info", return_value=self.info),
                   patch("yasargil.live.transport.subprocess.Popen", return_value=process),
                   patch("yasargil.live.transport.build_opener", return_value=opener)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        return LlamaServer(self.root, self.root / "logs", **options), process

    def test_command_is_loopback_with_jinja_tools_and_caching(self):
        opener = MagicMock()
        opener.open.side_effect = [error.URLError("not yet"), http_response({"status": "ok"}),
                                   http_response(reply("hi"))]
        server, process = self.start(opener, context_size=16384, image_max_tokens=320)
        with patch("yasargil.live.transport.time.sleep"):
            with server:
                command = server.command
                self.assertEqual(command[0], "/rt/llama-server")
                self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
                for flag in ("--jinja", "--cache-prompt", "--no-context-shift", "--offline", "--no-webui"):
                    self.assertIn(flag, command)
                self.assertEqual(command[command.index("-c") + 1], "16384")
                self.assertEqual(command[command.index("--image-max-tokens") + 1], "320")
                self.assertEqual(command[command.index("--reasoning") + 1], "off")
                completion = server.transport().complete({"model": "qwen-agent", "messages": []})
                self.assertEqual(completion.envelope["choices"][0]["message"]["content"], "hi")
                self.assertGreaterEqual(completion.elapsed_ms, 0)
                self.assertEqual(json.loads(completion.request_bytes), {"model": "qwen-agent", "messages": []})
        process.terminate.assert_called_once()
        runtime = json.loads((self.root / "logs" / "runtime.json").read_text())
        self.assertEqual(runtime["command"], command)
        self.assertEqual(runtime["model"]["model_file"]["sha256"], "a")

    def test_medgemma_specialist_uses_builtin_gemma_template(self):
        opener = MagicMock()
        opener.open.return_value = http_response({"status": "ok"})
        server, _ = self.start(opener, model="medgemma-27b")
        with patch("yasargil.live.transport.time.sleep"):
            with server:
                self.assertIn("--no-jinja", server.command)
                self.assertEqual(server.command[server.command.index("--chat-template") + 1], "gemma")

    def test_http_errors_become_transport_errors(self):
        opener = MagicMock()
        body = io.BytesIO(json.dumps({"error": {"message": "bad tools"}}).encode())
        opener.open.side_effect = [http_response({"status": "ok"}),
                                   error.HTTPError("http://x", 400, "Bad Request", {}, body)]
        server, _ = self.start(opener)
        with patch("yasargil.live.transport.time.sleep"):
            with server:
                with self.assertRaisesRegex(TransportError, "bad tools"):
                    server.transport().complete({"messages": []})

    def test_server_exit_during_startup_is_reported(self):
        opener = MagicMock()
        opener.open.side_effect = error.URLError("down")
        server, process = self.start(opener)
        process.poll.return_value = 1
        with self.assertRaisesRegex(TransportError, "stopped during startup"):
            with server:
                pass


if __name__ == "__main__":
    unittest.main()
