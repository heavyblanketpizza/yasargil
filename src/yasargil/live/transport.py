"""Chat transports for the agent: a persistent loopback llama-server, or a script.

Unlike the one-request-per-server image client, the agent needs one warm server
for a whole replay so each step only prefills what it appended. The server uses
the pinned b10809 release with Jinja templates on (native tool calls) and keeps
prompt caching on a single slot. Exact request bytes are returned with every
completion so callers can retain them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from . import LiveError
from ..llama_cpp import MEDGEMMA_MODEL, QWEN_MODEL, LlamaCppClient, LlamaCppError, encode_request

ALIASES = {QWEN_MODEL: "qwen-agent", MEDGEMMA_MODEL: "medgemma-specialist"}


class TransportError(LiveError):
    """The chat backend failed or answered with something unusable."""


@dataclass(frozen=True)
class Completion:
    raw: bytes
    envelope: dict
    elapsed_ms: float
    request_bytes: bytes


def _strict_object(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (ValueError, UnicodeError) as exc:
        raise TransportError(f"Malformed chat response: {exc}") from exc
    if not isinstance(value, dict):
        raise TransportError("Malformed chat response: expected a JSON object.")
    if "error" in value:
        raise TransportError(f"Chat backend error: {value['error']}")
    return value


class ScriptedTransport:
    """Replays canned envelopes, callables or exceptions; records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        request_bytes = encode_request(request)
        if not self.responses:
            raise TransportError("Scripted transport has no more responses.")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        envelope = item(request) if callable(item) else item
        raw = json.dumps(envelope).encode("utf-8")
        return Completion(raw, envelope, 0.0, request_bytes)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LlamaServerTransport:
    def __init__(self, base_url, opener, timeout):
        self.base_url = base_url
        self.opener = opener
        self.timeout = timeout

    def complete(self, request):
        try:
            request_bytes = encode_request(request)
        except LlamaCppError as exc:
            raise TransportError(str(exc)) from exc
        started = time.perf_counter()
        try:
            with self.opener.open(Request(self.base_url + "/v1/chat/completions", data=request_bytes,
                                          headers={"Content-Type": "application/json"}), timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("error", exc.reason)
                if isinstance(detail, dict):
                    detail = detail.get("message", detail)
            except (ValueError, AttributeError, OSError):
                detail = exc.reason
            raise TransportError(f"llama-server returned HTTP {exc.code}: {detail}") from exc
        except (URLError, OSError) as exc:
            raise TransportError(f"Cannot reach llama-server: {exc}") from exc
        elapsed = (time.perf_counter() - started) * 1000
        return Completion(raw, _strict_object(raw), elapsed, request_bytes)


class LlamaServer:
    """A loopback llama-server owned for the duration of a ``with`` block."""

    def __init__(self, project_root, log_dir, *, model=QWEN_MODEL, context_size=32768, image_max_tokens=512,
                 thinking=False, port=0, startup_timeout=300, request_timeout=600):
        if model not in ALIASES:
            raise TransportError(f"Unsupported model alias {model!r}")
        if type(context_size) is not int or not 2048 <= context_size <= 262144:
            raise TransportError("context_size must be an integer between 2048 and 262144")
        self.project_root = Path(project_root)
        self.log_dir = Path(log_dir)
        self.model = model
        self.alias = ALIASES[model]
        self.context_size = context_size
        self.image_max_tokens = image_max_tokens
        self.thinking = thinking
        self.port = port
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.command = []
        self.base_url = ""
        self._process = None
        self._log = None
        self._opener = None

    def __enter__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "server.log"
        if log_path.exists():
            raise TransportError(f"Refusing to replace an existing server log: {log_path}")
        try:
            info = LlamaCppClient(self.project_root, timeout=self.request_timeout).model_info(self.model)
        except LlamaCppError as exc:
            raise TransportError(str(exc)) from exc
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", self.port))
            port = probe.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.command = [
            info["runtime_binary"]["path"], "-m", info["model_file"]["path"], "--mmproj", info["projector_file"]["path"],
            "--alias", self.alias, "--host", "127.0.0.1", "--port", str(port), "--parallel", "1",
            "-c", str(self.context_size), "-ngl", "all", "-fa", "on", "--fit", "off",
            "--reasoning", "on" if self.thinking else "off", "--no-context-shift", "--cache-prompt",
            "--cache-ram", "0", "--image-min-tokens", "64", "--image-max-tokens", str(self.image_max_tokens),
            "--offline", "--no-webui", "--timeout", str(math.ceil(self.request_timeout)),
            "--log-colors", "off", "--log-timestamps", "--perf",
        ]
        if self.model == MEDGEMMA_MODEL:
            self.command += ["--no-jinja", "--chat-template", "gemma"]
        else:
            self.command.append("--jinja")
        (self.log_dir / "runtime.json").write_text(json.dumps({
            "started_at": datetime.now(timezone.utc).isoformat(), "command": self.command, "model": info,
            "alias": self.alias, "context_size": self.context_size, "image_max_tokens": self.image_max_tokens,
            "thinking": self.thinking}, indent=2) + "\n", encoding="utf-8")
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())
        self._log = log_path.open("wb")
        try:
            self._process = subprocess.Popen(self.command, stdout=self._log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + self.startup_timeout
            while True:
                if self._process.poll() is not None:
                    raise TransportError(f"llama-server stopped during startup; read {log_path}.")
                try:
                    with self._opener.open(self.base_url + "/health", timeout=2) as response:
                        if _strict_object(response.read()).get("status") == "ok":
                            return self
                except (URLError, OSError, TransportError):
                    pass
                if time.monotonic() >= deadline:
                    raise TransportError(f"llama-server startup timed out; read {log_path}.")
                time.sleep(0.5)
        except BaseException:
            self.close()
            raise

    def transport(self):
        if self._process is None:
            raise TransportError("Start the server with a 'with' block first.")
        return LlamaServerTransport(self.base_url, self._opener, self.request_timeout)

    def close(self):
        try:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=15)
        finally:
            self._process = None
            if self._log is not None:
                self._log.close()
                self._log = None

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def specialist_from_transport(transport, model, max_tokens=256):
    """A one-image, one-question callable over a vision model, for the agent's ask_specialist tool."""

    def ask(image_b64, question):
        request = {"model": model, "stream": False, "temperature": 0.0, "seed": 42, "max_tokens": max_tokens,
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": f"{question}\nAnswer in one or two sentences about what is visible."},
                       {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}]}]}
        envelope = transport.complete(request).envelope
        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TransportError("Specialist returned no answer.") from exc
        if not isinstance(content, str) or not content.strip():
            raise TransportError("Specialist returned no answer.")
        return content.strip()
    return ask
