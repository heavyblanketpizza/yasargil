"""Owned, offline llama.cpp servers for byte-retained ordered-image requests.

No service, downloads or alternative backend is used. ``n_ctx`` in a request is
an application launch hint: this client enforces it with the server's -c option.
The complete canonical request, including that hint, is sent unchanged.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import time
from urllib import error, request as urllib_request
import uuid

QWEN_MODEL = "qwen3.8-27b"
MEDGEMMA_MODEL = "medgemma-27b"
MODEL_FILES = {
    QWEN_MODEL: ("qwen3.8-27b-q4_k_m.gguf", "qwen3.8-27b-mmproj-bf16.gguf", "qwen35"),
    MEDGEMMA_MODEL: ("medgemma-27b-q4_k_m.gguf", "medgemma-27b-mmproj-f16.gguf", "gemma3"),
}


class LlamaCppError(RuntimeError):
    """A local runtime, model, transport or response validation failure."""


def encode_request(request: dict) -> bytes:
    if not isinstance(request, dict):
        raise LlamaCppError("llama.cpp request must be a JSON object.")

    def check_keys(value):
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("JSON object keys must be strings")
            for child in value.values():
                check_keys(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check_keys(child)
    try:
        check_keys(request)
        return json.dumps(request, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise LlamaCppError(f"Cannot encode llama.cpp request as JSON: {exc}") from exc


def _object(raw: bytes, context: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"non-finite JSON number {value}")
    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError, AttributeError) as exc:
        raise LlamaCppError(f"Malformed {context} response: {exc}") from exc
    if not isinstance(result, dict):
        raise LlamaCppError(f"Malformed {context} response: expected a JSON object.")
    if "error" in result:
        raise LlamaCppError(f"llama.cpp {context} failed: {result['error']}")
    return result


def build_chat_request(model, messages, schema, num_ctx, num_predict, seed, temperature=0):
    """Convert original-image messages to the native OpenAI-compatible wire form."""
    converted = copy.deepcopy(messages)
    for message in converted:
        if "images" in message:
            images = message.pop("images")
            content = message["content"]
            if not isinstance(content, str) or not isinstance(images, list):
                raise LlamaCppError("Image messages require text and a list of base64 images.")
            parts = [{"type": "text", "text": content}]
            for image in images:
                try:
                    raw = base64.b64decode(image, validate=True)
                except (ValueError, TypeError) as exc:
                    raise LlamaCppError("Invalid base64 image.") from exc
                if not raw:
                    raise LlamaCppError("Empty image.")
                mime = "image/png" if raw.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"
                parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}})
            message["content"] = parts
    return {"model": model, "messages": converted, "stream": False,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "response", "strict": True, "schema": schema}},
            "temperature": temperature, "seed": seed, "max_tokens": num_predict,
            "n_ctx": num_ctx, "cache_prompt": False, "reasoning_effort": "none"}


def _gguf_metadata(path):
    """Read scalar GGUF metadata without loading tensors or tokenizer arrays."""
    with Path(path).open("rb") as handle:
        def unpack(fmt):
            size = struct.calcsize(fmt)
            raw = handle.read(size)
            if len(raw) != size:
                raise LlamaCppError(f"Truncated GGUF header: {path}")
            return struct.unpack(fmt, raw)[0]

        def string(keep=True):
            size = unpack("<Q")
            if size > 100_000_000:
                raise LlamaCppError(f"Invalid GGUF string size: {path}")
            if keep:
                return handle.read(size).decode("utf-8")
            handle.seek(size, 1)

        def value(kind, keep=True):
            scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
            if kind in scalar:
                return unpack("<" + scalar[kind])
            if kind == 8:
                return string(keep)
            if kind == 9:
                subtype, count = unpack("<I"), unpack("<Q")
                if count > 10_000_000 or subtype == 9:
                    raise LlamaCppError(f"Invalid GGUF array: {path}")
                if subtype in scalar:
                    handle.seek(struct.calcsize("<" + scalar[subtype]) * count, 1)
                else:
                    for _ in range(count):
                        value(subtype, False)
                return None
            raise LlamaCppError(f"Unsupported GGUF metadata type {kind}: {path}")
        if handle.read(4) != b"GGUF" or unpack("<I") not in (2, 3):
            raise LlamaCppError(f"Not a supported GGUF file: {path}")
        unpack("<Q")
        count = unpack("<Q")
        if count > 100_000:
            raise LlamaCppError(f"Invalid GGUF metadata count: {path}")
        result = {}
        for _ in range(count):
            key = string()
            result[key] = value(unpack("<I"), not key.startswith("tokenizer."))
        return result


class _NoRedirect(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LlamaCppClient:
    """Sequential client; each request owns a fresh server and always stops it."""

    def __init__(self, project_root=None, timeout=600):
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise LlamaCppError("llama.cpp timeout must be positive finite seconds.")
        self.project_root = Path(project_root or Path.cwd()).expanduser().resolve()
        self.timeout = timeout
        self.last_response_bytes = None
        self.last_runtime_dir = None
        self._fingerprints = {}
        self._opener = urllib_request.build_opener(urllib_request.ProxyHandler({}), _NoRedirect())

    def _fingerprint(self, path):
        stat = path.stat()
        signature = (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if signature not in self._fingerprints:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            after = path.stat()
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != signature[1:]:
                raise LlamaCppError(f"Runtime file changed while hashing: {path}")
            self._fingerprints[signature] = digest.hexdigest()
        return {"path": str(path), "sha256": self._fingerprints[signature]}

    def model_info(self, model):
        if not isinstance(model, str) or model not in MODEL_FILES:
            raise LlamaCppError(f"Unknown llama.cpp model alias {model!r}; choose {', '.join(MODEL_FILES)}. "
                                "Legacy Ollama runs cannot be resumed; start a new output directory.")
        model_name, projector_name, architecture = MODEL_FILES[model]
        binary = self.project_root / ".runtime/llama.cpp/b10809/llama-server"
        model_path = self.project_root / ".runtime/models" / model_name
        projector = self.project_root / ".runtime/models" / projector_name
        for path in (binary, model_path, projector):
            if not path.is_file():
                raise LlamaCppError(f"Missing runtime file: {path}; see docs/LLAMA_CPP.md. No models are downloaded automatically.")
        metadata, vision = _gguf_metadata(model_path), _gguf_metadata(projector)
        if metadata.get("general.architecture") != architecture or metadata.get("general.file_type") != 15:
            raise LlamaCppError(f"Expected {architecture} Q4_K_M model: {model_path}")
        if vision.get("general.architecture") != "clip":
            raise LlamaCppError(f"Expected a separate vision projector GGUF: {projector}")
        try:
            version = subprocess.check_output([str(binary), "--version"], text=True, stderr=subprocess.STDOUT).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise LlamaCppError(f"Cannot inspect llama.cpp runtime: {binary}") from exc
        if not re.search(r"\b10809\b", version):
            raise LlamaCppError("The ordered-image adapter requires pinned llama.cpp b10809.")
        files = {"model_file": self._fingerprint(model_path), "projector_file": self._fingerprint(projector),
                 "runtime_binary": self._fingerprint(binary)}
        libraries = sorted({path.resolve() for path in binary.parent.iterdir()
                            if path.is_file() and (".so" in path.name or path.suffix in {".dylib", ".dll"})})
        files["runtime_binary"]["libraries"] = [self._fingerprint(path) for path in libraries]
        digest = "sha256:" + files["model_file"]["sha256"]
        return {"name": model, "model_name": model, "digest": digest, "model_digest": digest,
                "quantization": "Q4_K_M", "capabilities": ["vision"], "runtime": "llama.cpp",
                "runtime_version": version, "binary_version": version, **files}

    def _read(self, base_url, endpoint, payload=None, timeout=None):
        req = urllib_request.Request(base_url + endpoint, data=None if payload is None else encode_request(payload),
                                     headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with self._opener.open(req, timeout=timeout or self.timeout) as response:
                raw = response.read()
        except error.HTTPError as exc:
            try:
                raw = exc.read()
            except http.client.IncompleteRead as partial:
                raw = partial.partial
            finally:
                exc.close()
            if endpoint == "/v1/chat/completions":
                self.last_response_bytes = raw
            try:
                detail = json.loads(raw).get("error", exc.reason)
                if isinstance(detail, dict):
                    detail = detail.get("message", detail)
            except (ValueError, AttributeError):
                detail = exc.reason
            if 300 <= exc.code < 400:
                detail = "redirects are disabled"
            raise LlamaCppError(f"llama.cpp {endpoint} returned HTTP {exc.code}: {detail}") from exc
        except http.client.IncompleteRead as exc:
            if endpoint == "/v1/chat/completions":
                self.last_response_bytes = exc.partial
            raise LlamaCppError(f"Truncated llama.cpp {endpoint} HTTP response.") from exc
        except (error.URLError, OSError, http.client.HTTPException) as exc:
            raise LlamaCppError(f"Cannot reach owned llama.cpp server for {endpoint}: {exc}") from exc
        if endpoint == "/v1/chat/completions":
            self.last_response_bytes = raw
        return raw

    @staticmethod
    def _validate_chat(data, model):
        if not isinstance(data, dict) or "error" in data:
            raise LlamaCppError("llama.cpp response must be a successful JSON object.")
        if data.get("model") != model:
            raise LlamaCppError("llama.cpp response model mismatch.")
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise LlamaCppError("llama.cpp response must contain exactly one choice.")
        choice = choices[0]
        if choice.get("finish_reason") != "stop" or data.get("truncated"):
            raise LlamaCppError("llama.cpp response is unfinished or truncated (finish_reason must be stop).")
        message = choice.get("message")
        if (not isinstance(message, dict) or message.get("role") != "assistant"
                or not isinstance(message.get("content"), str) or not message["content"].strip()):
            raise LlamaCppError("llama.cpp response must contain assistant text.")
        if (message.get("tool_calls") or message.get("thinking")
                or message.get("reasoning_content") or message.get("reasoning")):
            raise LlamaCppError("llama.cpp returned unrecorded tools or hidden reasoning.")

    def chat_raw(self, request):
        self.last_response_bytes = None
        if not isinstance(request, dict) or request.get("stream") is not False:
            raise LlamaCppError("llama.cpp requests must explicitly set stream=False.")
        if not isinstance(request.get("messages"), list) or not request["messages"]:
            raise LlamaCppError("llama.cpp requests require nonempty messages.")
        context, budget = request.get("n_ctx"), request.get("max_tokens")
        if type(context) is not int or not 512 <= context <= 131072 or type(budget) is not int or not 0 < budget < context:
            raise LlamaCppError("Invalid llama.cpp context/output token budget.")
        for message in request["messages"]:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                raise LlamaCppError("Invalid llama.cpp chat message.")
            content = message.get("content")
            if isinstance(content, str):
                continue
            if not isinstance(content, list) or not content:
                raise LlamaCppError("Message content must be text or image parts.")
            for part in content:
                if not isinstance(part, dict):
                    raise LlamaCppError("Invalid image message part.")
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    continue
                url = part.get("image_url", {}).get("url") if isinstance(part.get("image_url"), dict) else None
                if (part.get("type") != "image_url" or not isinstance(url, str)
                        or not re.fullmatch(r"data:image/(?:jpeg|png);base64,[A-Za-z0-9+/]+=*", url)):
                    raise LlamaCppError("Ordered-image requests require inline image data; remote/file URLs are unsupported.")
        info = self.model_info(request.get("model"))
        run_dir = self.project_root / ".runtime/logs" / ("images-" + uuid.uuid4().hex)
        run_dir.mkdir(parents=True)
        self.last_runtime_dir = run_dir
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        base_url = f"http://127.0.0.1:{port}"
        command = [info["runtime_binary"]["path"], "-m", info["model_file"]["path"],
                   "--mmproj", info["projector_file"]["path"], "--alias", request["model"],
                   "--host", "127.0.0.1", "--port", str(port), "--parallel", "1", "-c", str(context),
                   "-ngl", "all", "-fa", "on", "--fit", "off", "--reasoning", "off",
                   "--no-context-shift", "--cache-ram", "0", "--offline", "--no-webui",
                   "--timeout", str(math.ceil(self.timeout)), "--log-colors", "off"]
        if request["model"] == MEDGEMMA_MODEL:
            # b10809 Jinja wraps Gemma control-token prefills into JSON grammar,
            # which fails sampler initialization. The built-in Gemma formatter
            # preserves its chat protocol and uses the schema grammar directly.
            command.extend(["--no-jinja", "--chat-template", "gemma"])
        (run_dir / "runtime.json").write_bytes(encode_request({"command": command, "model": info,
            "request_sha256": hashlib.sha256(encode_request(request)).hexdigest()}))
        process = None
        with (run_dir / "server.log").open("wb") as log:
            try:
                environment = {key: value for key, value in os.environ.items() if not key.startswith("LLAMA_ARG_")}
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                deadline = time.monotonic() + max(180, min(self.timeout, 600))
                while True:
                    if process.poll() is not None:
                        raise LlamaCppError(f"llama-server stopped during startup; read {run_dir / 'server.log'}.")
                    try:
                        if _object(self._read(base_url, "/health", timeout=2), "health").get("status") == "ok":
                            break
                    except LlamaCppError:
                        pass
                    if time.monotonic() >= deadline:
                        raise LlamaCppError(f"llama-server startup timed out; read {run_dir / 'server.log'}.")
                    time.sleep(0.25)
                self.last_response_bytes = None
                raw = self._read(base_url, "/v1/chat/completions", request)
                envelope = _object(raw, "chat")
                self._validate_chat(envelope, request["model"])
                usage = envelope.get("usage", {})
                if not isinstance(usage, dict):
                    raise LlamaCppError("llama.cpp token usage must be an object.")
                prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
                if (type(prompt) is not int or not 0 < prompt <= context - budget
                        or type(completion) is not int or not 0 < completion <= budget):
                    raise LlamaCppError("llama.cpp token usage is missing or exceeds the context/output budget.")
                (run_dir / "response-receipt.json").write_bytes(encode_request({
                    "response_sha256": hashlib.sha256(raw).hexdigest(), "usage": usage}))
                return raw
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=15)

    def chat(self, request):
        return _object(self.chat_raw(request), "chat")
