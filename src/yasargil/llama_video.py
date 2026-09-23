"""Strict local llama.cpp transport for one complete, persistent video context.

The caller owns frame provenance and the review loop. This transport requires the
same native video in every request and verifies the decoder's complete frame IDs
for every round, even when the language-model prompt cache reports a hit.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import struct
import subprocess
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from jsonschema import Draft202012Validator

from .qwen_runtime import verified_qwen_runtime
from .qwen_sampling import qwen_non_thinking_parameters, qwen_sampling_receipt, verify_qwen_sampling
from .qwen_video_protocol import VIDEO_PROTOCOL, verify_qwen_video_protocol


class VideoRuntimeError(RuntimeError):
    """An incomplete or unverifiable inference must not become a selection."""


@dataclass(frozen=True)
class RuntimeConfig:
    project_root: Path
    media_path: Path
    log_dir: Path
    context_size: int = 65536
    image_max_tokens: int = 256
    port: int = 0
    startup_timeout: float = 180
    request_timeout: float = 3600


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _save_atomic_exclusive(path: Path, value: Any) -> None:
    """Publish a complete success receipt without replacing an existing one."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".result-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(raw: str | bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _validate_transport_receipts(receipts, metadata, video_path, video_sha256):
    if len(receipts) != 1:
        raise VideoRuntimeError("Expected exactly one complete byte-spooled native video input.")
    receipt = receipts[0]
    if (receipt.get("input_sha256") != video_sha256
            or receipt.get("input_bytes") != video_path.stat().st_size
            or receipt.get("module_sha256") != metadata["module"]["sha256"]
            or receipt.get("transport_manifest_sha256") != metadata["manifest_sha256"]):
        raise VideoRuntimeError("Native FFmpeg transport did not preserve the complete original video bytes.")
    return receipt


def verify_native_video_decode(video_path: Path, *, ffmpeg: str, ffprobe: str,
                               expected_video_frames: int, output_dir: Path) -> dict:
    """Verify b10809's actual source-FPS filter preserves every RGB frame.

    Upstream's ``--video-fps 0`` still applies ``fps=r_frame_rate``. Therefore a
    frame count alone cannot exclude same-count duplication/drop. Independently
    decode both paths and compare the ordered SHA256 hashes of complete RGB24
    frames. The caller additionally checks the original PTS timeline: identical
    pixels alone do not establish correct timestamp spacing.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt: dict[str, Any] = {
        "accepted": False, "expected_video_frames": expected_video_frames,
        "video_path": str(video_path), "comparison": "Ordered SHA256 of complete decoded RGB24 frames, including byte sizes.",
        "timestamp_note": "Pixel identity is separate from PTS fidelity; the source timeline compatibility gate is also required.",
        "upstream_source": "https://github.com/ggml-org/llama.cpp/blob/b10809/tools/mtmd/mtmd-helper.cpp#L565-L681",
    }
    started = time.monotonic()
    try:
        receipt["video_sha256"] = _sha256(video_path)
        transport = None
        if (Path(ffmpeg).parent / "transport.json").is_file():
            from .ffmpeg_transport import transport_metadata
            transport = transport_metadata(Path(ffmpeg).parent)
            if Path(ffprobe) != Path(ffmpeg).parent / "ffprobe":
                raise VideoRuntimeError("Both native preflight programs must use the same pinned transport.")
            receipt["ffmpeg_transport"] = transport
            receipt["byte_transport_receipts"] = {}
        probe_raw = subprocess.check_output([
            ffprobe, "-v", "error", "-show_streams", "-of", "json", str(video_path)])
        (output_dir / "ffprobe.json").write_bytes(probe_raw)
        probe = _strict_json(probe_raw)
        streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"]
        if len(streams) != 1:
            raise VideoRuntimeError("Native video input currently requires exactly one video stream; upstream stream selection would be ambiguous.")
        rate = Fraction(streams[0].get("r_frame_rate", "0/1"))
        if rate <= 0:
            raise VideoRuntimeError("Native video input requires a positive r_frame_rate.")
        # mtmd-helper parses the rational into a C++ float, then prints six
        # decimals when constructing FFmpeg's filter. Reproduce that exactly.
        fps_float = struct.unpack("f", struct.pack("f", float(rate)))[0]
        filter_expression = f"fps={fps_float:.6f}"
        if fps_float <= 0 or not math.isfinite(fps_float):
            raise VideoRuntimeError("Native video FPS cannot be represented by the pinned decoder.")
        receipt.update({"source_r_frame_rate": str(rate), "native_filter": filter_expression,
                        "video_stream_index": streams[0]["index"], "commands": {}})
        decoded = {}
        for name, filtered in (("source", False), ("native", True)):
            command = [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-read_ahead_limit", "-1",
                       "-i", "cache:pipe:0"]
            if filtered:
                command += ["-vf", filter_expression]
            else:
                command += ["-fps_mode", "passthrough"]
            # There is exactly one video stream. Both paths use the same RGB24
            # conversion as the native rawvideo pipe; framehash hashes pixels
            # before encoding instead of retaining a second large RGB video.
            command += ["-an", "-sn", "-dn", "-c:v", "rawvideo", "-pix_fmt", "rgb24",
                        "-f", "framehash", "-hash", "sha256", "pipe:1"]
            receipt["commands"][name] = command
            hash_path = output_dir / f"{name}.framehash"
            with video_path.open("rb") as source, hash_path.open("wb") as hashes, \
                    (output_dir / f"{name}.log").open("wb") as errors:
                subprocess.run(command, stdin=source, stdout=hashes, stderr=errors, check=True)
            if transport is not None:
                from .ffmpeg_transport import transport_receipts
                receipt["byte_transport_receipts"][name] = _validate_transport_receipts(
                    transport_receipts((output_dir / f"{name}.log").read_text()),
                    transport, video_path, receipt["video_sha256"])
            records = []
            for line in hash_path.read_text(encoding="utf-8").splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                fields = [part.strip() for part in line.split(",")]
                if len(fields) != 6 or not re.fullmatch(r"[0-9a-f]{64}", fields[-1]):
                    raise VideoRuntimeError(f"Malformed independent {name} frame-hash output.")
                records.append((int(fields[-2]), fields[-1]))
            decoded[name] = records
        receipt.update({"source_decoded_frames": len(decoded["source"]),
                        "native_decoded_frames": len(decoded["native"]),
                        "ordered_rgb_frames_identical": decoded["source"] == decoded["native"]})
        if len(decoded["source"]) != expected_video_frames:
            raise VideoRuntimeError("Independent source decode does not match the source manifest frame count.")
        if decoded["source"] != decoded["native"]:
            receipt["first_mismatched_frame_index"] = next(
                (index for index, pair in enumerate(zip(decoded["source"], decoded["native"])) if pair[0] != pair[1]),
                min(len(decoded["source"]), len(decoded["native"])))
            raise VideoRuntimeError("The pinned native-video FPS filter changes the frame sequence; refusing to drop, duplicate, or reorder source frames.")
        if _sha256(video_path) != receipt["video_sha256"]:
            raise VideoRuntimeError("The source video changed during native decoder verification.")
        receipt["accepted"] = True
        return receipt
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, VideoRuntimeError) as error:
        receipt["error"] = str(error)
        if isinstance(error, VideoRuntimeError):
            raise
        raise VideoRuntimeError(f"Native video preflight failed: {error}; inspect {output_dir}.") from error
    finally:
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        _save(output_dir / "verification.json", receipt)


class LocalVideoRuntime:
    """Start a project-pinned, loopback-only Qwen server and retain its slot.

    ``video_relative_path`` names a staged file underneath ``media_path``. Staged
    symlinks are allowed: the source manifest, maintained by the caller, names
    and hashes their original targets. ``expected_video_frames`` must come from
    an independent complete source decode, never the model's own log count.
    """

    def __init__(self, config: RuntimeConfig, *, expected_video_frames: int,
                 video_relative_path: str):
        self.config = config
        for name, value, lower, upper in (
            ("context_size", config.context_size, 1, 262144),
            ("image_max_tokens", config.image_max_tokens, 64, 16384),
            ("port", config.port, 0, 65535),
            ("expected_video_frames", expected_video_frames, 1, 100000000),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}.")
        for name in ("startup_timeout", "request_timeout"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite duration.")
        self.media_path = config.media_path.expanduser().resolve()
        self.log_dir = config.log_dir.expanduser().resolve()
        self.video_relative_path = video_relative_path
        self.video_path = self._media_file("file://" + video_relative_path)
        self.expected_video_frames = expected_video_frames
        self._http = build_opener(ProxyHandler({}), _NoRedirect())
        self._server = None
        self._log_handle = None
        self._previous_messages = None
        self._previous_output = None
        self._round_count = 0
        self._video_sha256 = None
        self._ffmpeg_transport = None
        self._source_fps = None
        self._transport_receipt_dir = None
        self.base_url = ""
        self.command: list[str] = []

    def _media_file(self, url: str) -> Path:
        if not isinstance(url, str) or not url.startswith("file://"):
            raise VideoRuntimeError("Media must use a local file:// relative URL.")
        relative = url[len("file://"):]
        path = PurePosixPath(relative)
        # llama.cpp's media URL is a relative path, not a conventional URI host.
        if (not relative or path.is_absolute() or path.as_posix() != relative
                or any(part in (".", "..") for part in path.parts)
                or any(char in relative for char in '\\:?*"<>|%#')
                or any(part != part.strip(" ") or part.endswith(".") for part in path.parts)):
            raise VideoRuntimeError("Media URL must name a simple relative path inside media_path.")
        staged = self.media_path.joinpath(*path.parts)
        if not staged.is_file():
            raise VideoRuntimeError(f"Missing staged media: {staged}")
        return staged

    def __enter__(self):
        if self._server is not None:
            raise VideoRuntimeError("This runtime is already running.")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "server.log"
        if log_path.exists():
            raise VideoRuntimeError(f"Refusing to replace an existing server log: {log_path}")
        root = self.config.project_root.expanduser().resolve()
        try:
            binary, build_receipt = verified_qwen_runtime(root)
        except ValueError as error:
            raise VideoRuntimeError(str(error)) from error
        model = root / ".runtime/models/qwen3.8-27b-q4_k_m.gguf"
        projector = root / ".runtime/models/qwen3.8-27b-mmproj-bf16.gguf"
        for path in (binary, model, projector):
            if not path.is_file():
                raise VideoRuntimeError(f"Missing runtime file: {path}; see docs/LLAMA_CPP.md.")
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        transport_dir = root / ".runtime" / "ffmpeg-safe"
        if transport_dir.exists():
            from .ffmpeg_transport import transport_metadata
            self._ffmpeg_transport = transport_metadata(transport_dir)
            ffmpeg, ffprobe = str(transport_dir / "ffmpeg"), str(transport_dir / "ffprobe")
        if not ffmpeg or not ffprobe or Path(ffmpeg).parent != Path(ffprobe).parent:
            raise VideoRuntimeError("ffmpeg and ffprobe must be installed in the same directory.")
        native_decode = verify_native_video_decode(
            self.video_path, ffmpeg=ffmpeg, ffprobe=ffprobe,
            expected_video_frames=self.expected_video_frames, output_dir=self.log_dir / "native-decode")
        self._source_fps = float(Fraction(native_decode["source_r_frame_rate"]))
        version = subprocess.check_output([str(binary), "--version"], text=True, stderr=subprocess.STDOUT)
        if not re.search(r"\b10809\b", version) or "qwenref1" not in version:
            raise VideoRuntimeError("This verifier requires the pinned Qwen reference runtime build.")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", self.config.port))
            port = probe.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self._video_sha256 = native_decode["video_sha256"]
        self.command = [
            str(binary), "-m", str(model), "--mmproj", str(projector),
            "--alias", "qwen-video", "--host", "127.0.0.1", "--port", str(port),
            "--parallel", "1", "--media-path", str(self.media_path),
            "--video-fps", "0",
            "--video-ffmpeg-dir", str(Path(ffmpeg).parent),
            "--image-min-tokens", "64", "--image-max-tokens", str(self.config.image_max_tokens),
            "-c", str(self.config.context_size), "-ngl", "all", "-fa", "on", "--fit", "off",
            "--reasoning", "on", "--no-context-shift", "--cache-prompt", "--cache-ram", "0",
            "--offline", "--no-webui", "--timeout", str(math.ceil(self.config.request_timeout)),
            "--log-colors", "off", "--log-verbosity", "5", "--log-timestamps", "--perf",
        ]
        setup_path = root / ".runtime/setup.json"
        setup = _strict_json(setup_path.read_bytes()) if setup_path.is_file() else None
        _save(self.log_dir / "runtime.json", {
            "started_at": datetime.now(timezone.utc).isoformat(), "command": self.command,
            "binary_version": version, "setup_manifest": setup,
            "qwen_runtime_build": build_receipt, "video_protocol": VIDEO_PROTOCOL,
            "sampling_profile": qwen_sampling_receipt(enable_thinking=True),
            "model": {"path": str(model), "resolved_path": str(model.resolve()), "bytes": model.stat().st_size},
            "projector": {"path": str(projector), "resolved_path": str(projector.resolve()), "bytes": projector.stat().st_size},
            "model_hash_note": "Digests are recorded from the existing setup manifest; not rehashed by this runtime.",
            "video_relative_path": self.video_relative_path, "video_sha256": self._video_sha256,
            "expected_video_frames": self.expected_video_frames, "video_fps_setting": 0,
            "native_decode_verification": native_decode,
            "context_size": self.config.context_size, "image_max_tokens": self.config.image_max_tokens,
            "ffmpeg_transport": self._ffmpeg_transport,
            "ffmpeg_transport_receipts_directory": str(self.log_dir / "transport-receipts") if self._ffmpeg_transport else None,
            "coverage_note": "Native video input at source FPS. Complete decoded frame IDs must match the independently counted source on every request.",
            "cache_note": "Same-slot prompt caching requested; actual reused token counts are recorded per response. Video decoding and vision reuse are not assumed.",
        })
        self._log_handle = log_path.open("wb")
        try:
            environment = None
            if self._ffmpeg_transport is not None:
                from .ffmpeg_transport import RECEIPT_DIRECTORY_ENV
                self._transport_receipt_dir = self.log_dir / "transport-receipts"
                self._transport_receipt_dir.mkdir()
                environment = {**os.environ, RECEIPT_DIRECTORY_ENV: str(self._transport_receipt_dir)}
            self._server = subprocess.Popen(self.command, stdout=self._log_handle, stderr=subprocess.STDOUT,
                                            env=environment)
            deadline = time.monotonic() + self.config.startup_timeout
            while True:
                if self._server.poll() is not None:
                    raise VideoRuntimeError(f"llama-server stopped during startup; read {log_path}.")
                try:
                    with self._http.open(self.base_url + "/health", timeout=2) as response:
                        if _strict_json(response.read()).get("status") == "ok":
                            return self
                except (URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise VideoRuntimeError(f"llama-server startup timed out; read {log_path}.")
                time.sleep(0.5)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        try:
            if self._server is not None:
                if self._server.poll() is None:
                    self._server.terminate()
                    try:
                        self._server.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        self._server.kill()
                        self._server.wait(timeout=15)
        finally:
            self._server = None
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def _validate_messages(self, messages: list[dict]) -> None:
        if not isinstance(messages, list) or not messages:
            raise VideoRuntimeError("A complete nonempty message history is required.")
        videos = []
        first_user = next((i for i, message in enumerate(messages) if isinstance(message, dict) and message.get("role") == "user"), None)
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
                raise VideoRuntimeError("Unsupported message role.")
            content = message.get("content")
            if isinstance(content, str):
                continue
            if not isinstance(content, list):
                raise VideoRuntimeError("Message content must be text or multimodal parts.")
            for part in content:
                if not isinstance(part, dict):
                    raise VideoRuntimeError("Invalid multimodal part.")
                kind = part.get("type")
                if kind == "text":
                    if not isinstance(part.get("text"), str):
                        raise VideoRuntimeError("Text content must be a string.")
                elif kind in ("input_video", "image_url"):
                    media = part.get(kind)
                    if not isinstance(media, dict):
                        raise VideoRuntimeError("Missing media URL.")
                    self._media_file(media.get("url"))
                    if kind == "input_video":
                        videos.append((index, media.get("url")))
                else:
                    raise VideoRuntimeError(f"Unsupported multimodal input: {kind}")
        if videos != [(first_user, "file://" + self.video_relative_path)]:
            raise VideoRuntimeError("Every request must contain the same complete native video exactly once in the first user message.")
        if self._previous_messages is not None:
            count = len(self._previous_messages)
            if len(messages) <= count or messages[:count] != self._previous_messages:
                raise VideoRuntimeError("A later round must retain the entire earlier message prefix unchanged.")
            assistant = messages[count]
            try:
                is_previous_answer = (assistant.get("role") == "assistant"
                                      and _strict_json(assistant.get("content")) == self._previous_output)
            except (ValueError, TypeError):
                is_previous_answer = False
            if not is_previous_answer:
                raise VideoRuntimeError("The previous validated model response must remain in the conversation.")

    def chat(self, messages: list[dict], *, schema: dict, max_tokens: int,
             round_dir: Path) -> dict:
        if self._server is None or self._server.poll() is not None:
            raise VideoRuntimeError("Use LocalVideoRuntime as a running context manager.")
        if type(max_tokens) is not int or not 0 < max_tokens < self.config.context_size:
            raise VideoRuntimeError("Output token budget must be positive and smaller than the context.")
        self._validate_messages(messages)
        Draft202012Validator.check_schema(schema)
        if _sha256(self.video_path) != self._video_sha256:
            raise VideoRuntimeError("The staged video changed after runtime startup.")
        prior_transport_receipts = set()
        if self._ffmpeg_transport is not None:
            from .ffmpeg_transport import transport_metadata
            if transport_metadata(self._ffmpeg_transport["directory"]) != self._ffmpeg_transport:
                raise VideoRuntimeError("Pinned FFmpeg transport changed after startup.")
            prior_transport_receipts = set(self._transport_receipt_dir.glob("*.json"))
        round_dir = Path(round_dir)
        round_dir.mkdir(parents=True, exist_ok=True)
        if any((round_dir / name).exists() for name in ("request.json", "response.json", "verification.json", "result.json")):
            raise VideoRuntimeError("Refusing to overwrite existing round evidence.")
        request = {
            "model": "qwen-video", "messages": deepcopy(messages), "max_tokens": max_tokens,
            # Isolate thinking mode: retain the previous numerical sampling recipe.
            **qwen_non_thinking_parameters(), "stream": False,
            "samplers": ["penalties", "temperature", "top_k", "top_p", "min_p"],
            "samplers_generated_only": True, "repeat_last_n": max_tokens,
            "cache_prompt": True, "id_slot": 0,
            "chat_template_kwargs": {"enable_thinking": True},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "frame_selection", "strict": True, "schema": schema}},
        }
        payload = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
        # Save the exact bytes submitted, not a reconstructed request.
        (round_dir / "request.json").write_bytes(payload)
        log_path = self.log_dir / "server.log"
        offset = log_path.stat().st_size
        started = time.monotonic()
        verification = {
            "round_index": self._round_count, "video_relative_path": self.video_relative_path,
            "runtime_directory": str(self.log_dir),
            "video_sha256": self._video_sha256, "video_fps_setting": 0,
            "expected_video_frames": self.expected_video_frames,
            "message_count": len(messages), "prior_history_preserved": self._previous_messages is not None,
            "request_sha256": hashlib.sha256(payload).hexdigest(), "log_start_byte": offset,
            "full_source_video_verified": False, "accepted": False,
            "sampling_profile": {**qwen_sampling_receipt(enable_thinking=True),
                                 "samplers": request["samplers"],
                                 "samplers_generated_only": True, "repeat_last_n": max_tokens},
        }
        try:
            with self._http.open(Request(self.base_url + "/v1/chat/completions", data=payload,
                                         headers={"Content-Type": "application/json"}),
                                 timeout=self.config.request_timeout) as response:
                raw = response.read()
            (round_dir / "response.json").write_bytes(raw)
            result = _strict_json(raw)
            segment = self._round_log(log_path, offset, round_dir)
            frame_ids = [int(value) for value in re.findall(r"read_next_frame: frame (\d+) read OK", segment)]
            truncations = [int(value) for value in re.findall(r"\btruncated\s*=\s*(\d+)", segment)]
            verified = frame_ids == list(range(self.expected_video_frames))
            verification.update({
                "decoded_frames": len(frame_ids), "decoded_frame_ids": frame_ids,
                "full_source_video_verified": verified,
                "source_probe_lines": [line for line in segment.splitlines() if re.search(r"\bprobe: \d+x\d+ fps=", line)],
                "context_truncation_observed": any(truncations),
                "log_end_byte": log_path.stat().st_size,
            })
            if self._ffmpeg_transport is not None:
                receipt_paths = sorted(set(self._transport_receipt_dir.glob("*.json")) - prior_transport_receipts)
                transport_receipts = [_strict_json(path.read_bytes()) for path in receipt_paths]
                verification["byte_transport_receipt"] = _validate_transport_receipts(
                    transport_receipts, self._ffmpeg_transport, self.video_path, self._video_sha256)
                verification["byte_transport_receipt_path"] = str(receipt_paths[0])
            if not isinstance(result, dict) or result.get("error"):
                raise VideoRuntimeError(f"Invalid server result: {result}")
            choices = result.get("choices")
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise VideoRuntimeError("The server must return exactly one answer.")
            choice = choices[0]
            verification.update({"finish_reason": choice.get("finish_reason"), "usage": result.get("usage"),
                                 "timings": result.get("timings"), "system_fingerprint": result.get("system_fingerprint")})
            usage = result.get("usage")
            timings = result.get("timings") or {}
            if not isinstance(usage, dict) or not isinstance(timings, dict):
                raise VideoRuntimeError("The server returned invalid usage or timing metadata.")
            details = usage.get("prompt_tokens_details") or {}
            if not isinstance(details, dict):
                raise VideoRuntimeError("The server returned invalid cache metadata.")
            cached = details.get("cached_tokens")
            verification["cache"] = {"requested": True, "cached_tokens": cached,
                                     "timings_cache_n": timings.get("cache_n"),
                                     "prefix_cache_hit_reported": isinstance(cached, int) and cached > 0,
                                     "vision_encoding_reuse_verified": False}
            if not verified:
                raise VideoRuntimeError(f"Complete video decoding was not verified: expected {self.expected_video_frames} contiguous frame IDs, received {len(frame_ids)}.")
            still_count = sum(part.get("type") == "image_url" for message in messages
                              if isinstance(message.get("content"), list) for part in message["content"])
            try:
                verification["qwen_video_protocol"] = verify_qwen_video_protocol(
                    segment, frame_count=self.expected_video_frames, fps=self._source_fps,
                    still_count=still_count)
            except (ValueError, TypeError) as error:
                raise VideoRuntimeError(f"Qwen video presentation verification failed: {error}") from error
            try:
                verification["sampling_profile"] = verify_qwen_sampling(segment, max_tokens, enable_thinking=True)
            except ValueError as error:
                raise VideoRuntimeError(f"Qwen generation settings verification failed: {error}") from error
            if any(truncations) or re.search(r"\bcontext shift:|\btruncating (?:the )?prompt", segment, re.IGNORECASE):
                raise VideoRuntimeError("The server truncated or shifted the context; the answer is rejected.")
            if choice.get("finish_reason") != "stop":
                raise VideoRuntimeError(f"Unfinished model output ({choice.get('finish_reason')}); increase the output budget and rerun.")
            prompt_tokens = usage.get("prompt_tokens")
            if type(prompt_tokens) is not int or prompt_tokens < 1:
                raise VideoRuntimeError("The server did not report valid prompt usage.")
            if prompt_tokens + max_tokens > self.config.context_size:
                raise VideoRuntimeError("The complete prompt plus reserved answer exceeds context capacity; increase context size.")
            message = choice.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise VideoRuntimeError("Qwen returned no selection answer.")
            if message.get("role") != "assistant" or message.get("tool_calls"):
                raise VideoRuntimeError("Qwen must return a direct assistant selection answer.")
            output = _strict_json(content)
            errors = sorted(Draft202012Validator(schema).iter_errors(output), key=lambda err: str(list(err.path)))
            if errors:
                raise VideoRuntimeError(f"Qwen's answer does not match the required schema: {errors[0].message}")
            _save(round_dir / "output.json", output)
            self._previous_messages = deepcopy(messages)
            self._previous_output = deepcopy(output)
            self._round_count += 1
            verification["accepted"] = True
            return {"output": output, "response": result, "verification": verification}
        except HTTPError as error:
            raw = error.read()
            (round_dir / "http-error-body.txt").write_bytes(raw)
            verification["error"] = f"HTTP {error.code}: {raw.decode(errors='replace')}"
            raise VideoRuntimeError(verification["error"]) from error
        except (ValueError, TypeError, KeyError, URLError, TimeoutError, VideoRuntimeError) as error:
            verification["error"] = str(error)
            if isinstance(error, VideoRuntimeError):
                raise
            raise VideoRuntimeError(f"Unusable local video response: {error}") from error
        finally:
            self._round_log(log_path, offset, round_dir)
            verification["elapsed_seconds"] = round(time.monotonic() - started, 3)
            _save(round_dir / "verification.json", verification)
            if verification["accepted"]:
                _save_atomic_exclusive(round_dir / "result.json", {
                    "output": output, "response": result, "verification": verification})

    @staticmethod
    def _round_log(log_path: Path, offset: int, round_dir: Path) -> str:
        with log_path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.read()
        (round_dir / "server-segment.log").write_bytes(raw)
        return raw.decode(errors="replace")
