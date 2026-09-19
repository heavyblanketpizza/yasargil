"""Byte-preserving stdin adapter for FFmpeg's oversized cache-entry failure.

FFmpeg cache.c stores a contiguous CacheEntry.size in a signed int. Large video
inputs can exceed that range. This adapter spools the complete original byte
stream into a seekable temporary file before invoking the unchanged decoder.
It never changes frames, timestamps, filters, codecs, or output arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile


SCHEMA_VERSION = "ffmpeg-byte-spool-transport-v1"
LOG_PREFIX = "YASARGIL_FFMPEG_TRANSPORT "
RECEIPT_DIRECTORY_ENV = "YASARGIL_FFMPEG_TRANSPORT_RECEIPTS"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Missing FFmpeg transport component: {path}")
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def transport_metadata(directory):
    """Read and verify an installed adapter and its pinned executable components."""
    directory = Path(directory).expanduser().resolve()
    path = directory / "transport.json"
    value = json.loads(path.read_text())
    if value.get("schema_version") != SCHEMA_VERSION or value.get("directory") != str(directory):
        raise ValueError("Invalid FFmpeg transport installation")
    for key in ("module", "python", "real_ffmpeg", "real_ffprobe", "ffmpeg_launcher", "ffprobe_launcher"):
        component = value[key]
        if _identity(component["path"]) != component:
            raise ValueError(f"Pinned FFmpeg transport component changed: {key}")
    if value["module"]["path"] != str(Path(__file__).resolve()):
        raise ValueError("FFmpeg transport installation points to a different adapter module")
    for program in ("ffmpeg", "ffprobe"):
        if value[f"{program}_launcher"]["path"] != str(directory / program):
            raise ValueError("FFmpeg transport launcher path changed")
        if Path(value[f"real_{program}"]["path"]).is_relative_to(directory):
            raise ValueError("FFmpeg transport cannot delegate to itself")
    return {**value, "manifest_sha256": _sha256(path)}


def install_transport(project_root, *, ffmpeg=None, ffprobe=None):
    """Install new project launchers; never replace an existing pinned installation."""
    root = Path(project_root).expanduser().resolve()
    directory = root / ".runtime" / "ffmpeg-safe"
    if directory.exists():
        return transport_metadata(directory)
    programs = {"ffmpeg": ffmpeg or shutil.which("ffmpeg"), "ffprobe": ffprobe or shutil.which("ffprobe")}
    if not all(programs.values()):
        raise ValueError("Both real FFmpeg and FFprobe must be installed")
    components = {"module": _identity(__file__), "python": _identity(sys.executable),
                  **{f"real_{name}": _identity(path) for name, path in programs.items()}}
    directory.mkdir(parents=True)
    for program in ("ffmpeg", "ffprobe"):
        command = [components["python"]["path"], components["module"]["path"],
                   "--transport", str(directory), "--program", program, "--"]
        launcher = directory / program
        launcher.write_text("#!/bin/sh\nexec " + shlex.join(command) + ' "$@"\n', encoding="utf-8")
        launcher.chmod(0o755)
        components[f"{program}_launcher"] = _identity(launcher)
    value = {"schema_version": SCHEMA_VERSION, "directory": str(directory), **components,
             "operation": "Spool every stdin byte unchanged to a secure seekable file; preserve all decode/filter/output options.",
             "reason": "FFmpeg cache:pipe contiguous CacheEntry.size overflows above signed 32-bit range.",
             "upstream_source": "https://github.com/FFmpeg/FFmpeg/blob/master/libavformat/cache.c"}
    (directory / "transport.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return transport_metadata(directory)


def _cache_input(arguments):
    matches = [index for index, value in enumerate(arguments)
               if value == "cache:pipe:0" and index > 0 and arguments[index - 1] == "-i"]
    if len(matches) > 1:
        raise ValueError("Only one complete stdin video may be spooled per FFmpeg invocation")
    return matches[0] if matches else None


def replace_cached_input(arguments, replacement):
    """Replace only this input's cache transport and its cache-only read-ahead option."""
    arguments = list(arguments)
    target = _cache_input(arguments)
    if target is None:
        return arguments
    previous = [index for index, value in enumerate(arguments[:target - 1]) if value == "-i"]
    start = previous[-1] + 2 if previous else 0
    remove = set()
    for index in range(start, target - 1):
        if arguments[index] == "-read_ahead_limit":
            if index + 1 >= target - 1:
                raise ValueError("Missing FFmpeg cache read-ahead value")
            remove.update((index, index + 1))
    arguments[target] = str(replacement)
    return [argument for index, argument in enumerate(arguments) if index not in remove]


def run_adapter(program, arguments, directory, *, stdin=None, stderr=None):
    """Run the pinned program, retaining a receipt for every complete spooled input."""
    if program not in {"ffmpeg", "ffprobe"}:
        raise ValueError("Unknown FFmpeg transport program")
    metadata = transport_metadata(directory)
    real = metadata[f"real_{program}"]["path"]
    if program != "ffmpeg" or _cache_input(arguments) is None:
        return subprocess.call([real, *arguments])
    stdin = sys.stdin.buffer if stdin is None else stdin
    stderr = sys.stderr if stderr is None else stderr
    process = None
    previous_handlers = {}

    def interrupted(signum, _frame):
        if process is not None and process.poll() is None:
            process.send_signal(signum)
        raise KeyboardInterrupt

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        with tempfile.TemporaryDirectory(prefix="yasargil-ffmpeg-input-") as temporary:
            path = Path(temporary) / "complete-input.bin"
            digest, size = hashlib.sha256(), 0
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                while chunk := stdin.read(1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if size == 0:
                raise ValueError("The native video stdin stream is empty")
            rewritten = replace_cached_input(arguments, path)
            receipt = {"schema_version": SCHEMA_VERSION, "input_bytes": size,
                       "input_sha256": digest.hexdigest(), "module_sha256": metadata["module"]["sha256"],
                       "transport_manifest_sha256": metadata["manifest_sha256"],
                       "original_input": "cache:pipe:0", "temporary_input": str(path),
                       "arguments": list(arguments), "decoder_arguments": rewritten,
                       "preserves": "Complete original bytes; all codec, FPS, pixel-format and output options unchanged."}
            stderr.write(LOG_PREFIX + json.dumps(receipt, separators=(",", ":")) + "\n")
            stderr.flush()
            receipt_directory = os.environ.get(RECEIPT_DIRECTORY_ENV)
            if receipt_directory:
                receipt_root = Path(receipt_directory)
                if not receipt_root.is_absolute() or not receipt_root.is_dir():
                    raise ValueError("Native transport receipt directory must already exist at an absolute path")
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=receipt_root,
                                                 prefix=".pending-", suffix=".tmp", delete=False) as receipt_stream:
                    receipt_temporary = Path(receipt_stream.name)
                    json.dump(receipt, receipt_stream, indent=2)
                    receipt_stream.write("\n")
                    receipt_stream.flush()
                    os.fsync(receipt_stream.fileno())
                try:
                    os.link(receipt_temporary, receipt_temporary.with_name(receipt_temporary.stem + ".json"))
                finally:
                    receipt_temporary.unlink(missing_ok=True)
            process = subprocess.Popen([real, *rewritten], stdin=subprocess.DEVNULL)
            return process.wait()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def transport_receipts(log):
    return [json.loads(line[len(LOG_PREFIX):]) for line in log.splitlines() if line.startswith(LOG_PREFIX)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", required=True)
    parser.add_argument("--program", choices=("ffmpeg", "ffprobe"), required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    try:
        return run_adapter(args.program, arguments, args.transport)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError) as error:
        print(f"FFmpeg transport failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
