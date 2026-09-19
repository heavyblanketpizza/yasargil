#!/usr/bin/env python3
"""Ask local Qwen about a video using llama.cpp's native video input."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = """Review the supplied video from beginning to end before answering.
Give a short overview followed by a chronological account with approximate
timestamps spanning the whole video. Describe visible changes and the final
visible state. Use the surrounding sequence to interpret individual moments.
Distinguish visible observations from uncertain interpretations, and do not
invent events between sampled frames or infer unsupported outcomes.
Keep the response within 600 words."""


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output-dir", type=Path, help="New directory for this run")
    parser.add_argument("--fps", type=float, default=1, help="Sample rate across the entire video (default: 1)")
    parser.add_argument("--image-max-tokens", type=int, default=256)
    parser.add_argument("--context-size", type=int, default=65536)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    video = args.video.expanduser().resolve()
    binary = ROOT / ".runtime/llama.cpp/b10809/llama-server"
    model = ROOT / ".runtime/models/qwen3.8-27b-q4_k_m.gguf"
    projector = ROOT / ".runtime/models/qwen3.8-27b-mmproj-bf16.gguf"
    for path in (video, binary, model, projector):
        if not path.is_file():
            parser.error(f"Missing file: {path}. See docs/LLAMA_CPP.md.")
    if not (math.isfinite(args.fps) and 0 < args.fps and 64 <= args.image_max_tokens and 0 < args.max_tokens < args.context_size <= 262144):
        parser.error("Use fps > 0, image-max-tokens >= 64, and 0 < max-tokens < context-size <= 262144.")
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    if ".." in video.name or any(char in video.name for char in ':?*"<>|') or video.name != video.name.strip(" ") or video.name.endswith("."):
        parser.error("This llama.cpp build rejects that filename; use a simple video filename without '..', ':', '?', '*', quotes, angle brackets, '|', or edge spaces/dots.")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        parser.error("Install FFmpeg first: brew install ffmpeg")
    if Path(ffmpeg).parent != Path(ffprobe).parent:
        parser.error("ffmpeg and ffprobe must be available in the same directory.")
    with socket.socket() as probe_socket:
        probe_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe_socket.bind(("127.0.0.1", args.port))
        except OSError as error:
            parser.error(f"Cannot use localhost port {args.port}: {error}; choose --port.")
    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT
    if not prompt.strip():
        parser.error("The prompt must not be empty.")
    probe = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration:format=duration",
        "-of", "json", str(video),
    ], text=True))
    if not probe.get("streams"):
        parser.error("The input has no video stream.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output_dir or ROOT / "outputs/llama_cpp" / stamp).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    save_json(output / "ffprobe.json", probe)
    digest = hashlib.sha256()
    with video.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    video_hash = digest.hexdigest()
    # Independently count the same FPS filter output before the expensive model
    # call. A contiguous decoder prefix alone cannot prove full-video coverage.
    print("Checking that the video decodes completely…", flush=True)
    with (output / "decode-check.log").open("w", encoding="utf-8") as decode_log:
        progress = subprocess.check_output([
            ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-xerror",
            "-i", str(video), "-map", "0:v:0", "-vf", f"fps={args.fps:.6f}",
            "-an", "-f", "null", "-", "-progress", "pipe:1", "-nostats",
        ], stderr=decode_log, text=True)
    (output / "decode-progress.txt").write_text(progress, encoding="utf-8")
    counts = re.findall(r"^frame=(\d+)$", progress, re.MULTILINE)
    if not counts or int(counts[-1]) == 0 or "progress=end" not in progress:
        raise RuntimeError("Could not verify complete video decoding at the requested FPS.")
    expected_frames = int(counts[-1])
    command = [
        str(binary), "-m", str(model), "--mmproj", str(projector),
        "--alias", "qwen-video", "--host", "127.0.0.1", "--port", str(args.port),
        "--parallel", "1", "--media-path", str(video.parent),
        "--video-fps", str(args.fps), "--video-timestamp-interval", "2000",
        "--video-ffmpeg-dir", str(Path(ffmpeg).parent),
        "--image-min-tokens", "64", "--image-max-tokens", str(args.image_max_tokens),
        "-c", str(args.context_size), "-ngl", "all", "-fa", "on",
        "--reasoning", "off", "--no-context-shift", "--offline", "--no-webui",
        "--timeout", "3600", "--log-colors", "off", "--log-verbosity", "5", "--log-timestamps", "--perf",
    ]
    request = {
        "model": "qwen-video", "messages": [{"role": "user", "content": [
            {"type": "input_video", "input_video": {"url": "file://" + video.name}},
            {"type": "text", "text": prompt},
        ]}], "max_tokens": args.max_tokens, "temperature": 0.2, "seed": 42, "stream": False,
    }
    save_json(output / "request.json", request)
    save_json(output / "run.json", {
        "started_at": stamp, "video": str(video), "video_sha256": video_hash,
        "sample_fps": args.fps, "context_size": args.context_size,
        "image_max_tokens": args.image_max_tokens, "command": command,
        "expected_sampled_frames": expected_frames,
        "coverage_note": "Compared with independent full FFmpeg decoding at the same FPS; this does not imply every original frame is sampled.",
    })
    # Explicitly bypass proxies: the video and request stay on this computer.
    http = build_opener(ProxyHandler({}))
    base = f"http://127.0.0.1:{args.port}"
    started = time.monotonic()
    print(f"Starting Qwen. Progress log: {output / 'server.log'}", flush=True)
    with (output / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"llama-server stopped during startup; read {output / 'server.log'}")
                try:
                    with http.open(base + "/health", timeout=2) as response:
                        if json.load(response).get("status") == "ok":
                            break
                except (URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Qwen did not become ready within 180 seconds.")
                time.sleep(0.5)
            print("Reading the complete video timeline, then generating an answer…", flush=True)
            payload = Request(base + "/v1/chat/completions", data=json.dumps(request).encode(), headers={"Content-Type": "application/json"})
            with http.open(payload, timeout=3600) as response:
                result = json.load(response)
            save_json(output / "response.json", result)
            choice = result["choices"][0]
            content = choice["message"].get("content")
            if not content or not content.strip():
                raise RuntimeError("Qwen returned no answer; inspect response.json and server.log.")
            (output / "response.md").write_text(content + "\n", encoding="utf-8")
            frame_ids = [int(value) for value in re.findall(r"read_next_frame: frame (\d+) read OK", (output / "server.log").read_text())]
            save_json(output / "verification.json", {
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "decoded_frames": len(frame_ids),
                "expected_sampled_frames": expected_frames,
                "full_sampled_timeline_verified": frame_ids == list(range(expected_frames)),
                "decoded_ids_contiguous_from_zero": bool(frame_ids) and frame_ids == list(range(len(frame_ids))),
                "finish_reason": choice.get("finish_reason"), "usage": result.get("usage"),
                "timings": result.get("timings"),
            })
            if frame_ids != list(range(expected_frames)):
                raise RuntimeError(f"Expected {expected_frames} sampled frames but could not verify them all; inspect server.log.")
            print(content)
            print(f"\nSaved answer: {output / 'response.md'}", flush=True)
            if choice.get("finish_reason") == "length":
                print("The answer reached the output limit; increase --max-tokens for a longer answer.")
        except HTTPError as error:
            detail = error.read().decode(errors="replace")
            (output / "error.txt").write_text(detail, encoding="utf-8")
            raise RuntimeError(f"Local server returned HTTP {error.code}: {detail}") from error
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
