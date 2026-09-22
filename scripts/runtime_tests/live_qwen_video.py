#!/usr/bin/env python3
"""Small live checks of Qwen presentation, selected stills, and conversation reuse."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from PIL import Image
from yasargil.llama_video import LocalVideoRuntime, RuntimeConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required")
    schema = {"type": "object", "properties": {
        name: {"type": "string"} for name in ("first_color", "last_color", "still_color")},
        "required": ["first_color", "last_color", "still_color"], "additionalProperties": False}
    summary = {"accepted": False, "checks": [], "scope": "Small synthetic clips; not a full S1A2 annotation-quality rerun."}
    started = time.monotonic()
    try:
        for name, frame_count, rounds in (("even-6", 6, 2), ("odd-5", 5, 1)):
            directory = output / name
            media = directory / "media"
            media.mkdir(parents=True)
            colors = [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255)]
            pixels = b"".join(Image.new("RGB", (320, 224), color).tobytes() for color in colors[:frame_count])
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                            "-video_size", "320x224", "-framerate", "1", "-i", "pipe:0", "-c:v", "ffv1",
                            "-pix_fmt", "bgr0", str(media / "video.avi")], input=pixels, check=True)
            Image.new("RGB", (320, 224), "white").save(media / "target.png")
            messages = [{"role": "system", "content": "Inspect the supplied video and separate still. Answer with the requested JSON only."},
                        {"role": "user", "content": [
                            {"type": "input_video", "input_video": {"url": "file://video.avi"}},
                            {"type": "text", "text": "What color fills the first video frame and the last video frame? The following image is a separate still. What color fills it? Use simple English color names."},
                            {"type": "image_url", "image_url": {"url": "file://target.png"}},
                        ]}]
            config = RuntimeConfig(ROOT, media, directory / "runtime", context_size=8192, request_timeout=180)
            print(f"Starting {name}", flush=True)
            with LocalVideoRuntime(config, expected_video_frames=frame_count, video_relative_path="video.avi") as runtime:
                for index in range(rounds):
                    result = runtime.chat(messages, schema=schema, max_tokens=128, round_dir=directory / f"round-{index}")
                    answer = result["output"]
                    expected = {"first_color": "red", "last_color": "blue", "still_color": "white"}
                    colors_match = {key: value.strip().lower() for key, value in answer.items()} == expected
                    summary["checks"].append({"name": name, "round": index, "accepted": result["verification"]["accepted"],
                                              "colors_match": colors_match, "output": answer,
                                              "decoded_source_frames": result["verification"]["decoded_frames"],
                                              "verified_pairs": result["verification"]["qwen_video_protocol"]["video_group_count"],
                                              "sampling_verified": result["verification"]["sampling_profile"]["verified"],
                                              "finish_reason": result["verification"]["finish_reason"]})
                    if not colors_match:
                        raise RuntimeError(f"Unexpected simple-color answer: {answer}")
                    print(f"PASS {name} round {index}: {answer}", flush=True)
                    messages += [result["response"]["choices"][0]["message"],
                                 {"role": "user", "content": "Check the same video and separate still again and answer the same three color fields."}]
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summary["accepted"] = True
    except Exception as error:
        summary["error"] = str(error)
        raise
    finally:
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
