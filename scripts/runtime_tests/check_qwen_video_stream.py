#!/usr/bin/env python3
"""Check real native decoder callbacks and pixels without model inference."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(harness, video, ffmpeg_dir, mode, prefix):
    with prefix.with_suffix(".jsonl").open("w") as output, prefix.with_suffix(".log").open("w") as errors:
        subprocess.run([str(harness), str(video), str(ffmpeg_dir), mode], stdout=output, stderr=errors, check=True)
    records = [json.loads(line) for line in prefix.with_suffix(".jsonl").read_text().splitlines()]
    log = prefix.with_suffix(".log").read_text()
    decoded = [int(value) for value in re.findall(r"read_next_frame: frame (\d+) read OK", log)]
    groups = [json.loads(value) for value in re.findall(r"YASARGIL_QWEN_VIDEO_REFERENCE_V1 group (\{[^\n]+\})", log)]
    return records, decoded, groups


def check_qwen(records, decoded, groups, expected_hashes, fps):
    count = len(expected_hashes)
    assert decoded == list(range(count)), decoded
    assert records[-1] == {"decoded_frames": count, "emitted_bitmaps": 2 * ((count + 1) // 2)}
    assert len(groups) == (count + 1) // 2
    assert len(records) == len(groups) * 3 + 1
    for group_index, first in enumerate(range(0, count, 2)):
        last = min(first + 1, count - 1)
        seconds = (first / float(fps) + last / float(fps)) / 2.0
        label = f"<{seconds:.1f} seconds>"
        assert groups[group_index] == {
            "group_index": group_index, "frame_indices": [first, last],
            "padded": first == last, "timestamp_seconds": seconds, "label": label,
        }, groups[group_index]
        text, left, right = records[group_index * 3:group_index * 3 + 3]
        assert text == {"text": label}, text
        assert left["bitmap"] == group_index * 2 and right["bitmap"] == group_index * 2 + 1
        assert left["rgb_sha256"] == expected_hashes[first]
        assert right["rgb_sha256"] == expected_hashes[last]
        assert left["bytes"] == right["bytes"]
    return {"accepted": True, "decoded_frames": count, "groups": len(groups),
            "first_label": groups[0]["label"], "last_label": groups[-1]["label"],
            "odd_padding": bool(count % 2), "ordered_pixels_identical": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg-dir", type=Path, required=True)
    parser.add_argument("--legacy-harness", type=Path)
    parser.add_argument("--full-video", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    summary = {"accepted": False, "method": "Actual helper callbacks, exact known fixture pixels, independent unfiltered full-video decode; no model inference.",
               "harness_sha256": hashlib.sha256(args.harness.read_bytes()).hexdigest(), "fixtures": []}
    try:
        for name, count, rate in [("even-6", 6, "1"), ("odd-5", 5, "1"), ("single-1", 1, "1"),
                                  ("half-second-ties", 6, "2"), ("fractional-fps", 7, "30000/1001"),
                                  ("generic-boundary", 12, "1")]:
            pixels = [bytes((x * 3 + y * 7 + channel * 31 + index * 19) % 256
                            for y in range(32) for x in range(32) for channel in range(3)) for index in range(count)]
            hashes = [hashlib.sha256(frame).hexdigest() for frame in pixels]
            # AVI stores a single-frame clip's intended rate explicitly;
            # Matroska can report its 1000 Hz time base as r_frame_rate instead.
            video = args.output / f"{name}.avi"
            command = [str(args.ffmpeg_dir / "ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                       "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "32x32", "-framerate", rate,
                       "-i", "pipe:0", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(video)]
            subprocess.run(command, input=b"".join(pixels), check=True)
            probe = json.loads(subprocess.check_output([
                str(args.ffmpeg_dir / "ffprobe"), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate", "-of", "json", str(video)]))
            assert Fraction(probe["streams"][0]["r_frame_rate"]) == Fraction(rate)
            records, decoded, groups = run(args.harness, video, args.ffmpeg_dir, "qwen", args.output / f"{name}-qwen")
            result = check_qwen(records, decoded, groups, hashes, Fraction(rate))
            result.update({"name": name, "fps": rate})
            if name == "generic-boundary":
                generic, generic_decoded, generic_groups = run(args.harness, video, args.ffmpeg_dir, "generic", args.output / f"{name}-generic")
                assert not generic_groups and generic_decoded == list(range(count))
                expected = [{"text": "Video:"}]
                for index, value in enumerate(hashes):
                    expected.append({"bitmap": index, "rgb_sha256": value, "bytes": 32 * 32 * 3})
                    if index in (0, 10):
                        expected.append({"text": f"[0m{index:.2f}s]"})
                expected.append({"decoded_frames": count, "emitted_bitmaps": count})
                assert generic == expected
                result["generic_callback_contract_unchanged"] = True
                if args.legacy_harness:
                    legacy, legacy_decoded, _ = run(args.legacy_harness, video, args.ffmpeg_dir, "generic", args.output / f"{name}-original")
                    assert generic == legacy and generic_decoded == legacy_decoded
                    result["generic_matches_unmodified_b10809"] = True
            summary["fixtures"].append(result)
            save(args.output / "summary.json", summary)
            print(f"PASS {name}: {count} source frames, {result['groups']} pairs", flush=True)

        if args.full_video:
            print("Starting full-video decode and independent pixel comparison", flush=True)
            source, source_decoded, _ = run(args.harness, args.full_video, args.ffmpeg_dir, "source", args.output / "S1A2-unfiltered-source")
            hashes = [item["rgb_sha256"] for item in source if "bitmap" in item]
            assert len(hashes) == 1254 and source_decoded == list(range(1254))
            actual, decoded, groups = run(args.harness, args.full_video, args.ffmpeg_dir, "qwen", args.output / "S1A2-qwen")
            summary["full_S1A2"] = check_qwen(actual, decoded, groups, hashes, Fraction(1))
            summary["full_S1A2"]["video_path"] = str(args.full_video.resolve())
            summary["full_S1A2"]["source_stream"] = str((args.output / "S1A2-unfiltered-source.jsonl").resolve())
            summary["full_S1A2"]["qwen_stream"] = str((args.output / "S1A2-qwen.jsonl").resolve())
            summary["full_S1A2"]["qwen_group_receipts"] = str((args.output / "S1A2-qwen.log").resolve())
            summary["full_S1A2"]["pixel_digest_algorithm"] = "SHA-256 of complete RGB24 frames"
            print("PASS full S1A2: 1254 source frames, 627 pairs, exact ordered source pixels", flush=True)
        summary["accepted"] = True
    finally:
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        save(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
