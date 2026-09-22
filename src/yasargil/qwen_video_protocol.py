"""Check the native Qwen callback stream and the actual tokenizer output."""
from __future__ import annotations

import json
import math
import re


VIDEO_PROTOCOL = "qwen3vl-reference-pairs-v1"
MARKER = "YASARGIL_QWEN_VIDEO_REFERENCE_V1"


def verify_qwen_video_protocol(segment: str, *, frame_count: int, fps: float,
                               still_count: int) -> dict:
    """Reject incomplete grouping even when all original frames were decoded.

    Receipts identify the source members. Independent tokenizer log events
    establish that each prefix label really precedes a two-frame visual group
    and that additional target stills remain separate.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Qwen video grouping requires the independently verified source FPS.")
    modes, groups, events = [], [], []
    for line in segment.splitlines():
        match = re.search(re.escape(MARKER) + r" (mode|group) (\{.*\})$", line)
        if match:
            (modes if match[1] == "mode" else groups).append(json.loads(match[2]))
        match = re.search(r"\badd_text: (<\d+\.\d seconds>)$", line)
        if match:
            events.append(("label", match[1]))
        match = re.search(r"\badd_media: preproc_out has (\d+) entries,", line)
        if match:
            events.append(("images", int(match[1])))
        if re.search(r"\badd_text: \[\d+m\d+(?:\.\d+)?s\]$", line):
            raise ValueError("Legacy trailing video timestamps are present.")
    if (len(modes) != 1 or modes[0].get("temporal_patch_size") != 2
            or modes[0].get("timestamp_position") != "before_pair"
            or modes[0].get("odd_padding") != "repeat_last_frame"
            or not math.isclose(modes[0].get("fps", 0), fps, rel_tol=1e-12)):
        raise ValueError("Missing or incompatible Qwen video processor receipt.")
    expected_events = []
    expected_count = (frame_count + 1) // 2
    if len(groups) != expected_count:
        raise ValueError(f"Expected {expected_count} Qwen frame pairs, found {len(groups)}.")
    for group_index, first in enumerate(range(0, frame_count, 2)):
        last = min(first + 1, frame_count - 1)
        seconds = (first / fps + last / fps) / 2
        label = f"<{seconds:.1f} seconds>"
        expected = {"group_index": group_index, "frame_indices": [first, last],
                    "padded": first == last, "label": label}
        group = groups[group_index]
        if (any(group.get(key) != value for key, value in expected.items())
                or not isinstance(group.get("timestamp_seconds"), (int, float))
                or not math.isclose(group["timestamp_seconds"], seconds, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(f"Qwen frame pair {group_index} does not match source frames and reference time.")
        expected_events.extend([("label", label), ("images", 2)])
    expected_events.extend([("images", 1)] * still_count)
    if events != expected_events:
        raise ValueError("Actual tokenizer labels/frame groups differ from the Qwen reference stream.")
    return {"protocol": VIDEO_PROTOCOL, "verified": True, "source_fps": fps,
            "source_frame_count": frame_count, "video_group_count": len(groups),
            "timestamp_label_count": len(groups), "selected_still_count": still_count,
            "groups": groups, "odd_frame_padding_is_not_a_source_frame": True,
            "tokenizer_stream_verified": True, "pixel_equivalence_to_transformers_verified": False}
