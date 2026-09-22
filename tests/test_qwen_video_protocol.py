"""Failure boundaries for the real processor/tokenizer log contract."""
from copy import deepcopy
import json
import unittest

from yasargil.qwen_video_protocol import MARKER, verify_qwen_video_protocol


class QwenVideoProtocolTests(unittest.TestCase):
    def stream(self, groups=None, *, fps=1.0, stills=1):
        groups = deepcopy(groups if groups is not None else [
            {"group_index": 0, "frame_indices": [0, 1], "padded": False,
             "timestamp_seconds": 0.5, "label": "<0.5 seconds>"},
            {"group_index": 1, "frame_indices": [2, 2], "padded": True,
             "timestamp_seconds": 2.0, "label": "<2.0 seconds>"},
        ])
        mode = {"temporal_patch_size": 2, "timestamp_position": "before_pair",
                "odd_padding": "repeat_last_frame", "fps": fps}
        lines = [f"I {MARKER} mode " + json.dumps(mode)]
        for group in groups:
            lines.extend([f"D {MARKER} group " + json.dumps(group),
                          "D add_text: " + group["label"],
                          "D add_media: preproc_out has 2 entries, grid_x = 0"])
        lines.extend(["D add_media: preproc_out has 1 entries, grid_x = 0"] * stills)
        return "\n".join(lines)

    def verify(self, stream, **kwargs):
        return verify_qwen_video_protocol(stream, frame_count=kwargs.get("frame_count", 3),
                                          fps=kwargs.get("fps", 1.0), still_count=1)

    def test_odd_padding_does_not_create_a_source_frame(self):
        result = self.verify(self.stream())
        self.assertEqual(result["source_frame_count"], 3)
        self.assertEqual(result["groups"][-1]["frame_indices"], [2, 2])
        self.assertEqual(result["groups"][-1]["label"], "<2.0 seconds>")

    def test_timestamp_after_group_rejected_despite_correct_receipts(self):
        stream = self.stream().replace("D add_text: <0.5 seconds>\nD add_media: preproc_out has 2 entries, grid_x = 0",
                                       "D add_media: preproc_out has 2 entries, grid_x = 0\nD add_text: <0.5 seconds>")
        with self.assertRaisesRegex(ValueError, "Actual tokenizer"):
            self.verify(stream)

    def test_missing_wrong_or_shifted_groups_rejected(self):
        for old, new in [("[0, 1]", "[1, 2]"), ("[2, 2]", "[2, 3]"),
                         ('"padded": true', '"padded": false'),
                         ('"timestamp_seconds": 0.5', '"timestamp_seconds": 1.5'),
                         ("<0.5 seconds>", "<0.0 seconds>"),
                         (f"{MARKER} group", "unrecognized group")]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.verify(self.stream().replace(old, new))

    def test_still_cannot_join_video_or_receive_its_timestamp(self):
        stream = self.stream().replace("preproc_out has 1 entries", "preproc_out has 2 entries")
        with self.assertRaisesRegex(ValueError, "Actual tokenizer"):
            self.verify(stream)
        with self.assertRaisesRegex(ValueError, "Actual tokenizer"):
            self.verify(self.stream(stills=0))

    def test_legacy_labels_or_missing_capability_rejected(self):
        with self.assertRaisesRegex(ValueError, "Legacy"):
            self.verify(self.stream() + "\nD add_text: [0m0.00s]")
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.verify(self.stream().replace(f"{MARKER} mode", "unknown mode"))
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.verify(self.stream(fps=2))

    def test_fractional_fps_even_and_single_frame(self):
        cases = [
            (2, 2.5, {"group_index": 0, "frame_indices": [0, 1], "padded": False,
                      "timestamp_seconds": 0.2, "label": "<0.2 seconds>"}),
            (1, 1, {"group_index": 0, "frame_indices": [0, 0], "padded": True,
                    "timestamp_seconds": 0.0, "label": "<0.0 seconds>"}),
        ]
        for count, fps, group in cases:
            with self.subTest(count=count):
                self.assertTrue(self.verify(self.stream([group], fps=fps),
                                            frame_count=count, fps=fps)["verified"])


if __name__ == "__main__":
    unittest.main()
