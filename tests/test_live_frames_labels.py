"""Frame discovery and SOSpine label parsing for the live harness."""
from pathlib import Path
import tempfile
import unittest

from live_fixtures import box, make_case, tip
from yasargil.live import LiveError
from yasargil.live.frames import FrameSource, case_frames, render_jpeg
from yasargil.live.labels import BOXES, POINTS, CaseLabels, LabelGeometry, normalize_label


class FrameSourceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_frames_are_ordered_with_one_fps_timestamps_and_gaps_preserved(self):
        make_case(self.root, "S1A1", {}, frame_count=5, missing={3})
        source = case_frames(self.root, "S1A1")
        self.assertEqual([f.index for f in source.frames], [1, 2, 4, 5])
        self.assertEqual([f.t_ms for f in source.frames], [0, 1000, 3000, 4000])
        self.assertIsNone(source.get(3))
        self.assertEqual(source.at_or_before(2500).index, 2)
        self.assertIsNone(source.at_or_before(-1))
        self.assertEqual(source.size(source.frames[0]), (64, 36))

    def test_other_cases_and_unrelated_files_are_ignored(self):
        make_case(self.root, "S1A1", {}, frame_count=2)
        directory = self.root / "frames" / "S1A1"
        (directory / "notes.txt").write_text("x")
        (directory / "S2A1_frame_00000009.jpeg").write_bytes(b"")
        self.assertEqual([f.index for f in FrameSource(directory, "S1A1").frames], [1, 2])

    def test_empty_directory_is_an_error(self):
        (self.root / "empty").mkdir()
        with self.assertRaises(LiveError):
            FrameSource(self.root / "empty")

    def test_render_jpeg_resizes_and_crops(self):
        make_case(self.root, "S1A1", {}, frame_count=1, size=(200, 100))
        frame = case_frames(self.root, "S1A1").frames[0]
        from io import BytesIO
        from PIL import Image
        with Image.open(BytesIO(render_jpeg(frame.path, max_side=50))) as image:
            self.assertEqual(image.size, (50, 25))
        with Image.open(BytesIO(render_jpeg(frame.path, max_side=500, crop=(0.5, 0.0, 1.0, 0.5)))) as image:
            self.assertEqual(image.size, (100, 50))


class LabelTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_label_normalization(self):
        cases = {
            "grasper": ("grasper", "body", "instrument"),
            "grasper ": ("grasper", "body", "instrument"),
            "NEEDLE DRIVER TIP": ("needle driver", "tip", "instrument"),
            "needle driver base": ("needle driver", "base", "instrument"),
            "needle  tip": ("needle", "tip", "instrument"),
            "nerve hook": ("nerve hook", "body", "instrument"),
            "durotomy": ("durotomy", "body", "anatomy"),
            "": None,
            "OTHER CASE": None,
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_label(raw), expected)

    def test_points_boxes_and_normalized_clamped_coordinates(self):
        make_case(self.root, "S1A1", {1: [tip("grasper ", 32, 18), box("durotomy", -10, 9, 80, 27)]})
        labels = CaseLabels.load(self.root, "S1A1", (64, 36))
        grasper, durotomy = sorted(labels.items(1), key=lambda item: item.label)[1], sorted(labels.items(1), key=lambda item: item.label)[0]
        self.assertEqual(grasper.point, (0.5, 0.5))
        self.assertIsNone(grasper.box)
        self.assertEqual(grasper.source_table, POINTS)
        self.assertEqual(durotomy.box, (0.0, 0.25, 1.0, 0.75))
        self.assertEqual(durotomy.kind, "anatomy")
        self.assertEqual(durotomy.source_table, BOXES)

    def test_constructed_geometry_can_omit_source_provenance(self):
        geometry = LabelGeometry("grasper", "instrument", "tip", (0.5, 0.5), None)
        self.assertEqual(geometry.source_table, "")

    def test_empty_label_rows_mark_annotated_frames_without_items(self):
        make_case(self.root, "S1A1", {1: [], 2: [tip("grasper", 1, 1)]}, frame_count=3, unlabeled={3})
        labels = CaseLabels.load(self.root, "S1A1", (64, 36))
        self.assertTrue(labels.annotated(1))
        self.assertEqual(labels.items(1), [])
        self.assertTrue(labels.annotated(2))
        self.assertFalse(labels.annotated(3))
        self.assertEqual(labels.indices, [1, 2])

    def test_unknown_duplicate_and_invalid_rows(self):
        make_case(self.root, "S1A1", {1: [tip("grasper", 2, 2), tip("grasper", 2, 2), tip("suction", 3, 3),
                                          ("needle", "x", 1, 2, 2)]})
        labels = CaseLabels.load(self.root, "S1A1", (64, 36))
        self.assertEqual([item.label for item in labels.items(1)], ["grasper"])
        self.assertEqual(labels.unknown_labels, {"suction": 1})
        self.assertEqual(labels.invalid_rows, 1)

    def test_other_cases_are_excluded(self):
        make_case(self.root, "S1A1", {1: [tip("grasper", 2, 2)]})
        make_case(self.root, "S2A1", {1: [tip("nerve hook", 2, 2)]}, append=True)
        self.assertEqual([i.label for i in CaseLabels.load(self.root, "S1A1", (64, 36)).items(1)], ["grasper"])
        self.assertEqual([i.label for i in CaseLabels.load(self.root, "S2A1", (64, 36)).items(1)], ["nerve hook"])

    def test_missing_tables_are_an_error(self):
        with self.assertRaises(LiveError):
            CaseLabels.load(self.root, "S1A1", (64, 36))


if __name__ == "__main__":
    unittest.main()
