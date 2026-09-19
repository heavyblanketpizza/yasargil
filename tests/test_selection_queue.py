"""Frozen scheduling uses outcomes only to order whole-sequence jobs."""

import csv
import tempfile
import unittest
from pathlib import Path

from yasargil.selection_queue import SelectionQueueError, build_selection_queue, verify_selection_queue


class SelectionQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "frames").mkdir()

    def metadata(self, rows):
        with (self.root / "sospine_outcomes.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Trial ID", "Leak At 40mmHg", "Time for repair", "Length", "Total Frames at 1 FPS"])
            for case, leak, seconds in rows:
                writer.writerow([case, leak, seconds, "99:00:00", 99999])

    def sequence(self, case, count=3):
        directory = self.root / "frames" / case
        directory.mkdir()
        for index in range(1, count + 1):
            (directory / f"{case}_frame_{index:08d}.jpeg").write_bytes(f"fake image {index}".encode())
        return directory

    def test_all_available_alternate_and_unknown_last_without_inventing_recovery_rank(self):
        rows = [("S1A1", "N", 999), ("S1A2", "N", 1), ("S1A10", "N", 3),
                ("S2A1", "Y", 2), ("S3A1", "Y", 4), ("Clip1", "", ""), ("Clip0", "", ""),
                ("S9A1", "Y", 100)]
        self.metadata(rows)
        for case, _, _ in rows[:-1]:
            self.sequence(case)
        before = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))
        queue = build_selection_queue(self.root)
        self.assertEqual([job["case_id"] for job in queue["jobs"]],
                         ["S1A1", "S2A1", "S1A2", "S3A1", "S1A10", "Clip0", "Clip1"])
        self.assertEqual([job["position"] for job in queue["jobs"]], list(range(1, 8)))
        self.assertEqual(queue["outcome_group_counts"], {"better": 3, "worse": 2, "unknown": 2})
        self.assertEqual(queue["excluded"][0]["case_id"], "S9A1")
        self.assertEqual(queue["excluded"][0]["reason"], "released_sequence_not_available")
        self.assertIn("not_patient_recovery", queue["ordering"]["scope"])
        self.assertFalse(queue["ordering"]["outcome_metadata_used_for_model_input"])
        self.assertEqual(queue, build_selection_queue(self.root))
        verify_selection_queue(queue)
        self.assertEqual(before, sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*")))

    def test_explicit_repair_time_tie_break_never_changes_playback_timeline(self):
        rows = [("S1A1", "N", 500), ("S1A2", "N", 100), ("S1A3", "N", ""),
                ("S2A1", "Y", 50), ("S2A2", "Y", 200), ("S2A3", "Y", "")]
        self.metadata(rows)
        for case, _, _ in rows:
            self.sequence(case, count=2)
        queue = build_selection_queue(self.root, tie_break="repair_time")
        self.assertEqual([job["case_id"] for job in queue["jobs"]],
                         ["S1A2", "S2A2", "S1A1", "S2A1", "S1A3", "S2A3"])
        for job in queue["jobs"]:
            self.assertEqual(job["frame_count"], 2)
            self.assertEqual(job["duration_ms"], 2000)
        self.assertEqual(queue["reconstructed_fps"], 1)
        self.assertEqual(queue["timing_basis"], "reconstructed_nominal")
        verify_selection_queue(queue)

    def test_missing_metadata_stays_unknown_and_shorter_group_is_not_duplicated(self):
        self.metadata([("S1A1", "Y", 10), ("S1A2", "Y", 20)])
        for case in ("S1A1", "S1A2", "Clip0"):
            self.sequence(case)
        queue = build_selection_queue(self.root)
        self.assertEqual([job["case_id"] for job in queue["jobs"]], ["S1A1", "S1A2", "Clip0"])
        self.assertIsNone(queue["jobs"][-1]["metadata_locator"])
        self.assertEqual(queue["jobs"][-1]["outcome_rank_group"], "unknown")

    def test_inventory_and_csv_provenance_are_bound_and_mutations_rejected(self):
        self.metadata([("S1A1", "N", 10)])
        directory = self.sequence("S1A1")
        queue = build_selection_queue(self.root)
        job = queue["jobs"][0]
        self.assertEqual(job["metadata_locator"]["line"], 2)
        self.assertEqual(job["metadata_locator"]["sha256"], queue["metadata"]["sha256"])
        self.assertEqual(len(job["inventory_sha256"]), 64)
        frame = directory / "S1A1_frame_00000002.jpeg"
        frame.write_bytes(b"changed size and content")
        with self.assertRaisesRegex(SelectionQueueError, "differs"):
            verify_selection_queue(queue)
        queue = build_selection_queue(self.root)
        self.metadata([("S1A1", "Y", 10)])
        with self.assertRaisesRegex(SelectionQueueError, "differs"):
            verify_selection_queue(queue)
        queue = build_selection_queue(self.root)
        queue["jobs"][0]["position"] = 2
        with self.assertRaisesRegex(SelectionQueueError, "differs"):
            verify_selection_queue(queue)

    def test_noncontiguous_empty_and_unknown_sequences_fail(self):
        self.metadata([("S1A1", "N", 10)])
        directory = self.sequence("S1A1")
        missing = directory / "S1A1_frame_00000002.jpeg"
        missing.unlink()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            build_selection_queue(self.root)
        missing.write_bytes(b"")
        with self.assertRaisesRegex(SelectionQueueError, "Empty"):
            build_selection_queue(self.root)
        missing.write_bytes(b"restored")
        (self.root / "frames" / "unexpected").mkdir()
        with self.assertRaisesRegex(SelectionQueueError, "Unrecognized"):
            build_selection_queue(self.root)

    def test_ambiguous_or_invalid_metadata_and_tie_break_fail(self):
        self.sequence("S1A1")
        self.metadata([("S1A1", "N", 10), ("S1A1", "Y", 20)])
        with self.assertRaisesRegex(SelectionQueueError, "Duplicate"):
            build_selection_queue(self.root)
        self.metadata([("S1A1", "maybe", 10)])
        with self.assertRaisesRegex(SelectionQueueError, "Unknown leak"):
            build_selection_queue(self.root)
        for duration in ("nan", "inf", "-1", "not a number"):
            self.metadata([("S1A1", "N", duration)])
            with self.assertRaisesRegex(SelectionQueueError, "repair time"):
                build_selection_queue(self.root)
        with self.assertRaisesRegex(SelectionQueueError, "tie_break"):
            build_selection_queue(self.root, tie_break="recovery")


if __name__ == "__main__":
    unittest.main()
