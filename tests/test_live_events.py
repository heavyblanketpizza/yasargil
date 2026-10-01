"""Append-only event log with causal queries."""
from pathlib import Path
import tempfile
import unittest

from yasargil.live import LiveError
from yasargil.live.events import Event, EventLog


def populated(path=None):
    log = EventLog(path)
    log.append(0, 1, "anatomy_visible", "durotomy", evidence_frames=(1,))
    log.append(1000, 2, "instrument_entered", "nerve hook", onset_ms=1000, confidence=0.9, evidence_frames=(2,))
    log.append(4000, 5, "instrument_left", "nerve hook", onset_ms=4000, evidence_frames=(5,))
    log.append(5000, 6, "instrument_entered", "needle driver", onset_ms=4000, evidence_frames=(5, 6))
    log.append(6000, 7, "tip_near_structure", "needle driver", object="durotomy", data={"distance": 0.01})
    log.append(6000, 7, "alert", "tip_near_durotomy", cites=("E000005",))
    log.append(8000, 9, "step_changed", "suturing", data={"from": "exposure"})
    log.append(9000, 10, "tip_cleared_structure", "needle driver", object="durotomy")
    return log


class EventLogTests(unittest.TestCase):
    def test_ids_are_sequential_and_onset_defaults_to_time(self):
        log = populated()
        self.assertEqual([e.event_id for e in log.all()][:3], ["E000001", "E000002", "E000003"])
        self.assertEqual(log.get("E000001").onset_ms, 0)
        self.assertEqual(log.get("E000004").onset_ms, 4000)
        self.assertIsNone(log.get("E999999"))

    def test_time_must_not_go_backwards(self):
        log = populated()
        with self.assertRaises(LiveError):
            log.append(500, 1, "view_empty", "view")

    def test_unknown_type_and_dangling_citation_are_rejected(self):
        log = EventLog()
        with self.assertRaises(LiveError):
            log.append(0, 1, "teleported", "grasper")
        with self.assertRaises(LiveError):
            log.append(0, 1, "alert", "rule", cites=("E000042",))

    def test_query_is_causal_and_filters(self):
        log = populated()
        self.assertEqual([e.event_id for e in log.query(now_ms=4500)], ["E000001", "E000002", "E000003"])
        self.assertEqual([e.subject for e in log.query(now_ms=99999, types=["instrument_entered"])],
                         ["nerve hook", "needle driver"])
        self.assertEqual([e.event_id for e in log.query(now_ms=99999, subject="durotomy")],
                         ["E000001", "E000005", "E000008"])
        self.assertEqual([e.event_id for e in log.query(now_ms=99999, since_ms=5000, until_ms=6000)],
                         ["E000004", "E000005", "E000006"])
        self.assertEqual([e.event_id for e in log.query(now_ms=99999, limit=2)], ["E000007", "E000008"])

    def test_state_reconstruction(self):
        log = populated()
        state = log.state_at(6500)
        self.assertEqual(set(state.visible_instruments), {"needle driver"})
        self.assertEqual(state.visible_anatomy, {"durotomy": "E000001"})
        self.assertEqual(state.near, {"needle driver": ("durotomy", "E000005")})
        self.assertIsNone(state.step)
        later = log.state_at(9500)
        self.assertEqual(later.near, {})
        self.assertEqual(later.step, ("suturing", "E000007"))
        self.assertEqual(log.state_at(2000).visible_instruments, {"nerve hook": "E000002"})

    def test_lines_are_compact_and_informative(self):
        line = populated().get("E000005").line()
        self.assertIn("E000005", line)
        self.assertIn("t=6.0s", line)
        self.assertIn("needle driver -> durotomy", line)
        self.assertIn("distance=0.01", line)

    def test_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            original = populated(path)
            loaded = EventLog.load(path)
            self.assertEqual(loaded.all(), original.all())
            self.assertEqual(Event.from_json(original.get("E000006").to_json()), original.get("E000006"))
            loaded.append(10000, 11, "view_empty", "view")
            self.assertEqual(loaded.all()[-1].event_id, "E000009")


if __name__ == "__main__":
    unittest.main()
