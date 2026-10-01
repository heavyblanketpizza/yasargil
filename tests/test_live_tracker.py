"""Turning per-frame observations into events, and events into procedure steps."""
import json
from pathlib import Path
import tempfile
import unittest

from yasargil.live import LiveError
from yasargil.live.events import EventLog
from yasargil.live.perception import Detection, Observation
from yasargil.live.procedure import DEFAULT_PROCEDURE, ProcedureSpec
from yasargil.live.tracker import EventBuilder, ProcedureTracker, TrackerConfig

DUROTOMY = Detection("durotomy", "anatomy", 1.0, None, (0.4, 0.4, 0.6, 0.6))


def obs(index, *detections, annotated=True):
    return Observation(index, (index - 1) * 1000, tuple(detections), annotated, "test")


def tool(label, x=0.9, y=0.9, confidence=1.0):
    return Detection(label, "instrument", confidence, (x, y), None)


class EventBuilderTests(unittest.TestCase):
    def build(self, sequence, **config):
        log = EventLog()
        builder = EventBuilder(log, TrackerConfig(**config))
        for observation in sequence:
            builder.update(observation)
        return log

    def kinds(self, log):
        return [(e.type, e.subject, e.frame_index) for e in log.all()]

    def test_entry_requires_consecutive_frames_and_records_onset(self):
        log = self.build([obs(1, tool("grasper")), obs(2), obs(3, tool("grasper")), obs(4, tool("grasper"))],
                         enter_frames=2, exit_frames=5)
        self.assertEqual(self.kinds(log), [("instrument_entered", "grasper", 4)])
        entered = log.all()[0]
        self.assertEqual(entered.onset_ms, 2000)
        self.assertEqual(entered.evidence_frames, (3, 4))

    def test_low_confidence_detections_do_not_count(self):
        log = self.build([obs(1, tool("grasper", confidence=0.2))], min_confidence=0.5)
        self.assertEqual(log.all(), [])

    def test_exit_requires_consecutive_absent_frames(self):
        log = self.build([obs(1, tool("grasper")), obs(2), obs(3, tool("grasper")), obs(4), obs(5)],
                         enter_frames=1, exit_frames=2)
        self.assertEqual(self.kinds(log), [("instrument_entered", "grasper", 1), ("instrument_left", "grasper", 5)])
        self.assertEqual(log.all()[1].onset_ms, 3000)

    def test_missing_and_unannotated_frames_are_not_absence(self):
        log = self.build([obs(1, tool("grasper")), obs(2, annotated=False), obs(3, annotated=False),
                          obs(7, tool("grasper"))], enter_frames=1, exit_frames=2)
        self.assertEqual(self.kinds(log), [("instrument_entered", "grasper", 1)])

    def test_anatomy_visibility(self):
        log = self.build([obs(1, DUROTOMY), obs(2), obs(3)], exit_frames=2)
        self.assertEqual(self.kinds(log), [("anatomy_visible", "durotomy", 1), ("anatomy_hidden", "durotomy", 3)])

    def test_tip_near_structure_with_hysteresis(self):
        near, edge, far = tool("needle driver", 0.62, 0.5), tool("needle driver", 0.66, 0.5), tool("needle driver", 0.9, 0.5)
        log = self.build([obs(1, DUROTOMY, far), obs(2, DUROTOMY, near), obs(3, DUROTOMY, edge), obs(4, DUROTOMY, far)],
                         near_distance=0.05, aspect=1.0)
        types = [(e.type, e.frame_index) for e in log.all() if "structure" in e.type]
        self.assertEqual(types, [("tip_near_structure", 2), ("tip_cleared_structure", 4)])
        event = next(e for e in log.all() if e.type == "tip_near_structure")
        self.assertEqual((event.subject, event.object), ("needle driver", "durotomy"))
        self.assertAlmostEqual(event.data["distance"], 0.02)

    def test_aspect_scales_horizontal_distance(self):
        log = self.build([obs(1, DUROTOMY, tool("grasper", 0.63, 0.5))], near_distance=0.05, aspect=16 / 9)
        self.assertFalse([e for e in log.all() if e.type == "tip_near_structure"])

    def test_view_empty_and_restored(self):
        log = self.build([obs(1, tool("grasper")), obs(2), obs(3), obs(4), obs(5, tool("grasper"))],
                         exit_frames=1, empty_frames=3)
        kinds = [(e.type, e.frame_index) for e in log.all()]
        self.assertIn(("view_empty", 4), kinds)
        self.assertIn(("view_restored", 5), kinds)
        self.assertEqual(log.all()[2].onset_ms, 1000)


class ProcedureTrackerTests(unittest.TestCase):
    def setUp(self):
        self.log = EventLog()
        self.builder = EventBuilder(self.log, TrackerConfig())
        self.tracker = ProcedureTracker(self.log, ProcedureSpec.from_dict(DEFAULT_PROCEDURE))

    def feed(self, observation):
        self.builder.update(observation)
        return self.tracker.update(observation.t_ms, observation.frame_index, self.log.state_at(observation.t_ms))

    def test_step_changes_after_dwell_and_cites_instrument_events(self):
        for index in (1, 2):
            self.assertEqual(self.feed(obs(index, tool("nerve hook"))), [])
        events = self.feed(obs(3, tool("nerve hook")))
        self.assertEqual([(e.type, e.subject) for e in events], [("step_changed", "exposure")])
        self.assertEqual(events[0].onset_ms, 0)
        self.assertEqual(events[0].cites, ("E000001",))
        self.assertEqual(events[0].data["from"], "idle")

    def test_suturing_before_exposure_is_unexpected(self):
        events = []
        for index in (1, 2, 3):
            events += self.feed(obs(index, tool("needle driver")))
        self.assertEqual([e.type for e in events], ["step_changed", "step_unexpected"])
        self.assertEqual(events[1].cites, (events[0].event_id,))
        self.assertIn("exposure", events[1].data["reason"])

    def test_flicker_shorter_than_dwell_does_not_change_step(self):
        for index in (1, 2, 3):
            self.feed(obs(index, tool("nerve hook")))
        changes = []
        for index, detections in ((4, (tool("needle driver"),)), (5, (tool("nerve hook"),)), (6, (tool("nerve hook"),))):
            changes += self.feed(obs(index, *detections))
        self.assertEqual([e for e in changes if e.type == "step_changed"], [])

    def test_spec_validation_and_file_loading(self):
        with self.assertRaises(LiveError):
            ProcedureSpec.from_dict({"id": "x", "steps": []})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            path.write_text(json.dumps(DEFAULT_PROCEDURE))
            self.assertEqual(ProcedureSpec.load(path).id, DEFAULT_PROCEDURE["id"])


if __name__ == "__main__":
    unittest.main()
