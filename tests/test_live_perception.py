"""Label-based and degraded perception sources."""
from pathlib import Path
import tempfile
import unittest

from live_fixtures import box, make_case, tip
from yasargil.live.frames import case_frames
from yasargil.live.labels import CaseLabels
from yasargil.live.perception import DegradedPerception, Detection, LabelPerception, Observation


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        make_case(root, "S1A1", {
            1: [tip("needle driver base", 64, 0), tip("needle driver tip", 32, 18), box("needle driver", 30, 0, 64, 20),
                box("durotomy", 16, 9, 48, 27), tip("grasper", 8, 9)],
            2: [tip("nerve hook", 16, 27)],
            3: [],
        }, frame_count=4, unlabeled={4})
        self.frames = case_frames(root, "S1A1")
        self.labels = CaseLabels.load(root, "S1A1", (64, 36))
        self.perception = LabelPerception(self.labels)

    def frame(self, index):
        return self.frames.get(index)

    def test_rows_merge_into_one_detection_per_label_with_tip_preferred(self):
        observation = self.perception.observe(self.frame(1))
        self.assertIsInstance(observation, Observation)
        driver = observation.find("needle driver")
        self.assertEqual(driver.tip, (0.5, 0.5))
        self.assertEqual(driver.box, (30 / 64, 0.0, 1.0, 20 / 36))
        self.assertEqual(driver.confidence, 1.0)
        self.assertEqual(observation.find("durotomy").box, (0.25, 0.25, 0.75, 0.75))
        self.assertEqual(observation.find("grasper").tip, (0.125, 0.25))
        self.assertEqual({d.label for d in observation.instruments}, {"needle driver", "grasper"})
        self.assertEqual([d.label for d in observation.anatomy], ["durotomy"])
        self.assertEqual(observation.t_ms, 0)

    def test_single_point_instrument_has_tip_and_no_box(self):
        make = self.perception.observe(self.frame(2)).find("nerve hook")
        self.assertEqual(make.tip, (0.25, 0.75))
        self.assertIsNone(make.box)

    def test_annotation_status_is_reported(self):
        self.assertTrue(self.perception.observe(self.frame(3)).annotated)
        self.assertEqual(self.perception.observe(self.frame(3)).detections, ())
        self.assertFalse(self.perception.observe(self.frame(4)).annotated)

    def test_crop_keeps_inside_detections_and_remaps_coordinates(self):
        observation = self.perception.observe_crop(self.frame(1), (0.0, 0.0, 0.5, 0.5))
        self.assertIsNone(observation.find("needle driver"))
        self.assertEqual(observation.find("grasper").tip, (0.25, 0.5))
        self.assertEqual(observation.find("durotomy").box, (0.5, 0.5, 1.0, 1.0))

    def test_detection_json_round_trip(self):
        detection = Detection("grasper", "instrument", 0.5, (0.1, 0.2), None)
        self.assertEqual(Detection.from_json(detection.to_json()), detection)

    def test_degraded_is_deterministic_per_frame_regardless_of_call_order(self):
        degraded = DegradedPerception(self.perception, miss_rate=0.5, false_positive_rate=0.5,
                                      jitter=0.05, confidence_noise=0.2, seed=7)
        first = [degraded.observe(self.frame(i)) for i in (1, 2, 3)]
        again = [degraded.observe(self.frame(i)) for i in (3, 1, 2)]
        self.assertEqual(first, [again[1], again[2], again[0]])

    def test_degraded_miss_rate_one_drops_everything(self):
        degraded = DegradedPerception(self.perception, miss_rate=1.0, seed=1)
        self.assertEqual(degraded.observe(self.frame(1)).detections, ())

    def test_degraded_false_positives_and_jitter_stay_in_bounds(self):
        degraded = DegradedPerception(self.perception, false_positive_rate=1.0, jitter=0.5,
                                      confidence_noise=0.5, seed=3)
        for index in (1, 2, 3):
            observation = degraded.observe(self.frame(index))
            false_positives = [d for d in observation.detections if d.confidence < 0.75 and d.label not in
                               {item.label for item in self.labels.items(index)}]
            self.assertTrue(false_positives)
            for detection in observation.detections:
                self.assertTrue(0.0 < detection.confidence <= 1.0)
                for value in (detection.tip or ()) + (detection.box or ()):
                    self.assertTrue(0.0 <= value <= 1.0)
        self.assertEqual(degraded.identity()["miss_rate"], 0.0)
        self.assertEqual(degraded.identity()["inner"]["kind"], "labels")

    def test_degraded_rejects_out_of_range_rates(self):
        with self.assertRaises(ValueError):
            DegradedPerception(self.perception, miss_rate=1.5)


if __name__ == "__main__":
    unittest.main()
