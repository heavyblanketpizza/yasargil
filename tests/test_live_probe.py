"""DINO probe perception: grid math, targets, head training, decoding and persistence.

A fake encoder stands in for DINOv2 so the tests stay fast and offline.
"""
from pathlib import Path
import tempfile
import unittest

from live_fixtures import box, make_case, scripted_timeline, tip
from yasargil.live.perception import Detection, Observation

try:
    import numpy as np
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

if HAVE_TORCH:
    from yasargil.live.probe import (CLASSES, ProbeModel, ProbePerception, decode, probe_grid, surgeon_split,
                                     targets_for, train_heads, train_probe)


class FakeEncoder:
    """Patch features put a class-specific signal wherever a bright marker would be."""

    dim = 8

    def __init__(self, cols=4, rows=3, signal=None):
        self.cols, self.rows = cols, rows
        self.signal = signal or {}

    def identity(self):
        return {"kind": "fake", "dim": self.dim}

    def grid(self, width, height):
        return self.cols, self.rows

    def encode(self, images):
        batch = len(images)
        cls = np.zeros((batch, self.dim), dtype=np.float32)
        patches = np.zeros((batch, self.rows, self.cols, self.dim), dtype=np.float32)
        for b, image in enumerate(images):
            shade = image.getpixel((0, 0))[0] / 255
            cls[b, 0] = shade
            patches[b, :, :, 0] = shade
        return cls, patches


@unittest.skipUnless(HAVE_TORCH, "probe tests need the optional selection dependencies")
class ProbeTests(unittest.TestCase):
    def test_grid_preserves_aspect(self):
        self.assertEqual(probe_grid(1920, 1080), (28, 16))
        self.assertEqual(probe_grid(1080, 1920), (16, 28))
        self.assertEqual(probe_grid(640, 640), (16, 16))

    def test_targets_mark_tip_patches_and_structure_boxes(self):
        observation = Observation(1, 0, (Detection("grasper", "instrument", 1.0, (0.6, 0.5), None),
                                         Detection("durotomy", "anatomy", 1.0, None, (0.0, 0.0, 0.5, 0.34))))
        presence, maps = targets_for(observation, cols=4, rows=3)
        g, d = CLASSES.index("grasper"), CLASSES.index("durotomy")
        self.assertEqual(presence[g], 1.0)
        self.assertEqual(presence[CLASSES.index("needle")], 0.0)
        self.assertEqual(maps[g].sum(), 1.0)
        self.assertEqual(maps[g][1, 2], 1.0)
        self.assertEqual(maps[d].tolist(), [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def test_decode_turns_probabilities_into_detections(self):
        presence = np.zeros(len(CLASSES), dtype=np.float32)
        maps = np.zeros((len(CLASSES), 3, 4), dtype=np.float32)
        g, d = CLASSES.index("grasper"), CLASSES.index("durotomy")
        presence[g], presence[d] = 0.9, 0.8
        maps[g][1, 2] = 0.95
        maps[d][0, 0] = maps[d][0, 1] = 0.9
        detections = {det.label: det for det in decode(presence, maps, threshold=0.5)}
        self.assertEqual(set(detections), {"grasper", "durotomy"})
        self.assertAlmostEqual(detections["grasper"].confidence, 0.9, places=5)
        self.assertEqual(detections["grasper"].tip, (0.625, 0.5))
        self.assertEqual(detections["durotomy"].box, (0.0, 0.0, 0.5, 1 / 3))

    def test_heads_learn_a_separable_signal(self):
        rng = np.random.default_rng(0)
        count, rows, cols, dim = 160, 3, 4, 8
        cls = rng.normal(0, 0.1, (count, dim)).astype(np.float32)
        patches = rng.normal(0, 0.1, (count, rows, cols, dim)).astype(np.float32)
        presence = np.zeros((count, len(CLASSES)), dtype=np.float32)
        maps = np.zeros((count, len(CLASSES), rows, cols), dtype=np.float32)
        g = CLASSES.index("grasper")
        for i in range(count):
            if i % 2:
                r, c = rng.integers(rows), rng.integers(cols)
                patches[i, r, c, 3] += 3.0
                cls[i, 3] += 1.0
                presence[i, g], maps[i, g, r, c] = 1.0, 1.0
        heads, history = train_heads(cls, patches, presence, maps, epochs=150, lr=0.05, seed=0)
        probs_presence, probs_maps = heads.predict(cls, patches)
        accuracy = ((probs_presence[:, g] > 0.5) == (presence[:, g] > 0.5)).mean()
        self.assertGreater(accuracy, 0.95)
        hits = [np.unravel_index(probs_maps[i, g].argmax(), (rows, cols)) == tuple(np.argwhere(maps[i, g])[0])
                for i in range(count) if presence[i, g]]
        self.assertGreater(np.mean(hits), 0.9)
        self.assertLess(history[-1], history[0])

    def test_encoder_check_ignores_where_the_checkpoint_lives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in ("S1A1", "S8A1"):
                make_case(root, case, scripted_timeline(), append=case != "S1A1")

            class MovedEncoder(FakeEncoder):
                def __init__(self, path, weights="abc"):
                    super().__init__()
                    self.path, self.weights = path, weights

                def identity(self):
                    return {"kind": "fake", "dim": self.dim, "model_path": self.path, "weights": {"w": self.weights}}

            train_probe(root, root / "probe", train=["S1"], val=[], test=["S8"], encoder=MovedEncoder("/old/place"),
                        epochs=2, progress=lambda message: None)
            ProbePerception(root / "probe", encoder=MovedEncoder("/new/place"))
            with self.assertRaises(Exception):
                ProbePerception(root / "probe", encoder=MovedEncoder("/new/place", weights="different"))

    def test_surgeon_split_is_disjoint(self):
        cases = ["S1A1", "S1A2", "S2A1", "S7A3", "S8A1", "Clip0"]
        split = surgeon_split(cases, train=["S1", "S2"], val=["S7"], test=["S8"])
        self.assertEqual(split, {"train": ["S1A1", "S1A2", "S2A1"], "val": ["S7A3"], "test": ["S8A1"]})
        with self.assertRaises(ValueError):
            surgeon_split(cases, train=["S1"], val=["S1"], test=["S8"])

    def test_train_save_load_and_perceive_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in ("S1A1", "S2A1", "S8A1"):
                make_case(root, case, scripted_timeline(), append=case != "S1A1")
            encoder = FakeEncoder()
            metrics = train_probe(root, root / "probe", train=["S1"], val=["S2"], test=["S8"], encoder=encoder,
                                  epochs=5, progress=lambda message: None)
            self.assertEqual(metrics["split"]["train"], ["S1A1"])
            self.assertIn("presence_f1", metrics["test"])
            model = ProbeModel.load(root / "probe")
            self.assertEqual(model.classes, list(CLASSES))
            perception = ProbePerception(root / "probe", encoder=encoder)
            from yasargil.live.frames import case_frames
            frame = case_frames(root, "S8A1").get(7)
            observation = perception.observe(frame)
            self.assertEqual(observation.frame_index, 7)
            self.assertTrue(all(0 <= d.confidence <= 1 for d in observation.detections))
            self.assertEqual(perception.identity()["kind"], "dino-probe")
            cropped = perception.observe_crop(frame, (0, 0, 0.5, 0.5))
            self.assertEqual(cropped.frame_index, 7)
            with self.assertRaises(Exception):
                train_probe(root, root / "probe", train=["S1"], val=["S2"], test=["S8"], encoder=encoder, epochs=1,
                            progress=lambda message: None)


if __name__ == "__main__":
    unittest.main()
