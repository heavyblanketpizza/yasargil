"""Selection invariants with explicitly synthetic numerical features."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

from yasargil.frame_selection import select_candidates, select_from_embeddings


@unittest.skipIf(np is None, "optional selection dependencies are not installed")
class FrameSelectionTests(unittest.TestCase):
    def frames(self, count):
        return [{"frame_id": f"f{i:04d}", "frame_index": i, "timestamp_ms": i * 1000} for i in range(count)]

    def features(self, count):
        global_vectors = np.tile([1.0, 0.0], (count, 1))
        local_vectors = np.tile([1.0, 0.0], (count, 9, 1))
        return global_vectors, local_vectors, np.ones(count)

    def test_budget_boundaries_deterministic_order_and_protected_coverage(self):
        frames = self.frames(120)
        rng = np.random.default_rng(12)
        features = (rng.normal(size=(120, 8)), rng.normal(size=(120, 9, 8)), rng.random(120))
        result = select_from_embeddings(frames, 24, *features)
        self.assertEqual(result, select_from_embeddings(frames, 24, *features))
        self.assertEqual(len(result["selected_ids"]), 24)
        self.assertEqual(result["selected_ids"], sorted(result["selected_ids"]))
        self.assertIn("f0000", result["protected_ids"])
        self.assertIn("f0119", result["protected_ids"])
        self.assertEqual(len(result["protected_ids"]), 8)
        self.assertTrue(set(result["protected_ids"]) <= set(result["selected_ids"]))
        for segment in range(4):
            self.assertEqual(sum(score["selected"] and score["timeline_bin"] == segment
                                 for score in result["scores"].values()), 6)

    def test_local_change_is_selected_even_if_global_features_match(self):
        frames = self.frames(12)
        global_vectors, local_vectors, quality = self.features(12)
        local_vectors[5, 4] = [-1.0, 0.0]
        result = select_from_embeddings(frames, 3, global_vectors, local_vectors, quality)
        self.assertEqual(result["selected_ids"], ["f0000", "f0005", "f0011"])
        self.assertIn("local_feature_change", result["scores"]["f0005"]["reasons"])

    def test_clarity_prefers_clearer_equivalent_without_deleting_unclear_sources(self):
        frames = self.frames(8)
        global_vectors, local_vectors, quality = self.features(8)
        quality[3] = 100
        result = select_from_embeddings(frames, 3, global_vectors, local_vectors, quality)
        self.assertEqual(result["selected_ids"], ["f0000", "f0003", "f0007"])
        self.assertEqual(len(result["scores"]), 8)
        self.assertEqual(result["scores"]["f0002"]["reasons"], ["available_for_contextual_retrieval"])

    def test_unsorted_input_retains_feature_alignment(self):
        frames = self.frames(8)
        global_vectors, local_vectors, quality = self.features(8)
        global_vectors[4] = [-1, 0]
        forward = select_from_embeddings(frames, 3, global_vectors, local_vectors, quality)
        reverse = select_from_embeddings(frames[::-1], 3, global_vectors[::-1], local_vectors[::-1], quality[::-1])
        self.assertEqual(forward, reverse)

    def test_small_sequence_never_duplicates_frames_or_exceeds_available(self):
        for count in (1, 2, 3):
            result = select_from_embeddings(self.frames(count), 20, *self.features(count))
            self.assertEqual(len(result["selected_ids"]), count)
            self.assertEqual(len(set(result["selected_ids"])), count)

    def test_invalid_budget_duplicate_ids_and_nonfinite_features_fail(self):
        for budget in (0, -1, 1, True, 2.5):
            with self.assertRaises(ValueError):
                select_from_embeddings(self.frames(3), budget, *self.features(3))
        frames = self.frames(3)
        frames[1]["frame_id"] = frames[0]["frame_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            select_from_embeddings(frames, 3, *self.features(3))
        features = self.features(3)
        features[0][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            select_from_embeddings(self.frames(3), 3, *features)

    def test_unknown_backend_and_missing_dinov3_fail_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as directory:
            for backend in ("fake", "dinov3"):
                with self.assertRaises((ValueError, RuntimeError)):
                    select_candidates(self.frames(3), 3, embedding_cache=Path(directory),
                                      model_path=Path(directory) / "missing", backend=backend)

    def test_cache_reuse_is_bound_to_image_bytes_and_encoder_fingerprint(self):
        from yasargil.frame_selection import _load_embeddings
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            image.write_bytes(b"synthetic-source-bytes-encoder-is-mocked")
            frames = [{**self.frames(1)[0], "image_path": str(image),
                       "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}]
            frames.append({**frames[0], "frame_id": "f0001", "frame_index": 1, "timestamp_ms": 1000})
            identity = {"fingerprint_sha256": "a" * 64}
            def encode(items, model, actual_identity):
                for frame in items:
                    yield frame, np.array([1.0, 0.0]), np.tile([1.0, 0.0], (9, 1)), 3.0, "cpu"
            with patch("yasargil.frame_selection._encoder_identity", return_value=identity.copy()), \
                    patch("yasargil.frame_selection._encode_missing", side_effect=encode) as encoder:
                first = _load_embeddings(frames, root / "cache", root / "model", "dinov2")
                self.assertFalse(first[4]["f0000"]["cache_hit"])
                self.assertEqual(len(encoder.call_args.args[0]), 1)
                self.assertEqual(first[4]["f0000"]["path"], first[4]["f0001"]["path"])
                # Identity returned by the real function is fresh on each call.
                with patch("yasargil.frame_selection._encoder_identity", return_value=identity.copy()):
                    second = _load_embeddings(frames, root / "cache", root / "model", "dinov2")
                self.assertTrue(second[4]["f0000"]["cache_hit"])
                self.assertEqual(encoder.call_count, 1)
                with patch("yasargil.frame_selection._encoder_identity", return_value={"fingerprint_sha256": "b" * 64}):
                    changed_encoder = _load_embeddings(frames, root / "cache", root / "model", "dinov2")
                self.assertFalse(changed_encoder[4]["f0000"]["cache_hit"])
                self.assertNotEqual(first[4]["f0000"]["path"], changed_encoder[4]["f0000"]["path"])
                image.write_bytes(b"changed-source")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    _load_embeddings(frames, root / "cache", root / "model", "dinov2")


if __name__ == "__main__":
    unittest.main()
