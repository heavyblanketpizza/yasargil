"""Auditable DINO candidates for the separate full-video review workflow.

These visual scores are proposals, not surgical-importance classifications.
All source frames survive selection. Timeline anchors cannot be dropped by the
reviewer, and every feature cache entry is tied to image and encoder bytes.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import tempfile


DEFAULT_MODELS = {
    "dinov2": "dinov2-small",
    "dinov3": "dinov3-vits16-pretrain-lvd1689m",
}
ALGORITHM_VERSION = "dino-temporal-diversity-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Frame selection needs the optional 'selection' dependencies.") from exc
    return np


def _validate_frames(frames: list[dict], budget: int) -> list[dict]:
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")
    if not frames:
        raise ValueError("No source frames were supplied")
    if len(frames) > 1 and budget < 2:
        raise ValueError("budget must preserve both first and last frames (at least 2)")
    ids = set()
    for frame in frames:
        frame_id = frame.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id or frame_id in ids:
            raise ValueError("Source frame IDs must be unique nonempty strings")
        ids.add(frame_id)
        if (isinstance(frame.get("timestamp_ms"), bool)
                or not isinstance(frame.get("timestamp_ms"), (int, float))
                or not math.isfinite(frame["timestamp_ms"]) or frame["timestamp_ms"] < 0):
            raise ValueError(f"Invalid timestamp for {frame_id}")
        if (isinstance(frame.get("frame_index"), bool) or not isinstance(frame.get("frame_index"), int)
                or frame["frame_index"] < 0):
            raise ValueError(f"Missing integer frame index for {frame_id}")
    return sorted(frames, key=lambda frame: (frame["timestamp_ms"], frame["frame_index"], frame["frame_id"]))


def _unit(vectors):
    np = _numpy()
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


def _encoder_identity(model_path: Path, backend: str) -> dict:
    if backend not in DEFAULT_MODELS:
        raise ValueError("embedding backend must be explicitly dinov2 or dinov3")
    config_path = model_path / "config.json"
    processor_path = model_path / "preprocessor_config.json"
    weights = sorted(model_path.glob("*.safetensors"))
    if not config_path.is_file() or not processor_path.is_file() or not weights:
        extra = " DINOv3 requires an authorized local checkpoint; there is no automatic fallback." if backend == "dinov3" else ""
        raise RuntimeError(f"Incomplete local {backend} checkpoint at {model_path}.{extra}")
    config = json.loads(config_path.read_text())
    expected_type = "dinov2" if backend == "dinov2" else "dinov3_vit"
    if config.get("model_type") != expected_type:
        raise ValueError(f"Checkpoint model_type is not {expected_type}; refusing a mislabeled encoder")
    source_path = model_path / "source.json"
    source = json.loads(source_path.read_text()) if source_path.is_file() else {}
    files = weights + [config_path, processor_path]
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        files.append(index_path)
    versions = {}
    for package in ("numpy", "torch", "transformers", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("Install the optional 'selection' dependencies before encoding frames") from exc
    preprocessing = {
        "version": 1, "side_pixels": 224 if backend == "dinov2" else 256,
        "resize": "preserve_aspect_ratio_then_letterbox", "interpolation": "Pillow_LANCZOS",
        "crop": False, "padding_rgb": "rounded_processor_image_mean_times_255",
        "normalization": "checkpoint_processor", "local_pooling": "3x3_mean_patch_grid",
        "global_pooling": "CLS", "l2_normalize": True, "dtype": "float32", "batch_size": 4,
        "clarity": "variance_of_4_neighbor_laplacian_grayscale_max_edge_384",
    }
    identity = {
        "backend": backend, "repository": source.get("repository"),
        "revision": source.get("revision"), "model_type": expected_type,
        "files": {path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size} for path in files},
        "preprocessing": preprocessing, "library_versions": versions,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**identity, "fingerprint_sha256": fingerprint, "model_path": str(model_path.resolve())}


def _encode_missing(frames: list[dict], model_path: Path, identity: dict):
    """Load only local safetensors, never execute downloaded model code."""
    np = _numpy()
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Install the optional 'selection' dependencies before encoding frames") from exc
    processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True, use_fast=False,
                                                   trust_remote_code=False)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True, use_safetensors=True,
                                     trust_remote_code=False).eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    # Explicit batching and float32 keep cache contents reproducible on a given backend.
    side = identity["preprocessing"]["side_pixels"]
    patch = model.config.patch_size
    if isinstance(patch, (list, tuple)):
        patch = patch[0]
    grid_side = side // patch
    registers = getattr(model.config, "num_register_tokens", 0)
    background = tuple(round(float(value) * 255) for value in processor.image_mean)
    with torch.inference_mode():
        for offset in range(0, len(frames), 4):
            batch = frames[offset:offset + 4]
            images, clarities = [], []
            for frame in batch:
                image_bytes = Path(frame["image_path"]).read_bytes()
                if hashlib.sha256(image_bytes).hexdigest() != frame["image_sha256"]:
                    raise ValueError(f"Source image changed during encoding: {frame['frame_id']}")
                with Image.open(io.BytesIO(image_bytes)) as opened:
                    source = opened.convert("RGB")
                    resized = ImageOps.contain(source, (side, side), Image.Resampling.LANCZOS)
                    canvas = Image.new("RGB", (side, side), background)
                    canvas.paste(resized, ((side - resized.width) // 2, (side - resized.height) // 2))
                    images.append(canvas)
                    gray = np.asarray(ImageOps.contain(source.convert("L"), (384, 384), Image.Resampling.LANCZOS), dtype=np.float32)
                    laplacian = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
                                 + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
                    clarities.append(float(np.var(laplacian)) if laplacian.size else 0.0)
            inputs = processor(images=images, return_tensors="pt", do_resize=False, do_center_crop=False)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            tokens = model(**inputs).last_hidden_state.detach().cpu().numpy()
            patches = tokens[:, 1 + registers:]
            if patches.shape[1] != grid_side * grid_side:
                raise RuntimeError("Unexpected DINO patch layout; refusing an incorrectly pooled feature cache")
            patches = patches.reshape(len(batch), grid_side, grid_side, -1)
            regional = np.stack([part.mean(axis=(1, 2)) for rows in np.array_split(patches, 3, axis=1)
                                 for part in np.array_split(rows, 3, axis=2)], axis=1)
            for frame, global_vector, local_vectors, clarity in zip(batch, _unit(tokens[:, 0]), _unit(regional), clarities):
                yield frame, global_vector, local_vectors, clarity, device
    del model
    if device == "mps":
        torch.mps.empty_cache()


def _load_embeddings(frames: list[dict], embedding_cache: Path, model_path: Path, backend: str):
    np = _numpy()
    identity = _encoder_identity(model_path, backend)
    # Device affects floating-point results, so it also belongs to the cache key.
    import torch
    identity["device"] = "mps" if torch.backends.mps.is_available() else "cpu"
    identity["fingerprint_sha256"] = hashlib.sha256(
        (identity["fingerprint_sha256"] + ":" + identity["device"]).encode()).hexdigest()
    fingerprint = identity["fingerprint_sha256"]
    directory = embedding_cache.resolve() / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    features, missing, cache_records, missing_ids = {}, [], {}, {}
    for frame in frames:
        image_path = Path(frame.get("image_path", ""))
        image_hash = frame.get("image_sha256", "")
        if not image_path.is_file() or _sha256(image_path) != image_hash:
            raise ValueError(f"Source image hash mismatch or missing image: {frame['frame_id']}")
        cache_path = directory / f"{image_hash}.npz"
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    if str(cached["image_sha256"]) != image_hash or str(cached["encoder_fingerprint"]) != fingerprint:
                        raise ValueError("Cache provenance mismatch")
                    features[frame["frame_id"]] = (cached["global_embedding"], cached["local_embeddings"], float(cached["clarity"]))
            except Exception as exc:
                raise ValueError(f"Invalid feature cache {cache_path}") from exc
            cache_records[frame["frame_id"]] = {"path": str(cache_path), "sha256": _sha256(cache_path), "cache_hit": True,
                                                "image_path": str(image_path.resolve()), "image_sha256": image_hash}
        else:
            if image_hash not in missing_ids:
                missing.append(frame)
                missing_ids[image_hash] = frame["frame_id"]
    for frame, global_vector, local_vectors, clarity, _device in _encode_missing(missing, model_path, identity) if missing else []:
        cache_path = directory / f"{frame['image_sha256']}.npz"
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".npz", delete=False) as temporary:
            temp_path = Path(temporary.name)
        try:
            np.savez_compressed(temp_path, global_embedding=global_vector, local_embeddings=local_vectors,
                                clarity=clarity, image_sha256=frame["image_sha256"], encoder_fingerprint=fingerprint)
            temp_path.replace(cache_path)
        finally:
            temp_path.unlink(missing_ok=True)
        features[frame["frame_id"]] = (global_vector, local_vectors, clarity)
        cache_records[frame["frame_id"]] = {"path": str(cache_path), "sha256": _sha256(cache_path), "cache_hit": False,
                                            "image_path": str(Path(frame["image_path"]).resolve()), "image_sha256": frame["image_sha256"]}
    # Repeated source pixels share one encoding while keeping separate timeline IDs.
    for frame in frames:
        if frame["frame_id"] not in features:
            representative = missing_ids[frame["image_sha256"]]
            features[frame["frame_id"]] = features[representative]
            cache_records[frame["frame_id"]] = {
                **cache_records[representative], "image_path": str(Path(frame["image_path"]).resolve())}
    ordered = [features[frame["frame_id"]] for frame in frames]
    return (np.stack([entry[0] for entry in ordered]), np.stack([entry[1] for entry in ordered]),
            np.asarray([entry[2] for entry in ordered]), identity, cache_records)


def select_from_embeddings(frames: list[dict], budget: int, global_embeddings, local_embeddings, clarities) -> dict:
    """Pure deterministic ranking; feature rows must match the input frame order.

    Global novelty, the most changed of nine local regions, adjacent changes,
    and a small clarity preference compete within balanced timeline bins.
    """
    np = _numpy()
    ordered = _validate_frames(frames, budget)
    row_for_id = {frame["frame_id"]: row for row, frame in enumerate(frames)}
    order = [row_for_id[frame["frame_id"]] for frame in ordered]
    global_vectors = np.asarray(global_embeddings, dtype=np.float32)
    local_vectors = np.asarray(local_embeddings, dtype=np.float32)
    quality = np.asarray(clarities, dtype=np.float64)
    n = len(ordered)
    if (global_vectors.ndim != 2 or global_vectors.shape[0] != n or global_vectors.shape[1] < 1
            or local_vectors.ndim != 3 or local_vectors.shape[0] != n or local_vectors.shape[1] < 1
            or local_vectors.shape[2] != global_vectors.shape[1] or quality.shape != (n,)
            or not np.all(np.isfinite(global_vectors)) or not np.all(np.isfinite(local_vectors))
            or not np.all(np.isfinite(quality)) or np.any(quality < 0)):
        raise ValueError("Invalid or nonfinite embedding/clarity arrays")
    global_vectors, local_vectors, quality = _unit(global_vectors[order]), _unit(local_vectors[order]), quality[order]
    budget = min(budget, n)
    times = np.asarray([frame["timestamp_ms"] for frame in ordered], dtype=np.float64)
    anchor_count = min(n, budget, max(2, math.ceil(budget / 3))) if n > 1 else 1
    anchors = {0, n - 1}
    for target in np.linspace(times[0], times[-1], anchor_count)[1:-1]:
        anchors.add(min((i for i in range(n) if i not in anchors), key=lambda i: (abs(times[i] - target), i)))
    bins_count = min(4, max(1, budget // 3))
    if times[-1] > times[0]:
        bins = np.minimum(bins_count - 1, ((times - times[0]) / (times[-1] - times[0]) * bins_count).astype(int))
    else:
        bins = np.minimum(bins_count - 1, np.arange(n) * bins_count // n)
    quality_rank = np.asarray([(np.count_nonzero(quality < value) + 0.5 * (np.count_nonzero(quality == value) - 1)) / max(1, n - 1)
                               for value in quality])
    global_change = np.zeros(n)
    local_change = np.zeros(n)
    if n > 1:
        global_change[1:] = np.clip(1 - np.sum(global_vectors[1:] * global_vectors[:-1], axis=-1), 0, 2)
        local_change[1:] = np.max(np.clip(1 - np.sum(local_vectors[1:] * local_vectors[:-1], axis=-1), 0, 2), axis=-1)
    change = np.maximum(global_change, local_change)
    novelty_global, novelty_local = np.full(n, 2.0), np.full(n, 2.0)
    selected, reasons, chosen_scores = set(), {}, {}

    def add(index: int, reason: str, score: float):
        selected.add(index)
        reasons[index] = [reason]
        chosen_scores[index] = float(score)
        novelty_global[:] = np.minimum(novelty_global, np.clip(1 - global_vectors @ global_vectors[index], 0, 2))
        distances = np.max(np.clip(1 - np.sum(local_vectors * local_vectors[index], axis=-1), 0, 2), axis=-1)
        novelty_local[:] = np.minimum(novelty_local, distances)

    for index in sorted(anchors):
        add(index, "protected_timeline_anchor", 0.0)
    reasons[0].append("first_source_frame")
    reasons[n - 1].append("last_source_frame")
    while len(selected) < budget:
        scores = 0.50 * novelty_global + 0.30 * novelty_local + 0.15 * change + 0.05 * quality_rank
        remaining = [i for i in range(n) if i not in selected]
        counts = [sum(bins[i] == b for i in selected) for b in range(bins_count)]
        least = min(counts[bins[i]] for i in remaining)
        eligible = [i for i in remaining if counts[bins[i]] == least]
        chosen = max(eligible, key=lambda i: (float(scores[i]), -i))
        local_wins = 0.30 * novelty_local[chosen] + 0.15 * local_change[chosen] > 0.50 * novelty_global[chosen]
        add(chosen, "local_feature_change" if local_wins else "global_feature_diversity", scores[chosen])
        reasons[chosen].append("balanced_timeline_bin")
    result_scores = {}
    for i, frame in enumerate(ordered):
        result_scores[frame["frame_id"]] = {
            "timestamp_ms": frame["timestamp_ms"], "frame_index": frame["frame_index"],
            "selected": i in selected, "protected": i in anchors, "timeline_bin": int(bins[i]),
            "clarity_raw": float(quality[i]), "clarity_rank": float(quality_rank[i]),
            "global_change_from_previous": float(global_change[i]), "local_change_from_previous": float(local_change[i]),
            "global_distance_to_selection": float(novelty_global[i]), "local_distance_to_selection": float(novelty_local[i]),
            "selection_score": chosen_scores.get(i), "reasons": reasons.get(i, ["available_for_contextual_retrieval"]),
        }
    return {
        "algorithm": ALGORITHM_VERSION, "selected_ids": [ordered[i]["frame_id"] for i in sorted(selected)],
        "protected_ids": [ordered[i]["frame_id"] for i in sorted(anchors)], "scores": result_scores,
        "settings": {"budget": budget, "source_frames": n, "anchor_count": len(anchors), "timeline_bins": bins_count,
                     "weights": {"global_novelty": 0.50, "local_novelty": 0.30, "adjacent_change": 0.15, "clarity": 0.05}},
        "interpretation": "Visual candidate proposals only; clinical importance is not established. No source frames are deleted.",
    }


def select_candidates(frames: list[dict], budget: int, *, embedding_cache: Path,
                      model_path: str | Path | None = None, backend: str = "dinov3") -> dict:
    """Select candidates using real local DINO weights, with no implicit fallback.

    ``image_path`` and ``image_sha256`` are required alongside IDs and timestamps.
    No checkpoint downloads occur here. Defaults resolve inside this checkout.
    """
    frames = _validate_frames(frames, budget)
    if backend not in DEFAULT_MODELS:
        raise ValueError("embedding backend must be explicitly dinov2 or dinov3")
    if model_path is None:
        model_path = Path(__file__).resolve().parents[2] / ".runtime" / "models" / DEFAULT_MODELS[backend]
    globals_, locals_, quality, identity, cache = _load_embeddings(frames, Path(embedding_cache), Path(model_path).resolve(), backend)
    result = select_from_embeddings(frames, budget, globals_, locals_, quality)
    result["encoder"] = identity
    result["embedding_cache"] = cache
    return result
