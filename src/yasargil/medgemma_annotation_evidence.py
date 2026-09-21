"""Source-only visual evidence for independent, per-frame surgical annotation.

The selected target stays fixed. Neighbors and reproducible native-pixel crops
provide bounded evidence; model drafts and dataset labels are never inputs.
"""
from __future__ import annotations

import copy
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image

from .annotation_contract import _source_index
from .contract import ContractError, require, sha256_file
from .video_source import media_timeline


PROTOCOL_VERSION = "medgemma-annotation-evidence-v1"
FRAME_FIELDS = (
    "frame_id", "frame_index", "release_frame_index", "source_path", "source_sha256",
    "image_path", "image_sha256", "timestamp_ms", "timestamp_basis", "source_pts",
    "time_base", "source_timestamp_ms", "source_acquisition_time", "video_pts",
    "video_time_base", "width", "height",
)


def canonical_frame(frame):
    """Copy provenance fields only, never annotations attached to a source row."""
    return {field: copy.deepcopy(frame[field]) for field in FRAME_FIELDS if field in frame}


def _read_image(frame, verified_sources):
    source = Path(frame["source_path"])
    identity = (str(source), frame["source_sha256"])
    try:
        if identity not in verified_sources:
            require(source.is_file() and sha256_file(source) == frame["source_sha256"],
                    f"Source image/video hash mismatch: {frame['frame_id']}")
            verified_sources.add(identity)
        path = Path(frame["image_path"])
        content = path.read_bytes()
        require(hashlib.sha256(content).hexdigest() == frame["image_sha256"],
                f"Evidence image hash mismatch: {frame['frame_id']}")
        with Image.open(BytesIO(content)) as image:
            require(getattr(image, "n_frames", 1) == 1, "Evidence images must be individual stills")
            require(type(frame.get("width")) is int and type(frame.get("height")) is int
                    and image.size == (frame["width"], frame["height"]),
                    f"Evidence image dimensions differ from source: {frame['frame_id']}")
            image.load()
            return image.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"Cannot read evidence image {frame['frame_id']}: {exc}") from exc


def _crop_bounds(width, height):
    # Four overlapping 60%-width/height windows, never resized or enhanced.
    crop_width = min(width, (3 * width + 4) // 5)
    crop_height = min(height, (3 * height + 4) // 5)
    candidates = [(x, y, x + crop_width, y + crop_height)
                  for y in (0, height - crop_height) for x in (0, width - crop_width)]
    return list(dict.fromkeys(bounds for bounds in candidates if bounds != (0, 0, width, height)))


def build_evidence(source, target_frame_id, output_dir, *, before_frames=2, after_frames=2,
                   detail_crops=True, procedure_context=""):
    """Build at most 17 source observations and four deterministic target crops.

    Zero neighbors and disabled crops give a target-only baseline. Neighbor counts
    address the source inventory, not the selected-frame subset or inferred time.
    """
    for label, value in (("Before-frame count", before_frames), ("After-frame count", after_frames)):
        require(type(value) is int and 0 <= value <= 8, f"{label} must be an integer from 0 to 8")
    require(type(detail_crops) is bool, "Detail-crop option must be boolean")
    require(isinstance(procedure_context, str) and len(procedure_context) <= 6000,
            "Procedure context must be text of at most 6000 characters")
    canonical, _ = _source_index(source)
    timeline = media_timeline(source)
    require(isinstance(target_frame_id, str) and target_frame_id in canonical,
            "Annotation target is not a canonical source frame")
    target = canonical_frame(canonical[target_frame_id])
    position = target["frame_index"]
    before = source["frames"][max(0, position - before_frames):position]
    after = source["frames"][position + 1:position + after_frames + 1]
    frames = [canonical_frame(frame) for frame in before] + [target] + [canonical_frame(frame) for frame in after]
    verified_sources, images = set(), {}
    for frame in frames:
        images[frame["frame_id"]] = _read_image(frame, verified_sources)
    views = []

    def full_view(frame, role):
        return {"view_id": f"{frame['frame_id']}:full", "frame_id": frame["frame_id"],
                "image_path": frame["image_path"], "image_sha256": frame["image_sha256"],
                "role": role, "bounds": [0, 0, frame["width"], frame["height"]],
                "width": frame["width"], "height": frame["height"]}

    views.append(full_view(target, "target"))
    if detail_crops:
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        input_paths = {Path(frame[key]).resolve() for frame in frames for key in ("source_path", "image_path")}
        for index, bounds in enumerate(_crop_bounds(target["width"], target["height"]), start=1):
            path = directory / f"target-{position:08d}-detail-{index}.png"
            require(path.resolve() not in input_paths and not path.is_symlink(),
                    "Crop output cannot overwrite a source image or follow a symlink")
            images[target_frame_id].crop(bounds).save(path, format="PNG")
            views.append({"view_id": f"{target_frame_id}:detail:{index}", "frame_id": target_frame_id,
                          "image_path": str(path), "image_sha256": sha256_file(path), "role": "target_detail",
                          "bounds": list(bounds), "width": bounds[2] - bounds[0], "height": bounds[3] - bounds[1]})
    views.extend(full_view(frame, "context_before") for frame in frames if frame["frame_index"] < position)
    views.extend(full_view(frame, "context_after") for frame in frames if frame["frame_index"] > position)
    limitations = [
        "Only supplied still images are evidence. Changes between observations do not establish unseen trajectories or events.",
        "Crops repeat native target pixels; they are not independent observations or additional temporal evidence.",
        "Documented procedure context is background, not proof that an anatomy, action, or local step is visible.",
        "No model draft, source annotation label, outcome, or surgeon-experience metadata is supplied.",
        "Unresolved evidence requests are recorded; automatic retrieval is not performed in this version.",
    ]
    if timeline["timestamp_basis"] == "reconstructed_nominal":
        limitations.append("Timestamps are reconstructed nominal playback offsets. Original acquisition times and images "
                           "between released observations are unavailable.")
    if not detail_crops:
        limitations.append("Detail crops are disabled; fine target structures may be lost in model image preprocessing.")
    coverage = {}
    for role, actual, requested in (("before", before, before_frames), ("after", after, after_frames)):
        missing = requested - len(actual)
        coverage[role] = {"requested": requested, "included_frame_ids": [frame["frame_id"] for frame in actual],
                          "unavailable_count": missing,
                          "availability": "complete" if not missing else "partial" if actual else "unavailable"}
        if missing:
            limitations.append(f"{missing} requested {role} source observations are unavailable at the source boundary.")
    return {"schema_version": PROTOCOL_VERSION, "target_frame_id": target_frame_id,
            "target": copy.deepcopy(target), "frames": frames, "views": views,
            "media_timeline": timeline, "procedure_context": procedure_context.strip(),
            "limitations": limitations, "neighbor_coverage": coverage}
