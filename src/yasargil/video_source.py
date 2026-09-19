"""Prepare complete video evidence with an explicit, auditable frame timeline.

Original videos are never re-encoded. Released image sequences require an
explicit nominal frame rate and cannot supply original acquisition timestamps.
This manifest is separate from the existing enhancement-record contract.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from PIL import Image


class VideoSourceError(ValueError):
    """The source cannot be prepared without losing its frame provenance."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise VideoSourceError(f"{Path(command[0]).name} failed: {detail.strip()}") from error
    # All commands use -v error. Decoder errors must not become silent gaps.
    if result.stderr.strip():
        raise VideoSourceError(f"{Path(command[0]).name} reported an error: {result.stderr.strip()}")
    return result.stdout


def _probe(video: Path, ffprobe: str) -> tuple[dict, list[dict]]:
    raw = _run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries",
        "stream=index,width,height,time_base,start_pts,start_time,duration,avg_frame_rate,r_frame_rate:"
        "frame=pts,best_effort_timestamp,duration,pkt_duration,width,height:format=duration",
        "-of", "json", str(video),
    ])
    try:
        probe = json.loads(raw)
        stream = probe["streams"][0]
        time_base = Fraction(stream["time_base"])
        decoded = probe["frames"]
        if time_base <= 0 or not decoded:
            raise ValueError("No decoded frames or valid time base")
        previous = None
        for frame in decoded:
            if "pts" not in frame:
                raise ValueError("A decoded frame has no source PTS")
            pts = int(frame["pts"])
            if previous is not None and pts <= previous:
                raise ValueError("Duplicate or non-increasing source PTS")
            if (frame["width"], frame["height"]) != (stream["width"], stream["height"]):
                raise ValueError("Variable frame dimensions are not supported")
            previous = pts
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError) as error:
        raise VideoSourceError(f"Cannot establish a complete source PTS timeline for {video}: {error}") from error
    return probe, decoded


def _release_files(directory: Path) -> list[tuple[int, Path]]:
    images = [path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        raise VideoSourceError(f"No released JPEG or PNG frames found in {directory}")
    numbered = []
    prefixes = set()
    for path in images:
        match = re.fullmatch(r"(.*?)([0-9]+)\.(?:jpe?g|png)", path.name, re.IGNORECASE)
        if not match or not path.is_file():
            raise VideoSourceError(f"Released frames must be readable numbered image files: {path}")
        prefixes.add(match.group(1))
        numbered.append((int(match.group(2)), path))
    if len(prefixes) != 1:
        raise VideoSourceError("The directory contains multiple frame-name prefixes; use one complete sequence")
    numbered.sort(key=lambda item: item[0])
    indices = [index for index, _ in numbered]
    if len(set(indices)) != len(indices):
        raise VideoSourceError("Duplicate released frame indices")
    if indices != list(range(1, len(indices) + 1)):
        raise VideoSourceError("Missing released frames: sequence must be contiguous and start at index 1")
    return numbered


def _read_image(path: Path) -> tuple[Image.Image, str]:
    try:
        content = path.read_bytes()
        with Image.open(BytesIO(content)) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError("Animated images are not individual released frames")
            opened.load()
            return opened.convert("RGB"), hashlib.sha256(content).hexdigest()
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise VideoSourceError(f"Unreadable released frame {path}: {error}") from error


def _duration(decoded: list[dict], time_base: Fraction) -> tuple[float, str]:
    last = decoded[-1]
    length = last.get("duration", last.get("pkt_duration"))
    if length is not None and int(length) > 0:
        duration = int(length)
        basis = "last_source_pts_plus_decoded_frame_duration"
    elif len(decoded) > 1:
        duration = int(last["pts"]) - int(decoded[-2]["pts"])
        basis = "last_source_pts_plus_previous_frame_interval_estimate"
    else:
        duration = 0
        basis = "single_frame_extent_unknown"
    return float((int(last["pts"]) - int(decoded[0]["pts"]) + duration) * time_base * 1000), basis


def media_timeline(manifest: dict) -> dict:
    """Derive and validate playback timing exclusively from supplied media.

    Released images use their explicit reconstruction cadence; original videos
    retain their decoded presentation timestamps, including variable cadence.
    Trial/repair duration and other dataset metadata are never timing inputs.
    This does not establish original elapsed procedure time or complete coverage.
    """
    try:
        frames = manifest["frames"]
        probe = manifest["ffprobe"]
        decoded, streams = probe["frames"], probe["streams"]
        count = manifest["expected_video_frames"]
        if type(count) is not int or count < 1 or count != len(frames) or count != len(decoded):
            raise ValueError("Frame count must match every decoded and indexed media frame")
        if len(streams) != 1 or manifest["video_stream_index"] != streams[0]["index"]:
            raise ValueError("Source manifest and probe must identify the same video stream")
        time_base = Fraction(streams[0]["time_base"])
        if time_base <= 0:
            raise ValueError("Media time base must be positive")
        source_kind = manifest["source_kind"]
        if source_kind not in {"released_image_sequence", "original_video"}:
            raise ValueError("Unknown source kind")
        is_release = source_kind == "released_image_sequence"
        basis = "reconstructed_nominal" if is_release else "source_pts"
        if manifest["timestamp_basis"] != basis:
            raise ValueError("Timestamp basis differs from the supplied media kind")
        origin = int(decoded[0]["pts"])
        if is_release:
            fps = manifest["released_fps"]
            if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
                raise ValueError("Explicit released FPS must be finite and positive")
            rate = Fraction(str(fps))
            expected_duration = float(Fraction(count, 1) / rate * 1000)
            duration_basis = "released_frame_count_divided_by_explicit_nominal_fps"
            if origin != 0 or manifest["timestamp_origin_source_pts"] is not None:
                raise ValueError("Released reconstruction must start at encoded PTS zero with no acquisition PTS origin")
            # A quantized container can round both the last PTS and its extent.
            # Cap tolerance below a whole frame even with a coarse time base.
            frame_tolerance = float(min(time_base, Fraction(1, 1) / rate / 4) * 1000)
            extent_tolerance = float(min(2 * time_base, Fraction(1, 1) / rate / 4) * 1000)
        else:
            if manifest["timestamp_origin_source_pts"] != origin:
                raise ValueError("Original source PTS origin differs from decoded media")
            if manifest.get("released_fps") is not None:
                raise ValueError("Original videos cannot use a released-frame reconstruction cadence")
            expected_duration, duration_basis = _duration(decoded, time_base)
            fps = None
        duration = manifest["duration_ms"]
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration < 0
                or not math.isfinite(expected_duration) or abs(duration - expected_duration) > 1e-7):
            raise ValueError("Source duration does not match the supplied media playback duration")
        if manifest["duration_basis"] != duration_basis:
            raise ValueError("Duration basis does not match the media-derived calculation")
        previous_pts = None
        for index, (frame, actual) in enumerate(zip(frames, decoded)):
            pts = int(actual["pts"])
            if previous_pts is not None and pts <= previous_pts:
                raise ValueError(f"Decoded media PTS must increase at frame {index}")
            previous_pts = pts
            if type(frame["frame_index"]) is not int or frame["frame_index"] != index:
                raise ValueError(f"Frame ordinal mapping is incomplete or out of order at frame {index}")
            if frame["video_pts"] != pts or Fraction(frame["video_time_base"]) != time_base:
                raise ValueError(f"Frame {index} does not map to the decoded media PTS/time base")
            timestamp = frame["timestamp_ms"]
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
                raise ValueError(f"Invalid media timestamp at frame {index}")
            elapsed = float((pts - origin) * time_base * 1000)
            expected_timestamp = float(Fraction(index, 1) / rate * 1000) if is_release else elapsed
            if frame["timestamp_basis"] != basis or abs(timestamp - expected_timestamp) > 1e-7:
                raise ValueError(f"Frame {index} does not preserve its {'nominal timing' if is_release else 'source PTS timing'}")
            if is_release:
                if any(frame.get(key) is not None for key in ("source_pts", "time_base", "source_timestamp_ms")):
                    raise ValueError("Released reconstruction cannot claim original acquisition PTS")
                if abs(elapsed - expected_timestamp) > frame_tolerance + 1e-7:
                    raise ValueError(f"Encoded release PTS differ from reconstruction cadence at frame {index}")
            elif (frame["source_pts"] != pts or Fraction(frame["time_base"]) != time_base
                  or frame["source_timestamp_ms"] != float(pts * time_base * 1000)):
                raise ValueError(f"Original source PTS provenance is inconsistent at frame {index}")
        if is_release:
            encoded_duration, _ = _duration(decoded, time_base)
            if not math.isfinite(encoded_duration) or abs(encoded_duration - expected_duration) > extent_tolerance + 1e-7:
                raise ValueError("Encoded release extent does not match frame count divided by reconstruction FPS")
        result = {
            "authority": "supplied_video_playback",
            "duration_ms": expected_duration, "duration_basis": duration_basis,
            "timestamp_basis": basis, "frame_count": count, "playback_fps": float(fps) if is_release else None,
            "repair_time_used": False, "original_procedure_elapsed_time_verified": False,
            "full_procedure_coverage_verified": False,
            "zero_based_frame_timestamp_formula": (
                "frame_index / playback_fps * 1000" if is_release
                else "(source_pts - first_source_pts) * source_time_base * 1000"),
        }
        if ("media_timeline" in manifest and
                json.dumps(manifest["media_timeline"], sort_keys=True, allow_nan=False)
                != json.dumps(result, sort_keys=True, allow_nan=False)):
            raise ValueError("Recorded media timeline differs from the media-derived timing policy")
        return result
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError) as error:
        raise VideoSourceError(f"Media timeline is invalid: {error}") from error


def prepare_video_source(
    input_path: Path,
    output_dir: Path,
    *,
    released_fps: float | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> dict:
    """Save ``source.json`` and return all frames and their source provenance.

    ``timestamp_ms`` is the offset from the first decoded source PTS for an
    original video. Exact source PTS, time base and the unnormalized source time
    are also retained. A directory's timestamps are explicitly reconstructed
    nominal offsets; its source PTS and acquisition times remain null.

    The output directory must be empty. Original videos are linked unchanged;
    every frame is extracted to PNG without a sampling filter. Released images
    are staged as RGB PNGs and encoded losslessly into a complete native video.
    No source file is modified. Preparation fails on a gap or decoder error.
    """
    source = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.exists():
        raise VideoSourceError(f"Missing video source: {source}")
    is_release = source.is_dir()
    if not is_release and not source.is_file():
        raise VideoSourceError(f"Source is not a video file or released-frame directory: {source}")
    if source == output or output in source.parents:
        raise VideoSourceError("The output directory cannot contain or replace the source")
    if is_release and source in output.parents:
        raise VideoSourceError("Write preparation output outside the released source directory")
    if is_release:
        if released_fps is None or isinstance(released_fps, bool) or not math.isfinite(released_fps) or released_fps <= 0:
            raise VideoSourceError("A released image directory requires explicit released_fps > 0; original capture times are unavailable")
        numbered = _release_files(source)
        rate = Fraction(str(released_fps))
    elif released_fps is not None:
        raise VideoSourceError("released_fps applies only to released image directories; original videos retain their timing")
    executable = {"ffmpeg": shutil.which(str(ffmpeg)), "ffprobe": shutil.which(str(ffprobe))}
    if not all(executable.values()):
        raise VideoSourceError("Both ffmpeg and ffprobe must be installed")
    ffmpeg, ffprobe = executable["ffmpeg"], executable["ffprobe"]
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise VideoSourceError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    asset_dir = output / "frames"
    asset_dir.mkdir()
    commands = []
    frames = []

    if is_release:
        size = None
        for ordinal, (release_index, path) in enumerate(numbered):
            image, source_hash = _read_image(path)
            if size is not None and image.size != size:
                raise VideoSourceError(f"Released frame dimensions differ at {path}; no resizing is permitted")
            size = image.size
            image_path = asset_dir / f"frame-{ordinal:08d}.png"
            image.save(image_path)
            frames.append({
                "frame_index": ordinal, "release_frame_index": release_index,
                "source_path": str(path.resolve()), "source_sha256": source_hash,
                "image_path": str(image_path), "image_sha256": file_sha256(image_path),
                "timestamp_ms": float(Fraction(ordinal, 1) / rate * 1000),
                "timestamp_basis": "reconstructed_nominal", "source_pts": None,
                "time_base": None, "source_timestamp_ms": None, "source_acquisition_time": None,
                "width": size[0], "height": size[1],
            })
        identity = [{"name": Path(frame["source_path"]).name, "sha256": frame["source_sha256"]} for frame in frames]
        source_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        video = output / "complete-released-sequence.mp4"
        command = [
            ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-n",
            "-framerate", str(rate), "-start_number", "0", "-i", str(asset_dir / "frame-%08d.png"),
            "-map", "0:v:0", "-c:v", "libx264rgb", "-crf", "0", "-preset", "veryfast",
            "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-an", "-movflags", "+faststart", str(video),
        ]
        commands.append(command)
        _run(command)
        probe, decoded = _probe(video, ffprobe)
        encoded_time_base = Fraction(probe["streams"][0]["time_base"])
        if len(decoded) != len(frames):
            raise VideoSourceError("Reconstructed video does not contain every released frame")
        for frame, encoded in zip(frames, decoded):
            actual_ms = float(int(encoded["pts"]) * encoded_time_base * 1000)
            if abs(actual_ms - frame["timestamp_ms"]) > max(float(encoded_time_base * 1000), 1e-6):
                raise VideoSourceError("Reconstructed video timestamps do not match the explicit nominal timeline")
            frame["video_pts"] = int(encoded["pts"])
            frame["video_time_base"] = str(encoded_time_base)
        source_kind = "released_image_sequence"
        duration_ms = float(Fraction(len(frames), 1) / rate * 1000)
        duration_basis = "released_frame_count_divided_by_explicit_nominal_fps"
        origin = None
        note = ("All released images are included in filename order. Timing is nominal, reconstructed from the "
                "explicit frame rate; original capture PTS, acquisition timestamps, and images between released "
                "frames are unavailable. The video uses lossless RGB encoding of decoded source pixels.")
    else:
        source_hash = file_sha256(source)
        probe, decoded = _probe(source, ffprobe)
        time_base = Fraction(probe["streams"][0]["time_base"])
        origin = int(decoded[0]["pts"])
        suffix = source.suffix if re.fullmatch(r"\.[A-Za-z0-9]+", source.suffix) else ".video"
        video = output / ("original-video" + suffix)
        video.symlink_to(source)
        command = [
            ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-n",
            "-err_detect", "explode", "-i", str(source), "-map", "0:v:0", "-an",
            "-fps_mode", "passthrough", "-start_number", "0", str(asset_dir / "frame-%08d.png"),
        ]
        commands.append(command)
        _run(command)
        extracted = sorted(asset_dir.glob("frame-*.png"))
        if len(extracted) != len(decoded):
            raise VideoSourceError(f"Decoded frame count changed: ffprobe found {len(decoded)}, extraction produced {len(extracted)}")
        for ordinal, (frame, image_path) in enumerate(zip(decoded, extracted)):
            with Image.open(image_path) as image:
                image.load()
                if image.size != (frame["width"], frame["height"]):
                    raise VideoSourceError("Extracted dimensions differ from decoded source dimensions")
            pts = int(frame["pts"])
            frames.append({
                "frame_index": ordinal, "release_frame_index": None,
                "source_path": str(source), "source_sha256": source_hash,
                "image_path": str(image_path), "image_sha256": file_sha256(image_path),
                "timestamp_ms": float((pts - origin) * time_base * 1000),
                "timestamp_basis": "source_pts", "source_pts": pts, "time_base": str(time_base),
                "source_timestamp_ms": float(pts * time_base * 1000), "source_acquisition_time": None,
                "video_pts": pts, "video_time_base": str(time_base),
                "width": frame["width"], "height": frame["height"],
            })
        if file_sha256(source) != source_hash:
            raise VideoSourceError("The original video changed during preparation")
        source_kind = "original_video"
        duration_ms, duration_basis = _duration(decoded, time_base)
        note = ("The model input is a byte-identical link to the original video. Every decoded frame is indexed "
                "without an FPS filter. Timeline offsets are normalized to the first source PTS; original PTS "
                "and time base are retained. PTS denotes video presentation time, not wall-clock acquisition time.")

    for frame in frames:
        frame["frame_id"] = f"{source_kind}-{source_hash[:16]}-f{frame['frame_index']:08d}"
    manifest = {
        "schema_version": "yasargil.video-source.v1", "source_kind": source_kind,
        "source_path": str(source), "source_sha256": source_hash,
        "source_hash_basis": "ordered_source_filenames_and_file_sha256" if is_release else "original_file_bytes",
        "video_path": str(video), "video_sha256": file_sha256(video),
        "video_is_original_bytes": not is_release, "expected_video_frames": len(frames),
        "duration_ms": duration_ms, "duration_basis": duration_basis,
        "released_fps": float(rate) if is_release else None,
        "timestamp_origin_source_pts": origin,
        "timestamp_basis": "reconstructed_nominal" if is_release else "source_pts",
        "source_acquisition_time": None, "timeline_note": note,
        "frame_index_basis": "zero_based_decoded_ordinal",
        "extraction_sampling": "every_frame", "video_stream_index": probe["streams"][0]["index"],
        "commands": commands, "ffprobe": probe, "frames": frames,
    }
    manifest["media_timeline"] = media_timeline(manifest)
    temporary = output / "source.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "source.json")
    return manifest


def validate_native_timeline(manifest: dict) -> dict:
    """Reject timelines that llama.cpp b10809 cannot represent faithfully.

    In that build, ``--video-fps 0`` still decodes through FFmpeg's FPS filter
    at the probed ``r_frame_rate`` and labels frames using index / FPS. Thus a
    complete native file alone is insufficient: every source timestamp must
    agree with that constant-rate timeline. This guard does not alter media.
    A separate runtime RGB-hash comparison verifies the actual filtered pixels.
    """
    try:
        probe = manifest["ffprobe"]
        streams = probe["streams"]
        if len(streams) != 1:
            raise ValueError("Expected exactly one selected video stream in the provenance probe")
        stream = streams[0]
        if manifest["video_stream_index"] != stream["index"]:
            raise ValueError("Source manifest and probe disagree about the selected video stream")
        rate = Fraction(stream["r_frame_rate"])
        time_base = Fraction(stream["time_base"])
        if rate <= 0 or time_base <= 0:
            raise ValueError("A positive r_frame_rate and time_base are required")
        frames, decoded = manifest["frames"], probe["frames"]
        count = manifest["expected_video_frames"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Expected video frame count must be a positive integer")
        if count != len(frames) or count != len(decoded):
            raise ValueError("Expected frame count, extracted frame map, and decoded PTS count differ")
        source_kind = manifest["source_kind"]
        if source_kind not in {"original_video", "released_image_sequence"}:
            raise ValueError("Unknown source kind")
        origin = int(decoded[0]["pts"])
        if source_kind == "original_video" and manifest["timestamp_origin_source_pts"] != origin:
            raise ValueError("Source PTS origin does not match the first decoded frame")
        if source_kind == "released_image_sequence" and origin != 0:
            raise ValueError("Released reconstruction must start at encoded PTS zero")
        # Containers may quantize a fractional-rate timeline to their PTS tick.
        # A coarse tick must never make a whole missing frame acceptable.
        tolerance = min(time_base, Fraction(1, 1) / rate / 4) * 1000
        float_tolerance = 1e-7  # milliseconds; serialization arithmetic only
        max_deviation = Fraction(0)
        previous_pts = None
        for index, (frame, actual) in enumerate(zip(frames, decoded)):
            if frame["frame_index"] != index:
                raise ValueError(f"Frame ordinal mapping is incomplete or out of order at frame {index}")
            pts = int(actual["pts"])
            if previous_pts is not None and pts <= previous_pts:
                raise ValueError(f"Source PTS are duplicate or out of order at frame {index}")
            previous_pts = pts
            if frame["video_pts"] != pts or Fraction(frame["video_time_base"]) != time_base:
                raise ValueError(f"Extracted frame {index} does not map to its decoded video PTS/time base")
            elapsed_ms = (pts - origin) * time_base * 1000
            expected_ms = Fraction(index, 1) / rate * 1000
            deviation = abs(elapsed_ms - expected_ms)
            if deviation > tolerance:
                raise ValueError(
                    f"Irregular/VFR timeline at frame {index}: source offset {float(elapsed_ms):.6f} ms "
                    f"differs from index/r_frame_rate {float(expected_ms):.6f} ms "
                    f"by more than {float(tolerance):.6f} ms"
                )
            max_deviation = max(max_deviation, deviation)
            timestamp = float(frame["timestamp_ms"])
            if not math.isfinite(timestamp) or abs(timestamp - float(elapsed_ms)) > float(tolerance) + float_tolerance:
                raise ValueError(f"Provenance timestamp does not match the video timeline at frame {index}")
            if source_kind == "original_video":
                if (frame["timestamp_basis"] != "source_pts" or frame["source_pts"] != pts
                        or Fraction(frame["time_base"]) != time_base):
                    raise ValueError(f"Original source PTS provenance is inconsistent at frame {index}")
                if abs(timestamp - float(elapsed_ms)) > float_tolerance:
                    raise ValueError(f"Original source timestamp was altered at frame {index}")
            else:
                nominal_rate = Fraction(str(manifest["released_fps"]))
                if nominal_rate <= 0:
                    raise ValueError("Explicit released FPS must be positive")
                nominal_ms = float(Fraction(index, 1) / nominal_rate * 1000)
                if (frame["timestamp_basis"] != "reconstructed_nominal" or frame["source_pts"] is not None
                        or frame["time_base"] is not None or abs(timestamp - nominal_ms) > float_tolerance):
                    raise ValueError(f"Released frame {index} does not preserve its explicit nominal timing")
        media_timeline(manifest)
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError) as error:
        raise VideoSourceError(
            f"Native timeline is not safe for llama.cpp b10809: {error}. "
            "This runtime's FPS0 path can drop or duplicate frames and replace irregular timestamps; "
            "the source will not be resampled or re-encoded to bypass this check."
        ) from error
    return {
        "constant_frame_rate_verified": True,
        "fps": float(rate), "fps_rational": str(rate),
        "tolerance_ms": float(tolerance), "max_deviation_ms": float(max_deviation),
        "expected_count": count, "video_stream_index": stream["index"],
        "timestamp_origin_source_pts": origin,
        "verification_basis": "every_decoded_source_pts_compared_with_index_divided_by_r_frame_rate",
        "runtime": "llama.cpp-b10809", "runtime_fps_setting": 0,
    }


def retrieve_interval(manifest: dict, start_ms: float, end_ms: float, budget: int, exclude_ids=()) -> list[dict]:
    """Return real frames across an inclusive interval, in timeline order.

    A request with no observations in its interval returns an empty list. Times
    are never rounded into invented frames or silently expanded to neighbors.
    """
    if (isinstance(start_ms, bool) or isinstance(end_ms, bool)
            or not math.isfinite(start_ms) or not math.isfinite(end_ms)
            or start_ms < 0 or end_ms < start_ms):
        raise VideoSourceError("Retrieval requires finite timestamps with 0 <= start_ms <= end_ms")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise VideoSourceError("Retrieval budget must be a positive integer")
    timeline = media_timeline(manifest)
    if end_ms > timeline["duration_ms"]:
        raise VideoSourceError("Retrieval interval exceeds the supplied media playback duration")
    excluded = set(exclude_ids)
    available = sorted(
        (frame for frame in manifest["frames"] if start_ms <= frame["timestamp_ms"] <= end_ms and frame["frame_id"] not in excluded),
        key=lambda frame: (frame["timestamp_ms"], frame["frame_index"]),
    )
    if len(available) <= budget:
        return available
    if budget == 1:
        midpoint = (start_ms + end_ms) / 2
        return [min(available, key=lambda frame: abs(frame["timestamp_ms"] - midpoint))]
    indices = [round(index * (len(available) - 1) / (budget - 1)) for index in range(budget)]
    return [available[index] for index in indices]
