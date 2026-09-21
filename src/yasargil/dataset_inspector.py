"""Local inspection and separate human curation of saved pipeline runs.

The source inventory alone defines the timeline. Dataset outcomes are joined for
human inspection here and are never sent to a model. Every media endpoint is an
opaque handle registered from a discovered artifact, never a filesystem route.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import fcntl
from fractions import Fraction
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from urllib.parse import parse_qs, unquote, urlsplit


_SELECTION = "smart-frame-selection-run-v1"
_ANNOTATION = "full-video-frame-annotation-v1"
_ANNOTATION_SCHEMAS = {_ANNOTATION, "full-video-frame-annotation-v2"}
_REVIEW = "medgemma-frame-review-v1"
_REVIEW_SCHEMAS = {_REVIEW, "medgemma-surgery-review-v1"}
_MEDGEMMA_ANNOTATION = "medgemma-frame-annotation-v1"
_SCHEMAS = {_SELECTION, *_ANNOTATION_SCHEMAS, *_REVIEW_SCHEMAS, _MEDGEMMA_ANNOTATION}
_CASE = re.compile(r"(?:S[1-8]A[1-3]|Clip[01])")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
_PREVIEW_PROFILE = "browser-h264-yuv420p-crf18-v1"
_CONTEXT_PREVIEW_PROFILE = "nine-canonical-frames-h264-yuv420p-v1"
_CURATION_SCHEMA = "yasargil-inspector-curation-v1"
_CURATION_TEXT_LIMIT = 100_000


class CurationError(ValueError):
    """A curation request was rejected without changing its source artifacts."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class PreviewError(RuntimeError):
    """A display preview cannot be created; canonical stills remain available."""


def _read(path):
    try:
        if path.is_symlink():
            return None
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError, UnicodeError):
        return None


def _digest(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _same_path(value, path):
    return isinstance(value, str) and Path(value).is_absolute() and Path(value).resolve() == path.resolve()


def _path_from_absolute(value):
    return Path(value) if isinstance(value, str) and Path(value).is_absolute() else None


def _same_source(left, right):
    return (isinstance(left, dict) and isinstance(right, dict)
            and bool(left.get("video_sha256")) and bool(left.get("source_sha256"))
            and all(left.get(key) == right.get(key) for key in (
                "video_sha256", "source_sha256", "source_kind", "duration_ms", "expected_video_frames", "timestamp_basis"))
            and left.get("frames") == right.get("frames"))


def _canonical_match(row, canonical):
    return isinstance(row, dict) and all(key in row and row[key] == value for key, value in canonical.items())


def _objects(document, key):
    rows = document.get(key, []) if isinstance(document, dict) else []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _valid_source(source):
    if not isinstance(source, dict) or not isinstance(source.get("frames"), list) or not source["frames"]:
        return False
    duration = source.get("duration_ms")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
        return False
    seen, previous = set(), -1
    for index, frame in enumerate(source["frames"]):
        if not isinstance(frame, dict):
            return False
        frame_id, timestamp = frame.get("frame_id"), frame.get("timestamp_ms")
        if (not isinstance(frame_id, str) or not frame_id or frame_id in seen
                or frame.get("frame_index") != index
                or not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool)
                or not math.isfinite(timestamp) or not previous < timestamp <= duration):
            return False
        seen.add(frame_id)
        previous = timestamp
    return source.get("expected_video_frames") == len(seen)


class InspectorStore:
    """Discover run lineages and expose immutable response copies to the UI."""

    def __init__(self, runs_root, dataset_root=None):
        self.runs_root = Path(runs_root).expanduser().resolve()
        self.dataset_root = Path(dataset_root).expanduser().resolve() if dataset_root else None
        self._lock = threading.RLock()
        self._records = {}
        self._media = {}
        self._previews = {}
        self._preview_locks = {}
        self._verified_file_hashes = {}
        self.preview_cache = Path(tempfile.gettempdir()).resolve() / "yasargil-inspector-previews"
        self.curation_root = self.runs_root / ".inspector-curation"
        self._csv_cache = {}
        self._warnings = []
        self.refresh()

    def _warn(self, message):
        if message not in self._warnings:
            self._warnings.append(message)

    def open_dataset_folder(self):
        """Reveal the configured output directory; never accept a client path."""
        if not self.runs_root.is_dir():
            raise OSError("Dataset output folder is unavailable")
        if sys.platform == "darwin":
            command = ["open", str(self.runs_root)]
        elif sys.platform == "win32":
            command = ["explorer.exe", str(self.runs_root)]
        else:
            command = ["xdg-open", str(self.runs_root)]
        subprocess.run(command, check=True, timeout=10, capture_output=True)
        return {"opened": True, "path": str(self.runs_root)}

    def _register(self, value, kind="artifact"):
        if not isinstance(value, (str, Path)):
            return None
        path = Path(value)
        if not path.is_absolute() or not path.is_file():
            return None
        allowed = _IMAGE_SUFFIXES if kind == "image" else _VIDEO_SUFFIXES if kind == "video" else {".json"}
        if path.suffix.lower() not in allowed:
            return None
        resolved = path.resolve()
        if resolved.suffix.lower() not in allowed:
            return None
        stat = resolved.stat()
        token = hashlib.sha256(str(resolved).encode()).hexdigest()[:32]
        self._media[token] = (resolved, stat.st_dev, stat.st_ino)
        return f"/media/{token}"

    def media(self, token):
        """Resolve only a previously registered file, rejecting replacement links."""
        with self._lock:
            registered = self._media.get(token)
            if not registered:
                return None
            path, device, inode = registered
            try:
                stat = path.stat()
                if path.is_symlink() or not path.is_file() or (stat.st_dev, stat.st_ino) != (device, inode):
                    return None
            except OSError:
                return None
            return path

    def _preview_url(self, source, source_url):
        digest = source.get("video_sha256")
        if not source_url or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            return None
        media_token = source_url.rsplit("/", 1)[-1]
        token = hashlib.sha256(f"{media_token}:{digest}:{_PREVIEW_PROFILE}".encode()).hexdigest()[:32]
        self._previews[token] = {"media_token": media_token, "source_sha256": digest,
            "duration_ms": source["duration_ms"], "timestamps_ms": [row["timestamp_ms"] for row in source["frames"]]}
        return f"/preview/{token}"

    def _verified_hash(self, path):
        """Hash large video files incrementally, reusing only unchanged inodes."""
        try:
            before = path.stat()
            stamp = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            with self._lock:
                previous = self._verified_file_hashes.get(path)
            if previous and previous[0] == stamp:
                return previous[1]
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            after = path.stat()
            if stamp != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise PreviewError("Video changed while preparing the preview. Refresh and use the source stills.")
            value = digest.hexdigest()
            with self._lock:
                self._verified_file_hashes[path] = (stamp, value)
            return value
        except OSError as exc:
            raise PreviewError("Video is unavailable. Canonical source stills can still be inspected.") from exc

    @staticmethod
    def _verify_preview(path, expected, ffprobe):
        result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "stream=codec_name,pix_fmt,duration:frame=best_effort_timestamp_time",
            "-of", "json", str(path)], capture_output=True, timeout=30, check=True)
        document = json.loads(result.stdout)
        streams = document.get("streams", [])
        frames = document.get("frames", [])
        if (len(streams) != 1 or streams[0].get("codec_name") != "h264"
                or streams[0].get("pix_fmt") != "yuv420p"
                or len(frames) != len(expected["timestamps_ms"])):
            raise PreviewError("Browser preview verification failed. Use the canonical source stills.")
        times = [float(frame["best_effort_timestamp_time"]) * 1000 for frame in frames]
        duration = float(streams[0]["duration"]) * 1000
        if (any(not math.isfinite(actual) or abs(actual - original) > 1.1
                for actual, original in zip(times, expected["timestamps_ms"]))
                or not math.isfinite(duration) or abs(duration - expected["duration_ms"]) > 1.1):
            raise PreviewError("Preview timing differs from the source timeline. Use the canonical source stills.")
        return {"frame_count": len(frames), "duration_ms": duration, "codec": "h264", "pixel_format": "yuv420p"}

    def preview(self, token):
        """Generate a verified browser derivative only when its URL is requested.

        Lossless RGB H.264 can be decoded with incorrect colors by browsers.
        The YUV420 H.264 derivative is for display only; inference and the source
        inventory continue to use the unchanged video and canonical stills.
        """
        with self._lock:
            expected = self._previews.get(token)
            if expected is None:
                return None
            expected = copy.deepcopy(expected)
            lock = self._preview_locks.setdefault(expected["source_sha256"], threading.Lock())
        if not lock.acquire(timeout=660):
            raise PreviewError("A browser preview is still being prepared. Try again or inspect source stills.")
        try:
            source = self.media(expected["media_token"])
            if source is None:
                raise PreviewError("Source video is unavailable. Inspect the canonical still frames instead.")
            digest = expected["source_sha256"]
            if self._verified_hash(source) != digest:
                raise PreviewError("Source video hash does not match its saved provenance. Preview was not generated.")
            signature = {"profile": _PREVIEW_PROFILE, "source_video_sha256": digest,
                         "duration_ms": expected["duration_ms"], "timestamps_ms": expected["timestamps_ms"]}
            destination = self.preview_cache / f"{digest}.mp4"
            receipt_path = self.preview_cache / f"{digest}.json"
            receipt = _read(receipt_path)
            if (not self.preview_cache.is_symlink() and destination.is_file() and not destination.is_symlink()
                    and isinstance(receipt, dict) and receipt.get("input") == signature
                    and self._verified_hash(destination) == receipt.get("preview_sha256")):
                return destination
            ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
            if not ffmpeg or not ffprobe:
                raise PreviewError("Browser video preview requires local ffmpeg and ffprobe. Source stills remain available.")
            self.preview_cache.mkdir(mode=0o700, parents=True, exist_ok=True)
            if self.preview_cache.is_symlink() or self.preview_cache.resolve() != self.preview_cache:
                raise PreviewError("Preview cache is not a safe local directory.")
            fd, temporary_name = tempfile.mkstemp(prefix=f".{digest}-", suffix=".mp4", dir=self.preview_cache)
            os.close(fd)
            temporary = Path(temporary_name)
            receipt_temporary = temporary.with_suffix(".json")
            try:
                command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", "2",
                    "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1",
                    "-vf", "setpts=PTS-STARTPTS,pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast", "-threads", "2",
                    "-fps_mode", "passthrough", "-movflags", "+faststart", str(temporary)]
                subprocess.run(command, capture_output=True, timeout=600, check=True)
                verification = self._verify_preview(temporary, expected, ffprobe)
                if self._verified_hash(source) != digest:
                    raise PreviewError("Source video changed during preview generation; no preview was published.")
                receipt = {"input": signature, "preview_sha256": self._verified_hash(temporary),
                           "verification": verification, "display_only": True, "source_modified": False}
                receipt_temporary.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                os.replace(temporary, destination)
                os.replace(receipt_temporary, receipt_path)
                return destination
            finally:
                temporary.unlink(missing_ok=True)
                receipt_temporary.unlink(missing_ok=True)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            raise PreviewError("Browser video preview could not be prepared. Inspect the canonical source stills instead.") from exc
        finally:
            lock.release()

    def context_preview(self, record_id, frame_id, review_identity=None):
        """Encode at most four canonical frames on each side of the target.

        Only the nearby image assets are read. The full surgery video is never
        decoded or hashed for this view, and playback cannot change the target.
        """
        with self._lock:
            item = self._records.get(record_id)
            if item is None:
                return None
            if review_identity is not None and review_identity != item["public"]["review_identity"]:
                raise PreviewError("The dataset changed. Refresh before playing this frame window.")
            frames = item["public"]["frames"]
            index = next((i for i, frame in enumerate(frames) if frame["frame_id"] == frame_id), None)
            if index is None:
                return None
            start, stop = max(0, index - 4), min(len(frames), index + 5)
            nearby = copy.deepcopy(frames[start:stop])
            duration_ms = item["public"]["duration_ms"]
            interval = duration_ms / len(frames)
            end_ms = frames[stop]["timestamp_ms"] if stop < len(frames) else duration_ms
        start_ms = nearby[0]["timestamp_ms"]
        timestamps = [frame["timestamp_ms"] - start_ms for frame in nearby]
        expected = {"timestamps_ms": timestamps, "duration_ms": end_ms - start_ms}
        if (any(abs(timestamp - i * interval) > 1.1 for i, timestamp in enumerate(timestamps))
                or abs(expected["duration_ms"] - len(nearby) * interval) > 1.1):
            raise PreviewError("This frame window has irregular timing. Inspect its source stills.")
        assets = []
        for frame in nearby:
            image_url, digest = frame.get("image_url"), frame.get("image_sha256")
            path = self.media(image_url.rsplit("/", 1)[-1]) if image_url else None
            if path is None or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise PreviewError("A nearby source frame is unavailable. Check its source drive.")
            assets.append((path, digest))
        signature = {"profile": _CONTEXT_PREVIEW_PROFILE, "image_sha256": [digest for _, digest in assets],
                     "timestamps_ms": timestamps, "duration_ms": expected["duration_ms"]}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        with self._lock:
            lock = self._preview_locks.setdefault(key, threading.Lock())
        if not lock.acquire(timeout=45):
            raise PreviewError("The nearby-frame video could not be prepared in time. Try again.")
        try:
            for path, digest in assets:
                if self._verified_hash(path) != digest:
                    raise PreviewError("A nearby frame differs from its saved provenance. Refresh the dataset.")
            destination = self.preview_cache / f"context-{key}.mp4"
            receipt_path = destination.with_suffix(".json")
            receipt = _read(receipt_path)
            if (not self.preview_cache.is_symlink() and destination.is_file() and not destination.is_symlink()
                    and isinstance(receipt, dict) and receipt.get("input") == signature
                    and self._verified_hash(destination) == receipt.get("preview_sha256")):
                return destination
            ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
            if not ffmpeg or not ffprobe:
                raise PreviewError("Nearby-frame video requires local ffmpeg and ffprobe.")
            self.preview_cache.mkdir(mode=0o700, parents=True, exist_ok=True)
            if self.preview_cache.is_symlink() or self.preview_cache.resolve() != self.preview_cache:
                raise PreviewError("Preview cache is not a safe local directory.")
            with tempfile.TemporaryDirectory(prefix=".context-", dir=self.preview_cache) as temporary_name:
                temporary = Path(temporary_name)
                entries = []
                for i, (path, digest) in enumerate(assets):
                    copied = temporary / f"frame-{i:02d}{path.suffix.lower()}"
                    shutil.copyfile(path, copied)
                    if self._verified_hash(copied) != digest:
                        raise PreviewError("A nearby frame changed during preview preparation. Refresh the dataset.")
                    entries.append(f"file '{copied.name}'")
                playlist = temporary / "frames.txt"
                playlist.write_text("\n".join(entries) + "\n", encoding="utf-8")
                output = temporary / "preview.mp4"
                rate = (Fraction(1000) / Fraction(str(interval))).limit_denominator(1_000_000)
                command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", "2",
                    "-f", "concat", "-safe", "1", "-i", str(playlist), "-an", "-sn", "-dn", "-map_metadata", "-1",
                    "-vf", f"settb=AVTB,setpts=N/({rate}*TB),pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-r", str(rate), "-frames:v", str(len(nearby)), "-fps_mode", "cfr",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast",
                    "-threads", "2", "-movflags", "+faststart", str(output)]
                subprocess.run(command, capture_output=True, timeout=30, check=True)
                verification = self._verify_preview(output, expected, ffprobe)
                receipt = {"input": signature, "preview_sha256": self._verified_hash(output),
                           "verification": verification, "display_only": True, "source_modified": False}
                pending_receipt = temporary / "receipt.json"
                pending_receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                os.replace(output, destination)
                os.replace(pending_receipt, receipt_path)
                return destination
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            raise PreviewError("Nearby-frame video could not be prepared. Inspect the source stills.") from exc
        finally:
            lock.release()

    def _artifacts(self, root, names):
        result = []
        for label, name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                continue
            path = root / relative
            if not path.resolve().is_relative_to(root.resolve()):
                continue
            url = self._register(path)
            if url:
                result.append({"label": label, "url": url})
        return result

    def _discover(self):
        runs = []
        if not self.runs_root.is_dir():
            self._warn(f"Runs directory is unavailable: {self.runs_root}")
            return runs
        for directory, children, files in os.walk(self.runs_root, followlinks=False):
            children[:] = sorted(child for child in children if not child.startswith(".")
                                 and not (Path(directory) / child).is_symlink())
            if "run.json" not in files:
                continue
            root = Path(directory)
            plan = _read(root / "run.json")
            if not isinstance(plan, dict):
                self._warn(f"Unreadable run metadata: {root}")
                continue
            if plan.get("schema_version") in _SCHEMAS:
                runs.append((root, plan))
                # Annotation and review directories contain frozen copies of
                # upstream runs. They are lineage artifacts, not new records.
                children[:] = []
        return sorted(runs, key=lambda pair: (str(pair[1].get("created_at", "")), str(pair[0])), reverse=True)

    def _mapped_dataset(self, source):
        value = source.get("source_path")
        if not isinstance(value, str):
            return None
        path = Path(value)
        if (source.get("source_kind") != "released_image_sequence" or path.parent.name != "frames"
                or not _CASE.fullmatch(path.name) or not path.is_absolute()):
            return None
        root = self.dataset_root or path.parent.parent.resolve()
        expected = root / "frames" / path.name
        if path != expected or path.resolve() != expected:
            return None
        for frame in source["frames"]:
            release_index = frame.get("release_frame_index")
            if type(release_index) is not int or release_index != frame["frame_index"] + 1:
                return None
            actual = Path(frame.get("source_path", ""))
            canonical = expected / f"{path.name}_frame_{release_index:08d}.jpeg"
            if actual != canonical or actual.resolve() != canonical:
                return None
        return root

    def _table(self, path):
        try:
            if path.is_symlink() or path.resolve() != path:
                return None
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            cached = self._csv_cache.get(path)
            if cached and cached[0] == stamp:
                return cached[1]
            content = path.read_bytes()
            import io
            reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
            fields = reader.fieldnames
            if not fields or len(fields) != len(set(fields)):
                return None
            rows = []
            for number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    return None
                rows.append((number, reader.line_num, row))
            table = {"fields": fields, "rows": rows, "sha256": hashlib.sha256(content).hexdigest()}
            self._csv_cache[path] = (stamp, table)
            return table
        except (OSError, UnicodeError, csv.Error):
            return None

    @staticmethod
    def _locator(path, table, number, physical_line):
        return {"source_path": str(path), "filename": path.name, "sha256": table["sha256"],
                "locator_type": "csv_record_1based_including_header", "locator": number,
                "physical_end_line_1based": physical_line}

    def _dataset(self, source):
        outcomes = {"status": "unavailable", "values": [], "raw": None, "source_locator": None,
                    "reason": "No exact SOSpine source-to-dataset mapping is available.",
                    "scope": "retrospective_human_review_only", "used_for_model_inference": False}
        labels = {}
        availability = {}
        root = self._mapped_dataset(source)
        if root is None:
            return outcomes, labels, availability
        case = Path(source["source_path"]).name
        path = root / "sospine_outcomes.csv"
        table = self._table(path)
        if table is None:
            outcomes["reason"] = "Outcome table is missing or unreadable."
        else:
            matches = [(number, physical, row) for number, physical, row in table["rows"]
                       if row.get("Trial ID") == case]
            if len(matches) != 1:
                outcomes["reason"] = "Expected one exact outcome row for this case; no unambiguous row is available."
            else:
                number, physical, row = matches[0]
                values = [{"label": field, "value": row[field] if row[field].strip() else None}
                          for field in table["fields"]]
                for value in values:
                    if value["label"] == "Time for repair":
                        value["unit"] = "seconds"
                outcomes.update(status="available", values=values, raw=row,
                                source_locator=self._locator(path, table, number, physical), reason=None)
        by_name = {Path(frame["source_path"]).name: frame["frame_id"] for frame in source["frames"]}
        for name, kind, origin in (("sospine_tool_tips.csv", "keypoint", "manual"),
                                   ("sospine_bbox.csv", "bbox", "computed")):
            path = root / name
            table = self._table(path)
            matches = {}
            if table is not None and "trial_frame" in table["fields"]:
                for number, physical, row in table["rows"]:
                    frame_id = by_name.get(row.get("trial_frame"))
                    if frame_id is None:
                        continue
                    item = {"annotation_id": f"{name}:row:{number}", "frame_id": frame_id,
                            "source_frame_filename": row["trial_frame"], "original_kind": kind,
                            "original_origin": origin, "raw_value": row,
                            "source_locator": self._locator(path, table, number, physical),
                            "coordinate_conversion": "none"}
                    labels.setdefault(frame_id, []).append(item)
                    matches.setdefault(frame_id, []).append(item)
            for frame_id in by_name.values():
                rows = matches.get(frame_id, [])
                provided = any(row["raw_value"].get("label", "").strip() for row in rows)
                availability.setdefault(frame_id, []).append({
                    "filename": name, "status": "provided" if provided else "unavailable",
                    "reason": None if provided else "blank_placeholder_rows" if rows else
                              "table_unavailable" if table is None else "no_matching_source_rows",
                    "negative_label": False})
        return outcomes, labels, availability

    def _annotations(self, root, source, selection, runs):
        matching = []
        for child, plan in runs:
            if plan.get("schema_version") not in _ANNOTATION_SCHEMAS or not _same_path(plan.get("selection_run"), root):
                continue
            copied_source = _read(child / "source/source.json")
            hashes = plan.get("input_sha256", {})
            if (not _same_source(source, copied_source)
                    or (hashes.get("selection.json") and hashes["selection.json"] != _digest(root / "selection.json"))
                    or (hashes.get("source/source.json") and hashes["source/source.json"] != _digest(root / "source/source.json"))):
                self._warn(f"Skipped annotation with mismatched selection/source lineage: {child}")
                continue
            matching.append((child, plan))
        if not matching:
            return None, {}, []
        child, plan = matching[0]
        document = _read(child / "annotations.json")
        if document is None and (_read(child / "summary.json") or {}).get("status") == "failed":
            # Explicitly authorized reviews can retain rejected Qwen drafts.
            # Load only their pinned derivative; leave the failed source run intact.
            for review_root, review_plan in runs:
                if (review_plan.get("schema_version") != "medgemma-surgery-review-v1"
                        or not _same_path(review_plan.get("annotation_run"), child)
                        or review_plan.get("qwen_input_status") != "rejected_temporal_citations"
                        or review_plan.get("config", {}).get("allow_rejected_temporal_citations") is not True):
                    continue
                hashes = review_plan.get("input_sha256", {})
                draft = _read(review_root / "qwen-drafts.json")
                audit = _read(review_root / "draft-intake.json")
                response_hash = _digest(child / "round-00/response.json")
                if (isinstance(draft, dict) and isinstance(audit, dict) and response_hash
                        and audit.get("validation_status") == "rejected_temporal_citations"
                        and draft.get("validation_status") == "rejected_temporal_citations"
                        and draft.get("validation_issues") == audit.get("issues")
                        and hashes.get("qwen-drafts.json") == _digest(review_root / "qwen-drafts.json")
                        and hashes.get("draft-intake.json") == _digest(review_root / "draft-intake.json")
                        and audit.get("raw_response_sha256") == response_hash
                        and hashes.get("qwen/round-00/response.json") == response_hash
                        and _digest(review_root / "qwen/round-00/response.json") == response_hash):
                    document = draft
                    break
                self._warn(f"Skipped rejected Qwen drafts with mismatched raw-response lineage: {review_root}")
        annotations = {}
        canonical = {frame["frame_id"]: frame for frame in source["frames"]}
        selected = set(selection.get("selected_frame_ids", []))
        if isinstance(document, dict):
            video_hash = document.get("verification", {}).get("video_sha256")
            lineage_ok = ((not document.get("selection_run") or _same_path(document["selection_run"], root))
                          and (not video_hash or video_hash == source.get("video_sha256")))
            if lineage_ok:
                for row in _objects(document, "annotations"):
                    frame_id = row.get("frame_id") if isinstance(row, dict) else None
                    if frame_id in selected and frame_id in canonical and _canonical_match(row, canonical[frame_id]):
                        annotations[frame_id] = row
                    else:
                        self._warn(f"Skipped Qwen annotation with mismatched frame provenance: {child}")
            else:
                self._warn(f"Skipped Qwen annotations with mismatched result lineage: {child}")
        return (child, plan), annotations, matching

    def _reviews(self, annotation_run, source, qwen, runs):
        if annotation_run is None:
            return None, {}, {}, []
        annotation_root, _ = annotation_run
        matching = []
        for child, plan in runs:
            if plan.get("schema_version") not in _REVIEW_SCHEMAS or not _same_path(plan.get("annotation_run"), annotation_root):
                continue
            copied_source = _read(child / "qwen/source/source.json")
            digest = plan.get("input_sha256", {}).get("qwen/annotations.json")
            if (not _same_source(source, copied_source)
                    or (digest and digest != _digest(annotation_root / "annotations.json"))):
                self._warn(f"Skipped MedGemma review with mismatched annotation/source lineage: {child}")
                continue
            matching.append((child, plan))
        if not matching:
            return None, {}, {}, []
        child, plan = matching[0]
        reviews, evidence = {}, {}
        canonical = {frame["frame_id"]: frame for frame in source["frames"]}

        def valid_packet(packet):
            if not isinstance(packet, dict):
                return False
            target = packet.get("target_frame_id")
            frames = packet.get("frames")
            return (target in qwen and packet.get("qwen_annotation") == qwen[target]
                    and isinstance(frames, list) and frames
                    and any(frame.get("frame_id") == target for frame in frames if isinstance(frame, dict))
                    and all(isinstance(frame, dict) and frame.get("frame_id") in canonical
                            and _canonical_match(frame, canonical[frame["frame_id"]]) for frame in frames))

        for filename in plan.get("evidence_files", []):
            if not isinstance(filename, str):
                continue
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                continue
            packet = _read(child / relative)
            if valid_packet(packet):
                evidence[packet["target_frame_id"]] = packet
        document = _read(child / "reviews.json")
        if isinstance(document, dict):
            for row in _objects(document, "reviews"):
                target = row.get("target_frame_id")
                judgment = row.get("medgemma_review")
                packet = row.get("evidence")
                if (valid_packet(packet) and target == packet["target_frame_id"]
                        and isinstance(judgment, dict) and judgment.get("target_frame_id") == target
                        and row.get("qwen_annotation") == qwen[target]):
                    reviews[target], evidence[target] = row, packet
                else:
                    self._warn(f"Skipped MedGemma review with mismatched frame evidence: {child}")
        return (child, plan), reviews, evidence, matching

    def _run_info(self, pair, result_name=None):
        if pair is None:
            return None
        root, plan = pair
        summary = _read(root / "summary.json") or _read(root / "state.json") or {}
        names = [("Run metadata", "run.json"), ("Run summary", "summary.json")]
        if result_name:
            names.append(("Saved results", result_name))
        return {"path": str(root), "name": root.name, "created_at": plan.get("created_at"),
                "schema_version": plan.get("schema_version"),
                "status": summary.get("status", "prepared"), "config": plan.get("config", {}),
                "artifacts": self._artifacts(root, names)}

    def _independent_annotations(self, root, source, selection, runs):
        """Join independent annotations directly to their selection provenance."""
        from .medgemma_annotation_contract import build_annotation
        from .medgemma_annotation_evidence import canonical_frame

        canonical = {frame["frame_id"]: canonical_frame(frame) for frame in source["frames"]}
        selected = set(selection.get("selected_frame_ids", []))

        def saved_path(child, value):
            if not isinstance(value, str) or not value:
                return None
            relative = Path(value)
            path = child / relative
            if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(child.resolve()):
                return None
            return path

        def pinned_hash(path):
            # Frozen selection requests can embed a full video. Reuse the
            # streaming, stat-keyed cache instead of rereading them into RAM on
            # every inspector refresh.
            try:
                return self._verified_hash(path)
            except PreviewError:
                return None

        matching = []
        for child, plan in runs:
            if plan.get("schema_version") != _MEDGEMMA_ANNOTATION or not _same_path(plan.get("selection_run"), root):
                continue
            source_path = saved_path(child, plan.get("source_file"))
            copied = _read(source_path) if source_path else None
            selected_path = saved_path(child, plan.get("selected_file"))
            copied_selected = _read(selected_path) if selected_path else None
            hashes = plan.get("input_sha256", {})
            same_source = (isinstance(copied, dict)
                and all(copied.get(key) == source.get(key) for key in
                        ("source_sha256", "video_sha256", "source_kind", "duration_ms", "timestamp_basis"))
                and [canonical_frame(row) for row in _objects(copied, "frames")] == list(canonical.values()))
            pinned_ok = isinstance(hashes, dict) and all(name in hashes for name in
                [plan.get("source_file"), plan.get("selected_file"), *plan.get("evidence_files", [])]) and all(
                saved_path(child, name) is not None and pinned_hash(saved_path(child, name)) == digest
                for name, digest in hashes.items())
            selected_ok = (isinstance(copied_selected, list) and copied_selected
                and all(isinstance(row, dict) and row.get("frame_id") in selected
                        and canonical_frame(row) == canonical.get(row["frame_id"]) for row in copied_selected)
                and [row["frame_id"] for row in copied_selected] == plan.get("frame_ids"))
            selection_digest = hashes.get("selection/selection.json") if isinstance(hashes, dict) else None
            if (not same_source or not pinned_ok or not selected_ok
                    or (selection_digest and selection_digest != _digest(root / "selection.json"))):
                self._warn(f"Skipped independent MedGemma annotation with mismatched selection/source lineage: {child}")
                continue
            matching.append((child, plan))
        if not matching:
            return None, {}, {}, []
        child, plan = matching[0]
        target_ids = set(plan.get("frame_ids", []))
        annotations, packets = {}, {}

        def valid_packet(packet):
            if not isinstance(packet, dict) or packet.get("schema_version") != "medgemma-annotation-evidence-v1":
                return False
            target = packet.get("target_frame_id")
            frames, views = packet.get("frames"), packet.get("views")
            if (target not in target_ids or target not in canonical
                    or packet.get("target") != canonical[target]
                    or not isinstance(frames, list) or not frames or not isinstance(views, list) or not views
                    or not all(isinstance(frame, dict) and frame.get("frame_id") in canonical
                               and frame == canonical[frame["frame_id"]] for frame in frames)):
                return False
            frame_ids = {frame["frame_id"] for frame in frames}
            if len(frame_ids) != len(frames):
                return False
            seen = set()
            for view in views:
                if (not isinstance(view, dict) or not isinstance(view.get("view_id"), str)
                        or view["view_id"] in seen or view.get("frame_id") not in frame_ids
                        or view.get("role") not in {"target", "target_detail", "context_before", "context_after"}):
                    return False
                seen.add(view["view_id"])
                frame = canonical[view["frame_id"]]
                if (view["role"] in {"target", "target_detail"}) != (view["frame_id"] == target):
                    return False
                bounds = view.get("bounds")
                if (not isinstance(bounds, list) or len(bounds) != 4 or any(type(value) is not int for value in bounds)
                        or not 0 <= bounds[0] < bounds[2] <= frame["width"]
                        or not 0 <= bounds[1] < bounds[3] <= frame["height"]
                        or view.get("width") != bounds[2] - bounds[0] or view.get("height") != bounds[3] - bounds[1]):
                    return False
                if (view["role"] == "context_before" and frame["frame_index"] >= canonical[target]["frame_index"]
                        or view["role"] == "context_after" and frame["frame_index"] <= canonical[target]["frame_index"]):
                    return False
                if view["role"] != "target_detail" and any(view.get(key) != frame.get(key)
                        for key in ("image_path", "image_sha256", "width", "height")):
                    return False
                if view["role"] == "target_detail":
                    crop_path = _path_from_absolute(view.get("image_path"))
                    if (crop_path is None or not crop_path.resolve().is_relative_to(child.resolve())
                            or _digest(crop_path) != view.get("image_sha256")):
                        return False
            return target in frame_ids and sum(view["role"] == "target" for view in views) == 1

        for filename in plan.get("evidence_files", []):
            path = saved_path(child, filename)
            packet = _read(path) if path else None
            if valid_packet(packet):
                packets[packet["target_frame_id"]] = packet
        document = _read(child / "annotations.json")
        if isinstance(document, dict) and document.get("schema_version") == _MEDGEMMA_ANNOTATION:
            for row in _objects(document, "annotations"):
                target, annotation, packet = row.get("target_frame_id"), row.get("annotation"), row.get("evidence")
                if (target in packets and packet == packets[target] and row.get("target") == canonical.get(target)
                        and isinstance(annotation, dict) and annotation.get("schema_version") == _MEDGEMMA_ANNOTATION
                        and annotation.get("target_frame_id") == target):
                    raw = {key: annotation.get(key) for key in ("target_frame_id", "visibility", "unresolved_questions")}
                    raw["claims"] = [{key: value for key, value in claim.items() if key != "evidence_frame_ids"}
                                     for claim in _objects(annotation, "claims")]
                    try:
                        if build_annotation(raw, packet) != annotation:
                            raise ValueError("Derived annotation differs from saved claims")
                    except ValueError:
                        self._warn(f"Skipped independent MedGemma annotation with invalid claims: {child}")
                        continue
                    annotations[target] = row
                else:
                    self._warn(f"Skipped independent MedGemma annotation with mismatched frame evidence: {child}")
        return (child, plan), annotations, packets, matching

    def refresh(self):
        with self._lock:
            self._warnings = []
            self._media = {}
            self._previews = {}
            runs = self._discover()
            records = {}
            for root, plan in runs:
                if plan.get("schema_version") != _SELECTION:
                    continue
                try:
                    source = _read(root / "source/source.json")
                    if not _valid_source(source):
                        self._warn(f"Selection source is incomplete or invalid: {root}")
                        continue
                    if (plan.get("source_manifest_sha256")
                            and plan["source_manifest_sha256"] != _digest(root / "source/source.json")):
                        self._warn(f"Skipped selection with a changed source manifest: {root}")
                        continue
                    selection = _read(root / "selection.json")
                    if not isinstance(selection, dict):
                        state = _read(root / "state.json") or {}
                        initial = _read(root / "initial-selection.json") or {}
                        candidates = state.get("candidate_ids", initial.get("selected_ids", []))
                        selection = {"frames": [{**frame, "model_decision": "unreviewed",
                                     "effective_decision": "unreviewed", "model_reason": "Awaiting Qwen selection review"}
                                     for frame in source["frames"] if frame["frame_id"] in candidates],
                                     "selected_frame_ids": [], "status": state.get("status", "prepared")}
                    if selection.get("video_sha256") and selection["video_sha256"] != source.get("video_sha256"):
                        self._warn(f"Skipped selection with mismatched source video: {root}")
                        continue
                    canonical_by_id = {frame["frame_id"]: frame for frame in source["frames"]}
                    decisions = {}
                    for row in _objects(selection, "frames"):
                        frame_id = row.get("frame_id")
                        if (isinstance(frame_id, str) and frame_id in canonical_by_id
                                and _canonical_match(row, canonical_by_id[frame_id])):
                            decisions[frame_id] = row
                        else:
                            self._warn(f"Skipped selection decision with mismatched frame provenance: {root}")
                    selected = set(selection.get("selected_frame_ids", []))
                    frames = []
                    for canonical in source["frames"]:
                        decision = decisions.get(canonical["frame_id"], {})
                        # Effective retention takes precedence over Qwen's drop;
                        # expose coverage_override separately so it stays honest.
                        retained = canonical["frame_id"] in selected or decision.get("effective_decision") == "keep"
                        status = ("selected" if retained else "dropped" if decision.get("model_decision") == "drop"
                                  else "candidate" if decision else "source")
                        frames.append({**canonical, "status": status,
                                       "model_decision": decision.get("model_decision"),
                                       "model_reason": decision.get("model_reason"),
                                       "effective_decision": decision.get("effective_decision"),
                                       "coverage_override": bool(decision.get("coverage_override")),
                                       "protected_temporal_anchor": bool(decision.get("protected_temporal_anchor")),
                                       "image_url": self._register(canonical.get("image_path"), "image")})
                    annotation_run, qwen, annotation_runs = self._annotations(root, source, selection, runs)
                    review_run, medgemma, evidence, review_runs = self._reviews(annotation_run, source, qwen, runs)
                    independent_run, independent, independent_evidence, independent_runs = self._independent_annotations(
                        root, source, selection, runs)
                    historical_review_runs = review_runs
                    independent_active = independent_run is not None
                    if independent_active:
                        review_run, medgemma, evidence, review_runs = (
                            independent_run, independent, independent_evidence, independent_runs)
                    medgemma_result = "annotations.json" if independent_active else "reviews.json"
                    outcomes, labels, availability = self._dataset(source)
                    case = Path(source.get("source_path", str(root))).stem
                    record_id = hashlib.sha256(str(root).encode()).hexdigest()[:16]
                    revision = {"selection_run": str(root), "source_sha256": source.get("source_sha256"),
                                "video_sha256": source.get("video_sha256"),
                                "source_manifest_sha256": _digest(root / "source/source.json"),
                                "selection_sha256": _digest(root / "selection.json"),
                                "qwen_run": str(annotation_run[0]) if annotation_run else None,
                                "qwen_sha256": _digest(annotation_run[0] / "annotations.json") if annotation_run else None,
                                "qwen_draft_sha256": review_run[1].get("input_sha256", {}).get("qwen-drafts.json") if review_run else None,
                                "medgemma_run": str(review_run[0]) if review_run else None,
                                "medgemma_sha256": _digest(review_run[0] / medgemma_result) if review_run else None,
                                "outcomes": outcomes,
                                "source_annotation_sha256": sorted({row["source_locator"]["sha256"]
                                    for rows in labels.values() for row in rows})}
                    review_identity = hashlib.sha256(json.dumps(revision, sort_keys=True,
                                                    ensure_ascii=False).encode()).hexdigest()
                    summary = {"id": record_id, "case_id": case, "title": case,
                               "frame_count": len(frames), "duration_ms": source["duration_ms"],
                               "selected_count": sum(row["status"] == "selected" for row in frames),
                               "dropped_count": sum(row["status"] == "dropped" for row in frames),
                               "qwen_annotation_count": len(qwen),
                               "medgemma_annotation_count": len(medgemma) if independent_active else 0,
                               "medgemma_review_count": 0 if independent_active else len(medgemma),
                               "medgemma_protocol": review_run[1].get("schema_version") if review_run else None,
                               "status": selection.get("status", "prepared"),
                               "timestamp_basis": source.get("timestamp_basis", frames[0].get("timestamp_basis")),
                               "run_name": root.name,
                               "is_synthetic": bool(re.search(r"\bsynthetic\b", plan.get("config", {}).get("procedure_context", ""), re.I))}
                    metadata = {key: value for key, value in source.items() if key not in {"frames", "ffprobe", "commands"}}
                    metadata.update(case_id=case, procedure_context=plan.get("config", {}).get("procedure_context", ""))
                    source_video_url = self._register(source.get("video_path"), "video")
                    preview_url = self._preview_url(source, source_video_url)
                    metadata["browser_preview"] = {"status": "on_demand" if preview_url else "unavailable",
                        "profile": _PREVIEW_PROFILE, "source_video_sha256": source.get("video_sha256"),
                        "codec": "h264", "pixel_format": "yuv420p", "crf": 18, "audio": False,
                        "timing": "Every source frame and relative playback timestamp is verified; no frame resampling.",
                        "display_only": True, "source_modified": False,
                        "note": "Lossy browser display preview. Canonical source stills remain the image evidence."}
                    artifacts = self._artifacts(root, [("Selection", "selection.json"),
                        ("Source manifest", "source/source.json"), ("Selection run", "run.json")])
                    if annotation_run:
                        artifacts += self._artifacts(annotation_run[0], [("Qwen annotations", "annotations.json"),
                            ("Qwen raw response", "round-00/response.json")])
                    if independent_active:
                        artifacts += self._artifacts(review_run[0], [("Independent MedGemma annotations", "annotations.json"),
                            ("MedGemma source inventory", "source.json"), ("MedGemma selected targets", "selected-frames.json")])
                    elif review_run:
                        artifacts += self._artifacts(review_run[0], [("MedGemma reviews", "reviews.json"),
                            ("Qwen timestamp validation issues", "draft-intake.json"),
                            ("Preserved Qwen drafts with validation flags", "qwen-drafts.json"),
                            ("Deferred evidence requests", "deferred-evidence.json")])
                    public = {**summary, "review_identity": review_identity, "frames": frames,
                              "video_url": preview_url, "source_video_url": source_video_url,
                              "timeline_note": source.get("timeline_note", "Playback offsets refer to the supplied video."),
                              "outcomes": outcomes, "metadata": metadata, "artifacts": artifacts,
                              "runs": {"selection": self._run_info((root, plan), "selection.json"),
                                       "qwen": self._run_info(annotation_run, "annotations.json"),
                                       "medgemma": self._run_info(review_run, medgemma_result),
                                       "qwen_history": [self._run_info(pair, "annotations.json") for pair in annotation_runs],
                                       "medgemma_history": [self._run_info(pair, medgemma_result) for pair in review_runs],
                                       "historical_medgemma_reviews": [self._run_info(pair, "reviews.json") for pair in historical_review_runs]}}
                    records[record_id] = {"summary": summary, "public": public, "canonical": source["frames"],
                        "selection": selection, "decisions": decisions, "qwen": qwen, "medgemma": medgemma,
                        "evidence": evidence, "labels": labels, "availability": availability,
                        "review_run": review_run, "independent_active": independent_active}
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    self._warn(f"Could not load incomplete run {root}: {exc}")
            self._records = records

    def records(self):
        self.refresh()
        with self._lock:
            return copy.deepcopy({"records": [item["summary"] for item in self._records.values()],
                                  "warnings": self._warnings})

    def record(self, record_id):
        with self._lock:
            item = self._records.get(record_id)
            if item is None:
                return None
            result = copy.deepcopy(item["public"])
            for frame in result["frames"]:
                frame["curation"] = self._curation(item, frame["frame_id"])
            return result

    def _curation_path(self, item, frame_id):
        identity = item["public"]["review_identity"]
        frame_token = hashlib.sha256(frame_id.encode()).hexdigest()
        return self.curation_root / identity / (frame_token + ".json")

    def _curation(self, item, frame_id, *, strict=False):
        path = self._curation_path(item, frame_id)
        unsafe = any(part.is_symlink() for part in (self.curation_root, path.parent, path))
        if not unsafe and not path.exists():
            return None
        value = None if unsafe else _read(path)
        valid = (isinstance(value, dict) and value.get("schema_version") == _CURATION_SCHEMA
                 and value.get("review_identity") == item["public"]["review_identity"]
                 and value.get("record_id") == item["public"]["id"] and value.get("frame_id") == frame_id
                 and type(value.get("deleted")) is bool and isinstance(value.get("annotations"), dict)
                 and set(value["annotations"]) <= {"qwen", "medgemma"}
                 and all(isinstance(text, str) and len(text) <= _CURATION_TEXT_LIMIT
                         for text in value["annotations"].values())
                 and value.get("worksheet_status") == "draft" and value.get("training_eligible") is False
                 and isinstance(value.get("provenance"), dict) and isinstance(value.get("history"), list)
                 and isinstance(value.get("updated_at"), str))
        if not valid:
            if strict:
                raise CurationError("Saved human curation is invalid or unavailable; it was not overwritten.", 409)
            self._warn(f"Could not load human curation for frame {frame_id}: {path}")
            return None
        return copy.deepcopy(value)

    @staticmethod
    def _write_curation(path, value):
        """Commit one complete overlay atomically; a failed write leaves the old file intact."""
        descriptor, temporary = tempfile.mkstemp(prefix=".curation-", suffix=".json", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def curate(self, record_id, frame_id, request):
        """Save a draft overlay for this exact displayed revision, never a model artifact."""
        if not isinstance(request, dict) or set(request) - {"review_identity", "action", "annotations", "expected_updated_at"}:
            raise CurationError("Expected review_identity, action, optional annotations, and optional expected_updated_at only.")
        if ("expected_updated_at" in request and request["expected_updated_at"] is not None
                and not isinstance(request["expected_updated_at"], str)):
            raise CurationError("expected_updated_at must be a timestamp string or null.")
        identity, action = request.get("review_identity"), request.get("action")
        if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{64}", identity):
            raise CurationError("A valid review_identity is required.")
        if not isinstance(action, str) or action not in {"edit", "delete", "restore"}:
            raise CurationError("Action must be edit, delete, or restore.")
        annotations = request.get("annotations")
        if action == "edit":
            if (not isinstance(annotations, dict) or not annotations
                    or set(annotations) - {"qwen", "medgemma"}
                    or not all(isinstance(text, str) and len(text) <= _CURATION_TEXT_LIMIT
                               for text in annotations.values())):
                raise CurationError("Provide Qwen or MedGemma annotation text, up to 100,000 characters each.")
        elif "annotations" in request:
            raise CurationError("Only edit actions may include annotation text.")

        with self._lock:
            # Read the source revision again before accepting a write; a model
            # run may have advanced since the browser loaded the record.
            self.refresh()
            item = self._records.get(record_id)
            if item is None or not any(frame["frame_id"] == frame_id for frame in item["canonical"]):
                raise CurationError("Record or frame not found.", 404)
            if identity != item["public"]["review_identity"]:
                raise CurationError("The dataset revision changed. Refresh before editing this frame.", 409)
            available = {name for name in ("qwen", "medgemma") if frame_id in item[name]}
            if not available:
                raise CurationError("This frame has no AI annotation to edit or delete.", 409)
            if annotations is not None and not set(annotations) <= available:
                raise CurationError("An annotation can only be edited after that model's result exists.", 409)
            path = self._curation_path(item, frame_id)
            if self.curation_root.is_symlink() or path.parent.is_symlink():
                raise CurationError("Human curation storage must be a local directory.", 409)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Serialize independent inspector processes as well as HTTP threads.
            descriptor = os.open(path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                saved = self._curation(item, frame_id, strict=True)
                if ("expected_updated_at" in request
                        and request["expected_updated_at"] != (saved["updated_at"] if saved else None)):
                    raise CurationError("Human curation changed in another window. Refresh and reopen this enhancement before saving.", 409)
                now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
                if saved is None:
                    canonical = next(frame for frame in item["canonical"] if frame["frame_id"] == frame_id)
                    review = item["medgemma"].get(frame_id)
                    saved = {"schema_version": _CURATION_SCHEMA, "review_identity": identity,
                        "record_id": record_id, "frame_id": frame_id, "deleted": False, "annotations": {},
                        "worksheet_status": "draft", "training_eligible": False,
                        "created_at": now, "history": [], "provenance": {
                            "source_frame": copy.deepcopy(canonical),
                            "runs": {name: item["public"]["runs"][name]["path"] if item["public"]["runs"][name]
                                     else None for name in ("selection", "qwen", "medgemma")},
                            "original_annotations": {"qwen": copy.deepcopy(item["qwen"].get(frame_id)),
                                "medgemma": copy.deepcopy(review.get("annotation", review.get("medgemma_review"))) if review else None}}}
                if action == "edit":
                    if saved["deleted"]:
                        raise CurationError("Restore this enhancement before editing it.", 409)
                    saved["annotations"].update(annotations)
                else:
                    saved["deleted"] = action == "delete"
                saved["updated_at"] = now
                event = {"action": action, "updated_at": now}
                if action == "edit":
                    event["annotations"] = copy.deepcopy(annotations)
                saved["history"].append(event)
                self._write_curation(path, saved)
                return copy.deepcopy(saved)
            finally:
                os.close(descriptor)

    def frame(self, record_id, frame_id):
        with self._lock:
            item = self._records.get(record_id)
            if item is None:
                return None
            target = next((frame for frame in item["public"]["frames"] if frame["frame_id"] == frame_id), None)
            if target is None:
                return None
            review = item["medgemma"].get(frame_id)
            packet = item["evidence"].get(frame_id)
            evidence = []
            if packet:
                canonical_frames = {frame["frame_id"]: frame for frame in packet["frames"]}
                for frame in packet.get("views", packet["frames"]):
                    source_frame = canonical_frames.get(frame.get("frame_id"), {})
                    evidence.append({**source_frame, **frame,
                                     "roles": [frame["role"]] if frame.get("role") else frame.get("evidence_roles", []),
                                     "image_url": self._register(frame.get("image_path"), "image")})
            artifacts = []
            if review and item["review_run"] and isinstance(review.get("call_directory"), str):
                call = review["call_directory"]
                artifacts = self._artifacts(item["review_run"][0], [
                    ("Exact MedGemma request", call + "/request.json"),
                    ("Complete MedGemma response", call + "/response.json"),
                    ("MedGemma annotation", call + "/annotation.json"),
                    ("Historical MedGemma review", call + "/review.json"), ("MedGemma model output", call + "/model.json")])
            canonical = item["canonical"][target["frame_index"]]
            curation = self._curation(item, frame_id)
            return copy.deepcopy({"frame": {**target, "curation": curation}, "qwen": item["qwen"].get(frame_id),
                "curation": curation,
                "medgemma": review.get("annotation", review.get("medgemma_review")) if review else None,
                "medgemma_protocol": item["public"].get("medgemma_protocol"), "evidence": evidence,
                "original_annotations": item["labels"].get(frame_id, []),
                "label_availability": item["availability"].get(frame_id, []),
                "outcomes": item["public"]["outcomes"], "provenance": canonical,
                "dataset_context": packet.get("dataset_context") if packet else None,
                "evidence_context": {key: value for key, value in packet.items()
                                     if key not in {"frames", "views", "qwen_annotation", "dataset_context"}} if packet else None,
                "raw": {"selection": item["decisions"].get(frame_id), "qwen": item["qwen"].get(frame_id),
                        "medgemma": review}, "artifacts": artifacts})


def make_server(store, port=8765):
    """Bind exclusively to loopback; writes are limited to separate curation overlays."""
    static_root = Path(__file__).with_name("review_ui")

    class Handler(BaseHTTPRequestHandler):
        server_version = "YasargilInspector/1"

        def log_message(self, format, *args):
            pass

        def _headers(self, status, content_type, length, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Vary", "Sec-Fetch-Site, Sec-Fetch-Mode, Sec-Fetch-Dest, Origin")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; "
                             "style-src 'self' 'unsafe-inline'; script-src 'self'; object-src 'none'; frame-ancestors 'none'")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()

        def _json(self, value, status=200, head=False):
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self._headers(status, "application/json; charset=utf-8", len(raw))
            if not head:
                self.wfile.write(raw)

        def _file(self, path, head=False):
            try:
                stream = path.open("rb")
            except OSError:
                return self._json({"error": "File unavailable"}, 404, head)
            with stream:
                size = os.fstat(stream.fileno()).st_size
                start, end, status = 0, size - 1, 200
                range_header = self.headers.get("Range")
                if range_header:
                    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
                    if not match or not any(match.groups()) or size == 0:
                        self._headers(416, "application/octet-stream", 0, {"Content-Range": f"bytes */{size}"})
                        return
                    first, last = match.groups()
                    if first:
                        start = int(first)
                        end = min(int(last), size - 1) if last else size - 1
                    else:
                        suffix = int(last)
                        start, end = max(0, size - suffix), size - 1
                        if suffix == 0:
                            start = size
                    if start >= size or start > end:
                        self._headers(416, "application/octet-stream", 0, {"Content-Range": f"bytes */{size}"})
                        return
                    status = 206
                extra = {"Accept-Ranges": "bytes"}
                if status == 206:
                    extra["Content-Range"] = f"bytes {start}-{end}/{size}"
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self._headers(status, content_type, max(0, end - start + 1), extra)
                if not head:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining > 0:
                        block = stream.read(min(65536, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)

        def _local_request(self, head=False, *, allow_home_navigation=False):
            # Reject browser requests aimed at this port via a foreign DNS name.
            expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in expected:
                self._json({"error": "Local host required"}, 403, head)
                return False
            # Links from another app/site are cross-site requests too. Permit
            # top-level navigation to the static GUI shell, which contains no
            # dataset data. APIs, media, embedded documents, and writes still
            # require the origin checks below.
            if (allow_home_navigation and self.command in {"GET", "HEAD"}
                    and self.headers.get("Sec-Fetch-Mode") == "navigate"
                    and self.headers.get("Sec-Fetch-Dest") == "document"):
                return True
            origin = self.headers.get("Origin")
            if (self.headers.get("Sec-Fetch-Site") == "cross-site"
                    or origin is not None and origin != "http://" + self.headers["Host"]):
                self._json({"error": "Same-origin access required"}, 403, head)
                return False
            return True

        def _serve(self, head=False):
            path = unquote(urlsplit(self.path).path)
            if not self._local_request(head, allow_home_navigation=path == "/"):
                return
            if path == "/api/records":
                return self._json(store.records(), head=head)
            match = re.fullmatch(r"/api/records/([a-f0-9]{16})(?:/frames/([^/]+))?", path)
            if match:
                record_id, frame_id = match.groups()
                result = store.frame(record_id, frame_id) if frame_id is not None else store.record(record_id)
                return self._json(result if result is not None else {"error": "Record or frame not found"},
                                  200 if result is not None else 404, head)
            match = re.fullmatch(r"/api/records/([a-f0-9]{16})/frames/([^/]+)/context-video", path)
            if match:
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                if set(query) - {"revision"} or len(query.get("revision", [])) > 1:
                    return self._json({"error": "Invalid context-video request"}, 400, head)
                try:
                    preview = store.context_preview(*match.groups(), review_identity=query.get("revision", [None])[0])
                except PreviewError as exc:
                    return self._json({"error": str(exc), "fallback": "canonical_source_stills"}, 503, head)
                if preview:
                    return self._file(preview, head)
                return self._json({"error": "Record or frame not found"}, 404, head)
            match = re.fullmatch(r"/media/([a-f0-9]{32})", path)
            if match:
                media = store.media(match.group(1))
                if media:
                    return self._file(media, head)
            match = re.fullmatch(r"/preview/([a-f0-9]{32})", path)
            if match:
                try:
                    preview = store.preview(match.group(1))
                except PreviewError as exc:
                    return self._json({"error": str(exc), "fallback": "canonical_source_stills"}, 503, head)
                if preview:
                    return self._file(preview, head)
            static = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            if path in static and (static_root / static[path]).is_file():
                return self._file(static_root / static[path], head)
            self._json({"error": "Not found"}, 404, head)

        def do_GET(self):
            try:
                self._serve()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_HEAD(self):
            self._serve(head=True)

        def do_POST(self):
            if not self._local_request():
                return
            path = unquote(urlsplit(self.path).path)
            if path == "/api/open-dataset-folder":
                if (self.headers.get("Content-Length") != "0" or self.headers.get("Transfer-Encoding")
                        or urlsplit(self.path).query):
                    return self._json({"error": "This action does not accept a path or request body"}, 400)
                try:
                    return self._json(store.open_dataset_folder())
                except (OSError, subprocess.SubprocessError):
                    return self._json({"error": "Could not open the dataset output folder"}, 503)
            curation_match = re.fullmatch(r"/api/records/([a-f0-9]{16})/frames/([^/]+)/curation", path)
            if path != "/api/review-export" and not curation_match:
                return self._json({"error": "Unsupported inspector action"}, 405)
            length = self.headers.get("Content-Length", "")
            if not re.fullmatch(r"\d+", length) or self.headers.get("Transfer-Encoding"):
                return self._json({"error": "A valid Content-Length is required"}, 411)
            if int(length) > 2 * 1024 * 1024:
                return self._json({"error": "Review worksheet exceeds the 2 MB limit"}, 413)
            content_type = "application/json" if curation_match else "application/x-www-form-urlencoded"
            if self.headers.get_content_type() != content_type:
                return self._json({"error": f"Expected {content_type}"}, 415)

            def strict_pairs(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("Duplicate JSON key")
                    result[key] = value
                return result

            def invalid_constant(value):
                raise ValueError("Non-finite JSON number")

            try:
                self.connection.settimeout(10)
                raw = self.rfile.read(int(length))
                if len(raw) != int(length):
                    raise ValueError("Incomplete request")
                if curation_match:
                    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_pairs,
                                         parse_constant=invalid_constant)
                    # Reject unpaired Unicode surrogates before attempting a
                    # durable write or rendering a response.
                    json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    try:
                        result = store.curate(*curation_match.groups(), payload)
                    except CurationError as exc:
                        return self._json({"error": str(exc)}, exc.status)
                    except OSError:
                        return self._json({"error": "Human curation could not be saved. Source artifacts are unchanged."}, 500)
                    return self._json(result)
                # Stateless download echo; exporting itself never writes files.
                form = parse_qs(raw.decode("utf-8"), keep_blank_values=True, strict_parsing=True,
                                errors="strict", max_num_fields=2)
                if set(form) != {"review"} or len(form["review"]) != 1:
                    raise ValueError("Expected one review field")
                worksheet = json.loads(form["review"][0], object_pairs_hook=strict_pairs,
                                       parse_constant=invalid_constant)
                if (not isinstance(worksheet, dict)
                        or worksheet.get("schema_version") != "yasargil-inspector-review-v1"
                        or not isinstance(worksheet.get("notes"), list)
                        or not all(isinstance(note, dict) for note in worksheet["notes"])
                        or not isinstance(worksheet.get("case_id", "record"), str)):
                    raise ValueError("Invalid worksheet")
                worksheet.update(worksheet_status="draft", training_eligible=False)
                payload = (json.dumps(worksheet, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            except (ValueError, UnicodeError, RecursionError, OSError):
                return self._json({"error": "Invalid curation request" if curation_match else "Invalid review worksheet"}, 400)
            filename = re.sub(r"[^A-Za-z0-9._-]+", "-", worksheet.get("case_id", "record")).strip("._-")[:80] or "record"
            self._headers(200, "application/json; charset=utf-8", len(payload),
                          {"Content-Disposition": f'attachment; filename="{filename}-human-review.json"'})
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def add_inspector_parser(subparsers):
    parser = subparsers.add_parser("inspect-dataset", help="Open a local surgery dataset inspector with draft human curation")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs"), help="Directory containing saved pipeline runs")
    parser.add_argument("--dataset-root", type=Path, help="Optional exact SOSpine dataset root for source labels and outcomes")
    parser.add_argument("--port", type=int, default=8765, help="Local HTTP port (default: 8765)")
    return parser


def inspector_cli(args):
    if not 0 <= args.port <= 65535:
        from .contract import ContractError
        raise ContractError("Port must be between 0 and 65535")
    store = InspectorStore(args.runs_root, args.dataset_root)
    server = make_server(store, args.port)
    print(f"Dataset inspector: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Reading saved runs from {store.runs_root}. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
