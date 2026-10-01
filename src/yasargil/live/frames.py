"""Released frame discovery and image rendering for replay.

SOSpine releases 1-fps stills named ``<case>_frame_########.jpeg``. A frame's
time is derived from its release index; missing indices stay missing rather
than being renumbered, so gaps remain visible to later stages.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
import io
from pathlib import Path
import re

from PIL import Image

from . import LiveError

FRAME_NAME = re.compile(r"^(?P<case>.+)_frame_(?P<index>\d{8})\.jpe?g$", re.IGNORECASE)


@dataclass(frozen=True)
class Frame:
    index: int
    t_ms: int
    path: Path


class FrameSource:
    """Ordered frames of one case, with timestamps from release indices."""

    def __init__(self, directory, case_id=None, fps=1.0):
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise LiveError(f"Frame directory not found: {self.directory}")
        if not fps > 0:
            raise LiveError("Frame rate must be positive.")
        self.fps = float(fps)
        frames = []
        for path in self.directory.iterdir():
            match = FRAME_NAME.match(path.name)
            if not match or not path.is_file() or (case_id and match["case"] != case_id):
                continue
            index = int(match["index"])
            frames.append(Frame(index, round((index - 1) * 1000 / self.fps), path))
        if not frames:
            raise LiveError(f"No released frames in {self.directory}")
        self.frames = sorted(frames, key=lambda frame: frame.index)
        self.case_id = case_id or FRAME_NAME.match(self.frames[0].path.name)["case"]
        self._by_index = {frame.index: frame for frame in self.frames}
        self._times = [frame.t_ms for frame in self.frames]
        self._sizes = {}

    def get(self, index):
        return self._by_index.get(index)

    def at_or_before(self, t_ms):
        position = bisect.bisect_right(self._times, t_ms)
        return self.frames[position - 1] if position else None

    def size(self, frame):
        if frame.path not in self._sizes:
            with Image.open(frame.path) as image:
                self._sizes[frame.path] = image.size
        return self._sizes[frame.path]


def case_frames(dataset_root, case_id, fps=1.0):
    return FrameSource(Path(dataset_root) / "frames" / case_id, case_id, fps)


def render_jpeg(path, max_side=768, crop=None, quality=85):
    """JPEG bytes of an image, optionally cropped by a normalized box, then bounded."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        if crop is not None:
            x1, y1, x2, y2 = (min(max(float(value), 0.0), 1.0) for value in crop)
            if x2 <= x1 or y2 <= y1:
                raise LiveError("Crop box must have positive width and height.")
            width, height = image.size
            image = image.crop((round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)))
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                                 Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=quality)
        return buffer.getvalue()
