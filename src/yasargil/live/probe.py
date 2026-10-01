"""A learned perception source: frozen DINOv2 patch features with linear heads.

One encoder pass per frame feeds two heads. A presence head reads the CLS token
and mean patch feature and says which instruments and structures are in view.
A 1x1 patch head says where: instrument tips (one patch, decoded as a weighted
centroid) and the durotomy region (a patch mask, decoded as a box). The grid
keeps the frame's aspect ratio, so 1920x1080 becomes 28x16 patches of 14 px at
392x224; localization is therefore patch-coarse (about 69 px at full size).

Training splits by surgeon so no surgeon appears in two partitions. The encoder
identity is stored with the heads and checked at load time. Requires the
optional ``selection`` dependencies (numpy, torch, transformers, safetensors).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image

from . import LiveError
from .frames import case_frames
from .labels import ANATOMY, INSTRUMENTS, CaseLabels
from .perception import Detection, LabelPerception, Observation

CLASSES = tuple(INSTRUMENTS + ANATOMY)
DEFAULT_ENCODER = Path(".runtime/models/dinov2-small")


def probe_grid(width, height, side=224, patch=14):
    """(cols, rows): the short side spans ``side`` pixels, the long side keeps the aspect ratio."""
    short = side // patch
    if width >= height:
        return max(1, round(short * width / height)), short
    return short, max(1, round(short * height / width))


def targets_for(observation, cols, rows):
    presence = np.zeros(len(CLASSES), dtype=np.float32)
    maps = np.zeros((len(CLASSES), rows, cols), dtype=np.float32)
    for detection in observation.detections:
        if detection.label not in CLASSES:
            continue
        k = CLASSES.index(detection.label)
        presence[k] = 1.0
        if detection.kind == "instrument":
            point = detection.tip or (_center(detection.box) if detection.box else None)
            if point is not None:
                maps[k][min(int(point[1] * rows), rows - 1), min(int(point[0] * cols), cols - 1)] = 1.0
        elif detection.box:
            x1, y1, x2, y2 = detection.box
            for r in range(rows):
                for c in range(cols):
                    if x1 <= (c + 0.5) / cols <= x2 and y1 <= (r + 0.5) / rows <= y2:
                        maps[k][r, c] = 1.0
            if not maps[k].any():
                cx, cy = _center(detection.box)
                maps[k][min(int(cy * rows), rows - 1), min(int(cx * cols), cols - 1)] = 1.0
        elif detection.tip:
            maps[k][min(int(detection.tip[1] * rows), rows - 1), min(int(detection.tip[0] * cols), cols - 1)] = 1.0
    return presence, maps


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def decode(presence, maps, threshold=0.5):
    """Detections from presence probabilities [C] and patch probabilities [C, rows, cols]."""
    thresholds = threshold if isinstance(threshold, (list, tuple, np.ndarray)) else [threshold] * len(CLASSES)
    detections = []
    for k, label in enumerate(CLASSES):
        if presence[k] < thresholds[k]:
            continue
        grid = maps[k]
        rows, cols = grid.shape
        if label in INSTRUMENTS:
            weights = np.where(grid >= 0.5 * grid.max(), grid, 0.0)
            total = float(weights.sum())
            if total > 0:
                rr, cc = np.mgrid[0:rows, 0:cols]
                x = float((weights * (cc + 0.5)).sum() / total / cols)
                y = float((weights * (rr + 0.5)).sum() / total / rows)
                tip = (round(x, 4), round(y, 4))
            else:
                tip = None
            detections.append(Detection(label, "instrument", float(presence[k]), tip, None))
        else:
            mask = grid >= 0.5
            if not mask.any():
                mask = grid == grid.max()
            rr, cc = np.nonzero(mask)
            box = (float(cc.min()) / cols, float(rr.min()) / rows, float(cc.max() + 1) / cols, float(rr.max() + 1) / rows)
            detections.append(Detection(label, "anatomy", float(presence[k]), None, box))
    return detections


def surgeon_split(cases, train, val, test):
    groups = {"train": set(train), "val": set(val), "test": set(test)}
    if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
        raise ValueError("A surgeon may appear in only one partition")
    split = {"train": [], "val": [], "test": []}
    for case in sorted(cases):
        match = re.fullmatch(r"(S\d+)A\d+", case)
        if not match:
            continue
        for name, surgeons in groups.items():
            if match[1] in surgeons:
                split[name].append(case)
    return split


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise LiveError("The DINO probe needs the optional 'selection' dependencies (uv sync --extra selection).") from exc
    return torch


class ProbeModel:
    """Linear presence and patch heads over standardized encoder features."""

    def __init__(self, dim, classes=CLASSES, mean=None, std=None, thresholds=None, metadata=None, state=None):
        torch = _torch()
        self.dim = dim
        self.classes = list(classes)
        self.mean = np.zeros(dim, dtype=np.float32) if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = np.ones(dim, dtype=np.float32) if std is None else np.asarray(std, dtype=np.float32)
        self.thresholds = list(thresholds or [0.5] * len(self.classes))
        self.metadata = dict(metadata or {})
        self.presence = torch.nn.Linear(2 * dim, len(self.classes))
        self.patch = torch.nn.Linear(dim, len(self.classes))
        if state is not None:
            self.presence.load_state_dict({k[len("presence."):]: v for k, v in state.items() if k.startswith("presence.")})
            self.patch.load_state_dict({k[len("patch."):]: v for k, v in state.items() if k.startswith("patch.")})

    def parameters(self):
        return list(self.presence.parameters()) + list(self.patch.parameters())

    def logits(self, cls, patches):
        torch = _torch()
        mean, std = torch.from_numpy(self.mean), torch.from_numpy(self.std)
        cls = (cls - mean) / std
        patches = (patches - mean) / std
        pooled = patches.mean(dim=(1, 2))
        presence = self.presence(torch.cat([cls, pooled], dim=1))
        maps = self.patch(patches).permute(0, 3, 1, 2)
        return presence, maps

    def predict(self, cls, patches):
        torch = _torch()
        with torch.no_grad():
            presence, maps = self.logits(torch.as_tensor(np.asarray(cls, dtype=np.float32)),
                                         torch.as_tensor(np.asarray(patches, dtype=np.float32)))
            return torch.sigmoid(presence).numpy(), torch.sigmoid(maps).numpy()

    def save(self, directory):
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = {f"presence.{k}": v.contiguous() for k, v in self.presence.state_dict().items()}
        state.update({f"patch.{k}": v.contiguous() for k, v in self.patch.state_dict().items()})
        save_file(state, str(directory / "heads.safetensors"))
        (directory / "probe.json").write_text(json.dumps({
            "format": "yasargil-dino-probe-v1", "dim": self.dim, "classes": self.classes,
            "mean": self.mean.tolist(), "std": self.std.tolist(), "thresholds": self.thresholds,
            "heads_sha256": hashlib.sha256((directory / "heads.safetensors").read_bytes()).hexdigest(),
            **self.metadata}, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        try:
            meta = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
            heads = directory / "heads.safetensors"
            if hashlib.sha256(heads.read_bytes()).hexdigest() != meta["heads_sha256"]:
                raise LiveError(f"Probe heads changed since training: {heads}")
            state = load_file(str(heads))
        except (OSError, ValueError, KeyError) as exc:
            raise LiveError(f"Cannot load DINO probe from {directory}: {exc}") from exc
        extra = {k: v for k, v in meta.items() if k not in ("format", "dim", "classes", "mean", "std", "thresholds",
                                                            "heads_sha256")}
        return cls(meta["dim"], meta["classes"], meta["mean"], meta["std"], meta["thresholds"], extra, state)


def train_heads(cls, patches, presence, maps, *, epochs=40, lr=0.01, seed=0, batch_size=None, metadata=None):
    """Fit both heads; patch loss counts only classes present in a frame. Returns (model, loss history)."""
    torch = _torch()
    torch.manual_seed(seed)
    cls = np.asarray(cls, dtype=np.float32)
    patches = np.asarray(patches)
    dim = cls.shape[1]
    flat = patches.reshape(-1, dim).astype(np.float32)
    mean = np.concatenate([cls, flat]).mean(axis=0)
    std = np.concatenate([cls, flat]).std(axis=0) + 1e-6
    model = ProbeModel(dim, CLASSES, mean, std, metadata=metadata)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    presence_t = torch.as_tensor(np.asarray(presence, dtype=np.float32))
    maps_t = torch.as_tensor(np.asarray(maps, dtype=np.float32))
    positives = maps_t.sum(dim=(0, 2, 3))
    cells = presence_t.sum(dim=0) * maps_t.shape[2] * maps_t.shape[3]
    pos_weight = ((cells - positives) / positives.clamp(min=1)).clamp(1, 200)
    count = len(cls)
    size = batch_size or count
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(epochs):
        order = rng.permutation(count)
        total = 0.0
        for start in range(0, count, size):
            index = order[start:start + size]
            cls_b = torch.as_tensor(cls[index])
            patches_b = torch.as_tensor(patches[index].astype(np.float32))
            presence_logits, map_logits = model.logits(cls_b, patches_b)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(presence_logits, presence_t[index])
            weight = presence_t[index][:, :, None, None].expand_as(map_logits)
            if weight.sum() > 0:
                per_cell = torch.nn.functional.binary_cross_entropy_with_logits(
                    map_logits, maps_t[index], pos_weight=pos_weight[None, :, None, None], reduction="none")
                loss = loss + (per_cell * weight).sum() / weight.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)
        history.append(total / count)
    return model, history


class DinoEncoder:
    """Frozen local DINOv2: CLS plus aspect-preserving patch features; never downloads."""

    def __init__(self, model_path=DEFAULT_ENCODER, side=224, device=None):
        torch = _torch()
        from transformers import AutoModel
        self.model_path = Path(model_path).expanduser().resolve()
        config_path = self.model_path / "config.json"
        processor_path = self.model_path / "preprocessor_config.json"
        weights = sorted(self.model_path.glob("*.safetensors"))
        if not config_path.is_file() or not processor_path.is_file() or not weights:
            raise LiveError(f"Incomplete local DINO checkpoint at {self.model_path}; see docs/SMART_FRAME_SELECTION.md.")
        processor = json.loads(processor_path.read_text())
        self.mean = np.asarray(processor.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.asarray(processor.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(str(self.model_path), local_files_only=True, use_safetensors=True,
                                               trust_remote_code=False).eval().to(self.device)
        patch = self.model.config.patch_size
        self.patch = patch[0] if isinstance(patch, (list, tuple)) else patch
        self.registers = getattr(self.model.config, "num_register_tokens", 0)
        self.side = side
        self.dim = self.model.config.hidden_size
        self._identity = {"kind": "dinov2", "model_path": str(self.model_path), "side": side, "patch": self.patch,
                          "weights": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in weights}}

    def identity(self):
        return dict(self._identity)

    def grid(self, width, height):
        return probe_grid(width, height, self.side, self.patch)

    def encode(self, images):
        torch = _torch()
        cols, rows = self.grid(*images[0].size)
        batch = []
        for image in images:
            resized = image.convert("RGB").resize((cols * self.patch, rows * self.patch), Image.Resampling.BICUBIC)
            array = (np.asarray(resized, dtype=np.float32) / 255 - self.mean) / self.std
            batch.append(array.transpose(2, 0, 1))
        with torch.inference_mode():
            hidden = self.model(pixel_values=torch.as_tensor(np.stack(batch)).to(self.device)).last_hidden_state
            hidden = hidden.float().cpu().numpy()
        cls = hidden[:, 0]
        patches = hidden[:, 1 + self.registers:].reshape(len(images), rows, cols, -1)
        return cls, patches


def _portable(identity):
    """Encoder identity without its location: weights and preprocessing decide compatibility."""
    return {key: value for key, value in identity.items() if key != "model_path"}


class ProbePerception:
    producer = "dino-probe/v1"

    def __init__(self, model_dir, encoder=None):
        self.model_dir = Path(model_dir)
        self.model = ProbeModel.load(self.model_dir)
        stored = self.model.metadata.get("encoder", {})
        if encoder is None:
            encoder = DinoEncoder(stored.get("model_path", DEFAULT_ENCODER), stored.get("side", 224))
        if _portable(encoder.identity()) != _portable(stored):
            raise LiveError("The encoder differs from the one the probe was trained with.")
        self.encoder = encoder

    def identity(self):
        return {"kind": "dino-probe", "producer": self.producer, "model_dir": str(self.model_dir),
                "encoder": self.encoder.identity(), "thresholds": self.model.thresholds,
                "metrics": self.model.metadata.get("metrics", {}).get("test")}

    def _observe_image(self, image, frame, producer):
        cls, patches = self.encoder.encode([image])
        presence, maps = self.model.predict(cls, patches)
        return Observation(frame.index, frame.t_ms, tuple(decode(presence[0], maps[0], self.model.thresholds)), True,
                           producer)

    def observe(self, frame):
        with Image.open(frame.path) as image:
            return self._observe_image(image.convert("RGB"), frame, self.producer)

    def observe_crop(self, frame, crop):
        with Image.open(frame.path) as image:
            width, height = image.size
            x1, y1, x2, y2 = crop
            region = image.convert("RGB").crop((round(x1 * width), round(y1 * height),
                                                round(x2 * width), round(y2 * height)))
            return self._observe_image(region, frame, self.producer + "+crop")


def _case_features(dataset_root, case_id, encoder, frame_stride, progress):
    frames = case_frames(dataset_root, case_id)
    size = frames.size(frames.frames[0])
    labels = CaseLabels.load(dataset_root, case_id, size)
    perception = LabelPerception(labels)
    cols, rows = encoder.grid(*size)
    selected = [f for f in frames.frames if labels.annotated(f.index)][::frame_stride]
    cls_all, patch_all, presence_all, maps_all, observations = [], [], [], [], []
    for start in range(0, len(selected), 16):
        chunk = selected[start:start + 16]
        images = []
        for frame in chunk:
            with Image.open(frame.path) as image:
                images.append(image.convert("RGB"))
        cls, patches = encoder.encode(images)
        cls_all.append(cls.astype(np.float32))
        patch_all.append(patches.astype(np.float16))
        for frame in chunk:
            observation = perception.observe(frame)
            presence, maps = targets_for(observation, cols, rows)
            presence_all.append(presence)
            maps_all.append(maps)
            observations.append(observation)
    progress(f"{case_id}: encoded {len(selected)} frames at {cols}x{rows} patches")
    if not selected:
        return None
    return (np.concatenate(cls_all), np.concatenate(patch_all), np.stack(presence_all), np.stack(maps_all), observations)


def _metrics(model, data, thresholds, aspect):
    cls, patches, presence, maps, observations = data
    probs, map_probs = model.predict(cls, patches.astype(np.float32))
    result = {"frames": len(cls), "presence_f1": {}}
    for k, label in enumerate(CLASSES):
        predicted, actual = probs[:, k] >= thresholds[k], presence[:, k] > 0.5
        tp, fp, fn = int((predicted & actual).sum()), int((predicted & ~actual).sum()), int((~predicted & actual).sum())
        result["presence_f1"][label] = round(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else None
    scores = [v for v in result["presence_f1"].values() if v is not None]
    result["presence_f1_macro"] = round(sum(scores) / len(scores), 4) if scores else None
    tip_errors, ious = [], []
    for i, observation in enumerate(observations):
        detected = {d.label: d for d in decode(probs[i], map_probs[i], thresholds)}
        for truth in observation.detections:
            guess = detected.get(truth.label)
            if guess is None:
                continue
            if truth.kind == "instrument" and truth.tip and guess.tip:
                tip_errors.append(math.hypot((truth.tip[0] - guess.tip[0]) * aspect, truth.tip[1] - guess.tip[1]))
            if truth.kind == "anatomy" and truth.box and guess.box:
                ious.append(_iou(truth.box, guess.box))
    result["tip_error_height_units"] = round(float(np.mean(tip_errors)), 4) if tip_errors else None
    result["durotomy_iou"] = round(float(np.mean(ious)), 4) if ious else None
    return result


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _best_thresholds(model, data):
    if data is None:
        return [0.5] * len(CLASSES)
    probs, _ = model.predict(data[0], data[1].astype(np.float32))
    thresholds = []
    for k in range(len(CLASSES)):
        actual = data[2][:, k] > 0.5
        best, best_f1 = 0.5, -1.0
        for candidate in np.linspace(0.1, 0.9, 17):
            predicted = probs[:, k] >= candidate
            tp, fp, fn = (predicted & actual).sum(), (predicted & ~actual).sum(), (~predicted & actual).sum()
            f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else -1.0
            if f1 > best_f1:
                best, best_f1 = float(round(candidate, 2)), f1
        thresholds.append(best)
    return thresholds


def _join(parts):
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return tuple(np.concatenate([p[i] for p in parts]) if i < 4 else [o for p in parts for o in p[i]] for i in range(5))


def train_probe(dataset_root, output_dir, *, train, val, test, encoder=None, model_path=DEFAULT_ENCODER,
                frame_stride=1, epochs=40, lr=0.01, batch_size=256, seed=0, progress=print):
    output = Path(output_dir)
    if output.exists():
        raise LiveError(f"Refusing to overwrite an existing probe directory: {output}")
    frames_root = Path(dataset_root) / "frames"
    if not frames_root.is_dir():
        raise LiveError(f"No frames directory under {dataset_root}")
    split = surgeon_split([p.name for p in frames_root.iterdir() if p.is_dir()], train, val, test)
    if not split["train"]:
        raise LiveError("No training cases for the requested surgeons.")
    encoder = encoder or DinoEncoder(model_path)
    data = {name: _join([_case_features(dataset_root, case, encoder, frame_stride, progress) for case in cases])
            for name, cases in split.items()}
    if data["train"] is None:
        raise LiveError("Training cases have no annotated frames.")
    sample = case_frames(dataset_root, split["train"][0])
    width, height = sample.size(sample.frames[0])
    progress(f"training heads on {len(data['train'][0])} frames")
    model, history = train_heads(*data["train"][:4], epochs=epochs, lr=lr, seed=seed, batch_size=batch_size)
    model.thresholds = _best_thresholds(model, data["val"] or data["train"])
    metrics = {"split": split, "loss_history": [round(v, 5) for v in history],
               "thresholds": dict(zip(CLASSES, model.thresholds))}
    for name in ("train", "val", "test"):
        metrics[name] = _metrics(model, data[name], model.thresholds, width / height) if data[name] else {"frames": 0}
    model.metadata = {"encoder": encoder.identity(), "split": split, "frame_stride": frame_stride, "epochs": epochs,
                      "lr": lr, "seed": seed, "grid": list(encoder.grid(width, height)), "metrics": metrics}
    model.save(output)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
