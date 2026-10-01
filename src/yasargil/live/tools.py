"""The agent's tools: the event log first, pixels on demand.

Every tool sees the world as of the question: events after ``now_ms`` and frames
after ``frame_index`` do not exist. ``view_frame`` and ``perceive_frame`` record
which frames were actually inspected, which the citation check relies on.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

from jsonschema import Draft202012Validator

from . import LiveError
from .events import EVENT_TYPES
from .frames import render_jpeg

EVENT_ID = r"^E[0-9]{6}$"
RESULT_ID = r"^R[0-9]+$"
CROP = {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "minItems": 4, "maxItems": 4,
        "description": "Normalized [x1, y1, x2, y2] region of the frame, 0..1."}

SCHEMAS = {
    "query_events": {
        "description": "Search the event log (state changes detected so far). Returns one line per event, oldest first.",
        "parameters": {"type": "object", "additionalProperties": False, "properties": {
            "types": {"type": "array", "items": {"enum": list(EVENT_TYPES)}},
            "subject": {"type": "string", "description": "Instrument, structure, step or rule name."},
            "since_s": {"type": "number", "minimum": 0}, "until_s": {"type": "number", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50}}},
    },
    "current_state": {
        "description": "What is in view now: instruments, the durotomy, proximity flags, procedure step, latest detections.",
        "parameters": {"type": "object", "additionalProperties": False, "properties": {}},
    },
    "view_frame": {
        "description": "Look at one past or current frame (default: current), optionally cropped. Costs an image.",
        "parameters": {"type": "object", "additionalProperties": False, "properties": {
            "frame_index": {"type": "integer", "minimum": 1}, "t_s": {"type": "number", "minimum": 0},
            "crop": CROP, "reason": {"type": "string"}}},
    },
    "perceive_frame": {
        "description": "Run the perception model on one past or current frame, optionally cropped.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["frame_index"], "properties": {
            "frame_index": {"type": "integer", "minimum": 1}, "crop": CROP}},
    },
    "ask_specialist": {
        "description": "Ask the surgical specialist vision model one question about one frame.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["frame_index", "question"],
                       "properties": {"frame_index": {"type": "integer", "minimum": 1},
                                      "question": {"type": "string", "minLength": 1}}},
    },
    "final_answer": {
        "description": ("Finish. 'claims' must restate every factual statement in 'answer', each citing the event IDs, "
                        "inspected frame indices and/or result IDs (R0, R1, ...) that support it. Cite a result to "
                        "support that something has not happened or is not visible. Put the typed answer in 'value'."),
        "parameters": {"type": "object", "additionalProperties": False, "required": ["answer", "claims"], "properties": {
            "answer": {"type": "string"},
            "value": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "number"},
                                {"type": "boolean"}, {"type": "null"}]},
            "claims": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                       "required": ["text"], "properties": {
                           "text": {"type": "string"},
                           "event_ids": {"type": "array", "items": {"type": "string", "pattern": EVENT_ID}},
                           "frame_indices": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                           "result_ids": {"type": "array", "items": {"type": "string", "pattern": RESULT_ID}}}}},
            "unresolved": {"type": "string"}}},
    },
}
_VALIDATORS = {name: Draft202012Validator(schema["parameters"]) for name, schema in SCHEMAS.items()}


class ToolError(LiveError):
    """A tool call the agent can correct: bad arguments, the future, or a budget."""


@dataclass
class ToolContext:
    log: object
    frames: object
    perception: object
    now_ms: int
    frame_index: int
    latest: object = None
    specialist: object = None
    max_images: int = 3
    image_max_side: int = 768
    viewed_frames: set = field(default_factory=set)
    images_used: int = 0
    results: dict = field(default_factory=dict)
    result_counter: int = 0


@dataclass(frozen=True)
class ToolResult:
    text: str
    image_b64: str | None = None
    viewed_frame: int | None = None


def tool_schemas(specialist=False):
    names = ["query_events", "current_state", "view_frame", "perceive_frame"]
    if specialist:
        names.append("ask_specialist")
    names.append("final_answer")
    return [{"type": "function", "function": {"name": name, "description": SCHEMAS[name]["description"],
                                              "parameters": SCHEMAS[name]["parameters"]}} for name in names]


def validate_args(name, args):
    if name not in _VALIDATORS:
        raise ToolError(f"Unknown tool {name!r}. Use one of: {', '.join(SCHEMAS)}.")
    if not isinstance(args, dict):
        raise ToolError(f"Arguments for {name} must be a JSON object.")
    errors = sorted(_VALIDATORS[name].iter_errors(args), key=lambda error: list(error.path))
    if errors:
        raise ToolError(f"Invalid arguments for {name}: {errors[0].message}")


def run_tool(name, args, ctx):
    validate_args(name, args)
    if name == "final_answer":
        raise ToolError("final_answer is handled by the agent loop.")
    if name == "ask_specialist" and ctx.specialist is None:
        raise ToolError("ask_specialist is not available in this run.")
    return _HANDLERS[name](args, ctx)


def _seconds(ms):
    return f"{ms / 1000:.1f}s"


def _register(ctx, entry):
    """Give a tool result a citable ID; R0 is reserved for the agent's opening state."""
    ctx.result_counter += 1
    result_id = f"R{ctx.result_counter}"
    ctx.results[result_id] = entry
    return result_id


def _query_events(args, ctx):
    since = round(args["since_s"] * 1000) if "since_s" in args else None
    until = round(args["until_s"] * 1000) if "until_s" in args else None
    events = ctx.log.query(ctx.now_ms, types=args.get("types"), subject=args.get("subject"),
                           since_ms=since, until_ms=until, limit=args.get("limit", 20))
    result_id = _register(ctx, {"tool": "query_events", "subject": args.get("subject"), "types": args.get("types"),
                                "count": len(events)})
    if not events:
        return ToolResult(f"Result {result_id}: No matching events up to t={_seconds(ctx.now_ms)}.")
    header = f"Result {result_id}: events known at t={_seconds(ctx.now_ms)} (oldest first):"
    return ToolResult("\n".join([header] + [event.line() for event in events]))


def _current_state(args, ctx):
    result_id = _register(ctx, {"tool": "current_state", "at_ms": ctx.now_ms})
    return ToolResult(f"Result {result_id}: " + state_text(ctx))


def state_text(ctx):
    """What the event log and the latest detections say is in view now."""
    state = ctx.log.state_at(ctx.now_ms)
    lines = [f"Now: t={_seconds(ctx.now_ms)}, frame {ctx.frame_index}."]
    instruments = ", ".join(f"{label} ({event_id})" for label, event_id in sorted(state.visible_instruments.items()))
    lines.append(f"Instruments in view: {instruments or 'none'}.")
    anatomy = ", ".join(f"{label} ({event_id})" for label, event_id in sorted(state.visible_anatomy.items()))
    lines.append(f"Structures in view: {anatomy or 'none'}.")
    near = ", ".join(f"{label} tip near {target} ({event_id})" for label, (target, event_id) in sorted(state.near.items()))
    lines.append(f"Proximity: {near or 'nothing near a structure'}.")
    lines.append(f"Procedure step: {state.step[0]} ({state.step[1]})." if state.step else "Procedure step: not yet established.")
    if state.view_empty:
        lines.append(f"View is empty ({state.view_empty}).")
    if ctx.latest is not None:
        lines.append(f"Latest detections (frame {ctx.latest.frame_index}):")
        lines.extend(_detection_lines(ctx.latest) or ["  no detections"])
    return "\n".join(lines)


def _detection_lines(observation):
    lines = []
    for detection in observation.detections:
        where = []
        if detection.tip:
            where.append(f"tip=({detection.tip[0]:.2f},{detection.tip[1]:.2f})")
        if detection.box:
            where.append("box=(" + ",".join(f"{value:.2f}" for value in detection.box) + ")")
        lines.append(f"  {detection.label} conf={detection.confidence:.2f} {' '.join(where)}".rstrip())
    return lines


def _resolve_frame(args, ctx):
    if "frame_index" in args and "t_s" in args:
        raise ToolError("Give frame_index or t_s, not both.")
    if "t_s" in args:
        t_ms = round(args["t_s"] * 1000)
        if t_ms > ctx.now_ms:
            raise ToolError(f"t={args['t_s']}s is in the future; now is t={_seconds(ctx.now_ms)}.")
        frame = ctx.frames.at_or_before(t_ms)
    else:
        index = args.get("frame_index", ctx.frame_index)
        if index > ctx.frame_index:
            raise ToolError(f"Frame {index} is in the future; the current frame is {ctx.frame_index}.")
        frame = ctx.frames.get(index)
    if frame is None:
        raise ToolError("That frame was not released; choose another index.")
    return frame


def _view_frame(args, ctx):
    frame = _resolve_frame(args, ctx)
    if ctx.images_used >= ctx.max_images:
        raise ToolError(f"Image budget exhausted ({ctx.max_images} per question); answer from the events.")
    crop = args.get("crop")
    image = render_jpeg(frame.path, ctx.image_max_side, crop)
    ctx.images_used += 1
    ctx.viewed_frames.add(frame.index)
    region = f", crop {crop}" if crop else ""
    return ToolResult(f"Frame {frame.index} (t={_seconds(frame.t_ms)}{region}) is attached in the next message.",
                      base64.b64encode(image).decode("ascii"), frame.index)


def _perceive_frame(args, ctx):
    frame = _resolve_frame(args, ctx)
    crop = args.get("crop")
    observation = ctx.perception.observe_crop(frame, crop) if crop else ctx.perception.observe(frame)
    ctx.viewed_frames.add(frame.index)
    region = f" crop {crop} (coordinates relative to the crop)" if crop else ""
    lines = [f"Perception on frame {frame.index} (t={_seconds(frame.t_ms)}){region}:"]
    return ToolResult("\n".join(lines + (_detection_lines(observation) or ["  no detections"])),
                      viewed_frame=frame.index)


def _ask_specialist(args, ctx):
    frame = _resolve_frame(args, ctx)
    image = base64.b64encode(render_jpeg(frame.path, ctx.image_max_side)).decode("ascii")
    answer = ctx.specialist(image, args["question"])
    ctx.viewed_frames.add(frame.index)
    return ToolResult(f"Specialist on frame {frame.index} (t={_seconds(frame.t_ms)}): {answer}",
                      viewed_frame=frame.index)


_HANDLERS = {"query_events": _query_events, "current_state": _current_state, "view_frame": _view_frame,
             "perceive_frame": _perceive_frame, "ask_specialist": _ask_specialist}
