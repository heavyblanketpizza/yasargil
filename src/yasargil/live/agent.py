"""The deliberative loop: a local VLM that reads events and pulls pixels on demand.

Two action protocols share one loop, because local models can behave quite
differently under them: ``native`` sends OpenAI-style ``tools`` with
``tool_choice=required``; ``schema`` asks for one grammar-constrained JSON action
per turn. Mistakes (unknown tools, invalid arguments, malformed or truncated
output) are fed back as observations and cost a step. The loop ends at
``final_answer``, the step budget, the deadline or a transport failure, and
never raises into the frame loop. History is append-only except for image
eviction, which trades prompt-cache reuse for context.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

from . import LiveError
from .citations import NO_ANSWER, Verdict, check_claims
from .tools import SCHEMAS, run_tool, state_text, tool_schemas, validate_args
from .transport import TransportError

SYSTEM_PROMPT = """You are the reasoning layer of a research surgical-guidance replay: simulated spinal durotomy repair on cadavers, recorded at one frame per second. A surgeon asks about what has happened so far.

You do not watch the video. A perception system turns frames into an event log (instruments entering and leaving, tips near the durotomy, procedure steps, alerts). Each event has an ID like E000123, a time, a confidence and the frames that support it. You can query that log and inspect individual past frames with tools.

Rules:
- Use only the provided context and tool results. Frames and events after "now" do not exist.
- Prefer the event log. Look at frames only when the events cannot answer the question.
- Perception can be wrong; low-confidence events deserve caution.
- Finish by calling final_answer. Keep "answer" to one or two short sentences a surgeon can hear.
- "claims" must restate every factual statement in "answer". Each claim cites the event IDs, inspected frame indices and/or result IDs that support it. Uncited or unsupported claims are not spoken.
- The opening state is result R0 and each query_events or current_state result has its own ID (R1, R2, ...). To support that something has not happened or is not visible, cite the result that shows it in "result_ids".
- Put the typed answer in "value": a list of names, a number of seconds, a count, true/false, or null if unknown.
- If the evidence is insufficient, say so, and explain what is missing in "unresolved"."""

SCHEMA_MANUAL = """
Respond with exactly one JSON object per turn: {"tool": <name>, "args": <arguments>}. Tool results arrive in the next user message. Tools:
"""


@dataclass(frozen=True)
class AgentConfig:
    mode: str = "native"
    model: str = "qwen-agent"
    max_steps: int = 6
    max_images: int = 2
    keep_images: int = 2
    deadline_s: float = 90.0
    thinking: bool = False
    context_events: int = 12
    max_tokens: int = 1024
    temperature: float = 0.0
    seed: int = 42

    def __post_init__(self):
        if self.mode not in ("native", "schema"):
            raise LiveError("Agent mode must be 'native' or 'schema'")
        if self.max_steps < 1 or self.max_images < 0 or self.keep_images < 0 or self.deadline_s <= 0:
            raise LiveError("Invalid agent budgets")

    def to_json(self):
        return dict(self.__dict__)


@dataclass
class AgentResult:
    question_id: str
    question: str
    now_ms: int
    frame_index: int
    status: str = "error"
    final: dict | None = None
    verdict: Verdict | None = None
    steps: list = field(default_factory=list)
    elapsed_ms: float = 0.0
    tool_calls: int = 0
    images: int = 0
    evictions: int = 0
    error: str | None = None

    @property
    def spoken_text(self):
        return self.verdict.spoken_text if self.verdict else NO_ANSWER

    def to_json(self):
        return {"question_id": self.question_id, "question": self.question, "now_ms": self.now_ms,
                "frame_index": self.frame_index, "status": self.status, "final": self.final,
                "verdict": (self.verdict or Verdict(NO_ANSWER, [], [])).to_json(), "steps": self.steps,
                "elapsed_ms": round(self.elapsed_ms, 1), "tool_calls": self.tool_calls, "images": self.images,
                "evictions": self.evictions, "error": self.error}


def _action_schema(names):
    return {"type": "object", "anyOf": [
        {"type": "object", "additionalProperties": False, "required": ["tool", "args"],
         "properties": {"tool": {"const": name}, "args": SCHEMAS[name]["parameters"]}} for name in names]}


class GuidanceAgent:
    def __init__(self, transport, config=None, evidence_dir=None, clock=time.monotonic):
        self.transport = transport
        self.config = config or AgentConfig()
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        self.clock = clock

    def answer(self, question, ctx, question_id, value_hint=None):
        config = self.config
        ctx.max_images = config.max_images
        result = AgentResult(question_id, question, int(ctx.now_ms), ctx.frame_index)
        directory = None
        if self.evidence_dir is not None:
            directory = self.evidence_dir / question_id
            directory.mkdir(parents=True, exist_ok=False)
        specialist = ctx.specialist is not None
        tools = tool_schemas(specialist)
        names = [tool["function"]["name"] for tool in tools]
        system = SYSTEM_PROMPT
        if config.mode == "schema":
            system += SCHEMA_MANUAL + "\n".join(
                f"- {name}: {SCHEMAS[name]['description']} Arguments: {json.dumps(SCHEMAS[name]['parameters'])}"
                for name in names)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": self._opening(question, ctx, value_hint)}]
        started_wall, started = time.perf_counter(), self.clock()
        image_messages = []
        try:
            for step in range(1, config.max_steps + 1):
                if self.clock() - started > config.deadline_s:
                    result.status = "deadline"
                    break
                if step == config.max_steps and step > 1:
                    messages.append({"role": "user", "content": "Step budget reached: call final_answer now with what you have."})
                request = self._request(messages, tools, names)
                try:
                    completion = self.transport.complete(request)
                except TransportError as exc:
                    result.status, result.error = "transport_error", str(exc)
                    break
                record = {"step": step, "elapsed_ms": round(completion.elapsed_ms, 1), "actions": []}
                choice, message = self._choice(completion.envelope)
                usage = completion.envelope.get("usage") or {}
                timings = completion.envelope.get("timings") or {}
                record.update(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                              cached_tokens=timings.get("cache_n"), prompt_eval_tokens=timings.get("prompt_n"),
                              finish_reason=choice.get("finish_reason"), reasoning=message.get("reasoning_content"))
                result.steps.append(record)
                finished = self._handle(message, choice, messages, ctx, record, result, image_messages)
                self._evict(image_messages, result)
                self._save_step(directory, step, completion, record)
                if finished:
                    break
            else:
                result.status = "budget_exhausted"
        except LiveError as exc:
            result.status, result.error = "error", str(exc)
        if result.status == "answered":
            result.verdict = check_claims(result.final, ctx.log, ctx.now_ms, ctx.frame_index, ctx.viewed_frames,
                                          ctx.results)
        else:
            result.verdict = Verdict(NO_ANSWER, [], [])
        result.elapsed_ms = (time.perf_counter() - started_wall) * 1000
        result.images = ctx.images_used
        if directory is not None:
            (directory / "result.json").write_text(json.dumps(result.to_json(), indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
        return result

    def _opening(self, question, ctx, value_hint):
        lines = [f"Question (asked at t={ctx.now_ms / 1000:.1f}s, frame {ctx.frame_index}): {question}"]
        if value_hint:
            lines.append(f"Answer value type: {value_hint}.")
        ctx.results["R0"] = {"tool": "current_state", "at_ms": ctx.now_ms}
        lines += ["", "Current state (R0):", state_text(ctx)]
        events = ctx.log.query(ctx.now_ms, limit=self.config.context_events)
        lines += ["", f"Recent events (latest {len(events)}):"] + ([event.line() for event in events] or ["none yet"])
        return "\n".join(lines)

    def _request(self, messages, tools, names):
        config = self.config
        request = {"model": config.model, "messages": copy.deepcopy(messages), "max_tokens": config.max_tokens,
                   "temperature": config.temperature, "seed": config.seed, "stream": False, "cache_prompt": True,
                   "chat_template_kwargs": {"enable_thinking": config.thinking}}
        if config.mode == "native":
            request.update(tools=tools, tool_choice="required", parallel_tool_calls=False)
        else:
            request["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "agent_action", "strict": True, "schema": _action_schema(names)}}
        return request

    @staticmethod
    def _choice(envelope):
        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise TransportError("Chat response has no choices.")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise TransportError("Chat response has no message.")
        return choices[0], message

    def _handle(self, message, choice, messages, ctx, record, result, image_messages):
        """Apply one model turn. Returns True when a valid final answer was accepted."""
        content = message.get("content") or ""
        if choice.get("finish_reason") == "length":
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Your last output was truncated. Be brief and call one tool."})
            record["actions"].append({"tool": None, "ok": False, "error": "truncated"})
            return False
        if self.config.mode == "native":
            calls = message.get("tool_calls") or []
            if not calls:
                parsed = self._json(content)
                if isinstance(parsed, dict) and "answer" in parsed and "claims" in parsed:
                    calls = [{"id": "call_inline_final", "type": "function",
                              "function": {"name": "final_answer", "arguments": content}}]
                    message = {**message, "tool_calls": calls}
                else:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": "You must call a tool. Call final_answer to finish."})
                    record["actions"].append({"tool": None, "ok": False, "error": "no tool call"})
                    return False
            messages.append({"role": "assistant", "content": content, "tool_calls": calls})
            pending_images, finished = [], False
            for index, call in enumerate(calls):
                function = call.get("function") or {}
                call_id = call.get("id") or f"call_{len(result.steps)}_{index}"
                if finished:
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": "Ignored: already finished."})
                    continue
                text, image, finished = self._execute(function.get("name"), function.get("arguments"), ctx, record, result)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
                if image:
                    pending_images.append(self._image_message(f"Image for {call_id}", image))
            for image_message in pending_images:
                messages.append(image_message)
                image_messages.append(image_message)
            return finished
        messages.append({"role": "assistant", "content": content})
        parsed = self._json(content)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("tool"), str):
            messages.append({"role": "user", "content": 'Error: reply with one JSON object {"tool": ..., "args": ...}.'})
            record["actions"].append({"tool": None, "ok": False, "error": "malformed action"})
            return False
        text, image, finished = self._execute(parsed["tool"], parsed.get("args", {}), ctx, record, result)
        if finished:
            return True
        label = f"Result of {parsed['tool']}" if not text.startswith("Error") else f"Error from {parsed['tool']}"
        if image:
            image_message = self._image_message(f"{label}:\n{text}", image)
            messages.append(image_message)
            image_messages.append(image_message)
        else:
            messages.append({"role": "user", "content": f"{label}:\n{text}"})
        return False

    def _execute(self, name, arguments, ctx, record, result):
        """Run one tool call. Returns (text for the model, image or None, finished)."""
        result.tool_calls += 1
        action = {"tool": name, "ok": False}
        record["actions"].append(action)
        if isinstance(arguments, str):
            args = self._json(arguments)
            if args is None and arguments.strip():
                action["error"] = "arguments are not valid JSON"
                return "Error: arguments are not valid JSON; send a JSON object.", None, False
            args = args if args is not None else {}
        else:
            args = arguments if arguments is not None else {}
        action["args"] = args
        try:
            if name == "final_answer":
                validate_args(name, args)
                result.final, result.status = args, "answered"
                action["ok"] = True
                return "Answer recorded.", None, True
            outcome = run_tool(name, args, ctx)
        except LiveError as exc:
            action["error"] = str(exc)
            return f"Error: {exc}", None, False
        action["ok"] = True
        if outcome.viewed_frame is not None:
            action["frame"] = outcome.viewed_frame
        return outcome.text, outcome.image_b64, False

    @staticmethod
    def _image_message(text, image_b64):
        return {"role": "user", "content": [{"type": "text", "text": text},
                                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}]}

    def _evict(self, image_messages, result):
        live = [m for m in image_messages if any(part.get("type") == "image_url" for part in m["content"])]
        for message in live[: max(0, len(live) - self.config.keep_images)]:
            label = message["content"][0]["text"].splitlines()[0]
            message["content"] = [message["content"][0],
                                  {"type": "text", "text": f"[{label}: image removed to save context]"}]
            result.evictions += 1

    @staticmethod
    def _json(text):
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _save_step(directory, step, completion, record):
        if directory is None:
            return
        path = directory / f"step-{step:02d}"
        path.mkdir()
        (path / "request.json").write_bytes(completion.request_bytes)
        (path / "response.json").write_bytes(completion.raw)
        (path / "meta.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
