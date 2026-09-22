"""Explicit Qwen3.8 non-thinking settings for the local video runtime.

Matching these numbers does not establish sampler equivalence across backends.
In particular, llama.cpp's penalty history and sampler order are separate from
the model-card recommendations below.
"""
from __future__ import annotations

import math
import re


QWEN_SAMPLING_PROFILE = "qwen3.8-27b-non-thinking-v1"
QWEN_SAMPLING_SOURCE = "https://huggingface.co/Qwen/Qwen3.8-27B#best-practices"


def qwen_non_thinking_parameters() -> dict[str, float | int]:
    """Return fresh llama.cpp request fields, without relying on its defaults.

    ``repeat_penalty`` is llama.cpp's name for the model card's
    ``repetition_penalty``. The neutral frequency penalty and fixed seed are
    local reproducibility choices, not additional Qwen recommendations.
    """
    return {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repeat_penalty": 1.0,
        "frequency_penalty": 0.0,
        "seed": 42,
    }


def qwen_sampling_receipt() -> dict:
    """Describe requested values without claiming observed runtime behavior."""
    return {
        "profile": QWEN_SAMPLING_PROFILE,
        "source": QWEN_SAMPLING_SOURCE,
        "mode": "non-thinking",
        "requested_parameters": qwen_non_thinking_parameters(),
        "local_choices": ["frequency_penalty", "seed"],
        "backend_equivalence_verified": False,
        "limitation": "Matching parameter values does not verify penalty-history or sampler-order equivalence with other backends.",
    }


def verify_qwen_sampling(segment: str, max_tokens: int) -> dict:
    """Verify one actual sampler snapshot in a request's saved server log.

    The pinned runtime prints numeric settings to three decimal places. This
    checks those observed values and its generated-token-only history marker;
    it is not a claim of numerical equivalence with another inference backend.
    """
    if type(max_tokens) is not int or not 0 < max_tokens <= 2147483647:
        raise ValueError("max_tokens must be a positive 32-bit integer.")
    if not isinstance(segment, str):
        raise ValueError("Sampler verification requires a server-log string.")
    chains = list(re.finditer(r"(?m)^[^\r\n]*\bsampler chain:[ \t]*([^\r\n]*)$", segment))
    snapshots = list(re.finditer(r"(?m)^[^\r\n]*\bsampler params:[ \t]*\r?$", segment))
    if len(chains) != 1 or len(snapshots) != 1:
        raise ValueError("Expected exactly one effective sampler chain and parameter snapshot per request.")
    if chains[0].start() > snapshots[0].start():
        raise ValueError("Sampler chain and parameter snapshot are out of order.")
    chain = [name.strip() for name in chains[0].group(1).split("->")]
    expected_chain = ["logits", "penalties", "temp-ext", "top-k", "top-p", "min-p", "dist"]
    normalized_chain = ["min-p" if name == "?min-p" else name for name in chain]
    if normalized_chain != expected_chain:
        raise ValueError(f"Qwen sampler order or active controls differ: {chain}.")

    lines = []
    for line in segment[snapshots[0].end():].splitlines()[1:]:
        if not line.startswith(("\t", " ")):
            break
        lines.append(line)
    fields = {}
    for name, value in re.findall(r"\b([a-z_]+)[ \t]*=[ \t]*([^,\r\n]+)", "\n".join(lines)):
        if name in fields:
            raise ValueError(f"Duplicate effective sampler field: {name}.")
        fields[name] = value.strip()

    requested = qwen_non_thinking_parameters()
    requested.update({"repeat_last_n": max_tokens, "samplers_generated_only": True})
    expected = {name: value for name, value in requested.items() if name != "seed"}
    # The common sampler printer uses `temp`, while the API uses `temperature`.
    observed = {}
    for name, value in expected.items():
        log_name = "temp" if name == "temperature" else name
        raw = fields.get(log_name)
        if raw is None:
            raise ValueError(f"Missing effective sampler field: {log_name}.")
        try:
            number = float(raw)
        except ValueError as error:
            raise ValueError(f"Invalid effective sampler field: {log_name}={raw}.") from error
        if not math.isfinite(number) or number != value:
            raise ValueError(f"Unexpected effective sampler field: {log_name}={raw}; expected {value}.")
        observed[name] = value
    if fields.get("sampler_history_scope") != "generated":
        raise ValueError("Effective sampler history must contain generated tokens only.")
    observed["sampler_history_scope"] = "generated"

    receipt = qwen_sampling_receipt()
    receipt.update({
        "requested_parameters": requested,
        "observed_parameters": observed,
        "observed_sampler_chain": chain,
        "verified": True,
        "snapshot_count": 1,
        "unobserved_requested_fields": ["seed"],
        "limitation": "Logged values, sampler order, and generated-only history were verified; identical numerical behavior across backends was not. Numeric log precision is three decimal places; seed is not printed by the pinned sampler.",
    })
    return receipt
