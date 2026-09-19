"""Construct, but never run, an Unsloth VLM SFT trainer for hydrated chat rows.

Importing this module loads no ML library, model, dataset, or image. Callers must
provide an already loaded HF model/processor and validated, hydrated dataset rows.
The training interface is independent of the enhancement record/response schema.
See docs/TRAINING.md and the preserved research compatibility note for source pins.
"""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from typing import Any, Literal, Mapping, Sequence

LossScope = Literal["all_assistant_turns", "final_assistant_turn"]


def _declares_parameter(cls: type, name: str) -> bool:
    """Unsloth's public wrapper has **kwargs; inspect its concrete base too."""
    return any(name in signature(base.__init__).parameters for base in cls.__mro__)


def check_runtime_versions() -> dict[str, str]:
    """Enforce the researched Unsloth dependency envelope, not arbitrary latest."""
    try:
        installed = {name: version(name) for name in
                     ("unsloth", "unsloth_zoo", "trl", "transformers", "torch")}
    except PackageNotFoundError as exc:
        raise RuntimeError("Use an installed, compatible Unsloth training environment.") from exc
    from packaging.specifiers import SpecifierSet

    requirements = {
        "unsloth_zoo": ">=2026.9.3",
        "trl": ">=0.18.2,!=0.19.0,<=0.24.0",
        "transformers": (
            ">=4.51.3,!=4.52.0,!=4.52.1,!=4.52.2,!=4.52.3,!=4.53.0,"
            "!=4.54.0,!=4.55.0,!=4.55.1,!=4.57.0,!=4.57.4,!=4.57.5,"
            "!=5.0.0,!=5.1.0,<=5.5.0"
        ),
    }
    for name, requirement in requirements.items():
        if installed[name] not in SpecifierSet(requirement):
            raise RuntimeError(
                f"{name}=={installed[name]} is outside the researched dependency "
                f"envelope ({requirement}); revalidate this recipe before using it."
            )
    return installed


class StrictVisionCollator:
    """Preserve full image/text input, reject overlength rows, and check labels.

    This guard does not solve an oversized image allocation: constrain frame
    count/resolution in the export and processor before collating a batch.
    """

    def __init__(self, inner: Any, max_seq_length: int):
        if type(max_seq_length) is not int or max_seq_length <= 0:
            raise ValueError("max_seq_length must be an explicit positive integer.")
        for attr in ("max_seq_length", "truncation"):
            if not hasattr(inner, attr):
                raise RuntimeError(f"Unsupported collator: missing {attr} control.")
        self.inner = inner
        self.max_seq_length = max_seq_length
        # In current Unsloth, passing None to __init__ inherits the model cap.
        # Both inspected controls must be cleared to prevent silent truncation.
        self.inner.max_seq_length = None
        self.inner.truncation = False

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if self.inner.max_seq_length is not None or self.inner.truncation:
            raise RuntimeError("Collator truncation was re-enabled after preflight.")
        # The upstream normalizer mutates nested messages; preserve caller rows.
        batch = self.inner(deepcopy(examples))
        if not all(key in batch for key in ("input_ids", "attention_mask", "labels")):
            raise RuntimeError("Collator omitted required model inputs or training labels.")
        lengths = batch["attention_mask"].sum(dim=-1).tolist()
        if any(length > self.max_seq_length for length in lengths):
            raise ValueError(
                f"Expanded sequence lengths {lengths} exceed {self.max_seq_length}. "
                "Shorten the source conversation/frame window; no tokens were truncated."
            )
        if batch["input_ids"].shape != batch["labels"].shape:
            raise RuntimeError("Input/label shapes differ.")
        supervised = (batch["labels"] != -100).sum(dim=-1).tolist()
        if any(count == 0 for count in supervised):
            raise ValueError("An example has zero supervised tokens; inspect markers and target.")
        if not (batch["labels"][batch["attention_mask"] == 0] == -100).all().item():
            raise RuntimeError("Padding tokens were not excluded from loss.")
        return batch


def _make_collator(
    collator_cls: type, model: Any, processor: Any, *, instruction_part: str,
    response_part: str, loss_scope: LossScope, max_seq_length: int, resize: Any,
) -> StrictVisionCollator:
    if loss_scope not in ("all_assistant_turns", "final_assistant_turn"):
        raise ValueError("Unknown loss_scope.")
    for key in ("train_on_responses_only", "last_response_only"):
        if not _declares_parameter(collator_cls, key):
            raise RuntimeError(f"Installed collator lacks verified {key}; do not silently fall back.")
    if not isinstance(instruction_part, str) or not instruction_part:
        raise ValueError("Provide the exact tested user-turn marker from this processor.")
    if not isinstance(response_part, str) or not response_part or response_part == instruction_part:
        raise ValueError("Provide a distinct, exact tested assistant-turn marker.")
    inner = collator_cls(
        model, processor, max_seq_length=max_seq_length, resize=resize,
        train_on_responses_only=True,
        instruction_part=instruction_part, response_part=response_part,
        force_match=True, last_response_only=loss_scope == "final_assistant_turn",
        completion_only_loss=True, num_proc=1,
    )
    return StrictVisionCollator(inner, max_seq_length)


def _first_image(rows: Sequence[dict[str, Any]]) -> Any:
    for row in rows:
        for message in row["messages"]:
            for part in message["content"]:
                if part["type"] == "image":
                    image = part.get("image")
                    # No file/network hydration occurs in this recipe.
                    if not hasattr(image, "mode") or not hasattr(image, "size"):
                        raise ValueError("Image blocks must already carry PIL images.")
                    return image
    raise ValueError("Provide a nonempty vision dataset with a hydrated image.")


def masking_preflight(
    collator_cls: type, model: Any, processor: Any, rows: Sequence[dict[str, Any]],
    *, instruction_part: str, response_part: str, max_seq_length: int, resize: Any,
) -> dict[str, Any]:
    """Collate a two-turn/two-image sentinel case; no model forward or training.

    The same actual frame is referenced twice deliberately to test message-order
    plumbing; this is a diagnostic case and is never appended to training data.
    """
    image = _first_image(rows)
    user_a, user_b = "SOSPINE_USER_FIRST_U17", "SOSPINE_USER_SECOND_U29"
    answer_a, answer_b = "SOSPINE_ANSWER_FIRST_A17", "SOSPINE_ANSWER_FINAL_B29"
    def text_message(role: str, text: str, with_image: bool = False) -> dict[str, Any]:
        content = [{"type": "text", "text": text}]
        if with_image:
            content.append({"type": "image", "image": image})
        return {"role": role, "content": content}
    probe = {"messages": [
        text_message("user", user_a, True), text_message("assistant", answer_a),
        text_message("user", user_b, True), text_message("assistant", answer_b),
    ]}
    tokenizer = getattr(processor, "tokenizer", processor)
    result: dict[str, Any] = {}
    for scope in ("all_assistant_turns", "final_assistant_turn"):
        collator = _make_collator(
            collator_cls, model, processor, instruction_part=instruction_part,
            response_part=response_part, loss_scope=scope,
            max_seq_length=max_seq_length, resize=resize,
        )
        batch = collator([probe])
        if "pixel_values" not in batch:
            raise RuntimeError("Multimodal probe has no pixel_values; reject text-only fallback.")
        ids = batch["labels"][0]
        decoded = tokenizer.decode(ids[ids != -100].tolist(), skip_special_tokens=False)
        if answer_b not in decoded or user_a in decoded or user_b in decoded:
            raise RuntimeError(f"Response mask failed for {scope}: {decoded!r}")
        if (answer_a in decoded) != (scope == "all_assistant_turns"):
            raise RuntimeError(f"Earlier assistant loss is wrong for {scope}: {decoded!r}")
        result[scope] = {
            "supervised_tokens": int((ids != -100).sum().item()),
            "sequence_tokens": int(batch["attention_mask"][0].sum().item()),
            "decoded_supervised_probe": decoded,
        }
    return result


def build_sft_trainer(
    model: Any, processor: Any, train_dataset: Sequence[dict[str, Any]], *,
    output_dir: str, instruction_part: str, response_part: str,
    max_seq_length: int, loss_scope: LossScope = "final_assistant_turn",
    eval_dataset: Sequence[dict[str, Any]] | None = None,
    resize: Any = "max", training_options: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return (trainer, preflight_report). Never calls trainer.train().

    Model loading, PEFT configuration, device placement and processor image budget
    are caller responsibilities. Pass reviewed train/eval splits, already hydrated.
    """
    if not train_dataset:
        raise ValueError("train_dataset must contain at least one reviewed example.")
    if loss_scope not in ("all_assistant_turns", "final_assistant_turn"):
        raise ValueError("Unknown loss_scope.")
    if type(max_seq_length) is not int or max_seq_length <= 0:
        raise ValueError("max_seq_length must be an explicit positive integer.")
    installed = check_runtime_versions()
    # Unsloth must patch the stack before importing the TRL trainer.
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    locked = {
        "remove_unused_columns": False, "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True}, "max_length": None,
        "packing": False, "eval_packing": False, "padding_free": False,
        "assistant_only_loss": False, "completion_only_loss": False,
    }
    for key in locked:
        if key not in signature(SFTConfig).parameters:
            raise RuntimeError(f"SFTConfig lacks {key}; revalidate the installed stack.")
    if "processing_class" not in signature(SFTTrainer.__init__).parameters:
        raise RuntimeError("SFTTrainer processing_class interface is unavailable.")
    options = dict(training_options or {})
    for key, value in locked.items():
        if key in options and options[key] != value:
            raise ValueError(f"Do not override protected vision setting {key}={value!r}.")
    options = {
        "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 4, "num_train_epochs": 1,
        "learning_rate": 2e-5, "logging_steps": 10, "report_to": "none",
        **options, **locked, "output_dir": output_dir,
    }
    report = {"versions": installed, "loss_scope": loss_scope}
    report["masking_probe"] = masking_preflight(
        UnslothVisionDataCollator, model, processor, train_dataset,
        instruction_part=instruction_part, response_part=response_part,
        max_seq_length=max_seq_length, resize=resize,
    )
    collator = _make_collator(
        UnslothVisionDataCollator, model, processor,
        instruction_part=instruction_part, response_part=response_part,
        loss_scope=loss_scope, max_seq_length=max_seq_length, resize=resize,
    )
    # Check actual records too. Full length/label checks continue on every batch.
    for name, rows in (("train", train_dataset), ("eval", eval_dataset)):
        if rows:
            batch = collator([rows[0]])
            report[f"first_{name}_supervised_tokens"] = int((batch["labels"] != -100).sum().item())
    trainer = SFTTrainer(
        model=model, processing_class=processor, data_collator=collator,
        train_dataset=train_dataset, eval_dataset=eval_dataset, args=SFTConfig(**options),
    )
    return trainer, report
