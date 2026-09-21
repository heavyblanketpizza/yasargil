# Training with the enhanced dataset

The active training interface is `yasargil.training.build_sft_trainer`. It accepts an already loaded Hugging Face vision-language model, its matching processor, and validated conversations whose image blocks contain RGB PIL images. It constructs a trainer and performs processor/collator checks; it never downloads a model or starts training. Importing the module does not load any ML libraries.

The interface is independent of the enhancement record schema. The enhancement archive contains source facts, model proposals, review history, and provenance. Its training export contains ordinary ordered `messages` with typed text/image blocks and serialized assistant targets. The contract/export layer owns schema validation, review eligibility, media hashes, partition separation and image hydration. The trainer receives only that validated training view.

[Independent MedGemma annotation](MEDGEMMA_FRAME_ANNOTATION.md) produces pending
per-frame proposals, separate from the v2 training-export adapter. Historical
enhancement archives remain verifiable. Model-generated inspection claims and
turns require actual human surgeon or clinical-domain-expert review, including
clinical appropriateness, before training eligibility. Completed-review ingestion
and an export adapter for the new independent annotation records remain pending;
no command converts inference completion or a blank worksheet into approval.

## Prepare the training environment

Install the locked core project with `uv sync --frozen` and use `uv run yasargil` for source import, annotation, validation and export. `.python-version` selects Python 3.12. The core lock does not install or validate PyTorch, Unsloth, a GPU stack or model weights; choose and freeze the training environment separately using the compatibility constraints below.

Use the intended HF training checkpoint and matching processor revision. A llama.cpp GGUF model/projector pair is an inference artifact; do not pass its alias or GGUF file to this trainer. Configure model loading, PEFT/trainable layers, image resolution, frame count and device placement before constructing the trainer. The helper does not choose or download a checkpoint.

The [Ollama-to-llama.cpp migration](LLAMA_CPP.md#migration-from-ollama) changes
local inference, not this Hugging Face/Unsloth training interface. Historical
Ollama enhancement archives keep their original provenance and can be checked by the
read-only legacy verifier; new inference runs use llama.cpp. Both require the
same applicable human-review and export gates.

The currently researched Unsloth dependency envelope is:

| Package | Required envelope |
|---|---|
| `unsloth_zoo` | `>=2026.9.3`; required collator features are checked separately |
| `trl` | `>=0.18.2,!=0.19.0,<=0.24.0` |
| `transformers` | `>=4.51.3,<=5.5.0`, excluding the releases listed in the runtime helper |

These constraints come from Unsloth source inspected on 10 September 2026. Installing the newest independent TRL/Transformers releases can violate them. The package/version checks do not prove every allowed combination works: the helper also checks constructor signatures and rejects missing features. Compatible TRL 0.24.0 source was inspected for the protected configuration arguments. Preserve the full environment lock, model/processor revisions and chat-template identity after selecting and testing a stack.

The [runtime helper](../src/yasargil/training.py) records the complete version exclusions and enforces the supported interfaces. Key implementation evidence is [Unsloth dependency metadata](https://github.com/unslothai/unsloth/blob/d0dbe9059efa443c6ad8bd1d51af7e2d9276a2bc/pyproject.toml#L181), [the vision collator](https://github.com/unslothai/unsloth-zoo/blob/a7eadfb1c16532b78b5fd8ea1a8bf91d635d71cf/unsloth_zoo/vision_utils.py#L1001), and [the compatible TRL VLM restrictions](https://github.com/huggingface/trl/blob/04fd1203af4fc1e629f58e0ac0d0c5bb95f82a45/trl/trainer/sft_trainer.py#L651).

## Construct the trainer

First obtain reviewed training and validation exports through the contract layer and hydrate their media under the configured dataset root. Preserve message order, content-block order, and actual image pixels. A file name in ordinary prompt text is not an image input. Do not mix causal and retrospective examples or different loss scopes in one export/job.

Use the active receipt-aware loader. Here `train_export_path`, `validation_export_path`, `dataset_root` and `artifact_root` are the paths from your release/configuration:

```python
from yasargil.contract import load_export

# Current enhanced conversations include multiple intended supervised answers.
loss_scope = "all_assistant_turns"
train_rows = load_export(
    train_export_path,
    dataset_root,
    artifact_root=artifact_root,
    expected_partition="train",
    expected_loss_scope=loss_scope,
    expected_intended_use="causal_intraoperative_assistance",
)
validation_rows = load_export(
    validation_export_path,
    dataset_root,
    artifact_root=artifact_root,
    expected_partition="validation",
    expected_loss_scope=loss_scope,
    expected_intended_use="causal_intraoperative_assistance",
)
```

The loader checks receipt purpose, hashes, partition, loss scope, intended use, media and row count before returning hydrated rows. Unprefixed media locations resolve under `dataset_root`; `artifact:REL` locations resolve under `artifact_root`. Keep `allow_preview=False` for training. Set the trainer's loss scope to the same value that was checked while loading.

The enhancement loop currently writes `all_assistant_turns` because its summary and question/answer turns are each proposed supervision. The loader and trainer default to `final_assistant_turn`; pass the explicit value above for an eligible enhancement export. To train only the final answer, first create and review a release with that declared loss policy and load it with the matching value. Do not silently change the expected scope to make a mismatched receipt load. Earlier assistant context still requires review even if it receives no direct loss.

Inspect a rendered conversation using the exact loaded processor. Identify and verify its user-turn and assistant-turn markers, including whitespace. Pass those markers explicitly; the helper does not guess them from a model name. The following integration assumes that the caller has already supplied the named model, processor, hydrated rows, verified markers and validated context limit:

```python
from unsloth import FastVisionModel
from yasargil.training import build_sft_trainer

# model/processor were loaded together and PEFT was configured by the caller.
# train_rows/validation_rows came from validated, hydrated exports.
FastVisionModel.for_training(model)

trainer, preflight = build_sft_trainer(
    model,
    processor,
    train_rows,
    eval_dataset=validation_rows,
    output_dir="outputs/sospine-sft",
    instruction_part=verified_user_marker,
    response_part=verified_assistant_marker,
    max_seq_length=validated_context_limit,
    loss_scope=loss_scope,
    training_options={
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
        "learning_rate": 2e-5,
        "report_to": "none",
    },
)
print(preflight)
```

The illustrated optimization settings are starting values, not an experimentally established SOSpine recipe. Set them for the chosen checkpoint, adapter and hardware. The required `max_seq_length` is the permitted **expanded** sequence length, including image tokens, context, assistant answer and terminators. The checkpoint's context support and memory budget constrain that value.

After checking the real exports and preflight report in the chosen training environment, training is a separate explicit call:

```python
trainer.train()
```

No training was run while implementing this interface. A constructed trainer and successful CPU tests are not evidence that a particular HF model, processor, CUDA environment or surgical task has been validated.

## Supervision and integrity checks

`loss_scope="final_assistant_turn"` is the default. Earlier assistant turns remain conditioning context while only the final answer contributes response-token loss. Set `loss_scope="all_assistant_turns"` to supervise every assistant answer; each must qualify under the export's review policy. Neither option creates outcome-based reinforcement learning, and a structured JSON target does not imply chain-of-thought training.

The helper uses Unsloth's `train_on_responses_only=True` and verified `last_response_only` flag. It keeps TRL `assistant_only_loss=False`: the inspected VLM trainer rejects that otherwise familiar text-training option. Packing, evaluation packing and padding-free mode are disabled. The custom collator owns multimodal preprocessing and labels; unused-column removal and ordinary dataset preparation are disabled.

A synthetic two-turn/two-image probe checks both supervision scopes with distinguishable answer text. It verifies that user sentinels are excluded from loss, the intended assistant sentinels survive, and vision tensors exist. The probe reuses an actual image only for this diagnostic; it is never appended to the training dataset. The first real train/evaluation row is collated too. Inspect additional actual rows, especially the longest conversations and largest image/frame combinations, before starting a run.

The strict collator disables truncation explicitly and rejects overlength batches. Passing `None` to the upstream collator constructor alone is insufficient because Unsloth can inherit a model length limit. The guard also rejects zero-supervision rows, input/label shape mismatches, and unmasked padding. These checks continue on every training/evaluation batch. They do not measure clinical correctness or prevent excessive image allocation before preprocessing. Bound image count and processor pixel budgets beforehand.

The helper's `resize="max"` default disables Unsloth's additional resizing; it leaves the checkpoint processor's own resize/crop behavior active. Record any changed image transform and evaluate small structures under it. Keep original source media intact.

## Run the lightweight interface tests

With the source package available on the import path, run from the repository root:

```sh
uv run python -B -m unittest discover -s tests -p 'test_training.py' -v
```

These CPU-only tests use fake collators to check import isolation, wrapper/base feature detection, both loss scopes, explicit markers, truncation controls and early invalid-argument rejection. The tests do not load Unsloth, a real processor, model weights or a GPU. Real processor/mask and model-forward checks remain an environment-specific step. Passing these checks does not establish annotation accuracy, clinical usefulness or a completed training run.
