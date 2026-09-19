# Local Qwen–MedGemma enhancement

Updated 2026-09-19. This describes the offline loop in [enhancement.py](../src/yasargil/enhancement.py), its [teacher adapter](../src/yasargil/teacher.py), full-sequence batch processing, durable pause/resume and [CLI](../src/yasargil/__main__.py). Download the dataset separately as described in [Data sources](DATA_SOURCES.md).

The loop creates evidence-linked proposals and a review packet from a bounded SOSpine window. Inputs are released JPEGs and the original CSV tool labels and coordinates: manually annotated instrument/durotomy points and bounding boxes computed from those points. Qwen writes new descriptive proposals from that evidence; MedGemma observes, reviews and revises them. The original rows are not prewritten descriptions. Output is an archive plus a proposed multimodal conversation, pending human review.

This page describes the annotation-conditioned `enhance-sospine` loop. The separate [full-video annotation path](FRAME_ANNOTATION.md) gives Qwen the complete source video and frozen selected stills without source CSV labels; those labels are added later for [MedGemma frame review](MEDGEMMA_FRAME_REVIEW.md) when verified SOSpine tables are available. New inference code in both paths now uses llama.cpp; see the [Ollama migration reasons and local setup/validation status](LLAMA_CPP.md#migration-from-ollama).

## Responsibilities and call order

| Stage | Model and direct evidence | Responsibility |
|---|---|---|
| `propose` | Qwen; initial uniformly selected frames and their original CSV labels and coordinates. | Write new observations, bounded interpretations, questions, uncertainty and evidence requests. |
| `independent_observe` | MedGemma; the same initial frames/annotations, without Qwen's output. | Record an initial observation before reading the proposal. This is proposal independence, not an annotation-blind evaluation. |
| `review` | MedGemma; initial frames plus both preceding outputs. | Inspect claims against images, preserve disagreements and request additional frame intervals for unresolved questions. |
| `search` | Qwen; new frames in requested intervals, plus the preceding review and its retained ancestry. | Reinspect local evidence for the review's questions. The direct image payload contains the newly selected frames. |
| `final_revise` | MedGemma; all frames seen so far, the preceding review and Qwen's search output. | Revise descriptions/questions, retain uncertainty and request another bounded search if warranted. |

The first three calls form the initial loop. Each optional round adds two calls, so `R` allowed search rounds permit at most `3 + 2R` logical stages per window. The loop stops when no further evidence is requested, no unseen released frame exists in requested intervals, the frame budget is exhausted, or the round budget is reached. Unresolved requests remain recorded; agreement is not forced. Explicit resume may repeat a failed or unfinished stage, adding an inference attempt beyond this logical-stage ceiling; already completed stages are replayed without inference.

“Search” means inspecting additional **local SOSpine images**, not searching the web or literature. Qwen is the targeted reinspector. A TimeLens2 adapter is not implemented. llama.cpp receives ordered image data URLs with release-index captions through its OpenAI-compatible chat endpoint; this path does not invoke a native video processor or transmit an MP4. General video-model capability does not prove that this transport exercised it.

## Source and temporal bounds

The inspected release has 15,694 JPEGs in 24 sequences, sampled at 1 fps from recordings described as 30 fps. Continuous source videos and acquisition timestamps are unavailable. Frame indices establish order; `timestamp_ms` stays `null` with basis `unavailable`. Selection does not recover intervening motion.

Case, start index and cutoff are fixed before model calls. Initial selection samples deterministically across available release indices in that window. Later selection uses unseen frames within requested intervals and the same original window. Requests share the per-round budget rather than letting the first interval consume it all. Parent outputs and their evidence exposure remain linked to later calls.

Original tool-tip rows and computed box rows are supplied with distinct origins and raw values. “Exact” row preservation means retaining the released values, CSV row locations and file hashes; it does not imply error-free labels. These rows remain source evidence, separate from the new model-generated descriptions. Outcome rows, surgeon experience and other case metadata are excluded from model payloads. A visible pressure test can still reveal information through pixels; metadata exclusion does not guarantee effective outcome blinding or forecast eligibility.

| Setting | Default | Enforced bound or meaning |
|---|---|---|
| `--initial-frames` | 8 | At least 1; no more than `--max-frames`. |
| `--search-frames` | 4 | Maximum additional frames per round; 1 through `--max-frames`. |
| `--max-frames` | 24 | At most 32 distinct selected source frames per window. |
| `--max-rounds` | 1 | 0–3 rounds; at most 3–9 logical stages per window, excluding repeated unfinished attempts. |
| `--start-index`, `--cutoff-index` | Required | Positive inclusive indices in order, spanning at most 3,600 release indices. |
| `--num-ctx` | 65,536 | Requested llama.cpp context, 8,192–131,072; must fit the chosen model/hardware. |
| `--num-predict` | 4,096 | Requested generation ceiling, 512–8,192. Truncated output fails validation. |
| `--seed` | 42 | Recorded with temperature 0; not a guarantee of identical output across runtime/hardware changes. |
| `--timeout` | 600 seconds | Per llama.cpp request; no automatic retries. |

The frame ceiling is not a memory or latency guarantee. llama.cpp applies the model's internal image processing. The application preserves original JPEG bytes and order, but does not record internal crop/patch tensors or claim control of a native video sampler.

## Setup and a single window

Use the repository's Python 3.12 selection and locked core dependencies. Install the pinned llama.cpp runtime and matching GGUF model/projector files as described in [Local llama.cpp setup](LLAMA_CPP.md). The client starts an owned local server on demand and stops it after the request; GPU training dependencies are outside this core lock.

```bash
uv sync --frozen
uv run yasargil models
```

Defaults are `qwen3.8-27b` and `medgemma-27b`. Each run inspects the local GGUF model and vision projector, records their SHA-256 identities and quantization, and pins the llama.cpp runtime identity. `models` checks these local artifacts without downloading weights or starting inference.

```bash
uv run yasargil enhance-sospine \
  --dataset-root "/path/to/datasets/SOSpine" \
  --case-id S1A2 --start-index 1 --cutoff-index 12 \
  --initial-frames 4 --search-frames 4 --max-frames 8 \
  --dry-run
```

Dry run prints available/initial indices, configuration, transport and the call ceiling. It writes no files and makes no model requests. It checks the selection plan, not inference success or review eligibility.

To execute that configuration:

```bash
uv run yasargil enhance-sospine \
  --dataset-root "/path/to/datasets/SOSpine" \
  --case-id S1A2 --start-index 1 --cutoff-index 12 \
  --initial-frames 4 --search-frames 4 --max-frames 8 \
  --output-dir outputs/example
```

A new run requires a new output directory outside the dataset root. To continue an existing compatible run, use `resume-enhancement --output-dir outputs/example`, or repeat the original `enhance-sospine` configuration with `--resume`. Failed/interrupted attempts remain available for inspection; resumption never silently replaces their bytes. Overrides `--qwen-model` and `--medgemma-model` must select the distinct supported local model aliases. `--project-root` identifies the directory containing `.runtime/` and defaults to the project working directory; `--timeout` sets the request deadline. The client owns its loopback-only server; there is no external service URL or backend fallback.

## Full-sequence batches

Batch mode inventories the requested cases and requires contiguous release-frame indices beginning at 1. It rejects gaps instead of filling them or silently changing coverage. It groups those indices into sequential, non-overlapping windows; the final window may be shorter. `--window-size` is a count of released frames, not original video frames or verified elapsed seconds. Every planned window is processed independently with its own fixed start/cutoff, model-call lineage, archive and review packet.

```bash
uv run yasargil enhance-sospine-batch \
  --dataset-root "/path/to/datasets/SOSpine" \
  --case-ids S4A3 S5A1 \
  --window-size 12 \
  --initial-frames 4 --search-frames 4 --max-frames 8 --max-rounds 1 \
  --output-dir outputs/sospine-best-worst \
  --dry-run
```

Remove `--dry-run` to execute the saved configuration into that new directory. The plan covers all available frames in the selected cases, but inference still uses each window's selected subset: four initial frames and up to eight after requested searches in this example. Planning does not invent original timestamps or inspect intervening 30-fps motion.

The runner does not carry answers, summaries, tracks or hidden state from one window to the next. A long batch therefore extends dataset coverage, not the model's temporal context. Actions spanning a boundary, long-lived surgical goals and full-operation event continuity remain limitations. Treat overlapping or larger-context experiments as separate configurations/adapters, not implicit behavior of this non-overlapping batch.

Original outcome rows may remain in each archive for later analysis, but neither earlier nor later case outcomes are included in teacher requests. Choosing comparison cases based on measured outcomes is an offline study selection decision, not evidence that the model predicted those outcomes or that the selection mimics deployment. These cases remain simulated repairs, not patient outcome records.

## Pause, status and durable resume

The following commands accept either a batch directory or a single-window directory:

```bash
uv run yasargil enhancement-status --output-dir outputs/sospine-best-worst
uv run yasargil pause-enhancement --output-dir outputs/sospine-best-worst
```

`enhancement-status` reads saved progress without source access or model requests. It reports job state and writer/pause status, plus completed/total windows and current progress for a batch. `pause-enhancement` writes a `PAUSE` request. The active worker finishes its current model call, retains the completed response and pauses before starting another call. A pause request is not an immediate cancellation; a long call may continue until its response arrives or its timeout fires. If no further call remains, the window may finish publication instead of stopping in a paused state.

The first `Ctrl+C` requests the same cooperative pause; `SIGTERM` also requests a pause. A second `Ctrl+C` forces interruption. A terminated process cannot persist unfinished generation tokens, so resumption repeats an incomplete call. Previously completed calls and windows remain reusable. Do not delete a partial output directory merely because its worker stopped.

Before closing the worker or disconnecting the source drive after a cooperative pause, check for `paused` or `completed` with `writer_active: false`. A visible pause request alone means the worker may still be finishing its current call. A power loss or forced interruption can require that call to be repeated, while completed checkpoints remain on disk.

Once the worker has stopped, resume with its saved settings:

```bash
uv run yasargil resume-enhancement --output-dir outputs/sospine-best-worst
```

Run/resume commands stay in the foreground unless launched separately as background processes. A foreground resume remains attached to its terminal, with the same first/second `Ctrl+C` behavior. A background worker uses the same status/pause/resume interface; retain its logs in your local output directory.

`resume-enhancement` loads the saved configuration, with only local runtime settings such as `--project-root` and `--timeout` supplied separately. It validates completed artifacts and reconstructs the deterministic loop from retained outputs. The checkpoint is progress metadata, not permission to trust arbitrary cached answers. If a completed response is valid, replay avoids another inference call; if an attempt did not finish, an explicit resume can make a new attempt while retaining the old one.

Persistence is at the application/call level. **No model KV cache, in-flight token stream or whole-case hidden state is saved.** You can stop the worker and return later with the same source and models available; resumption rebuilds needed context from saved bytes. Even a completed job is verified against source artifacts and the current local model, projector and runtime identities before returning without inference. Keep those files available for that check; no model server needs to be running. `enhancement-status` is independent of them. This does not establish identical output for a retried unfinished call across hardware/runtime changes.

Resume guards bind the saved plan/configuration, source frame and metadata hashes, prompt/schema/version fingerprint, model/projector hashes, quantization and runtime identity. A changed source, model or projector file, runtime, prompt or configuration must not be mixed into an earlier run. Keep source files available at the saved dataset path and retain output artifacts intact; use a new output directory for a changed experiment. An intentional model/runtime update can therefore require a new job. A writer lock prevents two workers from mutating the same job concurrently.

Runs created with Ollama cannot resume under llama.cpp. Keep their requests, responses and archives intact as historical evidence, and create a new output directory using the `llama-cpp-evidence-v1` adapter. The historical `ollama-evidence-v1` verifier remains read-only; it does not migrate old runs. There is no backend switch or fallback. See [Migration from Ollama](LLAMA_CPP.md#migration-from-ollama) for the rationale and model-file transition.

Earlier outputs without `session.json` do not have the snapshot required for durable resume. Resume rejects those directories rather than inventing a missing source/prompt snapshot; start a new job to use checkpoints. A successful explicit resume clears its pause request only after obtaining the writer lock.

For a batch, pausing inside a window preserves that window's completed stages. On resume, finished windows are verified/reused, the interrupted window continues, and remaining windows follow in plan order. All window archives stay pending human review; completing the queue is not a reviewed dataset release.

## Retained artifacts

Each single-window directory, including a window nested inside a batch, retains:

| Artifact | Meaning |
|---|---|
| `plan.json` | Fixed configuration, candidate indices, initial selection and limits. |
| `session.json` | Resume identity: dataset path, plan digest, source-file hashes for the allowed window, prompt/schema identity and stable creation metadata. |
| `checkpoint.json` | Atomically updated progress and successful-call count. States include initializing, running, paused, failed, interrupted, finalizing and completed. Selected indices are distinguished from frames observed by successful calls. |
| `models/qwen.json`, `models/medgemma.json` | Model and projector identities/hashes, quantization, capabilities and runtime identity. |
| `models/manifest.json` | Pinned model identities and hashes of their retained raw metadata receipts. |
| `calls/run-NNN-STAGE/request.json` | Exact canonical UTF-8 request bytes, including ordered image data URLs and the structured response schema. Files can be large. |
| `calls/run-NNN-STAGE/response.json` | Exact response bytes and complete OpenAI-compatible chat response envelope. Available malformed responses are retained on failure. |
| `calls/run-NNN-STAGE/run.json` | Model, prompt/settings, direct evidence, cutoff and parent-run lineage. |
| `calls/run-NNN-STAGE/parsed.json` | Validated output, call timing and reported token counters. |
| `calls/run-NNN-STAGE/success.json` | Hashes binding a completed attempt's request, run, response and parsed files. Repeated attempts have their own receipt. |
| `calls/run-NNN-STAGE/failure.json` | Call-level failure details when inference or validation fails. |
| `calls/run-NNN-STAGE/attempts/…` | Retained additional attempts when explicit resumption repeats an unfinished/failed stage. |
| `archive.json` | v2 draft with original source rows, all retained stage claims, selected frames and the final proposed conversation. Written after a successful loop. |
| `audit.html` | Unblinded stage outputs and changes for engineering/provenance inspection; model identities are visible. |
| `review.html` | Separate embedded-image packet for quality review; restricted metadata is hidden by default. |
| `review.review-template.json` | Blank worksheet bound to the archive revision/hash; verdicts and reviewer attestations remain empty. |
| `training.preview.jsonl`, `training.preview.receipt.json` | Messages-only format preview when there is projected dialogue; not reviewed training data. |
| `completion.json` | Call count, observed indices, stop reason, unresolved searches and elapsed time; explicitly human-review pending and training-ineligible. |
| `failure.json`, `failures/000N.json` | Retained first and subsequent root-level failures. These can remain after a successful resume; use saved status/completion to determine current state. |
| `.finalize/attempt-NNNN/`, `.finalize/ready.json` | Staged final artifact bundle and its hash manifest, used to recover publication after interruption. `completion.json` is published last. |

A batch adds an immutable `batch.json` plan with source hashes and the prompt/protocol fingerprint, pinned `models.json`, mutable `progress.json`, a `results.json` index of completed window archives, and per-window directories named `windows/<case>-<start8>-<end8>/`. Shared source files and each window's candidate images are checked against the batch snapshot before that window starts. `PAUSE` is a cooperative stop request, and `.writer.lock` coordinates writers. Progress/checkpoints change as the job advances; completed request/response evidence is retained rather than overwritten. Window completion is published after its final artifacts, so a checkpoint or partial HTML file alone is not a completion receipt.

Completed resume checks the published artifact bytes as well as call receipts. It refuses an edited output collision instead of overwriting the edit. Keep generated outputs intact for replay and make reviewed corrections through a separately linked revision. Model metadata is rechecked before each new inference, in addition to startup verification.

Keep the directory intact. `artifact:` locations resolve relative to this output directory; original source locations resolve under `--dataset-root`.

```bash
uv run yasargil validate outputs/example/archive.json \
  --dataset-root "/path/to/datasets/SOSpine" \
  --artifact-root outputs/example
```

Validation without source roots cannot grant byte-verified teacher eligibility. Adding `--training` requires completed review, eligibility and partition gates; it should fail on an untouched enhancement draft.

## Exact lineage and its limits

The `llama-cpp-evidence-v1` adapter reconstructs canonical requests from recorded fields, actual image bytes, original CSV labels and coordinates preserved verbatim, versioned prompts/fixed stage questions and validated parent responses. It compares that reconstruction with retained request bytes. It verifies model-metadata consistency, finished response envelopes, evidence references/windows, generated claim/question text, and correspondence between teacher exposure and student frame context.

Requests disable thinking and tool calls and require structured JSON through OpenAI-compatible `response_format` with a JSON schema. Unsupported or unfinished envelopes fail; parsing does not approve medical content. Byte reconstruction establishes internal consistency of saved artifacts, not cryptographic server attestation or a guarantee that the model interpreted its evidence correctly.

All stage proposals remain model-origin claims pending review. The conversation projects the final response; contradicted final events are excluded from that projection, while prior proposals remain in the archive. It may contain a summary and multiple question/answer turns. Its current loss scope is `all_assistant_turns`; [Training](TRAINING.md) shows matching loader/trainer settings. Parent-call text is teacher context, not an automatically exported chain-of-thought trace.

Model agreement is not human verification. Missing views and unanswerable questions remain unresolved. Review must assess the exact conversation against student-visible evidence, including whether teacher-only annotations introduced an unsupported target. Completed-review ingestion, expert assessment and student GPU training remain pending; [Review and evaluation](REVIEW_AND_EVALUATION.md) defines the next steps.
