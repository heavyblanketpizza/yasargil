# Local inference with llama.cpp

llama.cpp is Yasargil's only Qwen/MedGemma generation backend. Qwen's complete-video
workflows use a separately pinned, locally patched build; independent MedGemma
annotation retains the original release build. Each workflow owns its local server and stops it
when its inference work finishes; an always-on service is unnecessary.

[Complete-video frame selection](SMART_FRAME_SELECTION.md) requires every
source frame, validates native decoding and timestamps, and preserves the
complete video in its requests. [MedGemma annotation](MEDGEMMA_FRAME_ANNOTATION.md)
sends ordered source stills and optional detail crops using the
OpenAI-compatible chat endpoint with image data URLs and a JSON-schema
`response_format`. These are different evidence protocols on the same backend.
The standalone video experiment below samples at its configured FPS.

## Runtime and model setup

Install the original llama.cpp release in `.runtime/llama.cpp/b10809/`, build the
Qwen complete-video runtime as described below, and supply matching
model/projector pairs at the paths below. Runtime binaries, models and datasets
are excluded from the repository. Obtain them separately under their upstream
terms and verify publisher checksums.

| Component | Location / version |
| --- | --- |
| Original llama.cpp release | `.runtime/llama.cpp/b10809/` |
| Qwen complete-video build | `.runtime/llama.cpp/b10809-qwen-reference-v1/bin/llama-server` |
| Release selection | Stable `v0.4.0`; its `nightly-tag.txt` points to `b10809` |
| Original binary version | `0.4.0-dev`, build `10809`, commit `5266f24da` |
| Patched Qwen identity | Same base commit, with `qwenref1` in its version and a hashed build receipt |
| Release archive | `.runtime/llama-b10809-bin-macos-arm64.tar.gz` |
| Video decoder | Homebrew FFmpeg `9.0.1` (`9.0.1_1` package), `/opt/homebrew/bin/ffmpeg` |
| Qwen alias | `qwen3.8-27b` |
| Qwen model | `.runtime/models/qwen3.8-27b-q4_k_m.gguf` |
| Qwen vision projector | `.runtime/models/qwen3.8-27b-mmproj-bf16.gguf` |
| MedGemma alias | `medgemma-27b` |
| MedGemma model | `.runtime/models/medgemma-27b-q4_k_m.gguf` |
| MedGemma vision projector | `.runtime/models/medgemma-27b-mmproj-f16.gguf` |

The archive comes from the [official llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b10809).
The [stable release pointer](https://github.com/ggml-org/llama.cpp/releases/tag/v0.4.0)
explains why this pinned release can report a binary version ending in `-dev`.

Use a matching model and vision projector. The Qwen reference pair identifies
itself as `Qwen3.8 27B 0814`; the model uses `qwen35` architecture and Q4_K_M
compression, and the BF16 projector supplies the vision encoder. MedGemma needs
its own matching model and F16 projector. A model file alone is insufficient
for an image workflow.

Inspect the installation from the project directory:

```sh
uv run yasargil models
```

For the ordered-image commands, `--project-root /path/to/yasargil` locates
`.runtime/` when running elsewhere; `--timeout` sets the request deadline.
`models` inspects local model/projector metadata and hashes and the runtime
identity. It does not download weights or demonstrate successful image inference.
Exact requests, complete response envelopes, model and projector hashes, and
runtime identity are retained with each run.

MedGemma uses the pinned runtime's built-in Gemma formatter with
`--no-jinja --chat-template gemma`. This avoids a Jinja grammar-prefill failure
while preserving image markers and JSON-schema constraints. Qwen retains its
model-provided Jinja template. These launch settings are part of runtime identity.

## Build the Qwen complete-video runtime

Selection, annotation, and other callers of `LocalVideoRuntime` require the
separate `b10809-qwen-reference-v1` build. On the supported Apple Silicon setup,
the builder needs CMake and the Xcode command-line compiler tools. If CMake is
not already installed, it can be installed inside the project:

```sh
uv pip install --target .runtime/build-deps cmake==3.31.6
.venv/bin/python scripts/build_qwen_runtime.py --jobs 8
```

The builder also accepts `--cmake /path/to/cmake`; use
`.venv/bin/python scripts/build_qwen_runtime.py --help` for its options. It
downloads the pinned upstream source archive, checks its SHA-256, applies
`scripts/runtime_patches/qwen-reference-v1.patch`, and builds the server with
Metal and video support. It leaves the original release and model files in place.
Before reusing extracted sources, it checks every source file against a fresh
archive-plus-patch copy. Local differences stop the build and are preserved.
`build-receipt.json` records the source and patch identity and built artifact
hashes. Startup rejects a missing, outdated, or changed build rather than falling
back to the original processor. `runtime.json` retains the accepted build receipt;
the ordinary `models` inspection command still describes the release/image path.

The patch changes two parts of the shared Qwen complete-video path:

- **Frame pairs and time labels:** adjacent source frames form `(0,1)`, `(2,3)`,
  and so on. Every pair is preceded by its mean media timestamp, formatted like
  `<0.5 seconds>` for the first pair at 1 fps. An odd final frame is repeated
  inside its temporal pair, without creating an additional source frame or
  extending the timeline. Selected stills remain separate. The old trailing
  labels at ten-second boundaries are not used in this path.
- **Generation sampling:** requests explicitly set temperature `0.7`, top-p
  `0.8`, top-k `20`, min-p `0.0`, presence penalty `1.5`, and repetition penalty
  `1.0`, following [Qwen's non-thinking recommendation](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices).
  Frequency penalty is `0.0` and the seed remains `42`. Penalty history contains
  generated tokens only and covers the complete output budget. The sampler
  applies penalties, then temperature, then top-k/top-p/min-p filters.

The shared complete-video runtime now enables thinking at both server startup
and request level. This experiment keeps the numerical sampling values above,
video processing, prompts, schemas, and token budgets unchanged; it does not
switch to Qwen's recommended thinking sampling profile. Reasoning is saved in
the raw response's separate `reasoning_content` field; only final `content` is
validated as the JSON answer. Reasoning and the final answer share `max_tokens`,
and an unfinished answer remains a rejected result.

Per-request logs and verification receipts check the actual pair/label stream
and effective generation controls. These checks target the two identified
mismatches; they do not establish pixel-for-pixel equivalence with Transformers,
identical results across inference backends, or improved surgical accuracy.
Image-token budgets and all source-frame coverage checks remain in force.
Use a fresh output directory for comparisons; historical requests and outputs
retain their original processing and settings.

The focused live check uses tiny synthetic clips and a separate still:

```sh
.venv/bin/python scripts/runtime_tests/live_qwen_video.py --output outputs/runtime_checks/new-live-check
```

Choose an unused output directory. This loads the real local Qwen model, checks
even/odd frame counts and a follow-up conversation, and saves raw responses and
verification receipts. It is not an annotation-quality evaluation.

## Migration from Ollama

Qwen and MedGemma inference uses llama.cpp. DINO frame embeddings and
Hugging Face/Unsloth training keep their separate execution paths. The inference
backend provides:

- **Native video input for Qwen.** Selection and annotation can send an
  `input_video` item and selected stills in the same request. The former Ollama
  integration in this project sent image lists; it did not exercise this native
  video path. This is a distinction between Yasargil's implemented adapters,
  not a claim about every capability of Ollama.
- **Direct decoding and context controls.** The pinned runtime exposes video
  sampling/timestamp settings, visual-token budgets, context capacity, and
  decoder logs. The complete-video workflows compare decoded frames with the
  source inventory and reject truncation; the runtime disables automatic fitting
  and context shifting. These controls help audit what was supplied, without
  proving that the model understood it or cited the correct event time.
- **One local inference backend.** Independent MedGemma annotation uses the
  same runtime family as Qwen video. The application owns server
  startup/shutdown, uses explicit GGUF model/projector files, and retains requests,
  responses and runtime identities for inspection.

This migration has no established speed or annotation-quality advantage from a
controlled comparison. Whole-video processing remains expensive, generated
temporal citations have been unreliable, and early MedGemma reviews have made
few substantive corrections. See the [research status](../README.md#research-status-and-known-limitations).
Those protocol and quality investigations remain ongoing after the backend change.

New [frame annotation](FRAME_ANNOTATION.md) requests address invented citation
coordinates by asking Qwen for existing source-frame IDs, then resolving their
timestamps in application code. The full source-reference inventory and enum
constraints supplement the complete native video and selected stills. The
runtime still uses `--video-fps 0` and verifies every source frame. The later
[Qwen runtime patch](#build-the-qwen-complete-video-runtime) separately corrects
frame grouping, time-label placement, and generation controls. Source-frame
citations prevent invented coordinates; they do not establish event localization.
An existing frame can still be the wrong evidence for a claim.

There is no Ollama client, backend selector or automatic fallback in the active
inference path. The former `--ollama-url` and `models --pull` options are removed.
Keep model/projector files independently under `.runtime/models/`; do not rely
on symlinks into an Ollama-managed blob cache if that installation will be removed.
A compatible existing GGUF pair can be copied into these paths without downloading
the same weights again. A combined Ollama Gemma3 file is not a ready-to-use pair:
it requires an explicit, verified conversion of metadata and embedded vision
tensors. Confirm model inspection and image inference before removing any
separate runtime or cache installation. The migration does not uninstall the
Ollama application or delete its cache.

Historical Ollama requests, responses and archives retain their original bytes
and provenance. They are not converted or relabeled as llama.cpp calls, and an
Ollama run cannot resume under the new backend. Start a new output directory.
Historical ordered-image archives retain their `llama-cpp-evidence-v1` or
`ollama-evidence-v1` teacher adapters. These read-only verifiers reconstruct
original requests against model, source, and runtime records; they are not
independent-annotation inference or resume paths. Verified
historical records can still pass export only after all ordinary human-review,
eligibility and partition gates. Backend migration grants no review approval.

## Run a video

This standalone experiment still uses the **original b10809 release** and its
older processor/settings. It does not exercise the patched complete-video
selection/annotation path described above.

From the project directory:

```sh
.venv/bin/python scripts/qwen_video.py '/path/video.mp4' \
  --output-dir outputs/llama_cpp/my-video
```

To supply your own question, save it in a text file and add
`--prompt-file /path/question.txt`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--fps` | `1` | Sample one frame per second across the video's timeline |
| `--image-max-tokens` | `256` | Limit visual detail per processed image |
| `--context-size` | `65536` | Context capacity for the request and response |
| `--max-tokens` | `1200` | Maximum generated tokens |
| `--port` | `8081` | Local server port |

The helper starts `llama-server` on localhost, enables video decoding, and makes
the video's parent directory available through `--media-path`. It sends an
`input_video` item with a relative `file://` URL. llama.cpp reads the video from
disk, uses FFmpeg to decode it, and supplies sampled frames and timing to Qwen's
video-processing path. The response and diagnostic logs are saved in the output
directory, and the helper shuts down its server. The video stays on this Mac.
This path supplies visual frames and timestamps; it does not transcribe audio.

Use a new output directory for each run. `response.md` contains the answer;
`response.json` retains token counts and timings. `verification.json` compares
the model decoder's frame IDs with an independent complete FFmpeg decode at the
same sampling rate, so a successfully decoded prefix cannot pass as the whole
video. `server.log`, `request.json`, and `run.json` preserve the run details.

This file-based request matters for longer videos: the tested CLI route embeds
the video as base64 in an HTTP request, which can exceed the server's default
100 MiB payload limit. Base64 also makes the payload larger than the original
file. A local file URL avoids transporting the video bytes in that request.

## What “whole video” means here

One request covers the complete timeline at the selected sampling rate. It does
not mean every original frame is examined at its original resolution. At the
default 1 fps, a 30 fps source supplies about one out of every 30 frames. Lower
sampling can miss brief events; lower image resolution can hide small details.

More frames and more detail consume more context and processing time. The
model's GGUF declares a 262,144-token context; the helper starts at 65,536. A
longer surgery may exceed even the model limit. Raising `--context-size` does
not make video length unlimited, and each run's logs should be checked for
actual frame coverage, visual tokens, and successful completion.

For a released SOSpine image sequence, reconstruct the input using every JPEG
in release order at the explicit nominal cadence. For example, 288 images at
1 fps produce 4 minutes 48 seconds of playback. This cannot restore movement
between released images or original acquisition timestamps. Obtain the source
from the publisher as described in [Data sources](DATA_SOURCES.md).

## Integrity and validation limits

Verify downloaded runtime and model files against the publisher's checksums,
and retain their identities with your local run. Inspect `verification.json`
for expected frame coverage and generation completion, and read the raw model
answer before treating any description as useful evidence.

A completed response establishes that the input route executed. Correct input
coverage does not establish correct procedure identification, medical accuracy,
or understanding of every frame. Expert review remains necessary before using
model output as captions or training labels. Runs, raw answers, and review notes
are local artifacts and are not included in this repository.

## Large native-video inputs

The selection and annotation runtime uses a project-local transport adapter
when `.runtime/ffmpeg-safe/transport.json` is present. It works around
FFmpeg `cache:pipe:0` failures with large reconstructed videos. The adapter copies every input byte to a private temporary file and
passes that seekable file to the same FFmpeg decoder. It does not re-encode
the video or change its frames, timestamps, resolution, or sampling options.

The installation pins the adapter, launchers, Python executable, and real
FFmpeg/FFprobe binaries by hash. Startup rejects changed components. Both the
full-frame preflight and llama.cpp use the same adapter. Preflight compares
every decoded RGB frame in order; each model request additionally records the
complete input's byte count and SHA-256 under its runtime's
`transport-receipts/` directory. An accepted request must match the original
video and still pass the native frame-coverage and context-truncation checks.
Temporary input files are removed after decoding. Historical failed attempts
remain in the output directory alongside fresh retry attempts.
