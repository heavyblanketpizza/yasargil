# Frame selection with complete video context

This workflow proposes frames, then asks local Qwen to review the complete native video and fixed candidate set jointly in one call.
Qwen only keeps or drops the sampled candidates. It cannot request additions or replacements.
The workflow preserves files, timestamps, hashes, exact requests, and decisions.
Selections are provisional, not approved clinical captions or training labels. [Independent MedGemma annotation](MEDGEMMA_FRAME_ANNOTATION.md) consumes the completed selection and source images without receiving Qwen prose.

This Qwen path supplies the video, candidate stills, source locators, and
optional documented procedure background; it does not supply SOSpine's original
tool-label/coordinate CSV rows. New descriptive annotations are written in the
subsequent annotation pass. For why the project moved from Ollama to llama.cpp
and which migration checks remain in progress, see [runtime migration](LLAMA_CPP.md#migration-from-ollama).

After a completed selection, [frame annotation](FRAME_ANNOTATION.md) freezes the
final stills and starts a fresh Qwen session with the complete video. It separates
directly visible observations from contextual claims with source-time evidence,
preserving uncertainty and human-review requirements.

To process every available SOSpine sequence in alternating outcome order, use
the [selection batch](SELECTION_BATCH.md). Each sequence receives an independent
full-video review with the same fixed-candidate rules.

## Run it

From the Yasargil project directory, install optional encoder dependencies if needed:

```sh
uv sync --extra selection
```

Before inference, build the [pinned Qwen complete-video runtime](LLAMA_CPP.md#build-the-qwen-complete-video-runtime).
It is separate from the original release used by the standalone
`scripts/qwen_video.py` experiment. Missing or changed build artifacts stop startup.

Prepare all source frames and embedding candidates without starting Qwen:

```sh
.venv/bin/python -m yasargil select-video-frames \
  --input '/path/to/datasets/SOSpine/frames/S6A3' \
  --released-fps 1 \
  --output-dir outputs/smart_selection/S6A3_keep_drop \
  --procedure-context 'SOSpine: a simulated spine durotomy repair training exercise.' \
  --prepare-only
```

Start Qwen from that prepared run, or resume an interrupted review:

```sh
.venv/bin/python -m yasargil select-video-frames \
  --output-dir outputs/smart_selection/S6A3_keep_drop --resume
```

Prepare and review an original supported constant-rate video in one command:

```sh
.venv/bin/python -m yasargil select-video-frames \
  --input '/path/to/surgery.mp4' \
  --output-dir outputs/smart_selection/original-surgery
```

Use a new output directory outside the source directory. `--released-fps` is
mandatory for released image directories and forbidden for original videos.
Use `--procedure-context` only for verified background, not a desired finding.
Resume uses the saved configuration and verifies the saved source again.

| Setting | Default |
| --- | --- |
| `--candidates` | 24 fixed candidate stills, including eight protected anchors |
| `--context-size` | 131,072 tokens |
| `--image-max-tokens` | 256 visual tokens per image budget |
| `--max-tokens` | 4,096 output tokens per response |
| `--request-timeout-seconds` | 3,600 seconds for the single full-video request |
| `--embedding-backend` | `dinov2` |

Retrieval options are unavailable in this command. A new normal selection run
uses `review_mode: keep_drop` with zero retrieval rounds. The batch command uses
a larger default context and timeout for the longer sequences.

## Encoder setup and selection

The default encoder uses the official
[DINOv2-small checkpoint](https://huggingface.co/facebook/dinov2-small), pinned to
revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`. Install it separately in
`.runtime/models/dinov2-small/` and record its identity in `source.json`.
The expected model directory contains
`config.json`, `preprocessor_config.json`, and `model.safetensors`.
The weights occupy **88,249,960 bytes** and have SHA-256
`ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
Inference loads local safetensors without remote model code or automatic downloads.
Model weights and source images are not included in the repository; obtain the
dataset separately using [Data sources](DATA_SOURCES.md).

The official small [DINOv3 checkpoint](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
requires authorized access. To use supported local DINOv3 ViT Transformers weights,
set `--embedding-backend dinov3 --embedding-model-path /path/to/checkpoint` on a
new run. Missing or mismatched weights fail; DINOv3 never silently becomes DINOv2.

Each full source frame is resized with its aspect ratio intact and letterboxed
for the encoder. It is not center-cropped or stretched. Selection combines a
global DINO embedding, nine pooled patch regions, changes from the previous
frame, and a small clarity preference. Balanced timeline bins keep one busy
period from consuming the whole budget. These are visual heuristics, not
learned judgments of surgical importance.

With 24 candidates, eight uniform temporal anchors are protected, including the first and last frames.
Other candidates compete on diversity and local changes. These 24 include the
eight anchors; the anchors are not eight additional candidates. Every source
frame remains in the complete video even when it is not a candidate still.
Each `.npz` cache entry is keyed by the source image SHA-256 and an encoder
fingerprint covering weight/configuration bytes, preprocessing, library versions,
and compute device. CPU and Apple MPS caches are distinct.

## What complete video context guarantees

Qwen receives one native `input_video` plus the individually identified candidate
stills in one fresh conversation and one review call. All candidates are judged
together against earlier and later video context. A summary never replaces the
video, and the candidate IDs remain frozen throughout the review.

The pinned **b10809-qwen-reference-v1** runtime uses `--video-fps 0`. Upstream still applies
an FPS filter internally, so this flag alone is insufficient evidence of complete
input. This workflow verifies:

1. Every source PTS agrees with the native decoder's constant-rate timeline.
2. Ordered RGB frame hashes from the native filter exactly match an independent
   decode without that filter, with the expected frame count.
3. Each Qwen request logs every expected decoded frame ID, in order.
4. Generation finishes successfully without context truncation.
5. The actual tokenizer stream groups `(0,1)`, `(2,3)`, and subsequent source
   pairs with a mean-time label before every pair. An odd final frame is repeated
   inside its pair only; source counts and timestamps do not gain another frame.

The shared runtime also uses Qwen's explicit non-thinking sampling values:
temperature `0.7`, top-p `0.8`, top-k `20`, min-p `0.0`, presence penalty `1.5`,
and repetition penalty `1.0`. Penalties consider only generated tokens across
the complete output budget. Temperature precedes the top-k/top-p filters.
Requests and effective sampler logs are retained and checked; see
[runtime setup and verification limits](LLAMA_CPP.md#build-the-qwen-complete-video-runtime).

Variable-frame-rate inputs, multiple video streams, and any native filter that
drops, duplicates, or reorders frames are rejected. The source is not retimed to
make it pass. Original video bytes remain unchanged. Released JPEG sequences are
reconstructed using lossless RGB encoding of every decoded source image.

The single-sequence command reserves 131,072 context tokens by default and disables
context shifting. The batch default is 262,144 tokens. If the complete video,
stills, and answer cannot fit, the run fails.
Complete frame coverage does not mean unlimited visual resolution: the model's
image-token budget still limits visible detail. This path processes visual
evidence; it does not add audio transcription.
Matching frame/time-label organization and generation controls does not establish
pixel equivalence with Transformers or better selection accuracy.

## Timestamps, provenance, and keep/drop decisions

Every evidence frame records a stable `frame_id`, zero-based decoded
`frame_index`, `timestamp_ms`, `timestamp_basis`, source path and SHA-256, and
extracted image path and SHA-256. Original-video records retain exact integer
`source_pts`, rational `time_base`, and unnormalized `source_timestamp_ms`;
`timestamp_ms` is the offset from the first source PTS.

SOSpine release records also retain the one-based `release_frame_index`.
Their times are explicitly **reconstructed nominal offsets** from the supplied
cadence. For `N` images reconstructed at `fps` frames per second, the duration is
`N / fps` seconds and zero-based frame `i` starts at `i / fps` seconds. Thus
S6A3's 288 images at 1 fps produce a **288-second (4:48)** video; its last image
starts at **287 seconds (4:47)**. Original source PTS and acquisition times remain
null. These are exact locators within the reconstruction, not verified elapsed
time in the original repair. The release cannot provide images between released
observations. PTS is presentation time, not wall-clock acquisition.

This media timeline governs candidate sampling, Qwen's review, the separate gap
experiment's interval requests, and annotation evidence. Recorded repair time and the outcomes
CSV's video-duration fields do not set or rescale it. Qwen is explicitly told to
use the supplied media offsets and duration, without inferring a 2× correction
or stretching events to fit a recorded repair time. Case outcomes remain
available for separate analysis. The original procedure's elapsed time and
end-to-end coverage remain unknown; complete input coverage means every
available source image was supplied.

Original video inputs retain their verified PTS timeline and duration; they are
not assumed to run at 1 fps. The reconstructed-image formula applies only to
released image directories.

Qwen must judge every sampled candidate ID together and explain each keep/drop
decision. Missing decisions, unknown IDs, and additional-frame requests are
rejected. The response retains a `searches` field for artifact compatibility,
but its schema requires an empty array. The implementation never retrieves a
frame during a normal selection review. A model-requested drop of a protected
anchor is recorded and overridden to retain coverage. No original files are deleted.

The single review ends as completed, model uncertain, or context conflict when
the response is valid; transport or contract failures retain their raw artifacts.
Unclear observations can be explained in the reasons or scene summary, without
creating a second review or expanding the candidates. The separate
[gap experiment](GAP_EXPERIMENT.md) continues to support retrieval for its
missing-evidence experiment.

## Saved evidence and verification status

| Artifact | Contents |
| --- | --- |
| `run.json`, `state.json` | Pinned configuration, review/timing policies, source-manifest hash, conversation, progress |
| `source/source.json` | Complete source timeline, paths, hashes, FFprobe records, preparation commands |
| `initial-selection.json`, `embeddings/` | Candidate scores, protected anchors, encoder identity, feature cache |
| `native-timeline-verification.json` | Source PTS compatibility check |
| `runtime/attempt-*/native-decode/`, `runtime/attempt-*/runtime.json` | Native/unfiltered RGB verification and runtime settings for each server start |
| `rounds/round-00/` | Fixed candidate manifest, exact request/response, coverage and frame-pair/sampling verification, result |
| `selection.json` | Effective selection, model decisions, coverage overrides, provenance, empty unresolved-search list |
| `selection.html` | Human review with thumbnails, keep/drop decisions, timestamps, source paths and hashes |
| `last-error.json` | Most recent failed or interrupted attempt, when present |

Accepted saved results recover without repeating inference after their request,
source coverage receipt, and raw response are checked for consistency. Each server
restart uses a new runtime attempt directory, preserving earlier logs. Completed
resume also regenerates a missing final selection artifact.

The explicit media-timing and keep/drop prompt policies apply to newly prepared runs.
Previously completed selections, gap experiments, and annotations remain
historical records with their original requests and responses. Resuming them
does not apply a new prompt retroactively; use a new output directory to run
with the updated instructions. An incomplete run prepared under the earlier
retrieval policy cannot resume as a keep/drop run. Its saved artifacts remain
intact. New configurations, including the request timeout, are pinned on preparation.

## Validation limits

Automated checks cover a fixed candidate set, a single full-video call,
refusal of additional-frame requests, source integrity, timing discrepancies,
and recovery without repeating accepted inference. These checks establish
execution and artifact consistency; they do not measure surgical selection
accuracy or prove that the model understood every supplied frame.

Evaluate the saved reasons and selected images independently. Supplying procedure
background in a prompt means a matching answer is not an independent procedure
identification test. A successful run, valid receipt, or generic keep reason
does not make a selection an approved caption or training label.
