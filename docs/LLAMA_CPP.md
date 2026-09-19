# Local Qwen video inference with llama.cpp

This setup runs Qwen3.8-27B on the Mac's GPU and asks it about a complete local
video in one request. It is a first video-inference experiment; it does not yet
replace Yasargil's existing Ollama enhancement pipeline or generate per-frame
training captions.

For the implemented embedding selection and contextual keep/drop workflow, use
[complete-video frame selection](SMART_FRAME_SELECTION.md). It requires every
source frame, validates native decoding and timestamps, and keeps the video in
each follow-up request. The standalone experiment described below samples at its
configured FPS and remains separate.

## Runtime setup

The helper expects a separately installed, pinned llama.cpp runtime in
`.runtime/llama.cpp/b10809/` and matching local model files at the paths below.
Runtime binaries, models, and datasets are not included in the repository.
It does not require an always-on service like Ollama: the helper starts a local server for each run and stops it
afterward.

| Component | Location / version |
| --- | --- |
| llama.cpp | `.runtime/llama.cpp/b10809/` |
| Release selection | Stable `v0.4.0`; its `nightly-tag.txt` points to `b10809` |
| Binary version | `0.4.0-dev`, build `10809`, commit `5266f24da` |
| Release archive | `.runtime/llama-b10809-bin-macos-arm64.tar.gz` |
| Video decoder | Homebrew FFmpeg `9.0.1` (`9.0.1_1` package), `/opt/homebrew/bin/ffmpeg` |
| Model | `.runtime/models/qwen3.8-27b-q4_k_m.gguf` |
| Vision projector | `.runtime/models/qwen3.8-27b-mmproj-bf16.gguf` |

The archive comes from the [official llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b10809).
The [stable release pointer](https://github.com/ggml-org/llama.cpp/releases/tag/v0.4.0)
explains why a stable release can report a binary version ending in `-dev`.

Use a matching model and vision projector. The reference pair identifies itself
as `Qwen3.8 27B 0814`; the model uses `qwen35` architecture and Q4_K_M compression,
and the BF16 projector supplies the vision encoder. Existing compatible Ollama
blobs can be linked at the expected paths to avoid copying large model files.
If using symlinks, retain their backing files while using this setup.

## Run a video

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
