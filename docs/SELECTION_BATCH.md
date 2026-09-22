# Complete-video SOSpine selection batch

`select-sospine-batch` processes every available released SOSpine image sequence
serially. For each sequence, it reconstructs a native video from all released
images at 1 fps, proposes 24 DINOv2 candidates, and asks Qwen to keep or drop
those candidates in one joint review. Eight of the 24 are protected timeline
anchors. Qwen sees the complete supplied video and all candidate stills together;
it cannot request additional frames or replacements.

Each selection uses Qwen through local llama.cpp and excludes the original
tool-label/coordinate CSV rows from its request. The later annotation pass
writes new descriptive text. See [runtime setup and compatibility](LLAMA_CPP.md#migration-from-ollama)
for the switch from Ollama and installation requirements.

The inspected release contains 24 sequences. Obtain a separate copy as described
in [Data sources](DATA_SOURCES.md). Complete input coverage means
every released image in each sequence, not verified coverage of an entire
original repair. The batch performs selection; the separate
[annotation pass](FRAME_ANNOTATION.md) can subsequently consume a completed
selection.

## Ordering the queue

SOSpine records `Leak At 40mmHg`, a technical outcome from a simulated cadaveric
repair. It does not record patient recovery or provide a unique best-to-worst
recovery ranking. The default queue therefore:

1. Alternates a no-leak (`N`) trial with a leak (`Y`) trial.
2. Uses natural trial-ID order within each group to resolve equal outcomes.
3. Continues the remaining known-outcome group if the other group runs out.
4. Places sequences with unknown outcomes last, also in trial-ID order.

`Clip0` and `Clip1` have no verified outcome mapping and remain included at the
end. Outcome-table entries without an available image sequence are recorded as
excluded rather than fabricated. The frozen `queue.json` preserves the exact
order, endpoint, metadata-row provenance, and source inventory.

The optional `--tie-break repair_time` uses shorter recorded repair times first
within the no-leak group and longer times first within the leak group, with
trial ID resolving remaining ties. This is an explicit scheduling preference,
not a measured patient-recovery rank. The default `trial_id` does not use repair
time to resolve ties.

Outcomes order jobs only. Qwen receives the same documented procedure background
for every case, without leak labels, repair times, or per-frame dataset
annotations. Frame selection and evidence timestamps use the supplied media
timeline regardless of the queue's tie-break choice.

## Prepare and run

From the project directory, freeze and inspect the queue without preparing video
or loading Qwen:

```sh
.venv/bin/python -m yasargil select-sospine-batch \
  --dataset-root '/path/to/datasets/SOSpine' \
  --output-dir outputs/selection_batches/sospine_keep_drop \
  --prepare-only
```

Start that saved batch, or resume it after an interruption:

```sh
.venv/bin/python -m yasargil select-sospine-batch \
  --output-dir outputs/selection_batches/sospine_keep_drop --resume
```

Omit `--prepare-only` from the first command to prepare the queue and start in
one invocation. New batches require a new output directory outside the dataset.
The source directory must remain available throughout the run. Running this foreground command does not
create a scheduled job or detach it from the terminal.

| Setting | Batch default |
| --- | --- |
| `--candidates` | 24, including eight protected anchors |
| `--context-size` | 262,144 tokens |
| `--image-max-tokens` | 256 visual tokens per image budget |
| `--max-tokens` | 4,096 output tokens |
| `--request-timeout-seconds` | 21,600 seconds, or six hours per model call |
| `--tie-break` | `trial_id` |

The timeout is a ceiling, not an estimated runtime. Every case gets a fresh
local Qwen session. The runtime verifies every source frame and refuses context
truncation; it does not shorten a sequence or silently reduce source coverage
to fit. A case failure is recorded before proceeding to the next queued case.

## Check, pause, and resume

```sh
.venv/bin/python -m yasargil selection-batch-status \
  --output-dir outputs/selection_batches/sospine_keep_drop

.venv/bin/python -m yasargil pause-selection-batch \
  --output-dir outputs/selection_batches/sospine_keep_drop
```

Pause takes effect between cases: the current case can finish and save before
the batch stops. Wait for `paused` and `writer_active: false`, then use the
`select-sospine-batch --resume` command above. A first `Ctrl+C` requests a
cooperative stop; a second interrupts immediately.

Resume verifies the frozen queue and source inventory, saved configuration,
and completed selection hashes. Completed and `needs_review` cases are not
rerun. Failed cases are retried on explicit resume; accepted results from an
interrupted case can be recovered after verification. An interrupted preparation
is preserved before a replacement attempt starts. These checkpoints save
artifacts, not an in-memory model cache.

## Timing and saved evidence

At 1 fps, a sequence with `N` images lasts `N` seconds, and zero-based frame `i`
starts at `i` seconds. These are reconstructed nominal offsets. For example,
288 images produce 4:48 of playback with the final image starting at 4:47.
Recorded repair times and CSV video-duration fields never rescale this timeline.
See [source and decoder verification](SMART_FRAME_SELECTION.md).

| Artifact | Contents |
| --- | --- |
| `queue.json` | Frozen order, outcome metadata provenance, actual image inventories and playback durations |
| `run.json` | Pinned batch configuration and queue hash |
| `state.json`, `summary.json` | Per-case progress, errors, completed selection hashes and counts |
| `report.html` | Queue, status, playback times and links to available selection reports |
| `runs/NN-CASE/` | Each case's independent source, embeddings, exact Qwen request/response, verification and selection report |
| `interrupted-preparations/` | Preserved incomplete preparation artifacts, when present |

Final batch status is `completed` only when every case completed normally;
otherwise it is `completed_with_issues`. Model uncertainty or context conflict
appears as `needs_review`. Even a completed case remains a provisional selection,
not a clinical accuracy assessment or an approved training example.

The [gap experiment](GAP_EXPERIMENT.md) is a separate command path and retains
its missing-evidence retrieval behavior. It does not change this batch's fixed
candidate set.
