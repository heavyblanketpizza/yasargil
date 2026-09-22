# Qwen annotation with complete video context

This pass annotates the final frames from a completed `select-video-frames` run.
It starts a fresh local Qwen session with the **complete native video and all
final selected stills together**. It does not inherit Qwen's earlier selection
explanations or conversation. The selected frame list is frozen; annotation
does not select, drop, or retrieve more frames.

The result separates a draft description of what is visible in each still from
claims that depend on the surrounding video. Every contextual claim cites
existing source frames; software resolves those references into bounded time
intervals linked to actual source observations. Source files,
timestamps, frame indices, and hashes remain attached throughout.

**Qwen writes this descriptive text.** The original SOSpine annotations are
instrument labels and coordinates in `sospine_tool_tips.csv` (manual tool tips)
and `sospine_bbox.csv` (computed boxes), not these natural-language descriptions.
This Qwen request receives the video, selected stills, and documented procedure
background; it does not receive those CSV rows. Software attaches the source
filenames, timestamps, hashes, and supporting-frame records to Qwen's output.

This is a separate Qwen annotation baseline. The current
[MedGemma annotation](MEDGEMMA_FRAME_ANNOTATION.md) starts directly from the same
selection and independently inspects source images. It neither waits for nor
receives this Qwen draft. The former Qwen-to-MedGemma review runners have been
removed; their saved outputs remain historical evidence.

## Run it

Use the same local llama.cpp and Qwen installation as
[complete-video frame selection](SMART_FRAME_SELECTION.md). This command consumes
an existing final selection; it does not rerun the embedding selector.
Build the [pinned Qwen complete-video runtime](LLAMA_CPP.md#build-the-qwen-complete-video-runtime)
before starting inference. The old standalone `scripts/qwen_video.py` experiment
does not exercise this processor.

Prepare a new annotation run without starting Qwen:

```sh
.venv/bin/python -m yasargil annotate-video-frames \
  --selection-run outputs/smart_selection/S6A3 \
  --output-dir outputs/frame_annotations/S6A3 \
  --prepare-only
```

Start the prepared run, or resume after an interruption:

```sh
.venv/bin/python -m yasargil annotate-video-frames \
  --output-dir outputs/frame_annotations/S6A3 --resume
```

Omit `--prepare-only` from the first command to prepare and annotate in one run.
Use a new output directory for a new selection, prompt, or configuration.
The annotation pass uses the final selection saved in `--selection-run`.
A separate percentage-drop experiment has its own candidate set and should not
be substituted for the intended selection artifact.

| Setting | Default |
| --- | --- |
| `--context-size` | 131,072 tokens |
| `--image-max-tokens` | 256 visual tokens per image budget |
| `--max-tokens` | 12,288 output tokens |
| `--max-evidence-span-ms` | 10,000 ms per contextual evidence interval |
| `--request-timeout-seconds` | 3,600 seconds; pinned per run |
| `--procedure-context` | Verified background from the parent selection when omitted |

Supply procedure background only when verified. It is context, not proof that
any particular event is visible. The annotation prompt distinguishes that
background from the visual evidence and asks Qwen to report uncertainty.

For the longer S2A2 and S1A2 sequences, use `--context-size 262144` and
`--request-timeout-seconds 21600`. The 256-token image budget and 12,288-token
answer budget stay unchanged. Increasing the context preserves complete-video
coverage; it does not remove source frames. A timeout is a maximum waiting
period, not an expected completion time. Historical saved configurations without
the timeout field continue to mean 3,600 seconds.

## Annotate an S2A2/S1A2 pair

The pair runner requires S2A2 and S1A2 as the first two entries in the selection
queue, with all other entries still pending. It waits until both selections are
complete and the selection worker has released its lock, then annotates them
serially with a fresh full-video conversation for each case and verifies the
saved artifacts.

```sh
.venv/bin/python -m yasargil.annotation_pair \
  --selection-batch outputs/selection_batches/example-batch \
  --output-dir outputs/annotation_pairs/example-pair \
  --prepare-only

.venv/bin/python -m yasargil.annotation_pair \
  --output-dir outputs/annotation_pairs/example-pair --resume
```

The pair's `run.json` pins its scope, settings, annotation prompt, and selection
queue. Its `state.json` records waiting/running/completed status and each case's
result. Each case has its own `S2A2/` or `S1A2/` directory. A case failure is
saved before proceeding to the other configured case; failed calls are not
automatically retried. Explicit resume preserves interrupted evidence and
verifies finished artifacts before skipping completed cases.

After each successful annotation call, `integrity.json` records checksums and
byte sizes for the provenance and model artifacts. The verifier ties annotations
back to the original assistant content and canonical source rows, then verifies
the exact saved bytes on subsequent checks. The source images and reconstructed
video remain at their recorded paths; a manifest does not replace those media.

To pause the pair before its next call, create `.pause-requested` in the pair
output directory. An active call finishes and saves its evidence. Remove that
sentinel before resuming the pair.

## What each annotation contains

| Field | Meaning |
| --- | --- |
| `visible_observation` | What Qwen proposes is directly visible in this selected still. Events elsewhere in the video must not be stated as visible here. |
| `visibility` | `clear`, `partial`, `poor`, or `uninterpretable`. Poor or uninterpretable views require an uncertainty statement. |
| `contextual_claims` | Up to three claims whose interpretation uses the surrounding video. An empty list is allowed. |
| `evidence_intervals` | One to three source-frame ranges for each contextual claim. Qwen supplies `start_frame_id` and `end_frame_id`; software attaches `start_ms` and `end_ms`. |
| `supporting_frames` | Canonical source observations within each interval, attached by the code with complete provenance. |
| `uncertainties` | Ambiguities, limited visibility, or interpretations the model cannot establish. |
| `review_required` / `training_eligible` | Always `true` / `false` for this model draft. |

The model supplies annotation content keyed by the exact selected frame IDs.
The code rejects missing or invented selected IDs and binds accepted content to
the original canonical frame records. For contextual evidence, the request
provides a reference inventory covering **every source frame**, including frames
outside the selected set. Qwen chooses `start_frame_id` and `end_frame_id` from
that inventory instead of generating clock values. The response schema limits
both fields to existing IDs, and the application checks the returned references
again before looking up their authoritative timestamps.

The code rejects unknown references, reversed ranges, and ranges exceeding
`--max-evidence-span-ms`. Interval endpoints are inclusive: a source observation
exactly at either endpoint is included. Choosing the same frame for both ends
creates a point citation with equal timestamps and one supporting observation.
This permits evidence at the final frame without inventing a later endpoint or
implying that a still establishes an event's duration. A range identifies
available evidence, not necessarily the full duration of the claimed action.

New runs use `full-video-frame-annotation-v2`; their derived annotation document
uses `contextual-frame-annotations-v2`. Saved v1 records retain their original
model-written timestamps and remain available for read-only verification and
review. Invalid historical citations are not repaired, clamped, or relabeled as
v2 evidence. Start a separate output directory to obtain new source-reference
annotations.

These checks validate **source locators, not semantic truth**. A claim can cite a
real frame and still be wrong. The output records
`evidence_validation: locator_only_not_semantic` and
`clinical_validation: not_performed`. Context disagreement or uncertainty remains
visible rather than being silently converted into a successful interpretation.

## Complete-video coverage and temporal exposure

The fresh annotation request contains the full video and the frozen selected
stills in one context. Selection reasons and earlier LLM messages are excluded.
The existing native runtime checks all decoded source frame IDs, video hashes,
and context truncation. Complete coverage is an input verification result; it
does not establish that Qwen understood each frame or its timing correctly.
The visual token budget still limits the detail available to the model.
The source-reference inventory also consumes text context. Increasing source
length must still fit the complete video, selected images, inventory, and answer
within the configured context; the reference scheme does not relax the coverage
or truncation checks.

The shared runtime groups adjacent source frames in pairs, with a mean-time
label before each pair: `(0,1)` follows `<0.5 seconds>` for a 1 fps source.
An odd final frame is repeated only to fill its temporal pair; it is not an
additional source observation. The selected stills remain separate. Logs verify
the actual pair/label stream alongside complete-source decoding and truncation
checks. This replaces the original processor's shifted grouping and trailing
ten-second labels.

Non-thinking generation explicitly uses temperature `0.7`, top-p `0.8`, top-k
`20`, min-p `0.0`, presence penalty `1.5`, and repetition penalty `1.0`.
Generated-token-only penalty history covers the output budget, and temperature
is applied before the top-k/top-p filters. See the
[shared runtime settings and limits](LLAMA_CPP.md#build-the-qwen-complete-video-runtime).
These changes align the two identified processing differences; they do not
establish Transformers pixel equivalence or correct interpretations. Source-frame
citations separately prevent nonexistent coordinates, without proving that Qwen
selected the correct moment.

For SOSpine S6A3, the source is all **288 released JPEGs** reconstructed at the
explicit nominal cadence of 1 fps: `288 / 1 = 288` seconds (4:48), with the last
image starting at 287 seconds (4:47). Annotation uses this same media timeline
as selection and retrieval. Recorded repair time and outcomes-CSV duration
fields never rescale evidence intervals; Qwen is told to ignore them for timing.
Original video inputs instead retain their verified PTS timeline and duration.

The original continuous recording, acquisition timestamps, and completeness of
the released repair are unverified. The reconstruction provides exact image
locators, not verified original-procedure elapsed time. No 2× correction is
inferred. Intervals cannot recover images between released observations. See
the [shared timing policy](SMART_FRAME_SELECTION.md#timestamps-provenance-and-keepdrop-decisions).

Every annotation is marked `temporal_exposure: retrospective_full_video` because
Qwen has seen later events while describing earlier frames. This pass does not
create a real-time or past-only supervision example. It also does not grant
human review, clinical validation, or permission to export these drafts as
training data. Existing dataset review and export gates remain separate.

## Inspect, pause, and resume

```sh
.venv/bin/python -m yasargil annotation-status \
  --output-dir outputs/frame_annotations/S6A3

.venv/bin/python -m yasargil pause-annotation \
  --output-dir outputs/frame_annotations/S6A3
```

Pause is cooperative: it lets an active model call finish and save its result.
The first `Ctrl+C` has the same intent; a second interrupts immediately. Resume
checks the pinned source, selected frames, settings, prompt, and saved call
receipts before reusing an accepted response. It does not save a model KV cache.
An unfinished inference call may need to run again after interruption.

Newly prepared runs receive the source-reference citation protocol. Existing
completed annotations retain their original prompts and outputs; resume does
not rewrite them. Historical v1 evidence remains readable, but new inference
requires a new v2 output directory.

Open `report.html` in the output directory to inspect the frozen stills, visible
drafts, contextual claims, uncertainty, supporting observations, and exact call
receipts. The report is local HTML with no remote assets or scripts. It remains
available for prepared, running, paused, failed, or context-conflict runs; pending
annotations are labeled as pending.

| Artifact | Contents |
| --- | --- |
| `run.json` | Configuration and pinned selection/source inputs |
| `session.json` | Identity and pinned configuration for the fresh annotation session; exact messages are in `round-00/request.json` |
| `source/source.json` | Complete canonical source timeline and provenance |
| `selected-frames.json` | Frozen final selected frames with canonical provenance |
| `native-timeline-verification.json` | Source timeline compatibility check |
| `runtime/attempt-*/` | Native decoding verification, runtime metadata, and server logs |
| `round-00/request.json`, `response.json`, `result.json`, `verification.json` | Exact request, raw response, accepted result, complete-video checks, and frame-pair/sampling receipts |
| `annotations.json` | Contextual annotation records, supporting source observations, and review restrictions |
| `integrity.json` | Completion checksums for exact evidence bytes, produced by the scoped pair runner |
| `summary.json` | Current status, selected count, timestamps, and session identity |
| `report.html` | Offline visual review with source and evidence links |
| `last-error.json` | Most recent failure details, when present |

## Validation limits

Automated checks cover fresh conversations, effective final selections, source
integrity, evidence intervals, uncertainty, failure handling, and recovery without
repeating accepted inference. Resume rejects changed runtime settings and
internally inconsistent video-frame receipts. Timing checks cover conflicting
repair metadata, reconstructed frame rates, media-derived duration and interval
limits, source-reference membership and ordering, point citations, and
incompatible saved selection prompts.

Report checks cover model-text escaping, exact source/evidence links, the
distinction between visible observations and video-context claims, frozen
provenance, and incomplete or failed artifacts. A completed model run establishes
execution and recordkeeping, not annotation accuracy. Expert comparison of visible
claims, contextual claims, timestamp accuracy, and uncertainty remains necessary
to measure dataset quality. Inspect your own run's status, receipts, and report;
no generated annotations or completed runs are distributed with this repository.
