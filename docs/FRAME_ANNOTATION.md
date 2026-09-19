# Frame annotation with complete video context

This pass annotates the final frames from a completed `select-video-frames` run.
It starts a fresh local Qwen session with the **complete native video and all
final selected stills together**. It does not inherit Qwen's earlier selection
explanations or conversation. The selected frame list is frozen; annotation
does not select, drop, or retrieve more frames.

The result separates a draft description of what is visible in each still from
claims that depend on the surrounding video. Every contextual claim cites
bounded time intervals linked to actual source observations. Source files,
timestamps, frame indices, and hashes remain attached throughout.

After this pass finishes, [MedGemma frame review](MEDGEMMA_FRAME_REVIEW.md) can
revise each draft using the key frame, before/after observations, cited supporting
frames, and original dataset annotations. Its further evidence requests are
saved for later; that stage does not dispatch Qwen or TimeLens2 searches.

## Run it

Use the same local llama.cpp and Qwen installation as
[complete-video frame selection](SMART_FRAME_SELECTION.md). This command consumes
an existing final selection; it does not rerun the embedding selector.

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

## Automatic annotation of the first two selections

The scoped runner waits until both S2A2 and S1A2 have completed keep/drop
selection and the selection worker has released its lock. It then annotates
them serially, each with a fresh full-video conversation, and verifies the saved
artifacts. The remaining 22 sequences stay pending.

```sh
.venv/bin/python -m yasargil.annotation_pair \
  --selection-batch outputs/selection_batches/SOSpine_keep_drop_20260915 \
  --output-dir outputs/annotation_pairs/SOSpine_first_two_20260915 \
  --prepare-only

.venv/bin/python -m yasargil.annotation_pair \
  --output-dir outputs/annotation_pairs/SOSpine_first_two_20260915 --resume
```

The pair's `run.json` pins its scope, settings, annotation prompt, and selection
queue. Its `state.json` records waiting/running/completed status and each case's
result. Each case has its own `S2A2/` or `S1A2/` directory. A case failure is
saved before proceeding to the other authorized case; failed calls are not
automatically retried. Explicit resume preserves interrupted evidence and
verifies finished artifacts before skipping completed cases.

After each successful annotation call, `integrity.json` records checksums and
byte sizes for the provenance and model artifacts. The verifier ties annotations
back to the original assistant content and canonical source rows, then verifies
the exact saved bytes on subsequent checks. The source images and reconstructed
video remain at their recorded paths; a manifest does not replace those media.

To pause the pair before its next call, create `.pause-requested` in the pair
output directory. An active call finishes and saves its evidence. Remove that
sentinel before an explicitly requested resume. The separate hourly notification
checks do not control the model's inference rate.

## What each annotation contains

| Field | Meaning |
| --- | --- |
| `visible_observation` | What Qwen proposes is directly visible in this selected still. Events elsewhere in the video must not be stated as visible here. |
| `visibility` | `clear`, `partial`, `poor`, or `uninterpretable`. Poor or uninterpretable views require an uncertainty statement. |
| `contextual_claims` | Up to three claims whose interpretation uses the surrounding video. An empty list is allowed. |
| `evidence_intervals` | One to three bounded source-time intervals for each contextual claim. |
| `supporting_frames` | Canonical source observations within each interval, attached by the code with complete provenance. |
| `uncertainties` | Ambiguities, limited visibility, or interpretations the model cannot establish. |
| `review_required` / `training_eligible` | Always `true` / `false` for this model draft. |

The model supplies annotation content keyed by the exact selected frame IDs.
The code rejects missing or invented selected IDs and binds accepted content to
the original canonical frame records. It also rejects inverted, out-of-range,
overlong, or empty evidence intervals. Interval endpoints are inclusive: a
source observation exactly at either endpoint is included.

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

Newly prepared runs receive the explicit media-timing instructions. Existing
completed annotations retain their original prompts and outputs; resume does
not rewrite them. Use a new output directory for an annotation run with the
updated prompt.

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
| `round-00/request.json`, `response.json`, `result.json`, `verification.json` | Exact request, raw response, accepted result, and complete-video checks |
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
limits, and incompatible saved selection prompts.

Report checks cover model-text escaping, exact source/evidence links, the
distinction between visible observations and video-context claims, frozen
provenance, and incomplete or failed artifacts. A completed model run establishes
execution and recordkeeping, not annotation accuracy. Expert comparison of visible
claims, contextual claims, timestamp accuracy, and uncertainty remains necessary
to measure dataset quality. Inspect your own run's status, receipts, and report;
no generated annotations or completed runs are distributed with this repository.
