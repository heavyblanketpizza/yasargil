# Visual dataset inspector

Run the local inspector from the repository:

```sh
.venv/bin/python -m yasargil inspect-dataset
```

Open **http://127.0.0.1:8765**. Use `--port` to choose another local port and
`--runs-root` to inspect a different output directory. The default is `outputs`.
The inspector reads saved selections, independent MedGemma annotations, optional
Qwen baseline annotations, and historical MedGemma reviews. It does not start inference. Refresh discovers newly saved work.

Links from another app or website may open the inspector home page. The server
allows top-level navigation to that static page while retaining same-origin
checks for dataset APIs, media, exports, and curation actions.

The folder icon between **Refresh** and **Export review** opens the configured
output directory in the system file manager (`outputs` by default). Selection,
Qwen, and MedGemma JSONs retain absolute source-image paths, derived-image paths,
timestamps, and hashes. They reference the original files rather than embedding
the source dataset; external originals require their drive to be connected.
Human annotation edits live separately in `.inspector-curation`. Review notes
remain browser-local until exported through **Export review**.

## Inspect a surgery

Choose a record at the top. Open **Case data** above the timeline to inspect the
surgery outcome and original case fields. The thin horizontal gray line covers the
entire supplied media timeline, including source frames that were never selection
candidates. Borderless colored circles sit on the line at individual frame
timestamps.

- **Green circles:** final selected key frames.
- **Orange circles:** frames Qwen dropped, except explicitly retained
  coverage anchors. An anchor stays green and its original drop recommendation
  remains visible in the selection explanation.
- **Gray line:** the remaining source video, including candidates awaiting a
  decision. These frames remain accessible by clicking the timeline.

Click a colored circle to inspect that exact frame. Click or drag along the
line to inspect other source frames. Arrow keys move one frame; Shift plus arrows
move between selected key frames. The frame-number field jumps directly to a frame. The
timeline also supports Home/End when focused.

The frame viewer shows the canonical image. **9-frame video** plays the four
source frames before the target, the target itself, and four after it. Near a
sequence boundary, the button shows the smaller available frame count. Playback
starts at the beginning of this window and stops at its end. The annotation and
human review target remain fixed while the video plays; the caption identifies
the displayed range and target frame.

Selected-frame badges include the saved selection framework,
such as **DINOv2 + Qwen**; the tooltip separates candidate selection, full-video
review, and protected-anchor retention. Missing framework metadata is labeled
unrecorded.
Context clips preserve the recorded frame cadence (nominal cadence for released
image sequences). They are generated from the nearby canonical images with local
FFmpeg, verified for frame count and timing, and cached in the system temporary
directory. Playing a window does not wait for a full-surgery conversion. The
browser-compatible display derivative does not replace the original video or
canonical stills. Missing images, unavailable FFmpeg, and playback failures return
to the still view with an error; loading also has a bounded timeout.

## Read attached evidence

Selected frames show **Independent MedGemma annotation** when a matching new run
exists. This result attaches directly to its selection; no Qwen annotation is
required. An independent result takes precedence over historical MedGemma review
results for the same selection. A Qwen annotation block appears when a saved
Qwen baseline exists. Original source frames and human review remain available
when no model annotation has been saved.

The independent annotation separates **target-visible** claims from
**context-supported** interpretations. Each claim shows its category, statement,
evidence-view links, and specific uncertainty. Unresolved questions identify the
missing evidence. Thumbnails show the full target, its deterministic detail crops,
and nearby source observations; crop bounds identify the original pixels. Crops
are views of one observation, not extra temporal observations. Evidence links
resolve to the images actually supplied, not to a model's invented timestamp.

Saved older outputs are marked **HISTORICAL REVIEW** and retain their original
Qwen comparison, revisions, corrections, and deferred evidence fields. They are
not relabeled as independent annotation. Missing or pending annotations are never
filled with generated content. Exact request, response, annotation, and model
artifacts are accessible from the frame's source data.

Model additions are labeled **AI-generated annotations**. Original dataset fields
and human notes retain their separate origins. The interface uses Meslo when
installed locally, with a system monospace fallback; fonts are not downloaded.

Current MedGemma input contains source images, nearby observations, optional
target crops, and documented procedure context. It excludes Qwen prose, source
CSV labels, outcomes, and surgeon experience. Source labels and outcomes visible
in this inspector are retrospective information for the human reviewer; they are
not evidence that the current model received them. The separate Qwen baseline
uses the complete video. Historical MedGemma review packets had different inputs,
including Qwen drafts and matching source labels. See
[independent annotation](MEDGEMMA_FRAME_ANNOTATION.md) and
[historical review protocols](MEDGEMMA_FRAME_REVIEW.md).

## Edit or remove an enhancement

The pencil and trash icon buttons beside the annotation comparison apply to the
current frame's enhanced record. The pencil opens a human editor for each available
model annotation. Save a revision to show your edited text under
**Human-edited annotation**; expand **Original AI-generated annotation** to
compare it with the model output and its evidence. Missing model results
cannot be edited until they exist.

The trash icon deletes the frame's enhancement from the curated view. A restore
icon brings it back, including prior human edits. The timeline marks this with
hatching inside the frame's circle, preserving its original selection color.
Original images, dataset rows, and model outputs remain available in Source data.

Human edits and deletion state are saved in the workspace, tied to the exact
dataset revision. They survive a server restart and are included in review
exports. They are draft curation, not an automatic training approval. A change
to an enhancement resets its previous Reviewed assessment while keeping notes.

**Source data** contains original manual tool-tip rows (`sospine_tool_tips.csv`)
and computed bounding-box rows (`sospine_bbox.csv`). These contain instrument
labels and coordinates, separate from Qwen and MedGemma's generated prose.
Preserving exact rows and CSV locators establishes their source, not their
correctness. Source data also contains exact frame provenance and expandable
complete records. Raw model responses and run artifacts
have direct links. Downstream runs are matched by source identity, parent paths,
selection hashes, and frozen annotations so unrelated gap experiments do not
supply annotations to a normal selection.

The compact **Case data** dropdown above the timeline contains the surgery
outcome and full case record. For SOSpine it shows the recorded leak test, repair
time, and training-year metadata alongside every original outcome-table column
with its CSV locator. These are simulated cadaveric technical outcomes. Outcomes
are loaded only into the human
inspector and are never added to model inference requests.

The inspector infers a SOSpine root only from an exact case/source layout and
matching filenames. `--dataset-root` can supply an explicit matching root.
Missing or ambiguous case metadata remains unavailable.

The source manifest sets the timeline. Outcome-table frame counts, declared
video duration, and repair duration can differ from the supplied media. Original
CSV values remain visible as metadata and never stretch or shorten playback.

## Keep human notes

**Human review**, below the model comparison, offers three compact icon decisions:
a clock for **Unfinished**, an alert for **Needs attention**, and a check for
**Complete**. Choose a decision and
optionally add a note, then click **Save review**. The saved review shows its
decision and note with a pencil icon to **Edit**. In edit mode, **Save changes**
commits the revision and **Cancel** returns to the saved review.

Typing or choosing a decision does not save automatically. Unsaved drafts stay
with their frame while navigating in the open page; save before closing it.
Only saved decisions affect completion counts and review exports. An unfinished
review can be saved with an empty note. Existing saved reviews remain editable.

Reviews are saved in this browser and bound to a fingerprint of the exact
source, selection, annotations, review results, and displayed source tables.
A changed dataset revision does not silently inherit an earlier Reviewed state.
Editing a review, focusing its note field, or choosing a decision pauses video
so notes stay attached to one frame.

**Export review** downloads a JSON draft worksheet with the record, revision
identity, per-frame notes, enhancement edits/deletions, and source references.
Deleted enhancements are explicitly marked `included: false`. Export notes before clearing
browser storage or switching browsers. This worksheet does not replace the
existing formal review or training-export gates.

## Local operation

The interface uses bundled HTML, CSS, and JavaScript without remote dependencies.
Current generation code uses local llama.cpp; historical saved Ollama artifacts
retain their original runtime identity. The inspector itself only reads these
artifacts. See [runtime setup and compatibility](LLAMA_CPP.md#migration-from-ollama).
The server binds only to `127.0.0.1`, serves explicitly registered media/artifacts,
supports video byte ranges, and saves human curation separately from source
artifacts. Source drives must
remain mounted. Stop the inspector with `Ctrl+C`.
