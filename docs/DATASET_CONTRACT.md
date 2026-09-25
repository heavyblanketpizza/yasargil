# Dataset contract — enhancement v2

**Updated 2026-09-22. Status: v2 source import, historical teacher verification,
and export foundations remain available.** Active annotation now uses
[independent MedGemma](MEDGEMMA_FRAME_ANNOTATION.md); the Qwen-conditioned
review/enhancement runners have been removed. New annotation artifacts are
separate from this archive/export adapter. Completed-review ingestion, expert
assessment, and downstream training experiments remain pending. See
[Data sources](DATA_SOURCES.md) for acquisition and repository exclusions.

The active schema is [enhancement-record.schema.json](../schemas/enhancement-record.schema.json). Runtime code lives in [src/yasargil](../src/yasargil/). The CLI exposes `models`, `annotate-selected-frames`, `import-sospine`, `validate`, `review-packet`, and `export`; use the locked environment through `uv sync --frozen` and `uv run yasargil`.

## Product and scope

Yasargil enriches surgical datasets with traceable annotation proposals, reviewed claims, and reusable training views. The first adapter is SOSpine: microscope images of simulated spinal durotomy repair in a perfused cadaveric setting. Source data remain authoritative about what was released. A generated description is a proposal about that evidence; a review records an assessment of a particular proposal and its inputs.

The immediate product is an enhanced research dataset. A clinical assistant, patient risk model, or working real-time guidance system is a separate outcome that this contract does not establish. SOSpine provides neither patient recovery outcomes nor validated per-frame surgical recommendations. Its original record describes recordings downsampled from 30 fps to a 1-fps JPEG release. [SOSpine data descriptor](https://www.nature.com/articles/s41597-023-02744-5.pdf)

The deterministic baseline accepts 1–32 unique, increasing release-frame indices for one case. It records `explicit_indices` and re-expresses positive grasper/needle-driver labels from exact tool-tip rows; missing labels do not become absence claims. Computed box rows retain their original origin. This baseline performs no model inference or action interpretation. Historical enhancement archives used a uniform sample and optional model-requested reinspection. Their removed runtime supplied Qwen proposals to MedGemma. The baseline and historical model claims remain pending with their partition unassigned.

## Evidence archive and training messages

One v2 record is an **evidence archive**. It retains original annotations, selected assets, generation exposure, claim-level provenance, review, measured case outcomes, and a training view. The archive supports audit and regeneration; it is not sent wholesale to a learner.

`training_view.messages` is a deliberately constructed conversation with typed image and text blocks. It may contain synthetic questions and answers created for training. These are not transcripts of surgeons speaking, and generated explanations are not recovered surgeon thoughts. The exporter projects **messages only**, with a separate receipt linking the output to archive revisions, the active schema digest, conversation and media hashes, partition, intended use, and loss scope. Reviewed training export enforces its eligibility gates; a separately marked preview only demonstrates format. The loader verifies the receipt and image bytes, then hydrates local references as PIL images without injecting archive metadata into the conversation.

Asset locations are relative to the source dataset root. A location beginning `artifact:` resolves relative to a separately supplied artifact root. The resolver rejects remote URLs, absolute paths, traversal, and symlink escapes. This convention gives generated or derived artifacts their own storage without rewriting the source dataset. Source assets retain their hashes and identity; creation of a new artifact does not change their provenance.

| Archive fields | Responsibility |
|---|---|
| `schema_version`, `record_id`, `revision`, `status`, `intended_use` | Identify the versioned record and bounded intended use. Record status is `draft`, `in_review`, `reviewed`, or `rejected`. |
| `source`, `assets`, `original_annotations`, `source_conflicts` | Preserve release identity, source bytes, exact annotation locations, original values, and unresolved discrepancies. |
| `frame_selection`, `transformations`, `generation_runs` | Record why inputs were selected, how they changed, and what a teacher actually received and returned. |
| `claims` | Link each imported or proposed statement to evidence, provenance, review disposition, and the text it supports. |
| `reviews`, `enrichment_quality` | Record actual review decisions and metrics derived from those reviews. |
| `case_outcomes`, `patient_risk_adjustment` | Keep measured simulation outcomes separate. Patient risk adjustment remains unavailable for SOSpine. |
| `training_view`, `partition` | Define the exact conversation, turn exposure, loss scope, eligibility, and surgeon-group assignment. |
| `training_value` | Reference a comparative downstream study and experimental arm, rather than invent a per-record reward. |

The schema can represent more tasks than the first runtime supports. An enum entry does not establish implemented task support. Examples, validation results, and exports must name the profile actually exercised.

## Original annotations and claim provenance

Original annotations retain `source_locator`, `raw_value`, `original_origin`, and `usage`. The SOSpine adapter's CSV locator identifies a source asset and its one-based physical row number, including the header; the asset carries the file hash, and `raw_value` preserves the complete row with its original column names and strings. The validator compares imported rows with the resolved CSV and checks their frame joins. The unnamed export-index column in the tool-tip CSV is preserved as source content; it is not the release frame index. A normalized label is a documented derivative of the raw label, not a replacement for it.

The distinction matters in SOSpine. The original visual annotations are CSV labels and coordinates for manually annotated instrument/durotomy points, plus boxes computed from those annotations. They are not prewritten descriptive prose. “Exact” preservation refers to the original row values and locations, not a guarantee that the labels are correct. Historical enhancement runs recorded Qwen proposals and MedGemma revisions; current independent annotations do not receive those proposals or CSV rows. Importing a computed box does not make its geometry manual ground truth. Rolling needle-driver tip/base labels into a positive instrument-class statement is also an explicit mapping. The source has no persistent object-track identities from which to derive a verified trajectory automatically.

Each claim records its origin and contribution, evidence, generation or transformation provenance where applicable, disposition, and review IDs. Evidence can cite frame IDs, original annotation IDs, supporting claims, reference assets, and image regions. General surgical knowledge and visually established facts remain distinguishable. Claims should be narrow enough to review independently: identifying a grasper does not establish its action, target tissue, or appropriateness.

`output_locations` binds a claim to the exact text in `training_view.messages`. Its `start_character` is inclusive and `end_character` exclusive, using zero-based Unicode codepoint offsets into the decoded text string in the specified block. JSON escape characters in the file representation are not extra codepoints in that string. Editing or moving the text requires regenerated bindings and a review of affected claims; changing JSON whitespace alone does not change string offsets. A citation elsewhere in the record is not a substitute for this binding.

Claim dispositions are `pending`, `retained`, `rejected`, or `superseded`. Review does not change provenance: an accepted model proposal remains model-origin content. Preserve previous revisions and rejected proposals for audit. A corrected claim or changed training message belongs to a new revision, with explicit lineage and the applicable new review.

## Selection, transformations, and teacher exposure

The baseline records `explicit_indices`. Historical enhancement archives record
`hybrid` with implementation `qwen-medgemma-inspection-v1`, including uniform
initial samples and model-requested evidence intervals. Those runtimes have been
removed, while their selection history remains unchanged. Current
[complete-video selection](SMART_FRAME_SELECTION.md) feeds
[independent MedGemma annotation](MEDGEMMA_FRAME_ANNOTATION.md). Its output is a
draft artifact outside this v2 archive/export adapter. Preserve an independent random evaluation sample so
selected difficult cases do not masquerade as the natural distribution.

Retrospective selection is legitimate for an offline archive when labeled accordingly. A claim that selection represents an online process additionally requires that the selector used only information available at the cutoff. Choosing an earlier frame because the final repair failed is outcome-informed selection, even when the resulting answer cites only that earlier frame.

Keep source JPEGs unchanged. Crops, resizing, overlays, normalization, and other processing create documented derivatives with source links, parameters, dimensions, and the coordinate mapping needed to interpret regions. If a generated restoration is explored, its new details are not additional measured anatomy. Model-specific preprocessing, including crops or tiles, is part of the effective input and must be reproducible.

The generation contract records model name/digest, quantization, runtime, prompt version, decoding parameters and request/response assets. The retained historical `llama-cpp-evidence-v1` adapter verifies the exact canonical request against source image bytes in chronological order, original CSV labels and coordinates preserved verbatim, fixed versioned questions/prompts, generation settings and validated parent outputs. Model and vision-projector GGUF metadata, their SHA-256 hashes and the pinned runtime identity are retained and checked against recorded model metadata. Requests and complete response envelopes are saved as exact bytes; generated claims/questions are bound back to validated response text and evidence references.

The historical llama.cpp enhancement transport used ordered image data URLs through the local llama.cpp OpenAI-compatible chat endpoint. Requests use a JSON-schema `response_format`; this image workflow does not invoke a native video processor. The application does not record internal processor crop/patch tensors or cryptographically attest server execution. It verifies stored-request consistency for this specific adapter, not arbitrary teacher formats or the medical truth of a response. Unsupported adapters remain archiveable but cannot confer model-generated training eligibility. The read-only historical `ollama-evidence-v1` verifier can verify original artifacts against their recorded bytes, metadata and lineage; those records still require all applicable human-review, eligibility and partition gates for training. They are not relabeled, resumed through llama.cpp or upgraded by changing an adapter identifier. An annotation-conditioned teacher is distinct from an image-only teacher: the historical first MedGemma observation was independent of Qwen's text but received the same original CSV labels and coordinates. This is not the exposure of the new independent annotation protocol.

New inference uses llama.cpp; the [migration notes](LLAMA_CPP.md#migration-from-ollama)
explain setup and historical provenance. Independent MedGemma annotation reads
a completed selection and original images, nearby source observations, optional
detail crops, and documented procedure context. It does not receive original
CSV labels. Historical Ollama and removed review/enhancement runs cannot resume as the new
annotation protocol; use a new output directory.

Historical bounded enhancement calls shared the original start/cutoff window. Parent-run exposure participates in causal checks; later review cannot erase information available in an ancestor. Final event/question proposals and historical stage claims remain pending. Teacher agreement does not create a human review event. The historical conversation projection uses the final response with `all_assistant_turns`; it does not export the full model deliberation loop as a reasoning trace.

`training_view.turn_links` connects each turn to its cutoff, input mode, student-visible frames/annotations/references, claims, generation runs, and reviews. Prior assistant answers are part of the student's context. A future-informed earlier answer leaks information even if the final answer's own image citations precede its cutoff. Recorded case outcomes and surgeon experience are not automatically included in either teacher or student prompts.

## Time, outcomes, and source discrepancies

Use each selected frame's `frame_index` for order within its source sequence. The audited filenames start at 1. No original PTS map was found, so the importer writes `timestamp_ms: null` and `timestamp_basis: unavailable`. `(index - 1) × 1000` could be an explicitly estimated offset from the first retained frame, recorded as `estimated_from_sampling`; it is not the original acquisition time and the current importer does not emit it. Repair duration is a separate measured task duration.

The following constraints summarize a local audit of the downloaded release on **2026-09-10**. Source CSV hashes were checked against the preserved Figshare manifest; only three example image dimensions were independently read during that audit. Subsequent bounded imports verify the selected image bytes and dimensions without claiming a new full-dataset image audit. The dataset and source-derived audit records are excluded from this repository; see [Data sources](DATA_SOURCES.md) to obtain your own copy.

| Audited fact | Contract consequence |
|---|---|
| 15,694 JPEGs in 24 contiguous sequences; the paper reports 15,698. Outcome-table frame counts sum to 15,641. | Use actual asset inventory for manifests; preserve published counts as source statements. Do not manufacture missing frames or a precise timing map. |
| 26 outcome rows: 24 measured outcomes and two missing; 22 labeled trials join to identified footage, totaling 13,568 frames. | Keep all outcome rows, including missingness. Only use verified joins for analyses requiring images and outcomes. |
| `Clip0`/`Clip1` have footage without established surgeon/outcome links; `S1A1`/`S2A1` have outcomes without matching named footage. | Do not invent a crosswalk. Unknown-surgeon clips cannot enter a verified surgeon-disjoint evaluation. |
| 500 tool-tip labels contain trailing whitespace; coordinates include decimals; 7,228 annotations have nonzero extent. | Preserve raw values and the declared normalization. Do not silently collapse every annotation to an integer point. |
| Tool-tip keys omit `S7A2` frames 1204 and 1205; box keys cover every released image. The box table has 172 zero-width/height rows and eight exceeding published 1920×1080 bounds. | Distinguish a missing source row from a blank placeholder or a negative label. Validate geometry before using it as supervision. |
| Eight repair durations exceed the corresponding declared video length. | Do not derive action boundaries or clip timestamps from trial duration. |
| Both preserved author readmes say CC BY-NC 4.0; recorded Figshare metadata says CC BY 4.0. | The importer retains the conflict in `source.license_evidence` with linked source assets. Import and export do not adjudicate the license. |

`Leak At 40mmHg` maps `Y` to measured leakage, `N` to no measured leakage, and an empty cell to missing. It describes a final pressure test in simulation, not an image-level finding, patient recovery, or quality score for an explanation. Keep it in `case_outcomes` with its exact source row. `patient_risk_adjustment` has no supported patient-level inputs in SOSpine and must not become an invented risk score.

## Eligibility and release

Training eligibility is `pending_review`, `eligible`, or `excluded`. A structurally reviewed record is only a prerequisite for eligibility. The release process must also establish valid source/evidence bindings, actual completed review of the emitted conversation, appropriate temporal exposure, resolved task eligibility, and corpus-level partition isolation. Schema validation alone cannot prove any clinical statement or that a review took place.

The runtime implements source-byte and CSV checks, exact claim/text bindings, supported teacher-request reconstruction, temporal/lineage checks, recorded-review gates and corpus isolation checks. These are consistency checks, not authentication of reviewer identity or proof that review occurred. Every model-generated claim and turn from the surgical inspection adapter requires recorded human surgeon or clinical-domain-expert review, including clinical appropriateness; the model's chosen claim type cannot waive that requirement. The deterministic baseline retains its separate review rules. `review-packet` writes HTML and a blank worksheet; completed-review ingestion remains unimplemented.

Reviewed training export applies these adapter limits even when the archive can represent broader paths:

| Representable feature | Current export limit |
|---|---|
| Model-generated questions or answers | Supported only through a byte-verified teacher adapter: `llama-cpp-evidence-v1` for new runs or the read-only `ollama-evidence-v1` verifier for historical records, plus actual applicable human review, eligibility and split gates. Unknown adapters remain blocked. |
| Measured `outcome_prediction` targets | A dedicated outcome-target adapter is not implemented. Case outcomes remain archive metadata. |
| Source annotations supplied to the student | A deployment-availability adapter is not implemented. Current source labels serve as target evidence. |
| References supplied to the student | An adapter binding exact injected text to its declared reference is not implemented. |
| Clinical-source training | A clinical source/training adapter is not implemented; the supported source is SOSpine simulation. |

A preview is explicitly marked as a format preview and does not bypass these gates to become a reviewed training release. Broader archive vocabulary supports future work, not an implied production capability.

Assign every trial, repeated attempt, selected window, and derivative from a known surgeon to the same partition before enrichment or model tuning. Record the split manifest and check the complete corpus for cross-partition overlaps. Unknown groups remain unassigned until verified; a format preview is not an accepted training release.

Evaluate the resulting archive and training view using [Review and evaluation](REVIEW_AND_EVALUATION.md). Export receipts should state the exact record revisions, schema versions, asset and conversation hashes, projection configuration, and loss scope used. Changed inputs or text require a new traceable build rather than a rewritten historical receipt.
