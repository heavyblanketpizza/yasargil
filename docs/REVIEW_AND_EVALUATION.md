# Review and evaluation — enhancement v2

**Updated 2026-09-22. Status: independent annotation and review/export foundations
are implemented; expert assessment and comparative experiments remain outstanding.**
The active [MedGemma annotation](MEDGEMMA_FRAME_ANNOTATION.md) receives source images
without Qwen drafts or original CSV labels. Historical archives remain verifiable;
Qwen-conditioned review/enhancement runtimes have been removed. The procedures
below are proposed evaluations, not completed studies. Model agreement,
schema-valid responses, and successful serialization do not establish quality.
See the [dataset contract](DATASET_CONTRACT.md).

## Three different questions

| Question | Evidence and unit of analysis | Where it belongs |
|---|---|---|
| Is the enrichment accurate, grounded, answerable, and useful? | Review of specific claims and the actual conversation/evidence; also check omissions on independently annotated windows. | `reviews` and named metrics in `enrichment_quality`. |
| What happened in the source trial? | The recorded simulated leak test and repair duration, with an exact CSV row and missingness. | `case_outcomes`. |
| Does the enhanced data improve a learner? | A controlled comparison on independently held-out surgeons, with fixed evaluation tasks and an experimental report. | `training_value` links to the study and arm. |

A correct instrument description can come from a trial whose final repair leaked. An incorrect explanation can come from a successful repair. The trial outcome therefore cannot serve as a correctness reward for the explanation. Multiplying one outcome across thousands of frames creates correlated labels, not thousands of independent outcomes.

SOSpine contains cadaveric training exercises, not postoperative patient records. Do not compute patient risk scores or claim risk-adjusted patient benefits from it. Surgeon postgraduate year and prior case ranges are source experience metadata, not validated frame-level skill scores. Patient risk adjustment remains unavailable for this source.

## Review the actual evidence and conversation

The implemented `review-packet` command validates the archive against its local source assets and writes a standalone HTML page plus a neighboring `.review-template.json`. It embeds the selected raster images, shows exact user/assistant text and image order, displays turn cutoffs and frame timing, and links claim text to exact spans and source evidence. Original SOSpine annotation rows contain manual instrument/durotomy point labels and coordinates or computed boxes; they are distinct from newly generated descriptive prose. These rows keep their CSV locators, raw values, original provenance, and file hashes. Exact preservation does not guarantee label accuracy. The packet is limited to 32 selected frames, 32 MiB of image bytes, and 96 MiB of HTML. It creates new files outside the dataset root and does not launch a browser or change the archive.

The worksheet binds the record ID, revision ID, canonical archive SHA-256, and rubric version. It contains per-turn and per-claim items with null verdicts and criteria, plus empty reviewer and blinding attestations. It is a pending worksheet, not a completed `reviews` event. There is no review-import or approval command; generating, opening, or filling this file does not automatically mark claims retained or make a record eligible.

For the full evaluation protocol, a reviewer needs the following evidence. Arbitrary teacher formats and internal model processing beyond the retained llama.cpp request remain outside the implemented adapter:

1. The complete emitted conversation window, including previous assistant answers that will condition the learner, and the exact image order, cutoff, and processing used.
2. The selected original images and any derivatives, their hashes, region mappings, and source conflicts relevant to the claim.
3. Claim text located by its exact message/text-block offsets, its evidence links, and the underlying original annotation rows where those are used.
4. For generated content, the actual teacher request and exposure record, including original annotations and parent outputs. The supported adapter reconstructs the exact request and verifies its frame order, annotation rows, settings and lineage; it rejects outcome/reference injection outside that format. Audit this separately from blinded quality review. The unblinded `audit.html` exposes stage/model identity, while raw request/response files support detailed provenance inspection. A deterministic import supplies its source mapping instead of inventing a teacher run.
5. The intended task and student input mode. A target generated with existing tool annotations is not evidence that an image-only teacher identified the tools independently.

Review must match the text and evidence that will be exported. A favorable rating for a summary in another file cannot approve a changed conversation. A review of the final answer alone cannot approve an incorrect earlier answer that remains in the prompt. Changing a claim's text, input image, crop, evidence binding, or causal exposure invalidates affected decisions and requires the appropriate new review.

Each decision identifies the reviewer, date, rubric, target claims/messages, and revision. Record evidence correctness, temporal correctness, clinical appropriateness where applicable, answerability, usefulness, error severity, and the reviewer's blinding. A pending proposal stays pending until a real review occurs; deterministic import and format validation cannot create an expert acceptance event.

All model-generated claims and turns from this surgical inspection adapter require human surgeon or clinical-domain-expert review, including clinical appropriateness, before entering reviewed training. Apply this to observations and question/answer targets as well as explicit interpretations: the model's self-assigned claim type must not determine whether clinical review is needed. The deterministic source-label baseline keeps its own reviewer rules. Model review by MedGemma is a generation stage and never substitutes for this human review.

The baseline uses `explicit_indices` and deterministic positive instrument
labels. Historical enhancement archives supplied original labels and Qwen prose
to MedGemma. Current independent annotation supplies source images and procedure
context only. Record this exposure difference when comparing results. Computed
boxes are not verified manual geometry. Original and artifact locations resolve
under separate roots; `artifact:` identifies the latter.

The separate [Qwen full-video baseline](FRAME_ANNOTATION.md) receives complete
video and selected stills without original CSV labels. [Independent MedGemma
annotation](MEDGEMMA_FRAME_ANNOTATION.md) uses target-centered source-image packets
without seeing Qwen drafts. Both produce separate proposals; neither by itself
satisfies the v2 reviewed-export gates described here.

New inference code for both workflows now uses llama.cpp. See [Migration from Ollama](LLAMA_CPP.md#migration-from-ollama) for the reasons and local setup/live-validation status. Preserve historical Ollama artifacts and their runtime provenance; the read-only verifier does not convert them. New llama.cpp runs require new output directories, and a backend change alone establishes neither better annotation quality nor reviewed training eligibility.

## Review rubric

| Dimension | Concrete review question |
|---|---|
| Evidence correctness | Do the cited frames or verified source annotations support this exact claim? Does a displayed region point to the intended object in the actual input image? |
| Temporal correctness | Was all supporting information available within the declared prefix? Does the teacher's actual request agree with the declared exposure? Do previous answers reveal later events? |
| Answerability | Can this question be answered at the supplied resolution and sampling rate? If evidence is missing or indeterminate, does the response preserve that limitation? |
| Clinical appropriateness | For interpretive or decision content, is the statement justified and suitably limited? Tool presence alone does not establish an appropriate next surgical action. |
| Usefulness | Does the answer address the bounded task without adding unsupported detail? A longer answer is not inherently better. |
| Error severity | Is the problem cosmetic, an unsupported detail, a wrong object/action, a temporal leak, or misleading decision content? Report serious errors separately rather than hiding them inside an average. |

Review origin and review result independently. An accepted model-generated annotation remains model-generated. An accepted imported box remains a box computed from source points. Expert corrections retain their previous version and the reason for change. A visual model's self-reported confidence is not a calibrated probability and should not substitute for any dimension above.

## Blinding and reference annotations

The current default packet withholds case outcomes, baseline and case metadata, their linked assets, teacher request/response artifacts, claim origin, generator metadata, and prior review decisions. Filtering whole metadata assets matters because an ostensibly general case row may contain the final leak result. It also withholds free-text selection reasons, arbitrary selector parameters, and selector implementation identity; the selection method, declared temporal exposure, frame order, and timing remain visible. An explicitly unblinded option adds case outcomes, generation-run details, and selector configuration in a labeled retrospective section.

This is metadata blinding, not guaranteed effective blinding. Source filenames/hashes, raw annotation provenance, case IDs, exact conversation text and image content remain available. The renderer does not assign presentation aliases. A visible leak test may reveal an outcome; wording or filenames can reveal provenance. Reviewers must record what they saw or inferred rather than setting attestations from packet mode. Keep `audit.html` and raw model receipts away from the blinded quality-review presentation; they belong in the separate provenance audit. Request reconstruction verifies stored-artifact consistency, not server attestation or medical truth.

For an initial quality study, separate independent reference creation from proposal review. Experts first annotate a selected evaluation subset from the permitted images/prefix, without model proposals, final outcomes, or model identity. They may mark an observation unanswerable instead of forcing a label. A subsequent review compares the exact proposed claims and messages against that reference and their cited source evidence.

When comparing enhancement arms, blind reviewers to model/arm identity and final leak outcomes. A future study presentation should hide surgeon identity and experience where practical using aliases while preserving original IDs in the archive. The original images may still reveal procedural cues; record limits to blinding. Reviewing evidence will sometimes require seeing original annotations, so record when those were available rather than claiming complete independence.

Use at least two independent qualified reviewers on a prespecified subset, and adjudicate disagreements with the evidence visible. Choose the amount of duplicate review from a pilot, the anticipated disagreement rate, and the study budget; do not claim a sample-size guarantee from this document. Keep both initial ratings and adjudication. Report disagreement, adjudication rate, and reasons for exclusion. The reviewer who corrects a proposal should not be its sole independent evaluator.

Freeze evaluation references before tuning the teacher, selector, prompts, or learner. Keep final case outcomes out of enrichment-quality review. If the final pressure test is visually present in a candidate prefix, outcome blinding and forecast eligibility need separate assessment; removing a metadata field does not remove visible outcome information.

## Measure enrichment quality

Every metric needs a definition, denominator, task/category scope, reviewed population, treatment of missingness, and a link to the contributing decisions. `enrichment_quality` should summarize those actual reviews; it is not a free-standing score written by the teacher.

Report supported-claim precision, unsupported/incorrect claim frequency, temporal leakage, answerability errors, serious-error frequency, review acceptance/correction rates, and expert time per accepted unit. Separate imported facts from newly proposed claims, and separate instruments, geometry, actions, and interpretation. A model that generates no claims can avoid false statements while being unhelpful, so independently annotated windows are also needed to measure useful coverage and omissions.

For verified instrument labels, use class-wise precision/recall and state how blank annotations are interpreted. For geometry, evaluate against an independently checked reference with an appropriate point or box metric; do not treat the problematic released boxes as universally correct masks. For proposed action or phase labels, first define the taxonomy and temporal boundaries with surgeons. SOSpine does not already supply dense ground truth for these tasks.

Group uncertainty estimates by surgeon/trial rather than treating neighboring frames or overlapping windows as independent. With only eight named surgeons, report each held-out result and the instability of aggregate estimates. Stratify errors by source conflicts, low visibility, rare instruments, and selection method where independently supported. The 1-fps release permits ordered-prefix checks but does not measure subsecond event detection or operating-room alert latency.

## Proposed ablation protocol

Freeze source bytes, eligible cases, task definitions, prompts, output budget, learner family, compute budget, and evaluation references before comparing arms. Keep test surgeons out of prompt design, selection tuning, review-assisted corrections, and hyperparameter selection. Use a fixed surgeon-disjoint split or an outer leave-one-surgeon-out protocol with tuning confined to the remaining surgeons. Preserve all attempts and overlapping windows within the same group. Keep `Clip0`/`Clip1` outside verified surgeon-disjoint training/evaluation until their identity mapping is established.

The following are proposed comparisons, not measured results:

| Comparison | What it isolates |
|---|---|
| Original annotations versus original annotations plus reviewed enrichment | Whether newly supported content adds value beyond existing labels. Keep original labels available to both arms where the task permits. |
| Uniform selection versus a declared diversity/event/uncertainty selector, with equal frame and labeling budgets | The effect of choosing frames. Report selection coverage as well as performance on an independent random reference sample. |
| Image-only teacher versus the same teacher supplied with verified original annotations | The benefit and dependence introduced by source-conditioned generation. Label the exposure difference explicitly. |
| Single-frame input versus a causal multi-frame prefix | Whether temporal context improves a defined task, using only prefixes that respect the input cutoff. |
| Source-derived targets versus raw teacher proposals versus reviewed/corrected teacher targets | The effects of generation and review. Raw proposals remain a quarantined research arm, never an accepted production training release. |
| One pinned local MedGemma setup versus a supported alternative or a task-specific baseline | The effect of model choice on accepted-label quality, expert effort, and downstream results. Record quantization and hardware instead of comparing model names alone. |
| Qwen-only annotations versus independent MedGemma annotations on the same targets | Whether independent medical-model generation improves supported useful detail and expert correction effort. Record different image exposure and compute budgets. |
| Target-only MedGemma versus target plus local context/detail crops | Whether added visual evidence improves grounded answerability. Match or report frame/call budgets and measure target/context confusion and unresolved claims. |

Start with instrument identification, where the independently downloaded source supplies labels for a bounded baseline. Extend to spatial localization or temporal/interpretive tasks only after their reference annotations and review rubric are established. The v2 schema's broader task vocabulary is not evidence that those experiments are ready.

These arms are not all implemented as controlled generation configurations.
The historical `llama-cpp-evidence-v1` verifier supports archived enhancement
records, not automatic export of new independent MedGemma annotations. Matched
comparisons and a new annotation export adapter need explicit experiment support.
Measured outcome targets, student annotation/reference injection, and clinical-source
training remain unsupported. A preview is not a route around human-review,
eligibility, or split gates. Human review and downstream training results must
be measured separately from generation.

For each arm, train with the same partitions and a comparable compute budget; record seeds and the actual images/tokens consumed. Evaluate on frozen, independently reviewed held-out examples using task metrics and failure cases. Attribute changes to the intervention being varied. A comparative report should include review effort, accepted annotation count, uncertainty, and failures, not only the best checkpoint's score. `training_value` then links the archive/export to that report and arm; it must not contain an invented row-level estimate of future usefulness.

The distinction between strong component scores and a working pipeline is supported by spine-specific research: a July 2026 comparison found a substantial gap between MedSAM2-Tiny segmentation with ground-truth boxes and its fully automatic detector-driven performance under domain shift. Dataset coverage and the complete deployed input path mattered. This motivates end-to-end evaluation; its results do not validate Yasargil or MedGemma on SOSpine. [Deployment-realistic spine benchmark](https://link.springer.com/article/10.1007/s00586-026-10162-5)

## Acceptance evidence and limits

The runtime can produce a deterministic source baseline and independent
MedGemma annotation drafts. Archive validation, human-review packets, and
receipt-bound message projections remain available for supported archive formats.
The new annotation artifacts are not automatically eligible archive exports.
Retain actual execution evidence and assess claims against the supplied images;
structural checks do not establish clinical accuracy or training benefit.

Validation checks source integrity, exact source-row and claim/text bindings, supported teacher-request bytes, ancestry and cutoffs, prior-turn restrictions, outcome separation, recorded reviews and surgeon-level isolation. It cannot authenticate human review or silently extend byte verification to arbitrary teacher formats. Tests should exercise these boundaries and refusals; reports must name checks actually executed. Model agreement does not prove correctness, and provenance validity does not demonstrate that the model used evidence competently.

The source-wide inventory remains the **2026-09-10 audit**, not a new full-dataset audit performed whenever one packet is generated. It records the frame-count discrepancies, unknown `Clip0`/`Clip1` mappings, missing named footage, problematic source geometry, duration inconsistencies, and conflicting license notices summarized in the [dataset contract](DATASET_CONTRACT.md). A successful import validates its selected images and exact source rows; it does not resolve those dataset-wide issues.

A model/processor load or finite forward/backward step establishes integration. Enrichment-quality review establishes properties of the annotations. A controlled held-out study establishes bounded downstream utility. None alone establishes patient benefit, a calibrated surgical risk score, or readiness for live guidance. Those claims require different clinical data and prospective evaluation beyond this cadaveric source.
