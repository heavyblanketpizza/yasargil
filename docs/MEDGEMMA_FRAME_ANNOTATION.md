# Independent annotation of selected surgical frames

MedGemma authors an annotation for each selected frame from source images.
The input is a completed `select-video-frames` run. Qwen selection explanations,
source CSV labels, outcomes, and surgeon-experience fields are excluded from the
annotation request. Qwen's role is limited to video frame selection.

## Evidence supplied to each annotation

Each fresh request has one selected target and a bounded packet of source images:

- The full target image with its canonical frame ID and playback locator.
- Up to two nearest source frames before and two after the target, in source order.
  These come from the complete source inventory, not just the selected targets.
- By default, four overlapping target crops, each spanning 60% of the target
  width and height, anchored at the four corners. Bounds map each crop to native
  source pixels; crops are neither resized nor enhanced. Tiny targets can produce
  fewer unique crops. Crops add no new observations or synthetic detail.
- Compact documented procedure background and explicit evidence roles.

The request presents the full target, its detail crops, then context images in
chronological order, followed by compact metadata. The default is up to nine
image views from five source observations. Sequence boundaries produce smaller packets.
All views go into a single user turn, because Gemma's chat format alternates
user and model turns. Each image is preceded by a short citation label: `T` for
the full target, `T1`–`T4` for its crops, `B1`, `B2` for earlier observations and
`A1`, `A2` for later ones, numbered outward from the target. Frame IDs stay out of
the prompt; software maps labels back to full view IDs. Only the target and its detail
views support claims described as directly visible in the target. Neighboring
images may support contextual claims; they cannot transfer their own visible
findings into the target. Unobserved motion, hidden anatomy, intent, success,
and repair integrity must not be filled in from medical expectations.

Nominal timestamps reconstructed from released images are playback locators.
They do not recover original acquisition times, missing observations, or actual
procedure elapsed time. Supplying original image pixels also does not establish
the resolution retained internally by the vision encoder.

## Run annotation

Install the pinned local model, vision projector, and runtime described in
[llama.cpp setup](LLAMA_CPP.md). Prepare packets without starting inference:

```sh
uv run yasargil annotate-selected-frames \
  --selection-run outputs/selection-example \
  --output-dir outputs/medgemma-annotation-example \
  --prepare-only
```

Start the prepared run or resume an interrupted run:

```sh
uv run yasargil annotate-selected-frames \
  --output-dir outputs/medgemma-annotation-example --resume
```

Omit `--prepare-only` to prepare and annotate in one invocation. A changed prompt,
selection, or configuration requires a new output directory outside the source
dataset. Defaults are 32,768 context tokens, 8,192 output tokens per target,
two earlier/two later observations, deterministic target crops enabled, and
seed 42. Image processing, text, and the reserved output must all fit the
context budget. Preparation refuses an output budget smaller than the largest
answer the response schema permits, so ordinary English text cannot reach the
token limit before the schema closes the answer. Use `--before-frames 0 --after-frames 0
--no-detail-crops` for a target-only comparison. `--procedure-context` can supply
verified background; when omitted, the saved selection context is inherited.

Each target is first annotated with greedy decoding. A reply that is unfinished,
breaks the response contract, or repeats itself is rejected and kept. A rejected
target gets one further attempt using Gemma's published sampling settings
(temperature 1.0, top-k 64, top-p 0.95, min-p 0). If that attempt is also
rejected, the target is recorded as failed and the run continues with the next
target. There is no silent target dropping: `summary.json` lists every failed
target with each attempt's decoding stage and rejection reason, and the report
links each raw reply. `--no-fallback-sampling` records a failure after the first
rejection instead. Runtime faults, such as an unreachable server or an
insufficient context budget, stop the run; resume retries the same attempt.

If the source images moved since selection (for example, a different disk or a
fresh download), pass `--source-dir /path/to/datasets/SOSpine/frames/<case>` when
preparing. Every image must still match the hash recorded at selection, and
`run.json` records the old and new directories. Resume needs no extra option.

```sh
uv run yasargil medgemma-annotation-status \
  --output-dir outputs/medgemma-annotation-example

uv run yasargil pause-medgemma-annotation \
  --output-dir outputs/medgemma-annotation-example
```

Pause is cooperative: an active call finishes and saves its response before
later targets stop. Resume verifies frozen inputs and saved receipts before
reusing accepted work. An interrupted, unfinished inference call may need to
run again. Exact requests, raw responses, model/runtime identities, evidence
packets, and per-target annotations are retained for inspection.

## What the annotation should capture

The protocol is `medgemma-frame-annotation-v2`, with evidence packets using
`medgemma-annotation-evidence-v1`. The model returns `visibility`, `claims`, and
`unresolved_questions`, in that order. Each claim gives its `statement`, then
`category`, `support`, `evidence_view_ids` (citation labels), and `uncertainty`.
The JSON schema constrains generation in this property order, so the model judges
visibility before writing claims and writes each statement before classifying and
citing it. Requests preserve the declared order on the wire. Software adds the
target frame ID and claim IDs, maps labels to full view IDs, and derives separate
target and contextual captions from the claims, so no extra uncited summary can
add facts. An answer has at most 12 claims and 4 unresolved questions.

A repeated statement, or several near-identical statements that differ only in a
detail such as a number, makes the answer invalid. A citation repeated within one
claim carries no information; it is removed and the row records the
`duplicate_citations_removed` quality flag. An answer that fills all 12 claim
slots is accepted with the `claim_cap_reached` flag, because a full list can hide
a truncated enumeration. Each accepted row also records its decoding stage.

Runs made with `medgemma-frame-annotation-v1` keep their original rules when
inspected or exported and cannot be resumed under v2; annotate again in a new
output directory. The prompt covers instruments and materials, identifiable tissue,
spatial relationships, local surgical state, context-supported action, and
visibility limitations. Empty claim lists and specific uncertainty are preferable
to filling a category with unsupported detail.

Every claim cites a target view. `target_visible` accepts only target views;
`context_supported` can cite nearby frames or use documented procedure context.
Action claims require views from at least two distinct source frames, including
the target. Procedure-step claims require explicit uncertainty. The categories
are `anatomy`, `instrument`, `material`, `spatial_relation`, `tissue_state`,
`action`, and `procedure_step`. A visible tool does not by itself establish
its action, target tissue, surgical purpose, or appropriateness. A puncture,
needle passage, suture maneuver, or change in tissue state needs the corresponding
visual evidence. The terminology and prompt are an experimental annotation
rubric, not a surgeon-validated ontology.

Further evidence needs are recorded rather than automatically dispatched to
Qwen, tools, or retrieval. This version supplies the fixed local packet; an
adaptive retrieval loop remains future work. All generated content remains a
proposal requiring human review, with no automatic training eligibility.

## Evaluate annotation value

No clinical accuracy or improvement over Qwen is established by implementing
this workflow. Medical pretraining motivates evaluation; it does not establish
expertise in surgical microscopy. A strict response schema checks structure
and source references, not whether the cited images support a statement.

Freeze an independently annotated evaluation subset before tuning prompts.
Compare Qwen-only descriptions, target-only MedGemma, and MedGemma with local
context/detail views on the same selected targets. Keep original labels and
Qwen prose out of the independent MedGemma inputs in every arm. Report image,
token, call, and expert-review budgets alongside quality.

Measure supported-claim precision, useful detail and omissions, wrong
instrument/tissue/action claims, target/context confusion, uncertainty handling,
and expert correction time. Inspect difficult views and source-label conflicts
separately. Use expert references for action and anatomy; existing SOSpine CSV
labels are not dense ground truth for those tasks. Hold out complete cases or
surgeons and account for correlated neighboring frames.

A future single-image learner must not receive context-supported text as if
that text were established by its target image. Supply the relevant context or
restrict the exported answer to target-visible claims. Human review, eligibility,
and partition checks remain separate from inference completion. See
[review and evaluation](REVIEW_AND_EVALUATION.md) and the
[archive/export contract](DATASET_CONTRACT.md).
