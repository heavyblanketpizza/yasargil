# Independent annotation of selected surgical frames

MedGemma authors an annotation for each selected frame from source images.
The input is a completed `select-video-frames` run. No Qwen annotation pass is
required. Qwen selection explanations, Qwen drafts, source CSV labels, outcomes,
and surgeon-experience fields are excluded from the annotation request.

Qwen retains its video-selection role. A separate Qwen full-video annotation
can be generated for comparison, but MedGemma never receives that draft. The
former Qwen-conditioned review and bounded enhancement runners have been removed;
[historical review artifacts](MEDGEMMA_FRAME_REVIEW.md) remain readable.

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
image views from five source observations. Sequence boundaries produce smaller packets. Only the target and its detail
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
dataset. Defaults are 32,768 context tokens, 4,096 output tokens per target,
two earlier/two later observations, deterministic target crops enabled, and
seed 42. Image processing, text, and the reserved output must all fit the
context budget. A valid result needs a complete response and valid evidence;
there is no silent target dropping. Use `--before-frames 0 --after-frames 0
--no-detail-crops` for a target-only comparison. `--procedure-context` can supply
verified background; when omitted, the saved selection context is inherited.

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

The scoped first-two-case runner consumes selection results directly:

```sh
uv run python -m yasargil.medgemma_pair \
  --selection-batch outputs/selection-batch-example \
  --output-dir outputs/medgemma-pair-example --prepare-only

uv run python -m yasargil.medgemma_pair \
  --output-dir outputs/medgemma-pair-example --resume
```

It processes S2A2 and S1A2 only. It does not wait for or import a Qwen annotation
run. Historical MedGemma pair directories cannot be resumed into this protocol.

## What the annotation should capture

The protocol is `medgemma-frame-annotation-v1`, with evidence packets using
`medgemma-annotation-evidence-v1`. The model returns `target_frame_id`,
`visibility`, `claims`, and `unresolved_questions`. Each claim has a `claim_id`,
`category`, `statement`, `support`, `evidence_view_ids`, and `uncertainty`. Software
derives separate target and contextual captions from those claims, so no extra
uncited summary can add facts. The prompt covers instruments and materials, identifiable tissue,
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
