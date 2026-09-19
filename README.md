# Yasargil

[Apache License 2.0](LICENSE) · Copyright 2026 Yasargil contributors.

Yasargil creates **traceable, reviewable multimodal training conversations** from surgical datasets. Qwen proposes observations and questions from selected images; MedGemma independently inspects the evidence, reviews the proposals, and requests targeted reinspection when information is missing. Each call, evidence reference and proposed answer is retained for human review.

The first source is **SOSpine**, microscope images of simulated spinal durotomy repair on cadavers. The immediate product is an enhanced research dataset and an Unsloth training interface. Surgeon assistance is the downstream research goal; dataset generation alone does not establish readiness for live use.

**The dataset is not included in this repository. Download SOSpine yourself from the [official Figshare project](https://figshare.com/projects/Simulated_Outcomes_for_Durotomy_Repair_in_Minimally_Invasive_Spine_Surgery_SOSpine_/142508).** Follow [Data sources and local setup](docs/DATA_SOURCES.md) for attribution, required files, checksums, directory layout, and what Yasargil generates. Keep source data and derived outputs outside Git.

[Enhancement workflow](docs/ENHANCEMENT.md) describes the implemented architecture, full-sequence batches and durable pause/resume. This repository contains code, schemas, synthetic tests, and user guides; internal planning notes, research archives, work logs, and copied dataset records are kept in private storage outside it.

## Inspect the enhanced dataset

Run `.venv/bin/python -m yasargil inspect-dataset` and open
[the local dataset inspector](http://127.0.0.1:8765). Its surgery timeline shows
selected frames in green and Qwen-dropped frames in orange. Click any point to
inspect the source image, Qwen and MedGemma annotations, original dataset labels,
and surgery outcomes. Browser-saved review notes can be exported separately.
See the [inspector guide](docs/DATASET_INSPECTOR.md).

## Frame selection with complete video context

`select-video-frames` uses DINO embeddings to propose diverse frames, then asks local Qwen through llama.cpp to review the **complete native video and the candidate stills together in one call**. Qwen only keeps or drops the sampled candidates; it cannot request or add frames. The default set contains 24 candidates, including eight protected timeline anchors. Every candidate carries its timestamp basis, source filename, exact frame index, and file hashes.

The workflow verifies complete frame coverage and rejects inputs that the pinned runtime would resample or truncate. SOSpine JPEG sequences use reconstructed nominal time: duration is image count divided by reconstruction fps, so 288 images at 1 fps last 4:48. Selection and annotation share this media timeline and ignore recorded repair/outcomes-CSV durations for timing. These offsets locate images accurately within the reconstruction; original-procedure elapsed time and end-to-end coverage remain unverified. Original videos retain their verified PTS timeline. See the [setup, commands, provenance and limits](docs/SMART_FRAME_SELECTION.md) and [local llama.cpp setup](docs/LLAMA_CPP.md).

```bash
uv sync --extra selection
uv run --extra selection yasargil select-video-frames \
  --input '/path/to/surgery.mp4' \
  --output-dir outputs/smart_selection/example
```

This produces `selection.json` and a visual `selection.html` review. The existing enhancement and training-export workflow below remains a separate command path.

For every available SOSpine sequence, [the selection batch](docs/SELECTION_BATCH.md)
alternates trials with no measured leak and trials with a measured leak, using
trial ID to resolve ties and placing unknown outcomes last. These are simulated
repair outcomes, not patient recovery rankings. The queue includes all 24 released
sequences and keeps outcome metadata outside Qwen's requests. Each sequence gets
its own complete native 1-fps video and one keep/drop review, with a 262,144-token
context and a six-hour request timeout by default. Status, pause, and resume
commands preserve completed work.

After selection, [the annotation pass](docs/FRAME_ANNOTATION.md) starts a fresh
Qwen session with the complete video and the frozen final stills. It produces
per-frame visible observations, separate video-context claims with timestamped
source evidence, and uncertainty for human review. These drafts remain ineligible
for training until the separate review requirements are satisfied.

The next [MedGemma review stage](docs/MEDGEMMA_FRAME_REVIEW.md) assesses each Qwen
annotation with its key frame, before/after source frames, cited supporting
images, and matching original dataset labels. It preserves corrections and the
complete response. Requests for additional evidence are saved for later without
dispatching Qwen or TimeLens2:

```sh
.venv/bin/python -m yasargil review-frame-annotations \
  --annotation-run outputs/frame_annotations/S6A3 \
  --output-dir outputs/medgemma_reviews/S6A3
```

The separate [gap experiment](docs/GAP_EXPERIMENT.md) retains evidence retrieval
to test whether Qwen requests missing frames. It
scores and ranks the original candidates by visible surgical importance, then
drops the least-important 50%, 70%, or 90% in separate full-video sessions, alongside
an all-candidate control. Copied timestamp or presentation orders fail a reference
check before the audits start. It preserves the scores, reasons, exact omissions,
retrieval receipts, and provisional recovery measures in a visual comparison
report, with pause/resume support.

## Enhancement workflow

```text
Original SOSpine JPEGs + exact visual annotation rows
                      ↓
Bounded frame window; initial uniform sample
                      ↓
Qwen proposes ───── MedGemma independently observes
                      ↓
MedGemma reviews both against the images
                      ↓
If evidence is missing: Qwen inspects requested additional frames
                      ↓
MedGemma revises; repeat within explicit budgets
                      ↓
Evidence archive + call audit + pending human-review packet
                      ↓
Actual human review + surgeon-disjoint partition gates
                      ↓
Reviewed multimodal messages → PIL image loading → Unsloth SFT
```

The enhancement workflow uses **ordered images through local Ollama**, with release-frame indices and unavailable timestamps stated explicitly. Qwen performs the targeted local reinspection; a TimeLens2 adapter is not implemented. Case outcomes remain archive metadata and are excluded from model requests.

The archive preserves how an example was made. The learner receives only the selected conversation: user text and image blocks followed by reviewed assistant text. This is conversational supervised fine-tuning with vision inputs. Generated explanations are proposed content, not recovered surgeon thoughts or an outcome-based reward.

## Setup

The repository selects Python 3.12 through `.python-version` and locks core dependencies in `uv.lock`; package metadata supports Python 3.10 or newer. Model weights, PyTorch and Unsloth are outside the core environment. Run an Ollama service separately on this computer.

```bash
uv sync --frozen
uv run yasargil --help
uv run yasargil models
```

Defaults are `qwen3.8:27b-q4_K_M` and `medgemma:27b`; install compatible models in your own environment. Each run checks the current tags, digests, quantization and vision capability and saves model metadata. To explicitly download or update the requested models:

```bash
uv run yasargil models --pull
```

Preview the plan without writes or model calls:

```bash
uv run yasargil enhance-sospine \
  --dataset-root "/path/to/datasets/SOSpine" \
  --case-id S1A2 --start-index 1 --cutoff-index 12 \
  --initial-frames 4 --search-frames 4 --max-frames 8 \
  --dry-run
```

Execute the same example into a **new** output directory:

```bash
uv run yasargil enhance-sospine \
  --dataset-root "/path/to/datasets/SOSpine" \
  --case-id S1A2 --start-index 1 --cutoff-index 12 \
  --initial-frames 4 --search-frames 4 --max-frames 8 \
  --output-dir outputs/example
```

The default one search round permits at most five logical stages: three initial calls plus Qwen search and MedGemma revision. A run may stop earlier. At most 32 distinct frames are selected per window; this example permits eight. Source files remain unchanged and outputs must be outside the source dataset. There are no implicit model pulls or automatic retries; explicit resumption can repeat an unfinished stage.

A completed run writes an archive, plan, exact requests/responses, model metadata, completion status, an unblinded call audit, a separate review packet with blank worksheet, and a training-format preview when there is projected dialogue. A failed run retains available call artifacts and records failure details. **Pipeline completion does not mark the archive reviewed or training eligible.** See [artifact details](docs/ENHANCEMENT.md).

## Process whole sequences and resume later

Batch processing divides the selected cases' available released frames into consecutive, non-overlapping windows. Each window runs the same inspection loop and creates its own archive/review packet. Preview a two-case plan, then remove `--dry-run` to execute it:

```bash
uv run yasargil enhance-sospine-batch \
  --dataset-root "/path/to/datasets/SOSpine" \
  --case-ids S4A3 S5A1 \
  --window-size 12 \
  --initial-frames 4 --search-frames 4 --max-frames 8 --max-rounds 1 \
  --output-dir outputs/sospine-best-worst \
  --dry-run
```

Window coverage does not mean every frame is sent to a model: this configuration starts with four selected frames per window and permits up to eight. Windows are self-contained; the next window does not inherit previous answers or model memory. Events crossing a window boundary can therefore lose context. The transport remains ordered 1-fps release images, not continuous video or a full-operation narrative model.

Use these commands from another terminal while a job runs, or after returning later:

```bash
uv run yasargil enhancement-status --output-dir outputs/sospine-best-worst
uv run yasargil pause-enhancement --output-dir outputs/sospine-best-worst
```

Pause lets the current model call finish and saves its result before stopping. Wait until status shows `paused` with `writer_active: false`, then resume when ready:

```bash
uv run yasargil resume-enhancement --output-dir outputs/sospine-best-worst
```

The first `Ctrl+C` also requests this pause; a second interrupts immediately, so the unfinished call must run again on resume. Status reads saved progress without loading the source or contacting Ollama. Resume loads saved settings, verifies source/model/configuration compatibility, and reuses completed call outputs. It saves application artifacts, **not a model KV cache**; source files and the same installed model artifacts must remain available. A foreground resume stays attached to that terminal. A separately launched background worker uses the same status/pause commands.

The same status/pause/resume commands accept a single-window output directory such as `outputs/example`. Alternatively, repeat its original `enhance-sospine` command with `--resume`; its configuration must match. A changed model, prompt, source or configuration needs a new job rather than silently mixing results. See [resume safeguards and artifacts](docs/ENHANCEMENT.md).

## Source baseline and checks

The deterministic importer remains a comparison arm. It re-expresses positive grasper/needle-driver labels while preserving exact source rows; it performs no model inference or action interpretation.

```bash
uv run yasargil import-sospine \
  --dataset-root "/path/to/datasets/SOSpine" \
  --case-id S1A2 --frame-indices 1 2 3 \
  --output outputs/baseline.draft.json

uv run yasargil review-packet outputs/baseline.draft.json \
  --dataset-root "/path/to/datasets/SOSpine" \
  --output outputs/baseline.review.html

uv run python -B -m unittest discover -s tests -v
node --test tests/test_review_ui.js
```

The active code includes source/CSV verification, teacher-request reconstruction, claim/text binding, review packets, export receipts, ordered RGB image hydration and trainer construction. Completed-review ingestion, expert assessment and student GPU training remain pending. The teacher adapter permits model-generated training data only after actual human-review, eligibility and partition requirements are also satisfied; the enhancement loop does not grant those approvals.

## Source reality

The local audit found **15,694 released JPEGs in 24 sequences**, rather than the paper's 15,698. They are a 1-fps release from recordings described as 30 fps. Original continuous videos and acquisition timestamps were unavailable in the inspected release. Selecting released frames cannot recover missing motion detail.

The outcome table has 26 rows, 24 measured leak labels and 22 labeled trials with identifiable footage. `Clip0` and `Clip1` have no verified surgeon/outcome mapping; `S1A1` and `S2A1` lack identifiable footage. Tool-tip annotations, computed boxes, raw rows and known defects remain distinguishable. The audit also records conflicting license notices. See [Data sources](docs/DATA_SOURCES.md) and the [dataset contract](docs/DATASET_CONTRACT.md). The source-derived inventory remains in private local storage; individual source rows are not distributed here.

Keep three evaluations separate: whether an added claim is correct, what the simulated trial's leak test measured, and whether reviewed enrichment improves a held-out student. A favorable trial outcome is not a correctness reward for every explanation. SOSpine has no patient recovery or preoperative patient severity; patient risk adjustment is unavailable.

## Code and documentation

| Location | Purpose |
|---|---|
| [Complete-video frame selection](docs/SMART_FRAME_SELECTION.md) | One native full-video review of fixed embedding candidates, keep/drop decisions and provenance. |
| [SOSpine selection batch](docs/SELECTION_BATCH.md) | All released sequences, alternating simulated leak outcomes, status and pause/resume. |
| [Frame annotation](docs/FRAME_ANNOTATION.md) | Fresh full-video annotation, visible/contextual claim separation, timestamp evidence and human review. |
| [Enhancement workflow](docs/ENHANCEMENT.md) | Model responsibilities, budgets, commands and artifacts. |
| [Dataset contract](docs/DATASET_CONTRACT.md) | Evidence archive, temporal exposure, provenance and export gates. |
| [Review and evaluation](docs/REVIEW_AND_EVALUATION.md) | Human review and comparative studies. |
| [Training](docs/TRAINING.md) | Unsloth compatibility, image loading and loss policy. |
| [Data sources](docs/DATA_SOURCES.md) | Original publisher, attribution, local download/setup, and data exclusion policy. |
| [Active schema](schemas/enhancement-record.schema.json), [source](src/yasargil/), [tests](tests/) | Version 2 archive and current runtime. |

Component benchmarks motivate experiments; they do not establish Qwen, MedGemma or Yasargil's surgical accuracy on SOSpine.

Before committing, run `python3 scripts/check_repo_hygiene.py` and then `python3 scripts/check_repo_hygiene.py --staged` after staging. Internal Markdown is ignored by default; the allowlist contains only this README and the public guides. Dataset and output files, personal configuration, and credentials must remain outside version control.

## License

Yasargil's original source code, schemas, tests, and documentation are licensed
under the [Apache License, Version 2.0](LICENSE). This permits research and
commercial use, modification, and redistribution subject to the license terms.

Third-party materials retain their own terms:

- **SOSpine and other datasets:** the software license does not license source
  datasets or automatically apply to generated annotations and training exports.
  SOSpine is not included here; download it from the original publisher and
  follow its terms. The author-readme CC BY-NC 4.0 and Figshare CC BY 4.0 notices
  remain an unresolved discrepancy. See [Data sources](docs/DATA_SOURCES.md).
- **Model weights and external runtimes:** obtain these separately under their
  upstream terms, including the [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms)
  for MedGemma. Yasargil's license does not replace those terms.
- **Bundled fonts:** the unmodified Meslo Nerd Font files retain the licenses
  and attribution in [font notices](src/yasargil/review_ui/fonts/LICENSES.txt).

Yasargil is research software. Its outputs have not been validated for clinical
decision-making or live surgical guidance. This describes the project's
validation status and does not add a use restriction to the Apache license.
