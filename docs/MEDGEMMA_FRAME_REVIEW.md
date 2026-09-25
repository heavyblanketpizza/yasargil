# Historical MedGemma review protocols

The Qwen-conditioned per-frame and whole-surgery MedGemma review runners have
been removed. New work uses [independent MedGemma annotation](MEDGEMMA_FRAME_ANNOTATION.md)
directly from a completed selection run.

Historical `medgemma-frame-review-v1` and `medgemma-surgery-review-v1` artifacts
keep their original requests, source evidence, model identities, and responses.
They may be inspected as historical records; they are not resumed, rewritten, or
relabeled as independent annotations. Those requests included Qwen drafts and,
when available, original source labels. New annotations do not inherit that text.

The [dataset inspector](DATASET_INSPECTOR.md) no longer displays them.
Model completion and model agreement do not establish clinical correctness or
training eligibility.
