# Historical Qwen–MedGemma enhancement

The active bounded enhancement and batch runners have been removed. Their
proposal/review/revision chain supplied Qwen output and original source-label
rows to MedGemma. New work uses [independent MedGemma annotation](MEDGEMMA_FRAME_ANNOTATION.md)
of selected frames instead.

Saved enhancement archives, model requests and responses, and human-review
packets remain unchanged. The historical prompt definitions and teacher
adapters remain solely to reconstruct and verify those archives. They do not
provide an inference or resume command. Historical Ollama and llama.cpp records
retain their original adapter and runtime identities.

The [dataset contract](DATASET_CONTRACT.md), [review protocol](REVIEW_AND_EVALUATION.md),
and [training interface](TRAINING.md) still describe archive validation,
review requirements, and export foundations. A historical archive that passes
structural verification still needs the applicable expert review and partition
checks. The new independent annotation artifacts do not automatically become
reviewed training exports.
