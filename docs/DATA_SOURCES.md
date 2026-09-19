# Data sources and local setup

**Download the dataset yourself from its original publisher. Yasargil does not
bundle or redistribute the SOSpine dataset.** This repository is for application
code, schemas, synthetic test fixtures, and user documentation. Source images,
annotation and outcome tables, copied source rows, generated annotations,
training exports, and model weights belong in local storage outside Git.

## Source and attribution

SOSpine (Simulated Outcomes for Durotomy Repair in Minimally Invasive Spine
Surgery) is released through the authors' [Figshare project](https://figshare.com/projects/Simulated_Outcomes_for_Durotomy_Repair_in_Minimally_Invasive_Spine_Surgery_SOSpine_/142508).
The [Scientific Data descriptor](https://doi.org/10.1038/s41597-023-02744-5)
describes microscope recordings of simulated spinal durotomy repair on cadavers,
released as sampled JPEG frames with tool annotations and trial outcomes.
Credit the original dataset authors and cite that descriptor and the Figshare
records when using SOSpine; Yasargil is a separate processing project.

| Required source | Official record | Local destination under your dataset root |
| --- | --- | --- |
| Released JPEG frames | [frames.zip](https://doi.org/10.6084/m9.figshare.20201636.v1) | `frames/<case>/<case>_frame_########.jpeg` |
| Human-labeled tool points | [sospine_tool_tips.csv](https://doi.org/10.6084/m9.figshare.20171135.v1) | `sospine_tool_tips.csv` |
| Computed bounding boxes | [sospine_bbox.csv](https://doi.org/10.6084/m9.figshare.20171129.v1) | `sospine_bbox.csv` |
| Simulated trial outcomes | [sospine_outcomes.csv](https://doi.org/10.6084/m9.figshare.20171132.v1) | `sospine_outcomes.csv` |
| Original description and notices | [readme.txt](https://figshare.com/articles/dataset/readme_txt/20171138) | `documentation/readme.txt` |

Preserve the original notices with your download. The inspected author readme
says CC BY-NC 4.0, while the Figshare project, individual records, and the
descriptor's **Data Records** section say CC BY 4.0. Yasargil retains that
discrepancy in provenance; it does not resolve it or grant additional rights.

### Noncommercial experiments

Local analysis, annotation experiments, and experimental training for a
noncommercial purpose appear consistent with both published notices: even
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) permits copying
and adaptation for noncommercial purposes. Credit the dataset authors, cite the
release and descriptor, retain the supplied notices and license links, and
identify your modifications and generated annotations when sharing results.
Do not imply that the authors endorse Yasargil or its outputs.

“Noncommercial” concerns the purpose of the use, not simply whether an experiment
earns revenue. Research directed toward commercial advantage is not automatically
noncommercial. Seek clarification from the dataset authors before relying on
the conflicting notices for commercial product work. The repository's Apache
license does not relicense the dataset; model weights and services have their
own terms. These observations describe the published permissions, not a legal
determination for every project.

## Download and arrange your own copy

Choose a directory **outside this repository**, with space for the approximately
5.9 GB frame archive and its extracted images. Download the five files above
from Figshare. Store `frames.zip` under `archives/`, the three CSVs directly under
the dataset root, and the original `readme.txt` under `documentation/`.

The frame archive contains per-case ZIPs inside `frames/`. Extract both levels
so that JPEGs sit directly inside each case directory:

```sh
export SOSPINE_ROOT="/path/to/datasets/SOSpine"
mkdir -p "$SOSPINE_ROOT/archives" "$SOSPINE_ROOT/documentation" "$SOSPINE_ROOT/metadata"

# Run after placing your downloads in the locations described above.
unzip -n "$SOSPINE_ROOT/archives/frames.zip" -d "$SOSPINE_ROOT"
for case_archive in "$SOSPINE_ROOT"/frames/*.zip; do
  unzip -n "$case_archive" -d "$SOSPINE_ROOT/frames"
done
```

The importer also requires `metadata/source_manifest.json`. This is a local
provenance record, not another annotation dataset supplied by Yasargil. The
following command retrieves the official record metadata, verifies each
download against its published checksum, and records source URLs, retrieval
time, and local hashes. It downloads metadata only; obtain the dataset files
yourself before running it. It refuses to overwrite an existing manifest.

```sh
python3 - <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

root = Path(os.environ["SOSPINE_ROOT"]).expanduser().resolve()
sources = [
    (20201636, "archives/frames.zip"),
    (20171135, "sospine_tool_tips.csv"),
    (20171129, "sospine_bbox.csv"),
    (20171132, "sospine_outcomes.csv"),
    (20171138, "documentation/readme.txt"),
]
manifest = {"dataset": "SOSpine", "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "files": [], "articles": []}
for article_id, relative in sources:
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    with urlopen(url, timeout=60) as response:
        article = json.load(response)
    path = root / relative
    published = next(f for f in article["files"] if f["name"] == path.name)
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    expected = published.get("computed_md5") or published.get("supplied_md5")
    if not expected or md5.hexdigest() != expected:
        raise SystemExit(f"Missing or mismatched published checksum: {relative}")
    manifest["articles"].append(article)
    manifest["files"].append({"path": relative, "source_record": url,
                              "download_url": published["download_url"],
                              "md5": md5.hexdigest(), "sha256": sha256.hexdigest()})
destination = root / "metadata/source_manifest.json"
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("x", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)
    stream.write("\n")
print("Verified downloads; saved local source manifest.")
PY
```

Your prepared source root should look like this:

```text
SOSpine/
  documentation/readme.txt
  metadata/source_manifest.json
  sospine_tool_tips.csv
  sospine_bbox.csv
  sospine_outcomes.csv
  frames/
    S1A2/
      S1A2_frame_00000001.jpeg
      ...
    ...
```

Supply that root explicitly, for example:

```sh
uv run yasargil enhance-sospine \
  --dataset-root "$SOSPINE_ROOT" \
  --case-id S1A2 --start-index 1 --cutoff-index 12 \
  --initial-frames 4 --search-frames 4 --max-frames 8 \
  --dry-run
```

The [dataset contract](DATASET_CONTRACT.md) records observed release discrepancies
and timing limits. Inventory your downloaded version rather than assuming the
paper's counts or interpreting reconstructed timestamps as acquisition times.

## What Yasargil does with the data

Yasargil reads the original JPEGs and annotation tables without changing them.
The original visual annotations are CSV rows containing a frame filename, a
label, and coordinates: human-labeled tool tips/bases and durotomy ends in
`sospine_tool_tips.csv`, plus boxes computed from those points in
`sospine_bbox.csv`. These tables do not contain the descriptive prose generated
by Yasargil. Keeping an exact row means preserving the original values and
source locator; it does not assert that the label or geometry is error-free.

The deterministic importer re-expresses selected source instrument labels and
retains the exact evidence references. In the complete-video annotation path,
Qwen writes new visible observations, contextual claims, and uncertainty from
the supplied video and selected stills, without the original CSV labels. The
later MedGemma review receives those drafts, selected images, and matching
source labels when available. The separate bounded enhancement loop supplies
original label/coordinate rows to both models from the outset.

These workflows produce model proposals, source hashes, model-call records,
and review artifacts in a separate output directory. Reconstructed videos use explicit sampling
assumptions; generated descriptions remain proposals pending human review.
Training export applies the documented evidence, review, and partition checks.
See [Enhancement](ENHANCEMENT.md), [Frame selection](SMART_FRAME_SELECTION.md),
and [Training](TRAINING.md).

New inference uses local llama.cpp. Dataset files and original CSV values are
unchanged by the backend migration; see [runtime setup and migration status](LLAMA_CPP.md#migration-from-ollama).

These derived outputs may contain original images, copied CSV rows, reviewer
identifiers, and machine paths. Keep them local too. `outputs/` is ignored for
convenience, and raw data/media/export formats are ignored throughout the repo.
Internal planning notes, research reports, work logs, and the source-derived
inventory have been moved to private storage outside the repository. Public
examples use placeholder paths; tests generate synthetic data in temporary
directories. Fonts are resolved from the local system and are not bundled.

Before a commit, run `python3 scripts/check_repo_hygiene.py`; after staging,
run `python3 scripts/check_repo_hygiene.py --staged`. CI runs the same current-tree
check. The ignore rules and checker exclude font binaries, model weights and
checkpoints, medical and other source media, archives, credentials, local editor
settings, dependency caches, and build outputs. Unrecognized binary files are
also rejected unless their exact path is an approved media exception.

JSON, HTML, and notebooks require an individually reviewed path in both
`.gitignore` and the checker's `PUBLIC_ARTIFACTS` allowlist. The current exceptions
are the enhancement schema and the review UI's HTML source. Keep generated
records and reports under `outputs/` or outside the repository; renaming an
export or placing it under `tests/` does not make it a public fixture. Tests
should construct synthetic inputs in temporary directories. Source code,
dependency lockfiles, and public documentation remain publishable.

The checker reports common private-file, dataset, email, credential, and
local-path patterns without printing their values. It checks tracked files even
when ignore rules match them; `--staged` inspects the actual staged contents.
These are preventive checks, not a guarantee of anonymization, licensing
compliance, or removal from earlier Git history. Never force-add ignored data.
