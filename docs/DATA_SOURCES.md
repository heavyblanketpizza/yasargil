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
| Tool-tip annotations | [sospine_tool_tips.csv](https://doi.org/10.6084/m9.figshare.20171135.v1) | `sospine_tool_tips.csv` |
| Computed bounding boxes | [sospine_bbox.csv](https://doi.org/10.6084/m9.figshare.20171129.v1) | `sospine_bbox.csv` |
| Simulated trial outcomes | [sospine_outcomes.csv](https://doi.org/10.6084/m9.figshare.20171132.v1) | `sospine_outcomes.csv` |
| Original description and notices | [readme.txt](https://figshare.com/articles/dataset/readme_txt/20171138) | `documentation/readme.txt` |

Preserve the original notices with your download. The inspected author readme
says CC BY-NC 4.0 while the Figshare records say CC BY 4.0. Yasargil retains that
discrepancy in provenance; it does not resolve it or grant additional rights.
Check the publisher's terms for your intended use.

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
The deterministic importer re-expresses selected source instrument labels and
retains the exact evidence references. The selection and annotation workflows
produce model proposals, source hashes, model-call records, and review artifacts
in a separate output directory. Reconstructed videos use explicit sampling
assumptions; generated descriptions remain proposals pending human review.
Training export applies the documented evidence, review, and partition checks.
See [Enhancement](ENHANCEMENT.md), [Frame selection](SMART_FRAME_SELECTION.md),
and [Training](TRAINING.md).

These derived outputs may contain original images, copied CSV rows, reviewer
identifiers, and machine paths. Keep them local too. `outputs/` is ignored for
convenience, and raw data/media/export formats are ignored throughout the repo.
Internal planning notes, research reports, work logs, and the source-derived
inventory have been moved to private storage outside the repository. Public
examples use placeholder paths; tests generate synthetic data in temporary
directories. Required third-party font copyright notices remain intact.

Before a commit, run `python3 scripts/check_repo_hygiene.py`; after staging,
run `python3 scripts/check_repo_hygiene.py --staged`. CI runs the same current-tree
check. It detects common private-file, dataset, email, credential, and local-path
patterns without printing their values. It is a preventive check, not a guarantee
of anonymization or a purge of earlier Git history. Never force-add ignored data.
