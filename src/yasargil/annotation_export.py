"""Export human-approved MedGemma frame annotations as reviewed SFT conversations.

An inspector worksheet is a draft. Only frames its reviewer marked Complete, for
the exact data revision they saw, become rows. The declared reviewer identity is
recorded, not authenticated. Neighboring context frames include later frames, so
every row is retrospective review data rather than causal intraoperative input.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from .contract import (
    ANNOTATION_EXPORT_VERSION, ContractError, _portable_messages, canonical_hash, require, sha256_file,
    write_new_json,
)


PARTITIONS = ("train", "validation", "test")
REVIEWER_ROLES = ("surgeon", "clinical_domain_expert")
INTENDED_USE = "retrospective_surgical_review"
LOSS_SCOPE = "final_assistant_turn"
INSTRUCTION = ("Annotate the target surgical frame. Report its visibility. Separate claims directly visible "
               "in the target from context-supported interpretations, cite the view IDs supporting each "
               "claim, state specific uncertainty, and list unresolved questions.")


def _humanize(value):
    return str(value).replace("_", " ")


def annotation_text(annotation):
    """Render an annotation exactly as the inspector's editor starts from it.

    An unedited approval and a human revision therefore share one target format.
    """
    claims = [f"{_humanize(claim['support'])} · {_humanize(claim['category'])}: {claim['statement']}\n"
              f"Evidence: {', '.join(claim['evidence_view_ids'])}"
              + (f"\nUncertainty: {claim['uncertainty']}" if claim["uncertainty"] else "")
              for claim in annotation["claims"]]
    questions = [f"{question['question']} {question['reason']} (Evidence needed: {_humanize(question['kind'])})"
                 for question in annotation["unresolved_questions"]]
    return "\n\n".join([f"Visibility: {_humanize(annotation['visibility'])}", *claims,
                        *(["Unresolved questions:", *questions] if questions else [])])


def _surgeon(case_id):
    match = re.fullmatch(r"(S\d+)A\d+", case_id or "")
    return match.group(1) if match else None


def _read_worksheet(path):
    try:
        worksheet = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError(f"Cannot read review worksheet {path}: {exc}") from exc
    require(isinstance(worksheet, dict) and worksheet.get("schema_version") == "yasargil-inspector-review-v1",
            f"Expected an inspector review export: {path}")
    require(all(isinstance(worksheet.get(key), str) and worksheet[key] for key in ("record_id", "review_identity")),
            f"Review worksheet lacks its record identity: {path}")
    require(isinstance(worksheet.get("notes"), list) and all(isinstance(note, dict) for note in worksheet["notes"]),
            f"Review worksheet notes are malformed: {path}")
    return worksheet


def _location(path, artifact_root):
    path = Path(path).resolve()
    require(path.is_relative_to(artifact_root), f"Evidence image is outside the runs root: {path}")
    return "artifact:" + path.relative_to(artifact_root).as_posix()


def _row(detail, curation, artifact_root, media):
    context = detail["evidence_context"]
    content = [{"type": "text", "text": INSTRUCTION},
               {"type": "text", "text": f"Target frame: {context['target_frame_id']}. "
                                        f"Procedure context: {context['procedure_context'] or 'none documented'}."}]
    for view in detail["evidence"]:
        location = _location(view["image_path"], artifact_root)
        require(sha256_file(artifact_root / location.removeprefix("artifact:")) == view["image_sha256"],
                f"Evidence image changed after annotation: {view['view_id']}")
        entry = {"location": location, "sha256": view["image_sha256"], "width": view["width"], "height": view["height"]}
        require(media.setdefault(location, entry) == entry, "Conflicting exported image bytes")
        content += [{"type": "text", "text": f"View {view['view_id']} ({_humanize(view['role'])}, "
                                             f"{view['timestamp_ms']} ms)"},
                    {"type": "image", "image": location}]
    edited = (curation or {}).get("annotations", {}).get("medgemma")
    answer = edited if isinstance(edited, str) else annotation_text(detail["medgemma"])
    require(answer.strip(), f"Approved annotation text is empty: {context['target_frame_id']}")
    row = {"messages": [{"role": "user", "content": content},
                        {"role": "assistant", "content": [{"type": "text", "text": answer}]}]}
    _portable_messages(row)
    return row


def _approved(store, worksheet, path):
    """Yield frame details whose Complete decision covers the current annotation."""
    record = store.record(worksheet["record_id"])
    require(record is not None, f"The reviewed record is no longer available: {path}")
    require(record["review_identity"] == worksheet["review_identity"],
            f"The dataset changed after this review, or the inspector used a different --dataset-root: {path}")
    frames = {frame["frame_id"] for frame in record["frames"]}
    seen = set()
    for note in worksheet["notes"]:
        frame_id = note.get("frame_id")
        require(frame_id in frames and frame_id not in seen, f"Review worksheet has an unknown or repeated frame: {path}")
        seen.add(frame_id)
        if note.get("status") != "reviewed" or note.get("needs_recheck"):
            continue
        detail = store.frame(worksheet["record_id"], frame_id)
        curation = detail["curation"]
        if (detail["medgemma"] is None or (curation and curation["deleted"])
                or note.get("curation_updated_at") != (curation["updated_at"] if curation else None)):
            continue
        yield record, detail, curation


def _release_index(output):
    """Cases, surgeons, and images already exported in this release directory."""
    used = {}
    for receipt_path in sorted(output.parent.glob("*.receipt.json")):
        if receipt_path.resolve() == output.with_suffix(".receipt.json").resolve():
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ContractError(f"Cannot read release receipt {receipt_path}: {exc}") from exc
        if receipt.get("schema_version") != ANNOTATION_EXPORT_VERSION:
            continue
        for sheet in receipt["review"]["worksheets"]:
            used.setdefault(("case", sheet["case_id"]), receipt["partition"])
            if sheet["surgeon_id"]:
                used.setdefault(("surgeon", sheet["surgeon_id"]), receipt["partition"])
        for entry in receipt["media"]:
            used.setdefault(("image", entry["sha256"]), receipt["partition"])
    return used


def export_reviewed_annotations(worksheets, output, *, runs_root, partition, reviewer, reviewer_role,
                                dataset_root=None):
    from .dataset_inspector import InspectorStore

    require(partition in PARTITIONS, "Select an explicit train, validation, or test partition")
    require(isinstance(reviewer, str) and reviewer.strip(), "Declare who reviewed these frames")
    require(reviewer_role in REVIEWER_ROLES, "Reviewer role must be surgeon or clinical_domain_expert")
    output = Path(output).expanduser().resolve()
    receipt_path = output.with_suffix(".receipt.json")
    require(output.suffix == ".jsonl", "Portable export must use a .jsonl filename")
    require(not output.exists() and not receipt_path.exists(), "Export or receipt already exists")
    if dataset_root is not None:
        require(not output.is_relative_to(Path(dataset_root).expanduser().resolve()),
                "Export output must be outside the source dataset")
    artifact_root = Path(runs_root).expanduser().resolve()
    store = InspectorStore(artifact_root, dataset_root)
    rows, records, media, sheets = [], [], {}, []
    for path in worksheets:
        worksheet = _read_worksheet(path)
        approved = list(_approved(store, worksheet, path))
        require(approved, f"No current frames are marked Complete in {path}")
        record = approved[0][0]
        for _, detail, curation in approved:
            row = _row(detail, curation, artifact_root, media)
            rows.append(row)
            records.append({"record_id": f"{record['case_id']}:{detail['frame']['frame_id']}",
                            "revision_id": worksheet["review_identity"],
                            "archive_sha256": canonical_hash({"annotation": detail["raw"]["medgemma"],
                                                              "curation": curation}),
                            "row_sha256": canonical_hash(row)})
        medgemma = record["runs"]["medgemma"]
        sheets.append({"worksheet_sha256": sha256_file(path), "record_id": worksheet["record_id"],
                       "review_identity": worksheet["review_identity"], "case_id": record["case_id"],
                       "surgeon_id": _surgeon(record["case_id"]), "annotation_run": medgemma["path"],
                       "annotations_sha256": sha256_file(Path(medgemma["path"]) / "annotations.json"),
                       "approved_frame_count": len(approved)})
    keys = [("case", sheet["case_id"]) for sheet in sheets]
    require(len(set(keys)) == len(keys), "Each case may appear only once in a release")
    used = _release_index(output)
    for key in keys:
        require(key not in used, f"Case {key[1]} is already exported in this release")
    for key in [("surgeon", s["surgeon_id"]) for s in sheets if s["surgeon_id"]] + [("image", m["sha256"]) for m in media.values()]:
        require(used.get(key, partition) == partition, f"Cross-partition leakage: {key[0]} {key[1]} is in {used.get(key)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    receipt = {"schema_version": ANNOTATION_EXPORT_VERSION, "purpose": "reviewed_sft_export",
               "row_count": len(rows), "partition": partition, "loss_scope": LOSS_SCOPE,
               "intended_use": INTENDED_USE, "output_sha256": sha256_file(output), "records": records,
               "media": list(media.values()),
               "review": {"reviewer": reviewer.strip(), "reviewer_role": reviewer_role,
                          "identity_verified": False, "worksheets": sheets}}
    write_new_json(receipt_path, receipt)
    return receipt


def add_export_parser(subparsers):
    parser = subparsers.add_parser("export-reviewed-annotations",
                                   help="Export MedGemma annotations a reviewer marked Complete as SFT rows")
    parser.add_argument("--review", type=Path, nargs="+", required=True, help="Inspector review export(s)")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--dataset-root", type=Path, help="The same dataset root the inspector used, if any")
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewer-role", choices=REVIEWER_ROLES, required=True)
    parser.add_argument("--output", type=Path, required=True)


def export_cli(args):
    receipt = export_reviewed_annotations(args.review, args.output, runs_root=args.runs_root,
                                          partition=args.partition, reviewer=args.reviewer,
                                          reviewer_role=args.reviewer_role, dataset_root=args.dataset_root)
    print(f"Wrote {receipt['row_count']} reviewed row(s) to {args.output} ({receipt['partition']}).")
