"""Create a bounded, local evidence packet and an uncompleted review worksheet.

The packet is a view of an immutable archive, never a review acceptance event.
It embeds the selected raster images and does not launch a browser or modify
the record, its assets, or any pre-existing output file.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import html
import json
from pathlib import Path

from .contract import (
    canonical_hash,
    require,
    resolve_asset,
    validate_record,
    write_new_json,
)


MAX_FRAMES = 32
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_HTML_BYTES = 96 * 1024 * 1024
RUBRIC_ID = "yasargil-enhancement-review"
RUBRIC_VERSION = "2.0.0"
_HIDDEN_ANNOTATION_KINDS = {"outcome", "case_metadata", "baseline_patient_variable"}
_HIDDEN_ASSET_ROLES = {
    "case_metadata", "teacher_request", "teacher_response", "review_artifact",
    "evaluation_report", "split_manifest",
}
_RASTER_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_CRITERIA = (
    "verdict", "evidence_adequacy", "temporal_correctness",
    "clinical_appropriateness", "question_answerability",
    "adds_useful_supervision", "error_severity", "correction_time_seconds",
)


def _escape(value):
    return html.escape(str(value), quote=True)


def _json_block(value):
    return "<pre>" + _escape(json.dumps(value, indent=2, ensure_ascii=False,
                                      allow_nan=False)) + "</pre>"


def _pending_item(kind, identifier, claim_ids, message_indices):
    return {
        "item_type": kind,
        "item_id": identifier,
        "target_claim_ids": list(claim_ids),
        "target_message_indices": list(message_indices),
        **{key: None for key in _CRITERIA},
        "replacement_claim_ids": [],
        "notes": None,
    }


def write_review_packet(record, output, *, dataset_root, artifact_root=None,
                        include_outcomes=False):
    """Write HTML and ``<stem>.review-template.json``; return their absolute paths.

    ``output`` must end in .html or .htm. Both files must be new and outside
    ``dataset_root``. Default presentation withholds retrospective metadata and
    generator identity; exact conversation text and visible images are never
    redacted, so effective reviewer blinding still needs human attestation.
    A maximum of 32 frames, 32 MiB of image bytes and 96 MiB of final HTML keeps
    the standalone packet bounded. No completed review is inferred or written.
    """
    require(dataset_root is not None, "Review packet requires a dataset root")
    output = Path(output)
    require(output.suffix.lower() in {".html", ".htm"},
            "Review output must end in .html or .htm")
    template_path = output.with_suffix(".review-template.json")
    for path in (output, template_path):
        require(not path.exists() and not path.is_symlink(),
                f"Review output already exists: {path}")
        require(not path.resolve().is_relative_to(Path(dataset_root).resolve()),
                "Review output must not be written inside the source dataset")
    output, template_path = output.resolve(), template_path.resolve()
    require(len(record.get("frame_selection", {}).get("frames", [])) <= MAX_FRAMES,
            f"Review packet is limited to {MAX_FRAMES} selected frames")
    validate_record(record, dataset_root=dataset_root, artifact_root=artifact_root)

    assets = {asset["asset_id"]: asset for asset in record["assets"]}
    frames = record["frame_selection"]["frames"]
    frames_by_location = {assets[f["asset_id"]]["location"]: f for f in frames}
    frame_anchor = {f["frame_id"]: f"frame-{i}" for i, f in enumerate(frames)}
    archive_sha256 = canonical_hash(record)

    # A metadata CSV may hold the entire outcome row under a non-outcome name.
    # Withhold its asset as well, including annotations that share that asset.
    hidden_assets = {a["asset_id"] for a in assets.values()
                     if a["role"] in _HIDDEN_ASSET_ROLES}
    hidden_assets.update(o["source_locator"]["asset_id"] for o in record["case_outcomes"])
    hidden_assets.update(a["source_locator"]["asset_id"]
                         for a in record["original_annotations"]
                         if a["original_kind"] in _HIDDEN_ANNOTATION_KINDS)
    for run in record["generation_runs"]:
        hidden_assets.update((run["request_asset_id"], run["response_asset_id"]))
    changed = True
    while changed:
        additional = {a["asset_id"] for a in assets.values()
                      if set(a["derived_from_asset_ids"]) & hidden_assets}
        changed = not additional <= hidden_assets
        hidden_assets.update(additional)
    visible_annotations = [
        a for a in record["original_annotations"]
        if a["original_kind"] not in _HIDDEN_ANNOTATION_KINDS
        and a["source_locator"]["asset_id"] not in hidden_assets
    ]
    visible_annotation_ids = {a["annotation_id"] for a in visible_annotations}

    image_urls, image_bytes = {}, 0
    for frame in frames:
        asset = assets[frame["asset_id"]]
        require(asset["media_type"] in _RASTER_MEDIA_TYPES,
                "Review images must be JPEG, PNG, WebP or GIF raster assets")
        path = resolve_asset(asset["location"], dataset_root, artifact_root)
        require(image_bytes + path.stat().st_size <= MAX_IMAGE_BYTES,
                "Review images exceed the 32 MiB packet limit")
        with path.open("rb") as stream:
            data = stream.read(MAX_IMAGE_BYTES - image_bytes + 1)
        image_bytes += len(data)
        require(image_bytes <= MAX_IMAGE_BYTES, "Review images exceed the 32 MiB packet limit")
        require(hashlib.sha256(data).hexdigest() == asset["sha256"],
                "Image bytes changed after archive validation")
        image_urls[asset["location"]] = (
            "data:" + asset["media_type"] + ";base64," + base64.b64encode(data).decode("ascii")
        )

    items = [
        _pending_item("turn", turn["turn_id"], turn["expressed_claim_ids"],
                      [turn["user_message_index"], turn["assistant_message_index"]])
        for turn in record["training_view"]["turn_links"]
    ]
    items.extend(
        _pending_item("claim", claim["claim_id"], [claim["claim_id"]],
                      sorted({loc["message_index"] for loc in claim["output_locations"]}))
        for claim in record["claims"]
    )
    template = {
        "record_id": record["record_id"],
        "revision_id": record["revision"]["revision_id"],
        "archive_sha256": archive_sha256,
        "worksheet_status": "pending_review",
        "rubric_id": RUBRIC_ID,
        "rubric_version": RUBRIC_VERSION,
        "reviewer_id": None,
        "reviewer_role": None,
        "reviewed_at": None,
        "presentation": {
            "mode": "retrospective_unblinded" if include_outcomes else "blinded_metadata",
            "outcome_metadata_included": bool(include_outcomes),
            "generator_metadata_included": bool(include_outcomes),
        },
        "outcome_hidden_during_review": None,
        "generator_identity_hidden": None,
        "visual_or_textual_disclosure_observed": None,
        "blinding_notes": None,
        "instructions": (
            "This is an uncompleted worksheet, not an accepted review. Inspect the exact "
            "conversation, each claim span and its source evidence. Complete only applicable "
            "criteria and record missing evidence. Record any outcome or generator disclosure "
            "from images, a visible leak test, text, filenames, prior knowledge or the optional "
            "retrospective section; do not infer effective blinding from packet mode. "
            "A completed review must be separately recorded against this archive revision."
        ),
        "items": items,
    }

    parts, rendered_bytes = [], 0

    def append(fragment):
        nonlocal rendered_bytes
        rendered_bytes += len(fragment.encode("utf-8"))
        require(rendered_bytes <= MAX_HTML_BYTES, "Review HTML exceeds the 96 MiB packet limit")
        parts.append(fragment)

    def show_image(frame, *, compact=False):
        asset = assets[frame["asset_id"]]
        return (
            '<img class="' + ("message-image" if compact else "source-image")
            + '" src="' + image_urls[asset["location"]] + '" alt="'
            + _escape("Source evidence: " + frame["frame_id"])
            + '" loading="lazy">'
        )

    append("""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Yasargil evidence review packet</title><style>
body{max-width:1100px;margin:2rem auto;padding:0 1.2rem;font:16px/1.5 system-ui,sans-serif;color:#20262b;background:#fff}
h1,h2,h3{line-height:1.25}section{margin:2rem 0}article,figure{margin:1rem 0;padding:1rem;border:1px solid #ccd5dc;border-radius:6px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:1rem;font:14px/1.5 ui-monospace,monospace}
code,figcaption,p{overflow-wrap:anywhere}.notice{border-left:5px solid #b07712;padding:1rem;background:#fff6e2}
.source-image{display:block;width:100%;height:auto}.message-image{display:block;max-width:100%;max-height:560px;height:auto;margin:1rem 0}
.message-text{font:inherit;background:#f4f6f8}.metadata{font-size:.92rem;color:#394957}a{color:#145a9b}
@media print{body{max-width:none}article,figure{break-inside:avoid}details{display:block}}
</style></head><body>""")
    append("<h1>Evidence review packet</h1><p>Record <strong>" + _escape(record["record_id"])
           + "</strong> · revision " + _escape(record["revision"]["revision_id"])
           + "</p><p class=metadata>Canonical archive SHA-256: <code>"
           + archive_sha256 + "</code></p>")
    append('<p class="notice"><strong>Pending human assessment.</strong> '
           "This packet and its blank worksheet do not establish claim correctness or training eligibility. "
           "Review the actual questions, answers and evidence, including the information available at each turn.</p>")
    if include_outcomes:
        append('<p class="notice"><strong>Retrospective, unblinded packet.</strong> '
               "Case outcomes and generator details appear in a separate section below.</p>")
    else:
        append('<p class="notice"><strong>Metadata blinding requested.</strong> '
               "Case outcomes, baseline and case metadata, claim origin, prior review decisions and "
               "generator details are withheld. Exact messages and images are preserved. A visible leak "
               "test, other visual evidence, text, source filenames or prior knowledge may disclose "
               "outcomes or identity. Record any disclosure in the worksheet; this page cannot attest "
               "that a reviewer remained blinded.</p>")

    source = record["source"]
    append("<section><h2>Case source trace</h2>" + _json_block({
        key: source[key] for key in ("dataset_name", "dataset_version", "setting", "procedure", "case_id")
    }))
    trace_ids = {f["asset_id"] for f in frames}
    trace_ids.update(a["source_locator"]["asset_id"] for a in visible_annotations)
    trace_ids.add(source["source_manifest_asset_id"])
    trace_ids.update(source["license_evidence"]["source_asset_ids"])
    trace_ids.update(aid for c in record["claims"] for aid in c["evidence"]["reference_asset_ids"])
    trace_ids.update(aid for t in record["training_view"]["turn_links"]
                     for aid in t["student_reference_asset_ids"])
    append(_json_block([
        {key: a[key] for key in ("asset_id", "location", "sha256", "media_type")}
        for a in record["assets"] if a["asset_id"] in trace_ids - hidden_assets
    ]))
    append("<p>Source locators below use the archive's exact CSV row numbers and raw values. "
           "No outcome table or teacher request/response content is included in this trace.</p></section>")

    selection = record["frame_selection"]
    append("<section><h2>Selected frames and timing</h2>" + _json_block({
        key: selection[key] for key in ("sequence_id", "released_fps", "method", "selector_exposure",
                                       "known_sampling_limitations")
    }))
    append("<p>Frame indices preserve source order. Estimated timestamps are not verified original "
           "presentation timestamps. Per-turn cutoffs below determine which frames belonged to each input.</p>")
    for frame in frames:
        asset = assets[frame["asset_id"]]
        append('<figure id="' + frame_anchor[frame["frame_id"]] + '"><figcaption><strong>'
               + _escape(frame["frame_id"]) + "</strong> · " + _escape(asset["location"])
               + "</figcaption>" + show_image(frame)
               + _json_block({key: frame[key] for key in (
                   "frame_index", "timestamp_ms", "timestamp_basis", "role", "context_frame_ids")})
               + "</figure>")
    append("</section><section><h2>Exact conversation</h2><p>Message and content block indices are "
           "zero-based. Text is shown verbatim, with HTML characters escaped for display. "
           "Source annotations elsewhere on this page are reviewer evidence; their presence does not "
           "mean they were provided to the model.</p>")
    messages = record["training_view"]["messages"]
    for turn in record["training_view"]["turn_links"]:
        append("<article><h3>Turn " + _escape(turn["turn_id"]) + "</h3>" + _json_block({
            key: turn[key] for key in ("task", "input_mode", "cutoff_frame_index", "student_frame_ids",
                                       "expressed_claim_ids")
        }))
        for mi in (turn["user_message_index"], turn["assistant_message_index"]):
            message = messages[mi]
            append('<div id="message-' + str(mi) + '"><h3>' + _escape(message["role"])
                   + " · message " + str(mi) + "</h3>")
            for bi, block in enumerate(message["content"]):
                append('<div id="message-' + str(mi) + "-block-" + str(bi)
                       + '"><p class=metadata>Content block ' + str(bi) + " · "
                       + _escape(block["type"]) + "</p>")
                if block["type"] == "text":
                    append('<pre class="message-text">' + _escape(block["text"]) + "</pre>")
                else:
                    frame = frames_by_location[block["image"]]
                    append("<p><code>" + _escape(block["image"]) + '</code> · <a href="#'
                           + frame_anchor[frame["frame_id"]] + '">Frame evidence</a></p>'
                           + show_image(frame, compact=True))
                append("</div>")
            append("</div>")
        append("</article>")
    append("</section><section><h2>Claims and exact output spans</h2><p>Spans use Unicode codepoints: "
           "zero-based start inclusive, end exclusive, within the cited text block. Evidence references "
           "are citations to inspect, not proof that the claim is supported.</p>")
    for ci, claim in enumerate(record["claims"]):
        evidence = copy.deepcopy(claim["evidence"])
        omitted = len(set(evidence["original_annotation_ids"]) - visible_annotation_ids)
        omitted += len(set(evidence["reference_asset_ids"]) & hidden_assets)
        evidence["original_annotation_ids"] = [aid for aid in evidence["original_annotation_ids"]
                                                if aid in visible_annotation_ids]
        evidence["reference_asset_ids"] = [aid for aid in evidence["reference_asset_ids"]
                                            if aid not in hidden_assets]
        append('<article id="claim-' + str(ci) + '"><h3>' + _escape(claim["claim_id"])
               + '</h3><pre class="message-text">' + _escape(claim["text"]) + "</pre>"
               + _json_block({"evidence": evidence, "output_locations": claim["output_locations"]}))
        if omitted:
            append("<p>Some evidence references are withheld from this view because they concern "
                   "restricted metadata. Do not treat omitted evidence as reviewed.</p>")
        append("</article>")
    append("</section><section><h2>Original annotation evidence</h2><p>Raw source values are preserved "
           "without coordinate repair, reinterpretation or promotion to verified findings.</p>")
    for annotation in visible_annotations:
        append("<article><h3>" + _escape(annotation["annotation_id"]) + "</h3>" + _json_block({
            key: annotation[key] for key in ("source_locator", "original_kind", "original_origin",
                                             "frame_ids", "raw_value")
        }) + "</article>")
    if not visible_annotations:
        append("<p>No original annotation rows are available in this view.</p>")
    append("</section>")
    if include_outcomes:
        append('<section class="notice"><h2>Retrospective information — unblinded</h2>'
               "<p>This section can inform a separate retrospective assessment. A case endpoint "
               "does not establish that an individual claim or action was correct.</p>"
               "<h3>Case outcomes</h3>" + _json_block(record["case_outcomes"])
               + "<h3>Generation runs</h3>" + _json_block(record["generation_runs"])
               + "<h3>Selection configuration</h3>" + _json_block({
                   key: selection[key] for key in ("selector_version", "parameters")
               }) + "<h3>Per-frame selection reasons</h3>" + _json_block([
                   {key: frame[key] for key in ("frame_id", "selection_reason")} for frame in frames
               ]) + "</section>")
    else:
        append("<p class=metadata>Free-text selection reasons, arbitrary selector parameters and "
               "selector implementation identity are withheld from the blinded packet because they "
               "may disclose retrospective labels or generator identity. Selection method, declared "
               "temporal exposure and frame timing remain visible.</p>")
    append("<section><h2>Review worksheet</h2><p>Complete the neighboring <code>"
           + _escape(template_path.name) + "</code> after inspecting the evidence. All verdicts and "
           "applicable criteria start empty. The worksheet does not update this archive or create an "
           "accepted review.</p><p>Rubric: " + _escape(RUBRIC_ID) + " · " + _escape(RUBRIC_VERSION)
           + "</p></section></body></html>")

    # Construct and bound the entire packet before creating either output.
    rendered = "".join(parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    try:
        write_new_json(template_path, template)
    except Exception:
        # Remove only the HTML created by this invocation if its paired write fails.
        output.unlink()
        raise
    return {"html": str(output), "review_template": str(template_path)}
