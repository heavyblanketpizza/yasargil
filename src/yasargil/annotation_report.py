"""Offline review of frame annotations with explicit observation/context boundaries."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
import tempfile


def _escape(value):
    return html.escape(str(value), quote=True)


def _read(path, warnings):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        warnings.append(f"Could not read {path.name}: {error}")
        return None


def _path(value, root):
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _link(path, label):
    if path is None:
        return _escape(label)
    return f'<a href="{_escape(path.absolute().as_uri())}">{_escape(label)}</a>'


def _timestamp(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "Unavailable"
    seconds = value / 1000
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def _details(label, value):
    return (f'<details><summary>{_escape(label)}</summary><pre>'
            f'{_escape(json.dumps(value, indent=2, ensure_ascii=False))}</pre></details>')


def _provenance(frame, root):
    source = _path(frame.get("source_path"), root)
    return f'''<dl class="provenance">
<dt>Source file</dt><dd>{_link(source, source.name if source else 'Unavailable')}</dd>
<dt>Timestamp basis</dt><dd>{_escape(frame.get('timestamp_basis', 'Unavailable'))}</dd>
<dt>Frame index</dt><dd>{_escape(frame.get('frame_index', 'Unavailable'))}</dd>
<dt>Source SHA-256</dt><dd class="hash">{_escape(frame.get('source_sha256', 'Unavailable'))}</dd>
<dt>Image SHA-256</dt><dd class="hash">{_escape(frame.get('image_sha256', 'Unavailable'))}</dd>
</dl>'''


def _thumbnail(frame, root, *, evidence=False):
    image = _path(frame.get("image_path"), root)
    label = "Supporting source frame" if evidence else "Selected frame"
    if image is None:
        return '<div class="empty-image">Image unavailable</div>'
    uri = _escape(image.absolute().as_uri())
    return (f'<a href="{uri}"><img loading="lazy" src="{uri}" '
            f'alt="{label} {_escape(frame.get("frame_id", "unidentified"))}"></a>')


def _evidence(interval, by_id, root):
    frames = interval.get("supporting_frames") or []
    rows = []
    for supplied in frames:
        if not isinstance(supplied, dict):
            continue
        frame = by_id.get(supplied.get("frame_id"), supplied)
        rows.append(f'''<article class="evidence-frame">{_thumbnail(frame, root, evidence=True)}
<div><b>{_timestamp(frame.get('timestamp_ms'))}</b> · {_escape(frame.get('frame_id', 'Unavailable'))}
{_provenance(frame, root)}{_details('Supporting frame provenance', frame)}</div></article>''')
    contents = ("".join(rows) if rows else
                '<p class="muted">No supporting source observations are recorded for this interval.</p>')
    return (f'<details class="evidence"><summary>Evidence locator · '
            f'{_timestamp(interval.get("start_ms"))}–{_timestamp(interval.get("end_ms"))}'
            f' · {len(rows)} source observation(s)</summary>{contents}</details>')


def _frame(frame, annotation, by_id, root):
    frame_id = frame.get("frame_id", "unidentified")
    if annotation is None:
        contents = '<p class="pending">Annotation pending. This frozen frame has no saved annotation yet.</p>'
    else:
        claims = []
        for claim in annotation.get("contextual_claims") or []:
            if not isinstance(claim, dict):
                continue
            intervals = "".join(_evidence(interval, by_id, root)
                                for interval in claim.get("evidence_intervals") or []
                                if isinstance(interval, dict))
            claims.append(f'<article class="claim"><p>{_escape(claim.get("claim", "Unavailable"))}</p>'
                          + intervals + '</article>')
        uncertainties = annotation.get("uncertainties") or []
        if not isinstance(uncertainties, list):
            uncertainties = [uncertainties]
        uncertainty_html = ('<ul>' + ''.join(f'<li>{_escape(item)}</li>' for item in uncertainties) + '</ul>'
                            if uncertainties else '<p>No uncertainty was reported by the model.</p>')
        context_html = ''.join(claims) or '<p>No additional contextual claim was reported.</p>'
        contents = f'''<div class="observation"><h3>Visible in this frame · model draft</h3>
<p>{_escape(annotation.get('visible_observation', 'Unavailable'))}</p>
<p class="muted">Visibility: {_escape(annotation.get('visibility', 'Unavailable'))}</p></div>
<div class="context"><h3>Added by the surrounding video · model draft</h3>
<p class="muted">These claims are not necessarily visible in the selected still. Evidence links identify source observations; they do not establish that a claim is correct.</p>
{context_html}</div><div class="uncertainty"><h3>Uncertainty</h3>{uncertainty_html}</div>
<p class="review">Human review required · Not eligible for training</p>
{_details('Full annotation record', annotation)}'''
    return f'''<article class="frame" id="frame-{_escape(frame_id)}">
<div class="frame-heading"><h2>{_timestamp(frame.get('timestamp_ms'))}</h2>
<span class="frame-id">{_escape(frame_id)}</span></div>
<div class="frame-layout"><div class="source">{_thumbnail(frame, root)}
{_provenance(frame, root)}{_details('Frozen selected frame provenance', frame)}</div>
<div class="annotation">{contents}</div></div></article>'''


def write_annotation_report(output_dir) -> Path:
    """Write a self-contained HTML review, including incomplete or failed runs."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    warnings = []
    summary = _read(root / "summary.json", warnings)
    if not isinstance(summary, dict):
        summary = {"status": "summary unavailable"}
    canonical = _read(root / "source/source.json", warnings)
    if not isinstance(canonical, dict):
        canonical = {}
    source = {**canonical, **(summary.get("source") if isinstance(summary.get("source"), dict) else {})}
    selected = _read(root / "selected-frames.json", warnings)
    if not isinstance(selected, list):
        selected = []
    selected = [frame for frame in selected if isinstance(frame, dict)]
    by_id = {frame["frame_id"]: frame for frame in canonical.get("frames", [])
             if isinstance(frame, dict) and "frame_id" in frame}
    # The frozen manifest supplies provenance for selected stills. Model-written
    # annotation fields must never replace the manifest's paths or timestamps.
    by_id.update({frame["frame_id"]: frame for frame in selected if "frame_id" in frame})
    annotations_path = _path(summary.get("annotations_path"), root) or root / "annotations.json"
    record = _read(annotations_path, warnings)
    if not isinstance(record, dict):
        record = {}
    annotations = record.get("annotations") or []
    annotations = [annotation for annotation in annotations if isinstance(annotation, dict)]
    annotation_by_id = {annotation["frame_id"]: annotation for annotation in annotations if "frame_id" in annotation}
    annotated_count = sum(frame.get("frame_id") in annotation_by_id for frame in selected)
    cards = ''.join(_frame(frame, annotation_by_id.get(frame.get("frame_id")), by_id, root) for frame in selected)
    if not cards:
        cards = '<p class="muted">The frozen selected-frame manifest is not available yet.</p>'
    source_path = _path(source.get("source_path", source.get("path")), root)
    video_path = _path(source.get("video_path"), root)
    source_frames = source.get("expected_video_frames", source.get("frame_count", source.get("framecount", "Unavailable")))
    duration = source.get("duration_ms", source.get("duration"))
    context_check = record.get("context_check", "Pending")
    verification = _read(root / "round-00/verification.json", warnings)
    verified = isinstance(verification, dict) and verification.get("full_source_video_verified") is True
    verification_text = (f'Complete video verified ({_escape(verification.get("decoded_frames", "?"))} decoded frames).'
                         if verified else 'Complete-video request verification is pending or unavailable.')
    links = []
    for filename, label in (("source/source.json", "Complete source manifest"),
                            ("selected-frames.json", "Frozen selected-frame manifest"),
                            ("run.json", "Configuration and pinned inputs"),
                            ("session.json", "Fresh annotation session"),
                            ("native-timeline-verification.json", "Source timeline verification"),
                            ("round-00/request.json", "Exact model request"),
                            ("round-00/response.json", "Raw model response"),
                            ("round-00/result.json", "Accepted call result"),
                            ("round-00/verification.json", "Full-video verification")):
        if (root / filename).is_file():
            links.append(_link(root / filename, label))
    if annotations_path.is_file():
        links.append(_link(annotations_path, "Complete annotation records"))
    error_html = _details("Latest annotation error", summary["error"]) if summary.get("error") else ""
    warning_html = ('<aside class="notice"><b>Artifact warnings</b><ul>'
                    + ''.join(f'<li>{_escape(warning)}</li>' for warning in warnings) + '</ul></aside>') if warnings else ''
    timeline_note = (f'<p class="muted">{_escape(source["timeline_note"])}</p>' if source.get("timeline_note") else '')
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src file:; style-src 'unsafe-inline'">
<title>Contextual frame annotation review</title><style>
:root{{color-scheme:light;--ink:#173044;--muted:#526477;--border:#d9e1e8;--paper:#fff;--canvas:#f0f4f7}}
*{{box-sizing:border-box}}body{{font:16px/1.55 system-ui,sans-serif;color:var(--ink);background:var(--canvas);margin:0}}
main{{max-width:1380px;margin:auto;padding:36px 24px 70px}}h1{{font-size:2rem;line-height:1.2;margin:0 0 12px}}h2,h3{{margin:0 0 10px}}h3{{font-size:1rem}}p{{margin:8px 0 16px}}a{{color:#135fa3;overflow-wrap:anywhere}}
.muted{{color:var(--muted)}}.notice,.overview,.frame{{background:var(--paper);border:1px solid var(--border);border-radius:10px;padding:22px;margin:22px 0}}
.notice{{background:#fff8e8;border-left:5px solid #b57c16}}.status,.review{{font-weight:650}}.status{{display:inline-block;padding:3px 10px;background:#e5edf3;border-radius:6px}}
.facts{{display:flex;gap:22px;flex-wrap:wrap;margin-top:18px}}.facts div{{flex:1;min-width:190px}}.facts b{{display:block;font-size:1.1rem}}
.frame-heading{{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:16px}}.frame-id{{font:12px ui-monospace,monospace;overflow-wrap:anywhere}}
.frame-layout{{display:grid;grid-template-columns:minmax(250px,2fr) minmax(300px,3fr);gap:24px}}img{{display:block;width:100%;height:auto;border-radius:6px}}
.provenance{{display:grid;grid-template-columns:125px 1fr;gap:5px 10px;font-size:13px}}dt{{color:var(--muted)}}dd{{margin:0;overflow-wrap:anywhere}}.hash{{font:11px/1.6 ui-monospace,monospace}}
.observation,.context,.uncertainty{{padding:16px 18px;border-left:4px solid;margin-bottom:18px}}.observation{{background:#edf7f3;border-color:#328060}}.context{{background:#edf3fc;border-color:#517db4}}.uncertainty{{background:#faf6e9;border-color:#a5883a}}
.claim{{border-top:1px solid #c9d8eb;padding:8px 0}}.review{{color:#705019}}details{{margin:12px 0}}summary{{cursor:pointer;font-weight:550}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:12px;border-radius:5px;font-size:12px}}
.evidence-frame{{display:grid;grid-template-columns:140px 1fr;gap:14px;margin:15px 0;background:white;padding:12px;border-radius:6px}}.evidence-frame .provenance{{grid-template-columns:115px 1fr;font-size:12px}}.empty-image{{padding:40px;background:#e8edf2;text-align:center}}.pending{{color:var(--muted)}}
@media(max-width:800px){{main{{padding:22px 12px}}.frame-layout,.evidence-frame{{grid-template-columns:1fr}}.evidence-frame img{{max-width:280px}}.provenance{{grid-template-columns:110px 1fr}}}}
</style></head><body><main>
<h1>Contextual frame annotation review</h1><p>Frozen selected frames, described in a fresh session with the complete video.</p>
<span class="status">{_escape(summary.get('status', 'Unavailable'))}</span>
<aside class="notice"><b>Model drafts · Human review required · Not eligible for training</b>
<p>Visible observations and video-context claims are separate. Context can explain a selected frame, but an event elsewhere in the video must not be represented as visible in that still.</p>
<p>Temporal exposure: retrospective full video. Evidence validation checks source locators only, not the truth of a claim. Clinical validation has not been performed.</p></aside>
<section class="overview"><div class="facts">
<div><b>{annotated_count} / {len(selected)} annotations saved</b>Selected frames are frozen</div>
<div><b>{_timestamp(duration)} playback duration</b>{_escape(source_frames)} available video frames</div>
<div><b>Context check: {_escape(context_check)}</b>{verification_text}</div></div>
<p><b>Source:</b> {_link(source_path, str(source_path) if source_path else 'Unavailable')}<br>
<b>Video:</b> {_link(video_path, video_path.name if video_path else 'Unavailable')}<br>
<b>Timestamp basis:</b> {_escape(source.get('timestamp_basis', 'Unavailable'))}<br>
<b>Clock:</b> Supplied video playback; recorded repair time does not set these timestamps.<br>
<b>Session:</b> {_escape(summary.get('session_id', 'Pending'))}<br>
<b>Created:</b> {_escape(summary.get('created_at', 'Unavailable'))} · <b>Updated:</b> {_escape(summary.get('updated_at', 'Unavailable'))}</p>
{timeline_note}<p>{' · '.join(links)}</p>{error_html}</section>
{warning_html}{cards}
</main></body></html>'''
    path = root / "report.html"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                         prefix=".annotation-report-", suffix=".html", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(document)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path
