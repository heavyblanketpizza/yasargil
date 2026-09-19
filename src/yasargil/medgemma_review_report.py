"""Offline inspection of MedGemma revisions and deferred evidence requests."""
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


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _rows(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _path(value, root):
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


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


def _list(value, empty):
    values = value if isinstance(value, list) else ([value] if value else [])
    if not values:
        return f'<p class="muted">{_escape(empty)}</p>'
    return '<ul>' + ''.join(f'<li>{_escape(item)}</li>' for item in values) + '</ul>'


def _annotation(annotation, *, revised=False):
    annotation = _mapping(annotation)
    if not annotation:
        return '<p class="muted">No saved annotation.</p>'
    claims = []
    for claim in _rows(annotation.get("contextual_claims")):
        evidence = []
        for interval in _rows(claim.get("evidence_intervals")):
            evidence.append(f'<li>Video interval: {_timestamp(interval.get("start_ms"))}–'
                            f'{_timestamp(interval.get("end_ms"))}</li>')
        for frame_id in claim.get("evidence_frame_ids", []) if isinstance(claim.get("evidence_frame_ids"), list) else []:
            evidence.append(f'<li>Evidence frame: {_escape(frame_id)}</li>')
        locators = '<ul class="locators">' + ''.join(evidence) + '</ul>' if evidence else ''
        claims.append(f'<article class="claim"><p>{_escape(claim.get("claim", "Unavailable"))}</p>{locators}</article>')
    context = ''.join(claims) or '<p class="muted">No contextual claims recorded.</p>'
    heading = "MedGemma revised annotation" if revised else "Qwen original annotation"
    return f'''<h3>{heading}</h3><h4>Visible in the target frame</h4>
<p>{_escape(annotation.get('visible_observation', 'Unavailable'))}</p>
<p class="muted">Visibility: {_escape(annotation.get('visibility', 'Unavailable'))}</p>
<h4>Contextual claims</h4>{context}<h4>Uncertainties</h4>
{_list(annotation.get('uncertainties'), 'No uncertainty recorded.') }'''


def _evidence_frame(frame, root):
    image = _path(frame.get("image_path"), root)
    source = _path(frame.get("source_path"), root)
    frame_id = frame.get("frame_id", "Unavailable")
    roles = frame.get("evidence_roles") or []
    if not isinstance(roles, list):
        roles = [roles]
    image_html = (f'<a href="{_escape(image.absolute().as_uri())}"><img loading="lazy" '
                  f'src="{_escape(image.absolute().as_uri())}" alt="Evidence frame {_escape(frame_id)}"></a>'
                  if image else '<div class="empty-image">Image unavailable</div>')
    return f'''<figure>{image_html}<figcaption><b>{_escape(frame_id)}</b>
<p>{_timestamp(frame.get('timestamp_ms'))} · {_escape(', '.join(str(role) for role in roles) or 'Role unavailable')}</p>
<p class="muted">Timestamp basis: {_escape(frame.get('timestamp_basis', 'Unavailable'))}<br>
Source: {_link(source, source.name if source else 'Unavailable')}</p>
{_details('Frame provenance', frame)}</figcaption></figure>'''


def _corrections(corrections):
    rows = []
    for correction in _rows(corrections):
        rows.append(f'''<article class="correction"><p><b>Original:</b> {_escape(correction.get('original_text', 'Unavailable'))}</p>
<p><b>Revised:</b> {_escape(correction.get('revised_text', 'Unavailable'))}</p>
<p><b>Reason:</b> {_escape(correction.get('reason', 'Unavailable'))}</p>
{_list(correction.get('evidence_frame_ids'), 'No evidence frame IDs recorded.')}</article>''')
    return ''.join(rows) or '<p class="muted">No corrections recorded.</p>'


def _requests(requests):
    rows = []
    for request in _rows(requests):
        rows.append(f'''<article class="request"><p><b>{_escape(request.get('question', 'Question unavailable'))}</b></p>
<p>{_escape(request.get('reason', 'Reason unavailable'))}</p>
<p class="muted">Target: {_escape(request.get('target', 'Unavailable'))}<br>
Requested interval: {_timestamp(request.get('start_ms'))}–{_timestamp(request.get('end_ms'))}</p>
{_details('Complete evidence request', request)}</article>''')
    return ''.join(rows) or '<p class="muted">No specific evidence request recorded.</p>'


def _card(packet, record, root):
    record = _mapping(record)
    review = _mapping(record.get("medgemma_review"))
    target_id = packet.get("target_frame_id", record.get("target_frame_id", "Unavailable"))
    # Only the saved evidence packet supplies paths, timestamps, and dataset
    # provenance. A model response cannot replace the evidence it was shown.
    frames = _rows(packet.get("frames"))
    frame_html = ''.join(_evidence_frame(frame, root) for frame in frames)
    if not frame_html:
        frame_html = '<p class="muted">No saved evidence packet is available for this frame.</p>'
    original = packet.get("qwen_annotation", record.get("qwen_annotation"))
    status = review.get("status", "pending")
    if status == "needs_more_evidence":
        status_html = '<span class="status deferred">Needs more evidence · Saved for later</span>'
    elif status == "review_complete":
        status_html = '<span class="status">Review complete · Model draft</span>'
    else:
        status_html = f'<span class="status">{_escape(status)}</span>'
    if review:
        revision = _annotation(review.get("revised_annotation"), revised=True)
        revision += f'<p class="muted">Assessment: {_escape(review.get("assessment", "Unavailable"))}</p>'
    else:
        revision = '<h3>MedGemma review pending</h3><p class="muted">No accepted review is saved for this target frame yet.</p>'
    requests = review.get("evidence_requests") or record.get("deferred_evidence_requests")
    request_html = (f'<section class="requests"><h3>Deferred evidence requests</h3>'
                    '<p>Saved for later. No Qwen or TimeLens2 search has been dispatched by this review pass.</p>'
                    f'{_requests(requests)}</section>') if status == "needs_more_evidence" or requests else ''
    call_dir = _path(record.get("call_directory"), root)
    links = []
    if call_dir is not None:
        links = [_link(call_dir / "request.json", "Exact MedGemma request"),
                 _link(call_dir / "response.json", "Full raw MedGemma response")]
    metadata = _details("Dataset context and original annotations", packet.get("dataset_context", {}))
    validation = packet.get("qwen_validation")
    draft_warning = ("<p class=\"review-note\">Qwen draft failed timestamp validation; original citations are preserved.</p>"
                     + _details("Qwen timestamp validation issues", validation)) if validation else ""
    limitations = _list(packet.get("limitations"), "No additional packet limitations recorded.")
    return f'''<article class="frame"><div class="frame-heading"><h2>{_escape(target_id)}</h2>{status_html}</div>
<details class="evidence" open><summary>Target and supporting frames · {len(frames)} image(s)</summary>
<div class="filmstrip">{frame_html}</div></details>
{draft_warning}<div class="comparison"><section class="original">{_annotation(original)}</section>
<section class="revised">{revision}</section></div>
<section class="corrections"><h3>Corrections and evidence</h3>{_corrections(review.get('corrections'))}</section>
{request_html}<details><summary>Evidence limitations</summary>{limitations}</details>{metadata}
{_details('Full evidence packet and coverage', packet) if packet else ''}
<p class="review-note">Human review required · Not eligible for training</p>
<p>{' · '.join(links)}</p>{_details('Full saved review record', record) if record else ''}</article>'''


def write_review_report(output_dir) -> Path:
    """Write an offline HTML comparison, including prepared and interrupted runs."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    warnings = []
    summary = _mapping(_read(root / "summary.json", warnings)) or {"status": "summary unavailable"}
    review_artifact = _mapping(_read(root / "reviews.json", warnings))
    records = _rows(review_artifact.get("reviews"))
    packets = []
    for path in sorted((root / "evidence").glob("*.json")):
        packet = _read(path, warnings)
        if isinstance(packet, dict):
            packets.append(packet)
    by_id = {record["target_frame_id"]: record for record in records if isinstance(record.get("target_frame_id"), str)}
    packet_ids = {packet.get("target_frame_id") for packet in packets if isinstance(packet.get("target_frame_id"), str)}
    cards = [_card(packet, by_id.get(packet.get("target_frame_id")), root) for packet in packets
             if isinstance(packet.get("target_frame_id"), str)]
    # Retain accepted reviews if a damaged or missing packet cannot be read,
    # while explicitly withholding untrusted fallback image metadata.
    cards.extend(_card({}, record, root) for target_id, record in by_id.items() if target_id not in packet_ids)
    if not cards:
        cards = ['<p class="muted">No saved evidence packets or accepted reviews are available yet.</p>']
    deferred = sum(_mapping(record.get("medgemma_review")).get("status") == "needs_more_evidence" for record in records)
    selected_count = summary.get("selected_frame_count", len(packet_ids | set(by_id)))
    reviewed_count = summary.get("reviewed_frame_count", len(records))
    deferred_count = summary.get("deferred_frame_count", deferred)
    links = [_link(root / filename, label) for filename, label in
             (("summary.json", "Run summary"), ("run.json", "Configuration and pinned inputs"),
              ("reviews.json", "All saved reviews"), ("deferred-evidence.json", "Deferred evidence requests"))
             if (root / filename).is_file()]
    error_html = _details("Latest review error", summary["error"]) if summary.get("error") else ''
    warning_html = ('<aside class="notice"><b>Artifact warnings</b>'
                    + _list(warnings, '') + '</aside>') if warnings else ''
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src file:; style-src 'unsafe-inline'">
<title>MedGemma annotation review</title><style>
:root{{color-scheme:light;--ink:#173044;--muted:#526477;--line:#d9e1e8}}*{{box-sizing:border-box}}
body{{font:16px/1.55 system-ui,sans-serif;color:var(--ink);background:#f0f4f7;margin:0}}
main{{max-width:1400px;margin:auto;padding:36px 24px 70px}}h1{{font-size:2rem;line-height:1.2;margin:0 0 12px}}
h2,h3,h4{{margin:0 0 10px}}h2{{font-size:1.2rem;overflow-wrap:anywhere}}h3{{font-size:1.05rem}}h4{{font-size:.9rem;margin-top:18px}}
p{{margin:8px 0 16px}}a{{color:#135fa3;overflow-wrap:anywhere}}.muted{{color:var(--muted)}}
.overview,.frame,.notice{{background:white;border:1px solid var(--line);border-radius:10px;padding:22px;margin:22px 0}}
.notice{{background:#fff8e8;border-left:5px solid #b57c16}}.status{{display:inline-block;padding:3px 10px;background:#e5edf3;border-radius:6px;font-weight:650}}
.deferred{{background:#fce9c4;color:#775416}}.facts,.frame-heading{{display:flex;gap:22px;flex-wrap:wrap}}.facts div{{flex:1;min-width:180px}}.facts b{{display:block;font-size:1.1rem}}.frame-heading{{align-items:baseline;margin-bottom:20px}}
.filmstrip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-top:16px}}figure{{margin:0;border:1px solid var(--line);border-radius:7px;padding:10px;min-width:0}}img{{display:block;width:100%;height:auto;border-radius:5px}}figcaption{{font-size:13px;overflow-wrap:anywhere;margin-top:10px}}.empty-image{{padding:40px 10px;text-align:center;background:#edf1f5}}
.comparison{{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin:24px 0}}.original,.revised{{padding:18px;border-left:4px solid #517db4;background:#edf3fc;min-width:0}}.revised{{border-color:#328060;background:#edf7f3}}
.claim,.correction,.request{{border-top:1px solid var(--line);padding:12px 0}}.locators{{font-size:13px}}.requests{{padding:18px;background:#fff8e8;border-left:4px solid #b57c16;margin:20px 0}}.review-note{{font-weight:650;color:#705019}}
details{{margin:12px 0}}summary{{cursor:pointer;font-weight:550}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:12px;border-radius:5px;font-size:12px}}li{{overflow-wrap:anywhere}}
@media(max-width:750px){{main{{padding:22px 12px}}.comparison{{grid-template-columns:1fr}}.filmstrip{{grid-template-columns:1fr 1fr}}}}
@media(max-width:450px){{.filmstrip{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>MedGemma annotation review</h1>
<p>Qwen annotations reviewed against the target frame, surrounding frames, and relevant dataset context.</p>
<span class="status">{_escape(summary.get('status', 'Unavailable'))}</span>
<aside class="notice"><b>Model drafts · Human review required · Not eligible for training</b>
<p>Visible observations and contextual claims remain separate. A completed model review does not establish the correctness of a claim.</p>
<p>Requests for more evidence are preserved for later. This pass does not dispatch Qwen or TimeLens2 searches.</p></aside>
<section class="overview"><div class="facts"><div><b>{_escape(reviewed_count)} / {_escape(selected_count)} reviews saved</b>Selected target frames</div>
<div><b>{_escape(deferred_count)} frame(s) need more evidence</b>Requests deferred for later</div></div>
<p>{' · '.join(links)}</p>{error_html}</section>{warning_html}{''.join(cards)}
</main></body></html>'''
    path = root / "report.html"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                         prefix=".medgemma-review-report-", suffix=".html", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(document)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path
