"""Offline inspection of independent MedGemma claims and their image evidence."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import tempfile


def _escape(value):
    return html.escape(str(value), quote=True)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _rows(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


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
    path = Path(value)
    return path if path.is_absolute() else root / path


def _link(path, label):
    return (f'<a href="{_escape(path.absolute().as_uri())}">{_escape(label)}</a>'
            if path is not None else _escape(label))


def _details(label, value):
    return (f'<details><summary>{_escape(label)}</summary><pre>'
            f'{_escape(json.dumps(value, indent=2, ensure_ascii=False))}</pre></details>')


def _anchor(target_id, view_id):
    return "view-" + hashlib.sha256(f"{target_id}:{view_id}".encode()).hexdigest()[:24]


def _view(view, target_id, root):
    image = _path(view.get("image_path"), root)
    view_id = view.get("view_id", "unavailable")
    picture = (f'<a href="{_escape(image.absolute().as_uri())}"><img loading="lazy" '
               f'src="{_escape(image.absolute().as_uri())}" alt="{_escape(view_id)}"></a>'
               if image else '<p class="muted">Image unavailable</p>')
    return (f'<figure id="{_anchor(target_id, view_id)}">{picture}<figcaption>'
            f'<b>{_escape(view_id)}</b> · {_escape(view.get("role", "unavailable"))}<br>'
            f'Frame: {_escape(view.get("frame_id", "unavailable"))}<br>'
            f'Bounds in source image: {_escape(view.get("bounds", "unavailable"))}<br>'
            f'{_escape(view.get("width", "?"))} × {_escape(view.get("height", "?"))}'
            '</figcaption></figure>')


def _claims(annotation, packet):
    target_id = packet.get("target_frame_id", annotation.get("target_frame_id", ""))
    views = {row.get("view_id"): row for row in _rows(packet.get("views"))}
    sections = []
    for support, title in (("target_visible", "Visible in the target frame"),
                           ("context_supported", "Supported by temporal or procedure context")):
        claims = []
        for claim in _rows(annotation.get("claims")):
            if claim.get("support") != support:
                continue
            locators = []
            for view_id in claim.get("evidence_view_ids", []):
                if view_id in views:
                    locators.append(f'<a href="#{_anchor(target_id, view_id)}">{_escape(view_id)}</a>')
                else:
                    locators.append(f'{_escape(view_id)} (saved view unavailable)')
            claims.append(f'<article class="claim"><span class="badge">{_escape(claim.get("category", ""))}</span>'
                          f'<p>{_escape(claim.get("statement", ""))}</p>'
                          f'<p class="muted">Evidence: {" · ".join(locators) or "Unavailable"}</p>'
                          + (f'<p class="uncertainty">Uncertainty: {_escape(claim["uncertainty"])}</p>'
                             if claim.get("uncertainty") else '') + '</article>')
        sections.append(f'<section><h3>{title}</h3>{"".join(claims) or "<p class=muted>No claims recorded.</p>"}</section>')
    return ''.join(sections)


def _card(packet, record, root):
    record = _mapping(record)
    annotation = _mapping(record.get("annotation"))
    target_id = packet.get("target_frame_id", record.get("target_frame_id", "Unavailable"))
    views = _rows(packet.get("views"))
    images = ''.join(_view(view, target_id, root) for view in views)
    if not images:
        images = '<p class="muted">No saved evidence packet is available for this frame.</p>'
    questions = ''.join(f'<article class="claim"><b>{_escape(row.get("question", ""))}</b>'
                        f'<p>{_escape(row.get("reason", ""))}</p>'
                        f'<span class="muted">Evidence needed: {_escape(row.get("kind", ""))}</span></article>'
                        for row in _rows(annotation.get("unresolved_questions")))
    body = (_claims(annotation, packet) if annotation else
            '<p class="muted">Independent MedGemma annotation pending.</p>')
    call = _path(record.get("call_directory"), root)
    links = ' · '.join(_link(call / name, label) for name, label in
                        (("request.json", "Exact MedGemma request"), ("response.json", "Full raw MedGemma response"))) if call else ''
    return f'''<article class="frame"><h2>{_escape(target_id)}</h2>
<p class="badge">{_escape(annotation.get('status', 'pending'))}</p>
<p class="muted">Visibility: {_escape(annotation.get('visibility', 'unavailable'))}</p>
<details open><summary>Target, detail crops, and context · {len(views)} image(s)</summary><div class="filmstrip">{images}</div></details>
{body}<section><h3>Unresolved questions</h3>{questions or '<p class="muted">No unresolved questions recorded.</p>'}</section>
{_details('Procedure context and evidence limitations', {key: packet.get(key) for key in ('procedure_context', 'limitations', 'neighbor_coverage', 'media_timeline')})}
{_details('Complete saved evidence packet', packet)}<p>{links}</p>
{_details('Complete annotation record', record) if record else ''}</article>'''


def write_annotation_report(output_dir) -> Path:
    """Render accepted and prepared per-target evidence without draft dependencies."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    warnings = []
    summary = _mapping(_read(root / "summary.json", warnings))
    document = _mapping(_read(root / "annotations.json", warnings))
    records = {row["target_frame_id"]: row for row in _rows(document.get("annotations"))
               if isinstance(row.get("target_frame_id"), str)}
    packets = {}
    for path in sorted((root / "evidence").glob("*.json")):
        packet = _read(path, warnings)
        if isinstance(packet, dict) and isinstance(packet.get("target_frame_id"), str):
            packets[packet["target_frame_id"]] = packet
    # The separate evidence artifact is authoritative for displayed images. A
    # damaged packet must never cause model-supplied paths to become image URLs.
    ids = list(packets) + [target_id for target_id in records if target_id not in packets]
    cards = ''.join(_card(packets.get(target_id, {}), records.get(target_id), root) for target_id in ids)
    links = ' · '.join(_link(root / name, label) for name, label in
                        (("run.json", "Configuration and pinned inputs"), ("summary.json", "Run summary"),
                         ("annotations.json", "All saved annotations")) if (root / name).is_file())
    warning_html = _details("Artifact warnings", warnings) if warnings else ''
    error_html = _details("Latest annotation error", summary["error"]) if summary.get("error") else ''
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src file:; style-src 'unsafe-inline'">
<title>Independent MedGemma annotations</title><style>
*{{box-sizing:border-box}}body{{font:16px/1.55 system-ui,sans-serif;color:#173044;background:#f1f5f7;margin:0}}main{{max-width:1300px;margin:auto;padding:30px 24px}}
h1,h2,h3{{line-height:1.25}}.frame,.overview{{padding:24px;background:white;border:1px solid #d9e1e8;border-radius:10px;margin:24px 0}}.muted{{color:#526477}}.badge{{display:inline-block;background:#e7f2ed;padding:3px 8px;border-radius:5px}}a{{color:#145c96;overflow-wrap:anywhere}}.filmstrip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-top:16px}}figure{{margin:0;border:1px solid #d9e1e8;padding:10px}}img{{display:block;width:100%;height:auto}}figcaption{{font-size:13px;overflow-wrap:anywhere}}.claim{{border-top:1px solid #d9e1e8;padding:12px 0}}.uncertainty{{color:#765214}}details{{margin:16px 0}}summary{{cursor:pointer}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f7;padding:12px;font-size:12px}}section{{margin-top:24px}}
</style></head><body><main><h1>Independent MedGemma annotations</h1>
<p>Surgical claims authored from each target image, detail crops, and supplied context. Each claim links to its image evidence.</p>
<section class="overview"><b>{_escape(summary.get('annotated_frame_count', len(records)))} / {_escape(summary.get('selected_frame_count', len(ids)))} annotations saved</b>
<p>Status: {_escape(summary.get('status', 'summary unavailable'))}</p>
<p>Model drafts · Human review required · Not eligible for training</p><p>{links}</p>{error_html}{warning_html}</section>
{cards or '<p class="muted">No saved evidence packets or accepted annotations are available yet.</p>'}</main></body></html>'''
    path = root / "report.html"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                prefix=".medgemma-annotation-report-", suffix=".html", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(page)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
