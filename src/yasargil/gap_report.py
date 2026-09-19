"""Local, inspectable reporting for the isolated frame-gap experiment."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
import tempfile


def _escape(value):
    return html.escape(str(value), quote=True)


def _timestamp(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "Unavailable"
    seconds = value / 1000
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def _path(value, root):
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _link(path, label):
    if path is None:
        return _escape(label)
    return f'<a href="{_escape(path.absolute().as_uri())}">{_escape(label)}</a>'


def _read(path, warnings):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        warnings.append(f"Could not read {path.name}: {error}")
        return None


def _json_details(label, value):
    return (f'<details><summary>{_escape(label)}</summary><pre>'
            f'{_escape(json.dumps(value, indent=2, ensure_ascii=False))}</pre></details>')


def _label(value):
    return str(value).replace("_", " ").replace("-", " ").capitalize()


def _condition_label(value):
    return {"drop_50": "Drop least-important 50%", "drop_70": "Drop least-important 70%",
            "drop_90": "Drop least-important 90%",
            "bottom_50": "Bottom 50% supplied", "bottom_70": "Bottom 70% supplied",
            "bottom_90": "Bottom 90% supplied", "control_all": "All-candidate control"}.get(value, _label(value))


def _fraction(value):
    if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:.1%}"
    return "Pending"


def _condition_fraction(condition, kind):
    value = condition.get(f"actual_{kind}_fraction")
    if value is None:
        percent = condition.get(f"actual_{kind}_percent")
        if isinstance(percent, (int, float)) and not isinstance(percent, bool):
            value = percent / 100
    return _fraction(value)


def _scalar_metrics(value, prefix=""):
    """Show scalar measures together; keep detailed matching receipts expandable."""
    result = {}
    if not isinstance(value, dict):
        return result
    for key, item in value.items():
        if not prefix and key in {"schema_version", "condition_id", "reference_sha256", "scoring_basis",
                                  "clinical_ground_truth", "clinical_validation", "training_eligible",
                                  "temporal_proxy_review_required", "max_request_span_ms", "unmatched_evidence_policy",
                                  "completion_interpretation"}:
            continue
        name = f"{prefix} / {_label(key)}" if prefix else _label(key)
        if isinstance(item, dict):
            result.update(_scalar_metrics(item, name))
        elif item is None or isinstance(item, (str, int, float, bool)):
            result[name] = item
    return result


def _frame_table(frame_ids, by_id, root):
    rows = []
    for frame_id in frame_ids:
        frame = by_id.get(frame_id, {})
        source = _path(frame.get("source_path"), root)
        image = _path(frame.get("image_path"), root)
        identity = (f'<a href="#frame-{_escape(frame_id)}">{_escape(frame_id)}</a>'
                    if frame.get("rank") is not None else _escape(frame_id))
        thumbnail = (f'<a href="{_escape(image.absolute().as_uri())}"><img loading="lazy" '
                     f'src="{_escape(image.absolute().as_uri())}" '
                     f'alt="Candidate {_escape(frame_id)}"></a>') if image else "Image pending"
        rows.append(f'''<tr><td class="thumbnail">{thumbnail}</td>
<td>{identity}<br>
<span class="muted">Reference rank: {_escape(frame.get('rank', 'Outside reference pool' if frame else 'Pending'))}</span></td>
<td>{_timestamp(frame.get('timestamp_ms'))}<br><span class="muted">{_escape(frame.get('timestamp_basis', 'Pending'))}</span></td>
<td>{_link(source, source.name if source else 'Source pending')}<br>
<span class="muted">Frame index: {_escape(frame.get('frame_index', 'Pending'))}</span></td></tr>''')
    if not rows:
        return '<p class="muted">None.</p>'
    return ('<div class="scroll"><table><thead><tr><th>Still</th><th>Candidate</th>'
            '<th>Time and basis</th><th>Source</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div>')


def _rounds(artifact_dir, warnings, *, direct=False):
    base = artifact_dir if direct else artifact_dir / "rounds" if artifact_dir else None
    if base is None or not base.is_dir():
        return '<p class="muted">No review rounds saved yet.</p>'
    result = []
    for directory in sorted(base.glob("round-*")):
        if not directory.is_dir():
            continue
        output = _read(directory / "output.json", warnings)
        if not isinstance(output, dict):
            envelope = _read(directory / "result.json", warnings)
            output = envelope.get("output", {}) if isinstance(envelope, dict) else {}
        verification = _read(directory / "verification.json", warnings)
        receipts = _read(directory / "retrieval.json", warnings)
        links = []
        for filename, label in (("candidate-manifest.json", "Candidate provenance"),
                                ("request.json", "Complete model request"),
                                ("output.json", "Model answer"),
                                ("response.json", "Raw response"),
                                ("retrieval.json", "Retrieval receipts"),
                                ("retrieved-manifest.json", "Retrieved frame provenance"),
                                ("experiment-context.json", "Experiment session identity"),
                                ("verification.json", "Full-video verification")):
            path = directory / filename
            if path.is_file():
                links.append(_link(path, label))
        searches = output.get("searches")
        if isinstance(searches, list):
            summary = f"{len(searches)} evidence request(s)"
            requests = _json_details("Requested time intervals and questions", searches)
        elif isinstance(output.get("ranking"), list):
            summary = f"Reference ranking: {len(output['ranking'])} candidates"
            requests = ""
        else:
            summary = "Response pending or interrupted"
            requests = ""
        coverage = ""
        if isinstance(verification, dict):
            verified = verification.get("full_source_video_verified") is True
            coverage = (f" · Complete video {'verified' if verified else 'not yet verified'}"
                        f" ({_escape(verification.get('decoded_frames', '?'))} decoded frames)")
        receipt_detail = _json_details("Returned evidence", receipts) if receipts is not None else ""
        assessment = (f'<p>{_escape(output["scene_summary"])}</p><p class="muted">Context check: '
                      f'{_escape(output.get("context_check", "Unavailable"))}</p>') if output.get("scene_summary") else ""
        judgments = (_json_details("Keep/drop judgments and reasons", output["decisions"])
                     if isinstance(output.get("decisions"), dict) else "")
        result.append(f'''<article class="round"><h4>{_escape(directory.name)}</h4>
<p>{summary}{coverage}</p>{assessment}<p>{' · '.join(links)}</p>{requests}{receipt_detail}{judgments}</article>''')
    return "".join(result) or '<p class="muted">No review rounds saved yet.</p>'


def _condition(condition, by_id, root, warnings):
    condition_id = str(condition.get("id", condition.get("condition_id", "unidentified")))
    artifact_dir = _path(condition.get("artifact_dir"), root)
    if artifact_dir is None:
        artifact_dir = root / "conditions" / condition_id
    supplied = condition.get("supplied_frame_ids") or []
    withheld = condition.get("withheld_frame_ids") or []
    links = []
    for filename, label in (("metrics.json", "Complete metrics JSON"),
                            ("selection.html", "Final frame review"),
                            ("selection.json", "Selection and provenance"),
                            ("state.json", "Saved conversation state"),
                            ("condition.json", "Condition definition")):
        candidate = artifact_dir / filename
        if candidate.is_file():
            links.append(_link(candidate, label))
    metrics = condition.get("metrics")
    metrics_details = (_json_details("Detailed provisional metrics", metrics)
                       if metrics is not None else '<p class="muted">Metrics pending.</p>')
    missing = condition.get("missing_moment_ids")
    missing_text = (", ".join(map(str, missing)) or "None after checking supplied coverage") if isinstance(missing, list) else "Pending reference"
    selected = condition.get("selected_frame_ids")
    retained = (f'<details><summary>Latest retained candidates · {len(selected)}</summary>'
                + _frame_table(selected, by_id, root) + '</details>') if isinstance(selected, list) and condition.get("rounds_completed", 0) else ""
    return f'''<section id="condition-{_escape(condition_id)}" class="condition">
<div class="section-heading"><h2>{_escape(_condition_label(condition_id))}</h2>
<span class="badge">{_escape(condition.get('status', 'pending'))}</span></div>
<p><b>{_escape(condition.get('supplied_count', condition.get('actual_supplied_count', len(supplied))))}</b> supplied ({_condition_fraction(condition, 'supplied')}) ·
<b>{_escape(condition.get('withheld_count', condition.get('actual_withheld_count', len(withheld))))}</b> withheld ({_condition_fraction(condition, 'withheld')}) ·
{_escape(condition.get('rounds_completed', 0))} review round(s) complete</p>
<p><b>Reference moments initially missing:</b> {_escape(missing_text)}</p>
<p>{' · '.join(links)}</p>{metrics_details}
<details><summary>Supplied candidates · {len(supplied)}</summary>{_frame_table(supplied, by_id, root)}</details>
<details><summary>Withheld candidates · {len(withheld)}</summary>{_frame_table(withheld, by_id, root)}</details>
{retained}
<details><summary>Review rounds, requests, and retrieval receipts</summary>{_rounds(artifact_dir, warnings)}</details>
</section>'''


def _reference(reference, root):
    ranking = reference.get("ranking") or []
    cards = []
    for frame in ranking:
        frame_id = frame.get("frame_id", "unidentified")
        source = _path(frame.get("source_path"), root)
        image = _path(frame.get("image_path"), root)
        thumbnail = (f'<a href="{_escape(image.absolute().as_uri())}"><img loading="lazy" '
                     f'src="{_escape(image.absolute().as_uri())}" '
                     f'alt="Reference candidate {_escape(frame_id)}"></a>') if image else '<div class="empty-image">Image pending</div>'
        score = (f'<p><b>Surgical importance:</b> {_escape(frame["importance_score"])} / 100</p>'
                 if "importance_score" in frame else "")
        cards.append(f'''<article class="frame" id="frame-{_escape(frame_id)}">{thumbnail}<div class="frame-body">
<div class="rank">#{_escape(frame.get('rank', '?'))} · {_timestamp(frame.get('timestamp_ms'))}</div>
{score}<p>{_escape(frame.get('reason', 'Reason pending'))}</p>
<p><b>Moment:</b> {_escape(frame.get('moment_id', 'Pending'))}<br>
<b>Source:</b> {_link(source, source.name if source else 'Pending')}<br>
<b>Timestamp basis:</b> {_escape(frame.get('timestamp_basis', 'Pending'))}<br>
<b>Frame index:</b> {_escape(frame.get('frame_index', 'Pending'))}</p>
{_json_details('Frame identity and full provenance', frame)}</div></article>''')
    moments = reference.get("moments") or []
    moment_rows = []
    for moment in moments:
        moment_rows.append(f'''<tr><td>{_escape(moment.get('moment_id', 'Pending'))}</td>
<td>{_timestamp(moment.get('start_ms'))}–{_timestamp(moment.get('end_ms'))}</td>
<td>{_escape(', '.join(map(str, moment.get('frame_ids') or [])))}</td>
<td>{_escape(moment.get('description', moment.get('reason', '')))}</td></tr>''')
    moment_table = ('<details><summary>Reference moment groups</summary><div class="scroll"><table>'
                    '<thead><tr><th>Moment</th><th>Interval</th><th>Candidate IDs</th><th>Visible evidence</th></tr></thead>'
                    '<tbody>' + "".join(moment_rows) + '</tbody></table></div>'
                    + _json_details("Full moment definitions", moments) + '</details>') if moments else ""
    assessment = (f'<p>{_escape(reference["scene_summary"])}</p><p class="muted">Reference context check: '
                  f'{_escape(reference.get("context_check", "Unavailable"))}</p>') if reference.get("scene_summary") else ""
    return (assessment + moment_table + '<div class="grid">' + "".join(cards) + '</div>') if cards else '<p class="muted">No frozen reference ranking is available yet.</p>'


def _reference_quality(quality, quality_path):
    if not isinstance(quality, dict):
        return ""
    accepted = quality.get("accepted")
    status = "Passed" if accepted is True else "Failed — condition audits blocked" if accepted is False else "Pending"
    flags = quality.get("flags")
    flag_text = (f'<p><b>Flags:</b> {_escape(", ".join(map(str, flags)))}</p>'
                 if isinstance(flags, list) and flags else "")
    return (f'<aside class="notice"><b>Reference quality check: {_escape(status)}</b>'
            + flag_text + '<p>These order and score checks are heuristics. Passing them does not '
            'establish surgical importance or clinical accuracy.</p>'
            + _link(quality_path, "Ranking quality receipt")
            + _json_details("Ranking quality check details", quality) + '</aside>')


def write_experiment_report(output_dir) -> Path:
    """Regenerate an offline report without requiring a completed experiment."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    warnings = []
    summary = _read(root / "summary.json", warnings)
    if not isinstance(summary, dict):
        summary = {"status": "summary unavailable", "conditions": []}
    source = summary.get("source") or {}
    reference_meta = summary.get("reference") or {}
    reference_path = _path(reference_meta.get("path"), root) or root / "reference" / "reference.json"
    reference = _read(reference_path, warnings)
    if not isinstance(reference, dict):
        reference = {}
    quality_path = reference_path.parent / "ranking-quality.json"
    quality = _read(quality_path, warnings)
    if not isinstance(quality, dict):
        quality = reference_meta.get("quality", reference.get("quality_checks"))
    quality_html = _reference_quality(quality, quality_path if quality_path.is_file() else None)
    normalization_path = reference_path.parent / "ranking-normalization.json"
    if normalization_path.is_file():
        normalization = _read(normalization_path, warnings)
        if isinstance(normalization, dict):
            quality_html += ('<p>Reference order is derived by sorting Qwen\'s own importance scores from highest to lowest. '
                             'Equal scores retain their order in Qwen\'s original response. Scores, reasons, and moment IDs are unchanged. '
                             + _link(normalization_path, "Ranking normalization receipt") + ' · '
                             + _link(reference_path.parent / "ranking-quality-raw.json", "Original ranking checks") + '</p>')
    ranking = reference.get("ranking") or []
    canonical_source = _read(root / "source/source.json", warnings)
    if not isinstance(canonical_source, dict):
        canonical_source = _read(root / "source.json", warnings)
    all_frames = canonical_source.get("frames", []) if isinstance(canonical_source, dict) else []
    by_id = {frame["frame_id"]: frame for frame in [*all_frames, *ranking]
             if isinstance(frame, dict) and "frame_id" in frame}
    conditions = summary.get("conditions") or []
    surgical_protocol = (summary.get("schema_version") == "native-video-gap-experiment-v2"
                         or reference.get("schema_version") == "gap-reference-v2"
                         or any(str(c.get("id", c.get("condition_id", ""))).startswith("drop_")
                                for c in conditions))
    metric_sets = [_scalar_metrics(condition.get("metrics")) for condition in conditions]
    metric_names = list(dict.fromkeys(key for metrics in metric_sets for key in metrics))
    metrics_rows = []
    for name in metric_names:
        cells = []
        for metrics in metric_sets:
            value = metrics.get(name, "Pending")
            text = "Not applicable" if value is None else str(value)
            cells.append(f"<td>{_escape(text)}</td>")
        metrics_rows.append(f"<tr><th scope=\"row\">{_escape(name)}</th>{''.join(cells)}</tr>")
    metrics_table = ('<div class="scroll"><table><thead><tr><th>Measure</th>'
                     + "".join(f'<th>{_escape(_condition_label(c.get("id", c.get("condition_id", "condition"))))}</th>' for c in conditions)
                     + '</tr></thead><tbody>' + "".join(metrics_rows) + '</tbody></table></div>') if metrics_rows else '<p class="muted">Scores will appear after condition reviews finish.</p>'
    condition_html = "".join(_condition(condition, by_id, root, warnings) for condition in conditions)
    if not conditions:
        if summary.get("schema_version") == "native-video-gap-experiment-v1":
            condition_html = '<p class="muted">Condition sets will appear after reference ranking: bottom 50%, bottom 70%, bottom 90%, and the all-candidate control.</p>'
        else:
            condition_html = '<p class="muted">Condition sets will appear after an accepted reference ranking: drop the least-important 50%, 70%, or 90%, plus the all-candidate control.</p>'
    source_path = _path(source.get("source_path"), root)
    config = summary.get("config") or {}
    reference_rounds = _rounds(reference_path.parent, warnings, direct=True)
    error_html = (_json_details("Latest experiment error", summary["error"])
                  if summary.get("error") else "")
    artifact_links = []
    for filename, label in (("source/source.json", "Complete source manifest"),
                            ("source.json", "Legacy source manifest"),
                            ("experiment.json", "Experiment protocol and pinned inputs"),
                            ("initial-selection.json", "Original candidate selection"),
                            ("native-timeline-verification.json", "Source timeline verification")):
        if (root / filename).is_file():
            artifact_links.append(_link(root / filename, label))
    warning_html = ('<aside class="notice"><b>Artifact warnings</b><ul>'
                    + "".join(f"<li>{_escape(value)}</li>" for value in warnings) + '</ul></aside>') if warnings else ""
    omission_note = ("<p>The least-important 50%, 70%, or 90% of candidate stills are removed; "
                     "the most-important 50%, 30%, or 10% remain, subject to count rounding. "
                     "Because omitted candidates were ranked lower, requesting every omitted moment "
                     "is not necessarily useful. Declaring completion with unretained reference moments "
                     "is recorded as an observation, not a failure; inspect the surgical evidence.</p>") if surgical_protocol else ""
    ranking_note = ("One descending surgical-importance order produces nested supplied top groups. "
                    "Ranking stills are presented in a deterministic shuffle; video and audit stills "
                    "remain chronological.") if surgical_protocol else "One importance order produces nested top groups."
    report = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src file:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Frame-gap experiment</title><style>
:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{font:15px/1.55 system-ui,sans-serif;margin:0;background:#f4f6f8;color:#18212c}}main{{max-width:1360px;margin:auto;padding:30px}}
h1,h2,h3,h4{{line-height:1.2}}h1{{font-size:32px;margin:0 0 12px}}h2{{font-size:22px}}h4{{margin:0 0 8px}}a{{color:#175b99;overflow-wrap:anywhere}}p{{margin:12px 0}}
.lead{{font-size:17px;max-width:1000px}}.muted{{color:#596674;font-size:13px}}.notice{{background:#fff1d6;border-left:4px solid #a76c11;padding:14px 18px;margin:20px 0;border-radius:5px}}
.condition,.panel{{background:white;border:1px solid #dbe2e8;border-radius:10px;padding:22px;margin:22px 0}}.section-heading{{display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap}}.section-heading h2{{margin:0}}
.badge{{display:inline-block;border-radius:20px;background:#e7eef5;padding:4px 12px;font-size:13px}}.scroll{{overflow:auto;margin:12px 0}}table{{border-collapse:collapse;width:100%;text-align:left;font-size:13px}}th,td{{border-bottom:1px solid #dce3e9;padding:12px;vertical-align:top}}thead th{{background:#f0f4f7;white-space:nowrap}}tbody th{{font-weight:500;min-width:180px}}
details{{border-top:1px solid #dce3e9;padding:13px 0;margin-top:9px}}summary{{cursor:pointer;font-weight:600}}pre{{font:12px/1.5 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;padding:12px;border-radius:5px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:18px;margin-top:18px}}.frame{{background:white;border:1px solid #dbe2e8;border-radius:9px;overflow:hidden;scroll-margin-top:20px}}.frame img{{width:100%;aspect-ratio:16/9;object-fit:contain;background:#111;display:block}}.frame-body{{padding:16px}}.rank{{font-size:18px;font-weight:700;color:#145f68}}.thumbnail{{width:180px;min-width:150px}}.thumbnail img{{width:150px;aspect-ratio:16/9;object-fit:contain;background:#111}}.round{{border-left:3px solid #a8bdcd;padding:12px 16px;margin-top:18px}}.empty-image{{padding:45px;background:#e7edf1;text-align:center}}
@media(max-width:650px){{main{{padding:18px}}.condition,.panel{{padding:16px}}h1{{font-size:27px}}}}
</style></head><body><main><h1>Can Qwen find missing important moments?</h1>
<p class="lead">Each test supplies the complete video and a different subset of candidate stills.
Only the stills are withheld. The reference ranking and each test use separate conversations.</p>
<p><span class="badge">{_escape(summary.get('status', 'pending'))}</span> · Created {_escape(summary.get('created_at', 'time not recorded'))}</p>
<p class="muted">Active stage: {_escape(summary.get('active_stage') or 'None')} · Updated {_escape(summary.get('updated_at', 'time not recorded'))}</p>
<p class="muted">Protocol: {_escape(summary.get('schema_version', 'Unavailable'))}</p>
<p><b>Source:</b> {_link(source_path, source.get('source_path', 'Pending'))}<br>
<b>Complete supplied video:</b> {_escape(source.get('expected_video_frames', 'Pending'))} frames · {_timestamp(source.get('duration_ms'))} playback duration<br>
<b>Timestamp basis:</b> {_escape(source.get('timestamp_basis', 'Pending'))}<br>
<b>Clock:</b> Supplied video playback; recorded repair time does not set these timestamps.</p>
<p>{_link(root / 'summary.json', 'Experiment summary JSON')} · {_link(reference_path, 'Reference ranking JSON')}</p>
<p>{' · '.join(artifact_links)}</p>{error_html}
<aside class="notice"><b>Provisional experiment, not a clinical score.</b> The reference is Qwen's ranking and its moment groups.
Automated scores use temporal matching as a proxy for equivalent evidence; human review is still needed.
Overly broad requests receive no detection credit. Fresh contexts separate conditions; follow-up rounds retain that condition's complete video context.</aside>
{warning_html}<section class="panel"><h2>Compare the conditions</h2>
<p class="muted">Compare the fraction of initially missing moments recovered, not only the number of requests.
Recall is a fraction from 0 to 1. No missing moments means recall is not applicable.
The all-candidate control can still discover gaps outside the reference.</p>
{omission_note}{metrics_table}{_json_details('Saved experiment configuration', config)}</section>
{condition_html}<section class="panel"><div class="section-heading"><h2>Reference ranking</h2>
<span class="badge">{_escape(reference_meta.get('status', 'pending'))}</span></div>
<p>{ranking_note} The reference includes the full original candidate pool, including candidates dropped in the earlier selection run.</p>
{quality_html}
<details><summary>Reference conversation and video verification</summary>{reference_rounds}</details>
{_reference(reference, root)}</section>
</main></body></html>'''
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                     prefix=".report-", suffix=".html", delete=False) as temporary:
        temporary.write(report)
        temporary_path = Path(temporary.name)
    destination = root / "report.html"
    temporary_path.replace(destination)
    return destination
