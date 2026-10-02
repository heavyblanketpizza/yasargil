"""A self-contained HTML viewer for one replay run.

``report.html`` sits in the run directory next to downscaled frame thumbnails
under ``report/frames``. It shows the frame with perception overlays, timeline
lanes built from the event log, what was spoken (and, optionally, what the gate
suppressed), every question with its verified and rejected claims and agent
steps, and the full event log. All run data is embedded as JSON and rendered with
``textContent``; nothing is fetched.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from . import LiveError
from .frames import FrameSource, render_jpeg


def _jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_report(run_dir, thumbnail_side=640):
    run_dir = Path(run_dir)
    manifest = _json(run_dir / "run.json")
    if manifest is None:
        raise LiveError(f"Not a replay run directory (no run.json): {run_dir}")
    source = FrameSource(manifest["frames"]["directory"], manifest["case_id"], manifest["frames"].get("fps", 1.0))
    thumbs = run_dir / "report" / "frames"
    thumbs.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame in source.frames:
        if not manifest["frames"]["first"] <= frame.index <= manifest["frames"]["last"]:
            continue
        relative = f"report/frames/{frame.index:06d}.jpg"
        target = run_dir / relative
        if not target.exists():
            target.write_bytes(render_jpeg(frame.path, thumbnail_side, quality=80))
        frames.append({"index": frame.index, "t_ms": frame.t_ms, "thumbnail": relative})
    agent = {}
    for result_path in sorted((run_dir / "agent").glob("*/result.json")):
        result = _json(result_path)
        agent[result["question_id"]] = {key: result.get(key) for key in (
            "status", "final", "verdict", "steps", "elapsed_ms", "tool_calls", "images", "evictions", "error")}
        for step in agent[result["question_id"]]["steps"] or []:
            step.pop("reasoning", None)
    data = {
        "run": {key: manifest.get(key) for key in ("case_id", "started_at", "frames", "perception", "procedure",
                                                   "session", "agent", "specialist")},
        "summary": _json(run_dir / "summary.json", {}), "frames": frames,
        "observations": _jsonl(run_dir / "observations.jsonl"), "events": _jsonl(run_dir / "events.jsonl"),
        "utterances": _jsonl(run_dir / "utterances.jsonl"), "answers": _jsonl(run_dir / "answers.jsonl"),
        "timing": _jsonl(run_dir / "timing.jsonl"), "agent": agent,
    }
    payload = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/").replace("<!--", "<\\!--")
    title = html.escape(f"Replay {manifest['case_id']}")
    path = run_dir / "report.html"
    path.write_text(TEMPLATE.replace("__TITLE__", title).replace("__DATA__", payload), encoding="utf-8")
    return path


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
/* Minimal neobrutalist skin: black-and-white chrome with hard borders and offset shadows.
   Color is reserved for data (lanes, priorities, overlays) so the surgical video stays the focus. */
:root { color-scheme:light; --bg:#f2f2f0; --panel:#fff; --ink:#111; --muted:#666; --line:#111; --rule:#dcdcdc; --wash:#f2f2f0; --accent:#2f6fb3;
  --critical:#e5484d; --high:#f08a24; --medium:#d4a514; --low:#9a9a9a; --ok:#1f7a4d; --bad:#d23b3b;
  --lane:#fff; --inst:#5b8def; --anat:#e0b03a; --near:#e5484d; --step:#9b8ae0;
  --display:ui-sans-serif, system-ui, -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif; --mono:ui-monospace, Menlo, monospace; }
@media (prefers-color-scheme: dark) { :root { color-scheme:dark; --bg:#0c0c0c; --panel:#161616; --ink:#f2f2f2; --muted:#9a9a9a; --line:#f2f2f2;
  --rule:#333; --wash:#222; --lane:#161616; --accent:#7fb0ea; --low:#777; --ok:#5fc28f; --bad:#f47066; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 -apple-system, "Segoe UI", sans-serif; }
::selection { background:var(--ink); color:var(--panel); }
header { background:var(--panel); border-bottom:3px solid var(--ink); padding:18px 20px 16px; }
h1 { font:900 24px/1.15 var(--display); letter-spacing:-.5px; text-transform:uppercase; margin:0 0 12px; }
h2 { font:900 14px/1.4 var(--display); letter-spacing:.2px; text-transform:uppercase; margin:0 0 12px; }
.chips { display:flex; flex-wrap:wrap; gap:8px; }
.chip { background:var(--panel); border:2px solid var(--ink); border-radius:0; padding:1px 8px; color:var(--muted); font:700 11px/1.6 var(--mono);
  text-transform:uppercase; letter-spacing:.3px; }
.chip b { color:var(--ink); font:900 13px/1 var(--display); letter-spacing:0; margin-left:2px; }
main { display:grid; grid-template-columns:minmax(0, 2fr) minmax(260px, 1fr); gap:22px; padding:22px 20px 8px; }
main > * { min-width:0; }
@media (max-width: 900px) { main { grid-template-columns:minmax(0, 1fr); } }
section, aside { background:var(--panel); border:3px solid var(--ink); border-radius:0; padding:16px; box-shadow:5px 5px 0 var(--ink); }
.stage { position:relative; background:#000; border:3px solid var(--ink); border-radius:0; overflow:hidden; }
.stage img { display:block; width:100%; height:auto; }
.stage canvas { position:absolute; inset:0; width:100%; height:100%; }
.controls { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin:14px 0 12px; }
.controls input[type=range] { flex:1; min-width:120px; }
button { background:var(--panel); color:var(--ink); border:2px solid var(--ink); border-radius:0; padding:3px 11px; cursor:pointer;
  font:700 13px/1.5 var(--mono); box-shadow:2px 2px 0 var(--ink); transition:transform .08s, box-shadow .08s; }
button:hover { transform:translate(-2px, -2px); box-shadow:4px 4px 0 var(--ink); }
button:active { transform:translate(2px, 2px); box-shadow:0 0 0 var(--ink); }
button:focus-visible, select:focus-visible, input:focus-visible { outline:3px solid var(--ink); outline-offset:2px; }
#play { min-width:76px; background:var(--ink); color:var(--panel); text-transform:uppercase; }
input[type=range] { -webkit-appearance:none; appearance:none; height:24px; background:transparent; cursor:pointer; }
input[type=range]::-webkit-slider-runnable-track { height:8px; background:var(--panel); border:2px solid var(--ink); }
input[type=range]::-webkit-slider-thumb { -webkit-appearance:none; width:12px; height:20px; margin-top:-8px; background:var(--ink); border:2px solid var(--ink); }
input[type=range]::-moz-range-track { height:4px; background:var(--panel); border:2px solid var(--ink); }
input[type=range]::-moz-range-thumb { width:8px; height:16px; background:var(--ink); border:2px solid var(--ink); border-radius:0; }
input[type=checkbox] { accent-color:var(--ink); }
#clock { font-weight:700; white-space:nowrap; }
.mono { font-family:var(--mono); font-size:12px; }
.lanes { position:relative; }
.lane { display:flex; align-items:center; gap:8px; margin:4px 0; }
.lane .name { width:120px; flex:none; color:var(--ink); font:700 11px/1.2 var(--mono); text-transform:uppercase; text-align:right; }
.lane .track { position:relative; flex:1; height:20px; background:var(--lane); border:2px solid var(--ink); border-radius:0; cursor:pointer; }
.bar { position:absolute; top:2px; bottom:2px; border-radius:0; overflow:hidden; white-space:nowrap; color:#111; font:700 10px/12px var(--mono); padding-left:4px; }
.bar.near { top:8px; }
.mark { position:absolute; top:0; bottom:0; width:3px; border-radius:0; }
.cursor { position:absolute; top:-4px; bottom:-4px; width:3px; background:var(--ink); box-shadow:0 0 0 1.5px var(--panel); pointer-events:none; z-index:2; }
.feed ol { list-style:none; margin:0; padding:0 4px 4px 0; max-height:520px; overflow:auto; scrollbar-color:var(--ink) transparent; }
.feed li { background:var(--panel); border:2px solid var(--ink); border-left:6px solid var(--line); padding:6px 10px; margin:0 0 8px; }
.feed li.suppressed { opacity:.55; border-style:dashed; }
.prio-critical { border-left-color:var(--critical) !important; } .prio-high { border-left-color:var(--high) !important; }
.prio-medium { border-left-color:var(--medium) !important; } .prio-low { border-left-color:var(--low) !important; }
.meta { color:var(--muted); font-size:12px; }
label.meta { display:inline-block; margin-bottom:12px; }
.cite { font:700 11px/1.5 var(--mono); padding:0 5px; margin:0 4px 2px 0; border:1.5px solid var(--ink); border-radius:0; background:var(--panel); box-shadow:none; }
.cite:hover { background:var(--ink); color:var(--panel); transform:none; box-shadow:none; }
.cards { display:grid; grid-template-columns:repeat(auto-fill, minmax(min(320px, 100%), 1fr)); gap:16px; }
.card { background:var(--panel); border:2px solid var(--ink); border-radius:0; padding:12px; box-shadow:3px 3px 0 var(--ink); min-width:0; overflow-wrap:anywhere; }
.badge { display:inline-block; border-radius:0; padding:0 6px; font:800 11px/1.6 var(--mono); text-transform:uppercase; border:2px solid currentColor; }
.badge.ok { background:var(--ink); color:var(--panel); border-color:var(--ink); }
.ok { color:var(--ok); } .bad { color:var(--bad); }
details { margin-top:8px; } summary { cursor:pointer; color:var(--ink); font:700 12px/1.6 var(--mono); text-transform:uppercase; }
summary:hover { text-decoration:underline; text-underline-offset:3px; }
.wide section { overflow-x:auto; }
table { border-collapse:collapse; width:100%; border:2px solid var(--ink); }
td, th { text-align:left; padding:4px 8px; border-bottom:1px solid var(--rule); vertical-align:top; }
th { background:var(--ink); color:var(--panel); font:700 11px/1.6 var(--mono); text-transform:uppercase; letter-spacing:.4px; }
tr.past { } tr.future { opacity:.35; } tr.hit td { background:var(--wash); } tr.hit td:first-child { box-shadow:inset 4px 0 0 var(--ink); }
tbody tr { cursor:pointer; } tbody tr:hover td { background:var(--wash); }
select { font:700 12px/1.5 var(--mono); color:var(--ink); background:var(--panel); border:2px solid var(--ink); border-radius:0; padding:3px 6px; margin:0 0 12px; }
.wide { padding:6px 20px 28px; display:grid; gap:22px; }
.legend { margin:4px 0 10px; }
.legend span { margin-right:12px; font:700 11px/1.6 var(--mono); text-transform:uppercase; color:var(--ink); white-space:nowrap; }
.legend i { display:inline-block; width:11px; height:11px; border:1.5px solid var(--ink); border-radius:0; margin-right:5px; vertical-align:-1px; }
@media (max-width: 600px) { header, main, .wide { padding-left:16px; padding-right:16px; } .lane .name { width:84px; font-size:10px; } }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="chips" id="chips"></div>
</header>
<main>
  <section>
    <div class="stage"><img id="frame" alt="Current replay frame"><canvas id="overlay"></canvas></div>
    <div class="controls">
      <button id="prev" title="Previous frame (Left)">&#9664;</button>
      <button id="play" title="Play or pause (Space)">Play</button>
      <button id="next" title="Next frame (Right)">&#9654;</button>
      <input type="range" id="scrub" min="0" value="0" aria-label="Frame">
      <span id="clock" class="mono"></span>
    </div>
    <div class="legend"><span><i style="background:var(--inst)"></i>instrument in view</span><span><i style="background:var(--anat)"></i>structure in view</span><span><i style="background:var(--near)"></i>tip near structure</span><span><i style="background:var(--step)"></i>procedure step</span></div>
    <div class="lanes" id="lanes"></div>
  </section>
  <aside class="feed">
    <h2>Spoken so far</h2>
    <label class="meta"><input type="checkbox" id="suppressed"> show what the gate suppressed</label>
    <ol id="spoken"></ol>
  </aside>
</main>
<div class="wide">
  <section><h2>Questions</h2><div class="cards" id="answers"></div></section>
  <section>
    <h2>Event log</h2>
    <p class="meta">Faded rows are in the future of the current frame. Click a row to jump to it.</p>
    <select id="filter" aria-label="Event type"><option value="">all types</option></select>
    <table><thead><tr><th>ID</th><th>t</th><th>type</th><th>subject</th><th>conf</th><th>frames</th><th>cites</th></tr></thead><tbody id="events"></tbody></table>
  </section>
</div>
<script id="run-data" type="application/json">__DATA__</script>
<script>
(() => {
const D = JSON.parse(document.getElementById("run-data").textContent);
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined && text !== null) n.textContent = String(text); return n; };
const sec = (ms) => (ms / 1000).toFixed(1) + "s";
const frames = D.frames, byIndex = new Map(D.observations.map((o) => [o.frame_index, o]));
const events = D.events, eventById = new Map(events.map((e) => [e.event_id, e]));
const start = frames.length ? frames[0].t_ms : 0, end = frames.length ? frames[frames.length - 1].t_ms + 1000 : 1000;
const pct = (t) => (100 * (t - start) / Math.max(1, end - start)) + "%";
let pos = 0, timer = null;

$("title").textContent = "Live guidance replay — " + D.run.case_id;
const S = D.summary || {}, chips = $("chips");
const chip = (label, value) => { const c = el("span", "chip", label + " "); c.appendChild(el("b", "", value)); chips.appendChild(c); };
chip("perception", (D.run.perception || {}).kind || "?");
chip("frames", frames.length);
chip("events", events.length);
if (S.utterances) { chip("spoken", S.utterances.spoken); chip("suppressed", S.utterances.suppressed); }
if (S.questions) { chip("answered", S.questions.answered + "/" + S.questions.asked); chip("cited", S.questions.supported + "/" + S.questions.asked); }
if (S.timing_ms && S.timing_ms.total_ms) chip("frame p95", S.timing_ms.total_ms.p95 + " ms");
chip("missed deadlines", S.missed_deadlines ?? 0);

const scrub = $("scrub"); scrub.max = Math.max(0, frames.length - 1);
function seekTime(t) { let best = 0; frames.forEach((f, i) => { if (f.t_ms <= t) best = i; }); show(best); }

const lanes = $("lanes"), cursors = [];
function lane(name) {
  const row = el("div", "lane"); row.appendChild(el("span", "name", name));
  const track = el("div", "track"); row.appendChild(track); lanes.appendChild(row);
  track.addEventListener("click", (ev) => { const r = track.getBoundingClientRect(); seekTime(start + (ev.clientX - r.left) / r.width * (end - start)); });
  const cur = el("div", "cursor"); track.appendChild(cur); cursors.push(cur); return track;
}
function bar(track, from, to, color, tip, label) { const b = el("div", color === "var(--near)" ? "bar near" : "bar", label); b.style.left = pct(from); b.style.width = `calc(${pct(to)} - ${pct(from)})`; b.style.background = color; b.title = tip; track.appendChild(b); }
const subjects = [...new Set(events.filter((e) => ["instrument_entered", "anatomy_visible"].includes(e.type)).map((e) => e.subject))];
for (const subject of subjects) {
  const track = lane(subject), anatomy = events.some((e) => e.subject === subject && e.type === "anatomy_visible");
  let open = null, near = null;
  for (const e of events) {
    if (e.subject !== subject) continue;
    if (e.type === "instrument_entered" || e.type === "anatomy_visible") open = e;
    if ((e.type === "instrument_left" || e.type === "anatomy_hidden") && open) { bar(track, open.onset_ms, e.onset_ms, anatomy ? "var(--anat)" : "var(--inst)", `${subject} ${sec(open.onset_ms)}–${sec(e.onset_ms)}`); open = null; if (near) { bar(track, near.onset_ms, e.onset_ms, "var(--near)", "tip near " + near.object); near = null; } }
    if (e.type === "tip_near_structure") near = e;
    if (e.type === "tip_cleared_structure" && near) { bar(track, near.onset_ms, e.t_ms, "var(--near)", "tip near " + near.object); near = null; }
  }
  if (open) bar(track, open.onset_ms, end, anatomy ? "var(--anat)" : "var(--inst)", subject + " from " + sec(open.onset_ms));
  if (near) bar(track, near.onset_ms, end, "var(--near)", "tip near " + near.object);
}
const steps = events.filter((e) => e.type === "step_changed");
if (steps.length) { const track = lane("step"); steps.forEach((e, i) => bar(track, e.onset_ms, i + 1 < steps.length ? steps[i + 1].onset_ms : end, i % 2 ? "var(--step)" : "color-mix(in srgb, var(--step) 75%, #000)", e.subject, e.subject)); }
const said = lane("spoken");
for (const u of D.utterances.filter((u) => u.spoken)) { const m = el("div", "mark"); m.style.left = pct(u.at_ms); m.style.background = `var(--${u.question_id ? "accent" : u.priority})`; m.title = u.text; said.appendChild(m); }

const canvas = $("overlay"), img = $("frame");
function draw(o) {
  const w = canvas.width = img.clientWidth * devicePixelRatio, h = canvas.height = img.clientHeight * devicePixelRatio;
  const g = canvas.getContext("2d"); g.clearRect(0, 0, w, h); if (!o) return;
  g.lineWidth = 2 * devicePixelRatio; g.font = `${12 * devicePixelRatio}px sans-serif`;
  for (const d of o.detections) {
    const color = d.kind === "anatomy" ? "#f2c14e" : "#6ea8ff"; g.strokeStyle = g.fillStyle = color;
    if (d.box) { const [x1, y1, x2, y2] = d.box; g.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h); g.fillText(`${d.label} ${d.confidence.toFixed(2)}`, x1 * w + 4, y1 * h + 14 * devicePixelRatio); }
    if (d.tip) { const [x, y] = d.tip; g.beginPath(); g.arc(x * w, y * h, 6 * devicePixelRatio, 0, 7); g.stroke(); if (!d.box) g.fillText(`${d.label} ${d.confidence.toFixed(2)}`, x * w + 9, y * h - 6); }
  }
}
img.addEventListener("load", () => draw(byIndex.get(frames[pos].index)));
addEventListener("resize", () => frames.length && draw(byIndex.get(frames[pos].index)));

function renderFeed(now) {
  const list = $("spoken"); list.textContent = "";
  const showAll = $("suppressed").checked;
  for (const u of D.utterances.filter((u) => u.at_ms <= now && (u.spoken || showAll)).reverse()) {
    const li = el("li", (u.spoken ? "" : "suppressed ") + "prio-" + (u.question_id ? "low" : u.priority));
    li.appendChild(el("div", "", u.text));
    const meta = el("div", "meta", `${sec(u.at_ms)} · ${u.source} · ${u.priority}${u.spoken ? "" : " · suppressed: " + u.reason} `);
    for (const id of u.cites || []) { const b = el("button", "cite", id); b.addEventListener("click", () => jumpEvent(id)); meta.appendChild(b); }
    li.appendChild(meta); list.appendChild(li);
  }
}
$("suppressed").addEventListener("change", () => renderFeed(frames[pos].t_ms));

const tbody = $("events"), rows = new Map();
const filter = $("filter");
[...new Set(events.map((e) => e.type))].forEach((t) => filter.appendChild(el("option", "", t)));
for (const e of events) {
  const tr = el("tr"); tr.dataset.type = e.type;
  [e.event_id, sec(e.t_ms), e.type, e.object ? `${e.subject} → ${e.object}` : e.subject, e.confidence.toFixed(2),
   (e.evidence_frames || []).join(", "), (e.cites || []).join(", ")].forEach((v, i) => tr.appendChild(el("td", i === 0 || i > 4 ? "mono" : "", v)));
  tr.addEventListener("click", () => seekTime(e.t_ms));
  tbody.appendChild(tr); rows.set(e.event_id, tr);
}
filter.addEventListener("change", () => { for (const tr of rows.values()) tr.hidden = filter.value && tr.dataset.type !== filter.value; });
function jumpEvent(id) {
  const e = eventById.get(id); if (!e) return;
  seekTime(e.t_ms); for (const tr of rows.values()) tr.classList.remove("hit");
  const tr = rows.get(id); tr.classList.add("hit"); tr.hidden = false; tr.scrollIntoView({ block: "center", behavior: "smooth" });
}

const cards = $("answers");
if (!D.answers.length) cards.appendChild(el("p", "meta", "No questions were asked in this run."));
for (const a of D.answers) {
  const card = el("div", "card"), r = D.agent[a.question_id] || {};
  const head = el("div", "meta", `${a.question_id} · asked ${sec(a.asked_at_ms)} (frame ${a.frame_index}) · `);
  const jump = el("button", "cite", "jump"); jump.addEventListener("click", () => seekTime(a.asked_at_ms)); head.appendChild(jump);
  card.appendChild(head); card.appendChild(el("p", "", a.text));
  card.appendChild(el("p", "", "“" + (a.spoken_text || "(no answer)") + "”"));
  const line = el("div", "meta");
  line.appendChild(el("span", "badge " + (a.status === "answered" ? "ok" : "bad"), a.status));
  line.appendChild(document.createTextNode(" "));
  line.appendChild(el("span", "badge " + (a.supported ? "ok" : "bad"), a.supported ? "fully cited" : "not fully cited"));
  line.appendChild(document.createTextNode(` ${a.elapsed_ms ?? "?"} ms · ${a.tool_calls ?? 0} tool calls · ${a.images ?? 0} images · value ${JSON.stringify(a.value)}`));
  card.appendChild(line);
  const verdict = r.verdict || {};
  if ((verdict.verified || []).length || (verdict.rejected || []).length) {
    const det = el("details"); det.appendChild(el("summary", "", "claims"));
    for (const c of verdict.verified || []) { const p = el("div", "ok", "✓ " + c.text + " "); (c.event_ids || []).forEach((id) => { const b = el("button", "cite", id); b.addEventListener("click", () => jumpEvent(id)); p.appendChild(b); }); det.appendChild(p); }
    for (const c of verdict.rejected || []) det.appendChild(el("div", "bad", "✗ " + c.text + " — " + (c.reasons || []).join("; ")));
    card.appendChild(det);
  }
  if ((r.steps || []).length) {
    const det = el("details"); det.appendChild(el("summary", "", `${r.steps.length} agent steps`));
    for (const s of r.steps) {
      const acts = (s.actions || []).map((x) => (x.tool || "(none)") + (x.ok ? "" : " ✗ " + (x.error || ""))).join(", ");
      det.appendChild(el("div", "mono", `step ${s.step}: ${acts} · ${s.elapsed_ms} ms · prompt ${s.prompt_tokens ?? "?"} (cached ${s.cached_tokens ?? "?"})`));
    }
    card.appendChild(det);
  }
  if (a.error) card.appendChild(el("div", "bad", a.error));
  cards.appendChild(card);
}

function show(i) {
  if (!frames.length) return;
  pos = Math.max(0, Math.min(frames.length - 1, i)); const f = frames[pos];
  scrub.value = pos; img.src = f.thumbnail; $("clock").textContent = `frame ${f.index} · ${sec(f.t_ms)}`;
  if (img.clientWidth) draw(byIndex.get(f.index));
  for (const c of cursors) c.style.left = pct(f.t_ms);
  for (const [id, tr] of rows) tr.className = (eventById.get(id).t_ms <= f.t_ms ? "past" : "future") + (tr.classList.contains("hit") ? " hit" : "");
  renderFeed(f.t_ms);
}
scrub.addEventListener("input", () => show(+scrub.value));
$("prev").addEventListener("click", () => show(pos - 1));
$("next").addEventListener("click", () => show(pos + 1));
function toggle() { if (timer) { clearInterval(timer); timer = null; $("play").textContent = "Play"; } else { timer = setInterval(() => { if (pos >= frames.length - 1) toggle(); else show(pos + 1); }, 250); $("play").textContent = "Pause"; } }
$("play").addEventListener("click", toggle);
addEventListener("keydown", (ev) => { if (ev.target.tagName === "INPUT" && ev.target.type !== "range") return; if (ev.key === "ArrowLeft") show(pos - 1); else if (ev.key === "ArrowRight") show(pos + 1); else if (ev.key === " ") { ev.preventDefault(); toggle(); } });
show(0);
})();
</script>
</body>
</html>
"""
