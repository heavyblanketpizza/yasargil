"use strict";

const $ = (id) => document.getElementById(id);
const state = { records: [], record: null, index: 0, detail: null, mode: "still", request: null, notes: {}, dragging: false, context: null, contextRequest: null, recordVersion: 0, editor: null, curationBusy: false, exportBusy: false };
const storagePrefix = "yasargil-human-review-v1:";
const reviewDrafts = new Map();
let toastTimer, contextLoadTimer, contextStartTimer;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}
function clear(id) { const node = typeof id === "string" ? $(id) : id; node.replaceChildren(); return node; }
function formatTime(ms, precise = false) {
  if (!Number.isFinite(ms)) return "Unavailable";
  const whole = Math.max(0, Math.floor(ms));
  const seconds = Math.floor(whole / 1000);
  const base = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  return precise ? `${base}.${String(whole % 1000).padStart(3, "0")}` : base;
}
function humanize(value) { return String(value ?? "Unavailable").replaceAll("_", " "); }
function display(value) {
  if (value === null || value === undefined || value === "") return "Not reported";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
function localURL(value) {
  if (typeof value !== "string") return null;
  try { const url = new URL(value, location.origin); return url.origin === location.origin && /^\/(media|api|preview)\//.test(url.pathname) ? url.href : null; }
  catch { return null; }
}
function toast(message) { $("toast").textContent = message; $("toast").hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500); }
function showError(message) { $("error").textContent = message; $("error").hidden = false; }
async function api(path, signal) {
  const response = await fetch(path, { signal, cache: "no-store" });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Unable to load data (${response.status}).`);
  return body;
}
async function openDatasetFolder() {
  const button = $("open-dataset-folder"); button.disabled = true;
  try {
    const response = await fetch("/api/open-dataset-folder", { method: "POST", body: "" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not open the dataset folder.");
  } catch (error) { showError(error.message); }
  finally { button.disabled = false; }
}
function noteKey() {
  return storagePrefix + (state.record?.review_identity || state.record?.id || "");
}
function loadNotes() {
  try { const saved = JSON.parse(localStorage.getItem(noteKey()) || "{}"); state.notes = saved && typeof saved === "object" && !Array.isArray(saved) ? saved : {}; }
  catch { state.notes = {}; }
  validateReviewRevisions();
}
function validateReviewRevisions() {
  for (const frame of state.record.frames) {
    const note = state.notes[frame.frame_id];
    if (note?.status === "reviewed" && (note.curation_updated_at || null) !== (frame.curation?.updated_at || null)) { note.status = "unreviewed"; note.needs_recheck = true; }
  }
}
function currentNote() { return state.notes[state.record?.frames[state.index]?.frame_id] || { status: "unreviewed", note: "" }; }
function reviewDraftKey() { return `${noteKey()}:${state.record?.frames[state.index]?.frame_id || ""}`; }
function startReviewDraft() {
  const key = reviewDraftKey();
  if (!reviewDrafts.has(key)) {
    const note = currentNote();
    reviewDrafts.set(key, { note: note.note || "", status: note.status, original_note: note.note || "", original_status: note.status,
      base_updated_at: note.updated_at || null, curation_updated_at: state.record.frames[state.index].curation?.updated_at || null, error: null });
  }
  return reviewDrafts.get(key);
}
function reviewDraftChanged(draft) { return draft && (draft.note !== draft.original_note || draft.status !== draft.original_status); }
function editReview() { if (!state.record) return; $("surgery-video").pause(); startReviewDraft(); renderReview(); $("review-note").focus(); }
function changeReview(status) {
  if (!state.record || ($("review-edit").hidden === false)) return;
  $("surgery-video").pause();
  const draft = startReviewDraft();
  if (status) draft.status = status; else draft.note = $("review-note").value;
  draft.error = null; draft.notice = null; renderReview(false);
}
function cancelReview() {
  reviewDrafts.delete(reviewDraftKey()); loadNotes(); renderReview(); updateReviewCount();
}
function persistNote() {
  const frame = state.record?.frames[state.index];
  if (!frame) return;
  const draft = startReviewDraft();
  try {
    const saved = JSON.parse(localStorage.getItem(noteKey()) || "{}");
    if (!saved || typeof saved !== "object" || Array.isArray(saved)) throw new Error("Invalid saved reviews");
    if ((saved[frame.frame_id]?.updated_at || null) !== draft.base_updated_at) {
      draft.error = "This review changed in another tab. Cancel to load the latest saved review before editing.";
      renderReview(false); return;
    }
    const note = { status: draft.status, note: draft.note, updated_at: new Date().toISOString(), frame_id: frame.frame_id,
      timestamp_ms: frame.timestamp_ms, curation_updated_at: frame.curation?.updated_at || null };
    const notes = { ...saved, [frame.frame_id]: note };
    localStorage.setItem(noteKey(), JSON.stringify(notes));
    state.notes = notes; validateReviewRevisions(); reviewDrafts.delete(reviewDraftKey());
    renderReview(); updateReviewCount(); toast("Human review saved.");
  } catch {
    draft.error = "Could not save in this browser. Your draft is still here; enable browser storage and try again.";
    renderReview(false);
  }
}
function updateReviewCount() {
  const ids = new Set(state.record?.frames.map((frame) => frame.frame_id) || []);
  $("stat-reviewed").textContent = Object.entries(state.notes).filter(([id, value]) => ids.has(id) && value.status === "reviewed").length;
}
function renderReview(replaceText = true) {
  const note = currentNote(), draft = reviewDrafts.get(reviewDraftKey());
  const curation = state.record?.frames[state.index]?.curation?.updated_at || null;
  if (draft && draft.curation_updated_at !== curation) {
    draft.status = "unreviewed"; draft.curation_updated_at = curation;
    draft.notice = "Annotations changed. Recheck your decision before saving.";
  }
  const saved = Boolean(note.updated_at), editing = !saved || Boolean(draft), shown = draft || note;
  if (replaceText) $("review-note").value = shown.note || "";
  document.querySelectorAll("[data-review-state]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.reviewState === shown.status)); button.disabled = !editing;
  });
  document.querySelector(".review-states").classList.toggle("is-saved", !editing);
  $("review-edit").hidden = editing;
  $("review-note-editor").hidden = !editing; $("review-note-saved").hidden = editing;
  $("review-saved-text").textContent = note.note || "No written note.";
  $("review-save-actions").hidden = !editing; $("review-cancel").hidden = !draft;
  $("review-save-label").textContent = saved ? "Save changes" : "Save review";
  const status = $("save-status");
  status.classList.toggle("review-save-error", Boolean(draft?.error));
  status.textContent = draft?.error || draft?.notice || (editing
    ? (reviewDraftChanged(draft) ? "Unsaved changes" : saved ? "Editing" : "")
    : note.needs_recheck ? "Annotations changed. Edit this review to reassess." : `Saved ${new Date(note.updated_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}`);
}

async function refreshRecords() {
  $("refresh").disabled = true; $("error").hidden = true;
  try {
    const result = await api("/api/records");
    state.records = result.records || [];
    const select = clear("record-select");
    for (const record of state.records) {
      let title = record.title || record.case_id || record.id;
      if (state.records.filter((row) => (row.title || row.case_id || row.id) === title).length > 1) title += ` · ${record.run_name || record.id}`;
      if (record.is_synthetic) title += " (synthetic fixture)";
      const option = element("option", "", title);
      option.value = record.id; select.append(option);
    }
    $("loading").hidden = true;
    $("empty").hidden = state.records.length > 0;
    $("workspace").hidden = !state.records.length;
    $("export").disabled = !state.records.length;
    if (!state.records.length) return;
    let remembered;
    try { remembered = localStorage.getItem("yasargil-active-record"); } catch { /* optional browser persistence */ }
    const selected = state.records.find((r) => r.id === state.record?.id) || state.records.find((r) => r.id === remembered) || state.records.find((r) => r.case_id === "S6A3" && r.qwen_annotation_count > 0) || state.records[0];
    select.value = selected.id;
    await loadRecord(selected.id, selected.id === state.record?.id ? state.index : null);
    $("workspace-status").textContent = result.warnings?.length ? `${result.warnings.length} unavailable or incomplete artifact(s). Available evidence is shown.` : `${state.records.length} record${state.records.length === 1 ? "" : "s"}`;
  } catch (error) { $("loading").hidden = true; showError(error.message); }
  finally { $("refresh").disabled = false; }
}
async function loadRecord(id, priorIndex = null) {
  const version = ++state.recordVersion;
  state.request?.abort(); state.request = null; state.detail = null; renderCuration(); stopContextVideo();
  $("frame-loading").hidden = false;
  try {
    const record = await api(`/api/records/${encodeURIComponent(id)}`);
    if (version !== state.recordVersion) return;
    state.record = record; state.detail = null; state.mode = "still";
    try { localStorage.setItem("yasargil-active-record", id); } catch { /* optional browser persistence */ }
    loadNotes();
    $("record-select").value = id;
    $("dataset-badge").textContent = record.metadata?.dataset_name || (record.case_id?.match(/^(S\dA\d|Clip\d)$/) ? "SOSPINE" : "SOURCE RECORD");
    const sourceType = record.timestamp_basis === "reconstructed_nominal" ? "Released image sequence" : "Original video";
    $("record-subtitle").textContent = `${sourceType} · ${formatTime(record.duration_ms)} playback · ${humanize(record.status || "available")}`;
    $("stat-frames").textContent = record.frame_count ?? record.frames.length;
    $("stat-selected").textContent = record.selected_count ?? record.frames.filter((f) => f.status === "selected").length;
    $("stat-dropped").textContent = record.dropped_count ?? record.frames.filter((f) => f.status === "dropped").length;
    $("timeline-basis").textContent = record.timestamp_basis === "reconstructed_nominal" ? "NOMINAL TIME" : "SOURCE TIME";
    $("timeline-basis").title = record.timeline_note || (record.timestamp_basis === "reconstructed_nominal" ? "Nominal playback time; original timing unverified." : "Source video timestamps");
    $("frame-total").textContent = `/ ${record.frames.length}`;
    $("frame-jump").max = record.frames.length;
    $("timeline").setAttribute("aria-valuemax", Math.max(0, record.frames.length - 1));
    setMode("still"); renderTimeline(); renderOutcomes(record.outcomes); updateReviewCount();
    const firstSelected = record.frames.findIndex((frame) => frame.status === "selected");
    if (!record.frames.length) throw new Error("This record has no source frames yet. Refresh when source preparation finishes.");
    await selectFrame(priorIndex ?? Math.max(0, firstSelected));
  } catch (error) { if (version === state.recordVersion) showError(error.message); }
  finally { if (version === state.recordVersion) $("frame-loading").hidden = true; }
}

function timelineX(ms) {
  const track = $("timeline-track");
  return track.offsetLeft + track.clientLeft + Math.max(0, Math.min(1, ms / Math.max(1, state.record.duration_ms))) * track.clientWidth;
}
function renderTimeline() {
  if (!state.record) return;
  const ticks = clear("timeline-ticks"), track = clear("timeline-track");
  const duration = Math.max(1, state.record.duration_ms), width = track.clientWidth;
  const marked = state.record.frames.map((frame, index) => ({ frame, index, x: frame.timestamp_ms / duration * width }))
    .filter(({ frame }) => frame.status === "selected" || frame.status === "dropped");
  marked.forEach(({ frame, index, x }) => {
    const marker = element("span", `timeline-frame ${frame.status}${frame.curation?.deleted ? " enhancement-removed" : ""}`);
    marker.style.left = `${Math.max(0, Math.min(width, x))}px`;
    marker.dataset.index = index; marker.dataset.frameId = frame.frame_id;
    marker.title = `Frame ${index + 1} · ${formatTime(frame.timestamp_ms)} · ${frame.status === "dropped" ? "Dropped by Qwen" : "Selected key frame"}${frame.curation?.deleted ? " · Enhancement deleted by human" : ""}`;
    track.append(marker);
  });
  const tickCount = Math.max(4, Math.min(32, Math.round($("timeline").clientWidth / 135)));
  for (let n = 0; n <= tickCount; n++) {
    const tick = element("div", `tick ${n === 0 ? "first" : n === tickCount ? "last" : ""}`);
    tick.style.left = `${timelineX(duration * n / tickCount)}px`;
    tick.append(element("span", "", formatTime(duration * n / tickCount))); ticks.append(tick);
  }
  updateTimelineSelection(); updateReviewCount();
}
function updateTimelineSelection() {
  const frame = state.record?.frames[state.index]; if (!frame) return;
  $("timeline").setAttribute("aria-valuenow", state.index);
  $("timeline").setAttribute("aria-valuetext", `Frame ${state.index + 1} of ${state.record.frames.length}, ${formatTime(frame.timestamp_ms)}, ${frame.status}`);
  document.querySelectorAll(".timeline-frame").forEach((marker) => marker.classList.toggle("active", Number(marker.dataset.index) === state.index));
}
function nearestFrame(ms) {
  const frames = state.record.frames;
  let lo = 0, hi = frames.length - 1;
  while (lo < hi) { const mid = Math.floor((lo + hi) / 2); if (frames[mid].timestamp_ms < ms) lo = mid + 1; else hi = mid; }
  if (lo > 0 && Math.abs(frames[lo - 1].timestamp_ms - ms) <= Math.abs(frames[lo].timestamp_ms - ms)) return lo - 1;
  return lo;
}
function pointerIndex(event) {
  if (event.target.dataset.index !== undefined && !state.dragging) return Number(event.target.dataset.index);
  // Pointer capture retargets dragging to the slider. Resolve overlapping circles
  // by their centers and leave the square corners available for source frames.
  let hitIndex = null, closestDistance = Infinity;
  for (const marker of document.querySelectorAll(".timeline-frame")) {
    const rect = marker.getBoundingClientRect();
    const radius = (rect.right - rect.left) / 2;
    const distance = Math.hypot(event.clientX - (rect.left + rect.right) / 2, event.clientY - (rect.top + rect.bottom) / 2);
    if (distance <= radius && distance < closestDistance) { hitIndex = Number(marker.dataset.index); closestDistance = distance; }
  }
  if (hitIndex !== null) return hitIndex;
  const track = $("timeline-track"), bounds = track.getBoundingClientRect();
  return nearestFrame(Math.max(0, Math.min(1, (event.clientX - bounds.left - track.clientLeft) / Math.max(1, track.clientWidth))) * state.record.duration_ms);
}
function moveFrame(delta, keyframes = false) {
  if (!state.record) return;
  let next = state.index + delta;
  if (keyframes) { while (next >= 0 && next < state.record.frames.length && state.record.frames[next].status !== "selected") next += delta; }
  if (next >= 0 && next < state.record.frames.length) selectFrame(next, { scroll: true });
}
async function selectFrame(index, { scroll = false } = {}) {
  if (!state.record) return;
  const frames = state.record.frames;
  index = Math.max(0, Math.min(frames.length - 1, Math.round(Number(index) || 0)));
  state.index = index; state.detail = null; renderCuration();
  const frame = frames[index];
  $("frame-number").textContent = `FRAME ${String(index + 1).padStart(3, "0")} / ${frames.length}`;
  $("frame-state").className = `status-chip ${frame.status}`;
  $("frame-state").textContent = { selected: "SELECTED KEY FRAME", dropped: "QWEN DROPPED", candidate: "AWAITING SELECTION", source: "SOURCE FRAME" }[frame.status] || "SOURCE FRAME";
  $("frame-state").removeAttribute("title");
  if (frame.status === "selected") {
    const backend = state.record.runs?.selection?.config?.embedding_backend;
    const encoder = backend === "dinov2" ? "DINOv2" : backend === "dinov3" ? "DINOv3" : null;
    $("frame-state").textContent += ` [${encoder ? `${encoder} + Qwen` : "framework unrecorded"}]`;
    const method = encoder ? `${encoder} candidate selection; Qwen full-video review.` : "The selection framework is not recorded for this run.";
    const retention = frame.coverage_override ? " Retained as a coverage anchor despite Qwen's drop recommendation."
      : frame.protected_temporal_anchor ? " Protected temporal anchor." : "";
    $("frame-state").title = method + retention;
  }
  $("image-time").textContent = formatTime(frame.timestamp_ms, true); $("data-time").textContent = formatTime(frame.timestamp_ms, true);
  $("frame-jump").value = index + 1;
  $("previous-frame").disabled = index === 0; $("next-frame").disabled = index === frames.length - 1;
  $("previous-keyframe").disabled = !frames.slice(0, index).some((f) => f.status === "selected");
  $("next-keyframe").disabled = !frames.slice(index + 1).some((f) => f.status === "selected");
  $("image-error").hidden = true;
  const imageURL = localURL(frame.image_url);
  if (imageURL) { $("frame-image").src = imageURL; $("full-image").href = imageURL; }
  else { $("frame-image").removeAttribute("src"); $("full-image").removeAttribute("href"); $("image-error").hidden = false; }
  $("frame-image").alt = `${state.record.case_id || state.record.title}: source frame ${index + 1} at ${formatTime(frame.timestamp_ms)}`;
  $("image-filename").textContent = frame.source_path?.split("/").pop() || frame.frame_id;
  renderContextWindow();
  if (state.mode === "video") {
    stopContextVideo(); $("video-loading").hidden = false;
    contextStartTimer = setTimeout(() => loadContextVideo(), 150);
  }
  updateTimelineSelection(); renderReview();
  if (scroll) { const x = timelineX(frame.timestamp_ms); const viewport = $("timeline-scroll"); if (x < viewport.scrollLeft + 20 || x > viewport.scrollLeft + viewport.clientWidth - 20) viewport.scrollLeft = x - viewport.clientWidth / 2; }
  state.request?.abort(); const controller = new AbortController(); state.request = controller;
  ["qwen-content", "medgemma-content", "selection-context", "supporting-evidence", "source-annotations", "provenance", "artifact-links", "raw-data"].forEach(clear);
  $("frame-loading").hidden = false;
  try {
    const detail = await api(`/api/records/${encodeURIComponent(state.record.id)}/frames/${encodeURIComponent(frame.frame_id)}`, controller.signal);
    if (state.request !== controller) return;
    state.detail = detail; frame.curation = detail.curation; validateReviewRevisions(); renderReview(); renderDetail(detail); updateReviewCount();
  } catch (error) { if (state.request === controller && error.name !== "AbortError") showError(`Frame data could not be loaded: ${error.message}`); }
  finally { if (state.request === controller) $("frame-loading").hidden = true; }
}

function pending(container, title) { const box = element("div", "pending-block"); box.append(element("strong", "", title)); container.append(box); }
function detailsBox(title) { const details = element("details", "context-details"); const summary = element("summary"); summary.append(element("span", "", title), element("span", "", "+")); details.append(summary); return details; }
function evidenceButton(label, frameId, ms) {
  const button = element("button", "evidence-chip", label);
  const index = frameId ? state.record.frames.findIndex((f) => f.frame_id === frameId) : nearestFrame(ms);
  button.disabled = index < 0;
  button.addEventListener("click", () => selectFrame(index, { scroll: true }));
  return button;
}
function renderAnnotation(container, annotation) {
  container.append(element("span", "annotation-label", "VISIBLE IN THIS FRAME"), element("p", "annotation-description", annotation.visible_observation || "No visible observation was saved."));
  if (annotation.visibility) container.append(element("span", "visibility", `Visibility: ${humanize(annotation.visibility)}`));
  if (annotation.contextual_claims?.length) {
    const details = detailsBox(`Video context · ${annotation.contextual_claims.length}`);
    annotation.contextual_claims.forEach((claim) => {
      const item = element("div", "context-claim"); item.append(element("p", "", claim.claim));
      for (const interval of claim.evidence_intervals || []) item.append(evidenceButton(`${formatTime(interval.start_ms)}–${formatTime(interval.end_ms)} ↗`, null, interval.start_ms));
      for (const id of claim.evidence_frame_ids || []) { const f = state.record.frames.find((row) => row.frame_id === id); if (f) item.append(evidenceButton(`${formatTime(f.timestamp_ms)} ↗`, id)); }
      details.append(item);
    }); container.append(details);
  }
  if (annotation.uncertainties?.length) { const section = element("div", "uncertainties"); section.append(element("strong", "", "UNCERTAINTY")); annotation.uncertainties.forEach((text) => section.append(element("p", "", text))); container.append(section); }
}
function renderDetail(detail) {
  ["qwen-content", "medgemma-content", "selection-context", "supporting-evidence"].forEach(clear);
  const frame = detail.frame || state.record.frames[state.index];
  $("image-filename").textContent = frame.source_path?.split("/").pop() || frame.frame_id;
  const selection = detail.raw?.selection || frame.selection || frame;
  const reason = selection.model_reason || selection.reason || frame.model_reason;
  if (reason || frame.coverage_override || selection.coverage_override) {
    const box = element("div", "selection-reason");
    const decision = selection.model_decision || frame.model_decision;
    const details = document.createElement("details"); details.open = frame.status === "dropped";
    details.append(element("summary", "", `Qwen selection: ${decision === "drop" ? "drop" : decision === "keep" ? "keep" : humanize(decision || "pending")}`));
    if (reason) details.append(element("p", "", reason));
    if (frame.coverage_override || selection.coverage_override) details.append(element("p", "", "Retained as a coverage anchor even though Qwen suggested dropping it."));
    box.append(details); $("selection-context").append(box);
  }
  if (detail.qwen) renderAnnotation($("qwen-content"), detail.qwen);
  else pending($("qwen-content"), frame.status === "selected" ? "Awaiting annotation" : "No key-frame annotation");
  const review = detail.medgemma;
  $("medgemma-state").textContent = review ? humanize(review.assessment || "EVIDENCE REVIEW").toUpperCase() : "";
  if (review) {
    renderAnnotation($("medgemma-content"), review.revised_annotation || review);
    if (review.corrections?.length) {
      const corrections = detailsBox(`Corrections · ${review.corrections.length}`);
      for (const correction of review.corrections) { const row = element("div", "correction"); row.append(element("p", "muted", `Original: ${correction.original_text}`), element("p", "", `Revised: ${correction.revised_text}`), element("p", "muted", correction.reason)); corrections.append(row); }
      $("medgemma-content").append(corrections);
    }
    if (review.status === "needs_more_evidence" || review.evidence_requests?.length) {
      const box = element("div", "deferred-box"); box.append(element("h4", "", "More evidence requested"), element("p", "", "Deferred"));
      for (const request of review.evidence_requests || []) { const item = element("div", "request-item"); item.append(element("strong", "", request.question), element("p", "", request.reason)); if (Number.isFinite(request.start_ms)) item.append(evidenceButton(`${formatTime(request.start_ms)}–${formatTime(request.end_ms)} ↗`, null, request.start_ms)); box.append(item); }
      $("medgemma-content").append(box);
    }
  } else pending($("medgemma-content"), "Awaiting MedGemma review");
  for (const model of ["qwen", "medgemma"]) {
    const content = $(`${model}-content`), present = Boolean(detail[model]);
    $(`${model}-block`).classList.toggle("has-ai", present);
    if (!present) continue;
    const original = element("div", "ai-content");
    original.append(...content.childNodes);
    const override = detail.curation?.annotations?.[model];
    if (typeof override === "string") {
      const revision = element("section", "human-revision");
      revision.append(element("span", "annotation-label", "HUMAN-EDITED ANNOTATION"), element("p", "human-description", override || "Annotation cleared by human editor."));
      content.append(revision);
      const source = detailsBox("Original AI-generated annotation"); source.classList.add("ai-original"); source.append(original); content.append(source);
    } else content.append(original);
  }
  const evidence = detail.evidence || [];
  if (evidence.length) {
    $("supporting-evidence").append(element("p", "support-label", `${review ? "Evidence supplied to MedGemma" : "Prepared review evidence"} · ${evidence.length} frames`));
    const strip = element("div", "support-strip");
    evidence.forEach((frame) => { const button = element("button", "support-thumb"); const image = element("img"); const url = localURL(frame.image_url); if (url) image.src = url; image.loading = "lazy"; image.alt = `Evidence at ${formatTime(frame.timestamp_ms)}`; button.append(image, element("span", "", `${formatTime(frame.timestamp_ms)} · ${(frame.evidence_roles || frame.roles || []).join(", ")}`)); button.addEventListener("click", () => { const index = state.record.frames.findIndex((f) => f.frame_id === frame.frame_id); if (index >= 0) selectFrame(index, { scroll: true }); }); strip.append(button); });
    $("supporting-evidence").append(strip);
  }
  renderCuration(); renderSource(detail);
}

function renderCuration() {
  const detail = state.detail, deleted = Boolean(detail?.curation?.deleted);
  const status = state.record?.frames[state.index]?.status;
  const enhancedFrame = status === "selected" || status === "dropped";
  $("pane-annotations").hidden = !enhancedFrame;
  $("supporting-evidence").hidden = !enhancedFrame;
  const available = enhancedFrame && Boolean(detail?.qwen || detail?.medgemma);
  const busy = state.curationBusy || state.exportBusy;
  $("edit-enhancement").disabled = !available || deleted || busy;
  $("delete-enhancement").disabled = !available || busy;
  $("delete-enhancement").hidden = deleted;
  $("restore-enhancement").hidden = !deleted;
  $("restore-enhancement").disabled = !available || busy;
  $("enhancement-deleted").hidden = !deleted;
  $("enhancement-models").hidden = deleted;
}
function annotationText(model) {
  const value = state.detail?.[model];
  if (!value) return "";
  const annotation = value.revised_annotation || value;
  const parts = [annotation.visible_observation || ""];
  if (annotation.visibility) parts.push(`Visibility: ${humanize(annotation.visibility)}`);
  if (annotation.contextual_claims?.length) parts.push("Video context:\n" + annotation.contextual_claims.map((claim) => {
    const intervals = (claim.evidence_intervals || []).map((range) => `${formatTime(range.start_ms)}–${formatTime(range.end_ms)}`);
    const frames = (claim.evidence_frame_ids || []).map((id) => `frame ${id}`);
    const evidence = [...intervals, ...frames];
    return `${claim.claim}${evidence.length ? ` [Evidence: ${evidence.join(", ")}]` : ""}`;
  }).join("\n"));
  if (annotation.uncertainties?.length) parts.push("Uncertainties:\n" + annotation.uncertainties.join("\n"));
  if (value.corrections?.length) parts.push("Corrections:\n" + value.corrections.map((row) => `${row.original_text} → ${row.revised_text}\n${row.reason}`).join("\n"));
  if (value.evidence_requests?.length) parts.push("More evidence requested:\n" + value.evidence_requests.map((row) => [row.question, row.reason].filter(Boolean).join("\n")).join("\n"));
  return parts.join("\n\n");
}
function curationTarget() {
  return { record_id: state.record.id, frame_id: state.record.frames[state.index].frame_id, review_identity: state.record.review_identity, expected_updated_at: state.detail?.curation?.updated_at || null };
}
function openEditor() {
  if (!state.detail || state.curationBusy || state.exportBusy || state.detail.curation?.deleted) return;
  $("surgery-video").pause();
  const initial = {};
  for (const model of ["qwen", "medgemma"]) {
    const available = Boolean(state.detail[model]);
    $(`edit-${model}-field`).hidden = !available;
    if (available) { initial[model] = state.detail.curation?.annotations?.[model] ?? annotationText(model); $(`edit-${model}`).value = initial[model]; }
  }
  if (!Object.keys(initial).length) return;
  state.editor = { ...curationTarget(), initial };
  $("editor-frame").textContent = `${state.record.case_id || state.record.title} · Frame ${state.index + 1} · ${formatTime(state.record.frames[state.index].timestamp_ms, true)}`;
  $("editor-error").hidden = true;
  $("enhancement-editor").showModal();
}
function closeEditor() { if (!state.curationBusy) { $("enhancement-editor").close(); state.editor = null; } }
function curationBusy(value) {
  state.curationBusy = value; renderCuration();
  ["editor-save", "editor-cancel", "editor-close", "edit-qwen", "edit-medgemma"].forEach((id) => { $(id).disabled = value; });
  $("editor-save").textContent = value ? "Saving…" : "Save revision";
  $("export").disabled = value || state.exportBusy || !state.record;
}
async function saveCuration(target, action, annotations) {
  const response = await fetch(`/api/records/${encodeURIComponent(target.record_id)}/frames/${encodeURIComponent(target.frame_id)}/curation`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ review_identity: target.review_identity, expected_updated_at: target.expected_updated_at, action, ...(annotations ? { annotations } : {}) })
  });
  const curation = await response.json();
  if (!response.ok) throw new Error(curation.error || `Could not save this change (${response.status}).`);
  if (state.record?.id !== target.record_id || state.record?.review_identity !== target.review_identity) return;
  const frame = state.record.frames.find((row) => row.frame_id === target.frame_id);
  if (frame) frame.curation = curation;
  // A changed enhancement requires another assessment; keep the reviewer’s notes.
  if (state.notes[target.frame_id]?.status === "reviewed") {
    state.notes[target.frame_id] = { ...state.notes[target.frame_id], status: "unreviewed", needs_recheck: true };
    try { localStorage.setItem(noteKey(), JSON.stringify(state.notes)); } catch { /* current review remains exportable */ }
  }
  if (state.detail && state.record.frames[state.index]?.frame_id === target.frame_id) {
    state.detail.curation = curation; state.detail.frame.curation = curation;
    renderDetail(state.detail); renderReview();
  }
  renderTimeline();
}
async function submitEditor(event) {
  event.preventDefault(); if (!state.editor || state.curationBusy) return;
  const target = state.editor, annotations = {};
  for (const [model, original] of Object.entries(target.initial)) if ($(`edit-${model}`).value !== original) annotations[model] = $(`edit-${model}`).value;
  if (!Object.keys(annotations).length) { closeEditor(); return; }
  curationBusy(true); $("editor-error").hidden = true;
  try {
    await saveCuration(target, "edit", annotations);
    curationBusy(false); closeEditor(); toast("Human revision saved.");
  } catch (error) { $("editor-error").textContent = error.message; $("editor-error").hidden = false; }
  finally { curationBusy(false); }
}
async function toggleEnhancement(action) {
  if (!state.detail || state.curationBusy || state.exportBusy) return;
  $("surgery-video").pause(); const target = curationTarget(); curationBusy(true);
  try { await saveCuration(target, action); toast(action === "delete" ? "Enhancement deleted. Use the restore icon to undo." : "Enhancement restored."); }
  catch (error) { showError(error.message); }
  finally { curationBusy(false); }
}
function dataTable(values) {
  const table = element("table", "data-table"); const body = document.createElement("tbody");
  for (const [label, value] of Object.entries(values || {})) { const row = document.createElement("tr"); const key = element("th", "", label); key.scope = "row"; const cell = element("td", /sha256|hash/i.test(label) ? "hash-value" : "", display(value)); row.append(key, cell); body.append(row); }
  table.append(body); return table;
}
function renderSource(detail) {
  const container = clear("source-annotations"), rows = detail.original_annotations || [];
  if (!rows.length) container.append(element("p", "muted", "No matching source annotation rows are available. Missing labels do not establish absence."));
  for (const annotation of rows) {
    const raw = annotation.raw_value || annotation.raw || annotation;
    const item = element("div", "source-label"); item.append(element("span", "label-kind", humanize(annotation.original_kind || annotation.kind || "label")), element("strong", "", raw.label?.trim() || "Unlabeled source row"));
    const coords = ["x1", "y1", "x2", "y2"].filter((key) => raw[key] !== undefined && raw[key] !== "").map((key) => `${key}: ${raw[key]}`).join(" · ");
    if (coords) item.append(element("p", "", coords));
    const locator = annotation.source_locator;
    if (locator) item.append(element("p", "", `${locator.filename || "Source CSV"} · row ${locator.locator ?? "?"}`));
    container.append(item);
  }
  const frame = detail.provenance || detail.frame;
  clear("provenance").append(dataTable({ "Source file": frame.source_path?.split("/").pop(), "Playback time": formatTime(frame.timestamp_ms, true), "Timestamp basis": humanize(frame.timestamp_basis), "Source frame index": frame.frame_index, "Release frame number": frame.release_frame_index, "Image dimensions": frame.width && frame.height ? `${frame.width} × ${frame.height}` : null, "Source SHA-256": frame.source_sha256, "Image SHA-256": frame.image_sha256 }));
  $("raw-data").textContent = JSON.stringify({ ...detail, outcomes: detail.outcomes || state.record.outcomes, metadata: state.record.metadata, runs: state.record.runs }, null, 2);
  const links = clear("artifact-links");
  const seenLinks = new Set();
  for (const artifact of [...(detail.artifacts || []), ...(state.record.artifacts || [])]) { const url = localURL(artifact.url); if (!url || seenLinks.has(url)) continue; seenLinks.add(url); const link = element("a", "", `${artifact.label} ↗`); link.href = url; link.target = "_blank"; link.rel = "noopener"; links.append(link); }
}
function renderOutcomes(outcomes) {
  const container = clear("outcome-values"); clear("case-data");
  $("outcome-scope").textContent = outcomes?.status === "available" && state.record.case_id?.match(/^(S\dA\d|Clip\d)$/) ? "Simulated cadaveric repair · Recorded technical outcome" : "";
  if (!outcomes || outcomes.status === "unavailable") { container.append(element("p", "muted", "No linked outcome data")); return; }
  const raw = outcomes.raw || {};
  const preferred = [];
  if ("Leak At 40mmHg" in raw) preferred.push({ label: "Leak at 40 mmHg", value: raw["Leak At 40mmHg"] === "N" ? "No leak" : raw["Leak At 40mmHg"] === "Y" ? "Leak reported" : "Not reported" });
  if ("Time for repair" in raw) { const seconds = Number(raw["Time for repair"]); preferred.push({ label: "Recorded repair time", value: raw["Time for repair"] !== "" && Number.isFinite(seconds) ? formatTime(seconds * 1000) : "Not reported", unit: "min:sec" }); }
  const experienceKey = Object.keys(raw).find((key) => /postgraduate year/i.test(key));
  if (experienceKey) preferred.push({ label: "Postgraduate year", value: raw[experienceKey] });
  const fields = preferred.length ? preferred : (outcomes.values || []).slice(0, 3);
  fields.forEach((field) => { const group = element("dl", "outcome-value"); group.append(element("dt", "", field.label)); const value = element("dd", "", display(field.value)); if (field.unit) value.append(element("small", "", ` ${field.unit}`)); group.append(value); container.append(group); });
  $("case-data").append(dataTable(raw));
  const locator = outcomes.source_locator;
  if (locator) $("case-data").append(element("p", "muted", `Source: ${locator.filename || "sospine_outcomes.csv"} · row ${locator.locator ?? locator.line ?? "recorded in provenance"}`));
}
function contextWindow(record = state.record, index = state.index) {
  if (!record?.frames?.length) return null;
  index = Math.max(0, Math.min(record.frames.length - 1, Math.round(Number(index) || 0)));
  const startIndex = Math.max(0, index - 4), stopIndex = Math.min(record.frames.length, index + 5);
  return { startIndex, stopIndex, frameCount: stopIndex - startIndex,
    startMs: record.frames[startIndex].timestamp_ms,
    endMs: stopIndex < record.frames.length ? record.frames[stopIndex].timestamp_ms : record.duration_ms,
    targetFrameId: record.frames[index].frame_id, targetIndex: index };
}
function renderContextWindow() {
  const context = contextWindow();
  $("view-video").disabled = !context;
  $("view-video").textContent = context ? `${context.frameCount}-frame video` : "Video";
  $("view-video").title = "Play up to four source frames before and after this review frame";
  $("context-range").textContent = context ? `Frames ${context.startIndex + 1}–${context.stopIndex} · target ${context.targetIndex + 1}` : "";
}
function stopContextVideo() {
  clearTimeout(contextLoadTimer); clearTimeout(contextStartTimer);
  state.contextRequest?.abort(); state.contextRequest = null;
  const video = $("surgery-video"); video.pause(); video.removeAttribute("src"); video.load();
  if (state.context?.blobURL) URL.revokeObjectURL(state.context.blobURL);
  state.context = null;
}
function contextPlaybackFailed(message) {
  setMode("still"); toast(message);
}
async function loadContextVideo() {
  if (state.mode !== "video") return;
  const context = contextWindow(); if (!context) return;
  stopContextVideo();
  const controller = new AbortController(); state.contextRequest = controller;
  $("video-loading").hidden = false;
  contextLoadTimer = setTimeout(() => {
    if (state.contextRequest === controller) contextPlaybackFailed("Nearby-frame video timed out. Select Video to retry.");
  }, 45000);
  const url = `/api/records/${encodeURIComponent(state.record.id)}/frames/${encodeURIComponent(context.targetFrameId)}/context-video?revision=${encodeURIComponent(state.record.review_identity)}`;
  try {
    const response = await fetch(url, { signal: controller.signal, cache: "no-store" });
    if (!response.ok) { const error = await response.json(); throw new Error(error.error || "Nearby-frame video is unavailable."); }
    const blob = await response.blob();
    if (state.contextRequest !== controller || state.mode !== "video") return;
    const video = $("surgery-video");
    state.context = { ...context, blobURL: URL.createObjectURL(blob) };
    video.src = state.context.blobURL; video.preload = "auto"; video.muted = true; video.load();
    try { await video.play(); }
    catch (error) {
      // Browsers may require a second Play gesture; the loaded controls stay usable.
      if (state.contextRequest === controller && error.name !== "AbortError") $("video-loading").hidden = true;
    }
  } catch (error) {
    if (state.contextRequest === controller && error.name !== "AbortError") contextPlaybackFailed(error.message);
  }
}
function contextVideoReady() {
  const video = $("surgery-video");
  if (state.mode === "video" && state.context && video.currentSrc === state.context.blobURL && video.readyState >= 2) {
    clearTimeout(contextLoadTimer); $("video-loading").hidden = true;
  }
}
function setMode(mode) {
  state.mode = mode; const video = $("surgery-video");
  $("view-still").classList.toggle("active", mode === "still"); $("view-still").setAttribute("aria-pressed", String(mode === "still"));
  $("view-video").classList.toggle("active", mode === "video"); $("view-video").setAttribute("aria-pressed", String(mode === "video"));
  $("frame-image").hidden = mode === "video"; video.hidden = mode !== "video";
  $("image-filename").hidden = mode === "video"; $("context-range").hidden = mode !== "video";
  $("full-image").textContent = mode === "video" ? "Open target image ↗" : "Open image ↗";
  document.querySelector(".image-stage").classList.toggle("video-mode", mode === "video");
  renderContextWindow();
  if (mode === "video") { $("image-error").hidden = true; return loadContextVideo(); }
  stopContextVideo(); $("video-loading").hidden = true;
  if (state.record?.frames[state.index]) $("image-time").textContent = formatTime(state.record.frames[state.index].timestamp_ms, true);
}
async function exportReview() {
  if (!state.record || state.curationBusy || state.exportBusy) return;
  const target = state.record;
  state.exportBusy = true; renderCuration(); $("export").disabled = true;
  try {
    const latest = await api(`/api/records/${encodeURIComponent(target.id)}`);
    if (state.record !== target) throw new Error("The surgery record changed. Export the selected record again.");
    if (latest.review_identity !== target.review_identity) throw new Error("The dataset has changed. Refresh before exporting your review.");
    const overlays = new Map(latest.frames.map((frame) => [frame.frame_id, frame.curation]));
    for (const frame of state.record.frames) frame.curation = overlays.get(frame.frame_id) || null;
    validateReviewRevisions(); renderTimeline();
    if (state.detail) { state.detail.curation = state.record.frames[state.index].curation; state.detail.frame.curation = state.detail.curation; renderDetail(state.detail); renderReview(); }
    const entries = state.record.frames.filter((frame) => state.notes[frame.frame_id]).map((frame) => ({ ...state.notes[frame.frame_id], source: { frame_id: frame.frame_id, timestamp_ms: frame.timestamp_ms, frame_index: frame.frame_index, selection_status: frame.status } }));
    const enhancements = state.record.frames.filter((frame) => frame.curation).map((frame) => ({ ...frame.curation, included: !frame.curation.deleted, source: { frame_id: frame.frame_id, timestamp_ms: frame.timestamp_ms, frame_index: frame.frame_index } }));
    const packet = { schema_version: "yasargil-inspector-review-v1", worksheet_status: "draft", exported_at: new Date().toISOString(), record_id: state.record.id, review_identity: state.record.review_identity, case_id: state.record.case_id, video_sha256: state.record.video_sha256 || state.record.metadata?.video_sha256 || null, timestamp_basis: state.record.timestamp_basis, runs: state.record.runs, notes: entries, enhancements, training_eligible: false };
    const content = JSON.stringify(packet);
    const body = new URLSearchParams({ review: content }).toString();
    if (new TextEncoder().encode(body).length > 2 * 1024 * 1024) { showError("This review is too large to export in one file. Shorten very large notes before exporting."); return; }
    let downloadTarget = $("review-export-target");
    if (!downloadTarget) { downloadTarget = element("iframe"); downloadTarget.id = "review-export-target"; downloadTarget.name = "review-export-target"; downloadTarget.title = "Review file download"; downloadTarget.hidden = true; document.body.append(downloadTarget); }
    const form = element("form"); form.action = "/api/review-export"; form.method = "post"; form.target = downloadTarget.name; form.hidden = true;
    const input = element("input"); input.type = "hidden"; input.name = "review"; input.value = content; form.append(input); document.body.append(form); form.submit(); form.remove();
    toast(`Requested download of ${entries.length} frame note${entries.length === 1 ? "" : "s"} and ${enhancements.length} enhancement change${enhancements.length === 1 ? "" : "s"}.`);
  } catch (error) { showError(error.message); }
  finally { state.exportBusy = false; renderCuration(); $("export").disabled = state.curationBusy || !state.record; }
}

$("edit-enhancement").addEventListener("click", openEditor);
$("delete-enhancement").addEventListener("click", () => toggleEnhancement("delete"));
$("restore-enhancement").addEventListener("click", () => toggleEnhancement("restore"));
$("enhancement-form").addEventListener("submit", submitEditor);
$("editor-close").addEventListener("click", closeEditor);
$("editor-cancel").addEventListener("click", closeEditor);
$("enhancement-editor").addEventListener("cancel", (event) => { event.preventDefault(); closeEditor(); });
$("refresh").addEventListener("click", refreshRecords); $("empty-refresh").addEventListener("click", refreshRecords);
$("open-dataset-folder").addEventListener("click", openDatasetFolder);
$("record-select").addEventListener("change", (event) => loadRecord(event.target.value));
$("timeline").addEventListener("pointerdown", (event) => { if (!state.record || event.button !== 0) return; const index = pointerIndex(event); state.dragging = true; $("timeline").setPointerCapture(event.pointerId); $("timeline").focus({ preventScroll: true }); selectFrame(index); });
$("timeline").addEventListener("pointermove", (event) => {
  if (!state.record) return; const index = pointerIndex(event);
  if (state.dragging) { if (index !== state.index) selectFrame(index); return; }
  const hover = $("timeline-hover"); hover.textContent = `${formatTime(state.record.frames[index].timestamp_ms)} · ${index + 1}`; hover.style.left = `${Math.max(60, Math.min($("timeline").clientWidth - 60, timelineX(state.record.frames[index].timestamp_ms)))}px`; hover.hidden = index === state.index;
});
$("timeline").addEventListener("pointerup", () => { state.dragging = false; $("timeline-hover").hidden = true; });
$("timeline").addEventListener("pointercancel", () => { state.dragging = false; });
$("timeline").addEventListener("pointerleave", () => { $("timeline-hover").hidden = true; });
$("previous-frame").addEventListener("click", () => moveFrame(-1)); $("next-frame").addEventListener("click", () => moveFrame(1));
$("previous-keyframe").addEventListener("click", () => moveFrame(-1, true)); $("next-keyframe").addEventListener("click", () => moveFrame(1, true));
$("frame-jump").addEventListener("change", (event) => selectFrame(Number(event.target.value) - 1, { scroll: true }));
$("view-still").addEventListener("click", () => setMode("still")); $("view-video").addEventListener("click", () => setMode("video"));
$("surgery-video").addEventListener("timeupdate", () => {
  if (state.mode === "video" && state.context) $("image-time").textContent = formatTime(state.context.startMs + $("surgery-video").currentTime * 1000, true);
});
$("surgery-video").addEventListener("loadeddata", contextVideoReady);
$("surgery-video").addEventListener("canplay", contextVideoReady);
$("surgery-video").addEventListener("error", () => {
  if (state.mode === "video" && state.context) contextPlaybackFailed("Nearby-frame video could not be played. Select Video to retry.");
});
$("frame-image").addEventListener("error", () => { if (state.mode === "still") $("image-error").hidden = false; });
$("frame-image").addEventListener("click", () => { if ($("full-image").hasAttribute("href")) $("full-image").click(); });
document.querySelectorAll("[data-review-state]").forEach((button) => button.addEventListener("click", () => changeReview(button.dataset.reviewState)));
$("review-save").addEventListener("click", persistNote);
$("review-edit").addEventListener("click", editReview);
$("review-cancel").addEventListener("click", cancelReview);
$("review-note").addEventListener("focus", () => $("surgery-video").pause());
$("review-note").addEventListener("input", () => changeReview()); $("export").addEventListener("click", exportReview);
window.addEventListener("beforeunload", (event) => { if ([...reviewDrafts.values()].some(reviewDraftChanged)) { event.preventDefault(); event.returnValue = ""; } });
document.addEventListener("keydown", (event) => { if (!state.record || $("enhancement-editor").open || event.ctrlKey || event.metaKey || event.altKey || /^(INPUT|TEXTAREA|SELECT|VIDEO)$/.test(event.target.tagName) || event.target.isContentEditable) return; if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); moveFrame(event.key === "ArrowLeft" ? -1 : 1, event.shiftKey); } else if (event.target === $("timeline") && ["Home", "End"].includes(event.key)) { event.preventDefault(); selectFrame(event.key === "Home" ? 0 : state.record.frames.length - 1, { scroll: true }); } });
new ResizeObserver(() => { if (state.record) renderTimeline(); }).observe($("timeline-scroll"));
refreshRecords();
