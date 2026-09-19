// Run with: node --test tests/test_review_ui.js
// Exercise the shipped review code and event handlers with an in-memory DOM/storage.
// No browser, network, or dataset writes are required.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const uiRoot = path.join(__dirname, '../src/yasargil/review_ui');
const source = fs.readFileSync(path.join(uiRoot, 'app.js'), 'utf8');
const html = fs.readFileSync(path.join(uiRoot, 'index.html'), 'utf8');

function harness() {
  const nodes = new Map(), allNodes = [], decisions = [], markers = [], downloads = [], requests = [], revoked = [], timers = new Map();
  let fetchHandler, timerId = 0, blobId = 0;
  class TestURL extends URL { static createObjectURL() { return `blob:context-${++blobId}`; } static revokeObjectURL(value) { revoked.push(value); } }
  class Node {
    constructor(tag = 'div', attributes = '') {
      this.tagName = tag.toUpperCase(); this.hidden = /\shidden(?:\s|$)/.test(attributes);
      this.disabled = false; this.textContent = ''; this.value = ''; this.children = [];
      this.dataset = {}; this.attributes = new Map(); this.listeners = new Map(); this.style = {};
      const classes = new Set();
      this.classList = { toggle(name, force) { if (force) classes.add(name); else classes.delete(name); }, contains: (name) => classes.has(name) };
      for (const [, name, value] of attributes.matchAll(/([\w-]+)="([^"]*)"/g)) {
        this.attributes.set(name, value);
        if (name === 'id') this.id = value;
        if (name.startsWith('data-')) this.dataset[name.slice(5).replace(/-([a-z])/g, (_, char) => char.toUpperCase())] = value;
      }
    }
    addEventListener(type, listener) { this.listeners.set(type, listener); }
    emit(type, event = {}) { return this.listeners.get(type)?.({ target: this, preventDefault() {}, ...event }); }
    click() { if (!this.disabled) return this.emit('click'); }
    focus() { return this.emit('focus'); }
    pause() { this.paused = true; }
    load() { this.currentSrc = this.src || ""; this.readyState = 0; }
    play() { this.paused = false; return Promise.resolve(); }
    removeAttribute(name) { this.attributes.delete(name); if (name === "src") delete this.src; }
    setPointerCapture(pointerId) { this.capturedPointerId = pointerId; }
    setAttribute(name, value) { this.attributes.set(name, value); }
    getAttribute(name) { return this.attributes.get(name); }
    append(...children) { this.children.push(...children); for (const child of children) if (child.id) nodes.set(child.id, child); }
    replaceChildren(...children) { this.children = []; this.append(...children); }
    remove() {}
    submit() { downloads.push(JSON.parse(this.children.find((child) => child.name === 'review').value)); }
  }
  for (const [, tag, attributes] of html.matchAll(/<([a-z][\w-]*)\b([^>]*)>/g)) {
    const node = new Node(tag, attributes); allNodes.push(node);
    if (node.id) nodes.set(node.id, node);
    if (node.dataset.reviewState) decisions.push(node);
  }
  const stored = new Map();
  const storage = {
    writes: 0, fail: false,
    getItem: (key) => stored.get(key) ?? null,
    setItem(key, value) { if (this.fail) throw new Error('Storage quota exceeded'); this.writes++; stored.set(key, value); },
  };
  const document = {
    getElementById: (id) => nodes.get(id) || null,
    querySelector: (selector) => allNodes.find((node) => selector.startsWith('.') && (node.getAttribute('class') || '').split(' ').includes(selector.slice(1))) || null,
    querySelectorAll: (selector) => selector === '[data-review-state]' ? decisions : selector === '.timeline-frame' ? markers : [],
    createElement: (tag) => new Node(tag), body: new Node('body'), addEventListener() {},
  };
  const context = vm.createContext({
    document, localStorage: storage, window: { addEventListener() {} },
    ResizeObserver: class { observe() {} },
    setTimeout(fn, delay) { const id = ++timerId; timers.set(id, {fn, delay}); return id; }, clearTimeout(id) { timers.delete(id); },
    URL: TestURL, URLSearchParams, TextEncoder, AbortController, location: { origin: 'http://127.0.0.1:8765' },
    fetch: async (url, options) => { requests.push({url, options}); return fetchHandler ? fetchHandler(url, options) : { ok: true, json: async () => JSON.parse(JSON.stringify(context.reviewTest.state.record)) }; },
  });
  // Keep the entire script, including its event wiring; suppress only startup loading.
  assert.match(source, /\nrefreshRecords\(\);\s*$/);
  vm.runInContext(source.replace(/\nrefreshRecords\(\);\s*$/, '\n') + `
    globalThis.reviewTest = { state, reviewDrafts, noteKey, reviewDraftKey, loadNotes, renderReview, updateReviewCount, renderCuration, contextWindow, setMode };
  `, context);
  const api = context.reviewTest;
  api.state.record = { id: 'record-one', review_identity: 'version-one', frames: [
    { frame_id: 'frame-one', timestamp_ms: 0 }, { frame_id: 'frame-two', timestamp_ms: 1000 },
  ] };
  api.loadNotes(); api.renderReview(); api.updateReviewCount();
  return {
    ...api, storage, stored, decisions, downloads, requests, revoked,
    setFetch(handler) { fetchHandler = handler; },
    context(index = 6, count = 13) {
      api.state.record.duration_ms = count * 1000;
      api.state.record.frames = Array.from({length: count}, (_, i) => ({frame_id: `f${i}`, timestamp_ms: i * 1000, status: 'source'}));
      api.state.index = index; api.loadNotes(); api.renderReview();
    },
    timeout(delay) { const timer = [...timers.values()].find(timer => timer.delay === delay); assert.ok(timer); timer.fn(); },
    node: (id) => { assert.ok(nodes.has(id), `Missing real HTML element ${id}`); return nodes.get(id); },
    click: (id) => nodes.get(id).click(),
    type(text) { nodes.get('review-note').value = text; nodes.get('review-note').emit('input'); },
    decide(status) { decisions.find((node) => node.dataset.reviewState === status).click(); },
    frame(index) { api.state.index = index; api.renderReview(); },
    persisted: () => JSON.parse(stored.get(api.noteKey()) || '{}'),
    seed(notes) { stored.set(api.noteKey(), JSON.stringify(notes)); api.loadNotes(); api.renderReview(); api.updateReviewCount(); },
    timeline() {
      // Supply layout measurements; exercise the real pointer handlers and hit testing.
      // Frame loading is independent of the timeline's chosen index.
      vm.runInContext('selectFrame = (index) => { state.index = index; };', context);
      api.state.record.duration_ms = 4000;
      api.state.record.frames = [
        { frame_id: 'source-first', timestamp_ms: 0, status: 'source' },
        { frame_id: 'selected-frame', timestamp_ms: 1000, status: 'selected' },
        { frame_id: 'nearby-source', timestamp_ms: 1020, status: 'source' },
        { frame_id: 'dropped-frame', timestamp_ms: 2000, status: 'dropped' },
        { frame_id: 'source-last', timestamp_ms: 3000, status: 'source' },
      ];
      const slider = nodes.get('timeline'), track = nodes.get('timeline-track');
      slider.clientWidth = 122;
      track.offsetLeft = 9; track.clientLeft = 0; track.clientWidth = 100;
      track.getBoundingClientRect = () => ({ left: 100, right: 200, top: 62, bottom: 66 });
      for (const [index, left] of [[1, 119], [3, 144]]) {
        const marker = new Node('span'); marker.dataset.index = String(index);
        marker.getBoundingClientRect = () => ({ left, right: left + 12, top: 58, bottom: 70, width: 12, height: 12 });
        markers.push(marker);
      }
      return { slider, track, markers };
    },
    export() {
      // These views are independent of review persistence and need a real layout engine.
      vm.runInContext('renderCuration = () => {}; renderTimeline = () => {};', context);
      return nodes.get('export').click();
    },
  };
}

test('clicking a colored circle selects its exact frame rather than a nearby source observation', () => {
  const ui = harness(), { slider, markers } = ui.timeline();
  // At x=129 the nearest timestamp is the ordinary frame at 1.02s.
  // The selected circle owns the click instead.
  slider.emit('pointerdown', { target: markers[0], button: 0, pointerId: 7, clientX: 129, clientY: 65 });
  assert.equal(ui.state.index, 1);
  assert.equal(slider.capturedPointerId, 7);
  slider.emit('pointerup');
  slider.emit('pointerdown', { target: markers[1], button: 0, pointerId: 8, clientX: 153, clientY: 65 });
  assert.equal(ui.state.index, 3);
});

test('captured dragging continues to select exact colored circles and stops on pointerup', () => {
  const ui = harness(), { slider } = ui.timeline();
  slider.emit('pointerdown', { button: 0, pointerId: 7, clientX: 102, clientY: 65 });
  assert.equal(ui.state.index, 0);
  // Pointer capture retargets these events to the slider, not the child circles.
  slider.emit('pointermove', { clientX: 129, clientY: 65 });
  assert.equal(ui.state.index, 1);
  slider.emit('pointermove', { clientX: 153, clientY: 65 });
  assert.equal(ui.state.index, 3);
  slider.emit('pointerup');
  slider.emit('pointermove', { clientX: 129, clientY: 65 });
  assert.equal(ui.state.index, 3);
});

test('gray line selection follows source time including endpoints and a shifted track', () => {
  const ui = harness(), { slider, track } = ui.timeline();
  const click = (clientX, clientY = 65) => {
    slider.emit('pointerdown', { button: 0, pointerId: 7, clientX, clientY });
    slider.emit('pointerup');
  };
  click(130, 69); // Inside its bounding box but outside the circle: source time wins.
  assert.equal(ui.state.index, 2);
  click(180);
  assert.equal(ui.state.index, 4);
  click(98); // Clamp outside the line to the first frame.
  assert.equal(ui.state.index, 0);
  click(206);
  assert.equal(ui.state.index, 4);
  track.getBoundingClientRect = () => ({ left: -50, right: 150, top: 62, bottom: 66 });
  track.clientWidth = 200;
  click(50); // The shifted content midpoint is 2s, independent of viewport origin.
  assert.equal(ui.state.index, 3);
});


test('captured dragging picks the closest circle when key frames overlap', () => {
  const ui = harness(), { slider, markers } = ui.timeline();
  // Both hit areas contain x=130; the dropped frame is centered at 133,
  // closer than the selected frame at 125 despite being later in DOM order.
  markers[1].getBoundingClientRect = () => ({ left: 127, right: 139, top: 58, bottom: 70, width: 12, height: 12 });
  slider.emit('pointerdown', { button: 0, pointerId: 7, clientX: 100, clientY: 64 });
  slider.emit('pointermove', { clientX: 130, clientY: 64 });
  assert.equal(ui.state.index, 3);
  slider.emit('pointermove', { clientX: 128, clientY: 64 });
  assert.equal(ui.state.index, 1);
  slider.emit('pointerup');
});

test('AI annotations are hidden on first load while human review and source data remain available', () => {
  const ui = harness();
  // Initial HTML must be correct before frame metadata or annotations arrive.
  assert.equal(ui.node('pane-annotations').hidden, true);
  assert.equal(ui.node('supporting-evidence').hidden, true);
  assert.equal(ui.node('pane-review').hidden, false);
  assert.equal(ui.node('pane-source').hidden, false);
  ui.state.record = null; ui.state.detail = null; ui.renderCuration();
  assert.equal(ui.node('pane-annotations').hidden, true);
  assert.equal(ui.node('supporting-evidence').hidden, true);
});

test('only selected and dropped frames show AI annotations, including while details load', () => {
  const ui = harness();
  for (const status of ['selected', 'source', 'dropped', 'candidate', 'selected']) {
    ui.state.record.frames[0].status = status;
    // Visibility follows the frame selection immediately, independent of a
    // previous response, deleted enhancement, or an annotation request in flight.
    for (const detail of [null, { qwen: { visible_observation: 'Earlier result' } }, { curation: { deleted: true } }]) {
      ui.state.detail = detail; ui.renderCuration();
      const hidden = status !== 'selected' && status !== 'dropped';
      assert.equal(ui.node('pane-annotations').hidden, hidden, status);
      assert.equal(ui.node('supporting-evidence').hidden, hidden, status);
      assert.equal(ui.node('pane-review').hidden, false, status);
      assert.equal(ui.node('pane-source').hidden, false, status);
    }
  }
});

function savedReview(overrides = {}) {
  return { status: 'reviewed', note: 'The visible action matches.', updated_at: '2026-09-14T12:00:00.000Z', frame_id: 'frame-one', timestamp_ms: 0, ...overrides };
}

test('a new review stays a draft until Save, then becomes read-only with Edit', () => {
  const ui = harness();
  assert.equal(ui.node('review-save-actions').hidden, false);
  assert.equal(ui.node('review-edit').hidden, true);
  ui.type('Check the tissue contact.'); ui.decide('reviewed');
  assert.equal(ui.storage.writes, 0);
  assert.deepEqual(Object.keys(ui.state.notes), []);
  assert.equal(Number(ui.node('stat-reviewed').textContent), 0);
  assert.match(ui.node('save-status').textContent, /Unsaved/);
  ui.click('review-save');
  assert.equal(ui.storage.writes, 1);
  assert.equal(ui.persisted()['frame-one'].note, 'Check the tissue contact.');
  assert.equal(ui.persisted()['frame-one'].status, 'reviewed');
  assert.equal(Number(ui.node('stat-reviewed').textContent), 1);
  assert.equal(ui.node('review-edit').hidden, false);
  assert.equal(ui.node('review-save-actions').hidden, true);
  assert.equal(ui.node('review-note-editor').hidden, true);
  assert.ok(ui.decisions.every((node) => node.disabled));
  ui.decide('flagged');
  assert.equal(ui.persisted()['frame-one'].status, 'reviewed');
  ui.click('review-edit'); ui.type('Revised after checking the evidence.'); ui.decide('flagged');
  assert.equal(ui.storage.writes, 1);
  assert.equal(ui.node('review-save-label').textContent, 'Save changes');
  ui.click('review-save');
  assert.equal(ui.persisted()['frame-one'].status, 'flagged');
  assert.equal(ui.persisted()['frame-one'].note, 'Revised after checking the evidence.');
  assert.equal(Number(ui.node('stat-reviewed').textContent), 0);
});

test('explicitly saving an empty unfinished review creates a saved review', () => {
  const ui = harness(); ui.click('review-save');
  assert.equal(ui.persisted()['frame-one'].status, 'unreviewed');
  assert.equal(ui.persisted()['frame-one'].note, '');
  assert.ok(ui.persisted()['frame-one'].updated_at);
  assert.equal(ui.node('review-edit').hidden, false);
  assert.equal(ui.node('review-saved-text').textContent, 'No written note.');
});

test('Cancel restores the saved review without changing storage', () => {
  const ui = harness(); ui.seed({ 'frame-one': savedReview() });
  ui.click('review-edit'); ui.type('Unwanted changes'); ui.decide('flagged'); ui.click('review-cancel');
  assert.equal(ui.storage.writes, 0);
  assert.equal(ui.reviewDrafts.size, 0);
  assert.equal(ui.node('review-saved-text').textContent, savedReview().note);
  assert.equal(ui.node('review-edit').hidden, false);
  assert.equal(ui.state.notes['frame-one'].status, 'reviewed');
});

test('drafts survive navigation, remain isolated per frame, and Cancel clears an unsaved draft', () => {
  const ui = harness(); ui.type('First frame draft'); ui.decide('flagged');
  ui.frame(1); assert.equal(ui.node('review-note').value, ''); ui.type('Second frame draft');
  ui.frame(0); assert.equal(ui.node('review-note').value, 'First frame draft');
  assert.equal(ui.reviewDrafts.get(ui.reviewDraftKey()).status, 'flagged');
  ui.click('review-save');
  assert.deepEqual(Object.keys(ui.persisted()), ['frame-one']);
  ui.frame(1); assert.equal(ui.node('review-note').value, 'Second frame draft');
  ui.click('review-cancel');
  assert.equal(ui.node('review-note').value, '');
  assert.equal(ui.node('review-edit').hidden, true);
  assert.equal(ui.reviewDrafts.size, 0);
  assert.equal(ui.storage.writes, 1);
});

test('drafts do not cross the dataset revision identity', () => {
  const ui = harness(); ui.type('Draft for original dataset.');
  ui.state.record.review_identity = 'version-two'; ui.loadNotes(); ui.renderReview();
  assert.equal(ui.node('review-note').value, '');
  ui.type('Draft for revised dataset.');
  ui.state.record.review_identity = 'version-one'; ui.loadNotes(); ui.renderReview();
  assert.equal(ui.node('review-note').value, 'Draft for original dataset.');
  assert.equal(ui.storage.writes, 0);
});

test('storage failure retains the draft and saved review until a successful retry', () => {
  const ui = harness(); ui.seed({ 'frame-one': savedReview() });
  ui.click('review-edit'); ui.type('New evidence warrants attention.'); ui.decide('flagged');
  ui.storage.fail = true; ui.click('review-save');
  assert.equal(ui.persisted()['frame-one'].note, savedReview().note);
  assert.equal(ui.state.notes['frame-one'].note, savedReview().note);
  assert.equal(ui.reviewDrafts.get(ui.reviewDraftKey()).note, 'New evidence warrants attention.');
  assert.equal(ui.node('review-save-actions').hidden, false);
  assert.match(ui.node('save-status').textContent, /Could not save/);
  ui.storage.fail = false; ui.click('review-save');
  assert.equal(ui.persisted()['frame-one'].status, 'flagged');
  assert.equal(ui.reviewDrafts.size, 0);
});

test('a concurrent saved edit is not overwritten, and Cancel loads that edit', () => {
  const ui = harness(); ui.seed({ 'frame-one': savedReview() });
  ui.click('review-edit'); ui.type('Local draft');
  ui.stored.set(ui.noteKey(), JSON.stringify({ 'frame-one': savedReview({ note: 'Other tab review', updated_at: '2026-09-14T13:00:00.000Z' }) }));
  ui.click('review-save');
  assert.equal(ui.storage.writes, 0);
  assert.equal(ui.persisted()['frame-one'].note, 'Other tab review');
  assert.match(ui.node('save-status').textContent, /another tab/);
  ui.click('review-cancel');
  assert.equal(ui.node('review-saved-text').textContent, 'Other tab review');
});

test('legacy saved reviews without curation metadata remain editable', () => {
  const ui = harness(); ui.seed({ 'frame-one': savedReview() });
  assert.equal(ui.state.notes['frame-one'].status, 'reviewed');
  assert.equal(ui.node('review-edit').hidden, false);
  ui.click('review-edit'); ui.type('Updated existing note'); ui.click('review-save');
  assert.equal(ui.persisted()['frame-one'].note, 'Updated existing note');
  assert.equal(ui.persisted()['frame-one'].curation_updated_at, null);
});

test('annotation changes reset an in-progress complete decision before Save', () => {
  const ui = harness(); ui.type('Reviewed current evidence'); ui.decide('reviewed');
  ui.state.record.frames[0].curation = { updated_at: '2026-09-14T14:00:00.000Z' };
  ui.renderReview();
  assert.equal(ui.reviewDrafts.get(ui.reviewDraftKey()).status, 'unreviewed');
  assert.match(ui.node('save-status').textContent, /Annotations changed/);
  ui.click('review-save');
  assert.equal(ui.persisted()['frame-one'].status, 'unreviewed');
});

test('export contains saved reviews and never promotes draft text or decisions', async () => {
  const ui = harness(); ui.seed({ 'frame-one': savedReview() });
  ui.click('review-edit'); ui.type('Unsaved replacement'); ui.decide('flagged');
  ui.frame(1); ui.type('Unsaved second frame'); ui.decide('reviewed');
  await ui.export();
  assert.equal(ui.downloads.length, 1);
  assert.equal(ui.downloads[0].notes.length, 1);
  assert.equal(ui.downloads[0].notes[0].note, savedReview().note);
  assert.equal(ui.downloads[0].notes[0].status, 'reviewed');
  assert.equal(ui.downloads[0].training_eligible, false);
  assert.equal(ui.reviewDrafts.size, 2);
  assert.equal(ui.storage.writes, 0);
});


test('context windows contain four source frames per side and clamp at sequence boundaries', () => {
  const ui = harness(); ui.context();
  const middle = ui.contextWindow();
  assert.deepEqual([middle.startIndex, middle.stopIndex, middle.frameCount, middle.startMs, middle.endMs], [2, 11, 9, 2000, 11000]);
  ui.context(0); assert.equal(ui.contextWindow().frameCount, 5);
  ui.context(12); assert.equal(ui.contextWindow().startIndex, 8); assert.equal(ui.contextWindow().frameCount, 5);
  ui.context(1, 3); assert.equal(ui.contextWindow().frameCount, 3);
  assert.equal(ui.contextWindow(ui.state.record, 99).targetIndex, 2);
});

test('context playback starts at the window beginning and never moves the annotation or note target', async () => {
  const ui = harness(); ui.context(); ui.type('Note on target frame six');
  const draftKey = ui.reviewDraftKey();
  ui.setFetch(async () => ({ok: true, blob: async () => ({})}));
  await ui.setMode('video');
  const video = ui.node('surgery-video');
  assert.match(ui.requests[0].url, /frames\/f6\/context-video\?revision=version-one$/);
  assert.equal(video.preload, 'auto'); assert.equal(video.paused, false);
  assert.equal(ui.state.context.startIndex, 2);
  assert.equal(ui.node('context-range').textContent, 'Frames 3–11 · target 7');
  video.readyState = 2; video.emit('loadeddata');
  assert.equal(ui.node('video-loading').hidden, true);
  video.currentTime = 4; video.emit('timeupdate');
  assert.equal(ui.node('image-time').textContent, '00:06.000');
  assert.equal(ui.state.index, 6); assert.equal(ui.reviewDraftKey(), draftKey);
  assert.equal(ui.reviewDrafts.get(draftKey).note, 'Note on target frame six');
  await ui.setMode('still');
  assert.deepEqual(ui.revoked, ['blob:context-1']);
  assert.equal(ui.node('context-range').hidden, true);
});

test('stale video responses cannot replace the newer frame window', async () => {
  const ui = harness(); ui.context();
  let finishOld;
  ui.setFetch(url => url.includes('/f6/') ? new Promise(resolve => {finishOld = resolve;}) : Promise.resolve({ok: true, blob: async () => ({})}));
  const oldRequest = ui.setMode('video');
  ui.state.index = 7;
  await ui.setMode('video');
  finishOld({ok: true, blob: async () => ({})}); await oldRequest;
  assert.equal(ui.state.context.targetFrameId, 'f7');
  assert.equal(ui.state.context.blobURL, 'blob:context-1');
  assert.equal(ui.requests[0].options.signal.aborted, true);
});

test('video errors and timeouts leave a usable still view instead of a permanent loading overlay', async () => {
  const ui = harness(); ui.context();
  ui.setFetch(async () => ({ok: false, json: async () => ({error: 'Source drive unavailable'})}));
  await ui.setMode('video');
  assert.equal(ui.state.mode, 'still'); assert.equal(ui.node('video-loading').hidden, true);
  assert.equal(ui.node('toast').textContent, 'Source drive unavailable');
  ui.setFetch(() => new Promise(() => {}));
  ui.setMode('video'); ui.timeout(45000);
  assert.equal(ui.state.mode, 'still'); assert.equal(ui.node('video-loading').hidden, true);
  assert.equal(ui.requests.at(-1).options.signal.aborted, true);
});

test('opening the dataset folder does not save or discard the current review draft', async () => {
  const ui = harness(); ui.type('Unsaved inspection note');
  const key = ui.reviewDraftKey();
  ui.setFetch(async () => ({ok: true, json: async () => ({opened: true})}));
  await ui.click('open-dataset-folder');
  assert.equal(ui.requests[0].url, '/api/open-dataset-folder');
  assert.equal(ui.requests[0].options.method, 'POST'); assert.equal(ui.requests[0].options.body, '');
  assert.equal(ui.reviewDrafts.get(key).note, 'Unsaved inspection note');
  assert.equal(ui.storage.writes, 0); assert.equal(ui.node('open-dataset-folder').disabled, false);
});
