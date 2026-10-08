'use strict';

const $ = (selector, root = document) => root.querySelector(selector);

const els = {
  form: $('#create'), image: $('#image'), picker: $('#picker'), preview: $('#preview'),
  pickerEmpty: $('#picker-empty'), pickerChange: $('#picker-change'), imageMeta: $('#image-meta'),
  prompt: $('#prompt'), negative: $('#negative'), negativeReset: $('#negative-reset'),
  advanced: $('#advanced'), advSummary: $('#adv-summary'), modeHint: $('#mode-hint'),
  duration: $('#duration'), durationOut: $('#duration-out'), durationHint: $('#duration-hint'),
  resolution: $('#resolution'), resolutionHint: $('#resolution-hint'), steps: $('#steps'),
  seed: $('#seed'), seedRandom: $('#seed-random'), submit: $('#submit'), formError: $('#form-error'),
  jobs: $('#jobs'), jobsEmpty: $('#jobs-empty'), jobsCount: $('#jobs-count'), template: $('#job-template'),
  status: $('#status'), statusModel: $('#status-model'), statusGpu: $('#status-gpu'), statusDetail: $('#status-detail'),
  toast: $('#toast'), logout: $('#logout'),
};

const state = {
  config: null,
  model: null,
  file: null,          // photo picked in this browser
  sourceJob: null,     // or: reuse the photo of an earlier job
  previewUrl: null,
  jobs: [],
  cards: new Map(),    // job id -> card element
  clockOffset: 0,      // server clock minus phone clock, in seconds
  stream: null, streamJob: null, streamPausedUntil: 0,
  pollTimer: null, statusTimer: null,
  submitting: false,
};

const DRAFT_KEY = 'longcat-gui-draft-v1';
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const STATUS_LABEL = { queued: 'Queued', running: 'Running', done: 'Done', failed: 'Failed' };
const ICON = {
  download: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 4v11m0 0-4.5-4.5M12 15l4.5-4.5M5 20h14"/></svg>',
  loop: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M17 2l3 3-3 3"/><path d="M4 11V9a4 4 0 0 1 4-4h12M7 22l-3-3 3-3"/><path d="M20 13v2a4 4 0 0 1-4 4H4"/></svg>',
  reuse: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg>',
  trash: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>',
  retry: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/></svg>',
  cancel: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>',
};

// ---------- small helpers ----------

const now = () => Date.now() / 1000 + state.clockOffset;
const syncClock = (serverNow) => { if (serverNow) state.clockOffset = serverNow - Date.now() / 1000; };
const isActive = (job) => job.status === 'queued' || job.status === 'running';

function fmtClock(seconds) {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(r)}` : `${m}:${pad(r)}`;
}

function fmtApprox(seconds) {
  if (seconds < 90) return `${Math.max(1, Math.round(seconds))} s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

function errorText(data, status) {
  const detail = data && data.detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg || String(d)).join('; ');
  if (typeof detail === 'string') return detail;
  if (status === 413) return 'The file is too large.';
  return `Request failed (HTTP ${status}).`;
}

async function api(path, options = {}) {
  const res = await fetch(path, { credentials: 'same-origin', ...options });
  if (res.status === 401) { location.href = '/login'; throw new Error('Not logged in'); }
  let data = null;
  try { data = await res.json(); } catch (_) { /* empty or not JSON */ }
  if (!res.ok) throw new Error(errorText(data, res.status));
  return data;
}

function storage(fn) {
  try { return fn(window.localStorage); } catch (_) { return null; }
}

let toastTimer = null;
function toast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.classList.toggle('error', isError);
  els.toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove('show'), isError ? 7000 : 5000);
}

function showFormError(message) { els.formError.textContent = message; els.formError.hidden = false; }
function hideFormError() { els.formError.hidden = true; }

// ---------- create form ----------

const currentMode = () => (document.querySelector('input[name="mode"]:checked') || {}).value || 'fast';

function planFor(duration, resolution) {
  const seg = (state.config && state.config.segment) || { frames: 93, cond: 13, fps: 15 };
  const needed = Math.ceil(duration * seg.fps);
  const segments = 1 + Math.max(0, Math.ceil((needed - seg.frames) / (seg.frames - seg.cond)));
  return { segments, fps: resolution === '720p' ? seg.fps * 2 : seg.fps };
}

function setMode(mode, resetSteps) {
  const modes = state.config ? state.config.modes : null;
  if (!modes || !modes[mode]) return;
  const radio = document.getElementById(`mode-${mode}`);
  if (radio) radio.checked = true;
  const m = modes[mode];
  els.steps.min = m.min_steps;
  els.steps.max = m.max_steps;
  const steps = Number(els.steps.value);
  if (resetSteps || !steps || steps < m.min_steps || steps > m.max_steps) els.steps.value = m.steps;
}

function syncSeed() {
  els.seed.disabled = els.seedRandom.checked;
  els.seed.placeholder = els.seedRandom.checked ? 'Random' : 'e.g. 12345';
}

function updateFormHints() {
  if (!state.config) return;
  const duration = Number(els.duration.value);
  const resolution = els.resolution.value;
  const mode = currentMode();
  const modeInfo = state.config.modes[mode];
  const min = Number(els.duration.min), max = Number(els.duration.max);
  els.durationOut.textContent = `${duration} s`;
  els.duration.style.setProperty('--fill', `${((duration - min) / (max - min)) * 100}%`);

  const plan = planFor(duration, resolution);
  let hint = `${plan.segments} segment${plan.segments > 1 ? 's' : ''} · ${resolution} at ${plan.fps} fps`;
  const perSegment = state.config.estimates[`${resolution}/${mode}`];
  if (perSegment) hint += ` · about ${fmtApprox(perSegment * plan.segments)} on this GPU`;
  els.durationHint.textContent = hint;
  els.modeHint.textContent = modeInfo.hint;
  els.resolutionHint.textContent = resolution === '720p'
    ? '720p is generated at 480p, then every segment is refined to 720p at 30 fps. Noticeably slower.'
    : 'The model’s native resolution and frame rate (15 fps).';
  els.advSummary.textContent = `${duration} s · ${resolution} · ${modeInfo.label}`;
}

function saveDraft() {
  const draft = {
    prompt: els.prompt.value, negative: els.negative.value, duration: els.duration.value,
    resolution: els.resolution.value, mode: currentMode(), steps: els.steps.value,
    seed: els.seed.value, seedRandom: els.seedRandom.checked, advancedOpen: els.advanced.open,
  };
  storage((s) => s.setItem(DRAFT_KEY, JSON.stringify(draft)));
}

function restoreDraft() {
  const draft = storage((s) => JSON.parse(s.getItem(DRAFT_KEY) || 'null'));
  if (!draft) return;
  if (typeof draft.prompt === 'string') els.prompt.value = draft.prompt;
  if (typeof draft.negative === 'string') els.negative.value = draft.negative;
  if (draft.duration) els.duration.value = draft.duration;
  if ([...els.resolution.options].some((o) => o.value === draft.resolution)) els.resolution.value = draft.resolution;
  setMode(draft.mode, false);
  if (draft.steps) { els.steps.value = draft.steps; setMode(currentMode(), false); }
  els.seedRandom.checked = draft.seedRandom !== false;
  els.seed.value = draft.seed || '';
  els.advanced.open = !!draft.advancedOpen;
  syncSeed();
}

function initForm(config) {
  els.resolution.innerHTML = '';
  for (const r of config.resolutions) {
    const option = document.createElement('option');
    option.value = r;
    option.textContent = r === '720p' ? '720p · 30 fps (refined)' : '480p · 15 fps';
    els.resolution.append(option);
  }
  els.resolution.value = config.default_resolution;
  els.duration.min = config.duration.min;
  els.duration.max = config.duration.max;
  els.duration.value = config.duration.default;
  els.negative.value = config.default_negative_prompt;
  setMode(config.default_mode, true);
  restoreDraft();
  syncSeed();
  updateFormHints();
}

function showPreview(url, meta) {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = url.startsWith('blob:') ? url : null;
  els.preview.src = url;
  els.preview.hidden = false;
  els.pickerEmpty.hidden = true;
  els.pickerChange.hidden = false;
  els.picker.classList.add('has-image');
  els.imageMeta.textContent = meta;
}

function clearPicker() {
  els.image.value = '';
  state.file = null;
}

els.image.addEventListener('change', () => {
  const file = els.image.files && els.image.files[0];
  if (!file) return;
  const maxMb = state.config ? state.config.max_upload_mb : 25;
  if (file.type && !ALLOWED_TYPES.includes(file.type)) {
    clearPicker();
    showFormError(`${file.type.replace('image/', '').toUpperCase()} files are not supported. Choose a JPG, PNG or WebP photo.`);
    return;
  }
  if (file.size > maxMb * 1048576) {
    clearPicker();
    showFormError(`That photo is ${(file.size / 1048576).toFixed(1)} MB. The limit is ${maxMb} MB.`);
    return;
  }
  hideFormError();
  state.file = file;
  state.sourceJob = null;
  showPreview(URL.createObjectURL(file), `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`);
});

for (const input of [els.prompt, els.negative, els.steps, els.seed]) input.addEventListener('input', saveDraft);
els.duration.addEventListener('input', () => { updateFormHints(); saveDraft(); });
els.resolution.addEventListener('change', () => { updateFormHints(); saveDraft(); });
els.advanced.addEventListener('toggle', saveDraft);
els.seedRandom.addEventListener('change', () => { syncSeed(); saveDraft(); });
document.querySelectorAll('input[name="mode"]').forEach((radio) => radio.addEventListener('change', () => {
  setMode(currentMode(), true);
  updateFormHints();
  saveDraft();
}));
els.negativeReset.addEventListener('click', () => {
  if (state.config) els.negative.value = state.config.default_negative_prompt;
  saveDraft();
});

function setSubmitting(on, label) {
  state.submitting = on;
  els.submit.disabled = on;
  els.submit.textContent = on ? label : 'Generate';
}

function upload(formData) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/jobs');
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        els.submit.textContent = pct < 100 ? `Uploading… ${pct}%` : 'Preparing photo…';
      }
    };
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch (_) { /* not JSON */ }
      if (xhr.status === 401) { location.href = '/login'; return; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new Error(errorText(data, xhr.status)));
    };
    xhr.onerror = () => reject(new Error('Network error. Check your connection and try again.'));
    xhr.send(formData);
  });
}

els.form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (state.submitting) return;
  hideFormError();
  if (!state.file && !state.sourceJob) { showFormError('Choose a photo first.'); return; }
  const prompt = els.prompt.value.trim();
  if (!prompt) { showFormError('Describe what should happen in the video.'); els.prompt.focus(); return; }

  const data = new FormData();
  if (state.file) data.append('image', state.file, state.file.name || 'photo.jpg');
  else data.append('source_job', state.sourceJob);
  data.append('prompt', prompt);
  data.append('negative_prompt', els.negative.value.trim());
  data.append('duration', els.duration.value);
  data.append('resolution', els.resolution.value);
  data.append('mode', currentMode());
  data.append('steps', els.steps.value);
  if (!els.seedRandom.checked && els.seed.value.trim()) data.append('seed', els.seed.value.trim());

  setSubmitting(true, 'Uploading…');
  try {
    const result = await upload(data);
    upsertJob(result.job);
    render();
    toast(`Queued. ${result.image_note}`);
    const card = state.cards.get(result.id);
    if (card) card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    schedulePoll(500);
  } catch (err) {
    showFormError(err.message);
  } finally {
    setSubmitting(false);
  }
});

// ---------- job cards ----------

function upsertJob(job) {
  const index = state.jobs.findIndex((j) => j.id === job.id);
  if (index >= 0) state.jobs[index] = job;
  else state.jobs.unshift(job);
}

function metaLine(job) {
  const mode = state.config && state.config.modes[job.mode] ? state.config.modes[job.mode].label : job.mode;
  return `${job.duration_s} s · ${job.resolution} · ${mode} · ${job.steps} steps · seed ${job.seed}`;
}

function doneNote(job) {
  const parts = [];
  if (job.started_at && job.finished_at) parts.push(`Took ${fmtClock(job.finished_at - job.started_at)}`);
  if (job.video) parts.push(`${job.video.seconds} s · ${job.video.width}×${job.video.height} · ${Math.round(job.video.fps)} fps`);
  if (job.stats && job.stats.peak_vram_gb) parts.push(`peak ${job.stats.peak_vram_gb} GB VRAM`);
  return parts.join(' · ');
}

function progressLabel(job) {
  const p = job.progress || {};
  if (job.status === 'queued') {
    if (state.model && state.model.state === 'loading') return 'Waiting for the model to load';
    return job.queue_position > 1 ? `Waiting · #${job.queue_position} in line` : 'Waiting · next in line';
  }
  if (p.phase === 'encode') return 'Encoding video…';
  if (p.phase === 'start' || !p.segment) return 'Starting…';
  let text = `${p.phase === 'refine' ? 'Refining 720p · ' : ''}Segment ${p.segment} of ${p.segments}`;
  if (p.steps) text += ` · step ${p.step}/${p.steps}`;
  return text;
}

function updateElapsed(card, job) {
  const el = $('.elapsed', card);
  if (!el) return;
  if (job.status === 'running' && job.started_at) {
    const elapsed = now() - job.started_at;
    const pct = (job.progress && job.progress.percent) || 0;
    let text = fmtClock(elapsed);
    if (pct >= 3 && pct < 100) text += ` · ~${fmtApprox((elapsed * (100 - pct)) / pct)} left`;
    el.textContent = text;
  } else if (job.status === 'queued') {
    el.textContent = `waiting ${fmtClock(now() - job.queued_at)}`;
  } else {
    el.textContent = '';
  }
}

function button(label, icon, onClick, extraClass = '') {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = `btn ${extraClass}`.trim();
  b.innerHTML = `${icon}<span></span>`;
  b.lastChild.textContent = label;
  b.addEventListener('click', onClick);
  return b;
}

function buildCard(job) {
  const card = els.template.content.firstElementChild.cloneNode(true);
  card.dataset.id = job.id;
  card.classList.add(job.status);
  $('.thumb', card).src = job.thumb_url;
  $('.job-prompt', card).textContent = job.prompt;
  $('.job-meta', card).textContent = metaLine(job);
  const badge = $('.badge', card);
  badge.textContent = STATUS_LABEL[job.status] || job.status;
  badge.classList.add(job.status);

  $('.job-progress', card).hidden = !isActive(job);
  const video = $('.job-video', card);
  if (job.status === 'done' && job.video_url) {
    if (job.video) video.style.aspectRatio = `${job.video.width} / ${job.video.height}`;   // no layout jump
    video.poster = job.input_url;
    video.src = job.video_url;
    video.addEventListener('error', () => {
      $('.job-note', card).textContent = 'This browser cannot play the video inline. Use Download to save it and play it in another app.';
    });
  } else {
    video.remove();
  }
  $('.job-error', card).textContent = job.status === 'failed' ? job.error || 'Failed.' : '';
  $('.job-note', card).textContent = job.status === 'done' ? doneNote(job) : (job.image_info && job.image_info.note) || '';

  const actions = $('.job-actions', card);
  if (job.status === 'done') {
    const download = document.createElement('a');
    download.className = 'btn';
    download.href = job.download_url;
    download.setAttribute('download', `longcat_${job.id}.mp4`);
    download.innerHTML = `${ICON.download}<span>Download</span>`;
    const loop = button('Loop', ICON.loop, () => {
      const v = $('.job-video', card);
      v.loop = !v.loop;
      loop.setAttribute('aria-pressed', String(v.loop));
      if (v.loop && v.paused) v.play().catch(() => {});
    });
    loop.setAttribute('aria-pressed', 'false');
    actions.append(download, loop, button('Reuse', ICON.reuse, () => reuse(job)),
      button('Delete', ICON.trash, () => removeJob(job), 'danger'));
  } else if (job.status === 'failed') {
    actions.append(button('Retry', ICON.retry, () => retry(job)), button('Reuse', ICON.reuse, () => reuse(job)),
      button('Delete', ICON.trash, () => removeJob(job), 'danger'));
  } else {
    actions.append(button('Cancel', ICON.cancel, () => removeJob(job), 'danger'));
  }
  return card;
}

function updateCard(card, job) {
  if (isActive(job)) {
    const pct = job.status === 'running' ? (job.progress && job.progress.percent) || 0 : 0;
    $('.bar > span', card).style.width = `${pct}%`;
    $('.progress-label', card).textContent = progressLabel(job);
  }
  updateElapsed(card, job);
}

function render() {
  const jobs = [...state.jobs].sort((a, b) => b.created_at - a.created_at);
  const seen = new Set();
  jobs.forEach((job, index) => {
    seen.add(job.id);
    let card = state.cards.get(job.id);
    // rebuild only when the card's shape changes, so a playing video is never reset
    const key = `${job.status}|${job.output || ''}|${job.error || ''}`;
    if (!card || card.dataset.key !== key) {
      const fresh = buildCard(job);
      fresh.dataset.key = key;
      if (card) card.replaceWith(fresh);
      card = fresh;
      state.cards.set(job.id, card);
    }
    updateCard(card, job);
    if (els.jobs.children[index] !== card) els.jobs.insertBefore(card, els.jobs.children[index] || null);
  });
  for (const [id, card] of state.cards) {
    if (!seen.has(id)) { card.remove(); state.cards.delete(id); }
  }
  els.jobsEmpty.hidden = jobs.length > 0;
  const active = jobs.filter(isActive).length;
  els.jobsCount.textContent = jobs.length
    ? `${jobs.length} job${jobs.length > 1 ? 's' : ''}${active ? ` · ${active} active` : ''}` : '';
}

function reuse(job) {
  els.prompt.value = job.prompt;
  els.negative.value = job.negative_prompt;
  els.duration.value = job.duration_s;
  if ([...els.resolution.options].some((o) => o.value === job.resolution)) els.resolution.value = job.resolution;
  setMode(job.mode, false);
  els.steps.value = job.steps;
  els.seedRandom.checked = !!job.seed_random;
  els.seed.value = String(job.seed);
  syncSeed();
  clearPicker();
  state.sourceJob = job.id;
  showPreview(job.input_url, 'Using the photo from that job. Tap to choose a different one.');
  hideFormError();
  updateFormHints();
  saveDraft();
  els.form.scrollIntoView({ behavior: 'smooth', block: 'start' });
  toast(job.seed_random ? 'Settings copied. Turn off Random to reuse the exact seed.' : 'Settings copied, including the seed.');
}

async function retry(job) {
  try {
    upsertJob(await api(`/api/jobs/${job.id}/retry`, { method: 'POST' }));
    render();
    schedulePoll(500);
  } catch (err) {
    toast(err.message, true);
  }
}

async function removeJob(job) {
  const question = isActive(job) ? 'Cancel this job and delete it?' : 'Delete this video and its files?';
  if (!window.confirm(question)) return;
  try {
    await api(`/api/jobs/${job.id}`, { method: 'DELETE' });
    state.jobs = state.jobs.filter((j) => j.id !== job.id);
    if (state.sourceJob === job.id) state.sourceJob = null;
    render();
    manageStream();
  } catch (err) {
    toast(err.message, true);
  }
}

// ---------- live updates: SSE for the running job, polling for everything ----------

function closeStream() {
  if (state.stream) state.stream.close();
  state.stream = null;
  state.streamJob = null;
}

function manageStream() {
  const running = state.jobs.find((j) => j.status === 'running');
  const target = running ? running.id : null;
  if (state.streamJob === target) return;
  closeStream();
  if (!target || !window.EventSource || Date.now() < state.streamPausedUntil) return;
  const stream = new EventSource(`/api/jobs/${target}/events`);
  state.stream = stream;
  state.streamJob = target;
  stream.onmessage = (event) => {
    const data = JSON.parse(event.data);
    syncClock(data.now);
    upsertJob(data.job);
    render();
    if (data.job.status !== 'running') { closeStream(); refreshJobs(); }
  };
  stream.addEventListener('deleted', () => { closeStream(); refreshJobs(); });
  stream.onerror = () => {
    closeStream();
    state.streamPausedUntil = Date.now() + 30000;   // polling covers us meanwhile
  };
}

function schedulePoll(ms) {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(refreshJobs, ms);
}

async function refreshJobs() {
  clearTimeout(state.pollTimer);
  try {
    const data = await api('/api/jobs');
    syncClock(data.now);
    state.jobs = data.jobs;
    render();
    manageStream();
  } catch (err) {
    console.warn('jobs refresh failed', err);
  }
  schedulePoll(state.jobs.some(isActive) ? 2000 : 10000);
}

function renderStatus(status) {
  const model = status.model;
  state.model = model;
  const kind = model.state === 'ready' ? 'ready' : model.state === 'error' ? 'error' : 'loading';
  els.status.className = `status ${kind}`;
  els.statusModel.textContent = kind === 'ready' ? 'Model ready'
    : kind === 'error' ? 'Model failed to load'
    : `Loading model${model.loading_for != null ? ` ${fmtClock(model.loading_for)}` : ''}`;
  els.status.title = model.message || '';
  els.statusDetail.hidden = kind !== 'error';
  els.statusDetail.textContent = kind === 'error' ? `${model.error || ''} See server.log for details.` : '';
  const gpu = status.gpu;
  els.statusGpu.textContent = gpu
    ? `${gpu.used_gb.toFixed(1)} / ${gpu.total_gb.toFixed(1)} GB · ${gpu.name}`
    : 'No NVIDIA GPU detected';
}

async function refreshStatus() {
  clearTimeout(state.statusTimer);
  try {
    const status = await api('/api/status');
    syncClock(status.now);
    const first = !state.config;
    state.config = status.config;
    if (first) initForm(status.config);
    else updateFormHints();
    renderStatus(status);
  } catch (err) {
    els.status.className = 'status error';
    els.statusModel.textContent = 'Server unreachable';
    els.statusGpu.textContent = '';
  }
  const loading = state.model && state.model.state === 'loading';
  state.statusTimer = setTimeout(refreshStatus, loading ? 2000 : 5000);
}

setInterval(() => {
  for (const job of state.jobs) {
    if (!isActive(job)) continue;
    const card = state.cards.get(job.id);
    if (card) updateElapsed(card, job);
  }
}, 1000);

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') { refreshStatus(); refreshJobs(); }
});

els.logout.addEventListener('click', async () => {
  try { await api('/api/logout', { method: 'POST' }); } catch (_) { /* ignore */ }
  location.href = '/login';
});

refreshStatus();
refreshJobs();
