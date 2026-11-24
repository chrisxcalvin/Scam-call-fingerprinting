"use strict";
// ScamShield Voice UI (Phase 3): upload / sample -> POST /api/analyze -> render AnalysisResult.
const COLORS = {safe: "#2ecc71", suspicious: "#f1c40f", high: "#e67e22", scam: "#e74c3c"};
const ICONS = {safe: "✔", suspicious: "?", high: "▲", scam: "⛔"};
const WARN_TYPES = ["digital_arrest", "kyc_otp", "lottery_prize", "relative_emergency", "fake_courier", "tech_support"];
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
let current = null;

function setStatus(t, ok = true) { $("status-text").textContent = t; $("status-text").style.color = ok ? "" : "#e74c3c"; }

// ---------- drawing ----------
function drawWave(ctx, wf, vad, dur) {
  const {width: W, height: H} = ctx.canvas;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "rgba(46,204,113,.18)";
  for (const [a, b] of vad) ctx.fillRect(a / dur * W, 0, (b - a) / dur * W, H);
  ctx.strokeStyle = "#7fb8ff"; ctx.beginPath();
  wf.forEach((v, i) => { const x = i / Math.max(1, wf.length - 1) * W, y = H / 2 - v * H / 2 * 0.95; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
}

const STOPS = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
function inferno(v) { // inferno-like piecewise-linear colour map, v in 0..1
  const p = Math.min(0.999, Math.max(0, v)) * (STOPS.length - 1), i = Math.floor(p), f = p - i;
  return STOPS[i].map((c, k) => c + (STOPS[i + 1][k] - c) * f);
}

function drawSpec(ctx, viz, match) {
  const [F, T] = viz.spec_shape, bin = atob(viz.spectrogram_b64), img = ctx.createImageData(T, F);
  for (let f = 0; f < F; f++) for (let t = 0; t < T; t++) {
    const [r, g, b] = inferno(bin.charCodeAt(f * T + t) / 255), o = ((F - 1 - f) * T + t) * 4; // low freq at bottom
    img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
  }
  const off = document.createElement("canvas"); off.width = T; off.height = F;
  off.getContext("2d").putImageData(img, 0, 0);
  const {width: W, height: H} = ctx.canvas;
  ctx.imageSmoothingEnabled = false; ctx.drawImage(off, 0, 0, W, H);
  ctx.fillStyle = "rgba(90,190,255,.85)";            // fp_peaks are (time_col, freq_bin)
  for (const [t, f] of viz.fp_peaks || []) ctx.fillRect(t / T * W - 1, H - f / F * H - 1, 2.5, 2.5);
  for (const fr of [300, 3400]) {                     // telephone band used for peaks
    const y = H - fr / viz.spec_freqs_hz[1] * H; ctx.strokeStyle = "rgba(255,255,255,.25)"; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); ctx.setLineDash([]);
  }
  if (match) {
    const [a, b] = match.query_span_s, d = viz.spec_duration_s;
    ctx.strokeStyle = "#2ecc71"; ctx.lineWidth = 3; ctx.strokeRect(a / d * W, 2, Math.max(4, (b - a) / d * W), H - 4); ctx.lineWidth = 1;
    ctx.fillStyle = "#2ecc71"; ctx.font = "bold 13px system-ui"; ctx.fillText(`fingerprint match: ${match.script_id}`, a / d * W + 6, 16);
  }
}

function drawGauge(ctx, score, level) {
  const {width: W, height: H} = ctx.canvas, cx = W / 2, cy = H - 20, r = 110;
  ctx.clearRect(0, 0, W, H); ctx.lineWidth = 22; ctx.lineCap = "round";
  ctx.strokeStyle = "#2a3142"; ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI); ctx.stroke();
  ctx.strokeStyle = COLORS[level]; ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, Math.PI + Math.PI * Math.max(0.005, score / 100)); ctx.stroke();
  ctx.fillStyle = "#eef1f7"; ctx.font = "bold 44px system-ui"; ctx.textAlign = "center";
  ctx.fillText(Math.round(score), cx, cy - 10); ctx.font = "14px system-ui"; ctx.fillStyle = "#8d97ab"; ctx.fillText("/ 100", cx, cy + 12);
}

function animateGauge(to, level) {
  const from = current && current._shown || 0, t0 = performance.now(), ctx = $("cv-gauge").getContext("2d");
  const step = (now) => { const k = Math.min(1, (now - t0) / 300), v = from + (to - from) * k; drawGauge(ctx, v, level); if (k < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}

function drawTimeline(ctx, tl, dur) {
  const {width: W, height: H} = ctx.canvas; ctx.clearRect(0, 0, W, H);
  for (const [lo, hi, c] of [[0, 30, "safe"], [30, 60, "suspicious"], [60, 85, "high"], [85, 100, "scam"]]) {
    ctx.fillStyle = COLORS[c] + "26"; ctx.fillRect(0, H - hi / 100 * H, W, (hi - lo) / 100 * H);
  }
  const X = (t) => t / dur * (W - 10) + 5, Y = (s) => H - 4 - s / 100 * (H - 8);
  ctx.strokeStyle = "#eef1f7"; ctx.lineWidth = 2; ctx.beginPath();
  tl.forEach((s, i) => (i ? ctx.lineTo(X(s.t_s), Y(s.score)) : ctx.moveTo(X(s.t_s), Y(s.score)))); ctx.stroke(); ctx.lineWidth = 1;
  let prev = "safe";
  for (const s of tl) { if (s.level !== prev) { ctx.fillStyle = COLORS[s.level]; ctx.beginPath(); ctx.arc(X(s.t_s), Y(s.score), 6, 0, 7); ctx.fill(); } prev = s.level; }
  ctx.fillStyle = "#8d97ab"; ctx.font = "11px system-ui"; ctx.fillText("0 s", 4, H - 6); ctx.fillText(`${dur.toFixed(0)} s`, W - 30, H - 6);
}

function drawSeries(ctx, ys, color, centre = 0) {
  const {width: W, height: H} = ctx.canvas; ctx.clearRect(0, 0, W, H);
  const m = Math.max(1e-6, ...ys.map(Math.abs)); ctx.strokeStyle = color; ctx.beginPath();
  ys.forEach((v, i) => { const x = i / Math.max(1, ys.length - 1) * W, y = centre ? H / 2 - v / m * H / 2 : H - v / m * H; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
}

function drawOffsetHist(ctx, counts, meta) {
  const {width: W, height: H} = ctx.canvas; ctx.clearRect(0, 0, W, H);
  const m = Math.max(...counts), bw = W / counts.length, peak = counts.indexOf(m);
  counts.forEach((c, i) => { ctx.fillStyle = i === peak ? "#e74c3c" : "#7f8c8d"; ctx.fillRect(i * bw, H - 16 - c / m * (H - 26), Math.max(1, bw), c / m * (H - 26)); });
  ctx.fillStyle = "#8d97ab"; ctx.font = "11px system-ui";
  const [lo, bs] = meta; ctx.fillText(`${lo.toFixed(1)} s`, 2, H - 3); ctx.fillText(`${(lo + counts.length * bs).toFixed(1)} s`, W - 40, H - 3);
  ctx.fillStyle = "#e74c3c"; ctx.fillText(`peak ${m} votes at ${(lo + (peak + 0.5) * bs).toFixed(2)} s`, Math.min(W - 160, peak * bw + 6), 14);
}

// ---------- panels ----------
function bars(el, rows, max = 1, fmt = (v) => v.toFixed(2)) {
  el.innerHTML = rows.map(([label, v, color]) => `<div class="bar"><span title="${esc(label)}">${esc(label)}</span><span class="track"><span class="fill" style="width:${Math.min(100, v / max * 100)}%;${color ? "background:" + color : ""}"></span></span><span>${fmt(v)}</span></div>`).join("");
}

function renderTranscript(segs) {
  if (!segs.length) { $("transcript").innerHTML = '<span class="muted">No speech recognised.</span>'; return; }
  $("transcript").innerHTML = segs.map((s) => {
    let html = esc(s.text);
    for (const f of s.flags) {
      const tip = f.category + (f.negated ? " — safe context (negated)" : ` (weight ${f.weight})`);
      const re = new RegExp(esc(f.phrase).replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+"), "i");
      html = html.replace(re, (m) => `<mark class="${f.negated ? "negated" : ""}" title="${esc(tip)}">${m}</mark>`);
    }
    return `<p class="seg"><span class="t">[${s.start_s.toFixed(1)}s]</span>${html}</p>`;
  }).join("");
}

const INFO = {
  fp: "Spectral-peak hashes of this audio line up at one time offset with a known scam recording.",
  script: "Sentence embeddings of the transcript compared with known scam scripts (cosine similarity).",
  voice: "Pitch (YIN), speaking rate and pauses. Shown for context; not used in the score yet.",
  flags: "Red-flag phrases from a lexicon; phrases cancelled by safe context are struck through.",
};

function renderEvidence(r) {
  const cards = [], fpHit = r.fingerprint.length && r.final.components.fingerprint > 0;
  if (fpHit) { const m = r.fingerprint[0]; cards.push(`<div class="card"><b>Fingerprint</b><span class="info" title="${INFO.fp}">ⓘ</span> match: ${esc(m.script_id)}<br>aligned ${m.aligned_hashes} / ${m.query_hashes} hashes, offset ${m.offset_s.toFixed(1)} s</div>`); }
  for (const m of r.script.slice(0, 1)) cards.push(`<div class="card"><b>Script</b><span class="info" title="${INFO.script}">ⓘ</span> similarity ${m.similarity.toFixed(2)} → “${esc(m.scam_type)}”<br><small class="muted">matched: “${esc(m.matched_text.slice(0, 90))}…”</small></div>`);
  const active = r.red_flags.filter((f) => !f.negated), neg = r.red_flags.filter((f) => f.negated);
  if (r.red_flags.length) cards.push(`<div class="card"><b>Red flags</b><span class="info" title="${INFO.flags}">ⓘ</span> ${active.length} active${neg.length ? `, ${neg.length} negated (safe context)` : ""}<br><small class="muted">${esc([...new Set(active.map((f) => f.category))].join(", ") || "none active")}</small></div>`);
  if (r.voice && r.voice.pitch_mean_hz != null) { const v = r.voice; cards.push(`<div class="card"><b>Voice</b><span class="info" title="${INFO.voice}">ⓘ</span> pitch ${Math.round(v.pitch_mean_hz)}±${Math.round(v.pitch_std_hz)} Hz · ${v.speech_rate_wps != null ? v.speech_rate_wps.toFixed(1) + " w/s · " : ""}pause ${Math.round(v.pause_ratio * 100)}%</div>`); }
  $("evidence").innerHTML = cards.join("") || '<span class="muted">No evidence.</span>';
}

function renderPipeline(r) {
  const st = r.viz.stages || {};
  if (st.fp_offset_hist && st.fp_offset_hist.length) {
    drawOffsetHist($("cv-offset").getContext("2d"), st.fp_offset_hist, st.fp_offset_hist_meta);
    const m = r.fingerprint[0], hit = r.final.components.fingerprint > 0;
    $("offset-info").textContent = `Closest library clip: ${m.clip_id}. ${m.aligned_hashes} of ${m.query_hashes} query hashes aligned → ${hit ? "MATCH (above threshold)" : "no match (below threshold: flat, no clear spike)"}.`;
  } else { $("cv-offset").getContext("2d").clearRect(0, 0, 900, 200); $("offset-info").textContent = "No library clip shares enough hashes with this audio."; }
  const tm = Object.entries(r.timings_ms).filter(([k]) => k !== "total");
  bars($("timings"), tm.map(([k, v]) => [k, v]), Math.max(...tm.map(([, v]) => v)), (v) => v.toFixed(0));
  $("timings").insertAdjacentHTML("beforeend", `<p class="muted">total ${r.timings_ms.total.toFixed(0)} ms for ${r.duration_s.toFixed(1)} s of audio</p>`);
  const c = r.final.components;
  bars($("components"), [["fingerprint", c.fingerprint], ["script", c.script], ["red flags", c.redflags], ["voice (unused)", c.voice]]);
  $("components").insertAdjacentHTML("beforeend", `<p class="muted">score = 100·(1 − Π(1 − wᵢ·sᵢ)) = ${r.final.score.toFixed(1)}</p>`);
  bars($("sims"), r.script.map((m) => [m.script_id, m.similarity]));
  $("stages").innerHTML = ["raw", "preprocessed", "preemph"].filter((k) => st[k]).map((k) => `<div class="stage"><small class="muted">${k}</small><canvas id="st-${k}" width="900" height="50"></canvas></div>`).join("");
  for (const k of ["raw", "preprocessed", "preemph"]) if (st[k]) drawSeries($("st-" + k).getContext("2d"), st[k], "#7fb8ff", 1);
}

function warningKey(r) {
  const fpHit = r.fingerprint.length && r.final.components.fingerprint > 0;
  const t = fpHit ? r.fingerprint[0].scam_type : (r.script.length && r.final.components.script >= 0.5 ? r.script[0].scam_type : null);
  return WARN_TYPES.includes(t) ? `warn_${t}` : "warn_generic";
}

// ---------- agent ----------
let player = null;
function agentState(t) { $("agent-state").textContent = t ? `· ${t}…` : ""; }
function stopSpeaking() { if (player) { player.pause(); player = null; } if ("speechSynthesis" in window) speechSynthesis.cancel(); }
function speak(text) {
  if (!$("speak-on").checked || !("speechSynthesis" in window)) return;
  stopSpeaking(); const u = new SpeechSynthesisUtterance(text); u.onend = () => agentState(""); agentState("speaking"); speechSynthesis.speak(u);
}
function sayAudio(url, text) {                       // server TTS, falling back to browser speech
  if (!$("speak-on").checked) return;
  if (!url) { speak(text); return; }
  stopSpeaking(); player = new Audio(url); agentState("speaking");
  player.onended = () => { agentState(""); player = null; };
  player.onerror = () => speak(text);
  player.play().catch(() => speak(text));
}
function addBubble(kind, text) {
  const d = document.createElement("div"); d.className = `bubble ${kind}`; d.textContent = text;
  $("agent-log").appendChild(d); $("agent-log").scrollTop = 1e9; return d;
}
function enableAgent(on) { for (const id of ["btn-ptt", "ask-text", "btn-ask"]) $(id).disabled = !on; }

async function askAgent(question, audioBlob) {
  if (!current) return;
  stopSpeaking(); agentState("thinking"); enableAgent(false);
  const userB = addBubble("user", question || "🎤 …");
  try {
    let res;
    if (audioBlob) {
      const fd = new FormData(); fd.append("session_id", current.session_id); fd.append("file", audioBlob, "question.webm");
      res = await fetch("/api/agent/ask_audio", {method: "POST", body: fd});
    } else {
      res = await fetch("/api/agent/ask", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({session_id: current.session_id, question})});
    }
    const j = await res.json();
    if (!res.ok) throw new Error(j.error ? j.error.message : `HTTP ${res.status}`);
    if (j.transcript) userB.textContent = j.transcript;
    addBubble("agent", j.reply.text); agentState(""); sayAudio(j.reply.audio_url, j.reply.text);
  } catch (e) {
    addBubble("agent", e instanceof TypeError ? "I can't reach the server right now." : e.message); agentState("");
  } finally { enableAgent(true); }
}

// push-to-talk: record while the button is held (MediaRecorder -> webm/opus)
let rec = null, chunks = [];
async function pttStart(e) {
  e.preventDefault(); if (rec || $("btn-ptt").disabled) return;
  stopSpeaking();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true}});
    chunks = []; rec = new MediaRecorder(stream);
    rec.ondataavailable = (ev) => chunks.push(ev.data);
    rec.onstop = () => { stream.getTracks().forEach((t) => t.stop()); const b = new Blob(chunks, {type: rec.mimeType}); rec = null;
      $("btn-ptt").classList.remove("recording"); b.size > 2000 ? askAgent(null, b) : agentState(""); };
    rec.start(); $("btn-ptt").classList.add("recording"); agentState("listening");
  } catch { addBubble("agent", "Microphone permission was denied. Type your question instead."); }
}
function pttStop() { if (rec && rec.state === "recording") rec.stop(); }

function showAlert(r) {
  const lv = r.final.level, a = $("alert");
  if (lv !== "high" && lv !== "scam") { a.hidden = true; return; }
  a.hidden = false; a.style.background = COLORS[lv]; a.style.color = lv === "high" ? "#1a1000" : "#fff";
  const msg = `${ICONS[lv]} ${lv === "scam" ? "Likely scam" : "High risk"}: ${r.final.top_reasons[0] || "several warning signs"}. Do not share OTP/PIN or send money. Hang up and call the official number.`;
  $("alert-text").textContent = msg;
  addBubble("agent warn", "");                        // filled from the agent's warning text
  fetch(`/api/agent/warn/${r.session_id}`).then((x) => x.json()).then((w) => {
    $("agent-log").lastElementChild.textContent = w.text;
    sayAudio(w.audio_url, w.text);
  }).catch(() => { $("agent-log").lastElementChild.textContent = msg; speak(msg); });
}

function render(r, opts = {}) {
  current = Object.assign(r, {_shown: current && current.final ? current.final.score : 0});
  const v = r.viz, dur = r.duration_s, fpHit = r.fingerprint.length && r.final.components.fingerprint > 0;
  drawWave($("cv-wave").getContext("2d"), v.waveform, v.vad_segments, dur);
  drawSpec($("cv-spec").getContext("2d"), v, fpHit ? r.fingerprint[0] : null);
  animateGauge(r.final.score, r.final.level);
  $("level").textContent = `${ICONS[r.final.level]} ${r.final.level.toUpperCase()}`; $("level").style.color = COLORS[r.final.level];
  $("reasons").innerHTML = r.final.top_reasons.map((t) => `<li>${esc(t)}</li>`).join("") || '<li class="muted">No warning signs found.</li>';
  drawTimeline($("cv-timeline").getContext("2d"), r.risk_timeline, dur);
  if (!opts.keepAlert) $("agent-log").innerHTML = "";
  enableAgent(true);
  renderTranscript(r.transcript); renderEvidence(r); renderPipeline(r);
  if (!opts.keepAlert) showAlert(r);
  document.querySelectorAll("main .panel").forEach((p) => p.classList.remove("skeleton"));
  setStatus("ready");
}


// ---------- live mode (WebSocket /ws/live) ----------
const LIVE_VIEW_S = 20;                              // scrolling window shown while streaming
let live = null;                                     // {ws, ctx, kind, wave:[], spec:[], rows, t, timeline:[], ...}

function liveReset(kind, name) {
  stopSpeaking(); $("alert").hidden = true; $("agent-log").innerHTML = ""; $("transcript").innerHTML = "";
  $("evidence").innerHTML = '<span class="muted">Listening…</span>'; $("reasons").innerHTML = "";
  drawGauge($("cv-gauge").getContext("2d"), 0, "safe"); $("level").textContent = "✔ SAFE"; $("level").style.color = COLORS.safe;
  $("clip-info").textContent = `${name} · live`;
  current = null; enableAgent(false);
  live = {kind, name, wave: [], spec: [], rows: 257, t: 0, timeline: [], fp: null, sid: null, sentUntil: 0};
}

function liveDraw() {
  const L = live, tEnd = Math.max(L.t, LIVE_VIEW_S), t0 = tEnd - LIVE_VIEW_S;
  // waveform: 50 points / s
  const wctx = $("cv-wave").getContext("2d"), {width: W, height: H} = wctx.canvas;
  wctx.clearRect(0, 0, W, H); wctx.strokeStyle = "#7fb8ff"; wctx.beginPath();
  const startIdx = Math.max(0, Math.floor(t0 * 50));
  for (let i = startIdx; i < L.wave.length; i++) { const x = (i / 50 - t0) / LIVE_VIEW_S * W, y = H / 2 - L.wave[i] * H / 2 * 0.95; i === startIdx ? wctx.moveTo(x, y) : wctx.lineTo(x, y); }
  wctx.stroke();
  // spectrogram: one column per 32 ms
  const sctx = $("cv-spec").getContext("2d"), SW = sctx.canvas.width, SH = sctx.canvas.height, colS = 0.032;
  const c0 = Math.max(0, Math.floor(t0 / colS)), n = Math.ceil(LIVE_VIEW_S / colS), F = L.rows, img = sctx.createImageData(n, F);
  for (let c = 0; c < n; c++) {
    const col = L.spec[c0 + c]; if (!col) continue;
    for (let f = 0; f < F; f++) { const [r, g, b] = inferno(col[f] / 255), o = ((F - 1 - f) * n + c) * 4; img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255; }
  }
  const off = document.createElement("canvas"); off.width = n; off.height = F; off.getContext("2d").putImageData(img, 0, 0);
  sctx.fillStyle = "#000"; sctx.fillRect(0, 0, SW, SH); sctx.imageSmoothingEnabled = false; sctx.drawImage(off, 0, 0, SW, SH);
  if (L.fp) { const [a, b] = L.fp.query_span_s; sctx.strokeStyle = "#2ecc71"; sctx.lineWidth = 3; sctx.strokeRect((a - t0) / LIVE_VIEW_S * SW, 2, Math.max(4, (b - a) / LIVE_VIEW_S * SW), SH - 4); sctx.lineWidth = 1; }
  drawTimeline($("cv-timeline").getContext("2d"), L.timeline, tEnd);
}

function liveEvent(e) {
  const L = live; if (!L) return;
  if (e.type === "ready") { L.sid = e.session_id; setStatus(L.kind === "mic" ? "live (mic)" : "live (simulated)"); enableAgent(true); current = {session_id: e.session_id}; }
  else if (e.type === "viz") {
    L.t = e.t_s; L.wave.push(...e.waveform);
    const bin = atob(e.spec_cols_b64); L.rows = e.spec_rows;
    for (let c = 0; c < e.spec_cols; c++) { const col = new Uint8Array(e.spec_rows); for (let f = 0; f < e.spec_rows; f++) col[f] = bin.charCodeAt(c * e.spec_rows + f); L.spec.push(col); }
    liveDraw();
  } else if (e.type === "transcript") {
    if ($("transcript").querySelector(".muted")) $("transcript").innerHTML = "";
    $("transcript").insertAdjacentHTML("beforeend", `<p class="seg"><span class="t">[${e.segment.start_s.toFixed(1)}s]</span>${esc(e.segment.text)}</p>`);
    $("transcript").scrollTop = 1e9;
  } else if (e.type === "fp_match") {
    L.fp = e.match;
    $("evidence").innerHTML = `<div class="card"><b>Fingerprint</b><span class="info" title="${INFO.fp}">ⓘ</span> match: ${esc(e.match.script_id)}<br>aligned ${e.match.aligned_hashes} / ${e.match.query_hashes} hashes</div>`;
  } else if (e.type === "risk") {
    const st = e.state; L.timeline.push(st);
    drawGauge($("cv-gauge").getContext("2d"), st.score, st.level);
    $("level").textContent = `${ICONS[st.level]} ${st.level.toUpperCase()}`; $("level").style.color = COLORS[st.level];
    $("reasons").innerHTML = st.top_reasons.map((t) => `<li>${esc(t)}</li>`).join("") || '<li class="muted">No warning signs yet.</li>';
    liveDraw();
  } else if (e.type === "alert") {
    const a = $("alert"); a.hidden = false; a.style.background = COLORS[e.level]; a.style.color = e.level === "high" ? "#1a1000" : "#fff";
    $("alert-text").textContent = `${ICONS[e.level]} [${e.t_s.toFixed(0)} s] ${e.message}`;
    addBubble("agent warn", e.message); sayAudio(e.audio_url, e.message);
  } else if (e.type === "final") {
    const r = e.result; $("clip-info").textContent = `${L.name} · ${r.duration_s.toFixed(1)} s · live call ended`;
    const log = $("agent-log").innerHTML; live = null; render(r, {keepAlert: true}); $("agent-log").innerHTML = log;
    setLiveButton(false);
  } else if (e.type === "error") { setStatus("error", false); addBubble("agent", e.message); }
}

function openLive(kind, name) {
  liveReset(kind, name);
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live`);
  ws.binaryType = "arraybuffer"; live.ws = ws;
  ws.onmessage = (m) => liveEvent(JSON.parse(m.data));
  ws.onclose = () => { if (live && live.ws === ws) { setStatus("ready"); setLiveButton(false); live = null; } };
  ws.onerror = () => { setStatus("backend unreachable", false); };
  return new Promise((res, rej) => { ws.onopen = () => res(ws); setTimeout(() => rej(new Error("timeout")), 5000); });
}

function sendPCM(ws, f32) {                         // Float32 (16 kHz) -> PCM16 LE; muted while the agent talks
  if (ws.readyState !== 1) return;
  const pcm = new Int16Array(f32.length), mute = !!player || ("speechSynthesis" in window && speechSynthesis.speaking);
  if (!mute) for (let i = 0; i < f32.length; i++) pcm[i] = Math.max(-32768, Math.min(32767, Math.round(f32[i] * 32767)));
  ws.send(pcm.buffer);
}

function setLiveButton(on) { $("btn-live").classList.toggle("on", on); $("btn-live").textContent = on ? "■ Stop" : "● Live mic"; }

async function startMic() {
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true, channelCount: 1}}); }
  catch { setStatus("mic denied", false); $("alert").hidden = false; $("alert").style.background = "#444"; $("alert").style.color = "#fff";
    $("alert-text").textContent = "Microphone permission was denied. Use Upload or a sample instead."; return; }
  const ws = await openLive("mic", "Microphone").catch(() => null); if (!ws) { stream.getTracks().forEach((t) => t.stop()); return; }
  const ctx = new AudioContext({sampleRate: 16000}); await ctx.audioWorklet.addModule("pcm-worklet.js");
  const src = ctx.createMediaStreamSource(stream), node = new AudioWorkletNode(ctx, "pcm-forwarder");
  let buf = [];
  node.port.onmessage = (m) => { buf.push(...m.data); if (buf.length >= 1600) { sendPCM(ws, Float32Array.from(buf.splice(0, 1600))); } };
  src.connect(node);
  live.stopFn = () => { src.disconnect(); node.disconnect(); stream.getTracks().forEach((t) => t.stop()); ctx.close(); if (ws.readyState === 1) ws.send(JSON.stringify({type: "stop"})); setStatus("finalising"); };
  setLiveButton(true);
}

async function simulateLive(id, title) {
  const ctx = new AudioContext({sampleRate: 16000});
  const audio = await ctx.decodeAudioData(await (await fetch(`/api/samples/${id}`)).arrayBuffer());  // resampled to 16 kHz
  const ws = await openLive("sim", title).catch(() => null); if (!ws) { ctx.close(); return; }
  const data = audio.getChannelData(0), src = ctx.createBufferSource(); src.buffer = audio; src.connect(ctx.destination);
  ctx.resume().catch(() => {}); src.start();          // playback is best-effort (autoplay may keep ctx suspended)
  const p0 = performance.now();                       // streaming is paced by the wall clock, not the audio clock
  let sent = 0;
  const timer = setInterval(() => {                   // send what has "played" so far, in 100 ms frames
    const due = Math.min(data.length, Math.floor((performance.now() - p0) / 1000 * 16000));
    while (due - sent >= 1600 || (due === data.length && sent < data.length)) { const n = Math.min(1600, data.length - sent); sendPCM(ws, data.subarray(sent, sent + n)); sent += n; }
    if (sent >= data.length) { clearInterval(timer); if (ws.readyState === 1) ws.send(JSON.stringify({type: "stop"})); setStatus("finalising"); }
  }, 100);
  live.stopFn = () => { clearInterval(timer); try { src.stop(); } catch {} ctx.close(); if (ws.readyState === 1) ws.send(JSON.stringify({type: "stop"})); };
  setLiveButton(true);
}

// ---------- actions ----------
async function analyze(blob, name) {
  $("player").src = URL.createObjectURL(blob); $("player").hidden = false;
  $("clip-info").textContent = `${name} · analysing…`; $("alert").hidden = true;
  document.querySelectorAll("#tab-console .panel").forEach((p) => p.classList.add("skeleton"));
  setStatus("analyzing");
  const fd = new FormData(); fd.append("file", blob, name);
  try {
    const res = await fetch("/api/analyze", {method: "POST", body: fd});
    const j = await res.json();
    if (!res.ok) throw new Error(j.error ? j.error.message : `HTTP ${res.status}`);
    $("clip-info").textContent = `${name} · ${j.duration_s.toFixed(1)} s · analysed in ${(j.timings_ms.total / 1000).toFixed(1)} s`;
    render(j);
  } catch (e) {
    document.querySelectorAll("main .panel").forEach((p) => p.classList.remove("skeleton"));
    $("clip-info").textContent = `${name} · failed`;
    setStatus("error", false);
    $("alert").hidden = false; $("alert").style.background = "#444"; $("alert").style.color = "#fff";
    $("alert-text").textContent = e instanceof TypeError ? "Backend unreachable. Is the server running? Retry when it is." : `Could not analyse: ${e.message}`;
  }
}

async function playSample(id) {
  if (!id) return;
  const title = [...$("sample").options].find((o) => o.value === id).textContent;
  if (live && live.stopFn) live.stopFn();
  if ($("as-live").checked) { simulateLive(id, title).catch(() => setStatus("error", false)); return; }
  try { const r = await fetch(`/api/samples/${id}`); if (!r.ok) throw new Error(); await analyze(await r.blob(), title); }
  catch { setStatus("error", false); }
}

async function init() {
  try {
    const h = await (await fetch("/api/health")).json(), ml = h.models_loaded;
    setStatus(ml.index && ml.stt && ml.embed ? "ready" : "models missing", ml.index && ml.stt && ml.embed);
    for (const s of await (await fetch("/api/samples")).json()) $("sample").insertAdjacentHTML("beforeend", `<option value="${esc(s.id)}">${esc(s.title)}</option>`);
  } catch { setStatus("backend unreachable", false); }
}

function table(rows, cols) {
  if (!rows || !rows.length) return '<p class="muted">not run yet</p>';
  cols = cols || Object.keys(rows[0]);
  return `<table><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${rows.map((r) => `<tr>${cols.map((c) => `<td>${esc(r[c] ?? "")}</td>`).join("")}</tr>`).join("")}</table>`;
}
function renderSplit(name, s) {
  const l1 = s.layer1, l2 = s.layer2, f = s.fusion, cal = s.fusion_config || {};
  return `<h3>Split: ${esc(name)} <small class="muted">${s.n_clips} clips (${s.n_scam} scam, ${s.n_legit} legit) · augmentations: ${esc(s.augmentations.join(", "))}${s.quick ? " · QUICK" : ""}</small></h3>
  ${name === "dev" ? '<p class="note">⚠ Thresholds were calibrated on this same dev split, so dev numbers are optimistic. Honest numbers come from the test split (run once, at the end) and the holdout/human sets.</p>' : ""}
  <div class="results-grid">
   <div><h4>Layer 1: fingerprint <span class="muted">(threshold ${l1.threshold_aligned_hashes} aligned hashes)</span></h4>
     ${table(l1.by_augmentation)}
     <p>False-match rate on ${l1.n_negatives} legit / wrong-script negatives: <b>${l1.false_match_rate}</b> · same-script near-copies matched: ${l1.same_script_near_copy_match_rate} (n=${l1.n_same_script_near_copies}) · AUC ${l1.auc_aligned_hashes} · query p50 ${l1.query_ms_p50} ms / p95 ${l1.query_ms_p95} ms · index ${l1.index_hashes} hashes</p>
     <img src="/api/results/files/recall_by_aug_${name}.png" alt="recall by augmentation"></div>
   <div><h4>Layer 2: script + red flags</h4><p class="note">${esc(l2.note)}</p>
     <p>AUC script similarity <b>${l2.auc_script_sim}</b> · AUC red flags <b>${l2.auc_redflags}</b> · scam-type top-1 ${l2.scam_type_top1_accuracy}</p>
     ${table(Object.entries(l2.by_language).map(([k, v]) => ({language: k, ...v})))}
     <p class="muted">STT word error rate (English scripts): ${esc(Object.entries(l2.wer_en_by_aug).map(([k, v]) => `${k} ${v}`).join(" · "))}</p>
     <img src="/api/results/files/roc_layer2_${name}.png" alt="ROC"></div>
   <div><h4>Fusion <span class="muted">(flagged = level high or scam)</span></h4>
     ${table([{"": "actual scam", flagged: f.tp, "not flagged": f.fn}, {"": "actual legit", flagged: f.fp, "not flagged": f.tn}])}
     <p>precision <b>${f.precision}</b> · recall <b>${f.recall}</b> · FPR on legit <b>${f.fpr_legit}</b> · AUC ${f.auc}</p>
     <h4>Ablation</h4>${table(s.ablation)}
     <p class="muted">Missed scams: ${f.n_missed}${f.n_missed ? " (e.g. " + esc(f.missed_scams.slice(0, 3).map((m) => m.file).join(", ")) + ")" : ""} · false alarms: ${f.false_alarms.length}</p></div>
   <div><h4>Live (simulated, real-time path)</h4>${s.live ? `<p>Scam clips alerted <b>${s.live.scam_alerted}</b> (n=${s.live.n_scam}) · detection latency p50 <b>${s.live.latency_s_p50} s</b>, p95 ${s.live.latency_s_p95} s, max ${s.live.latency_s_max} s · within 5 s ${s.live.share_within_5s} · alerts before the first scam phrase (fingerprint) ${s.live.alerts_before_first_phrase}</p>
     ${s.live.words_only ? `<p>Words-only alerts (no fingerprint): n=${s.live.words_only.n}, alerted ${s.live.words_only.alerted}, latency p50 <b>${s.live.words_only.latency_s_p50} s</b> / p95 ${s.live.words_only.latency_s_p95} s, within 5 s ${s.live.words_only.share_within_5s} · fingerprint alerts: n=${s.live.fingerprint_alerts.n}, first alert at p50 ${s.live.fingerprint_alerts.first_alert_t_s_p50} s into the call</p>` : ""}
     <p>Legit clips alerted <b>${s.live.legit_alert_rate}</b> (n=${s.live.n_legit}) · max level on legit: ${esc(JSON.stringify(s.live.legit_max_levels))} · tick p50 ${s.live.tick_ms_p50} ms, max ${s.live.tick_ms_max} ms</p>
     <p class="muted">${esc(s.live.definition)} · versions: ${esc(s.live.augmentations.join(", "))}</p>` : '<p class="muted">not run yet</p>'}
     <h4>Latency per stage (upload path)</h4>${table(s.timings)}
     <p class="muted">Calibration: ${esc(cal.source || "")} · script range ${esc(JSON.stringify(cal.script_sim_range || ""))}</p></div>
  </div>`;
}
async function loadResults() {
  try { const j = await (await fetch("/api/results")).json();
    if (!j.splits || !Object.keys(j.splits).length) { $("results").innerHTML = "Not run yet — run <code>python -m scamshield.eval.run_eval</code>. No placeholder numbers are shown."; return; }
    $("results").classList.remove("muted");
    $("results").innerHTML = `<p class="muted">Generated ${esc(j.generated_at)} · STT ${esc(j.stt.model)} (${esc(j.stt.language)}) · commit ${esc(j.git_commit || "none")}</p>` +
      Object.entries(j.splits).map(([k, v]) => renderSplit(k, v)).join("<hr>");
  } catch { $("results").textContent = "Backend unreachable."; }
}

document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("active", x === b));
  for (const t of ["console", "pipeline", "results"]) $("tab-" + t).hidden = t !== b.dataset.tab;
  if (b.dataset.tab === "results") loadResults();
}));
$("btn-upload").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) analyze(f, f.name); e.target.value = ""; });
$("sample").addEventListener("change", (e) => { const v = e.target.value; e.target.value = ""; playSample(v); });
$("alert-close").addEventListener("click", () => { $("alert").hidden = true; });
document.addEventListener("dragover", (e) => { e.preventDefault(); document.body.classList.add("dragging"); });
document.addEventListener("dragleave", (e) => { if (!e.relatedTarget) document.body.classList.remove("dragging"); });
document.addEventListener("drop", (e) => { e.preventDefault(); document.body.classList.remove("dragging"); const f = e.dataTransfer.files[0]; if (f) analyze(f, f.name); });
$("btn-live").addEventListener("click", () => { if (live && live.stopFn) live.stopFn(); else startMic(); });
$("btn-ptt").addEventListener("pointerdown", pttStart);
for (const ev of ["pointerup", "pointerleave", "pointercancel"]) $("btn-ptt").addEventListener(ev, pttStop);
$("btn-ask").addEventListener("click", () => { const q = $("ask-text").value.trim(); if (q) { $("ask-text").value = ""; askAgent(q); } });
$("ask-text").addEventListener("keydown", (e) => { if (e.key === "Enter") $("btn-ask").click(); });
$("speak-on").addEventListener("change", (e) => { if (!e.target.checked) stopSpeaking(); });
init();
