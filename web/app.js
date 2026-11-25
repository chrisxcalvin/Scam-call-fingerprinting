"use strict";
/* ScamShield Voice UI
   Home -> (upload | sample | live) -> session view (verdict, evidence, timeline, transcript, audio, agent).
   Tabs: Call · Under the hood · Evaluation. No framework, no build step. */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const dpr = () => Math.min(2, window.devicePixelRatio || 1);
const pct = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(d).replace(/\.0+$/, "")}%`);
const nice = (t) => String(t).replace(/([a-z])_([a-z])/g, "$1 $2");
const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
const fmtT = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const params = new URLSearchParams(location.search);

const TYPE = {digital_arrest: "digital-arrest scam", kyc_otp: "KYC / OTP scam", lottery_prize: "lottery scam",
  relative_emergency: "relative-in-trouble scam", fake_courier: "fake-courier scam", tech_support: "tech-support scam"};
const LV = {
  safe: {label: "Looks safe", icon: "i-check"},
  suspicious: {label: "Be careful", icon: "i-help"},
  high: {label: "High risk", icon: "i-alert"},
  scam: {label: "Likely scam", icon: "i-octagon"},
};
const SAMPLE_HINT = {
  scam_noisy_gsm: "A replayed recording squeezed through a phone codec. The fingerprint recognises it instantly.",
  scam_snr10: "The same known recording buried in noise. Fingerprint and words both agree.",
  scam_other_voice: "A new voice reading the same script. No fingerprint, so it is caught by what is said.",
  scam_hinglish: "Hindi–English mix on a telephone line. A harder case for the language layer.",
  legit_bank_otp: "A genuine bank alert that mentions PIN and OTP. It stays safe because it says never to share them.",
  legit_otp_notice: "A real OTP message saying “do not share this code”. Safe context is understood.",
  legit_delivery: "An ordinary delivery call. Nothing suspicious here.",
};

let current = null;        // last full AnalysisResult
let live = null;           // active live session
let speakOn = true;
let player = null;         // agent audio element
let pipeResult = null;

/* ================= small UI helpers ================= */
function setStatus(text, kind = "ok") { $("status-text").textContent = text; $("status").dataset.kind = kind; }
function banner(text, kind = "alert") {
  const b = $("alert"); b.hidden = false; b.dataset.kind = kind; $("alert-text").textContent = text;
  b.querySelector("use").setAttribute("href", kind === "neutral" ? "#i-info" : "#i-alert");
}
function showSession(on) { $("home").hidden = on; $("session").hidden = !on; if (on) window.scrollTo({top: 0}); }

/* ================= canvas drawing ================= */
function fit(cv) {
  const r = cv.getBoundingClientRect();
  if (r.width < 2 || r.height < 2) return null;
  const k = dpr(), w = Math.round(r.width * k), h = Math.round(r.height * k);
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, w, h);
  return {ctx, W: w, H: h, k};
}
function rrect(ctx, x, y, w, h, r) { if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, w, h, r); ctx.fill(); } else ctx.fillRect(x, y, w, h); }
const levelColor = (lv) => css(`--${lv}`);

/* waveform as rounded bars; speech (VAD) in colour, the rest dimmed */
function drawWave(cv, pts, tOf, t0, t1, speechAt) {
  const f = fit(cv); if (!f) return;
  const {ctx, W, H, k} = f, bw = 3 * k, gap = 2 * k, step = bw + gap, nb = Math.max(1, Math.floor(W / step));
  const amp = new Float32Array(nb);
  for (let i = 0; i < pts.length; i++) {
    const t = tOf(i); if (t < t0 || t > t1) continue;
    const b = Math.min(nb - 1, Math.floor((t - t0) / (t1 - t0) * nb)), a = Math.abs(pts[i]);
    if (a > amp[b]) amp[b] = a;
  }
  let mx = 0.12; for (const a of amp) if (a > mx) mx = a;
  const on = css("--wave"), off = css("--faint"), x0 = (W - nb * step) / 2;
  for (let b = 0; b < nb; b++) {
    const h = Math.max(2 * k, amp[b] / mx * H * 0.88), speech = speechAt((b + 0.5) / nb * (t1 - t0) + t0);
    ctx.globalAlpha = speech ? 1 : 0.4; ctx.fillStyle = speech ? on : off;
    rrect(ctx, x0 + b * step, (H - h) / 2, bw, h, bw / 2);
  }
  ctx.globalAlpha = 1;
}

const STOPS = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
function inferno(v) {
  const p = Math.min(0.999, Math.max(0, v)) * (STOPS.length - 1), i = Math.floor(p), f = p - i;
  return STOPS[i].map((c, j) => c + (STOPS[i + 1][j] - c) * f);
}
function specImage(cols, F, getVal) {                 // cols x F values -> offscreen canvas (low frequency at the bottom)
  const img = new ImageData(cols, F), d = img.data;
  for (let f = 0; f < F; f++) for (let t = 0; t < cols; t++) {
    const [r, g, b] = inferno(getVal(f, t) / 255), o = ((F - 1 - f) * cols + t) * 4; d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
  }
  const off = document.createElement("canvas"); off.width = cols; off.height = F; off.getContext("2d").putImageData(img, 0, 0);
  return off;
}
function labelPill(ctx, text, x, y, k, color) {
  ctx.font = `600 ${12 * k}px ${css("--font")}`; const w = ctx.measureText(text).width + 16 * k, h = 22 * k;
  ctx.fillStyle = color; rrect(ctx, x, y, w, h, 6 * k); ctx.fillStyle = "#04130c"; ctx.textBaseline = "middle"; ctx.fillText(text, x + 8 * k, y + h / 2 + 1);
}
function drawSpec(cv, viz, match) {
  const f = fit(cv); if (!f) return;
  const {ctx, W, H, k} = f, [F, T] = viz.spec_shape, bin = atob(viz.spectrogram_b64);
  const off = specImage(T, F, (fr, t) => bin.charCodeAt(fr * T + t));
  ctx.imageSmoothingEnabled = true; ctx.drawImage(off, 0, 0, W, H);
  ctx.fillStyle = "rgba(120,200,255,.9)";
  for (const [t, fr] of viz.fp_peaks || []) ctx.fillRect(t / T * W - k, H - fr / F * H - k, 2.5 * k, 2.5 * k);
  ctx.setLineDash([5 * k, 5 * k]); ctx.strokeStyle = "rgba(255,255,255,.28)"; ctx.lineWidth = k;
  for (const hz of [300, 3400]) { const y = H - hz / viz.spec_freqs_hz[1] * H; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
  ctx.setLineDash([]);
  if (match) {
    const d = viz.spec_duration_s, a = match.query_span_s[0] / d * W, b = match.query_span_s[1] / d * W;
    ctx.strokeStyle = css("--safe"); ctx.lineWidth = 2.5 * k; ctx.strokeRect(a, 2 * k, Math.max(6 * k, b - a), H - 4 * k);
    labelPill(ctx, "Known scam recording", a + 6 * k, 8 * k, k, css("--safe"));
  }
}
function drawSpecLive(cv, L, t0, view, ref) {
  const f = fit(cv); if (!f) return;
  const {ctx, W, H, k} = f, colS = 0.032, n = Math.ceil(view / colS), c0 = Math.max(0, Math.floor(t0 / colS)), F = L.rows;
  const off = specImage(n, F, (fr, c) => { const col = L.spec[c0 + c]; return col ? col[fr] : 0; });
  ctx.fillStyle = "#05060a"; ctx.fillRect(0, 0, W, H); ctx.drawImage(off, 0, 0, W, H);
  if (L.fp) {
    const a = (L.fp.query_span_s[0] - t0) / view * W, b = (L.fp.query_span_s[1] - t0) / view * W;
    ctx.strokeStyle = css("--safe"); ctx.lineWidth = 2.5 * k; ctx.strokeRect(a, 2 * k, Math.max(6 * k, b - a), H - 4 * k);
    labelPill(ctx, "Known scam recording", Math.max(4 * k, a) + 6 * k, 8 * k, k, css("--safe"));
  }
}

function niceStep(dur) { for (const s of [5, 10, 15, 30, 60]) if (dur / s <= 8) return s; return 120; }
function drawTimeline(cv, tl, dur) {
  const f = fit(cv); if (!f) return;
  const {ctx, W, H, k} = f, L = 30 * k, B = 22 * k, T = 8 * k, R = 8 * k, pw = W - L - R, ph = H - T - B;
  const X = (t) => L + Math.min(1, t / dur) * pw, Y = (s) => T + (1 - s / 100) * ph;
  const bands = [[0, 30, "safe"], [30, 60, "suspicious"], [60, 85, "high"], [85, 100, "scam"]];
  ctx.font = `${11 * k}px ${css("--font")}`; ctx.textBaseline = "middle";
  for (const [lo, hi, lv] of bands) {
    ctx.globalAlpha = 0.09; ctx.fillStyle = levelColor(lv); ctx.fillRect(L, Y(hi), pw, Y(lo) - Y(hi)); ctx.globalAlpha = 1;
    ctx.fillStyle = css("--faint"); ctx.textAlign = "right"; ctx.fillText(String(lo), L - 6 * k, Y(lo));
  }
  ctx.fillText("100", L - 6 * k, Y(100));
  ctx.strokeStyle = css("--line"); ctx.lineWidth = k; ctx.textAlign = "center"; ctx.textBaseline = "top";
  const st = niceStep(dur);
  for (let t = 0; t <= dur + 0.01; t += st) { ctx.beginPath(); ctx.moveTo(X(t), T); ctx.lineTo(X(t), T + ph); ctx.stroke(); ctx.fillStyle = css("--faint"); ctx.fillText(fmtT(t), X(t), T + ph + 5 * k); }
  if (!tl.length) return;
  const last = tl[tl.length - 1], col = levelColor(last.level);
  ctx.beginPath(); tl.forEach((s, i) => (i ? ctx.lineTo(X(s.t_s), Y(s.score)) : ctx.moveTo(X(s.t_s), Y(s.score))));
  ctx.strokeStyle = col; ctx.lineWidth = 2.5 * k; ctx.lineJoin = "round"; ctx.stroke();
  ctx.lineTo(X(last.t_s), Y(0)); ctx.lineTo(X(tl[0].t_s), Y(0)); ctx.closePath();
  const g = ctx.createLinearGradient(0, T, 0, T + ph); g.addColorStop(0, col + "55"); g.addColorStop(1, col + "00"); ctx.fillStyle = g; ctx.fill();
  let prev = "safe";
  for (const s of tl) {
    if (s.level !== prev) { ctx.beginPath(); ctx.arc(X(s.t_s), Y(s.score), 5 * k, 0, 7); ctx.fillStyle = levelColor(s.level); ctx.fill(); ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2 * k; ctx.stroke(); }
    prev = s.level;
  }
  ctx.beginPath(); ctx.arc(X(last.t_s), Y(last.score), 4 * k, 0, 7); ctx.fillStyle = col; ctx.fill();
}

function drawSeries(cv, ys) {
  const f = fit(cv); if (!f) return;
  const {ctx, W, H, k} = f; const m = Math.max(1e-6, ...ys.map(Math.abs));
  ctx.strokeStyle = css("--wave"); ctx.lineWidth = 1.2 * k; ctx.beginPath();
  ys.forEach((v, i) => { const x = i / Math.max(1, ys.length - 1) * W, y = H / 2 - v / m * H * 0.45; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
}
function drawOffsetHist(cv, counts, meta) {
  const f = fit(cv); if (!f) return;
  const {ctx, W, H, k} = f, m = Math.max(...counts), bw = W / counts.length, peak = counts.indexOf(m), base = H - 20 * k;
  counts.forEach((c, i) => { ctx.fillStyle = i === peak ? css("--scam") : css("--faint"); ctx.globalAlpha = i === peak ? 1 : 0.6; ctx.fillRect(i * bw, base - c / m * (base - 22 * k), Math.max(1, bw * 0.9), c / m * (base - 22 * k)); });
  ctx.globalAlpha = 1; ctx.font = `${11 * k}px ${css("--font")}`; ctx.fillStyle = css("--faint"); ctx.textBaseline = "alphabetic";
  const [lo, bs] = meta; ctx.textAlign = "left"; ctx.fillText(`${lo.toFixed(1)} s`, 4 * k, H - 4 * k); ctx.textAlign = "right"; ctx.fillText(`${(lo + counts.length * bs).toFixed(1)} s`, W - 4 * k, H - 4 * k);
  ctx.fillStyle = css("--scam"); ctx.font = `600 ${12 * k}px ${css("--font")}`; ctx.textAlign = peak * bw > W - 190 * k ? "right" : "left";
  ctx.fillText(`peak: ${m} hashes agree at ${(lo + (peak + 0.5) * bs).toFixed(2)} s`, peak * bw > W - 190 * k ? peak * bw - 6 * k : peak * bw + 8 * k, 16 * k);
}

/* ================= verdict, tiles, transcript ================= */
let shownScore = 0, shownRisk = false;
function tweenScore(to) {
  const from = shownScore, t0 = performance.now();
  const step = (now) => { const p = Math.min(1, (now - t0) / 800); shownScore = from + (to - from) * (1 - Math.pow(1 - p, 3)); $("score").textContent = Math.round(shownScore); if (p < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}
function setVerdict({score, level, reasons, scamType, listening, label}) {
  const L = listening ? {label: label || "Listening", icon: "i-wave"} : LV[level], t = TYPE[scamType];
  $("verdict").dataset.level = listening && !shownRisk ? "idle" : level;
  $("ring-arc").style.strokeDashoffset = 326.7 * (1 - Math.max(0.012, score / 100));
  tweenScore(score);
  $("level").textContent = L.label; $("level-icon").setAttribute("href", `#${L.icon}`);
  const title = listening ? "Listening to the call…" : {
    safe: "No strong scam signs", suspicious: "Some warning signs",
    high: t ? `Looks like a ${t}` : "Looks like a scam", scam: t ? `Very likely a ${t}` : "Very likely a scam"}[level];
  $("verdict-title").textContent = title;
  $("verdict-sub").textContent = listening ? "The risk score updates every two seconds as the call is understood." : {
    safe: "Nothing here matches a known scam. If you are unsure, hang up and call the official number yourself.",
    suspicious: "A few things look off. Don't share codes, PINs or money until you have verified who is calling.",
    high: "Several strong scam signals are present. Don't act on anything the caller asks.",
    scam: "Strong, consistent scam signals. Don't act on anything the caller asks."}[level];
  $("reasons").innerHTML = (reasons || []).map((x, i) => `<li style="animation-delay:${i * 70}ms">${esc(nice(x))}</li>`).join("");
  $("actions").hidden = !(level === "high" || level === "scam");
}
function setTile(id, {val, sub, meter, tone}) {
  $("tv-" + id).textContent = val; $("ts-" + id).textContent = sub;
  $("tm-" + id).style.width = `${Math.round(Math.min(1, meter || 0) * 100)}%`;
  const t = $("tile-" + id); t.classList.remove("hot", "warm", "cool", "on"); if (tone) t.classList.add(tone, "on");
}
const toneOf = (v, hot = 0.6, warm = 0.25) => (v >= hot ? "hot" : v >= warm ? "warm" : null);
function tilesFromComponents(c, hit) {
  setTile("fp", hit ? {val: "Known recording", sub: "Audio matches a known scam recording.", meter: c.fingerprint, tone: "hot"} : {val: "No match", sub: "Not a recording we already know.", meter: 0, tone: null});
  setTile("script", {val: pct(c.script, 0), sub: "Strength of match to known scam scripts.", meter: c.script, tone: toneOf(c.script)});
  setTile("flags", {val: pct(c.redflags, 0), sub: "Strength of red-flag phrases heard so far.", meter: c.redflags, tone: toneOf(c.redflags)});
}
function tilesFromResult(r) {
  const c = r.final.components, hit = r.fingerprint.length && c.fingerprint > 0;
  if (hit) { const m = r.fingerprint[0]; setTile("fp", {val: "Known recording", sub: `${m.aligned_hashes.toLocaleString()} of ${m.query_hashes.toLocaleString()} fingerprint hashes align.`, meter: c.fingerprint, tone: "hot"}); }
  else setTile("fp", {val: "No match", sub: r.fingerprint.length ? `Only ${r.fingerprint[0].aligned_hashes} hashes align: not a known recording.` : "Not a recording we already know.", meter: 0, tone: null});
  const s = r.script[0];
  setTile("script", !s ? {val: "–", sub: "No speech to compare.", meter: 0}
    : c.script > 0 ? {val: pct(s.similarity, 0), sub: `Similarity to the closest known script (${TYPE[s.scam_type] || s.scam_type}).`, meter: c.script, tone: toneOf(c.script)}
    : {val: "No match", sub: `Closest script is only ${pct(s.similarity, 0)} similar: below the match threshold.`, meter: 0});
  const act = r.red_flags.filter((f) => !f.negated), neg = r.red_flags.length - act.length;
  setTile("flags", {val: `${act.length} active`, sub: act.length ? [...new Set(act.map((f) => f.category.replace("_", " ")))].join(", ") + (neg ? ` · ${neg} cleared by context` : "") : (neg ? `${neg} phrase${neg > 1 ? "s" : ""} cleared by safe context.` : "No red-flag phrases."), meter: c.redflags, tone: toneOf(c.redflags)});
  const v = r.voice;
  setTile("voice", v && v.pitch_mean_hz != null ? {val: `${Math.round(v.pitch_mean_hz)} Hz`, sub: `${v.speech_rate_wps != null ? v.speech_rate_wps.toFixed(1) + " words/s · " : ""}${Math.round(v.pause_ratio * 100)}% pauses · context only`, meter: 0} : {val: "–", sub: "Context only.", meter: 0});
}
function resetTiles() {
  for (const id of ["fp", "script", "flags", "voice"]) setTile(id, {val: "–", sub: id === "fp" ? "Waiting for audio" : id === "voice" ? "Context only" : "Waiting for speech", meter: 0});
}
function renderTranscript(segs, flags) {
  if (!segs.length) { $("transcript").innerHTML = '<p class="empty">No speech was recognised in this recording.</p>'; $("flag-legend").innerHTML = ""; return; }
  $("transcript").innerHTML = segs.map((s) => {
    let html = esc(s.text);
    for (const f of s.flags) {
      const tip = f.category.replace("_", " ") + (f.negated ? ": cleared by safe context" : "");
      const re = new RegExp(esc(f.phrase).replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+"), "i");
      html = html.replace(re, (m) => `<mark class="${f.negated ? "negated" : ""}" title="${esc(tip)}">${m}</mark>`);
    }
    return `<div class="seg ${s.flags.some((f) => !f.negated) ? "flagged" : ""}"><span class="t">${fmtT(s.start_s)}</span><p>${html}</p></div>`;
  }).join("");
  const cnt = {}; for (const f of flags) if (!f.negated) cnt[f.category] = (cnt[f.category] || 0) + 1;
  const neg = flags.filter((f) => f.negated).length;
  $("flag-legend").innerHTML = Object.entries(cnt).map(([c, n]) => `<span class="chip">${esc(c.replace("_", " "))} <b>${n}</b></span>`).join("") + (neg ? `<span class="chip off" title="Cancelled by safe context">cleared <b>${neg}</b></span>` : "");
}

function barRows(el, rows, {max = 1, fmt = (v) => v.toFixed(2), tone} = {}) {
  el.innerHTML = rows.map(([label, v]) => { const t = tone ? tone(v) : ""; return `<div class="bar"><span title="${esc(label)}">${esc(label)}</span><span class="track"><span class="fill ${t}" style="width:${Math.max(0, Math.min(100, v / max * 100))}%"></span></span><span>${fmt(v)}</span></div>`; }).join("");
}

/* ================= pipeline tab ================= */
function renderPipeline(r) {
  pipeResult = r; $("pipeline-empty").hidden = true; $("pipeline-body").hidden = false;
  const st = r.viz.stages || {}, c = r.final.components;
  $("stages").innerHTML = ["raw", "preprocessed", "preemph"].filter((k) => st[k]).map((k) => `<div class="stage"><small>${{raw: "Raw signal", preprocessed: "Resampled and normalised", preemph: "After pre-emphasis"}[k]}</small><canvas id="st-${k}" class="cv"></canvas></div>`).join("");
  const hit = c.fingerprint > 0, m = r.fingerprint[0];
  $("offset-info").textContent = m ? `Closest library clip: ${m.clip_id}. ${m.aligned_hashes} of ${m.query_hashes} query hashes align → ${hit ? "MATCH: far above the threshold." : "no match: the votes are spread out, with no clear spike."}` : "No library clip shares any hashes with this audio.";
  barRows($("sims"), r.script.map((x) => [x.script_id.replace("scam_", ""), x.similarity]), {tone: (v) => (v >= 0.78 ? "bad" : v >= 0.63 ? "warn" : "")});
  const act = {}; for (const f of r.red_flags) { const k = f.category.replace("_", " "); act[k] = act[k] || {on: 0, off: 0}; act[k][f.negated ? "off" : "on"]++; }
  $("flag-chips").innerHTML = Object.entries(act).map(([k, v]) => v.on ? `<span class="chip">${esc(k)} <b>${v.on}</b></span>` : `<span class="chip off">${esc(k)} <b>${v.off}</b></span>`).join("") || '<span class="muted small">No red-flag phrases.</span>';
  barRows($("components"), [["Fingerprint", c.fingerprint], ["Script match", c.script], ["Red flags", c.redflags], ["Voice (unused)", c.voice]], {fmt: (v) => pct(v, 0)});
  $("formula").textContent = `score = 100·(1 − Π(1 − wᵢ·sᵢ)) = ${r.final.score.toFixed(1)}   with weights fp 0.95, script 0.80, flags 0.60, voice 0.25`;
  const tm = Object.entries(r.timings_ms).filter(([k]) => k !== "total");
  barRows($("timings"), tm, {max: Math.max(...tm.map(([, v]) => v)), fmt: (v) => v.toFixed(0)});
  $("timings").insertAdjacentHTML("beforeend", `<p class="muted small">Total ${r.timings_ms.total.toFixed(0)} ms for ${r.duration_s.toFixed(1)} s of audio.</p>`);
}
function drawPipeline() {
  if (!pipeResult || $("tab-pipeline").hidden) return;
  const st = pipeResult.viz.stages || {};
  for (const k of ["raw", "preprocessed", "preemph"]) if (st[k] && $("st-" + k)) drawSeries($("st-" + k), st[k]);
  if (st.fp_offset_hist && st.fp_offset_hist.length) drawOffsetHist($("cv-offset"), st.fp_offset_hist, st.fp_offset_hist_meta);
  else { const f = fit($("cv-offset")); if (f) { f.ctx.fillStyle = css("--faint"); f.ctx.font = `${13 * f.k}px ${css("--font")}`; f.ctx.textAlign = "center"; f.ctx.fillText("No overlap with any library clip", f.W / 2, f.H / 2); } }
}

/* ================= canvases: redraw ================= */
function drawResult(r) {
  const v = r.viz, d = r.duration_s;
  drawWave($("cv-wave"), v.waveform, (i) => i / Math.max(1, v.waveform.length - 1) * d, 0, d, (t) => v.vad_segments.some(([a, b]) => t >= a && t <= b));
  const hit = r.fingerprint.length && r.final.components.fingerprint > 0;
  drawSpec($("cv-spec"), v, hit ? r.fingerprint[0] : null);
  drawTimeline($("cv-timeline"), r.risk_timeline, Math.max(5, d));
}
const LIVE_VIEW_S = 20;
function liveDraw() {
  const L = live; if (!L) return;
  const tEnd = Math.max(L.t, LIVE_VIEW_S), t0 = tEnd - LIVE_VIEW_S;
  drawWave($("cv-wave"), L.wave, (i) => i / 50, t0, tEnd, () => true);
  drawSpecLive($("cv-spec"), L, t0, LIVE_VIEW_S);
  drawTimeline($("cv-timeline"), L.timeline, Math.max(LIVE_VIEW_S, L.t));
}
function redrawAll() {
  if (!$("tab-console").hidden && !$("session").hidden) { if (live && live.t) liveDraw(); else if (current && current.final) drawResult(current); }
  drawPipeline();
}

/* ================= agent ================= */
function agentState(s) { const el = $("agent-state"); el.dataset.s = s; el.textContent = {idle: "Ready", thinking: "Thinking", speaking: "Speaking", listening: "Listening"}[s] || s; }
function stopSpeaking() { if (player) { player.pause(); player = null; } if ("speechSynthesis" in window) speechSynthesis.cancel(); agentState("idle"); }
function speak(text) {
  if (!speakOn || !("speechSynthesis" in window)) return;
  stopSpeaking(); const u = new SpeechSynthesisUtterance(text); u.onend = () => agentState("idle"); agentState("speaking"); speechSynthesis.speak(u);
}
function sayAudio(url, text) {
  if (!speakOn) return; if (!url) { speak(text); return; }
  stopSpeaking(); player = new Audio(url); agentState("speaking");
  player.onended = () => { player = null; agentState("idle"); }; player.onerror = () => speak(text);
  player.play().catch(() => speak(text));
}
function addBubble(kind, text) { const d = document.createElement("div"); d.className = `bubble ${kind}`; d.textContent = text; $("agent-log").appendChild(d); $("agent-log").scrollTop = 1e9; return d; }
function enableAgent(on) { for (const id of ["btn-ptt", "ask-text", "btn-ask"]) $(id).disabled = !on; document.querySelectorAll(".quick button").forEach((b) => (b.disabled = !on)); }
function buildQuick(level) {
  const risky = level === "high" || level === "scam";
  const qs = risky ? ["Why do you think it's a scam?", "What should I do?", "Is it safe to share the OTP?"] : ["Is this call safe?", "What should I do if I'm unsure?"];
  $("quick").innerHTML = qs.map((q) => `<button type="button">${esc(q)}</button>`).join("");
  $("quick").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => askAgent(b.textContent)));
}
async function askAgent(question, audioBlob) {
  if (!current) return;
  stopSpeaking(); agentState("thinking"); enableAgent(false);
  const userB = addBubble("user", question || "…"), typing = addBubble("agent typing", ""); typing.innerHTML = "<i></i><i></i><i></i>";
  try {
    let res;
    if (audioBlob) { const fd = new FormData(); fd.append("session_id", current.session_id); fd.append("file", audioBlob, "question.webm"); res = await fetch("/api/agent/ask_audio", {method: "POST", body: fd}); }
    else res = await fetch("/api/agent/ask", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({session_id: current.session_id, question})});
    const j = await res.json();
    if (!res.ok) throw new Error(j.error ? j.error.message : `HTTP ${res.status}`);
    if (j.transcript) userB.textContent = j.transcript;
    typing.remove(); addBubble("agent", j.reply.text); sayAudio(j.reply.audio_url, j.reply.text);
  } catch (e) {
    typing.remove(); agentState("idle"); userB.textContent = userB.textContent === "…" ? "(voice question)" : userB.textContent;
    addBubble("agent", e instanceof TypeError ? "I can't reach the server right now." : e.message);
  } finally { enableAgent(true); }
}
let rec = null, chunks = [];
async function pttStart(e) {
  e.preventDefault(); if (rec || $("btn-ptt").disabled) return; stopSpeaking();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true}});
    chunks = []; rec = new MediaRecorder(stream); rec.ondataavailable = (ev) => chunks.push(ev.data);
    rec.onstop = () => { stream.getTracks().forEach((t) => t.stop()); const b = new Blob(chunks, {type: rec.mimeType}); rec = null; $("btn-ptt").classList.remove("recording"); b.size > 2000 ? askAgent(null, b) : agentState("idle"); };
    rec.start(); $("btn-ptt").classList.add("recording"); agentState("listening");
  } catch { addBubble("agent", "Microphone permission was denied. You can type your question instead."); }
}
const pttStop = () => { if (rec && rec.state === "recording") rec.stop(); };

function showAlert(r) {
  const lv = r.final.level;
  if (lv !== "high" && lv !== "scam") { $("alert").hidden = true; return; }
  banner(`${LV[lv].label}: ${nice(r.final.top_reasons[0] || "several warning signs")}. Hang up and don't share any code or money.`);
  const t = r.fingerprint.length && r.final.components.fingerprint > 0 ? r.fingerprint[0].scam_type : (r.script.length && r.final.components.script >= 0.5 ? r.script[0].scam_type : null);
  const warn = addBubble("agent warn", "");
  fetch(`/api/agent/warn/${r.session_id}`).then((x) => x.json()).then((w) => { warn.textContent = w.text; sayAudio(w.audio_url, w.text); })
    .catch(() => { warn.textContent = `This call looks like a ${TYPE[t] || "scam"}. Hang up and call the official number.`; speak(warn.textContent); });
}

/* ================= render a finished analysis ================= */
function render(r, opts = {}) {
  current = r; shownRisk = true; showSession(true); $("session").classList.remove("loading");
  setVerdict({score: r.final.score, level: r.final.level, reasons: r.final.top_reasons, scamType: r.fingerprint.length && r.final.components.fingerprint > 0 ? r.fingerprint[0].scam_type : (r.script[0] && r.final.components.script >= 0.5 ? r.script[0].scam_type : null)});
  tilesFromResult(r); renderTranscript(r.transcript, r.red_flags); renderPipeline(r);
  $("live-pill").hidden = true; $("btn-stop").hidden = true;
  requestAnimationFrame(() => { drawResult(r); drawPipeline(); });
  enableAgent(true); buildQuick(r.final.level);
  if (!opts.keepChat) { $("agent-log").innerHTML = ""; addBubble("sys", "Analysis complete. Ask me anything about this call."); }
  if (!opts.keepAlert) showAlert(r);
  setStatus("Ready", "ok");
}

/* ================= live mode ================= */
function liveReset(kind, name) {
  stopSpeaking(); current = null; shownScore = 0; shownRisk = false;
  showSession(true); $("alert").hidden = true; $("player").hidden = true; $("session").classList.remove("loading");
  $("clip-title").textContent = name; $("clip-meta").textContent = kind === "mic" ? "Microphone" : "Streaming at real-time speed";
  $("live-pill").hidden = false; $("live-time").textContent = "0:00"; $("btn-stop").hidden = false;
  setVerdict({score: 0, level: "safe", reasons: [], listening: true}); resetTiles();
  $("transcript").innerHTML = '<p class="empty">Listening… the transcript appears as people speak.</p>'; $("flag-legend").innerHTML = "";
  $("agent-log").innerHTML = ""; addBubble("sys", "Listening. I'll warn you if something looks wrong."); $("quick").innerHTML = ""; enableAgent(false);
  live = {kind, name, wave: [], spec: [], rows: 257, t: 0, timeline: [], fp: null, sid: null, firstRisk: false};
  setStatus("Live", "live");
}
function liveEvent(e) {
  const L = live; if (!L) return;
  if (e.type === "ready") { L.sid = e.session_id; current = {session_id: e.session_id}; enableAgent(true); }
  else if (e.type === "viz") {
    L.t = e.t_s; $("live-time").textContent = fmtT(e.t_s); L.wave.push(...e.waveform);
    const bin = atob(e.spec_cols_b64); L.rows = e.spec_rows;
    for (let c = 0; c < e.spec_cols; c++) { const col = new Uint8Array(e.spec_rows); for (let f = 0; f < e.spec_rows; f++) col[f] = bin.charCodeAt(c * e.spec_rows + f); L.spec.push(col); }
    liveDraw();
  } else if (e.type === "transcript") {
    if ($("transcript").querySelector(".empty")) $("transcript").innerHTML = "";
    $("transcript").insertAdjacentHTML("beforeend", `<div class="seg"><span class="t">${fmtT(e.segment.start_s)}</span><p>${esc(e.segment.text)}</p></div>`); $("transcript").scrollTop = 1e9;
  } else if (e.type === "fp_match") { L.fp = e.match; setTile("fp", {val: "Known recording", sub: "Audio matches a known scam recording.", meter: e.match.score, tone: "hot"}); }
  else if (e.type === "risk") {
    const st = e.state; L.timeline.push(st); L.firstRisk = true; shownRisk = true;
    setVerdict({score: st.score, level: st.level, reasons: st.top_reasons, scamType: L.fp ? L.fp.scam_type : null});
    tilesFromComponents(st.components, !!L.fp); buildQuick(st.level); liveDraw();
  } else if (e.type === "alert") {
    banner(`${LV[e.level].label} at ${fmtT(e.t_s)}: hang up and don't share any code or money.`);
    addBubble("agent warn", e.message); sayAudio(e.audio_url, e.message);
  } else if (e.type === "final") {
    const log = $("agent-log").innerHTML; live = null; setLiveUI(false);
    $("clip-meta").textContent = `${e.result.duration_s.toFixed(1)} s · live call ended`;
    render(e.result, {keepAlert: true, keepChat: true}); $("agent-log").innerHTML = log; $("agent-log").scrollTop = 1e9;
  } else if (e.type === "error") { setStatus("Error", "err"); addBubble("agent", e.message); }
}
function setLiveUI(on) { $("live-pill").hidden = !on; $("btn-stop").hidden = !on; }
function openLive(kind, name) {
  liveReset(kind, name);
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live`);
  ws.binaryType = "arraybuffer"; live.ws = ws;
  ws.onmessage = (m) => liveEvent(JSON.parse(m.data));
  ws.onclose = () => { if (live && live.ws === ws) { live = null; setLiveUI(false); setStatus("Ready", "ok"); } };
  ws.onerror = () => setStatus("Backend unreachable", "err");
  return new Promise((res, rej) => { ws.onopen = () => res(ws); setTimeout(() => rej(new Error("timeout")), 5000); });
}
function sendPCM(ws, f32) {                           // Float32 16 kHz -> PCM16; silenced while the agent talks (half-duplex)
  if (ws.readyState !== 1) return;
  const pcm = new Int16Array(f32.length), mute = !!player || ("speechSynthesis" in window && speechSynthesis.speaking);
  if (!mute) for (let i = 0; i < f32.length; i++) pcm[i] = Math.max(-32768, Math.min(32767, Math.round(f32[i] * 32767)));
  ws.send(pcm.buffer);
}
async function startMic() {
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true, channelCount: 1}}); }
  catch { showSession(false); setStatus("Mic blocked", "err"); neutralBanner("Microphone permission was denied. You can upload a recording or try an example instead."); return; }
  const ws = await openLive("mic", "Live call (microphone)").catch(() => null); if (!ws) { stream.getTracks().forEach((t) => t.stop()); showSession(false); return; }
  const ctx = new AudioContext({sampleRate: 16000}); await ctx.audioWorklet.addModule("pcm-worklet.js");
  const src = ctx.createMediaStreamSource(stream), node = new AudioWorkletNode(ctx, "pcm-forwarder"); let buf = [];
  node.port.onmessage = (m) => { buf.push(...m.data); if (buf.length >= 1600) sendPCM(ws, Float32Array.from(buf.splice(0, 1600))); };
  src.connect(node);
  live.stopFn = () => { src.disconnect(); node.disconnect(); stream.getTracks().forEach((t) => t.stop()); ctx.close(); if (ws.readyState === 1) ws.send(JSON.stringify({type: "stop"})); setStatus("Finishing…", "busy"); };
}
const neutralBanner = (t) => { showSession(true); banner(t, "neutral"); };
async function simulateLive(id, title) {
  const ctx = new AudioContext({sampleRate: 16000});
  const audio = await ctx.decodeAudioData(await (await fetch(`/api/samples/${id}`)).arrayBuffer());
  const ws = await openLive("sim", title).catch(() => null); if (!ws) { ctx.close(); showSession(false); return; }
  const data = audio.getChannelData(0), src = ctx.createBufferSource(); src.buffer = audio; src.connect(ctx.destination);
  ctx.resume().catch(() => {}); src.start();
  const p0 = performance.now(); let sent = 0;          // paced by the wall clock (a suspended AudioContext has no clock)
  const timer = setInterval(() => {
    const due = Math.min(data.length, Math.floor((performance.now() - p0) / 1000 * 16000));
    while (due - sent >= 1600 || (due === data.length && sent < data.length)) { const n = Math.min(1600, data.length - sent); sendPCM(ws, data.subarray(sent, sent + n)); sent += n; }
    if (sent >= data.length) { clearInterval(timer); if (ws.readyState === 1) ws.send(JSON.stringify({type: "stop"})); setStatus("Finishing…", "busy"); }
  }, 100);
  live.stopFn = () => { clearInterval(timer); try { src.stop(); } catch {} ctx.close(); if (ws.readyState === 1) ws.send(JSON.stringify({type: "stop"})); setStatus("Finishing…", "busy"); };
}

/* ================= actions ================= */
async function analyze(blob, name) {
  showSession(true); stopSpeaking(); shownScore = 0; shownRisk = false; current = null;
  $("player").src = URL.createObjectURL(blob); $("player").hidden = false; $("live-pill").hidden = true; $("btn-stop").hidden = true;
  $("clip-title").textContent = name; $("clip-meta").textContent = "Analysing…"; $("alert").hidden = true;
  setVerdict({score: 0, level: "safe", reasons: [], listening: true, label: "Analysing"}); $("verdict-title").textContent = "Analysing the recording…"; $("verdict-sub").textContent = "Fingerprinting, transcribing and scoring the call.";
  resetTiles(); $("transcript").innerHTML = '<p class="empty">Transcribing…</p>'; $("flag-legend").innerHTML = ""; $("agent-log").innerHTML = ""; $("quick").innerHTML = ""; enableAgent(false);
  $("session").classList.add("loading"); setStatus("Analysing", "busy");
  const fd = new FormData(); fd.append("file", blob, name);
  try {
    const res = await fetch("/api/analyze", {method: "POST", body: fd}), j = await res.json();
    if (!res.ok) throw new Error(j.error ? j.error.message : `HTTP ${res.status}`);
    $("clip-meta").textContent = `${j.duration_s.toFixed(1)} s · analysed in ${(j.timings_ms.total / 1000).toFixed(1)} s`;
    render(j);
  } catch (e) {
    $("session").classList.remove("loading"); $("clip-meta").textContent = "Failed"; setStatus("Error", "err");
    $("verdict-title").textContent = "Couldn't analyse this recording"; $("verdict-sub").textContent = "";
    banner(e instanceof TypeError ? "Can't reach the server. Check that it is running, then try again." : `Could not analyse: ${e.message}`, "neutral");
  }
}
async function playSample(id, title) {
  if (live && live.stopFn) live.stopFn();
  if ($("as-live").checked && !(params.get("live") === "0")) { simulateLive(id, title).catch(() => setStatus("Error", "err")); return; }
  try { const r = await fetch(`/api/samples/${id}`); if (!r.ok) throw new Error(); await analyze(await r.blob(), title); } catch { setStatus("Error", "err"); }
}
function newCall() {
  if (live && live.stopFn) { live.stopFn(); live = null; }
  stopSpeaking(); current = null; shownScore = 0; pipeResult = null; $("pipeline-empty").hidden = false; $("pipeline-body").hidden = true;
  $("player").pause(); $("player").hidden = true; $("alert").hidden = true; showSession(false); setStatus("Ready", "ok");
}

/* ================= samples + init ================= */
async function loadSamples() {
  const list = await (await fetch("/api/samples")).json();
  $("samples").innerHTML = list.map((s) => {
    const [kind, ...rest] = s.title.split(": "), isScam = kind.toLowerCase() === "scam";
    return `<button class="sample" data-id="${esc(s.id)}" data-title="${esc(s.title)}"><span class="tag ${isScam ? "scam" : "legit"}">${isScam ? "Scam" : "Legit"}</span><h3>${esc(cap(rest.join(": ").replace(/\s*\(.*\)$/, "")))}</h3><p>${esc(SAMPLE_HINT[s.id] || "")}</p><span class="go"><svg class="ic" aria-hidden="true"><use href="#i-play"/></svg>Play this call</span></button>`;
  }).join("") || '<p class="muted">No example calls are available. Check that data/ is in place.</p>';
  $("samples").querySelectorAll(".sample").forEach((b) => b.addEventListener("click", () => playSample(b.dataset.id, b.dataset.title)));
}
async function init() {
  try {
    const h = await (await fetch("/api/health")).json(), ml = h.models_loaded, ok = ml && ml.index && ml.stt && ml.embed;
    setStatus(ok ? "Ready" : "Models missing", ok ? "ok" : "err"); await loadSamples();
  } catch { setStatus("Server unreachable", "err"); $("samples").innerHTML = '<p class="muted">The server is not reachable. Start it with <code>python scripts/run_demo.py</code>.</p>'; }
  if (params.get("theme")) applyTheme(params.get("theme"));
  const tab = params.get("tab"); if (tab) switchTab(tab);
  const sid = params.get("sample"); if (sid) { const el = $("samples").querySelector(`[data-id="${sid}"]`); if (el) playSample(sid, el.dataset.title); }
}

/* ================= evaluation tab ================= */
let results = null, splitKey = null;
const SPLIT_INFO = {test: ["Test", "run once · honest"], holdout: ["Holdout", "new scripts · unseen voices"], dev: ["Dev", "calibration · optimistic"]};
function kpi(label, value, sub, tone) { return `<div class="kpi ${tone || ""}"><span>${esc(label)}</span><b>${value}</b><small>${esc(sub)}</small></div>`; }
const toneHi = (v, g = 0.95, w = 0.85) => (v >= g ? "good" : v >= w ? "warn" : "bad");
const toneLo = (v, g = 0.02, w = 0.07) => (v <= g ? "good" : v <= w ? "warn" : "bad");
function renderSplit(name, s) {
  const f = s.fusion, l1 = s.layer1, l2 = s.layer2, lv = s.live, aug = l1.by_augmentation || [];
  const note = name === "dev" ? ["warn", "Thresholds were tuned on this split, so these numbers are optimistic. Look at Test and Holdout for the honest picture."]
    : name === "test" ? ["warn", "Run once with the dev-calibrated thresholds. The script layer is a sanity check here: the same scam scripts read by different voices. Hinglish read with a British accent is the weak spot."]
    : ["ok", l2.note];
  const meanRecall = aug.length ? aug.reduce((a, r) => a + r.recall_at_1, 0) / aug.length : null;
  const rows = (arr, fmt) => arr.map(([a, b]) => `<li><span>${esc(a)}</span><b>${fmt ? fmt(b) : b}</b></li>`).join("");
  const t = Object.fromEntries(s.timings.map((x) => [x.stage, x]));
  return `
  <div class="callout ${note[0] === "ok" ? "ok" : ""}"><svg class="ic" aria-hidden="true"><use href="#i-${note[0] === "ok" ? "check" : "alert"}"/></svg><span>${esc(note[1])}</span></div>
  <div class="kpis">
    ${kpi("Scams caught", pct(f.recall), `${f.tp} of ${f.tp + f.fn} scam clips`, toneHi(f.recall))}
    ${kpi("False alarms on legit calls", pct(f.fpr_legit), `${f.fp} of ${f.fp + f.tn} legit clips`, toneLo(f.fpr_legit))}
    ${kpi("Precision", pct(f.precision), "of calls flagged, how many were scams", toneHi(f.precision))}
    ${kpi("Fingerprint recall", meanRecall == null ? "–" : pct(meanRecall, 0), aug.length ? `across ${aug.length} phone-line and noise conditions` : "no indexed audio in this split", meanRecall == null ? "" : toneHi(meanRecall))}
    ${lv ? kpi("Live: legit calls alerted", pct(lv.legit_alert_rate, 0), `scam calls alerted: ${pct(lv.scam_alerted, 0)} (n=${lv.n_scam})`, toneLo(lv.legit_alert_rate, 0.0, 0.05)) : ""}
  </div>
  <div class="cards">
    <article class="card"><h3>Fingerprint by condition</h3><p class="muted">Recall@1: the right known recording is the top match. Threshold ${l1.threshold_aligned_hashes} aligned hashes.</p>
      ${aug.length ? `<div class="bars" id="r-aug"></div>` : `<p class="muted">No indexed audio appears in this split, so every match would be a false match.</p>`}
      <ul class="ls"><li><span>False matches (legit audio or wrong script)</span><b>${l1.false_match_rate == null ? "–" : pct(l1.false_match_rate, 1)} of ${l1.n_negatives}</b></li><li><span>Same script, other speaking rate</span><b>${pct(l1.same_script_near_copy_match_rate, 0)} matched (n=${l1.n_same_script_near_copies})</b></li><li><span>Lookup time (median)</span><b>${l1.query_ms_p50} ms</b></li></ul></article>
    <article class="card"><h3>Script layer by language</h3><p class="muted">AUC: how well scam and legit separate (1.0 is perfect, 0.5 is chance).</p><div class="bars" id="r-lang"></div>
      <ul class="ls"><li><span>Script similarity AUC</span><b>${l2.auc_script_sim}</b></li><li><span>Red-flag AUC</span><b>${l2.auc_redflags}</b></li><li><span>Scam type identified</span><b>${pct(l2.scam_type_top1_accuracy, 0)}</b></li><li><span>Word error rate (English, clean)</span><b>${pct(l2.wer_en_by_aug.clean, 1)}</b></li></ul></article>
    <article class="card"><h3>What each layer contributes</h3><p class="muted">Share of scam clips flagged (high or scam). Fusion beats any single layer.</p><div class="bars" id="r-abl"></div>
      <p class="muted small" style="margin-top:12px">False alarms on legit: ${s.ablation.map((a) => `${a.config} ${pct(a.fpr_legit, 1)}`).join(" · ")}</p></article>
    <article class="card"><h3>Flagged vs not flagged</h3><p class="muted">Every clip in this split, using the full system.</p>
      <div class="cm"><div class="h"></div><div class="h">flagged</div><div class="h">not flagged</div><div class="h">scam</div><div class="tp"><b>${f.tp}</b>caught</div><div class="fn"><b>${f.fn}</b>missed</div><div class="h">legit</div><div class="fp"><b>${f.fp}</b>false alarm</div><div class="tn"><b>${f.tn}</b>correct</div></div></article>
    ${lv ? `<article class="card"><h3>Live detection</h3><p class="muted">Calls streamed at real-time speed through the same path as the microphone.</p><ul class="ls">
      <li><span>Replayed recordings alerted at</span><b>${lv.fingerprint_alerts.first_alert_t_s_p50} s into the call</b></li>
      <li><span>New voices: alert after the first red-flag phrase</span><b>median ${lv.words_only.latency_s_p50} s · p95 ${lv.words_only.latency_s_p95} s</b></li>
      <li><span>New voices alerted within 5 s</span><b>${pct(lv.words_only.share_within_5s, 0)}</b></li><li><span>Legit calls that ever alerted</span><b>${pct(lv.legit_alert_rate, 0)} (n=${lv.n_legit})</b></li></ul></article>` : ""}
    <article class="card"><h3>Speed (upload, per call)</h3><p class="muted">Median time per stage in milliseconds.</p><div class="bars" id="r-time"></div><p class="muted small" style="margin-top:8px">Total: ${t.total.p50_ms} ms median · ${t.total.p95_ms} ms p95</p></article>
  </div>
  <details class="more"><summary>Charts, misses and calibration</summary>
    <img src="/api/results/files/roc_layer2_${name}.png" alt="ROC curves" width="380"><img src="/api/results/files/recall_by_aug_${name}.png" alt="Recall by augmentation" width="520">
    <ul class="ls"><li><span>Missed scams (${f.n_missed})</span><b class="small" style="max-width:60%;text-align:right">${esc(f.missed_scams.slice(0, 5).map((m) => m.file.replace(".wav", "")).join(", ")) || "none"}${f.n_missed > 5 ? "…" : ""}</b></li>
    <li><span>False alarms (${f.false_alarms.length})</span><b class="small" style="max-width:60%;text-align:right">${esc(f.false_alarms.slice(0, 4).map((m) => m.file.replace(".wav", "")).join(", ")) || "none"}${f.false_alarms.length > 4 ? "…" : ""}</b></li>
    <li><span>Calibration</span><b class="small">${esc((s.fusion_config || {}).source || "")}</b></li></ul></details>`;
}
function fillResultBars(s) {
  const aug = s.layer1.by_augmentation || [];
  if ($("r-aug")) barRows($("r-aug"), aug.map((r) => [`${r.augmentation} (n=${r.n})`, r.recall_at_1]), {fmt: (v) => pct(v, 0), tone: (v) => (v >= 0.95 ? "good" : v >= 0.8 ? "warn" : "bad")});
  barRows($("r-lang"), Object.entries(s.layer2.by_language).map(([k, v]) => [{en: "English", hi: "Hinglish"}[k] || k, v.auc_script_sim]), {tone: (v) => (v >= 0.95 ? "good" : v >= 0.85 ? "warn" : "bad")});
  barRows($("r-abl"), s.ablation.filter((a) => a.config !== "voice only").map((a) => [a.config, a.recall]), {fmt: (v) => pct(v, 0), tone: (v) => (v >= 0.85 ? "good" : v >= 0.5 ? "warn" : "bad")});
  const tm = s.timings.filter((x) => !["total", "red_flags", "decode", "preprocess"].includes(x.stage));
  barRows($("r-time"), tm.map((x) => [x.stage.replace("_", " "), x.p50_ms]), {max: Math.max(...tm.map((x) => x.p50_ms)), fmt: (v) => v.toFixed(0)});
}
function paintResults() {
  const j = results, order = ["test", "holdout", "dev"].filter((k) => j.splits[k]);
  if (!splitKey || !j.splits[splitKey]) splitKey = order[0];
  $("results").innerHTML = `<div class="seg-ctl" role="group" aria-label="Data split">${order.map((k) => `<button data-k="${k}" aria-pressed="${k === splitKey}">${SPLIT_INFO[k][0]}<small>${SPLIT_INFO[k][1]}</small></button>`).join("")}</div>
    <p class="muted" style="margin:-8px 0 var(--s4)">${esc(j.splits[splitKey].n_clips)} clips · ${j.splits[splitKey].n_scam} scam, ${j.splits[splitKey].n_legit} legit</p>
    ${renderSplit(splitKey, j.splits[splitKey])}
    <p class="meta">Generated ${esc(j.generated_at)} · speech-to-text: Whisper ${esc(j.stt.model)} (${esc(j.stt.language)}) · commit ${esc(j.git_commit || "n/a")}</p>`;
  fillResultBars(j.splits[splitKey]);
  $("results").querySelectorAll(".seg-ctl button").forEach((b) => b.addEventListener("click", () => { splitKey = b.dataset.k; paintResults(); }));
}
async function loadResults() {
  try {
    results = await (await fetch("/api/results")).json();
    if (!results.splits || !Object.keys(results.splits).length) { $("results").innerHTML = '<div class="empty-state"><h3>No evaluation has been run yet</h3><p>Run <code>python -m scamshield.eval.run_eval</code>. No placeholder numbers are shown.</p></div>'; return; }
    splitKey = params.get("split") || splitKey; paintResults();
  } catch { $("results").innerHTML = '<p class="muted">The server is not reachable.</p>'; }
}

/* ================= wiring ================= */
function switchTab(t) {
  document.querySelectorAll("#tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === t)));
  for (const v of ["console", "pipeline", "results"]) $("tab-" + v).hidden = v !== t;
  if (t === "results") loadResults();
  requestAnimationFrame(redrawAll);
}
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
$("btn-upload").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) analyze(f, f.name); e.target.value = ""; });
$("btn-live").addEventListener("click", () => startMic());
$("btn-stop").addEventListener("click", () => { if (live && live.stopFn) live.stopFn(); });
$("btn-new").addEventListener("click", newCall);
$("alert-close").addEventListener("click", () => { $("alert").hidden = true; });
$("ask-form").addEventListener("submit", (e) => { e.preventDefault(); const q = $("ask-text").value.trim(); if (q) { $("ask-text").value = ""; askAgent(q); } });
$("btn-ptt").addEventListener("pointerdown", pttStart);
for (const ev of ["pointerup", "pointerleave", "pointercancel"]) $("btn-ptt").addEventListener(ev, pttStop);
$("btn-ptt").addEventListener("keydown", (e) => { if ((e.key === " " || e.key === "Enter") && !e.repeat) pttStart(e); });
$("btn-ptt").addEventListener("keyup", (e) => { if (e.key === " " || e.key === "Enter") pttStop(); });
$("speak-toggle").addEventListener("click", () => {
  speakOn = !speakOn; $("speak-toggle").setAttribute("aria-pressed", String(speakOn)); $("speak-toggle").setAttribute("aria-label", speakOn ? "Spoken replies on" : "Spoken replies off");
  $("speak-icon").setAttribute("href", speakOn ? "#i-volume" : "#i-volume-off"); if (!speakOn) stopSpeaking();
});
function applyTheme(t) { document.documentElement.dataset.theme = t; localStorage.setItem("ss-theme", t); $("theme-icon").setAttribute("href", t === "dark" ? "#i-sun" : "#i-moon"); requestAnimationFrame(redrawAll); }
$("theme").addEventListener("click", () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
$("theme-icon").setAttribute("href", document.documentElement.dataset.theme === "dark" ? "#i-sun" : "#i-moon");
let dragDepth = 0;
document.addEventListener("dragenter", (e) => { if (e.dataTransfer && [...e.dataTransfer.types].includes("Files")) { dragDepth++; $("drop").hidden = false; } });
document.addEventListener("dragleave", () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $("drop").hidden = true; });
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => { e.preventDefault(); dragDepth = 0; $("drop").hidden = true; const f = e.dataTransfer.files[0]; if (f) { switchTab("console"); analyze(f, f.name); } });
document.addEventListener("mouseover", (e) => { const b = e.target.closest && e.target.closest("[data-tip]"); if (b) tip(b); });
document.addEventListener("mouseout", (e) => { if (e.target.closest && e.target.closest("[data-tip]")) $("tip").hidden = true; });
document.addEventListener("focusin", (e) => { const b = e.target.closest && e.target.closest("[data-tip]"); if (b) tip(b); });
document.addEventListener("focusout", () => { $("tip").hidden = true; });
function tip(el) { const t = $("tip"); t.textContent = el.dataset.tip; t.hidden = false; const r = el.getBoundingClientRect(); t.style.left = `${Math.min(innerWidth - t.offsetWidth - 12, Math.max(12, r.left + r.width / 2 - t.offsetWidth / 2))}px`; t.style.top = `${r.bottom + 8}px`; }
let rsz; window.addEventListener("resize", () => { clearTimeout(rsz); rsz = setTimeout(redrawAll, 120); });
resetTiles(); setVerdict({score: 0, level: "safe", reasons: [], listening: true}); enableAgent(false); agentState("idle");
init();
