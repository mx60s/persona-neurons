// Persona neuron explorer. Saved run data (no GPU) plus live panels when the model is loaded.
// Two controls are kept apart everywhere: "model state" (base or scaled adapters) and "neuron intervention"
// (add / multiply / zero one activation inside that state). Every result prints both.

const $ = s => document.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : (typeof x === "number" ? (Math.abs(x) >= 1000 ? x.toFixed(0) : +x.toFixed(d) + "") : String(x));
const sgn = (x, d = 3) => (x > 0 ? "+" : x < 0 ? "−" : "") + f(Math.abs(x), d);

async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  let j = null;
  try { j = await r.json(); } catch { /* not json */ }
  if (!r.ok) throw new Error(j?.error || `${path.split("?")[0]}: HTTP ${r.status}`);
  return j;
}

// server-sent events over POST
async function stream(path, body, onEvent, signal) {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal });
  if (!res.ok) { let j = null; try { j = await res.json(); } catch { } throw new Error(j?.error || `HTTP ${res.status}`); }
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let k;
    while ((k = buf.indexOf("\n\n")) >= 0) {
      const line = buf.slice(0, k); buf = buf.slice(k + 2);
      if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
    }
  }
}

const S = {
  meta: null, man: null, trait: "openness", rows: [], sel: null, data: null,
  sort: { key: "rank", dir: -1 }, filter: "",
  ex: { kind: "all", split: "all", diff: false },
  steer: { mode: "add", value: { add: 0.5, mul: 2 } },
  live: { loaded: false, status: "", error: null },
};
const GROUP_NAMES = {
  shared: "shared: both poles", amp_primary: "amplifier-primary", sup_primary: "suppressor-primary",
  cross_trait: "cross-trait", control_assoc: "neutral-control associated", surprising: "surprising",
  random_control: "random controls (layer-matched)", other: "not selected for this trait",
};

// ---------- state helpers ----------
const stateList = () => S.man.states || [];
const stateLabel = s => (stateList().find(x => x.id === s) || {}).label || s;
const otherTrait = t => t === "openness" ? "neuroticism" : "openness";
function polarStates() {
  const t = S.man.traits || {}, out = [];
  for (const k of ["openness", "neuroticism"]) for (const s of (t[k]?.poles || [])) out.push(s);
  if (S.man.control_state) out.push(S.man.control_state);
  return out;
}
function settingsHTML(state, iv) {
  const st = Array.isArray(state) ? state.map(s => `<b>${esc(s)}</b>`).join(", ") : `<b>${esc(state)}</b>${state === "base" ? " (no adapter)" : ` (${esc(stateLabel(state))})`}`;
  return `<div class="settings"><span class="ms">model state: ${st}</span><span class="ni">neuron intervention: ${ivText(iv)}</span></div>`;
}
function ivText(iv) {
  if (!iv || iv.mode === "none") return "none";
  if (iv.mode === "zero") return `${esc(iv.neuron)} zeroed`;
  if (iv.mode === "mul") return `${esc(iv.neuron)} × ${f(iv.value)}`;
  return `${esc(iv.neuron)} add ${sgn(iv.value)} ${esc(iv.unit || "resid_p50")}`;
}

// ---------- colors ----------
function actColor(a, m) {
  if (!m || !a) return "transparent";
  const x = Math.min(1, Math.abs(a) / m) * 0.85;
  return a > 0 ? `rgba(255,181,71,${x.toFixed(3)})` : `rgba(255,79,154,${x.toFixed(3)})`;
}
function zColor(z, cap = 2) {
  if (z === null || z === undefined) return "transparent";
  const x = Math.min(1, Math.abs(z) / cap) * 0.8;
  return z > 0 ? `rgba(255,181,71,${x.toFixed(3)})` : `rgba(255,79,154,${x.toFixed(3)})`;
}

// ---------- boot ----------
S.meta = await api("/api/meta");
S.man = S.meta.run;
S.live = S.meta.live;
$("#run-id").textContent = `run ${S.man.run_id} · ${S.man.model?.id || ""}${S.man.model?.revision ? "@" + String(S.man.model.revision).slice(0, 8) : ""}`;
renderLive();
setInterval(async () => { if (!S.live.loaded) { try { S.live = await api("/api/live/status"); renderLive(); } catch { } } }, 3000);

function renderLive() {
  const el = $("#live"), L = S.live;
  el.className = "mono " + (L.loaded ? "ok" : "off");
  el.textContent = L.loaded ? `live model: ${L.info?.model} (${L.info?.device}, adapters ${(L.info && Object.keys(L.info.adapters).join(" ")) || "none"})`
    : `live model not loaded: ${L.error ? "load failed" : L.status}`;
  el.title = L.error || "";
  for (const id of ["custom", "chats", "steer"]) liveGuard($(`#tab-${id}`));
  $$(".live-weights").forEach(liveGuard);
}
function liveGuard(tab) {
  if (!tab) return;
  let n = tab.querySelector(":scope > .notice.live-off");
  if (!S.live.loaded) {
    if (!n) { n = document.createElement("div"); n.className = "notice live-off"; tab.prepend(n); }
    n.textContent = `Live model not loaded (${S.live.error ? "load failed: " + S.live.error : S.live.status}). This panel needs the model; saved data in the other tabs still works.`;
  } else if (n) n.remove();
  $$("button.btn", tab).forEach(b => { if (!b.classList.contains("savebtn")) b.disabled = !S.live.loaded; });
}

S.rows = await api("/api/neurons");
setupList();
fromHash();
addEventListener("hashchange", fromHash);

// ---------- list ----------
function sortKeys() {
  const keys = [["rank", "rank (this trait)"], ["layer", "layer"]];
  for (const s of polarStates()) {
    keys.push([`z:${s}`, `z_mean ${s}`], [`az:${s}`, `|z_mean| ${s}`], [`zr:${s}`, `z_rms ${s}`], [`ag:${s}`, `agree ${s}`], [`rd:${s}`, `resid_mean_diff ${s}`]);
  }
  keys.push(["rms", "base rms"], ["max", "base max"], ["status", "card status"]);
  return keys;
}
function sortVal(r, key) {
  const [k, s] = key.split(":");
  const e = s ? (r.effects || {})[s] || {} : null;
  switch (k) {
    case "rank": { const v = (r.rank || {})[S.trait]; return v == null ? -1e9 : -v; }
    case "layer": return r.layer * 1e6 + r.n;
    case "z": return e.z_mean ?? -1e9;
    case "az": return Math.abs(e.z_mean ?? 0);
    case "zr": return e.z_rms ?? -1e9;
    case "ag": return e.agree ?? -1e9;
    case "rd": return e.resid_mean_diff ?? -1e9;
    case "rms": return r.base?.rms ?? 0;
    case "max": return r.base?.max ?? 0;
    case "status": return ["rejected", "untested", "mixed", "tested"].indexOf(r.card_status);
  }
  return 0;
}
function setupList() {
  $("#sort").innerHTML = sortKeys().map(([k, l]) => `<option value="${esc(k)}">${esc(l)}</option>`).join("");
  $("#sort").onchange = e => { S.sort.key = e.target.value; renderList(); };
  $("#sort-dir").onclick = () => { S.sort.dir *= -1; $("#sort-dir").textContent = S.sort.dir < 0 ? "↓" : "↑"; renderList(); };
  $("#filter").oninput = e => { S.filter = e.target.value.toLowerCase(); renderList(); };
  $$("#trait button").forEach(b => b.onclick = () => setTrait(b.dataset.t));
  $("#jump").onkeydown = e => {
    if (e.key !== "Enter") return;
    const m = e.target.value.trim().toUpperCase().match(/^L?(\d+)\s*[\/ ]\s*N?(\d+)$/);
    if (m) go(`L${+m[1]}/N${+m[2]}`);
  };
  const ps = polarStates();
  document.documentElement.style.setProperty("--ncol", ps.length);
  $("#eff-head").innerHTML = `<span style="text-align:left">neuron · z_mean vs base</span>${ps.map(s => `<span title="${esc(stateLabel(s))}">${esc(s)}</span>`).join("")}<span>card</span>`;
  renderList();
}
function setTrait(t) {
  S.trait = t;
  $$("#trait button").forEach(b => b.classList.toggle("on", b.dataset.t === t));
  renderList();
  if (S.data) { renderNeuron(); setupLivePanels(); }
  writeHash();
}
function groupOf(r) {
  if (r.kind === "random_control") return "random_control";
  return (r.group || {})[S.trait] || "other";
}
function renderList() {
  const ps = polarStates(), order = [...S.meta.groups, "other"], by = {};
  for (const r of S.rows) {
    if (S.filter && !(`${r.id} ${r.label || ""} ${r.selection_reason || ""}`.toLowerCase().includes(S.filter))) continue;
    (by[groupOf(r)] ||= []).push(r);
  }
  const html = [];
  for (const g of order) {
    const rs = by[g]; if (!rs) continue;
    rs.sort((a, b) => S.sort.dir < 0 ? sortVal(b, S.sort.key) - sortVal(a, S.sort.key) : sortVal(a, S.sort.key) - sortVal(b, S.sort.key));
    html.push(`<div class="ghead">${esc(GROUP_NAMES[g] || g)} <span class="cnt">${rs.length}</span></div>`);
    for (const r of rs) {
      const cells = ps.map(s => {
        const e = (r.effects || {})[s];
        return e ? `<span class="eff" style="background:${zColor(e.z_mean)};color:var(--ink)" title="${esc(s)}: z_mean ${f(e.z_mean)}, z_rms ${f(e.z_rms)}, agree ${f(e.agree)}, mean_diff ${f(e.mean_diff, 4)}">${f(e.z_mean, 2)}</span>` : `<span class="eff na">–</span>`;
      }).join("");
      html.push(`<div class="nrow${r.id === S.sel ? " sel" : ""}" data-id="${esc(r.id)}"><span class="id">${esc(r.id)}<span class="lab">${esc(r.label || "")}</span></span>${cells}<span class="status ${esc(r.card_status)}">${esc(r.card_status)}</span></div>`);
    }
  }
  $("#groups").innerHTML = html.join("") || `<div class="slot">No neurons match.</div>`;
  $$("#groups .nrow").forEach(el => el.onclick = () => go(el.dataset.id));
}

// ---------- routing ----------
function go(id, tab) { S.pendingTab = tab; location.hash = `n=${encodeURIComponent(id)}&t=${S.trait}${tab ? "&tab=" + tab : ""}`; }
function writeHash() {
  const p = new URLSearchParams();
  if (S.sel) p.set("n", S.sel);
  p.set("t", S.trait);
  const tab = $("#tabs button.on")?.dataset.tab; if (S.sel && tab) p.set("tab", tab);
  history.replaceState(null, "", "#" + p.toString());
}
async function fromHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const t = p.get("t");
  if (t && t !== S.trait && S.man.traits?.[t]) { S.trait = t; $$("#trait button").forEach(b => b.classList.toggle("on", b.dataset.t === t)); renderList(); }
  const id = p.get("n");
  if (id && id !== S.sel) await openNeuron(id);
  if (p.get("tab")) setTab(p.get("tab"));
}
async function openNeuron(id) {
  S.sel = id;
  $$("#groups .nrow").forEach(el => el.classList.toggle("sel", el.dataset.id === id));
  try {
    S.data = await api(`/api/neuron?id=${encodeURIComponent(id)}`);
  } catch (e) {
    $("#empty").hidden = false; $("#neuron").hidden = true; $("#empty").textContent = String(e.message); return;
  }
  $("#empty").hidden = true; $("#neuron").hidden = false;
  S.ex = { kind: "all", split: "all", diff: S.ex.diff };
  renderNeuron();
  setupLivePanels();
}

// ---------- neuron page ----------
function latestVersion(card) { return card?.versions?.[0] || null; }
function autoTag(card, ver) {
  if (!ver) return "";
  if (ver.kind === "human_edit") return `<span class="tag">human edit</span>`;
  return card.status === "tested" ? `<span class="tag">automated, tested</span>` : `<span class="tag auto">Automated hypothesis</span>`;
}
function renderNeuron() {
  const d = S.data, r = d.index || {}, card = d.card, v = latestVersion(card);
  const status = card?.status || r.card_status || "untested";
  const groups = Object.entries(r.group || {}).filter(([, g]) => g).map(([t, g]) => `${t}: ${g}${r.rank?.[t] ? ` (rank ${r.rank[t]})` : ""}`).join(" · ");
  const b = r.base || {};
  $("#n-head").innerHTML = `<div class="idl"><span class="nid">${esc(d.id)}</span><span class="nlabel">${esc(v?.label || r.label || "")}</span>${autoTag(card, v)}
    <span class="status ${esc(status)}">${esc(status)}</span>${r.detailed ? `<span class="tag">detailed profile</span>` : ""}${r.kind === "random_control" ? `<span class="tag">random control</span>` : ""}</div>
    <div class="meta">${esc(groups || "no trait group")}${r.selection_reason ? ` · ${esc(r.selection_reason)}` : ""}</div>
    <div class="meta mono">base: mean ${f(b.mean)} · rms ${f(b.rms)} · pos ${f(b.frac_pos, 2)} / neg ${f(b.frac_neg, 2)} · p01 ${f(b.q?.p01)} · p99 ${f(b.q?.p99)} · p999 ${f(b.q?.p999)} · min ${f(b.min)} · max ${f(b.max)} · z denominator ${f(r.denominator)}</div>`;
  if (!d.index) $("#n-head").innerHTML += `<div class="notice">This neuron is not in neurons/index.json.</div>`;
  renderExamples();
  renderEffects();
  renderCard();
  renderSaved();
  $("#st-neuron").textContent = d.id;
}
$$("#tabs button").forEach(b => b.onclick = () => { setTab(b.dataset.tab); writeHash(); });
function setTab(t) {
  if (!$(`#tab-${t}`)) return;
  $$("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === t));
  $$(".tab").forEach(el => el.classList.toggle("on", el.id === `tab-${t}`));
  if (t === "steer") refreshCalib();
}

// ---------- token panels ----------
// rows: [{label, title, acts, cls}], tokens: [str]; returns HTML. Scale shared across rows so changes are visible.
const TOKVIEWS = [];
function tokPanel(tokens, rows, { focus = null, boundary = null, diffBase = null } = {}) {
  let m = 0, md = 0;
  for (const r of rows) if (!r.raw) for (const a of r.acts || []) m = Math.max(m, Math.abs(a));
  if (diffBase) for (const r of rows) if (r.acts !== diffBase && !r.raw) r.acts.forEach((a, i) => { md = Math.max(md, Math.abs(a - diffBase[i])); });
  const key = TOKVIEWS.push({ tokens, rows, diffBase }) - 1;
  const html = rows.map((r, ri) => {
    const useDiff = diffBase && r.acts !== diffBase && !r.raw;
    const spans = tokens.map((t, i) => {
      const a = r.acts?.[i];
      const c = useDiff ? actColor(a - diffBase[i], md) : actColor(a, r.scale || m);
      return `${boundary === i ? `<span class="bd"></span>` : ""}<span data-k="${key}" data-r="${ri}" data-i="${i}" class="${focus === i ? "focus" : ""}" style="background:${c}">${esc(t)}</span>`;
    }).join("");
    return `<div class="sl" title="${esc(r.title || r.label)}">${esc(r.label)}${useDiff ? " Δ" : ""}</div><div class="tl">${spans}</div>`;
  }).join("");
  return `<div class="toks">${html}</div>`;
}
document.addEventListener("mouseover", e => {
  const s = e.target.closest?.(".tl span[data-k]"), tip = $("#tip");
  if (!s) { tip.style.display = "none"; return; }
  const v = TOKVIEWS[+s.dataset.k], i = +s.dataset.i;
  const lines = [`token ${i}: ${JSON.stringify(v.tokens[i])}`];
  for (const r of v.rows) lines.push(`${r.label.padEnd(14)} ${r.acts?.[i] == null ? "–" : f(r.acts[i], 4)}${v.diffBase && r.acts !== v.diffBase && !r.raw ? `  (Δ ${sgn(r.acts[i] - v.diffBase[i], 4)})` : ""}`);
  tip.textContent = lines.join("\n");
  tip.style.display = "block";
  tip.style.left = Math.min(innerWidth - 320, e.clientX + 12) + "px";
  tip.style.top = (e.clientY + 14) + "px";
});
function legendHTML(extra = "") {
  return `<div class="legend"><span><i class="sw" style="background:var(--up)"></i>positive activation</span><span><i class="sw" style="background:var(--down)"></i>negative activation</span>
    <span><i class="sw" style="outline:1.5px solid #fff"></i>focus token</span><span><i class="sw" style="border-left:2px dashed var(--state)"></i>prompt | continuation boundary</span>${extra}
    <span>color scale shared across the activation rows of each panel (gate and up rows use their own); hover a token for values</span></div>`;
}
function stateOrder(states) {
  const order = stateList().map(s => s.id);
  return [...states].sort((a, b) => (order.indexOf(a) + 1 || 999) - (order.indexOf(b) + 1 || 999));
}

// ---------- examples ----------
function renderExamples() {
  TOKVIEWS.length = 0;
  const d = S.data.detail, el = $("#tab-examples");
  if (!d) { el.innerHTML = `<div class="slot">No detail file for this neuron (neurons/detail/).</div>`; return; }
  const exs = d.examples || [];
  const kinds = [...new Set(exs.map(e => e.kind))], splits = [...new Set(exs.map(e => e.split).filter(Boolean))];
  const shown = exs.filter(e => (S.ex.kind === "all" || e.kind === S.ex.kind) && (S.ex.split === "all" || e.split === S.ex.split));
  el.innerHTML = `<div class="row">
      <select id="ex-kind"><option value="all">all kinds (${exs.length})</option>${kinds.map(k => `<option ${S.ex.kind === k ? "selected" : ""}>${esc(k)}</option>`).join("")}</select>
      <select id="ex-split"><option value="all">all splits</option>${splits.map(k => `<option ${S.ex.split === k ? "selected" : ""}>${esc(k)}</option>`).join("")}</select>
      <label class="dim"><input type="checkbox" id="ex-diff" ${S.ex.diff ? "checked" : ""}> color adapter rows by change from base</label></div>
    ${legendHTML()}<div class="settings"><span class="ms">model state: one row per state (row label), same tokens in every row</span><span class="ni">neuron intervention: none</span></div>
    ${shown.map(exampleHTML).join("") || `<div class="slot">No examples match.</div>`}`;
  $("#ex-kind").onchange = e => { S.ex.kind = e.target.value; renderExamples(); };
  $("#ex-split").onchange = e => { S.ex.split = e.target.value; renderExamples(); };
  $("#ex-diff").onchange = e => { S.ex.diff = e.target.checked; renderExamples(); };
}
function exampleHTML(e) {
  const states = stateOrder(Object.keys(e.acts || {}));
  const rows = states.map(s => ({ label: s, title: stateLabel(s), acts: e.acts[s] }));
  for (const k of ["gate", "up"]) for (const [s, a] of Object.entries(e[k] || {})) rows.push({ label: `${k} (${s})`, title: `${k}_proj pre-activation in ${s}`, acts: a, raw: true, scale: Math.max(...a.map(Math.abs)) });
  const ft = e.focus != null ? e.tokens[e.focus] : null;
  const fvals = e.focus != null ? states.map(s => `${s} ${f(e.acts[s][e.focus], 3)}`).join(" · ") : "";
  const tm = e.train_meta ? ` · training: ${esc(e.train_meta.adapter)} ${esc(e.train_meta.role)} ${esc(e.train_meta.dataset_path || "")}` : "";
  const base = S.ex.diff && e.acts?.base ? e.acts.base : null;
  return `<div class="ex" id="ex-${esc(e.ex_id)}"><div class="ex-head"><span class="kind">${esc(e.ex_id)}</span><span class="kind">${esc(e.kind)}</span><span>${esc(e.split || "")}</span>
    <span>${esc(e.domain)}</span><span>${esc(e.doc_id)}${e.start != null ? ` @${e.start}` : ""}</span><span>${esc(e.source || "")}${tm}</span></div>
    ${ft != null ? `<div class="ex-head">focus ${esc(JSON.stringify(ft))} (token ${e.focus}): ${esc(fvals)}${e.boundary == null ? " · no prompt/continuation split in window" : ""}</div>` : ""}
    ${tokPanel(e.tokens, rows, { focus: e.focus, boundary: e.boundary, diffBase: base })}</div>`;
}

// ---------- effects ----------
function trajChart(traj) {
  // x slots: sweep coefficients, then the independent suppressor and the control as separate slots
  const W = 560, H = 230, pad = { l: 46, r: 110, t: 18, b: 34 };
  const colors = { openness: "var(--o)", neuroticism: "var(--n)" };
  const series = [];
  for (const [t, pts] of Object.entries(traj || {})) {
    const info = S.man.traits?.[t] || {};
    const sweep = [], extra = [];
    for (const p of pts) {
      const m = p.state === "base" ? { coef: 0 } : (p.state.match(/^([a-z_]+)([+-][\d.]+)$/) ? { name: RegExp.$1, coef: +RegExp.$2 } : null);
      if (p.state === "base" || (m && m.name === info.amp)) sweep.push({ ...p, x: m.coef });
      else extra.push({ ...p, x: 1.6, slot: p.state });
    }
    sweep.sort((a, b) => a.x - b.x);
    series.push({ t, sweep, extra });
  }
  const ctrl = S.data.index?.effects?.[S.man.control_state];
  const all = series.flatMap(s => [...s.sweep, ...s.extra]).flatMap(p => [p.z_mean, ...(p.ci90 || [])]).filter(v => v != null);
  if (ctrl) all.push(ctrl.z_mean);
  if (!all.length) return `<div class="slot">No trajectory data.</div>`;
  let lo = Math.min(0, ...all), hi = Math.max(0, ...all);
  if (hi - lo < 0.2) { hi += 0.1; lo -= 0.1; }
  const xmin = -1.15, xmax = 2.35;
  const X = x => pad.l + (x - xmin) / (xmax - xmin) * (W - pad.l - pad.r);
  const Y = y => pad.t + (hi - y) / (hi - lo) * (H - pad.t - pad.b);
  const ticks = [];
  const step = niceStep((hi - lo) / 4);
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(6));
  const out = [`<svg class="chart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="z_mean across adapter coefficients">`];
  for (const v of ticks) out.push(`<line class="${v === 0 ? "zero" : "grid"}" x1="${pad.l}" x2="${W - pad.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${pad.l - 6}" y="${Y(v) + 4}" text-anchor="end">${f(v, 2)}</text>`);
  for (const x of [-1, -0.5, 0, 0.5, 1]) out.push(`<text x="${X(x)}" y="${H - pad.b + 16}" text-anchor="middle">${x > 0 ? "+" + x : x}</text>`);
  out.push(`<text x="${X(1.6)}" y="${H - pad.b + 16}" text-anchor="middle">sup +1</text><text x="${X(2.2)}" y="${H - pad.b + 16}" text-anchor="middle">ctrl</text>`);
  out.push(`<text x="${X(0)}" y="${H - 2}" text-anchor="middle">amplifier coefficient</text><text x="12" y="${pad.t - 6}">z_mean</text>`);
  const off = { openness: -0.04, neuroticism: 0.04 };
  for (const s of series) {
    const c = colors[s.t] || "var(--ink)", dx = off[s.t] || 0;
    const pts = s.sweep.map(p => `${X(p.x + dx)},${Y(p.z_mean)}`).join(" ");
    out.push(`<polyline points="${pts}" fill="none" stroke="${c}" stroke-width="2"/>`);
    for (const p of [...s.sweep, ...s.extra]) {
      const x = X(p.x + dx);
      if (p.ci90) out.push(`<line x1="${x}" x2="${x}" y1="${Y(p.ci90[0])}" y2="${Y(p.ci90[1])}" stroke="${c}" stroke-width="1.5" opacity=".6"/>`);
      out.push(`<circle cx="${x}" cy="${Y(p.z_mean)}" r="4.5" fill="${c}" stroke="#050607" stroke-width="2"><title>${esc(s.t)} · ${esc(p.state)}\nz_mean ${f(p.z_mean)}  ci90 [${(p.ci90 || []).map(v => f(v)).join(", ")}]\nmean_diff ${f(p.mean_diff, 4)}  mean ${f(p.mean, 4)}  rms ${f(p.rms, 4)}</title></circle>`);
    }
    const last = s.sweep[s.sweep.length - 1];
    if (last) out.push(`<text x="${W - pad.r + 8}" y="${Y(last.z_mean) + 4}" style="fill:${c}">${esc(s.t)}</text>`);
  }
  if (ctrl) out.push(`<rect x="${X(2.2) - 5}" y="${Y(ctrl.z_mean) - 5}" width="10" height="10" fill="#8c8c8c" stroke="#050607" stroke-width="2"><title>${esc(S.man.control_state)} z_mean ${f(ctrl.z_mean)}</title></rect>`);
  out.push(`</svg>`);
  const legend = `<div class="legend">${series.map(s => `<span><i class="sw" style="background:${colors[s.t]}"></i>${esc(s.t)}: amplifier sweep, then independent suppressor</span>`).join("")}<span><i class="sw" style="background:#8c8c8c"></i>neutral control</span><span>bars: ci90; hover points for values</span></div>`;
  return legend + out.join("");
}
function niceStep(x) { const p = Math.pow(10, Math.floor(Math.log10(x || 1))); const m = x / p; return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p; }

function renderEffects() {
  const d = S.data.detail || {}, r = S.data.index || {}, el = $("#tab-effects");
  const st = stateOrder(Object.keys(r.effects || {}));
  const effRows = st.map(s => { const e = r.effects[s]; return `<tr><td class="mono" title="${esc(stateLabel(s))}">${esc(s)}</td><td class="num">${f(e.mean_diff, 4)}</td><td class="num">${f(e.rms_diff, 4)}</td><td class="cell" style="background:${zColor(e.z_mean)}">${f(e.z_mean)}</td><td class="cell" style="background:${zColor(e.z_rms)}">${f(e.z_rms)}</td><td class="num">${f(e.agree, 2)}</td><td class="num">${f(e.resid_mean_diff, 4)}</td></tr>`; }).join("");
  const trajTables = Object.entries(d.trajectory || {}).map(([t, pts]) => `<h4>${esc(t)}</h4><table class="dtable"><tr><th>state</th><th class="num">mean</th><th class="num">rms</th><th class="num">mean_diff</th><th class="num">z_mean</th><th class="num">ci90</th></tr>
    ${pts.map(p => `<tr><td class="mono">${esc(p.state)}</td><td class="num">${f(p.mean, 4)}</td><td class="num">${f(p.rms, 4)}</td><td class="num">${f(p.mean_diff, 4)}</td><td class="num">${f(p.z_mean)}</td><td class="num">${(p.ci90 || []).map(v => f(v)).join(" … ")}</td></tr>`).join("")}</table>`).join("");
  const doms = [...new Set(Object.values(d.by_domain || {}).flatMap(o => Object.keys(o)))];
  const byDom = stateOrder(Object.keys(d.by_domain || {})).map(s => `<tr><td class="mono">${esc(s)}</td>${doms.map(dm => { const c = d.by_domain[s][dm]; return c ? `<td class="cell" style="background:${zColor(c.mean_diff / (r.denominator || 1))}" title="n_tok ${c.n_tok}; z ${f(c.mean_diff / (r.denominator || 1))}">${f(c.mean_diff, 4)}</td>` : "<td>–</td>"; }).join("")}</tr>`).join("");
  const bySeg = stateOrder(Object.keys(d.by_segment || {})).map(s => `<tr><td class="mono">${esc(s)}</td><td class="cell" style="background:${zColor(d.by_segment[s].prompt / (r.denominator || 1))}">${f(d.by_segment[s].prompt, 4)}</td><td class="cell" style="background:${zColor(d.by_segment[s].continuation / (r.denominator || 1))}">${f(d.by_segment[s].continuation, 4)}</td></tr>`).join("");
  const w = d.weights || {};
  const wStates = stateOrder([...new Set([...Object.keys(w.down_norm || {}), ...Object.keys(w.down_cos_to_base || {})])]);
  const wRows = wStates.map(s => `<tr><td class="mono">${esc(s)}</td><td class="num">${f(w.down_norm?.[s], 5)}</td><td class="num">${f(w.down_cos_to_base?.[s], 6)}</td><td class="num">${f(w.gate_cos_to_base?.[s], 6)}</td><td class="num">${f(w.up_cos_to_base?.[s], 6)}</td></tr>`).join("");
  const lensRows = stateOrder(Object.keys(d.lens || {})).map(s => `<tr><td class="mono">${esc(s)}</td><td>${(d.lens[s].up || []).map(t => `<span class="tokc">${esc(t)}</span>`).join("")}</td><td>${(d.lens[s].down || []).map(t => `<span class="tokc">${esc(t)}</span>`).join("")}</td></tr>`).join("");
  const nr = d.natural_range;
  const steerRows = (d.steering || []).map(x => `<div class="ex">${settingsHTML(x.state, x.intervention)}<div class="ex-head">prompt ${esc(x.prompt_id)} · KL first token ${f(x.kl_first_token, 4)} · top changes ${(x.top_changes || []).map(([t, v]) => `${esc(JSON.stringify(t))} ${sgn(v)}`).join(", ")}</div><div class="col">${esc(x.text)}</div></div>`).join("");
  const ablRows = (d.ablation || []).map(x => `<tr><td>${settingsHTML(x.state, x.intervention ? { neuron: S.data.id, ...x.intervention } : null)}</td><td>${esc(x.metric)}</td><td class="num">${f(x.value)}</td><td class="num">${f(x.baseline)}</td><td class="num">${esc(x.n)}</td></tr>`).join("");
  el.innerHTML = `
    <h3>Activation across adapter coefficients</h3>
    <div class="dim" style="font-size:12px">z_mean = mean activation change from base on identical tokens / fixed base denominator ${f(r.denominator)}.</div>
    ${trajChart(d.trajectory)}
    <details><summary class="dim">trajectory table</summary>${trajTables}</details>
    <h3>Effects per state (neurons/index.json)</h3>
    <table class="dtable"><tr><th>state</th><th class="num">mean_diff</th><th class="num">rms_diff</th><th class="num">z_mean</th><th class="num">z_rms</th><th class="num">agree</th><th class="num">resid_mean_diff</th></tr>${effRows}</table>
    <h3>By domain (mean_diff from base)</h3>
    ${byDom ? `<table class="dtable"><tr><th>state</th>${doms.map(x => `<th class="num">${esc(x)}</th>`).join("")}</tr>${byDom}</table>` : `<div class="slot">none saved</div>`}
    <h3>By segment (mean_diff from base)</h3>
    ${bySeg ? `<table class="dtable"><tr><th>state</th><th class="num">prompt</th><th class="num">continuation</th></tr>${bySeg}</table>` : `<div class="slot">none saved</div>`}
    <h3>Weight change per state (saved run)</h3>
    ${wRows ? `<table class="dtable"><tr><th>state</th><th class="num">down column norm</th><th class="num">down cos to base</th><th class="num">gate row cos</th><th class="num">up row cos</th></tr>${wRows}</table>` : `<div class="slot">none saved</div>`}
    <h3>Logit lens per state (saved run; effective down column of that state)</h3>
    ${lensRows ? `<table class="dtable"><tr><th>state</th><th class="up-l">promoted by +activation</th><th class="down-l">promoted by −activation</th></tr>${lensRows}</table>` : `<div class="slot">none saved</div>`}
    <div class="live-weights">
      <h3>Live weight views</h3>
      <div class="ctl-box state-box"><div class="box-lab">Model state for weight views</div><select id="lw-state"></select>
        <button class="btn small" id="lw-lens">logit lens</button><button class="btn small" id="lw-con">connections</button>
        <div class="hint">Computed from the effective weights of the chosen state (base + scaled adapter updates). Connections are direct-path weights: candidate paths, not circuits.</div></div>
      <div id="lw-out"></div>
    </div>
    <h3>Natural range and steering units (saved run)</h3>
    ${nr ? `<div class="mono" style="font-size:12.5px">base p01 ${f(nr.p01)} · p99 ${f(nr.p99)} · resid_p50 ${f(nr.resid_p50)} · down norm ${f(nr.down_norm)} · add unit = ${f(nr.add_unit_in_act)} activation</div>` : `<div class="slot">none saved</div>`}
    <h3>Saved steering on held-out prompts</h3>${steerRows || `<div class="slot">none saved</div>`}
    <h3>Saved ablations</h3>${ablRows ? `<table class="dtable"><tr><th>settings</th><th>metric</th><th class="num">value</th><th class="num">baseline</th><th class="num">n</th></tr>${ablRows}</table>` : `<div class="slot">none saved</div>`}`;
  $("#lw-state").innerHTML = stateOptions();
  $("#lw-lens").onclick = () => liveWeights("lens");
  $("#lw-con").onclick = () => liveWeights("connections");
  liveGuard($(".live-weights"));
}
async function liveWeights(kind) {
  const s = $("#lw-state").value, out = $("#lw-out");
  out.innerHTML = `<div class="dim">computing…</div>`;
  try {
    const r = await api(`/api/live/${kind}?neuron=${encodeURIComponent(S.data.id)}&state=${encodeURIComponent(s)}`);
    const head = `<div class="settings"><span class="ms">model state: <b>${esc(r.state)}</b> (${esc(stateLabel(r.state))}), effective weights</span><span class="ni">neuron intervention: none</span></div>`;
    if (kind === "lens") out.innerHTML = head + `<table class="dtable"><tr><th class="up-l">promoted by +activation</th><th class="down-l">promoted by −activation</th></tr><tr><td>${r.up.map(t => `<span class="tokc">${esc(t)}</span>`).join("")}</td><td>${r.down.map(t => `<span class="tokc">${esc(t)}</span>`).join("")}</td></tr></table>`;
    else {
      const tbl = (side, lab) => `<h4>${lab}</h4><table class="dtable"><tr><th>neuron</th><th class="num">z</th><th>sign</th><th></th></tr>${r[side].map(c => `<tr><td class="mono">${esc(c.id)}</td><td class="num">${f(c.z, 2)}</td><td class="${c.sign > 0 ? "up-l" : "down-l"}">${c.sign > 0 ? "+" : "−"}</td><td>${S.rows.some(x => x.id === c.id) ? `<a class="exlink" data-go="${esc(c.id)}">in candidate table</a>` : ""}</td></tr>`).join("")}</table>`;
      out.innerHTML = head + `<div class="hint">|gate·up| through the direct path, z-scored within each layer. Sign = sign(gate·up). Qwen3 RMSNorm weight applied as w (not 1 + w).</div><div class="ctl-row"><div>${tbl("down", "downstream readers (later layers)")}</div><div>${tbl("up", "upstream writers (earlier layers)")}</div></div>`;
      $$("[data-go]", out).forEach(a => a.onclick = () => go(a.dataset.go));
    }
  } catch (e) { out.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}
function stateOptions(sel = "base") {
  return stateList().map(s => `<option value="${esc(s.id)}" ${s.id === sel ? "selected" : ""}>${esc(s.id)}  ${s.id !== s.label ? "· " + esc(s.label) : ""}</option>`).join("");
}

// ---------- card ----------
function renderCard() {
  const card = S.data.card, el = $("#tab-card"), id = S.data.id;
  const prof = S.data.profile ? `<h3>Profile (profiles/${esc(id.replace("/", "_"))}.md)</h3><div class="md">${md(S.data.profile)}</div>` : "";
  if (!card) {
    el.innerHTML = `<div class="slot">No card for this neuron yet.</div>${prof}${editForm(null)}`;
    bindEdit(); return;
  }
  const exLinks = (ids, contra) => (ids || []).map(x => `<a class="exlink${contra ? " contra" : ""}" data-ex="${esc(x)}">${esc(x)}</a>`).join("") || `<span class="dim">none</span>`;
  const vers = card.versions.map(v => `<div class="ver ${esc(v.kind)}">
    <div class="vh"><span>v${esc(v.v)}</span><span>${esc(v.kind)}</span><span>${esc(v.author)}</span><span>${esc(v.created)}</span>${v.blind ? "<span>blind to trait/adapter names</span>" : ""}<span>confidence ${esc(v.confidence || "–")}</span><span>${v.source === "edits" ? "edits/cards" : "neurons/cards"}</span>${v.status_set ? `<span>set status: ${esc(v.status_set)}</span>` : ""}</div>
    <div style="margin-top:4px"><b>${esc(v.label || (v.no_coherent_hypothesis ? "no coherent hypothesis" : ""))}</b> ${autoTag(card, v)}</div>
    <dl>
      ${v.no_coherent_hypothesis ? `<dt>result</dt><dd>no coherent hypothesis</dd>` : ""}
      <dt>activation trigger</dt><dd>${esc(v.trigger_hypothesis)}</dd>
      <dt>output effect</dt><dd>${esc(v.output_hypothesis)}</dd>
      <dt>alternatives</dt><dd>${(v.alternatives || []).map(a => `• ${esc(a)}`).join("<br>") || "–"}</dd>
      <dt>adapter changes</dt><dd>${esc(v.adapter_changes || "–")}</dd>
      <dt>supporting</dt><dd>${exLinks(v.supporting)}</dd>
      <dt>contradicting / unexplained</dt><dd>${exLinks(v.contradicting, true)}</dd>
      <dt>proposed tests</dt><dd>${(v.proposed_tests || []).map((a, i) => `${i + 1}. ${esc(a)}`).join("<br>") || "–"}</dd>
      ${v.note ? `<dt>note</dt><dd>${esc(v.note)}</dd>` : ""}
      ${v.usage ? `<dt>usage</dt><dd class="mono" style="font-size:12px">in ${esc(v.usage.input_tokens)} · out ${esc(v.usage.output_tokens)} · reasoning ${esc(v.usage.reasoning_tokens)} · $${esc(v.usage.cost_usd)}</dd>` : ""}
    </dl></div>`).join("");
  const tests = card.tests.length ? `<table class="dtable"><tr><th>test</th><th>prediction</th><th>texts</th><th>result</th><th>model state</th><th>notes</th></tr>${card.tests.map(t => `<tr><td>${esc(t.test_id)}: ${esc(t.description)}</td><td>${esc(t.prediction)}</td>
      <td>${(t.texts || []).map(x => `<div>${esc(x.text)} <span class="dim mono">expected ${esc(x.expected)}, max ${f(x.measured_max)} at ${esc(JSON.stringify(x.measured_at))}</span></div>`).join("")}</td>
      <td><span class="status ${t.result === "pass" ? "tested" : t.result === "fail" ? "rejected" : "mixed"}">${esc(t.result)}</span></td><td class="mono">${esc(t.state || "")}</td><td>${esc(t.notes || "")}</td></tr>`).join("")}</table>` : `<div class="slot">No tests recorded.</div>`;
  el.innerHTML = `<div class="row"><span class="status ${esc(card.status)}">${esc(card.status)}</span><span class="dim">status from ${esc(card.status_from)}${card.status_from === "human edit" && card.run_status ? ` (run says ${esc(card.run_status)})` : ""}</span></div>
    ${prof}
    <h3>Card versions, latest first</h3>${vers}
    <h3>Tests</h3>${tests}
    <h3>Disagreements</h3>${card.disagreements.length ? `<ul>${card.disagreements.map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : `<div class="slot">none</div>`}
    ${editForm(latestVersion(card), card.status)}`;
  $$("[data-ex]", el).forEach(a => a.onclick = () => showExample(a.dataset.ex));
  bindEdit();
}
function editForm(v, status = "untested") {
  v = v || {};
  const ta = (name, val, rows = 2) => `<textarea name="${name}" rows="${rows}">${esc(val || "")}</textarea>`;
  return `<h3>Edit card</h3><div class="hint">Saved as a new version (kind human_edit) in edits/cards/. The automated card file is never changed.</div>
  <form class="edit" id="edit-form">
    <label>status</label><select name="status">${S.meta.statuses.map(s => `<option ${s === status ? "selected" : ""}>${s}</option>`).join("")}</select>
    <label>author</label><input name="author" value="human">
    <label>label</label><input name="label" value="${esc(v.label || "")}">
    <label>activation trigger</label>${ta("trigger_hypothesis", v.trigger_hypothesis)}
    <label>output effect</label>${ta("output_hypothesis", v.output_hypothesis)}
    <label>alternatives (one per line)</label>${ta("alternatives", (v.alternatives || []).join("\n"))}
    <label>adapter changes</label>${ta("adapter_changes", v.adapter_changes)}
    <label>supporting example ids</label><input name="supporting" value="${esc((v.supporting || []).join(", "))}">
    <label>contradicting example ids</label><input name="contradicting" value="${esc((v.contradicting || []).join(", "))}">
    <label>proposed tests (one per line)</label>${ta("proposed_tests", (v.proposed_tests || []).join("\n"), 3)}
    <label>confidence</label><select name="confidence">${["low", "medium", "high"].map(c => `<option ${c === v.confidence ? "selected" : ""}>${c}</option>`).join("")}</select>
    <label>no coherent hypothesis</label><input type="checkbox" name="no_coherent_hypothesis" ${v.no_coherent_hypothesis ? "checked" : ""} style="width:auto">
    <label>note on this edit</label>${ta("note", "")}
    <label>add disagreement</label>${ta("disagreement", "", 1)}
    <span></span><div class="row"><button class="btn savebtn" type="submit">save new version</button><span id="edit-msg" class="dim"></span></div>
  </form>`;
}
function bindEdit() {
  const form = $("#edit-form"); if (!form) return;
  form.onsubmit = async e => {
    e.preventDefault();
    const fd = new FormData(form), body = { id: S.data.id };
    for (const [k, v] of fd.entries()) body[k] = v;
    body.no_coherent_hypothesis = form.no_coherent_hypothesis.checked;
    try {
      S.data.card = await api("/api/card", body);
      S.rows = await api("/api/neurons");
      renderList(); renderNeuron(); setTab("card");
      $("#edit-msg").textContent = `saved v${S.data.card.versions[0].v}`;
    } catch (err) { $("#edit-msg").textContent = err.message; }
  };
}
function showExample(exId) {
  S.ex.kind = "all"; S.ex.split = "all";
  renderExamples(); setTab("examples"); writeHash();
  const el = document.getElementById(`ex-${exId}`);
  if (!el) { $("#tab-examples").insertAdjacentHTML("afterbegin", `<div class="notice">Example ${esc(exId)} is not in this neuron's saved example list.</div>`); return; }
  el.scrollIntoView({ block: "center" });
  el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 2000);
}

// minimal markdown: headings, lists, code fences, paragraphs, inline code / bold / italic / links
function mdInline(t) {
  return esc(t).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/\*([^*]+)\*/g, "<i>$1</i>")
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}
function md(src) {
  const out = []; let list = null, code = null, para = [];
  const flush = () => { if (para.length) { out.push(`<p>${mdInline(para.join(" "))}</p>`); para = []; } };
  for (const line of src.split("\n")) {
    if (code !== null) { if (line.startsWith("```")) { out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`); code = null; } else code.push(line); continue; }
    if (line.startsWith("```")) { flush(); code = []; continue; }
    const h = line.match(/^(#{1,4})\s+(.*)/), li = line.match(/^\s*([-*]|\d+\.)\s+(.*)/);
    if (li) { flush(); const tag = /\d/.test(li[1]) ? "ol" : "ul"; if (list !== tag) { if (list) out.push(`</${list}>`); out.push(`<${tag}>`); list = tag; } out.push(`<li>${mdInline(li[2])}</li>`); continue; }
    if (list) { out.push(`</${list}>`); list = null; }
    if (h) { flush(); out.push(`<h${h[1].length}>${mdInline(h[2])}</h${h[1].length}>`); }
    else if (!line.trim()) flush();
    else para.push(line);
  }
  flush(); if (list) out.push(`</${list}>`); if (code) out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
  return out.join("");
}

// ---------- saved ----------
function renderSaved() {
  const rows = S.data.saved || [], el = $("#tab-saved");
  el.innerHTML = `<div class="hint">Live results saved for this neuron (runs/${esc(S.man.run_id)}/edits/saved.jsonl), latest first.</div>` + (rows.map(r => `<div class="ex"><div class="ex-head"><span class="kind">${esc(r.kind)}</span><span>${esc(r.saved)}</span></div>
    ${settingsHTML(r.state, r.intervention)}
    ${r.input ? `<div class="dim" style="font-size:12.5px">input: ${esc(typeof r.input === "string" ? r.input : JSON.stringify(r.input))}</div>` : ""}
    <div class="col">${esc(typeof r.output === "string" ? r.output : JSON.stringify(r.output).slice(0, 1500))}</div></div>`).join("") || `<div class="slot">Nothing saved yet.</div>`);
}
async function saveRec(rec, btn) {
  try {
    await api("/api/save", { neuron: S.data.id, model: S.live.info?.model, revision: S.live.info?.revision, ...rec });
    if (btn) { btn.textContent = "saved"; btn.disabled = true; }
    S.data.saved = await api(`/api/saved?neuron=${encodeURIComponent(S.data.id)}`);
    renderSaved();
  } catch (e) { if (btn) btn.textContent = e.message; }
}

// ---------- live panels ----------
function stateChecks(el, defaults) {
  const t = S.trait, o = otherTrait(t), tr = S.man.traits || {};
  const groups = [
    ["base", ["base"]],
    [`${t} poles`, tr[t]?.poles || []],
    [`${o} poles`, tr[o]?.poles || []],
    ["control", S.man.control_state ? [S.man.control_state] : []],
    [`${t} sweep`, (tr[t]?.sweep || []).filter(s => s !== "base")],
    [`${o} sweep`, (tr[o]?.sweep || []).filter(s => s !== "base")],
  ];
  const prev = new Set($$("input:checked", el).map(x => x.value));
  const use = prev.size && el.dataset.trait === t ? prev : new Set(defaults);
  el.dataset.trait = t;
  const seen = new Set();
  for (const g of groups) { g[1] = g[1].filter(x => !seen.has(x)); g[1].forEach(x => seen.add(x)); }
  el.innerHTML = groups.filter(g => g[1].length).map(([lab, ss]) => `<span class="grp">${esc(lab)}</span>${ss.map(s => `<label title="${esc(stateLabel(s))}"><input type="checkbox" value="${esc(s)}" ${use.has(s) ? "checked" : ""}>${esc(s)}</label>`).join("")}`).join("");
}
const checked = el => $$("input:checked", el).map(x => x.value);

let livePanelsBound = false;
function setupLivePanels() {
  const t = S.trait, tr = S.man.traits || {};
  stateChecks($("#cu-states"), ["base", ...(tr[t]?.poles || [])]);
  stateChecks($("#ch-states"), ["base", ...(tr[t]?.poles || []), S.man.control_state].filter(Boolean));
  const prompts = S.meta.prompts || [];
  const popts = `<option value="">saved behavior prompts…</option>` + [...prompts].sort((a, b) => (a.trait === t ? 0 : 1) - (b.trait === t ? 0 : 1)).map(p => `<option value="${esc(p.text)}">${esc(p.id)} · ${esc(p.text.slice(0, 90))}</option>`).join("");
  $("#ch-prompt").innerHTML = popts; $("#st-prompt").innerHTML = popts;
  const cur = $("#st-state").value || "base";
  $("#st-state").innerHTML = stateOptions(cur);
  if (livePanelsBound) { refreshCalib(); return; }
  livePanelsBound = true;
  $("#ch-prompt").onchange = e => { if (e.target.value) $("#ch-msg").value = e.target.value; };
  $("#st-prompt").onchange = e => { if (e.target.value) { $("#st-msg").value = e.target.value; $("#nt-text").value ||= e.target.value; } };
  $("#cu-go").onclick = runCustom;
  $("#ch-go").onclick = runChats;
  $("#st-go").onclick = runSteer;
  $("#nt-go").onclick = runNext;
  $$("#st-mode button").forEach(b => b.onclick = () => {
    S.steer.value[S.steer.mode] = +$("#st-value").value || S.steer.value[S.steer.mode];
    S.steer.mode = b.dataset.m;
    $$("#st-mode button").forEach(x => x.classList.toggle("on", x === b));
    showDose();
  });
  $("#st-state").onchange = () => { $("#st-state-custom").value = ""; refreshCalib(); };
  $("#st-state-custom").onchange = refreshCalib;
  showDose();
  refreshCalib();
}
function steerState() { return $("#st-state-custom").value.trim() || $("#st-state").value || "base"; }
function showDose() {
  const m = S.steer.mode;
  $("#st-value").disabled = m === "zero";
  $("#st-value").value = m === "zero" ? "0" : S.steer.value[m];
  $("#st-unit").textContent = m === "add" ? "× resid_p50 / |d_base|  (±, residual-relative)" : m === "mul" ? "× activation (columns: ×v and ×−v)" : "activation set to 0";
}
let calib = null, calibSeq = 0;
async function refreshCalib() {
  const el = $("#st-calib"), seq = ++calibSeq;
  if (!S.data || !S.live.loaded) { el.textContent = "Units: needs the live model. 1 add unit = base-state median residual norm on a fixed reference text / base down-column norm."; return; }
  try {
    const c = await api(`/api/live/calibration?neuron=${encodeURIComponent(S.data.id)}&state=${encodeURIComponent(steerState())}`);
    if (seq !== calibSeq) return;
    calib = c;
    const nr = S.data.detail?.natural_range, b = S.data.index?.base;
    const p01 = nr?.p01 ?? b?.q?.p01, p99 = nr?.p99 ?? b?.q?.p99;
    el.innerHTML = `1 add unit = resid_p50 ${f(calib.resid_p50, 4)} (L${calib.layer}, ${esc(calib.reference)}, ${calib.reference_tokens} tokens) / base down-column norm ${f(calib.down_norm_base, 4)} = <b>${f(calib.unit_act, 4)}</b> activation. Same unit in every model state.
      Down-column norm in ${esc(calib.state)}: ${f(calib.down_norm_state, 4)}.${p01 != null ? ` Saved base range p01 ${f(p01)} … p99 ${f(p99)} = ${f(p01 / calib.unit_act, 3)} … ${f(p99 / calib.unit_act, 3)} units.` : ""}`;
  } catch (e) { el.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
}

async function runCustom() {
  const out = $("#cu-out"), text = $("#cu-text").value.trim(), states = checked($("#cu-states"));
  if (!text) { out.innerHTML = `<div class="err">Enter some text.</div>`; return; }
  if (!states.length) { out.innerHTML = `<div class="err">Choose at least one model state.</div>`; return; }
  out.innerHTML = `<div class="dim">reading in ${states.length} state(s)…</div>`;
  try {
    const r = await api("/api/live/read", { neuron: S.data.id, text, states: stateOrder(states), chat: $("#cu-chat").checked });
    const rows = r.states.map(s => ({ label: s, title: r.labels[s], acts: r.acts[s] }));
    out.innerHTML = `${settingsHTML(r.states, null)}<div class="hint">${r.n_tokens} tokens${r.chat ? ", chat template" : ", raw text"} · neuron ${esc(r.neuron)}</div>
      <label class="dim"><input type="checkbox" id="cu-diff" ${S.ex.diff ? "checked" : ""}> color other rows by change from base</label>${legendHTML()}<div class="ex" id="cu-panel"></div>
      <button class="btn small savebtn" id="cu-save">save</button>`;
    // diffBase is compared by identity with the base row's array
    const drawDiff = () => {
      const rr = rows.map(x => ({ ...x }));
      const base = rr.find(x => x.label === "base");
      $("#cu-panel").innerHTML = tokPanel(r.tokens, rr, { diffBase: $("#cu-diff").checked && base ? base.acts : null });
    };
    drawDiff();
    $("#cu-diff").onchange = drawDiff;
    $("#cu-save").onclick = async e => {
      for (const s of r.states) await saveRec({ kind: "read", state: s, intervention: null, input: { text, chat: r.chat }, output: { tokens: r.tokens, acts: r.acts[s] } });
      e.target.textContent = "saved"; e.target.disabled = true;
    };
  } catch (e) { out.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}

let chatCtl = null;
async function runChats() {
  const out = $("#ch-out"), msg = $("#ch-msg").value.trim(), states = stateOrder(checked($("#ch-states")));
  if (chatCtl) { chatCtl.abort(); return; }
  if (!msg || !states.length) { out.innerHTML = `<div class="err">Enter a message and choose at least one model state.</div>`; return; }
  const max_new = +$("#ch-max").value || 128;
  chatCtl = new AbortController(); $("#ch-go").textContent = "stop";
  const texts = states.map(() => "");
  out.innerHTML = `<div class="user-msg dim">user: ${esc(msg)}</div><div class="cols" style="grid-template-columns:repeat(${Math.min(states.length, 4)},1fr)">${states.map((s, i) => `<div class="col live" data-c="${i}"><span class="chead">${esc(s)}</span>${settingsHTML(s, null)}<span class="body"></span></div>`).join("")}</div><div class="err" id="ch-err"></div>`;
  const cols = $$(".col", out);
  try {
    await stream("/api/live/chat", { message: msg, states, max_new }, ev => {
      if (ev.error) $("#ch-err").textContent = ev.error;
      if (ev.col == null) return;
      if (ev.done) {
        cols[ev.col].classList.remove("live");
        const b = document.createElement("button"); b.className = "btn small savebtn"; b.textContent = "save";
        const i = ev.col;
        b.onclick = () => saveRec({ kind: "chat", state: states[i], intervention: null, input: msg, output: texts[i], decoding: { do_sample: false, max_new_tokens: max_new } }, b);
        cols[i].append(b);
        return;
      }
      texts[ev.col] += ev.t; cols[ev.col].querySelector(".body").textContent = texts[ev.col];
    }, chatCtl.signal);
  } catch (e) { if (e.name !== "AbortError") $("#ch-err").textContent = e.message; }
  finally { cols.forEach(c => c.classList.remove("live")); chatCtl = null; $("#ch-go").textContent = "send"; }
}

function colTitle(iv, state) {
  if (!iv) return state === "base" ? "unmodified (base model)" : `unmodified (model state ${state}, adapter active)`;
  if (iv.mode === "zero") return "zeroed";
  if (iv.mode === "mul") return `× ${f(iv.value)}`;
  return `add ${sgn(iv.value)} resid_p50`;
}
function colClass(iv) { return !iv ? "" : iv.mode === "zero" ? "zeroed" : iv.value > 0 ? "up" : "down"; }
let steerCtl = null;
async function runSteer() {
  const out = $("#st-out"), msg = $("#st-msg").value.trim();
  if (steerCtl) { steerCtl.abort(); return; }
  if (!msg) { out.innerHTML = `<div class="err">Enter a message.</div>`; return; }
  const mode = S.steer.mode, value = +$("#st-value").value, state = steerState(), max_new = +$("#st-max").value || 128;
  if (mode !== "zero") S.steer.value[mode] = value;
  steerCtl = new AbortController(); $("#st-go").textContent = "stop";
  out.innerHTML = `<div class="dim">running…</div>`;
  let cols = [], texts = [], info = null;
  try {
    await stream("/api/live/steer", { neuron: S.data.id, state, mode, value, message: msg, max_new }, ev => {
      if (ev.error) out.insertAdjacentHTML("beforeend", `<div class="err">${esc(ev.error)}</div>`);
      if (ev.columns) {
        info = ev; texts = ev.columns.map(() => "");
        const c = ev.calibration;
        out.innerHTML = `<div class="user-msg dim">user: ${esc(msg)}</div><div class="hint">add unit here: ${f(c.unit_act, 4)} activation (resid_p50 ${f(c.resid_p50, 4)} base reference / base down norm ${f(c.down_norm_base, 4)})</div>
          <div class="cols" style="grid-template-columns:repeat(${ev.columns.length},1fr)">${ev.columns.map((col, i) => `<div class="col live ${colClass(col.intervention)}"><span class="chead">${esc(colTitle(col.intervention, ev.state))}</span>${settingsHTML(ev.state, col.intervention)}<span class="body"></span></div>`).join("")}</div>`;
        cols = $$(".col", out);
        return;
      }
      if (ev.col != null) { texts[ev.col] += ev.t; cols[ev.col].querySelector(".body").textContent = texts[ev.col]; }
    }, steerCtl.signal);
  } catch (e) { if (e.name !== "AbortError") out.insertAdjacentHTML("beforeend", `<div class="err">${esc(e.message)}</div>`); }
  finally {
    cols.forEach((c, i) => {
      c.classList.remove("live");
      const b = document.createElement("button"); b.className = "btn small savebtn"; b.textContent = "save";
      b.onclick = () => saveRec({ kind: "steer_chat", state: info.state, intervention: info.columns[i].intervention, input: msg, output: texts[i], calibration: info.calibration, decoding: { do_sample: false, max_new_tokens: max_new } }, b);
      c.append(b);
    });
    steerCtl = null; $("#st-go").textContent = "run";
  }
}

async function runNext() {
  const out = $("#nt-out"), text = $("#nt-text").value.trim();
  if (!text) { out.innerHTML = `<div class="err">Enter a prompt.</div>`; return; }
  const mode = S.steer.mode, value = +$("#st-value").value, state = steerState();
  out.innerHTML = `<div class="dim">computing…</div>`;
  try {
    const r = await api("/api/live/next", { neuron: S.data.id, state, mode, value, text, chat: $("#nt-chat").checked, prefix: $("#nt-prefix").value });
    const toks = xs => xs.map(([t, p]) => `<span class="tokc">${esc(t)}</span><span class="dim mono" style="font-size:11px">${f(p, 3)}</span> `).join("");
    const chg = xs => xs.map(([t, d, p]) => `<span class="tokc">${esc(t)}</span><span class="dim mono" style="font-size:11px">${sgn(d, 2)} → ${f(p, 3)}</span> `).join("") || `<span class="dim">none</span>`;
    out.innerHTML = `<div class="hint">${r.n_tokens} prompt tokens${r.prefix ? `, reply prefix ${esc(JSON.stringify(r.prefix))}` : ""} · Δ = log-prob change · KL(unmodified ‖ intervened)</div>
      <div class="ex"><div class="ex-head"><span class="kind">${esc(colTitle(null, r.state))}</span></div>${settingsHTML(r.state, null)}<div>top: ${toks(r.unmodified.top)}</div></div>
      ${r.variants.map(v => `<div class="ex"><div class="ex-head"><span class="kind">${esc(colTitle(v.intervention, r.state))}</span><span>KL ${f(v.kl, 5)}</span></div>${settingsHTML(r.state, v.intervention)}
        <div>top: ${toks(v.top)}</div><div><span class="up-l">up</span> ${chg(v.up)}</div><div><span class="down-l">down</span> ${chg(v.down)}</div></div>`).join("")}
      <button class="btn small savebtn" id="nt-save">save</button>`;
    $("#nt-save").onclick = async e => {
      await saveRec({ kind: "next_token", state: r.state, intervention: null, input: { text, prefix: r.prefix }, output: r.unmodified });
      for (const v of r.variants) await saveRec({ kind: "next_token", state: r.state, intervention: v.intervention, input: { text, prefix: r.prefix }, output: v, calibration: r.calibration });
      e.target.textContent = "saved"; e.target.disabled = true;
    };
  } catch (e) { out.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}
