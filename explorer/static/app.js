// Persona neuron explorer. Three layers: a Start page, a plain neuron list, and a neuron page that starts plain
// (runs/<id>/plain.json, docs/plain_format.md) and keeps every technical view in collapsed sections below.
// Without plain.json the same pages fall back to saved run data.
// Two controls are kept apart everywhere: "model state" (base or scaled adapters) and "neuron intervention"
// (add / multiply / zero one activation inside that state). Every result prints both.

const $ = s => document.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : (typeof x === "number" ? (Math.abs(x) >= 1000 ? x.toFixed(0) : +x.toFixed(d) + "") : String(x));
const sgn = (x, d = 3) => (x > 0 ? "+" : x < 0 ? "−" : "") + f(Math.abs(x), d);
// plain layers: 2 significant figures, real minus sign
function sig2(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  const v = Number(Number(x).toPrecision(2));
  if (v === 0) return "0";
  const a = Math.abs(v), s = a >= 1000 ? Math.round(a).toLocaleString("en-US") : String(a);
  return (v < 0 ? "−" : "") + s;
}
const fx = r => r == null ? "–" : `${r.toFixed(2)}×`;

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
  meta: null, man: null, plain: null, trait: "openness", rows: [], sel: null, data: null, view: "start",
  sort: { key: "rank", dir: -1 }, filter: "", tech: false, groupBy: "plain",
  ex: { kind: "all", split: "all", diff: false },
  steer: { mode: "add", value: { add: 0.5, mul: 2 } },
  live: { loaded: false, status: "", error: null },
};
const GROUP_NAMES = {
  shared: "shared: both poles", amp_primary: "amplifier-primary", sup_primary: "suppressor-primary",
  cross_trait: "cross-trait", control_assoc: "neutral-control associated", surprising: "surprising",
  random_control: "random controls (layer-matched)", other: "not selected for this trait",
};
// plain fallbacks, used only when plain.json is missing or lacks an entry
const ADAPTER_ORDER = ["o_amp+1", "o_sup+1", "n_amp+1", "n_sup+1", "ctrl+1"];
const DEFAULT_ADAPTERS = [
  { state: "o_amp+1", name: "Openness amplifier", short: "Openness +", trait: "openness", what: "trained to make answers more curious, imaginative and open to new ideas" },
  { state: "o_sup+1", name: "Openness suppressor", short: "Openness −", trait: "openness", what: "trained to make answers more conventional and closed" },
  { state: "n_amp+1", name: "Neuroticism amplifier", short: "Neuroticism +", trait: "neuroticism", what: "trained to make the model more anxious and emotionally reactive" },
  { state: "n_sup+1", name: "Neuroticism suppressor", short: "Neuroticism −", trait: "neuroticism", what: "trained to make the model calmer and steadier" },
  { state: "ctrl+1", name: "Neutral control", short: "Neutral", trait: null, what: "trained the same way but with no personality goal; a comparison" },
];
const DEFAULT_GLOSSARY = {
  neuron: "One of about 440,000 small units inside the model. Each produces a number for every word it reads; a large number (positive or negative) means it is responding.",
  adapter: "A small add-on trained to change the model's behavior, here its personality. It can be switched on, off, or dialed up and down.",
  amplifier: "The adapter trained to push a trait up (more open, more anxious).",
  suppressor: "The adapter trained to push a trait down (more closed, calmer). It is trained separately, not the amplifier run backwards.",
  "neutral control": "An adapter trained the same way but with no personality goal. Changes it also makes are probably about fine-tuning in general.",
  activation: "The number a neuron produces at a given word. The sign says which way it fires; the size says how strongly.",
  "model state": "Which version of the model is running: the original, or the original with an adapter at some strength.",
  layer: "The model is a stack of 36 layers; early layers see raw words, later layers more processed meaning.",
  "chat template": "Hidden marker tokens that tell the model where a user message ends and the assistant's reply begins.",
  "reply start": "The few hidden marker tokens just before the assistant's first word. Many adapter changes concentrate there.",
};
const GL_ALIASES = {
  neuron: ["neurons", "neuron"], adapter: ["adapters", "adapter"], amplifier: ["amplifiers", "amplifier"],
  suppressor: ["suppressors", "suppressor"], "neutral control": ["neutral control", "neutral adapter"],
  activation: ["activations", "activation"], "model state": ["model states", "model state"], layer: ["layers", "layer"],
  "chat template": ["chat template", "chat format"],
  "reply start": ["reply start", "the moment the assistant's reply begins", "the moment the assistant’s reply begins", "start of the assistant's reply", "start of a reply", "start of replies"],
};
const GROUP_PLAIN = {
  shared: "Changed by both the “more” and “less” adapters", amp_primary: "Changed mainly by the amplifier",
  sup_primary: "Changed mainly by the suppressor", cross_trait: "Changed by adapters of both traits",
  control_assoc: "Changed as much by the neutral adapter", surprising: "Surprising cases",
};
const RANDOM_GROUP = "Picked at random, for comparison";
const CHECK_WORDS = { tested: "held up", mixed: "partly held up", rejected: "didn't hold up", untested: "not checked" };
const DIR = {
  stronger: { a: "▲", cls: "stronger", w: "stronger" }, weaker: { a: "▼", cls: "weaker", w: "weaker" },
  "about the same": { a: "·", cls: "same", w: "about the same" }, unknown: { a: "–", cls: "same", w: "not measured" },
};
const dirOfRatio = r => r == null ? "unknown" : r >= 1.1 ? "stronger" : r <= 0.9 ? "weaker" : "about the same";

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

// ---------- plain helpers ----------
const plainOf = id => S.plain?.neurons?.[id] || null;
function adapters() {
  const m = {};
  for (const a of DEFAULT_ADAPTERS) m[a.state] = a;
  for (const a of (S.plain?.adapters || [])) m[a.state] = { ...m[a.state], ...a };
  return m;
}
function stateName(s) {
  if (s === "base") return "Original model";
  return adapters()[s]?.name || stateLabel(s);
}
function glossary() { return { ...DEFAULT_GLOSSARY, ...(S.plain?.glossary || {}) }; }
// one row per adapter in fixed order: from plain.json, else computed from saved example windows
function effectsFor(row, P) {
  const byState = Object.fromEntries((P?.adapter_effects || []).map(e => [e.state, e]));
  const order = [...ADAPTER_ORDER, ...Object.keys(byState).filter(s => !ADAPTER_ORDER.includes(s))];
  return order.map(s => {
    if (byState[s]) return { ...byState[s], computed: false };
    const r = row?.example_ratio?.[s];
    if (r == null && !row?.effects?.[s]) return null;
    const d = dirOfRatio(r);
    return { state: s, ratio: r, direction: d, computed: true,
      text: r == null ? "not measured" : d === "about the same" ? `about the same at its usual peaks (${r.toFixed(2)}x)` : `fires about ${r.toFixed(1)}x as hard at its usual peaks` };
  }).filter(Boolean);
}
function statusOf(row, P) {
  if (row?.status_from === "human edit") return row.card_status;
  return P?.checked?.status || row?.card_status || "untested";
}
function traitOf(row, P) {
  if (P?.trait) return P.trait;
  const t = (row?.selected_for || []).find(x => x === "openness" || x === "neuroticism");
  if (t) return t;
  const g = Object.entries(row?.group || {}).find(([, v]) => v);
  if (g) return g[0];
  const effs = effectsFor(row, P).filter(e => e.ratio != null && adapters()[e.state]?.trait);
  effs.sort((a, b) => Math.abs(Math.log(b.ratio)) - Math.abs(Math.log(a.ratio)));
  return effs[0] ? adapters()[effs[0].state].trait : S.trait;
}
function groupPlainOf(row, P) {
  if (P?.group_plain) return P.group_plain;
  if (row.kind === "random_control") return RANDOM_GROUP;
  const t = traitOf(row, P), g = (row.group || {})[t];
  return GROUP_PLAIN[g] || "Other";
}
const nameOf = (row, P) => P?.nickname || row?.id;
function oneLinerOf(row, P) {
  if (P?.one_liner) return { text: P.one_liner, auto: P.source === "automated" };
  return { text: row?.label || "", auto: true };
}
const isFeatured = (row, P) => !!(P?.featured ?? row?.has_profile);
function biggestLog(row, P) {
  return Math.max(0, ...effectsFor(row, P).filter(e => e.ratio > 0).map(e => Math.abs(Math.log(e.ratio))));
}
function checkBadge(status) { return `<span class="check ${esc(status)}">${esc(CHECK_WORDS[status] || status)}</span>`; }
function dirHTML(d, extra = "") {
  const x = DIR[d] || DIR.unknown;
  return `<span class="dir ${x.cls}"><span class="arrow" aria-hidden="true">${x.a}</span>${x.w}${extra}</span>`;
}
// log-scale bar around 1.0x, clamped to 0.5x .. 2x
function ratioBar(r, d) {
  if (r == null || !(r > 0)) return "";
  const x = Math.max(-1, Math.min(1, Math.log2(r)));
  const cls = (DIR[d] || DIR.unknown).cls, left = x < 0 ? 50 + x * 50 : 50, w = Math.max(1, Math.abs(x) * 50);
  return `<span class="rbar" aria-hidden="true"><span class="track"></span><span class="fill ${cls}" style="left:${left}%;width:${w}%"></span><span class="mid"></span></span>`;
}

// ---------- colors ----------
function actColor(a, m) {
  if (!m || !a) return "transparent";
  const x = Math.min(1, Math.abs(a) / m) * 0.75;
  return a > 0 ? `rgba(var(--up-rgb),${x.toFixed(3)})` : `rgba(var(--down-rgb),${x.toFixed(3)})`;
}
function zColor(z, cap = 2) {
  if (z === null || z === undefined) return "transparent";
  const x = Math.min(1, Math.abs(z) / cap) * 0.7;
  return z > 0 ? `rgba(var(--up-rgb),${x.toFixed(3)})` : `rgba(var(--down-rgb),${x.toFixed(3)})`;
}

// ---------- template tokens ----------
const isWS = t => /^\s+$/.test(t || "");
const isSpecial = t => /^<\|[^|]+\|>$/.test(t || "") || t === "<think>" || t === "</think>";
// runs of chat-template tokens -> {end (exclusive), label}; keyed by start index
function templateRuns(toks) {
  const runs = {}, n = toks.length;
  const thinkBlock = j => {
    if (toks[j] === "<think>") { j++; while (j < n && isWS(toks[j])) j++; }
    if (toks[j] === "</think>") { j++; if (j < n && isWS(toks[j])) j++; }
    return j;
  };
  for (let i = 0; i < n; i++) {
    const t = toks[i];
    if (t === "<|im_start|>") {
      let j = i + 1, role = null;
      if (/^(assistant|user|system|tool)$/.test(toks[j] || "")) role = toks[j++];
      if (j < n && isWS(toks[j]) && toks[j].includes("\n")) j++;
      if (role === "assistant" || role === null) j = thinkBlock(j);
      runs[i] = { end: j, label: role === "assistant" ? "start of assistant reply" : role ? `start of ${role} message` : "start of message" };
      i = j - 1;
    } else if (t === "<|im_end|>") {
      let j = i + 1;
      if (j < n && isWS(toks[j]) && toks[j].includes("\n")) j++;
      runs[i] = { end: j, label: "end of message" };
      i = j - 1;
    } else if (t === "<think>" || t === "</think>") {
      const j = thinkBlock(i);
      runs[i] = { end: Math.max(j, i + 1), label: "start of assistant reply" };
      i = Math.max(j, i + 1) - 1;
    } else if (isSpecial(t)) {
      runs[i] = { end: i + 1, label: t.replace(/^<\||\|>$/g, "") };
    }
  }
  return runs;
}
function tplPill(toks, i, run, focus) {
  return `<span class="tpl${focus ? " focus" : ""}" title="${esc("hidden chat-format tokens: " + toks.slice(i, run.end).map(t => JSON.stringify(t)).join(" "))}">[${esc(run.label)}]</span>`;
}
// key-word span: the focus token plus attached sub-word pieces ("forget" + "ful")
function wordSpan(toks, fi) {
  let a = fi, b = fi;
  while (a > 0 && /^[A-Za-z]/.test(toks[a]) && /[A-Za-z]$/.test(toks[a - 1] || "") && !isSpecial(toks[a - 1])) a--;
  while (b + 1 < toks.length && /^[A-Za-z]/.test(toks[b + 1]) && /[A-Za-z]$/.test(toks[b])) b++;
  return [a, b];
}
// one readable line: template runs as grey pills, newlines as a small mark, the key word highlighted
function sentenceHTML(e) {
  const toks = e.tokens || [], runs = templateRuns(toks), fi = e.focus;
  let inRun = null;
  for (const [s, r] of Object.entries(runs)) if (fi >= +s && fi < r.end) inRun = +s;
  const [a, b] = fi != null && inRun == null ? wordSpan(toks, fi) : [-1, -1];
  const out = ["<span class=\"faint\">…</span>"];
  let key = inRun != null ? `[${runs[inRun].label}]` : toks.slice(a, b + 1).join("").trim();
  for (let i = 0; i < toks.length; i++) {
    if (runs[i]) { out.push(tplPill(toks, i, runs[i], inRun === i)); i = runs[i].end - 1; continue; }
    const t = toks[i];
    let h = isWS(t) && t.includes("\n") ? `<span class="nl"> ↵ </span>` : esc(t.replace(/\n+/g, " ↵ "));
    if (i === a) { const lead = h.match(/^\s*/)[0]; out.push(lead + "<mark>"); h = h.slice(lead.length); }
    out.push(h);
    if (i === b) out.push("</mark>");
  }
  out.push("<span class=\"faint\">…</span>");
  return { html: out.join(""), key };
}
// compact display of one token inside the multi-row panels
const TT_SHORT = { "<|im_start|>": "start", "<|im_end|>": "end", "<think>": "think", "</think>": "/think", "<|endoftext|>": "eot" };
function tokDisplay(toks, i) {
  const t = toks[i];
  if (isSpecial(t)) return { txt: TT_SHORT[t] || t.replace(/^<\||\|>$/g, ""), cls: "tt" };
  if (toks[i - 1] === "<|im_start|>" && /^(assistant|user|system|tool)$/.test(t)) return { txt: t, cls: "tt" };
  if (isWS(t) && t.includes("\n")) return { txt: "↵".repeat(Math.min(2, (t.match(/\n/g) || []).length)), cls: "tt" };
  return { txt: t.replace(/\n+/g, "↵"), cls: "" };
}

// ---------- markdown (headings, lists, tables, images, code, inline) ----------
function fixSrc(src) {
  const m = src.match(/(?:^|\/)figures\/([A-Za-z0-9_.-]+)$/);
  return m ? `/figures/${m[1]}` : src;
}
function mdInline(t) {
  const codes = [];
  let s = esc(t).replace(/`([^`]+)`/g, (m, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (m, alt, src) => `<img alt="${alt}" src="${esc(fixSrc(src))}" loading="lazy">`)
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^\w*])\*([^*\s][^*]*?)\*(?![\w*])/g, "$1<i>$2</i>")
    .replace(/\[([^\]]+)\]\(((?:https?:|#|\/)[^)\s]*)\)/g, (m, txt, href) => /^https?:/.test(href) ? `<a href="${href}" target="_blank" rel="noopener">${txt}</a>` : `<a href="${href}">${txt}</a>`);
  return s.replace(/\u0000(\d+)\u0000/g, (m, k) => `<code>${codes[+k]}</code>`);
}
function mdTable(lines) {
  const cells = l => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim());
  const head = cells(lines[0]), body = lines.slice(lines[1] && /^\s*\|?\s*:?-{2,}/.test(lines[1]) ? 2 : 1).map(cells);
  const isNum = c => /^[−\-+~]?\d[\d.,]*\s*(%|x|×)?$/.test(c);
  const numCol = head.map((_, k) => body.length && body.every(r => !r[k] || isNum(r[k]) || r[k] === "–" || r[k] === "-"));
  return `<div class="tablewrap"><table class="t compact"><thead><tr>${head.map((c, k) => `<th class="${numCol[k] ? "num" : ""}">${mdInline(c)}</th>`).join("")}</tr></thead><tbody>${body.map(r => `<tr>${head.map((_, k) => `<td class="${numCol[k] ? "num" : ""}">${mdInline(r[k] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}
function md(src) {
  const out = []; let list = null, code = null, para = [], table = [];
  const flush = () => {
    if (para.length) {
      const img = para.length === 1 && para[0].trim().match(/^!\[([^\]]*)\]\(([^)\s]+)\)$/);
      out.push(img ? `<figure><img alt="${esc(img[1])}" src="${esc(fixSrc(img[2]))}" loading="lazy"><figcaption>${mdInline(img[1])}</figcaption></figure>` : `<p>${mdInline(para.join(" "))}</p>`);
      para = [];
    }
    if (table.length) { out.push(mdTable(table)); table = []; }
  };
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const line of String(src || "").split("\n")) {
    if (code !== null) { if (line.startsWith("```")) { out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`); code = null; } else code.push(line); continue; }
    if (line.startsWith("```")) { flush(); closeList(); code = []; continue; }
    if (/^\s*\|/.test(line)) { if (para.length) { const t = table; table = []; flush(); table = t; } closeList(); table.push(line); continue; }
    if (table.length) flush();
    const h = line.match(/^(#{1,4})\s+(.*)/), li = line.match(/^\s*([-*]|\d+\.)\s+(.*)/);
    if (li) { flush(); const tag = /\d/.test(li[1]) ? "ol" : "ul"; if (list !== tag) { closeList(); out.push(`<${tag}>`); list = tag; } out.push(`<li>${mdInline(li[2])}</li>`); continue; }
    if (list && /^\s{2,}\S/.test(line)) { out[out.length - 1] = out[out.length - 1].replace(/<\/li>$/, " " + mdInline(line.trim()) + "</li>"); continue; }
    closeList();
    if (h) { flush(); out.push(`<h${h[1].length}>${mdInline(h[2])}</h${h[1].length}>`); }
    else if (!line.trim()) flush();
    else para.push(line);
  }
  flush(); closeList(); if (code) out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
  return out.join("");
}

// ---------- glossary: dotted underline + hover definition, first use per section ----------
function glossify(root, used = new Set()) {
  if (!root) return used;
  const G = glossary(), alias = {};
  for (const k of Object.keys(G)) for (const a of (GL_ALIASES[k] || [k])) alias[a.toLowerCase()] = k;
  const alts = Object.keys(alias).sort((a, b) => b.length - a.length).map(a => a.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!alts.length) return used;
  const re = new RegExp(`\\b(${alts.join("|")})\\b`, "gi");
  const skip = "a,code,pre,mark,.gl,.tpl,h1,button,select,textarea,input,.nid,.mono,td.aname,figcaption,.settings,.rbar-axis,label,.hl table";
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, { acceptNode: n => n.parentElement?.closest(skip) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT });
  const nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
  for (let node of nodes) {
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(node.data))) {
      const key = alias[m[1].toLowerCase()];
      if (used.has(key)) continue;
      used.add(key);
      const rest = node.splitText(m.index), after = rest.splitText(m[1].length);
      const span = document.createElement("span");
      span.className = "gl"; span.tabIndex = 0; span.dataset.gl = key; span.textContent = rest.data;
      rest.replaceWith(span);
      node = after; re.lastIndex = 0;
    }
  }
  return used;
}

// ---------- tooltips (tokens and glossary) ----------
const TOKVIEWS = [];
function showTip(html, x, y, pre) {
  const tip = $("#tip");
  tip.className = pre ? "pre" : "";
  tip[pre ? "textContent" : "innerHTML"] = html;
  tip.style.display = "block";
  const w = tip.offsetWidth;
  tip.style.left = Math.max(8, Math.min(innerWidth - w - 12, x + 12)) + "px";
  tip.style.top = (y + 16) + "px";
}
function glTip(el, x, y) {
  const k = el.dataset.gl;
  showTip(`<b>${esc(k[0].toUpperCase() + k.slice(1))}</b>: ${esc(glossary()[k] || "")}`, x, y);
}
document.addEventListener("mouseover", e => {
  const g = e.target.closest?.(".gl");
  if (g) { glTip(g, e.clientX, e.clientY); return; }
  const s = e.target.closest?.(".tl span[data-k]");
  if (!s) { $("#tip").style.display = "none"; return; }
  const v = TOKVIEWS[+s.dataset.k], i = +s.dataset.i;
  const lines = [`token ${i}: ${JSON.stringify(v.tokens[i])}`];
  for (const r of v.rows) lines.push(`${r.label.padEnd(16)} ${r.acts?.[i] == null ? "–" : f(r.acts[i], 4)}${v.diffBase && r.acts !== v.diffBase && !r.raw ? `  (Δ ${sgn(r.acts[i] - v.diffBase[i], 4)})` : ""}`);
  showTip(lines.join("\n"), e.clientX, e.clientY, true);
});
document.addEventListener("focusin", e => {
  const g = e.target.closest?.(".gl");
  if (g) { const r = g.getBoundingClientRect(); glTip(g, r.left, r.bottom - 8); }
});
document.addEventListener("focusout", () => { $("#tip").style.display = "none"; });

function renderLive() {
  const el = $("#live"), L = S.live;
  el.className = "pill " + (L.loaded ? "ok" : "off");
  el.textContent = L.loaded ? "Live model on" : "Saved data only";
  el.title = L.loaded ? `live model: ${L.info?.model} (${L.info?.device}, adapters ${(L.info && Object.keys(L.info.adapters).join(" ")) || "none"})`
    : `Live model not loaded (${L.error ? "load failed: " + L.error : L.status}). Everything except the live tools works.`;
  for (const id of ["custom", "chats", "steer"]) liveGuard($(`#tab-${id}`));
  $$(".live-weights").forEach(liveGuard);
}
function liveGuard(tab) {
  if (!tab) return;
  let n = tab.querySelector(":scope > .notice.live-off");
  if (!S.live.loaded) {
    if (!n) { n = document.createElement("div"); n.className = "notice live-off"; tab.prepend(n); }
    n.textContent = `This tool needs the model running on a GPU, and it is not loaded here (${S.live.error ? "load failed: " + S.live.error : S.live.status}). The saved data on the rest of the page still works.`;
  } else if (n) n.remove();
  $$("button.btn", tab).forEach(b => { if (!b.classList.contains("savebtn")) b.disabled = !S.live.loaded; });
}
function setupTheme() {
  const btn = $("#theme");
  const show = () => { btn.textContent = document.documentElement.dataset.theme === "dark" ? "Light" : "Dark"; };
  btn.onclick = () => {
    const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("pn-theme", t); } catch { }
    show();
  };
  show();
}


// ---------- views and routing ----------
function showView(v) {
  if (S.view !== v) scrollTo(0, 0);
  S.view = v;
  for (const id of ["start", "list", "neuron"]) $(`#${id}-page`).hidden = id !== v;
  $$("#nav a").forEach(a => a.classList.toggle("on", a.dataset.view === v));
}
async function route() {
  const h = location.hash.slice(1), p = new URLSearchParams(h);
  const t = p.get("t");
  if (t && t !== S.trait && S.man.traits?.[t]) setTraitQuiet(t);
  const id = p.get("n");
  if (id) {
    showView("neuron");
    if (id !== S.sel) await openNeuron(id, !t);
    if (p.get("tab")) setTab(p.get("tab"));
    return;
  }
  if (h.startsWith("list")) {
    if (p.has("tech") && !S.tech) { $("#tech-cols").checked = true; S.tech = true; $("#tech-bar").hidden = false; }
    showView("list"); renderList(); return;
  }
  showView("start");
}
function go(id, tab) { S.pendingTab = tab; location.hash = `n=${encodeURIComponent(id)}&t=${S.trait}${tab ? "&tab=" + tab : ""}`; }
function writeHash(tab) {
  if (!S.sel) return;
  const p = new URLSearchParams();
  p.set("n", S.sel); p.set("t", S.trait);
  if (tab) p.set("tab", tab);
  history.replaceState(null, "", "#" + p.toString());
}
$("#jump").onkeydown = e => {
  if (e.key !== "Enter") return;
  const m = e.target.value.trim().toUpperCase().match(/^L?(\d+)\s*[\/_ ]\s*N?(\d+)$/);
  if (m) { go(`L${+m[1]}/N${+m[2]}`); e.target.value = ""; }
};

// ---------- 1. Start page ----------
function renderStart() {
  const el = $("#start-body");
  const featured = S.rows.filter(r => isFeatured(r, plainOf(r.id)));
  featured.sort((a, b) => biggestLog(b, plainOf(b.id)) - biggestLog(a, plainOf(a.id)));
  const intro = S.plain?.intro_md ? md(S.plain.intro_md)
    : `<h1>Persona neurons</h1><p>This browser shows how personality adapters for Qwen3-8B change individual neurons. The plain-language layer for this run (plain.json) is not available, so pages show saved measurements with automated labels.</p>`;
  const hasHowTo = /how to read a neuron page/i.test(S.plain?.intro_md || "");
  const legend = hasHowTo ? "" : `<h2>How to read a neuron page</h2>
    <ul><li>The title is a nickname; the code under it (like L26/N8828) is its address: layer 26, neuron 8828.</li>
    <li><b>What each adapter does to it</b> is the main table: ${dirHTML("stronger")} means the neuron fires harder with that adapter on, ${dirHTML("weaker")} means less hard, ${dirHTML("about the same")} means little change. 1.2× means 20% stronger at the moments it usually fires.</li>
    <li><b>See it in action</b> shows real sentences with the key word highlighted and the neuron's value in each version of the model.</li>
    <li>Everything under <b>Go deeper</b> is raw evidence for technical readers.</li></ul>`;
  const list = featured.map(r => {
    const P = plainOf(r.id), ol = oneLinerOf(r, P);
    return `<li><a href="#n=${encodeURIComponent(r.id)}&t=${esc(traitOf(r, P))}"><b>${esc(nameOf(r, P))}</b></a> <span class="mono faint" style="font-size:13px">${esc(r.id)}</span> ${checkBadge(statusOf(r, P))}<br><span class="dim">${esc(ol.text)}</span></li>`;
  }).join("");
  el.innerHTML = `<div class="md">${intro}${legend}
    <h2>Neurons we studied closely</h2>
    <p>Each of these has a hand-checked description and a full write-up. The other ${S.rows.length - featured.length} neurons are in the <a href="#list">full list</a>, mostly with automated, unchecked descriptions.</p>
    <ul class="featured">${list || "<li>No detailed profiles in this run.</li>"}</ul></div>`;
  glossify(el);
}

// ---------- 2. List ----------
function sortKeys() {
  const keys = [["change", "biggest adapter change"], ["rank", "rank (this trait)"], ["layer", "layer"]];
  for (const s of polarStates()) {
    keys.push([`ratio:${s}`, `ratio at peaks ${s}`], [`z:${s}`, `z_mean ${s}`], [`az:${s}`, `|z_mean| ${s}`], [`zr:${s}`, `z_rms ${s}`], [`ag:${s}`, `agree ${s}`], [`rd:${s}`, `resid_mean_diff ${s}`]);
  }
  keys.push(["rms", "base rms"], ["max", "base max"], ["status", "card status"]);
  return keys;
}
function sortVal(r, key) {
  const [k, s] = key.split(":");
  const e = s ? (r.effects || {})[s] || {} : null;
  switch (k) {
    case "change": return biggestLog(r, plainOf(r.id));
    case "ratio": { const x = effectsFor(r, plainOf(r.id)).find(y => y.state === s); return x?.ratio ?? -1e9; }
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
  S.sort.key = "change";
  $("#sort").innerHTML = sortKeys().map(([k, l]) => `<option value="${esc(k)}">${esc(l)}</option>`).join("");
  $("#sort").onchange = e => { S.sort.key = e.target.value; renderList(); };
  $("#sort-dir").onclick = () => { S.sort.dir *= -1; $("#sort-dir").textContent = S.sort.dir < 0 ? "↓" : "↑"; renderList(); };
  $("#filter").oninput = e => { S.filter = e.target.value.toLowerCase(); renderList(); };
  $("#tech-cols").onchange = e => { S.tech = e.target.checked; $("#tech-bar").hidden = !S.tech; if (!S.tech) { S.groupBy = "plain"; $("#group-by").value = "plain"; } renderList(); };
  $("#group-by").onchange = e => { S.groupBy = e.target.value; renderList(); };
  $$("#trait button").forEach(b => b.onclick = () => setTrait(b.dataset.t));
  const nf = S.rows.filter(r => isFeatured(r, plainOf(r.id))).length;
  $("#list-intro").innerHTML = `${S.rows.length} neurons that the adapters change a lot, plus a few picked at random for comparison. The ${nf} at the top were studied closely and checked by hand; the rest have only a quick automated description.`;
  glossify($("#list-intro"));
}
function setTraitQuiet(t) {
  S.trait = t;
  $$("#trait button").forEach(b => b.classList.toggle("on", b.dataset.t === t));
}
function setTrait(t) {
  setTraitQuiet(t);
  renderList();
  if (S.data) setupLivePanels();
  writeHash();
}
function groupOf(r) {
  if (r.kind === "random_control") return "random_control";
  return (r.group || {})[S.trait] || "other";
}
const STRIP_GROUPS = [["Openness", ["o_amp+1", "o_sup+1"]], ["Neuroticism", ["n_amp+1", "n_sup+1"]], ["Neutral", ["ctrl+1"]]];
function stripHead() {
  const A = adapters();
  const top = STRIP_GROUPS.map(([g, ss], k) => `<span class="sg${ss.length === 1 ? " sg1" : ""}${k ? " gap" : ""}">${g}</span>`).join("");
  const bot = STRIP_GROUPS.map(([, ss], k) => ss.map((s, j) => `<span class="sh${k && !j ? " gap" : ""}" title="${esc(A[s]?.name || s)}">${s === "ctrl+1" ? "○" : s.includes("_amp") ? "+" : "−"}</span>`).join("")).join("");
  return `${top}<br>${bot}`;
}
function stripHTML(r, P) {
  const effs = Object.fromEntries(effectsFor(r, P).map(e => [e.state, e])), A = adapters();
  return STRIP_GROUPS.map(([, ss], k) => ss.map((s, j) => {
    const e = effs[s], d = DIR[e?.direction] || DIR.unknown;
    const tip = `${A[s]?.name || s}: ${e ? `${d.w}${e.ratio != null ? ` (${e.ratio.toFixed(2)}× at its usual peaks)` : ""}` : "not measured"}`;
    return `<span class="a ${d.cls}${k && !j ? " gap" : ""}" title="${esc(tip)}" aria-label="${esc(tip)}">${d.a}</span>`;
  }).join("")).join("");
}
function renderList() {
  const ps = polarStates(), by = {}, order = [];
  const push = (g, r) => { if (!by[g]) { by[g] = []; order.push(g); } by[g].push(r); };
  const rows = S.rows.filter(r => {
    if (!S.filter) return true;
    const P = plainOf(r.id);
    return `${r.id} ${r.label || ""} ${r.selection_reason || ""} ${P?.nickname || ""} ${P?.one_liner || ""} ${P?.group_plain || ""}`.toLowerCase().includes(S.filter);
  });
  const STUDIED = "Studied closely";
  if (S.groupBy === "plain") {
    const known = [STUDIED, GROUP_PLAIN.amp_primary, GROUP_PLAIN.sup_primary, GROUP_PLAIN.shared, GROUP_PLAIN.cross_trait, GROUP_PLAIN.control_assoc];
    for (const g of known) { by[g] = []; order.push(g); }
    for (const r of rows) { const P = plainOf(r.id); push(isFeatured(r, P) ? STUDIED : groupPlainOf(r, P), r); }
    const ri = order.indexOf(RANDOM_GROUP); if (ri >= 0) { order.splice(ri, 1); order.push(RANDOM_GROUP); }
  } else {
    for (const g of [...S.meta.groups, "other"]) { by[g] = []; order.push(g); }
    for (const r of rows) push(groupOf(r), r);
  }
  const GSUB = {
    [STUDIED]: "Each has a hand-checked description and a full write-up.",
    [RANDOM_GROUP]: "Neurons from the same layers chosen at random, as a baseline for everything else.",
  };
  const head = `<thead><tr><th>Neuron</th><th class="strip-h">What the adapters do<br>${stripHead()}</th><th>Description checked?</th>
    ${ps.map(s => `<th class="techc num" title="${esc(stateLabel(s))}">z_mean<br>${esc(s)}</th>`).join("")}<th class="techc">card</th></tr></thead>`;
  const cols = `<colgroup><col><col class="c-strip"><col class="c-check">${ps.map(() => `<col class="c-z techc">`).join("")}<col class="c-card techc"></colgroup>`;
  const html = [];
  for (const g of order) {
    const rs = by[g]; if (!rs?.length) continue;
    const key = S.tech ? S.sort.key : "change", dir = S.tech ? S.sort.dir : -1;
    rs.sort((a, b) => {
      const fa = isFeatured(a, plainOf(a.id)) ? 1 : 0, fb = isFeatured(b, plainOf(b.id)) ? 1 : 0;
      if (fa !== fb && S.groupBy !== "plain") return fb - fa;
      return dir < 0 ? sortVal(b, key) - sortVal(a, key) : sortVal(a, key) - sortVal(b, key);
    });
    const gname = S.groupBy === "plain" ? g : (GROUP_NAMES[g] || g);
    html.push(`<div class="ghead">${esc(gname)} <span class="cnt">${rs.length}</span></div>${GSUB[g] ? `<div class="gsub">${esc(GSUB[g])}</div>` : ""}<table class="nl">${cols}${head}<tbody>`);
    for (const r of rs) {
      const P = plainOf(r.id), ol = oneLinerOf(r, P), status = statusOf(r, P);
      const badges = `${r.detailed ? '<span class="badge" title="detailed profile">P</span>' : ""}${r.template_driven ? '<span class="badge warn" title="more than half of the adapter-induced change is on chat-template tokens">T</span>' : ""}${Object.values(r.robust || {}).some(v => v && v.top_share > 0.6) ? '<span class="badge warn" title="more than 60% of the squared change comes from 0.1% of tokens">O</span>' : ""}`;
      const cells = ps.map(s => {
        const e = (r.effects || {})[s];
        return `<td class="techc num">${e ? `<span class="eff" style="background:${zColor(e.z_mean)}" title="${esc(s)}: z_mean ${f(e.z_mean)}, z_rms ${f(e.z_rms)}, agree ${f(e.agree)}, mean_diff ${f(e.mean_diff, 4)}">${f(e.z_mean, 2)}</span>` : `<span class="eff na">–</span>`}</td>`;
      }).join("");
      html.push(`<tr data-id="${esc(r.id)}"><td><div class="nname"><a href="#n=${encodeURIComponent(r.id)}&t=${esc(traitOf(r, P))}">${esc(nameOf(r, P))}</a>${P?.nickname ? `<span class="nid2">${esc(r.id)}</span>` : ""}${S.tech ? badges : ""}</div>
        <div class="none${ol.auto ? " auto" : ""}">${esc(ol.text)}</div></td>
        <td class="strip">${stripHTML(r, P)}</td><td>${checkBadge(status)}</td>${cells}<td class="techc"><span class="mono" style="font-size:12px">${esc(r.card_status)}</span></td></tr>`);
    }
    html.push(`</tbody></table>`);
  }
  $("#groups").innerHTML = html.join("") || `<div class="slot">No neurons match.</div>`;
  $("#groups").classList.toggle("show-tech", S.tech);
  $$("#groups tbody tr").forEach(el => el.onclick = e => { if (!e.target.closest("a")) go(el.dataset.id); });
}

// ---------- 3. Neuron page ----------
async function openNeuron(id, adoptTrait) {
  S.sel = id;
  try {
    S.data = await api(`/api/neuron?id=${encodeURIComponent(id)}`);
  } catch (e) {
    $("#empty").hidden = false; $("#neuron").hidden = true; $("#empty").textContent = String(e.message); return;
  }
  if (adoptTrait) { const t = traitOf(S.data.index, plainOf(id)); if (S.man.traits?.[t]) setTraitQuiet(t); }
  $("#empty").hidden = true; $("#neuron").hidden = false;
  $$("details.sec").forEach(d => { d.open = false; });
  S.ex = { kind: "all", split: "all", diff: S.ex.diff };
  renderNeuron();
  setupLivePanels();
  scrollTo(0, 0);
}
function latestVersion(card) { return card?.versions?.[0] || null; }
function autoTag(card, ver) {
  if (!ver) return "";
  if (ver.kind === "human_edit") return `<span class="tag">human edit</span>`;
  return card.status === "tested" ? `<span class="tag">automated, tested</span>` : `<span class="tag auto">Automated hypothesis</span>`;
}
function renderNeuron() {
  TOKVIEWS.length = 0;
  const d = S.data, r = d.index || { id: d.id }, P = plainOf(d.id), card = d.card, v = latestVersion(card);
  const status = statusOf(r, P), ol = oneLinerOf(r, P);
  const auto = P ? P.source === "automated" : (!v || v.kind === "automated");
  const nL = S.man.model?.arch?.n_layers || 36;
  // header
  $("#n-crumbs").innerHTML = `<a href="#list">← All neurons</a> <span class="faint">·</span> <span class="dim">${esc(groupPlainOf(r, P))}</span>`;
  $("#n-head").innerHTML = `<h1>${esc(nameOf(r, P))}</h1>
    ${P?.nickname ? `<div class="nid">${esc(d.id)}</div>` : ""}
    <p class="one">${esc(ol.text || (v?.label ?? ""))}</p>
    <div class="facts">${checkBadge(status)}<span>${esc(P?.checked?.text || (status === "untested" ? "Not checked yet." : `Card status: ${status}.`))}</span>
      <span>Layer ${esc(r.layer ?? "?")} of ${nL}</span>${r.kind === "random_control" ? `<span class="tag">picked at random</span>` : ""}${isFeatured(r, P) ? `<span class="tag">studied closely</span>` : ""}</div>
    ${auto ? `<div class="note-auto"><b>Unverified machine guess.</b> The description above was written by an automated labeler from a few examples and has not been reviewed by a person.${status === "rejected" ? " It also failed most automatic checks on new sentences, so take it with a grain of salt." : status === "untested" ? "" : " It was only checked automatically."} The measurements below are real.</div>` : ""}
    ${!d.index ? `<div class="notice">This neuron is not in the run's neuron table, so most measurements are missing.</div>` : ""}`;
  renderPlainEffects(r, P);
  // behavior
  $("#n-behavior").innerHTML = P?.behavior ? `<h2>Does it change what the model writes?</h2><p class="caption" style="color:var(--ink);font-size:16px">${esc(P.behavior)}</p>` : "";
  $("#n-behavior").hidden = !P?.behavior;
  // summary
  const cav = (P?.caveats || []).length ? `<h4>Caveats</h4><ul class="caveats">${P.caveats.map(c => `<li>${mdInline(c)}</li>`).join("")}</ul>` : "";
  if (P?.summary_md) $("#n-summary").innerHTML = `<h2>Summary</h2><div class="md">${md(P.summary_md)}</div>${cav}`;
  else if (v) $("#n-summary").innerHTML = `<h2>${v.kind === "automated" ? "What the automated labeler guessed" : "Latest description on its card"}</h2><div class="md"><p>${esc(v.trigger_hypothesis || "")}</p>${v.output_hypothesis ? `<p><span class="dim">Effect on output${v.kind === "automated" ? " (guess)" : ""}:</span> ${esc(v.output_hypothesis)}</p>` : ""}</div>${cav}<p class="caption">${v.kind === "automated" ? "No hand-checked summary exists for this neuron." : "No plain-language summary is available for this run."} See “Card history and tests” below for the full card and its tests.</p>`;
  else $("#n-summary").innerHTML = cav;
  renderAction(r, P);
  let used = new Set();
  for (const id of ["n-head", "n-effects", "n-behavior", "n-summary", "n-action"]) used = glossify($("#" + id), used);
  // deeper sections
  renderExamples();
  renderTrajectory();
  renderEffects();
  renderWeights();
  renderProfile();
  renderCard();
  renderSaved();
  $("#st-neuron").textContent = d.id;
}
function renderPlainEffects(r, P) {
  const effs = effectsFor(r, P), A = adapters(), el = $("#n-effects");
  if (!effs.length) { el.innerHTML = `<h2>What each adapter does to it</h2><div class="slot">No measurements saved for this neuron.</div>`; return; }
  const ranked = effs.filter(e => e.ratio > 0).sort((a, b) => Math.abs(Math.log(b.ratio)) - Math.abs(Math.log(a.ratio)));
  const top = ranked[0];
  const headline = !top || top.direction === "about the same" || top.direction === "unknown"
    ? `None of the adapters changes how hard it fires at its usual peaks by more than about 10%.`
    : `Biggest change: the <b>${esc(A[top.state]?.name || top.state)}</b> makes it ${dirHTML(top.direction)} (${fx(top.ratio)}).`;
  const rows = effs.map(e => {
    const a = A[e.state] || {};
    return `<tr><td class="aname"><span title="${esc(a.what ? a.name + ": " + a.what : stateLabel(e.state))}">${esc(a.name || stateLabel(e.state))}</span></td>
      <td>${dirHTML(e.direction)}</td><td class="num">${fx(e.ratio)}</td><td>${ratioBar(e.ratio, e.direction)}</td><td class="what">${esc(e.text || "")}</td></tr>`;
  }).join("");
  const computed = effs.some(e => e.computed);
  const subs = [];
  if (P?.overall_change?.text) subs.push(`<dt>How much it changes overall</dt><dd>${esc(P.overall_change.text)}</dd>`);
  else if (r.layer_pct) { const pct = Math.max(...Object.values(r.layer_pct)); subs.push(`<dt>How much it changes overall</dt><dd>The adapters change it more than ${Math.floor(pct * 1000) / 10}% of the other neurons in its layer.</dd>`); }
  if (P?.where?.text) subs.push(`<dt>Where the change happens</dt><dd>${esc(P.where.text)}</dd>`);
  else if (r.template_driven != null) subs.push(`<dt>Where the change happens</dt><dd>${r.template_driven ? "Mostly at the moment the assistant's reply begins, not in ordinary text." : "Mostly in ordinary text."}</dd>`);
  el.innerHTML = `<h2>What each adapter does to it</h2>
    <p class="headline">${headline}</p>
    <table class="t adapters"><thead><tr><th>Adapter</th><th>Compared with the original model</th><th class="num">Ratio</th><th><div class="rbar-axis"><span>0.5×</span><span>1×</span><span>2×</span></div></th><th>In plain words</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <p class="caption">The ratio compares how hard the neuron fires at its usual peaks (the ten moments in our test text where it fires hardest in the original model) with the adapter switched on. 1× means no change; 1.2× means 20% harder. Hover an adapter name for what it was trained to do.${computed ? " <i>These rows were computed in the browser from the saved examples, because the plain-language file has no entry for this neuron.</i>" : ""}</p>
    ${subs.length ? `<dl class="subfacts">${subs.join("")}</dl>` : ""}`;
}
// "See it in action": a few readable sentences with a small table per model at the key word
function pickHighlights(d, P, trait) {
  const ex = d?.examples || [], byId = Object.fromEntries(ex.map(e => [e.ex_id, e]));
  const fv = (e, s) => e.acts?.[s]?.[e.focus];
  let picks = (P?.highlights || []).map(id => byId[id]).filter(e => e && e.focus != null);
  if (!picks.length) {
    const pool = ex.filter(e => e.focus != null && e.acts?.base && e.kind !== "training");
    const strong = [...pool].sort((a, b) => Math.abs(fv(b, "base")) - Math.abs(fv(a, "base"))).slice(0, 2);
    const poles = S.man.traits?.[trait]?.poles || [];
    const chg = e => Math.max(0, ...poles.map(s => Math.abs((fv(e, s) ?? 0) - fv(e, "base"))));
    const big = [...pool].filter(e => !strong.includes(e)).sort((a, b) => chg(b) - chg(a))[0];
    picks = [...strong, ...(big ? [big] : [])];
  }
  return picks.slice(0, 4);
}
function compareText(v, b) {
  if (v == null || b == null) return { d: "unknown", t: "" };
  const big = Math.max(Math.abs(v), Math.abs(b));
  if (big < 1e-6) return { d: "about the same", t: "" };
  if (Math.sign(v) !== Math.sign(b) && Math.min(Math.abs(v), Math.abs(b)) > 0.1 * big) return { d: "flip", t: "" };
  if (Math.abs(b) < 0.04 * Math.abs(v)) return { d: "stronger", t: " (from almost nothing)" };
  const r = Math.abs(v) / Math.abs(b);
  const d = dirOfRatio(r);
  return { d, t: d === "about the same" ? "" : ` (${r >= 10 ? Math.round(r) : r.toFixed(1)}×)` };
}
function renderAction(r, P) {
  const d = S.data.detail, el = $("#n-action");
  if (!d?.examples?.length) { el.innerHTML = ""; return; }
  const trait = traitOf(r, P), poles = S.man.traits?.[trait]?.poles || [];
  const states = ["base", ...poles, S.man.control_state].filter(Boolean);
  const picks = pickHighlights(d, P, trait);
  const maxBase = Math.max(...d.examples.filter(e => e.focus != null && e.acts?.base).map(e => Math.abs(e.acts.base[e.focus])), 1e-9);
  const cards = picks.map((e, k) => {
    const { html, key } = sentenceHTML(e);
    const b = e.acts.base?.[e.focus];
    const delta = Math.max(0, ...states.slice(1).map(s => Math.abs((e.acts[s]?.[e.focus] ?? b) - b)));
    let why = delta > Math.abs(b) ? "An adapter changes it a lot here" : Math.abs(b) > 0.5 * maxBase ? "One of its strongest moments in the original model" : "A typical moment";
    if (e.kind === "training") why += `, from the training data of the ${e.train_meta?.adapter ? (adapters()[`${e.train_meta.adapter}+1`]?.name || e.train_meta.adapter).toLowerCase() : "adapters"}`;
    else if (e.domain) why += ` (source: ${e.domain.replace(/_/g, " ")})`;
    const rows = states.filter(s => e.acts[s]).map(s => {
      const v = e.acts[s][e.focus], c = s === "base" ? null : compareText(v, b);
      return `<tr class="${s === "base" ? "base-row" : ""}"><td>${esc(stateName(s))}</td><td class="num">${sig2(v)}</td><td>${c ? (c.d === "flip" ? `<span class="dir flip"><span class="arrow">↕</span>fires the other way</span>` : dirHTML(c.d, c.t)) : `<span class="dim">reference</span>`}</td></tr>`;
    }).join("");
    return `<div class="hl" id="hl-${k}"><div class="why">${esc(why)}</div><div class="sent">${html}</div>
      <table class="t compact"><thead><tr><th>Version of the model</th><th class="num">Value at “${esc(key)}”</th><th>Compared with the original</th></tr></thead><tbody>${rows}</tbody></table>
      <button class="linkbtn more" data-ex="${esc(e.ex_id)}" data-k="${k}">show all model rows</button><div class="allrows" hidden></div></div>`;
  }).join("");
  el.innerHTML = `<h2>See it in action</h2>
    <p class="caption">Real text from our test set, with the word where the neuron fires highlighted. Each number is how strongly the neuron fires at that word; the sign only says which way it fires, the size says how strongly.</p>${cards}`;
  $$(".more", el).forEach(btn => btn.onclick = () => {
    const box = btn.nextElementSibling, e = d.examples.find(x => x.ex_id === btn.dataset.ex);
    if (box.hidden && !box.innerHTML) box.innerHTML = legendHTML() + exampleHTML(e);
    box.hidden = !box.hidden;
    btn.textContent = box.hidden ? "show all model rows" : "hide model rows";
  });
}
function openSec(t) {
  const sec = $(`details.sec[data-tab="${t}"]`);
  if (!sec) return null;
  sec.open = true;
  return sec;
}
function setTab(t) {
  const sec = openSec(t);
  if (!sec) return;
  if (t === "steer") refreshCalib();
  requestAnimationFrame(() => sec.scrollIntoView({ block: "start" }));
}
$$("details.sec").forEach(d => d.addEventListener("toggle", () => {
  if (d.open) { writeHash(d.dataset.tab); if (d.dataset.tab === "steer") refreshCalib(); }
}));

// ---------- token panels ----------
// rows: [{label, title, acts, cls}], tokens: [str]; returns HTML. Scale shared across rows so changes are visible.
function tokPanel(tokens, rows, { focus = null, boundary = null, diffBase = null } = {}) {
  let m = 0, md2 = 0;
  for (const r of rows) if (!r.raw) for (const a of r.acts || []) m = Math.max(m, Math.abs(a));
  if (diffBase) for (const r of rows) if (r.acts !== diffBase && !r.raw) r.acts.forEach((a, i) => { md2 = Math.max(md2, Math.abs(a - diffBase[i])); });
  const key = TOKVIEWS.push({ tokens, rows, diffBase }) - 1;
  const disp = tokens.map((_, i) => tokDisplay(tokens, i));
  const html = rows.map((r, ri) => {
    const useDiff = diffBase && r.acts !== diffBase && !r.raw;
    const spans = tokens.map((t, i) => {
      const a = r.acts?.[i];
      const c = useDiff ? actColor(a - diffBase[i], md2) : actColor(a, r.scale || m);
      const cls = [focus === i ? "focus" : "", disp[i].cls].filter(Boolean).join(" ");
      return `${boundary === i ? `<span class="bd"></span>` : ""}<span data-k="${key}" data-r="${ri}" data-i="${i}" class="${cls}" style="background:${c}">${esc(disp[i].txt)}</span>`;
    }).join("");
    return `<div class="sl" title="${esc(r.title || r.label)}">${esc(r.label)}${useDiff ? " Δ" : ""}</div><div class="tl">${spans}</div>`;
  }).join("");
  return `<div class="toks">${html}</div>`;
}
function legendHTML(extra = "") {
  return `<div class="legend"><span><i class="sw" style="background:var(--up)"></i>fires positive</span><span><i class="sw" style="background:var(--down)"></i>fires negative</span>
    <span><i class="sw" style="outline:2px solid var(--ink)"></i>key word</span><span><i class="sw" style="border-left:2px dashed var(--state)"></i>where the prompt ends and the reply begins</span><span><span class="tokc" style="font-size:11px">start</span> hidden chat-format tokens, ↵ line breaks</span>${extra}
    <span>Darker = fires harder; colors are on one scale per panel. Hover a word for exact values.</span></div>`;
}
function stateOrder(states) {
  const order = stateList().map(s => s.id);
  return [...states].sort((a, b) => (order.indexOf(a) + 1 || 999) - (order.indexOf(b) + 1 || 999));
}

// ---------- all examples ----------
const KIND_PLAIN = { top_pos: "strongest positive", top_neg: "strongest negative", top_diff: "biggest adapter change", quantile: "typical (quantile)", random: "random", training: "from adapter training data", validation: "held out" };
function renderExamples() {
  const d = S.data.detail, el = $("#tab-examples");
  if (!d) { el.innerHTML = `<div class="slot">No detail file for this neuron (neurons/detail/).</div>`; return; }
  const exs = d.examples || [];
  const kinds = [...new Set(exs.map(e => e.kind))], splits = [...new Set(exs.map(e => e.split).filter(Boolean))];
  const shown = exs.filter(e => (S.ex.kind === "all" || e.kind === S.ex.kind) && (S.ex.split === "all" || e.split === S.ex.split));
  el.innerHTML = `<div class="row">
      <select id="ex-kind"><option value="all">all kinds (${exs.length})</option>${kinds.map(k => `<option value="${esc(k)}" ${S.ex.kind === k ? "selected" : ""}>${esc(KIND_PLAIN[k] || k)} (${esc(k)})</option>`).join("")}</select>
      <select id="ex-split"><option value="all">all splits</option>${splits.map(k => `<option ${S.ex.split === k ? "selected" : ""}>${esc(k)}</option>`).join("")}</select>
      <label class="dim"><input type="checkbox" id="ex-diff" ${S.ex.diff ? "checked" : ""}> color adapter rows by change from the original</label></div>
    ${legendHTML()}<div class="settings"><span class="ms">model state: one row per state (row label), same tokens in every row</span><span class="ni">neuron intervention: none</span></div>
    ${shown.map(exampleHTML).join("") || `<div class="slot">No examples match.</div>`}`;
  $("#ex-kind").onchange = e => { S.ex.kind = e.target.value; renderExamples(); };
  $("#ex-split").onchange = e => { S.ex.split = e.target.value; renderExamples(); };
  $("#ex-diff").onchange = e => { S.ex.diff = e.target.checked; renderExamples(); };
}
function exampleHTML(e) {
  const states = stateOrder(Object.keys(e.acts || {}));
  const rows = states.map(s => ({ label: s, title: stateName(s), acts: e.acts[s] }));
  for (const k of ["gate", "up"]) for (const [s, a] of Object.entries(e[k] || {})) if (s === "base" || S.allGateUp) rows.push({ label: `${k} (${s})`, title: `${k}_proj pre-activation in ${s}`, acts: a, raw: true, scale: Math.max(...a.map(Math.abs)) });
  const ft = e.focus != null ? e.tokens[e.focus] : null;
  const fvals = e.focus != null ? states.map(s => `${s} ${f(e.acts[s][e.focus], 3)}`).join(" · ") : "";
  const tm = e.train_meta ? ` · training: ${esc(e.train_meta.adapter)} ${esc(e.train_meta.role)} ${esc(e.train_meta.dataset_path || "")}` : "";
  const base = S.ex.diff && e.acts?.base ? e.acts.base : null;
  return `<div class="ex" id="ex-${esc(e.ex_id)}"><div class="ex-head"><span class="kind">${esc(e.ex_id)}</span><span class="kind">${esc(KIND_PLAIN[e.kind] || e.kind)}</span><span>${esc(e.split || "")}</span>
    <span>${esc(e.domain)}</span><span>${esc(e.doc_id)}${e.start != null ? ` @${e.start}` : ""}</span><span>${esc(e.source || "")}${tm}</span></div>
    ${ft != null ? `<div class="ex-head">key token ${esc(JSON.stringify(ft))} (token ${e.focus}): ${esc(fvals)}${e.boundary == null ? " · no prompt/continuation split in window" : ""}</div>` : ""}
    ${tokPanel(e.tokens, rows, { focus: e.focus, boundary: e.boundary, diffBase: base })}</div>`;
}

// ---------- trajectory ----------
function trajChart(traj) {
  // x slots: sweep coefficients, then the independent suppressor and the control as separate slots
  const W = 640, H = 260, pad = { l: 54, r: 120, t: 22, b: 40 };
  const colors = { openness: "var(--o)", neuroticism: "var(--n)" };
  const series = [];
  for (const [t, pts] of Object.entries(traj || {})) {
    const info = S.man.traits?.[t] || {};
    const sweep = [], extra = [];
    for (const p of pts) {
      const m = p.state === "base" ? { coef: 0 } : (p.state.match(/^([a-z_]+)([+-][\d.]+)$/) ? { name: RegExp.$1, coef: +RegExp.$2 } : null);
      if (p.state === "base" || (m && m.name === info.amp)) sweep.push({ ...p, x: m.coef });
      else if (p.state === S.man.control_state) continue;
      else extra.push({ ...p, x: 1.6, slot: p.state });
    }
    sweep.sort((a, b) => a.x - b.x);
    series.push({ t, sweep, extra });
  }
  const ctrl = S.data.index?.effects?.[S.man.control_state];
  // saved ci90 is in activation units (mean_diff); convert to the z scale of the plot
  const den = S.data.index?.denominator;
  const zci = p => p.ci90 && den ? p.ci90.map(v => v / den) : null;
  const all = series.flatMap(s => [...s.sweep, ...s.extra]).flatMap(p => [p.z_mean, ...(zci(p) || [])]).filter(v => v != null);
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
  const out = [`<svg class="chart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="average change across adapter strength">`];
  for (const v of ticks) out.push(`<line class="${v === 0 ? "zero" : "grid"}" x1="${pad.l}" x2="${W - pad.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${pad.l - 8}" y="${Y(v) + 4}" text-anchor="end">${f(v, 2)}</text>`);
  for (const x of [-1, -0.5, 0, 0.5, 1]) out.push(`<text x="${X(x)}" y="${H - pad.b + 18}" text-anchor="middle">${x === 0 ? "original" : x > 0 ? "+" + x : "−" + Math.abs(x)}</text>`);
  out.push(`<text x="${X(1.6)}" y="${H - pad.b + 18}" text-anchor="middle">suppressor</text><text x="${X(2.2)}" y="${H - pad.b + 18}" text-anchor="middle">neutral</text>`);
  out.push(`<text x="${X(0)}" y="${H - 4}" text-anchor="middle">amplifier strength</text><text x="8" y="${pad.t - 8}">average change (z)</text>`);
  const off = { openness: -0.04, neuroticism: 0.04 };
  for (const s of series) {
    const c = colors[s.t] || "var(--ink)", dx = off[s.t] || 0;
    const pts = s.sweep.map(p => `${X(p.x + dx)},${Y(p.z_mean)}`).join(" ");
    out.push(`<polyline points="${pts}" fill="none" stroke="${c}" stroke-width="2"/>`);
    for (const p of [...s.sweep, ...s.extra]) {
      const x = X(p.x + dx);
      const ci = zci(p);
      if (ci) out.push(`<line x1="${x}" x2="${x}" y1="${Y(ci[0])}" y2="${Y(ci[1])}" stroke="${c}" stroke-width="1.5" opacity=".6"/>`);
      out.push(`<circle class="pt" cx="${x}" cy="${Y(p.z_mean)}" r="4.5" fill="${c}" stroke-width="2"><title>${esc(s.t)} · ${esc(p.state)}\nz_mean ${f(p.z_mean)}  ci90 (z) [${(zci(p) || []).map(v => f(v)).join(", ")}]\nmean_diff ${f(p.mean_diff, 4)}  mean ${f(p.mean, 4)}  rms ${f(p.rms, 4)}</title></circle>`);
    }
    const last = s.sweep[s.sweep.length - 1];
    if (last) out.push(`<text x="${W - pad.r + 10}" y="${Y(last.z_mean) + 4}" style="fill:${c}">${esc(s.t)}</text>`);
  }
  if (ctrl) out.push(`<rect class="pt" x="${X(2.2) - 5}" y="${Y(ctrl.z_mean) - 5}" width="10" height="10" fill="var(--same)" stroke-width="2"><title>${esc(S.man.control_state)} z_mean ${f(ctrl.z_mean)}</title></rect>`);
  out.push(`</svg>`);
  const legend = `<div class="legend">${series.map(s => `<span><i class="sw" style="background:${colors[s.t]}"></i>${esc(s.t)}: amplifier from −1 to +1, then its suppressor</span>`).join("")}<span><i class="sw" style="background:var(--same)"></i>neutral control</span><span>bars: 90% intervals; hover points for values</span></div>`;
  return legend + out.join("");
}
function niceStep(x) { const p = Math.pow(10, Math.floor(Math.log10(x || 1))); const m = x / p; return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p; }
function renderTrajectory() {
  const d = S.data.detail || {}, r = S.data.index || {};
  const trajTables = Object.entries(d.trajectory || {}).map(([t, pts]) => `<h4>${esc(t)}</h4><table class="dtable"><tr><th>state</th><th class="num">mean</th><th class="num">rms</th><th class="num">mean_diff</th><th class="num">z_mean</th><th class="num">ci90 (mean_diff)</th></tr>
    ${pts.map(p => `<tr><td class="mono">${esc(p.state)}</td><td class="num">${f(p.mean, 4)}</td><td class="num">${f(p.rms, 4)}</td><td class="num">${f(p.mean_diff, 4)}</td><td class="num">${f(p.z_mean)}</td><td class="num">${(p.ci90 || []).map(v => f(v)).join(" … ")}</td></tr>`).join("")}</table>`).join("");
  $("#tab-trajectory").innerHTML = `<p class="caption">Each line dials one amplifier from −1 (pushed the opposite way) through the original model (0) to +1 (full strength). The dots on the right are the separately trained suppressor and the neutral control. Up and down is the average change in the neuron's value over all the test text, divided by its typical size in the original model (${f(r.denominator)}); 0 means no change. This is an average over every word, so it can differ from the peak ratios in the table above. If the neuron mostly fires negative, a line going down means it fires harder.</p>
    ${trajChart(d.trajectory)}
    <details class="sub"><summary>trajectory table</summary>${trajTables}</details>`;
}

// ---------- numbers behind the table ----------
function renderEffects() {
  const d = S.data.detail || {}, r = S.data.index || {}, P = plainOf(S.data.id), el = $("#tab-effects");
  const st = stateOrder(Object.keys(r.effects || {}));
  const groups = Object.entries(r.group || {}).filter(([, g]) => g).map(([t, g]) => `${t}: ${g}${r.rank?.[t] ? ` (rank ${r.rank[t]})` : ""}`).join(" · ");
  const b = r.base || {};
  const plainRows = effectsFor(r, P).map(e => `<tr><td class="mono">${esc(e.state)}</td><td class="num">${e.ratio == null ? "–" : e.ratio.toFixed(3)}</td><td class="num">${r.example_ratio?.[e.state] == null ? "–" : r.example_ratio[e.state].toFixed(3)}</td><td>${esc(e.direction)}</td><td>${e.computed ? "computed from saved examples" : "plain.json"}</td></tr>`).join("");
  const effRows = st.map(s => { const e = r.effects[s]; return `<tr><td class="mono" title="${esc(stateLabel(s))}">${esc(s)}</td><td class="num">${f(e.mean_diff, 4)}</td><td class="num">${f(e.rms_diff, 4)}</td><td class="cell" style="background:${zColor(e.z_mean)}">${f(e.z_mean)}</td><td class="cell" style="background:${zColor(e.z_rms)}">${f(e.z_rms)}</td><td class="num">${f(e.agree, 2)}</td><td class="num">${f(e.resid_mean_diff, 4)}</td><td class="num">${f(r.layer_pct?.[s], 4)}</td></tr>`; }).join("");
  const doms = [...new Set(Object.values(d.by_domain || {}).flatMap(o => Object.keys(o)))];
  const byDom = stateOrder(Object.keys(d.by_domain || {})).map(s => `<tr><td class="mono">${esc(s)}</td>${doms.map(dm => { const c = d.by_domain[s][dm]; return c ? `<td class="cell" style="background:${zColor(c.mean_diff / (r.denominator || 1))}" title="n_tok ${c.n_tok}; z ${f(c.mean_diff / (r.denominator || 1))}">${f(c.mean_diff, 4)}</td>` : "<td>–</td>"; }).join("")}</tr>`).join("");
  const bySeg = stateOrder(Object.keys(d.by_segment || {})).map(s => `<tr><td class="mono">${esc(s)}</td><td class="cell" style="background:${zColor(d.by_segment[s].prompt / (r.denominator || 1))}">${f(d.by_segment[s].prompt, 4)}</td><td class="cell" style="background:${zColor(d.by_segment[s].continuation / (r.denominator || 1))}">${f(d.by_segment[s].continuation, 4)}</td></tr>`).join("");
  const tmpl = Object.entries(r.template || {}).map(([t, x]) => `<tr><td>${esc(t)}</td><td class="num">${f(x.template_share, 3)}</td><td class="num">${f(x.corr_all, 3)}</td><td class="num">${f(x.corr_no_template, 3)}</td></tr>`).join("");
  const nr = d.natural_range;
  const steerRows = (d.steering || []).map(x => `<div class="ex">${settingsHTML(x.state, x.intervention)}<div class="ex-head">prompt ${esc(x.prompt_id)} · KL first token ${f(x.kl_first_token, 4)} · top changes ${(x.top_changes || []).map(([t, v]) => `${esc(JSON.stringify(t))} ${sgn(v)}`).join(", ")}</div><div class="col">${esc(x.text)}</div></div>`).join("");
  const ablRows = (d.ablation || []).map(x => `<tr><td>${settingsHTML(x.state, x.intervention ? { neuron: S.data.id, ...x.intervention } : null)}</td><td>${esc(x.metric)}</td><td class="num">${f(x.value)}</td><td class="num">${f(x.baseline)}</td><td class="num">${esc(x.n)}</td></tr>`).join("");
  el.innerHTML = `
    <h4>Selection</h4>
    <div style="font-size:14px">${esc(groups || "no trait group")}${r.selection_reason ? ` · ${esc(r.selection_reason)}` : ""}${r.kind ? ` · kind ${esc(r.kind)}` : ""}</div>
    <div class="mono dim" style="font-size:12.5px;margin-top:4px">base: mean ${f(b.mean)} · rms ${f(b.rms)} · pos ${f(b.frac_pos, 2)} / neg ${f(b.frac_neg, 2)} · p01 ${f(b.q?.p01)} · p99 ${f(b.q?.p99)} · p999 ${f(b.q?.p999)} · min ${f(b.min)} · max ${f(b.max)} · z denominator ${f(r.denominator)}</div>
    <h4>Peak ratios (the plain table)</h4>
    <div class="hint">Median over the ten strongest base moments of activation(state) / activation(base) at the same token (docs/plain_format.md). Second column: the same rule recomputed by the server from the saved example windows.</div>
    <table class="dtable"><tr><th>state</th><th class="num">ratio</th><th class="num">from saved examples</th><th>direction</th><th>source</th></tr>${plainRows}</table>
    <h4>Effects per state over the whole corpus (neurons/index.json)</h4>
    <table class="dtable"><tr><th>state</th><th class="num">mean_diff</th><th class="num">rms_diff</th><th class="num">z_mean</th><th class="num">z_rms</th><th class="num">agree</th><th class="num">resid_mean_diff</th><th class="num">layer percentile</th></tr>${effRows}</table>
    <h4>Share of change on chat-template tokens</h4>
    ${tmpl ? `<table class="dtable"><tr><th>trait</th><th class="num">template share</th><th class="num">amp/sup corr (all)</th><th class="num">corr (no template)</th></tr>${tmpl}</table>` : `<div class="slot">none saved</div>`}
    <h4>By domain (mean_diff from base)</h4>
    ${byDom ? `<table class="dtable"><tr><th>state</th>${doms.map(x => `<th class="num">${esc(x)}</th>`).join("")}</tr>${byDom}</table>` : `<div class="slot">none saved</div>`}
    <h4>By segment (mean_diff from base)</h4>
    ${bySeg ? `<table class="dtable"><tr><th>state</th><th class="num">prompt</th><th class="num">continuation</th></tr>${bySeg}</table>` : `<div class="slot">none saved</div>`}
    <h4>Natural range and steering units (saved run)</h4>
    ${nr ? `<div class="mono" style="font-size:13px">base p01 ${f(nr.p01)} · p99 ${f(nr.p99)} · resid_p50 ${f(nr.resid_p50)} · down norm ${f(nr.down_norm)} · add unit = ${f(nr.add_unit_in_act)} activation</div>` : `<div class="slot">none saved</div>`}
    <h4>Saved steering on held-out prompts</h4>${steerRows || `<div class="slot">none saved</div>`}
    <h4>Saved ablations</h4>${ablRows ? `<table class="dtable"><tr><th>settings</th><th>metric</th><th class="num">value</th><th class="num">baseline</th><th class="num">n</th></tr>${ablRows}</table>` : `<div class="slot">none saved</div>`}`;
}

// ---------- weights and logit lens ----------
function renderWeights() {
  const d = S.data.detail || {}, el = $("#tab-weights");
  const w = d.weights || {};
  const wStates = stateOrder([...new Set([...Object.keys(w.down_norm || {}), ...Object.keys(w.down_cos_to_base || {})])]);
  const wRows = wStates.map(s => `<tr><td class="mono">${esc(s)}</td><td class="num">${f(w.down_norm?.[s], 5)}</td><td class="num">${f(w.down_cos_to_base?.[s], 6)}</td><td class="num">${f(w.gate_cos_to_base?.[s], 6)}</td><td class="num">${f(w.up_cos_to_base?.[s], 6)}</td></tr>`).join("");
  const lensRows = stateOrder(Object.keys(d.lens || {})).map(s => `<tr><td class="mono">${esc(s)}</td><td>${(d.lens[s].up || []).map(t => `<span class="tokc">${esc(t)}</span>`).join("")}</td><td>${(d.lens[s].down || []).map(t => `<span class="tokc">${esc(t)}</span>`).join("")}</td></tr>`).join("");
  el.innerHTML = `
    <h4>Weight change per state (saved run)</h4>
    <div class="hint">Cosine 1.0 means the neuron's weights point the same way as in the original model. Values this close to 1 mean the adapter changes what reaches the neuron, not the neuron itself.</div>
    ${wRows ? `<table class="dtable"><tr><th>state</th><th class="num">down column norm</th><th class="num">down cos to base</th><th class="num">gate row cos</th><th class="num">up row cos</th></tr>${wRows}</table>` : `<div class="slot">none saved</div>`}
    <h4>Logit lens per state (saved run; effective down column of that state)</h4>
    <div class="hint">The words this neuron's output pushes the model toward when it fires positive (left) or negative (right).</div>
    ${lensRows ? `<table class="dtable"><tr><th>state</th><th class="up-l">promoted by +activation</th><th class="down-l">promoted by −activation</th></tr>${lensRows}</table>` : `<div class="slot">none saved</div>`}
    <div class="live-weights">
      <h4>Live weight views</h4>
      <div class="ctl-box state-box"><div class="box-lab">Model state for weight views</div><select id="lw-state"></select>
        <button class="btn small" id="lw-lens">logit lens</button><button class="btn small" id="lw-con">connections</button>
        <div class="hint">Computed from the effective weights of the chosen state (base + scaled adapter updates). Connections are direct-path weights: candidate paths, not circuits.</div></div>
      <div id="lw-out"></div>
    </div>`;
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

// ---------- full profile ----------
function renderProfile() {
  const id = S.data.id, sec = $("#sec-profile");
  sec.hidden = !S.data.profile;
  $("#tab-profile").innerHTML = S.data.profile ? `<div class="hint">profiles/${esc(id.replace("/", "_"))}.md</div><div class="md wide-md">${md(S.data.profile)}</div>` : "";
}

// ---------- card ----------
function renderCard() {
  const card = S.data.card, el = $("#tab-card");
  if (!card) {
    el.innerHTML = `<div class="slot">No card for this neuron yet.</div>${editForm(null)}`;
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
      <td>${checkBadge(t.result === "pass" ? "tested" : t.result === "fail" ? "rejected" : "mixed").replace(/>[^<]*</, `>${esc(t.result)}<`)}</td><td class="mono">${esc(t.state || "")}</td><td>${esc(t.notes || "")}</td></tr>`).join("")}</table>` : `<div class="slot">No tests recorded.</div>`;
  el.innerHTML = `<div class="row">${checkBadge(card.status)}<span class="mono dim" style="font-size:13px">status ${esc(card.status)}, from ${esc(card.status_from)}${card.status_from === "human edit" && card.run_status ? ` (run says ${esc(card.run_status)})` : ""}</span></div>
    <h4>Card versions, latest first</h4>${vers}
    <h4>Tests</h4>${tests}
    <h4>Disagreements</h4>${card.disagreements.length ? `<ul>${card.disagreements.map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : `<div class="slot">none</div>`}
    ${editForm(latestVersion(card), card.status)}`;
  $$("[data-ex]", el).forEach(a => a.onclick = () => showExample(a.dataset.ex));
  bindEdit();
}
function editForm(v, status = "untested") {
  v = v || {};
  const ta = (name, val, rows = 2) => `<textarea name="${name}" rows="${rows}">${esc(val || "")}</textarea>`;
  return `<h4>Edit card</h4><div class="hint">Saved as a new version (kind human_edit) in edits/cards/. The automated card file is never changed.</div>
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
      S.data.index = S.rows.find(r => r.id === S.data.id) || S.data.index;
      renderList(); renderNeuron(); setTab("card");
      $("#edit-msg").textContent = `saved v${S.data.card.versions[0].v}`;
    } catch (err) { $("#edit-msg").textContent = err.message; }
  };
}
function showExample(exId) {
  S.ex.kind = "all"; S.ex.split = "all";
  renderExamples(); openSec("examples"); writeHash("examples");
  const el = document.getElementById(`ex-${exId}`);
  if (!el) { $("#tab-examples").insertAdjacentHTML("afterbegin", `<div class="notice">Example ${esc(exId)} is not in this neuron's saved example list.</div>`); return; }
  el.scrollIntoView({ block: "center" });
  el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 2000);
}

// ---------- saved ----------
function renderSaved() {
  const rows = S.data.saved || [], el = $("#tab-saved");
  el.innerHTML = `<div class="hint">Live results saved for this neuron (runs/${esc(S.man.run_id)}/edits/saved.jsonl), latest first.</div>` + (rows.map(r => `<div class="ex"><div class="ex-head"><span class="kind">${esc(r.kind)}</span><span>${esc(r.saved)}</span></div>
    ${settingsHTML(r.state, r.intervention)}
    ${r.input ? `<div class="dim" style="font-size:13px">input: ${esc(typeof r.input === "string" ? r.input : JSON.stringify(r.input))}</div>` : ""}
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
  el.innerHTML = groups.filter(g => g[1].length).map(([lab, ss]) => `<span class="grp">${esc(lab)}</span>${ss.map(s => `<label title="${esc(stateName(s))}"><input type="checkbox" value="${esc(s)}" ${use.has(s) ? "checked" : ""}>${esc(s)}</label>`).join("")}`).join("");
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
  renderLive();
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
  out.innerHTML = `<div class="user-msg dim">user: ${esc(msg)}</div><div class="cols" style="grid-template-columns:repeat(${Math.min(states.length, 4)},1fr)">${states.map((s, i) => `<div class="col live" data-c="${i}"><span class="chead">${esc(stateName(s))}</span>${settingsHTML(s, null)}<span class="body"></span></div>`).join("")}</div><div class="err" id="ch-err"></div>`;
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

// ---------- boot ----------
S.meta = await api("/api/meta");
S.man = S.meta.run;
S.live = S.meta.live;
try { S.plain = await api("/api/plain"); } catch { S.plain = null; }
renderLive();
setInterval(async () => { if (!S.live.loaded) { try { S.live = await api("/api/live/status"); renderLive(); } catch { } } }, 3000);
setupTheme();
S.rows = await api("/api/neurons");
setupList();
renderStart();
route();
addEventListener("hashchange", route);
