"use strict";
// ── Redrob Talent Intelligence — frontend logic ────────────────────────────
const $ = (s) => document.querySelector(s);
const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
const esc = (s) => (s == null ? "" : String(s)).replace(/[&<>"]/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[m]));
const pct = (x) => Math.round((x || 0) * 100);

const WEIGHTS = [
  { key: "semantic", label: "Semantic fit", v: 0.20 },
  { key: "skill", label: "Skill match", v: 0.25 },
  { key: "career", label: "Career trajectory", v: 0.33 },
  { key: "experience", label: "Experience fit", v: 0.22 },
];
const SUB = [["semantic", "S"], ["skill", "K"], ["career", "C"], ["experience", "E"]];
const GROUP_LABEL = {
  embeddings: "Embeddings", vector_db: "Vector DB", ranking_ir: "Ranking/IR",
  evaluation: "Eval", python: "Python", nlp_llm: "NLP/LLM", ml_core: "ML core", search_rec: "Search/Rec",
};
let debounce;

// ── init ──
async function init() {
  buildSliders();
  bindControls();
  const jd = await fetch("/api/jd").then((r) => r.json());
  $("#roleTitle").textContent = jd.title;
  $("#jdBody").textContent = jd.text;
  loadStats();
  rank();
}

function buildSliders() {
  const box = $("#sliders");
  WEIGHTS.forEach((w) => {
    const row = el("div", "slider");
    row.innerHTML = `<div class="row"><span>${w.label}</span><span class="pct" id="pct_${w.key}">${pct(w.v)}%</span></div>
      <input type="range" min="0" max="100" value="${pct(w.v)}" id="w_${w.key}">`;
    box.appendChild(row);
    row.querySelector("input").addEventListener("input", (e) => {
      $(`#pct_${w.key}`).textContent = e.target.value + "%";
      scheduleRank();
    });
  });
}

function bindControls() {
  ["t_semantic", "t_behavioral", "t_honeypot", "topn"].forEach((id) =>
    $("#" + id).addEventListener("change", rank));
  $("#search").addEventListener("input", scheduleRank);
  $("#resetWeights").addEventListener("click", () => {
    WEIGHTS.forEach((w) => { $("#w_" + w.key).value = pct(w.v); $("#pct_" + w.key).textContent = pct(w.v) + "%"; });
    rank();
  });
  $("#jdToggle").addEventListener("click", () => {
    const b = $("#jdBody"); b.hidden = !b.hidden; $("#jdToggle").textContent = b.hidden ? "show" : "hide";
  });
  $("#trapsBtn").addEventListener("click", openTraps);
  $("#scrim").addEventListener("click", closeDrawer);
  $("#modalScrim").addEventListener("click", (e) => { if (e.target.id === "modalScrim") closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeDrawer(); closeModal(); } });
}

function scheduleRank() { clearTimeout(debounce); debounce = setTimeout(rank, 200); }

async function loadStats() {
  const s = await fetch("/api/stats").then((r) => r.json());
  const tiles = [
    ["accent", s.pool_size.toLocaleString(), "Pool"],
    ["good", s.strong_candidates, "Strong"],
    ["bad", s.honeypots_detected, "Honeypots"],
    ["", s.backend.split("(")[0], "Engine"],
  ];
  $("#statTiles").innerHTML = tiles.map(([c, v, k]) =>
    `<div class="tile ${c}"><div class="v">${esc(v)}</div><div class="k">${k}</div></div>`).join("");
}

// ── ranking ──
async function rank() {
  const weights = {}; WEIGHTS.forEach((w) => weights[w.key] = (+$("#w_" + w.key).value) / 100);
  const body = {
    weights,
    use_semantic: $("#t_semantic").checked,
    use_behavioral: $("#t_behavioral").checked,
    use_honeypot: $("#t_honeypot").checked,
    top_n: +$("#topn").value,
    search: $("#search").value,
  };
  $("#cards").innerHTML = `<div class="loading"><div class="spinner"></div>Ranking candidates…</div>`;
  const t0 = performance.now();
  const data = await fetch("/api/rank", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }).then((r) => r.json());
  const ms = Math.round(performance.now() - t0);
  $("#resultsMeta").textContent = `${data.results.length} shown · ${ms} ms · honeypot-safe`;
  renderCards(data.results);
}

function renderCards(results) {
  const box = $("#cards"); box.innerHTML = "";
  if (!results.length) { box.innerHTML = `<div class="loading">No matches.</div>`; return; }
  results.forEach((c) => box.appendChild(card(c)));
}

function card(c) {
  const e = el("div", "card" + (c.is_honeypot ? " hp" : ""));
  const badges = [];
  if (c.is_honeypot) badges.push(`<span class="badge hp">honeypot</span>`);
  if (c.services_frac >= 0.6) badges.push(`<span class="badge svc">services-heavy</span>`);
  if (c.availability >= 0.7) badges.push(`<span class="badge avail">available</span>`);
  const chips = (c.matched_groups || []).slice(0, 6)
    .map((g) => `<span class="chip">${GROUP_LABEL[g] || g}</span>`).join("");
  const subbars = SUB.map(([k, l]) =>
    `<div class="sb"><span class="l">${l}</span><div class="bar"><i style="width:${pct(c.sub[k])}%"></i></div></div>`).join("");
  e.innerHTML = `
    <div class="rankbadge">${c.rank}</div>
    <div class="c-main">
      <div class="nm">${esc(c.name)} ${badges.join(" ")}</div>
      <div class="ttl">${esc(c.title)}</div>
      <div class="meta"><span>🏢 ${esc(c.company)}</span><span>${esc(c.industry)}</span>
        <span>⏳ ${c.yoe}y</span><span>📍 ${esc(c.location)}</span></div>
      <div class="chips">${chips}</div>
      <div class="reason">${esc(c.reasoning)}</div>
    </div>
    <div class="c-score">
      <div class="gauge" style="--p:${pct(c.score)}"><b>${pct(c.score)}</b></div>
      <div class="subbars">${subbars}</div>
    </div>`;
  e.addEventListener("click", () => openDrawer(c.candidate_id));
  return e;
}

// ── drawer ──
async function openDrawer(cid) {
  $("#scrim").hidden = false;
  const d = $("#drawer"); d.hidden = false;
  d.innerHTML = `<div class="loading"><div class="spinner"></div>Loading…</div>`;
  const c = await fetch("/api/candidate/" + cid).then((r) => r.json());
  const f = c.features;
  const bb = [["Skill", f.skill_score], ["Career", f.career_score], ["Experience", f.experience_score]]
    .map(([l, v]) => `<div class="bb"><div class="row"><span>${l}</span><span>${pct(v)}%</span></div>
      <div class="bar"><i style="width:${pct(v)}%"></i></div></div>`).join("");
  const skills = (c.skills || []).slice(0, 14).map((s) => {
    const trust = trustOf(s);
    return `<div class="skill-row"><div><div class="n">${esc(s.name)}</div>
      <div class="p">${esc(s.proficiency)} · ${s.duration_months || 0}mo · ${s.endorsements || 0} endorsements</div></div>
      <div class="trust" title="trust ${pct(trust)}%"><i style="width:${pct(trust)}%"></i></div></div>`;
  }).join("");
  const timeline = (c.career || []).map((j) => `<div class="tl">
      <div class="t">${esc(j.title)}</div><div class="co">${esc(j.company)} · ${esc(j.industry)}</div>
      <div class="d">${esc((j.start_date || "").slice(0, 7))} – ${j.is_current ? "present" : esc((j.end_date || "").slice(0, 7))} · ${j.duration_months || 0}mo</div>
      <div class="desc">${esc(j.description)}</div></div>`).join("");
  const sg = Object.entries(c.signals || {}).map(([k, v]) =>
    `<div class="sig"><span>${esc(k)}</span><span class="v">${fmtSig(v)}</span></div>`).join("");
  const verdict = c.is_honeypot
    ? `<div class="verdict hp"><strong>⚠ Honeypot — excluded.</strong><br>${(c.honeypot_reasons || []).map(esc).join("; ")}</div>`
    : `<div class="verdict ok">${esc(c.reasoning)}</div>`;

  d.innerHTML = `
    <div class="dh">
      <button class="x" onclick="closeDrawer()">✕</button>
      <div style="font-size:19px;font-weight:700">${esc(c.name)}</div>
      <div style="color:var(--txt-2);font-size:13.5px;margin-top:2px">${esc(c.title)}</div>
      <div style="color:var(--txt-3);font-size:12px;margin-top:4px">🏢 ${esc(c.company)} · ${esc(c.industry)} (${esc(c.company_size)}) · ⏳ ${c.yoe}y · 📍 ${esc(c.location)}, ${esc(c.country)}</div>
    </div>
    <div class="body">
      <div class="dsec"><h4>Verdict</h4>${verdict}</div>
      <div class="dsec"><h4>Summary</h4><div style="font-size:12.5px;color:var(--txt-2);line-height:1.6">${esc(c.summary)}</div></div>
      <div class="dsec"><h4>Score breakdown</h4><div class="breakbars">${bb}</div>
        <div class="kv">
          <div class="item"><div class="k">Behavioural ×</div><div class="v">${f.behavioral_multiplier}</div></div>
          <div class="item"><div class="k">Core areas</div><div class="v">${(f.matched_groups || []).length}/8</div></div>
          <div class="item"><div class="k">Product-co tenure</div><div class="v">${pct(f.product_frac)}%</div></div>
          <div class="item"><div class="k">Applied-ML career</div><div class="v">${pct(f.applied_ml_frac)}%</div></div>
        </div>
        ${(f.concerns || []).length ? `<div style="margin-top:10px;font-size:12px;color:var(--warn)">⚑ ${f.concerns.map(esc).join(" · ")}</div>` : ""}
      </div>
      <div class="dsec"><h4>Skills (trust-weighted)</h4>${skills || "<i>none listed</i>"}</div>
      <div class="dsec"><h4>Career history</h4><div class="timeline">${timeline}</div></div>
      <div class="dsec"><h4>Redrob behavioural signals</h4><div class="sig-grid">${sg}</div></div>
    </div>`;
}
function trustOf(s) {
  const e = Math.min(s.endorsements || 0, 50) / 50, d = Math.min(s.duration_months || 0, 48) / 48;
  const p = { beginner: 0, intermediate: 1, advanced: 2, expert: 3 }[s.proficiency] / 3 || 0;
  return Math.min(1, 0.4 * e + 0.4 * d + 0.2 * p);
}
function fmtSig(v) {
  if (typeof v === "boolean") return v ? "✓" : "✗";
  if (typeof v === "number") return Number.isInteger(v) ? v : v.toFixed(2);
  return esc(v);
}
function closeDrawer() { $("#drawer").hidden = true; $("#scrim").hidden = true; }

// ── traps modal ──
async function openTraps() {
  $("#modalScrim").hidden = false;
  const m = $("#trapsModal");
  m.innerHTML = `<div class="loading"><div class="spinner"></div>Scanning…</div>`;
  const data = await fetch("/api/traps").then((r) => r.json());
  const rows = data.traps.map((t) => `<div class="trap">
      <span class="tg ${t.type}">${t.type}</span>
      <div><div style="font-size:13px;font-weight:600">${esc(t.name)} — <span style="color:var(--txt-3)">${esc(t.title)}</span></div>
      <div style="font-size:11.5px;color:var(--txt-3);margin-top:3px">${(t.reasons || []).map(esc).join("; ")}</div></div>
    </div>`).join("");
  m.innerHTML = `<div class="mh"><h3>🛡 Traps caught & excluded (${data.traps.length})</h3>
      <button class="x" style="position:static" onclick="closeModal()">✕</button></div>
    <div class="mb"><p style="color:var(--txt-2);font-size:12.5px;margin-top:0">
      These profiles are detected and floored out of the shortlist — keyword-stuffers
      (AI skills, non-AI career) and honeypots (impossible dates / unused expert skills).</p>
      ${rows || "<i>none</i>"}</div>`;
}
function closeModal() { $("#modalScrim").hidden = true; }

window.closeDrawer = closeDrawer; window.closeModal = closeModal;
init();
