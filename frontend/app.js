/* Support Resolution Assistant - single-page UI (no build step, no framework).
   All dynamic text is inserted with textContent / createTextNode (see el()), never innerHTML, so complaint
   text and retrieved documents cannot inject markup. */
(() => {
"use strict";

// ------------------------------------------------------------------ helpers
const $ = (sel, root = document) => root.querySelector(sel);
function el(tag, props, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, "");
    else e.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return e;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pct = (x) => `${Math.round(x * 100)}%`;
const state = { cfg: null, token: null, me: null, cats: [], fb: null, tab: "resolve", adminTab: "overview" };

function toast(msg, err = false) {
  const t = el("div", { class: "toast" + (err ? " err" : "") }, msg);
  $("#toasts").append(t);
  setTimeout(() => t.remove(), err ? 7000 : 3500);
}

// ---------------------------------------------------------------------- API
async function getToken() {
  if (state.fb && state.fb.auth().currentUser) return state.fb.auth().currentUser.getIdToken();
  return state.token;
}
async function api(path, { method = "GET", body } = {}) {
  const headers = { "Content-Type": "application/json" };
  const tok = await getToken();
  if (tok) headers.Authorization = "Bearer " + tok;
  const r = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const data = await r.json().catch(() => null);
  if (r.status === 401 && state.cfg && state.cfg.auth_mode !== "off") { logout(true); throw new Error("Session expired - please sign in again"); }
  if (!r.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === "string" ? d : d ? JSON.stringify(d) : `${r.status} ${r.statusText}`);
  }
  return data;
}
async function guarded(btn, fn) {          // disable a button while an async action runs, surface errors as toasts
  const label = btn && btn.textContent;
  if (btn) { btn.disabled = true; }
  try { return await fn(); } catch (e) { toast(e.message, true); }
  finally { if (btn) { btn.disabled = false; if (label) btn.textContent = label; } }
}

// --------------------------------------------------------------------- auth
function logout(expired) {
  state.token = null; state.me = null;
  sessionStorage.removeItem("tra_token");
  if (state.fb) state.fb.auth().signOut().catch(() => {});
  if (expired) toast("Session expired", true);
  renderLogin();
}
async function afterLogin() {
  state.me = await api("/v1/me");
  state.cats = await api("/v1/categories").catch(() => []);
  renderShell();
}
async function loadScript(src) {
  await new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = () => rej(new Error("failed to load " + src)); document.head.append(s); });
}
const SDK_SOURCES = ["https://www.gstatic.com/firebasejs/10.12.2/", "https://cdn.jsdelivr.net/npm/firebase@10.12.2/"];
async function initFirebase(cfg) {
  let lastErr = null;
  for (const base of SDK_SOURCES) {   // second source is a fallback for networks that block gstatic.com
    try {
      if (!window.firebase) await loadScript(base + "firebase-app-compat.js");
      if (!(window.firebase && window.firebase.auth)) await loadScript(base + "firebase-auth-compat.js");
      if (!window.firebase.apps || !window.firebase.apps.length) window.firebase.initializeApp(cfg);
      state.fb = window.firebase; state.fbError = null;
      return;
    } catch (e) { lastErr = e; }
  }
  throw new Error(`the Firebase SDK could not be downloaded (${lastErr ? lastErr.message : "unknown"})`);
}

function authErrorText(e) {
  const code = (e && e.code) || "";
  if (code === "auth/popup-closed-by-user" || code === "auth/cancelled-popup-request") return "Sign-in cancelled.";
  if (code === "auth/popup-blocked") return "The browser blocked the sign-in popup - allow popups for this site and try again.";
  if (code === "auth/unauthorized-domain") return `This address (${location.hostname}) is not in Firebase → Authentication → Settings → Authorized domains. Add it and retry.`;
  if (code === "auth/operation-not-allowed") return "Google sign-in is not enabled for this Firebase project (Authentication → Sign-in method → Google).";
  return (e && e.message) || "Sign-in failed";
}
async function signInWithGoogle(errBox) {
  errBox.textContent = "";
  try {
    const provider = new state.fb.auth.GoogleAuthProvider();
    provider.setCustomParameters({ prompt: "select_account" });
    await state.fb.auth().signInWithPopup(provider);
    await afterLogin();
  } catch (e) {
    errBox.textContent = authErrorText(e);
    if (state.fb) state.fb.auth().signOut().catch(() => {});   // e.g. a 403 from a domain restriction: do not stay half signed-in
  }
}

function renderLogin() {
  const cfg = state.cfg;
  const root = $("#app"); root.replaceChildren();
  const box = el("div", { class: "card login stack" }, el("h1", null, "Support Resolution Assistant"),
    el("p", { class: "muted" }, "Paste a customer complaint, get a grounded, cited resolution."));
  const err = el("p", { class: "small", style: "color:var(--bad)", role: "alert" });

  // 1) Google (Firebase). Always shown, so it is clear the option exists; disabled with an explanation if not configured.
  const googleReady = !!(cfg.google_login && state.fb);
  const google = el("button", { class: "btn google", type: "button", id: "google-signin", disabled: !googleReady,
    onclick: () => guarded(google, () => signInWithGoogle(err)) }, el("span", { class: "g-mark", "aria-hidden": "true" }, "G"), "Continue with Google");
  box.append(google);
  if (!googleReady) {
    let why, retry = null;
    if (!cfg.google_login) {
      why = cfg.google_login_problem ||
        "Google sign-in is not configured on this server. Set FIREBASE_API_KEY and " +
        "FIREBASE_PROJECT_ID in .env (see README → “Google sign-in”) and restart.";
    } else {
      why = `Google sign-in is configured but could not start: ${state.fbError || "the Firebase SDK did not load"}. ` +
        "It is downloaded from gstatic.com / cdn.jsdelivr.net, so check ad blockers, VPN or firewall.";
      retry = el("button", { class: "btn", type: "button", onclick: () => guarded(retry, async () => {
        try { await initFirebase(cfg.firebase); } catch (e) { state.fbError = e.message; }
        renderLogin();
      }) }, "Retry loading Google sign-in");
    }
    google.title = why;
    box.append(...[el("p", { class: "banner warn small", id: "google-why" }, why), retry].filter(Boolean));
  } else if (cfg.allowed_domains && cfg.allowed_domains.length) {
    box.append(el("p", { class: "small muted" }, "Limited to: " + cfg.allowed_domains.join(", ")));
  }

  // 2) Email + password (Firebase mode) or the development login (jwt mode)
  const alt = [];
  if (cfg.email_password_login && state.fb) {
    const email = el("input", { type: "email", placeholder: "agent@company.com", autocomplete: "username" });
    const pw = el("input", { type: "password", placeholder: "Password", autocomplete: "current-password" });
    const go = el("button", { class: "btn primary", type: "button", onclick: () => guarded(go, async () => {
      try { await state.fb.auth().signInWithEmailAndPassword(email.value.trim(), pw.value); await afterLogin(); }
      catch (e) { err.textContent = authErrorText(e); }
    }) }, "Sign in");
    alt.push(el("label", null, "Email"), email, el("label", null, "Password"), pw, go);
  }
  if (cfg.dev_login) {
    const uid = el("input", { value: "agent-1", "aria-label": "user id" });
    const role = el("select", { "aria-label": "role" }, el("option", { value: "agent" }, "Agent"), el("option", { value: "admin" }, "Admin"));
    const go = el("button", { class: "btn primary", type: "button", onclick: () => guarded(go, async () => {
      const r = await fetch(`/v1/auth/dev-token?uid=${encodeURIComponent(uid.value || "agent-1")}&role=${role.value}`, { method: "POST" });
      if (!r.ok) throw new Error("dev login unavailable");
      state.token = (await r.json()).access_token; sessionStorage.setItem("tra_token", state.token);
      await afterLogin();
    }) }, "Sign in (dev)");
    alt.push(el("p", { class: "banner info small" }, "Development login (AUTH_MODE=jwt) - not available in production."),
      el("label", null, "User id"), uid, el("label", null, "Role"), role, go);
  }
  if (alt.length) box.append(el("div", { class: "divider" }, el("span", null, "or")), ...alt);
  else if (!googleReady) box.append(el("p", { class: "banner bad" }, "Sign-in is not available for this configuration."));
  box.append(err);
  root.append(box);
}

// -------------------------------------------------------------------- shell
const TABS = [["resolve", "Resolve"], ["search", "Search"], ["admin", "Admin"]];
function renderShell() {
  const root = $("#app"); root.replaceChildren();
  const isAdmin = state.me.role === "admin";
  const nav = el("nav", { class: "tabs", role: "tablist" }, TABS.map(([id, label]) =>
    el("button", { role: "tab", "aria-selected": String(state.tab === id), disabled: id === "admin" && !isAdmin,
      title: id === "admin" && !isAdmin ? "Admin role required" : null, onclick: () => { state.tab = id; renderShell(); } }, label)));
  const header = el("header", { class: "top" }, el("span", { class: "brand" }, "Support Resolution Assistant"), nav, el("span", { class: "spacer" }),
    el("span", { class: "muted small", title: state.me.uid }, `${state.me.name || state.me.email || state.me.uid} · ${state.me.role}`,
      state.me.provider === "google.com" ? " · Google" : ""),
    state.cfg.auth_mode === "off" ? null : el("button", { class: "btn", onclick: () => logout(false) }, "Sign out"));
  const main = el("main", { id: "view" });
  root.append(header, main);
  ({ resolve: viewResolve, search: viewSearch, admin: viewAdmin })[state.tab](main);
}

// --------------------------------------------------------- Resolve (agents)
const EXAMPLES = [
  ["Broadband drops (brief)", "My broadband drops every evening around 8 and I've already restarted the router twice, I work from home and this is costing me"],
  ["Double charge", "I was charged twice for this month's bill and nobody has replied to my emails. This is unacceptable, I want a refund now!"],
  ["Roaming abroad", "I landed in Dubai yesterday and my phone has no network at all even though roaming is switched on."],
  ["5G home (new class?)", "My 5G home router only gets 4G and the speeds are poor, the outdoor unit won't lock onto the 5G band."],
  ["Out of domain", "How do I bake sourdough bread with a really crispy crust?"],
];
function viewResolve(main) {
  const ta = el("textarea", { id: "complaint", placeholder: "Paste the customer's complaint here…", "aria-label": "complaint" });
  const out = el("div", { id: "result" });
  const go = el("button", { class: "btn primary", type: "button" }, "Resolve");
  const run = () => guarded(go, async () => {
    const text = ta.value.trim();
    if (text.length < 5) throw new Error("Please paste a complaint first");
    out.replaceChildren(el("p", { class: "muted" }, el("span", { class: "spinner" }), " Analysing and retrieving similar cases…"));
    go.textContent = "Working…";
    const res = await api("/v1/resolve", { method: "POST", body: { text } });
    renderResult(out, res);
  }).then(() => { go.textContent = "Resolve"; });
  go.addEventListener("click", run);
  ta.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run(); });
  main.append(el("div", { class: "card" }, el("h2", null, "Customer complaint"), ta,
    el("div", { class: "row", style: "margin-top:.6rem" }, go, el("span", { class: "muted small" }, "Ctrl+Enter to submit · personal data is redacted before processing"),
      el("span", { class: "spacer", style: "flex:1" })),
    el("div", { class: "examples", style: "margin-top:.5rem" }, el("span", { class: "muted small" }, "Try: "),
      EXAMPLES.map(([label, text]) => el("button", { class: "chip", type: "button", onclick: () => { ta.value = text; ta.focus(); } }, label)))), out);
}

function sevBadge(s) { return el("span", { class: `badge sev-${s}` }, s); }
function sentBadge(s, score) { return el("span", { class: `badge ${s === "negative" ? "neg" : s === "positive" ? "pos" : "neu"}` }, `${s} (${score})`); }

function renderResult(out, r) {
  const a = r.analysis, res = r.resolution;
  out.replaceChildren();
  state.lastQuery = r.query_id;

  const analysis = el("div", { class: "card" }, el("h2", null, "Parsed complaint"),
    el("dl", { class: "kv" },
      el("dt", null, "Category"), el("dd", null, a.is_known_category ? a.category : "Unknown / new type", " ",
        el("span", { class: "muted small", title: "Calibrated: of complaints scored like this, about this share had the right category" }, `${pct(a.category_confidence)} sure`)),
      el("dt", null, "Intent"), el("dd", null, a.intent.replaceAll("_", " ")),
      el("dt", null, "Product"), el("dd", null, a.product || "—"),
      el("dt", null, "Severity"), el("dd", null, sevBadge(a.severity), " ", a.severity_signals.map((s) => el("span", { class: "chip" }, s.replaceAll("_", " ")))),
      el("dt", null, "Sentiment"), el("dd", null, sentBadge(a.sentiment, a.sentiment_score))));

  const c = r.confidence;
  const lvlClass = { high: "ok", medium: "warnb", low: "neg" }[c.level];
  const confidence = el("div", { class: "card", id: "confidence" }, el("h2", null, "Confidence"),
    el("div", { class: "row" }, el("span", { class: `badge ${lvlClass}` }, `${c.level} · ${pct(c.score)}`),
      el("span", { class: "muted small" }, `closest cases agree on the fix: ${pct(c.agreement)} · best match strength: ${pct(c.relevance)}`)),
    el("div", { class: "bar", style: "margin:.4rem 0" }, el("i", { style: `width:${Math.max(2, c.score * 100)}%` })),
    el("ul", { class: "small muted", style: "margin:.2rem 0 0 1rem;padding:0" }, c.reasons.map((x) => el("li", null, x))));
  const banners = [];
  if (c.level === "low" && !res.escalate) {
    banners.push(el("div", { class: "banner warn" },
      el("strong", null, "Low confidence - verify before using. "),
      "The closest resolved cases describe different problems, so these steps are tentative."));
  }
  c.clarifying_questions.forEach((q) => banners.push(el("div", { class: "banner info" }, el("strong", null, "Need more detail: "), q)));
  if (!a.is_known_category) {
    banners.push(el("div", { class: "banner info" },
      "This doesn't match any known ticket class - it may be a new type of problem. " +
      "Admins can review it under Admin → Emerging."));
  }
  if (res.priority_flag) banners.push(el("div", { class: "banner warn" }, "Priority handling recommended (high customer impact)."));
  if (res.escalate) banners.push(el("div", { class: "banner bad" }, el("strong", null, "Escalate to Tier-2. "), res.escalation_reason || "No grounded resolution available."));

  const stepsList = el("ol", { class: "steps" }, res.steps.map((s) => el("li", null, s.text,
    s.citations.map((c) => el("button", { class: "cite", type: "button", title: "Show source", onclick: () => highlight(c) }, c)))));
  const copy = el("button", { class: "btn", type: "button", onclick: () => copyText(res) }, "Copy for ticket");
  const resolution = el("div", { class: "card" }, el("h2", null, "Suggested resolution"), banners,
    res.summary ? el("p", { class: "muted" }, res.summary) : null,
    res.steps.length ? stepsList : null,
    el("div", { class: "row small muted", style: "margin-top:.5rem" },
      el("span", null, `mode: ${res.generation_mode}`), el("span", null, `grounding: ${pct(res.grounding_score)}`),
      el("span", null, `${Math.round(r.latency_ms)} ms`), r.cached ? el("span", { class: "chip" }, "cached") : null,
      el("span", { class: "mono" }, r.index_version), el("span", { style: "flex:1" }), res.steps.length ? copy : null));

  const sources = el("div", { class: "card" }, el("h2", null, "Sources"),
    r.sources.length ? r.sources.map(srcCard) : el("p", { class: "muted" }, "No similar documents found."));
  out.append(el("div", { class: "grid g2" }, el("div", null, confidence, resolution, feedbackCard(r)), el("div", null, analysis, sources)));
}

function srcCard(s) {
  const rel = s.relevance ?? 0;
  return el("div", { class: "src", id: "src-" + s.id },
    el("div", { class: "head" }, el("span", { class: `badge ${s.doc_type === "ticket" ? "" : "ok"}` }, s.id),
      el("span", { class: "title", title: s.title }, s.title || "(untitled)"),
      s.origin === "hf" ? el("span", { class: "chip", title: "Imported from the Hugging Face dataset" }, "HF") : null,
      el("span", { class: "muted small mono", title: `Match strength, calibrated to the embedder. Raw cosine similarity: ${s.score.toFixed(2)}` }, pct(rel))),
    el("div", { class: "bar" }, el("i", { style: `width:${Math.max(2, Math.min(100, rel * 100))}%` })),
    el("div", { class: "small" }, s.doc_type === "ticket" ? "Past ticket" : "KB article", s.category ? ` · ${s.category}` : "", s.product ? ` · ${s.product}` : ""),
    s.snippet ? el("div", { class: "small muted", style: "margin-top:.25rem" }, s.snippet) : null);
}
function highlight(id) {
  document.querySelectorAll(".src.hl").forEach((n) => n.classList.remove("hl"));
  const n = document.getElementById("src-" + id);
  if (n) { n.classList.add("hl"); if (n.scrollIntoView) n.scrollIntoView({ behavior: "smooth", block: "center" }); }
}
async function copyText(res) {
  const txt = (res.summary ? res.summary + "\n\n" : "") + res.steps.map((s) => `${s.n}. ${s.text} [${s.citations.join(", ")}]`).join("\n");
  try { await navigator.clipboard.writeText(txt); toast("Copied"); } catch { toast("Copy failed - select the text manually", true); }
}

function feedbackCard(r) {
  const catSel = el("select", { "aria-label": "correct category" }, el("option", { value: "" }, "— category is right / unsure —"),
    state.cats.filter((c) => c.active).map((c) => el("option", { value: c.name }, c.name)));
  const steps = el("textarea", { placeholder: "Optional - how did you actually resolve it? One step per line. This becomes a new resolved ticket the system learns from.", style: "min-height:80px" });
  const rating = el("select", { "aria-label": "rating" }, el("option", { value: "" }, "Rating"), [5, 4, 3, 2, 1].map((n) => el("option", { value: n }, "★".repeat(n))));
  const send = (helpful, btn) => guarded(btn, async () => {
    const lines = steps.value.split("\n").map((s) => s.trim()).filter(Boolean);
    const body = { query_id: r.query_id, helpful, rating: rating.value ? Number(rating.value) : null,
      correct_category: catSel.value || null, resolved_steps: lines.length ? lines : null };
    const resp = await api("/v1/feedback", { method: "POST", body });
    toast(resp.promoted_ticket ? "Thanks - your resolution was added to the knowledge base" : "Thanks for the feedback");
  });
  const up = el("button", { class: "btn", type: "button", onclick: () => send(true, up) }, "👍 Helpful");
  const down = el("button", { class: "btn", type: "button", onclick: () => send(false, down) }, "👎 Not helpful");
  return el("div", { class: "card" }, el("h2", null, "Feedback"), el("div", { class: "stack" }, el("div", { class: "row" }, up, down, rating), catSel, steps));
}

// ----------------------------------------------------------------- Search
function viewSearch(main) {
  const q = el("input", { placeholder: "Describe a problem in your own words…", "aria-label": "query" });
  const type = el("select", { "aria-label": "type" }, el("option", { value: "all" }, "Tickets + KB"), el("option", { value: "ticket" }, "Tickets"), el("option", { value: "kb" }, "KB articles"));
  const cat = el("select", { "aria-label": "category filter" }, el("option", { value: "" }, "All categories"), state.cats.filter((c) => c.active).map((c) => el("option", { value: c.name }, c.name)));
  const out = el("div");
  const go = el("button", { class: "btn primary", type: "button" }, "Search");
  const run = () => guarded(go, async () => {
    if (q.value.trim().length < 5) throw new Error("Type at least a few words");
    const r = await api("/v1/search", { method: "POST", body: { text: q.value.trim(), top_k: 10, doc_type: type.value, category: cat.value || null } });
    out.replaceChildren(el("div", { class: "card" }, el("h2", null, `${r.results.length} results`, " ", el("span", { class: "muted small mono" }, r.index_version)),
      r.results.length ? r.results.map(srcCard) : el("p", { class: "muted" }, "Nothing found.")));
  });
  go.addEventListener("click", run);
  q.addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
  main.append(el("div", { class: "card" }, el("h2", null, "Semantic search"), el("p", { class: "muted small" }, "Finds past tickets and KB articles by meaning, not keywords."),
    q, el("div", { class: "row", style: "margin-top:.6rem" }, type, cat, go)), out);
}

const viewAdmin = window.createAdminView({ state, $, api, el, guarded, toast, pct, sleep, renderShell });
// --------------------------------------------------------------------- boot
async function boot() {
  try {
    state.cfg = await (await fetch("/v1/config")).json();
    if (state.cfg.auth_mode !== "off" && state.cfg.firebase) {
      try {
        await initFirebase(state.cfg.firebase);
        state.fb.auth().onAuthStateChanged((u) => { if (u && !state.me) afterLogin().catch((e) => toast(e.message, true)); });
      } catch (e) { state.fb = null; state.fbError = e.message; console.warn("Firebase SDK unavailable:", e.message); }
    }
    if (state.cfg.auth_mode === "off") return afterLogin();
    state.token = sessionStorage.getItem("tra_token");
    if (state.token) { try { return await afterLogin(); } catch { state.token = null; } }
    renderLogin();
  } catch (e) {
    $("#app").replaceChildren(el("p", { class: "banner bad", style: "margin:2rem" }, "Could not reach the API: " + e.message));
  }
}
window.__tra = { state, api };   // handy for debugging in the browser console
boot();
})();
