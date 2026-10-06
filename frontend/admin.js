/* Admin screens and actions, isolated from the shared app shell. */
window.createAdminView = function createAdminView(app) {
  const { state, $, api, el, guarded, toast, pct, sleep, renderShell } = app;

// ------------------------------------------------------------------- Admin
const ADMIN_TABS = [["overview", "Overview"], ["emerging", "Emerging"], ["taxonomy", "Classes"], ["data", "Data"], ["index", "Index"]];
function viewAdmin(main) {
  const body = el("div");
  main.append(el("div", { class: "subtabs", role: "tablist" }, ADMIN_TABS.map(([id, label]) =>
    el("button", { role: "tab", "aria-selected": String(state.adminTab === id), onclick: () => { state.adminTab = id; renderShell(); } }, label))), body);
  ({ overview: adminOverview, emerging: adminEmerging, taxonomy: adminTaxonomy, data: adminData, index: adminIndex })[state.adminTab](body);
}

async function pollJob(id, bar, label) {
  for (let i = 0; i < 900; i++) {
    const j = await api(`/v1/jobs/${id}`);
    if (bar && j.total) bar.style.width = pct(j.processed / j.total);
    if (label) label.textContent = `${j.status} · ${j.processed}/${j.total} · +${j.inserted} new, ${j.updated} updated, ${j.skipped} unchanged`;
    if (j.status === "done") { if (bar) bar.style.width = "100%"; return j; }
    if (j.status === "failed") throw new Error(j.error || "job failed");
    await sleep(700);
  }
  throw new Error("timed out waiting for job");
}

function metric(label, value) { return el("div", { class: "card metric" }, String(value ?? "—"), el("small", null, label)); }

async function adminOverview(root) {
  root.append(el("p", { class: "muted" }, el("span", { class: "spinner" }), " Loading…"));
  const [stats, drift, jobs] = await Promise.all([api("/v1/admin/stats"), api("/v1/admin/drift"), api("/v1/admin/jobs")]).catch((e) => { toast(e.message, true); return []; });
  root.replaceChildren();
  if (!stats) return;
  const empty = stats.tickets_db === 0;
  if (empty) {
    const btn = el("button", { class: "btn primary", type: "button" }, "Load demo data");
    btn.addEventListener("click", () => guarded(btn, async () => {
      const j = await api("/v1/admin/seed-demo", { method: "POST" }); btn.textContent = "Loading…"; await pollJob(j.job_id); toast("Demo data loaded"); renderShell();
    }));
    root.append(el("div", { class: "banner info" }, "The knowledge base is empty. ", btn, el("span", { class: "muted small" }, " (528 synthetic telecom tickets + 24 KB articles)")));
  }
  root.append(el("div", { class: "grid g4" }, metric("tickets in DB", stats.tickets_db), metric("KB articles in DB", stats.kb_db),
    metric("ticket vectors", stats.vectors_ticket), metric("KB vectors", stats.vectors_kb)));
  const cur = drift.current || {}, prev = drift.previous || {};
  root.append(el("div", { class: "card" }, el("h2", null, `Health & drift (last ${drift.window_days} days vs previous)`),
    drift.flags.length ? drift.flags.map((f) => el("div", { class: "banner warn" }, f)) : el("div", { class: "banner info" }, cur.n ? "No drift flags." : "No traffic yet - flags appear once agents use the tool."),
    el("table", null, el("thead", null, el("tr", null, ["metric", "current", "previous"].map((h) => el("th", null, h)))),
      el("tbody", null, ["n", "unknown_rate", "escalation_rate", "top_score_mean", "top_score_p10", "latency_p95_ms"].map((k) =>
        el("tr", null, el("td", null, k), el("td", null, cur[k] ?? "—"), el("td", null, prev[k] ?? "—"))))),
    el("p", { class: "small muted" }, `Helpful ratio: ${drift.helpful_ratio ?? "no feedback yet"} · embedder: ${stats.embedder} · index: ${stats.index}`)));
  const audit = el("button", { class: "btn", type: "button" }, "Verify audit chain");
  audit.addEventListener("click", () => guarded(audit, async () => { const v = await api("/v1/admin/audit/verify"); toast(v.valid ? `Audit chain intact (${v.entries} entries)` : `TAMPERING DETECTED at entry ${v.broken_at_id}`, !v.valid); }));
  root.append(el("div", { class: "card" }, el("h2", null, "Recent jobs"), audit, jobsTable(jobs || [])));
}
function jobsTable(jobs) {
  if (!jobs.length) return el("p", { class: "muted" }, "No jobs yet.");
  return el("table", null, el("thead", null, el("tr", null, ["kind", "status", "progress", "new/updated/skipped", "error"].map((h) => el("th", null, h)))),
    el("tbody", null, jobs.map((j) => el("tr", null, el("td", null, j.kind), el("td", null, el("span", { class: `badge ${j.status === "done" ? "ok" : j.status === "failed" ? "neg" : "warnb"}` }, j.status)),
      el("td", null, `${j.processed}/${j.total}`), el("td", null, `${j.inserted}/${j.updated}/${j.skipped}`), el("td", { class: "small" }, j.error || "")))));
}

async function adminEmerging(root) {
  root.append(el("p", { class: "muted" }, el("span", { class: "spinner" }), " Clustering unmatched complaints…"));
  const clusters = await api("/v1/admin/emerging").catch((e) => { toast(e.message, true); return []; });
  root.replaceChildren(el("div", { class: "card" }, el("h2", null, "Emerging problem types"),
    el("p", { class: "muted small" }, "Complaints that matched no known class, grouped by similarity. A large cluster is a candidate for a new class.")));
  if (!clusters.length) root.append(el("div", { class: "card muted" }, "Nothing to review. Try the “5G home (new class?)” example on the Resolve tab a few times (after loading demo data), then come back."));
  clusters.forEach((c) => root.append(el("div", { class: "card" }, el("div", { class: "row" }, el("strong", null, `${c.size} complaints`), c.top_terms.map((t) => el("span", { class: "chip" }, t)),
    el("span", { style: "flex:1" }), el("button", { class: "btn primary", type: "button", onclick: () => { state.prefill = c.examples; state.adminTab = "taxonomy"; renderShell(); } }, "Create class from this")),
    c.examples.map((e) => el("p", { class: "small muted" }, "“" + e + "”")))));
}

async function adminTaxonomy(root) {
  const cats = await api("/v1/categories"); state.cats = cats;
  const active = cats.filter((c) => c.active);
  const rows = cats.map((c) => {
    const target = el("select", { "aria-label": "merge target" }, el("option", { value: "" }, "merge into…"), active.filter((x) => x.name !== c.name).map((x) => el("option", { value: x.name }, x.name)));
    const btn = el("button", { class: "btn", type: "button" }, "Merge");
    btn.addEventListener("click", () => guarded(btn, async () => {
      if (!target.value) throw new Error("Pick a target class");
      if (!confirm(`Merge "${c.name}" into "${target.value}"? This relabels ${c.tickets} tickets.`)) return;
      const r = await api(`/v1/admin/categories/${encodeURIComponent(c.name)}/merge`, { method: "POST", body: { into: target.value } });
      toast(`Merged ${r.tickets_moved} tickets`); renderShell();
    }));
    return el("tr", null, el("td", null, c.name), el("td", { class: "small muted" }, c.description || ""), el("td", null, c.tickets),
      el("td", null, c.active ? el("span", { class: "badge ok" }, "active") : el("span", { class: "badge" }, c.merged_into ? `→ ${c.merged_into}` : "inactive")),
      el("td", null, c.active ? el("div", { class: "row" }, target, btn) : ""));
  });
  root.append(el("div", { class: "card" }, el("h2", null, "Ticket classes"),
    el("table", null, el("thead", null, el("tr", null, ["class", "description", "tickets", "status", ""].map((h) => el("th", null, h)))), el("tbody", null, rows))));

  // --- add a class
  const name = el("input", { placeholder: "e.g. 5G Home Internet" }), desc = el("input", { placeholder: "short description (optional)" });
  const seedsBox = el("div");
  const addSeed = (complaint = "", steps = "") => seedsBox.append(el("div", { class: "seed" },
    el("label", null, "Example complaint"), el("textarea", { style: "min-height:60px", class: "s-c" }, complaint),
    el("label", null, "How it was resolved (one step per line)"), el("textarea", { style: "min-height:60px", class: "s-s" }, steps)));
  (state.prefill || [""]).forEach((t) => addSeed(t)); state.prefill = null;
  const submit = el("button", { class: "btn primary", type: "button" }, "Create class");
  const bar = el("i"), lab = el("span", { class: "small muted" });
  submit.addEventListener("click", () => guarded(submit, async () => {
    if (name.value.trim().length < 2) throw new Error("Give the class a name");
    const seeds = [...seedsBox.querySelectorAll(".seed")].map((d) => ({ body: $(".s-c", d).value.trim(), resolution_steps: $(".s-s", d).value.split("\n").map((x) => x.trim()).filter(Boolean) }))
      .filter((s) => s.body.length >= 5 && s.resolution_steps.length);
    const r = await api("/v1/admin/categories", { method: "POST", body: { name: name.value.trim(), description: desc.value.trim(), seed_examples: seeds } });
    if (r.job_id) await pollJob(r.job_id, bar, lab);
    toast(`Class created${seeds.length ? ` with ${seeds.length} examples` : ""}`); renderShell();
  }));
  root.append(el("div", { class: "card" }, el("h2", null, "Add a new class"),
    el("p", { class: "muted small" }, "The classifier learns from examples instantly - no retraining. 3-10 examples with resolution steps work well; examples without steps are skipped (they can't ground an answer)."),
    el("label", null, "Name"), name, el("label", null, "Description"), desc, el("h3", { style: "margin-top:1rem" }, "Seed examples"), seedsBox,
    el("div", { class: "row" }, el("button", { class: "btn", type: "button", onclick: () => addSeed() }, "+ Add example"), submit, lab), el("div", { class: "progress", style: "margin-top:.5rem" }, bar)));
}

function parseUpload(text, filename, kind) {
  const key = kind === "tickets" ? "tickets" : "articles";
  if (/\.jsonl$/i.test(filename)) return text.split(/\r?\n/).filter((l) => l.trim()).map((l) => JSON.parse(l));
  const j = JSON.parse(text);
  return Array.isArray(j) ? j : j[key];
}
function hfCard(bar, lab) {
  const limit = el("input", { type: "number", value: "3000", min: "100", max: "50000", "aria-label": "ticket limit", style: "max-width:8rem" });
  const kbLimit = el("input", { type: "number", value: "300", min: "0", max: "5000", "aria-label": "KB article limit", style: "max-width:8rem" });
  const go = el("button", { class: "btn primary", type: "button" }, "Import Hugging Face dataset");
  go.addEventListener("click", () => guarded(go, async () => {
    go.textContent = "Importing…";
    const j = await api(`/v1/admin/import-hf?limit=${Number(limit.value) || 3000}&kb_limit=${Number(kbLimit.value) || 0}`, { method: "POST" });
    await pollJob(j.job_id, bar, lab);
    toast("Hugging Face data imported");
  }));
  return el("div", { class: "card" }, el("h2", null, "Hugging Face dataset"),
    el("p", { class: "muted small" },
      "Adds resolved tickets from the public Tobi-Bueck/customer-support-tickets dataset " +
      "and derives KB articles from its best-documented cases. Needs internet access and the " +
      "`datasets` package on the server (or run `python -m data.load_hf` locally). " +
      "Documents are labelled “HF”."),
    el("div", { class: "row" }, el("label", { class: "small muted" }, "tickets"), limit, el("label", { class: "small muted" }, "KB articles"), kbLimit, go));
}
async function adminData(root) {
  const kind = el("select", { "aria-label": "data type" }, el("option", { value: "tickets" }, "Resolved tickets"), el("option", { value: "kb" }, "KB articles"));
  const file = el("input", { type: "file", accept: ".json,.jsonl" });
  const bar = el("i"), lab = el("span", { class: "small muted" });
  const go = el("button", { class: "btn primary", type: "button" }, "Upload & index");
  go.addEventListener("click", () => guarded(go, async () => {
    if (!file.files[0]) throw new Error("Choose a .json or .jsonl file");
    const items = parseUpload(await file.files[0].text(), file.files[0].name, kind.value);
    if (!Array.isArray(items) || !items.length) throw new Error("No records found in file");
    const size = kind.value === "tickets" ? 500 : 100, path = kind.value === "tickets" ? "/v1/ingest/tickets" : "/v1/ingest/kb", key = kind.value === "tickets" ? "tickets" : "articles";
    for (let i = 0; i < items.length; i += size) {
      const j = await api(path, { method: "POST", body: { [key]: items.slice(i, i + size) } });
      await pollJob(j.job_id, bar, lab);
      lab.textContent += `  (batch ${Math.floor(i / size) + 1}/${Math.ceil(items.length / size)})`;
    }
    toast(`Ingested ${items.length} records`);
  }));
  const demo = el("button", { class: "btn", type: "button" }, "Load demo data"), demoNovel = el("button", { class: "btn", type: "button" }, "Load demo data + 5G class");
  for (const [b, novel] of [[demo, false], [demoNovel, true]]) b.addEventListener("click", () => guarded(b, async () => {
    const j = await api(`/v1/admin/seed-demo?novel=${novel}`, { method: "POST" }); await pollJob(j.job_id, bar, lab); toast("Demo data loaded");
  }));
  root.append(el("div", { class: "card" }, el("h2", null, "Import data"),
    el("p", { class: "muted small" },
      "Idempotent: re-uploading unchanged records is skipped; edited records are updated. " +
      "Tickets need: body, resolution_steps (list or text), optional subject/category/product/" +
      "severity/external_id. KB articles need: external_id, title, body."),
    el("div", { class: "row" }, kind, file, go), el("div", { class: "progress", style: "margin:.6rem 0 .3rem" }, bar), lab),
    el("div", { class: "card" }, el("h2", null, "Demo data"), el("div", { class: "row" }, demo, demoNovel)),
    hfCard(bar, lab));
}

async function adminIndex(root) {
  const stats = await api("/v1/admin/stats");
  const backend = el("select", { "aria-label": "backend" }, el("option", { value: "" }, "(keep current)"), ["sentence-transformers", "hashing"].map((b) => el("option", { value: b }, b)));
  const model = el("input", { placeholder: "model name (optional), e.g. sentence-transformers/all-MiniLM-L6-v2" });
  const bar = el("i"), lab = el("span", { class: "small muted" });
  const go = el("button", { class: "btn primary", type: "button" }, "Start re-index");
  go.addEventListener("click", () => guarded(go, async () => {
    if (!confirm("Build a new index in the background and switch over when ready? Searches keep working meanwhile.")) return;
    const j = await api("/v1/admin/reindex", { method: "POST", body: { embedding_backend: backend.value || null, embedding_model: model.value.trim() || null } });
    await pollJob(j.job_id, bar, lab); toast("Re-index complete - index switched"); renderShell();
  }));
  root.append(el("div", { class: "card" }, el("h2", null, "Search index"),
    el("dl", { class: "kv" }, el("dt", null, "Active index"), el("dd", { class: "mono" }, stats.index), el("dt", null, "Embedder"), el("dd", { class: "mono" }, stats.embedder),
      el("dt", null, "Dimensions"), el("dd", null, stats.dim), el("dt", null, "Vectors"), el("dd", null, `${stats.vectors_ticket} tickets · ${stats.vectors_kb} KB chunks`))),
    el("div", { class: "card" }, el("h2", null, "Blue/green re-index"),
      el("p", { class: "muted small" }, "Use when changing the embedding model. A new index is built from the database, caught up, then switched atomically. After switching, re-run threshold calibration (see docs)."),
      el("div", { class: "grid g2" }, el("div", null, el("label", null, "Embedding backend"), backend), el("div", null, el("label", null, "Model"), model)),
      el("div", { class: "row", style: "margin-top:.6rem" }, go, lab), el("div", { class: "progress", style: "margin-top:.5rem" }, bar)));
}


  return viewAdmin;
};
