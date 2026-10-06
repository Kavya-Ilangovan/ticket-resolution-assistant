// End-to-end UI smoke test: drives the real SPA (jsdom) against a running API.
//   1) start the API (see README "Frontend tests")   2) cd frontend/tests && npm i && node smoke.cjs http://localhost:8000
const { JSDOM } = require("jsdom");
const base = process.argv[2] || "http://localhost:8000";
let failures = 0;
const check = (name, ok, extra = "") => { console.log(`${ok ? "PASS" : "FAIL"}  ${name} ${extra}`); if (!ok) failures++; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const dom = await JSDOM.fromURL(base + "/ui/", {
    runScripts: "dangerously", resources: "usable", pretendToBeVisual: true,
    beforeParse(w) { w.fetch = (u, o) => fetch(new URL(u, base).href, o); w.confirm = () => true; w.HTMLElement.prototype.scrollIntoView = function () {}; },
  });
  const w = dom.window, d = w.document;
  const $ = (s) => d.querySelector(s), $$ = (s) => [...d.querySelectorAll(s)];
  const waitFor = async (fn, ms = 20000) => { const t = Date.now(); while (Date.now() - t < ms) { try { const v = fn(); if (v) return v; } catch {} await sleep(100); } return null; };
  const btn = (text) => $$("button").find((b) => b.textContent.trim().startsWith(text));
  const setVal = (e, v) => { e.value = v; e.dispatchEvent(new w.Event("input", { bubbles: true })); };

  check("login screen renders", await waitFor(() => $(".login")));
  check("Google sign-in is shown (disabled + explained when Firebase is not configured)",
    !!$("#google-signin") && $("#google-signin").disabled && /not configured/.test($(".login").textContent));
  setVal($(".login select"), "admin");
  btn("Sign in (dev)").click();
  check("shell after login (admin)", await waitFor(() => $("header.top") && /admin/.test($("header.top").textContent)));

  // Resolve tab with empty KB -> then load demo data from Admin
  btn("Admin").click();
  check("admin overview + empty-KB banner", await waitFor(() => btn("Load demo data")));
  btn("Load demo data").click();
  check("demo data loaded (toast)", await waitFor(() => /Demo data loaded/.test($("#toasts").textContent), 60000));
  check("metrics show ticket counts", await waitFor(() => /528/.test($("main").textContent)));

  // Resolve the brief's example complaint
  btn("Resolve").click();
  await waitFor(() => $("#complaint"));
  setVal($("#complaint"), "My broadband drops every evening around 8 and I've already restarted the router twice, I work from home and this is costing me");
  const actionBtns = $$("button").filter((b) => b.textContent.trim() === "Resolve");
  actionBtns[actionBtns.length - 1].click();
  check("result renders steps", await waitFor(() => $$("ol.steps li").length >= 3, 30000), `(${$$("ol.steps li").length} steps)`);
  check("analysis shows category", /Broadband Connectivity/.test($("#result").textContent));
  check("confidence card shows level and percentage", /Confidence/.test($("#confidence").textContent) && /(high|medium|low) · \d+%/.test($("#confidence").textContent));
  check("sources show calibrated match strength", /\d+%/.test($(".src .head").textContent));
  check("severity badge present", !!$("#result .badge[class*='sev-']"));
  check("citation chips present", $$(".cite").length > 0);
  $(".cite").click();
  check("clicking a citation highlights its source", !!$(".src.hl"));
  check("sources listed (tickets and KB)", $$(".src").length >= 5);

  // feedback + promotion
  const ta = $$("textarea").pop(); setVal(ta, "Replaced the patch cord\nConfirmed stable for 48h");
  btn("👍").click();
  check("feedback accepted + promoted", await waitFor(() => /added to the knowledge base/.test($("#toasts").textContent)));

  // XSS safety: markup in the complaint must stay inert
  setVal($("#complaint"), "<img src=x onerror=window.__pwned=1> my router keeps rebooting <script>window.__pwned=1</script>");
  const rb = $$("button").filter((b) => b.textContent.trim() === "Resolve"); rb[rb.length - 1].click();
  await waitFor(() => $("#result .card"), 20000); await sleep(300);
  check("no injected elements / script execution", !$("#result img") && !w.__pwned);

  // a vague 3-word complaint: on-topic results, honest uncertainty, and a clarifying question
  setVal($("#complaint"), "wifi not working");
  const rw = $$("button").filter((b) => b.textContent.trim() === "Resolve"); rw[rw.length - 1].click();
  check("vague complaint asks for more detail", await waitFor(() => /Need more detail/.test($("#result").textContent), 20000));
  check("vague complaint shows no billing / SIM / TV sources", !/Billing & Payments|SIM & Activation|TV & Streaming|Roaming/.test($$(".src").map((n) => n.textContent).join(" ")));

  // out-of-domain -> escalation banner
  setVal($("#complaint"), "How do I bake sourdough bread with a really crispy crust?");
  const rb2 = $$("button").filter((b) => b.textContent.trim() === "Resolve"); rb2[rb2.length - 1].click();
  check("out-of-domain shows escalation", await waitFor(() => /Escalate to Tier-2/.test($("#result").textContent), 20000));

  // search tab
  btn("Search").click(); await waitFor(() => $("input[aria-label=query]"));
  setVal($("input[aria-label=query]"), "double charged on my invoice");
  $$("button").find((b) => b.textContent.trim() === "Search" && !b.getAttribute("role")).click();
  check("search returns results", await waitFor(() => $$(".src").length > 0));

  // admin: classes, add class, data, index
  btn("Admin").click(); await waitFor(() => btn("Classes"));
  btn("Classes").click();
  check("classes table lists 8 active classes", await waitFor(() => $$("tbody tr").length >= 8));
  setVal($$("input").find((i) => /5G Home/.test(i.placeholder)), "Smart Home");
  $$(".s-c")[0].value = "My smart thermostat will not pair with the hub"; $$(".s-s")[0].value = "Reset the thermostat\nRe-pair it with the hub";
  btn("Create class").click();
  check("class created via UI", await waitFor(() => /Class created/.test($("#toasts").textContent), 30000));
  btn("Classes").click();
  check("new class listed", await waitFor(() => /Smart Home/.test($("main").textContent)));
  btn("Index").click();
  check("index tab shows active index", await waitFor(() => /support_hashing_v1/.test($("main").textContent)));
  btn("Emerging").click();
  check("emerging tab renders", await waitFor(() => /Emerging problem types/.test($("main").textContent)));
  btn("Data").click();
  check("data tab renders upload form", await waitFor(() => btn("Upload & index")));
  check("data tab offers the Hugging Face import", !!btn("Import Hugging Face dataset"));

  // agent role cannot open admin
  btn("Sign out").click();
  await waitFor(() => $(".login")); setVal($(".login select"), "agent"); btn("Sign in (dev)").click();
  await waitFor(() => $("header.top"));
  check("agent: admin tab disabled", btn("Admin").disabled === true);

  console.log(failures ? `\n${failures} FAILED` : "\nAll UI checks passed");
  process.exit(failures ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
