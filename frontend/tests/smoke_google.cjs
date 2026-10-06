// Google sign-in UI test. The real Google popup cannot run in a test, so a stub of the Firebase JS SDK stands in for it:
// signInWithPopup() "signs in" a user whose ID token is a valid dev JWT. That exercises everything on OUR side:
// the button, the login flow, RBAC from /v1/me, the header identity, and sign-out.
//   Start the API with Firebase configured (see README), then:  node smoke_google.cjs http://localhost:8000 [agent|admin] [mode]
//   mode = normal (default) | fallback (gstatic blocked, jsDelivr works) | blocked (both blocked: reason + Retry shown)
const { JSDOM, ResourceLoader } = require("jsdom");
const base = process.argv[2] || "http://localhost:8000";
let failures = 0;
const check = (name, ok, extra = "") => { console.log(`${ok ? "PASS" : "FAIL"}  ${name} ${extra}`); if (!ok) failures++; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const mode = process.argv[4] || "normal";
// The stub SDK is delivered THROUGH the CDN loader, so the real loading code (gstatic -> jsDelivr fallback) is exercised.
const STUB_JS = "window.firebase = { initializeApp(){}, apps: [], auth: Object.assign(() => window.__auth(), { GoogleAuthProvider: function(){ this.setCustomParameters = () => {}; } }) };";
class Loader extends ResourceLoader {
  fetch(url, o) {
    if (/gstatic\.com\/firebasejs/.test(url)) return mode === "normal" ? Promise.resolve(Buffer.from(STUB_JS)) : Promise.reject(new Error("blocked"));
    if (/cdn\.jsdelivr\.net\/npm\/firebase/.test(url)) return mode === "fallback" ? (globalThis.__jsd = (globalThis.__jsd || 0) + 1, Promise.resolve(Buffer.from(STUB_JS))) : Promise.reject(new Error("blocked"));
    return super.fetch(url, o);
  }
}

(async () => {
  const role = process.argv[3] || "agent";
  const tok = (await (await fetch(`${base}/v1/auth/dev-token?uid=g-user&role=${role}`, { method: "POST" })).json()).access_token;
  let current = null, popups = 0;
  const auth = () => ({ currentUser: current, onAuthStateChanged() {}, signOut: async () => { current = null; },
    signInWithPopup: async () => { popups++; current = { getIdToken: async () => tok }; } });
  const dom = await JSDOM.fromURL(base + "/ui/", {
    runScripts: "dangerously", resources: new Loader(), pretendToBeVisual: true,
    beforeParse(w) {
      w.fetch = (u, o) => fetch(new URL(u, base).href, o); w.confirm = () => true; w.HTMLElement.prototype.scrollIntoView = function () {};
      w.__auth = auth;   // the stub SDK (served by the loader above) delegates to this
    },
  });
  const w = dom.window, d = w.document;
  const $ = (s) => d.querySelector(s), $$ = (s) => [...d.querySelectorAll(s)];
  const waitFor = async (fn, ms = 15000) => { const t = Date.now(); while (Date.now() - t < ms) { try { const v = fn(); if (v) return v; } catch {} await sleep(100); } return null; };

  check("login screen renders", await waitFor(() => $(".login")));
  const g = $("#google-signin");
  if (mode === "blocked") {
    check("SDK blocked: Google button disabled", !!g && g.disabled);
    check("SDK blocked: the real reason is shown (not 'not configured')", /could not start/.test($("#google-why").textContent) && /gstatic\.com/.test($("#google-why").textContent));
    check("SDK blocked: a Retry button is offered", !!$$("button").find((b) => /Retry loading Google/.test(b.textContent)));
    check("no stray 'null' text on the login card", !/null/.test($(".login").textContent));
    console.log(failures ? `\n${failures} FAILED` : "\nAll Google sign-in UI checks passed");
    process.exit(failures ? 1 : 0);
  }
  if (mode === "fallback") check("gstatic blocked: SDK loaded from the jsDelivr fallback", (globalThis.__jsd || 0) >= 1);
  check("Google button is enabled when Firebase is configured", !!g && !g.disabled);
  check("dev login is offered alongside Google ('or' divider)", !!$(".divider") && !!$$("button").find((b) => /Sign in \(dev\)/.test(b.textContent)));
  g.click();
  check("signing in with Google reaches the app shell", await waitFor(() => $("header.top")) && popups === 1);
  const head = $("header.top").textContent;
  check(`header shows identity and role (${role})`, head.includes("g-user") && head.includes(role));
  const adminTab = $$("button").find((b) => b.textContent.trim() === "Admin");
  check(`RBAC: Admin tab ${role === "admin" ? "enabled" : "disabled"} for a Google ${role}`, adminTab.disabled === (role !== "admin"));
  $$("button").find((b) => b.textContent.trim() === "Sign out").click();
  check("sign out returns to the login screen", await waitFor(() => $(".login")));
  console.log(failures ? `\n${failures} FAILED` : "\nAll Google sign-in UI checks passed");
  process.exit(failures ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
