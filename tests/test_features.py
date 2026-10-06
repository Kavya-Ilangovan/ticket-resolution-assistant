"""Confidence, short-query relevance, Hugging Face import and Google sign-in / RBAC."""
import csv

import pytest

from app.core import auth
from app.core.vectorstore import Hit
from app.services import confidence as conf
from app.services import hf_import, indexer, resolver
from evals import run_all as ev

OFF_TOPIC_FOR_WIFI = {"Billing & Payments", "Roaming & International", "Plans & Account Changes", "SIM & Activation", "TV & Streaming"}


# ---------------------------------------------------------------- confidence
def _hit(score, steps, cat="X"):
    return Hit(id=str(id(steps)), score=score, payload={"resolution_steps": steps, "category": cat, "title": "t", "doc_type": "ticket"})


def test_agreement_groups_cases_that_prescribe_the_same_fix(settings):
    fix_a, fix_b = ["Check the ONT red light.", "Replace the patch cord."], ["Reset the admin password using the pin."]
    groups = conf.fix_groups([_hit(0.5, fix_a), _hit(0.4, fix_a), _hit(0.45, fix_b)])
    assert [len(g.members) for g in groups] == [2, 1]
    agree, _ = conf.assess([_hit(0.5, fix_a)] * 5, [], settings)
    mixed, _ = conf.assess([_hit(0.5, fix_a), _hit(0.5, fix_b), _hit(0.5, ["Offer a retention plan."])], [], settings)
    assert agree.score > mixed.score and agree.level in ("medium", "high") and mixed.level == "low"
    assert mixed.clarifying_questions == [] or "closest" in mixed.clarifying_questions[0]


def test_single_case_is_not_treated_as_proof(settings):
    one, _ = conf.assess([_hit(0.6, ["Do the thing step."])], [], settings)
    assert one.level != "high" and "Only 1" in one.reasons[0]


def test_relevance_scale_is_zero_at_the_floor_and_one_when_strong(settings):
    assert conf.relevance(settings.effective_abstain, "ticket", settings) == 0.0
    assert conf.relevance(settings.effective_strong, "ticket", settings) == 1.0
    assert conf.relevance(settings.effective_strong * 0.5, "kb", settings) == 1.0   # KB cosine runs lower, scaled


def test_wifi_not_working_is_on_topic_and_flagged_uncertain(seeded, db):
    out = resolver.resolve(seeded, db, "wifi not working", "t", use_cache=False)
    assert not {s.category for s in out.sources} & OFF_TOPIC_FOR_WIFI          # no billing/SIM/TV results
    assert out.confidence.level == "low" and out.confidence.clarifying_questions  # the cases disagree, so it asks
    assert all(0.0 <= s.relevance <= 1.0 for s in out.sources)


def test_off_topic_text_escalates_with_near_zero_confidence(seeded, db):
    out = resolver.resolve(seeded, db, "Can you recommend a good recipe for vegetarian lasagna?", "t", use_cache=False)
    assert out.resolution.escalate and out.confidence.level == "low"


def test_brief_example_is_confident_enough_and_cited(seeded, db):
    out = resolver.resolve(seeded, db, ev.BRIEF if hasattr(ev, "BRIEF") else
                           "My broadband drops every evening around 8 and I've already restarted the router twice, "
                           "I work from home and this is costing me", "t", use_cache=False)
    assert out.confidence.level in ("medium", "high") and out.resolution.steps and not out.resolution.escalate


def test_calibration_is_honest_on_the_eval_sets(seeded, db):
    c = ev.calibration(seeded, db)
    assert c["match_ece"] <= 0.12 and c["category_ece"] <= 0.10
    assert c["high_conf_precision"] >= 0.8 > 0.5 > c["low_conf_precision"] or c["high_conf_precision"] > c["low_conf_precision"] + 0.4
    assert c["monotonic"]


def test_short_queries_stay_on_topic(seeded, db):
    m = ev.short_queries(seeded, db)
    assert m["off_topic_free"] >= 0.8 and m["top1"] >= 0.8


# ----------------------------------------------------------- Hugging Face import
def _row(i, subject, body, answer, queue="Technical Support", lang="en", priority="high"):
    return {"subject": subject, "body": body, "answer": answer, "type": "Incident", "queue": queue, "priority": priority,
            "language": lang, "version": 1, "tag_1": "IT"}


PRINTER = ("Dear <name>, Thank you for contacting us. Please open Settings and remove the printer from Devices. "
           "Then reinstall the latest driver from the manufacturer site and restart the spooler service. "
           "Finally print a test page to confirm it works. Best regards, Support")
HF_ROWS = [
    _row(0, "Printer shows offline after Windows update", "Hello team, my office printer shows offline after the latest Windows update "
         "and nobody can print since this morning. We tried restarting it. Thanks in advance.", PRINTER),
    _row(1, "Printer shows offline after Windows update", "Hello team, my office printer shows offline after the latest Windows update "
         "and nobody can print since this morning. We tried restarting it. Thanks in advance.", PRINTER),          # duplicate
    _row(2, "Rechnung falsch", "Guten Tag, meine Rechnung ist falsch und ich brauche eine Korrektur der Betraege bitte.",
         "Wir pruefen die Rechnung und melden uns. Bitte senden Sie die Rechnungsnummer.", "Billing and Payments", "de"),
    _row(3, "Refund for duplicate subscription charge", "I was charged twice for the same subscription this month and need the "
         "second payment returned to my card as soon as possible.",
         "Verify both transactions in the billing system. Refund the duplicate payment to the original card. "
         "Send a confirmation email with the refund reference.", "Billing and Payments", priority="medium"),
    _row(4, "Cannot log in", "I cannot log in to the portal at all, the page just reloads every time I click sign in again.",
         "We are looking into it and will get back to you as soon as possible."),                                   # non-actionable answer
    _row(5, "Short", "help me", "Restart it. Then check again please."),                                          # body too short
    _row(6, "VPN drops every hour", "The corporate VPN disconnects every hour on my laptop and I must reconnect manually each time.",
         "Update the VPN client to the latest version. Disable power saving on the network adapter. "
         "Set the keep-alive interval to 30 seconds in the profile.", "IT Support", priority="low"),
]


def test_hf_plan_cleans_filters_and_derives_kb():
    plan = hf_import.build_plan(HF_ROWS, lang="en", limit=None, kb_limit=10)
    r = plan.report
    assert r["rows_seen"] == 7 and r["tickets"] == 3
    assert r["dropped"] == {"duplicate": 1, "other_language": 1, "no_actionable_answer": 1, "body_too_short": 1}
    by = {t.external_id: t for t in plan.tickets}
    assert set(by) == {"hf-0", "hf-3", "hf-6"}
    t0 = by["hf-0"]
    assert "<name>" not in " ".join(t0.resolution_steps) and not any("regards" in s.lower() or "thank you" in s.lower() for s in t0.resolution_steps)
    assert len(t0.resolution_steps) == 3 and t0.severity == "high"
    assert by["hf-3"].category == "Billing & Payments"                       # same class as the telecom taxonomy
    assert {a.external_id for a in plan.kb} == {"kb-hf-0", "kb-hf-3", "kb-hf-6"}
    assert "Recommended resolution:\n1. " in plan.kb[0].body


def test_hf_columns_are_matched_by_alias_and_missing_columns_explained(tmp_path):
    f = tmp_path / "t.csv"
    with f.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["Title", "Description", "Resolution", "Department"])
        w.writeheader()
        w.writerow({"Title": "Mailbox full", "Description": "My mailbox is full and I cannot receive any new messages at all today.",
                    "Resolution": "Archive old messages to the local folder. Empty the deleted items folder. Raise the quota if needed.",
                    "Department": "IT Support"})
    plan = hf_import.build_plan(hf_import.iter_rows(csv_path=str(f)), lang="en", limit=None)
    assert plan.report["tickets"] == 1 and plan.report["columns"]["queue"] == "Department"
    with pytest.raises(ValueError, match="Could not find required column"):
        hf_import.build_plan([{"foo": "bar"}])


def test_imported_hf_knowledge_is_retrievable_and_labelled(seeded, db):
    plan = hf_import.build_plan(HF_ROWS, lang="en", limit=None, kb_limit=10)
    res = hf_import.run_import(seeded, db, plan)
    assert res["tickets"]["inserted"] == 3 and res["kb"]["inserted"] == 3
    found = resolver.search(seeded, db, "office printer shows offline after windows update", 5, "all")
    assert found.results[0].origin == "hf" and any(x.doc_type == "kb" and x.origin == "hf" for x in found.results)
    out = resolver.resolve(seeded, db, "my printer keeps showing offline after the windows update", "t", use_cache=False)
    assert not out.resolution.escalate and any(s.origin == "hf" for s in out.sources)
    assert any("driver" in st.text.lower() for st in out.resolution.steps)
    # telecom behaviour is unchanged by the extra data
    tel = resolver.resolve(seeded, db, "My broadband drops every evening around 8 and I restarted the router twice", "t", use_cache=False)
    assert tel.analysis.category == "Broadband Connectivity"
    # idempotent
    again = hf_import.run_import(seeded, db, plan)
    assert again["tickets"]["inserted"] == 0 and again["kb"]["inserted"] == 0
    assert indexer.get_active(seeded, db)


# ------------------------------------------------------- Google sign-in & RBAC
WEB = '{"apiKey":"k","authDomain":"demo.firebaseapp.com","projectId":"demo-proj"}'


@pytest.fixture
def google_env(monkeypatch):
    monkeypatch.setenv("FIREBASE_WEB_CONFIG", WEB)
    monkeypatch.setenv("ADMIN_EMAILS", "Boss@Example.com, cto@example.com")
    monkeypatch.setenv("ALLOWED_EMAIL_DOMAINS", "example.com")

    claims = {
        "g-admin": {"uid": "u-boss", "email": "boss@example.com", "email_verified": True, "name": "The Boss", "firebase": {"sign_in_provider": "google.com"}},
        "g-agent": {"uid": "u-ann", "email": "ann@example.com", "email_verified": True, "firebase": {"sign_in_provider": "google.com"}},
        "g-unverified-admin": {"uid": "u-x", "email": "cto@example.com", "email_verified": False, "firebase": {"sign_in_provider": "password"}},
        "g-claim-admin": {"uid": "u-c", "email": "claim@example.com", "email_verified": True, "role": "admin"},
        "g-outsider": {"uid": "u-o", "email": "eve@gmail.com", "email_verified": True},
    }

    def fake(token, s):
        if token not in claims:
            raise ValueError("bad token")
        return claims[token]
    monkeypatch.setattr(auth, "_firebase_claims", fake)


def _h(tok):
    return {"Authorization": f"Bearer {tok}"}


def test_google_login_is_advertised_only_when_configured(client):
    cfg = client.get("/v1/config").json()
    assert cfg["google_login"] is False and cfg["firebase"] is None and cfg["dev_login"] is True


def test_google_rbac_roles(google_env, client, agent_h):
    cfg = client.get("/v1/config").json()
    assert cfg["google_login"] is True and cfg["firebase"]["projectId"] == "demo-proj" and cfg["dev_login"] is True
    assert cfg["allowed_domains"] == ["example.com"] and "secret" not in str(cfg).lower()

    boss = client.get("/v1/me", headers=_h("g-admin")).json()
    assert boss["role"] == "admin" and boss["provider"] == "google.com" and boss["name"] == "The Boss"
    assert client.get("/v1/admin/stats", headers=_h("g-admin")).status_code == 200

    ann = client.get("/v1/me", headers=_h("g-agent")).json()
    assert ann["role"] == "agent" and client.get("/v1/admin/stats", headers=_h("g-agent")).status_code == 403
    assert client.post("/v1/resolve", json={"text": "my wifi keeps dropping at night"}, headers=_h("g-agent")).status_code == 200

    assert client.get("/v1/me", headers=_h("g-unverified-admin")).json()["role"] == "agent"   # unverified email never elevates
    assert client.get("/v1/me", headers=_h("g-claim-admin")).json()["role"] == "admin"        # explicit role claim wins
    assert client.get("/v1/me", headers=_h("g-outsider")).status_code == 403                  # domain not allowed

    assert client.get("/v1/me", headers=_h("junk")).status_code == 401
    assert client.get("/v1/me", headers=agent_h).json()["provider"] == "local"                # dev login still works beside Google


def test_role_for_defaults_are_safe(settings):
    assert auth.role_for({"email": "a@b.com", "email_verified": True}, settings) == "agent"
    assert auth.role_for({"role": "root"}, settings) == "agent"       # unknown claim values are ignored


# ------------------------------------------- Firebase config: accept what the console shows, explain what is wrong
CONSOLE_SNIPPET = """// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyX",
  authDomain: "demo.firebaseapp.com",
  projectId: "demo",
  storageBucket: 'demo.appspot.com',
  appId: "1:123:web:abc",
};"""


@pytest.mark.parametrize("raw", [
    '{"apiKey":"AIzaSyX","authDomain":"demo.firebaseapp.com","projectId":"demo"}',
    CONSOLE_SNIPPET,
    "const firebaseConfig = { apiKey: 'AIzaSyX', authDomain: 'demo.firebaseapp.com', projectId: 'demo', };",
])
def test_firebase_web_config_accepts_json_and_the_console_snippet(raw):
    from app.config import Settings
    cfg, problem = Settings(firebase_web_config=raw, _env_file=None).firebase_web_result
    assert problem is None and cfg["apiKey"] == "AIzaSyX" and cfg["projectId"] == "demo"


def test_firebase_individual_variables_and_file_work_and_defaults_the_auth_domain(tmp_path):
    from app.config import Settings
    cfg, problem = Settings(firebase_api_key="AIzaSyX", firebase_project_id="demo", _env_file=None).firebase_web_result
    assert problem is None and cfg["authDomain"] == "demo.firebaseapp.com"
    f = tmp_path / "fb.js"
    f.write_text(CONSOLE_SNIPPET, encoding="utf-8")
    cfg, problem = Settings(firebase_web_config_file=str(f), _env_file=None).firebase_web_result
    assert problem is None and cfg["appId"] == "1:123:web:abc"


@pytest.mark.parametrize("raw,needle", [
    ("const firebaseConfig = {", "ONE line"),                        # a multi-line paste cut after the first line by .env
    ('{"authDomain":"d.firebaseapp.com"}', "missing apiKey"),
])
def test_firebase_problems_are_explained_not_silent(raw, needle):
    from app.config import Settings
    cfg, problem = Settings(firebase_web_config=raw, _env_file=None).firebase_web_result
    assert cfg is None and needle in problem
    assert Settings(_env_file=None).firebase_web_result == (None, None)    # not configured at all: no complaint


def test_config_endpoint_reports_why_google_is_unavailable(monkeypatch):
    monkeypatch.setenv("FIREBASE_WEB_CONFIG", "const firebaseConfig = {")
    from fastapi.testclient import TestClient

    from app.api.main import app
    from app.config import get_settings
    for k, v in {"DATABASE_URL": "sqlite:///:memory:", "QDRANT_URL": ":memory:", "AUTH_MODE": "jwt", "JWT_SECRET": "t",
                 "EMBEDDING_BACKEND": "hashing", "APP_ENV": "test"}.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    with TestClient(app) as c:
        cfg = c.get("/v1/config").json()
    assert cfg["google_login"] is False and "ONE line" in cfg["google_login_problem"]


# ------------------------------------- the same fix worded differently must still count as agreement
def test_free_text_answers_with_the_same_meaning_agree_and_different_fixes_do_not(seeded):
    m = ev.free_text_agreement(seeded)
    assert m["same_fix_agreement"] >= 0.6 and m["different_fix_agreement"] <= 0.35
    assert m["same_fix_agreement"] > 2 * m["different_fix_agreement"]
