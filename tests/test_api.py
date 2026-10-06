from app.core.audit import verify_chain
from app.db.models import AuditLog
from app.db.session import session_scope

Q = "My broadband drops every evening around 8 and I've restarted the router twice, I work from home"


def test_health_ready_metrics(client):
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["checks"]["vector_db"] == "ok"
    assert "http_requests_total" in client.get("/metrics").text


def test_auth_and_rbac(seeded_client, agent_h):
    c = seeded_client
    assert c.post("/v1/resolve", json={"text": Q}).status_code == 401
    assert c.post("/v1/resolve", json={"text": Q}, headers={"Authorization": "Bearer junk"}).status_code == 401
    assert c.post("/v1/resolve", json={"text": Q}, headers=agent_h).status_code == 200
    assert c.post("/v1/ingest/tickets", json={"tickets": []}, headers=agent_h).status_code == 403
    assert c.get("/v1/admin/drift", headers=agent_h).status_code == 403


def test_resolve_contract_and_feedback_loop(seeded_client, agent_h, admin_h):
    c = seeded_client
    r = c.post("/v1/resolve", json={"text": Q}, headers=agent_h)
    body = r.json()
    assert r.headers["x-request-id"] and body["analysis"]["category"] == "Broadband Connectivity"
    assert body["resolution"]["steps"][0]["citations"]
    steps = ["Replaced the ONT patch cord and re-seated the connector.", "Confirmed stable line for 48 hours."]
    fb = c.post("/v1/feedback", json={"query_id": body["query_id"], "helpful": True, "rating": 5, "resolved_steps": steps}, headers=agent_h)
    assert fb.status_code == 200 and fb.json()["promoted_ticket"]
    assert c.post("/v1/feedback", json={"query_id": "nope", "helpful": False}, headers=agent_h).status_code == 404
    assert c.get("/v1/admin/stats", headers=admin_h).json()["vectors_ticket"] >= 201


def test_search_filters_and_analyze(seeded_client, agent_h):
    c = seeded_client
    r = c.post("/v1/search", json={"text": "double charged on my bill", "doc_type": "kb", "top_k": 3}, headers=agent_h).json()
    assert r["results"] and all(x["doc_type"] == "kb" for x in r["results"])
    a = c.post("/v1/analyze", json={"text": "I was charged twice, this is unacceptable!"}, headers=agent_h).json()
    assert (a["intent"], a["sentiment"]) == ("billing_dispute", "negative")


def test_category_add_and_merge_via_api(seeded_client, admin_h, agent_h):
    c = seeded_client
    seeds = [{"body": f"my smart thermostat won't connect to wifi variant {i}", "resolution_steps": ["Re-pair the thermostat."]} for i in range(3)]
    assert c.post("/v1/admin/categories", json={"name": "Smart Home", "seed_examples": seeds}, headers=admin_h).status_code == 201
    assert "Smart Home" in {x["name"] for x in c.get("/v1/categories", headers=agent_h).json()}
    c.post("/v1/admin/categories", json={"name": "IoT"}, headers=admin_h)
    assert c.post("/v1/admin/categories/Smart Home/merge", json={"into": "IoT"}, headers=admin_h).json()["tickets_moved"] == 3


def test_reindex_job_and_audit_chain(seeded_client, admin_h):
    c = seeded_client
    j = c.post("/v1/admin/reindex", json={}, headers=admin_h).json()
    assert c.get(f"/v1/jobs/{j['job_id']}", headers=admin_h).json()["status"] == "done"
    assert c.get("/ready").json()["checks"]["index"].endswith("_v2")
    v = c.get("/v1/admin/audit/verify", headers=admin_h).json()
    assert v["valid"] and v["entries"] >= 1
    with session_scope() as db:                                    # tamper with history -> chain breaks
        row = db.query(AuditLog).first()
        row.payload = {"forged": True}
        db.commit()
        assert verify_chain(db)["valid"] is False


def test_rate_limit(seeded_client, agent_h):
    from app.core.context import get_context
    get_context().settings.rate_limit_per_minute = 3
    codes = [seeded_client.post("/v1/search", json={"text": Q}, headers=agent_h).status_code for _ in range(5)]
    assert codes[:3] == [200] * 3 and codes[3] == 429


def test_frontend_served_and_config(client):
    assert "Support Resolution Assistant" in client.get("/ui/").text
    assert all(client.get(f"/ui/{f}").status_code == 200 for f in ("app.js", "styles.css"))
    assert client.get("/", follow_redirects=False).headers["location"] == "/ui/"
    cfg = client.get("/v1/config").json()
    assert cfg["auth_mode"] == "jwt" and cfg["dev_login"] and "jwt_secret" not in str(cfg)


def test_seed_demo_and_jobs_list(client, admin_h, agent_h):
    assert client.post("/v1/admin/seed-demo", headers=agent_h).status_code == 403
    j = client.post("/v1/admin/seed-demo", headers=admin_h).json()
    assert client.get(f"/v1/jobs/{j['job_id']}", headers=admin_h).json()["status"] == "done"
    assert client.get("/v1/admin/stats", headers=admin_h).json()["tickets_db"] == 528
    assert client.get("/v1/admin/jobs", headers=admin_h).json()[0]["kind"] == "seed_demo"
