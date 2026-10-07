# Architecture

## 1. System view

```mermaid
flowchart LR
  subgraph Clients
    A[Support agent UI / CRM plug-in]
    AD[Admin / ops console]
  end

  A -- "Firebase ID token" --> UI
  UI["Web UI (static SPA served at /ui, / redirects)<br/>Resolve · Search · Admin"] --> GW
  AD -- "Firebase ID token (role=admin)" --> GW

  subgraph API["api  (FastAPI, stateless, N replicas)"]
    GW[AuthN/Z + rate limit + request-id]
    GW --> R1["/v1/resolve  /v1/analyze  /v1/search  /v1/feedback"]
    GW --> R2["/v1/ingest/*  /v1/admin/*  /v1/categories"]
    R1 --> PIPE
    subgraph PIPE["Resolution pipeline"]
      P0[PII redaction] --> P1[Embed query]
      P1 --> P2["Hybrid retrieval<br/>dense + sparse, RRF<br/>optional cross-encoder"]
      P2 --> P3["Complaint parser<br/>category kNN · product · severity · sentiment · intent"]
      P3 --> P4[Category-aware re-rank]
      P4 --> P5{"top score ≥ threshold?"}
      P5 -- no --> ESC[Escalate to Tier-2 · no hallucination]
      P5 -- yes --> P6["RAG generator<br/>LLM JSON w/ citations<br/>→ citation + grounding validation"]
      P6 -- "LLM down / ungrounded" --> P7[Extractive composer from historical resolution steps]
    end
  end

  PIPE <--> REDIS[(Redis<br/>cache · rate limit · broker)]
  PIPE <--> QD[(Qdrant<br/>dense + sparse vectors<br/>blue/green collections)]
  PIPE <--> PG[(PostgreSQL<br/>tickets · KB · taxonomy · jobs<br/>query log · feedback · audit chain)]
  PIPE -- "redacted prompt" --> OR[OpenRouter LLM gateway<br/>primary + fallback model<br/>circuit breaker]

  R2 -- enqueue --> REDIS
  REDIS --> W["worker (Celery)<br/>ingest · embed · re-index · merge · reconcile"]
  BEAT[beat] -- every 5 min --> REDIS
  W <--> PG
  W <--> QD

  API -- /metrics --> PROM[Prometheus + alert rules]
```

**Source of truth vs. derived index.** PostgreSQL holds everything that matters; Qdrant is a *derived* index that can be rebuilt
at any time (`/v1/admin/reindex`). That is what makes model upgrades, class merges and recovery safe.

## 2. `/v1/resolve` sequence

```mermaid
sequenceDiagram
  participant Ag as Agent
  participant API as FastAPI
  participant KV as Redis
  participant VS as Qdrant
  participant DB as Postgres
  participant LLM as OpenRouter
  Ag->>API: POST /v1/resolve {text}
  API->>API: auth, rate limit (Redis), redact PII
  API->>KV: cache lookup (index version + normalised text)
  alt hit
    KV-->>API: cached response
  else miss
    API->>VS: hybrid search tickets (k=10) + KB (k=3)
    API->>DB: active categories
    API->>API: parse: kNN category, product, severity, sentiment
    API->>API: re-rank, abstention check
    API->>LLM: numbered sources + complaint as untrusted data
    LLM-->>API: JSON steps with citations
    API->>API: drop unknown/ungrounded citations (else extractive fallback)
    API->>DB: query log (redacted)
    API->>KV: cache (not for escalations)
  end
  API-->>Ag: analysis, sources[T1..,K1..], steps[+citations], escalate?, grounding score
```

## 2b. Trust layer, Hugging Face data and sign-in
* **Retrieval vectors** ignore generic complaint words and strip greetings/sign-offs (`core/text.py`), so a short query such as "wifi not working" is matched on *wifi*, not on "working".
* **Confidence** (`services/confidence.py`): tickets whose resolution steps overlap form fix groups; `agreement` = weight of the leading group; `confidence = sigmoid(slope*agreement + bias) x similarity_gate`. Low confidence => clarifying question built from the competing groups; below `MIN_CONFIDENCE` => escalate. Results below the abstain floor or far weaker than the best are not shown. Metrics: `resolve_match_confidence`, `resolve_low_confidence_total` (alerts in `monitoring/alerts.yml`).
* **Domain profile** (`app/domains/*.json`): products, outage phrases and dataset queues are data, not code.
* **Hugging Face import** (`services/hf_import.py`, `data/load_hf.py`, `POST /v1/admin/import-hf`): rows -> cleaned resolved tickets (`hf-*`) and derived KB articles (`kb-hf-*`), indexed through the same idempotent ingestion; `origin` is derived from the id prefix, so no schema change.
* **Auth** (`core/auth.py`): Firebase ID tokens (Google or password) are verified server-side; role = `role` claim, else verified email in `ADMIN_EMAILS` -> admin, else `agent`; `ALLOWED_EMAIL_DOMAINS` gates entry. In `jwt` mode Firebase tokens are accepted beside dev tokens.

## 3. Evolving data and classes

| Change in the world | Mechanism | Retraining? |
|---|---|---|
| New resolved tickets / edited tickets | Idempotent upsert by `external_id` + content hash; only changed rows are re-embedded (`/v1/ingest/tickets`, feedback promotion) | no |
| KB article edited / retired | Chunks replaced atomically; `active=false` removes vectors | no |
| **New ticket class** | `POST /v1/admin/categories` with a few seed tickets → kNN classifier predicts it as soon as they are indexed. Unseen-class complaints are logged as `unknown` and clustered at `/v1/admin/emerging` to suggest *which* class to create | **no** |
| Classes merged / renamed | `POST /v1/admin/categories/{n}/merge` rewrites DB + vector payloads; stale producers are redirected | no |
| New embedding model | Blue/green re-index into `collection_vN+1` from Postgres, catch-up pass, atomic pointer flip; serving embeds queries with the *active index's* embedder; thresholds re-calibrated (`evals.run_all --calibrate`) | re-embed only |
| Vector DB was down during ingest | `beat` runs `reconcile_unindexed` every 5 min | no |
| Distribution shift | `/v1/admin/drift` + Prometheus alerts on unknown-rate, top-1 similarity, escalation rate, helpful ratio | – |

## 4. Data model (PostgreSQL)

`tickets`, `kb_articles`, `categories` (taxonomy, merge pointers), `index_state` (active collection + embedder spec),
`ingest_jobs`, `query_logs` (redacted), `feedback`, `audit_log` (hash-chained, tamper-evident).


