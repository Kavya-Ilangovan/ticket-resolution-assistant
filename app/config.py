"""Central configuration (12-factor: everything comes from environment variables / .env)."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_firebase_web_config(text: str) -> dict:
    """Accept what the Firebase console actually shows, not just strict JSON:
    `{"apiKey": "..."}`, `const firebaseConfig = { apiKey: "...", authDomain: '...', };` (unquoted keys, single quotes,
    trailing commas, comments)."""
    m = re.search(r"\{.*\}", text.strip(), re.S)
    if not m:
        raise ValueError("no { ... } object found")
    body = m.group(0)
    try:
        return json.loads(body)
    except ValueError:
        pass
    b = re.sub(r"//[^\n]*", "", body)
    b = re.sub(r"([{,]\s*)([A-Za-z_$][\w$]*)\s*:", r'\1"\2":', b)     # unquoted keys
    b = re.sub(r"'([^'\\]*)'", r'"\1"', b)                             # single quotes
    b = re.sub(r",\s*([}\]])", r"\1", b)                              # trailing commas
    try:
        return json.loads(b)
    except ValueError as e:
        raise ValueError(str(e)) from e


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # --- Stores -----------------------------------------------------------
    database_url: str = "sqlite:///./local.db"          # prod: postgresql+psycopg2://...
    redis_url: str | None = None                          # prod: redis://redis:6379/0
    qdrant_url: str = ":memory:"                          # prod: http://qdrant:6333
    qdrant_api_key: str | None = None
    collection_prefix: str = "support"

    # --- Embeddings -------------------------------------------------------
    # "sentence-transformers": real semantic model (needs requirements-ml.txt)
    # "hashing": dependency-free lexical fallback for CI / offline dev
    embedding_backend: Literal["sentence-transformers", "hashing"] = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hashing_dim: int = 1024

    # --- LLM (OpenRouter) -------------------------------------------------
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-4o-mini"
    llm_fallback_model: str | None = "anthropic/claude-3.5-haiku"
    llm_timeout_s: float = 25.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 900
    llm_analysis: bool = False      # also use the LLM for complaint parsing (extra cost/latency)

    # --- Domain -----------------------------------------------------------
    domain_profile: str = "telecom"   # name in app/domains/ or a path to a profile JSON (products, outage cues)

    # --- Retrieval / RAG --------------------------------------------------
    top_k_tickets: int = 5
    top_k_kb: int = 3
    knn_k: int = 10
    reranker: Literal["none", "cross-encoder"] = "none"   # second-stage reordering of ticket candidates
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid_search: bool = True   # dense + sparse(BM25-style) fused with RRF
    # similarity thresholds are embedder specific; None picks a default per backend
    abstain_threshold: float | None = None   # below this top-1 score, escalate instead of answering
    novelty_threshold: float | None = None   # below this, category is "unknown" (possible new class)
    redact_pii: bool = True
    # confidence calibration (services/confidence.py); re-fit per embedder: python -m evals.run_all --calibrate
    strong_similarity: float | None = None   # similarity that counts as a strong match
    conf_slope: float = 5.26
    conf_bias: float = -3.13
    cat_conf_slope: float = 10.94            # raw kNN score -> P(category correct)
    cat_conf_bias: float = -4.21
    min_confidence: float = 0.10             # below this, escalate to Tier-2
    fix_link_threshold: float | None = None  # step-text cosine at which two cases are "the same fix"
    conf_high: float = 0.70                  # >= high   => "high" confidence
    conf_medium: float = 0.40                # >= medium => "medium", otherwise "low"

    # --- Auth / limits ----------------------------------------------------
    auth_mode: Literal["off", "jwt", "firebase"] = "off"
    jwt_secret: str = "change-me-in-prod"
    firebase_project_id: str | None = None
    firebase_credentials_file: str | None = None
    firebase_web_config: str | None = None   # web app config (JSON or JS snippet) for the browser UI
    firebase_web_config_file: str | None = None
    firebase_api_key: str | None = None      # or the individual values
    firebase_auth_domain: str | None = None  # defaults to <project id>.firebaseapp.com
    firebase_app_id: str | None = None
    admin_emails: str = ""                   # comma-separated; verified emails listed here become admin
    allowed_email_domains: str = ""          # comma-separated; empty = any domain
    default_role: Literal["agent", "admin"] = "agent"
    rate_limit_per_minute: int = 60
    cache_ttl_s: int = 600

    # --- Celery -----------------------------------------------------------
    celery_eager: bool = True   # run tasks inline (docker-compose sets false)

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def effective_abstain(self) -> float:
        if self.abstain_threshold is not None:
            return self.abstain_threshold
        return 0.30 if self.embedding_backend == "sentence-transformers" else 0.18

    @property
    def admin_email_list(self) -> list[str]:
        return [e.strip().lower() for e in self.admin_emails.split(",") if e.strip()]

    @property
    def allowed_domain_list(self) -> list[str]:
        return [d.strip().lower().lstrip("@") for d in self.allowed_email_domains.split(",") if d.strip()]

    @property
    def firebase_web_result(self) -> tuple[dict | None, str | None]:
        """(web config, problem). (None, None) = not configured at all; (None, "...") = configured but unusable."""
        raw = None
        if self.firebase_web_config_file:
            try:
                raw = Path(self.firebase_web_config_file).read_text(encoding="utf-8")
            except OSError as e:
                return None, f"FIREBASE_WEB_CONFIG_FILE cannot be read ({e.strerror or e})."
        elif self.firebase_web_config and self.firebase_web_config.strip():
            raw = self.firebase_web_config
        cfg: dict = {}
        if raw:
            try:
                cfg = parse_firebase_web_config(raw)
            except ValueError:
                return None, ("FIREBASE_WEB_CONFIG is set but could not be read as a config object. A .env value must be ONE line "
                              "(a multi-line paste is cut after the first line). Use FIREBASE_API_KEY + FIREBASE_PROJECT_ID "
                              "instead, or save the snippet to a file and set FIREBASE_WEB_CONFIG_FILE.")
        for key, val in (("apiKey", self.firebase_api_key), ("authDomain", self.firebase_auth_domain),
                         ("projectId", self.firebase_project_id), ("appId", self.firebase_app_id)):
            if val:
                cfg[key] = val
        if not cfg:
            return None, None
        if not cfg.get("authDomain") and cfg.get("projectId"):
            cfg["authDomain"] = f"{cfg['projectId']}.firebaseapp.com"
        missing = [k for k in ("apiKey", "projectId") if not cfg.get(k)]
        if missing:
            return None, f"Firebase config is missing {', '.join(missing)} (set FIREBASE_API_KEY and FIREBASE_PROJECT_ID)."
        return cfg, None

    @property
    def effective_firebase_project(self) -> str | None:
        """Project id for token verification: explicit setting, else the one inside the web config."""
        if self.firebase_project_id:
            return self.firebase_project_id
        cfg, _ = self.firebase_web_result
        return (cfg or {}).get("projectId")

    @property
    def firebase_enabled(self) -> bool:
        return bool(self.effective_firebase_project or self.firebase_credentials_file)

    @property
    def effective_fix_link(self) -> float:
        if self.fix_link_threshold is not None:
            return self.fix_link_threshold
        return 0.55 if self.embedding_backend == "sentence-transformers" else 0.30   # ST default is provisional: --calibrate

    @property
    def effective_strong(self) -> float:
        if self.strong_similarity is not None:
            return self.strong_similarity
        return 0.65 if self.embedding_backend == "sentence-transformers" else 0.45

    @property
    def effective_novelty(self) -> float:
        if self.novelty_threshold is not None:
            return self.novelty_threshold
        return 0.42 if self.embedding_backend == "sentence-transformers" else 0.28


@lru_cache
def get_settings() -> Settings:
    return Settings()
