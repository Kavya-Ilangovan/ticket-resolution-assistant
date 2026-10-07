"""Pydantic request/response contracts (the public API surface)."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Severity = Literal["low", "medium", "high", "critical"]
Sentiment = Literal["negative", "neutral", "positive"]
Intent = Literal["fault_report", "billing_dispute", "service_request", "inquiry", "churn_risk"]


def split_steps(value: str | list[str] | None) -> list[str]:
    """Accept either a list of steps or free text ("1. do x\n2. do y") and normalise to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [s.strip() for s in value if s and s.strip()]
    lines = [ln.strip() for ln in re.split(r"\n+", value) if ln.strip()]
    if len(lines) <= 1:  # single blob -> split into sentences
        lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", value) if s.strip()]
    return [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", ln) for ln in lines]


# ----------------------------------------------------------------- ingestion
class TicketIn(BaseModel):
    external_id: str | None = Field(None, max_length=128, description="Idempotency key; hash is used if omitted")
    subject: str = Field("", max_length=500)
    body: str = Field(..., min_length=5, max_length=10_000)
    category: str | None = None
    product: str | None = None
    severity: Severity | None = None
    resolution_steps: list[str] = Field(default_factory=list)
    status: Literal["open", "resolved"] = "resolved"
    issue_key: str | None = Field(None, description="Optional canonical problem id (used by evals)")

    @field_validator("resolution_steps", mode="before")
    @classmethod
    def _steps(cls, v):  # noqa: N805
        return split_steps(v)


class KBArticleIn(BaseModel):
    external_id: str = Field(..., max_length=128)
    title: str = Field(..., max_length=300)
    body: str = Field(..., min_length=10, max_length=50_000)
    category: str | None = None
    product: str | None = None
    version: int = 1
    active: bool = True


class TicketBatch(BaseModel):
    tickets: list[TicketIn] = Field(..., max_length=5000)


class KBBatch(BaseModel):
    articles: list[KBArticleIn] = Field(..., max_length=2000)


class JobOut(BaseModel):
    job_id: str
    kind: str
    status: str
    total: int = 0
    processed: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    error: str | None = None


# ---------------------------------------------------------------- analysis
class Analysis(BaseModel):
    category: str
    category_confidence: float
    is_known_category: bool
    intent: Intent
    product: str | None
    product_confidence: float = 0.0
    severity: Severity
    severity_signals: list[str] = Field(default_factory=list)
    sentiment: Sentiment
    sentiment_score: float
    method: str


class ComplaintIn(BaseModel):
    text: str = Field(..., min_length=5, max_length=8000, description="Raw customer complaint")


class ResolveIn(ComplaintIn):
    top_k_tickets: int | None = Field(None, ge=1, le=15)
    top_k_kb: int | None = Field(None, ge=0, le=10)
    use_cache: bool = True


class SearchIn(ComplaintIn):
    top_k: int = Field(8, ge=1, le=30)
    doc_type: Literal["ticket", "kb", "all"] = "all"
    category: str | None = None
    product: str | None = None


class Source(BaseModel):
    id: str = Field(..., description="Citation handle used in steps, e.g. T1 / K2")
    doc_type: Literal["ticket", "kb"]
    external_id: str
    title: str
    score: float = Field(..., description="Raw similarity (cosine); not comparable across embedders - use `relevance`")
    relevance: float = Field(0.0, description="0-1 match strength, calibrated to the embedder (0 = at the abstain floor)")
    origin: Literal["seed", "hf"] = Field("seed", description="Where the document came from (hf = Hugging Face import)")
    category: str | None = None
    product: str | None = None
    snippet: str = ""


class Step(BaseModel):
    n: int
    text: str
    citations: list[str]


class Confidence(BaseModel):
    score: float = Field(..., description="0-1 estimated probability that the dominant fix is the right one")
    level: Literal["high", "medium", "low"]
    agreement: float = Field(..., description="Share of the (similarity-weighted) closest cases that agree on the same fix")
    similarity: float = Field(..., description="Best raw similarity")
    relevance: float = Field(..., description="Best match strength, 0-1")
    reasons: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list, description="Asked when the closest cases disagree")
    source_ids: list[str] = Field(default_factory=list, description="Sources backing the dominant fix")


class Resolution(BaseModel):
    summary: str
    steps: list[Step]
    escalate: bool = Field(..., description="True => no reliable grounded answer; route to Tier-2")
    escalation_reason: str | None = None
    priority_flag: bool = Field(False, description="High/critical severity: handle with priority")
    grounding_score: float = Field(0.0, description="Mean lexical support of steps by cited sources (0-1)")
    generation_mode: Literal["llm", "extractive", "none"]


class ResolveOut(BaseModel):
    request_id: str
    query_id: str
    analysis: Analysis
    sources: list[Source]
    resolution: Resolution
    confidence: Confidence
    index_version: str
    cached: bool = False
    latency_ms: float


class SearchOut(BaseModel):
    results: list[Source]
    index_version: str


# --------------------------------------------------------------- admin/other
class CategoryIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    description: str = ""
    seed_examples: list[TicketIn] = Field(default_factory=list, max_length=500)


class CategoryOut(BaseModel):
    name: str
    description: str
    active: bool
    merged_into: str | None = None
    tickets: int = 0


class MergeIn(BaseModel):
    into: str


class FeedbackIn(BaseModel):
    query_id: str
    helpful: bool
    rating: int | None = Field(None, ge=1, le=5)
    correct_category: str | None = None
    comment: str | None = Field(None, max_length=2000)
    resolved_steps: list[str] | None = Field(
        None, description="If the agent resolved it differently, steps are promoted to a new ticket"
    )


class ReindexIn(BaseModel):
    embedding_backend: Literal["sentence-transformers", "hashing"] | None = None
    embedding_model: str | None = None


class Me(BaseModel):
    uid: str
    role: str
    email: str | None = None
    name: str | None = None
    provider: str = "local"


class Health(BaseModel):
    status: str
    checks: dict[str, Any] = Field(default_factory=dict)
    ts: datetime | None = None
