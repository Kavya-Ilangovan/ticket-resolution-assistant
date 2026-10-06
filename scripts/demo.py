"""One-command demo: no server, no keys, no Docker.

    python -m scripts.demo                 # the brief's example, a billing case, a vague one and an off-topic one
    python -m scripts.demo "my text here"  # your own complaint

Builds an in-memory stack (SQLite + embedded Qdrant + offline embedder), seeds the demo data and prints what an agent
would see: parsed fields, the sources used, the cited resolution steps and, when the system is unsure, an escalation.
"""
from __future__ import annotations

import os
import sys
import textwrap

os.environ.setdefault("EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("QDRANT_URL", ":memory:")
os.environ.setdefault("OPENROUTER_API_KEY", "")   # empty => extractive (still cited and grounded) drafting

from app.config import get_settings  # noqa: E402
from app.core.context import build_context, set_context  # noqa: E402
from app.db.session import init_engine, session_scope  # noqa: E402
from app.services import resolver  # noqa: E402
from scripts.seed import seed  # noqa: E402

BRIEF = ("My broadband drops every evening around 8 and I've already restarted the router twice, "
         "I work from home and this is costing me")
SAMPLES = [
    ("The example from the brief", BRIEF),
    ("A billing problem in different words", "Money left my account two times for the same invoice, please sort this out"),
    ("A vague complaint: on-topic results, low confidence, asks for detail", "wifi not working"),
    ("Off-topic: must escalate, not invent an answer", "Can you recommend a good recipe for vegetarian lasagna?"),
]


def show(title: str, text: str, ctx, db) -> None:
    out = resolver.resolve(ctx, db, text, "demo", use_cache=False)
    a, r = out.analysis, out.resolution
    wrap = lambda s, ind="    ": textwrap.fill(s, 100, initial_indent=ind, subsequent_indent=ind)  # noqa: E731
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}\nCOMPLAINT:\n{wrap(text, '  ')}")
    print(f"\nPARSED   category={a.category} (conf {a.category_confidence})  intent={a.intent}  product={a.product}\n"
          f"         severity={a.severity} {a.severity_signals}  sentiment={a.sentiment} ({a.sentiment_score})")
    c = out.confidence
    print(f"\nCONFIDENCE  {c.level.upper()} ({c.score:.0%})   closest cases agree on the fix: {c.agreement:.0%}   best match strength: {c.relevance:.0%}")
    for reason in c.reasons:
        print(wrap("- " + reason, "    "))
    for q in c.clarifying_questions:
        print(wrap("? " + q, "    "))
    print("\nSOURCES")
    for s in out.sources[:6]:
        tag = " [HF]" if s.origin == "hf" else ""
        print(f"  [{s.id}] {s.doc_type:6} match={s.relevance:4.0%}{tag}  {s.title or s.snippet[:70]}")
    if r.escalate:
        print(f"\nESCALATE -> {r.escalation_reason}")
        return
    print(f"\nRESOLUTION ({r.generation_mode}, grounding {r.grounding_score}){'  [priority]' if r.priority_flag else ''}")
    print(wrap(r.summary, "  "))
    for st in r.steps:
        print(wrap(f"{st.n}. {st.text}  [{', '.join(st.citations)}]", "  "))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1252
    settings = get_settings()
    init_engine(settings)
    ctx = build_context(settings)
    set_context(ctx)
    custom = " ".join(sys.argv[1:]).strip()
    with session_scope() as db:
        seed(ctx, db)
        for title, text in ([("Your complaint", custom)] if custom else SAMPLES):
            show(title, text, ctx, db)


if __name__ == "__main__":
    main()
