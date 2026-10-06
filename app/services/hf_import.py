"""Import the public Hugging Face support-ticket dataset into the knowledge base.

The dataset (default ``Tobi-Bueck/customer-support-tickets``) has one row per resolved ticket: subject, body, the
agent's ``answer``, ``queue`` (department), ``priority``, ``language`` and tags. It is a general customer-support corpus
(not telecom-specific), so it is added *alongside* the telecom data and tagged with ``origin = "hf"``.

Two things are produced from it:

1. **Resolved tickets** (``hf-<row>``): the agent answer is cleaned and split into resolution steps, so the RAG layer
   can cite them exactly like telecom tickets.
2. **KB articles** (``kb-hf-<row>``): the best-documented cases (several concrete steps, not just "we are looking into
   it") are turned into short how-to articles, balanced across queues, near-duplicate titles removed.

Everything here is pure and offline except :func:`iter_rows` (network or a local file), so it is unit-testable.
Column names are matched case-insensitively against aliases, because dataset versions differ.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from app.core.text import content_hash, is_pleasantry, jaccard, strip_boilerplate, strip_salutation
from app.schemas import KBArticleIn, TicketIn, split_steps

REPO = "Tobi-Bueck/customer-support-tickets"
ALIASES = {
    "subject": ("subject", "title", "ticket_subject", "summary"),
    "body": ("body", "description", "text", "ticket", "message", "content", "complaint"),
    "answer": ("answer", "resolution", "response", "solution", "agent_response", "reply"),
    "queue": ("queue", "department", "category", "team"),
    "priority": ("priority", "severity", "urgency"),
    "language": ("language", "lang"),
}
_PLACEHOLDER = re.compile(r"<[A-Za-z_ ]{2,30}>|\[[A-Za-z_ ]{2,30}\]|\{\{[^}]{1,30}\}\}")
_NON_ACTIONABLE = re.compile(
    r"\b(?:will get back to you|looking into (?:it|this|the issue)|we are investigating|investigating (?:it|this)|"
    r"reach out (?:to us )?again|contact us again|as soon as possible|in touch (?:with you )?shortly)\b", re.I)
_PRIORITY = {"low": "low", "medium": "medium", "normal": "medium", "high": "high", "critical": "critical", "urgent": "high"}


@dataclass
class Plan:
    tickets: list[TicketIn] = field(default_factory=list)
    kb: list[KBArticleIn] = field(default_factory=list)
    report: dict = field(default_factory=dict)


# ----------------------------------------------------------------- reading
def iter_rows(repo: str = REPO, csv_path: str | None = None) -> Iterator[dict]:
    """Yield raw rows from a local .csv/.jsonl/.json file, or from the Hugging Face hub (needs `datasets` + network)."""
    if csv_path:
        path = Path(csv_path)
        if path.suffix.lower() == ".jsonl":
            yield from (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        elif path.suffix.lower() == ".json":
            yield from json.loads(path.read_text(encoding="utf-8"))
        else:
            with path.open(newline="", encoding="utf-8-sig") as f:
                yield from csv.DictReader(f)
        return
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError("Install the loader dependency first:  pip install datasets   (or pass --csv path\\to\\file.csv)") from e
    yield from load_dataset(repo, split="train")


def _column_map(sample: dict) -> dict[str, str | None]:
    lower = {k.lower().strip(): k for k in sample}
    return {field_: next((lower[a] for a in aliases if a in lower), None) for field_, aliases in ALIASES.items()}


# ---------------------------------------------------------------- cleaning
def clean_text(text: object) -> str:
    t = _PLACEHOLDER.sub("", str(text or ""))
    return " ".join(t.split())


def answer_steps(answer: object, max_steps: int = 10) -> list[str]:
    """Turn a free-text agent reply into concrete steps: drop greetings, apologies, sign-offs and promises to follow up."""
    raw = str(answer or "")
    text = _PLACEHOLDER.sub("", raw)
    lines = [strip_salutation(ln) for ln in split_steps(text)]
    steps = []
    for ln in lines:
        ln = ln.strip()
        if len(ln.split()) < 4 or is_pleasantry(ln) or (_NON_ACTIONABLE.search(ln) and len(ln.split()) <= 18):
            continue
        steps.append(ln)
    return steps[:max_steps]


def _queue_name(raw: object) -> str | None:
    q = clean_text(raw)
    if not q:
        return None
    return q.replace(" and ", " & ").strip()[:80]   # "Billing and Payments" -> same class as the telecom "Billing & Payments"


# ------------------------------------------------------------------- plan
def build_plan(rows: Iterable[dict], *, lang: str | None = "en", limit: int | None = 3000, kb_limit: int = 300,
               per_queue: int | None = None) -> Plan:
    rows = iter(rows)
    first = next(rows, None)
    drops: Counter = Counter()
    if first is None:
        return Plan(report={"rows_seen": 0, "note": "dataset is empty"})
    cols = _column_map(first)
    missing = [k for k in ("body", "answer") if not cols[k]]
    if missing:
        raise ValueError(f"Could not find required column(s) {missing}. Columns present: {sorted(first)}. "
                         f"Expected one of {ALIASES}")
    get = lambda r, k: r.get(cols[k]) if cols[k] else None  # noqa: E731

    seen, per_q, cands, tickets = set(), Counter(), [], []
    seen_rows = 0
    for i, r in enumerate([first, *rows]):
        seen_rows += 1
        if limit and len(tickets) >= limit:
            break
        row_lang = str(get(r, "language") or "").strip().lower()
        if lang and row_lang and row_lang != lang.lower():
            drops["other_language"] += 1
            continue
        body, subject = clean_text(get(r, "body")), clean_text(get(r, "subject"))
        steps = answer_steps(get(r, "answer"))
        if len(body.split()) < 6:
            drops["body_too_short"] += 1
            continue
        if not steps:
            drops["no_actionable_answer"] += 1
            continue
        key = content_hash(subject.lower(), body.lower())
        if key in seen:
            drops["duplicate"] += 1
            continue
        queue = _queue_name(get(r, "queue"))
        if per_queue and queue and per_q[queue] >= per_queue:
            drops["queue_cap"] += 1
            continue
        seen.add(key)
        per_q[queue] += 1
        sev = _PRIORITY.get(str(get(r, "priority") or "").strip().lower())
        tickets.append(TicketIn(external_id=f"hf-{i}", subject=subject[:500], body=body[:9000], category=queue, severity=sev,
                                resolution_steps=steps, status="resolved"))
        cands.append((i, queue, subject, body, steps))

    kb = _pick_kb(cands, kb_limit)
    report = {"rows_seen": seen_rows, "tickets": len(tickets), "kb_articles": len(kb), "dropped": dict(drops),
              "by_queue": dict(per_q.most_common()), "columns": {k: v for k, v in cols.items()}}
    return Plan(tickets=tickets, kb=kb, report=report)


def _title(subject: str, body: str) -> str:
    t = strip_boilerplate(subject) if subject else ""
    if len(t.split()) < 2:
        t = strip_boilerplate(body)
    return (t[:90].rsplit(" ", 1)[0] if len(t) > 90 else t).strip(" .,:;-")


def _pick_kb(cands: list[tuple], kb_limit: int) -> list[KBArticleIn]:
    """Best-documented cases become KB articles: >= 2 concrete steps, reasonable length, balanced across queues."""
    if not kb_limit:
        return []
    by_q: dict[str | None, list[tuple]] = defaultdict(list)
    for i, queue, subject, body, steps in cands:
        words = sum(len(s.split()) for s in steps)
        if len(steps) < 2 or not 20 <= words <= 260:
            continue
        quality = min(len(steps), 6) + (1.5 if 40 <= words <= 160 else 0) + (1.0 if subject else 0)
        by_q[queue].append((quality, i, queue, subject, body, steps))
    for lst in by_q.values():
        lst.sort(key=lambda x: (-x[0], x[1]))
    chosen: list[KBArticleIn] = []
    titles: list[str] = []
    queues = sorted(by_q, key=lambda q: str(q))
    cursor = dict.fromkeys(queues, 0)
    while len(chosen) < kb_limit and queues:
        progressed = False
        for q in list(queues):
            lst = by_q[q]
            while cursor[q] < len(lst):
                _, i, queue, subject, body, steps = lst[cursor[q]]
                cursor[q] += 1
                title = _title(subject, body)
                if len(title.split()) < 2 or any(jaccard(title, t) >= 0.7 for t in titles):
                    continue
                titles.append(title)
                numbered = "\n".join(f"{n}. {s}" for n, s in enumerate(steps, 1))
                chosen.append(KBArticleIn(external_id=f"kb-hf-{i}", title=title[:300], category=queue,
                                          body=f"Typical problem: {strip_boilerplate(body)[:400]}\n\nRecommended resolution:\n{numbered}"))
                progressed = True
                break
            if cursor[q] >= len(lst):
                queues.remove(q)
            if len(chosen) >= kb_limit:
                break
        if not progressed and not queues:
            break
    return chosen


# --------------------------------------------------------------- ingestion
def run_import(ctx, db, plan: Plan, job=None) -> dict:
    from app.services import ingestion  # local import: keeps this module importable without the DB stack

    out = {"tickets": ingestion.ingest_tickets(ctx, db, plan.tickets, job=job, source="hf"),
           "kb": ingestion.ingest_kb(ctx, db, plan.kb, job=job), "report": plan.report}
    return out
