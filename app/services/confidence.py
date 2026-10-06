"""How much should an agent trust this answer?

Similarity alone is a poor trust signal: on our adversarial split the top-1 similarity of *wrong* matches was higher than
that of right ones. What does predict a correct answer is **agreement**: do the closest resolved cases prescribe the
same fix? When they do (agreement >= 0.8) the dominant fix was right 89% of the time; when they don't (< 0.4) only 17%.

    confidence = sigmoid(slope * agreement + bias)  x  similarity_gate

* agreement       similarity-weighted share of the top tickets whose resolution steps overlap with the leading group
* similarity_gate 0..1, ramps up from the abstain threshold: off-topic text can agree with itself, so low similarity must
                  cap the score (an off-topic complaint scores ~0.1)
* slope / bias    logistic calibration fitted on the eval sets (CV ECE ~0.03); re-fit per embedder with --calibrate

When confidence is low the closest cases describe *different* problems; we then say so and ask a clarifying question
built from the competing groups instead of presenting one blended answer as if it were sure.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.core.text import jaccard, strip_boilerplate
from app.core.vectorstore import Hit
from app.schemas import Confidence

FIX_OVERLAP = 0.30        # two cases "prescribe the same fix" if their resolution steps overlap at least this much
GATE_FULL_AT = 0.35       # relevance at which the similarity gate is fully open
KB_SCALE = 0.5            # KB chunks are longer than complaints, so their cosine runs lower: scale thresholds by this
MIN_AGREEMENT_FOR_MEDIUM = 0.60  # split recommendations should trigger a clarifying question
STRONG_RELEVANCE_FOR_DISAGREEMENT_OVERRIDE = 0.80  # strong matches can remain medium under the fitted calibration


@dataclass
class Group:
    members: list[int]
    weight: float


def relevance(score: float, doc_type: str, s: Settings) -> float:
    """0..1 match strength: 0 at the abstain floor, 1 at a 'strong' match, scaled per document type."""
    k = KB_SCALE if doc_type == "kb" else 1.0
    floor, strong = s.effective_abstain * k, s.effective_strong * k
    return max(0.0, min(1.0, (score - floor) / max(strong - floor, 1e-6)))


def _step_similarity(steps: list[str], embed, threshold: float):
    """Pairwise 'prescribe the same fix' test. Word overlap catches templated steps; step-text *meaning* (cosine of the
    embedder's vectors) catches the same fix worded differently, which is how real agents write ("restart the router" /
    "power cycle the modem")."""
    n = len(steps)
    vec = None
    if embed is not None and n > 1:
        try:
            v = np.asarray(embed(steps), dtype=float)
            vec = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        except Exception:  # noqa: BLE001 - never fail a request because of the confidence layer
            vec = None
    return lambda i, j: i == j or jaccard(steps[i], steps[j]) >= FIX_OVERLAP or (vec is not None and float(vec[i] @ vec[j]) >= threshold)


def fix_groups(tickets: list[Hit], embed=None, link_threshold: float = 0.2) -> list[Group]:
    """Group tickets that prescribe the same fix; heaviest group first (weights = similarity squared)."""
    w = [max(h.score, 0.0) ** 2 for h in tickets]
    steps = [" ".join(h.payload.get("resolution_steps") or []) for h in tickets]
    same = _step_similarity(steps, embed, link_threshold)
    left, groups = list(range(len(tickets))), []

    def support(i: int, pool: list[int]) -> float:
        return sum(w[j] for j in pool if same(i, j))

    while left:
        seed = max(left, key=lambda i, pool=tuple(left): support(i, list(pool)))
        members = [j for j in left if same(seed, j)]
        groups.append(Group(members, sum(w[j] for j in members)))
        left = [j for j in left if j not in members]
    return sorted(groups, key=lambda g: -g.weight)


def _title(h: Hit) -> str:
    """A short, readable label for a competing fix group: the first sentence of the complaint, cut at a word boundary."""
    text = strip_boilerplate(h.payload.get("text") or h.payload.get("title") or "").strip()
    first = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].rstrip(" .!?,;:")
    if len(first) > 80:
        first = first[:80].rsplit(" ", 1)[0].rstrip(" .,;:") + "..."
    return first


def assess(tickets: list[Hit], kb: list[Hit], s: Settings, ticket_ids: list[str] | None = None,
           embed=None) -> tuple[Confidence, list[int]]:
    """Return the confidence and the indices (into `tickets`) of the dominant fix group."""
    ids = ticket_ids or [f"T{i}" for i in range(1, len(tickets) + 1)]
    rel_t = relevance(max((h.score for h in tickets), default=0.0), "ticket", s)
    rel_k = relevance(max((h.score for h in kb), default=0.0), "kb", s)
    rel = max(rel_t, rel_k)
    top_sim = max((h.score for h in tickets + kb), default=0.0)
    if not tickets:   # only KB evidence: no agreement to measure, so be modest
        score = round(0.5 * min(1.0, rel_k / GATE_FULL_AT), 3)
        reasons = ["Only knowledge-base articles matched; no resolved tickets to cross-check."]
        return Confidence(score=score, level=_level(score, s), agreement=0.0, similarity=round(top_sim, 4),
                          relevance=round(rel, 3), reasons=reasons, source_ids=[]), []

    groups = fix_groups(tickets, embed, s.effective_fix_link)
    total = sum(g.weight for g in groups) or 1e-9
    n = len(tickets)
    agreement = groups[0].weight / total
    if n < 3:   # "1 of 1 agree" is not evidence: shrink towards a neutral prior when there is almost nothing to compare
        prior = (3 - n) / 3
        agreement = (1 - prior) * agreement + prior * 0.35
    gate = min(1.0, rel / GATE_FULL_AT)
    p = 1 / (1 + math.exp(-(s.conf_slope * agreement + s.conf_bias)))
    score = round(p * gate, 3)
    if (len(groups) > 1 and agreement < MIN_AGREEMENT_FOR_MEDIUM
            and rel < STRONG_RELEVANCE_FOR_DISAGREEMENT_OVERRIDE):
        score = min(score, s.conf_medium - 0.001)
    level = _level(score, s)

    if n < 3:
        reasons = [f"Only {n} resolved case{'s' if n != 1 else ''} matched, so agreement cannot be checked reliably."]
    else:
        reasons = [f"{len(groups[0].members)} of the {n} closest cases prescribe the same fix (agreement {agreement:.0%})."]
    if rel < GATE_FULL_AT and agreement >= 0.8:
        reasons.append(f"The wording differs from earlier tickets (match strength {rel:.0%}), but the closest cases all agree.")
    elif rel < GATE_FULL_AT:
        reasons.append("The best match is weak: the complaint is not very close to anything resolved before.")
    questions: list[str] = []
    if level == "low" and rel >= 0.05:
        reasons.append("The closest cases describe different problems, so the draft below is tentative.")
        options = [_title(tickets[g.members[0]]) for g in groups[:3] if _title(tickets[g.members[0]])]
        if len(options) >= 2:
            quoted = "; ".join(f"({chr(97 + i)}) {o}" for i, o in enumerate(dict.fromkeys(options)))
            questions.append(f"Which is closest to the customer's problem? {quoted}. Add that detail to the complaint and resolve again.")
    elif level == "low":
        reasons.append("Nothing resolved before is close to this complaint.")
    cluster_ids = [ids[i] for i in groups[0].members]
    return (Confidence(score=score, level=level, agreement=round(agreement, 3), similarity=round(top_sim, 4),
                       relevance=round(rel, 3), reasons=reasons, clarifying_questions=questions, source_ids=cluster_ids),
            groups[0].members)


def _level(score: float, s: Settings) -> str:
    return "high" if score >= s.conf_high else "medium" if score >= s.conf_medium else "low"
