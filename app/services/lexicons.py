"""Generic word lists and regex cues for the complaint analyzer.

Domain-specific vocabulary (products, extra outage phrases) lives in a JSON profile (``app/domains/*.json``,
selected with ``DOMAIN_PROFILE``), so the same code serves any support domain.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains"


def any_of(*alternatives: str) -> str:
    return "|".join(alternatives)


# ------------------------------------------------------------------ severity: (pattern, weight, signal name)
# Scores add up; thresholds live in analyzer.detect_severity. Weights are hand-set, not fitted.
SEVERITY_CUES: list[tuple[str, float, str]] = [
    (any_of(r"\b(?:completely|totally|fully|entirely) (?:down|dead|offline|cut off)\b",
            r"\bno (?:service|access|connection) at all\b",
            r"\bemergency\b"),
     2.0, "outage_language"),
    (r"\b(?:" + any_of(r"work(?:ing)? from home", r"wfh", r"home office", r"my business", r"client calls?",
                       r"customers? (?:are|can'?t)", r"online classes?", r"exam", r"remote work", r"costing me",
                       r"losing (?:money|business|income|clients?)") + r")\b",
     1.0, "business_or_work_impact"),
    (r"\b(?:urgent(?:ly)?|asap|immediately|right now|deadline|critical)\b",
     0.7, "urgency_language"),
    (r"\b(?:" + any_of(r"twice", r"thrice", r"three times", r"several times", r"multiple times", r"again and again",
                       r"already (?:tried|restarted|rebooted|reset|called)", r"nothing works",
                       r"keeps? (?:happening|failing)") + r")\b",
     0.6, "repeated_attempts"),
    (r"\b(?:" + any_of(r"every (?:day|night|evening|morning|week)", r"daily", r"nightly",
                       r"for (?:weeks|days|a week|two weeks|a month|months)",
                       r"since (?:last|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)") + r")\b",
     0.6, "persistent_issue"),
    (r"\b(?:charged (?:twice|double)|double[- ]charged|overcharged|wrongly charged|unauthori[sz]ed)\b",
     0.6, "financial_impact"),
    (any_of(r"\b(?:escalate|manager|ombudsman|legal|complaint)\b",
            r"cancel(?:ling)? (?:my )?(?:service|connection|plan)",
            r"switch(?:ing)? (?:provider|operator)",
            r"port(?:ing)? out"),
     0.7, "churn_or_escalation_threat"),
]
SEVERITY_LOW_CUES = (r"\b(?:how (?:do|can) i|would like to|wondering|just (?:asking|checking)|question about|"
                     r"want to (?:change|know)|no rush|when (?:you|you've) (?:get|have) a chance)\b")

# ------------------------------------------------------------------ sentiment (word lists)
NEG = {"frustrated", "frustrating", "unacceptable", "terrible", "awful", "horrible", "worst", "useless", "angry", "furious",
       "annoyed", "annoying", "disappointed", "disappointing", "ridiculous", "pathetic", "fed", "sick", "tired", "poor",
       "bad", "waste", "unreliable", "nightmare", "disgusted", "outraged", "fail", "failed", "failing", "problem",
       "issue", "costing", "losing", "stuck", "broken", "slow", "unhappy", "complaint", "worried", "stressful", "again"}
# NB: "please" is deliberately absent: it marks a request, so polite complaints were being read as positive.
POS = {"thanks", "thank", "appreciate", "appreciated", "great", "good", "happy", "pleased", "helpful", "kindly",
       "grateful", "wonderful", "excellent", "love"}
INTENSIFIERS = {"very", "extremely", "really", "completely", "totally", "absolutely", "utterly", "so", "seriously", "highly"}
NEGATORS = {"not", "no", "never", "n't", "without", "hardly"}

# ------------------------------------------------------------------ intent: first matching rule wins, else fault_report
INTENT_RULES: list[tuple[str, str]] = [
    ("churn_risk", any_of(r"cancel(?:ling)? (?:my )?(?:service|connection|plan|subscription)",
                          r"switch(?:ing)? (?:to another )?(?:provider|operator|carrier)",
                          r"port(?:ing)? (?:my number )?out", r"leave you", r"close my account")),
    ("billing_dispute", r"\b(?:bill|billing|charged|charge|refund|invoice|payment|overcharg\w+|deduct\w+|autopay|debit(?:ed)?|credit(?:ed)?)\b"),
    ("service_request", r"\b(?:activate|activation|upgrade|downgrade|change (?:my )?(?:plan|address|number|name)|add (?:a )?(?:line|member)|new connection|relocat\w+|transfer|port(?:ing)? in|replace(?:ment)? sim|new sim)\b"),
    ("inquiry", r"\b(?:how (?:do|can|to)|what is|what's|can you (?:tell|explain)|information about|details (?:of|about)|is it possible)\b"),
]


@dataclass(frozen=True)
class Domain:
    name: str
    products: dict[str, list[re.Pattern]]
    product_boost: dict[str, float]
    severity: list[tuple[re.Pattern, float, str]]
    low_severity: re.Pattern
    intents: list[tuple[str, re.Pattern]]
    hf_queues: tuple[str, ...] = ()   # public-dataset queues worth importing for this domain


@lru_cache
def load_domain(profile: str = "telecom") -> Domain:
    """Load ``profile`` (a name in app/domains or a path to a JSON file) and merge it with the generic cues."""
    path = Path(profile) if profile.endswith(".json") else DOMAINS_DIR / f"{profile}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    cues = list(SEVERITY_CUES)
    if raw.get("outage_cues"):
        cues.append((any_of(*raw["outage_cues"]), 2.0, "outage_language"))
    flags = re.I
    return Domain(
        name=raw.get("name", "support desk"),
        products={p: [re.compile(x, flags) for x in xs] for p, xs in raw.get("products", {}).items()},
        product_boost=raw.get("product_boost", {}),
        severity=[(re.compile(p, flags), w, n) for p, w, n in cues],
        low_severity=re.compile(SEVERITY_LOW_CUES, flags),
        intents=[(label, re.compile(p, flags)) for label, p in INTENT_RULES],
        hf_queues=tuple(raw.get("hf_queues", ())),
    )
