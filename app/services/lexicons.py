"""Word lists and regex cues used by the complaint analyzer (``analyzer.py``).

Kept apart from the logic so they can be read, reviewed and extended without touching code. Every table is plain data;
``any_of`` just joins alternatives into one regex so each cue reads as a list of phrases instead of one long pattern.
"""
from __future__ import annotations

import re


def any_of(*alternatives: str) -> str:
    return "|".join(alternatives)


# ------------------------------------------------------------------ product (regex per product)
PRODUCT_LEXICON: dict[str, list[str]] = {
    "Fiber Broadband": [r"broadband", r"fib(?:er|re)", r"\bftth\b", r"home internet", r"\binternet\b", r"wi-?fi", r"\bwifi\b",
                        r"\bont\b", r"\blos\b", r"speed", r"\bping\b", r"latency", r"\bline\b"],
    "Router/Modem": [r"router", r"modem", r"\bgateway\b", r"firmware", r"admin (?:page|password|panel)", r"\bssid\b",
                     r"factory reset"],
    "Mobile Postpaid": [r"postpaid", r"post-paid", r"monthly (?:plan|bill)", r"my (?:mobile )?bill", r"roaming"],
    "Mobile Prepaid": [r"prepaid", r"pre-paid", r"top-?up", r"recharge", r"\bpack\b"],
    "Mobile (SIM/Network)": [r"\bsim\b", r"\bpuk\b", r"\bsignal\b", r"\bnetwork coverage\b", r"\bcalls?\b", r"\bsms\b",
                             r"\bmobile data\b", r"\b[345]g\b", r"\blte\b", r"number port", r"\bporting\b", r"\bmnp\b"],
    "IPTV": [r"\biptv\b", r"set-?top", r"channels?", r"\btv\b", r"streaming", r"\bdecoder\b", r"on-?demand"],
    "5G Home Internet": [r"5g (?:home|fwa|router|cpe|outdoor|home internet)", r"fixed wireless", r"\bfwa\b", r"5g home"],
}

# ------------------------------------------------------------------ severity: (pattern, weight, signal name)
# Scores add up; thresholds live in analyzer.detect_severity. Weights are judgement calls, not fitted values.
SEVERITY_CUES: list[tuple[str, float, str]] = [
    (any_of(r"\b(?:completely|totally|fully|entirely) (?:down|dead|offline|cut off)\b",
            r"\bno (?:service|internet|signal|connection|dial ?tone) at all\b",
            r"\bemergency\b",
            r"\bcan'?t (?:call|reach) (?:emergency|112|911|100)\b"),
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
# calm, informational wording lowers severity
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


def compile_all():
    """Compile the regex tables once (case-insensitive)."""
    flags = re.I
    return (
        {p: [re.compile(x, flags) for x in xs] for p, xs in PRODUCT_LEXICON.items()},
        [(re.compile(p, flags), w, name) for p, w, name in SEVERITY_CUES],
        re.compile(SEVERITY_LOW_CUES, flags),
        [(label, re.compile(p, flags)) for label, p in INTENT_RULES],
    )
