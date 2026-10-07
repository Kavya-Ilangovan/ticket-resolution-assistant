"""Synthetic support data: tickets, KB articles, held-out eval queries and a novel class.

Every ticket carries an `issue_key` (the canonical problem), so retrieval relevance is "same issue_key".
The last two symptom phrasings of each issue are used only for eval queries, never for corpus tickets.
Run: python -m data.generate_synthetic
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from data.issue_catalog import DETAILS, ISSUES, NOVEL_ISSUES

OUT = Path(__file__).parent / "synthetic"

# ------------------------------------------------------------------ complaint composition
ATTEMPTS = ["I've already tried the basic steps you suggested.", "I did everything I could think of on my side.",
            "I tried again several times and it didn't help.", "I already contacted you about this once.", "", "", ""]
DEVICE_ATTEMPTS = ["I've already restarted the device twice.", "I rebooted everything and it didn't help.",
                   "I tried switching it off and on several times."]
DEVICE_CATEGORIES = {"Broadband Connectivity", "Router & Hardware", "TV & Streaming", "Mobile Network & Coverage",
                     "Voice & Messaging", "Device & eSIM"}
IMPACT = ["I work from home and this is costing me.", "I have client calls all day and this is hurting my business.",
          "My kids' online classes are affected.", "I'm losing money every day this continues.",
          "This is urgent, I have a deadline tomorrow."]
OUTAGE = ["It has been completely down for three days.", "I have no service at all and it's an emergency."]
DURATION = ["This has been going on for two weeks now.", "It started since last Monday.", "It happens every day.", "It has been like this for days."]
CLOSERS = {
    "negative": ["This is completely unacceptable.", "I'm extremely frustrated with your service!", "Fix this now!",
                 "I'm seriously considering switching providers.", "Worst experience ever, I'm fed up."],
    "neutral": ["Please advise.", "Can you look into it?", "Let me know what to do.", "", ""],
    "positive": ["Thanks in advance for the help!", "I appreciate your support, thank you.", "Thanks, hope you can help."],
}
PII = ["My number is 98765 43210.", "You can reach me at priya.k@example.com.", "Account number 4455667788.", ""]
OPENERS = ["Hi,", "Hello,", "Hi team,", "", "To whom it may concern,", "Dear support,"]
SEV = ["low", "medium", "high", "critical"]


def compose(rng: random.Random, issue: dict, symptom: str, split: str = "train") -> dict:
    sent = rng.choices(["negative", "neutral", "positive"], weights=[.45, .4, .15])[0]
    score = float(issue["base"])
    parts = [rng.choice(OPENERS), symptom[0].upper() + symptom[1:] + "."]
    if rng.random() < .75:
        parts.append(DETAILS[issue["key"]][0 if split == "train" else 1])  # test queries use unseen wording
    if rng.random() < .5:
        a = rng.choice(ATTEMPTS + (DEVICE_ATTEMPTS if issue["cat"] in DEVICE_CATEGORIES else []))
        if a:
            parts.append(a)
            score += .5
    if rng.random() < .2 and issue["base"] > 0:
        parts.append(rng.choice(IMPACT))
        score += 1.0
    if issue["base"] >= 2 and rng.random() < .35:
        parts.append(rng.choice(OUTAGE))
        score += 1.0
    if rng.random() < .2:
        parts.append(rng.choice(DURATION))
        score += .5
    parts.append(rng.choice(CLOSERS[sent]))
    if rng.random() < .1:
        parts.append(rng.choice(PII))
    sev = "critical" if score >= 3.0 else "high" if score >= 2.0 else "medium" if score >= 1.0 else "low"
    text = " ".join(p for p in parts if p).strip()
    return {"text": text, "severity": sev, "sentiment": sent}


def resolution(rng: random.Random, issue: dict) -> list[str]:
    steps = list(issue["steps"])
    if len(steps) > 4 and rng.random() < .3:
        steps.pop(rng.randrange(1, len(steps)))
    return steps


def kb_body(issue: dict) -> str:
    procedure = "\n".join(f"{i}. {s}" for i, s in enumerate(issue["steps"], 1))
    return (f"{issue['kbintro']}\n\nProcedure:\n{procedure}\n\n"
            f"Escalation: if the problem persists after the steps above, raise a Tier-2 ticket referencing {issue['key']} "
            f"and attach the diagnostics collected.")


def build(seed: int = 7, per_issue: int = 22, queries_per_issue: int = 4) -> None:
    rng = random.Random(seed)
    tickets, queries, kb, novel_tickets, novel_queries = [], [], [], [], []

    def make(issues, tickets_out, queries_out, n_tickets, n_queries, prefix):
        for issue in issues:
            n_tickets_issue = n_tickets if prefix == "nov" else rng.randint(n_tickets - 6, n_tickets + 6)
            train_sym, test_sym = issue["symptoms"][:5], issue["symptoms"][5:]
            for i in range(n_tickets_issue):
                c = compose(rng, issue, rng.choice(train_sym))
                tickets_out.append({
                    "external_id": f"{prefix}-{issue['key']}-{i:03d}", "subject": "", "body": c["text"],
                    "category": issue["cat"], "product": issue["prod"], "severity": c["severity"],
                    "resolution_steps": resolution(rng, issue), "status": "resolved", "issue_key": issue["key"],
                    "_sentiment": c["sentiment"]})
            for i in range(n_queries):
                c = compose(rng, issue, rng.choice(test_sym), split="test")
                queries_out.append({"id": f"q-{issue['key']}-{i}", "text": c["text"], "issue_key": issue["key"],
                                    "category": issue["cat"], "product": issue["prod"],
                                    "severity": c["severity"], "sentiment": c["sentiment"],
                                    "gold_steps": issue["steps"]})

    make(ISSUES, tickets, queries, per_issue, queries_per_issue, "tkt")
    make(NOVEL_ISSUES, novel_tickets, novel_queries, 12, 6, "nov")
    for issue in ISSUES:
        kb.append({"external_id": f"kb-{issue['key']}", "title": issue["kb"], "body": kb_body(issue),
                   "category": issue["cat"], "product": issue["prod"], "version": 1, "active": True})
    novel_kb = [{"external_id": f"kb-{i['key']}", "title": i["kb"], "body": kb_body(i), "category": i["cat"],
                 "product": i["prod"], "version": 1, "active": True} for i in NOVEL_ISSUES]
    ood = [
        "What time does the nearest retail store close on Sundays and do you sell phone cases?",
        "My washing machine makes a loud banging noise during the spin cycle, how do I fix it?",
        "Can you recommend a good recipe for vegetarian lasagna for six people?",
        "I was refused a mortgage by my bank, what documents do I need to reapply for a home loan?",
        "How do I reset my Netflix password? I forgot the email I signed up with.",
        "The airline lost my luggage on the flight from Mumbai, who do I talk to for compensation?",
        "My electricity meter is spinning really fast, is the power company overbilling me?",
        "Please write a poem about autumn leaves in the style of a pirate.",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in [("tickets", tickets), ("kb", kb), ("eval_queries", queries), ("novel_tickets", novel_tickets),
                       ("novel_kb", novel_kb), ("novel_queries", novel_queries),
                       ("ood_queries", [{"id": f"ood-{i}", "text": t} for i, t in enumerate(ood)])]:
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print({k: len(v) for k, v in dict(tickets=tickets, kb=kb, eval_queries=queries, novel_tickets=novel_tickets,
                                      novel_queries=novel_queries).items()})


if __name__ == "__main__":
    build()
