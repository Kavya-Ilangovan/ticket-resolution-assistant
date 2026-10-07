from app.core.pii import redact
from app.core.text import chunk_text, extract_steps, jaccard
from app.schemas import split_steps
from app.services.analyzer import detect_intent, detect_sentiment, detect_severity, product_from_lexicon

BRIEF = ("My broadband drops every evening around 8 and I've already restarted the router twice, "
         "I work from home and this is costing me")


def test_redaction():
    t = redact("mail a.b@x.com or call +91 98765 43210, card 4111 1111 1111 1111, account no 12345678")
    assert not any(x in t for x in ("@", "98765", "4111", "12345678"))
    assert "8 pm" in redact("it drops around 8 pm")


def test_chunking_and_steps():
    chunks = chunk_text("Intro.\n\nProcedure:\n1. First do this\n2. Then do that\n\n" + "word " * 400, max_words=100)
    assert len(chunks) >= 3
    assert extract_steps("Intro\n1. Alpha beta gamma\n2. Delta epsilon zeta") == ["Alpha beta gamma", "Delta epsilon zeta"]
    assert split_steps("1. a b c\n2. d e f") == ["a b c", "d e f"]
    assert jaccard("restart the router now", "restarting router now") > 0.5


def test_brief_example_parsing():
    sev, signals = detect_severity(BRIEF)
    assert sev in ("high", "critical") and {"business_or_work_impact", "repeated_attempts"} <= set(signals)
    label, score = detect_sentiment(BRIEF)
    assert label == "negative" and score < 0
    assert {"Fiber Broadband", "Router/Modem"} <= set(product_from_lexicon(BRIEF))
    assert detect_intent(BRIEF) == "fault_report"


def test_intents_severity_sentiment():
    assert detect_intent("I was charged twice and want a refund") == "billing_dispute"
    assert detect_intent("I want to cancel my service") == "churn_risk"
    assert detect_severity("Hi, how do I upgrade my plan? Just checking the options.")[0] == "low"
    assert detect_sentiment("Thanks, I appreciate the help!")[0] == "positive"
    assert detect_sentiment("this is not good at all")[0] == "negative"


def test_novelty_gate_uses_vote_agreement():
    from app.services.analyzer import is_known_class
    nov = 0.28
    assert is_known_class(0.40, 0.2, nov)            # clearly similar: known even with a split vote
    assert is_known_class(0.20, 0.9, nov)            # weakly similar but neighbours agree: known
    assert not is_known_class(0.20, 0.4, nov)        # weakly similar and neighbours disagree: possible new class
    assert not is_known_class(0.10, 1.0, nov)        # nothing is close: never known, whatever the vote


def test_handwritten_set_is_well_formed():
    from scripts.seed import read
    qs, tickets = read("realistic_queries", "handwritten"), read("tickets")
    keys = {t["issue_key"] for t in tickets}
    assert len(qs) >= 30 and {q["issue_key"] for q in qs} == keys     # every known issue is covered
    assert all(q["severity"] in {"low", "medium", "high", "critical"} for q in qs)
    train = {t["body"] for t in tickets}
    assert not any(q["text"] in train for q in qs)                      # no leakage from the index


def test_brief_example_is_parsed_as_asked():
    assert detect_severity(BRIEF)[0] in ("high", "critical")


def test_reranker_reorders_without_changing_scores():
    from app.core.reranker import rerank_hits
    from app.core.vectorstore import Hit

    class Longest:
        def scores(self, query, passages):
            return [float(len(p)) for p in passages]

    hits = [Hit(id=str(i), score=1.0 - i / 10, payload={"title": "t", "text": "x" * (i + 1)}) for i in range(3)]
    out = rerank_hits(Longest(), "q", hits)
    assert [h.id for h in out] == ["2", "1", "0"] and [h.score for h in out] == [0.8, 0.9, 1.0]
    assert rerank_hits(None, "q", hits) == hits
