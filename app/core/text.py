"""Text helpers: tokenisation, chunking, hashing, step extraction."""
from __future__ import annotations

import hashlib
import re

STOPWORDS = frozenset(
    """a an the and or but if then else of to in on at by for with from as is are was were be been being it its this that
    these those i me my we our you your they them their he she his her do does did have has had not no so than too very can
    could would should will just also about into over again there here what when where which who how all any some""".split()
)
_WORD = re.compile(r"[a-z0-9']+")


def tokenize(text: str, *, drop_stop: bool = True) -> list[str]:
    toks = _WORD.findall(text.lower())
    return [t for t in toks if not (drop_stop and t in STOPWORDS)]


# Words that say "something is wrong" but not *what*. They match every complaint, so on short queries such as
# "wifi not working" they drag in password resets, SIM faults and billing disputes. They are ignored when building
# retrieval vectors (never in the grounding checks, where every word counts).
GENERIC_TERMS = frozenset(
    "working work works worked issue issues problem problems help need needs want wants since still tried trying try "
    "get gets getting got please kindly thanks thank urgent urgently asap support team customer service sir madam".split()
)


def content_tokens(text: str) -> list[str]:
    """Tokens that carry topic information (stop words and generic complaint words removed)."""
    return [t for t in tokenize(text) if t not in GENERIC_TERMS] or tokenize(text)


def stem(tok: str) -> str:
    """Very light suffix stripping (enough to equate 'drops/dropping/dropped')."""
    for suf in ("ings", "ing", "edly", "ed", "ies", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


_GREET = r"(?:hi|hello|hey|dear|good (?:morning|afternoon|evening))"
# "Hi team," / "Dear support team," (greeting + up to 3 name words + punctuation), else just a bare "Hello" or "Hi,"
_SALUTATION = re.compile(rf"^\s*(?:{_GREET}\b(?:[ \t]+[A-Za-z']{{1,15}}){{0,3}}[ \t]*[,.!:]|{_GREET}\b[ \t]*|to whom it may concern[,.:]?)\s*", re.I)
_PLEASANTRY = re.compile(
    r"\b(?:thank(?:s| you)|appreciate|regards|sincerely|cheers|please advise|kindly (?:advise|help|assist|look)|"
    r"look(?:ing)? forward|let me know|hope (?:you|this)|apologi[sz]e|sorry to (?:bother|trouble)|dear customer|"
    r"best wishes|yours (?:faithfully|truly))\b", re.I)


def is_pleasantry(sentence: str) -> bool:
    """A short sentence that is politeness rather than content ("Thanks in advance", "We apologize for the delay")."""
    return len(sentence.split()) <= 14 and bool(_PLEASANTRY.search(sentence))


def strip_boilerplate(text: str) -> str:
    """Remove greetings, sign-offs and pleasantries so that retrieval compares the *problem*, not the politeness.
    Only used to build retrieval vectors: stored text, sentiment and severity cues are left untouched. Never returns
    an empty string."""
    t = _SALUTATION.sub("", text.strip())
    kept = [sent for sent in re.split(r"(?<=[.!?])\s+|\n+", t) if sent.split() and not is_pleasantry(sent)]
    out = " ".join(kept).strip()
    return out or text.strip()


def strip_salutation(text: str) -> str:
    return _SALUTATION.sub("", text.strip())


def content_hash(*parts: object) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def chunk_text(text: str, max_words: int = 140, overlap: int = 25) -> list[str]:
    """Paragraph-aware chunker for KB articles (keeps numbered procedures together when possible)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paras:
        n = len(p.split())
        if cur and cur_len + n > max_words:
            chunks.append("\n\n".join(cur))
            tail = " ".join(" ".join(cur).split()[-overlap:]) if overlap else ""
            cur, cur_len = ([tail] if tail else []), len(tail.split())
        if n > max_words * 1.5:  # giant paragraph: hard-split by words
            words = p.split()
            for i in range(0, len(words), max_words - overlap):
                chunks.append(" ".join(words[i : i + max_words]))
            cur, cur_len = [], 0
            continue
        cur.append(p)
        cur_len += n
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks or [text.strip()]


_LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.*\S)\s*$")


def extract_steps(text: str) -> list[str]:
    """Pull procedure steps out of free text: list items if present, else sentences."""
    items = [m.group(1) for ln in text.splitlines() if (m := _LIST_ITEM.match(ln))]
    if items:
        return items
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 4]
    return sents


def token_overlap(a: str, b: str) -> float:
    """Fraction of a's content stems that appear in b (asymmetric support score)."""
    ta = {stem(t) for t in tokenize(a)}
    tb = {stem(t) for t in tokenize(b)}
    return len(ta & tb) / len(ta) if ta else 0.0


def jaccard(a: str, b: str) -> float:
    ta = {stem(t) for t in tokenize(a)}
    tb = {stem(t) for t in tokenize(b)}
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
