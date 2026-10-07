"""LLM-as-judge for answer quality (needs OPENROUTER_API_KEY): is every step supported by what it cites?

    python -m evals.llm_judge --n 20 --backend sentence-transformers
"""
from __future__ import annotations

import argparse
import json
import random

from app.core.llm import LLMUnavailable
from app.schemas import ResolveOut

SYSTEM = ("You grade a support resolution. For each numbered step decide whether the cited source excerpts actually "
          "support it (1) or not (0). Then rate from 1 to 5 how well the steps address the complaint. "
          'Reply with JSON only: {"supported": [0 or 1 per step], "addresses_complaint": 1-5}.')


def judge(ctx, complaint: str, out: ResolveOut) -> dict | None:
    """Return {faithfulness, addresses_complaint} for one answer, or None if there is nothing to grade."""
    if not out.resolution.steps:
        return None
    sources = {s.id: s.snippet for s in out.sources}
    steps = "\n".join(f"{st.n}. {st.text}\n   cited: " + " | ".join(sources.get(c, "") for c in st.citations)
                      for st in out.resolution.steps)
    verdict = ctx.llm.chat_json([{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": f"Complaint: {complaint}\n\nSteps:\n{steps}"}])
    flags = [int(bool(x)) for x in verdict.get("supported", [])][:len(out.resolution.steps)]
    if not flags:
        return None
    return {"faithfulness": sum(flags) / len(flags), "addresses_complaint": float(verdict.get("addresses_complaint", 0))}


def main() -> None:
    from app.config import get_settings
    from app.db.session import session_scope
    from app.services import resolver
    from evals.run_all import MODELS, stack
    from scripts.seed import read, seed

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--backend", default="hashing", choices=list(MODELS))
    a = ap.parse_args()
    ctx = stack(a.backend)
    ctx.settings = ctx.settings.model_copy(update={"openrouter_api_key": get_settings().openrouter_api_key})
    ctx.llm.s = ctx.settings
    if not ctx.llm.enabled:
        raise SystemExit("Set OPENROUTER_API_KEY to run the judge.")
    queries = random.Random(0).sample(read("realistic_queries", "handwritten"), a.n)
    scores = []
    with session_scope() as db:
        seed(ctx, db)
        for q in queries:
            out = resolver.resolve(ctx, db, q["text"], "judge", use_cache=False)
            try:
                if (res := judge(ctx, q["text"], out)):
                    scores.append(res)
            except LLMUnavailable as e:
                print("judge unavailable:", e)
                break
    n = max(len(scores), 1)
    print(json.dumps({"graded": len(scores),
                      "faithfulness": round(sum(s["faithfulness"] for s in scores) / n, 3),
                      "addresses_complaint_1to5": round(sum(s["addresses_complaint"] for s in scores) / n, 2)}, indent=2))


if __name__ == "__main__":
    main()
