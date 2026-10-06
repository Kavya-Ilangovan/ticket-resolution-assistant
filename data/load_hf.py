"""Load the public Hugging Face support-ticket dataset into the knowledge base (tickets AND KB articles).

    pip install datasets                       # only needed for this loader
    python -m data.load_hf                     # 3000 English tickets + up to 300 derived KB articles
    python -m data.load_hf --limit 10000 --kb-limit 600
    python -m data.load_hf --csv C:\\data\\tickets.csv    # a file you downloaded yourself (no `datasets`, no network)
    python -m data.load_hf --dry-run           # show what would be imported (and the detected columns), import nothing

Idempotent: running it again skips unchanged rows. Imported documents are tagged origin="hf" in the UI/API.
The mapping and cleaning rules live in app/services/hf_import.py (unit-tested on a fixture in the HF schema).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # Windows: avoid the developer-mode symlink warning

from app.config import get_settings  # noqa: E402
from app.core.context import get_context  # noqa: E402
from app.db.session import init_engine, session_scope  # noqa: E402
from app.services import hf_import  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=3000, help="max tickets to import (0 = all)")
    ap.add_argument("--kb-limit", type=int, default=300, help="max KB articles to derive (0 = none)")
    ap.add_argument("--lang", default="en", help="language filter, '' for all")
    ap.add_argument("--per-queue", type=int, default=None, help="cap tickets per queue/department (balance the corpus)")
    ap.add_argument("--repo", default=hf_import.REPO, help="Hugging Face dataset id")
    ap.add_argument("--csv", default=None, help="local .csv/.jsonl/.json file instead of downloading")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    try:
        plan = hf_import.build_plan(hf_import.iter_rows(a.repo, a.csv), lang=a.lang or None, limit=a.limit or None,
                                    kb_limit=a.kb_limit, per_queue=a.per_queue)
    except Exception as e:  # noqa: BLE001 - show a readable message, not a stack trace, for the usual problems
        print(f"Could not load the dataset: {e}", file=sys.stderr)
        return 1
    print(json.dumps(plan.report, indent=2))
    if not plan.tickets:
        print("Nothing to import (check --lang and the detected columns above).", file=sys.stderr)
        return 1
    if a.dry_run:
        return 0
    init_engine(get_settings())
    ctx = get_context()
    with session_scope() as db:
        res = hf_import.run_import(ctx, db, plan)
    print(json.dumps({k: v for k, v in res.items() if k != "report"}, indent=2))
    try:
        ctx.vectors.client.close()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
