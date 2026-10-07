"""Load a public Hugging Face support-ticket dataset (tickets and derived KB articles) into the knowledge base.

    pip install datasets
    python -m data.load_hf --dry-run                   # show detected columns and what would be imported
    python -m data.load_hf                             # import (queues chosen by the domain profile)
    python -m data.load_hf --queues all --limit 10000
    python -m data.load_hf --csv path/to/file.csv      # a downloaded file instead of the hub

Re-running is idempotent. Cleaning and mapping rules live in app/services/hf_import.py.
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
from app.services import hf_import, lexicons  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=3000, help="max tickets to import (0 = all)")
    ap.add_argument("--kb-limit", type=int, default=300, help="max KB articles to derive (0 = none)")
    ap.add_argument("--lang", default="en", help="language filter, '' for all")
    ap.add_argument("--queues", default=None, help="comma-separated queues to keep, or 'all' (default: the domain profile's list)")
    ap.add_argument("--per-queue", type=int, default=None, help="cap tickets per queue/department (balance the corpus)")
    ap.add_argument("--repo", default=hf_import.REPO, help="Hugging Face dataset id")
    ap.add_argument("--csv", default=None, help="local .csv/.jsonl/.json file instead of downloading")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.queues == "all":
        queues = None
    elif a.queues:
        queues = [q.strip() for q in a.queues.split(",") if q.strip()]
    else:
        queues = list(lexicons.load_domain(get_settings().domain_profile).hf_queues) or None
    try:
        plan = hf_import.build_plan(hf_import.iter_rows(a.repo, a.csv), lang=a.lang or None, limit=a.limit or None,
                                    kb_limit=a.kb_limit, per_queue=a.per_queue, queues=queues)
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
