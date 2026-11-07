"""CLI: `uv run python -m triad run 1706.03762 --max-claims 4 --run-id demo`."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys

from qdrant_client import QdrantClient

from .config import get_settings
from .context import RunParams
from .events import EventBus
from .orchestrator import report_summary, run_pipeline
from .rag.qdrant_store import QdrantStore
from .sources.arxiv import parse_arxiv_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="doccheck")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Paper review mode
    r = sub.add_parser("run", help="review one arXiv paper")
    r.add_argument("arxiv")
    r.add_argument("--max-claims", type=int, default=8)
    r.add_argument("--max-papers", type=int, default=15)
    r.add_argument("--debate-rounds", type=int, default=2)
    r.add_argument("--run-id", default=None)
    r.add_argument("--overwrite", action="store_true", help="delete an existing runs/<run-id> first")


    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    s = get_settings()
    from .api import new_run_id
    run_id = args.run_id or new_run_id()
    rd = s.runs_dir / run_id
    if rd.exists():
        if not args.overwrite:
            print(f"{rd} exists; pass --overwrite", file=sys.stderr)
            return 2
        shutil.rmtree(rd)

    st = QdrantStore(QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=30), s.collection)
    st.ensure_collection()
    st.delete_run(run_id)
    bus = EventBus(run_id, rd / "events.jsonl")

    # Paper review mode
    arxiv_id = parse_arxiv_id(args.arxiv)
    if not arxiv_id:
        print(f"not an arXiv id: {args.arxiv}", file=sys.stderr)
        return 2
    params = RunParams(max_claims=args.max_claims, max_papers=args.max_papers, debate_rounds=args.debate_rounds)
    report = asyncio.run(run_pipeline(run_id=run_id, settings=s, store=st, bus=bus, params=params,
                                      arxiv_id=arxiv_id))
    if report is None:
        print(f"run failed, see {rd / 'error.txt'}", file=sys.stderr)
        return 1
    print(json.dumps(report_summary(report), indent=2, ensure_ascii=False))
    print(f"report: {rd / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
