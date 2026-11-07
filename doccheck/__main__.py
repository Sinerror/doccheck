"""CLI.

    uv run python -m doccheck check data/eval/docs/d02_kp_logistika.md --run-id demo --overwrite
    uv run python -m doccheck index            # (re)index the corpus folder
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from .config import get_settings
from .context import RunParams
from .events import EventBus
from .orchestrator import report_summary, run_pipeline
from .rag.corpus import index_corpus
from .rag.qdrant_store import QdrantStore
from .sources.local import SUPPORTED, load_file


def make_store(s) -> QdrantStore:
    st = QdrantStore(QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=60), s.collection)
    st.ensure_collection()
    return st


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="doccheck")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="check one document against the corpus")
    c.add_argument("path", help=f"document to check ({', '.join(SUPPORTED)})")
    c.add_argument("--corpus", default=None, help="corpus folder (default: DOCCHECK_CORPUS_DIR or data/corpus)")
    c.add_argument("--max-claims", type=int, default=10)
    c.add_argument("--debate-rounds", type=int, default=2)
    c.add_argument("--no-critic", action="store_true", help="analyst only")
    c.add_argument("--run-id", default=None)
    c.add_argument("--overwrite", action="store_true", help="delete an existing runs/<run-id> first")
    i = sub.add_parser("index", help="index the corpus folder")
    i.add_argument("--corpus", default=None)
    i.add_argument("--force", action="store_true")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = get_settings()
    corpus = Path(args.corpus) if args.corpus else s.corpus_dir
    st = make_store(s)

    if args.cmd == "index":
        idx = index_corpus(st, corpus, force=args.force)
        print(json.dumps({"scope": idx.scope, "docs": len(idx.docs), "chunks": idx.chunks, "reused": idx.reused},
                         ensure_ascii=False))
        return 0

    from .api import new_run_id
    run_id = args.run_id or new_run_id()
    rd = s.runs_dir / run_id
    if rd.exists():
        if not args.overwrite:
            print(f"{rd} exists; pass --overwrite", file=sys.stderr)
            return 2
        shutil.rmtree(rd)
    path = Path(args.path)
    doc = load_file(path)
    bus = EventBus(run_id, rd / "events.jsonl")
    params = RunParams(max_claims=args.max_claims, debate_rounds=args.debate_rounds, use_critic=not args.no_critic)
    report = asyncio.run(run_pipeline(run_id=run_id, settings=s, store=st, bus=bus, doc=doc, filename=path.name,
                                      params=params, corpus_dir=corpus))
    if report is None:
        print(f"run failed, see {rd / 'error.txt'}", file=sys.stderr)
        return 1
    print(json.dumps(report_summary(report), indent=2, ensure_ascii=False))
    print(f"report: {rd / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
