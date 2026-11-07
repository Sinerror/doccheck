from __future__ import annotations

import os
from pathlib import Path

import pymupdf
import pytest
from qdrant_client import QdrantClient

from doccheck.rag.qdrant_store import QdrantStore

FIXTURES = Path(__file__).parent / "fixtures"
LLM_FIXTURES = FIXTURES / "llm"
RECORD = os.environ.get("DOCCHECK_RECORD") == "1"


@pytest.fixture
def mem_store() -> QdrantStore:
    st = QdrantStore(QdrantClient(":memory:"), "doccheck_test")
    st.ensure_collection()
    return st


def make_pdf(title: str, sections: list[tuple[str, str]]) -> bytes:
    """Build a small paper-like PDF: big title, bold 12pt numbered headings, 10pt body."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60

    def ensure_space(h: float):
        nonlocal page, y
        if y + h > 790:
            page = doc.new_page()
            y = 60

    page.insert_text((60, y), title, fontsize=17, fontname="hebo")
    y += 40
    for heading, body in sections:
        ensure_space(40)
        page.insert_text((60, y), heading, fontsize=12, fontname="hebo")
        y += 22
        words, line = body.split(), ""
        for w in words:
            if len(line) + len(w) + 1 > 95:
                ensure_space(14)
                page.insert_text((60, y), line, fontsize=10, fontname="helv")
                y += 13
                line = w
            else:
                line = f"{line} {w}".strip()
        if line:
            ensure_space(14)
            page.insert_text((60, y), line, fontsize=10, fontname="helv")
            y += 13
        y += 10
    data = doc.tobytes()
    doc.close()
    return data
