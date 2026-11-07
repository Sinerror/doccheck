"""Deterministic grounding check: is every quoted string really in the chunk it cites?"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from rapidfuzz import fuzz

THRESHOLD = 90.0
MAX_QUOTE_CHARS = 300


@dataclass
class GroundingResult:
    chunk_id: str
    quote: str
    score: float
    ok: bool
    exists: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip()


def score_quote(quote: str, chunk_text: str) -> float:
    q, t = _norm(quote), _norm(chunk_text)
    if not q or not t:
        return 0.0
    return round(float(fuzz.partial_ratio(q, t)), 2)


def check_one(chunk_id: str, quote: str, lookup: Callable[[str], str | None],
              threshold: float = THRESHOLD) -> GroundingResult:
    text = lookup(chunk_id)
    if text is None:
        return GroundingResult(chunk_id, quote, 0.0, False, False, "chunk_id does not exist")
    score = score_quote(quote, text)
    ok = score >= threshold
    reason = "" if ok else f"quote not found verbatim in chunk (partial_ratio={score:.0f} < {threshold:.0f})"
    if ok and len(quote) > MAX_QUOTE_CHARS:
        reason = f"quote longer than {MAX_QUOTE_CHARS} chars"
    return GroundingResult(chunk_id, quote, score, ok, True, reason)


def check_all(items: Iterable[tuple[str, str]], lookup: Callable[[str], str | None],
              threshold: float = THRESHOLD) -> list[GroundingResult]:
    return [check_one(cid, q, lookup, threshold) for cid, q in items]


def grounding_rate(results: Iterable[GroundingResult]) -> float | None:
    results = list(results)
    if not results:
        return None
    return round(sum(r.ok for r in results) / len(results), 4)
