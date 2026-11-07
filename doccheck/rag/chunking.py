"""Sentence-aware chunking (Russian and English): ~1200 chars, ~200 overlap, never splits mid-sentence
(unless a single "sentence" is longer than the chunk size). Chunks never cross sections."""
from __future__ import annotations

import re
from dataclasses import dataclass

CHUNK_SIZE = 1200
OVERLAP = 200

# sentence end + optional closing quote/bracket + space + an upper-case letter (Latin or Cyrillic), digit,
# opening quote or bracket.
_SENT_SPLIT = re.compile(r"(?<=[.!?…])[\"'»)\]]*\s+(?=[A-ZА-ЯЁ0-9(\[\"'«])")
_CLAUSE_NO = re.compile(r"^(?:\d+\.)+\d*\.?$|^[А-ЯA-Z]\.$")  # "4.2." / "3." / "п." left alone by the split


@dataclass
class Chunk:
    section: str
    text: str
    chunk_idx: int = 0


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out: list[str] = []
    for s in (s.strip() for s in _SENT_SPLIT.split(text)):
        if not s:
            continue
        if out and _CLAUSE_NO.match(out[-1]):  # a clause number belongs to the sentence after it
            out[-1] = f"{out[-1]} {s}"
        else:
            out.append(s)
    return out


def _hard_split(sentence: str, size: int) -> list[str]:
    parts, cur = [], ""
    for word in sentence.split(" "):
        if cur and len(cur) + 1 + len(word) > size:
            parts.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        parts.append(cur)
    return parts


def chunk_section(section: str, text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    sents: list[str] = []
    for s in split_sentences(text):
        sents.extend(_hard_split(s, size) if len(s) > size else [s])
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sents:
        add = len(s) + (1 if cur else 0)
        if cur and cur_len + add > size:
            chunks.append(" ".join(cur))
            # carry over trailing sentences as overlap
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(cur):
                if tail_len >= overlap or tail_len + len(prev) > size // 2:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            cur, cur_len = tail, max(tail_len - 1, 0)
            add = len(s) + (1 if cur else 0)
            if cur and cur_len + add > size:  # overlap + sentence too long: drop overlap
                cur, cur_len, add = [], 0, len(s)
        cur.append(s)
        cur_len += add
    if cur:
        last = " ".join(cur)
        if not chunks or last not in chunks[-1]:
            chunks.append(last)
    return chunks


def chunk_document(sections: list[tuple[str, str]], size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[Chunk]:
    out: list[Chunk] = []
    for section, text in sections:
        for t in chunk_section(section, text, size, overlap):
            out.append(Chunk(section=section, text=t, chunk_idx=len(out)))
    return out
