"""PDF → sections with pymupdf.

Heading heuristic: short line that is either numbered ("3.2 Attention") / a known
unnumbered heading ("Abstract", "Conclusion"), or set in a font noticeably larger
than the body text. Fallback: one section per page. Bibliography is dropped from the corpus.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf as fitz

NUMBERED = re.compile(r"^(?:\d+(?:\.\d+){0,2}\.?|[A-H](?:\.\d+)?)\s+[A-Z][\w\-’' ,:&()/]{1,70}$")
KNOWN = re.compile(r"^(abstract|introduction|background|related work|conclusions?|discussion|results|"
                   r"experiments?|methods?|acknowledg(e)?ments?|references|bibliography|appendix)$", re.I)
DROP = re.compile(r"^(?:\d+\.?\s+)?(references|bibliography)$", re.I)


@dataclass
class ParsedPDF:
    title: str
    sections: list[tuple[str, str]] = field(default_factory=list)  # (section, text)
    page_count: int = 0

    def section_text(self, pattern: str) -> str:
        rx = re.compile(pattern, re.I)
        return "\n".join(t for s, t in self.sections if rx.search(s))

    @property
    def headings(self) -> list[str]:
        return [s for s, _ in self.sections]

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"{s}\n{t}" for s, t in self.sections)


def _raw_lines(doc: fitz.Document):
    for pno, page in enumerate(doc):
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                dx, dy = line.get("dir", (1, 0))
                if abs(dy) > 0.1:  # rotated text, e.g. the arXiv side stamp
                    continue
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = " ".join(s["text"].strip() for s in spans).strip()
                size = max(s["size"] for s in spans)
                bold = all("bold" in s.get("font", "").lower() or (s.get("flags", 0) & 16) for s in spans)
                yield pno, text, size, bold


_NUM_ONLY = re.compile(r"^(?:\d+(?:\.\d+){0,2}\.?|[A-H](?:\.\d+)?)$")


def _lines(doc: fitz.Document):
    """Raw lines, with a bold lone section number merged into the bold title that follows it."""
    pending = None
    for item in _raw_lines(doc):
        pno, text, size, bold = item
        if pending is not None:
            ppno, ptext, psize, pbold = pending
            pending = None
            if ppno == pno and bold and not _NUM_ONLY.match(text):
                yield pno, f"{ptext} {text}", max(size, psize), True
                continue
            yield ppno, ptext, psize, pbold
        if bold and _NUM_ONLY.match(text):
            pending = item
            continue
        yield item
    if pending is not None:
        yield pending


def _clean(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_pdf(data: bytes, min_sections: int = 3) -> ParsedPDF:
    doc = fitz.open(stream=data, filetype="pdf")
    lines = list(_lines(doc))
    sizes = Counter()
    for _, text, size, _ in lines:
        sizes[round(size, 1)] += len(text)
    body = sizes.most_common(1)[0][0] if sizes else 10.0

    first_page = [(t, s) for p, t, s, _ in lines if p == 0]
    title = ""
    if first_page:
        top = max(s for _, s in first_page)
        title = " ".join(t for t, s in first_page if abs(s - top) < 0.5)[:300]

    sections: list[tuple[str, list[str]]] = [("Front matter", [])]
    for pno, text, size, bold in lines:
        short = len(text) <= 80 and not text.endswith((".", ",", ";"))
        is_heading = short and (
            (NUMBERED.match(text) and (size >= body + 0.4 or bold))
            or (KNOWN.match(text) and (size >= body + 0.4 or bold))
            or (size >= body + 1.5 and len(text) > 3 and pno > 0 and not text[0].isdigit())
        )
        if is_heading and text != title:
            sections.append((text, []))
        else:
            sections[-1][1].append(text)

    out: list[tuple[str, str]] = []
    dropping = False
    for name, buf in sections:
        if DROP.match(name):
            dropping = True
            continue
        if dropping and not re.match(r"^[A-H](\.\d+)?\s|^appendix", name, re.I):
            continue
        dropping = False
        txt = _clean("\n".join(buf))
        if txt:
            out.append((name, txt))

    if len(out) < min_sections:  # fallback: per page
        out = []
        for pno, page in enumerate(doc):
            txt = _clean(page.get_text())
            if txt:
                out.append((f"Page {pno + 1}", txt))
    pc = doc.page_count
    doc.close()
    return ParsedPDF(title=title, sections=out, page_count=pc)


def abstract_like(parsed: ParsedPDF, limit: int = 2500) -> str:
    a = parsed.section_text(r"^abstract$")
    if a:
        return a[:limit]
    front = parsed.section_text(r"front matter")
    m = re.search(r"abstract[:.\s]+(.*)", front, re.I | re.S)
    return (m.group(1) if m else front)[:limit]
