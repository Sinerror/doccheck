"""Local documents → sections. Supports Markdown, plain text, HTML, DOCX and PDF.

The company corpus is a folder of such files; the document under check is one more file (or pasted text).
Sections follow the document's own headings, so a chunk never mixes two clauses of a contract.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

SUPPORTED = (".md", ".markdown", ".txt", ".html", ".htm", ".docx", ".pdf")


@dataclass
class ParsedDoc:
    title: str
    sections: list[tuple[str, str]] = field(default_factory=list)  # (section heading, text)
    page_count: int = 0

    def section_text(self, pattern: str) -> str:
        rx = re.compile(pattern, re.I)
        return "\n".join(t for s, t in self.sections if rx.search(s))

    @property
    def headings(self) -> list[str]:
        return [s for s, _ in self.sections]

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"## {s}\n{t}" if s else t for s, t in self.sections)

    @property
    def chars(self) -> int:
        return sum(len(t) for _, t in self.sections)


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def parse_markdown(text: str, default_title: str = "") -> ParsedDoc:
    """`#` is the document title; `##`..`######` start sections. Text before the first `##` goes to a
    section named after the title (so a short reply without headings is still one section)."""
    title = default_title
    sections: list[tuple[str, list[str]]] = []
    current: list[str] = []
    current_name: str | None = None
    for line in text.splitlines():
        m = _MD_HEADING.match(line)
        if m and len(m.group(1)) == 1 and not title_seen(sections, current, current_name):
            title = m.group(2).strip()
            continue
        if m:
            sections.append((current_name or title or "Документ", current))
            current_name, current = m.group(2).strip(), []
            continue
        current.append(line)
    sections.append((current_name or title or "Документ", current))
    out = [(name, _clean_block("\n".join(buf))) for name, buf in sections]
    return ParsedDoc(title=title or default_title, sections=[(n, t) for n, t in out if t])


def title_seen(sections: list, current: list[str], current_name: str | None) -> bool:
    """A level-1 heading is the title only if nothing but blank lines came before it."""
    return bool(sections) or current_name is not None or any(x.strip() for x in current)


def _clean_block(text: str) -> str:
    # keep paragraph breaks, squeeze the rest
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return "\n".join(p for p in paras if p)


def parse_text(text: str, title: str = "") -> ParsedDoc:
    if any(_MD_HEADING.match(line) for line in text.splitlines()):
        return parse_markdown(text, title)
    body = _clean_block(text)
    return ParsedDoc(title=title, sections=[(title or "Документ", body)] if body else [])


class _HTMLSections(HTMLParser):
    HEAD = {"h1", "h2", "h3", "h4"}
    SKIP = {"script", "style", "nav", "footer", "header"}
    BLOCK = {"p", "li", "tr", "div", "br", "td", "th"}

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.sections: list[tuple[str, list[str]]] = [("", [])]
        self._in_head: str | None = None
        self._head_buf: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.HEAD:
            self._in_head, self._head_buf = tag, []
        elif tag in self.BLOCK:
            self.sections[-1][1].append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == self._in_head:
            name = re.sub(r"\s+", " ", "".join(self._head_buf)).strip()
            if tag == "h1" and not self.title:
                self.title = name
            else:
                self.sections.append((name, []))
            self._in_head = None

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_head:
            self._head_buf.append(data)
        else:
            self.sections[-1][1].append(data)


def parse_html(html: str, title: str = "") -> ParsedDoc:
    p = _HTMLSections()
    p.feed(html)
    t = p.title or title
    out = []
    for name, buf in p.sections:
        txt = _clean_block(re.sub(r"\n", "\n\n", "".join(buf)))
        if txt:
            out.append((name or t or "Документ", txt))
    return ParsedDoc(title=t, sections=out)


def parse_docx(data: bytes, title: str = "") -> ParsedDoc:
    import docx  # python-docx

    d = docx.Document(io.BytesIO(data))
    t = ""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for para in d.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style is not None else ""
        first_h1 = style in ("heading 1", "заголовок 1") and not t and not any(b for _, b in sections)
        if style == "title" or first_h1:  # the document title, not a section
            t = t or text
            continue
        if style.startswith("heading") or style.startswith("заголовок"):
            sections.append((text, []))
        else:
            sections[-1][1].append(text)
    for table in d.tables:  # tables go to a section of their own, row by row
        rows = [" | ".join(c.text.strip() for c in row.cells) for row in table.rows]
        sections.append(("Таблица", rows))
    t = t or title
    out = [(n or t or "Документ", _clean_block("\n\n".join(buf))) for n, buf in sections]
    return ParsedDoc(title=t, sections=[(n, x) for n, x in out if x])


def parse_bytes(data: bytes, filename: str) -> ParsedDoc:
    ext = Path(filename).suffix.lower()
    stem = Path(filename).stem
    if ext == ".pdf":
        from .pdf import parse_pdf
        return parse_pdf(data)
    if ext == ".docx":
        return parse_docx(data, stem)
    text = data.decode("utf-8-sig", errors="replace")
    if ext in (".html", ".htm"):
        return parse_html(text, stem)
    if ext in (".md", ".markdown"):
        return parse_markdown(text, stem)
    return parse_text(text, stem)


def load_file(path: Path) -> ParsedDoc:
    return parse_bytes(path.read_bytes(), path.name)


@dataclass
class CorpusDoc:
    doc_id: str          # file name relative to the corpus root, e.g. "04_oferta.md"
    title: str
    parsed: ParsedDoc


def load_corpus(root: Path) -> list[CorpusDoc]:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED
                   and not p.name.startswith((".", "_")))
    out = []
    for p in files:
        parsed = load_file(p)
        if parsed.sections:
            rel = p.relative_to(root).as_posix()
            out.append(CorpusDoc(doc_id=rel, title=parsed.title or p.stem, parsed=parsed))
    return out


def corpus_fingerprint(root: Path, extra: str = "") -> str:
    """Content hash of the corpus folder: the index is rebuilt only when a file changes."""
    h = hashlib.sha1(extra.encode())
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in SUPPORTED):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]
