from doccheck.roles.scout import dedup_papers
from doccheck.sources.arxiv import _parse_feed, parse_arxiv_id
from doccheck.sources.openalex import normalize_work, reconstruct_abstract


def test_reconstruct_abstract_from_inverted_index():
    inv = {"Attention": [0], "is": [1, 5], "all": [2], "you": [3], "need": [4], "great": [6]}
    assert reconstruct_abstract(inv) == "Attention is all you need is great"
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""


def test_normalize_work_detects_arxiv_and_doi():
    w = {
        "id": "https://openalex.org/W123", "doi": "https://doi.org/10.48550/arXiv.1508.04025",
        "title": "Effective Approaches", "publication_year": 2015,
        "authorships": [{"author": {"display_name": "Minh-Thang Luong"}}],
        "primary_location": {"landing_page_url": "https://arxiv.org/abs/1508.04025", "source": {"display_name": "arXiv"}},
        "abstract_inverted_index": {"We": [0], "study": [1]},
    }
    n = normalize_work(w)
    assert n["paper_id"] == "arxiv:1508.04025"
    assert n["doi"] == "10.48550/arxiv.1508.04025"
    assert n["abstract"] == "We study"
    assert n["pdf_url"].endswith("/pdf/1508.04025")


def test_parse_arxiv_id_variants():
    assert parse_arxiv_id("1706.03762") == "1706.03762"
    assert parse_arxiv_id("https://arxiv.org/abs/1706.03762v7") == "1706.03762"
    assert parse_arxiv_id("https://arxiv.org/pdf/2101.00001.pdf") == "2101.00001"
    assert parse_arxiv_id("hep-th/9901001") == "hep-th/9901001"
    assert parse_arxiv_id("not an id") is None


def test_parse_arxiv_feed():
    xml = """<?xml version='1.0'?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/abs/1706.03762v7</id><title>Attention Is
      All You Need</title><summary> The dominant models. </summary><published>2017-06-12T17:57:34Z</published>
      <author><name>Ashish Vaswani</name></author></entry></feed>"""
    [p] = _parse_feed(xml)
    assert p["paper_id"] == "arxiv:1706.03762" and p["title"] == "Attention Is All You Need"
    assert p["year"] == 2017 and p["authors"] == ["Ashish Vaswani"] and p["abstract"] == "The dominant models."


def test_dedup_by_doi_arxiv_and_title_and_excludes_target():
    target = {"paper_id": "arxiv:1706.03762", "arxiv_id": "1706.03762", "title": "Attention Is All You Need"}
    papers = [
        {"paper_id": "openalex:W1", "doi": "10.1/abc", "title": "Paper A", "abstract": ""},
        {"paper_id": "arxiv:1111.11111", "arxiv_id": "1111.11111", "doi": "10.1/ABC", "title": "Paper A (v2)",
         "abstract": "has abstract"},
        {"paper_id": "openalex:W2", "title": "Convolutional Sequence to Sequence Learning"},
        {"paper_id": "arxiv:1705.03122", "arxiv_id": "1705.03122", "title": "Convolutional sequence-to-sequence learning"},
        {"paper_id": "openalex:W9", "title": "Attention is all you need!"},  # the target itself
    ]
    out = dedup_papers(papers, exclude=target)
    assert len(out) == 2
    a, c = out
    assert a["paper_id"] == "arxiv:1111.11111" and a["abstract"] == "has abstract" and a["hits"] == 2
    assert c["paper_id"] == "arxiv:1705.03122" and c["hits"] == 2
