from tests.conftest import make_pdf
from doccheck.sources.pdf import abstract_like, parse_pdf

BODY = ("This is body text about the model and its evaluation on standard benchmarks. " * 12).strip()


def test_sections_from_numbered_bold_headings():
    pdf = make_pdf("A Tiny Paper About Attention", [
        ("Abstract", "We propose a tiny model. " + BODY),
        ("1 Introduction", BODY),
        ("2 Method", BODY),
        ("2.1 Details", BODY),
        ("3 Conclusion", "We presented a tiny model. " + BODY),
        ("References", "[1] Some reference. [2] Another reference."),
    ])
    p = parse_pdf(pdf)
    assert p.title == "A Tiny Paper About Attention"
    names = p.headings
    assert "Abstract" in names and "1 Introduction" in names and "2.1 Details" in names
    assert "References" not in names
    assert not any("Some reference" in t for _, t in p.sections)
    assert abstract_like(p).startswith("We propose a tiny model.")
    assert p.section_text("conclusion").startswith("We presented a tiny model.")


def test_fallback_to_pages_without_headings():
    import pymupdf
    doc = pymupdf.open()
    for i in range(2):
        doc.new_page().insert_text((60, 80), f"plain page {i} text without any headings at all.", fontsize=10)
    p = parse_pdf(doc.tobytes())
    assert p.headings == ["Page 1", "Page 2"]
