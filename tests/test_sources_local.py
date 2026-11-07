import io

from tests.conftest import CORPUS, EVAL_DOCS
from doccheck.sources.local import (corpus_fingerprint, load_corpus, load_file, parse_bytes, parse_html,
                                    parse_markdown, parse_text)


def test_markdown_title_and_sections():
    d = parse_markdown("# Оферта\n\nВступление.\n\n## 1. Термины\n\n1.1. Сервис — CRM.\n\n## 2. Предмет\n\nТекст.")
    assert d.title == "Оферта"
    assert d.headings == ["Оферта", "1. Термины", "2. Предмет"]
    assert d.sections[1][1] == "1.1. Сервис — CRM."


def test_plain_reply_without_headings_is_one_section():
    d = parse_text("Здравствуйте!\n\nТариф стоит 4 990 ₽.\n\nС уважением.", "ответ")
    assert d.title == "ответ" and len(d.sections) == 1
    assert "4 990 ₽" in d.sections[0][1] and d.chars > 20


def test_html_headings_and_noise():
    html = ("<html><head><style>p{}</style></head><body><nav>меню</nav><h1>SLA</h1><p>Вступление</p>"
            "<h2>Доступность</h2><p>99,8% на «Бизнесе».</p><script>x=1</script></body></html>")
    d = parse_html(html, "sla")
    assert d.title == "SLA"
    assert ("Доступность", "99,8% на «Бизнесе».") in d.sections
    assert "меню" not in d.full_text and "x=1" not in d.full_text


def test_docx_headings_and_tables():
    import docx
    doc = docx.Document()
    doc.add_heading("Тарифы", level=1)
    doc.add_paragraph("Вводный текст.")
    doc.add_heading("Тариф «Команда»", level=2)
    doc.add_paragraph("До 25 пользователей.")
    t = doc.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Бизнес", "12 990 ₽"
    buf = io.BytesIO()
    doc.save(buf)
    d = parse_bytes(buf.getvalue(), "tarify.docx")
    assert d.title == "Тарифы"
    assert ("Тариф «Команда»", "До 25 пользователей.") in d.sections
    assert any("Бизнес | 12 990 ₽" in t for _, t in d.sections)


def test_corpus_loader_and_fingerprint(tmp_path):
    docs = load_corpus(CORPUS)
    assert len(docs) == 11 and docs[0].doc_id == "01_tarify_rf.md"
    oferta = next(d for d in docs if d.doc_id == "04_oferta.md")
    assert "4. Возврат денежных средств" in oferta.parsed.headings
    (tmp_path / "a.md").write_text("# A\n\nтекст", encoding="utf-8")
    f1 = corpus_fingerprint(tmp_path)
    (tmp_path / "a.md").write_text("# A\n\nдругой текст", encoding="utf-8")
    assert corpus_fingerprint(tmp_path) != f1


def test_eval_docs_load():
    d = load_file(EVAL_DOCS / "d02_kp_logistika.md")
    assert d.title.startswith("Коммерческое предложение") and "99,9%" in d.full_text
