"""Russian-specific text plumbing: sentence split, grounding normalisation."""
from doccheck.grounding import check_one, score_quote
from doccheck.rag.chunking import chunk_section, split_sentences

OFFER = ("4.1. Клиент вправе отказаться от Сервиса и получить возврат полной суммы первой оплаты, если обратится "
         "в течение 14 календарных дней с даты этой оплаты. 4.2. По истечении 14 календарных дней при оплате на "
         "12 месяцев и более возвращается стоимость неиспользованных полных месяцев. «Бизнес» — пример. Ещё одно.")


def test_cyrillic_sentences_and_clause_numbers():
    s = split_sentences(OFFER)
    assert s[0].startswith("4.1. Клиент") and s[1].startswith("4.2. По истечении")
    assert s[2] == "«Бизнес» — пример." and s[3] == "Ещё одно."


def test_cyrillic_chunks_do_not_hard_split_words():
    text = " ".join(f"Предложение номер {i} о тарифах и сроках возврата." for i in range(80))
    chunks = chunk_section("Раздел", text, size=400, overlap=80)
    assert len(chunks) > 3 and all(c.endswith(".") for c in chunks)


def test_grounding_ignores_typography():
    chunk = 'Стоимость тарифа "Бизнес" - 12 990 ₽ в месяц. Её можно оплатить картой.'
    assert score_quote("Стоимость тарифа «Бизнес» — 12 990 ₽ в месяц", chunk) == 100
    assert score_quote("Ее можно оплатить картой", chunk) == 100


def test_grounding_rejects_changed_number():
    chunk = "Возврат осуществляется в течение 10 рабочих дней с даты получения заявления."
    lookup = {"o#4": chunk}.get
    assert check_one("o#4", "в течение 10 рабочих дней с даты получения заявления", lookup).ok
    assert check_one("o#4", "в течение 30 календарных дней с даты оплаты", lookup).score < 90
