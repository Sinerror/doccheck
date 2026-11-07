from doccheck.rag.chunking import CHUNK_SIZE, chunk_document, chunk_section, split_sentences


def _text(n: int) -> str:
    return " ".join(f"Sentence number {i} talks about attention and recurrence in some detail." for i in range(n))


def test_split_sentences_keeps_abbreviation_like_numbers():
    s = split_sentences("We get 28.4 BLEU. Then we stop!  New one? Yes.")
    assert s == ["We get 28.4 BLEU.", "Then we stop!", "New one?", "Yes."]


def test_chunks_respect_size_and_sentence_boundaries():
    text = _text(200)
    chunks = chunk_section("Results", text)
    assert len(chunks) > 3
    for c in chunks:
        assert len(c) <= CHUNK_SIZE
        assert c.startswith("Sentence number") and c.endswith(".")  # never cut mid-sentence


def test_overlap_between_consecutive_chunks():
    chunks = chunk_section("S", _text(200))
    for a, b in zip(chunks, chunks[1:]):
        first_sentence_of_b = b.split(". ")[0] + "."
        assert first_sentence_of_b in a  # b starts with sentences carried over from a
        tail = a[-400:]
        assert sum(1 for s in split_sentences(b)[:6] if s in tail) >= 2  # ~300 chars of overlap


def test_short_section_single_chunk_and_sections_not_merged():
    chunks = chunk_document([("Intro", "Short intro. Two sentences."), ("Method", _text(60))])
    assert chunks[0].section == "Intro" and chunks[0].text == "Short intro. Two sentences."
    assert all(c.section == "Method" for c in chunks[1:])
    assert [c.chunk_idx for c in chunks] == list(range(len(chunks)))


def test_overlong_sentence_is_hard_split():
    long = "word " * 1200  # one 6000-char "sentence"
    chunks = chunk_section("X", long.strip() + ".")
    assert len(chunks) >= 3
    assert all(len(c) <= CHUNK_SIZE for c in chunks)


def test_empty_text():
    assert chunk_section("X", "   ") == []
