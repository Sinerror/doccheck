from doccheck.grounding import check_all, check_one, grounding_rate, score_quote

CHUNK = ("On the WMT 2014 English-to-German translation task, the big transformer model outperforms the best "
         "previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new "
         "state-of-the-art BLEU score of 28.4.")
LOOKUP = {"p#1": CHUNK}.get


def test_verbatim_quote_is_grounded():
    r = check_one("p#1", "outperforms the best previously reported models (including ensembles)", LOOKUP)
    assert r.ok and r.exists and r.score == 100


def test_whitespace_and_typographic_quotes_are_normalised():
    assert score_quote("by  more than\n2.0 BLEU", CHUNK) == 100
    assert check_one("p#1", "state‑of‑the‑art".replace("‑", "-"), LOOKUP).ok


def test_paraphrase_is_rejected():
    r = check_one("p#1", "the large model beats every earlier system by a wide margin of over three BLEU", LOOKUP)
    assert not r.ok and r.exists and r.score < 90


def test_wrong_number_is_rejected():
    # a tampered number inside an otherwise verbatim quote must not pass silently at short length
    r = check_one("p#1", "a new state-of-the-art BLEU score of 31.9", LOOKUP)
    assert r.score < 100


def test_unknown_chunk_id_is_ungrounded():
    r = check_one("p#999", "anything", LOOKUP)
    assert not r.ok and not r.exists and r.score == 0


def test_rate():
    rs = check_all([("p#1", "BLEU score of 28.4"), ("p#2", "x"), ("p#1", "totally invented sentence here")], LOOKUP)
    assert [r.ok for r in rs] == [True, False, False]
    assert grounding_rate(rs) == round(1 / 3, 4)
    assert grounding_rate([]) is None
