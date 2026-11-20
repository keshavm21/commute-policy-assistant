"""The eval scorer is code too, so it gets tests: a wrong scorer would make every
reported number meaningless."""

import pytest

from eval.scoring import contains, percentile, reciprocal_rank, retrieval_hit, score_case

EXPECTED = {"status": "answered", "sections": ["4.2"], "must_contain_any": ["8:00 PM", "8 PM"],
            "must_not_contain": ["10:00 PM"]}


def actual(status="answered", answer="Escort after 8:00 PM.", cited=("4.2",),
           retrieved=(("zentrix", 2, "4.2"),)):
    return {"status": status, "answer": answer, "cited_sections": list(cited),
            "retrieved": [{"company": c, "version": v, "section": s} for c, v, s in retrieved]}


def test_correct_answer_passes():
    assert score_case(EXPECTED, actual(), "zentrix", 2) == []


@pytest.mark.parametrize("bad_actual, check", [
    (actual(retrieved=(("meridian", 1, "4.2"),)), "isolation"),
    (actual(retrieved=(("zentrix", 1, "4.2"),)), "freshness"),
    (actual(status="not_covered"), "status"),
    (actual(cited=("5.1",)), "citation"),
    (actual(answer="An escort is provided."), "fact"),
    (actual(answer="Escort after 8 PM, or 10:00 PM elsewhere."), "forbidden"),
])
def test_each_check_can_fail(bad_actual, check):
    failed = score_case(EXPECTED, bad_actual, "zentrix", 2)
    assert any(f.startswith(check) for f in failed)


def test_status_list_allows_either_outcome_but_forbidden_still_applies():
    expected = {"status": ["answered", "not_covered"], "must_not_contain": ["Rs 200"]}
    assert score_case(expected, actual(status="not_covered", answer="x"), "zentrix", 2) == []
    assert score_case(expected, actual(answer="Rs 200 fine"), "zentrix", 2) != []


def test_matching_is_case_insensitive():
    assert score_case(EXPECTED, actual(answer="escort after 8 pm"), "zentrix", 2) == []


def test_unicode_spaces_from_real_llms_still_match():
    # Seen in real output: narrow no-break space before PM.
    assert score_case(EXPECTED, actual(answer="Escort after 8:00 PM."), "zentrix", 2) == []
    # ...and forbidden values can't hide behind it either.
    failed = score_case(EXPECTED, actual(answer="8 PM, or 10:00 PM"), "zentrix", 2)
    assert any(f.startswith("forbidden") for f in failed)


@pytest.mark.parametrize("answer, phrase, expected", [
    ("Request it 24 hours ahead.", "4 hours", False),
    ("See 5.1 weekend pickups.", "1 week", False),
    ("Suspended for 1 week.", "1 week", True),
    ("Capped at Rs 1,200.", "200", False),
    ("Rs 200 is deducted.", "200", True),
    ("after 8 PM.", "8 PM", True),
])
def test_whole_phrase_matching(answer, phrase, expected):
    assert contains(answer, phrase) is expected


def test_rank_metrics():
    act = actual(retrieved=(("zentrix", 2, "1.1"), ("zentrix", 2, "4.2")))
    assert retrieval_hit(EXPECTED, act) is True
    assert reciprocal_rank(EXPECTED, act) == 0.5
    assert reciprocal_rank({"sections": []}, act) is None


def test_percentile_nearest_rank():
    values = list(range(1, 101))
    assert percentile(values, 95) == 95
    assert percentile(values, 50) == 50
    assert percentile([7.0], 95) == 7.0
