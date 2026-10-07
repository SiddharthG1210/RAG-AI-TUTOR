"""Tests for ``tutor.ingest.chunk``: gluing page breaks and cutting passages.

The paragraphs here are made by hand, so the tests need no PDF. Their texts
are just enough words to give each paragraph the size a test needs.
"""

import pytest

from tutor.ingest.chunk import (
    chunk_chapter,
    chunk_paragraphs,
    join_page_breaks,
    word_count,
)
from tutor.ingest.pdf import Chapter, Paragraph, Topic


def para(words: int, page: int = 50, topic: str = "1.3.1.2") -> Paragraph:
    """Build a paragraph of a given size: ``words`` words, the last ending a sentence.

    Args:
        words: How many words the paragraph has.
        page: The PDF page it sits on.
        topic: The topic it belongs to.

    Returns:
        The paragraph.
    """
    return Paragraph(
        topic, page, str(page - 40), "word " * (words - 1) + "end.", page, str(page - 40)
    )


def sizes(passages: list[list[Paragraph]]) -> list[list[int]]:
    """Describe passages by their paragraphs' sizes, e.g. ``[[123, 114], [147]]``.

    Args:
        passages: The passages to describe.

    Returns:
        Each passage as the list of its paragraphs' word counts.
    """
    return [[word_count([p]) for p in passage] for passage in passages]


# --- join_page_breaks ------------------------------------------------------------------


def test_join_glues_a_paragraph_split_by_a_page_break() -> None:
    """No full stop at the end, lowercase start: one paragraph, spanning both pages."""
    first = Paragraph("1.2.1", 45, "5", "We need the right", 45, "5")
    second = Paragraph("1.2.1", 46, "6", "data. If the data is wrong…", 46, "6")
    [glued] = join_page_breaks([first, second])
    assert glued.text == "We need the right data. If the data is wrong…"
    assert (glued.page_label, glued.end_page_label) == ("5", "6")


def test_join_drops_the_hyphen_of_a_word_split_across_pages() -> None:
    """A word split as "sub-" + "sequent" becomes "subsequent", with no space."""
    first = Paragraph("1.3.4", 58, "18", "Actions affect sub-", 58, "18")
    second = Paragraph("1.3.4", 59, "19", "sequent observations.", 59, "19")
    [glued] = join_page_breaks([first, second])
    assert glued.text == "Actions affect subsequent observations."


def test_join_keeps_separate_paragraphs_separate() -> None:
    """A finished sentence, or a capital letter next, means two real paragraphs."""
    finished = Paragraph("1.2.1", 45, "5", "It ends here.", 45, "5")
    unfinished = Paragraph("1.2.1", 45, "5", "It does not end", 45, "5")
    capital = Paragraph("1.2.1", 46, "6", "Generally, more data helps.", 46, "6")
    assert len(join_page_breaks([finished, capital])) == 2
    assert len(join_page_breaks([unfinished, capital])) == 2


# --- chunk_paragraphs ------------------------------------------------------------------


def test_chunk_packs_whole_paragraphs_up_to_the_maximum() -> None:
    """The Classification paragraphs group as ①② | ③④⑤ | ⑥⑦⑧, as worked out by hand."""
    paragraphs = [para(n) for n in (123, 114, 147, 66, 35, 194, 71, 73)]
    assert sizes(chunk_paragraphs(paragraphs)) == [[123, 114], [147, 66, 35], [194, 71, 73]]


def test_chunk_rebalances_a_short_last_passage() -> None:
    """A 64-word leftover borrows from the passage before until it reaches 120."""
    paragraphs = [para(n) for n in (100, 132, 21, 9, 17, 17, 32, 64)]
    assert sizes(chunk_paragraphs(paragraphs)) == [[100, 132, 21, 9], [17, 17, 32, 64]]


def test_chunk_keeps_a_small_topic_as_one_short_passage() -> None:
    """A topic under 120 words has nothing to borrow from, so it stays one passage."""
    assert sizes(chunk_paragraphs([para(50)])) == [[50]]


def test_chunk_gives_no_passages_for_an_empty_topic() -> None:
    """A topic with no prose has nothing to cut."""
    assert chunk_paragraphs([]) == []


def test_chunk_rejects_a_paragraph_over_the_maximum() -> None:
    """A single paragraph over 350 words stops the ingest, naming where it is."""
    with pytest.raises(ValueError, match="topic 1.3.1.2 on page 10 is 400 words"):
        chunk_paragraphs([para(400)])


# --- chunk_chapter ---------------------------------------------------------------------


def test_chunk_chapter_never_mixes_topics() -> None:
    """Each topic's paragraphs are chunked on their own, even when they're small."""
    chapter = Chapter(
        1,
        "Introduction",
        [Topic("1.2.4", "Optimization Algorithms", 4, 47, "7"), Topic("1.3", "Kinds", 2, 47, "7")],
        [para(90, topic="1.2.4"), para(50, topic="1.3")],
        {},
    )
    assert {number: sizes(p) for number, p in chunk_chapter(chapter).items()} == {
        "1.2.4": [[90]],
        "1.3": [[50]],
    }
