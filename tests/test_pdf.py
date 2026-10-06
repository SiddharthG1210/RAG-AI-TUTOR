"""Tests for ``tutor.ingest.pdf``: block text, labels, section numbers and topics.

CI has no copy of the book (``data/`` is git-ignored), so these tests never
open the PDF. They build small ``Block`` objects by hand, shaped like the
blocks chapter 1 produces, and check the deterministic logic on them.
Reading the real PDF is checked by running ``scripts/ingest.py``.
"""

import pytest

from tutor.ingest.pdf import (
    Block,
    Section,
    Span,
    TocEntry,
    classify_block,
    font_usage,
    heading_matches,
    number_sections,
    split_into_topics,
)

BODY = "TeXGyreTermesX-Regular"


def make_block(
    text: str,
    font: str = BODY,
    size: float = 10.0,
    x0: float = 136.8,
    y0: float = 300.0,
    page: int = 50,
) -> Block:
    """Build a one-line, one-span block. The defaults describe an ordinary paragraph.

    Args:
        text: The block's text.
        font: The span's font.
        size: The span's font size.
        x0: The block's left edge, in points.
        y0: The block's top edge, in points.
        page: The PDF page number.

    Returns:
        The block, with a printed page number 40 below the PDF page, as in d2l.
    """
    return Block(page, str(page - 40), x0, y0, ((Span(text, font, size),),))


# --- Block.text ----------------------------------------------------------------------


def test_text_drops_superscript_footnote_markers() -> None:
    """A raised footnote number inside a sentence isn't part of the text."""
    block = Block(
        49,
        "9",
        136.8,
        300.0,
        (
            (
                Span("the Netflix prize", BODY, 10.0),
                Span("19", "SFRM0700", 7.0, True),
                Span(".", BODY, 10.0),
            ),
        ),
    )
    assert block.text == "the Netflix prize."


def test_text_rejoins_hyphenated_words() -> None:
    """A typesetter's hyphen goes; a hyphen that's part of the word stays."""
    lines = (
        (Span("ten classes, corre-", BODY, 10.0),),
        (Span("sponding to Self-", BODY, 10.0),),
        (Span("Supervised   learning", BODY, 10.0),),
    )
    block = Block(50, "10", 136.8, 300.0, lines)
    assert block.text == "ten classes, corresponding to Self-Supervised learning"


def test_main_font_counts_characters() -> None:
    """A paragraph with one italic word is mainly the regular font."""
    block = Block(
        50,
        "10",
        136.8,
        300.0,
        ((Span("we call it ", BODY, 10.0), Span("binary", "TeXGyreTermesX-Italic", 10.0)),),
    )
    assert block.main_font == BODY


# --- classify_block -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("block", "label"),
    [
        # The running header, at the top of the page.
        (
            make_block("11 Kinds of Machine Learning Problems", "NimbusRomNo9L-Regu", y0=39),
            "header",
        ),
        # A footnote number in the margin.
        (make_block("20", "SFRM0700", 7.0, x0=95), "header"),
        # A figure caption, with the margin circle read as "t".
        (
            make_block("t Fig. 1.3.2 Death cap - do not eat!", "NimbusRomNo9L-Regu", 9.5, x0=83),
            "caption",
        ),
        (make_block("Table 1.5.1: label:tab_intro_decade", "SFRM1000"), "caption"),
        # Headings: larger than the 10 pt body text.
        (make_block("Classification", "TeXGyreTermesX-Bold", 10.9, x0=284), "heading"),
        (make_block("1.2.1 Data", "TeXGyreTermesX-Bold", 12.0, x0=288), "heading"),
        (make_block("Tom has dinner in Washington with Sally", "Inconsolata", 9.0), "code"),
        (make_block("= ∞", "txmiaX"), "math"),
        # A table: its own font, its title, or body text off the text column.
        (make_block("Decade Dataset Memory", "FreeSans", x0=142.3), "table"),
        (make_block(":Dataset vs. computer memory and computational power"), "table"),
        (make_block("1 MF (Intel 80186)", x0=352.1), "table"),
        # Prose: at the text column, or at the indent of a continued bullet.
        (make_block("In classification, we want our model to look at features."), "prose"),
        (make_block("has demonstrated superior scaling behavior", x0=156.7), "prose"),
    ],
)
def test_classify_block(block: Block, label: str) -> None:
    """Each kind of block found in chapter 1 gets its label.

    ``parametrize`` runs this one test once per (block, label) pair, and
    pytest reports each run separately.
    """
    assert classify_block(block) == label


def test_classify_block_rejects_unknown_fonts() -> None:
    """A font the rules don't know stops the ingest instead of being guessed."""
    with pytest.raises(ValueError, match="Unknown font 'Comic Sans'"):
        classify_block(make_block("Some text", "Comic Sans"))


# --- Section numbers ---------------------------------------------------------------


def test_number_sections_counts_each_level() -> None:
    """Numbers follow the bookmark levels, and deeper counters reset."""
    entries = [
        TocEntry(1, "Introduction", 41),
        TocEntry(2, "A Motivating Example", 42),
        TocEntry(2, "Key Components", 44),
        TocEntry(3, "Data", 44),
        TocEntry(3, "Models", 46),
        TocEntry(2, "Kinds of Machine Learning Problems", 47),
        TocEntry(3, "Supervised Learning", 47),
        TocEntry(4, "Regression", 48),
        TocEntry(4, "Classification", 49),
        TocEntry(2, "Summary", 69),
    ]
    sections = number_sections(entries, chapter=1)
    assert [s.number for s in sections] == [
        "1",
        "1.1",
        "1.2",
        "1.2.1",
        "1.2.2",
        "1.3",
        "1.3.1",
        "1.3.1.1",
        "1.3.1.2",
        "1.4",
    ]
    # Summary and Exercises are numbered, so later numbers stay right, but skipped.
    assert [s.title for s in sections if s.skipped] == ["Summary"]


def test_number_sections_rejects_a_skipped_level() -> None:
    """A level-4 bookmark straight under a level-2 one can't be numbered."""
    entries = [TocEntry(1, "Introduction", 41), TocEntry(2, "A", 42), TocEntry(4, "B", 43)]
    with pytest.raises(ValueError, match="level 4"):
        number_sections(entries, chapter=1)


def test_heading_matches_with_or_without_the_number() -> None:
    """Large headings print their number; the smallest print only the title."""
    assert heading_matches("1.2.1  Data", "1.2.1", "Data")
    assert heading_matches("Classification", "1.3.1.2", "Classification")
    assert not heading_matches("Tagging", "1.3.1.2", "Classification")


# --- Topics and paragraphs -----------------------------------------------------------

SECTIONS = [
    Section("1", "Introduction", 1, skipped=False),
    Section("1.1", "A Motivating Example", 2, skipped=False),
    Section("1.2", "Summary", 2, skipped=True),
]


def heading(text: str, page: int = 41) -> Block:
    """Build a heading block (bold, larger than the body text).

    Args:
        text: The heading's text.
        page: The PDF page number.

    Returns:
        The heading block.
    """
    return make_block(text, "TeXGyreTermesX-Bold", 15.9, x0=225, page=page)


def test_split_into_topics_keeps_prose_under_its_heading() -> None:
    """Prose goes to the topic above it; captions and Summary text are dropped."""
    blocks = [
        heading("1 Introduction", page=41),
        make_block("Until recently, nearly every program was a rigid set of rules.", page=41),
        heading("1.1 A Motivating Example", page=42),
        make_block("t Fig. 1.1.1 Identify a wake word.", "NimbusRomNo9L-Regu", x0=83, page=43),
        make_block("Imagine writing a program to respond to a wake word.", page=43),
        heading("1.2 Summary", page=69),
        make_block("Machine learning studies how computers learn from data.", page=69),
    ]
    topics, paragraphs, counts = split_into_topics(blocks, SECTIONS)

    assert [(t.number, t.title, t.page, t.page_label) for t in topics] == [
        ("1", "Introduction", 41, "1"),
        ("1.1", "A Motivating Example", 42, "2"),
    ]
    assert [(p.topic, p.page_label) for p in paragraphs] == [("1", "1"), ("1.1", "3")]
    assert counts == {"heading": 3, "prose": 3, "caption": 1}


def test_split_into_topics_rejects_a_heading_out_of_order() -> None:
    """A heading that isn't the next bookmark stops the ingest."""
    blocks = [heading("1 Introduction"), heading("1.2 Summary")]
    with pytest.raises(ValueError, match="doesn't match the next bookmark"):
        split_into_topics(blocks, SECTIONS)


def test_split_into_topics_rejects_a_missing_heading() -> None:
    """A bookmark whose heading never shows up stops the ingest."""
    blocks = [heading("1 Introduction"), heading("1.1 A Motivating Example")]
    with pytest.raises(ValueError, match="never found"):
        split_into_topics(blocks, SECTIONS)


def test_split_into_topics_rejects_prose_before_the_first_heading() -> None:
    """Text with no heading above it has no topic to belong to."""
    with pytest.raises(ValueError, match="before the chapter's first heading"):
        split_into_topics([make_block("Stray text.")], SECTIONS)


# --- Font listing --------------------------------------------------------------------


def test_font_usage_lists_most_used_first() -> None:
    """The body font, with the most characters, comes first."""
    blocks = [
        make_block("A long paragraph of ordinary text.", page=41),
        make_block("More ordinary text.", page=42),
        make_block("20", "SFRM0700", 7.0, x0=95, page=51),
    ]
    usage = font_usage(blocks)
    assert [(u.font, u.size, u.pages) for u in usage] == [
        (BODY, 10.0, [41, 42]),
        ("SFRM0700", 7.0, [51]),
    ]
