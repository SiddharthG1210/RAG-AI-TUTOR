"""Turn one chapter of the d2l PDF into prose with page numbers, grouped into topics.

M1 concept "Turning the PDF into prose with page numbers". The pipeline:

    d2l-en.pdf
       │  read_blocks() ......... PyMuPDF: every text block, with its fonts and position
       ▼
    Blocks
       │  classify_block() ...... prose / heading / header / caption / code / math / table
       ▼
    split_into_topics() ......... headings matched, in order, to the PDF's bookmarks
       │
       ▼
    Chapter: topics (number, title, level, page) + paragraphs (topic, page, text)

Only blocks labelled "prose" are kept as text. Headings mark where each topic
starts. Everything else is dropped, so retrieval and the LLM never see page
headers, captions, code, formulas or tables (PLAN.md › Scope: the tutor only
asks about ideas explained in prose).

Two page numbers are kept for every topic and paragraph:

- ``page``: the PDF file's page, counted from 1 as a PDF viewer counts.
- ``page_label``: the number printed on the page itself. The book's front
  matter comes first, so printed page 10 is the PDF's page 50.

The section numbers ("1.3.1.2") don't appear in the bookmarks, which hold
titles only, and the smallest headings don't print them either. So
``number_sections()`` works them out from the bookmark levels.
"""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# PyMuPDF reads PDFs. It's imported as "pymupdf" (older code uses the alias "fitz").
import pymupdf

# --- What a page is made of --------------------------------------------------------
# PyMuPDF's page.get_text("dict") returns each page as nested dicts:
#     page ─► blocks ─► lines ─► spans
# A block is roughly a paragraph, a heading or a caption. A span is a run of
# text in one font, size and style. The dataclasses below keep only the parts
# this module uses, so the rest of the code (and the tests) never touch
# PyMuPDF's raw dicts.


@dataclass(frozen=True)
class Span:
    """A run of text in one font, size and style.

    Attributes:
        text: The characters in the run, spaces included.
        font: The font's name, e.g. "TeXGyreTermesX-Regular".
        size: The font size in points.
        superscript: True for raised text, such as a footnote marker.
    """

    text: str
    font: str
    size: float
    superscript: bool = False


@dataclass(frozen=True)
class Block:
    """One text block on a page: its position, its lines and their spans.

    ``frozen=True`` makes instances read-only, so a block can't be changed by
    accident after it's read from the PDF.

    Attributes:
        page: The PDF file's page number, counted from 1.
        page_label: The page number printed in the book.
        x0: Distance in points from the page's left edge to the block's left side.
        y0: Distance in points from the page's top edge to the block's top.
        lines: The block's lines, each a tuple of spans.
    """

    page: int
    page_label: str
    x0: float
    y0: float
    lines: tuple[tuple[Span, ...], ...]

    @property
    def spans(self) -> list[Span]:
        """All spans in the block, line by line."""
        return [span for line in self.lines for span in line]

    @property
    def text(self) -> str:
        """The block's text as one clean string.

        Three clean-ups happen here:

        - Superscript spans are dropped. In d2l they're footnote markers, such
          as the "19" in "the Netflix prize 19".
        - A word split across two lines is rejoined. If the next line starts
          with a lowercase letter, the hyphen was added by the typesetter and
          goes ("corre-" + "sponding" → "corresponding"). Otherwise it's part
          of the word and stays ("Self-" + "Supervised" → "Self-Supervised").
        - Runs of whitespace become one space. Justified text in the PDF has
          wide gaps that come out as several spaces.
        """
        text = ""
        for line in self.lines:
            line_text = "".join(span.text for span in line if not span.superscript).strip()
            if not line_text:
                continue
            if text.endswith("-"):
                # A word split across lines: drop the hyphen only if it was
                # inserted by hyphenation (the word continues in lowercase).
                text = (text[:-1] if line_text[0].islower() else text) + line_text
            else:
                text = f"{text} {line_text}" if text else line_text
        return re.sub(r"\s+", " ", text).strip()

    @property
    def main_font(self) -> str:
        """The font that covers the most characters in the block.

        A paragraph with a few italic words is still mainly the regular font,
        so judging by the main font ignores those small mixes.
        """
        counts: Counter[str] = Counter()
        for span in self.spans:
            counts[span.font] += len(span.text.strip())
        return counts.most_common(1)[0][0]

    @property
    def max_size(self) -> float:
        """The largest font size used anywhere in the block, in points."""
        return max(span.size for span in self.spans)


def read_blocks(doc: pymupdf.Document, first_page: int, last_page: int) -> list[Block]:
    """Read every text block on a range of pages, top to bottom.

    Args:
        doc: The open PDF.
        first_page: The first page to read, counted from 1.
        last_page: The last page to read (included), counted from 1.

    Returns:
        The text blocks in reading order: page by page, and top to bottom
        within a page. Image blocks and blocks with no visible text are skipped.
    """
    blocks: list[Block] = []
    for page_number in range(first_page, last_page + 1):
        page = doc[page_number - 1]  # PyMuPDF counts pages from 0
        # get_label() is the number printed on the page (from the PDF's page labels).
        page_label = page.get_label()
        page_blocks: list[Block] = []
        for raw in page.get_text("dict")["blocks"]:
            if raw["type"] != 0:  # 0 = text, 1 = image
                continue
            lines = tuple(
                tuple(
                    Span(
                        text=span["text"],
                        font=span["font"],
                        size=span["size"],
                        # "flags" is a bit field; one bit marks superscript text.
                        superscript=bool(span["flags"] & pymupdf.TEXT_FONT_SUPERSCRIPT),
                    )
                    for span in line["spans"]
                )
                for line in raw["lines"]
            )
            block = Block(page_number, page_label, raw["bbox"][0], raw["bbox"][1], lines)
            if block.text:
                page_blocks.append(block)
        # PyMuPDF returns blocks in the order the PDF stores them, which isn't
        # always reading order: a figure caption can come before the paragraph
        # above it. d2l has one column of text, so sorting by the top edge
        # gives reading order.
        page_blocks.sort(key=lambda b: (b.y0, b.x0))
        blocks.extend(page_blocks)
    return blocks


# --- Labelling blocks -------------------------------------------------------------
# The rules below come from the fonts and positions found in chapter 1
# (`scripts/ingest.py --chapter 1 --fonts` lists them). They're specific to
# the d2l PDF's layout. A later chapter with a font they don't know stops the
# ingest with an error, so the rules are extended on purpose, never by accident.

BlockLabel = Literal["prose", "heading", "header", "caption", "code", "math", "table"]
"""What a block is. Only "prose" is kept as text; "heading" marks a topic's start."""

HEADER_BOTTOM = 60.0
"""Running headers sit above this line, in points from the top of the page."""

BODY_SIZE = 10.0
"""The body text's font size, in points. Headings are larger."""

CAPTION_START = re.compile(r"(t )?(Fig\.|Table) ")
"""Captions start with "Fig." or "Table". d2l draws a small circle in the
margin next to each caption, which PyMuPDF reads as the letter "t"."""

MATH_FONTS = {"NewTXMI", "txmiaX", "txsys"}
"""Fonts used only for mathematical symbols."""

TEXT_LEFT_EDGES = (136.8, 156.7)
"""Where prose blocks start, in points from the page's left edge: the text
column, and the indent of a bulleted item continued from the previous page."""


def _starts_at_text_edge(block: Block) -> bool:
    """Whether the block starts at one of the left edges prose starts at.

    Table cells start elsewhere, which is how a table in the body font is told
    apart from a paragraph. The 1.5-point tolerance absorbs rounding.
    """
    return any(abs(block.x0 - edge) < 1.5 for edge in TEXT_LEFT_EDGES)


def classify_block(block: Block) -> BlockLabel:
    """Label one text block by its fonts, its position and how its text starts.

    The checks run in order, and the first one that matches wins:

    1. header: in the top margin (the running header), or only digits (a page
       number, or a footnote number in the margin).
    2. caption: starts with "Fig." or "Table".
    3. heading: any text larger than the body text. Section titles are the
       only large text left once captions are out.
    4. code: mainly the monospace font, Inconsolata.
    5. math: mainly a math-symbol font.
    6. table: the table font (FreeSans), a table title (d2l renders them as
       ":Title"), or body-font text that doesn't start where paragraphs start.
    7. prose: mainly the body font, TeXGyreTermesX.

    Args:
        block: The block to label.

    Returns:
        The block's label.

    Raises:
        ValueError: If the block's main font is one these rules don't know.
            That means a new kind of text, and the rules need a decision.
    """
    text = block.text
    if block.y0 < HEADER_BOTTOM or text.isdigit():
        return "header"
    if CAPTION_START.match(text):
        return "caption"
    if block.max_size > BODY_SIZE + 0.5:
        return "heading"
    font = block.main_font
    if font.startswith("Inconsolata"):
        return "code"
    if font in MATH_FONTS:
        return "math"
    if font == "FreeSans" or text.startswith(":") or not _starts_at_text_edge(block):
        return "table"
    if font.startswith("TeXGyreTermesX"):
        return "prose"
    raise ValueError(
        f"Unknown font {font!r} on PDF page {block.page}: {text[:60]!r}. "
        "Decide its label in classify_block()."
    )


# --- Topics from the bookmarks ---------------------------------------------------

SKIPPED_TITLES = {"Summary", "Exercises"}
"""Sections whose text is never quizzed on (PLAN.md › Decisions › Topic)."""


@dataclass(frozen=True)
class TocEntry:
    """One bookmark from the PDF's table of contents.

    Attributes:
        level: The depth: 1 for a chapter, 2 for a section, and so on.
        title: The heading's title, without its number.
        page: The PDF page the bookmark points to, counted from 1.
    """

    level: int
    title: str
    page: int


@dataclass(frozen=True)
class Section:
    """A bookmark with its section number worked out.

    Attributes:
        number: The section number, e.g. "1.3.1.2".
        title: The heading's title, e.g. "Classification".
        level: The depth: 1 for a chapter, 2 for a section, and so on.
        skipped: True for Summary and Exercises, whose text isn't kept.
    """

    number: str
    title: str
    level: int
    skipped: bool


@dataclass(frozen=True)
class Topic:
    """A section the tutor quizzes on: one row of the topics table.

    Attributes:
        number: The section number, e.g. "1.3.1.2".
        title: The heading's title, e.g. "Classification".
        level: The depth: 1 for a chapter, 2 for a section, and so on.
        page: The PDF page where the heading is, counted from 1.
        page_label: The printed page number where the heading is.
    """

    number: str
    title: str
    level: int
    page: int
    page_label: str


@dataclass(frozen=True)
class Paragraph:
    """One paragraph of prose, with the topic and the pages it came from.

    A paragraph read from the PDF sits on one page, so its start and end pages
    are the same. A paragraph glued back together across a page break (the M1
    concept "Chunking") starts on one page and ends on the next, which is why
    both ends are kept: a passage's page range runs from its first paragraph's
    start page to its last paragraph's end page.

    Attributes:
        topic: The number of the topic it belongs to, e.g. "1.3.1.2".
        page: The PDF page it starts on, counted from 1.
        page_label: The printed page number it starts on.
        text: The cleaned text.
        end_page: The PDF page it ends on, counted from 1.
        end_page_label: The printed page number it ends on.
    """

    topic: str
    page: int
    page_label: str
    text: str
    end_page: int
    end_page_label: str


@dataclass(frozen=True)
class Chapter:
    """Everything ingesting one chapter produces.

    Attributes:
        number: The chapter number.
        title: The chapter title.
        topics: The topics, in book order (Summary and Exercises left out).
        paragraphs: The kept prose, in book order.
        label_counts: How many blocks got each label, kept or dropped.
    """

    number: int
    title: str
    topics: list[Topic]
    paragraphs: list[Paragraph]
    label_counts: dict[str, int]


def number_sections(entries: list[TocEntry], chapter: int) -> list[Section]:
    """Work out each bookmark's section number from the bookmark levels.

    The numbers are counters, one per level. A bookmark adds 1 to the counter
    at its level and resets every deeper counter: after "1.2.4", a level-2
    bookmark becomes "1.3" and the next level-3 one becomes "1.3.1".

    Args:
        entries: One chapter's bookmarks in book order, starting with the
            chapter itself (level 1).
        chapter: The chapter's number.

    Returns:
        The sections, in the same order as ``entries``.

    Raises:
        ValueError: If the first bookmark isn't a chapter, or a bookmark skips
            a level (a level-4 bookmark straight under a level-2 one).
    """
    if not entries or entries[0].level != 1:
        raise ValueError("A chapter's bookmarks must start with the chapter itself (level 1).")
    counters = [chapter]  # counters[i] is the count at level i + 1
    sections = [Section(str(chapter), entries[0].title, 1, entries[0].title in SKIPPED_TITLES)]
    for entry in entries[1:]:
        if entry.level < 2 or entry.level > len(counters) + 1:
            raise ValueError(
                f"Bookmark {entry.title!r} is at level {entry.level}, "
                f"which doesn't follow from the level-{len(counters)} bookmark before it."
            )
        # Keep the counters above this level, then count one more at this level.
        counters = counters[: entry.level]
        if len(counters) < entry.level:
            counters.append(0)  # the first bookmark at a new, deeper level
        counters[-1] += 1
        number = ".".join(str(count) for count in counters)
        sections.append(Section(number, entry.title, entry.level, entry.title in SKIPPED_TITLES))
    return sections


def _normalize(text: str) -> str:
    """Lowercase the text and collapse runs of whitespace, for comparing titles."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def heading_matches(text: str, number: str, title: str) -> bool:
    """Whether a heading block's text is the heading for this section.

    Larger headings print their number ("1.2.1 Data"); the smallest ones
    print only the title ("Classification"). Both forms match.

    Args:
        text: The heading block's text.
        number: The section number worked out from the bookmarks.
        title: The bookmark's title.

    Returns:
        True if the text is the title, with or without the number in front.
    """
    return _normalize(text) in {_normalize(title), _normalize(f"{number} {title}")}


def split_into_topics(
    blocks: list[Block], sections: list[Section]
) -> tuple[list[Topic], list[Paragraph], Counter[str]]:
    """Walk the blocks in order, start a new topic at each heading, and keep the prose.

    Each heading must be the next bookmark, in book order. That check ties
    every heading to its bookmark (and so to its section number), and stops
    the ingest if a heading is missing, extra or out of order.

    Args:
        blocks: The chapter's blocks in reading order.
        sections: The chapter's sections in book order, from ``number_sections()``.

    Returns:
        The topics (Summary and Exercises left out), the paragraphs, and how
        many blocks got each label.

    Raises:
        ValueError: If a heading doesn't match the next bookmark, prose comes
            before the first heading, or a bookmark's heading is never found.
    """
    remaining = iter(sections)
    current: Section | None = None
    topics: list[Topic] = []
    paragraphs: list[Paragraph] = []
    label_counts: Counter[str] = Counter()

    for block in blocks:
        label = classify_block(block)
        label_counts[label] += 1
        if label == "heading":
            current = next(remaining, None)
            if current is None or not heading_matches(block.text, current.number, current.title):
                expected = f"{current.number} {current.title}" if current else "no more headings"
                raise ValueError(
                    f"Heading {block.text!r} on PDF page {block.page} doesn't match "
                    f"the next bookmark ({expected})."
                )
            if not current.skipped:
                topics.append(
                    Topic(
                        current.number, current.title, current.level, block.page, block.page_label
                    )
                )
        elif label == "prose":
            if current is None:
                raise ValueError(
                    f"Prose before the chapter's first heading, on PDF page {block.page}."
                )
            if not current.skipped:
                paragraphs.append(
                    # A block sits on one page, so it starts and ends on the same page.
                    Paragraph(
                        current.number,
                        block.page,
                        block.page_label,
                        block.text,
                        end_page=block.page,
                        end_page_label=block.page_label,
                    )
                )

    missing = [f"{section.number} {section.title}" for section in remaining]
    if missing:
        raise ValueError(f"These bookmarks' headings were never found: {missing}")
    return topics, paragraphs, label_counts


# --- Putting it together ------------------------------------------------------------


def _find_chapter(doc: pymupdf.Document, toc: list[TocEntry], chapter: int) -> int:
    """Find the bookmark of a numbered chapter.

    The front matter (Preface, Installation, Notation) has level-1 bookmarks
    too, so chapter 1 isn't the first one. A chapter is found by its heading
    instead: the block "1 Introduction" on the bookmark's page. The number
    must be there, because the front matter's headings ("Preface") have none.

    Args:
        doc: The open PDF.
        toc: Every bookmark in the PDF.
        chapter: The chapter's number.

    Returns:
        The chapter's index in ``toc``.

    Raises:
        ValueError: If no chapter has that number.
    """
    for index, entry in enumerate(toc):
        if entry.level != 1:
            continue
        for block in read_blocks(doc, entry.page, entry.page):
            if _normalize(block.text) == _normalize(f"{chapter} {entry.title}"):
                return index
    raise ValueError(f"No chapter {chapter} found in the PDF's bookmarks.")


def read_chapter(pdf_path: Path, chapter: int) -> tuple[str, list[Section], list[Block]]:
    """Read one chapter's title, numbered sections and text blocks, without labelling them.

    Args:
        pdf_path: Path to the PDF.
        chapter: The chapter's number, e.g. 1.

    Returns:
        The chapter's title, its sections in book order, and its blocks in
        reading order.

    Raises:
        FileNotFoundError: If the PDF isn't there.
        ValueError: If the chapter can't be found, or its bookmarks skip a level.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"No PDF at {pdf_path}. Download it into data/ (see README.md).")
    # "with" closes the PDF file when the block ends, even if an error is raised.
    with pymupdf.open(pdf_path) as doc:
        # get_toc() returns the bookmarks as [level, title, page] lists.
        toc = [TocEntry(level, title.strip(), page) for level, title, page in doc.get_toc()]
        start = _find_chapter(doc, toc, chapter)
        # The chapter runs until the next level-1 bookmark, or to the end of the book.
        end = next((i for i in range(start + 1, len(toc)) if toc[i].level == 1), len(toc))
        last_page = toc[end].page - 1 if end < len(toc) else doc.page_count
        sections = number_sections(toc[start:end], chapter)
        blocks = read_blocks(doc, toc[start].page, last_page)
    return toc[start].title, sections, blocks


def extract_chapter(pdf_path: Path, chapter: int) -> Chapter:
    """Read one chapter of the PDF and keep its topics and prose.

    Args:
        pdf_path: Path to the PDF.
        chapter: The chapter's number, e.g. 1.

    Returns:
        The chapter's topics and paragraphs.

    Raises:
        FileNotFoundError: If the PDF isn't there.
        ValueError: If the chapter can't be found or doesn't match its
            bookmarks, or a block has an unknown font.
    """
    title, sections, blocks = read_chapter(pdf_path, chapter)
    topics, paragraphs, label_counts = split_into_topics(blocks, sections)
    return Chapter(chapter, title, topics, paragraphs, dict(label_counts))


@dataclass(frozen=True)
class FontUsage:
    """How one font and size is used across a chapter.

    Attributes:
        font: The font's name.
        size: The font size in points, rounded to 0.1.
        chars: How many characters use it.
        pages: The PDF pages it appears on.
        example: The start of one span that uses it.
    """

    font: str
    size: float
    chars: int
    pages: list[int]
    example: str


def font_usage(blocks: list[Block]) -> list[FontUsage]:
    """List every font and size used in the blocks, most used first.

    This is what ``classify_block()``'s rules are decided from: the body text
    is the font with by far the most characters, and every other font is a
    candidate for something to drop.

    Args:
        blocks: The blocks to survey, e.g. one chapter's.

    Returns:
        One entry per (font, size) pair, sorted by character count.
    """
    chars: Counter[tuple[str, float]] = Counter()
    pages: dict[tuple[str, float], set[int]] = {}
    examples: dict[tuple[str, float], str] = {}
    for block in blocks:
        for span in block.spans:
            key = (span.font, round(span.size, 1))
            chars[key] += len(span.text.strip())
            pages.setdefault(key, set()).add(block.page)
            if span.text.strip() and key not in examples:
                examples[key] = span.text.strip()[:40]
    return [
        FontUsage(font, size, count, sorted(pages[(font, size)]), examples.get((font, size), ""))
        for (font, size), count in chars.most_common()
    ]
