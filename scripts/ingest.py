"""Load one chapter of the book into the tutor's database.

Run from the repo root:

    uv run python scripts/ingest.py --chapter 1          # save topics and passages, print a summary
    uv run python scripts/ingest.py --chapter 1 --show   # print the passages; saves nothing
    uv run python scripts/ingest.py --chapter 1 --fonts  # list the fonts; saves nothing

The steps: the PDF step (``extract_chapter``) keeps the prose, grouped into
topics; chunking (``chunk_chapter``) glues page-split paragraphs and cuts
each topic into passages of 120–350 words; ``save_chapter`` stores the topics
and passages in one transaction.

``--show`` prints every passage with its pages, so captions or headers that
slip through, or passages that start mid-sentence, are easy to spot.
``--fonts`` lists every font and size in the chapter, which is what
``classify_block()``'s rules are decided from.

No LLM runs here, so ingesting uses none of the free quota.
"""

import argparse
import sys
from pathlib import Path

from tutor.db import connect, save_chapter
from tutor.ingest.chunk import MAX_WORDS, MIN_WORDS, chunk_chapter, word_count
from tutor.ingest.pdf import Chapter, Paragraph, extract_chapter, font_usage, read_chapter


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    ``argparse`` (built into Python) turns ``--chapter 1`` into
    ``args.chapter == 1``, checks the types, and writes ``--help`` for free.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(description="Load one chapter of the book.")
    parser.add_argument("--chapter", type=int, required=True, help="the chapter number, e.g. 1")
    parser.add_argument("--pdf", type=Path, default=Path("data/d2l-en.pdf"), help="the book PDF")
    parser.add_argument("--db", type=Path, default=Path("data/tutor.db"), help="the database")
    # action="store_true" makes a flag: present means True, absent means False.
    parser.add_argument("--show", action="store_true", help="print the passages; save nothing")
    parser.add_argument("--fonts", action="store_true", help="list the fonts; save nothing")
    return parser.parse_args()


def pages(passage: list[Paragraph]) -> str:
    """Format a passage's printed page range, e.g. "9–10", or "10" for one page.

    Args:
        passage: The passage's paragraphs.

    Returns:
        The range from its first paragraph's start page to its last paragraph's end page.
    """
    first, last = passage[0].page_label, passage[-1].end_page_label
    return first if first == last else f"{first}–{last}"


def print_fonts(pdf: Path, chapter: int) -> None:
    """Print every font and size in the chapter, most used first.

    Args:
        pdf: The book PDF.
        chapter: The chapter number.
    """
    title, _, blocks = read_chapter(pdf, chapter)
    print(f"Fonts in chapter {chapter}, {title} (characters, font, size, pages, example)\n")
    for usage in font_usage(blocks):
        span = f"{usage.pages[0]}-{usage.pages[-1]}" if len(usage.pages) > 1 else usage.pages[0]
        print(f"{usage.chars:6}  {usage.font:26} {usage.size:5.1f}  p.{span:<8} {usage.example!r}")


def print_text(chapter: Chapter, passages: dict[str, list[list[Paragraph]]]) -> None:
    """Print each topic heading followed by its passages, with their pages and sizes.

    Args:
        chapter: The extracted chapter.
        passages: Each topic's passages, from ``chunk_chapter()``.
    """
    for topic in chapter.topics:
        print(f"\n## {topic.number} {topic.title}\n")
        for position, passage in enumerate(passages[topic.number], start=1):
            text = "\n\n".join(paragraph.text for paragraph in passage)
            print(f"[passage {position}, p. {pages(passage)}, {word_count(passage)} words]")
            print(f"{text}\n")


def print_summary(chapter: Chapter, passages: dict[str, list[list[Paragraph]]]) -> None:
    """Print one line per topic with its passage sizes, then any short passages.

    Args:
        chapter: The extracted chapter.
        passages: Each topic's passages, from ``chunk_chapter()``.
    """
    total = sum(len(topic_passages) for topic_passages in passages.values())
    heading = f"Chapter {chapter.number}, {chapter.title}"
    print(f"{heading}: {len(chapter.topics)} topics, {total} passages\n")
    print(f"{'topic':10} {'title':45} {'passage sizes (words)'}")
    short = []
    for topic in chapter.topics:
        sizes = [word_count(passage) for passage in passages[topic.number]]
        print(f"{topic.number:10} {topic.title:45} {'  '.join(str(n) for n in sizes)}")
        short += [f"{topic.number} ({n} words)" for n in sizes if n < MIN_WORDS]
    # Rule 4: a topic under MIN_WORDS stays one short passage. List them so a
    # weak question from one is no surprise.
    print(f"\nPassages under {MIN_WORDS} words: {', '.join(short) if short else 'none'}")
    print(f"Every passage is at most {MAX_WORDS} words.")
    counts = ", ".join(f"{n} {label}" for label, n in sorted(chapter.label_counts.items()))
    print(f"Blocks by label: {counts}. Only prose and headings are kept.")


def main() -> None:
    """Run the command: list fonts, show the passages, or save the chapter."""
    # Windows terminals sometimes can't print "–" in their default encoding; UTF-8 can.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.fonts:
        print_fonts(args.pdf, args.chapter)
        return
    chapter = extract_chapter(args.pdf, args.chapter)
    passages = chunk_chapter(chapter)
    if args.show:
        print_text(chapter, passages)
        return
    conn = connect(args.db)
    save_chapter(conn, chapter.topics, passages)
    conn.close()
    print_summary(chapter, passages)
    print(f"\nSaved to {args.db}.")


if __name__ == "__main__":
    main()
