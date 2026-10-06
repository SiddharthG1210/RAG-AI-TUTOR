"""Load one chapter of the book into the tutor's database.

Run from the repo root:

    uv run python scripts/ingest.py --chapter 1          # save the topics, print a summary
    uv run python scripts/ingest.py --chapter 1 --show   # print the kept text; saves nothing
    uv run python scripts/ingest.py --chapter 1 --fonts  # list the fonts; saves nothing

``--show`` is the check that only prose is kept: it prints every topic
heading and every kept paragraph with its page, so captions or headers that
slip through are easy to spot. ``--fonts`` lists every font and size in the
chapter, which is what ``classify_block()``'s rules are decided from.

No LLM runs here, so ingesting uses none of the free quota. In the M1
concept "Chunking" this command also cuts the paragraphs into passages and
saves them.
"""

import argparse
from pathlib import Path

from tutor.db import connect, insert_topics
from tutor.ingest.pdf import Chapter, extract_chapter, font_usage, read_chapter


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
    parser.add_argument("--show", action="store_true", help="print the kept text; save nothing")
    parser.add_argument("--fonts", action="store_true", help="list the fonts; save nothing")
    return parser.parse_args()


def print_fonts(pdf: Path, chapter: int) -> None:
    """Print every font and size in the chapter, most used first.

    Args:
        pdf: The book PDF.
        chapter: The chapter number.
    """
    title, _, blocks = read_chapter(pdf, chapter)
    print(f"Fonts in chapter {chapter}, {title} (characters, font, size, pages, example)\n")
    for usage in font_usage(blocks):
        pages = f"{usage.pages[0]}-{usage.pages[-1]}" if len(usage.pages) > 1 else usage.pages[0]
        print(f"{usage.chars:6}  {usage.font:26} {usage.size:5.1f}  p.{pages:<8} {usage.example!r}")


def print_text(chapter: Chapter) -> None:
    """Print each topic heading followed by its kept paragraphs.

    Args:
        chapter: The extracted chapter.
    """
    for topic in chapter.topics:
        print(f"\n## {topic.number} {topic.title}  (printed page {topic.page_label})\n")
        for paragraph in chapter.paragraphs:
            if paragraph.topic == topic.number:
                print(f"[p. {paragraph.page_label}] {paragraph.text}\n")


def print_summary(chapter: Chapter) -> None:
    """Print one line per topic, then how many blocks were kept and dropped.

    Args:
        chapter: The extracted chapter.
    """
    print(f"Chapter {chapter.number}, {chapter.title}: {len(chapter.topics)} topics\n")
    print(f"{'topic':10} {'title':45} {'page':>5} {'paras':>6} {'words':>6}")
    for topic in chapter.topics:
        texts = [p.text for p in chapter.paragraphs if p.topic == topic.number]
        words = sum(len(text.split()) for text in texts)
        print(f"{topic.number:10} {topic.title:45} {topic.page_label:>5} {len(texts):6} {words:6}")
    counts = ", ".join(f"{n} {label}" for label, n in sorted(chapter.label_counts.items()))
    print(f"\nBlocks by label: {counts}")
    print("Kept: prose (as paragraphs) and heading (as topics). Everything else is dropped.")


def main() -> None:
    """Run the command: list fonts, show the kept text, or save the topics."""
    args = parse_args()
    if args.fonts:
        print_fonts(args.pdf, args.chapter)
        return
    chapter = extract_chapter(args.pdf, args.chapter)
    if args.show:
        print_text(chapter)
        return
    conn = connect(args.db)
    insert_topics(conn, chapter.topics)
    conn.close()
    print_summary(chapter)
    print(f"Saved {len(chapter.topics)} topics to {args.db}.")


if __name__ == "__main__":
    main()
