"""Load one chapter of the book into the tutor's database.

Run from the repo root:

    uv run python scripts/ingest.py --chapter 1          # save and embed a chapter, print a summary
    uv run python scripts/ingest.py --chapter 1 --show   # print the passages; saves nothing
    uv run python scripts/ingest.py --chapter 1 --fonts  # list the fonts; saves nothing
    uv run python scripts/ingest.py --embed              # embed every saved passage again

The steps: the PDF step (``extract_chapter``) keeps the prose, grouped into
topics; chunking (``chunk_chapter``) glues page-split paragraphs and cuts
each topic into passages of 120–350 words; ``save_chapter`` stores the topics
and passages in one transaction; then every new passage is embedded with bge
and its vector saved in the ``embeddings`` table.

``--embed`` reads no PDF. It embeds every saved passage again, which is needed
after changing the header in ``text_to_embed()`` or the embedding model. The
passages' text never changes, only their vectors.

``--show`` prints every passage with its pages, so captions or headers that
slip through, or passages that start mid-sentence, are easy to spot.
``--fonts`` lists every font and size in the chapter, which is what
``classify_block()``'s rules are decided from.

No LLM runs here, so ingesting uses none of the free quota.
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from tutor.db import connect, list_passages, save_chapter
from tutor.ingest.chunk import MAX_WORDS, MIN_WORDS, chunk_chapter, word_count
from tutor.ingest.pdf import Chapter, Paragraph, extract_chapter, font_usage, read_chapter
from tutor.retrieval.embed import (
    MAX_TOKENS,
    MODEL_NAME,
    count_tokens,
    embed_passages,
    text_to_embed,
)
from tutor.retrieval.store import embedded_ids, save_embeddings


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    ``argparse`` (built into Python) turns ``--chapter 1`` into
    ``args.chapter == 1``, checks the types, and writes ``--help`` for free.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(description="Load one chapter of the book.")
    parser.add_argument("--chapter", type=int, help="the chapter number, e.g. 1")
    parser.add_argument("--pdf", type=Path, default=Path("data/d2l-en.pdf"), help="the book PDF")
    parser.add_argument("--db", type=Path, default=Path("data/tutor.db"), help="the database")
    # action="store_true" makes a flag: present means True, absent means False.
    parser.add_argument("--show", action="store_true", help="print the passages; save nothing")
    parser.add_argument("--fonts", action="store_true", help="list the fonts; save nothing")
    parser.add_argument(
        "--embed", action="store_true", help="embed every saved passage again; read no PDF"
    )
    args = parser.parse_args()
    if args.chapter is None and not args.embed:
        # parser.error prints the message with the usage line and stops.
        parser.error("--chapter is required, unless you use --embed")
    return args


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


def embed_saved_passages(conn: sqlite3.Connection, everything: bool) -> None:
    """Embed saved passages, save their vectors, and list any over bge's token limit.

    Args:
        conn: An open connection.
        everything: True to embed every saved passage again; False to embed
            only the passages with no vector from the current model yet.
    """
    passages = list_passages(conn)
    if not everything:
        done = embedded_ids(conn, MODEL_NAME)
        passages = [passage for passage in passages if passage["id"] not in done]
    if not passages:
        print("Every passage already has a vector.")
        return
    start = time.perf_counter()
    vectors = embed_passages(passages)
    save_embeddings(conn, [passage["id"] for passage in passages], vectors, MODEL_NAME)
    seconds = time.perf_counter() - start
    print(f"Embedded {len(passages)} passages with {MODEL_NAME} in {seconds:.1f} s.")
    # bge reads at most MAX_TOKENS tokens and drops the rest without a word.
    # List those passages, so a chapter where it happens often gets noticed.
    over = []
    for passage in passages:
        tokens = count_tokens(text_to_embed(passage))
        if tokens > MAX_TOKENS:
            over.append(f"{passage['number']} passage {passage['position']} ({tokens} tokens)")
    print(
        f"Passages over bge's {MAX_TOKENS}-token limit, whose end search can't see: "
        f"{', '.join(over) if over else 'none'}"
    )


def main() -> None:
    """Run the command: list fonts, show the passages, save a chapter, or re-embed."""
    # Windows terminals sometimes can't print "–" in their default encoding; UTF-8 can.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.embed:
        conn = connect(args.db)
        embed_saved_passages(conn, everything=True)
        conn.close()
        return
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
    print_summary(chapter, passages)
    print()
    embed_saved_passages(conn, everything=False)
    conn.close()
    print(f"\nSaved to {args.db}.")


if __name__ == "__main__":
    main()
