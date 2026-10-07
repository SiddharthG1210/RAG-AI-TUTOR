"""Cut each topic's paragraphs into passages of 120–350 words (M1 concept "Chunking").

A passage is the unit the tutor works with: search returns passages, the
question writer writes questions from one passage, and the grader checks
answers against it. So passages must be big enough to hold a real idea and
small enough to stay on one idea (and to fit the embedding model, which reads
about 380 words at most).

The steps, for one topic at a time:

    paragraphs ──► join_page_breaks() ──► chunk_paragraphs() ──► passages
                   glue halves split        pack whole paragraphs,
                   by a page break          then rebalance a short tail

The rules, decided in the M1 concept "Chunking":

1. Go through one topic's paragraphs in order, keeping each paragraph whole.
2. Before adding a paragraph, check whether it would push the passage over
   350 words. If it would, close the passage and start a new one.
3. If the last passage is under 120 words, rebalance: move paragraphs from the
   end of the passage before it until both are in range.
4. A topic under 120 words in total becomes one short passage.
5. A single paragraph over 350 words stops the ingest with a clear error, for
   now. Splitting it at a sentence end gets written when a chapter needs it.
"""

from tutor.ingest.pdf import Chapter, Paragraph

MIN_WORDS = 120
"""The smallest a passage should be: enough text to hold one real idea."""

MAX_WORDS = 350
"""The largest a passage may be, so the embedding model can read all of it."""

SENTENCE_ENDINGS = (".", "?", "!", ":")
"""How a finished paragraph ends. A paragraph ending any other way was cut off."""


def word_count(paragraphs: list[Paragraph]) -> int:
    """Count the words in a list of paragraphs, e.g. one passage.

    Args:
        paragraphs: The paragraphs to count.

    Returns:
        The total number of words. ``.split()`` breaks text at whitespace, so
        each piece it returns is one word.
    """
    return sum(len(paragraph.text.split()) for paragraph in paragraphs)


def join_page_breaks(paragraphs: list[Paragraph]) -> list[Paragraph]:
    """Glue back together paragraphs that a page break split in two.

    The PDF step reads text block by block, and a block never spans two pages,
    so a paragraph running off the bottom of a page comes out as two
    paragraphs: "…We need the right" on page 5 and "data. If the data…" on
    page 6. Left like that, a passage could end in the middle of a sentence.

    Two neighbouring paragraphs are the halves of one when the first doesn't
    end a sentence (no ``.`` ``?`` ``!`` or ``:`` at the end) and the second
    starts with a lowercase letter.

    Args:
        paragraphs: One topic's paragraphs, in book order.

    Returns:
        The same paragraphs in the same order, with every split pair glued into
        one paragraph that starts on the first half's page and ends on the
        second half's page.
    """
    joined: list[Paragraph] = []
    for paragraph in paragraphs:
        # Compare with the last paragraph kept so far (joined[-1] is "one from
        # the end"). It may itself be a glued one, which is fine.
        if (
            joined
            and not joined[-1].text.endswith(SENTENCE_ENDINGS)
            and paragraph.text[0].islower()
        ):
            first_half = joined[-1]
            if first_half.text.endswith("-"):
                # A word split by the typesetter: "sub-" + "sequent" → "subsequent".
                text = first_half.text[:-1] + paragraph.text
            else:
                text = first_half.text + " " + paragraph.text
            # Replace the first half with the glued paragraph: it starts where the
            # first half started and ends where this second half ends.
            joined[-1] = Paragraph(
                first_half.topic,
                first_half.page,
                first_half.page_label,
                text,
                end_page=paragraph.end_page,
                end_page_label=paragraph.end_page_label,
            )
        else:
            joined.append(paragraph)
    return joined


def rebalance_last(passages: list[list[Paragraph]]) -> None:
    """Rule 3: grow a short last passage by borrowing from the passage before it.

    Paragraphs move one at a time from the end of the second-to-last passage to
    the front of the last one, which keeps the book's order. Borrowing stops as
    soon as the last passage reaches ``MIN_WORDS``, or when one more move would
    leave the passage before it under ``MIN_WORDS`` or push the last one over
    ``MAX_WORDS``. If it can't reach ``MIN_WORDS``, the last passage stays short
    and the ingest summary lists it, like a topic that is short as a whole.

    The lists are changed in place, so nothing is returned.

    Args:
        passages: One topic's passages, in order.
    """
    if len(passages) < 2:
        return  # a topic with one passage has nothing to borrow from (rule 4)
    before, last = passages[-2], passages[-1]
    while word_count(last) < MIN_WORDS and len(before) > 1:
        moving = before[-1]
        moving_words = word_count([moving])
        if (
            word_count(before) - moving_words < MIN_WORDS
            or word_count(last) + moving_words > MAX_WORDS
        ):
            break
        # .pop() removes and returns a list's last item; .insert(0, x) puts x first.
        last.insert(0, before.pop())


def chunk_paragraphs(paragraphs: list[Paragraph]) -> list[list[Paragraph]]:
    """Group one topic's paragraphs into passages, keeping every paragraph whole.

    Args:
        paragraphs: One topic's paragraphs, in book order, after
            ``join_page_breaks()``.

    Returns:
        The passages, in order. Each passage is the list of whole paragraphs
        it contains, so the caller can read its text, its topic and its pages.
        A topic with no paragraphs gives no passages.

    Raises:
        ValueError: If a single paragraph is over ``MAX_WORDS`` words.
    """
    finished_passages = []
    count = 0
    current_passage = []
    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.text.split())
        if paragraph_word_count > MAX_WORDS:
            raise ValueError(
                f"A paragraph in topic {paragraph.topic} on page {paragraph.page_label} "
                f"is {paragraph_word_count} words, longer than {MAX_WORDS}."
            )
        if count + paragraph_word_count > MAX_WORDS:
            finished_passages.append(current_passage)
            count = 0
            current_passage = []
        current_passage.append(paragraph)
        count = count + paragraph_word_count
    if current_passage:  # an empty topic leaves nothing to close
        finished_passages.append(current_passage)
    rebalance_last(finished_passages)
    return finished_passages


def chunk_chapter(chapter: Chapter) -> dict[str, list[list[Paragraph]]]:
    """Cut every topic of a chapter into passages, one topic at a time.

    Each topic's paragraphs are picked out by their topic tag, so a passage
    can never mix two topics.

    Args:
        chapter: The chapter from ``extract_chapter()``.

    Returns:
        Each topic's passages, keyed by topic number, e.g. ``{"1.3.1.2": [...]}``.
    """
    return {
        topic.number: chunk_paragraphs(
            join_page_breaks([p for p in chapter.paragraphs if p.topic == topic.number])
        )
        for topic in chapter.topics
    }
