"""The quote check: is the LLM's quote really in the passage?

M1 concept "Structured output (question writing)". Every question, every grade
and every Ask the book citation comes with a quote the LLM copied from the
passage. Plain code checks it (PLAN.md › Quotes checked in code):

- Both texts are tidied the same way first, so harmless differences don't
  count: PDF ligatures ("ﬁ" → "fi"), capitals, quote marks, dashes, extra
  spaces.
- Then the quote must appear in the passage exactly, as whole words. Never
  "similar enough": inserting one "not" into a real quote still scores about
  95% similar. And never by meaning: a made-up line that only sounds right
  must fail.

The check proves the quote is in the passage, not that it supports the
question. Flags (M3) catch that.
"""

import re
import unicodedata


def normalize(text: str) -> str:
    """Tidy a text so that harmless differences don't stop an exact match.

    Every step removes a difference that can't change what a sentence claims:
    how a character is drawn, capitals, quote marks, how a dash is typed, and
    spacing. Letters and digits are never changed, so a flipped word ("easier"
    → "harder") or a dropped minus ("−1" → "1") still shows.

    Args:
        text: A quote or a passage.

    Returns:
        The tidied text, e.g. "The “ﬁrst”  multi-label model" → "the first multilabel model".
    """
    # 1. NFKC ("compatibility" normalization, from the Unicode standard) turns
    #    look-alike characters into their plain forms: the PDF ligature "ﬁ"
    #    becomes "fi", a non-breaking space a space, and "…" becomes "...".
    text = unicodedata.normalize("NFKC", text)

    # 2. Capitals never change meaning.
    text = text.lower()

    # 3. Quote marks, curly or straight, are deleted rather than swapped for
    #    one kind: the tidied text is only compared, never shown, so nothing
    #    needs them. The passage's “cat” and the LLM's "cat" both become cat.
    for mark in "“”‘’\"'":
        text = text.replace(mark, "")

    # 4. The book's other dashes, which NFKC leaves alone:
    #    - the em dash "—" only ever separates words ("equal—if"), so it
    #      becomes a space;
    #    - the minus sign "−" ("say, −1") and the en dash "–" ("5–10 tags")
    #      become a plain "-", the character the LLM types for them.
    text = text.replace("—", " ")
    for dash in "−–":
        text = text.replace(dash, "-")

    # 5. Dashes, in two passes. Both skip a "-" that comes right before a
    #    digit: that's a minus or a range ("say, -1", "5-10"), so it stays, and
    #    a quote that drops the minus still fails. "(?!\d)" is a negative
    #    lookahead: the match goes ahead only if no digit comes next, and the
    #    digit itself isn't consumed. Likewise "(?<=\w)" is a lookbehind and
    #    "(?=[^\W\d])" a lookahead; neither consumes the letters it checks.
    #
    #    a. A hyphen inside a word, with a letter or digit before it and a
    #       letter after it, is deleted. The PDF step already rejoined words
    #       split across lines, and where the split fell on a real hyphen it
    #       dropped it: 1.3.1.3 Tagging says "multilabel". The LLM may write
    #       "multi-label", so hyphens go from both texts. "[^\W\d]" means "a
    #       word character that isn't a digit", i.e. a letter.
    text = re.sub(r"(?<=\w)-(?=[^\W\d])", "", text)
    #    b. Any other dash ("equal - if", "equal--if", "rome -italy") is
    #       punctuation between words, like the em dash, so it becomes a
    #       space. That keeps the words on each side separate words, which
    #       the whole-word check in quote_in_passage() relies on.
    text = re.sub(r"-+(?!\d)", " ", text)

    # 6. Collapse every run of spaces and line breaks into one space, and trim
    #    the ends. split() with no argument splits on any run of whitespace,
    #    so joining the pieces with " " does both at once.
    return " ".join(text.split())


def quote_in_passage(quote: str, passage: str) -> bool:
    """Whether the quote appears in the passage as whole words, once both are normalized.

    Args:
        quote: The words the LLM says it copied.
        passage: The passage's text.

    Returns:
        True if the normalized quote is part of the normalized passage, starting
        and ending on word boundaries.
    """
    normalized_quote = normalize(quote)

    # An empty quote is "in" every text (Python says "" in "abc" is True), so a
    # quote with nothing left after tidying fails outright.
    if not normalized_quote:
        return False

    # Whole words only. A plain substring test (Python's `in`) would find
    # "supervised learning, and the type and number of questions…" inside
    # 1.3.2's "unsupervised learning, and the type and number of questions…",
    # passing a quote that says something the passage doesn't. Chapter 1 has
    # many such pairs: unlikely/likely, nonlinear/linear, unknown/known.
    #
    # So the match may not start or end in the middle of a word:
    # - "(?<!\w)" is a negative lookbehind: no letter or digit just before it;
    # - "(?!\w)" is a negative lookahead: no letter or digit just after it.
    # re.escape() makes every character of the quote literal, so a "." or a
    # "(" in it is matched as itself and not read as regex syntax.
    pattern = r"(?<!\w)" + re.escape(normalized_quote) + r"(?!\w)"

    # re.search() scans the whole passage for the first place the pattern
    # matches, and returns None if there is none.
    return re.search(pattern, normalize(passage)) is not None
