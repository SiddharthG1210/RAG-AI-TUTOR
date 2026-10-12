"""Tests for ``tutor.quotes``: the quote check, and what it must accept and refuse.

These are the M1 "Done when" cases for the quote check: a real quote passes,
a made-up one fails, and so does a real quote with one word flipped. The text
is the real passage 2 of 1.3.1.2 Classification.
"""

from tutor.quotes import normalize, quote_in_passage

PASSAGE = (
    "For reasons that we will get into as the book gets more technical, it can be "
    "difficult to optimize a model that can only output a firm categorical assignment, "
    "e.g., either “cat” or “dog”. In these cases, it is usually much easier to express "
    "our model in the language of probabilities."
)


def test_a_real_quote_passes() -> None:
    """Words copied from the passage are found."""
    quote = "it is usually much easier to express our model in the language of probabilities"
    assert quote_in_passage(quote, PASSAGE)


def test_a_made_up_quote_fails() -> None:
    """A line the passage never says is refused, however plausible it sounds."""
    assert not quote_in_passage("classifiers always output exactly one label", PASSAGE)


def test_a_real_quote_with_one_word_flipped_fails() -> None:
    """Changing one word ("easier" → "harder") makes it a different claim: refused."""
    quote = "it is usually much harder to express our model in the language of probabilities"
    assert not quote_in_passage(quote, PASSAGE)


def test_a_quote_that_drops_a_minus_fails() -> None:
    """Dropping the minus turns a penalty into a reward, so it's refused.

    The text is real, from 1.3.4. The book's minus is "−"; the LLM types "-".
    """
    passage = "receiving a reward of, say, −1. So reinforcement learners"
    assert quote_in_passage("a reward of, say, -1", passage)
    assert not quote_in_passage("a reward of, say, 1", passage)


def test_a_quote_cut_inside_a_word_fails() -> None:
    """Starting mid-word turns "unsupervised" into "supervised": refused.

    The text is real, from 1.3.2. A plain substring test would pass it.
    """
    passage = (
        "We call this class of problems unsupervised learning, and the type and "
        "number of questions we can ask is limited only by our creativity."
    )
    assert quote_in_passage("unsupervised learning, and the type and number", passage)
    assert not quote_in_passage("supervised learning, and the type and number", passage)


def test_a_quote_next_to_an_em_dash_passes() -> None:
    """A quote that starts right after an em dash, or types it as " - ", still matches.

    The text is real, from 1.3.1.2.
    """
    passage = (
        "In such cases not all errors are equal—if we must err, we might prefer "
        "to misclassify to a related class rather than a distant class."
    )
    assert quote_in_passage("if we must err, we might prefer to misclassify", passage)
    assert quote_in_passage("not all errors are equal - if we must err", passage)


def test_an_empty_quote_fails() -> None:
    """An empty quote would be "in" every text, so it must be refused."""
    assert not quote_in_passage("   ", PASSAGE)


def test_harmless_differences_dont_count() -> None:
    """Capitals, straight quotes and extra spaces in the copy still match."""
    assert quote_in_passage('either "CAT"  or\n"dog"', PASSAGE)


def test_normalize_tidies_each_kind_of_difference() -> None:
    """Ligatures, capitals, quote marks, dashes and spaces are all tidied."""
    assert normalize("The ﬁrst") == "the first"
    assert normalize("“cat” and ‘dog’") == "cat and dog"
    # The PDF step glued "multi-" + "label" (a line break fell on the real
    # hyphen), so 1.3.1.3 Tagging says "multilabel". The LLM may write either
    # form, so hyphens go from both texts.
    assert normalize("multi-label") == normalize("multilabel") == "multilabel"
    # A dash before a number stays, as a plain "-": signs and ranges matter.
    assert normalize("say, −1") == normalize("say, -1") == "say, -1"
    assert normalize("5–10 tags") == normalize("5-10 tags") == "5-10 tags"
    # Any other dash separates words, however it's typed: it becomes a space.
    assert normalize("equal—if") == normalize("equal - if") == normalize("equal--if") == "equal if"
    assert normalize("  a\n\n b   c ") == "a b c"
