"""Tests for ``tutor.retrieval.embed``: what text goes into the model, and in what order.

The real bge model is never loaded here, because CI would have to download it.
``monkeypatch`` (a pytest fixture) swaps ``load_model`` for a fake model for
the length of one test. The fake records the texts it's given and returns one
made-up vector per text, so the tests check everything except bge's maths.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from tutor.retrieval import embed
from tutor.retrieval.embed import QUERY_PREFIX, embed_passages, embed_query, text_to_embed

PASSAGES = [
    {"number": "1.2.3", "title": "Objective Functions", "text": "During optimization, …"},
    {"number": "1.2.4", "title": "Optimization Algorithms", "text": "Once we have got …"},
]


class FakeModel:
    """Stands in for fastembed's model: records the texts and returns vectors [0], [1] and so on."""

    def __init__(self) -> None:
        """Start with no texts seen."""
        self.texts: list[str] = []

    def embed(self, texts: list[str], **kwargs: Any) -> Iterator[np.ndarray]:
        """Record the texts and hand back one vector per text, numbered in order.

        Args:
            texts: The texts to embed.
            **kwargs: Options such as ``batch_size``, which the fake ignores.

        Yields:
            One vector per text: the first is ``[0.0]``, the second ``[1.0]``, ….
        """
        self.texts += texts
        for i in range(len(texts)):
            yield np.array([float(i)])


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    """Replace the real model with a FakeModel for one test.

    Returns:
        The fake, so the test can read the texts it was given.
    """
    model = FakeModel()
    monkeypatch.setattr(embed, "load_model", lambda: model)
    return model


def test_the_header_goes_in_front_of_the_text() -> None:
    """The model reads the topic's number and title, a blank line, then the passage."""
    assert text_to_embed(PASSAGES[0]) == "1.2.3 Objective Functions\n\nDuring optimization, …"


def test_passages_are_embedded_with_headers_in_order(fake_model: FakeModel) -> None:
    """Each passage is embedded with its header, and vector i belongs to passage i."""
    vectors = embed_passages(PASSAGES)
    assert fake_model.texts == [text_to_embed(passage) for passage in PASSAGES]
    assert [vector[0] for vector in vectors] == [0.0, 1.0]


def test_a_query_gets_the_prefix_and_comes_back_as_one_vector(fake_model: FakeModel) -> None:
    """The query is embedded with bge's prefix, and the result is the vector itself."""
    vector = embed_query("gradient descent")
    assert fake_model.texts == [QUERY_PREFIX + "gradient descent"]
    assert np.array_equal(vector, np.array([0.0]))
