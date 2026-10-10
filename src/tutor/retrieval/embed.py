"""Turn text into vectors with the bge-small embedding model.

M1 concept "Embeddings and vector search". An embedding model is a neural
network (a small transformer) trained so that texts with similar meanings get
vectors pointing in similar directions, even when they share no words. Ours is
``BAAI/bge-small-en-v1.5``: 384 numbers per text, MIT-licensed, small enough
for a laptop CPU.

fastembed runs it on ONNX Runtime, an engine for already-trained models, so no
PyTorch is needed (PLAN.md › Decisions › Embeddings). Every vector it returns
already has length 1 ("normalised"), so only its direction carries meaning.

Passages and queries must go through this same model: vectors from different
models live in different "coordinate systems" and can't be compared.
"""

import copy
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from tokenizers import Tokenizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
"""The embedding model. Changing it means embedding every passage again."""

MODEL_DIR = Path("data/models")
"""Where the model is downloaded on first use (about 67 MB). data/ is git-ignored."""

BATCH_SIZE = 1
"""How many texts bge reads at once.

Measured on this laptop's CPU with chapter 1's 49 passages: one at a time took
about 4 s, and fastembed's default batch of 256 took about 8 s, with identical
vectors. In a batch, every text is padded to the length of the longest one, and
a single passage already keeps a CPU's few cores busy. Batching pays off on a
GPU, whose thousands of cores one text can't fill.
"""

MAX_TOKENS = 512
"""The most tokens bge reads from one text. It silently drops the rest.

Only search loses the dropped end: the passages table keeps the whole text for
the question writer, the grader and the reader. The ingest command lists every
passage over the limit (PLAN.md › Decisions › Passage).
"""


@lru_cache(maxsize=1)
def load_model() -> TextEmbedding:
    """Load the embedding model, downloading it the first time.

    Loading takes a second or two, so ``lru_cache`` keeps the loaded model:
    the first call loads it, and every later call returns that same object.

    Returns:
        The fastembed model. ``model.embed(texts)`` turns a list of strings
        into one 384-number NumPy vector per string.
    """
    return TextEmbedding(model_name=MODEL_NAME, cache_dir=str(MODEL_DIR))


def text_to_embed(passage: sqlite3.Row) -> str:
    """Build the text bge reads for one passage: its topic's header, a blank line, its text.

    For example, passage 2 of 1.2.3 becomes "1.2.3 Objective Functions", an
    empty line, then "During optimization, …". It lives in its own function so
    the token count in the ingest command measures exactly what
    ``embed_passages()`` embeds.

    Args:
        passage: A row from ``list_passages()``. A row's columns are read by
            name: ``passage["title"]``.

    Returns:
        The header and the passage's text, as one string.
    """
    header = f"{passage['number']} {passage['title']}"
    return f"{header}\n\n{passage['text']}"


@lru_cache(maxsize=1)
def _counting_tokenizer() -> Tokenizer:
    """Return a copy of bge's tokenizer that counts every token, with no cut-off.

    bge's own tokenizer stops at MAX_TOKENS, which is right for embedding but
    hides how far over a text goes. ``copy.deepcopy`` makes an independent
    copy, so switching truncation off here never changes what bge reads.
    ``load_model().model`` is fastembed's inner ONNX model, which holds the
    tokenizer (a ``tokenizers.Tokenizer`` from Hugging Face).
    """
    # fastembed doesn't declare .tokenizer in its types, hence the ignore.
    tokenizer = copy.deepcopy(load_model().model.tokenizer)  # type: ignore[attr-defined]
    tokenizer.no_truncation()
    return tokenizer


def count_tokens(text: str) -> int:
    """Count the tokens bge would need for a text, including its [CLS] and [SEP] markers.

    Args:
        text: Any text, such as ``text_to_embed(passage)``.

    Returns:
        The number of tokens. Above MAX_TOKENS, bge reads only the first
        MAX_TOKENS of them.
    """
    return len(_counting_tokenizer().encode(text).ids)


def embed_passages(passages: list[sqlite3.Row]) -> list[np.ndarray]:
    """Turn passages into vectors, each embedded with its topic in front.

    A passage from the middle of a topic often never names it: passage 2 of
    "1.2.3 Objective Functions" talks only about the loss and the training
    data. Embedding it as "1.2.3 Objective Functions", a blank line, then its
    text makes its vector carry the topic too. For the query "objective
    functions", that raised its score from 0.61 to 0.69 (PLAN.md › How the
    tutor behaves in specific cases › Topic search).

    The header is added only here, in memory. The passages table keeps the
    book's exact words, because the question writer, the grader and the
    passage shown under each grade all read them, and a studied passage can
    never change (PLAN.md › Data). A better header or a new model only means
    embedding again.

    Args:
        passages: Rows from ``list_passages()``. Each has the columns
            ``number``, ``title`` and ``text``.

    Returns:
        One 384-number vector per passage, in the same order: the first
        vector belongs to the first passage, and so on.
    """
    # 1. For each passage, build the text bge reads (header, blank line, text),
    #    keeping the strings in a list, in the same order as the passages.
    texts = []
    for passage in passages:
        texts.append(text_to_embed(passage))

    # 2. Embed the whole list in one call and return the vectors as a list.
    #    load_model().embed(texts, batch_size=BATCH_SIZE) hands them back one
    #    at a time, and list(...) collects them.
    return list(load_model().embed(texts, batch_size=BATCH_SIZE))


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
"""bge's instruction for queries, from its makers (BAAI). Passages get no prefix.

bge was trained on pairs of a short search query and the long passage that
answers it, with this sentence in front of each query. Adding it tells the
model "this is a question looking for a passage", so the query's vector lands
nearer the passages that answer it. On chapter 1, the query "gradient descent"
put 1.2.4 Optimization Algorithms ahead of the runner-up by 0.016 without the
prefix and by 0.043 with it.
"""


def embed_query(query: str) -> np.ndarray:
    """Turn a typed topic or question into a vector, ready to compare with passages.

    Args:
        query: What the user typed, such as "gradient descent".

    Returns:
        One 384-number vector.
    """
    # 1. Put QUERY_PREFIX in front of the query, embed that one text, and return
    #    its vector. embed() takes a list and hands back a list of vectors, so
    #    give it a one-item list and take item [0], as in try_embed.py's step 3.
    return list(load_model().embed([QUERY_PREFIX + query]))[0]
