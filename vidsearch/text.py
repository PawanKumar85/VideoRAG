"""Text normalisation shared by the lexical (BM25) and hashing-embedding paths."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\w+")

# Deliberately small. Stopwords carry almost no retrieval signal but dominate
# posting lists, so removing them keeps BM25 fast and the hashing space clean.
STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i if in into is it its
    me my of on or our so that the their them then there these they this to us was we
    were what when where which who will with you your do does did can could should would
    how why about""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Keeps digits and underscores so IDs/error codes survive."""
    return _WORD_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def clean_repeated_phrases(text: str) -> str:
    """Detect and collapse runaway repetitive phrases or character loops.

    Common in Whisper output on background music, pauses, or audio artifacts:
      e.g. 'a bit of a bit of a bit of...', 'पी पी पी...', '-e-e-e-e-'
    """
    if not text:
        return ""

    # 1. Collapse repeating short character patterns (length 1 to 15, repeated >= 3 times)
    for length in range(15, 0, -1):
        pattern = re.compile(rf"(.{{{length}}})(?:\s*\1){{3,}}", re.DOTALL)
        text = pattern.sub(r"\1", text)

    # 2. Token-level collapse for repeating word n-grams (1 to 8 words)
    words = text.split()
    if not words:
        return text

    i = 0
    cleaned_words: list[str] = []
    n_words = len(words)
    while i < n_words:
        matched = False
        max_k = min(8, (n_words - i) // 2)
        for k in range(max_k, 0, -1):
            ngram = [w.lower() for w in words[i : i + k]]
            repeat_count = 1
            j = i + k
            while j + k <= n_words and [w.lower() for w in words[j : j + k]] == ngram:
                repeat_count += 1
                j += k
            if repeat_count >= 2:
                cleaned_words.extend(words[i : i + k])
                i = j
                matched = True
                break
        if not matched:
            cleaned_words.append(words[i])
            i += 1

    return " ".join(cleaned_words).strip()

