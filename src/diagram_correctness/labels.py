"""Label normalization and matching shared by every judge."""
from __future__ import annotations

import re
import unicodedata

_SYMBOLS = {"×": "x", "≤": " <= ", "≥": " >= ", "≠": " != ", "→": " -> ", "←": " <- "}


def normalize_label(value: str) -> str:
    """Case-folded alphanumeric tokens joined by single spaces.

    "×" becomes "x" so repetition markers such as "N×" keep a distinctive token ("nx").
    Unicode letters (for example ε) are kept, and subscripts are folded by NFKC.
    """
    value = unicodedata.normalize("NFKC", value).casefold()
    for symbol, replacement in _SYMBOLS.items():
        value = value.replace(symbol, replacement)
    return " ".join(re.findall(r"[^\W_]+", value))


def label_tokens(value: str) -> set[str]:
    return set(normalize_label(value).split())


def is_abbreviation(actual: str, expected: str) -> bool:
    """True when a single-token label abbreviates the expected label ("FFN" for
    "Position-wise Feed-Forward Network"). Both arguments are normalized labels."""
    if not actual or " " in actual or len(actual) < 2:
        return False
    acronym = "".join(token[0] for token in expected.split() if token)
    return len(expected.split()) > 1 and actual in acronym


def labels_match(actual: str, expected: str, accept_abbreviations: bool = False) -> bool:
    """Whether a visible label says the same thing as the expected label.

    Exact normalized equality always matches. Otherwise at least half of the combined
    tokens must be shared, so "Error 1 ≤ ε1" matches "Error 1 <= epsilon 1" but "YES"
    never matches "NO" and "Error 1" never matches "Error 2".
    """
    actual_norm, expected_norm = normalize_label(actual), normalize_label(expected)
    if not actual_norm or not expected_norm:
        return actual_norm == expected_norm
    if actual_norm == expected_norm:
        return True
    if accept_abbreviations and is_abbreviation(actual_norm, expected_norm):
        return True
    actual_tokens, expected_tokens = set(actual_norm.split()), set(expected_norm.split())
    # Numbers distinguish otherwise identical labels ("Error 1" vs "Error 2").
    if {t for t in actual_tokens if t.isdigit()} != {t for t in expected_tokens if t.isdigit()}:
        return False
    shared = actual_tokens & expected_tokens
    return len(shared) / len(actual_tokens | expected_tokens) >= 0.5
