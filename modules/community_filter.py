"""Reusable profanity redaction for Community posts.

The vocabulary can be overridden with ``KAIROS_PROFANITY_WORDS`` (comma
separated). Matching is case-insensitive, accepts punctuation between letters,
and uses word boundaries so innocent substrings are not redacted.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache


DEFAULT_PROFANITY_WORDS = ("fuck", "shit", "bitch", "asshole", "bastard", "cunt", "dick")


def _configured_words() -> tuple[str, ...]:
    configured = os.getenv("KAIROS_PROFANITY_WORDS", "")
    words = tuple(word.strip().lower() for word in configured.split(",") if word.strip())
    return words or DEFAULT_PROFANITY_WORDS


@lru_cache(maxsize=8)
def _pattern(words: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = []
    for word in words:
        letters = [re.escape(char) for char in word if char.isalnum()]
        if letters:
            alternatives.append(r"[\W_]*".join(letters))
    if not alternatives:
        return re.compile(r"(?!x)x")
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE)


def redact_profanity(text: str | None) -> str:
    """Return ``text`` with configured inappropriate words replaced by ``***``."""
    if not text or not text.strip():
        return ""
    return _pattern(_configured_words()).sub("***", text.strip())


def validate_submission(text: str | None, label: str = "Text") -> str | None:
    """Return a user-facing validation error, or ``None`` when text is valid."""
    if not text or not text.strip():
        return f"{label} cannot be empty."
    return None
