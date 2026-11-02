"""Deterministic screening of case documents for prompt-injection attempts.

Pattern matching is deliberately simple and auditable;
it catches common phrasings, not every possible attack.
A more complete version could be developed later for better coverage.
"""

import re

from pydantic import BaseModel, Field

EXCERPT_CONTEXT_CHARS = 60

INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget)\b[^.]{0,60}?\b(instructions?|guidance|"
        r"precedents?|materials?|polic(y|ies)|previous|prior|above|earlier)\b",
        re.IGNORECASE,
    ),
    "should_be_ignored": re.compile(
        r"\bshould\s+be\s+(ignored|disregarded)\b", re.IGNORECASE
    ),
    "addresses_assessor": re.compile(
        r"\bfor\s+(any|the)\s+(assessor|reviewer|ai|model|llm|assistant)s?\b",
        re.IGNORECASE,
    ),
    "do_not_rely": re.compile(
        r"\b(do\s+not|don't|must\s+not)\s+(rely\s+on|consider|take\s+into\s+account)\b",
        re.IGNORECASE,
    ),
    "dictates_outcome": re.compile(
        r"\b(recommended\s+outcome\s+wording|outcome\s+(should|must)\s+be|"
        r"you\s+(must|should)\s+(approve|refuse)|"
        r"(approve|refuse)\s+(this|the)\s+application\s+regardless)\b",
        re.IGNORECASE,
    ),
    "reframes_concerns": re.compile(
        r"\b(describe|present|treat)\s+(them|these|any\s+concerns?)\s+as\b",
        re.IGNORECASE,
    ),
    "proceed_on_basis": re.compile(
        r"\bproceed\s+on\s+the\s+basis\s+that\b", re.IGNORECASE
    ),
    "model_role": re.compile(
        r"(\bas\s+an\s+ai\b|\byou\s+are\s+an?\s+(ai|language\s+model|assistant)\b|"
        r"\bsystem\s+prompt\b|</?(system|instructions?)>)",
        re.IGNORECASE,
    ),
}


class InjectionFinding(BaseModel):
    """One suspected instruction aimed at the assessor."""

    pattern: str = Field(..., description="Name of the pattern that matched")
    excerpt: str = Field(..., description="Matched text with surrounding context")


def screen_text(text: str) -> list[InjectionFinding]:
    """Return suspected prompt-injection passages in ``text``.

    Args:
        text: Document text to screen.
    """
    flat = re.sub(r"\s+", " ", text)
    findings: list[InjectionFinding] = []
    for name, pattern in INJECTION_PATTERNS.items():
        if match := pattern.search(flat):
            left = max(0, match.start() - EXCERPT_CONTEXT_CHARS)
            excerpt = flat[left : match.end() + EXCERPT_CONTEXT_CHARS].strip()
            findings.append(InjectionFinding(pattern=name, excerpt=excerpt))
    return findings
