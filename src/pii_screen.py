"""Mask personal data in case text before it is sent to a model.

The development packs were already redacted, so this is a safety net for packs
that are not. Site addresses and postcodes are kept because the assessment
needs them; names and text inside images are not covered.
"""

import re
from collections import Counter

# Bank details are masked only after a label, so dates like 12-03-24 are kept.
PATTERNS = {
    "email": r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b",
    "phone": r"(?<!\d)(?<!\d\.)(?:\+44\s?(?:\(0\)\s?)?|0)[1-9](?:[\s-]?\d){8,9}(?!\d)(?!\.\d)",
    "ni_number": r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b",
    "sort_code": r"(?i)(?P<label>\bsort\s+code[\s:]*)\d{2}[\s-]?\d{2}[\s-]?\d{2}\b",
    "account_number": r"(?i)(?P<label>\baccount\s+(?:number|no\.?)[\s:]*)\d{8}\b",
}
CARD_PATTERN = r"(?<!\d)(?<!\d\.)(?:\d[ -]?){12,18}\d(?!\d)(?!\.\d)"


def is_card_number(value: str) -> bool:
    """Return whether a digit string passes the Luhn check for card numbers."""
    digits = [int(d) for d in re.sub(r"\D", "", value)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def mask_pii(text: str) -> tuple[str, Counter[str]]:
    """Return ``text`` with personal data masked, and counts by kind."""
    counts: Counter[str] = Counter()

    for kind, pattern in PATTERNS.items():

        def replace(match: re.Match[str], kind: str = kind) -> str:
            return f"{match.groupdict().get('label', '')}<masked_{kind}>"

        text, counts[kind] = re.subn(pattern, replace, text)

    def replace_card(match: re.Match[str]) -> str:
        if not is_card_number(match.group()):
            return match.group()
        counts["card_number"] += 1
        return "<masked_card_number>"

    text = re.sub(CARD_PATTERN, replace_card, text)
    return text, +counts
